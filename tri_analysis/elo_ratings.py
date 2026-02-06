"""
Elo rating system for World Triathlon athletes.

Computes Elo ratings from pairwise race results, processing all races
chronologically from 2008 onward. Designed for mass-start events where
each race produces a complete finishing order.

Key design decisions:
- Pairwise decomposition: Each race generates C(n,2) pairwise outcomes
- K scaled by field size: K_scaled = K_base / sqrt(n_opponents) to prevent
  rating inflation in large fields
- Tier-weighted K: Higher-tier events cause larger rating changes
  (WTCS=32, WorldCup=24, Regional=16, Other=12)
- Gender-specific: Separate rating pools for men and women
- Starting rating: 1500 for all athletes

Usage:
    python -m tri_analysis.elo_ratings               # Full rebuild from 2008
    python -m tri_analysis.elo_ratings --since 2020  # Rebuild from 2020
    python -m tri_analysis.elo_ratings --dry-run     # Compute but don't write to DB
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

try:
    # Works when executed as a module: `python -m tri_analysis.elo_ratings`
    from .database import get_engine
except ImportError:  # pragma: no cover
    # Allows direct script execution: `python .\tri_analysis\elo_ratings.py`
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from tri_analysis.database import get_engine

logger = logging.getLogger(__name__)

# Default starting Elo rating
STARTING_ELO = 1500.0

# Tier-weighted K factors (base K before field-size scaling)
TIER_K_FACTORS = {
    1: 32.0,   # WTCS / Championship Series
    2: 24.0,   # World Cup
    3: 16.0,   # Regional / Continental
    4: 12.0,   # Other
}

# Event tier classification (reuse the same patterns from features.py)
EVENT_TIER_PATTERNS = {
    1: [
        "Championship Series", "WTCS", "Championship Finals", "Grand Final",
        "Olympic", "Olympics", "World Championship",
    ],
    2: [
        "World Triathlon Cup", "ITU Triathlon World Cup",
    ],
    3: [
        "PATCO", "ATU", "OTU", "ASTC", "ETU", "CAMTRI",
        "Continental Cup", "Continental Championships",
        "African Championships", "Asian Championships",
        "Oceania Championships", "Pan-American", "European Championships",
        "European Cup", "Europe Triathlon Cup",
        "Oceania Cup", "Panamerican Cup", "Pan-American Cup",
        "Asia Triathlon Cup", "Africa Triathlon Cup", "Africa Triathlon Premium Cup",
        "Americas Triathlon Cup", "Americas Cup",
    ],
}


def classify_event_tier(event_name: str) -> int:
    """Classify an event into a tier based on its name."""
    if not event_name:
        return 4
    name_upper = event_name.upper()
    for tier in sorted(EVENT_TIER_PATTERNS.keys()):
        for pattern in EVENT_TIER_PATTERNS[tier]:
            if pattern.upper() in name_upper:
                return tier
    return 4


def expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def fetch_all_races(
    engine: Engine,
    since_year: int = 2008,
    elite_only: bool = True,
) -> pd.DataFrame:
    """
    Fetch all race results chronologically for Elo computation.

    Returns one row per athlete per race, ordered by event_date then event_id.
    Only includes finishers with valid finish positions.
    """
    where_clauses = [
        "EXTRACT(YEAR FROM e.event_date) >= :since_year",
        "rr.finish_status = 'FINISH'",
        "rr.finish_position IS NOT NULL",
    ]
    params: dict = {"since_year": since_year}

    if elite_only:
        where_clauses.append("e.prog_name IN ('Elite Men', 'Elite Women')")

    where_sql = " AND ".join(where_clauses)

    query = text(f"""
        SELECT
            rr.event_id,
            rr.prog_id,
            rr.athlete_id,
            rr.finish_position,
            e.event_date,
            e.event_name,
            e.prog_name
        FROM race_results rr
        JOIN events e
            ON rr.event_id = e.event_id
            AND rr.prog_id = e.prog_id
        WHERE {where_sql}
        ORDER BY e.event_date, rr.event_id, rr.prog_id, rr.finish_position
    """)

    df = pd.read_sql(query, engine, params=params)
    logger.info(
        f"Fetched {len(df)} finisher rows across "
        f"{df[['event_id', 'prog_id']].drop_duplicates().shape[0]} races "
        f"(since {since_year})"
    )
    return df


def compute_elo_ratings(
    races_df: pd.DataFrame,
) -> dict[int, dict]:
    """
    Process all races chronologically and compute Elo ratings.

    For each race:
    1. Classify event tier -> determine K_base
    2. Determine gender from prog_name
    3. For each pair of finishers: update ratings based on who beat whom
    4. Scale K by 1/sqrt(n_opponents) to prevent inflation

    Args:
        races_df: DataFrame from fetch_all_races, sorted by event_date

    Returns:
        Dict mapping athlete_id -> {
            'elo_rating': float,
            'elo_peak': float,
            'elo_races': int,
            'gender': str,
            'last_race_date': date,
        }
    """
    # athlete_id -> current state
    ratings: dict[int, dict] = {}

    # Group by race (event_id, prog_id), already sorted by date
    race_groups = races_df.groupby(["event_id", "prog_id"], sort=False)

    n_races = 0
    n_updates = 0

    for (event_id, prog_id), race_df in race_groups:
        race_df = race_df.sort_values("finish_position")
        n_finishers = len(race_df)

        if n_finishers < 2:
            continue

        # Determine tier and K
        event_name = race_df["event_name"].iloc[0] or ""
        tier = classify_event_tier(event_name)
        k_base = TIER_K_FACTORS.get(tier, 12.0)

        # Scale K by field size: prevents inflation in large fields
        n_opponents = n_finishers - 1
        k_scaled = k_base / math.sqrt(n_opponents)

        # Determine gender
        prog_name = str(race_df["prog_name"].iloc[0] or "").lower()
        if "women" in prog_name or "female" in prog_name:
            gender = "female"
        else:
            gender = "male"

        event_date = race_df["event_date"].iloc[0]

        # Initialize any new athletes
        athlete_ids = race_df["athlete_id"].values
        for aid in athlete_ids:
            aid = int(aid)
            if aid not in ratings:
                ratings[aid] = {
                    "elo_rating": STARTING_ELO,
                    "elo_peak": STARTING_ELO,
                    "elo_races": 0,
                    "gender": gender,
                    "last_race_date": None,
                }

        # Get current ratings for this field
        positions = race_df["finish_position"].values
        athlete_list = [int(a) for a in athlete_ids]

        # Accumulate deltas then apply (so all updates use pre-race ratings)
        deltas = {aid: 0.0 for aid in athlete_list}

        # Pairwise updates: for each pair (i, j) where i finished ahead of j
        for i in range(n_finishers):
            aid_i = athlete_list[i]
            pos_i = positions[i]
            r_i = ratings[aid_i]["elo_rating"]

            for j in range(i + 1, n_finishers):
                aid_j = athlete_list[j]
                pos_j = positions[j]
                r_j = ratings[aid_j]["elo_rating"]

                # Actual score: 1 if i beat j, 0.5 if tie
                if pos_i < pos_j:
                    s_i = 1.0
                elif pos_i > pos_j:
                    s_i = 0.0
                else:
                    s_i = 0.5

                e_i = expected_score(r_i, r_j)
                delta = k_scaled * (s_i - e_i)

                deltas[aid_i] += delta
                deltas[aid_j] -= delta
                n_updates += 1

        # Apply deltas
        for aid in athlete_list:
            ratings[aid]["elo_rating"] += deltas[aid]
            ratings[aid]["elo_peak"] = max(
                ratings[aid]["elo_peak"], ratings[aid]["elo_rating"]
            )
            ratings[aid]["elo_races"] += 1
            ratings[aid]["last_race_date"] = event_date

        n_races += 1

        if n_races % 500 == 0:
            logger.info(f"Processed {n_races} races, {n_updates} pairwise updates")

    logger.info(
        f"Elo computation complete: {n_races} races, {len(ratings)} athletes, "
        f"{n_updates} total pairwise updates"
    )
    return ratings


def write_elo_to_database(
    engine: Engine,
    ratings: dict[int, dict],
) -> int:
    """
    Write computed Elo ratings to the athlete_elo_ratings table.

    Performs a full replace (truncate + insert) to ensure consistency.

    Returns:
        Number of rows written
    """
    if not ratings:
        logger.warning("No ratings to write")
        return 0

    rows = []
    now = datetime.now(timezone.utc)
    for aid, data in ratings.items():
        rows.append({
            "athlete_id": int(aid),
            "gender": data["gender"],
            "elo_rating": round(data["elo_rating"], 2),
            "elo_peak": round(data["elo_peak"], 2),
            "elo_races": data["elo_races"],
            "last_race_date": data["last_race_date"],
            "last_updated": now,
        })

    df = pd.DataFrame(rows)

    with engine.begin() as conn:
        # Create table if it doesn't exist
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS athlete_elo_ratings (
                athlete_id INTEGER NOT NULL,
                gender VARCHAR NOT NULL,
                elo_rating DOUBLE PRECISION NOT NULL,
                elo_peak DOUBLE PRECISION NOT NULL,
                elo_races INTEGER NOT NULL,
                last_race_date DATE,
                last_updated TIMESTAMP NOT NULL,
                PRIMARY KEY (athlete_id, gender)
            )
        """))
        conn.execute(text("DELETE FROM athlete_elo_ratings"))
        df.to_sql("athlete_elo_ratings", conn, if_exists="append", index=False)

    logger.info(f"Wrote {len(df)} Elo ratings to database")
    return len(df)


def print_top_ratings(ratings: dict[int, dict], engine: Engine, n: int = 20):
    """Print the top N rated athletes per gender for verification."""
    if not ratings:
        return

    # Collect the athlete IDs we'll actually display (top N per gender)
    display_ids: set[int] = set()
    for gender in ["male", "female"]:
        gender_ratings = [
            (aid, d) for aid, d in ratings.items()
            if d["gender"] == gender and d["elo_races"] >= 5
        ]
        gender_ratings.sort(key=lambda x: x[1]["elo_rating"], reverse=True)
        for aid, _ in gender_ratings[:n]:
            display_ids.add(int(aid))

    if not display_ids:
        return

    # Fetch names: try athlete table first, then fall back to race_results
    id_list = ", ".join(str(a) for a in display_ids)
    name_map: dict[int, str] = {}

    try:
        q1 = text(f"SELECT athlete_id, full_name FROM athlete WHERE athlete_id IN ({id_list})")
        df1 = pd.read_sql(q1, engine)
        for _, row in df1.iterrows():
            if pd.notna(row["full_name"]):
                name_map[int(row["athlete_id"])] = row["full_name"]
    except Exception:
        pass

    # Fill gaps from race_results (athlete_full_name)
    missing = display_ids - set(name_map.keys())
    if missing:
        try:
            miss_list = ", ".join(str(a) for a in missing)
            q2 = text(f"""
                SELECT DISTINCT ON (athlete_id) athlete_id, athlete_full_name
                FROM race_results
                WHERE athlete_id IN ({miss_list}) AND athlete_full_name IS NOT NULL
                ORDER BY athlete_id
            """)
            df2 = pd.read_sql(q2, engine)
            for _, row in df2.iterrows():
                if pd.notna(row["athlete_full_name"]):
                    name_map[int(row["athlete_id"])] = row["athlete_full_name"]
        except Exception:
            pass

    for gender in ["male", "female"]:
        gender_ratings = [
            (aid, d) for aid, d in ratings.items()
            if d["gender"] == gender and d["elo_races"] >= 5
        ]
        gender_ratings.sort(key=lambda x: x[1]["elo_rating"], reverse=True)

        label = "Men" if gender == "male" else "Women"
        print(f"\n{'='*60}")
        print(f"Top {n} {label} (min 5 races):")
        print(f"{'='*60}")
        print(f"{'Rank':<5} {'Name':<30} {'Elo':>7} {'Peak':>7} {'Races':>6}")
        print(f"{'-'*5} {'-'*30} {'-'*7} {'-'*7} {'-'*6}")

        for i, (aid, data) in enumerate(gender_ratings[:n], 1):
            name = name_map.get(aid, f"ID:{aid}")
            print(
                f"{i:<5} {name:<30} "
                f"{data['elo_rating']:>7.1f} "
                f"{data['elo_peak']:>7.1f} "
                f"{data['elo_races']:>6}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Compute Elo ratings for World Triathlon athletes"
    )
    parser.add_argument(
        "--since", type=int, default=2008,
        help="Start year for processing races (default: 2008)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute ratings but don't write to database"
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of top athletes to display per gender (default: 20)"
    )
    parser.add_argument(
        "--include-age-group", action="store_true",
        help="Include non-elite programs (Junior, U23, etc.)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine = get_engine()
    elite_only = not args.include_age_group

    logger.info(f"Fetching races since {args.since} (elite_only={elite_only})")
    races_df = fetch_all_races(engine, since_year=args.since, elite_only=elite_only)

    if races_df.empty:
        logger.warning("No races found, exiting")
        return

    logger.info("Computing Elo ratings...")
    ratings = compute_elo_ratings(races_df)

    print_top_ratings(ratings, engine, n=args.top)

    if not args.dry_run:
        n_written = write_elo_to_database(engine, ratings)
        print(f"\nWrote {n_written} Elo ratings to database.")
    else:
        print(f"\nDry run: {len(ratings)} ratings computed but not written.")


if __name__ == "__main__":
    main()
