"""
Per-race Elo trajectory computation for para triathlon athletes.

Unlike `elo_ratings.py` (which retains only each athlete's final rating in a
snapshot table), this module emits the full per-race Elo *trajectory* so we can
plot rating vs. years-of-experience "performance funnels".

Key design decisions:
- One Elo pool per para category (prog_name). Para categories never race each
  other, and within a category all sub-classes (B1/B2/B3, H1/H2) race together on
  factored times, so the competition pool == prog_name exactly.
- Reuses the exact pairwise, field-size-scaled-K update from elo_ratings.py.
- Years of experience = time since the athlete's first para race *in our DB*
  (within the pool). Para history floor is 2017-03-11, so earlier starters are
  left-censored (their true experience may be higher).

Usage:
    python -m tri_analysis.elo_trajectory --check        # sanity-check pools
    python -m tri_analysis.elo_trajectory --category "PTWC Women"
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

try:
    from .database import get_engine
    from .elo_ratings import (
        STARTING_ELO,
        TIER_K_FACTORS,
        classify_event_tier,
        expected_score,
    )
except ImportError:  # pragma: no cover - direct script execution
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from tri_analysis.database import get_engine
    from tri_analysis.elo_ratings import (
        STARTING_ELO,
        TIER_K_FACTORS,
        classify_event_tier,
        expected_score,
    )

logger = logging.getLogger(__name__)

# Earliest para race present in the DB. First races at/before this date are
# left-censored: the athlete may have competed before our data window.
PARA_HISTORY_FLOOR = pd.Timestamp("2017-03-12")

CATEGORIES = ["PTVI Women", "PTVI Men", "PTS3 Women", "PTWC Women"]


def fetch_para_races(engine: Engine, prog_name: str, since_year: int = 2008) -> pd.DataFrame:
    """Fetch all finisher rows for one para category, ordered chronologically."""
    query = text(
        """
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
        WHERE e.is_para = TRUE
          AND e.prog_name = :prog_name
          AND EXTRACT(YEAR FROM e.event_date) >= :since_year
          AND rr.finish_status = 'FINISH'
          AND rr.finish_position IS NOT NULL
        ORDER BY e.event_date, rr.event_id, rr.prog_id, rr.finish_position
        """
    )
    df = pd.read_sql(query, engine, params={"prog_name": prog_name, "since_year": since_year})
    df["event_date"] = pd.to_datetime(df["event_date"])
    logger.info(
        "[%s] fetched %d finisher rows across %d races",
        prog_name, len(df), df[["event_id", "prog_id"]].drop_duplicates().shape[0],
    )
    return df


def compute_elo_trajectory(races_df: pd.DataFrame) -> pd.DataFrame:
    """Process races chronologically, emitting one row per (athlete, race).

    Uses the same pairwise field-size-scaled-K update as elo_ratings.compute_elo_ratings,
    but records each athlete's rating *after* every race rather than only the final value.

    Returns long DataFrame:
        athlete_id, event_id, prog_id, event_date, elo_after, race_number, n_finishers, tier
    """
    ratings: dict[int, float] = {}
    race_counts: dict[int, int] = {}
    out_rows: list[dict] = []

    race_groups = races_df.groupby(["event_id", "prog_id"], sort=False)

    for (event_id, prog_id), race_df in race_groups:
        race_df = race_df.sort_values("finish_position")
        n_finishers = len(race_df)
        if n_finishers < 2:
            continue

        event_name = race_df["event_name"].iloc[0] or ""
        tier = classify_event_tier(event_name)
        k_base = TIER_K_FACTORS.get(tier, 12.0)
        k_scaled = k_base / math.sqrt(n_finishers - 1)
        event_date = race_df["event_date"].iloc[0]

        athlete_list = [int(a) for a in race_df["athlete_id"].values]
        positions = race_df["finish_position"].values

        for aid in athlete_list:
            ratings.setdefault(aid, STARTING_ELO)
            race_counts.setdefault(aid, 0)

        deltas = {aid: 0.0 for aid in athlete_list}
        for i in range(n_finishers):
            aid_i = athlete_list[i]
            pos_i = positions[i]
            r_i = ratings[aid_i]
            for j in range(i + 1, n_finishers):
                aid_j = athlete_list[j]
                pos_j = positions[j]
                r_j = ratings[aid_j]
                if pos_i < pos_j:
                    s_i = 1.0
                elif pos_i > pos_j:
                    s_i = 0.0
                else:
                    s_i = 0.5
                delta = k_scaled * (s_i - expected_score(r_i, r_j))
                deltas[aid_i] += delta
                deltas[aid_j] -= delta

        for aid in athlete_list:
            ratings[aid] += deltas[aid]
            race_counts[aid] += 1
            out_rows.append({
                "athlete_id": aid,
                "event_id": int(event_id),
                "prog_id": int(prog_id),
                "event_date": event_date,
                "elo_after": round(ratings[aid], 2),
                "race_number": race_counts[aid],
                "n_finishers": n_finishers,
                "tier": tier,
            })

    return pd.DataFrame(out_rows)


def attach_experience(traj: pd.DataFrame) -> pd.DataFrame:
    """Add years_experience (since first para race in pool) and left_censored flag."""
    if traj.empty:
        return traj.assign(first_date=pd.NaT, years_experience=float("nan"), left_censored=False)
    first = traj.groupby("athlete_id")["event_date"].transform("min")
    traj = traj.copy()
    traj["first_date"] = first
    traj["years_experience"] = (traj["event_date"] - first).dt.days / 365.25
    traj["left_censored"] = first <= PARA_HISTORY_FLOOR
    return traj


def _name_map(engine: Engine, athlete_ids: list[int]) -> dict[int, str]:
    if not athlete_ids:
        return {}
    id_list = ", ".join(str(int(a)) for a in athlete_ids)
    names: dict[int, str] = {}
    try:
        q = text(f"SELECT athlete_id, full_name FROM athlete WHERE athlete_id IN ({id_list})")
        for _, row in pd.read_sql(q, engine).iterrows():
            if pd.notna(row["full_name"]):
                names[int(row["athlete_id"])] = row["full_name"]
    except Exception:  # pragma: no cover
        pass
    return names


def top_athletes(traj: pd.DataFrame, engine: Engine, n: int = 10) -> pd.DataFrame:
    """Final Elo per athlete (latest race), name-joined, sorted desc."""
    if traj.empty:
        return pd.DataFrame()
    latest = traj.sort_values("event_date").groupby("athlete_id").tail(1)
    latest = latest.sort_values("elo_after", ascending=False).head(n).copy()
    names = _name_map(engine, latest["athlete_id"].tolist())
    latest["name"] = latest["athlete_id"].map(names).fillna(latest["athlete_id"].astype(str))
    return latest[["athlete_id", "name", "elo_after", "race_number", "first_date"]]


def build_pool_trajectory(engine: Engine, prog_name: str, since_year: int = 2008) -> pd.DataFrame:
    """Convenience: fetch -> compute -> attach experience for one category."""
    races = fetch_para_races(engine, prog_name, since_year=since_year)
    traj = compute_elo_trajectory(races)
    return attach_experience(traj)


def _check(engine: Engine):
    """Verification gate: print top athletes per pool and assert known champions."""
    expectations = {
        "PTWC Women": (122515, "Lauren Parker"),
        "PTVI Women": (40701, "Susana Rodriguez"),
    }
    for category in CATEGORIES:
        traj = build_pool_trajectory(engine, category)
        top = top_athletes(traj, engine, n=10)
        print(f"\n{'='*64}\n{category} -- top 10 by final Elo  (pool mean ~ {traj['elo_after'].mean():.0f})\n{'='*64}")
        if top.empty:
            print("  (no data)")
            continue
        for i, row in enumerate(top.itertuples(index=False), 1):
            print(f"  {i:>2} {row.name:<28} {row.elo_after:>7.1f}  ({row.race_number} races)")
        if category in expectations:
            exp_id, exp_name = expectations[category]
            top_id = int(top.iloc[0]["athlete_id"])
            status = "OK" if top_id == exp_id else "WARN"
            print(f"  [{status}] expected {exp_name} ({exp_id}) at #1; got id {top_id}")


def main():
    parser = argparse.ArgumentParser(description="Para Elo trajectory computation")
    parser.add_argument("--check", action="store_true", help="Run sanity checks per pool")
    parser.add_argument("--category", type=str, help="Print trajectory head for one category")
    parser.add_argument("--since", type=int, default=2008)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    engine = get_engine()

    if args.check:
        _check(engine)
        return
    if args.category:
        traj = build_pool_trajectory(engine, args.category, since_year=args.since)
        print(traj.head(30).to_string(index=False))
        print(f"\n{len(traj)} trajectory rows, {traj['athlete_id'].nunique()} athletes")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
