"""
Build the historical Strength-of-Field (SoF) reference distribution.

For every Elite Men / Elite Women race since 2021, computes three raw SoF metrics:
    mean_elo_top10        — mean Elo of the 10 best-Elo athletes in the field
    mean_wt_rank_top10    — mean WT rank of the 10 best-ranked athletes (lower = stronger)
    n_top50_world_ranked  — count of athletes ranked top-50 globally

Then per gender saves the empirical 0–100 percentile array for each metric.
At dashboard prediction time, the SoF service computes the same raw metrics for the
current field and maps to 0–100 by looking up where they fall in the CDF.

Caveats:
- Elo is taken from the CURRENT snapshot (`athlete_elo_ratings`) since we don't store
  historical Elo. This biases stronger athletes' early-career races slightly upward, but
  the bias is systematic across all races so PERCENTILE rankings are still meaningful.
- WT rank IS historical (via `computed_weekly_rankings`), joined to the closest weekly
  ranking date <= the event date.
- Per-distance stratification skipped (per user direction — gender-only).

Usage:
    python scripts/build_sof_reference.py
    # writes outputs/sof_reference.json
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

from tri_analysis.database import get_engine

OUTPUT_PATH = REPO_ROOT / "outputs" / "sof_reference.json"
START_DATE = "2021-01-01"

# ranking_cat_id mapping (confirmed via prog_name lookup)
GENDER_CAT_ID = {"men": 13, "women": 14}


def fetch_race_finishers(engine, gender_prog: str) -> pd.DataFrame:
    """Pull every finisher in every Elite race for one gender since START_DATE."""
    query = text("""
        SELECT
            rr.event_id, rr.prog_id, e.event_date, e.event_name,
            rr.athlete_id, rr.finish_position
        FROM race_results rr
        JOIN events e ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
        WHERE e.prog_name = :pn
          AND e.event_date >= :start_date
          AND e.event_date < CURRENT_DATE
          AND rr.finish_status = 'FINISH'
          AND rr.finish_position IS NOT NULL
          AND COALESCE(e.prog_distance_category, '') != 'long_distance'
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"pn": gender_prog, "start_date": START_DATE})
    return df


def fetch_current_elo(engine, gender: str) -> pd.DataFrame:
    """Pull current Elo snapshot for athletes of one gender."""
    query = text("""
        SELECT athlete_id, elo_rating
        FROM athlete_elo_ratings
        WHERE LOWER(gender) IN (:g1, :g2)
          AND elo_rating IS NOT NULL
    """)
    g_variants = {
        "men":   {"g1": "male",   "g2": "men"},
        "women": {"g1": "female", "g2": "women"},
    }
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=g_variants[gender])
    return df


def fetch_wt_rankings(engine, cat_id: int) -> pd.DataFrame:
    """Pull all weekly rankings for one gender. We'll merge_asof to event_date."""
    query = text("""
        SELECT ranking_date, athlete_id, rank_position
        FROM computed_weekly_rankings
        WHERE ranking_cat_id = :cat
        ORDER BY ranking_date, athlete_id
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"cat": cat_id})
    df["ranking_date"] = pd.to_datetime(df["ranking_date"])
    return df


def join_athlete_wt_rank_asof(finishers: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """
    For each (event_date, athlete_id), find the most-recent WT rank at or before event_date.
    Uses pandas merge_asof grouped by athlete_id.
    """
    f = finishers.copy()
    f["event_date"] = pd.to_datetime(f["event_date"])
    # merge_asof requires the ON column sorted globally (not just within `by` groups)
    f_sorted = f.sort_values("event_date").reset_index()  # preserve original index
    r_sorted = rankings.sort_values("ranking_date").reset_index(drop=True)

    merged = pd.merge_asof(
        f_sorted,
        r_sorted.rename(columns={"ranking_date": "_rk_date"}),
        left_on="event_date",
        right_on="_rk_date",
        by="athlete_id",
        direction="backward",
        tolerance=pd.Timedelta(days=180),  # ignore rank older than 6 months
    )
    # Restore original ordering
    return merged.sort_values("index").drop(columns=["index"]).reset_index(drop=True)


def compute_field_metrics(group: pd.DataFrame) -> dict:
    """Per-race SoF raw metrics."""
    elo_vals = group["elo_rating"].dropna()
    elo_top10 = elo_vals.nlargest(10) if len(elo_vals) > 0 else pd.Series(dtype=float)

    wt_vals = group["rank_position"].dropna()
    wt_top10 = wt_vals.nsmallest(10) if len(wt_vals) > 0 else pd.Series(dtype=float)
    n_top50 = int((wt_vals <= 50).sum())

    return {
        "event_id": group["event_id"].iloc[0],
        "prog_id": group["prog_id"].iloc[0],
        "event_date": group["event_date"].iloc[0],
        "event_name": group["event_name"].iloc[0],
        "field_size": len(group),
        "n_with_elo": len(elo_vals),
        "n_with_wt_rank": len(wt_vals),
        "mean_elo_top10": float(elo_top10.mean()) if len(elo_top10) > 0 else np.nan,
        "mean_wt_rank_top10": float(wt_top10.mean()) if len(wt_top10) > 0 else np.nan,
        "n_top50_world_ranked": n_top50,
    }


def quantile_array(series: pd.Series, n: int = 101) -> list:
    """Return raw values at percentiles 0, 1, 2, ..., 100. Sorted ascending."""
    s = series.dropna()
    if len(s) == 0:
        return [None] * n
    qs = np.linspace(0, 1, n)
    return [float(s.quantile(q)) for q in qs]


def main():
    engine = get_engine()
    reference = {
        "metadata": {
            "built_at": pd.Timestamp.now().isoformat(),
            "start_date": START_DATE,
            "note": "Elo from current snapshot (proxy); WT rank historical via weekly rankings",
        }
    }

    print("=" * 80)
    print("Building SoF reference distributions")
    print("=" * 80)

    for gender, prog_name in [("men", "Elite Men"), ("women", "Elite Women")]:
        print(f"\n--- {gender.upper()} ({prog_name}) ---")
        finishers = fetch_race_finishers(engine, prog_name)
        print(f"  Finishers since {START_DATE}: {len(finishers):,}")

        elo = fetch_current_elo(engine, gender)
        print(f"  Current Elo rows: {len(elo):,}")

        rankings = fetch_wt_rankings(engine, GENDER_CAT_ID[gender])
        print(f"  Weekly rankings rows: {len(rankings):,}")

        # Join Elo (one-to-one current snapshot)
        f1 = finishers.merge(elo, on="athlete_id", how="left")
        # Join WT rank as-of event_date
        f2 = join_athlete_wt_rank_asof(f1, rankings)

        # Aggregate to per-race metrics
        per_race = []
        for (eid, pid), grp in f2.groupby(["event_id", "prog_id"], sort=False):
            per_race.append(compute_field_metrics(grp))
        per_race_df = pd.DataFrame(per_race)
        print(f"  Distinct programs: {len(per_race_df):,}")

        # Diagnostics
        print(f"  mean_elo_top10:       median={per_race_df['mean_elo_top10'].median():.0f}, "
              f"p10={per_race_df['mean_elo_top10'].quantile(0.10):.0f}, "
              f"p90={per_race_df['mean_elo_top10'].quantile(0.90):.0f}")
        print(f"  mean_wt_rank_top10:   median={per_race_df['mean_wt_rank_top10'].median():.0f}, "
              f"p10={per_race_df['mean_wt_rank_top10'].quantile(0.10):.0f}, "
              f"p90={per_race_df['mean_wt_rank_top10'].quantile(0.90):.0f}")
        print(f"  n_top50_world_ranked: median={per_race_df['n_top50_world_ranked'].median():.0f}, "
              f"p10={per_race_df['n_top50_world_ranked'].quantile(0.10):.0f}, "
              f"p90={per_race_df['n_top50_world_ranked'].quantile(0.90):.0f}")

        reference[gender] = {
            "n_reference_races": len(per_race_df),
            "metrics": {
                "mean_elo_top10": {
                    "higher_is_better": True,
                    "percentiles": quantile_array(per_race_df["mean_elo_top10"]),
                },
                "mean_wt_rank_top10": {
                    "higher_is_better": False,
                    "percentiles": quantile_array(per_race_df["mean_wt_rank_top10"]),
                },
                "n_top50_world_ranked": {
                    "higher_is_better": True,
                    "percentiles": quantile_array(per_race_df["n_top50_world_ranked"]),
                },
            },
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
