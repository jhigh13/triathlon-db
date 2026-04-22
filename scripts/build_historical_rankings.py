"""
Build historical World Triathlon Rankings from race results.

Usage:
    python scripts/build_historical_rankings.py --phase=all
    python scripts/build_historical_rankings.py --phase=points
    python scripts/build_historical_rankings.py --phase=rankings
    python scripts/build_historical_rankings.py --phase=validate
    python scripts/build_historical_rankings.py --phase=rankings --start=2024-01-01
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(override=True)

from tri_analysis.database import get_engine
from tri_analysis.ranking_points import (
    compute_event_points_batch,
    simulate_weekly_rankings,
    validate_against_official,
)


def phase_points(engine):
    print("=" * 60)
    print("PHASE 3: Computing event points for all elite results")
    print("=" * 60)
    n = compute_event_points_batch(engine)
    print(f"\nDone. {n:,} rows written to computed_event_points.")


def phase_rankings(engine, start: date, end: date):
    print("=" * 60)
    print(f"PHASE 4: Simulating weekly rankings {start} to {end}")
    print("=" * 60)
    n = simulate_weekly_rankings(engine, start, end)
    print(f"\nDone. {n:,} rows written to computed_weekly_rankings.")


def phase_validate(engine):
    print("=" * 60)
    print("PHASE 5: Validating against official API snapshots")
    print("=" * 60)
    for cat_id, label in [(13, "World Rankings — Men"), (14, "World Rankings — Women")]:
        print(f"\n--- {label} ---")
        results = validate_against_official(engine, cat_id)
        if not results:
            print("  No comparable snapshots found.")
            continue
        for snap_date, metrics in sorted(results.items()):
            top10 = metrics["top10_overlap"]
            top10_n = metrics["top10_official"]
            delta = metrics["top50_mean_rank_delta"]
            sp = metrics["top50_spearman"]
            print(
                f"  {snap_date}  "
                f"top10={top10}/{top10_n}  "
                f"top50_avg_delta={delta}  "
                f"spearman={sp}  "
                f"common={metrics['common_athletes']}"
            )


def main():
    parser = argparse.ArgumentParser(description="Build historical WT rankings")
    parser.add_argument(
        "--phase",
        choices=["points", "rankings", "validate", "all"],
        default="all",
        help="Which phase to run",
    )
    parser.add_argument(
        "--start",
        default="2022-01-02",
        help="Start date for weekly rankings simulation (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=str(date.today()),
        help="End date for weekly rankings simulation (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    engine = get_engine()

    if args.phase in ("points", "all"):
        phase_points(engine)

    if args.phase in ("rankings", "all"):
        phase_rankings(engine, start_date, end_date)

    if args.phase in ("validate", "all"):
        phase_validate(engine)

    print("\nAll done.")


if __name__ == "__main__":
    main()
