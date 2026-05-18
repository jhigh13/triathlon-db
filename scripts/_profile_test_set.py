"""Quick one-off: profile the H2 2025 backtest test set + training availability."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sqlalchemy import text
from tri_analysis.database import get_engine

eng = get_engine()
with eng.connect() as c:
    # Field size distribution for the test set
    fs = pd.read_sql(text("""
        SELECT e.event_id, e.prog_id,
               e.event_date, e.event_name, e.prog_name,
               COUNT(*) AS field_size
        FROM events e
        JOIN race_results rr USING (event_id, prog_id)
        WHERE e.event_date >= '2025-07-01' AND e.event_date <= '2025-12-31'
          AND e.prog_name ~* '(Elite Men|Elite Women)'
          AND COALESCE(e.prog_distance_category, '') != 'long_distance'
        GROUP BY e.event_id, e.prog_id, e.event_date, e.event_name, e.prog_name
        HAVING COUNT(*) >= 10
    """), c)

    print("=== H2 2025 test set ===")
    print(f"Programs: {len(fs)}")
    print(f"Distinct events: {fs.event_id.nunique()}")
    print(f"Distinct dates: {fs.event_date.nunique()}")
    print(f"Date range: {fs.event_date.min()} to {fs.event_date.max()}")
    print()
    print(f"Field size — min={fs.field_size.min()}, p25={fs.field_size.quantile(.25):.0f}, "
          f"median={fs.field_size.median():.0f}, p75={fs.field_size.quantile(.75):.0f}, "
          f"max={fs.field_size.max()}, total athlete-races={fs.field_size.sum()}")
    print()

    print("Field size buckets:")
    for label, lo, hi in [("<20", 0, 20), ("20-30", 20, 30), ("30-40", 30, 40),
                          ("40-50", 40, 50), ("50+", 50, 999)]:
        n = ((fs.field_size >= lo) & (fs.field_size < hi)).sum()
        print(f"  {label:>6}: {n} programs")
    print()

    # Distance distribution
    print("=== Distance distribution ===")
    dist = pd.read_sql(text("""
        SELECT e.prog_distance_category, COUNT(*) AS n_programs
        FROM events e
        JOIN race_results rr USING (event_id, prog_id)
        WHERE e.event_date >= '2025-07-01' AND e.event_date <= '2025-12-31'
          AND e.prog_name ~* '(Elite Men|Elite Women)'
        GROUP BY e.event_id, e.prog_id, e.prog_distance_category
        HAVING COUNT(*) >= 10
    """), c)
    print(dist.groupby("prog_distance_category").size())
    print()

    # Training data availability
    n_train = pd.read_sql(text("""
        SELECT COUNT(*) AS n
        FROM events e
        JOIN race_results rr USING (event_id, prog_id)
        WHERE e.event_date >= '2021-01-01' AND e.event_date <= '2025-06-30'
          AND e.prog_name ~* '(Elite Men|Elite Women)'
    """), c).iloc[0]["n"]
    print(f"Training rows (2021-01-01 → 2025-06-30): {n_train}")

    # 2026 H1 events that could become a future test set
    n_h1_2026 = pd.read_sql(text("""
        SELECT COUNT(*) AS n
        FROM events e
        JOIN race_results rr USING (event_id, prog_id)
        WHERE e.event_date >= '2026-01-01' AND e.event_date < CURRENT_DATE
          AND e.prog_name ~* '(Elite Men|Elite Women)'
    """), c).iloc[0]["n"]
    print(f"2026 H1 athlete-races already finished (potential future eval): {n_h1_2026}")
