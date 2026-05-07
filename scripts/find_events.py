"""List upcoming Elite events in the next 8 weeks with start-list availability."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from sqlalchemy import text

from tri_analysis.database import get_engine

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.max_rows", 200)

WINDOW_WEEKS = 8

engine = get_engine()

print("=" * 90)
print(f"Upcoming Elite events in the next {WINDOW_WEEKS} weeks")
print("=" * 90)

with engine.connect() as conn:
    df = pd.read_sql(
        text(
            """
            SELECT
                e.event_date,
                e.event_id,
                e.prog_id,
                e.event_name,
                e.prog_name,
                COALESCE(pe.n_entries, 0) AS n_entries
            FROM events e
            LEFT JOIN (
                SELECT event_id, prog_id, COUNT(*) AS n_entries
                FROM program_entries
                WHERE is_active = TRUE
                GROUP BY event_id, prog_id
            ) pe
              ON pe.event_id = e.event_id
             AND pe.prog_id = e.prog_id
            WHERE e.event_date >= CURRENT_DATE
              AND e.event_date < CURRENT_DATE + (:weeks * INTERVAL '7 days')
              AND e.prog_name IN ('Elite Men', 'Elite Women')
            ORDER BY e.event_date, e.event_name, e.prog_name
            """
        ),
        conn,
        params={"weeks": WINDOW_WEEKS},
    )

if df.empty:
    print(f"\nNo upcoming Elite events found in the next {WINDOW_WEEKS} weeks.")
    print("Tip: refresh start lists with  `python -m tri_analysis.update_program_entries`")
else:
    with_entries = df[df["n_entries"] > 0]
    without_entries = df[df["n_entries"] == 0]

    print(f"\nTotal: {len(df)} programs across {df['event_id'].nunique()} events")
    print(f"  with start lists:    {len(with_entries)}")
    print(f"  without start lists: {len(without_entries)}")

    if not with_entries.empty:
        print("\n--- With active start lists ---")
        print(with_entries.to_string(index=False))

    if not without_entries.empty:
        print("\n--- No start list yet ---")
        print(without_entries.to_string(index=False))

    print("\nTip: refresh start lists with  `python -m tri_analysis.update_program_entries`")
