"""
Sync computed_weekly_rankings from local PostgreSQL to Supabase.

Usage:
    python scripts/sync_weekly_rankings_to_supabase.py
    python scripts/sync_weekly_rankings_to_supabase.py --truncate
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(override=True)

# Also try PodiumDashboard's .env for TRIATHLON_DATABASE_URL
_podium_env = Path(__file__).parent.parent.parent / "PodiumDashboard" / "PodiumDashboard" / ".env"
if _podium_env.exists():
    load_dotenv(_podium_env, override=False)

from sqlalchemy import create_engine, text

BATCH_SIZE = 500   # smaller batches to stay within Supabase statement timeout
MAX_RETRIES = 5

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS computed_weekly_rankings (
    ranking_date     DATE    NOT NULL,
    ranking_cat_id   INTEGER NOT NULL,
    athlete_id       INTEGER NOT NULL,
    rank_position    INTEGER NOT NULL,
    total_points     DOUBLE PRECISION NOT NULL,
    events_current   INTEGER,
    events_previous  INTEGER,
    PRIMARY KEY (ranking_date, ranking_cat_id, athlete_id)
)
"""

INSERT_SQL = text("""
    INSERT INTO computed_weekly_rankings
        (ranking_date, ranking_cat_id, athlete_id, rank_position,
         total_points, events_current, events_previous)
    VALUES
        (:ranking_date, :ranking_cat_id, :athlete_id, :rank_position,
         :total_points, :events_current, :events_previous)
    ON CONFLICT (ranking_date, ranking_cat_id, athlete_id)
    DO UPDATE SET
        rank_position   = EXCLUDED.rank_position,
        total_points    = EXCLUDED.total_points,
        events_current  = EXCLUDED.events_current,
        events_previous = EXCLUDED.events_previous
""")


def get_local_engine():
    from tri_analysis.database import get_engine
    return get_engine()


def get_supabase_engine():
    url = os.environ.get("TRIATHLON_DATABASE_URL")
    if not url:
        raise RuntimeError("TRIATHLON_DATABASE_URL not set in .env")
    # pool_pre_ping ensures stale connections are detected and reconnected
    return create_engine(url, pool_pre_ping=True, pool_size=1, max_overflow=0)


def insert_batch(engine, batch, attempt=1):
    params = [
        {
            "ranking_date": r[0],
            "ranking_cat_id": r[1],
            "athlete_id": r[2],
            "rank_position": r[3],
            "total_points": r[4],
            "events_current": r[5],
            "events_previous": r[6],
        }
        for r in batch
    ]
    # Each batch gets its own connection + transaction so a dropped connection
    # only rolls back one batch, not the whole run.
    with engine.begin() as conn:
        conn.execute(INSERT_SQL, params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truncate", action="store_true",
                        help="Truncate destination table before inserting")
    args = parser.parse_args()

    local_engine = get_local_engine()
    supa_engine = get_supabase_engine()

    print("Creating table on Supabase if needed...")
    with supa_engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))

    if args.truncate:
        print("Dropping and recreating destination table...")
        with supa_engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS computed_weekly_rankings"))
            conn.execute(text(CREATE_TABLE_SQL))

    print("Fetching all rows from local DB...")
    with local_engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT ranking_date, ranking_cat_id, athlete_id, rank_position, "
            "total_points, events_current, events_previous "
            "FROM computed_weekly_rankings ORDER BY ranking_date, ranking_cat_id, rank_position"
        )).fetchall()

    total = len(rows)
    print(f"Syncing {total:,} rows to Supabase in batches of {BATCH_SIZE}...")

    inserted = 0
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                insert_batch(supa_engine, batch)
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    print(f"\nBatch {i//BATCH_SIZE + 1} failed after {MAX_RETRIES} retries: {e}")
                    raise
                wait = 2 ** attempt
                print(f"\n  Retry {attempt}/{MAX_RETRIES} for batch {i//BATCH_SIZE + 1} after {wait}s: {e}")
                time.sleep(wait)
        inserted += len(batch)
        pct = 100 * inserted / total
        print(f"  {inserted:,}/{total:,} ({pct:.0f}%)", end="\r", flush=True)

    print(f"\nDone. {inserted:,} rows upserted to Supabase.")


if __name__ == "__main__":
    main()
