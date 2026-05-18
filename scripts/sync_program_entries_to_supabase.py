"""
Sync events + program_entries from local Postgres to Supabase.

Scoped by default to events with event_date >= today (start lists for upcoming
races, which is what the PodiumDashboard prediction page needs). Both tables
must already exist on Supabase with matching schemas.

Order of operations:
  1. Upsert events for the chosen scope
  2. Upsert active program_entries for those (event_id, prog_id) pairs
  3. Deactivate Supabase entries inside the same scope that aren't in the
     local active set (mirrors local `deactivate_missing_program_entries`)

Usage (PowerShell):
  python scripts/sync_program_entries_to_supabase.py
  python scripts/sync_program_entries_to_supabase.py --since 2026-04-01
  python scripts/sync_program_entries_to_supabase.py --event-id 195144 --prog-id 678661
  python scripts/sync_program_entries_to_supabase.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(override=True)
_podium_env = REPO_ROOT.parent / "PodiumDashboard" / "PodiumDashboard" / ".env"
if _podium_env.exists():
    load_dotenv(_podium_env, override=False)

from sqlalchemy import create_engine, text

BATCH_SIZE = 500
MAX_RETRIES = 5

def _table_columns(engine, table_name):
    with engine.connect() as c:
        rows = c.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name=:t ORDER BY ordinal_position"
        ), {"t": table_name}).all()
    return [r[0] for r in rows]


def _column_intersection(src_engine, dst_engine, table_name):
    src_cols = _table_columns(src_engine, table_name)
    dst_cols = set(_table_columns(dst_engine, table_name))
    common = [c for c in src_cols if c in dst_cols]
    skipped = [c for c in src_cols if c not in dst_cols]
    return common, skipped


def _build_upsert(table, columns, pk):
    placeholders = ", ".join(f":{c}" for c in columns)
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c not in pk)
    pk_clause = ", ".join(pk)
    return text(
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT ({pk_clause}) DO UPDATE SET {update_set}"
    )

DEACTIVATE_SQL = text("""
    UPDATE program_entries
       SET is_active = FALSE,
           removed_at = COALESCE(removed_at, NOW())
     WHERE event_id = :event_id
       AND prog_id  = :prog_id
       AND is_active = TRUE
       AND entry_id <> ALL(:keep_entry_ids)
""")


def _local_engine():
    from tri_analysis.database import get_engine
    return get_engine()


def _supabase_engine():
    url = os.environ.get("TRIATHLON_DATABASE_URL")
    if not url:
        raise RuntimeError("TRIATHLON_DATABASE_URL not set in .env")
    return create_engine(url, pool_pre_ping=True, pool_size=1, max_overflow=0)


def _retry_batch(engine, sql, params, label):
    """Execute one batch in its own transaction; retry on transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with engine.begin() as conn:
                conn.execute(sql, params)
            return
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = min(2 ** attempt, 30)
            print(f"  [{label}] attempt {attempt} failed ({e.__class__.__name__}: {str(e)[:120]}); retrying in {wait}s")
            time.sleep(wait)


def _batched(rows, n):
    for i in range(0, len(rows), n):
        yield rows[i : i + n]


def _scope_clause(args):
    where = []
    params = {}
    if args.event_id is not None and args.prog_id is not None:
        where.append("event_id = :event_id AND prog_id = :prog_id")
        params.update(event_id=args.event_id, prog_id=args.prog_id)
    elif args.since:
        where.append("event_date >= :since")
        params["since"] = args.since
    else:
        where.append("event_date >= :since")
        params["since"] = dt.date.today()
    return " AND ".join(where), params


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", type=lambda s: dt.date.fromisoformat(s),
                    help="Only sync events with event_date >= YYYY-MM-DD (default: today)")
    ap.add_argument("--event-id", type=int, help="Limit to a single event_id (use with --prog-id)")
    ap.add_argument("--prog-id", type=int, help="Limit to a single prog_id (use with --event-id)")
    ap.add_argument("--dry-run", action="store_true", help="Show counts and exit without writing")
    ap.add_argument("--no-deactivate", action="store_true",
                    help="Skip the deactivation step (pure additive upsert)")
    args = ap.parse_args()

    if (args.event_id is None) != (args.prog_id is None):
        ap.error("--event-id and --prog-id must be used together")

    where, params = _scope_clause(args)

    src = _local_engine()
    dst = _supabase_engine()

    print("=" * 80)
    print(f"Sync scope: {where} {params}")
    print("=" * 80)

    event_cols, event_skipped = _column_intersection(src, dst, "events")
    entry_cols, entry_skipped = _column_intersection(src, dst, "program_entries")
    if event_skipped:
        print(f"Skipping events columns missing on Supabase: {event_skipped}")
    if entry_skipped:
        print(f"Skipping program_entries columns missing on Supabase: {entry_skipped}")

    upsert_event = _build_upsert("events", event_cols, ("event_id", "prog_id"))
    upsert_entry = _build_upsert("program_entries", entry_cols, ("event_id", "prog_id", "entry_id"))

    with src.connect() as sc:
        events = sc.execute(text(
            f"SELECT {', '.join(event_cols)} FROM events WHERE {where}"
        ), params).mappings().all()

        entries = sc.execute(text(
            f"""
            SELECT {', '.join(entry_cols)}
              FROM program_entries
             WHERE is_active = TRUE
               AND (event_id, prog_id) IN (
                   SELECT event_id, prog_id FROM events WHERE {where}
               )
            """
        ), params).mappings().all()

    print(f"Local events in scope:           {len(events)}")
    print(f"Local active program_entries:    {len(entries)}")

    if not events and not entries:
        print("Nothing to sync.")
        return

    keep_by_prog: dict[tuple[int, int], list[int]] = {}
    for r in entries:
        keep_by_prog.setdefault((r["event_id"], r["prog_id"]), []).append(r["entry_id"])

    print(f"Distinct (event_id, prog_id):    {len(keep_by_prog)}")

    if args.dry_run:
        print("\n--- DRY RUN — no writes ---")
        with dst.connect() as dc:
            n_events = dc.execute(text(
                f"SELECT COUNT(*) FROM events WHERE {where}"
            ), params).scalar()
            n_entries = dc.execute(text(
                f"""
                SELECT COUNT(*) FROM program_entries
                 WHERE is_active=TRUE AND (event_id, prog_id) IN (
                    SELECT event_id, prog_id FROM events WHERE {where}
                 )
                """
            ), params).scalar()
        print(f"Supabase already has: events={n_events}, active entries={n_entries}")
        print(f"Would upsert {len(events)} events and {len(entries)} entries.")
        if not args.no_deactivate:
            print(f"Would run deactivate-missing across {len(keep_by_prog)} programs.")
        return

    # Per-batch transactions so a failure in one batch doesn't poison the rest.
    for batch in _batched(events, BATCH_SIZE):
        _retry_batch(dst, upsert_event, [dict(r) for r in batch], "events")
    print(f"Upserted events:                 {len(events)}")

    for batch in _batched(entries, BATCH_SIZE):
        _retry_batch(dst, upsert_entry, [dict(r) for r in batch], "program_entries")
    print(f"Upserted program_entries:        {len(entries)}")

    if not args.no_deactivate:
        deact = 0
        with dst.begin() as dc:
            for (event_id, prog_id), keep_ids in keep_by_prog.items():
                res = dc.execute(DEACTIVATE_SQL, {
                    "event_id": event_id,
                    "prog_id": prog_id,
                    "keep_entry_ids": keep_ids,
                })
                deact += res.rowcount or 0
        print(f"Deactivated stale entries:       {deact}")

    print("\nDone.")


if __name__ == "__main__":
    main()
