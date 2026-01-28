"""Pull upcoming program start lists and store them in the database.

This script fetches upcoming events within a date window, discovers programs for each event,
then calls the Program Entries endpoint to retrieve start lists and persists them to
`program_entries`.

Notes:
- Designed to be fast and incremental (upsert + deactivate missing entries per program).
- Does NOT store waitlists (type=wait) by default.

Usage (PowerShell):
- `python -m tri_analysis.update_program_entries`

Env overrides:
- START_DATE: YYYY-MM-DD (default: today)
- END_DATE:   YYYY-MM-DD (default: today + 60 days)
- EVENTS_PER_PAGE: integer (default: 500)
- PROGRAM_WORKERS: integer (default: 50)
- ENTRY_WORKERS:   integer (default: 20)
"""

from __future__ import annotations

import os
import datetime
import concurrent.futures
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv

try:
    from tri_analysis.database import get_engine, initialize_database
    from tri_analysis.api_handling import (
        fetch_events_ids,
        fetch_program_ids,
        fetch_program_entries,
        normalize_program_entries,
    )
    from tri_analysis.upsert_tables import upsert_program_entries, deactivate_missing_program_entries
except ImportError:  # pragma: no cover
    from database import get_engine, initialize_database  # type: ignore
    from api_handling import (  # type: ignore
        fetch_events_ids,
        fetch_program_ids,
        fetch_program_entries,
        normalize_program_entries,
    )
    from upsert_tables import upsert_program_entries, deactivate_missing_program_entries  # type: ignore


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _date_env(name: str, default: datetime.date) -> str:
    raw = os.getenv(name)
    if raw and raw.strip():
        return raw.strip()
    return default.isoformat()


def _pairs(events: Iterable[int], program_id_lists: Iterable[list[int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for event_id, prog_ids in zip(events, program_id_lists):
        for prog_id in prog_ids or []:
            if prog_id is None:
                continue
            out.append((event_id, prog_id))
    return out


def main():
    load_dotenv(override=True)

    engine = get_engine()
    # Ensure schema exists (safe to run repeatedly)
    initialize_database()

    today = datetime.date.today()
    start_date = _date_env("START_DATE", today)
    end_date = _date_env("END_DATE", today + datetime.timedelta(days=60))

    per_page = _env_int("EVENTS_PER_PAGE", 500)
    program_workers = _env_int("PROGRAM_WORKERS", 50)
    entry_workers = _env_int("ENTRY_WORKERS", 20)

    print(f"Fetching events from {start_date} to {end_date}...")
    event_ids = fetch_events_ids(start_date=start_date, end_date=end_date, per_page=per_page)
    print(f"Found {len(event_ids)} events.")

    if not event_ids:
        print("No events found in this window.")
        return

    print(f"Fetching program IDs with {program_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=program_workers) as ex:
        program_id_lists = list(ex.map(fetch_program_ids, event_ids))

    pairs = _pairs(event_ids, program_id_lists)
    print(f"Found {len(pairs)} (event_id, prog_id) pairs.")

    if not pairs:
        print("No programs found for events in this window.")
        return

    now_utc = pd.Timestamp.now(tz="UTC").to_pydatetime()

    def _fetch_one(pair: tuple[int, int]):
        event_id, prog_id = pair
        try:
            payload = fetch_program_entries(event_id=event_id, prog_id=prog_id, entry_type="start")
            df = normalize_program_entries(payload, entry_type="start")
            return event_id, prog_id, df, None
        except Exception as e:
            return event_id, prog_id, None, str(e)

    print(f"Fetching program entries with {entry_workers} workers...")
    successes = 0
    failures = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=entry_workers) as ex:
        for event_id, prog_id, df, err in ex.map(_fetch_one, pairs):
            if err is not None:
                failures += 1
                print(f"WARN: entries fetch failed for event_id={event_id}, prog_id={prog_id}: {err}")
                continue

            if df is None or df.empty:
                # Still deactivate if the fetch succeeded (empty start list means no entries yet)
                deactivate_missing_program_entries(
                    engine,
                    event_id=event_id,
                    prog_id=prog_id,
                    entry_type="start",
                    active_entry_ids=[],
                    removed_at=now_utc,
                )
                successes += 1
                continue

            # Upsert rows and then deactivate missing entries for this program
            upsert_program_entries(df, engine)
            active_ids = [r[0] for r in df[["entry_id"]].dropna().values.tolist()]
            deactivate_missing_program_entries(
                engine,
                event_id=event_id,
                prog_id=prog_id,
                entry_type="start",
                active_entry_ids=active_ids,
                removed_at=now_utc,
            )
            successes += 1

    print(f"Done. Successful programs: {successes}; failures: {failures}")


if __name__ == "__main__":
    main()
