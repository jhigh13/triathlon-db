"""Sync a date window from local Postgres to Supabase using upserts.

Usage (PowerShell):
  c:/Users/johnk/VSCode/triathlon-db/.venv/Scripts/python.exe scripts/sync_local_to_supabase.py --start-date 2026-01-01 --end-date 2026-04-20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tri_analysis.upsert_tables import upsert_dataframe


def _get_target_columns(engine, table_name: str) -> list[str]:
    query = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = :table_name
        ORDER BY ordinal_position
        """
    )
    with engine.connect() as conn:
        cols = [row[0] for row in conn.execute(query, {"table_name": table_name}).fetchall()]
    if not cols:
        raise RuntimeError(f"No columns found for table '{table_name}' on target DB.")
    return cols


def _load_events(source_engine, start_date: str, end_date: str) -> pd.DataFrame:
    query = text(
        """
        SELECT *
        FROM events
        WHERE event_date BETWEEN :start_date AND :end_date
        """
    )
    return pd.read_sql(query, source_engine, params={"start_date": start_date, "end_date": end_date})


def _load_race_results(source_engine, start_date: str, end_date: str) -> pd.DataFrame:
    query = text(
        """
        SELECT rr.*
        FROM race_results rr
        INNER JOIN events e
          ON e.event_id = rr.event_id
         AND e.prog_id = rr.prog_id
        WHERE e.event_date BETWEEN :start_date AND :end_date
        """
    )
    return pd.read_sql(query, source_engine, params={"start_date": start_date, "end_date": end_date})


def _load_athletes(source_engine, start_date: str, end_date: str) -> pd.DataFrame:
    query = text(
        """
        SELECT a.*
        FROM athlete a
        WHERE a.athlete_id IN (
            SELECT DISTINCT rr.athlete_id
            FROM race_results rr
            INNER JOIN events e
              ON e.event_id = rr.event_id
             AND e.prog_id = rr.prog_id
            WHERE e.event_date BETWEEN :start_date AND :end_date
              AND rr.athlete_id IS NOT NULL
        )
        """
    )
    return pd.read_sql(query, source_engine, params={"start_date": start_date, "end_date": end_date})


def _filter_to_target_columns(df: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    cols = [c for c in target_columns if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=target_columns)
    return df[cols].copy()


def _upsert_dynamic(
    df: pd.DataFrame,
    table_name: str,
    conflict_cols: list[str],
    target_columns: list[str],
    target_engine,
) -> None:
    if df.empty:
        return
    common_cols = [c for c in target_columns if c in df.columns]
    update_cols = [c for c in common_cols if c not in conflict_cols]
    if not update_cols:
        raise RuntimeError(f"No update columns available for table '{table_name}'.")
    upsert_dataframe(
        df[common_cols].copy(),
        table_name,
        conflict_cols,
        update_cols,
        target_engine,
    )


def main() -> int:
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Sync local triathlon DB date window to Supabase")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--source-uri",
        default=os.getenv("DB_URI"),
        help="Source DB URI (defaults to DB_URI)",
    )
    parser.add_argument(
        "--target-uri",
        default=os.getenv("TRIATHLON_DATABASE_URL"),
        help="Target DB URI (defaults to TRIATHLON_DATABASE_URL)",
    )
    args = parser.parse_args()

    if not args.source_uri:
        raise RuntimeError("Missing source URI. Set DB_URI or pass --source-uri.")
    if not args.target_uri:
        raise RuntimeError("Missing target URI. Set TRIATHLON_DATABASE_URL or pass --target-uri.")

    source_engine = create_engine(args.source_uri)
    target_engine = create_engine(args.target_uri)

    print(f"Loading local rows from {args.start_date} to {args.end_date}...")
    events_df = _load_events(source_engine, args.start_date, args.end_date)
    race_results_df = _load_race_results(source_engine, args.start_date, args.end_date)
    athlete_df = _load_athletes(source_engine, args.start_date, args.end_date)

    print(f"Local extract -> events: {len(events_df)}, race_results: {len(race_results_df)}, athlete: {len(athlete_df)}")

    if events_df.empty:
        print("No events in range; nothing to sync.")
        return 0

    if "total_time" in race_results_df.columns:
        before = len(race_results_df)
        race_results_df = race_results_df.dropna(subset=["total_time"])
        dropped = before - len(race_results_df)
        if dropped:
            print(f"Dropped {dropped} race_results rows with null total_time before upsert.")

    events_cols = _get_target_columns(target_engine, "events")
    race_results_cols = _get_target_columns(target_engine, "race_results")
    athlete_cols = _get_target_columns(target_engine, "athlete")

    events_df = _filter_to_target_columns(events_df, events_cols)
    race_results_df = _filter_to_target_columns(race_results_df, race_results_cols)
    athlete_df = _filter_to_target_columns(athlete_df, athlete_cols)

    print("Upserting to Supabase...")
    _upsert_dynamic(
        athlete_df,
        "athlete",
        ["athlete_id"],
        athlete_cols,
        target_engine,
    )
    _upsert_dynamic(
        events_df,
        "events",
        ["event_id", "prog_id"],
        events_cols,
        target_engine,
    )
    _upsert_dynamic(
        race_results_df,
        "race_results",
        ["athlete_id", "prog_id", "total_time"],
        race_results_cols,
        target_engine,
    )

    verify_query = text(
        """
        SELECT COUNT(*)
        FROM events
        WHERE event_date BETWEEN :start_date AND :end_date
        """
    )
    with target_engine.connect() as conn:
        target_event_count = conn.execute(
            verify_query,
            {"start_date": args.start_date, "end_date": args.end_date},
        ).scalar()

    print("Sync complete.")
    print(f"Target verification -> events in range: {target_event_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
