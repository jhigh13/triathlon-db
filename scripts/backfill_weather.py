#!/usr/bin/env python
"""
Backfill missing weather data for events using the Open-Meteo historical API.

Usage:
    python scripts/backfill_weather.py                          # backfill all missing
    python scripts/backfill_weather.py --start_date 2020-01-01  # from 2020 onwards
    python scripts/backfill_weather.py --overwrite              # re-fetch everything
    python scripts/backfill_weather.py --dry_run                # preview counts only

Events must have latitude/longitude to fetch weather. Events with existing
weather_source='api' (from World Triathlon) are preserved unless --overwrite.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(override=True)

from tri_analysis.database import get_engine, initialize_database
from tri_analysis.weather import backfill_event_weather, backfill_missing_coordinates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill missing weather data using Open-Meteo API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Only process events on or after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="Only process events on or before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-fetch weather even for events with existing data",
    )
    parser.add_argument(
        "--geocode",
        action="store_true",
        help="Geocode events with venue name but no coordinates before backfilling",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview how many events would be processed (no API calls)",
    )

    args = parser.parse_args()

    engine = get_engine()

    # Ensure weather columns exist (runs ALTER TABLE ADD COLUMN IF NOT EXISTS)
    logger.info("Ensuring database schema is up to date...")
    initialize_database()

    # Step 1: Geocode events with venue but no coordinates
    if args.geocode:
        print(f"\n{'='*60}")
        print("GEOCODING EVENTS WITH VENUE BUT NO COORDINATES")
        print(f"{'='*60}")
        geo_stats = backfill_missing_coordinates(engine, dry_run=args.dry_run)
        print(f"  Events with venue, no coords: {geo_stats['total']}")
        print(f"  Successfully geocoded:        {geo_stats['geocoded']}")
        print(f"  Failed:                       {geo_stats['failed']}")
        print(f"{'='*60}\n")

    # Show current weather data coverage before backfill
    from sqlalchemy import text
    with engine.connect() as conn:
        total_events = conn.execute(text("SELECT COUNT(*) FROM events")).scalar()
        with_coords = conn.execute(text(
            "SELECT COUNT(*) FROM events WHERE event_latitude IS NOT NULL AND event_longitude IS NOT NULL"
        )).scalar()
        with_temp = conn.execute(text(
            "SELECT COUNT(*) FROM events WHERE temperature_air IS NOT NULL AND temperature_air != ''"
        )).scalar()
        with_source = conn.execute(text(
            "SELECT weather_source, COUNT(*) FROM events GROUP BY weather_source ORDER BY COUNT(*) DESC"
        )).fetchall()

    print(f"\n{'='*60}")
    print("WEATHER DATA COVERAGE (before backfill)")
    print(f"{'='*60}")
    print(f"  Total events:              {total_events}")
    print(f"  With coordinates:          {with_coords} ({with_coords/total_events*100:.1f}%)")
    print(f"  With temperature_air:      {with_temp} ({with_temp/total_events*100:.1f}%)")
    print(f"  Missing temperature_air:   {total_events - with_temp}")
    print(f"\n  Weather source breakdown:")
    for source, count in with_source:
        print(f"    {source or '(none)':20s}: {count}")
    print(f"{'='*60}\n")

    # Run backfill
    stats = backfill_event_weather(
        engine,
        start_date=args.start_date,
        end_date=args.end_date,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("BACKFILL RESULTS")
    print(f"{'='*60}")
    print(f"  Events to process:   {stats['total']}")
    print(f"  Successfully fetched: {stats['fetched']}")
    print(f"  Skipped (has API):    {stats['skipped']}")
    print(f"  Failed:               {stats['failed']}")
    print(f"  Missing coordinates:  {stats['no_coords']}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("(Dry run — no changes made. Remove --dry_run to execute.)")


if __name__ == "__main__":
    main()
