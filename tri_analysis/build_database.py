#build_database.py
import sys
import os
import concurrent.futures
import datetime
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
try:
    from tri_analysis.database import get_engine, initialize_database
except ImportError:  # pragma: no cover
    from database import get_engine, initialize_database  # type: ignore

try:
    from tri_analysis.config import (
        ATHLETE_TABLE_NAME,
        RACE_RESULTS_TABLE_NAME,
        EVENTS_TABLE_NAME,
        RANKINGS_RESULTS_TABLE_NAME,
        METRICS_TABLE_NAME,
    )
except ImportError:  # pragma: no cover
    from config import (  # type: ignore
        ATHLETE_TABLE_NAME,
        RACE_RESULTS_TABLE_NAME,
        EVENTS_TABLE_NAME,
        RANKINGS_RESULTS_TABLE_NAME,
        METRICS_TABLE_NAME,
    )
from tri_analysis.api_handling import (
    fetch_athlete_id_search,
    fetch_athlete_id_ranking,
    fetch_athlete_info,
    fetch_race_results,
    fetch_events_ids,
    fetch_program_ids,
    process_program_data,
    fetch_and_process_program_results,
)
from tri_analysis.upsert_tables import upsert_athlete, upsert_events, upsert_race_results
load_dotenv(override=True)

def is_valid_df(df):
    return isinstance(df, pd.DataFrame) and not df.empty and not df.isna().all(axis=None)

def process_pair(pair):
    event_id, program_id = pair
    try:
        event_data = process_program_data(event_id, program_id)
    except Exception as e:
        print(f"Skipping event_id={event_id}, program_id={program_id} in process_program_data due to error: {e}")
        event_data = None
    try:
        result_data = fetch_and_process_program_results(event_id, program_id)
    except Exception as e:
        print(f"Skipping event_id={event_id}, program_id={program_id} in fetch_and_process_program_results due to error: {e}")
        result_data = None
    return event_data, result_data

def fetch_and_validate_athlete_info(athlete_id):
    try:
        df = fetch_athlete_info(athlete_id)
        return df if is_valid_df(df) else None
    except Exception as e:
        print(f"Skipping athlete_id={athlete_id} due to error: {e}")
        return None

def main(): 
# Get database engine, drop existing tables, and initialize the database
    engine = get_engine()
    '''with engine.begin() as conn:
        # Recreate all core tables to guarantee correct column types and avoid legacy narrow types
        #conn.execute(text(f'DROP TABLE IF EXISTS "{RACE_RESULTS_TABLE_NAME}" CASCADE'))
        #conn.execute(text(f'DROP TABLE IF EXISTS "{ATHLETE_TABLE_NAME}" CASCADE'))
        #conn.execute(text(f'DROP TABLE IF EXISTS "{EVENTS_TABLE_NAME}" CASCADE'))
        #conn.execute(text(f'DROP TABLE IF EXISTS "{RANKINGS_RESULTS_TABLE_NAME}" CASCADE'))
        #conn.execute(text(f'DROP TABLE IF EXISTS "{METRICS_TABLE_NAME}" CASCADE'))
        print("Dropped existing tables")'''
    initialize_database()

    # Allow overriding the date window via environment variables for backfills (e.g., para-only last 4 years)
    start_date = os.environ.get("START_DATE") or datetime.date(2026, 1, 1).strftime("%Y-%m-%d")
    end_date = os.environ.get("END_DATE") or datetime.date(2026, 12, 31).strftime("%Y-%m-%d")

    # Initialize as an empty list to collect DataFrames from each process_program_data call
    event_df = []
    race_results_df = []
    athletes_df = []


    print("Fetching event IDs...")    # Fetch event IDs since the specified start date
    events_id = fetch_events_ids(start_date=start_date, end_date=end_date)  
    print(f"Found {len(events_id)} events from {start_date} to {end_date}.")

    print("Fetching program IDs for each event concurrently...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
        program_id_lists = list(executor.map(fetch_program_ids, events_id))

    # Flatten the list of lists and keep track of (event_id, program_id) pairs
    event_program_pairs = [
        (event_id, prog_id)
        for event_id, prog_ids in zip(events_id, program_id_lists)
        for prog_id in prog_ids
    ]

    print("Processing program data and race results concurrently...") # Process each event
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for event_data, result_data in executor.map(process_pair, event_program_pairs):
            if is_valid_df(event_data):
                event_df.append(event_data)
            if is_valid_df(result_data):
                race_results_df.append(result_data)

    event_df = pd.concat(event_df, ignore_index=True) if event_df else pd.DataFrame()
    race_results_df = pd.concat(race_results_df, ignore_index=True) if race_results_df else pd.DataFrame()
    print(f"Processed {len(event_df)} programs and {len(race_results_df)} race results.")

    race_results_df = race_results_df.drop_duplicates(subset=["athlete_id", "prog_id", "total_time"]).copy()


    print("Fetching athlete information concurrently...")
    unique_athlete_ids = race_results_df["athlete_id"].dropna().unique().tolist()
    athletes_df_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        for df in executor.map(fetch_and_validate_athlete_info, unique_athlete_ids):
            if df is not None:
                athletes_df_list.append(df)
    athletes_df = pd.concat(athletes_df_list, ignore_index=True) if athletes_df_list else pd.DataFrame()

    # Remove rows with null athlete_id before writing to race_results
    race_results_df = race_results_df.dropna(subset=["athlete_id"])

    # Remove rows with null total_time before writing to race_results
    before = len(race_results_df)
    race_results_df = race_results_df.dropna(subset=["total_time"])
    after = len(race_results_df)
    if before != after:
        print(f"Warning: Dropped {before - after} rows from race_results_df due to null total_time.")

    # Write DataFrames to database
    print("Writing DataFrames to database...")

    # Write athletes_df
    if not athletes_df.empty:
        upsert_athlete(athletes_df, engine)
        print(f"Wrote {len(athletes_df)} rows to {ATHLETE_TABLE_NAME}")

    # Write event_df
    if not event_df.empty:
        event_df.columns = [c.lower() for c in event_df.columns]
        # Normalize: numeric columns should not receive empty strings; convert blanks/NaN to None
        # Treat lap columns as floats (to allow NaN) along with other float columns
        float_cols = [
            "swim_laps", "bike_laps", "run_laps",
            "swim_distance", "bike_distance", "run_distance",
            "event_latitude", "event_longitude"
        ]

        def _clean_int(v):
            if v is None:
                return None
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            if isinstance(v, str):
                s = v.strip()
                if s == "":
                    return None
                if s.isdigit():
                    return int(s)
                # non-numeric string -> treat as NULL
                return None
            if isinstance(v, float):
                try:
                    if pd.isna(v):
                        return None
                except Exception:
                    pass
                return int(v)
            if isinstance(v, int):
                return v
            return None

        def _clean_float(v):
            if v is None:
                return None
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            if isinstance(v, str):
                s = v.strip()
                if s == "":
                    return None
                try:
                    return float(s)
                except Exception:
                    return None
            if isinstance(v, (int, float)):
                try:
                    f = float(v)
                    if pd.isna(f):
                        return None
                    return f
                except Exception:
                    return None
            return None

        for col in float_cols:
            if col in event_df.columns:
                event_df[col] = event_df[col].apply(_clean_float)

        # Weather metrics now stored as strings; coerce to string or None
        weather_cols = ["temperature_water", "temperature_air", "humidity", "wbgt", "wind"]
        for col in weather_cols:
            if col in event_df.columns:
                event_df[col] = event_df[col].apply(lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
        upsert_events(event_df, engine)
        print(f"Wrote {len(event_df)} rows to {EVENTS_TABLE_NAME}")
        event_df['event_id'].drop_duplicates().to_csv('latest_events.txt', index=False, header=False)

    # Write race_results_df
    if not race_results_df.empty:
        race_results_df.columns = [c.lower() for c in race_results_df.columns]
        # Normalize NaN to None for nullable columns to satisfy SQL types
        nullable_cols = [
            "finish_position", "position_sort", "start_num", "position",
            "swimtime", "t1time", "biketime", "t2time", "runtime", "total_time",
        ]
        for col in nullable_cols:
            if col in race_results_df.columns:
                race_results_df[col] = race_results_df[col].where(~race_results_df[col].isna(), None)

        # Ensure finish_position and position_sort are strings (or None) to match DB schema
        def _to_str_or_none(x):
            if x is None:
                return None
            try:
                if pd.isna(x):
                    return None
            except Exception:
                pass
            return str(x)
        for col in ["finish_position", "position_sort"]:
            if col in race_results_df.columns:
                race_results_df[col] = race_results_df[col].apply(_to_str_or_none)

        # start_num can be mixed; keep as string to match DB schema and avoid int overflows
        if "start_num" in race_results_df.columns:
            race_results_df["start_num"] = race_results_df["start_num"].apply(lambda x: None if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
        upsert_race_results(race_results_df, engine)
        print(f"Wrote {len(race_results_df)} rows to {RACE_RESULTS_TABLE_NAME}")

if __name__ == "__main__":
    main()
    print("Database build complete.")