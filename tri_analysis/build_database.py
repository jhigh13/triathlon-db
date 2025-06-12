#build_database.py
import sys
import os
import concurrent.futures
import datetime
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from database import get_engine, initialize_database
from config import ATHLETE_TABLE_NAME, RACE_RESULTS_TABLE_NAME, EVENTS_TABLE_NAME, RANKINGS_RESULTS_TABLE_NAME, METRICS_TABLE_NAME
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
load_dotenv()

def is_valid_df(df):
    return isinstance(df, pd.DataFrame) and not df.empty and not df.isna().all(axis=None)

def process_pair(pair):
    event_id, program_id = pair
    event_data = process_program_data(event_id, program_id)
    result_data = fetch_and_process_program_results(event_id, program_id)
    return event_data, result_data

def fetch_and_validate_athlete_info(athlete_id):
    try:
        df = fetch_athlete_info(athlete_id)
        return df if is_valid_df(df) else None
    except Exception as e:
        print(f"Skipping athlete_id={athlete_id} due to error: {e}")
        return None


# Get database engine, drop existing tables, and initialize the database
engine = get_engine()
with engine.begin() as conn:
    conn.execute(text(f'DROP TABLE IF EXISTS "{RACE_RESULTS_TABLE_NAME}" CASCADE'))
    conn.execute(text(f'DROP TABLE IF EXISTS "{ATHLETE_TABLE_NAME}" CASCADE'))
    conn.execute(text(f'DROP TABLE IF EXISTS "{EVENTS_TABLE_NAME}" CASCADE'))
    conn.execute(text(f'DROP TABLE IF EXISTS "{RANKINGS_RESULTS_TABLE_NAME}" CASCADE'))
    conn.execute(text(f'DROP TABLE IF EXISTS "{METRICS_TABLE_NAME}" CASCADE'))
    print("Dropped existing tables")
initialize_database()

    
start_date = datetime.date(2012, 1, 1).strftime("%Y-%m-%d")    

# Initialize as an empty list to collect DataFrames from each process_program_data call
event_df = []
race_results_df = []
athletes_df = []


print("Fetching event IDs...")    # Fetch event IDs since the specified start date
events_id = fetch_events_ids(start_date=start_date)  
print(f"Found {len(events_id)} events since {start_date}.")

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
with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
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
    athletes_df.to_sql(ATHLETE_TABLE_NAME, engine, if_exists='append', index=False, method='multi')
    print(f"Wrote {len(athletes_df)} rows to {ATHLETE_TABLE_NAME}")

# Write event_df
if not event_df.empty:
    event_df.to_sql(EVENTS_TABLE_NAME, engine, if_exists='append', index=False, method='multi')
    print(f"Wrote {len(event_df)} rows to {EVENTS_TABLE_NAME}")

# Write race_results_df
if not race_results_df.empty:
    race_results_df.to_sql(RACE_RESULTS_TABLE_NAME, engine, if_exists='append', index=False, method='multi')
    print(f"Wrote {len(race_results_df)} rows to {RACE_RESULTS_TABLE_NAME}")