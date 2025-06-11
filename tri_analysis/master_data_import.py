# Data_Import/master_data_import.py
import sys
import os
import pandas as pd
import concurrent.futures
from sqlalchemy import text
from Data_Import.database import get_engine, initialize_database
from config.config import ATHLETE_TABLE_NAME, EVENTS_TABLE_NAME, RACE_RESULTS_TABLE_NAME
from Data_Import.athlete_ids import get_top_athlete_ids
from Data_Import.athlete_data import get_athlete_results_df, get_athlete_info_df

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def fetch_concurrent(athlete_ids, func, max_workers=100):
    """
    Concurrently fetch DataFrames using `func` for each athlete ID.

    Returns a list of non-empty DataFrames.
    """
    dfs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {executor.submit(func, aid): aid for aid in athlete_ids}
        for future in concurrent.futures.as_completed(future_to_id):
            aid = future_to_id[future]
            try:
                df = future.result()
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                print(f"Error fetching data for athlete {aid}: {e}")
    return dfs


def build_event_and_fact_tables(df):
    """
    Split combined race results into event dimension and race_results fact tables.
    """
    event_cols = ["EventID", "EventName", "EventDate", "Venue", "Country", "CategoryName", "EventSpecifications"]
    events_df = df[event_cols].drop_duplicates().reset_index(drop=True)
    fact_df = df.drop(columns=["EventName", "EventDate", "Venue", "Country",])
    return events_df, fact_df

def import_master_data():
    """
    Single-step process to import athletes, events, and race_results tables.
    """
    engine = get_engine()
    
    # Clean up any existing tables and constraints first
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{RACE_RESULTS_TABLE_NAME}" CASCADE'))
        conn.execute(text(f'DROP TABLE IF EXISTS "{ATHLETE_TABLE_NAME}" CASCADE'))
        conn.execute(text(f'DROP TABLE IF EXISTS "{EVENTS_TABLE_NAME}" CASCADE'))
        conn.execute(text('DROP TABLE IF EXISTS athlete_rankings CASCADE'))
        print("Dropped existing tables")
    
    initialize_database()
    athlete_ids = get_top_athlete_ids()

    print("Fetching athlete profiles...")
    athlete_dfs = fetch_concurrent(athlete_ids, get_athlete_info_df)
    athletes_df = pd.concat(athlete_dfs, ignore_index=True) if athlete_dfs else pd.DataFrame()

    print("Fetching race results...")
    result_dfs = fetch_concurrent(athlete_ids, get_athlete_results_df)
    race_results_df = pd.concat(result_dfs, ignore_index=True) if result_dfs else pd.DataFrame()

    if athletes_df.empty or race_results_df.empty:
        print("No data available to import.")
        return

    print("Building event and fact tables...")
    events_df, fact_df = build_event_and_fact_tables(race_results_df)

    # Remove duplicates from fact_df based on unique constraint
    fact_df = fact_df.drop_duplicates(subset=["athlete_id", "EventID", "TotalTime"]).copy()

    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"TRUNCATE TABLE \"{ATHLETE_TABLE_NAME}\", \"{EVENTS_TABLE_NAME}\", \"{RACE_RESULTS_TABLE_NAME}\""
        )
    athletes_df.to_sql(ATHLETE_TABLE_NAME, engine, if_exists="append", index=False)
    events_df.to_sql(EVENTS_TABLE_NAME, engine, if_exists="append", index=False)
    fact_df.to_sql(RACE_RESULTS_TABLE_NAME, engine, if_exists="append", index=False)

    print("Master data import completed.")

if __name__ == "__main__":
    import_master_data()