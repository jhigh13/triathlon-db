# Data_Import/master_data_import.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
import concurrent.futures
from Data_Import.database import get_engine, initialize_database
from Data_Import.athlete_ids import get_top_athlete_ids
from Data_Import.athlete_data import get_athlete_results_df, get_athlete_info_df


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

    print("Saving tables to database...")

    with engine.begin() as conn:
        conn.exec_driver_sql("TRUNCATE TABLE athlete, events, race_results")


    athletes_df.to_sql("athlete", engine, if_exists="append", index=False)
    events_df.to_sql("events", engine, if_exists="append", index=False)
    fact_df.to_sql("race_results", engine, if_exists="append", index=False)

    print("Master data import completed.")


if __name__ == "__main__":
    import_master_data()
