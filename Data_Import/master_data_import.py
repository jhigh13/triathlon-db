# Data_Import/master_data_import.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
import concurrent.futures
from sqlalchemy import text
from Data_Import.database import get_engine, initialize_database
from config.config import ATHLETE_TABLE_NAME, EVENTS_TABLE_NAME, RACE_RESULTS_TABLE_NAME
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
    before = len(fact_df)
    fact_df = fact_df.drop_duplicates(subset=["athlete_id", "EventID", "TotalTime"]).copy()
    after = len(fact_df)
    print(f"Removed {before - after} duplicate rows from race_results (kept {after}).")

    # Helper to parse time strings into seconds
    def parse_time_to_secs(t):
        if pd.isna(t) or t == "":
            return 0
        parts = str(t).split(":")
        try:
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + int(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + int(s)
        except ValueError:
            return 0
        return 0

    # Calculate elapsed times at each checkpoint (cumulative)
    fact_df['ElapsedSwim'] = fact_df['SwimTime'].apply(parse_time_to_secs)
    fact_df['ElapsedT1'] = fact_df['ElapsedSwim'] + fact_df['T1'].apply(parse_time_to_secs)
    fact_df['ElapsedBike'] = fact_df['ElapsedT1'] + fact_df['BikeTime'].apply(parse_time_to_secs)
    fact_df['ElapsedT2'] = fact_df['ElapsedBike'] + fact_df['T2'].apply(parse_time_to_secs)
    fact_df['ElapsedRun'] = fact_df['ElapsedT2'] + fact_df['RunTime'].apply(parse_time_to_secs)    # Ensure all elapsed/behind columns are int64 and no NaN/inf
    for col in ['ElapsedSwim','ElapsedT1','ElapsedBike','ElapsedT2','ElapsedRun',
                'BehindSwim','BehindT1','BehindBike','BehindT2','BehindRun']:
        if col in fact_df:
            fact_df[col] = pd.to_numeric(fact_df[col], errors='coerce').fillna(0).astype('int64')

    # Ensure all elapsed/behind columns are int64 and no NaN/inf
    for col in ['ElapsedSwim','ElapsedT1','ElapsedBike','ElapsedT2','ElapsedRun',
                'BehindSwim','BehindT1','BehindBike','BehindT2','BehindRun']:
        if col in fact_df:
            fact_df[col] = pd.to_numeric(fact_df[col], errors='coerce').fillna(0).astype('int64')

    # Parse individual split seconds
    fact_df['SwimSecs'] = fact_df['SwimTime'].apply(parse_time_to_secs)
    fact_df['T1Secs']   = fact_df['T1'].apply(parse_time_to_secs)
    fact_df['BikeSecs'] = fact_df['BikeTime'].apply(parse_time_to_secs)
    fact_df['T2Secs']   = fact_df['T2'].apply(parse_time_to_secs)
    fact_df['RunSecs']  = fact_df['RunTime'].apply(parse_time_to_secs)

    # Compute seconds behind leader per EventID + Program, filtering only athletes who recorded that split
    # Swim
    min_swim = fact_df[fact_df['SwimSecs'] > 0].groupby(['EventID','Program'])['ElapsedSwim'] \
                  .transform('min').fillna(0)
    fact_df['BehindSwim'] = fact_df['ElapsedSwim'] - min_swim
    # T1
    min_t1 = fact_df[fact_df['T1Secs'] > 0].groupby(['EventID','Program'])['ElapsedT1'] \
                .transform('min').fillna(0)
    fact_df['BehindT1'] = fact_df['ElapsedT1'] - min_t1
    # Bike
    min_bike = fact_df[fact_df['BikeSecs'] > 0].groupby(['EventID','Program'])['ElapsedBike'] \
                  .transform('min').fillna(0)
    fact_df['BehindBike'] = fact_df['ElapsedBike'] - min_bike
    # T2
    min_t2 = fact_df[fact_df['T2Secs'] > 0].groupby(['EventID','Program'])['ElapsedT2'] \
                .transform('min').fillna(0)
    fact_df['BehindT2'] = fact_df['ElapsedT2'] - min_t2
    # Run
    min_run = fact_df[fact_df['RunSecs'] > 0].groupby(['EventID','Program'])['ElapsedRun'] \
                 .transform('min').fillna(0)
    fact_df['BehindRun'] = fact_df['ElapsedRun'] - min_run
    
    # Calculate positions at each checkpoint (only for athletes with non-zero times)
    print("Calculating position rankings at each checkpoint...")
    
    # Position at Swim (rank by ElapsedSwim, only athletes with SwimSecs > 0)
    mask_swim = fact_df['SwimSecs'] > 0
    fact_df.loc[mask_swim, 'Position_at_Swim'] = fact_df.loc[mask_swim].groupby(['EventID', 'Program'])['ElapsedSwim'].rank(method='min')
    
    # Position at T1 (rank by ElapsedT1, only athletes with T1Secs > 0)
    mask_t1 = fact_df['T1Secs'] > 0
    fact_df.loc[mask_t1, 'Position_at_T1'] = fact_df.loc[mask_t1].groupby(['EventID', 'Program'])['ElapsedT1'].rank(method='min')
    
    # Position at Bike (rank by ElapsedBike, only athletes with BikeSecs > 0)
    mask_bike = fact_df['BikeSecs'] > 0
    fact_df.loc[mask_bike, 'Position_at_Bike'] = fact_df.loc[mask_bike].groupby(['EventID', 'Program'])['ElapsedBike'].rank(method='min')
    
    # Position at T2 (rank by ElapsedT2, only athletes with T2Secs > 0)
    mask_t2 = fact_df['T2Secs'] > 0
    fact_df.loc[mask_t2, 'Position_at_T2'] = fact_df.loc[mask_t2].groupby(['EventID', 'Program'])['ElapsedT2'].rank(method='min')
    
    # Position at Run/Finish (rank by ElapsedRun, only athletes with RunSecs > 0)
    mask_run = fact_df['RunSecs'] > 0
    fact_df.loc[mask_run, 'Position_at_Run'] = fact_df.loc[mask_run].groupby(['EventID', 'Program'])['ElapsedRun'].rank(method='min')
    
    # Calculate position changes between checkpoints (negative = gained positions)
    print("Calculating position changes between checkpoints...")
    
    # Swim to T1 position change
    fact_df['Swim_to_T1_pos_change'] = fact_df['Position_at_T1'] - fact_df['Position_at_Swim']
    
    # T1 to Bike position change
    fact_df['T1_to_Bike_pos_change'] = fact_df['Position_at_Bike'] - fact_df['Position_at_T1']
    
    # Bike to T2 position change
    fact_df['Bike_to_T2_pos_change'] = fact_df['Position_at_T2'] - fact_df['Position_at_Bike']
    
    # T2 to Run position change
    fact_df['T2_to_Run_pos_change'] = fact_df['Position_at_Run'] - fact_df['Position_at_T2']
    
    # Convert position columns to nullable integers (Int64) to handle NaN values properly
    position_cols = ['Position_at_Swim', 'Position_at_T1', 'Position_at_Bike', 'Position_at_T2', 'Position_at_Run',
                     'Swim_to_T1_pos_change', 'T1_to_Bike_pos_change', 'Bike_to_T2_pos_change', 'T2_to_Run_pos_change']
    
    for col in position_cols:
        fact_df[col] = fact_df[col].astype('Int64')  # Nullable integer type
    
    # Drop temporary split-second columns
    fact_df.drop(columns=['SwimSecs','T1Secs','BikeSecs','T2Secs','RunSecs'], inplace=True)

    print("Saving tables to database...")

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