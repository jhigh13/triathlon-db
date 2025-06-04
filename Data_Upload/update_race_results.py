import requests
import pandas as pd
from sqlalchemy import create_engine, text
from config.config import HEADERS, BASE_URL, DB_URI
from config.config import ATHLETE_TABLE_NAME, EVENTS_TABLE_NAME, RACE_RESULTS_TABLE_NAME
from Data_Import.database import initialize_database
from concurrent.futures import ThreadPoolExecutor, as_completed

# API endpoints
events_url = f"{BASE_URL}/events"
results_url = f"{BASE_URL}/events/{{event_id}}/programs/{{program_id}}/results"
CATEGORY_IDS = "356|357|351"


def get_last_event_date(engine):
    """Get the date 30 days ago to fetch recent events."""
    from datetime import datetime, timedelta
    thirty_days_ago = datetime.now() - timedelta(days=30)
    return thirty_days_ago.strftime("%Y-%m-%d")


def fetch_new_events(since_date, per_page=100, category_ids=CATEGORY_IDS):
    """Fetch events from the API since a given date."""
    events = []
    params = {"per_page": per_page, "start_date": since_date, "order": "asc", "page": 1, "category_ids": category_ids}

    while True:
        resp = requests.get(events_url, headers=HEADERS, params=params)
        resp.raise_for_status()
        payload = resp.json().get("data") or []
        if not payload:
            break
        for ev in payload:
            cats = ev.get("event_categories") or []
            ev["CategoryName"] = ", ".join(c.get("cat_name", "") for c in cats) if cats else None
            specs = ev.get("event_specifications") or []
            ev["EventSpecifications"] = ", ".join(s.get("cat_name", "") for s in specs) if specs else None
        events.extend(payload)
        if not resp.json().get("next_page_url"):
            break
        params["page"] += 1
    return events


def get_elite_programs(event_id):
    """Get Elite Men and Elite Women program IDs for a given event."""
    resp = requests.get(f"{BASE_URL}/events/{event_id}", headers=HEADERS)
    resp.raise_for_status()
    progs = resp.json().get("data", {}).get("programs") or []
    return [(p.get("prog_id"), p.get("prog_name")) for p in progs if p.get("prog_name") in ("Elite Men", "Elite Women")]


def fetch_race_results(event_id, program_id, limit=50):
    """Fetch race results for a specific event and program."""
    url = results_url.format(event_id=event_id, program_id=program_id)
    resp = requests.get(url, headers=HEADERS, params={"limit": limit})
    resp.raise_for_status()
    data = resp.json().get("data") or {}
    return data.get("results") or []


def process_race_results(results, event, program_id, program_name):
    """Process raw race results into a DataFrame with proper column structure."""
    if not results:
        return pd.DataFrame()
    
    df = pd.json_normalize(results)
    df["athlete_id"] = df["athlete_id"]
    df["EventID"] = event.get("event_id")
    df["ProgID"] = program_id
    df["Program"] = program_name
    df["CategoryName"] = event.get("CategoryName")
    df["EventSpecifications"] = event.get("EventSpecifications")

    # Extract split times from the splits array
    df["SwimTime"] = df.get("splits").apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    df["T1"] = df.get("splits").apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None)
    df["BikeTime"] = df.get("splits").apply(lambda x: x[2] if isinstance(x, list) and len(x) > 2 else None)
    df["T2"] = df.get("splits").apply(lambda x: x[3] if isinstance(x, list) and len(x) > 3 else None)
    df["RunTime"] = df.get("splits").apply(lambda x: x[4] if isinstance(x, list) and len(x) > 4 else None)

    return df[[
        "athlete_id", "EventID", "ProgID", "Program", "CategoryName", "EventSpecifications",  
        "position", "total_time", "SwimTime", "T1", "BikeTime", "T2", "RunTime"
    ]].rename(columns={"position": "Position", "total_time": "TotalTime"})


def calculate_position_metrics(df):
    """Calculate position rankings and position changes for race results."""
    if df.empty:
        return df
    
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
    df['ElapsedSwim'] = df['SwimTime'].apply(parse_time_to_secs)
    df['ElapsedT1'] = df['ElapsedSwim'] + df['T1'].apply(parse_time_to_secs)
    df['ElapsedBike'] = df['ElapsedT1'] + df['BikeTime'].apply(parse_time_to_secs)
    df['ElapsedT2'] = df['ElapsedBike'] + df['T2'].apply(parse_time_to_secs)
    df['ElapsedRun'] = df['ElapsedT2'] + df['RunTime'].apply(parse_time_to_secs)

    # Ensure all elapsed columns are int64 and no NaN/inf
    for col in ['ElapsedSwim','ElapsedT1','ElapsedBike','ElapsedT2','ElapsedRun']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

    # Parse individual split seconds for filtering
    df['SwimSecs'] = df['SwimTime'].apply(parse_time_to_secs)
    df['T1Secs'] = df['T1'].apply(parse_time_to_secs)
    df['BikeSecs'] = df['BikeTime'].apply(parse_time_to_secs)
    df['T2Secs'] = df['T2'].apply(parse_time_to_secs)
    df['RunSecs'] = df['RunTime'].apply(parse_time_to_secs)

    # Compute seconds behind leader per EventID + Program
    # Swim
    min_swim = df[df['SwimSecs'] > 0].groupby(['EventID','Program'])['ElapsedSwim'].transform('min').fillna(0)
    df['BehindSwim'] = df['ElapsedSwim'] - min_swim
    # T1
    min_t1 = df[df['T1Secs'] > 0].groupby(['EventID','Program'])['ElapsedT1'].transform('min').fillna(0)
    df['BehindT1'] = df['ElapsedT1'] - min_t1
    # Bike
    min_bike = df[df['BikeSecs'] > 0].groupby(['EventID','Program'])['ElapsedBike'].transform('min').fillna(0)
    df['BehindBike'] = df['ElapsedBike'] - min_bike
    # T2
    min_t2 = df[df['T2Secs'] > 0].groupby(['EventID','Program'])['ElapsedT2'].transform('min').fillna(0)
    df['BehindT2'] = df['ElapsedT2'] - min_t2
    # Run
    min_run = df[df['RunSecs'] > 0].groupby(['EventID','Program'])['ElapsedRun'].transform('min').fillna(0)
    df['BehindRun'] = df['ElapsedRun'] - min_run

    # Calculate positions at each checkpoint (only for athletes with non-zero times)
    # Position at Swim
    mask_swim = df['SwimSecs'] > 0
    df.loc[mask_swim, 'Position_at_Swim'] = df.loc[mask_swim].groupby(['EventID', 'Program'])['ElapsedSwim'].rank(method='min')
    
    # Position at T1
    mask_t1 = df['T1Secs'] > 0
    df.loc[mask_t1, 'Position_at_T1'] = df.loc[mask_t1].groupby(['EventID', 'Program'])['ElapsedT1'].rank(method='min')
    
    # Position at Bike
    mask_bike = df['BikeSecs'] > 0
    df.loc[mask_bike, 'Position_at_Bike'] = df.loc[mask_bike].groupby(['EventID', 'Program'])['ElapsedBike'].rank(method='min')
    
    # Position at T2
    mask_t2 = df['T2Secs'] > 0
    df.loc[mask_t2, 'Position_at_T2'] = df.loc[mask_t2].groupby(['EventID', 'Program'])['ElapsedT2'].rank(method='min')
    
    # Position at Run/Finish
    mask_run = df['RunSecs'] > 0
    df.loc[mask_run, 'Position_at_Run'] = df.loc[mask_run].groupby(['EventID', 'Program'])['ElapsedRun'].rank(method='min')
    
    # Calculate position changes between checkpoints (negative = gained positions)
    df['Swim_to_T1_pos_change'] = df['Position_at_T1'] - df['Position_at_Swim']
    df['T1_to_Bike_pos_change'] = df['Position_at_Bike'] - df['Position_at_T1']
    df['Bike_to_T2_pos_change'] = df['Position_at_T2'] - df['Position_at_Bike']
    df['T2_to_Run_pos_change'] = df['Position_at_Run'] - df['Position_at_T2']
    
    # Convert position columns to nullable integers (Int64) to handle NaN values properly
    position_cols = ['Position_at_Swim', 'Position_at_T1', 'Position_at_Bike', 'Position_at_T2', 'Position_at_Run',
                     'Swim_to_T1_pos_change', 'T1_to_Bike_pos_change', 'Bike_to_T2_pos_change', 'T2_to_Run_pos_change']
    
    for col in position_cols:
        df[col] = df[col].astype('Int64')  # Nullable integer type

    # Ensure behind columns are int64
    for col in ['BehindSwim','BehindT1','BehindBike','BehindT2','BehindRun']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
    
    # Drop temporary split-second columns
    df.drop(columns=['SwimSecs','T1Secs','BikeSecs','T2Secs','RunSecs'], inplace=True)
    
    return df


def upsert_events(events, engine):
    """Upsert events into the events table."""
    if not events:
        return
    
    df = pd.DataFrame(events)
    df = df.rename(columns={
        "event_id": "EventID", "event_title": "EventName", "event_date": "EventDate",
        "event_venue": "Venue", "event_country": "Country"
    })
    cols = ["EventID","EventName","EventDate","Venue","Country","CategoryName","EventSpecifications"]
    df = df[cols].drop_duplicates(subset=["EventID"])
    if df.empty:
        return
    
    records = df.to_dict(orient="records")
    with engine.begin() as conn:
        for record in records:
            conn.execute(text(f"""
                INSERT INTO "{EVENTS_TABLE_NAME}" ("EventID", "EventName", "EventDate", "Venue", "Country", "CategoryName", "EventSpecifications")
                VALUES (:EventID, :EventName, :EventDate, :Venue, :Country, :CategoryName, :EventSpecifications)
                ON CONFLICT ("EventID") DO UPDATE SET
                    "EventName" = EXCLUDED."EventName",
                    "EventDate" = EXCLUDED."EventDate",
                    "Venue" = EXCLUDED."Venue",
                    "Country" = EXCLUDED."Country",
                    "CategoryName" = EXCLUDED."CategoryName",
                    "EventSpecifications" = EXCLUDED."EventSpecifications"
            """), record)
    print(f"Upserted {len(records)} events.")


def upsert_race_results(df, engine):
    """Upsert race results into the race_results table."""
    if df.empty:
        return
    
    # Filter out rows with NULL values in key columns to prevent constraint violations
    df = df.dropna(subset=["athlete_id", "EventID", "TotalTime"])
    
    if df.empty:
        print("No valid race results to upsert (all had NULL key values).")
        return
    
    # Remove duplicates based on unique constraint
    before = len(df)
    df = df.drop_duplicates(subset=["athlete_id", "EventID", "TotalTime"])
    after = len(df)
    if before != after:
        print(f"Removed {before - after} duplicate rows from race_results (kept {after}).")
    
    records = df.to_dict(orient="records")
    with engine.begin() as conn:
        for record in records:
            conn.execute(text(f"""
                INSERT INTO "{RACE_RESULTS_TABLE_NAME}" (
                    athlete_id, "EventID", "ProgID", "Program", "CategoryName", "EventSpecifications",
                    "Position", "TotalTime", "SwimTime", "T1", "BikeTime", "T2", "RunTime",
                    "ElapsedSwim", "ElapsedT1", "ElapsedBike", "ElapsedT2", "ElapsedRun",
                    "BehindSwim", "BehindT1", "BehindBike", "BehindT2", "BehindRun",
                    "Position_at_Swim", "Position_at_T1", "Position_at_Bike", "Position_at_T2", "Position_at_Run",
                    "Swim_to_T1_pos_change", "T1_to_Bike_pos_change", "Bike_to_T2_pos_change", "T2_to_Run_pos_change"
                )
                VALUES (
                    :athlete_id, :EventID, :ProgID, :Program, :CategoryName, :EventSpecifications,
                    :Position, :TotalTime, :SwimTime, :T1, :BikeTime, :T2, :RunTime,
                    :ElapsedSwim, :ElapsedT1, :ElapsedBike, :ElapsedT2, :ElapsedRun,
                    :BehindSwim, :BehindT1, :BehindBike, :BehindT2, :BehindRun,
                    :Position_at_Swim, :Position_at_T1, :Position_at_Bike, :Position_at_T2, :Position_at_Run,
                    :Swim_to_T1_pos_change, :T1_to_Bike_pos_change, :Bike_to_T2_pos_change, :T2_to_Run_pos_change
                )
                ON CONFLICT (athlete_id, "EventID", "TotalTime") DO UPDATE SET
                    "ProgID" = EXCLUDED."ProgID",
                    "Program" = EXCLUDED."Program",
                    "CategoryName" = EXCLUDED."CategoryName",
                    "EventSpecifications" = EXCLUDED."EventSpecifications",
                    "Position" = EXCLUDED."Position",
                    "SwimTime" = EXCLUDED."SwimTime",
                    "T1" = EXCLUDED."T1",
                    "BikeTime" = EXCLUDED."BikeTime",
                    "T2" = EXCLUDED."T2",
                    "RunTime" = EXCLUDED."RunTime",
                    "ElapsedSwim" = EXCLUDED."ElapsedSwim",
                    "ElapsedT1" = EXCLUDED."ElapsedT1",
                    "ElapsedBike" = EXCLUDED."ElapsedBike",
                    "ElapsedT2" = EXCLUDED."ElapsedT2",
                    "ElapsedRun" = EXCLUDED."ElapsedRun",
                    "BehindSwim" = EXCLUDED."BehindSwim",
                    "BehindT1" = EXCLUDED."BehindT1",
                    "BehindBike" = EXCLUDED."BehindBike",
                    "BehindT2" = EXCLUDED."BehindT2",
                    "BehindRun" = EXCLUDED."BehindRun",
                    "Position_at_Swim" = EXCLUDED."Position_at_Swim",
                    "Position_at_T1" = EXCLUDED."Position_at_T1",
                    "Position_at_Bike" = EXCLUDED."Position_at_Bike",
                    "Position_at_T2" = EXCLUDED."Position_at_T2",
                    "Position_at_Run" = EXCLUDED."Position_at_Run",
                    "Swim_to_T1_pos_change" = EXCLUDED."Swim_to_T1_pos_change",
                    "T1_to_Bike_pos_change" = EXCLUDED."T1_to_Bike_pos_change",
                    "Bike_to_T2_pos_change" = EXCLUDED."Bike_to_T2_pos_change",
                    "T2_to_Run_pos_change" = EXCLUDED."T2_to_Run_pos_change"
            """), record)
    print(f"Upserted {len(records)} race results.")


def update_race_results(engine, max_workers=10):
    """Main function to update race results with new events."""
    since_date = get_last_event_date(engine)
    print(f"Fetching events since {since_date}...")
    
    events = fetch_new_events(since_date)
    if not events:
        print("No new events found.")
        return
    
    print(f"Found {len(events)} new events.")
    upsert_events(events, engine)
    
    all_results_dfs = []
    
    def process_event_program(event, program_id, program_name):
        try:
            results = fetch_race_results(event.get("event_id"), program_id)
            if results:
                df = process_race_results(results, event, program_id, program_name)
                return df
        except Exception as e:
            print(f"Error processing event {event.get('event_id')} program {program_id}: {e}")
        return pd.DataFrame()
    
    # Process events concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for event in events:
            elite_programs = get_elite_programs(event.get("event_id"))
            for program_id, program_name in elite_programs:
                future = executor.submit(process_event_program, event, program_id, program_name)
                futures.append(future)
        
        for future in as_completed(futures):
            df = future.result()
            if not df.empty:
                all_results_dfs.append(df)
    
    if all_results_dfs:
        combined_df = pd.concat(all_results_dfs, ignore_index=True)
        print(f"Processing {len(combined_df)} race results...")
        
        # Calculate position metrics
        combined_df = calculate_position_metrics(combined_df)
        
        # Upsert to database
        upsert_race_results(combined_df, engine)
        print("Race results update completed.")
    else:
        print("No race results found for the new events.")


if __name__ == "__main__":
    engine = create_engine(DB_URI)
    update_race_results(engine)