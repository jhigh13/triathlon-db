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

CATEGORY_IDS = "340|341|342|623|343|352|347|640|624|351|348|349" #Add Para afterwards
SPEC_IDS = "356|357"


def get_last_event_date(engine):
    """Get the date 30 days ago to fetch recent events."""
    from datetime import datetime, timedelta
    thirty_days_ago = datetime.now() - timedelta(days=30)
    return thirty_days_ago.strftime("%Y-%m-%d")


 def fetch_new_events(since_date, per_page=500, spec_ids=SPEC_IDS):
    """Fetch events from the API since a given date."""
    events = []
    params = {"per_page": per_page, "start_date": since_date, "order": "asc", "page": 1, "category_id": category_ids, "specification_id": spec_ids}

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
    
    # Filter out rows with NULL values in key columns to prevent constraint violations; remove duplicates
    df = df.dropna(subset=["athlete_id", "EventID", "TotalTime"])
    df = df.drop_duplicates(subset=["athlete_id", "EventID", "TotalTime"])
    
    records = df.to_dict(orient="records")
    with engine.begin() as conn:
        for record in records:
            conn.execute(text(f"""                INSERT INTO "{RACE_RESULTS_TABLE_NAME}" (
                    athlete_id, "EventID", "ProgID", "Program", "CategoryName", "EventSpecifications",
                    "Position", "TotalTime", "SwimTime", "T1", "BikeTime", "T2", "RunTime"
                )
                VALUES (
                    :athlete_id, :EventID, :ProgID, :Program, :CategoryName, :EventSpecifications,
                    :Position, :TotalTime, :SwimTime, :T1, :BikeTime, :T2, :RunTime
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
                    "RunTime" = EXCLUDED."RunTime"
            """), record)
    print(f"Upserted {len(records)} race results.")

def update_race_results(engine, max_workers=200):
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