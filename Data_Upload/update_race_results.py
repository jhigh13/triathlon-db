import requests
import pandas as pd
from sqlalchemy import create_engine, text
from config.config import HEADERS, BASE_URL, DB_URI
from concurrent.futures import ThreadPoolExecutor, as_completed

# API endpoints
events_url = f"{BASE_URL}/events"
results_url = f"{BASE_URL}/events/{{event_id}}/programs/{{program_id}}/results"
CATEGORY_IDS = "356|357|351"


def get_last_event_date(engine):
    with engine.connect() as conn:
        result = conn.execute(text('SELECT MAX("EventDate") FROM events'))
        max_date = result.scalar()
        if max_date is None:
            return "2025-05-10"
        if isinstance(max_date, str):
            return max_date
        return max_date.strftime("%Y-%m-%d")


def fetch_new_events(since_date, per_page=100, category_ids=CATEGORY_IDS):
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
    resp = requests.get(f"{BASE_URL}/events/{event_id}", headers=HEADERS)
    resp.raise_for_status()
    progs = resp.json().get("data", {}).get("programs") or []
    return [(p.get("prog_id"), p.get("prog_name")) for p in progs if p.get("prog_name") in ("Elite Men", "Elite Women")]


def fetch_race_results(event_id, program_id, limit=25):
    url = results_url.format(event_id=event_id, program_id=program_id)
    resp = requests.get(url, headers=HEADERS, params={"limit": limit})
    resp.raise_for_status()
    data = resp.json().get("data") or {}
    return data.get("results") or []


def process_race_results(results, event, program_id, program_name):
    if not results:
        return None
    df = pd.json_normalize(results)
    df["athlete_id"] = df["athlete_id"]
    df["EventID"] = event.get("event_id")
    df["ProgID"] = program_id
    df["Program"] = program_name
    df["CategoryName"] = event.get("CategoryName")
    df["EventSpecifications"] = event.get("EventSpecifications")

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
    insert_sql = """
        INSERT INTO events ("EventID","EventName","EventDate","Venue","Country","CategoryName","EventSpecifications")
        VALUES (:EventID,:EventName,:EventDate,:Venue,:Country,:CategoryName,:EventSpecifications)
        ON CONFLICT ("EventID") DO UPDATE SET
            "EventName" = EXCLUDED."EventName",
            "EventDate" = EXCLUDED."EventDate",
            "Venue" = EXCLUDED."Venue",
            "Country" = EXCLUDED."Country",
            "CategoryName" = EXCLUDED."CategoryName",
            "EventSpecifications" = EXCLUDED."EventSpecifications";
    """
    with engine.begin() as conn:
        conn.execute(text(insert_sql), records)


def upsert_race_results(df, engine):
    if df is None or df.empty:
        return
    records = df.to_dict(orient="records")
    insert_sql = """
        INSERT INTO race_results (
            athlete_id, "EventID", "ProgID", "Program", "CategoryName", "EventSpecifications",
            "Position", "TotalTime", "SwimTime", "T1", "BikeTime", "T2", "RunTime"
        ) VALUES (
            :athlete_id, :EventID, :ProgID, :Program, :CategoryName, :EventSpecifications,
            :Position, :TotalTime, :SwimTime, :T1, :BikeTime, :T2, :RunTime
        )
    """
    """
        ON CONFLICT (athlete_id, "EventID") DO UPDATE SET
            "Position" = EXCLUDED."Position",
            "TotalTime" = EXCLUDED."TotalTime",
            "SwimTime" = EXCLUDED."SwimTime",
            "T1" = EXCLUDED."T1",
            "BikeTime" = EXCLUDED."BikeTime",
            "T2" = EXCLUDED."T2",
            "RunTime" = EXCLUDED."RunTime",
            "ProgName" = EXCLUDED."Program",
            "ProgID" = EXCLUDED."ProgID",
            "CategoryName" = EXCLUDED."CategoryName",
            "EventSpecifications" = EXCLUDED."EventSpecifications";
    """
    with engine.begin() as conn:
        conn.execute(text(insert_sql), records)


def update_race_results(engine, max_workers=10):
    from datetime import datetime, timedelta
    last_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    evs = fetch_new_events(last_date)

    # Parallelize program discovery
    elite_evs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(get_elite_programs, ev.get("event_id")): ev for ev in evs}
        for future in as_completed(future_map):
            ev = future_map[future]
            try:
                progs = future.result()
                if progs:
                    ev['elite_programs'] = progs
                    elite_evs.append(ev)
            except Exception as e:
                print(f"Error fetching programs for event {ev.get('event_id')}: {e}")

    if not elite_evs:
        print("No new events with elite programs to upsert.")
    else:
        upsert_events(elite_evs, engine)

    # Parallelize race results retrieval
    all_dfs = []
    def fetch_and_process(ev, pid, pname):
        results = fetch_race_results(ev.get("event_id"), pid)
        return process_race_results(results, ev, pid, pname)

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ev in elite_evs:
            for pid, pname in ev['elite_programs']:
                futures.append(executor.submit(fetch_and_process, ev, pid, pname))
        for future in as_completed(futures):
            try:
                df = future.result()
                if df is not None:
                    all_dfs.append(df)
            except Exception as e:
                print(f"Error fetching results: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        upsert_race_results(combined, engine)
        print(f"Upserted {len(combined)} race results.")
    else:
        print("No race results to upsert.")


if __name__ == "__main__":
    engine = create_engine(DB_URI)
    update_race_results(engine)
