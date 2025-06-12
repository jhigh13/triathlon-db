import json
import requests
import pandas as pd
from config import (
    HEADERS, ATHLETE_RESULTS_URL, ATHLETE_DATA_URL, ATHLETE_SEARCH_URL, RANKING_URL, EVENT_LISTING_URL,
    PROGRAM_LISTING_URL, PROGRAM_RESULTS_URL, PROGRAM_DETAILS_URL, BASE_URL, SPEC_IDS, CATEGORY_IDS
)

def fetch_athlete_id_search(athlete_name: str) -> int:
    """
    Fetch athlete ID based on a user-provided name.
    """
    params = {"query": athlete_name}
    response = requests.get(ATHLETE_SEARCH_URL, params=params, headers=HEADERS)
    if response.status_code == 200:
        data = response.json().get("data", [])
        if data:
            return data[0]["athlete_id"]
        else:
            raise ValueError(f"Athlete '{athlete_name}' not found.")
    else:
        raise Exception(f"Search API failed with status code {response.status_code}")

def fetch_athlete_id_ranking(ranking_id: int) -> list:
    """
    Fetch athlete IDs from a ranking category ID. Return a list of athlete IDs.
    """
    response = requests.get(RANKING_URL.format(ranking_id=ranking_id), headers=HEADERS)
    response.raise_for_status()
    rankings = response.json().get("data", {}).get("rankings", [])
    return [entry["athlete_id"] for entry in rankings]

def fetch_athlete_info(athlete_id: int) -> pd.DataFrame:
    """
    Fetch basic athlete details from the API, given an athlete ID. Return a DataFrame with information.
    """
    url = ATHLETE_DATA_URL.format(athlete_id=athlete_id)
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json().get("data", {})

    categories_raw = data.get("categories", "{}")
    try:
        categories = json.loads(categories_raw)
    except json.JSONDecodeError:
        categories = {}

    info = {
        "athlete_id": data.get("athlete_id"),
        "full_name": data.get("athlete_full_name"),
        "gender": data.get("athlete_gender"),
        "country": data.get("athlete_country_name"),
        "age": data.get("athlete_age"),
        "category_to": categories.get("to", False),
        "category_coach": categories.get("coach", False),
        "category_athlete": categories.get("athlete", False),
        "category_medical": categories.get("medical", False),
        "category_paratriathlete": categories.get("paratriathlete", False),
    }
    return pd.DataFrame([info]) if info else pd.DataFrame()

def fetch_race_results(athlete_id: int) -> list:
    """
    Fetch all race results for an athleteID, handling pagination. Return list of event dictionaries to process.
    """
    url = ATHLETE_RESULTS_URL.format(athlete_id=athlete_id)
    results = []
    while url:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        page = response.json()
        results.extend(page.get("data", []))
        url = page.get("next_page_url")
    return results

def fetch_events_ids(start_date, per_page=500, spec_ids=SPEC_IDS, category_ids=CATEGORY_IDS) -> list:
    """
    Fetch events from the API since a given date. Returns a list of event ids.
    """
    event_ids = []
    params = {
        "per_page": per_page,
        "start_date": start_date,
        "order": "asc",
        "page": 1,
        "category_id": category_ids,
        "specification_id": spec_ids
    }
    while True:
        resp = requests.get(EVENT_LISTING_URL, headers=HEADERS, params=params)
        resp.raise_for_status()
        payload = resp.json().get("data") or []
        if not payload:
            break
        for ev in payload:
            event_ids.append(ev["event_id"])
        if not resp.json().get("next_page_url"):
            break
        params["page"] += 1
    return event_ids

def fetch_program_ids(event_id) -> list:
    """
    Get program IDs for a given event for the following categories:
    Elite Men, Elite Women, U23 Men, U23 Women, Junior Men, Junior Women, Mixed Relay.
    """
    target_names = {
        "Elite Men", "Elite Women", "U23 Men", "U23 Women",
        "Junior Men", "Junior Women", "Mixed Relay"
    }
    params = {"is_race": "true"}
    resp = requests.get(PROGRAM_LISTING_URL.format(event_id=event_id), headers=HEADERS, params=params)
    resp.raise_for_status()
    data = resp.json().get("data")
    if not data:
        # data is None or empty, so return an empty list
        return []
    if not isinstance(data, list):
        #print(f"Warning: Unexpected program data type for event_id={event_id}: {type(data)} value: {data}")
        return []
    prog_ids = [p.get("prog_id") for p in data if p and p.get("prog_name") in target_names]
    return prog_ids if prog_ids else []

def process_program_data(event_id, program_id) -> pd.DataFrame:
    """
    Fetch program details for a specific event and program ID.
    Returns a DataFrame with selected program, event, and meta details in a single row.
    """
    url = PROGRAM_DETAILS_URL.format(event_id=event_id, program_id=program_id)
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json().get("data", {})

    # Basic program fields
    row = {
        "prog_id": data.get("prog_id"),
        "event_id": data.get("event_id"),
        "prog_name": data.get("prog_name"),
        "prog_distance_category": data.get("prog_distance_category"),
    }

    # prog_distances: extract laps and distance for Swim, Bike, Run
    seg_map = {"Swim": ("Swim_laps", "Swim_distance"),
               "Bike": ("Bike_laps", "Bike_distance"),
               "Run": ("Run_laps", "Run_distance")}
    for seg in ["Swim", "Bike", "Run"]:
        laps_key, dist_key = seg_map[seg]
        laps = None
        dist = None
        for d in data.get("prog_distances", []):
            if d.get("segment") == seg:
                laps = d.get("laps")
                dist = d.get("distance")
                break
        row[laps_key] = laps
        row[dist_key] = dist

    # Event info
    event = data.get("event", {})
    row["event_name"] = event.get("event_title")
    row["event_venue"] = event.get("event_venue")
    row["event_date"] = event.get("event_date")
    row["event_country"] = event.get("event_country")
    row["event_latitude"] = event.get("event_latitude")
    row["event_longitude"] = event.get("event_longitude")

    # event_categories: concatenate all cat_name values
    event_categories = event.get("event_categories", [])
    row["cat_name"] = ", ".join([cat.get("cat_name") for cat in event_categories if cat.get("cat_name")])

    # meta: add all meta fields except head_referee and competition_jury
    meta = data.get("meta", {})
    for k, v in meta.items():
        if k not in ("head_referee", "competition_jury"):
            row[k] = v

    return pd.DataFrame([row])

def fetch_and_process_program_results(event_id, program_id, limit=50) -> pd.DataFrame:
    """
    Given an event ID and program ID, fetch and process race results for a specific event and program.
    Returns a DataFrame with unique rows for each athlete_id, including split times and key result fields.
    Stores all splits as available, filling missing with None.
    """
    url = PROGRAM_RESULTS_URL.format(event_id=event_id, program_id=program_id)
    resp = requests.get(url, headers=HEADERS, params={"limit": limit})
    resp.raise_for_status()
    data = resp.json().get("data", {})
    results = data.get("results")
    if not isinstance(results, list):
        # Debug print to help you see the unexpected value
        print(f"Warning: Unexpected results type for event_id={event_id}, program_id={program_id}: {type(results)} value: {results}")
        results = []

    rows = []
    for r in results:
        splits = r.get("splits", [])
        row = {
            "event_id": event_id,
            "prog_id": program_id,
            "athlete_id": r.get("athlete_id"),
            "athlete_full_name": r.get("athlete_full_name"),
            "SwimTime": splits[0] if len(splits) > 0 else None,
            "T1Time": splits[1] if len(splits) > 1 else None,
            "BikeTime": splits[2] if len(splits) > 2 else None,
            "T2Time": splits[3] if len(splits) > 3 else None,
            "RunTime": splits[4] if len(splits) > 4 else None,
            "position": r.get("position"),
            "total_time": r.get("total_time"),
            "start_num": r.get("start_num"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["athlete_id"])
    return df

def fetch_rankings(ranking_cat_id: int, limit: int = 200) -> pd.DataFrame:
    """
    Pulls the ranking snapshot for a given category, returns a normalized DataFrame.
    """
    url = f"{BASE_URL}/rankings/{ranking_cat_id}"
    resp = requests.get(url, headers=HEADERS, params={'limit': limit})
    resp.raise_for_status()
    js = resp.json()['data']

    # build records list
    records = []
    for r in js.get('rankings', []):
        records.append({
            'athlete_id':     r['athlete_id'],
            'athlete_name':   r['athlete_full_name'],
            'ranking_cat_name': js['ranking_cat_name'],
            'ranking_cat_id': js['ranking_id'],
            'rank_position':  r['rank'],
            'total_points':   r['total'],
            'year':           2025,  # Assuming all rankings are for the year 2025
            'retrieved_at':   pd.to_datetime(js['published']).date()
        })
    return pd.DataFrame(records)

