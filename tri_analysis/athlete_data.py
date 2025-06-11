# Data_Import/athlete_data.py
import json
import requests
import pandas as pd
from config.config import HEADERS, ATHLETE_RESULTS_URL, ATHLETE_DATA_URL

def get_athlete_info_df(athlete_id: int) -> pd.DataFrame:
    """
    Retrieve basic athlete details from the API.

    Parameters:
        athlete_id (int): The athlete's ID.

    Return a single-row DataFrame of athlete basic information.
    """
    url = ATHLETE_DATA_URL.format(athlete_id=athlete_id)
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json().get("data", {})

    # Parse categories JSON if provided
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
        # boolean flags from categories
        "category_to": categories.get("to", False),
        "category_coach": categories.get("coach", False),
        "category_athlete": categories.get("athlete", False),
        "category_medical": categories.get("medical", False),
        "category_paratriathlete": categories.get("paratriathlete", False),
    }
    return pd.DataFrame([info]) if info else pd.DataFrame()


def fetch_race_results(athlete_id: int) -> list:
    """
    Retrieve raw race results JSON for an athlete, handling pagination.

    Parameters:
        athlete_id (int): The athlete's ID.

    Returns:
        list: List of race event dictionaries.
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

def get_athlete_results_df(athlete_id: int) -> pd.DataFrame:
    """
    Return a DataFrame of structured race results for an athlete.
    """
    raw_events = fetch_race_results(athlete_id)
    records = []

    for event in raw_events:
        prog_name = event.get("prog_name")
        prog_id = event.get("prog_id")

        splits = event.get("splits", [])
        category_names = event.get("event_categories") or []
        spec_names = event.get("event_specifications") or []

        records.append({
            "athlete_id": athlete_id,
            "EventID": event.get("event_id"),
            "ProgID": prog_id,
            "Program": prog_name,
            "EventName": event.get("event_title"),
            "EventDate": event.get("event_date"),
            "Venue": event.get("event_venue"),
            "Country": event.get("event_country"),
            "CategoryName": ", ".join([c.get("cat_name") for c in category_names]) or None,
            "EventSpecifications": ", ".join([s.get("cat_name") for s in spec_names]) or None,
            "Position": event.get("position"),
            "TotalTime": event.get("total_time"),
            "SwimTime": splits[0] if len(splits) > 0 else None,
            "T1": splits[1] if len(splits) > 1 else None,
            "BikeTime": splits[2] if len(splits) > 2 else None,
            "T2": splits[3] if len(splits) > 3 else None,
            "RunTime": splits[4] if len(splits) > 4 else None,
        })

    if records: 
        df = pd.DataFrame(records)
        df["TotalTime"] = df["TotalTime"].fillna("NA")
    else:
        df = pd.DataFrame()

    return df