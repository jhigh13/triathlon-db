import json
import time
import requests
import pandas as pd
from datetime import datetime
try:
    from tri_analysis.config import (
        HEADERS,
        ATHLETE_RESULTS_URL,
        ATHLETE_DATA_URL,
        ATHLETE_SEARCH_URL,
        RANKING_URL,
        EVENT_LISTING_URL,
        PROGRAM_LISTING_URL,
        PROGRAM_RESULTS_URL,
        PROGRAM_DETAILS_URL,
        PROGRAM_ENTRIES_URL,
        BASE_URL,
        SPEC_IDS,
        CATEGORY_IDS,
    )
except ImportError:  # pragma: no cover
    from config import (  # type: ignore
        HEADERS,
        ATHLETE_RESULTS_URL,
        ATHLETE_DATA_URL,
        ATHLETE_SEARCH_URL,
        RANKING_URL,
        EVENT_LISTING_URL,
        PROGRAM_LISTING_URL,
        PROGRAM_RESULTS_URL,
        PROGRAM_DETAILS_URL,
        PROGRAM_ENTRIES_URL,
        BASE_URL,
        SPEC_IDS,
        CATEGORY_IDS,
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
    

def fetch_athlete_country_search(athlete_name: str) -> str:
    """
    Fetch athlete country based on a user-provided name.
    """
    params = {"query": athlete_name}
    response = requests.get(ATHLETE_SEARCH_URL, params=params, headers=HEADERS)
    if response.status_code == 200:
        data = response.json().get("data", [])
        if data:
            return data[0]["athlete_country_name"]
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


def normalize_athlete_mixed_relay_results(
    athlete_id: int,
    athlete_results: list,
    valid_program_pairs: set[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """
    Normalize athlete-history mixed relay rows into the race_results schema.

    The program results endpoint returns team-level rows for Mixed Relay, so this
    fallback uses athlete history to recover leg-level athlete results for those
    relay programs.
    """
    rows = []
    for result in athlete_results or []:
        prog_name = result.get("prog_name") or ""
        event_id = result.get("event_id")
        prog_id = result.get("prog_id")
        if not event_id or not prog_id:
            continue
        if "relay" not in str(prog_name).lower():
            continue
        pair = (int(event_id), int(prog_id))
        if valid_program_pairs is not None and pair not in valid_program_pairs:
            continue

        splits = result.get("splits", []) or []
        pos_raw = result.get("position")
        pos_str = str(pos_raw).strip() if pos_raw is not None else None
        if pos_str and pos_str.isdigit():
            finish_position = pos_str
            finish_status = "FINISH"
            position_sort = pos_str
        else:
            finish_position = None
            finish_status = pos_str.upper() if pos_str else None
            code_map = {
                "DNF": "1000001",
                "DNS": "1000002",
                "DSQ": "1000003",
                "LAP": "1000004",
            }
            position_sort = code_map.get(finish_status, "1000009")

        row = {
            "event_id": pair[0],
            "prog_id": pair[1],
            "athlete_id": athlete_id,
            "athlete_full_name": None,
            "SwimTime": splits[0] if len(splits) > 0 else None,
            "T1Time": splits[1] if len(splits) > 1 else None,
            "BikeTime": splits[2] if len(splits) > 2 else None,
            "T2Time": splits[3] if len(splits) > 3 else None,
            "RunTime": splits[4] if len(splits) > 4 else None,
            "position": pos_raw,
            "finish_status": finish_status,
            "finish_position": finish_position,
            "position_sort": position_sort,
            "total_time": result.get("total_time"),
            "start_num": result.get("team_order") or result.get("start_num"),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def fetch_events_ids(start_date, end_date, per_page=500, spec_ids=SPEC_IDS, category_ids=CATEGORY_IDS) -> list:
    """
    Fetch events from the API since a given date. Returns a list of event ids.
    """
    event_ids = []
    params = {
        "per_page": per_page,
        "start_date": start_date,
        "order": "asc",
        "page": 1,
        "end_date": end_date
    }
    # Only include filters when provided (allows ALL/empty to mean no filter)
    if category_ids and str(category_ids).upper() not in {"ALL", ""}:
        params["category_id"] = category_ids
    if spec_ids and str(spec_ids).upper() not in {"ALL", ""}:
        params["specification_id"] = spec_ids
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

def fetch_program_ids(event_id, relay_only: bool = False) -> list:
    """
    Get program IDs for a given event. Includes:
    - Elite pipeline: Elite Men/Women, U23 Men/Women, Junior Men/Women, Mixed Relay
    - Para pipeline: PTWC, PTS2–PTS5, PTVI classes and related guide/handler programs
    """
    # Existing elite programs
    target_names = {
        "Elite Men", "Elite Women", "U23 Men", "U23 Women",
        "Junior Men", "Junior Women", "Mixed Relay"
    }
    # Para exact program names provided and commonly seen
    para_exact_names = {
        # Men
        "PTWC Men", "Personal Handler PTWC M", "PTS2 Men", "PTS3 Men", "PTS4 Men", "PTS5 Men", "PTVI Men", "Male Guides",
        # Women
        "PTWC Women", "Personal Handler PTWC F", "PTS2 Women", "PTS3 Women", "PTS4 Women", "PTS5 Women", "PTVI Women", "Female Guides",
    }
    # Para class prefixes to be robust to minor naming variations
    para_prefixes = ("PTWC", "PTS2", "PTS3", "PTS4", "PTS5", "PTVI")
    params = {"is_race": "true"}
    last_error = None
    data = None
    for attempt in range(4):
        try:
            resp = requests.get(
                PROGRAM_LISTING_URL.format(event_id=event_id),
                headers=HEADERS,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json().get("data")
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    if last_error and data is None:
        raise last_error
    if not data:
        # data is None or empty, so return an empty list
        return []
    if not isinstance(data, list):
        #print(f"Warning: Unexpected program data type for event_id={event_id}: {type(data)} value: {data}")
        return []
    prog_ids = []
    for p in data:
        if not p:
            continue
        name = p.get("prog_name") or ""
        if relay_only:
            if "relay" in name.lower():
                prog_ids.append(p.get("prog_id"))
            continue
        if (
            name in target_names
            or name in para_exact_names
            or name.startswith(para_prefixes)
        ):
            prog_ids.append(p.get("prog_id"))
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

    # Event info (needed early for fallback logic)
    event = data.get("event") or {}

    # Fallback: infer prog_distance_category from event_specifications if missing/empty
    if not row.get("prog_distance_category"):
        event_specs = event.get("event_specifications") or []
        spec_names = ", ".join([spec.get("cat_name") for spec in event_specs if spec.get("cat_name")])
        if "Sprint" in spec_names:
            row["prog_distance_category"] = "sprint"
        elif "Standard" in spec_names:
            row["prog_distance_category"] = "standard"
        elif "Olympic" in spec_names:
            row["prog_distance_category"] = "olympic"

    # Derive is_para from program name (robust to variations)
    prog_name = (row.get("prog_name") or "").upper()
    row["is_para"] = (
        prog_name.startswith("PTWC")
        or prog_name.startswith("PTS2")
        or prog_name.startswith("PTS3")
        or prog_name.startswith("PTS4")
        or prog_name.startswith("PTS5")
        or prog_name.startswith("PTVI")
        or " PARA" in prog_name
    )

    # prog_distances: extract laps and distance for Swim, Bike, Run
    seg_map = {"Swim": ("Swim_laps", "Swim_distance"),
               "Bike": ("Bike_laps", "Bike_distance"),
               "Run": ("Run_laps", "Run_distance")}
    for seg in ["Swim", "Bike", "Run"]:
        laps_key, dist_key = seg_map[seg]
        laps = None
        dist = None
        # Some events return prog_distances as null; treat as empty list
        for d in (data.get("prog_distances") or []):
            if d.get("segment") == seg:
                laps = d.get("laps")
                dist = d.get("distance")
                break
        row[laps_key] = laps
        row[dist_key] = dist

    # Event info
    row["event_name"] = event.get("event_title")
    row["event_venue"] = event.get("event_venue")
    row["event_date"] = event.get("event_date")
    row["event_country"] = event.get("event_country")
    row["event_latitude"] = event.get("event_latitude")
    row["event_longitude"] = event.get("event_longitude")

    # event_categories: concatenate all cat_name values
    event_categories = event.get("event_categories") or []
    row["cat_name"] = ", ".join([cat.get("cat_name") for cat in event_categories if cat.get("cat_name")])

    

    # meta: add all meta fields except head_referee and competition_jury
    meta = data.get("meta") or {}
    for k, v in meta.items():
        if k not in ("head_referee", "competition_jury"):
            row[k] = v

    return pd.DataFrame([row])

def fetch_and_process_program_results(event_id, program_id, limit=75) -> pd.DataFrame:
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
        # Derive finish fields from the raw position value
        pos_raw = r.get("position")
        pos_str = (str(pos_raw).strip() if pos_raw is not None else None)
        if pos_str and pos_str.isdigit():
            # For finishers, store numeric place as strings to match DB schema
            finish_position = pos_str
            finish_status = "FINISH"
            position_sort = pos_str
        else:
            # For non-finishers, store sentinel sort keys as strings
            finish_position = None
            finish_status = (pos_str.upper() if pos_str else None)
            code_map = {
                "DNF": "1000001",
                "DNS": "1000002",
                "DSQ": "1000003",
                "LAP": "1000004",
            }
            position_sort = code_map.get(finish_status, "1000009")
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
            "finish_status": finish_status,
            "finish_position": finish_position,
            "position_sort": position_sort,
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


def fetch_program_entries(event_id: int, prog_id: int, entry_type: str = "start", limit: int | None = None) -> dict:
    """Fetch program entries for a given (event_id, prog_id).

    This is used for start lists (default) and can support wait lists in the future.

    Notes:
    - Wait lists may return 403 until 30 days pre-event (per API docs).
    """
    params: dict = {"type": entry_type}
    if limit is not None:
        params["limit"] = limit
    url = PROGRAM_ENTRIES_URL.format(event_id=event_id, prog_id=prog_id)
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()


def normalize_program_entries(payload: dict, entry_type: str = "start") -> pd.DataFrame:
    """Normalize the Program Entries API payload into a single DataFrame.

    Produces one row per entry in payload['data']['entries'].
    """
    data = (payload or {}).get("data") or {}
    event_id = data.get("event_id")
    prog_id = data.get("prog_id")
    entries = data.get("entries") or []
    if not isinstance(entries, list) or not entries:
        return pd.DataFrame(columns=[
            "event_id", "prog_id", "entry_id", "entry_type",
            "approved", "start_num", "api_updated_at", "last_fetched_at",
            "is_active", "removed_at",
            "athlete_id", "athlete_full_name", "athlete_first", "athlete_last",
            "athlete_gender", "athlete_country_id", "athlete_country_name",
            "athlete_noc", "athlete_yob", "athlete_slug", "validated",
        ])

    now_utc = pd.Timestamp.now(tz="UTC")
    rows: list[dict] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        api_updated_at = pd.to_datetime(e.get("updated_at"), errors="coerce", utc=True)
        row = {
            "event_id": event_id,
            "prog_id": prog_id,
            "entry_id": e.get("entry_id"),
            "entry_type": entry_type,
            "approved": e.get("approved"),
            "start_num": None if e.get("start_num") is None else str(e.get("start_num")),
            "api_updated_at": None if pd.isna(api_updated_at) else api_updated_at.to_pydatetime(),
            "last_fetched_at": now_utc.to_pydatetime(),
            "is_active": True,
            "removed_at": None,
            "athlete_id": e.get("athlete_id"),
            "athlete_full_name": e.get("athlete_full_name") or e.get("athlete_title"),
            "athlete_first": e.get("athlete_first"),
            "athlete_last": e.get("athlete_last"),
            "athlete_gender": e.get("athlete_gender"),
            "athlete_country_id": e.get("athlete_country_id"),
            "athlete_country_name": e.get("athlete_country_name"),
            "athlete_noc": e.get("athlete_noc"),
            "athlete_yob": e.get("athlete_yob"),
            "athlete_slug": e.get("athlete_slug"),
            "validated": e.get("validated"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure nullable columns don't get pandas NaN
    df = df.where(~df.isna(), None)
    return df


def fetch_athlete_ranking_breakdown(ranking_cat_id: int, athlete_id: int) -> list[dict]:
    """
    Fetch the per-event ranking breakdown for a single athlete in a given category.

    URL: GET /rankings/{ranking_cat_id}/breakdown/{athlete_id}

    Returns a list of dicts, one per scored event, with keys:
        event_id, event_title, event_finish_date, points, period, position
    Period 1 = current window (last 0-1 years), Period 2 = previous window (last 1-2 years).
    Returns [] on error.
    """
    url = f"{BASE_URL}/rankings/{ranking_cat_id}/breakdown/{athlete_id}"
    for attempt in range(4):
        try:
            resp = requests.get(
                url,
                headers=HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            breakdown = data.get("breakdown", [])
            records = []
            for b in breakdown:
                records.append({
                    "event_id":          b.get("event_id"),
                    "event_title":       b.get("event_title"),
                    "event_finish_date": b.get("event_finish_date"),
                    "points":            b.get("points"),
                    "period":            b.get("period"),
                    "position":          b.get("position"),
                    "included":          b.get("include", True),
                })
            return records
        except requests.RequestException:
            if attempt == 3:
                return []
            time.sleep(1.5 * (attempt + 1))
    return []
