# --- Rankings import: fetch World Triathlon rankings and upsert to DB ---
import time

import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

try:
    from tri_analysis.config import HEADERS, BASE_URL
    from tri_analysis.database import get_engine
except ImportError:  # pragma: no cover
    from config import HEADERS, BASE_URL  # type: ignore
    from database import get_engine  # type: ignore

load_dotenv(override=True)

# Ranking category IDs to import
# 13-16: Current World Triathlon global/series rankings used in the prediction stack.
# 35-44: Continental Points Lists (Europe, Americas, Africa, Asia, Oceania x Men/Women)
RANKING_CATEGORIES = {
    # Olympic qualification window (LA 2028 cycle)
    11: "Olympic Qualification - Male",
    12: "Olympic Qualification - Female",
    # Global
    13: "World Rankings - Male",
    14: "World Rankings - Female",
    15: "World Triathlon Series - Male",
    16: "World Triathlon Series - Female",
    # Continental Points Lists
    35: "Continental Points List - Europe Elite Men",
    36: "Continental Points List - Europe Elite Women",
    37: "Continental Points List - Americas Elite Men",
    38: "Continental Points List - Americas Elite Women",
    39: "Continental Points List - Africa Elite Men",
    40: "Continental Points List - Africa Elite Women",
    41: "Continental Points List - Asia Elite Men",
    42: "Continental Points List - Asia Elite Women",
    43: "Continental Points List - Oceania Elite Men",
    44: "Continental Points List - Oceania Elite Women",
}


def fetch_rankings(ranking_cat_id: int, limit: int = 500) -> pd.DataFrame:
    """
    Pulls the ranking snapshot for a given category, returns a normalized DataFrame.
    """
    url = f"{BASE_URL}/rankings/{ranking_cat_id}"
    last_error = None
    js = None
    for attempt in range(4):
        try:
            resp = requests.get(url, headers=HEADERS, params={'limit': limit}, timeout=30)
            resp.raise_for_status()
            js = resp.json()['data']
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    if last_error and js is None:
        raise last_error

    published_date = pd.to_datetime(js['published']).date()
    ranking_year = published_date.year

    # build records list
    records = []
    for r in js.get('rankings', []):
        records.append({
            'athlete_id':     r['athlete_id'],
            'athlete_name':   r['athlete_full_name'],
            'ranking_cat_name': RANKING_CATEGORIES.get(ranking_cat_id, js['ranking_cat_name']),
            'ranking_cat_id': js['ranking_id'],
            'rank_position':  r['rank'],
            'total_points':   r['total'],
            'year':           ranking_year,
            'retrieved_at':   published_date,
            'events_current_period':  r.get('events_current_period'),
            'events_previous_period': r.get('events_previous_period'),
        })
    return pd.DataFrame(records)


def upsert_rankings(df: pd.DataFrame, engine):
    """
    Batch upsert into athlete_rankings with ON CONFLICT on (athlete_name, ranking_cat_name, year, retrieved_at).
    """
    if df.empty:
        return

    insert_sql = text(
        """
        INSERT INTO athlete_rankings (
            athlete_id, athlete_name, ranking_cat_name, ranking_cat_id, rank_position, total_points, year, retrieved_at,
            events_current_period, events_previous_period
        ) VALUES (
            :athlete_id, :athlete_name, :ranking_cat_name, :ranking_cat_id, :rank_position, :total_points, :year, :retrieved_at,
            :events_current_period, :events_previous_period
        )
        ON CONFLICT (athlete_name, ranking_cat_name, year, retrieved_at) DO UPDATE SET
            rank_position           = EXCLUDED.rank_position,
            total_points            = EXCLUDED.total_points,
            events_current_period   = EXCLUDED.events_current_period,
            events_previous_period  = EXCLUDED.events_previous_period;
        """
    )
    with engine.begin() as conn:
        conn.execute(insert_sql, df.to_dict(orient='records'))


def import_rankings(category_ids: list[int] | None = None, limit: int = 500):
    """
    Fetch and persist rankings for each desired ranking category.

    Args:
        category_ids: Specific category IDs to import. Defaults to all in RANKING_CATEGORIES.
        limit: Max athletes per category (default 500).
    """
    engine = get_engine()
    ids_to_fetch = category_ids or list(RANKING_CATEGORIES.keys())

    all_dfs = []
    for cat_id in ids_to_fetch:
        label = RANKING_CATEGORIES.get(cat_id, f"Category {cat_id}")
        try:
            df = fetch_rankings(cat_id, limit=limit)
            if not df.empty:
                year = df["year"].iloc[0]
                print(f"  [{cat_id}] {label}: {len(df)} athletes (year={year})")
                all_dfs.append(df)
            else:
                print(f"  [{cat_id}] {label}: empty response")
        except Exception as e:
            print(f"  [{cat_id}] {label}: ERROR - {e}")

    if not all_dfs:
        print("No ranking data fetched.")
        return

    full = pd.concat(all_dfs, ignore_index=True)

    # Keep all rankings — athletes will have both world and continental entries.
    # The SQL layer pivots these into separate columns (wt_rank_position vs
    # wt_continental_rank) so both signals are available to the model.
    n_athletes = full["athlete_id"].nunique()

    upsert_rankings(full, engine)
    print(f"Imported {len(full)} ranking records ({n_athletes} unique athletes) across {len(ids_to_fetch)} categories.")


if __name__ == '__main__':
    import_rankings()
