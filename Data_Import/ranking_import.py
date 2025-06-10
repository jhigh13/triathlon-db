# --- New script: Data_Import/rankings_import.py ---

from dotenv import load_dotenv
load_dotenv()
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from config.config import HEADERS, BASE_URL, DB_URI

# helper to get engine
from Data_Import.database import get_engine

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


def upsert_rankings(df: pd.DataFrame, engine):
    """
    Batch upsert into athlete_rankings with ON CONFLICT on (athlete_name, ranking_cat_name, year, retrieved_at).
    """
    if df.empty:
        return

    insert_sql = text(
        """
        INSERT INTO athlete_rankings (
            athlete_id, athlete_name, ranking_cat_name, ranking_cat_id, rank_position, total_points, year, retrieved_at
        ) VALUES (
            :athlete_id, :athlete_name, :ranking_cat_name, :ranking_cat_id, :rank_position, :total_points, :year, :retrieved_at
        )
        ON CONFLICT (athlete_name, ranking_cat_name, year, retrieved_at) DO UPDATE SET
            rank_position = EXCLUDED.rank_position,
            total_points  = EXCLUDED.total_points;
        """
    )
    with engine.begin() as conn:
        conn.execute(insert_sql, df.to_dict(orient='records'))


def import_rankings():
    """
    Fetch and persist rankings for each desired ranking category.
    """
    engine = get_engine()
    # list of ranking_cat_ids to import (e.g. 13 for Elite Men, 14 for Elite Women)
    ranking_ids = [13, 14, 15, 16]
    all_dfs = []
    for cat in ranking_ids:
        df = fetch_rankings(cat)
        all_dfs.append(df)

    if not all_dfs:
        print("No ranking data fetched.")
        return
    full = pd.concat(all_dfs, ignore_index=True)
    upsert_rankings(full, engine)
    print(f"Imported {len(full)} ranking records.")


if __name__ == '__main__':
    import_rankings()
