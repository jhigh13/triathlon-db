# Data_Import/athlete_ids.py
import requests
from config.config import HEADERS, RANKING_URL

def get_gender_ids(ranking_id):
    response = requests.get(RANKING_URL.format(ranking_id=ranking_id), headers=HEADERS)
    response.raise_for_status()
    rankings = response.json().get("data", {}).get("rankings", [])
    return [entry["athlete_id"] for entry in rankings]

def get_top_athlete_ids():
    male_ids = get_gender_ids(13)
    female_ids = get_gender_ids(14)
    return male_ids + female_ids
