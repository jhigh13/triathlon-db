# config/config.py
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("TRI_API_KEY")
HEADERS = {"apikey": API_KEY}

NUMBER_OF_ATHLETES = 1000
BASE_URL = "https://api.triathlon.org/v1"

# Athlete endpoints
ATHLETE_SEARCH_URL   = f"{BASE_URL}/search/athletes"
ATHLETE_RESULTS_URL  = f"{BASE_URL}/athletes/{{athlete_id}}/results"
ATHLETE_DATA_URL     = f"{BASE_URL}/athletes/{{athlete_id}}?ouput=basic"

# Ranking endpoint
RANKING_URL          = f"{BASE_URL}/rankings/{{ranking_id}}?limit={NUMBER_OF_ATHLETES}"

# Event & Program endpoints
EVENT_DETAILS_URL    = f"{BASE_URL}/events/{{event_id}}"
PROGRAM_RESULTS_URL  = f"{BASE_URL}/programs/{{program_id}}/results"

# Database
DB_URI = os.environ.get(
    "DB_URI",
    "postgresql+psycopg2://postgres:Bc020406!@localhost:5432/triathlon_results"
)
# Table name overrides (via env vars for testing)
ATHLETE_TABLE_NAME       = os.getenv('ATHLETE_TABLE_NAME', 'athlete')
EVENTS_TABLE_NAME        = os.getenv('EVENTS_TABLE_NAME', 'events')
RACE_RESULTS_TABLE_NAME  = os.getenv('RACE_RESULTS_TABLE_NAME', 'race_results')