# config/config.py
from dotenv import load_dotenv
import os
load_dotenv(override=True)

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
EVENT_LISTING_URL  = f"{BASE_URL}/events"    
EVENT_DETAILS_URL    = f"{BASE_URL}/events/{{event_id}}"
PROGRAM_LISTING_URL  = f"{BASE_URL}/events/{{event_id}}/programs"
PROGRAM_DETAILS_URL   = f"{BASE_URL}/events/{{event_id}}/programs/{{program_id}}"
PROGRAM_RESULTS_URL = f"{BASE_URL}/events/{{event_id}}/programs/{{program_id}}/results"

# Database
DB_URI_OLD = os.environ.get(
    "DB_URI",
    "postgresql://postgres.kzddtovwtnsfxccyicjd:6mBe-4ZKA_YWawR@aws-0-us-east-2.pooler.supabase.com:5432/postgres"
)

DB_URI = os.environ.get(
    "DB_URI",
    "postgresql+psycopg2://postgres:Bc020406!@localhost:5432/triathlon_results"
)

# Table name overrides (via env vars for testing)
ATHLETE_TABLE_NAME       = os.getenv('ATHLETE_TABLE_NAME', 'athlete')
EVENTS_TABLE_NAME        = os.getenv('EVENTS_TABLE_NAME', 'events')
RACE_RESULTS_TABLE_NAME  = os.getenv('RACE_RESULTS_TABLE_NAME', 'race_results')
RANKINGS_RESULTS_TABLE_NAME  = os.getenv('RANKINGS_RESULTS_TABLE_NAME', 'rankings')
METRICS_TABLE_NAME        = os.getenv('METRICS_TABLE_NAME', 'metrics')

# ID for filtering events
CATEGORY_IDS = "340|341|342|623|343|352|347|640|624|351|348|349|623|449|448|350" #Add Para afterwards
SPEC_IDS = "356|357"
# ID for filtering events (allow env overrides so we can run para-only backfills)
CATEGORY_IDS = os.getenv(
    "CATEGORY_IDS",
    "340|341|342|623|343|352|347|640|624|351|348|349|623|449|448|350"  # Elite/U23/Junior/MR defaults; extend with para via env
)
SPEC_IDS = os.getenv(
    "SPEC_IDS",
    "356|357"  # default specifications; extend/override via env for para
)