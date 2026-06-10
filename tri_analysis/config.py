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
ATHLETE_DATA_URL     = f"{BASE_URL}/athletes/{{athlete_id}}?output=basic"

# Ranking endpoint
RANKING_URL          = f"{BASE_URL}/rankings/{{ranking_id}}?limit={NUMBER_OF_ATHLETES}"

# Event & Program endpoints
EVENT_LISTING_URL  = f"{BASE_URL}/events"    
EVENT_DETAILS_URL    = f"{BASE_URL}/events/{{event_id}}"
PROGRAM_LISTING_URL  = f"{BASE_URL}/events/{{event_id}}/programs"
PROGRAM_DETAILS_URL   = f"{BASE_URL}/events/{{event_id}}/programs/{{program_id}}"
PROGRAM_RESULTS_URL = f"{BASE_URL}/events/{{event_id}}/programs/{{program_id}}/results"
PROGRAM_ENTRIES_URL = f"{BASE_URL}/events/{{event_id}}/programs/{{prog_id}}/entries"

# Database — always local for model training (fast reads/writes).
# Dashboard reads from Supabase via its own TRIATHLON_DATABASE_URL setting.
DB_URI = os.environ.get(
    "DB_URI",
    "postgresql+psycopg://postgres:Bc020406%21@localhost:5432/triathlon_results"
)


# Table name overrides (via env vars for testing)
ATHLETE_TABLE_NAME       = os.getenv('ATHLETE_TABLE_NAME', 'athlete')
EVENTS_TABLE_NAME        = os.getenv('EVENTS_TABLE_NAME', 'events')
RACE_RESULTS_TABLE_NAME  = os.getenv('RACE_RESULTS_TABLE_NAME', 'race_results')
RANKINGS_RESULTS_TABLE_NAME  = os.getenv('RANKINGS_RESULTS_TABLE_NAME', 'rankings')
METRICS_TABLE_NAME        = os.getenv('METRICS_TABLE_NAME', 'metrics')

# ID for filtering events (allow env overrides so we can run para-only backfills)
# Para Category IDs:
#   343 = Major Games (includes Paralympic Games and Para World Championships)
#   449 = Para Cup
#   448 = Para Series
# Elite/U23/Junior/Mixed Relay IDs: 340|341|342|623|352|347|640|624|351|348|349|350
CATEGORY_IDS = os.getenv(
    "CATEGORY_IDS",
    "340|341|342|623|343|352|347|640|624|351|348|349|449|448|350"  # Elite/U23/Junior/MR + Para defaults
)
SPEC_IDS = os.getenv(
    "SPEC_IDS",
    "356|357"  # default specifications; extend/override via env for para
)

# ── INSCYD External API ──────────────────────────────────────────────
# Metabolic test data (VO2/lactate/fat/carb polynomials + scalars).
# Set INSCYD_API_KEY and INSCYD_HOST in .env. The host has no scheme,
# e.g. INSCYD_HOST=app.inscyd.com  (base URL is derived from it).
INSCYD_API_KEY = os.getenv("INSCYD_API_KEY")
INSCYD_HOST = os.getenv("INSCYD_HOST")
INSCYD_BASE_URL = os.getenv(
    "INSCYD_BASE_URL",
    f"https://{INSCYD_HOST}/api/external" if INSCYD_HOST else None,
)
# Default cycling sport id can be pinned once known (see API "sport_id").
INSCYD_BIKE_SPORT_ID = os.getenv("INSCYD_BIKE_SPORT_ID")