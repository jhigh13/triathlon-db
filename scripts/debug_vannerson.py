"""Debug why Vannerson prediction is so wrong."""
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv('DB_URI'))

with engine.connect() as conn:
    # Get Vannerson's race history
    query = text("""
SELECT 
    rr.athlete_id,
    rr.athlete_full_name,
    e.event_date,
    e.event_name,
    e.event_tier,
    e.prog_distance_category,
    rr.total_time_sec,
    rr.swim_sec,
    rr.bike_sec,
    rr.run_sec,
    rr."Position"
FROM race_results rr
JOIN events e ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
WHERE rr.athlete_full_name ILIKE '%vannerson%'
ORDER BY e.event_date DESC
""")
    df = pd.read_sql(query, conn)
    print("=== Vannerson race history ===")
    print(df.to_string())
    print()

    # Get Lamprecht's race history  
    query2 = text("""
SELECT 
    rr.athlete_id,
    rr.athlete_full_name,
    e.event_date,
    e.event_name,
    e.event_tier,
    e.prog_distance_category,
    rr.total_time_sec,
    rr.swim_sec,
    rr.bike_sec,
    rr.run_sec,
    rr."Position"
FROM race_results rr
JOIN events e ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
WHERE rr.athlete_full_name ILIKE '%lamprecht%'
ORDER BY e.event_date DESC
""")
    df2 = pd.read_sql(query2, conn)
    print("=== Lamprecht race history ===")
    print(df2.to_string())
