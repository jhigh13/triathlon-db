import os, sys
sys.path.insert(0, os.path.abspath('.'))
from dotenv import load_dotenv
load_dotenv(override=True)
from tri_analysis.database import get_engine
from sqlalchemy import text
import pandas as pd

engine = get_engine()

# 1. All distinct distance categories with counts
with engine.connect() as conn:
    result = conn.execute(text("SELECT prog_distance_category, COUNT(*) as n FROM events GROUP BY 1 ORDER BY 2 DESC"))
    print("=== Distance Categories ===")
    for row in result:
        print(f"  {row[0]}: {row[1]}")

# 2. Find Rodrigo Gonzalez Lopez athlete_id
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT pe.athlete_id, pe.athlete_full_name
        FROM program_entries pe
        WHERE pe.event_id = 195253 AND pe.prog_id = 676981
        AND pe.athlete_full_name ILIKE '%Rodrigo Gonzalez%'
    """))
    rows = result.fetchall()
    print("\n=== Rodrigo in start list ===")
    for r in rows:
        print(f"  athlete_id={r[0]}, name={r[1]}")
    rodrigo_id = rows[0][0] if rows else None

# 3. Show Rodrigo's FULL race history with distance labels and times
if rodrigo_id:
    df = pd.read_sql(text("""
        SELECT e.event_date, e.event_name, e.prog_name, e.prog_distance_category,
               rr.swimtime, rr.biketime, rr.runtime, rr.total_time, rr.finish_status
        FROM race_results rr
        JOIN events e ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
        WHERE rr.athlete_id = :aid
        ORDER BY e.event_date DESC
        LIMIT 30
    """), engine, params={"aid": rodrigo_id})
    print(f"\n=== Rodrigo ({rodrigo_id}) Full Race History ===")
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False))

    # Show just the sprint-labeled races
    sprint_df = df[df['prog_distance_category'].str.lower() == 'sprint']
    print(f"\n=== Rodrigo's 'sprint'-labeled races only ({len(sprint_df)} races) ===")
    print(sprint_df.to_string(index=False))

# 4. Also check Dylan Campa (another bad case)
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT pe.athlete_id, pe.athlete_full_name
        FROM program_entries pe
        WHERE pe.event_id = 195253 AND pe.prog_id = 676981
        AND pe.athlete_full_name ILIKE '%Dylan%Campa%'
    """))
    rows = result.fetchall()
    dylan_id = rows[0][0] if rows else None

if dylan_id:
    df = pd.read_sql(text("""
        SELECT e.event_date, e.event_name, e.prog_name, e.prog_distance_category,
               rr.swimtime, rr.biketime, rr.runtime, rr.total_time, rr.finish_status
        FROM race_results rr
        JOIN events e ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
        WHERE rr.athlete_id = :aid
        ORDER BY e.event_date DESC
        LIMIT 20
    """), engine, params={"aid": dylan_id})
    print(f"\n=== Dylan Campa ({dylan_id}) Full Race History ===")
    print(df.to_string(index=False))

    sprint_df = df[df['prog_distance_category'].str.lower() == 'sprint']
    print(f"\n=== Dylan's 'sprint'-labeled races only ({len(sprint_df)} races) ===")
    print(sprint_df.to_string(index=False))
