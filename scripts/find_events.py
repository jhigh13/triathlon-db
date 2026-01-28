"""Quick script to find Cuba events and upcoming races with start lists."""
from tri_analysis.database import get_engine
import pandas as pd
from sqlalchemy import text

pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 60)
engine = get_engine()

print("=" * 80)
print("Searching for Cuba/Havana events...")
print("=" * 80)

with engine.connect() as conn:
    df = pd.read_sql(
        text("""
        SELECT DISTINCT event_id, prog_id, event_name, prog_name, event_date
        FROM events 
        WHERE event_name ILIKE :pattern1 OR event_name ILIKE :pattern2
        ORDER BY event_date DESC
        LIMIT 20
        """),
        conn,
        params={"pattern1": "%cuba%", "pattern2": "%havana%"}
    )
    print(df.to_string())

print("\n" + "=" * 80)
print("Events with active program_entries (start lists)...")
print("=" * 80)

with engine.connect() as conn:
    df2 = pd.read_sql(
        text("""
        SELECT DISTINCT e.event_id, e.prog_id, e.event_name, e.prog_name, e.event_date,
               COUNT(pe.athlete_id) as n_entries
        FROM events e 
        JOIN program_entries pe ON e.event_id=pe.event_id AND e.prog_id=pe.prog_id 
        WHERE pe.is_active=TRUE 
        GROUP BY e.event_id, e.prog_id, e.event_name, e.prog_name, e.event_date
        ORDER BY e.event_date
        LIMIT 30
        """),
        conn
    )
    if df2.empty:
        print("No active start lists found.")
    else:
        print(df2.to_string())

print("\n" + "=" * 80)
print("February 2025 events (recent Americas Cup events)...")
print("=" * 80)

with engine.connect() as conn:
    df3 = pd.read_sql(
        text("""
        SELECT DISTINCT event_id, prog_id, event_name, prog_name, event_date
        FROM events 
        WHERE event_date BETWEEN '2025-02-01' AND '2025-02-28'
          AND (prog_name = 'Elite Men' OR prog_name = 'Elite Women')
        ORDER BY event_date
        """),
        conn
    )
    print(df3.to_string())
