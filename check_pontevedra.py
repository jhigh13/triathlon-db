from tri_analysis.database import get_engine
from sqlalchemy import text
import pandas as pd

engine = get_engine()

# Check for Pontevedra 2023 PTS3 Women
query = text("""
SELECT event_id, prog_id, event_name, event_date, prog_name, cat_name
FROM events 
WHERE event_name ILIKE :pattern 
  AND prog_name = :prog
ORDER BY event_date
""")

df = pd.read_sql(query, engine, params={"pattern": "%Pontevedra%", "prog": "PTS3 Women"})

if df.empty:
    print("No Pontevedra PTS3 Women events found in database")
    print("\nSearching for any 2023 PTS3 Women events...")
    query2 = text("""
    SELECT event_id, prog_id, event_name, event_date, prog_name, cat_name
    FROM events 
    WHERE prog_name = :prog
      AND EXTRACT(YEAR FROM event_date) = 2023
    ORDER BY event_date
    """)
    df2 = pd.read_sql(query2, engine, params={"prog": "PTS3 Women"})
    if df2.empty:
        print("No 2023 PTS3 Women events found at all")
    else:
        print(df2.to_string(index=False))
else:
    print("Pontevedra PTS3 Women event found:")
    print(df.to_string(index=False))
    
    # Check for results
    event_id = int(df.iloc[0]['event_id'])
    prog_id = int(df.iloc[0]['prog_id'])
    
    query3 = text("""
    SELECT a.full_name, rr.finish_position, rr.total_time
    FROM race_results rr
    JOIN athlete a ON a.athlete_id = rr.athlete_id
    WHERE rr.event_id = :eid AND rr.prog_id = :pid
      AND rr.finish_position IN (1,2,3)
      AND rr.finish_status = 'FINISH'
    ORDER BY rr.finish_position
    """)
    results = pd.read_sql(query3, engine, params={"eid": event_id, "pid": prog_id})
    
    if not results.empty:
        print("\nTop 3 finishers:")
        print(results.to_string(index=False))
    else:
        print("\nNo results found for this event")
