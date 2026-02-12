from tri_analysis.database import get_engine
import pandas as pd
from sqlalchemy import text

engine = get_engine()

# Check for Wollongong PTS3 programs
query = text("""
SELECT 
    event_id, 
    prog_id, 
    event_name, 
    event_date, 
    prog_name, 
    cat_name
FROM events 
WHERE event_name ILIKE '%Wollongong%' 
  AND prog_name LIKE 'PTS%'
ORDER BY prog_name
""")

df = pd.read_sql(query, engine)
print("\nWollongong PTS programs found:")
print(df.to_string(index=False))

# Also check if we have race results
print("\n\nChecking race results for PTS3 programs:")
result_query = text("""
SELECT 
    e.event_id,
    e.prog_id,
    e.event_name,
    e.prog_name,
    COUNT(*) as num_finishers,
    COUNT(CASE WHEN rr.finish_position IN (1,2,3) THEN 1 END) as medalists
FROM events e
JOIN race_results rr ON rr.event_id = e.event_id AND rr.prog_id = e.prog_id
WHERE e.event_name ILIKE '%Wollongong%'
  AND e.prog_name LIKE 'PTS3%'
  AND rr.finish_status = 'FINISH'
GROUP BY e.event_id, e.prog_id, e.event_name, e.prog_name
ORDER BY e.prog_name
""")
df_results = pd.read_sql(result_query, engine)
print(df_results.to_string(index=False))
