from tri_analysis.database import get_engine
import pandas as pd
from sqlalchemy import text

engine = get_engine()

# Check top 3 finishers for PTS3 Men and Women at Wollongong
query = text("""
SELECT 
    e.prog_name,
    a.full_name,
    rr.finish_position,
    rr.total_time
FROM race_results rr
JOIN athlete a ON a.athlete_id = rr.athlete_id
JOIN events e ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
WHERE rr.event_id = 188993 
  AND rr.prog_id IN (674975, 674967)
  AND rr.finish_position IN (1,2,3)
  AND rr.finish_status = 'FINISH'
ORDER BY e.prog_name, rr.finish_position
""")

df = pd.read_sql(query, engine)
print("\nWollongong 2025 PTS3 Top 3 Finishers:")
print(df.to_string(index=False))
