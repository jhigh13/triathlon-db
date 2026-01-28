"""Check tier assignments for H2 2025 events."""
from tri_analysis.database import get_engine
import pandas as pd
from sqlalchemy import text

engine = get_engine()

query = text("""
SELECT 
    event_date, 
    event_name,
    CASE
        WHEN event_name ~* '(WTCS|Championship Series|World Championship Finals)' THEN 1
        WHEN event_name ~* 'World Cup|World Triathlon Cup' THEN 2
        WHEN event_name ~* '(Continental Cup|Americas Cup|European Cup|Asian Cup|African Cup|Oceania Cup|Americas Triathlon Cup|Africa Triathlon Cup|Europe Triathlon Cup)' THEN 3
        ELSE 4
    END AS event_tier
FROM events
WHERE event_date >= '2025-07-01' 
AND event_date <= '2025-12-31'
AND prog_name ~* '(Elite Men|Elite Women)'
ORDER BY event_date
LIMIT 20
""")

df = pd.read_sql(query, engine)
print(df.to_string(index=False))
