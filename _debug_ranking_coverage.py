import os, sys
sys.path.insert(0, os.path.abspath('.'))
from dotenv import load_dotenv
load_dotenv(override=True)
from tri_analysis.database import get_engine
from sqlalchemy import text
import pandas as pd

engine = get_engine()

# 1. What ranking categories exist?
df = pd.read_sql(text("SELECT ranking_cat_name, ranking_cat_id, year, COUNT(*) as n_athletes FROM athlete_rankings GROUP BY 1,2,3 ORDER BY 3 DESC, 1"), engine)
print("=== Ranking Categories ===")
print(df.to_string(index=False))

# 2. How many La Paz athletes have rankings vs don't?
df2 = pd.read_sql(text("""
    SELECT pe.athlete_full_name, ar.rank_position, ar.total_points, ar.ranking_cat_name
    FROM program_entries pe
    LEFT JOIN athlete_rankings ar ON pe.athlete_id = ar.athlete_id
    WHERE pe.event_id = 195253 AND pe.prog_id = 676981
    ORDER BY ar.rank_position ASC NULLS LAST
"""), engine)
n_with = df2['rank_position'].notna().sum()
n_without = df2['rank_position'].isna().sum()
print(f"\n=== La Paz Ranking Coverage ===")
print(f"Athletes with rankings: {n_with}")
print(f"Athletes WITHOUT rankings: {n_without}")
print(f"\nTop 10 ranked athletes:")
print(df2.head(10).to_string(index=False))
print(f"\nUnranked athletes:")
unranked = df2[df2['rank_position'].isna()]
print(unranked[['athlete_full_name']].to_string(index=False))

# 3. How many athletes in Elo table?
elo_df = pd.read_sql(text("""
    SELECT COUNT(*) as total_elo_athletes,
           AVG(elo_rating) as avg_elo,
           MIN(elo_rating) as min_elo,
           MAX(elo_rating) as max_elo
    FROM athlete_elo_ratings
"""), engine)
print(f"\n=== Elo Ratings Coverage ===")
print(elo_df.to_string(index=False))

# 4. La Paz athletes Elo coverage
elo_lapaz = pd.read_sql(text("""
    SELECT pe.athlete_full_name, er.elo_rating, er.elo_peak, er.elo_races
    FROM program_entries pe
    LEFT JOIN athlete_elo_ratings er ON pe.athlete_id = er.athlete_id
    WHERE pe.event_id = 195253 AND pe.prog_id = 676981
    ORDER BY er.elo_rating DESC NULLS LAST
"""), engine)
n_with_elo = elo_lapaz['elo_rating'].notna().sum()
n_without_elo = elo_lapaz['elo_rating'].isna().sum()
print(f"\nLa Paz Elo Coverage:")
print(f"  With Elo: {n_with_elo}")
print(f"  Without Elo: {n_without_elo}")
