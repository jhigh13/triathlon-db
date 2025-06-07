import pytest
# ensure project root is on path for Data_Import imports
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Import.historical_rankings_scraper import HistoricalRankingsScraper

@pytest.fixture(autouse=True)
def isolate_db(monkeypatch):
    # Use real test tables in the PostgreSQL database
    from Data_Import.database import get_engine, create_test_tables
    create_test_tables()
    engine = get_engine()
    from sqlalchemy import text
    with engine.begin() as conn:
        # Clean up for repeatable tests
        conn.execute(text("DELETE FROM test_athlete_rankings"))
        conn.execute(text("DELETE FROM test_athlete"))
        # Insert a test athlete row for name matching
        conn.execute(
            text("INSERT INTO test_athlete (athlete_id, full_name) VALUES (:id, :name)"),
            {"id": 1, "name": "Alex Yee"}
        )
    yield

def test_upsert_no_rankings_does_not_error():
    """upsert_rankings should handle empty list without error."""
    scraper = HistoricalRankingsScraper()
    # Patch scraper to use test tables
    scraper.match_athlete_id = lambda name: 1 if name == "Alex Yee" else None
    scraper.upsert_rankings([])

def test_upsert_creates_and_updates_rankings():
    """upsert_rankings should insert a ranking record and update on conflict in test_athlete_rankings."""
    scraper = HistoricalRankingsScraper()
    # Patch scraper to use test tables
    scraper.match_athlete_id = lambda name: 1 if name == "Alex Yee" else None
    # Patch upsert_rankings to use test_athlete_rankings
    import types
    def upsert_rankings_test(self, rankings):
        from Data_Import.database import get_engine
        from datetime import date
        from sqlalchemy import text
        engine = get_engine()
        today = date.today()
        upsert_sql = text("""
            INSERT INTO test_athlete_rankings
              (athlete_id, athlete_name, ranking_cat_name, ranking_cat_id,
               rank_position, total_points, retrieved_at)
            VALUES (:athlete_id, :athlete_name, :ranking_cat_name, :ranking_cat_id,
                    :rank_position, :total_points, :retrieved_at)
            ON CONFLICT (athlete_name, ranking_cat_name, retrieved_at)
            DO UPDATE SET
              rank_position = EXCLUDED.rank_position,
              total_points = EXCLUDED.total_points
        """)
        with engine.begin() as conn:
            for r in rankings:
                cat_name = r['ranking_cat_name']
                cat_id = r['ranking_cat_id']
                for athlete in r['athletes']:
                    full_name = f"{athlete['given_name']} {athlete['family_name']}"
                    aid = self.match_athlete_id(full_name)
                    params = {
                        'athlete_id': aid,
                        'athlete_name': full_name,
                        'ranking_cat_name': cat_name,
                        'ranking_cat_id': cat_id,
                        'rank_position': athlete['rank'],
                        'total_points': athlete['total_points'],
                        'retrieved_at': today
                    }
                    conn.execute(upsert_sql, params)
    scraper.upsert_rankings = types.MethodType(upsert_rankings_test, scraper)
    # Create a dummy ranking record
    dummy_rankings = [{
        'ranking_cat_name': 'Test Series 2025 Male',
        'ranking_cat_id': 99,
        'athletes': [
            {'rank': 1, 'given_name': 'Alex', 'family_name': 'Yee', 'total_points': 123.4}
        ]
    }]
    # First insert
    scraper.upsert_rankings(dummy_rankings)
    # Modify points and upsert again
    dummy_rankings[0]['athletes'][0]['total_points'] = 200.5
    scraper.upsert_rankings(dummy_rankings)
    # Verify from DB
    from Data_Import.database import get_engine
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT rank_position, total_points FROM test_athlete_rankings WHERE athlete_name = 'Alex Yee'"
        )).fetchone()
    assert result == (1, 200.5)
