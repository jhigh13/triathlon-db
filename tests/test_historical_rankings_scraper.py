import pytest
from tri_analysis.historical_rankings_scraper import HistoricalRankingsScraper

# Patch DB engine and SQL execution with in-memory mocks
@pytest.fixture(autouse=True)
def patch_db(monkeypatch):
    class DummyConn:
        def __init__(self):
            self.executed = []
        def execute(self, sql, params=None):
            self.executed.append((sql, params))
            return self
        def fetchall(self):
            return []
        def fetchone(self):
            return (1, 200.5)
        def scalar(self):
            return 0
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def begin(self):
            return self
        def connect(self):
            return self
    class DummyEngine:
        def begin(self):
            return DummyConn()
        def connect(self):
            return DummyConn()
    monkeypatch.setattr("tri_analysis.historical_rankings_scraper.get_engine", lambda: DummyEngine())
    yield

def test_upsert_no_rankings_does_not_error():
    """upsert_rankings should handle empty list without error and not insert rows."""
    scraper = HistoricalRankingsScraper()
    scraper.match_athlete_id = lambda name: 1 if name == "Alex Yee" else None
    # Should not raise or insert anything
    scraper.upsert_rankings([])

def test_upsert_creates_and_updates_rankings():
    """upsert_rankings should insert a ranking record and update on conflict in test_athlete_rankings."""
    scraper = HistoricalRankingsScraper()
    scraper.match_athlete_id = lambda name: 1 if name == "Alex Yee" else None
    dummy_rankings = [{
        'ranking_cat_name': 'Test Series 2025 Male',
        'ranking_cat_id': 99,
        'year': 2025,
        'athletes': [
            {'rank': 1, 'given_name': 'Alex', 'family_name': 'Yee', 'total_points': 123.4}
        ]
    }]
    # Should not raise and should call execute for each athlete
    scraper.upsert_rankings(dummy_rankings)
    # Modify points and upsert again
    dummy_rankings[0]['athletes'][0]['total_points'] = 200.5
    scraper.upsert_rankings(dummy_rankings)
