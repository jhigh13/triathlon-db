"""
Unit tests for tri_analysis.api_handling with mocked requests.
Save as tests/test_master_import.py
"""
import pandas as pd
import pytest
import requests
from tri_analysis import api_handling

# -- Fixtures -----------------------------------------------------------------

@pytest.fixture
def mock_requests_get(monkeypatch):
    """
    Fixture to monkeypatch requests.get for all tests.
    """
    def _patch(json_data, status_code=200):
        class MockResponse:
            def __init__(self, json_data, status_code):
                self._json = json_data
                self.status_code = status_code
            def json(self):
                return self._json
            def raise_for_status(self):
                if self.status_code != 200:
                    raise requests.HTTPError(f"Status {self.status_code}")
        monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse(json_data, status_code))
    return _patch

# -- Tests --------------------------------------------------------------------

def test_fetch_athlete_id_search_found(mock_requests_get):
    mock_requests_get({"data": [{"athlete_id": 12345}]})
    result = api_handling.fetch_athlete_id_search("Test Athlete")
    assert result == 12345

def test_fetch_athlete_id_search_not_found(mock_requests_get):
    mock_requests_get({"data": []})
    with pytest.raises(ValueError):
        api_handling.fetch_athlete_id_search("Missing Athlete")

def test_fetch_athlete_id_ranking(mock_requests_get):
    mock_requests_get({"data": {"rankings": [{"athlete_id": 1}, {"athlete_id": 2}]}})
    ids = api_handling.fetch_athlete_id_ranking(999)
    assert ids == [1, 2]

def test_fetch_athlete_info(mock_requests_get):
    mock_requests_get({
        "data": {
            "athlete_id": 42,
            "athlete_full_name": "Jane Doe",
            "athlete_gender": "F",
            "athlete_country_name": "USA",
            "athlete_age": 30,
            "categories": "{}"
        }
    })
    df = api_handling.fetch_athlete_info(42)
    assert df.iloc[0]["athlete_id"] == 42
    assert df.iloc[0]["full_name"] == "Jane Doe"
    assert df.iloc[0]["gender"] == "F"

def test_fetch_race_results_pagination(monkeypatch):
    # Simulate two pages of results
    calls = []
    def fake_get(url, headers=None):
        if not calls:
            calls.append(1)
            return type("Resp", (), {
                "status_code": 200,
                "json": lambda: {"data": ["A"], "next_page_url": "url2"},
                "raise_for_status": lambda self=None: None
            })()
        else:
            return type("Resp", (), {
                "status_code": 200,
                "json": lambda: {"data": ["B"], "next_page_url": None},
                "raise_for_status": lambda self=None: None
            })()
    monkeypatch.setattr(requests, "get", fake_get)
    results = api_handling.fetch_race_results(123)
    assert results == ["A", "B"]
