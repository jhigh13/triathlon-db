"""
Unit tests for master_data_import with mocked API calls.
Save as tests/test_master_import.py
"""
import pandas as pd
import pytest

# -- Fixtures -----------------------------------------------------------------

@pytest.fixture
def fake_athlete_ids(monkeypatch):
    """
    Monkey-patch get_top_athlete_ids to return a tiny stable list.
    """
    from Data_Import import athlete_ids
    monkeypatch.setattr(athlete_ids, "get_top_athlete_ids", lambda: [111, 222])
    return [111, 222]

@pytest.fixture
def fake_athlete_data(monkeypatch):
    """
    Monkey-patch both get_athlete_info_df and get_athlete_results_df
    to return deterministic DataFrames.
    """
    from Data_Import import athlete_data

    def _info_df(aid):
        return pd.DataFrame(
            [{
                "athlete_id": aid,
                "full_name": f"Test {aid}",
                "gender": "M",
                "country": "USA",
                "age": 25,
            }]
        )

    def _results_df(aid):
        return pd.DataFrame(
            [{
                "athlete_id": aid,
                "EventID": 99,
                "EventName": "Mock Event",
                "EventDate": "2025-05-30",
                "Venue": "Nowhere",
                "Country": "USA",
                "CategoryName": "Elite Men",
                "EventSpecifications": None,
                "Position": 1,
                "TotalTime": 3600,
                "SwimTime": 900,
                "T1": 60,
                "BikeTime": 1800,
                "T2": 60,
                "RunTime": 780,
            }]
        )

    monkeypatch.setattr(athlete_data, "get_athlete_info_df", _info_df)
    monkeypatch.setattr(athlete_data, "get_athlete_results_df", _results_df)

# -- Tests --------------------------------------------------------------------

def test_fetch_concurrent_returns_two_dfs(fake_athlete_ids, fake_athlete_data):
    """
    Ensure fetch_concurrent returns a list of DataFrames
    matching the number of athlete IDs.
    """
    from Data_Import.master_data_import import fetch_concurrent
    from Data_Import.athlete_data import get_athlete_info_df

    dfs = fetch_concurrent(fake_athlete_ids, get_athlete_info_df, max_workers=2)
    assert len(dfs) == 2
    assert all(isinstance(df, pd.DataFrame) for df in dfs)
    # confirm each df has exactly one row
    assert all(df.shape[0] == 1 for df in dfs)

def test_build_event_and_fact_tables(fake_athlete_ids, fake_athlete_data):
    """
    Validate the split between event dimension and race_results fact tables.
    """
    from Data_Import.master_data_import import build_event_and_fact_tables
    from Data_Import.athlete_data import get_athlete_results_df

    combined = pd.concat(
        [get_athlete_results_df(aid) for aid in fake_athlete_ids], ignore_index=True
    )
    events_df, fact_df = build_event_and_fact_tables(combined)

    # events_df should have one row because EventID is the same for mocks
    assert events_df.shape[0] == 1
    # fact_df should keep all rows and drop EventName/EventDate/Venue/Country
    assert fact_df.shape[0] == combined.shape[0]
    for col in ["EventName", "EventDate", "Venue", "Country"]:
        assert col not in fact_df.columns
