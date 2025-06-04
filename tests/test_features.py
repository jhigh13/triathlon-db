import pytest
import pandas as pd
from ml.features import FeatureEngineer, engineer_features_for_dataframe

def sample_df():
    return pd.DataFrame({
        "athlete_id": [1, 1, 2, 2, 3],
        "EventDate": pd.to_datetime([
            "2023-01-01", "2023-03-01", "2023-02-01", "2023-05-01", "2023-01-15"
        ]),
        "TotalTime": ["01:00:00", "00:59:10", "01:01:40", "01:00:50", "00:58:20"],
        "Position": [10, 8, 15, 13, 5],
        "EventSpecifications": [
            "Triathlon, Sprint", "Triathlon, Sprint", "Triathlon, Sprint", "Triathlon, Sprint", "Triathlon, Sprint"
        ],
        "BikeTime": ["00:30:00"]*5,
        "T2": ["00:01:00"]*5,
        "SwimTime": ["00:15:00"]*5,
        "RunTime": ["00:14:00"]*5,
        "T1": ["00:01:00"]*5,
        "age": [25, 25, 30, 30, 22],
    })

def test_clean_initial_data():
    df = sample_df()
    engineer = FeatureEngineer()
    cleaned = engineer.clean_initial_data(df)
    assert isinstance(cleaned, pd.DataFrame)
    assert "EventSpecifications" in cleaned.columns

def test_process_event_specifications():
    df = sample_df()
    engineer = FeatureEngineer()
    processed = engineer.process_event_specifications(df)
    assert "race_type" in processed.columns
    assert "distance" in processed.columns
    assert (processed["race_type"] == "Triathlon").all()
    assert (processed["distance"] == "Sprint").all()

def test_create_event_mode_feature():
    df = sample_df()
    engineer = FeatureEngineer()
    processed = engineer.process_event_specifications(df)
    event_mode = engineer.create_event_mode_feature(processed)
    assert "event_mode" in event_mode.columns
    assert (event_mode["event_mode"] == "individual").all()

def test_handle_dnf_flags():
    df = sample_df()
    engineer = FeatureEngineer()
    processed = engineer.process_event_specifications(df)
    event_mode = engineer.create_event_mode_feature(processed)
    dnf = engineer.handle_dnf_flags(event_mode)
    assert "DNF_flag" in dnf.columns
    assert (dnf["DNF_flag"] == 0).all()  # all valid times
    assert "recent_5_DNF_rate" in dnf.columns

def test_create_rolling_features():
    df = sample_df()
    engineer = FeatureEngineer()
    processed = engineer.process_event_specifications(df)
    event_mode = engineer.create_event_mode_feature(processed)
    dnf = engineer.handle_dnf_flags(event_mode)
    rolling = engineer.create_rolling_features(dnf)
    assert "rolling3_time_dist" in rolling.columns
    assert "rolling5_time_dist" in rolling.columns
    assert "rolling5_time_all" in rolling.columns
    assert "rolling3_pos" in rolling.columns
    assert "rolling5_pos" in rolling.columns

def test_create_time_features():
    df = sample_df()
    engineer = FeatureEngineer()
    processed = engineer.process_event_specifications(df)
    event_mode = engineer.create_event_mode_feature(processed)
    dnf = engineer.handle_dnf_flags(event_mode)
    rolling = engineer.create_rolling_features(dnf)
    time_feats = engineer.create_time_features(rolling)
    assert "days_since_last" in time_feats.columns

def test_encode_categorical_features():
    df = sample_df()
    engineer = FeatureEngineer()
    processed = engineer.process_event_specifications(df)
    event_mode = engineer.create_event_mode_feature(processed)
    dnf = engineer.handle_dnf_flags(event_mode)
    rolling = engineer.create_rolling_features(dnf)
    time_feats = engineer.create_time_features(rolling)
    encoded = engineer.encode_categorical_features(time_feats)
    assert any(col.startswith("rt_") for col in encoded.columns)
    assert any(col.startswith("dist_") for col in encoded.columns)
    assert any(col.startswith("mode_") for col in encoded.columns)

def test_engineer_features_for_dataframe():
    df = sample_df()
    result = engineer_features_for_dataframe(df)
    # Should have engineered columns and all column names as str
    assert isinstance(result, pd.DataFrame)
    assert all(isinstance(col, str) for col in result.columns)
