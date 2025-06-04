import pytest
import pandas as pd
from ml.label_generation import LabelGenerator, generate_labels_for_dataframe

def sample_df():
    return pd.DataFrame({
        "athlete_id": [1, 1, 2, 2, 3],
        "EventDate": pd.to_datetime([
            "2023-01-01", "2023-03-01", "2023-02-01", "2023-05-01", "2023-01-15"
        ]),
        "TotalTime_sec": [3600, 3550, 3700, 3650, 3500],
        "Position": [10, 8, 15, 13, 5],
    })

def test_create_next_race_labels():
    df = sample_df()
    generator = LabelGenerator()
    df_labeled = generator.create_next_race_labels(df)
    # Check that next_time_sec and next_position are present
    assert "next_time_sec" in df_labeled.columns
    assert "next_position" in df_labeled.columns
    # Check shifting: athlete 1's first row next_time_sec should match their second TotalTime_sec
    athlete1 = df_labeled[df_labeled["athlete_id"] == 1].sort_values("EventDate")
    assert athlete1.iloc[0]["next_time_sec"] == athlete1.iloc[1]["TotalTime_sec"]
    assert pd.isna(athlete1.iloc[-1]["next_time_sec"])  # last race should be NaN

def test_filter_for_modeling():
    df = sample_df()
    generator = LabelGenerator()
    df_labeled = generator.create_next_race_labels(df)
    df_model = generator.filter_for_modeling(df_labeled)
    # Should drop last race per athlete (no next race)
    assert all(~df_model["next_time_sec"].isna())
    # Number of rows should be less than input
    assert len(df_model) < len(df_labeled)

def test_generate_labels_for_dataframe():
    df = sample_df()
    df_model = generate_labels_for_dataframe(df)
    # Should have next_time_sec and next_position, and no NaNs in those columns
    assert "next_time_sec" in df_model.columns
    assert "next_position" in df_model.columns
    assert df_model["next_time_sec"].notna().all()
    assert df_model["next_position"].notna().all()
