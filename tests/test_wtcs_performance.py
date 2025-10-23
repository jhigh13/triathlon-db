import pandas as pd
from tri_analysis.wtcs_performance import WTCSFilters, aggregate_checkpoint_metrics


def test_aggregate_checkpoint_metrics_basic():
    # Synthetic minimal dataset with two athletes, two events
    data = [
        {
            "athlete_id": 1,
            "full_name": "Athlete One",
            "event_id": 100,
            "prog_id": 10,
            "event_date": pd.Timestamp("2025-05-01"),
            "finish_position": 5,
            "position_at_swim": 10,
            "position_at_t1": 9,
            "position_at_bike": 7,
            "position_at_t2": 6,
            "position_at_run": 5,
            "behindswim": 20,
            "behindt1": 18,
            "behindbike": 25,
            "behindt2": 30,
            "behindrun": 32,
            "swimrank": 12,
            "t1rank": 9,
            "bikerank": 8,
            "t2rank": 7,
            "runrank": 5,
            "swim_to_t1_pos_change": -1,
            "t1_to_bike_pos_change": -2,
            "bike_to_t2_pos_change": -1,
            "t2_to_run_pos_change": -1,
        },
        {
            "athlete_id": 1,
            "full_name": "Athlete One",
            "event_id": 101,
            "prog_id": 10,
            "event_date": pd.Timestamp("2025-06-01"),
            "finish_position": 7,
            "position_at_swim": 9,
            "position_at_t1": 8,
            "position_at_bike": 8,
            "position_at_t2": 7,
            "position_at_run": 7,
            "behindswim": 18,
            "behindt1": 17,
            "behindbike": 28,
            "behindt2": 34,
            "behindrun": 40,
            "swimrank": 11,
            "t1rank": 8,
            "bikerank": 8,
            "t2rank": 7,
            "runrank": 6,
            "swim_to_t1_pos_change": -1,
            "t1_to_bike_pos_change": 0,
            "bike_to_t2_pos_change": -1,
            "t2_to_run_pos_change": 0,
        },
        {
            "athlete_id": 2,
            "full_name": "Athlete Two",
            "event_id": 100,
            "prog_id": 10,
            "event_date": pd.Timestamp("2025-05-01"),
            "finish_position": 12,
            "position_at_swim": 20,
            "position_at_t1": 22,
            "position_at_bike": 18,
            "position_at_t2": 15,
            "position_at_run": 12,
            "behindswim": 60,
            "behindt1": 70,
            "behindbike": 90,
            "behindt2": 110,
            "behindrun": 120,
            "swimrank": 21,
            "t1rank": 22,
            "bikerank": 19,
            "t2rank": 16,
            "runrank": 13,
            "swim_to_t1_pos_change": 2,
            "t1_to_bike_pos_change": -4,
            "bike_to_t2_pos_change": -3,
            "t2_to_run_pos_change": -3,
        },
    ]
    df = pd.DataFrame(data)
    filters = WTCSFilters(min_events=1)
    summary = aggregate_checkpoint_metrics(df, filters)
    assert not summary.empty
    # Validate expected columns
    expected_cols = {
        "athlete_id", "full_name", "races", "avg_finish_position", "avg_pos_swim",
        "avg_gap_swim", "avg_swim_rank", "avg_delta_swim_t1"
    }
    assert expected_cols.issubset(set(summary.columns))
    # Athlete 1 averages
    a1 = summary[summary.athlete_id == 1].iloc[0]
    assert a1["races"] == 2
    assert round(a1["avg_finish_position"], 2) == 6.0  # (5+7)/2
    # Athlete 2 single race
    a2 = summary[summary.athlete_id == 2].iloc[0]
    assert a2["races"] == 1
