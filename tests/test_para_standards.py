import pandas as pd

from tri_analysis.time_utils import pace_sec_per_100m, pace_sec_per_km, seconds_to_hms, time_to_seconds


def test_distance_panel_inference_sprint_when_missing_category():
    import pandas as pd

    from para_triathlon_analysis.para_standards import compute_metrics

    df = pd.DataFrame(
        [
            {
                "swimtime": "00:12:00",
                "t1time": "00:01:00",
                "biketime": "00:40:00",
                "t2time": "00:01:00",
                "runtime": "00:15:00",
                "total_time": "01:09:00",
                "prog_distance_category": None,
                "swim_distance": 750.0,
                "bike_distance": 22500.0,
                "run_distance": 5000.0,
            }
        ]
    )
    out = compute_metrics(df)
    assert out.loc[0, "distance_panel"] == "Sprint"


def test_compute_metrics_turns_zero_splits_into_missing_and_filters_nonfinish():
    from para_triathlon_analysis.para_standards import compute_metrics

    df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "prog_id": 10,
                "event_name": "Test Event",
                "event_date": "2024-01-01",
                "prog_distance_category": "Sprint",
                "swim_distance": 750.0,
                "bike_distance": 20000.0,
                "run_distance": 5000.0,
                "athlete_id": 111,
                "full_name": "USA Athlete",
                "finish_status": "FINISH",
                "finish_position": 5,
                "swimtime": "00:00:00",
                "biketime": "00:30:00",
                "runtime": "00:15:00",
                "t1time": "00:00:00",
                "t2time": "00:00:00",
                "total_time": "01:00:00",
            },
            {
                "event_id": 1,
                "prog_id": 10,
                "event_name": "Test Event",
                "event_date": "2024-01-01",
                "prog_distance_category": "Sprint",
                "swim_distance": 750.0,
                "bike_distance": 20000.0,
                "run_distance": 5000.0,
                "athlete_id": 222,
                "full_name": "Opponent",
                "finish_status": "DNF",
                "finish_position": None,
                "swimtime": "00:10:00",
                "biketime": "00:00:00",
                "runtime": "00:00:00",
                "t1time": "00:00:00",
                "t2time": "00:00:00",
                "total_time": "00:00:00",
            },
        ]
    )

    out = compute_metrics(df)
    usa_row = out[out["athlete_id"] == 111].iloc[0]
    opp_row = out[out["athlete_id"] == 222].iloc[0]

    # FINISH row: placeholder 00:00:00 splits become missing (not 0)
    assert pd.isna(usa_row["swimtime_sec"])
    assert pd.isna(usa_row["t1time_sec"])
    assert pd.isna(usa_row["t2time_sec"])

    # Non-finish rows: never keep split seconds for plotting
    assert pd.isna(opp_row["swimtime_sec"])
    assert pd.isna(opp_row["biketime_sec"])
    assert pd.isna(opp_row["runtime_sec"])


def test_composite_score_positive_when_usa_better():
    from para_triathlon_analysis.para_standards import compute_composite_scores

    df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "prog_id": 1,
                "event_label": "Test Event",
                "event_date": "2024-01-01",
                "athlete_id": 100,
                "finish_status": "FINISH",
                # USA better: lower swim/run pace + lower transitions + higher bike speed
                "swim_pace_s_per_100m": 90.0,
                "bike_speed_kmh": 36.0,
                "run_pace_s_per_km": 220.0,
                "t1time_sec": 50.0,
                "t2time_sec": 45.0,
            },
            {
                "event_id": 1,
                "prog_id": 1,
                "event_label": "Test Event",
                "event_date": "2024-01-01",
                "athlete_id": 200,
                "finish_status": "FINISH",
                # Benchmark worse
                "swim_pace_s_per_100m": 95.0,
                "bike_speed_kmh": 34.0,
                "run_pace_s_per_km": 230.0,
                "t1time_sec": 60.0,
                "t2time_sec": 55.0,
            },
        ]
    )

    composite, _scales = compute_composite_scores(df, usa_athlete_id=100, athlete_weights={200: 1.0})
    assert len(composite) == 1
    assert float(composite.loc[0, "score"]) > 0


def test_time_factor_normalization_ptvi_b3_2025_subtracts_from_swim_and_total():
    from para_triathlon_analysis.para_standards import compute_metrics

    df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "prog_id": 1,
                "event_name": "Test Event",
                "event_date": "2025-06-01",
                "prog_name": "PTVI Men",
                "prog_distance_category": "Sprint",
                "swim_distance": 750.0,
                "bike_distance": 20000.0,
                "run_distance": 5000.0,
                "athlete_id": 10,
                "full_name": "Oscar Kelly B3",
                "finish_status": "FINISH",
                "finish_position": 1,
                # Includes factor in swim (2:41 for PTVI Men)
                "swimtime": "00:12:00",
                "t1time": "00:00:30",
                "biketime": "00:30:00",
                "t2time": "00:00:30",
                "runtime": "00:15:00",
                "total_time": "00:58:00",
            }
        ]
    )

    out = compute_metrics(df)
    row = out.iloc[0]
    assert row["para_subclass"] == "B3"
    assert row["time_factor_year"] == 2025
    assert row["time_factor_sec"] == 161
    assert bool(row["time_factor_applied"]) is True
    assert float(row["swimtime_sec_adjusted"]) == 12 * 60
    # Swim splits are not adjusted (feeds are inconsistent); raw == adjusted
    assert float(row["swimtime_sec_raw"]) == 12 * 60

    assert float(row["total_time_sec_adjusted"]) == 58 * 60
    # Effort total from sum of splits: 12:00 + 0:30 + 30:00 + 0:30 + 15:00 = 58:00
    assert float(row["sum_splits_sec"]) == 58 * 60
    assert float(row["total_time_sec_raw"]) == 58 * 60
    # Default behavior: total_time_sec remains adjusted-for-placing
    assert float(row["total_time_sec"]) == 58 * 60


def test_time_factor_segment_matches_adjusted_minus_effort_total():
    from para_triathlon_analysis.para_standards import compute_metrics

    df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "prog_id": 1,
                "event_name": "Test Event",
                "event_date": "2025-06-01",
                "prog_name": "PTWC Women",
                "prog_distance_category": "Sprint",
                "swim_distance": 750.0,
                "bike_distance": 20000.0,
                "run_distance": 5000.0,
                "athlete_id": 10,
                "full_name": "Kendall Gretsch H2",
                "finish_status": "FINISH",
                "finish_position": 1,
                # Swim split is effort; adjusted total includes factor (3:38)
                "swimtime": "00:11:51",
                "t1time": "00:00:51",
                "biketime": "00:35:37",
                "t2time": "00:00:33",
                "runtime": "00:12:55",
                "total_time": "01:05:25",
            }
        ]
    )

    out = compute_metrics(df)
    row = out.iloc[0]
    assert bool(row["time_factor_applied"]) is True
    assert float(row["time_factor_sec"]) == 218
    assert float(row["sum_splits_sec"]) == (11 * 60 + 51) + (0 * 60 + 51) + (35 * 60 + 37) + (0 * 60 + 33) + (12 * 60 + 55)
    assert float(row["total_time_sec_adjusted"]) == (65 * 60 + 25)
    assert float(row["total_time_sec_raw"]) == float(row["sum_splits_sec"])
    assert float(row["time_factor_segment_sec"]) == 218


def test_time_factor_not_applied_for_pre_2024_years():
    from para_triathlon_analysis.para_standards import compute_metrics

    df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "prog_id": 1,
                "event_name": "Test Event",
                "event_date": "2023-06-01",
                "prog_name": "PTWC Women",
                "prog_distance_category": "Sprint",
                "swim_distance": 750.0,
                "bike_distance": 20000.0,
                "run_distance": 5000.0,
                "athlete_id": 10,
                "full_name": "Kendall Gretsch H2",
                "finish_status": "FINISH",
                "finish_position": 1,
                "swimtime": "00:12:00",
                "t1time": "00:00:30",
                "biketime": "00:30:00",
                "t2time": "00:00:30",
                "runtime": "00:15:00",
                "total_time": "01:00:00",
            }
        ]
    )

    out = compute_metrics(df)
    row = out.iloc[0]
    assert row["para_subclass"] == "H2"
    assert pd.isna(row["time_factor_sec"])
    assert pd.isna(row["time_factor_year"])
    assert bool(row["time_factor_applied"]) is False


def test_time_to_seconds_formats():
    assert time_to_seconds("0:59") == 59
    assert time_to_seconds("1:00") == 60
    assert time_to_seconds("1:02:03") == 3723
    assert time_to_seconds("15") == 15
    assert time_to_seconds(12) == 12
    assert time_to_seconds(None) is None
    assert time_to_seconds("") is None


def test_seconds_to_hms():
    assert seconds_to_hms(59) == "00:59"
    assert seconds_to_hms(60) == "01:00"
    assert seconds_to_hms(3723) == "1:02:03"
    assert seconds_to_hms(None) is None


def test_pace_calcs():
    # swim: 1500m in 18:00 -> 1080s -> 72.0 sec/100m
    assert pace_sec_per_100m(1080, 1500.0) == 72.0

    # run: 10k in 35:00 -> 2100s -> 210 sec/km
    assert pace_sec_per_km(2100, 10000.0) == 210.0

    assert pace_sec_per_km(None, 10000.0) is None
    assert pace_sec_per_km(100, None) is None
    assert pace_sec_per_km(100, 0.0) is None
