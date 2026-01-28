"""
Tests for the prediction pipeline.

Run with: pytest tests/test_prediction.py -v
"""

import pytest
from datetime import date, timedelta
import pandas as pd
import numpy as np

from tri_analysis.prediction.utils_time import (
    parse_time_to_seconds,
    seconds_to_hms,
    parse_time_columns,
)


class TestParseTimeToSeconds:
    """Test suite for parse_time_to_seconds function."""

    def test_hh_mm_ss(self):
        assert parse_time_to_seconds("1:02:03") == 3723
        assert parse_time_to_seconds("01:02:03") == 3723
        assert parse_time_to_seconds("0:59:59") == 3599

    def test_mm_ss(self):
        assert parse_time_to_seconds("59:30") == 3570
        assert parse_time_to_seconds("05:30") == 330
        assert parse_time_to_seconds("0:45") == 45

    def test_ss_only(self):
        assert parse_time_to_seconds("45") == 45
        assert parse_time_to_seconds("0") == 0

    def test_none_and_empty(self):
        assert parse_time_to_seconds(None) is None
        assert parse_time_to_seconds("") is None
        assert parse_time_to_seconds("   ") is None

    def test_invalid_strings(self):
        assert parse_time_to_seconds("DNF") is None
        assert parse_time_to_seconds("DNS") is None
        assert parse_time_to_seconds("DSQ") is None
        assert parse_time_to_seconds("LAP") is None
        assert parse_time_to_seconds("invalid") is None

    def test_numeric_input(self):
        assert parse_time_to_seconds(3600) == 3600
        assert parse_time_to_seconds(3600.5) == 3600
        assert parse_time_to_seconds(0) == 0

    def test_nan_input(self):
        assert parse_time_to_seconds(float("nan")) is None

    def test_bool_input(self):
        # bool is subclass of int, should return None
        assert parse_time_to_seconds(True) is None
        assert parse_time_to_seconds(False) is None


class TestSecondsToHms:
    """Test suite for seconds_to_hms function."""

    def test_with_hours(self):
        assert seconds_to_hms(3723) == "1:02:03"
        assert seconds_to_hms(3600) == "1:00:00"
        assert seconds_to_hms(7200) == "2:00:00"

    def test_without_hours(self):
        assert seconds_to_hms(3570) == "59:30"
        assert seconds_to_hms(330) == "05:30"
        assert seconds_to_hms(45) == "00:45"
        assert seconds_to_hms(0) == "00:00"

    def test_none_input(self):
        assert seconds_to_hms(None) is None

    def test_negative(self):
        assert seconds_to_hms(-60) == "-01:00"
        assert seconds_to_hms(-3661) == "-1:01:01"

    def test_float_input(self):
        assert seconds_to_hms(3600.4) == "1:00:00"
        assert seconds_to_hms(3600.6) == "1:00:01"  # rounds

    def test_nan_input(self):
        assert seconds_to_hms(float("nan")) is None


class TestParseTimeColumns:
    """Test suite for parse_time_columns helper."""

    def test_basic_parsing(self):
        df = pd.DataFrame({
            "swimtime": ["18:30", "19:00", "DNF"],
            "biketime": ["59:30", "1:00:00", ""],
        })
        parse_time_columns(df, ["swimtime", "biketime"])
        
        assert df["swimtime_sec"].iloc[0] == 1110
        assert df["swimtime_sec"].iloc[1] == 1140
        assert pd.isna(df["swimtime_sec"].iloc[2])
        
        assert df["biketime_sec"].iloc[0] == 3570
        assert df["biketime_sec"].iloc[1] == 3600
        assert pd.isna(df["biketime_sec"].iloc[2])

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        # Should not raise, just skip missing columns
        parse_time_columns(df, ["swimtime"])
        assert "swimtime_sec" not in df.columns


class TestFeatures:
    """Test suite for feature engineering."""

    def test_compute_athlete_form_features_empty(self):
        from tri_analysis.prediction.features import compute_athlete_form_features
        
        empty_df = pd.DataFrame()
        features = compute_athlete_form_features(empty_df, date.today())
        
        assert features["ema_swim_sec_5"] is None
        assert features["races_12m"] == 0

    def test_compute_athlete_form_features_basic(self):
        from tri_analysis.prediction.features import compute_athlete_form_features
        
        # Create mock history with 3 races
        history_df = pd.DataFrame({
            "event_date": [
                date.today() - timedelta(days=30),
                date.today() - timedelta(days=60),
                date.today() - timedelta(days=90),
            ],
            "swimtime": ["18:30", "18:45", "19:00"],
            "biketime": ["59:00", "59:30", "1:00:00"],
            "runtime": ["32:00", "32:30", "33:00"],
            "total_time": ["1:52:00", "1:53:15", "1:54:30"],
            "finish_status": ["FINISH", "FINISH", "FINISH"],
        })
        
        features = compute_athlete_form_features(history_df, date.today())
        
        # Should have computed EWMAs
        assert features["ema_swim_sec_5"] is not None
        assert features["ema_bike_sec_5"] is not None
        assert features["ema_run_sec_5"] is not None
        assert features["ema_total_sec_5"] is not None
        
        # Should have last total (most recent race)
        assert features["last_total_sec"] == parse_time_to_seconds("1:52:00")
        
        # Should have race counts
        assert features["races_12m"] == 3
        
        # Days since last race
        assert features["days_since_last_race"] == 30


class TestSimulation:
    """Test suite for Monte Carlo simulation."""

    def test_run_monte_carlo_basic(self):
        from tri_analysis.prediction.simulate import run_monte_carlo
        
        # Create mock prediction data
        pred_df = pd.DataFrame({
            "athlete_id": [1, 2, 3],
            "athlete_full_name": ["Alice", "Bob", "Charlie"],
            "pred_total_sec": [6600, 6700, 6800],  # Alice fastest
            "std_total_sec_24m": [60, 80, 100],
            "front_pack_rate": [0.8, 0.5, 0.3],
        })
        
        result_df = run_monte_carlo(pred_df, n_sims=1000, random_state=42)
        
        # Should have probability columns
        assert "prob_win" in result_df.columns
        assert "prob_podium" in result_df.columns
        assert "prob_top5" in result_df.columns
        assert "expected_rank" in result_df.columns
        
        # Alice (fastest predicted) should have highest win prob
        alice = result_df[result_df["athlete_id"] == 1].iloc[0]
        bob = result_df[result_df["athlete_id"] == 2].iloc[0]
        assert alice["prob_win"] > bob["prob_win"]
        
        # Probabilities should sum to ~1 across athletes
        assert abs(result_df["prob_win"].sum() - 1.0) < 0.01
        
        # Rank intervals should be reasonable
        assert all(result_df["rank_p10"] <= result_df["rank_p90"])


class TestModelBundle:
    """Test suite for model training and persistence."""

    def test_model_bundle_save_load(self, tmp_path):
        from tri_analysis.prediction.train import (
            ModelBundle,
            save_model_bundle,
            load_model_bundle,
        )
        
        # Create a mock bundle
        bundle = ModelBundle(
            model_swim=None,
            model_bike=None,
            model_run=None,
            model_total=None,
            feature_columns=["feat1", "feat2"],
            version="test_v1",
            metadata={"test": "data"},
        )
        
        # Save and load
        path = str(tmp_path / "test_bundle.joblib")
        save_model_bundle(bundle, path)
        loaded = load_model_bundle(path)
        
        assert loaded.feature_columns == ["feat1", "feat2"]
        assert loaded.version == "test_v1"
        assert loaded.metadata == {"test": "data"}


class TestEvaluation:
    """Test suite for evaluation metrics."""

    def test_precision_at_k(self):
        from tri_analysis.prediction.evaluate import precision_at_k
        
        # Perfect prediction
        assert precision_at_k([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], k=3) == 1.0
        
        # Completely wrong top-3
        assert precision_at_k([4, 5, 6, 1, 2], [1, 2, 3, 4, 5], k=3) == 0.0
        
        # Partial overlap
        assert precision_at_k([1, 4, 3, 2, 5], [1, 2, 3, 4, 5], k=3) == pytest.approx(2/3, rel=0.01)

    def test_spearman_rank_corr(self):
        from tri_analysis.prediction.evaluate import spearman_rank_corr
        
        # Perfect correlation
        pred = pd.Series([1, 2, 3, 4, 5], index=[10, 20, 30, 40, 50])
        true = pd.Series([1, 2, 3, 4, 5], index=[10, 20, 30, 40, 50])
        assert spearman_rank_corr(pred, true) == pytest.approx(1.0)
        
        # Perfect negative correlation
        pred = pd.Series([5, 4, 3, 2, 1], index=[10, 20, 30, 40, 50])
        true = pd.Series([1, 2, 3, 4, 5], index=[10, 20, 30, 40, 50])
        assert spearman_rank_corr(pred, true) == pytest.approx(-1.0)
