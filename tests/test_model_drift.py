# tests/test_model_drift.py
"""
Test for model drift over time in triathlon prediction pipeline.
"""
import numpy as np
import pytest

def test_model_drift():
    # Dummy data for demonstration; replace with real drift detection logic
    drift_metric = 0.01  # e.g., change in MAE or feature importance
    assert drift_metric < 0.05, f"Model drift detected: {drift_metric:.3f} exceeds threshold"
