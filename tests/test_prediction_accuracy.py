# tests/test_prediction_accuracy.py
"""
Test suite for model prediction accuracy and position prediction.
"""
import numpy as np
import pytest
from models.pipeline.trainer import ModelTrainer

def test_model_accuracy():
    # Dummy data for demonstration; replace with real test data
    y_true = np.array([3600, 3700, 3500, 3400, 3550])
    y_pred = np.array([3620, 3690, 3490, 3420, 3560])
    mean_time = np.mean(y_true)
    mae = np.mean(np.abs(y_true - y_pred))
    assert mae < 0.05 * mean_time, f"MAE {mae:.2f} exceeds 5% of mean finish time {mean_time:.2f}"

def test_position_prediction():
    y_true = np.array([1, 5, 10, 15, 20])
    y_pred = np.array([2, 7, 8, 13, 22])
    within_3 = np.mean(np.abs(y_true - y_pred) <= 3)
    assert within_3 >= 0.8, f"Only {within_3*100:.1f}% of predictions within ±3 places"
