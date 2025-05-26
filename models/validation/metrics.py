# models/validation/metrics.py
"""
Custom metrics for model evaluation in triathlon race prediction pipeline.
Includes MAE, position accuracy, and confidence interval coverage.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

def mae_within_percent(y_true, y_pred, percent=0.05):
    mean_time = np.mean(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    return mae < percent * mean_time

def position_accuracy(y_true, y_pred, tolerance=3):
    return np.mean(np.abs(y_true - y_pred) <= tolerance)

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)
