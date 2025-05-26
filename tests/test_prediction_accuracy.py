import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# tests/test_prediction_accuracy.py
"""
Test suite for model prediction accuracy and position prediction.
"""
import numpy as np
import pytest
from models.pipeline.trainer import ModelTrainer

def test_model_accuracy():
    import pandas as pd
    from models.validation.metrics import mae_within_percent
    from models.validation.cross_validator import TemporalCrossValidator
    # Load real feature-engineered dataset
    df = pd.read_parquet('data/ml_model_dataset.parquet')
    # Use only rows with next_time_sec label
    df = df.dropna(subset=['next_time_sec'])
    # Use only numeric features for modeling
    exclude_cols = ['athlete_id', 'EventID', 'EventDate', 'next_time_sec', 'next_position', 'next_time', 'Position', 'TotalTime', 'SwimTime', 'BikeTime', 'RunTime']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].select_dtypes(include=[float, int])
    y = df['next_time_sec']
    # Cross-validated MAE using temporal splits
    cv = TemporalCrossValidator(n_splits=5, time_col='EventDate', group_col='athlete_id')
    maes = []
    preds = []
    trues = []
    from models.pipeline.trainer import ModelTrainer
    for train_idx, test_idx in cv.split(df):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # Use a simple model for now: RandomForestRegressor on all features
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        maes.append(np.mean(np.abs(y_test - y_pred)))
        preds.extend(y_pred)
        trues.extend(y_test)
    mean_time = np.mean(trues)
    mae = np.mean(np.abs(np.array(trues) - np.array(preds)))
    assert mae < 0.05 * mean_time, f"MAE {mae:.2f} exceeds 5% of mean finish time {mean_time:.2f}"

def test_position_prediction():
    import pandas as pd
    from models.validation.metrics import position_accuracy
    from models.validation.cross_validator import TemporalCrossValidator
    # Load real feature-engineered dataset
    df = pd.read_parquet('data/ml_model_dataset.parquet')
    df = df.dropna(subset=['next_position'])
    # Use only numeric features for modeling
    exclude_cols = ['athlete_id', 'EventID', 'EventDate', 'next_time_sec', 'next_position', 'next_time', 'Position', 'TotalTime', 'SwimTime', 'BikeTime', 'RunTime']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].select_dtypes(include=[float, int])
    y = df['next_position']
    cv = TemporalCrossValidator(n_splits=5, time_col='EventDate', group_col='athlete_id')
    preds = []
    trues = []
    from sklearn.ensemble import RandomForestRegressor
    for train_idx, test_idx in cv.split(df):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds.extend(y_pred)
        trues.extend(y_test)
    within_3 = position_accuracy(np.array(trues), np.array(preds), tolerance=3)
    assert within_3 >= 0.8, f"Only {within_3*100:.1f}% of predictions within ±3 places"
