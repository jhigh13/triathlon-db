# models/split_predictor.py
"""
Split Time Predictor for triathlon race pipeline.
Implements a wrapper for LightGBM regression model for split (swim, bike, run) prediction.
"""
import lightgbm as lgb

class SplitTimePredictor:
    def __init__(self, split_name):
        self.model = lgb.LGBMRegressor()
        self.split = split_name
        self.features = [
            f'{split_name}_sec',
            f'{split_name}_rolling3',
            'age_at_race',
            'days_since_last',
            # Add more features as needed
        ]

    def fit(self, X, y):
        self.model.fit(X[self.features], y)

    def predict(self, X):
        return self.model.predict(X[self.features])
