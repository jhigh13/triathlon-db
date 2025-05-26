# models/dnf_classifier.py
"""
DNF (Did Not Finish) Risk Classifier for triathlon race prediction pipeline.
Implements a wrapper for XGBoost classifier with feature selection for DNF risk.
"""
from xgboost import XGBClassifier

class DNFRiskClassifier:
    def __init__(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.features = [
            'recent_5_DNF_rate',
            'days_since_last',
            'age_at_race',
            'rolling3_time_dist',
            'rolling5_time_dist',
            'rolling3_pos',
            'rolling5_pos',
            # Add more features as needed
        ]

    def fit(self, X, y):
        self.model.fit(X[self.features], y)

    def predict(self, X):
        return self.model.predict(X[self.features])

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.features])
