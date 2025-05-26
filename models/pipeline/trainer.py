# models/pipeline/trainer.py
"""
ModelTrainer: End-to-end training pipeline for triathlon race prediction models.
Trains DNF, split, and position models in sequence.
"""
from models.dnf_classifier import DNFRiskClassifier
from models.split_predictor import SplitTimePredictor
from models.position_predictor import PositionPredictor
import pandas as pd

class ModelTrainer:
    def __init__(self):
        self.dnf_model = DNFRiskClassifier()
        self.split_models = {
            'SwimTime': SplitTimePredictor('SwimTime'),
            'BikeTime': SplitTimePredictor('BikeTime'),
            'RunTime': SplitTimePredictor('RunTime'),
        }
        self.position_model = PositionPredictor()

    def train_dnf_model(self, X, y):
        self.dnf_model.fit(X, y)

    def train_split_models(self, X_dict, y_dict):
        for split, model in self.split_models.items():
            model.fit(X_dict[split], y_dict[split])

    def train_position_model(self, X, y):
        self.position_model.fit(X, y)

    def train_all_models(self, dnf_X, dnf_y, split_X_dict, split_y_dict, pos_X, pos_y):
        self.train_dnf_model(dnf_X, dnf_y)
        self.train_split_models(split_X_dict, split_y_dict)
        self.train_position_model(pos_X, pos_y)
