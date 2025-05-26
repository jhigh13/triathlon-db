# models/validation/cross_validator.py
"""
Temporal cross-validation for time-based ML validation in triathlon prediction pipeline.
Splits data by time, preserving athlete grouping.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

class TemporalCrossValidator(BaseCrossValidator):
    def __init__(self, n_splits=5, time_col='EventDate', group_col='athlete_id'):
        self.n_splits = n_splits
        self.time_col = time_col
        self.group_col = group_col

    def split(self, X, y=None, groups=None):
        df = X.copy()
        df = df.sort_values(self.time_col)
        unique_times = df[self.time_col].sort_values().unique()
        fold_sizes = np.full(self.n_splits, len(unique_times) // self.n_splits, dtype=int)
        fold_sizes[:len(unique_times) % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            test_times = unique_times[current:current + fold_size]
            train_idx = df[~df[self.time_col].isin(test_times)].index
            test_idx = df[df[self.time_col].isin(test_times)].index
            # Ensure athlete grouping in test set
            test_athletes = set(df.loc[test_idx, self.group_col])
            train_idx = df[~df[self.group_col].isin(test_athletes)].index
            yield train_idx, test_idx
            current += fold_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
