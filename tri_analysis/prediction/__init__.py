"""
Triathlon Race Prediction Package (WTCS Draft-Legal, Pre-Race)

Modules:
- utils_time: Time string parsing and formatting
- sql: Database queries for results, history, pack membership, start lists
- features: Feature engineering (athlete form, pack metrics, field context)
- train: Model training and persistence
- predict: Deterministic predictions
- simulate: Monte Carlo simulation for probabilities
"""

from .utils_time import parse_time_to_seconds, seconds_to_hms
from .sql import ProgramKey

__all__ = [
    "parse_time_to_seconds",
    "seconds_to_hms",
    "ProgramKey",
]
