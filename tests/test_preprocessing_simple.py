"""
Simple preprocessing test to debug pytest collection issue
"""

import pandas as pd
import numpy as np
from ml.preprocessing import TriathlonPreprocessor, create_preprocessor

def test_simple_initialization():
    """Simple test to see if pytest collects it."""
    preprocessor = TriathlonPreprocessor()
    assert preprocessor.apply_pca is True

def test_create_sample_data():
    """Test creating sample data with pandas and numpy."""
    data = pd.DataFrame({
        'TotalTime_sec': [3600, 3700, 3550],
        'age': [25, 30, 28],
        'dist_Sprint': [1, 0, 1],
        'next_time_sec': [3580, 3650, 3590],
        'athlete_id': [1, 2, 3]
    })
    assert len(data) == 3
    assert 'TotalTime_sec' in data.columns
