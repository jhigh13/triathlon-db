"""
Unit tests for ml.preprocessing module

Tests the TriathlonPreprocessor class and its methods for feature preprocessing,
scaling, imputation, and PCA dimensionality reduction.
"""

import pytest
import pandas as pd
import numpy as np
from ml.preprocessing import TriathlonPreprocessor, create_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


@pytest.fixture
def sample_feature_data():
    """Create sample engineered feature data for testing."""
    return pd.DataFrame({
        'TotalTime_sec': [3600, 3700, 3550, 3800, 3650],
        'days_since_last': [30, 45, 20, 60, 35],
        'age': [25, 30, 28, 32, 27],
        'rolling3_time_dist': [3580, 3680, 3570, 3750, 3620],
        'rolling5_time_dist': [3590, 3690, 3580, 3760, 3630],
        'rolling5_time_all': [3585, 3685, 3575, 3755, 3625],
        'rolling3_pos': [5.0, 8.0, 3.0, 12.0, 6.0],
        'rolling5_pos': [6.0, 9.0, 4.0, 13.0, 7.0],
        'recent_5_DNF_rate': [0.1, 0.0, 0.2, 0.0, 0.1],
        # One-hot encoded categorical features
        'rt_Triathlon': [1, 1, 1, 1, 1],
        'rt_Duathlon': [0, 0, 0, 0, 0],
        'dist_Sprint': [1, 0, 1, 0, 1],
        'dist_Standard': [0, 1, 0, 1, 0],
        'mode_individual': [1, 1, 1, 1, 1],
        'mode_relay': [0, 0, 0, 0, 0],
        # Target and grouping columns
        'next_time_sec': [3580, 3650, 3590, 3720, 3610],
        'next_position': [4, 7, 3, 11, 5],
        'athlete_id': [1, 2, 3, 4, 5]
    })


@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values for testing imputation."""
    return pd.DataFrame({
        'TotalTime_sec': [3600, np.nan, 3550, 3800, 3650],
        'days_since_last': [30, 45, np.nan, 60, 35],
        'age': [25, 30, 28, np.nan, 27],
        'rolling3_time_dist': [3580, 3680, 3570, 3750, np.nan],
        'rolling5_time_dist': [3590, 3690, 3580, 3760, 3630],
        'rolling5_time_all': [3585, 3685, 3575, 3755, 3625],
        'rolling3_pos': [5.0, 8.0, 3.0, 12.0, 6.0],
        'rolling5_pos': [6.0, 9.0, 4.0, 13.0, 7.0],
        'recent_5_DNF_rate': [0.1, 0.0, 0.2, 0.0, 0.1],
        # Categorical features
        'rt_Triathlon': [1, 1, 1, 1, 1],
        'dist_Sprint': [1, 0, 1, 0, 1],
        'dist_Standard': [0, 1, 0, 1, 0],
        'mode_individual': [1, 1, 1, 1, 1],
        # Target and grouping
        'next_time_sec': [3580, 3650, 3590, 3720, 3610],
        'athlete_id': [1, 2, 3, 4, 5]
    })


def test_preprocessor_initialization():
    """Test TriathlonPreprocessor initialization with different parameters."""
    # Default initialization
    preprocessor = TriathlonPreprocessor()
    assert preprocessor.apply_pca is True
    assert preprocessor.pca_components is None
    assert preprocessor.preprocessor is None
    assert preprocessor.feature_names is None
    
    # Custom initialization
    preprocessor_custom = TriathlonPreprocessor(apply_pca=False, pca_components=10)
    assert preprocessor_custom.apply_pca is False
    assert preprocessor_custom.pca_components == 10


def test_factory_function():
    """Test the create_preprocessor factory function."""
    preprocessor = create_preprocessor(apply_pca=True, pca_components=5)
    assert isinstance(preprocessor, TriathlonPreprocessor)
    assert preprocessor.apply_pca is True
    assert preprocessor.pca_components == 5


def test_get_feature_columns(sample_feature_data):
    """Test feature column identification."""
    preprocessor = TriathlonPreprocessor()
    numeric_features, categorical_features = preprocessor.get_feature_columns(sample_feature_data)
    
    # Check that numeric features are identified correctly
    expected_numeric = [
        'TotalTime_sec', 'days_since_last', 'age', 'rolling3_time_dist',
        'rolling5_time_dist', 'rolling5_time_all', 'rolling3_pos', 
        'rolling5_pos', 'recent_5_DNF_rate'
    ]
    assert set(numeric_features) == set(expected_numeric)
    
    # Check that categorical features are identified correctly
    expected_categorical = [
        'rt_Triathlon', 'rt_Duathlon', 'dist_Sprint', 'dist_Standard',
        'mode_individual', 'mode_relay'
    ]
    assert set(categorical_features) == set(expected_categorical)


def test_get_feature_columns_with_missing_columns():
    """Test feature column identification when some expected columns are missing."""
    # Create data with only some expected columns
    partial_data = pd.DataFrame({
        'TotalTime_sec': [3600, 3700],
        'age': [25, 30],
        'dist_Sprint': [1, 0],
        'mode_individual': [1, 1]
    })
    
    preprocessor = TriathlonPreprocessor()
    numeric_features, categorical_features = preprocessor.get_feature_columns(partial_data)
    
    # Should only include columns that actually exist
    assert 'TotalTime_sec' in numeric_features
    assert 'age' in numeric_features
    assert 'rolling3_time_dist' not in numeric_features  # Missing column
    
    assert 'dist_Sprint' in categorical_features
    assert 'mode_individual' in categorical_features
    assert 'rt_Triathlon' not in categorical_features  # Missing column


def test_create_preprocessing_pipeline(sample_feature_data):
    """Test preprocessing pipeline creation."""
    preprocessor = TriathlonPreprocessor(apply_pca=True, pca_components=5)
    numeric_features, categorical_features = preprocessor.get_feature_columns(sample_feature_data)
    
    pipeline = preprocessor.create_preprocessing_pipeline(numeric_features, categorical_features)
    
    assert isinstance(pipeline, Pipeline)
    assert 'preprocessing' in pipeline.named_steps
    assert 'pca' in pipeline.named_steps  # Should have PCA step


def test_create_preprocessing_pipeline_no_pca(sample_feature_data):
    """Test preprocessing pipeline creation without PCA."""
    preprocessor = TriathlonPreprocessor(apply_pca=False)
    numeric_features, categorical_features = preprocessor.get_feature_columns(sample_feature_data)
    
    pipeline = preprocessor.create_preprocessing_pipeline(numeric_features, categorical_features)
    
    assert isinstance(pipeline, Pipeline)
    assert 'preprocessing' in pipeline.named_steps
    assert 'pca' not in pipeline.named_steps  # Should not have PCA step


def test_prepare_features_and_target(sample_feature_data):
    """Test feature and target preparation."""
    preprocessor = TriathlonPreprocessor()
    X, y, groups = preprocessor.prepare_features_and_target(sample_feature_data, 'next_time_sec')
    
    # Check shapes and types
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(groups, pd.Series)
    assert len(X) == len(y) == len(groups) == 5
    
    # Check that target values are correct
    assert y.name == 'next_time_sec'
    assert list(y.values) == [3580, 3650, 3590, 3720, 3610]
    
    # Check that groups are athlete IDs
    assert list(groups.values) == [1, 2, 3, 4, 5]
    
    # Check that feature names are stored
    assert preprocessor.feature_names is not None
    assert len(preprocessor.feature_names) > 0


def test_prepare_features_with_alternative_target(sample_feature_data):
    """Test feature preparation with alternative target column."""
    preprocessor = TriathlonPreprocessor()
    X, y, groups = preprocessor.prepare_features_and_target(sample_feature_data, 'next_position')
    
    assert y.name == 'next_position'
    assert list(y.values) == [4, 7, 3, 11, 5]


def test_fit_preprocessor(sample_feature_data):
    """Test fitting the preprocessor."""
    preprocessor = TriathlonPreprocessor(apply_pca=True, pca_components=5)
    X, y, groups = preprocessor.prepare_features_and_target(sample_feature_data)
    
    fitted_pipeline = preprocessor.fit_preprocessor(X, y)
    
    assert isinstance(fitted_pipeline, Pipeline)
    assert preprocessor.preprocessor is not None
    assert 'preprocessing' in fitted_pipeline.named_steps
    assert 'pca' in fitted_pipeline.named_steps
    
    # Check that PCA was fitted
    pca_step = fitted_pipeline.named_steps['pca']
    assert isinstance(pca_step, PCA)
    assert pca_step.n_components_ <= 5  # Should have at most 5 components


def test_transform_features(sample_feature_data):
    """Test feature transformation."""
    preprocessor = TriathlonPreprocessor(apply_pca=True, pca_components=3)
    X, y, groups = preprocessor.prepare_features_and_target(sample_feature_data)
    
    # Fit then transform
    preprocessor.fit_preprocessor(X, y)
    X_transformed = preprocessor.transform_features(X)
    
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == 5  # Same number of samples
    assert X_transformed.shape[1] <= 3  # Should have at most 3 components after PCA


def test_transform_features_not_fitted():
    """Test that transform_features raises error when not fitted."""
    preprocessor = TriathlonPreprocessor()
    sample_X = pd.DataFrame({'feature1': [1, 2, 3]})
    
    with pytest.raises(ValueError, match="Preprocessor not fitted"):
        preprocessor.transform_features(sample_X)


def test_fit_transform(sample_feature_data):
    """Test fit_transform method."""
    preprocessor = TriathlonPreprocessor(apply_pca=True, pca_components=4)
    X, y, groups = preprocessor.prepare_features_and_target(sample_feature_data)
    
    X_transformed = preprocessor.fit_transform(X, y)
    
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == 5
    assert X_transformed.shape[1] <= 4
    assert preprocessor.preprocessor is not None


def test_preprocessing_with_missing_values(sample_data_with_missing):
    """Test preprocessing handles missing values correctly."""
    preprocessor = TriathlonPreprocessor(apply_pca=False)
    X, y, groups = preprocessor.prepare_features_and_target(sample_data_with_missing)
    
    # Should handle missing values without error
    X_transformed = preprocessor.fit_transform(X, y)
    
    assert isinstance(X_transformed, np.ndarray)
    assert not np.isnan(X_transformed).any()  # No missing values after preprocessing
    assert X_transformed.shape[0] == 5


def test_preprocessing_no_pca(sample_feature_data):
    """Test preprocessing without PCA."""
    preprocessor = TriathlonPreprocessor(apply_pca=False)
    X, y, groups = preprocessor.prepare_features_and_target(sample_feature_data)
    
    X_transformed = preprocessor.fit_transform(X, y)
    
    # Without PCA, should have same number of features as input
    expected_features = len(preprocessor.feature_names)
    assert X_transformed.shape[1] == expected_features


def test_preprocessing_pca_variance_retention():
    """Test PCA with variance retention (95% default)."""
    # Create data with more features to test PCA variance retention
    high_dim_data = pd.DataFrame({
        'TotalTime_sec': np.random.normal(3600, 100, 20),
        'days_since_last': np.random.normal(30, 10, 20),
        'age': np.random.normal(28, 5, 20),
        'rolling3_time_dist': np.random.normal(3580, 100, 20),
        'rolling5_time_dist': np.random.normal(3590, 100, 20),
        'rolling5_time_all': np.random.normal(3585, 100, 20),
        'rolling3_pos': np.random.normal(6, 3, 20),
        'rolling5_pos': np.random.normal(7, 3, 20),
        'recent_5_DNF_rate': np.random.uniform(0, 0.3, 20),
        'rt_Triathlon': [1] * 20,
        'dist_Sprint': np.random.choice([0, 1], 20),
        'dist_Standard': np.random.choice([0, 1], 20),
        'mode_individual': [1] * 20,
        'next_time_sec': np.random.normal(3600, 100, 20),
        'athlete_id': range(1, 21)
    })
    
    preprocessor = TriathlonPreprocessor(apply_pca=True, pca_components=None)  # Auto-select
    X, y, groups = preprocessor.prepare_features_and_target(high_dim_data)
    
    X_transformed = preprocessor.fit_transform(X, y)
    
    # Should have fewer components than original features
    assert X_transformed.shape[1] < X.shape[1]
    
    # Check that PCA explained variance is stored
    pca_step = preprocessor.preprocessor.named_steps['pca']
    assert pca_step.explained_variance_ratio_.sum() >= 0.90  # Should explain at least 90% variance


def test_column_name_handling():
    """Test that column names are properly converted to strings."""
    # Create data with potential pandas Index column names
    data = pd.DataFrame({
        'TotalTime_sec': [3600, 3700],
        'age': [25, 30],
        'dist_Sprint': [1, 0],
        'next_time_sec': [3580, 3650],
        'athlete_id': [1, 2]
    })
    
    # Force column names to be Index objects (simulate pandas get_dummies output)
    data.columns = pd.Index(data.columns)
    
    preprocessor = TriathlonPreprocessor()
    X, y, groups = preprocessor.prepare_features_and_target(data)
    
    # Should handle Index objects and convert to strings
    assert all(isinstance(col, str) for col in X.columns)
    assert preprocessor.feature_names is not None
    assert all(isinstance(name, str) for name in preprocessor.feature_names)
