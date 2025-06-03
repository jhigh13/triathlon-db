"""
Configuration Module for Triathlon ML Pipeline

This module contains configuration settings for the machine learning pipeline,
including model parameters, feature engineering settings, and evaluation criteria.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""

    # Target performance criteria (MAE in seconds)
    target_mae_sprint: float = 600.0
    target_mae_standard: float = 600.0
    target_mae_super_sprint: float = 600.0

    # Cross-validation settings
    cv_folds: int = 5
    random_state: int = 42

    # Model hyperparameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_n_jobs: int = -1

    # PCA settings
    pca_variance_threshold: float = 0.95
    pca_components: Optional[int] = None  # None for auto-selection


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Rolling window sizes
    rolling_windows: List[int] = None

    # Minimum time threshold (seconds) - filters out unrealistic times
    min_time_threshold: float = 600.0  # 10 minutes

    # DNF rolling window
    dnf_rolling_window: int = 5

    # Allowed event types
    allowed_race_types: List[str] = None
    allowed_distances: List[str] = None
    allowed_event_modes: List[str] = None

    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [3, 5]

        if self.allowed_race_types is None:
            self.allowed_race_types = ["Triathlon"]

        if self.allowed_distances is None:
            self.allowed_distances = ["Sprint", "Standard", "Super Sprint"]

        if self.allowed_event_modes is None:
            self.allowed_event_modes = [
                "individual",
                "mixed_relay",
                "paratriathlon",
                "para_mixed_relay",
            ]


@dataclass
class DataConfig:
    """Configuration for data extraction and processing."""

    # Database connection settings (uses existing database.py config)
    use_existing_engine: bool = True

    # Data filtering settings
    min_races_per_athlete: int = 2  # Minimum races to include athlete in training
    exclude_dnf_from_training: bool = True

    # Train/validation split
    test_size: float = 0.2
    validation_method: str = "group_kfold"  # or "time_split"

    # Data caching
    cache_processed_data: bool = True
    cache_directory: str = "data/cache"


@dataclass
class MLPipelineConfig:
    """Complete configuration for the ML pipeline."""

    model: ModelConfig = None
    feature: FeatureConfig = None
    data: DataConfig = None

    # Output settings
    model_output_dir: str = "models"
    results_output_dir: str = "results"
    log_level: str = "INFO"

    # Experiment tracking
    use_mlflow: bool = False
    mlflow_experiment_name: str = "triathlon_race_prediction"
    mlflow_tracking_uri: str = "file:./mlruns"

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.feature is None:
            self.feature = FeatureConfig()
        if self.data is None:
            self.data = DataConfig()


# Default configuration instance
DEFAULT_CONFIG = MLPipelineConfig()


def get_config() -> MLPipelineConfig:
    """
    Get the default configuration.

    Returns:
        MLPipelineConfig instance
    """
    return DEFAULT_CONFIG


def get_target_mae_for_distance(distance: str) -> float:
    """
    Get target MAE for a specific distance.

    Args:
        distance: Distance name (Sprint, Standard, Super Sprint)

    Returns:
        Target MAE in seconds
    """
    config = get_config()

    distance_targets = {
        "Sprint": config.model.target_mae_sprint,
        "Standard": config.model.target_mae_standard,
        "Super Sprint": config.model.target_mae_super_sprint,
    }

    return distance_targets.get(distance, 600.0)


def validate_config(config: MLPipelineConfig) -> bool:
    """
    Validate configuration settings.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    # Validate model config
    if config.model.cv_folds < 2:
        raise ValueError("cv_folds must be >= 2")

    if (
        config.model.pca_variance_threshold <= 0
        or config.model.pca_variance_threshold > 1
    ):
        raise ValueError("pca_variance_threshold must be between 0 and 1")

    # Validate feature config
    if config.feature.min_time_threshold <= 0:
        raise ValueError("min_time_threshold must be positive")

    if not all(isinstance(w, int) and w > 0 for w in config.feature.rolling_windows):
        raise ValueError("rolling_windows must be positive integers")

    # Validate data config
    if config.data.test_size <= 0 or config.data.test_size >= 1:
        raise ValueError("test_size must be between 0 and 1")

    if config.data.min_races_per_athlete < 1:
        raise ValueError("min_races_per_athlete must be >= 1")

    return True


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("Default configuration:")
    print(f"Target MAE for Sprint: {get_target_mae_for_distance('Sprint')} seconds")
    print(f"CV folds: {config.model.cv_folds}")
    print(f"Rolling windows: {config.feature.rolling_windows}")
    print(f"PCA variance threshold: {config.model.pca_variance_threshold}")

    # Validate configuration
    try:
        validate_config(config)
        print("Configuration is valid ✓")
    except ValueError as e:
        print(f"Configuration error: {e}")
