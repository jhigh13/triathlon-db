"""
Model training for the prediction pipeline.

Trains baseline regression models for swim, bike, run, and total time prediction.
Uses sklearn HistGradientBoostingRegressor as the default (no external dependencies).

ModelBundle dataclass stores all trained models and metadata for persistence.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import logging

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

# Try to import LightGBM; fall back to sklearn if not available
try:
    from lightgbm import LGBMRegressor
    USE_LIGHTGBM = True
    logger.info("Using LightGBM for regression models")
except ImportError:
    USE_LIGHTGBM = False
    logger.info("LightGBM not available, using sklearn HistGradientBoostingRegressor")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelBundle:
    """
    Container for all trained models and metadata.

    Attributes:
        model_swim: Regressor for swim_sec prediction
        model_bike: Regressor for bike_sec prediction
        model_run: Regressor for run_sec prediction
        model_total: Regressor for total_sec prediction
        model_t1: Regressor for t1_sec (optional, usually not trained)
        model_t2: Regressor for t2_sec (optional, usually not trained)
        model_front_pack: Classifier for front pack probability (optional)
        feature_columns: List of feature column names used in training
        created_at: Timestamp of model creation
        version: Version string
        metadata: Additional metadata (training params, metrics, etc.)
    """
    model_swim: Any
    model_bike: Any
    model_run: Any
    model_total: Any
    model_t1: Optional[Any] = None
    model_t2: Optional[Any] = None
    model_front_pack: Optional[Any] = None
    feature_columns: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "v1.0"
    metadata: dict = field(default_factory=dict)


def create_regressor() -> Pipeline:
    """
    Create a regression pipeline with imputation and model.

    Returns a Pipeline that:
    1. Imputes missing values with median
    2. Fits a gradient boosting regressor
    """
    if USE_LIGHTGBM:
        model = LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=6,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42,
        )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", model),
    ])

    return pipeline


def train_baseline_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: dict[str, str] | None = None,
    use_sample_weights: bool = True
) -> ModelBundle:
    """
    Train baseline regression models for split and total time prediction.

    Args:
        train_df: Training DataFrame with features and labels
        feature_cols: List of feature column names
        target_cols: Dict mapping model name to target column name
                     Default: {"swim": "swim_sec", "bike": "bike_sec", ...}
        use_sample_weights: If True, weight samples by event tier (WTCS counts more)

    Returns:
        ModelBundle with trained models

    Raises:
        ValueError: If required columns are missing
    """
    from .features import TIER_SAMPLE_WEIGHTS
    
    if target_cols is None:
        target_cols = {
            "swim": "swim_sec",
            "bike": "bike_sec",
            "run": "run_sec",
            "total": "total_sec",
        }

    # Verify feature columns exist
    missing_features = [c for c in feature_cols if c not in train_df.columns]
    if missing_features:
        logger.warning(f"Missing feature columns (will be filled): {missing_features}")

    # Prepare feature matrix
    X = train_df[feature_cols].copy()
    
    # Compute sample weights from event_tier
    sample_weights = None
    if use_sample_weights and "event_tier" in train_df.columns:
        sample_weights = train_df["event_tier"].map(TIER_SAMPLE_WEIGHTS).fillna(1.0)
        tier_counts = train_df["event_tier"].value_counts().sort_index()
        logger.info(f"Using tier-based sample weights: {dict(TIER_SAMPLE_WEIGHTS)}")
        logger.info(f"Samples by tier: {tier_counts.to_dict()}")

    # Train models for each target
    models = {}
    metrics = {}

    for name, target_col in target_cols.items():
        if target_col not in train_df.columns:
            logger.warning(f"Target column {target_col} not found, skipping {name} model")
            models[name] = None
            continue

        # Filter to rows with valid target
        mask = train_df[target_col].notna()
        X_train = X.loc[mask]
        y_train = train_df.loc[mask, target_col]
        weights_train = sample_weights.loc[mask] if sample_weights is not None else None

        if len(y_train) < 10:
            logger.warning(f"Insufficient training samples for {name} ({len(y_train)}), skipping")
            models[name] = None
            continue

        logger.info(f"Training {name} model on {len(y_train)} samples")

        model = create_regressor()
        # Pass sample weights to fit (works with LightGBM and sklearn)
        if weights_train is not None:
            model.fit(X_train, y_train, regressor__sample_weight=weights_train.values)
        else:
            model.fit(X_train, y_train)
        models[name] = model

        # Compute training metrics
        y_pred = model.predict(X_train)
        mae = np.mean(np.abs(y_train - y_pred))
        metrics[name] = {"mae": float(mae), "n_samples": len(y_train)}
        logger.info(f"  {name} training MAE: {mae:.1f} sec")

    # Build bundle
    bundle = ModelBundle(
        model_swim=models.get("swim"),
        model_bike=models.get("bike"),
        model_run=models.get("run"),
        model_total=models.get("total"),
        feature_columns=feature_cols,
        metadata={"training_metrics": metrics},
    )

    return bundle


def save_model_bundle(bundle: ModelBundle, path: str) -> str:
    """
    Save a ModelBundle to disk using joblib.

    Args:
        bundle: ModelBundle to save
        path: Output file path (e.g., 'models/bundle.joblib')

    Returns:
        The path where the bundle was saved
    """
    joblib.dump(bundle, path)
    logger.info(f"Saved model bundle to {path}")
    return path


def load_model_bundle(path: str) -> ModelBundle:
    """
    Load a ModelBundle from disk.

    Args:
        path: Path to the saved bundle file

    Returns:
        ModelBundle
    """
    bundle = joblib.load(path)
    logger.info(f"Loaded model bundle from {path} (version: {bundle.version})")
    return bundle


def build_training_dataset(
    engine,
    start_date: str,
    end_date: str,
    min_finishers: int = 10,
    elite_only: bool = True,
    distance_categories: list[str] = None,
    match_distance: bool = True
) -> pd.DataFrame:
    """
    Build a training dataset by fetching results and computing features for historical programs.

    Args:
        engine: SQLAlchemy Engine
        start_date: Start date for training window (YYYY-MM-DD)
        end_date: End date for training window (YYYY-MM-DD)
        min_finishers: Minimum finishers per program to include
        elite_only: If True, only train on Elite Men/Women programs (recommended)
        distance_categories: If provided, filter to these distances (e.g., ['sprint', 'standard'])
        match_distance: If True, filter athlete history to matching distance category

    Returns:
        DataFrame with features and labels for all athlete-program combinations
    """
    from .sql import fetch_training_programs, fetch_program_results, ProgramKey
    from .features import build_features_for_program, get_feature_columns
    from .utils_time import parse_time_to_seconds

    # Get list of programs to use for training
    programs_df = fetch_training_programs(
        engine, start_date, end_date, min_finishers,
        elite_only=elite_only,
        distance_categories=distance_categories
    )
    logger.info(f"Found {len(programs_df)} programs for training (elite_only={elite_only})")

    all_rows = []

    for _, prog_row in programs_df.iterrows():
        key = ProgramKey(event_id=prog_row["event_id"], prog_id=prog_row["prog_id"])

        try:
            # Build features (using results as athlete list, not start list)
            features_df = build_features_for_program(
                engine, key, 
                use_start_list=False,
                match_distance=match_distance,
                elite_only=elite_only
            )

            if features_df.empty:
                continue

            # Fetch actual results to get labels
            results_df = fetch_program_results(engine, key)

            # Parse time labels to seconds
            # Map original column names to target label names
            time_col_map = {
                "swimtime": "swim_sec",
                "t1time": "t1_sec",
                "biketime": "bike_sec",
                "t2time": "t2_sec",
                "runtime": "run_sec",
                "total_time": "total_sec",
            }
            for src_col, tgt_col in time_col_map.items():
                if src_col in results_df.columns:
                    results_df[tgt_col] = results_df[src_col].apply(parse_time_to_seconds)

            # Merge features with labels
            merged = features_df.merge(
                results_df[["athlete_id", "swim_sec", "t1_sec", "bike_sec", "t2_sec", "run_sec", "total_sec",
                            "finish_status", "finish_position", "position_sort"]],
                on="athlete_id",
                how="left"
            )

            # Only keep finishers for training
            merged = merged[merged["finish_status"] == "FINISH"]

            all_rows.append(merged)

        except Exception as e:
            logger.warning(f"Error processing {key}: {e}")
            continue

    if not all_rows:
        logger.error("No training data collected")
        return pd.DataFrame()

    train_df = pd.concat(all_rows, ignore_index=True)
    logger.info(f"Built training dataset with {len(train_df)} rows")

    return train_df
