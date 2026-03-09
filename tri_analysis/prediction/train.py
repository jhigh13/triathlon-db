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
    from lightgbm import LGBMRegressor, LGBMRanker
    USE_LIGHTGBM = True
    logger.info("Using LightGBM for regression models")
except ImportError:
    USE_LIGHTGBM = False
    logger.info("LightGBM not available, using sklearn HistGradientBoostingRegressor")

try:
    import xgboost as xgb
    USE_XGBOOST = True
except ImportError:
    USE_XGBOOST = False

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
    model_total_pct: Optional[Any] = None  # Percentile model (finish_position / n_finishers)
    model_ranker: Optional[Any] = None  # LGBMRanker for direct rank optimization
    ranker_imputer: Optional[Any] = None  # Imputer for ranker (not in pipeline)
    model_xgb_ranker: Optional[Any] = None  # XGBRanker for ensemble diversity
    xgb_ranker_imputer: Optional[Any] = None  # Imputer for XGB ranker
    distance_split_models: dict = field(default_factory=dict)
    # e.g., {"sprint": {"swim": model, "bike": model, "run": model},
    #        "standard": {"swim": model, "bike": model, "run": model}}
    feature_columns: list[str] = field(default_factory=list)
    split_feature_columns: list[str] = field(default_factory=list)  # Reduced set for split models
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "v1.0"
    metadata: dict = field(default_factory=dict)


# Default model hyperparameters
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}


# ---- Split Time Outlier Thresholds ----
# Per-distance (min, max) in seconds for each split.
# Values below min or above max are treated as data errors or
# distance-category mislabeling (e.g., super-sprint results in sprint bucket).
# Thresholds are set generously below/above the 1st/99th percentile of
# legitimate data to avoid removing valid results.
SPLIT_OUTLIER_THRESHOLDS = {
    "super_sprint": {
        "swim_sec":  (60, 720),       # 1:00 - 12:00
        "t1_sec":    (5, 180),        # 0:05 - 3:00
        "bike_sec":  (180, 1500),     # 3:00 - 25:00
        "t2_sec":    (5, 120),        # 0:05 - 2:00
        "run_sec":   (90, 1200),      # 1:30 - 20:00
        "total_sec": (600, 3000),     # 10:00 - 50:00
    },
    "sprint": {
        "swim_sec":  (240, 1500),     # 4:00 - 25:00
        "t1_sec":    (5, 180),        # 0:05 - 3:00
        "bike_sec":  (900, 3000),     # 15:00 - 50:00
        "t2_sec":    (5, 120),        # 0:05 - 2:00
        "run_sec":   (480, 2400),     # 8:00 - 40:00
        "total_sec": (2400, 6000),    # 40:00 - 100:00
    },
    "standard": {
        "swim_sec":  (600, 2400),     # 10:00 - 40:00
        "t1_sec":    (5, 240),        # 0:05 - 4:00
        "bike_sec":  (2400, 6000),    # 40:00 - 100:00
        "t2_sec":    (5, 180),        # 0:05 - 3:00
        "run_sec":   (1200, 4200),    # 20:00 - 70:00
        "total_sec": (4800, 12000),   # 80:00 - 200:00
    },
    "middle_distance": {
        "swim_sec":  (600, 4200),     # 10:00 - 70:00
        "t1_sec":    (5, 600),        # 0:05 - 10:00
        "bike_sec":  (4800, 14400),   # 80:00 - 240:00
        "t2_sec":    (5, 300),        # 0:05 - 5:00
        "run_sec":   (2700, 9000),    # 45:00 - 150:00
        "total_sec": (9000, 25200),   # 150:00 - 420:00
    },
    "long_distance": {
        "swim_sec":  (1800, 6000),    # 30:00 - 100:00
        "t1_sec":    (5, 900),        # 0:05 - 15:00
        "bike_sec":  (9000, 21600),   # 150:00 - 360:00
        "t2_sec":    (5, 600),        # 0:05 - 10:00
        "run_sec":   (5400, 16200),   # 90:00 - 270:00
        "total_sec": (18000, 43200),  # 300:00 - 720:00
    },
}


def filter_split_outliers(
    df: pd.DataFrame,
    distance_category: str,
    split_col: str,
) -> pd.Series:
    """
    Return a boolean mask of rows with valid (non-outlier) split times.

    Rows where the split value falls outside the (min, max) threshold for the
    given distance category are marked False. Rows with NaN splits are kept
    (marked True) so they can be excluded later by the normal notna() filter.

    Args:
        df: DataFrame containing the split column
        distance_category: Normalized distance key (e.g., 'sprint', 'standard')
        split_col: Column name to check (e.g., 'swim_sec', 'bike_sec')

    Returns:
        Boolean Series aligned with df index (True = keep, False = outlier)
    """
    thresholds = SPLIT_OUTLIER_THRESHOLDS.get(distance_category)
    if thresholds is None or split_col not in thresholds:
        return pd.Series(True, index=df.index)

    lo, hi = thresholds[split_col]
    vals = df[split_col]

    # NaN values pass through (they'll be filtered by notna() downstream)
    is_nan = vals.isna()
    in_range = (vals >= lo) & (vals <= hi)

    return is_nan | in_range


def filter_outliers_by_distance(
    df: pd.DataFrame,
    split_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Filter outlier split times from a mixed-distance training DataFrame.

    For each row, looks up the distance category and applies the corresponding
    thresholds. Outlier split values are set to NaN (per-split) so the row
    can still be used for other targets. Rows where total_sec is an outlier
    are dropped entirely.

    Args:
        df: Training DataFrame with 'prog_distance_category' and split columns
        split_cols: Split columns to check. Default: swim_sec, bike_sec, run_sec, total_sec

    Returns:
        Filtered DataFrame (may be shorter if total_sec outliers were dropped)
    """
    if "prog_distance_category" not in df.columns:
        return df

    if split_cols is None:
        split_cols = ["swim_sec", "t1_sec", "bike_sec", "t2_sec", "run_sec", "total_sec"]

    df = df.copy()
    dist_norm = df["prog_distance_category"].str.lower().str.strip().replace({"olympic": "standard"})

    total_outlier_mask = pd.Series(True, index=df.index)
    n_nulled = {col: 0 for col in split_cols}

    for dist_cat, thresholds in SPLIT_OUTLIER_THRESHOLDS.items():
        dist_mask = dist_norm == dist_cat
        if not dist_mask.any():
            continue

        for col in split_cols:
            if col not in thresholds or col not in df.columns:
                continue

            lo, hi = thresholds[col]
            vals = df.loc[dist_mask, col]
            bad = vals.notna() & ((vals < lo) | (vals > hi))

            if col == "total_sec":
                # Drop entire row if total is an outlier
                total_outlier_mask.loc[bad[bad].index] = False
            else:
                # Null out individual split outliers (keep row for other targets)
                df.loc[bad[bad].index, col] = np.nan

            n_nulled[col] += bad.sum()

    # Log summary
    total_dropped = (~total_outlier_mask).sum()
    for col, count in n_nulled.items():
        if count > 0:
            action = "dropped" if col == "total_sec" else "nulled"
            logger.info(f"Outlier filter: {action} {count} rows for {col}")

    return df.loc[total_outlier_mask].reset_index(drop=True)


def create_regressor(params: dict | None = None) -> Pipeline:
    """
    Create a regression pipeline with imputation and model.

    Args:
        params: Optional hyperparameter overrides. Keys:
                n_estimators, max_depth, learning_rate, num_leaves, min_child_samples.
                Unspecified keys fall back to DEFAULT_PARAMS.

    Returns a Pipeline that:
    1. Imputes missing values with median
    2. Fits a gradient boosting regressor
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    if USE_LIGHTGBM:
        model = LGBMRegressor(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            num_leaves=p.get("num_leaves", 31),
            min_child_samples=p.get("min_child_samples", 20),
            reg_alpha=p.get("reg_alpha", 0.0),
            reg_lambda=p.get("reg_lambda", 0.0),
            random_state=42,
            verbose=-1,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            min_samples_leaf=p.get("min_child_samples", 20),
            l2_regularization=p.get("reg_lambda", 0.0),
            random_state=42,
        )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", model),
    ])

    return pipeline


def _compute_time_decay_weights(
    train_df: pd.DataFrame,
    half_life_days: float = 365.0,
) -> pd.Series | None:
    """
    Compute exponential time-decay weights so recent races count more.

    Uses exponential decay: weight = 2^(-days_ago / half_life_days).
    A race from half_life_days ago gets weight 0.5, a race from 2*half_life
    ago gets 0.25, etc.

    Args:
        train_df: DataFrame with 'event_date' column
        half_life_days: Half-life in days (default 365 = 1 year)

    Returns:
        Series of weights aligned with train_df index, or None if no date column
    """
    if "event_date" not in train_df.columns:
        return None

    dates = pd.to_datetime(train_df["event_date"])
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.clip(lower=0)
    weights = np.power(2.0, -days_ago / half_life_days)
    return weights


def train_baseline_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: dict[str, str] | None = None,
    use_sample_weights: bool = True,
    model_params: dict | None = None,
    time_decay_half_life: float = 365.0,
) -> ModelBundle:
    """
    Train baseline regression models for split and total time prediction.

    Args:
        train_df: Training DataFrame with features and labels
        feature_cols: List of feature column names
        target_cols: Dict mapping model name to target column name
                     Default: {"swim": "swim_sec", "bike": "bike_sec", ...}
        use_sample_weights: If True, weight samples by event tier (WTCS counts more)
        time_decay_half_life: Half-life in days for time-decay weighting.
                             Set to 0 to disable. Default 365 (1 year).

    Returns:
        ModelBundle with trained models

    Raises:
        ValueError: If required columns are missing
    """
    from .features import TIER_SAMPLE_WEIGHTS

    if target_cols is None:
        target_cols = {
            "swim": "swim_sec",
            "t1": "t1_sec",
            "bike": "bike_sec",
            "t2": "t2_sec",
            "run": "run_sec",
            "total": "total_sec",
            "total_pct": "finish_pct",  # Percentile target (0=winner, 1=last)
        }

    # Verify feature columns exist
    missing_features = [c for c in feature_cols if c not in train_df.columns]
    if missing_features:
        logger.warning(f"Missing feature columns (will be filled): {missing_features}")

    # Prepare feature matrix
    X = train_df[feature_cols].copy()

    # Compute sample weights: tier * time_decay
    sample_weights = None
    if use_sample_weights and "event_tier" in train_df.columns:
        sample_weights = train_df["event_tier"].map(TIER_SAMPLE_WEIGHTS).fillna(1.0)
        tier_counts = train_df["event_tier"].value_counts().sort_index()
        logger.info(f"Using tier-based sample weights: {dict(TIER_SAMPLE_WEIGHTS)}")
        logger.info(f"Samples by tier: {tier_counts.to_dict()}")

    # Time-decay weighting: recent races count more
    if time_decay_half_life > 0:
        decay_weights = _compute_time_decay_weights(train_df, time_decay_half_life)
        if decay_weights is not None:
            if sample_weights is not None:
                sample_weights = sample_weights * decay_weights
            else:
                sample_weights = decay_weights
            logger.info(
                f"Time-decay weighting: half_life={time_decay_half_life:.0f} days, "
                f"min_weight={decay_weights.min():.3f}, median_weight={decay_weights.median():.3f}"
            )

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

        model = create_regressor(params=model_params)
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

    # ---- Train LGBMRanker for direct rank optimization ----
    # The ranker directly optimizes ranking (NDCG) within each race group,
    # which is the actual evaluation metric (P@K). This complements the
    # percentile regressor which optimizes MSE on finish_pct.
    model_ranker = None
    ranker_imputer = None

    if USE_LIGHTGBM and "finish_position" in train_df.columns:
        try:
            # Build group structure: each race is a group
            # Need event_id + prog_id to identify unique races
            if "event_id" in train_df.columns and "prog_id" in train_df.columns:
                train_df = train_df.copy()
                train_df["_race_key"] = (
                    train_df["event_id"].astype(str) + "_" + train_df["prog_id"].astype(str)
                )

                # Filter to rows with valid finish position and enough athletes per race
                rank_mask = train_df["finish_position"].notna() & train_df["_race_key"].notna()
                rank_df = train_df[rank_mask].copy()

                # Count athletes per race and filter to races with >= 10 finishers
                race_counts = rank_df.groupby("_race_key").size()
                valid_races = race_counts[race_counts >= 10].index
                rank_df = rank_df[rank_df["_race_key"].isin(valid_races)]

                if len(rank_df) > 100:
                    # Sort by race key for group structure
                    rank_df = rank_df.sort_values("_race_key").reset_index(drop=True)

                    # Compute relevance: higher = better finish
                    # Use inverted position within each race, capped to 31 levels (0-30)
                    # LightGBM lambdarank default label_cnt=31, so labels must be 0-30
                    MAX_RELEVANCE = 30
                    race_sizes = rank_df.groupby("_race_key")["finish_position"].transform("count")
                    raw_relevance = (race_sizes - rank_df["finish_position"]).clip(lower=0)
                    rank_df["_relevance"] = raw_relevance.clip(upper=MAX_RELEVANCE).astype(int)

                    # Group sizes array (ordered)
                    group_sizes = rank_df.groupby("_race_key").size().values

                    # Prepare features
                    X_rank = rank_df[feature_cols].copy()
                    y_rank = rank_df["_relevance"]

                    # Impute first (ranker can't use Pipeline)
                    from sklearn.impute import SimpleImputer as _SI
                    ranker_imputer = _SI(strategy="median")
                    X_rank_imputed = ranker_imputer.fit_transform(X_rank)

                    # Sample weights for ranker: tier * field-size * time-decay
                    from .features import TIER_SAMPLE_WEIGHTS
                    rank_weights = None
                    if use_sample_weights and "event_tier" in rank_df.columns:
                        tier_w = rank_df["event_tier"].map(TIER_SAMPLE_WEIGHTS).fillna(1.0)
                        # Upweight athletes in large fields (>30) where P@10 is hardest
                        field_sizes = rank_df.groupby("_race_key")["finish_position"].transform("count")
                        field_w = np.where(field_sizes >= 50, 2.0,
                                  np.where(field_sizes >= 40, 1.5,
                                  np.where(field_sizes >= 30, 1.2, 1.0)))
                        rank_weights = (tier_w * field_w).values
                    # Add time-decay to ranker weights
                    if time_decay_half_life > 0 and "event_date" in rank_df.columns:
                        decay_w = _compute_time_decay_weights(rank_df, time_decay_half_life)
                        if rank_weights is not None:
                            rank_weights = rank_weights * decay_w.values
                        else:
                            rank_weights = decay_w.values

                    p = {**DEFAULT_PARAMS, **(model_params or {})}
                    model_ranker = LGBMRanker(
                        objective="lambdarank",
                        lambdarank_truncation_level=10,  # Focus on top-10 ranking (NDCG@10)
                        n_estimators=300,  # More trees for ranking task
                        max_depth=p["max_depth"],
                        learning_rate=0.05,  # Slower LR with more trees
                        num_leaves=p.get("num_leaves", 31),
                        min_child_samples=p.get("min_child_samples", 20),
                        reg_alpha=p.get("reg_alpha", 0.0),
                        reg_lambda=p.get("reg_lambda", 0.0),
                        random_state=42,
                        verbose=-1,
                    )

                    fit_kwargs = {"group": group_sizes}
                    if rank_weights is not None:
                        fit_kwargs["sample_weight"] = rank_weights

                    model_ranker.fit(X_rank_imputed, y_rank, **fit_kwargs)

                    logger.info(
                        f"  ranker: trained on {len(rank_df)} samples across "
                        f"{len(group_sizes)} races (LGBMRanker lambdarank)"
                    )
                    metrics["ranker"] = {
                        "n_samples": len(rank_df),
                        "n_races": len(group_sizes),
                    }
                else:
                    logger.warning(f"Not enough data for ranker ({len(rank_df)} rows)")
        except Exception as e:
            logger.warning(f"Failed to train ranker: {e}")
            model_ranker = None
            ranker_imputer = None

    # ---- Train XGBRanker for ensemble diversity ----
    model_xgb_ranker = None
    xgb_ranker_imputer = None

    if USE_XGBOOST and model_ranker is not None:
        try:
            # Reuse the same rank_df, relevance labels, and group structure from LGBMRanker
            from sklearn.impute import SimpleImputer as _SI2
            xgb_ranker_imputer = _SI2(strategy="median")
            X_xgb_imputed = xgb_ranker_imputer.fit_transform(X_rank)

            model_xgb_ranker = xgb.XGBRanker(
                objective="rank:ndcg",
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                tree_method="hist",
            )

            # XGBoost ranking weights must be per-group, not per-sample
            # Aggregate sample weights to group-level by averaging within each group
            xgb_fit_kwargs = {"group": group_sizes}
            if rank_weights is not None:
                group_weights = []
                offset = 0
                for gs in group_sizes:
                    group_weights.append(np.mean(rank_weights[offset:offset + gs]))
                    offset += gs
                xgb_fit_kwargs["sample_weight"] = np.array(group_weights)

            model_xgb_ranker.fit(X_xgb_imputed, y_rank, **xgb_fit_kwargs)

            logger.info(
                f"  xgb_ranker: trained on {len(rank_df)} samples across "
                f"{len(group_sizes)} races (XGBRanker rank:ndcg)"
            )
            metrics["xgb_ranker"] = {
                "n_samples": len(rank_df),
                "n_races": len(group_sizes),
            }
        except Exception as e:
            logger.warning(f"Failed to train XGB ranker: {e}")
            model_xgb_ranker = None
            xgb_ranker_imputer = None

    # Build bundle
    bundle = ModelBundle(
        model_swim=models.get("swim"),
        model_bike=models.get("bike"),
        model_run=models.get("run"),
        model_total=models.get("total"),
        model_t1=models.get("t1"),
        model_t2=models.get("t2"),
        model_total_pct=models.get("total_pct"),
        model_ranker=model_ranker,
        ranker_imputer=ranker_imputer,
        model_xgb_ranker=model_xgb_ranker,
        xgb_ranker_imputer=xgb_ranker_imputer,
        feature_columns=feature_cols,
        metadata={
            "training_metrics": metrics,
            "model_params": model_params or DEFAULT_PARAMS,
        },
    )

    return bundle


def train_distance_split_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    min_samples: int = 200,
    use_sample_weights: bool = True,
    model_params: dict | None = None,
    split_feature_cols: list[str] | None = None,
    exclude_pack_features: bool = False,
    time_decay_half_life: float = 365.0,
) -> tuple[dict, list[str]]:
    """
    Train distance-specific split and total models for each distance category.

    Uses a reduced feature set (split_feature_cols) for individual split models
    that focuses on individual athlete ability rather than field-context or
    ranking features. Also trains a distance-specific total model using the
    full feature set — this ensures predicted total time matches the distance's
    actual time scale (e.g., sprint ~55min vs standard ~110min).

    The unified percentile model (model_total_pct) still handles rankings since
    percentile targets are naturally distance-agnostic.

    Args:
        train_df: Training DataFrame with features, labels, and 'prog_distance_category'
        feature_cols: Full feature column names (used for total model + fallback)
        min_samples: Minimum rows per distance to train models (skip if fewer)
        use_sample_weights: Whether to use tier-based sample weights
        model_params: Hyperparameter overrides for the regressors
        split_feature_cols: Reduced feature set for split models. If None,
                           uses get_split_feature_columns() from features.py.
        exclude_pack_features: If True, removes pack dynamics features from
                              split models to produce "ability-only" predictions.
                              The simulation then adds all pack effects, avoiding
                              double-counting between model and simulation.

    Returns:
        Tuple of (distance_models dict, split_feature_columns list).
        distance_models: {distance: {"swim": model, ..., "total": model}}
        split_feature_columns: the feature columns used for training split models
    """
    from .features import TIER_SAMPLE_WEIGHTS, get_split_feature_columns

    # Use reduced feature set for split models
    if split_feature_cols is None:
        split_feature_cols = get_split_feature_columns(exclude_pack=exclude_pack_features)

    # Filter to columns that actually exist in the training data
    available_cols = [c for c in split_feature_cols if c in train_df.columns]
    missing = set(split_feature_cols) - set(available_cols)
    if missing:
        logger.warning(f"Split feature columns not in training data: {missing}")
    split_feature_cols = available_cols

    logger.info(
        f"Split models using {len(split_feature_cols)} features "
        f"(reduced from {len(feature_cols)} full features)"
    )

    if "prog_distance_category" not in train_df.columns:
        logger.warning("No 'prog_distance_category' column; skipping distance-specific split models")
        return {}, split_feature_cols

    # Normalize distance categories
    df = train_df.copy()
    df["_dist_norm"] = df["prog_distance_category"].str.lower().str.strip()

    # Group standard/olympic together
    df["_dist_norm"] = df["_dist_norm"].replace({"olympic": "standard"})

    distance_models = {}
    split_targets = {"swim": "swim_sec", "t1": "t1_sec", "bike": "bike_sec", "t2": "t2_sec", "run": "run_sec"}
    # Total model trained per-distance with FULL feature set for accurate absolute times
    total_target = {"total": "total_sec"}

    # Prepare full feature columns (available in training data)
    full_feature_cols = [c for c in feature_cols if c in train_df.columns]

    for dist_cat, dist_df in df.groupby("_dist_norm"):
        if dist_df[split_targets["swim"]].notna().sum() < min_samples:
            logger.info(
                f"Distance '{dist_cat}': {len(dist_df)} rows, "
                f"<{min_samples} with valid splits — skipping"
            )
            continue

        logger.info(f"Training distance-specific models for '{dist_cat}' ({len(dist_df)} rows)")

        X_split = dist_df[split_feature_cols].copy()
        X_full = dist_df[full_feature_cols].copy()

        # Sample weights: tier * time_decay
        sample_weights = None
        if use_sample_weights and "event_tier" in dist_df.columns:
            sample_weights = dist_df["event_tier"].map(TIER_SAMPLE_WEIGHTS).fillna(1.0)

        if time_decay_half_life > 0:
            decay_weights = _compute_time_decay_weights(dist_df, time_decay_half_life)
            if decay_weights is not None:
                if sample_weights is not None:
                    sample_weights = sample_weights * decay_weights
                else:
                    sample_weights = decay_weights

        models = {}

        # Train split models (swim, t1, bike, t2, run) with reduced feature set
        for split_name, target_col in split_targets.items():
            if target_col not in dist_df.columns:
                logger.warning(f"  {dist_cat}/{split_name}: target '{target_col}' not found, skipping")
                continue

            mask = dist_df[target_col].notna()

            # Apply outlier filtering for this split
            outlier_mask = filter_split_outliers(dist_df, dist_cat, target_col)
            n_outliers = mask.sum() - (mask & outlier_mask).sum()
            if n_outliers > 0:
                logger.info(
                    f"  {dist_cat}/{split_name}: removed {n_outliers} outlier rows "
                    f"({n_outliers / mask.sum() * 100:.1f}% of valid data)"
                )
            mask = mask & outlier_mask

            X_train = X_split.loc[mask]
            y_train = dist_df.loc[mask, target_col]
            w_train = sample_weights.loc[mask] if sample_weights is not None else None

            if len(y_train) < 50:
                logger.warning(f"  {dist_cat}/{split_name}: only {len(y_train)} samples, skipping")
                continue

            model = create_regressor(params=model_params)
            if w_train is not None:
                model.fit(X_train, y_train, regressor__sample_weight=w_train.values)
            else:
                model.fit(X_train, y_train)

            # Training MAE
            y_pred = model.predict(X_train)
            mae = np.mean(np.abs(y_train - y_pred))
            logger.info(f"  {dist_cat}/{split_name}: MAE={mae:.1f}s on {len(y_train)} samples")
            models[split_name] = model

        # Train distance-specific total model with FULL feature set
        # This gives accurate absolute times for the distance, avoiding the
        # systematic bias from a unified total model across all distances.
        for total_name, target_col in total_target.items():
            if target_col not in dist_df.columns:
                continue

            mask = dist_df[target_col].notna()
            outlier_mask = filter_split_outliers(dist_df, dist_cat, target_col)
            mask = mask & outlier_mask

            X_train = X_full.loc[mask]
            y_train = dist_df.loc[mask, target_col]
            w_train = sample_weights.loc[mask] if sample_weights is not None else None

            if len(y_train) < 50:
                logger.warning(f"  {dist_cat}/{total_name}: only {len(y_train)} samples, skipping")
                continue

            model = create_regressor(params=model_params)
            if w_train is not None:
                model.fit(X_train, y_train, regressor__sample_weight=w_train.values)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_train)
            mae = np.mean(np.abs(y_train - y_pred))
            logger.info(f"  {dist_cat}/{total_name}: MAE={mae:.1f}s on {len(y_train)} samples (full features)")
            models[total_name] = model

        if models:
            distance_models[dist_cat] = models
            logger.info(f"  Trained {len(models)} models for '{dist_cat}' (incl. total)")

    logger.info(f"Distance-specific split models trained for: {list(distance_models.keys())}")
    return distance_models, split_feature_cols


def compute_residual_stats(
    train_df: pd.DataFrame,
    bundle: "ModelBundle",
    feature_cols: list[str],
    split_feature_cols: list[str],
) -> dict:
    """
    Compute empirical residual covariance matrices from training data predictions.

    For each distance category with trained split models, predicts all 5 splits
    (swim, t1, bike, t2, run) on training data, computes residuals (actual - predicted),
    and builds a 5×5 covariance matrix. This captures the natural correlation structure
    between split prediction errors (e.g., fast swimmers tend to have fast T1s).

    Args:
        train_df: Training DataFrame with features and actual split times
        bundle: Trained ModelBundle with distance_split_models
        feature_cols: Full feature column names
        split_feature_cols: Reduced feature columns used for split models

    Returns:
        Dict with structure:
        {
            "overall": {"cov_matrix": [[...]], "per_split_sigma": {"swim": 12.3, ...}, "n_samples": N},
            "sprint": {"cov_matrix": [[...]], "per_split_sigma": {...}, "n_samples": N},
            "standard": {"cov_matrix": [[...]], "per_split_sigma": {...}, "n_samples": N},
        }
    """
    SPLIT_ORDER = ["swim", "t1", "bike", "t2", "run"]
    SPLIT_COLS = ["swim_sec", "t1_sec", "bike_sec", "t2_sec", "run_sec"]

    def _compute_for_subset(df_subset, models, feat_cols):
        """Compute residuals and covariance for a DataFrame subset using given models."""
        residuals = {}
        for split_name, actual_col in zip(SPLIT_ORDER, SPLIT_COLS):
            model = models.get(split_name)
            if model is None or actual_col not in df_subset.columns:
                return None  # Need all 5 splits

            mask = df_subset[actual_col].notna()
            if mask.sum() < 50:
                return None

            X = df_subset.loc[mask, feat_cols]
            y_actual = df_subset.loc[mask, actual_col].values
            y_pred = model.predict(X)
            residuals[split_name] = pd.Series(y_actual - y_pred, index=df_subset.index[mask])

        # Align residuals — only keep rows where all 5 splits have values
        resid_df = pd.DataFrame(residuals)
        resid_df = resid_df.dropna()

        if len(resid_df) < 50:
            return None

        resid_matrix = resid_df.values  # shape (n_samples, 5)
        cov_matrix = np.cov(resid_matrix.T)  # shape (5, 5)
        per_split_sigma = {
            name: float(np.sqrt(cov_matrix[i, i]))
            for i, name in enumerate(SPLIT_ORDER)
        }

        return {
            "cov_matrix": cov_matrix.tolist(),
            "per_split_sigma": per_split_sigma,
            "n_samples": len(resid_df),
        }

    results = {}

    # Normalize distance column
    if "prog_distance_category" in train_df.columns:
        dist_norm = train_df["prog_distance_category"].str.lower().str.strip().replace({"olympic": "standard"})
    else:
        dist_norm = pd.Series("unknown", index=train_df.index)

    # Per-distance residual stats
    for dist_cat, dist_models in bundle.distance_split_models.items():
        dist_mask = dist_norm == dist_cat
        if dist_mask.sum() < 50:
            logger.info(f"Residual stats: {dist_cat} has only {dist_mask.sum()} rows, skipping")
            continue

        # Split models use reduced feature set
        available_feat = [c for c in split_feature_cols if c in train_df.columns]
        result = _compute_for_subset(train_df[dist_mask], dist_models, available_feat)
        if result:
            results[dist_cat] = result
            logger.info(
                f"Residual stats ({dist_cat}): n={result['n_samples']}, "
                f"sigmas: " + ", ".join(f"{k}={v:.1f}s" for k, v in result["per_split_sigma"].items())
            )

    # Overall stats (all distances combined, using unified models)
    overall_models = {
        "swim": bundle.model_swim,
        "t1": bundle.model_t1,
        "bike": bundle.model_bike,
        "t2": bundle.model_t2,
        "run": bundle.model_run,
    }
    # Unified models use full feature set
    available_full = [c for c in feature_cols if c in train_df.columns]
    overall_result = _compute_for_subset(train_df, overall_models, available_full)
    if overall_result:
        results["overall"] = overall_result
        logger.info(
            f"Residual stats (overall): n={overall_result['n_samples']}, "
            f"sigmas: " + ", ".join(f"{k}={v:.1f}s" for k, v in overall_result["per_split_sigma"].items())
        )

    if not results:
        logger.warning("Could not compute residual stats for any distance or overall")

    return results


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
    match_distance: bool = True,
    gender: str | None = None,
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

    # Filter by gender if specified
    if gender:
        if gender.lower() in ("men", "male"):
            programs_df = programs_df[programs_df["prog_name"] == "Elite Men"]
        elif gender.lower() in ("women", "female"):
            programs_df = programs_df[programs_df["prog_name"] == "Elite Women"]
        logger.info(f"Filtered to gender={gender}: {len(programs_df)} programs remaining")

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

            # Compute finish percentile target (0 = best, 1 = last)
            n_finishers = len(merged)
            if n_finishers > 0:
                merged["finish_pct"] = merged["finish_position"].astype(float) / n_finishers

            all_rows.append(merged)

        except Exception as e:
            logger.warning(f"Error processing {key}: {e}")
            continue

    if not all_rows:
        logger.error("No training data collected")
        return pd.DataFrame()

    train_df = pd.concat(all_rows, ignore_index=True)
    logger.info(f"Built training dataset with {len(train_df)} rows (before outlier filtering)")

    # Apply outlier filtering based on distance-specific split thresholds
    n_before = len(train_df)
    train_df = filter_outliers_by_distance(train_df)
    n_after = len(train_df)
    if n_before != n_after:
        logger.info(
            f"Outlier filtering: {n_before} → {n_after} rows "
            f"({n_before - n_after} dropped, {(n_before - n_after) / n_before * 100:.1f}%)"
        )

    return train_df


def time_based_cv_splits(
    train_df: pd.DataFrame,
    folds: list[tuple[str, str]] | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create time-based cross-validation splits (expanding window).

    Each fold uses all data up to a cutoff for training and the next period
    for validation. Requires an 'event_date' column in train_df.

    Args:
        train_df: Full training DataFrame with 'event_date' column
        folds: Optional list of (train_end, val_end) date strings.
               Default folds:
                 Fold 1: train ≤2021-12-31, val 2022
                 Fold 2: train ≤2022-12-31, val 2023
                 Fold 3: train ≤2023-12-31, val 2024H1
                 Fold 4: train ≤2024-06-30, val 2024H2

    Returns:
        List of (train_split, val_split) DataFrame tuples
    """
    if "event_date" not in train_df.columns:
        raise ValueError("train_df must have an 'event_date' column for time-based CV")

    dates = pd.to_datetime(train_df["event_date"])

    if folds is None:
        folds = [
            ("2021-12-31", "2022-12-31"),
            ("2022-12-31", "2023-12-31"),
            ("2023-12-31", "2024-06-30"),
            ("2024-06-30", "2024-12-31"),
        ]

    splits = []
    for i, (train_end, val_end) in enumerate(folds):
        train_mask = dates <= pd.Timestamp(train_end)
        val_mask = (dates > pd.Timestamp(train_end)) & (dates <= pd.Timestamp(val_end))

        train_split = train_df.loc[train_mask]
        val_split = train_df.loc[val_mask]

        if len(train_split) < 50 or len(val_split) < 10:
            logger.warning(
                f"Fold {i+1}: insufficient data (train={len(train_split)}, val={len(val_split)}), skipping"
            )
            continue

        logger.info(
            f"Fold {i+1}: train ≤{train_end} ({len(train_split)} rows), "
            f"val ({train_end}, {val_end}] ({len(val_split)} rows)"
        )
        splits.append((train_split, val_split))

    return splits


def cross_validate(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    model_params: dict | None = None,
    folds: list[tuple[str, str]] | None = None,
    use_sample_weights: bool = True,
) -> dict:
    """
    Run time-based cross-validation, training a finish_pct model per fold.

    For each fold:
    1. Train a single percentile model (fast)
    2. Group val predictions by (event_id, prog_id)
    3. Compute per-race Spearman correlation

    Args:
        train_df: Full training DataFrame with features, labels, and 'event_date'
        feature_cols: Feature column names
        model_params: Hyperparameters for the regressor
        folds: Optional CV fold definitions (passed to time_based_cv_splits)
        use_sample_weights: Whether to use tier-based sample weights

    Returns:
        Dict with keys: mean_spearman, mean_mae, fold_results (list of per-fold dicts)
    """
    from .features import TIER_SAMPLE_WEIGHTS
    from scipy.stats import spearmanr

    splits = time_based_cv_splits(train_df, folds=folds)
    if not splits:
        logger.error("No valid CV splits")
        return {"mean_spearman": np.nan, "mean_mae": np.nan, "fold_results": []}

    fold_results = []

    for fold_idx, (tr, va) in enumerate(splits):
        # Train percentile model only
        target_col = "finish_pct"
        if target_col not in tr.columns:
            logger.warning(f"Fold {fold_idx+1}: '{target_col}' not in data, skipping")
            continue

        tr_mask = tr[target_col].notna()
        va_mask = va[target_col].notna()

        X_tr = tr.loc[tr_mask, feature_cols]
        y_tr = tr.loc[tr_mask, target_col]
        X_va = va.loc[va_mask, feature_cols]
        y_va = va.loc[va_mask, target_col]

        if len(y_tr) < 50 or len(y_va) < 10:
            logger.warning(f"Fold {fold_idx+1}: insufficient labeled data, skipping")
            continue

        # Sample weights
        w_tr = None
        if use_sample_weights and "event_tier" in tr.columns:
            w_tr = tr.loc[tr_mask, "event_tier"].map(TIER_SAMPLE_WEIGHTS).fillna(1.0)

        model = create_regressor(params=model_params)
        if w_tr is not None:
            model.fit(X_tr, y_tr, regressor__sample_weight=w_tr.values)
        else:
            model.fit(X_tr, y_tr)

        # Predict on validation
        va_pred = va.loc[va_mask].copy()
        va_pred["pred_pct"] = model.predict(X_va)

        # Per-race Spearman
        race_spearmans = []
        race_maes = []
        for _, race_group in va_pred.groupby(["event_id", "prog_id"]):
            if len(race_group) < 5:
                continue
            corr, _ = spearmanr(race_group["pred_pct"], race_group[target_col])
            race_spearmans.append(corr)
            race_maes.append(np.mean(np.abs(race_group["pred_pct"] - race_group[target_col])))

        fold_spearman = np.mean(race_spearmans) if race_spearmans else np.nan
        fold_mae = np.mean(race_maes) if race_maes else np.nan

        logger.info(
            f"Fold {fold_idx+1}: Spearman={fold_spearman:.3f}, MAE={fold_mae:.4f}, "
            f"n_races={len(race_spearmans)}"
        )

        fold_results.append({
            "fold": fold_idx + 1,
            "spearman": fold_spearman,
            "mae": fold_mae,
            "n_races": len(race_spearmans),
            "n_train": len(y_tr),
            "n_val": len(y_va),
        })

    mean_spearman = np.mean([f["spearman"] for f in fold_results]) if fold_results else np.nan
    mean_mae = np.mean([f["mae"] for f in fold_results]) if fold_results else np.nan

    return {
        "mean_spearman": mean_spearman,
        "mean_mae": mean_mae,
        "fold_results": fold_results,
    }


def cross_validate_splits(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    split_feature_cols: list[str] | None = None,
    model_params: dict | None = None,
    folds: list[tuple[str, str]] | None = None,
    use_sample_weights: bool = True,
    exclude_pack_features: bool = True,
) -> dict:
    """
    Run time-based cross-validation tracking per-split MAE independently.

    For each fold, trains distance-specific split models (swim, t1, bike, t2, run)
    and evaluates MAE per split on the validation set. Also computes a split-total
    consistency metric (how well the sum of split predictions matches the total
    model prediction).

    Args:
        train_df: Full training DataFrame with features, labels, and 'event_date'
        feature_cols: Full feature column names
        split_feature_cols: Reduced feature set for split models. If None, uses default.
        model_params: Hyperparameters for the regressors
        folds: Optional CV fold definitions (passed to time_based_cv_splits)
        use_sample_weights: Whether to use tier-based sample weights
        exclude_pack_features: Whether to exclude pack features from split models

    Returns:
        Dict with per-split MAE averages and per-fold breakdown
    """
    from .features import get_split_feature_columns

    if split_feature_cols is None:
        split_feature_cols = get_split_feature_columns(exclude_pack=exclude_pack_features)

    splits = time_based_cv_splits(train_df, folds=folds)
    if not splits:
        logger.error("No valid CV splits for per-split validation")
        return {}

    SPLIT_TARGETS = {
        "swim": "swim_sec", "t1": "t1_sec", "bike": "bike_sec",
        "t2": "t2_sec", "run": "run_sec",
    }

    fold_results = []

    for fold_idx, (tr, va) in enumerate(splits):
        # Train distance-specific models on the fold
        dist_models, used_split_cols = train_distance_split_models(
            tr, feature_cols,
            model_params=model_params,
            use_sample_weights=use_sample_weights,
            split_feature_cols=list(split_feature_cols),
            exclude_pack_features=exclude_pack_features,
        )

        fold_metrics = {"fold": fold_idx + 1, "n_train": len(tr), "n_val": len(va)}

        # Evaluate per-split MAE on validation set, per distance
        for dist_cat, models in dist_models.items():
            dist_norm = va["prog_distance_category"].str.lower().str.strip().replace({"olympic": "standard"})
            va_dist = va[dist_norm == dist_cat]
            if len(va_dist) < 10:
                continue

            X_va_split = va_dist[[c for c in used_split_cols if c in va_dist.columns]]

            split_sum = np.zeros(len(va_dist))
            for split_name, target_col in SPLIT_TARGETS.items():
                model = models.get(split_name)
                if model is None or target_col not in va_dist.columns:
                    continue

                mask = va_dist[target_col].notna()
                if mask.sum() < 5:
                    continue

                y_pred = model.predict(X_va_split.loc[mask])
                y_actual = va_dist.loc[mask, target_col].values
                mae = float(np.mean(np.abs(y_actual - y_pred)))
                fold_metrics[f"{dist_cat}_{split_name}_mae"] = mae
                split_sum[mask.values] += y_pred

            # Split-total consistency: compare split sum to actual total
            if "total_sec" in va_dist.columns:
                total_mask = va_dist["total_sec"].notna()
                if total_mask.sum() > 5:
                    actual_total = va_dist.loc[total_mask, "total_sec"].values
                    pred_split_sum = split_sum[total_mask.values]
                    # Only compare where split_sum is non-zero (all splits predicted)
                    valid = pred_split_sum > 0
                    if valid.sum() > 5:
                        consistency_mae = float(np.mean(np.abs(actual_total[valid] - pred_split_sum[valid])))
                        fold_metrics[f"{dist_cat}_split_sum_mae"] = consistency_mae

        fold_results.append(fold_metrics)
        logger.info(f"Fold {fold_idx + 1} split CV: {fold_metrics}")

    # Aggregate across folds
    summary = {"fold_results": fold_results}
    all_metric_keys = set()
    for fr in fold_results:
        all_metric_keys.update(k for k in fr.keys() if k not in ("fold", "n_train", "n_val"))

    for key in sorted(all_metric_keys):
        values = [fr[key] for fr in fold_results if key in fr]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))

    return summary


# Default hyperparameter search grid
DEFAULT_PARAM_GRID = {
    "n_estimators": [200, 350, 500],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "min_child_samples": [10, 20, 50],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0],
}


def tune_hyperparameters(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    param_grid: dict | None = None,
    folds: list[tuple[str, str]] | None = None,
    use_sample_weights: bool = True,
) -> dict:
    """
    Grid search over hyperparameters using time-based cross-validation.

    Optimizes for Spearman correlation on the percentile model.

    Args:
        train_df: Full training DataFrame
        feature_cols: Feature column names
        param_grid: Dict of param_name -> list of values to search.
                    Default: DEFAULT_PARAM_GRID (54 combinations)
        folds: Optional CV fold definitions
        use_sample_weights: Whether to use tier-based sample weights

    Returns:
        Dict with keys: best_params, best_score, all_results (sorted list)
    """
    from itertools import product

    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    logger.info(f"Hyperparameter tuning: {len(combinations)} combinations to evaluate")

    all_results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())

        logger.info(f"[{i+1}/{len(combinations)}] {params_str}")

        cv_result = cross_validate(
            train_df, feature_cols,
            model_params=params,
            folds=folds,
            use_sample_weights=use_sample_weights,
        )

        all_results.append({
            "params": params,
            "mean_spearman": cv_result["mean_spearman"],
            "mean_mae": cv_result["mean_mae"],
            "fold_results": cv_result["fold_results"],
        })

    # Sort by Spearman (higher is better)
    all_results.sort(key=lambda x: x["mean_spearman"], reverse=True)

    best = all_results[0] if all_results else {"params": DEFAULT_PARAMS, "mean_spearman": np.nan}

    logger.info(f"\n--- Tuning Results (top 5) ---")
    for rank, res in enumerate(all_results[:5], 1):
        params_str = ", ".join(f"{k}={v}" for k, v in res["params"].items())
        logger.info(f"  #{rank}: Spearman={res['mean_spearman']:.4f} | {params_str}")

    logger.info(f"\nBest params: {best['params']}")
    logger.info(f"Best Spearman: {best['mean_spearman']:.4f}")

    return {
        "best_params": best["params"],
        "best_score": best["mean_spearman"],
        "all_results": all_results,
    }
