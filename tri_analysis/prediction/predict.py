"""
Prediction module for the pipeline.

Produces deterministic predictions (split times, total time, ranking) from features.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from .train import ModelBundle
from .utils_time import seconds_to_hms

logger = logging.getLogger(__name__)


def predict_splits_and_total(
    features_df: pd.DataFrame,
    bundle: ModelBundle,
    distance_category: str | None = None,
) -> pd.DataFrame:
    """
    Generate predictions for all athletes in the features DataFrame.

    Uses distance-specific split models when available in the bundle for
    accurate absolute time predictions (swim, bike, run). Falls back to
    the unified split models otherwise. Rankings always use the unified
    percentile model (model_total_pct) since percentile targets are
    distance-agnostic.

    Args:
        features_df: DataFrame with feature columns (from build_features_for_program)
        bundle: Trained ModelBundle
        distance_category: Race distance (e.g., 'sprint', 'standard') for
                          selecting distance-specific split models. If None,
                          uses unified models.

    Returns:
        DataFrame with original columns plus:
        - pred_swim_sec, pred_bike_sec, pred_run_sec, pred_total_sec
        - pred_swim_hms, pred_bike_hms, pred_run_hms, pred_total_hms (formatted strings)
        - predicted_rank (1=fastest predicted total)
    """
    if features_df.empty:
        logger.warning("Empty features_df, returning empty DataFrame")
        return features_df.copy()

    df = features_df.copy()
    feature_cols = bundle.feature_columns

    # Ensure all feature columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing feature columns: {missing_cols}")
        for col in missing_cols:
            df[col] = np.nan

    # Prepare feature matrix
    X = df[feature_cols].copy()

    # Replace any remaining NaNs with median (defensive)
    X = X.fillna(X.median())

    # Select distance-specific split models if available
    dist_models = None
    dist_key = None
    if distance_category and hasattr(bundle, "distance_split_models") and bundle.distance_split_models:
        dist_key = distance_category.lower().strip()
        # Normalize olympic → standard
        if dist_key == "olympic":
            dist_key = "standard"
        dist_models = bundle.distance_split_models.get(dist_key)
        if dist_models:
            logger.info(f"Using distance-specific split models for '{dist_key}' (splits: {list(dist_models.keys())})")
        else:
            logger.info(f"No distance-specific models for '{dist_key}', using unified split models")

    # Predict each split (distance-specific if available, else unified)
    predictions = {}

    swim_model = (dist_models or {}).get("swim") or bundle.model_swim
    bike_model = (dist_models or {}).get("bike") or bundle.model_bike
    run_model = (dist_models or {}).get("run") or bundle.model_run

    if swim_model is not None:
        predictions["pred_swim_sec"] = swim_model.predict(X)
    else:
        predictions["pred_swim_sec"] = np.full(len(df), np.nan)

    if bike_model is not None:
        predictions["pred_bike_sec"] = bike_model.predict(X)
    else:
        predictions["pred_bike_sec"] = np.full(len(df), np.nan)

    if run_model is not None:
        predictions["pred_run_sec"] = run_model.predict(X)
    else:
        predictions["pred_run_sec"] = np.full(len(df), np.nan)

    if bundle.model_total is not None:
        predictions["pred_total_sec"] = bundle.model_total.predict(X)
    else:
        # Fall back to sum of splits if total model not available
        predictions["pred_total_sec"] = (
            predictions["pred_swim_sec"] +
            predictions["pred_bike_sec"] +
            predictions["pred_run_sec"]
        )

    # Add predictions to DataFrame
    for col, vals in predictions.items():
        df[col] = vals

    # ========== Prediction Anchoring ==========
    # Prevent predictions from being unreasonably slow compared to historical performance.
    # If pred_total_sec is more than 10% slower than EMA, anchor it closer to EMA.
    # This addresses model overfitting that penalizes athletes with unusually fast history.
    MAX_SLOWDOWN_FACTOR = 1.10  # Max 10% slower than EMA
    
    if "ema_total_sec_5" in df.columns:
        ema_total = df["ema_total_sec_5"]
        pred_total = df["pred_total_sec"]
        max_allowed = ema_total * MAX_SLOWDOWN_FACTOR
        
        # Where prediction exceeds max allowed, cap it firmly
        over_limit = pred_total > max_allowed
        if over_limit.any():
            df.loc[over_limit, "pred_total_sec"] = max_allowed[over_limit]
            n_adjusted = over_limit.sum()
            logger.info(f"Anchored {n_adjusted} predictions that exceeded {MAX_SLOWDOWN_FACTOR:.0%} slowdown from EMA")

    # ========== Percentile Model (Two-Stage Ranking) ==========
    # If a percentile model exists, predict finish_pct for ranking.
    # This is distance-agnostic (target is always 0-1) and focuses on relative
    # position rather than absolute seconds, improving ranking accuracy.
    has_pct_model = getattr(bundle, "model_total_pct", None) is not None
    if has_pct_model:
        df["pred_finish_pct"] = bundle.model_total_pct.predict(X)
        # Clip to valid range
        df["pred_finish_pct"] = df["pred_finish_pct"].clip(0.001, 1.0)
        logger.info("Using percentile model for ranking")

    # Format as human-readable times
    df["pred_swim_hms"] = df["pred_swim_sec"].apply(lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None)
    df["pred_bike_hms"] = df["pred_bike_sec"].apply(lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None)
    df["pred_run_hms"] = df["pred_run_sec"].apply(lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None)
    df["pred_total_hms"] = df["pred_total_sec"].apply(lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None)

    # Compute predicted rank
    # Use percentile model for ranking if available (better cross-distance accuracy),
    # otherwise fall back to ranking by predicted total seconds
    if has_pct_model:
        df["predicted_rank"] = df["pred_finish_pct"].rank(method="min", ascending=True).astype(int)
    else:
        df["predicted_rank"] = df["pred_total_sec"].rank(method="min", ascending=True).astype(int)

    # Sort by predicted rank
    df = df.sort_values("predicted_rank").reset_index(drop=True)

    logger.info(f"Generated predictions for {len(df)} athletes")

    return df


def format_prediction_output(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format prediction output for display/export.

    Returns a clean DataFrame with key columns for reporting.
    """
    output_cols = [
        "predicted_rank",
        "athlete_full_name",
        "athlete_country_name",
        "pred_total_hms",
        "pred_swim_hms",
        "pred_bike_hms",
        "pred_run_hms",
        "pred_total_sec",
        "pred_finish_pct",
        "ema_total_sec_5",
        "front_pack_rate",
        "seed_total_rank",
    ]

    # Only include columns that exist
    available_cols = [c for c in output_cols if c in pred_df.columns]

    output_df = pred_df[available_cols].copy()
    output_df = output_df.rename(columns={
        "predicted_rank": "Rank",
        "athlete_full_name": "Athlete",
        "athlete_country_name": "Country",
        "pred_total_hms": "Pred Total",
        "pred_swim_hms": "Pred Swim",
        "pred_bike_hms": "Pred Bike",
        "pred_run_hms": "Pred Run",
    })

    return output_df
