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
    bundle: ModelBundle
) -> pd.DataFrame:
    """
    Generate predictions for all athletes in the features DataFrame.

    Args:
        features_df: DataFrame with feature columns (from build_features_for_program)
        bundle: Trained ModelBundle

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

    # Predict each split
    predictions = {}

    if bundle.model_swim is not None:
        predictions["pred_swim_sec"] = bundle.model_swim.predict(X)
    else:
        predictions["pred_swim_sec"] = np.full(len(df), np.nan)

    if bundle.model_bike is not None:
        predictions["pred_bike_sec"] = bundle.model_bike.predict(X)
    else:
        predictions["pred_bike_sec"] = np.full(len(df), np.nan)

    if bundle.model_run is not None:
        predictions["pred_run_sec"] = bundle.model_run.predict(X)
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

    # Format as human-readable times
    df["pred_swim_hms"] = df["pred_swim_sec"].apply(lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None)
    df["pred_bike_hms"] = df["pred_bike_sec"].apply(lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None)
    df["pred_run_hms"] = df["pred_run_sec"].apply(lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None)
    df["pred_total_hms"] = df["pred_total_sec"].apply(lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None)

    # Compute predicted rank (1 = fastest)
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
