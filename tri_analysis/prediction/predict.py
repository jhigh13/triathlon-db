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
        - pred_swim_sec, pred_t1_sec, pred_bike_sec, pred_t2_sec, pred_run_sec, pred_total_sec
        - pred_swim_hms, pred_t1_hms, pred_bike_hms, pred_t2_hms, pred_run_hms, pred_total_hms
        - predicted_rank (1=fastest predicted total)
    """
    if features_df.empty:
        logger.warning("Empty features_df, returning empty DataFrame")
        return features_df.copy()

    df = features_df.copy()
    feature_cols = bundle.feature_columns

    # Split models use a reduced feature set (no field-context/ranking features)
    split_feature_cols = getattr(bundle, "split_feature_columns", None) or feature_cols

    # Ensure all feature columns exist
    for col_list in [feature_cols, split_feature_cols]:
        missing_cols = [c for c in col_list if c not in df.columns]
        if missing_cols:
            logger.warning(f"Missing feature columns: {missing_cols}")
            for col in missing_cols:
                df[col] = np.nan

    # Prepare feature matrices
    X_full = df[feature_cols].copy()
    X_full = X_full.fillna(X_full.median())

    X_split = df[split_feature_cols].copy()
    X_split = X_split.fillna(X_split.median())

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
            logger.info(
                f"Using distance-specific split models for '{dist_key}' "
                f"(splits: {list(dist_models.keys())}, "
                f"features: {len(split_feature_cols)}/{len(feature_cols)})"
            )
        else:
            logger.info(f"No distance-specific models for '{dist_key}', using unified split models")

    # Predict each split
    # Distance-specific models use X_split (reduced features)
    # Unified fallback models use X_full (full features)
    predictions = {}

    if dist_models and "swim" in dist_models:
        predictions["pred_swim_sec"] = dist_models["swim"].predict(X_split)
    elif bundle.model_swim is not None:
        predictions["pred_swim_sec"] = bundle.model_swim.predict(X_full)
    else:
        predictions["pred_swim_sec"] = np.full(len(df), np.nan)

    # T1 prediction
    if dist_models and "t1" in dist_models:
        predictions["pred_t1_sec"] = dist_models["t1"].predict(X_split)
    elif getattr(bundle, "model_t1", None) is not None:
        predictions["pred_t1_sec"] = bundle.model_t1.predict(X_full)
    else:
        predictions["pred_t1_sec"] = np.full(len(df), np.nan)

    if dist_models and "bike" in dist_models:
        predictions["pred_bike_sec"] = dist_models["bike"].predict(X_split)
    elif bundle.model_bike is not None:
        predictions["pred_bike_sec"] = bundle.model_bike.predict(X_full)
    else:
        predictions["pred_bike_sec"] = np.full(len(df), np.nan)

    # T2 prediction
    if dist_models and "t2" in dist_models:
        predictions["pred_t2_sec"] = dist_models["t2"].predict(X_split)
    elif getattr(bundle, "model_t2", None) is not None:
        predictions["pred_t2_sec"] = bundle.model_t2.predict(X_full)
    else:
        predictions["pred_t2_sec"] = np.full(len(df), np.nan)

    if dist_models and "run" in dist_models:
        predictions["pred_run_sec"] = dist_models["run"].predict(X_split)
    elif bundle.model_run is not None:
        predictions["pred_run_sec"] = bundle.model_run.predict(X_full)
    else:
        predictions["pred_run_sec"] = np.full(len(df), np.nan)

    # Add predictions to DataFrame (before total, since we may derive total from splits)
    # Force float dtype to prevent TypeError from mixed int/None object columns
    for col, vals in predictions.items():
        df[col] = pd.to_numeric(pd.Series(vals, index=df.index), errors="coerce")

    # T1/T2 fallback: fill missing T1/T2 predictions from total model budget
    # Uses .any() so partial missing values (some athletes missing) also get filled.
    # Need a temporary total for the fallback calculation
    if bundle.model_total is not None:
        _total_for_fallback = bundle.model_total.predict(X_full)
    elif dist_models and "total" in dist_models:
        _total_for_fallback = dist_models["total"].predict(X_full)
    else:
        _total_for_fallback = None

    t1_has_any_missing = df["pred_t1_sec"].isna().any()
    t2_has_any_missing = df["pred_t2_sec"].isna().any()
    if (t1_has_any_missing or t2_has_any_missing) and _total_for_fallback is not None:
        split_sum = df["pred_swim_sec"] + df["pred_bike_sec"] + df["pred_run_sec"]
        transition_budget = (pd.Series(_total_for_fallback, index=df.index) - split_sum).clip(lower=0)

        t1_mask = df["pred_t1_sec"].isna()
        t2_mask = df["pred_t2_sec"].isna()
        both_mask = t1_mask & t2_mask

        n_t1_filled = 0
        n_t2_filled = 0

        if both_mask.any():
            df.loc[both_mask, "pred_t1_sec"] = (transition_budget[both_mask] * 0.55)
            df.loc[both_mask, "pred_t2_sec"] = (transition_budget[both_mask] * 0.45)
            n_t1_filled += both_mask.sum()
            n_t2_filled += both_mask.sum()

        t1_only = t1_mask & ~t2_mask
        if t1_only.any():
            df.loc[t1_only, "pred_t1_sec"] = (transition_budget[t1_only] - df.loc[t1_only, "pred_t2_sec"]).clip(lower=5)
            n_t1_filled += t1_only.sum()

        t2_only = t2_mask & ~t1_mask
        if t2_only.any():
            df.loc[t2_only, "pred_t2_sec"] = (transition_budget[t2_only] - df.loc[t2_only, "pred_t1_sec"]).clip(lower=5)
            n_t2_filled += t2_only.sum()

        if n_t1_filled or n_t2_filled:
            logger.info(f"Filled missing transitions: {n_t1_filled} T1, {n_t2_filled} T2 (from total model budget)")

    # Total prediction: use SPLIT SUM when all 5 splits are available.
    # This ensures Det. Total is consistent with what the simulation uses
    # (sim_total = swim + T1 + bike + T2 + run), eliminating the split-total gap.
    # Fall back to total model when splits are incomplete.
    all_splits_available = all(
        col in df.columns and df[col].notna().all()
        for col in ["pred_swim_sec", "pred_t1_sec", "pred_bike_sec", "pred_t2_sec", "pred_run_sec"]
    )

    # ========== Total Prediction ==========
    # First, compute a "reference total" from the total model (or dist-specific total).
    # This serves as the ground truth for split sanity checks and anchoring.
    if dist_models and "total" in dist_models:
        ref_total = pd.Series(dist_models["total"].predict(X_full), index=df.index)
    elif bundle.model_total is not None:
        ref_total = pd.Series(bundle.model_total.predict(X_full), index=df.index)
    else:
        ref_total = None

    # ========== Split Sanity Check ==========
    # Distance-specific split models can produce wildly wrong predictions for
    # athletes with sparse/no history in that distance. When the split sum
    # diverges significantly from the reference total, proportionally rescale
    # splits to match the reference. This prevents garbage splits from corrupting
    # both rankings and simulation input.
    SPLIT_RESCALE_THRESHOLD = 0.20  # Rescale if split sum > 20% off reference total

    if all_splits_available and ref_total is not None:
        split_sum = (
            df["pred_swim_sec"] + df["pred_t1_sec"] +
            df["pred_bike_sec"] + df["pred_t2_sec"] +
            df["pred_run_sec"]
        )
        ratio = split_sum / ref_total
        needs_rescale = (ratio - 1.0).abs() > SPLIT_RESCALE_THRESHOLD
        if needs_rescale.any():
            n_rescaled = needs_rescale.sum()
            scale_factor = ref_total[needs_rescale] / split_sum[needs_rescale]
            for col in ["pred_swim_sec", "pred_t1_sec", "pred_bike_sec", "pred_t2_sec", "pred_run_sec"]:
                df.loc[needs_rescale, col] = df.loc[needs_rescale, col] * scale_factor
            logger.info(
                f"Rescaled splits for {n_rescaled} athletes where split sum diverged "
                f">{SPLIT_RESCALE_THRESHOLD:.0%} from total model "
                f"(worst ratio: {ratio[needs_rescale].max():.2f}x)"
            )

    if all_splits_available:
        df["pred_total_sec"] = (
            df["pred_swim_sec"] + df["pred_t1_sec"] +
            df["pred_bike_sec"] + df["pred_t2_sec"] +
            df["pred_run_sec"]
        )
        logger.info("Using split sum as pred_total_sec (all 5 splits available)")
    elif ref_total is not None:
        df["pred_total_sec"] = ref_total
        logger.info("Using total model (incomplete splits)")
    else:
        df["pred_total_sec"] = (
            df["pred_swim_sec"].fillna(0) + df["pred_bike_sec"].fillna(0) +
            df["pred_run_sec"].fillna(0)
        )
        logger.warning("No total model and incomplete splits — using partial split sum")

    # ========== Prediction Anchoring ==========
    # Prevent predictions from being unreasonably slow compared to historical performance.
    # If pred_total_sec is more than 10% slower than EMA, anchor it closer to EMA.
    # When anchoring adjusts the total, proportionally rescale splits to maintain consistency.
    MAX_SLOWDOWN_FACTOR = 1.10  # Max 10% slower than EMA

    if "ema_total_sec_5" in df.columns:
        ema_total = df["ema_total_sec_5"]
        pred_total = df["pred_total_sec"]
        max_allowed = ema_total * MAX_SLOWDOWN_FACTOR

        # Where prediction exceeds max allowed, cap it and rescale splits
        over_limit = pred_total > max_allowed
        if over_limit.any():
            anchor_scale = max_allowed[over_limit] / pred_total[over_limit]
            df.loc[over_limit, "pred_total_sec"] = max_allowed[over_limit]
            # Proportionally rescale splits to match new total
            for col in ["pred_swim_sec", "pred_t1_sec", "pred_bike_sec", "pred_t2_sec", "pred_run_sec"]:
                if col in df.columns:
                    df.loc[over_limit, col] = df.loc[over_limit, col] * anchor_scale
            n_adjusted = over_limit.sum()
            logger.info(f"Anchored {n_adjusted} predictions that exceeded {MAX_SLOWDOWN_FACTOR:.0%} slowdown from EMA (splits rescaled)")

    # ========== Percentile Model (Two-Stage Ranking) ==========
    # If a percentile model exists, predict finish_pct for ranking.
    # This is distance-agnostic (target is always 0-1) and focuses on relative
    # position rather than absolute seconds, improving ranking accuracy.
    has_pct_model = getattr(bundle, "model_total_pct", None) is not None
    if has_pct_model:
        df["pred_finish_pct"] = bundle.model_total_pct.predict(X_full)
        # Clip to valid range
        df["pred_finish_pct"] = df["pred_finish_pct"].clip(0.001, 1.0)
        logger.info("Using percentile model for ranking")

    # Format as human-readable times
    df["pred_swim_hms"] = df["pred_swim_sec"].apply(lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None)
    df["pred_t1_hms"] = df["pred_t1_sec"].apply(lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None)
    df["pred_bike_hms"] = df["pred_bike_sec"].apply(lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None)
    df["pred_t2_hms"] = df["pred_t2_sec"].apply(lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None)
    df["pred_run_hms"] = df["pred_run_sec"].apply(lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None)
    df["pred_total_hms"] = df["pred_total_sec"].apply(lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None)

    # Log split-total accounting diagnostic
    active_total = df["pred_swim_sec"] + df["pred_t1_sec"] + df["pred_bike_sec"] + df["pred_t2_sec"] + df["pred_run_sec"]
    delta = df["pred_total_sec"] - active_total
    abs_delta = delta.abs()
    logger.info(
        f"Split-total accounting: mean delta={delta.mean():.1f}s, "
        f"median={delta.median():.1f}s (total_model - sum_of_splits)"
    )

    # Warn on large per-athlete discrepancies
    ANCHORING_WARN_THRESHOLD = 60.0  # seconds
    large_discrepancy = abs_delta > ANCHORING_WARN_THRESHOLD
    if large_discrepancy.any():
        n_bad = large_discrepancy.sum()
        worst_idx = abs_delta.idxmax()
        worst_name = df.loc[worst_idx, "athlete_full_name"] if "athlete_full_name" in df.columns else f"idx={worst_idx}"
        logger.warning(
            f"Split-total anchoring: {n_bad}/{len(df)} athletes have |delta| > {ANCHORING_WARN_THRESHOLD:.0f}s. "
            f"Worst: {worst_name} (delta={delta.loc[worst_idx]:.1f}s). "
            f"Consider retraining split models or checking feature coverage."
        )

    # Store diagnostic columns for downstream use
    df["pred_split_sum_sec"] = active_total
    df["pred_split_total_delta"] = delta

    # Compute predicted rank
    # Priority: (1) LGBMRanker if available, (2) percentile model, (3) absolute time
    has_ranker = (
        hasattr(bundle, "model_ranker") and bundle.model_ranker is not None
        and hasattr(bundle, "ranker_imputer") and bundle.ranker_imputer is not None
    )

    if has_ranker and has_pct_model:
        try:
            X_rank = df[bundle.feature_columns].copy()
            X_rank_imputed = bundle.ranker_imputer.transform(X_rank)
            ranker_scores = bundle.model_ranker.predict(X_rank_imputed)
            df["ranker_score"] = ranker_scores
            # Ranker rank: higher score = better finish
            ranker_rank = df["ranker_score"].rank(method="average", ascending=False)
            # Percentile rank: lower pct = better finish
            pct_rank = df["pred_finish_pct"].rank(method="average", ascending=True)

            RANKER_WEIGHT = 0.5
            df["ensemble_rank_score"] = RANKER_WEIGHT * ranker_rank + (1 - RANKER_WEIGHT) * pct_rank
            logger.info(f"Using ensemble ranking (ranker={RANKER_WEIGHT:.0%}, pct={1-RANKER_WEIGHT:.0%})")

            df["predicted_rank"] = df["ensemble_rank_score"].rank(method="min", ascending=True).astype(int)
        except Exception as e:
            logger.warning(f"Ensemble ranking failed, falling back to percentile: {e}")
            df["predicted_rank"] = df["pred_finish_pct"].rank(method="min", ascending=True).astype(int)
    elif has_ranker:
        try:
            X_rank = df[bundle.feature_columns].copy()
            X_rank_imputed = bundle.ranker_imputer.transform(X_rank)
            ranker_scores = bundle.model_ranker.predict(X_rank_imputed)
            df["ranker_score"] = ranker_scores
            df["predicted_rank"] = df["ranker_score"].rank(method="min", ascending=False).astype(int)
            logger.info("Using LGBMRanker for ranking")
        except Exception as e:
            logger.warning(f"Ranker prediction failed, falling back: {e}")
            df["predicted_rank"] = df["pred_total_sec"].rank(method="min", ascending=True).astype(int)
    elif has_pct_model:
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
        "pred_t1_hms",
        "pred_bike_hms",
        "pred_t2_hms",
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
