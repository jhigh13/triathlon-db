"""
Evaluation metrics and backtesting for the prediction pipeline.

Provides:
- Precision@K: Fraction of true top-K in predicted top-K
- Spearman rank correlation
- MAE for time predictions
- Backtest harness for historical evaluation
"""

from __future__ import annotations
from typing import Optional
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def precision_at_k(
    pred_order: list[int] | pd.Series,
    true_order: list[int] | pd.Series,
    k: int
) -> float:
    """
    Compute Precision@K: fraction of true top-K athletes in predicted top-K.

    Args:
        pred_order: List of athlete_ids sorted by predicted finish (best first)
        true_order: List of athlete_ids sorted by actual finish (best first)
        k: Number of top positions to evaluate

    Returns:
        Precision@K as a float in [0, 1]

    Example:
        >>> precision_at_k([1, 2, 3, 4, 5], [2, 1, 3, 5, 4], k=3)
        1.0
        >>> precision_at_k([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], k=3)
        0.333...
    """
    pred_topk = set(list(pred_order)[:k])
    true_topk = set(list(true_order)[:k])

    if not true_topk:
        return 0.0

    overlap = len(pred_topk & true_topk)
    return overlap / k


def spearman_rank_corr(
    pred_rank: pd.Series,
    true_rank: pd.Series
) -> float:
    """
    Compute Spearman rank correlation between predicted and actual ranks.

    Args:
        pred_rank: Series with athlete_id as index, predicted rank as values
        true_rank: Series with athlete_id as index, actual rank as values

    Returns:
        Spearman correlation coefficient in [-1, 1]
    """
    # Align on common athletes
    common_idx = pred_rank.index.intersection(true_rank.index)

    if len(common_idx) < 2:
        return np.nan

    p = pred_rank.loc[common_idx]
    t = true_rank.loc[common_idx]

    corr, _ = spearmanr(p, t)
    return corr


def compute_mae(
    pred_sec: pd.Series,
    true_sec: pd.Series
) -> float:
    """
    Compute Mean Absolute Error for time predictions (in seconds).

    Args:
        pred_sec: Predicted times in seconds
        true_sec: Actual times in seconds

    Returns:
        MAE in seconds
    """
    # Filter to valid pairs
    valid = pred_sec.notna() & true_sec.notna()
    if valid.sum() == 0:
        return np.nan

    return np.abs(pred_sec[valid] - true_sec[valid]).mean()


def evaluate_program_predictions(
    pred_df: pd.DataFrame,
    results_df: pd.DataFrame
) -> dict:
    """
    Evaluate predictions against actual results for a single program.

    Args:
        pred_df: DataFrame with predictions (athlete_id, pred_total_sec, predicted_rank)
        results_df: DataFrame with actual results (athlete_id, total_sec, finish_position)

    Returns:
        Dict of metric name -> value
    """
    # Merge predictions with results
    merged = pred_df.merge(
        results_df[["athlete_id", "total_sec", "finish_position", "finish_status"]],
        on="athlete_id",
        how="inner",
        suffixes=("_pred", "_actual")
    )

    # Filter to finishers
    merged = merged[merged["finish_status"] == "FINISH"]

    if merged.empty:
        return {"error": "No finishers to evaluate"}

    metrics = {}

    # Precision@K
    pred_order = merged.sort_values("predicted_rank")["athlete_id"].tolist()
    true_order = merged.sort_values("finish_position")["athlete_id"].tolist()

    for k in [3, 5, 10, 20]:
        if k <= len(true_order):
            metrics[f"precision_at_{k}"] = precision_at_k(pred_order, true_order, k)

    # Spearman correlation
    pred_rank = merged.set_index("athlete_id")["predicted_rank"]
    true_rank = merged.set_index("athlete_id")["finish_position"]
    metrics["spearman_corr"] = spearman_rank_corr(pred_rank, true_rank)

    # MAE for times
    if "pred_total_sec" in merged.columns and "total_sec" in merged.columns:
        metrics["mae_total_sec"] = compute_mae(merged["pred_total_sec"], merged["total_sec"])

    metrics["n_evaluated"] = len(merged)

    return metrics


def backtest_events(
    engine,
    event_prog_keys: list[tuple[int, int]],
    bundle_path: str,
    feature_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Run backtests on multiple historical events.

    For each event:
    1. Build features using only data before event_date
    2. Generate predictions with the model
    3. Compare to actual results
    4. Compute evaluation metrics

    Args:
        engine: SQLAlchemy Engine
        event_prog_keys: List of (event_id, prog_id) tuples to evaluate
        bundle_path: Path to saved ModelBundle
        feature_cols: Feature columns (uses bundle defaults if None)

    Returns:
        DataFrame with one row per event and metric columns
    """
    from .sql import ProgramKey, fetch_program_results
    from .features import build_features_for_program, fill_missing_features, get_feature_columns
    from .train import load_model_bundle
    from .predict import predict_splits_and_total
    from .utils_time import parse_time_to_seconds

    bundle = load_model_bundle(bundle_path)
    if feature_cols is None:
        feature_cols = bundle.feature_columns or get_feature_columns()

    results = []

    for event_id, prog_id in event_prog_keys:
        key = ProgramKey(event_id=event_id, prog_id=prog_id)
        logger.info(f"Backtesting {key}")

        try:
            # Build features (using results as athlete list)
            features_df = build_features_for_program(engine, key, use_start_list=False)

            if features_df.empty:
                logger.warning(f"No features for {key}")
                continue

            # Fill missing features
            features_df = fill_missing_features(features_df, feature_cols)

            # Generate predictions
            pred_df = predict_splits_and_total(features_df, bundle)

            # Get actual results
            results_df = fetch_program_results(engine, key)
            results_df["total_sec"] = results_df["total_time"].apply(parse_time_to_seconds)

            # Evaluate
            metrics = evaluate_program_predictions(pred_df, results_df)
            metrics["event_id"] = event_id
            metrics["prog_id"] = prog_id

            results.append(metrics)

        except Exception as e:
            logger.error(f"Error backtesting {key}: {e}")
            continue

    if not results:
        return pd.DataFrame()

    backtest_df = pd.DataFrame(results)

    # Compute summary statistics
    logger.info(f"Backtest complete: {len(backtest_df)} events evaluated")
    for col in ["precision_at_3", "precision_at_10", "spearman_corr", "mae_total_sec"]:
        if col in backtest_df.columns:
            logger.info(f"  Mean {col}: {backtest_df[col].mean():.3f}")

    return backtest_df
