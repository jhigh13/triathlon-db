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
import traceback

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


def compute_start_num_baseline(
    engine,
    event_prog_keys: list[tuple[int, int]],
) -> pd.DataFrame:
    """
    Compute a naive baseline that ranks athletes by start_num (bib number).

    In World Triathlon Elite events, start numbers are typically assigned
    based on world ranking/seeding. This baseline measures how well seeding
    alone predicts race outcomes.

    Args:
        engine: SQLAlchemy Engine
        event_prog_keys: List of (event_id, prog_id) tuples

    Returns:
        DataFrame with one row per event and metric columns (same format as backtest_events)
    """
    from sqlalchemy import text
    from .utils_time import parse_time_to_seconds

    results = []
    for event_id, prog_id in event_prog_keys:
        try:
            query = text("""
                SELECT rr.athlete_id, rr.start_num,
                       rr.finish_position, rr.finish_status,
                       rr.total_time
                FROM race_results rr
                WHERE rr.event_id = :event_id
                  AND rr.prog_id = :prog_id
                ORDER BY rr.position_sort
            """)
            with engine.connect() as conn:
                df = pd.read_sql(query, conn, params={"event_id": int(event_id), "prog_id": int(prog_id)})

            if df.empty:
                continue

            # Parse start_num to numeric (stored as VARCHAR like "12.0")
            df["start_num_int"] = pd.to_numeric(df["start_num"], errors="coerce")

            # Filter to finishers with valid start numbers
            finishers = df[
                (df["finish_status"] == "FINISH") & df["start_num_int"].notna()
            ].copy()

            if len(finishers) < 3:
                continue

            # Rank by start_num (lowest = predicted best)
            pred_order = finishers.sort_values("start_num_int")["athlete_id"].tolist()
            true_order = finishers.sort_values("finish_position")["athlete_id"].tolist()

            metrics = {"event_id": event_id, "prog_id": prog_id}

            for k in [3, 5, 10, 20]:
                if k <= len(true_order):
                    metrics[f"precision_at_{k}"] = precision_at_k(pred_order, true_order, k)

            # Spearman: correlate start_num rank with finish position
            pred_rank = finishers.set_index("athlete_id")["start_num_int"].rank()
            true_rank = finishers.set_index("athlete_id")["finish_position"]
            metrics["spearman_corr"] = spearman_rank_corr(pred_rank, true_rank)

            # MAE not applicable (start_num doesn't predict time)
            metrics["mae_total_sec"] = np.nan
            metrics["n_evaluated"] = len(finishers)

            results.append(metrics)

        except Exception as e:
            logger.error(f"Error computing start_num baseline for ({event_id}, {prog_id}): {e}")
            continue

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def backtest_events(
    engine,
    event_prog_keys: list[tuple[int, int]],
    bundle_path: str | dict[str, str],
    feature_cols: Optional[list[str]] = None,
    run_simulation: bool = False,
) -> pd.DataFrame:
    """
    Run backtests on multiple historical events.

    For each event:
    1. Build features using only data before event_date
    2. Generate predictions with the model
    3. Compare to actual results
    4. Compute evaluation metrics
    5. Optionally run Monte Carlo simulation and evaluate sim-based rankings

    Args:
        engine: SQLAlchemy Engine
        event_prog_keys: List of (event_id, prog_id) tuples to evaluate
        bundle_path: Path to saved ModelBundle, or dict like
                     {"men": "path_men.joblib", "women": "path_women.joblib"}
                     for gender-specific models
        feature_cols: Feature columns (uses bundle defaults if None)
        run_simulation: If True, also run MC simulation and evaluate sim rankings

    Returns:
        DataFrame with one row per event and metric columns.
        When run_simulation=True, includes sim_ prefixed columns alongside deterministic ones.
    """
    from .sql import ProgramKey, fetch_program_results, fetch_event_metadata
    from .features import build_features_for_program, fill_missing_features, get_feature_columns
    from .train import load_model_bundle
    from .predict import predict_splits_and_total
    from .utils_time import parse_time_to_seconds

    if run_simulation:
        from .simulate import run_monte_carlo, get_distance_pack_params, get_distance_merge_params

    # Load bundle(s)
    if isinstance(bundle_path, dict):
        bundles = {k: load_model_bundle(v) for k, v in bundle_path.items()}
        # Use feature cols from first bundle
        if feature_cols is None:
            first_bundle = next(iter(bundles.values()))
            feature_cols = first_bundle.feature_columns or get_feature_columns()
    else:
        bundle = load_model_bundle(bundle_path)
        bundles = None
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

            # Select the right bundle for gender-specific models
            if bundles is not None:
                # Determine gender from program results
                results_df = fetch_program_results(engine, key)
                prog_name = results_df["prog_name"].iloc[0] if not results_df.empty else ""
                if "women" in prog_name.lower():
                    active_bundle = bundles.get("women", next(iter(bundles.values())))
                else:
                    active_bundle = bundles.get("men", next(iter(bundles.values())))
            else:
                active_bundle = bundle
                results_df = None

            # Get distance category for distance-specific split models
            event_meta = fetch_event_metadata(engine, key)
            distance_cat = event_meta.get("prog_distance_category") if event_meta else None

            # Generate predictions
            pred_df = predict_splits_and_total(features_df, active_bundle, distance_category=distance_cat)

            # Get actual results (reuse if already fetched above)
            if results_df is None:
                results_df = fetch_program_results(engine, key)
            results_df["total_sec"] = results_df["total_time"].apply(parse_time_to_seconds)

            # Evaluate deterministic predictions
            metrics = evaluate_program_predictions(pred_df, results_df)
            metrics["event_id"] = event_id
            metrics["prog_id"] = prog_id

            # Optionally run Monte Carlo simulation and evaluate
            if run_simulation:
                try:
                    pack_params = get_distance_pack_params(active_bundle.metadata, distance_cat)
                    merge_params = get_distance_merge_params(active_bundle.metadata, distance_cat)

                    sim_df = run_monte_carlo(
                        pred_df,
                        n_sims=10000,
                        random_state=42,
                        pack_params=pack_params,
                        merge_params=merge_params,
                        distance_category=distance_cat,
                        bundle_metadata=active_bundle.metadata,
                    )

                    # Build evaluation df using sim rankings
                    sim_eval_df = sim_df.copy()
                    sim_eval_df["predicted_rank"] = sim_eval_df["expected_rank"]
                    sim_eval_df["pred_total_sec"] = sim_eval_df["total_p50"]

                    sim_metrics = evaluate_program_predictions(sim_eval_df, results_df)

                    # Store with sim_ prefix
                    for k, v in sim_metrics.items():
                        metrics[f"sim_{k}"] = v

                except Exception as e:
                    logger.warning(f"Simulation failed for {key}: {e}")

            results.append(metrics)

        except Exception as e:
            logger.error(f"Error backtesting {key}: {e}\n{traceback.format_exc()}")
            continue

    if not results:
        return pd.DataFrame()

    backtest_df = pd.DataFrame(results)

    # Compute summary statistics
    logger.info(f"Backtest complete: {len(backtest_df)} events evaluated")
    for col in ["precision_at_3", "precision_at_10", "spearman_corr", "mae_total_sec"]:
        if col in backtest_df.columns:
            logger.info(f"  Mean {col}: {backtest_df[col].mean():.3f}")
    if run_simulation:
        for col in ["sim_precision_at_3", "sim_precision_at_10", "sim_spearman_corr", "sim_mae_total_sec"]:
            if col in backtest_df.columns:
                logger.info(f"  Mean {col}: {backtest_df[col].mean():.3f}")

    return backtest_df


def compute_simulation_calibration(
    sim_df: pd.DataFrame,
    results_df: pd.DataFrame,
    athlete_id_col: str = "athlete_id",
) -> dict:
    """
    Check what fraction of actual results fall within simulation confidence intervals.

    Merges simulation output (with rank/time percentile columns) against actual
    results and computes coverage: the fraction of athletes whose actual outcome
    falls within the predicted [p10, p90] interval.

    Well-calibrated simulations should show ~80% coverage for [p10, p90].
    Higher = uncertainty too wide (overconfident intervals paradoxically means
    too-narrow would be LOWER coverage). Lower = uncertainty too narrow.

    Args:
        sim_df: Simulation output DataFrame with columns:
            total_p10, total_p90, rank_p10, rank_p90, athlete_id
        results_df: Actual results DataFrame with columns:
            athlete_id, total_sec (or total_time_sec), finish_position
        athlete_id_col: Column name for joining

    Returns:
        Dict with calibration metrics:
        {
            "time_coverage_80": float,   # fraction within [p10, p90] for total time
            "rank_coverage_80": float,   # fraction within [p10, p90] for rank
            "n_athletes": int,           # number of athletes evaluated
            "time_bias": float,          # mean(actual - sim_median), positive = sim too fast
            "rank_bias": float,          # mean(actual_rank - expected_rank)
        }
    """
    # Determine actual total time column
    time_col = None
    for col in ["total_sec", "total_time_sec"]:
        if col in results_df.columns:
            time_col = col
            break

    rank_col = None
    for col in ["finish_position", "position_sort"]:
        if col in results_df.columns:
            rank_col = col
            break

    # Merge simulation with actuals
    merge_cols = [athlete_id_col]
    result_cols = merge_cols.copy()
    if time_col:
        result_cols.append(time_col)
    if rank_col:
        result_cols.append(rank_col)

    merged = sim_df.merge(
        results_df[result_cols].drop_duplicates(subset=[athlete_id_col]),
        on=athlete_id_col,
        how="inner",
    )

    if merged.empty:
        logger.warning("No matching athletes between sim and results for calibration")
        return {
            "time_coverage_80": np.nan,
            "rank_coverage_80": np.nan,
            "n_athletes": 0,
            "time_bias": np.nan,
            "rank_bias": np.nan,
        }

    n_athletes = len(merged)
    result = {"n_athletes": n_athletes}

    # Time coverage: actual total within [p10, p90]
    if time_col and "total_p10" in merged.columns and "total_p90" in merged.columns:
        actual_time = pd.to_numeric(merged[time_col], errors="coerce")
        valid_time = actual_time.notna()
        if valid_time.sum() > 0:
            in_interval = (
                (actual_time[valid_time] >= merged.loc[valid_time, "total_p10"]) &
                (actual_time[valid_time] <= merged.loc[valid_time, "total_p90"])
            )
            result["time_coverage_80"] = float(in_interval.mean())

            # Bias: positive = sim predicted too fast (actual slower than median)
            if "total_p50" in merged.columns:
                result["time_bias"] = float(
                    (actual_time[valid_time] - merged.loc[valid_time, "total_p50"]).mean()
                )
            else:
                result["time_bias"] = np.nan
        else:
            result["time_coverage_80"] = np.nan
            result["time_bias"] = np.nan
    else:
        result["time_coverage_80"] = np.nan
        result["time_bias"] = np.nan

    # Rank coverage: actual rank within [p10, p90]
    if rank_col and "rank_p10" in merged.columns and "rank_p90" in merged.columns:
        actual_rank = pd.to_numeric(merged[rank_col], errors="coerce")
        valid_rank = actual_rank.notna()
        if valid_rank.sum() > 0:
            in_interval = (
                (actual_rank[valid_rank] >= merged.loc[valid_rank, "rank_p10"]) &
                (actual_rank[valid_rank] <= merged.loc[valid_rank, "rank_p90"])
            )
            result["rank_coverage_80"] = float(in_interval.mean())

            # Rank bias
            if "expected_rank" in merged.columns:
                result["rank_bias"] = float(
                    (actual_rank[valid_rank] - merged.loc[valid_rank, "expected_rank"]).mean()
                )
            else:
                result["rank_bias"] = np.nan
        else:
            result["rank_coverage_80"] = np.nan
            result["rank_bias"] = np.nan
    else:
        result["rank_coverage_80"] = np.nan
        result["rank_bias"] = np.nan

    logger.info(
        f"Simulation calibration ({n_athletes} athletes): "
        f"time_coverage={result.get('time_coverage_80', 'N/A'):.1%}, "
        f"rank_coverage={result.get('rank_coverage_80', 'N/A'):.1%}, "
        f"time_bias={result.get('time_bias', 'N/A'):.1f}s, "
        f"rank_bias={result.get('rank_bias', 'N/A'):.1f}"
    )

    return result
