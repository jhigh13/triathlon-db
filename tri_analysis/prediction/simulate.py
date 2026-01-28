"""
Monte Carlo simulation for probability estimates.

Simulates many races to compute:
- Win/podium/top5/top10/top20 probabilities
- Expected rank and rank intervals (p10, p90)
- Total time intervals (p10, p50, p90)

Uses a simple uncertainty model based on historical std_total_sec_24m.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default uncertainty parameters
DEFAULT_SIGMA_TOTAL = 90.0  # Default std dev for total time (seconds)
DEFAULT_SIGMA_SWIM = 15.0
DEFAULT_SIGMA_BIKE = 45.0
DEFAULT_SIGMA_RUN = 30.0
PACK_BONUS_SEC = -30.0  # Seconds faster for front pack (bike)
PACK_PENALTY_SEC = 20.0  # Seconds slower if miss front pack (bike)


def estimate_uncertainty(
    pred_df: pd.DataFrame,
    sigma_total_col: str = "std_total_sec_24m",
    default_sigma: float = DEFAULT_SIGMA_TOTAL
) -> pd.DataFrame:
    """
    Add uncertainty (sigma) columns to prediction DataFrame.

    Uses per-athlete std_total_sec_24m if available, otherwise default.

    Args:
        pred_df: DataFrame with predictions
        sigma_total_col: Column name for athlete's historical std dev
        default_sigma: Default sigma if not available

    Returns:
        DataFrame with sigma_total column added
    """
    df = pred_df.copy()

    # Use athlete's historical std if available, else default
    if sigma_total_col in df.columns:
        df["sigma_total"] = df[sigma_total_col].fillna(default_sigma)
        # Clamp to reasonable range [30, 300] seconds
        df["sigma_total"] = df["sigma_total"].clip(30, 300)
    else:
        df["sigma_total"] = default_sigma

    # Simple split sigmas (could be refined later)
    df["sigma_swim"] = DEFAULT_SIGMA_SWIM
    df["sigma_bike"] = DEFAULT_SIGMA_BIKE
    df["sigma_run"] = DEFAULT_SIGMA_RUN

    return df


def run_monte_carlo(
    pred_df: pd.DataFrame,
    n_sims: int = 10000,
    random_state: int = 42,
    use_pack_effects: bool = True
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation to estimate probabilities and intervals.

    For each simulation:
    1. Sample a "race day form" delta per athlete (shared across splits)
    2. Sample individual split noise
    3. Optionally apply pack effects (front pack -> faster bike)
    4. Compute total time and rank
    5. Track outcomes

    Args:
        pred_df: DataFrame with predictions and sigma columns
        n_sims: Number of simulations (default 10000)
        random_state: Random seed for reproducibility
        use_pack_effects: Whether to apply draft-legal pack bonuses

    Returns:
        DataFrame with original columns plus probability/interval columns:
        - prob_win, prob_podium, prob_top5, prob_top10, prob_top20
        - expected_rank, rank_p10, rank_p50, rank_p90
        - total_p10, total_p50, total_p90 (seconds)
    """
    if pred_df.empty:
        logger.warning("Empty pred_df, returning empty DataFrame")
        return pred_df.copy()

    # Ensure uncertainty columns exist
    df = estimate_uncertainty(pred_df)

    n_athletes = len(df)
    rng = np.random.default_rng(random_state)

    # Extract arrays for vectorized simulation
    pred_total = df["pred_total_sec"].values
    sigma_total = df["sigma_total"].values
    front_pack_rate = df["front_pack_rate"].fillna(0.5).values if "front_pack_rate" in df.columns else np.full(n_athletes, 0.5)

    # Track outcomes across simulations
    rank_matrix = np.zeros((n_sims, n_athletes), dtype=np.int32)
    total_matrix = np.zeros((n_sims, n_athletes), dtype=np.float64)

    logger.info(f"Running {n_sims} Monte Carlo simulations for {n_athletes} athletes")

    for sim in range(n_sims):
        # 1. Sample race-day form factor (shared delta per athlete)
        #    This introduces correlation across splits
        form_delta = rng.normal(0, sigma_total * 0.5)  # 50% of variance from shared form

        # 2. Sample individual noise for this race
        individual_noise = rng.normal(0, sigma_total * 0.5)  # 50% from race-specific noise

        # 3. Apply pack effects (optional)
        pack_effect = np.zeros(n_athletes)
        if use_pack_effects:
            # Sample front pack membership for each athlete
            front_pack = rng.random(n_athletes) < front_pack_rate
            pack_effect = np.where(front_pack, PACK_BONUS_SEC, PACK_PENALTY_SEC)

        # 4. Compute simulated total
        sim_total = pred_total + form_delta + individual_noise + pack_effect
        total_matrix[sim, :] = sim_total

        # 5. Compute ranks (1 = fastest)
        ranks = np.argsort(np.argsort(sim_total)) + 1
        rank_matrix[sim, :] = ranks

    # Aggregate outcomes
    logger.info("Aggregating simulation results")

    # Probability calculations
    prob_win = (rank_matrix == 1).mean(axis=0)
    prob_podium = (rank_matrix <= 3).mean(axis=0)
    prob_top5 = (rank_matrix <= 5).mean(axis=0)
    prob_top10 = (rank_matrix <= 10).mean(axis=0)
    prob_top20 = (rank_matrix <= 20).mean(axis=0)

    # Rank statistics
    expected_rank = rank_matrix.mean(axis=0)
    rank_p10 = np.percentile(rank_matrix, 10, axis=0)
    rank_p50 = np.percentile(rank_matrix, 50, axis=0)
    rank_p90 = np.percentile(rank_matrix, 90, axis=0)

    # Time statistics
    total_p10 = np.percentile(total_matrix, 10, axis=0)
    total_p50 = np.percentile(total_matrix, 50, axis=0)
    total_p90 = np.percentile(total_matrix, 90, axis=0)

    # Add results to DataFrame
    result_df = df.copy()
    result_df["prob_win"] = prob_win
    result_df["prob_podium"] = prob_podium
    result_df["prob_top5"] = prob_top5
    result_df["prob_top10"] = prob_top10
    result_df["prob_top20"] = prob_top20
    result_df["expected_rank"] = expected_rank
    result_df["rank_p10"] = rank_p10.astype(int)
    result_df["rank_p50"] = rank_p50.astype(int)
    result_df["rank_p90"] = rank_p90.astype(int)
    result_df["total_p10"] = total_p10
    result_df["total_p50"] = total_p50
    result_df["total_p90"] = total_p90

    # Sort by expected rank
    result_df = result_df.sort_values("expected_rank").reset_index(drop=True)

    logger.info(f"Simulation complete. Top 3 win probabilities: {prob_win[:3]}")

    return result_df


def format_simulation_output(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format simulation output for display/export.

    Returns a clean DataFrame with key columns for reporting.
    """
    from .utils_time import seconds_to_hms

    output_df = sim_df.copy()

    # Format probabilities as percentages
    for col in ["prob_win", "prob_podium", "prob_top5", "prob_top10", "prob_top20"]:
        if col in output_df.columns:
            output_df[col + "_pct"] = (output_df[col] * 100).round(1)

    # Format time intervals as HH:MM:SS
    for col in ["total_p10", "total_p50", "total_p90"]:
        if col in output_df.columns:
            output_df[col + "_hms"] = output_df[col].apply(
                lambda x: seconds_to_hms(int(x)) if pd.notna(x) else None
            )

    # Build rank interval string
    if all(c in output_df.columns for c in ["rank_p10", "rank_p90"]):
        output_df["rank_interval"] = output_df.apply(
            lambda r: f"{int(r['rank_p10'])}-{int(r['rank_p90'])}", axis=1
        )

    # Select display columns
    display_cols = [
        "predicted_rank",
        "athlete_full_name",
        "athlete_country_name",
        "prob_win_pct",
        "prob_podium_pct",
        "prob_top5_pct",
        "prob_top10_pct",
        "expected_rank",
        "rank_interval",
        "total_p50_hms",
        "pred_total_hms",
    ]

    available_cols = [c for c in display_cols if c in output_df.columns]

    result = output_df[available_cols].copy()
    result = result.rename(columns={
        "predicted_rank": "Det. Rank",
        "athlete_full_name": "Athlete",
        "athlete_country_name": "Country",
        "prob_win_pct": "Win %",
        "prob_podium_pct": "Podium %",
        "prob_top5_pct": "Top5 %",
        "prob_top10_pct": "Top10 %",
        "expected_rank": "E[Rank]",
        "rank_interval": "Rank 80% CI",
        "total_p50_hms": "Median Time",
        "pred_total_hms": "Pred Total",
    })

    return result
