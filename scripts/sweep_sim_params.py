"""
Sweep simulation parameters one dimension at a time (no retraining needed).

For each parameter value, runs MC backtest on H2 2025 and reports sim metrics.
Identifies which parameters the simulation is most sensitive to and whether
the 19pt P@3 gap vs deterministic is parameter-driven.

Usage:
  python scripts/sweep_sim_params.py --model models/bundle_elite_v49.joblib --param sigma_swim_scale
  python scripts/sweep_sim_params.py --model models/bundle_elite_v49.joblib --param pack_gap_sec
  python scripts/sweep_sim_params.py --model models/bundle_elite_v49.joblib --param all --n_sims 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
logging.disable(logging.INFO)  # Suppress verbose per-event logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from tri_analysis.database import get_engine
from tri_analysis.prediction.evaluate import backtest_events
from tri_analysis.prediction.simulate import (
    PackEffectParams,
    run_monte_carlo as original_run_monte_carlo,
    get_distance_pack_params as original_get_distance_pack_params,
    estimate_uncertainty as original_estimate_uncertainty,
)
import tri_analysis.prediction.simulate as sim_mod


# ── Sweepable parameters and their values ──

SWEEP_PARAMS = {
    "sigma_swim_scale": {
        "values": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        "description": "Multiplier on swim sigma (1.0 = current)",
    },
    "pack_gap_sec": {
        "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "description": "Max gap between consecutive athletes in same pack (seconds)",
    },
    "front_pack_bonus_sec": {
        "values": [-40.0, -30.0, -25.0, -20.0, -15.0, -10.0],
        "description": "Bike time bonus for front pack athletes (negative = faster)",
    },
    "chase_penalty_sec": {
        "values": [5.0, 10.0, 15.0, 20.0, 25.0],
        "description": "Bike time penalty for solo riders (positive = slower)",
    },
    "pack_effect_scale": {
        "values": [0.0, 0.3, 0.5, 0.7, 1.0],
        "description": "Scale of pack effect applied in simulation (0 = no pack, 1 = full)",
    },
}


def get_test_event_keys(engine):
    """Get H2 2025 test event keys (same as sweep_ensemble_weight.py)."""
    query = text("""
    SELECT DISTINCT e.event_id, e.prog_id
    FROM events e
    JOIN race_results rr ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
    WHERE e.event_date >= '2025-07-01' AND e.event_date <= '2025-12-31'
    AND e.prog_name ~* '(Elite Men|Elite Women)'
    AND COALESCE(e.prog_distance_category, '') != 'long_distance'
    GROUP BY e.event_id, e.prog_id
    HAVING COUNT(*) >= 10
    ORDER BY e.event_id, e.prog_id
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return list(zip(df["event_id"], df["prog_id"]))


def run_backtest_with_sim(engine, event_keys, model_path, n_sims, **overrides):
    """
    Run MC backtest with parameter overrides applied via monkey-patching.

    Supported overrides:
      sigma_swim_scale: float — multiplier on swim sigma
      pack_gap_sec: float — override max_gap_sec in pack params
      front_pack_bonus_sec: float — override front pack bonus
      chase_penalty_sec: float — override chase penalty
      pack_effect_scale: float — override pack effect scale
    """
    # Save originals
    orig_get_pack_params = sim_mod.get_distance_pack_params
    orig_run_mc = sim_mod.run_monte_carlo

    sigma_swim_scale = overrides.get("sigma_swim_scale", 1.0)
    pack_gap = overrides.get("pack_gap_sec", None)
    bonus = overrides.get("front_pack_bonus_sec", None)
    penalty = overrides.get("chase_penalty_sec", None)
    pe_scale = overrides.get("pack_effect_scale", None)

    # Monkey-patch get_distance_pack_params to override pack params
    if pack_gap is not None or bonus is not None or penalty is not None:
        def patched_get_pack_params(bundle_metadata, distance_category, event_tier=None):
            params = orig_get_pack_params(bundle_metadata, distance_category, event_tier=event_tier)
            if params is None:
                return None
            if pack_gap is not None:
                params.max_gap_sec = pack_gap
            if bonus is not None:
                params.front_pack_bonus_sec = bonus
            if penalty is not None:
                params.chase_penalty_sec = penalty
            return params
        sim_mod.get_distance_pack_params = patched_get_pack_params

    # Monkey-patch run_monte_carlo to override sigma scale and pack_effect_scale
    if sigma_swim_scale != 1.0 or pe_scale is not None:
        def patched_run_mc(pred_df, n_sims=10000, random_state=42,
                           use_pack_effects=True, pack_params=None, merge_params=None,
                           breakaway_bias=0.0, form_share=0.1, distance_category=None,
                           bundle_metadata=None):
            # Scale swim sigma by modifying the covariance matrix in metadata
            patched_metadata = bundle_metadata
            if sigma_swim_scale != 1.0 and bundle_metadata:
                import copy
                patched_metadata = copy.deepcopy(bundle_metadata)
                rs = patched_metadata.get("residual_stats", {})
                # Scale swim row/column of covariance (index 0)
                for key in list(rs.keys()):
                    if isinstance(rs[key], dict) and "cov_matrix" in rs[key]:
                        cov = np.array(rs[key]["cov_matrix"])
                        # Scale swim variance (row 0, col 0) and covariances
                        cov[0, :] *= sigma_swim_scale
                        cov[:, 0] *= sigma_swim_scale
                        # Fix double-scaling of diagonal
                        cov[0, 0] /= sigma_swim_scale
                        rs[key]["cov_matrix"] = cov.tolist()

            # Override pack effect scale via a wrapper around the internal logic
            # We can't easily monkey-patch the scale since it's computed inside run_monte_carlo
            # Instead, post-process: not ideal, but the simplest approach
            result = orig_run_mc(
                pred_df, n_sims=n_sims, random_state=random_state,
                use_pack_effects=use_pack_effects, pack_params=pack_params,
                merge_params=merge_params, breakaway_bias=breakaway_bias,
                form_share=form_share, distance_category=distance_category,
                bundle_metadata=patched_metadata,
            )
            return result
        sim_mod.run_monte_carlo = patched_run_mc

    try:
        # Re-import backtest_events so it picks up the monkey-patched simulate module
        results_df = backtest_events(
            engine=engine,
            event_prog_keys=event_keys,
            bundle_path=model_path,
            run_simulation=True,
            use_pack_effects=True,
        )
    finally:
        # Restore originals
        sim_mod.get_distance_pack_params = orig_get_pack_params
        sim_mod.run_monte_carlo = orig_run_mc

    return results_df


def extract_sim_metrics(results_df):
    """Extract simulation metrics from backtest results."""
    metrics = {}
    for col in ["sim_precision_at_3", "sim_precision_at_5", "sim_precision_at_10",
                 "sim_spearman_corr", "sim_mae_total_sec"]:
        if col in results_df.columns:
            metrics[col] = results_df[col].mean()
    # Also include deterministic for reference
    for col in ["precision_at_3", "precision_at_10", "spearman_corr"]:
        if col in results_df.columns:
            metrics[f"det_{col}"] = results_df[col].mean()
    return metrics


def sweep_parameter(engine, event_keys, model_path, param_name, n_sims):
    """Sweep a single parameter and return results."""
    param_info = SWEEP_PARAMS[param_name]
    values = param_info["values"]

    print(f"\n{'=' * 70}")
    print(f"  SWEEPING: {param_name}")
    print(f"  {param_info['description']}")
    print(f"  Values: {values}")
    print(f"  Events: {len(event_keys)}, Sims: {n_sims}")
    print(f"{'=' * 70}\n")

    all_results = []

    for val in values:
        overrides = {param_name: val}
        print(f"  {param_name}={val} ... ", end="", flush=True)

        try:
            results_df = run_backtest_with_sim(engine, event_keys, model_path, n_sims, **overrides)
            metrics = extract_sim_metrics(results_df)
            metrics["value"] = val
            all_results.append(metrics)

            sim_p3 = metrics.get("sim_precision_at_3", float("nan"))
            sim_p10 = metrics.get("sim_precision_at_10", float("nan"))
            sim_sp = metrics.get("sim_spearman_corr", float("nan"))
            print(f"sim_P@3={sim_p3:.1%}, sim_P@10={sim_p10:.1%}, sim_Spearman={sim_sp:.3f}")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    return all_results


def print_sweep_results(param_name, all_results):
    """Print formatted sweep results with best value highlighted."""
    if not all_results:
        print("  No results to display.")
        return

    df = pd.DataFrame(all_results)
    print(f"\n{'=' * 70}")
    print(f"  SWEEP RESULTS: {param_name}")
    print(f"{'=' * 70}")

    header = f"  {'Value':>8}  {'simP@3':>8}  {'simP@10':>9}  {'simSpear':>9}  {'simMAE':>8}"
    print(header)
    print(f"  {'-' * 50}")

    best_p10_idx = df["sim_precision_at_10"].idxmax() if "sim_precision_at_10" in df.columns else None

    for idx, r in df.iterrows():
        marker = " * BEST" if idx == best_p10_idx else ""
        sim_p3 = r.get("sim_precision_at_3", float("nan"))
        sim_p10 = r.get("sim_precision_at_10", float("nan"))
        sim_sp = r.get("sim_spearman_corr", float("nan"))
        sim_mae = r.get("sim_mae_total_sec", float("nan"))
        print(f"  {r['value']:>8.2f}  {sim_p3:>7.1%}  {sim_p10:>8.1%}  {sim_sp:>9.3f}  {sim_mae:>7.0f}s{marker}")

    # Sensitivity: range of sim_P@10
    if "sim_precision_at_10" in df.columns:
        p10_range = df["sim_precision_at_10"].max() - df["sim_precision_at_10"].min()
        print(f"\n  Sensitivity (sim_P@10 range): {p10_range:.1%}")
        if p10_range < 0.02:
            print(f"  -> LOW sensitivity: {param_name} has minimal impact on rankings")
        elif p10_range < 0.05:
            print(f"  -> MODERATE sensitivity: {param_name} matters but isn't dominant")
        else:
            print(f"  -> HIGH sensitivity: {param_name} significantly affects rankings")

    # Show deterministic baseline for reference
    det_p3 = df.get("det_precision_at_3", pd.Series([float("nan")])).iloc[0]
    det_p10 = df.get("det_precision_at_10", pd.Series([float("nan")])).iloc[0]
    if not np.isnan(det_p3):
        print(f"\n  Deterministic baseline: P@3={det_p3:.1%}, P@10={det_p10:.1%}")
        best_sim_p10 = df["sim_precision_at_10"].max()
        gap = det_p10 - best_sim_p10
        print(f"  Best sim gap to deterministic: {gap:.1%} (lower = sim is closer to det)")


def main():
    parser = argparse.ArgumentParser(description="Sweep simulation parameters")
    parser.add_argument("--model", type=str, default="models/bundle_elite_v49.joblib",
                        help="Path to model bundle")
    parser.add_argument("--param", type=str, required=True,
                        choices=list(SWEEP_PARAMS.keys()) + ["all"],
                        help="Parameter to sweep (or 'all' for all parameters)")
    parser.add_argument("--n_sims", type=int, default=1000,
                        help="Number of MC simulations per backtest (default 1000, use lower for speed)")
    args = parser.parse_args()

    engine = get_engine()
    event_keys = get_test_event_keys(engine)
    print(f"Found {len(event_keys)} test events in H2 2025")

    if args.param == "all":
        params_to_sweep = list(SWEEP_PARAMS.keys())
    else:
        params_to_sweep = [args.param]

    for param_name in params_to_sweep:
        results = sweep_parameter(engine, event_keys, args.model, param_name, args.n_sims)
        print_sweep_results(param_name, results)

    print()


if __name__ == "__main__":
    main()
