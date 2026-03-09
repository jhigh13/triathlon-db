#!/usr/bin/env python
"""
Predict race outcomes for an upcoming WTCS program.

Usage:
    python scripts/predict_program.py --event_id 123 --prog_id 456 --model_path models/bundle.joblib

Options:
    --event_id      Event ID for the upcoming race
    --prog_id       Program ID (e.g., Men Elite, Women Elite)
    --model_path    Path to saved ModelBundle (default: models/bundle.joblib)
    --n_sims        Number of Monte Carlo simulations (default: 10000)
    --output_dir    Output directory for CSV (default: outputs/)
    --no_mc         Skip Monte Carlo simulation (deterministic only)

Output:
    - Prints top 20 predictions to console
    - Saves full predictions CSV to outputs/predictions_{event_id}_{prog_id}.csv
"""

from __future__ import annotations
import argparse
import logging
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from tri_analysis.database import get_engine
from tri_analysis.prediction.sql import ProgramKey, fetch_event_metadata
from tri_analysis.prediction.features import (
    build_features_for_program,
    fill_missing_features,
    get_feature_columns,
    classify_event_tier,
)
from tri_analysis.prediction.train import load_model_bundle
from tri_analysis.prediction.predict import predict_splits_and_total, format_prediction_output
from tri_analysis.prediction.simulate import (
    run_monte_carlo,
    format_simulation_output,
    print_sim_diagnostics,
    get_distance_pack_params,
    get_distance_merge_params,
    PackEffectParams,
    MergeParams,
    DRAFT_LEGAL_DISTANCES,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_race_context(
    sim_df: pd.DataFrame,
    pack_params: PackEffectParams,
    distance_category: str | None,
    event_meta: dict,
    merge_params: "MergeParams | None" = None,
    breakaway_bias: float = 0.0,
):
    """Print race context: pack formation analysis, field strength, model parameters."""
    from tri_analysis.prediction.simulate import (
        assign_packs_chain,
        continuous_gap_bike_effect,
        apply_pack_merges,
        DEFAULT_SIGMA_SWIM,
        DEFAULT_SIGMA_T1,
        DEFAULT_SIGMA_BIKE,
        DEFAULT_SIGMA_T2,
        DEFAULT_SIGMA_RUN,
        DISTANCE_SIGMA_MULTIPLIER,
        DEFAULT_FORM_SHARE,
    )
    from tri_analysis.prediction.utils_time import seconds_to_hms

    print("\n" + "=" * 80)
    print("RACE CONTEXT")
    print("=" * 80)

    # ── 1. Predicted Pack Formations ──
    # Use swim + T1 for pack formation (bike entry order), matching the simulation
    if "pred_swim_sec" in sim_df.columns and sim_df["pred_swim_sec"].notna().all():
        swim_times = sim_df["pred_swim_sec"].values
        has_t1 = "pred_t1_sec" in sim_df.columns and sim_df["pred_t1_sec"].notna().all()
        if has_t1:
            bike_entry_times = swim_times + sim_df["pred_t1_sec"].values
            pack_basis_label = "swim + T1"
        else:
            bike_entry_times = swim_times
            pack_basis_label = "swim only (no T1 predictions)"

        sorted_idx = bike_entry_times.argsort()
        sorted_entry = bike_entry_times[sorted_idx]
        sorted_names = sim_df["athlete_full_name"].values[sorted_idx]
        sorted_swim = swim_times[sorted_idx]

        pack_ids = assign_packs_chain(sorted_entry, max_gap_sec=pack_params.max_gap_sec)

        print(f"\n--- Predicted Pack Formations (based on {pack_basis_label}) ---")
        print(f"  Pack gap threshold: {pack_params.max_gap_sec:.1f}s")

        n_packs = pack_ids.max() + 1 if len(pack_ids) > 0 else 0
        for p in range(n_packs):
            mask = pack_ids == p
            pack_size = mask.sum()
            pack_athletes = sorted_names[mask]
            pack_swim_times = sorted_swim[mask]
            swim_min = seconds_to_hms(int(pack_swim_times.min()))
            swim_max = seconds_to_hms(int(pack_swim_times.max()))

            if p == 0:
                label = "Front Pack"
            elif pack_size == 1:
                label = "Solo"
            else:
                label = f"Chase Pack {p}"

            time_range = swim_min if swim_min == swim_max else f"{swim_min}-{swim_max}"
            print(f"\n  {label} ({pack_size} athletes, swim: {time_range}):")

            display_limit = 10
            for i, name in enumerate(pack_athletes[:display_limit]):
                print(f"    {name} ({seconds_to_hms(int(pack_swim_times[i]))})")
            if len(pack_athletes) > display_limit:
                print(f"    ... and {len(pack_athletes) - display_limit} more")

        # ── Pack effect on bike (based on bike entry times = swim + T1) ──
        # Use apply_pack_merges (matching simulation) when merge_params available,
        # otherwise fall back to continuous_gap_bike_effect
        if merge_params is not None:
            rng_display = np.random.default_rng(42)
            bike_effects = apply_pack_merges(
                bike_entry_times, pack_params, merge_params, rng_display, breakaway_bias
            )
            effect_label = "Pack Effect on Bike (with dynamic merging)"
        else:
            bike_effects = continuous_gap_bike_effect(bike_entry_times, pack_params)
            effect_label = "Pack Effect on Bike (static)"

        print(f"\n--- {effect_label} ---")
        for p in range(min(n_packs, 5)):
            p_mask = pack_ids == p
            if not p_mask.any():
                continue
            p_size = p_mask.sum()
            avg_effect = bike_effects[sorted_idx[p_mask]].mean()
            if p == 0:
                label = "Front Pack"
            elif p_size == 1:
                label = f"Solo rider (pack {p})"
            else:
                label = f"Chase Pack {p}"
            print(f"  {label:25s}: {avg_effect:+.1f}s  ({p_size} athletes)")
        if n_packs > 5:
            remaining = sum(1 for pid in range(5, n_packs) if (pack_ids == pid).any())
            print(f"  ... and {remaining} more pack(s)")

    # ── 2. Field Strength ──
    print(f"\n--- Field Strength ---")
    n_entrants = len(sim_df)
    print(f"  Entrants: {n_entrants}")

    if "elo_rating" in sim_df.columns and sim_df["elo_rating"].notna().any():
        elo_vals = sim_df["elo_rating"].dropna()
        print(f"  Elo rating:  mean={elo_vals.mean():.0f}, "
              f"max={elo_vals.max():.0f}, "
              f"median={elo_vals.median():.0f}")

    if "wt_total_points" in sim_df.columns and sim_df["wt_total_points"].notna().any():
        wt_vals = sim_df["wt_total_points"].dropna()
        print(f"  WT points:   mean={wt_vals.mean():.0f}, "
              f"max={wt_vals.max():.0f}")

    if "wt_rank_position" in sim_df.columns and sim_df["wt_rank_position"].notna().any():
        wt_rank = sim_df["wt_rank_position"].dropna()
        print(f"  WT rank:     best={wt_rank.min():.0f}, "
              f"median={wt_rank.median():.0f}")

    if "front_pack_rate" in sim_df.columns and sim_df["front_pack_rate"].notna().any():
        fpr = sim_df["front_pack_rate"].dropna()
        print(f"  Front pack rate (field avg): {fpr.mean():.2f}")

    # ── Weather Conditions ──
    if "temperature_air" in sim_df.columns and sim_df["temperature_air"].notna().any():
        temp = sim_df["temperature_air"].iloc[0]
        print(f"\n--- Weather Conditions ---")
        print(f"  Air temperature:  {temp}°C")
        if "apparent_temp" in sim_df.columns and sim_df["apparent_temp"].notna().any():
            print(f"  Feels like:       {sim_df['apparent_temp'].iloc[0]}°C")
        if "humidity" in sim_df.columns and sim_df["humidity"].notna().any():
            print(f"  Humidity:         {sim_df['humidity'].iloc[0]}%")
        if "wbgt" in sim_df.columns and sim_df["wbgt"].notna().any():
            wbgt_val = sim_df["wbgt"].iloc[0]
            heat_flag = " (HOT)" if float(wbgt_val) > 25 else ""
            print(f"  WBGT:             {wbgt_val}°C{heat_flag}")
        if "wind_speed_kmh" in sim_df.columns and sim_df["wind_speed_kmh"].notna().any():
            wind = sim_df["wind_speed_kmh"].iloc[0]
            gust = sim_df.get("wind_gust_kmh", pd.Series()).iloc[0] if "wind_gust_kmh" in sim_df.columns else None
            gust_str = f", gusts {gust} km/h" if gust and pd.notna(gust) else ""
            print(f"  Wind:             {wind} km/h{gust_str}")
        if "precipitation_mm" in sim_df.columns and sim_df["precipitation_mm"].notna().any():
            precip = sim_df["precipitation_mm"].iloc[0]
            if float(precip) > 0:
                print(f"  Precipitation:    {precip} mm")
            else:
                print(f"  Precipitation:    Dry")
        if "cloud_cover_pct" in sim_df.columns and sim_df["cloud_cover_pct"].notna().any():
            cloud = sim_df["cloud_cover_pct"].iloc[0]
            print(f"  Cloud cover:      {cloud}%")
        weather_src = event_meta.get("weather_source", "unknown")
        print(f"  Source:           {weather_src}")

    # ── 3. Model Parameters ──
    print(f"\n--- Model Parameters ---")
    print(f"  Distance category:    {distance_category or 'unknown'}")

    dist_mult = DISTANCE_SIGMA_MULTIPLIER.get(
        (distance_category or "").lower().strip(), 1.0
    )
    print(f"  Sigma multiplier:     {dist_mult:.2f} "
          f"(swim={DEFAULT_SIGMA_SWIM * dist_mult:.0f}s, "
          f"T1={DEFAULT_SIGMA_T1 * dist_mult:.0f}s, "
          f"bike={DEFAULT_SIGMA_BIKE * dist_mult:.0f}s, "
          f"T2={DEFAULT_SIGMA_T2 * dist_mult:.0f}s, "
          f"run={DEFAULT_SIGMA_RUN * dist_mult:.0f}s)")
    print(f"  Form share:           {DEFAULT_FORM_SHARE:.0%}")
    print(f"  Pack bonus (front):   {pack_params.front_pack_bonus_sec:.1f}s")
    print(f"  Pack penalty (chase): {pack_params.chase_penalty_sec:+.1f}s")
    print(f"  Pack gap threshold:   {pack_params.max_gap_sec:.1f}s")
    print(f"  Min pack size draft:  {pack_params.min_pack_size_for_draft}")
    print(f"  Draft size scale:     {pack_params.draft_size_scale:.2f}")
    if pack_params.n_observations > 0:
        print(f"  Learned from:         {pack_params.n_observations} observations, "
              f"{pack_params.n_races} races")

    # ── 4. Dynamic Merge Parameters ──
    if merge_params is not None:
        print(f"\n--- Dynamic Pack Merging ---")
        print(f"  Model:                P(merge) = sigmoid(beta_0 + beta_gap*gap + beta_chase*size)")
        print(f"  beta_0 (intercept):   {merge_params.beta_0:.3f}")
        print(f"  beta_gap:             {merge_params.beta_gap:.3f}")
        print(f"  beta_chase_size:      {merge_params.beta_chase_size:.3f}")
        print(f"  Max merge gap:        {merge_params.max_merge_gap_sec:.1f}s")
        print(f"  Min chase size:       {merge_params.min_chase_size_for_merge}")
        if merge_params.n_observations > 0:
            print(f"  Learned from:         {merge_params.n_observations} merge situations, "
                  f"{merge_params.n_races} races")
        # Show example probabilities
        import math
        for gap, cs, label in [(2, 4, "2s gap, 4 chasers"), (5, 5, "5s gap, 5 chasers"),
                                (8, 3, "8s gap, 3 chasers"), (12, 4, "12s gap, 4 chasers")]:
            lo = merge_params.beta_0 + merge_params.beta_gap * gap + merge_params.beta_chase_size * cs - breakaway_bias
            p = 1.0 / (1.0 + math.exp(-lo))
            print(f"  P(merge | {label}): {p:.0%}")
        if breakaway_bias != 0.0:
            print(f"  Breakaway bias:       {breakaway_bias:+.1f} (coach override)")
    else:
        print(f"\n--- Pack Merging: disabled (no merge params in model) ---")

    print("=" * 80 + "\n")


def print_simulation_diagnostics(
    sim_df: pd.DataFrame,
    n_sims: int,
    top_n: int = 25,
):
    """
    Print detailed diagnostics explaining how E[Rank] is derived from simulations.

    Shows per-athlete breakdown of:
    - Deterministic vs simulation ranking and what drives the difference
    - Split predictions (moved from main display)
    - Pack dynamics contribution
    - Uncertainty profile
    - Rank distribution across simulations
    """
    from tri_analysis.prediction.utils_time import seconds_to_hms
    from tri_analysis.prediction.simulate import DEFAULT_FORM_SHARE

    print("\n" + "=" * 80)
    print("SIMULATION DIAGNOSTICS")
    print("=" * 80)

    # ── 1. Methodology Explanation ──
    form_pct = f"{DEFAULT_FORM_SHARE:.0%}"
    noise_pct = f"{1 - DEFAULT_FORM_SHARE:.0%}"
    median_sigma = f"{sim_df['sigma_total'].median():.0f}" if "sigma_total" in sim_df.columns else "?"
    print(f"""
--- How E[Rank] is Computed ---
  1. DETERMINISTIC PREDICTION: Model predicts split times (swim, T1, bike, T2, run)
     and a total time. Det. Rank = rank by predicted total (or percentile model).
  2. MONTE CARLO ({n_sims:,} sims): Each sim perturbs all splits with:
     - Shared form factor ({median_sigma}s σ_total for median athlete)
       → Good/bad day correlation across splits ({form_pct} of variance)
     - Per-split noise (independent, remaining {noise_pct})
     - Causal pack formation: sim_swim + sim_t1 → bike entry → pack assignment → bike effect
  3. RANK: In each sim, athletes are ranked by sim_total = swim + T1 + bike + T2 + run.
     E[Rank] = average rank across all {n_sims:,} simulations.
  4. WHY E[Rank] ≠ Det. Rank:
     - Pack dynamics: strong swimmers get front pack MORE often → consistent bike advantage
     - Uncertainty: athletes with low σ (consistent) hold rank; high σ = volatile
     - Non-linearity: rank is ordinal — a 2s gap means nothing if 5 athletes are within it
""")

    # ── 2. Per-Athlete Diagnostics Table ──
    subset = sim_df.head(top_n).copy()

    print(f"--- Per-Athlete Breakdown (Top {top_n}) ---\n")

    # Table header
    header = (
        f"{'Det.Rk':>6} {'E[Rk]':>6} {'Δ':>5} │ "
        f"{'Rk p10':>6} {'Rk p50':>6} {'Rk p90':>6} │ "
        f"{'FPk%':>5} {'AvgPkFx':>8} │ "
        f"{'σ_total':>7} │ "
        f"{'Swim':>8} {'T1':>6} {'Bike':>9} {'T2':>6} {'Run':>8} {'Total':>9} │ "
        f"{'Athlete'}"
    )
    print(header)
    print("─" * len(header))

    for _, row in subset.iterrows():
        det_rank = int(row.get("predicted_rank", 0))
        e_rank = row.get("expected_rank", 0)
        delta = e_rank - det_rank

        rank_p10 = int(row.get("rank_p10", 0))
        rank_p50 = int(row.get("rank_p50", 0))
        rank_p90 = int(row.get("rank_p90", 0))

        fpk_pct = row.get("sim_front_pack_pct", 0) * 100 if "sim_front_pack_pct" in row.index else 0
        avg_pk = row.get("sim_avg_pack_effect", 0) if "sim_avg_pack_effect" in row.index else 0

        sigma_t = row.get("sigma_total", 0)

        # Split predictions
        swim = seconds_to_hms(int(row["pred_swim_sec"])) if pd.notna(row.get("pred_swim_sec")) else "-"
        t1 = seconds_to_hms(int(row["pred_t1_sec"])) if pd.notna(row.get("pred_t1_sec")) else "-"
        bike = seconds_to_hms(int(row["pred_bike_sec"])) if pd.notna(row.get("pred_bike_sec")) else "-"
        t2 = seconds_to_hms(int(row["pred_t2_sec"])) if pd.notna(row.get("pred_t2_sec")) else "-"
        run = seconds_to_hms(int(row["pred_run_sec"])) if pd.notna(row.get("pred_run_sec")) else "-"
        total = seconds_to_hms(int(row["pred_total_sec"])) if pd.notna(row.get("pred_total_sec")) else "-"

        name = row.get("athlete_full_name", "Unknown")[:25]

        delta_str = f"{delta:+.1f}" if abs(delta) >= 0.5 else f"{delta:+.1f}"

        print(
            f"{det_rank:>6} {e_rank:>6.1f} {delta_str:>5} │ "
            f"{rank_p10:>6} {rank_p50:>6} {rank_p90:>6} │ "
            f"{fpk_pct:>5.0f} {avg_pk:>+8.1f} │ "
            f"{sigma_t:>7.0f} │ "
            f"{swim:>8} {t1:>6} {bike:>9} {t2:>6} {run:>8} {total:>9} │ "
            f"{name}"
        )

    # ── 3. Rank Movement Analysis ──
    print(f"\n--- Rank Movement Analysis ---")
    subset["rank_delta"] = subset["expected_rank"] - subset["predicted_rank"]
    risers = subset.nsmallest(5, "rank_delta")
    fallers = subset.nlargest(5, "rank_delta")

    print("\n  Biggest risers (E[Rank] < Det. Rank → simulation favors them):")
    for _, row in risers.iterrows():
        name = row.get("athlete_full_name", "?")[:25]
        det_r = int(row["predicted_rank"])
        e_r = row["expected_rank"]
        fpk = row.get("sim_front_pack_pct", 0) * 100 if "sim_front_pack_pct" in row.index else 0
        sigma = row.get("sigma_total", 0)
        reasons = []
        if fpk > 60:
            reasons.append(f"front pack {fpk:.0f}% of sims")
        if sigma < 60:
            reasons.append(f"low uncertainty (σ={sigma:.0f}s)")
        if "front_pack_rate" in row.index and pd.notna(row.get("front_pack_rate")) and row["front_pack_rate"] > 0.6:
            reasons.append(f"historically strong swimmer (FPR={row['front_pack_rate']:.2f})")
        reason_str = "; ".join(reasons) if reasons else "small sim variance"
        print(f"    {name:25s}: {det_r:>3} → {e_r:.1f} ({row['rank_delta']:+.1f}) — {reason_str}")

    print("\n  Biggest fallers (E[Rank] > Det. Rank → simulation hurts them):")
    for _, row in fallers.iterrows():
        name = row.get("athlete_full_name", "?")[:25]
        det_r = int(row["predicted_rank"])
        e_r = row["expected_rank"]
        fpk = row.get("sim_front_pack_pct", 0) * 100 if "sim_front_pack_pct" in row.index else 0
        sigma = row.get("sigma_total", 0)
        reasons = []
        if fpk < 30:
            reasons.append(f"front pack only {fpk:.0f}% of sims")
        if sigma > 100:
            reasons.append(f"high uncertainty (σ={sigma:.0f}s)")
        if "front_pack_rate" in row.index and pd.notna(row.get("front_pack_rate")) and row["front_pack_rate"] < 0.3:
            reasons.append(f"historically weak swimmer (FPR={row['front_pack_rate']:.2f})")
        reason_str = "; ".join(reasons) if reasons else "large sim variance / crowded field"
        print(f"    {name:25s}: {det_r:>3} → {e_r:.1f} ({row['rank_delta']:+.1f}) — {reason_str}")

    # ── 4. Split-Total Accounting ──
    print(f"\n--- Split-Total Accounting ---")
    if all(c in sim_df.columns for c in ["pred_swim_sec", "pred_t1_sec", "pred_bike_sec", "pred_t2_sec", "pred_run_sec", "pred_total_sec"]):
        split_sum = (
            sim_df["pred_swim_sec"] + sim_df["pred_t1_sec"] +
            sim_df["pred_bike_sec"] + sim_df["pred_t2_sec"] +
            sim_df["pred_run_sec"]
        )
        delta = sim_df["pred_total_sec"] - split_sum
        print(f"  Total model vs sum-of-splits: mean Δ={delta.mean():.1f}s, median={delta.median():.1f}s")
        print(f"  Range: [{delta.min():.1f}s, {delta.max():.1f}s]")
        if abs(delta.mean()) > 60:
            print(f"  ⚠ Large gap — total model includes time not accounted for by split models")
    else:
        print("  (not all split columns available)")

    # ── 5. Simulation vs Deterministic Consistency ──
    print(f"\n--- Simulation vs Deterministic Consistency ---")
    if "total_p50" in sim_df.columns and "pred_total_sec" in sim_df.columns:
        sim_median = sim_df["total_p50"]
        pred_total = sim_df["pred_total_sec"]
        offset = sim_median - pred_total
        print(f"  Sim median vs pred total: mean offset={offset.mean():.1f}s, std={offset.std():.1f}s")
        if abs(offset.mean()) > 10:
            print(f"  ⚠ Simulation median is biased relative to deterministic prediction")
        else:
            print(f"  ✓ Simulation median closely matches deterministic prediction (well-calibrated)")

    # ── 6. Uncertainty Profile ──
    print(f"\n--- Uncertainty Profile ---")
    if "sigma_total" in sim_df.columns:
        sigma = sim_df["sigma_total"]
        print(f"  σ_total: mean={sigma.mean():.0f}s, min={sigma.min():.0f}s, max={sigma.max():.0f}s")
        low_sigma = (sigma < 50).sum()
        high_sigma = (sigma > 120).sum()
        print(f"  Low uncertainty (<50s): {low_sigma} athletes  |  High uncertainty (>120s): {high_sigma} athletes")

    # ── 7. Sim Front Pack vs Historical ──
    if "sim_front_pack_pct" in sim_df.columns and "front_pack_rate" in sim_df.columns:
        print(f"\n--- Sim Front Pack % vs Historical Front Pack Rate ---")
        valid = sim_df[["sim_front_pack_pct", "front_pack_rate"]].dropna()
        if len(valid) > 0:
            corr = valid["sim_front_pack_pct"].corr(valid["front_pack_rate"])
            diff = (valid["sim_front_pack_pct"] - valid["front_pack_rate"]).abs()
            print(f"  Correlation: {corr:.3f}")
            print(f"  Mean |sim - historical|: {diff.mean():.2f}")
            # Flag large divergences
            big_diff = sim_df[
                (sim_df["sim_front_pack_pct"] - sim_df["front_pack_rate"]).abs() > 0.3
            ].head(5)
            if len(big_diff) > 0:
                print(f"  Athletes with largest divergence (>30pp):")
                for _, row in big_diff.iterrows():
                    name = row.get("athlete_full_name", "?")[:25]
                    sim_fp = row["sim_front_pack_pct"] * 100
                    hist_fp = row["front_pack_rate"] * 100
                    print(f"    {name:25s}: sim={sim_fp:.0f}%, hist={hist_fp:.0f}%")
            else:
                print(f"  ✓ No large divergences (all within 30pp)")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict WTCS race outcomes for an upcoming program",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--event_id", type=int, required=True, help="Event ID")
    parser.add_argument("--prog_id", type=int, required=True, help="Program ID")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/bundle.joblib",
        help="Path to saved ModelBundle",
    )
    parser.add_argument(
        "--n_sims",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--no_mc",
        action="store_true",
        help="Skip Monte Carlo simulation (deterministic predictions only)",
    )
    parser.add_argument(
        "--breakaway_bias",
        type=float,
        default=0.0,
        help=(
            "Coach override for breakaway vs merge tendency. "
            "0.0 = use learned probabilities (default). "
            "Positive (e.g. +1.0) = breakaways more likely to stick. "
            "Negative (e.g. -1.0) = packs more likely to come together. "
            "Range: roughly -3 to +3."
        ),
    )
    parser.add_argument(
        "--no_packs",
        action="store_true",
        help="Disable pack effects in MC simulation (pure individual noise). "
             "Useful for isolating whether pack logic helps or hurts accuracy.",
    )

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        logger.info("Train a model first using train.build_training_dataset() and train.train_baseline_models()")
        sys.exit(1)

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize
    key = ProgramKey(event_id=args.event_id, prog_id=args.prog_id)
    logger.info(f"Predicting for {key}")

    # Connect to database
    engine = get_engine()

    # Get event metadata for display
    event_meta = fetch_event_metadata(engine, key)
    if event_meta is None:
        logger.error(f"Event not found: {key}")
        sys.exit(1)

    logger.info(f"Event: {event_meta.get('prog_name', 'Unknown')} - {event_meta.get('event_date')}")
    logger.info(f"Location: {event_meta.get('event_venue', 'Unknown')}, {event_meta.get('event_country', 'Unknown')}")

    # Fetch weather if missing — geocode if needed
    has_weather = event_meta.get("temperature_air") is not None
    has_coords = event_meta.get("event_latitude") is not None and event_meta.get("event_longitude") is not None

    # Try to geocode if we have a venue but no coordinates
    if not has_coords and event_meta.get("event_venue"):
        try:
            from tri_analysis.weather import geocode_venue
            venue = event_meta["event_venue"]
            country = event_meta.get("event_country")
            logger.info(f"Geocoding venue '{venue}' ({country})...")
            coords = geocode_venue(venue, country)
            if coords:
                lat, lon = coords
                event_meta["event_latitude"] = lat
                event_meta["event_longitude"] = lon
                has_coords = True
                logger.info(f"Geocoded to {lat}, {lon}")
                # Persist to DB so we don't re-geocode next time
                from sqlalchemy import text as sa_text
                with engine.begin() as conn:
                    conn.execute(sa_text(
                        "UPDATE events SET event_latitude = :lat, event_longitude = :lon "
                        "WHERE event_id = :eid AND event_latitude IS NULL"
                    ), {"lat": lat, "lon": lon, "eid": key.event_id})
            else:
                logger.warning(f"Could not geocode venue '{venue}'")
        except Exception as e:
            logger.warning(f"Geocoding failed: {e}")

    if not has_weather and has_coords:
        try:
            from tri_analysis.weather import fetch_weather_smart
            event_date = event_meta.get("event_date")
            lat = event_meta["event_latitude"]
            lon = event_meta["event_longitude"]

            logger.info("Fetching weather from Open-Meteo (smart routing)...")
            weather = fetch_weather_smart(lat, lon, event_date)

            if weather:
                source = weather.get("weather_source", "unknown")
                # Merge into event_meta for feature building
                for k, v in weather.items():
                    if v is not None:
                        event_meta[k] = v
                logger.info(
                    f"Weather ({source}): {weather.get('temperature_air')}°C, "
                    f"wind {weather.get('wind_speed_kmh')} km/h, "
                    f"precip {weather.get('precipitation_mm')} mm"
                )
            else:
                logger.warning("No weather data available (no source returned data)")
        except Exception as e:
            logger.warning(f"Could not fetch weather: {e}")

    # Load model bundle
    logger.info(f"Loading model from {args.model_path}")
    bundle = load_model_bundle(args.model_path)

    # Build features for entrants
    logger.info("Building features for start list...")
    features_df = build_features_for_program(engine, key, use_start_list=True, event_meta_override=event_meta)

    if features_df.empty:
        logger.error("No athletes found in start list. Check program_entries table.")
        sys.exit(1)

    logger.info(f"Found {len(features_df)} athletes in start list")

    # Fill missing features
    feature_cols = bundle.feature_columns or get_feature_columns()
    features_df = fill_missing_features(features_df, feature_cols)

    # Generate deterministic predictions
    logger.info("Generating predictions...")
    distance_cat = event_meta.get("prog_distance_category")
    pred_df = predict_splits_and_total(features_df, bundle, distance_category=distance_cat)

    # Run Monte Carlo simulation (unless skipped)
    if not args.no_mc:
        logger.info(f"Running {args.n_sims} Monte Carlo simulations...")

        # Derive event tier for tier-specific pack/merge params
        event_tier = classify_event_tier(event_meta.get("event_name", ""))

        # Load distance-specific pack effect params (falls back to tier → distance → overall)
        pack_params = get_distance_pack_params(bundle.metadata, distance_cat, event_tier=event_tier)
        if pack_params is None:
            # Non-drafting distance or no params at all
            dist_norm = (distance_cat or "").lower().strip()
            if dist_norm and dist_norm not in DRAFT_LEGAL_DISTANCES:
                logger.info(f"Non-drafting distance '{distance_cat}', disabling pack effects")
                pack_params = PackEffectParams(front_pack_bonus_sec=0.0, chase_penalty_sec=0.0)
            else:
                pack_params = PackEffectParams()
                logger.info("No learned pack effects in bundle, using defaults")

        # Load distance-specific merge params (falls back to tier → distance → overall)
        merge_params = get_distance_merge_params(bundle.metadata, distance_cat, event_tier=event_tier)
        if merge_params is not None:
            logger.info(
                f"Merge params: beta_0={merge_params.beta_0:.2f}, "
                f"beta_gap={merge_params.beta_gap:.3f}, "
                f"beta_chase_size={merge_params.beta_chase_size:.3f} "
                f"({merge_params.n_observations} obs)"
            )
        else:
            logger.info("No merge params available, using static pack effects")

        if args.breakaway_bias != 0.0:
            logger.info(f"Breakaway bias: {args.breakaway_bias:+.1f}")

        use_packs = not args.no_packs
        if not use_packs:
            logger.info("Pack effects DISABLED (--no_packs flag)")

        sim_df = run_monte_carlo(
            pred_df,
            n_sims=args.n_sims,
            random_state=42,
            pack_params=pack_params,
            merge_params=merge_params,
            breakaway_bias=args.breakaway_bias,
            distance_category=distance_cat,
            bundle_metadata=bundle.metadata,
            use_pack_effects=use_packs,
        )
        output_df = sim_df
        display_df = format_simulation_output(sim_df)
    else:
        logger.info("Skipping Monte Carlo (--no_mc flag)")
        output_df = pred_df
        display_df = format_prediction_output(pred_df)

    # Print top 65 to console  
    print("\n" + "=" * 80)
    print(f"PREDICTIONS: {event_meta.get('prog_name', 'Unknown')} - {event_meta.get('event_date')}")
    print("=" * 80)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)
    print(display_df.head(65).to_string(index=False))
    print("=" * 80)

    # Print race context section (only with Monte Carlo)
    if not args.no_mc:
        print_race_context(
            sim_df=sim_df,
            pack_params=pack_params,
            distance_category=distance_cat,
            event_meta=event_meta,
            merge_params=merge_params,
            breakaway_bias=args.breakaway_bias,
        )

        # Print detailed simulation diagnostics
        print_simulation_diagnostics(
            sim_df=sim_df,
            n_sims=args.n_sims,
            top_n=25,
        )

        # Print MC simulation diagnostics (per-split bias, pack rates, etc.)
        print_sim_diagnostics(sim_df)

    # Save full results to CSV
    output_file = os.path.join(
        args.output_dir,
        f"predictions_{args.event_id}_{args.prog_id}.csv"
    )
    output_df.to_csv(output_file, index=False)
    logger.info(f"Saved predictions to {output_file}")

    # Print summary statistics
    if not args.no_mc:
        print("\n--- Summary ---")
        top_3 = output_df.head(3)
        print("Top 3 favorites:")
        for _, row in top_3.iterrows():
            name = row.get("athlete_full_name", "Unknown")
            win_pct = row.get("prob_win", 0) * 100
            podium_pct = row.get("prob_podium", 0) * 100
            print(f"  {name}: {win_pct:.1f}% win, {podium_pct:.1f}% podium")

    logger.info("Done!")


if __name__ == "__main__":
    main()
