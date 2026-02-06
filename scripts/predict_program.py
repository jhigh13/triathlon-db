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

import pandas as pd

from tri_analysis.database import get_engine
from tri_analysis.prediction.sql import ProgramKey, fetch_event_metadata
from tri_analysis.prediction.features import (
    build_features_for_program,
    fill_missing_features,
    get_feature_columns,
)
from tri_analysis.prediction.train import load_model_bundle
from tri_analysis.prediction.predict import predict_splits_and_total, format_prediction_output
from tri_analysis.prediction.simulate import (
    run_monte_carlo,
    format_simulation_output,
    pack_params_from_dict,
    merge_params_from_dict,
    PackEffectParams,
    MergeParams,
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
        DEFAULT_SIGMA_SWIM,
        DEFAULT_SIGMA_BIKE,
        DEFAULT_SIGMA_RUN,
        DISTANCE_SIGMA_MULTIPLIER,
        DEFAULT_FORM_SHARE,
    )
    from tri_analysis.prediction.utils_time import seconds_to_hms

    print("\n" + "=" * 80)
    print("RACE CONTEXT")
    print("=" * 80)

    # ── 1. Predicted Pack Formations ──
    if "pred_swim_sec" in sim_df.columns and sim_df["pred_swim_sec"].notna().all():
        swim_times = sim_df["pred_swim_sec"].values
        sorted_idx = swim_times.argsort()
        sorted_swim = swim_times[sorted_idx]
        sorted_names = sim_df["athlete_full_name"].values[sorted_idx]

        pack_ids = assign_packs_chain(sorted_swim, max_gap_sec=pack_params.max_gap_sec)

        print("\n--- Predicted Pack Formations (from deterministic swim predictions) ---")
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

        # ── Pack effect on bike ──
        bike_effects = continuous_gap_bike_effect(swim_times, pack_params)
        print(f"\n--- Pack Effect on Bike ---")
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

    # ── 3. Model Parameters ──
    print(f"\n--- Model Parameters ---")
    print(f"  Distance category:    {distance_category or 'unknown'}")

    dist_mult = DISTANCE_SIGMA_MULTIPLIER.get(
        (distance_category or "").lower().strip(), 1.0
    )
    print(f"  Sigma multiplier:     {dist_mult:.2f} "
          f"(swim={DEFAULT_SIGMA_SWIM * dist_mult:.0f}s, "
          f"bike={DEFAULT_SIGMA_BIKE * dist_mult:.0f}s, "
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

    # Load model bundle
    logger.info(f"Loading model from {args.model_path}")
    bundle = load_model_bundle(args.model_path)

    # Build features for entrants
    logger.info("Building features for start list...")
    features_df = build_features_for_program(engine, key, use_start_list=True)

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

        # Load learned pack effect params from bundle if available
        pack_params = None
        pack_params_dict = bundle.metadata.get("pack_effect_params")
        if pack_params_dict:
            pack_params = pack_params_from_dict(pack_params_dict)
            logger.info(
                f"Using learned pack effects: bonus={pack_params.front_pack_bonus_sec:.1f}s, "
                f"penalty={pack_params.chase_penalty_sec:.1f}s"
            )
        else:
            pack_params = PackEffectParams()
            logger.info("No learned pack effects in bundle, using defaults")

        # Load learned merge params from bundle if available
        merge_params = None
        merge_params_dict = bundle.metadata.get("merge_params")
        if merge_params_dict:
            merge_params = merge_params_from_dict(merge_params_dict)
            logger.info(
                f"Using learned merge params: beta_0={merge_params.beta_0:.2f}, "
                f"beta_gap={merge_params.beta_gap:.3f}, "
                f"beta_chase_size={merge_params.beta_chase_size:.3f} "
                f"({merge_params.n_observations} obs)"
            )
        else:
            logger.info("No merge params in bundle, using static pack effects")

        if args.breakaway_bias != 0.0:
            logger.info(f"Breakaway bias: {args.breakaway_bias:+.1f}")

        sim_df = run_monte_carlo(
            pred_df,
            n_sims=args.n_sims,
            random_state=42,
            pack_params=pack_params,
            merge_params=merge_params,
            breakaway_bias=args.breakaway_bias,
            distance_category=distance_cat,
        )
        output_df = sim_df
        display_df = format_simulation_output(sim_df)
    else:
        logger.info("Skipping Monte Carlo (--no_mc flag)")
        output_df = pred_df
        display_df = format_prediction_output(pred_df)

    # Print top 20 to console
    print("\n" + "=" * 80)
    print(f"PREDICTIONS: {event_meta.get('prog_name', 'Unknown')} - {event_meta.get('event_date')}")
    print("=" * 80)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)
    print(display_df.head(50).to_string(index=False))
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
