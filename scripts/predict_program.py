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
from tri_analysis.prediction.simulate import run_monte_carlo, format_simulation_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    pred_df = predict_splits_and_total(features_df, bundle)

    # Run Monte Carlo simulation (unless skipped)
    if not args.no_mc:
        logger.info(f"Running {args.n_sims} Monte Carlo simulations...")
        sim_df = run_monte_carlo(pred_df, n_sims=args.n_sims, random_state=42)
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
    print(display_df.head(20).to_string(index=False))
    print("=" * 80 + "\n")

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
