#!/usr/bin/env python
"""
Train baseline prediction models for WTCS race prediction.

Usage:
    python scripts/train_models.py --start_date 2018-01-01 --end_date 2025-06-30 --output models/bundle.joblib

This script:
1. Builds a training dataset from historical race results
2. Computes features for each athlete-program combination
3. Trains regression models for swim, bike, run, and total time
4. Saves the ModelBundle to disk

Prerequisites:
- Database must have race_results, events, position_metrics, wtcs_pack_membership populated
- Run the ETL pipeline (build_database.py) first
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

from tri_analysis.database import get_engine
from tri_analysis.prediction.train import (
    build_training_dataset,
    train_baseline_models,
    save_model_bundle,
)
from tri_analysis.prediction.features import get_feature_columns, fill_missing_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train WTCS prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2021-01-01",
        help="Start date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2025-06-30",
        help="End date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/bundle_elite_v6.joblib",
        help="Output path for ModelBundle",
    )
    parser.add_argument(
        "--min_finishers",
        type=int,
        default=10,
        help="Minimum finishers per program to include",
    )
    parser.add_argument(
        "--save_dataset",
        type=str,
        default=None,
        help="Optional: Save training dataset to this path (parquet)",
    )
    parser.add_argument(
        "--elite_only",
        action="store_true",
        default=True,
        help="Only train on Elite Men/Women programs (excludes Junior, U23, Para). Default: True",
    )
    parser.add_argument(
        "--all_categories",
        action="store_true",
        help="Include all race categories (Junior, U23, Para, etc.). Overrides --elite_only",
    )
    parser.add_argument(
        "--distances",
        type=str,
        nargs="+",
        default=None,
        help="Distance categories to include (e.g., sprint standard). Default: all",
    )
    parser.add_argument(
        "--no_match_distance",
        action="store_true",
        help="Don't filter athlete history by distance category",
    )

    args = parser.parse_args()
    
    # Handle elite_only vs all_categories
    elite_only = not args.all_categories

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Connect to database
    engine = get_engine()

    # Build training dataset
    logger.info(f"Building training dataset from {args.start_date} to {args.end_date}...")
    logger.info(f"  elite_only={elite_only}, distances={args.distances}, match_distance={not args.no_match_distance}")
    
    train_df = build_training_dataset(
        engine,
        start_date=args.start_date,
        end_date=args.end_date,
        min_finishers=args.min_finishers,
        elite_only=elite_only,
        distance_categories=args.distances,
        match_distance=not args.no_match_distance,
    )

    if train_df.empty:
        logger.error("No training data found. Check your date range and database.")
        sys.exit(1)

    logger.info(f"Training dataset: {len(train_df)} rows")

    # Optionally save dataset
    if args.save_dataset:
        train_df.to_parquet(args.save_dataset, index=False)
        logger.info(f"Saved training dataset to {args.save_dataset}")

    # Get feature columns and fill missing
    feature_cols = get_feature_columns()
    train_df = fill_missing_features(train_df, feature_cols)

    # Train models
    logger.info("Training baseline models...")
    bundle = train_baseline_models(train_df, feature_cols)

    # Save bundle
    save_model_bundle(bundle, args.output)
    logger.info(f"Model bundle saved to {args.output}")

    # Print summary
    print("\n--- Training Summary ---")
    print(f"Training samples: {len(train_df)}")
    print(f"Feature columns: {len(feature_cols)}")
    if bundle.metadata.get("training_metrics"):
        for name, metrics in bundle.metadata["training_metrics"].items():
            print(f"  {name}: MAE={metrics.get('mae', 'N/A'):.1f}s, n={metrics.get('n_samples', 'N/A')}")
    print(f"\nModel saved to: {args.output}")
    print("\nTo run predictions:")
    print(f"  python scripts/predict_program.py --event_id <ID> --prog_id <ID> --model_path {args.output}")


if __name__ == "__main__":
    main()
