#!/usr/bin/env python
"""
Analyze prediction feature contributions for athletes.

Loads the latest prediction CSV and model bundle to show:
1. Feature importance from the model
2. Feature values and contributions for selected athletes
3. Comparison between athletes

Usage:
    # Show top 5 athletes by predicted rank
    python scripts/analyze_predictions.py

    # Show specific athletes
    python scripts/analyze_predictions.py --athletes "Reese Vannerson" "Sullivan Middaugh"

    # Compare two athletes
    python scripts/analyze_predictions.py --athletes "Reese Vannerson" --compare "Sullivan Middaugh"

    # Use specific CSV and model
    python scripts/analyze_predictions.py --csv outputs/predictions_195251_676986.csv --model models/bundle_claudev16.joblib
"""

from __future__ import annotations
import argparse
import glob
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from tri_analysis.prediction.train import load_model_bundle
from tri_analysis.prediction.predict import predict_splits_and_total
from tri_analysis.prediction.utils_time import seconds_to_hms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_latest_predictions_csv(output_dir: str = "outputs") -> str | None:
    """Find the most recently modified predictions CSV in the output directory."""
    pattern = os.path.join(output_dir, "predictions_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def get_model_feature_importance(bundle, top_n: int = 20) -> pd.DataFrame | None:
    """Extract feature importance from model if available."""
    model = bundle.model_total
    if model is None:
        return None

    feature_cols = bundle.feature_columns

    # Try different methods to get feature importance
    if hasattr(model, 'feature_importances_'):
        # Tree-based models (RandomForest, GradientBoosting, XGBoost, etc.)
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importance = np.abs(model.coef_).flatten()
    else:
        return None

    df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return df.head(top_n)


def compute_feature_contributions(
    athlete_features: np.ndarray,
    baseline_features: np.ndarray,
    model,
    feature_cols: list[str],
    target_name: str = "prediction"
) -> list[tuple]:
    """
    Compute how much each feature contributes to the difference between two athletes.

    For each feature, swap the athlete's value with the baseline's value and measure
    the change in prediction.

    Returns list of (feature, impact_sec, athlete_value, baseline_value) sorted by |impact|.
    """
    athlete_pred = model.predict(athlete_features.reshape(1, -1))[0]

    contributions = []
    for i, feat in enumerate(feature_cols):
        # Create modified feature vector with baseline's value for this feature
        modified = athlete_features.copy()
        modified[i] = baseline_features[i]
        new_pred = model.predict(modified.reshape(1, -1))[0]

        # Impact: how much the prediction changes when we use baseline's value
        # Positive impact = athlete would be slower (higher seconds) with baseline's value
        impact = new_pred - athlete_pred

        contributions.append((
            feat,
            impact,
            athlete_features[i],
            baseline_features[i]
        ))

    # Sort by absolute impact
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions


def format_value(val, feat_name: str) -> str:
    """Format feature value for display based on feature type."""
    if pd.isna(val):
        return "N/A"

    # Time-based features (seconds)
    if any(x in feat_name for x in ['_sec_', '_gap_sec']):
        if val >= 60:
            return seconds_to_hms(int(val))
        return f"{val:.1f}s"

    # Percentile/rate features (0-1 scale)
    if any(x in feat_name for x in ['_pct_', '_rate', 'finish_pct', 'perf_ratio']):
        return f"{val:.2f}"

    # Integer features
    if feat_name in ['seed_total_rank', 'n_entrants', 'event_tier', 'races_12m',
                     'elo_races', 'wt_rank_position', 'distance_category_encoded']:
        return f"{int(val)}"

    # Days
    if feat_name == 'days_since_last_race':
        return f"{int(val)}d"

    # Default
    return f"{val:.1f}"


def print_athlete_analysis(
    df: pd.DataFrame,
    athlete_name: str,
    bundle,
    feature_cols: list[str],
    compare_athlete: str | None = None,
    field_median: np.ndarray | None = None,
):
    """Print detailed feature analysis for a single athlete."""
    # Find athlete row
    mask = df['athlete_full_name'].str.contains(athlete_name, case=False, na=False)
    if not mask.any():
        print(f"  Athlete '{athlete_name}' not found in predictions")
        return

    athlete_row = df[mask].iloc[0]
    athlete_features = athlete_row[feature_cols].values.astype(float)

    print(f"\n{'='*80}")
    print(f"ATHLETE: {athlete_row['athlete_full_name']} ({athlete_row['athlete_country_name']})")
    print(f"{'='*80}")

    # Key predictions summary (recalculated from --model)
    print(f"\n--- Predictions (from specified model) ---")
    print(f"  Predicted Rank:  {athlete_row.get('predicted_rank', 'N/A')}")
    print(f"  Pred Total:      {seconds_to_hms(int(athlete_row['pred_total_sec']))} ({athlete_row['pred_total_sec']:.0f}s)")
    if 'pred_swim_sec' in athlete_row and pd.notna(athlete_row['pred_swim_sec']):
        print(f"  Pred Swim:       {seconds_to_hms(int(athlete_row['pred_swim_sec']))}")
    if 'pred_bike_sec' in athlete_row and pd.notna(athlete_row['pred_bike_sec']):
        print(f"  Pred Bike:       {seconds_to_hms(int(athlete_row['pred_bike_sec']))}")
    if 'pred_run_sec' in athlete_row and pd.notna(athlete_row['pred_run_sec']):
        print(f"  Pred Run:        {seconds_to_hms(int(athlete_row['pred_run_sec']))}")

    # Historical context (from data, model-independent)
    print(f"\n--- Historical Context (from data) ---")
    ema_total = athlete_row.get('ema_total_sec_5')
    if pd.notna(ema_total):
        print(f"  EMA Total (5):   {seconds_to_hms(int(ema_total))} ({ema_total:.0f}s)")
        gap = athlete_row['pred_total_sec'] - ema_total
        direction = "slower" if gap > 0 else "faster"
        print(f"  Pred vs EMA:     {abs(gap):.0f}s {direction}")

    ema_swim = athlete_row.get('ema_swim_sec_5')
    if pd.notna(ema_swim):
        print(f"  EMA Swim (5):    {seconds_to_hms(int(ema_swim))}")

    best_total = athlete_row.get('best_total_sec_24m')
    if pd.notna(best_total):
        print(f"  Best Total 24m:  {seconds_to_hms(int(best_total))}")

    # Get model for contribution analysis
    model = bundle.model_total
    if model is None:
        print("\n  (No model available for feature contribution analysis)")
        return

    # Determine baseline for comparison
    if compare_athlete:
        compare_mask = df['athlete_full_name'].str.contains(compare_athlete, case=False, na=False)
        if compare_mask.any():
            baseline_row = df[compare_mask].iloc[0]
            baseline_features = baseline_row[feature_cols].values.astype(float)
            baseline_name = baseline_row['athlete_full_name']
        else:
            print(f"\n  Comparison athlete '{compare_athlete}' not found, using field median")
            baseline_features = field_median
            baseline_name = "Field Median"
    elif field_median is not None:
        baseline_features = field_median
        baseline_name = "Field Median"
    else:
        print("\n  (No baseline available for feature contribution analysis)")
        return

    # Compute feature contributions
    contributions = compute_feature_contributions(
        athlete_features, baseline_features, model, feature_cols
    )

    print(f"\n--- Feature Contributions (vs {baseline_name}) ---")
    print(f"  {'Feature':<28} {'Impact':>10} {'Athlete':>12} {'Baseline':>12}")
    print(f"  {'-'*62}")

    # Show top 15 most impactful features
    for feat, impact, a_val, b_val in contributions[:15]:
        if abs(impact) < 0.5:  # Skip negligible impacts
            continue
        sign = '+' if impact >= 0 else ''
        a_str = format_value(a_val, feat)
        b_str = format_value(b_val, feat)
        print(f"  {feat:<28} {sign}{impact:>8.0f}s {a_str:>12} {b_str:>12}")

    # Compute total explained difference
    athlete_pred = model.predict(athlete_features.reshape(1, -1))[0]
    baseline_pred = model.predict(baseline_features.reshape(1, -1))[0]
    print(f"\n  Model prediction: {athlete_pred:.0f}s vs {baseline_name}: {baseline_pred:.0f}s")
    print(f"  Difference: {athlete_pred - baseline_pred:+.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prediction feature contributions for athletes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to predictions CSV (default: most recent in outputs/)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/bundle_claudev23_men.joblib",
        help="Path to model bundle"
    )
    parser.add_argument(
        "--athletes",
        type=str,
        nargs="*",
        default=None,
        help="Athlete names to analyze (default: top N by predicted rank)"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Athlete name to compare against (default: field median)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top athletes to show when --athletes not specified (default: 5)"
    )
    parser.add_argument(
        "--show-importance",
        action="store_true",
        help="Show model feature importance ranking"
    )

    args = parser.parse_args()

    # Find predictions CSV
    csv_path = args.csv
    if csv_path is None:
        csv_path = find_latest_predictions_csv()
        if csv_path is None:
            logger.error("No predictions CSV found in outputs/")
            sys.exit(1)

    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(1)

    logger.info(f"Loading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Load model bundle
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    logger.info(f"Loading model from: {args.model}")
    bundle = load_model_bundle(args.model)
    feature_cols = bundle.feature_columns

    # Check that feature columns exist in CSV
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing {len(missing_cols)} feature columns in CSV: {missing_cols[:5]}...")

    # Re-predict using the specified model so predictions match the --model arg
    distance_category = df["prog_distance_category"].iloc[0] if "prog_distance_category" in df.columns else None
    df = predict_splits_and_total(df, bundle, distance_category=distance_category)
    logger.info(f"Re-predicted using model: {args.model}")

    # Compute field median for baseline comparison
    feature_df = df[feature_cols].copy()
    field_median = feature_df.median().values.astype(float)

    print("\n" + "="*80)
    print("PREDICTION ANALYSIS")
    print("="*80)
    print(f"CSV:       {csv_path}")
    print(f"Model:     {args.model}")
    print(f"Distance:  {distance_category or 'unknown'}")
    print(f"Athletes:  {len(df)}")

    # Show model feature importance if requested
    if args.show_importance:
        print("\n" + "="*80)
        print("MODEL FEATURE IMPORTANCE (Top 20)")
        print("="*80)
        importance_df = get_model_feature_importance(bundle, top_n=20)
        if importance_df is not None:
            for _, row in importance_df.iterrows():
                bar = "█" * int(row['importance'] * 50)
                print(f"  {row['feature']:<30} {row['importance']:.3f} {bar}")
        else:
            print("  (Feature importance not available for this model type)")

    # Determine which athletes to analyze
    if args.athletes:
        athlete_names = args.athletes
    else:
        # Use top N by predicted rank
        if 'predicted_rank' in df.columns:
            top_df = df.nsmallest(args.top, 'predicted_rank')
        else:
            top_df = df.head(args.top)
        athlete_names = top_df['athlete_full_name'].tolist()
        logger.info(f"Analyzing top {args.top} athletes by predicted rank")

    # Analyze each athlete
    for athlete_name in athlete_names:
        print_athlete_analysis(
            df=df,
            athlete_name=athlete_name,
            bundle=bundle,
            feature_cols=feature_cols,
            compare_athlete=args.compare,
            field_median=field_median,
        )

    # If comparing two specific athletes, show head-to-head
    if args.compare and args.athletes and len(args.athletes) == 1:
        athlete1 = args.athletes[0]
        athlete2 = args.compare

        mask1 = df['athlete_full_name'].str.contains(athlete1, case=False, na=False)
        mask2 = df['athlete_full_name'].str.contains(athlete2, case=False, na=False)

        if mask1.any() and mask2.any():
            row1 = df[mask1].iloc[0]
            row2 = df[mask2].iloc[0]
            name1 = row1['athlete_full_name']
            name2 = row2['athlete_full_name']

            print("\n" + "="*80)
            print(f"HEAD-TO-HEAD: {name1} vs {name2}")
            print("="*80)

            # Model predictions (recalculated from --model)
            print(f"\n--- Model Predictions (from {os.path.basename(args.model)}) ---")
            print(f"  {'':28} {name1[:15]:>15} {name2[:15]:>15} {'Diff':>10}")
            print(f"  {'-'*70}")
            for split in ['total', 'swim', 'bike', 'run']:
                col = f'pred_{split}_sec'
                if col in row1.index and pd.notna(row1[col]) and pd.notna(row2[col]):
                    v1, v2 = row1[col], row2[col]
                    diff = v1 - v2
                    print(f"  {'Pred ' + split.capitalize():<28} {seconds_to_hms(int(v1)):>15} {seconds_to_hms(int(v2)):>15} {diff:>+10.0f}s")
            if 'predicted_rank' in row1.index:
                print(f"  {'Predicted Rank':<28} {int(row1['predicted_rank']):>15} {int(row2['predicted_rank']):>15}")

            # Feature values (from historical data, same regardless of model)
            features1 = row1[feature_cols].values.astype(float)
            features2 = row2[feature_cols].values.astype(float)

            print(f"\n--- Feature Values (from historical data, model-independent) ---")
            print(f"  {'Feature':<28} {name1[:15]:>15} {name2[:15]:>15} {'Diff':>10}")
            print(f"  {'-'*70}")

            # Show all features with meaningful differences
            diffs = []
            for i, feat in enumerate(feature_cols):
                v1, v2 = features1[i], features2[i]
                if pd.notna(v1) and pd.notna(v2):
                    diff = v1 - v2
                    if abs(diff) > 0.01:  # Skip tiny differences
                        diffs.append((feat, v1, v2, diff))

            # Sort by absolute difference (normalized by typical scale)
            diffs.sort(key=lambda x: abs(x[3]), reverse=True)

            for feat, v1, v2, diff in diffs[:20]:
                v1_str = format_value(v1, feat)
                v2_str = format_value(v2, feat)
                diff_str = f"{diff:+.1f}" if abs(diff) < 100 else f"{diff:+.0f}"
                print(f"  {feat:<28} {v1_str:>15} {v2_str:>15} {diff_str:>10}")

    print("\n" + "="*80)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
