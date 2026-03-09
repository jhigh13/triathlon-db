#!/usr/bin/env python
"""
Ablation study: Pack dynamics features in split models.

Trains TWO model bundles from the same training data:
  1. "with_pack"  — Standard split models (includes ema_bike_pack_7, ema_bike_gap_sec_7, etc.)
  2. "without_pack" — Ability-only split models (pack features excluded)

Then runs backtest on H2 2025 for both and compares P@3, P@10, Spearman, MAE.

This answers the Phase 3 question: do pack features in split models cause
double-counting with the simulation's pack effects?

Usage:
    python scripts/ablation_pack_features.py
    python scripts/ablation_pack_features.py --start_date 2018-01-01 --end_date 2025-06-30
    python scripts/ablation_pack_features.py --gender men
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tri_analysis.database import get_engine
from tri_analysis.prediction.train import (
    build_training_dataset,
    train_baseline_models,
    train_distance_split_models,
    save_model_bundle,
    compute_residual_stats,
)
from tri_analysis.prediction.features import (
    get_feature_columns,
    fill_missing_features,
    PACK_DYNAMIC_FEATURES,
)
from tri_analysis.prediction.evaluate import backtest_events
from tri_analysis.prediction.sql import (
    fetch_pack_effect_data,
    fetch_swim_to_bike_transitions,
)
from tri_analysis.prediction.simulate import (
    learn_pack_effects_from_data,
    pack_params_to_dict,
    learn_merge_params_from_data,
    merge_params_to_dict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_bundle(
    engine,
    train_df,
    feature_cols,
    model_params,
    start_date,
    end_date,
    elite_only,
    exclude_pack: bool,
    output_path: str,
):
    """Train a full model bundle with or without pack features."""
    label = "ABILITY-ONLY (no pack features)" if exclude_pack else "STANDARD (with pack features)"
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {label}")
    logger.info(f"{'='*60}")

    # Train baseline models (these use full features, unaffected by pack ablation)
    bundle = train_baseline_models(train_df, feature_cols, model_params=model_params)

    # Train distance-specific split models — THIS is where the ablation happens
    dist_split_models, split_feature_cols = train_distance_split_models(
        train_df, feature_cols, model_params=model_params,
        exclude_pack_features=exclude_pack,
    )
    bundle.distance_split_models = dist_split_models
    bundle.split_feature_columns = split_feature_cols
    bundle.metadata["exclude_pack_features"] = exclude_pack

    # Compute residual stats
    residual_stats = compute_residual_stats(
        train_df, bundle, feature_cols, split_feature_cols
    )
    if residual_stats:
        bundle.metadata["residual_stats"] = residual_stats

    # Learn pack effects (same for both — this is simulation params, not model features)
    pack_data = fetch_pack_effect_data(engine, start_date, end_date, elite_only=elite_only)
    if not pack_data.empty:
        learned_pack_params = learn_pack_effects_from_data(pack_data)
        bundle.metadata["pack_effect_params"] = pack_params_to_dict(learned_pack_params)

        # Distance-specific
        pack_params_by_distance = {}
        for dist_cat in ["sprint", "standard"]:
            dist_pack_data = fetch_pack_effect_data(
                engine, start_date, end_date, elite_only=elite_only,
                distance_category=dist_cat,
            )
            if not dist_pack_data.empty and len(dist_pack_data) >= 100:
                dist_params = learn_pack_effects_from_data(dist_pack_data)
                dist_params.distance_category = dist_cat
                pack_params_by_distance[dist_cat] = pack_params_to_dict(dist_params)
        if pack_params_by_distance:
            bundle.metadata["pack_effect_params_by_distance"] = pack_params_by_distance

    # Learn merge params
    transition_data = fetch_swim_to_bike_transitions(
        engine, start_date, end_date, elite_only=elite_only,
    )
    if not transition_data.empty:
        learned_merge = learn_merge_params_from_data(transition_data)
        bundle.metadata["merge_params"] = merge_params_to_dict(learned_merge)

        merge_by_dist = {}
        for dist_cat in ["sprint", "standard"]:
            dist_trans = fetch_swim_to_bike_transitions(
                engine, start_date, end_date, elite_only=elite_only,
                distance_category=dist_cat,
            )
            if not dist_trans.empty and len(dist_trans) >= 50:
                dist_merge = learn_merge_params_from_data(dist_trans)
                merge_by_dist[dist_cat] = merge_params_to_dict(dist_merge)
        if merge_by_dist:
            bundle.metadata["merge_params_by_distance"] = merge_by_dist

    save_model_bundle(bundle, output_path)
    logger.info(f"Saved to {output_path}")

    # Log feature counts
    n_split = len(split_feature_cols)
    pack_used = [f for f in PACK_DYNAMIC_FEATURES if f in split_feature_cols]
    logger.info(f"Split features: {n_split} (pack features: {len(pack_used)})")
    if pack_used:
        logger.info(f"  Pack features in model: {pack_used}")
    else:
        logger.info("  No pack features in model (ability-only)")

    return bundle


def run_comparison_backtest(engine, model_with_path, model_without_path, gender=None):
    """Run backtest for both models and compare."""
    import pandas as pd
    from sqlalchemy import text

    # Get H2 2025 test events (same schema as run_backtest.py — prog info is on events table)
    query = text("""
        SELECT DISTINCT e.event_id, e.prog_id, e.prog_name,
               e.event_name, e.event_date,
               e.prog_distance_category
        FROM events e
        JOIN race_results rr ON e.event_id = rr.event_id AND e.prog_id = rr.prog_id
        WHERE e.event_date >= '2025-07-01'
          AND e.event_date <= '2025-12-31'
          AND e.prog_name ~* '(Elite Men|Elite Women)'
          AND COALESCE(e.prog_distance_category, '') != 'long_distance'
        GROUP BY e.event_id, e.prog_id, e.event_date, e.event_name, e.prog_name, e.prog_distance_category
        HAVING COUNT(*) >= 10
        ORDER BY e.event_date
    """)
    events_df = pd.read_sql(query, engine)

    if events_df.empty:
        logger.error("No test events found")
        return

    # Filter by gender if specified
    if gender == "men":
        events_df = events_df[events_df["prog_name"].str.contains("Men")]
    elif gender == "women":
        events_df = events_df[events_df["prog_name"].str.contains("Women")]

    programs = list(zip(events_df["event_id"], events_df["prog_id"]))
    logger.info(f"Backtesting on {len(programs)} programs")

    results = {}
    for label, model_path in [("with_pack", model_with_path), ("without_pack", model_without_path)]:
        logger.info(f"\n--- Backtesting: {label} ({model_path}) ---")
        bt_df = backtest_events(
            engine, programs, model_path,
        )
        if bt_df.empty:
            logger.warning(f"No backtest results for {label}")
            continue

        results[label] = {
            "n_events": len(bt_df),
            "precision_at_3": bt_df["precision_at_3"].mean(),
            "precision_at_5": bt_df["precision_at_5"].mean(),
            "precision_at_10": bt_df["precision_at_10"].mean(),
            "spearman_corr": bt_df["spearman_corr"].mean(),
            "mae_total_sec": bt_df["mae_total_sec"].mean(),
            "bt_df": bt_df,
        }

    if len(results) < 2:
        logger.error("Could not produce results for both models")
        return

    # Print comparison
    print("\n" + "=" * 80)
    print("ABLATION RESULTS: Pack Features in Split Models")
    print("=" * 80)
    print(f"\nPack dynamic features: {PACK_DYNAMIC_FEATURES}")
    print(f"\nTest set: H2 2025 ({results['with_pack']['n_events']} events)")

    header = f"{'Metric':<20} {'With Pack':>12} {'Without Pack':>14} {'Delta':>10} {'Winner':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    metrics = [
        ("P@3", "precision_at_3", True),
        ("P@5", "precision_at_5", True),
        ("P@10", "precision_at_10", True),
        ("Spearman", "spearman_corr", True),
        ("MAE (s)", "mae_total_sec", False),
    ]

    for display_name, key, higher_is_better in metrics:
        with_val = results["with_pack"][key]
        without_val = results["without_pack"][key]
        delta = without_val - with_val
        if higher_is_better:
            winner = "without" if delta > 0 else "with" if delta < 0 else "tie"
        else:
            winner = "without" if delta < 0 else "with" if delta > 0 else "tie"

        if key == "mae_total_sec":
            fmt = f"{display_name:<20} {with_val:>12.1f} {without_val:>14.1f} {delta:>+10.1f} {winner:>10}"
        elif key == "spearman_corr":
            fmt = f"{display_name:<20} {with_val:>12.3f} {without_val:>14.3f} {delta:>+10.3f} {winner:>10}"
        else:
            fmt = f"{display_name:<20} {with_val*100:>11.1f}% {without_val*100:>13.1f}% {delta*100:>+9.1f}% {winner:>10}"

        print(fmt)

    # Per-tier comparison if tier info available
    for label in ["with_pack", "without_pack"]:
        bt_df = results[label]["bt_df"]
        if "event_tier" in bt_df.columns:
            print(f"\n--- {label.upper()} by Tier ---")
            for tier, group in bt_df.groupby("event_tier"):
                print(
                    f"  Tier {tier}: P@3={group['precision_at_3'].mean()*100:.1f}%, "
                    f"P@10={group['precision_at_10'].mean()*100:.1f}%, "
                    f"Spearman={group['spearman_corr'].mean():.3f}, "
                    f"MAE={group['mae_total_sec'].mean():.0f}s, n={len(group)}"
                )

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  If 'without_pack' wins → pack features caused double-counting")
    print("    → Use ability-only models + full sim pack effects (Option A)")
    print("  If 'with_pack' wins → pack features provide useful signal")
    print("    → Keep pack features but scale down sim pack effects (Option B)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: pack dynamics features in split models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start_date", default="2021-01-01")
    parser.add_argument("--end_date", default="2025-06-30")
    parser.add_argument("--gender", choices=["men", "women"], default=None)
    parser.add_argument("--output_dir", default="models")
    parser.add_argument(
        "--backtest_only",
        action="store_true",
        help="Skip training, just run backtest comparison using existing model files",
    )
    args = parser.parse_args()

    engine = get_engine()
    os.makedirs(args.output_dir, exist_ok=True)

    gender_suffix = f"_{args.gender}" if args.gender else ""
    with_path = os.path.join(args.output_dir, f"bundle_ablation_withpack{gender_suffix}.joblib")
    without_path = os.path.join(args.output_dir, f"bundle_ablation_nopack{gender_suffix}.joblib")

    if not args.backtest_only:
        # Build training data once (shared between both)
        logger.info("Building training dataset...")
        train_df = build_training_dataset(
            engine, args.start_date, args.end_date,
            elite_only=True, gender=args.gender,
        )
        if train_df.empty:
            logger.error("No training data")
            return

        feature_cols = get_feature_columns()
        train_df = fill_missing_features(train_df, feature_cols)
        logger.info(f"Training dataset: {len(train_df)} rows, {len(feature_cols)} features")

        # Train both variants
        train_bundle(
            engine, train_df, feature_cols, model_params=None,
            start_date=args.start_date, end_date=args.end_date,
            elite_only=True, exclude_pack=False, output_path=with_path,
        )
        train_bundle(
            engine, train_df, feature_cols, model_params=None,
            start_date=args.start_date, end_date=args.end_date,
            elite_only=True, exclude_pack=True, output_path=without_path,
        )

    # Run comparison backtest
    if not os.path.exists(with_path) or not os.path.exists(without_path):
        logger.error(f"Model files not found: {with_path}, {without_path}")
        return

    run_comparison_backtest(engine, with_path, without_path, gender=args.gender)


if __name__ == "__main__":
    main()
