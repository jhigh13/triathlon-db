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

import mlflow

from tri_analysis.database import get_engine
from tri_analysis.prediction.train import (
    build_training_dataset,
    train_baseline_models,
    train_distance_split_models,
    save_model_bundle,
    cross_validate,
    cross_validate_splits,
    tune_hyperparameters,
    compute_residual_stats,
    learn_swim_gap_distributions,
    optimize_ensemble_weight,
)
from tri_analysis.prediction.features import get_feature_columns, fill_missing_features
from tri_analysis.prediction.sql import fetch_pack_effect_data, fetch_swim_to_bike_transitions
from tri_analysis.prediction.simulate import (
    learn_pack_effects_from_data,
    pack_params_to_dict,
    learn_merge_params_from_data,
    merge_params_to_dict,
)

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
        help="End date for training data (YYYY-MM-DD). Default 2025-06-30 to avoid H2 2025 backtest leakage.",
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
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run time-based cross-validation on the training data (no model saved)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning via grid search before training",
    )
    parser.add_argument(
        "--gender",
        type=str,
        choices=["men", "women"],
        default=None,
        help="Train on a single gender only (men or women)",
    )
    parser.add_argument(
        "--train_both_genders",
        action="store_true",
        help="Train separate models for men and women. Output files get _men/_women suffix.",
    )
    parser.add_argument(
        "--time_decay_half_life",
        type=float,
        default=365.0,
        help="Half-life in days for time-decay sample weighting (0 = disable). Default: 365",
    )
    parser.add_argument(
        "--no_time_decay",
        action="store_true",
        help="Disable time-decay sample weighting",
    )
    parser.add_argument(
        "--cv_splits",
        action="store_true",
        help="Run per-split cross-validation (track swim/t1/bike/t2/run MAE per fold)",
    )
    parser.add_argument(
        "--include_pack_features",
        action="store_true",
        help="Include pack dynamics features (ema_bike_pack_7, ema_bike_gap_sec_7, etc.) "
             "in split models. Default is ability-only (excluded) since ablation showed "
             "pack features add no ranking signal and the simulation handles pack effects.",
    )
    parser.add_argument(
        "--optimize_weight",
        action="store_true",
        help="Optimize ensemble ranker/percentile blend weight via time-based CV. "
             "Stores optimal weight in bundle metadata (default: 0.5).",
    )

    args = parser.parse_args()

    # Handle elite_only vs all_categories
    elite_only = not args.all_categories

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Configure MLflow
    mlflow.set_tracking_uri(f"file:///{os.path.join(project_root, 'mlruns').replace(os.sep, '/')}")
    mlflow.set_experiment("triathlon-prediction")

    # Connect to database
    engine = get_engine()

    # Determine gender(s) to train
    if args.train_both_genders:
        genders = ["men", "women"]
    elif args.gender:
        genders = [args.gender]
    else:
        genders = [None]  # None = both genders combined (original behavior)

    for gender in genders:
        gender_label = gender or "all"
        logger.info(f"\n{'='*60}")
        logger.info(f"Training pipeline for gender={gender_label}")
        logger.info(f"{'='*60}")

        # Build training dataset
        logger.info(f"Building training dataset from {args.start_date} to {args.end_date}...")
        logger.info(f"  elite_only={elite_only}, distances={args.distances}, "
                     f"match_distance={not args.no_match_distance}, gender={gender_label}")

        train_df = build_training_dataset(
            engine,
            start_date=args.start_date,
            end_date=args.end_date,
            min_finishers=args.min_finishers,
            elite_only=elite_only,
            distance_categories=args.distances,
            match_distance=not args.no_match_distance,
            gender=gender,
        )

        if train_df.empty:
            logger.error(f"No training data found for gender={gender_label}. Check your date range and database.")
            continue

        logger.info(f"Training dataset: {len(train_df)} rows")

        # Optionally save dataset
        if args.save_dataset:
            ds_path = args.save_dataset
            if gender:
                base, ext = os.path.splitext(ds_path)
                ds_path = f"{base}_{gender}{ext}"
            train_df.to_parquet(ds_path, index=False)
            logger.info(f"Saved training dataset to {ds_path}")

        # Get feature columns and fill missing
        feature_cols = get_feature_columns()
        train_df = fill_missing_features(train_df, feature_cols)

        # Cross-validation only mode
        if args.cv:
            logger.info("Running time-based cross-validation...")
            cv_result = cross_validate(train_df, feature_cols)
            print(f"\n--- Cross-Validation Results (gender={gender_label}) ---")
            print(f"Mean Spearman: {cv_result['mean_spearman']:.4f}")
            print(f"Mean MAE (pct): {cv_result['mean_mae']:.4f}")
            for fold in cv_result["fold_results"]:
                print(f"  Fold {fold['fold']}: Spearman={fold['spearman']:.3f}, "
                      f"MAE={fold['mae']:.4f}, n_races={fold['n_races']}, "
                      f"train={fold['n_train']}, val={fold['n_val']}")
            if not args.tune and not args.cv_splits:
                continue  # CV-only mode: skip training

        # Per-split cross-validation
        if args.cv_splits:
            logger.info("Running per-split cross-validation...")
            split_cv = cross_validate_splits(
                train_df, feature_cols,
                exclude_pack_features=not args.include_pack_features,
            )
            print(f"\n--- Per-Split CV Results (gender={gender_label}) ---")
            for key, val in sorted(split_cv.items()):
                if key == "fold_results":
                    continue
                print(f"  {key}: {val:.1f}")
            if split_cv.get("fold_results"):
                for fr in split_cv["fold_results"]:
                    fold_num = fr.get("fold", "?")
                    metrics_str = ", ".join(
                        f"{k}={v:.1f}" for k, v in sorted(fr.items())
                        if k not in ("fold", "n_train", "n_val") and isinstance(v, (int, float))
                    )
                    print(f"  Fold {fold_num}: {metrics_str}")
            if not args.tune:
                continue  # CV-only mode: skip training

        # Hyperparameter tuning
        model_params = None
        if args.tune:
            logger.info("Running hyperparameter tuning...")
            tune_result = tune_hyperparameters(train_df, feature_cols)
            model_params = tune_result["best_params"]

            print(f"\n--- Tuning Results (gender={gender_label}) ---")
            print(f"Best Spearman: {tune_result['best_score']:.4f}")
            print(f"Best params: {model_params}")
            print("\nTop 5 configurations:")
            for rank, res in enumerate(tune_result["all_results"][:5], 1):
                params_str = ", ".join(f"{k}={v}" for k, v in res["params"].items())
                print(f"  #{rank}: Spearman={res['mean_spearman']:.4f} | {params_str}")

        # ── Start MLflow training run ──
        run_name = os.path.splitext(os.path.basename(args.output))[0]
        if gender:
            run_name = f"{run_name}_{gender}"

        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params({
                "start_date": args.start_date,
                "end_date": args.end_date,
                "elite_only": elite_only,
                "gender": gender_label,
                "match_distance": not args.no_match_distance,
                "distances": str(args.distances),
                "min_finishers": args.min_finishers,
                "n_features": len(feature_cols),
                "n_training_samples": len(train_df),
                "tuned": args.tune,
                "exclude_pack_features": not args.include_pack_features,
                "time_decay_half_life": 0.0 if args.no_time_decay else args.time_decay_half_life,
            })
            if model_params:
                mlflow.log_params({f"hp_{k}": v for k, v in model_params.items()})

            # Log feature list as artifact
            feature_list_path = os.path.join(project_root, "mlruns", "_tmp_features.txt")
            os.makedirs(os.path.dirname(feature_list_path), exist_ok=True)
            with open(feature_list_path, "w") as f:
                f.write("\n".join(feature_cols))
            mlflow.log_artifact(feature_list_path, "config")
            os.remove(feature_list_path)

            # Train final models
            logger.info("Training models...")
            decay_hl = 0.0 if args.no_time_decay else args.time_decay_half_life
            bundle = train_baseline_models(
                train_df, feature_cols, model_params=model_params,
                time_decay_half_life=decay_hl,
            )

            # Log training metrics
            if bundle.metadata.get("training_metrics"):
                for name, metrics in bundle.metadata["training_metrics"].items():
                    mlflow.log_metric(f"train_mae_{name}", metrics.get("mae", 0))
                    mlflow.log_metric(f"train_n_{name}", metrics.get("n_samples", 0))

            # Train distance-specific split models (swim, bike, run per distance)
            # Uses reduced feature set focused on individual athlete ability
            if not args.include_pack_features:
                logger.info("Training distance-specific split models (ABILITY-ONLY: pack features excluded)...")
            else:
                logger.info("Training distance-specific split models...")
            dist_split_models, split_feature_cols = train_distance_split_models(
                train_df, feature_cols, model_params=model_params,
                exclude_pack_features=not args.include_pack_features,
                time_decay_half_life=decay_hl,
            )
            bundle.distance_split_models = dist_split_models
            bundle.split_feature_columns = split_feature_cols
            if dist_split_models:
                mlflow.set_tag("distance_split_distances", ",".join(dist_split_models.keys()))
                for dist, models in dist_split_models.items():
                    mlflow.set_tag(f"dist_splits_{dist}", ",".join(models.keys()))
                logger.info(f"Distance-specific split models: {list(dist_split_models.keys())}")
            else:
                logger.info("No distance-specific split models trained (insufficient data)")

            # Store training mode in metadata so simulation can adjust pack effect scaling
            bundle.metadata["exclude_pack_features"] = not args.include_pack_features

            # Optimize ensemble weight via CV (if requested)
            if args.optimize_weight and bundle.model_ranker is not None:
                logger.info("Optimizing ensemble ranker/percentile blend weight via CV...")
                weight_result = optimize_ensemble_weight(
                    train_df, feature_cols,
                    model_params=model_params,
                    time_decay_half_life=decay_hl,
                )
                best_w = weight_result["best_weight"]
                bundle.metadata["ensemble_ranker_weight"] = best_w
                mlflow.log_metric("ensemble_ranker_weight", best_w)
                logger.info(f"Optimal ensemble weight: {best_w:.2f}")
                print(f"\n--- Ensemble Weight Optimization ---")
                for w in sorted(weight_result["weight_scores"].keys()):
                    s = weight_result["weight_scores"][w]
                    marker = " * BEST" if w == best_w else (" <-- 50/50" if w == 0.5 else "")
                    print(f"  {w:.2f}: P@10={s['P@10']:.1%}, P@3={s['P@3']:.1%}{marker}")
                print(f"  Stored weight: {best_w:.2f}")
            elif args.optimize_weight:
                logger.warning("Cannot optimize weight: no LGBMRanker trained")

            # Compute empirical residual covariance from training data predictions
            # This produces per-distance 5×5 covariance matrices used by the MVN noise
            # model in Monte Carlo simulation (Phase 2 improvement)
            logger.info("Computing empirical residual covariance matrices...")
            residual_stats = compute_residual_stats(
                train_df, bundle, feature_cols, split_feature_cols
            )
            if residual_stats:
                bundle.metadata["residual_stats"] = residual_stats
                # Log per-split sigmas to MLflow
                for dist_key, stats in residual_stats.items():
                    if "per_split_sigma" in stats:
                        for split_name, sigma in stats["per_split_sigma"].items():
                            mlflow.log_metric(f"resid_sigma_{dist_key}_{split_name}", sigma)
                        mlflow.log_metric(f"resid_n_samples_{dist_key}", stats["n_samples"])
                logger.info(f"Residual stats computed for: {list(residual_stats.keys())}")
            else:
                logger.warning("Could not compute residual stats (insufficient data or models)")

            # Learn swim gap distributions for percentile-based simulation
            logger.info("Learning swim gap-to-leader distributions...")
            swim_gap_dists = learn_swim_gap_distributions(train_df)
            if swim_gap_dists:
                bundle.metadata["swim_gap_distributions"] = swim_gap_dists
                for dist_key, stats in swim_gap_dists.items():
                    mlflow.log_metrics({
                        f"swim_gap_{dist_key}_n_races": stats["n_races"],
                        f"swim_gap_{dist_key}_median": stats["gap_p50"],
                        f"swim_gap_{dist_key}_p90": stats["gap_p90"],
                    })
                logger.info(f"Swim gap distributions learned for: {list(swim_gap_dists.keys())}")
            else:
                logger.warning("Could not learn swim gap distributions")

            # Learn pack effects from historical data and store in bundle
            # Overall params (all distances combined) for backward compatibility
            logger.info("Learning pack effects from historical data...")
            pack_data = fetch_pack_effect_data(
                engine,
                start_date=args.start_date,
                end_date=args.end_date,
                elite_only=elite_only,
            )
            if not pack_data.empty:
                learned_pack_params = learn_pack_effects_from_data(pack_data)
                bundle.metadata["pack_effect_params"] = pack_params_to_dict(learned_pack_params)
                mlflow.log_metrics({
                    "pack_front_bonus_sec": learned_pack_params.front_pack_bonus_sec,
                    "pack_chase_penalty_sec": learned_pack_params.chase_penalty_sec,
                    "pack_n_observations": learned_pack_params.n_observations,
                    "pack_n_races": learned_pack_params.n_races,
                })
                logger.info(
                    f"Learned pack effects (overall): bonus={learned_pack_params.front_pack_bonus_sec:.1f}s, "
                    f"penalty={learned_pack_params.chase_penalty_sec:.1f}s "
                    f"(from {learned_pack_params.n_observations} observations, "
                    f"{learned_pack_params.n_races} races)"
                )

                # Distance-specific pack params — draft-legal distances only
                pack_params_by_distance = {}
                for dist_cat in ["sprint", "standard"]:
                    dist_pack_data = fetch_pack_effect_data(
                        engine,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        elite_only=elite_only,
                        distance_category=dist_cat,
                    )
                    if not dist_pack_data.empty and len(dist_pack_data) >= 100:
                        dist_params = learn_pack_effects_from_data(dist_pack_data)
                        dist_params.distance_category = dist_cat
                        pack_params_by_distance[dist_cat] = pack_params_to_dict(dist_params)
                        mlflow.log_metrics({
                            f"pack_{dist_cat}_front_bonus": dist_params.front_pack_bonus_sec,
                            f"pack_{dist_cat}_chase_penalty": dist_params.chase_penalty_sec,
                            f"pack_{dist_cat}_n_obs": dist_params.n_observations,
                        })
                        logger.info(
                            f"  {dist_cat}: bonus={dist_params.front_pack_bonus_sec:.1f}s, "
                            f"penalty={dist_params.chase_penalty_sec:.1f}s "
                            f"({dist_params.n_observations} obs, {dist_params.n_races} races)"
                        )
                    else:
                        n = len(dist_pack_data) if not dist_pack_data.empty else 0
                        logger.info(f"  {dist_cat}: insufficient data ({n} rows), will use overall")

                if pack_params_by_distance:
                    bundle.metadata["pack_effect_params_by_distance"] = pack_params_by_distance

                # Tier-specific pack params — learn different pack dynamics per tier
                # WTCS (tier 1) has tighter packs, Continental (tier 3+) has wider gaps
                pack_params_by_tier = {}
                for tier_val, tier_label in [(1, "tier1"), (2, "tier2")]:
                    tier_pack_data = fetch_pack_effect_data(
                        engine,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        elite_only=elite_only,
                        event_tier=tier_val,
                    )
                    if not tier_pack_data.empty and len(tier_pack_data) >= 50:
                        tier_params = learn_pack_effects_from_data(tier_pack_data)
                        pack_params_by_tier[tier_label] = pack_params_to_dict(tier_params)
                        mlflow.log_metrics({
                            f"pack_{tier_label}_front_bonus": tier_params.front_pack_bonus_sec,
                            f"pack_{tier_label}_chase_penalty": tier_params.chase_penalty_sec,
                            f"pack_{tier_label}_n_obs": tier_params.n_observations,
                        })
                        logger.info(
                            f"  {tier_label}: bonus={tier_params.front_pack_bonus_sec:.1f}s, "
                            f"penalty={tier_params.chase_penalty_sec:.1f}s "
                            f"({tier_params.n_observations} obs, {tier_params.n_races} races)"
                        )
                    else:
                        n = len(tier_pack_data) if not tier_pack_data.empty else 0
                        logger.info(f"  {tier_label}: insufficient data ({n} rows), will use overall")

                if pack_params_by_tier:
                    bundle.metadata["pack_effect_params_by_tier"] = pack_params_by_tier
            else:
                logger.warning("No pack effect data available, using defaults")

            # Learn merge parameters from swim-to-bike pack transitions
            logger.info("Learning pack merge parameters from swim-to-bike transitions...")
            transition_data = fetch_swim_to_bike_transitions(
                engine,
                start_date=args.start_date,
                end_date=args.end_date,
                elite_only=elite_only,
            )
            if not transition_data.empty:
                learned_merge_params = learn_merge_params_from_data(transition_data)
                bundle.metadata["merge_params"] = merge_params_to_dict(learned_merge_params)
                mlflow.log_metrics({
                    "merge_beta_0": learned_merge_params.beta_0,
                    "merge_beta_gap": learned_merge_params.beta_gap,
                    "merge_beta_chase_size": learned_merge_params.beta_chase_size,
                    "merge_n_observations": learned_merge_params.n_observations,
                    "merge_n_races": learned_merge_params.n_races,
                })
                logger.info(
                    f"Learned merge params (overall): beta_0={learned_merge_params.beta_0:.3f}, "
                    f"beta_gap={learned_merge_params.beta_gap:.3f}, "
                    f"beta_chase_size={learned_merge_params.beta_chase_size:.3f} "
                    f"(from {learned_merge_params.n_observations} merge situations, "
                    f"{learned_merge_params.n_races} races)"
                )

                # Distance-specific merge params
                merge_params_by_distance = {}
                for dist_cat in ["sprint", "standard"]:
                    dist_trans_data = fetch_swim_to_bike_transitions(
                        engine,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        elite_only=elite_only,
                        distance_category=dist_cat,
                    )
                    if not dist_trans_data.empty and len(dist_trans_data) >= 50:
                        dist_merge = learn_merge_params_from_data(dist_trans_data)
                        merge_params_by_distance[dist_cat] = merge_params_to_dict(dist_merge)
                        mlflow.log_metrics({
                            f"merge_{dist_cat}_beta_0": dist_merge.beta_0,
                            f"merge_{dist_cat}_beta_gap": dist_merge.beta_gap,
                            f"merge_{dist_cat}_n_obs": dist_merge.n_observations,
                        })
                        logger.info(
                            f"  {dist_cat}: beta_0={dist_merge.beta_0:.3f}, "
                            f"beta_gap={dist_merge.beta_gap:.3f} "
                            f"({dist_merge.n_observations} obs)"
                        )
                    else:
                        n = len(dist_trans_data) if not dist_trans_data.empty else 0
                        logger.info(f"  {dist_cat}: insufficient data ({n} rows), will use overall")

                if merge_params_by_distance:
                    bundle.metadata["merge_params_by_distance"] = merge_params_by_distance

                # Tier-specific merge params
                merge_params_by_tier = {}
                for tier_val, tier_label in [(1, "tier1"), (2, "tier2")]:
                    tier_trans_data = fetch_swim_to_bike_transitions(
                        engine,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        elite_only=elite_only,
                        event_tier=tier_val,
                    )
                    if not tier_trans_data.empty and len(tier_trans_data) >= 30:
                        tier_merge = learn_merge_params_from_data(tier_trans_data)
                        merge_params_by_tier[tier_label] = merge_params_to_dict(tier_merge)
                        mlflow.log_metrics({
                            f"merge_{tier_label}_beta_0": tier_merge.beta_0,
                            f"merge_{tier_label}_beta_gap": tier_merge.beta_gap,
                            f"merge_{tier_label}_n_obs": tier_merge.n_observations,
                        })
                        logger.info(
                            f"  {tier_label}: beta_0={tier_merge.beta_0:.3f}, "
                            f"beta_gap={tier_merge.beta_gap:.3f} "
                            f"({tier_merge.n_observations} obs)"
                        )
                    else:
                        n = len(tier_trans_data) if not tier_trans_data.empty else 0
                        logger.info(f"  {tier_label}: insufficient data ({n} rows), will use overall")

                if merge_params_by_tier:
                    bundle.metadata["merge_params_by_tier"] = merge_params_by_tier
            else:
                logger.warning("No transition data available, merge params will use defaults")

            # Determine output path
            out_path = args.output
            if gender and args.train_both_genders:
                base, ext = os.path.splitext(out_path)
                out_path = f"{base}_{gender}{ext}"

            save_model_bundle(bundle, out_path)
            logger.info(f"Model bundle saved to {out_path}")

            # Log model bundle as artifact
            mlflow.log_artifact(out_path, "model")

            # Store the run_id in the bundle metadata for linking backtest results
            bundle.metadata["mlflow_run_id"] = run.info.run_id
            save_model_bundle(bundle, out_path)  # Re-save with run_id

            # Tag with output path for easy lookup
            mlflow.set_tag("model_path", out_path)

        # Print summary
        print(f"\n--- Training Summary (gender={gender_label}) ---")
        print(f"Training samples: {len(train_df)}")
        print(f"Feature columns: {len(feature_cols)}")
        print(f"MLflow run ID: {run.info.run_id}")
        if model_params:
            print(f"Tuned params: {model_params}")
        if bundle.metadata.get("training_metrics"):
            for name, metrics in bundle.metadata["training_metrics"].items():
                mae_val = metrics.get('mae', 0)
                unit = "pct" if "pct" in name else "s"
                fmt = f"{mae_val:.4f}" if "pct" in name else f"{mae_val:.1f}"
                print(f"  {name}: MAE={fmt}{unit}, n={metrics.get('n_samples', 'N/A')}")
        if bundle.distance_split_models:
            print(f"Distance-specific split models: {list(bundle.distance_split_models.keys())}")
            print(f"Split model features: {len(bundle.split_feature_columns)} (reduced from {len(feature_cols)})")
        print(f"\nModel saved to: {out_path}")
        print(f"\nView experiment: mlflow ui --backend-store-uri file:///{os.path.join(project_root, 'mlruns').replace(os.sep, '/')}")
        print("\nTo run predictions:")
        print(f"  python scripts/predict_program.py --event_id <ID> --prog_id <ID> --model_path {out_path}")


if __name__ == "__main__":
    main()
