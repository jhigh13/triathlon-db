"""
Training Orchestration Script for Triathlon ML Pipeline

This script coordinates the complete machine learning pipeline:
1. Data extraction from PostgreSQL
2. Feature engineering
3. Label generation
4. Preprocessing with PCA
5. Model training and evaluation per distance
6. Model persistence

Usage:
    python train.py [--distance DISTANCE] [--no-pca] [--pca-components N]

Example:
    python train.py --distance Sprint --pca-components 10
    python train.py --no-pca  # Train without PCA
"""

import argparse
import logging
import joblib
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.pipeline import Pipeline

# Import our custom modules
from ml.data_extraction import DataExtractor
from ml.features import FeatureEngineer
from ml.label_generation import LabelGenerator
from ml.preprocessing import TriathlonPreprocessor
from Data_Import.database import get_engine

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TriathlonMLTrainer:
    """Orchestrates the complete triathlon ML pipeline."""

    def __init__(self, apply_pca: bool = True, pca_components: int = None):
        """
        Initialize trainer.

        Args:
            apply_pca: Whether to apply PCA preprocessing
            pca_components: Number of PCA components (None for auto)
        """
        self.apply_pca = apply_pca
        self.pca_components = pca_components
        self.models = {}
        self.results = {}

        # Create output directories
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

        # Initialize pipeline components
        self.data_extractor = None
        self.feature_engineer = FeatureEngineer()
        self.label_generator = LabelGenerator()
        self.preprocessor = TriathlonPreprocessor(apply_pca, pca_components)

    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load data from database and prepare for modeling.

        Returns:
            DataFrame ready for model training
        """
        logger.info("Starting data loading and preparation...")

        # Get database engine
        engine = get_engine(echo=False)
        self.data_extractor = DataExtractor(engine)

        # Extract raw data
        raw_df = self.data_extractor.extract_base_dataset()

        # Engineer features
        featured_df = self.feature_engineer.engineer_features(raw_df)

        # Generate labels
        model_df = self.label_generator.generate_labels(featured_df)

        logger.info(f"Data preparation complete. Final dataset shape: {model_df.shape}")
        return model_df

    def create_models(self) -> dict:
        """
        Create model instances to evaluate.

        Returns:
            Dictionary of model name -> model instance
        """
        return {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
        }

    def evaluate_model_for_distance(
        self, df: pd.DataFrame, distance: str, cv_folds: int = 5
    ) -> dict:
        """
        Evaluate models for a specific distance.

        Args:
            df: Complete dataset
            distance: Distance to filter for (e.g., 'Sprint', 'Standard')
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating models for distance: {distance}")

        # Filter for specific distance
        distance_col = f"dist_{distance}"
        if distance_col not in df.columns:
            logger.warning(
                f"Distance column {distance_col} not found. Available: {[c for c in df.columns if c.startswith('dist_')]}"
            )
            return {}

        df_dist = df[df[distance_col] == 1].copy()

        if len(df_dist) == 0:
            logger.warning(f"No data found for distance: {distance}")
            return {}

        logger.info(f"Training on {len(df_dist)} samples for {distance}")

        # Prepare features and target
        X, y, groups = self.preprocessor.prepare_features_and_target(df_dist)

        # Remove rows with missing target
        valid_mask = ~y.isna()
        X, y, groups = X[valid_mask], y[valid_mask], groups[valid_mask]

        # Fit preprocessor
        X_processed = self.preprocessor.fit_transform(X, y)

        # Cross-validation setup
        cv = GroupKFold(n_splits=cv_folds)
        neg_mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

        # Evaluate each model
        models = self.create_models()
        results = {}

        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name} for {distance}...")

            try:
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model,
                    X_processed,
                    y,
                    cv=cv,
                    groups=groups,                    scoring=neg_mae_scorer,
                    n_jobs=-1,
                )

                mae_scores = -cv_scores

                results[model_name] = {
                    "mae_mean": float(mae_scores.mean()),
                    "mae_std": float(mae_scores.std()),
                    "mae_scores": mae_scores.tolist(),
                    "n_samples": int(len(df_dist)),
                }

                logger.info(
                    f"{model_name} - MAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f} seconds"
                )

                # Train final model on all data for persistence
                model.fit(X_processed, y)

                # Save model with preprocessor
                model_pipeline = Pipeline(
                    [
                        ("preprocessing", self.preprocessor.preprocessor),
                        ("model", model),
                    ]
                )

                model_filename = f"{distance}_{model_name}_{'pca' if self.apply_pca else 'nopca'}.pkl"
                model_path = self.model_dir / model_filename
                joblib.dump(model_pipeline, model_path)

                logger.info(f"Saved model to {model_path}")

            except Exception as e:
                logger.error(f"Error evaluating {model_name} for {distance}: {e}")
                results[model_name] = {"error": str(e)}

        return results

    def calculate_baseline_metrics(self, df: pd.DataFrame, distance: str) -> dict:
        """
        Calculate baseline metrics for comparison.

        Args:
            df: Dataset
            distance: Distance to evaluate

        Returns:
            Dictionary with baseline metrics
        """
        distance_col = f"dist_{distance}"
        if distance_col not in df.columns:
            return {}

        df_dist = df[df[distance_col] == 1].copy()

        if len(df_dist) == 0:
            return {}

        # Athlete-specific mean baseline
        athlete_means = df_dist.groupby("athlete_id")["TotalTime_sec"].mean()
        baseline_preds = df_dist["athlete_id"].map(athlete_means)        # Calculate MAE where both prediction and target are available
        valid_mask = ~df_dist["next_time_sec"].isna() & ~baseline_preds.isna()

        if valid_mask.sum() > 0:
            mae = abs(
                df_dist.loc[valid_mask, "next_time_sec"]
                - baseline_preds.loc[valid_mask]
            ).mean()
            return {"athlete_mean_baseline_mae": float(mae), "n_samples": int(valid_mask.sum())}

        return {}

    def train_models(self, distances: list = None) -> dict:
        """
        Train models for specified distances.

        Args:
            distances: List of distances to train for (None for all)

        Returns:
            Complete results dictionary
        """
        logger.info("Starting model training pipeline...")

        # Load and prepare data
        df = self.load_and_prepare_data()

        # Determine distances to train on
        if distances is None:
            available_distances = [
                col.replace("dist_", "")
                for col in df.columns
                if col.startswith("dist_")
            ]
            distances = [
                d
                for d in ["Sprint", "Standard", "Super Sprint"]
                if d in available_distances
            ]

        logger.info(f"Training models for distances: {distances}")

        # Train models for each distance
        all_results = {}

        for distance in distances:
            logger.info(f"\\n{'='*50}")
            logger.info(f"TRAINING FOR DISTANCE: {distance}")
            logger.info(f"{'='*50}")

            # Calculate baseline
            baseline_results = self.calculate_baseline_metrics(df, distance)

            # Train and evaluate models
            model_results = self.evaluate_model_for_distance(df, distance)

            # Combine results
            all_results[distance] = {
                "baseline": baseline_results,
                "models": model_results,
                "timestamp": datetime.now().isoformat(),
            }

            # Log summary
            if baseline_results:
                logger.info(
                    f"Baseline MAE: {baseline_results.get('athlete_mean_baseline_mae', 'N/A'):.2f} seconds"
                )

            for model_name, result in model_results.items():
                if "mae_mean" in result:
                    logger.info(
                        f"{model_name} MAE: {result['mae_mean']:.2f} ± {result['mae_std']:.2f} seconds"
                    )

        # Save results summary
        results_path = (
            self.model_dir
            / f"training_results_{'pca' if self.apply_pca else 'nopca'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        import json

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Training complete. Results saved to {results_path}")

        self.results = all_results
        return all_results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train triathlon race time prediction models"
    )
    parser.add_argument(
        "--distance",
        type=str,
        choices=["Sprint", "Standard", "Super Sprint"],
        help="Specific distance to train (default: all)",
    )
    parser.add_argument(
        "--no-pca", action="store_true", help="Disable PCA preprocessing"
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        help="Number of PCA components (default: auto-select for 95%% variance)",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5, help="Number of cross-validation folds"
    )

    args = parser.parse_args()

    # Configure trainer
    apply_pca = not args.no_pca
    distances = [args.distance] if args.distance else None

    # Initialize trainer
    trainer = TriathlonMLTrainer(
        apply_pca=apply_pca, pca_components=args.pca_components
    )

    # Train models
    results = trainer.train_models(distances=distances)    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for distance, dist_results in results.items():
        print(f"\n{distance}:")

        if "baseline" in dist_results and dist_results["baseline"]:
            baseline_mae = dist_results["baseline"].get("athlete_mean_baseline_mae")
            if baseline_mae:
                print(f"  Baseline MAE: {baseline_mae:.2f} seconds")

        if "models" in dist_results:
            for model_name, model_result in dist_results["models"].items():
                if "mae_mean" in model_result:
                    mae_mean = model_result["mae_mean"]
                    mae_std = model_result["mae_std"]
                    print(f"  {model_name}: {mae_mean:.2f} ± {mae_std:.2f} seconds")


if __name__ == "__main__":
    main()
