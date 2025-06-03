"""
Preprocessing Module for Triathlon ML Pipeline

This module handles preprocessing steps including feature selection, scaling,
imputation, and PCA dimensionality reduction before model training.

Based on the exploration in notebooks/model_pipeline.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import logging
from typing import List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TriathlonPreprocessor:
    """Handles preprocessing and PCA for triathlon ML pipeline."""

    def __init__(self, apply_pca: bool = True, pca_components: Optional[int] = None):
        """
        Initialize preprocessor.

        Args:
            apply_pca: Whether to apply PCA dimensionality reduction
            pca_components: Number of PCA components (None for auto-selection)
        """
        self.apply_pca = apply_pca
        self.pca_components = pca_components
        self.preprocessor = None
        self.feature_names = None

    def get_feature_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numeric and categorical feature columns.

        Args:
            df: DataFrame with engineered features

        Returns:
            Tuple of (numeric_features, categorical_features)
        """
        # Define expected numeric features based on notebook exploration
        numeric_features = [
            "TotalTime_sec",
            "days_since_last",
            "age",
            "rolling3_time_dist",
            "rolling5_time_dist",
            "rolling5_time_all",
            "rolling3_pos",
            "rolling5_pos",
            "recent_5_DNF_rate",
        ]

        # Find one-hot encoded categorical features (created by pd.get_dummies)
        categorical_features = [
            col
            for col in df.columns
            if any(col.startswith(prefix) for prefix in ["rt_", "dist_", "mode_"])
        ]

        # Filter to only include columns that exist in the dataframe
        numeric_features = [col for col in numeric_features if col in df.columns]
        categorical_features = [
            col for col in categorical_features if col in df.columns
        ]

        logger.info(
            f"Identified {len(numeric_features)} numeric and {len(categorical_features)} categorical features"
        )

        return numeric_features, categorical_features

    def create_preprocessing_pipeline(
        self, numeric_features: List[str], categorical_features: List[str]
    ) -> Pipeline:
        """
        Create scikit-learn preprocessing pipeline.

        Args:
            numeric_features: List of numeric feature column names
            categorical_features: List of categorical feature column names

        Returns:
            Preprocessing pipeline
        """
        logger.info("Creating preprocessing pipeline...")

        # Numeric pipeline: impute missing values and scale
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Categorical features are already one-hot encoded, just treat as numeric
        # (they're 0/1 values from pd.get_dummies)
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
            ]
        )
        # Combine numeric and categorical preprocessing
        preprocessor_steps = [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]

        column_transformer = ColumnTransformer(
            transformers=preprocessor_steps,
            remainder="drop",  # Drop any columns not specified
        )

        # Create full pipeline with optional PCA
        pipeline_steps = [("preprocessing", column_transformer)]

        if self.apply_pca:
            if self.pca_components is None:
                # Auto-select number of components (95% variance)
                pca_step = PCA(n_components=0.95, random_state=42)
            else:
                pca_step = PCA(n_components=self.pca_components, random_state=42)

            pipeline_steps.append(("pca", pca_step))

        return Pipeline(pipeline_steps)

    def prepare_features_and_target(
        self, df: pd.DataFrame, target_col: str = "next_time_sec"
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare features (X), target (y), and groups for model training.

        Args:
            df: DataFrame with engineered features and labels
            target_col: Name of target column

        Returns:
            Tuple of (X, y, groups) where groups is athlete_id for GroupKFold
        """
        logger.info(f"Preparing features and target ({target_col})...")        # Ensure all column names are strings (handle pandas Index objects)
        df.columns = [str(col) for col in df.columns]

        numeric_features, categorical_features = self.get_feature_columns(df)

        # Ensure all feature columns are numeric
        all_features = numeric_features + categorical_features
        for col in all_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Prepare X, y, groups
        X = df[all_features]
        y = df[target_col]
        groups = df["athlete_id"]        # Ensure X column names are strings (handle pandas Index objects)
        X.columns = [str(col) for col in X.columns]

        # Store feature names for later reference
        self.feature_names = all_features

        logger.info(f"Prepared {X.shape[1]} features for {len(X)} samples")

        return X, y, groups

    def fit_preprocessor(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        """
        Fit the preprocessing pipeline on training data.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Fitted preprocessing pipeline
        """
        logger.info("Fitting preprocessing pipeline...")

        numeric_features, categorical_features = self.get_feature_columns(X)
        self.preprocessor = self.create_preprocessing_pipeline(
            numeric_features, categorical_features
        )

        # Fit the preprocessor
        self.preprocessor.fit(X, y)

        # Log PCA info if applied
        if self.apply_pca and "pca" in self.preprocessor.named_steps:
            pca = self.preprocessor.named_steps["pca"]
            logger.info(
                f"PCA retained {pca.n_components_} components explaining "
                f"{pca.explained_variance_ratio_.sum():.3f} of variance"
            )

        return self.preprocessor

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessor.

        Args:
            X: Feature matrix to transform

        Returns:
            Transformed feature matrix
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_preprocessor first.")

        return self.preprocessor.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Fit preprocessor and transform features in one step.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Transformed feature matrix
        """
        self.fit_preprocessor(X, y)
        return self.transform_features(X)


def create_preprocessor(
    apply_pca: bool = True, pca_components: Optional[int] = None
) -> TriathlonPreprocessor:
    """
    Factory function to create a preprocessor instance.

    Args:
        apply_pca: Whether to apply PCA
        pca_components: Number of PCA components

    Returns:
        TriathlonPreprocessor instance
    """
    return TriathlonPreprocessor(apply_pca=apply_pca, pca_components=pca_components)


if __name__ == "__main__":
    # For testing
    logger.info("Preprocessing module loaded successfully")

    # Test with sample data
    sample_data = pd.DataFrame(
        {
            "TotalTime_sec": [3600, 3700, 3550],
            "age": [25, 30, 28],
            "rolling3_time_dist": [3580, 3680, 3570],
            "dist_Sprint": [1, 0, 1],
            "dist_Standard": [0, 1, 0],
            "rt_Triathlon": [1, 1, 1],
            "mode_individual": [1, 1, 1],
            "next_time_sec": [3580, 3650, 3590],
            "athlete_id": [1, 2, 3],
        }
    )

    preprocessor = TriathlonPreprocessor(apply_pca=True)
    X, y, groups = preprocessor.prepare_features_and_target(sample_data)
    print(f"Sample preprocessing: X shape {X.shape}, y shape {y.shape}")
