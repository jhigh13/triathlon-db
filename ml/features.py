"""
Feature Engineering Module for Triathlon ML Pipeline

This module handles data cleaning, categorical feature engineering, rolling averages,
and other transformations to prepare features for machine learning models.

Based on the exploration in notebooks/model_pipeline.ipynb
"""

import pandas as pd
import numpy as np
import logging
from typing import Set, Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for triathlon ML pipeline."""

    def __init__(self):
        """Initialize FeatureEngineer with predefined categories."""
        self.race_types = {
            "Triathlon",
            "Duathlon",
            "Aquathlon",
            "Winter Duathlon",
            "Aquabike",
        }
        self.distances = {
            "Sprint",
            "Standard",
            "Super Sprint",
            "Long Distance",
            "Middle Distance",
            "Olympic",
            "Ironman",
        }
        self.other_flags = {"Paratriathlon", "Mixed Relay", "Relay", "Aquabike"}
        self.allowed_flags = {"Mixed Relay", "Paratriathlon"}

    def clean_initial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform initial data cleaning steps.

        Args:
            df: Raw joined DataFrame from data extraction

        Returns:
            Cleaned DataFrame
        """
        logger.info("Performing initial data cleaning...")

        # Drop duplicate columns from join
        df = df.drop(
            columns=["EventSpecifications_y", "CategoryName_y", "category_medical"],
            errors="ignore",
        )
        df = df.rename(
            columns={
                "EventSpecifications_x": "EventSpecifications",
                "CategoryName_x": "CategoryName",
            },
            errors="ignore",
        )

        logger.info(f"Shape after initial cleaning: {df.shape}")
        return df

    def parse_event_specs(self, spec_string) -> pd.Series:
        """
        Parse EventSpecifications string into race_type, distance, and event_flags.

        Args:
            spec_string: EventSpecifications column value

        Returns:
            Series with [race_type, distance, event_flags]
        """
        if not isinstance(spec_string, str):
            return pd.Series([None, None, None])

        items = [s.strip() for s in spec_string.split(",")]

        # Find matches
        race = next((x for x in items if x in self.race_types), None)
        distance = next((x for x in items if x in self.distances), None)

        # Other flags: those in other_flags, or anything else not found above
        others = [
            x
            for x in items
            if x in self.other_flags
            or (x not in self.race_types and x not in self.distances)
        ]

        return pd.Series([race, distance, ", ".join(others) if others else None])

    def process_event_specifications(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse EventSpecifications and filter for triathlon events only.

        Args:
            df: DataFrame with EventSpecifications column

        Returns:
            DataFrame with parsed race_type, distance, event_flags columns
        """
        logger.info("Processing event specifications...")

        # Parse event specifications
        df[["race_type", "distance", "event_flags"]] = df["EventSpecifications"].apply(
            self.parse_event_specs
        )
        # Filter for triathlon events only
        df = df[
            ~df["race_type"].isin(
                ["Duathlon", "Aquathlon", "Winter Duathlon", "Aquabike"]
            )
        ].copy()

        # Drop rows with missing critical data
        df = df.dropna(
            subset=[
                "BikeTime",
                "T2",
                "SwimTime",
                "RunTime",
                "T1",
                "age",
                "EventSpecifications",
            ]
        )

        logger.info(f"Shape after event specification processing: {df.shape}")
        return df

    def create_event_mode_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create simplified event_mode categorical feature from event_flags.

        Args:
            df: DataFrame with event_flags column

        Returns:
            DataFrame with event_mode column
        """
        logger.info("Creating event_mode feature...")

        # Split event_flags into lists, handling NaNs
        df["flags_list"] = (
            df["event_flags"]
            .fillna("")
            .apply(lambda x: [s.strip() for s in x.split(",") if s.strip()])
        )

        # Create boolean indicator columns
        df["is_mixed_relay"] = (
            df["flags_list"].apply(lambda lst: "Mixed Relay" in lst).astype(int)
        )
        df["is_paratriathlon"] = (
            df["flags_list"].apply(lambda lst: "Paratriathlon" in lst).astype(int)
        )
        # Drop rows with flags other than our allowed ones
        df = df[
            df["flags_list"].apply(lambda lst: set(lst).issubset(self.allowed_flags))
        ].copy()

        # Create event_mode categorical feature
        def determine_event_mode(row):
            if row["is_mixed_relay"] and row["is_paratriathlon"]:
                return "para_mixed_relay"
            if row["is_mixed_relay"]:
                return "mixed_relay"
            if row["is_paratriathlon"]:
                return "paratriathlon"
            return "individual"

        df["event_mode"] = df.apply(determine_event_mode, axis=1)

        # Clean up helper columns
        df.drop(
            columns=["event_flags", "flags_list", "EventSpecifications"], inplace=True
        )

        logger.info(f"Event mode distribution:\\n{df['event_mode'].value_counts()}")
        return df

    def handle_dnf_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create DNF flags and recent DNF rates, then filter for valid times.

        Args:
            df: DataFrame with TotalTime column

        Returns:
            DataFrame with DNF features and valid times only
        """
        logger.info("Handling DNF flags...")

        # Flag DNF/DNS rows
        df["DNF_flag"] = df["TotalTime"].isin(["DNF", "DNS"]).astype(int)

        # Calculate recent DNF rates (rolling window)
        df = df.sort_values(["athlete_id", "EventDate"])
        df["recent_5_DNF_rate"] = (
            df.groupby("athlete_id")["DNF_flag"]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        # Filter for valid times only (format: HH:MM:SS)
        df_valid = df[
            df["TotalTime"].str.match(r"^\d{2}:\d{2}:\d{2}$", na=False)
        ].copy()

        logger.info(f"Shape after DNF handling (valid times only): {df_valid.shape}")
        return df_valid

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling average features for times and positions.

        Args:
            df: DataFrame with TotalTime and Position columns

        Returns:
            DataFrame with rolling average features
        """
        logger.info("Creating rolling features...")

        # Convert finish time to seconds for numeric operations
        df["TotalTime_sec"] = pd.to_timedelta(df["TotalTime"]).dt.total_seconds()
        df["Position"] = pd.to_numeric(df["Position"], errors="coerce")

        # Filter out extremely short times (< 10 minutes)
        df = df[df["TotalTime_sec"] > 600]

        # Sort for rolling calculations
        df = df.sort_values(["athlete_id", "EventDate"])

        # Rolling averages by distance (more relevant for prediction)
        df["rolling3_time_dist"] = (
            df.groupby(["athlete_id", "distance"])["TotalTime_sec"]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        df["rolling5_time_dist"] = (
            df.groupby(["athlete_id", "distance"])["TotalTime_sec"]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        # Rolling averages across all distances
        df["rolling5_time_all"] = (
            df.groupby("athlete_id")["TotalTime_sec"]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Rolling position averages
        df["rolling3_pos"] = (
            df.groupby("athlete_id")["Position"]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling5_pos"] = (
            df.groupby("athlete_id")["Position"]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        logger.info("Rolling features created successfully")
        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features like days since last race.

        Args:
            df: DataFrame with EventDate column

        Returns:
            DataFrame with time-based features
        """
        logger.info("Creating time-based features...")

        # Days since last race
        df["days_since_last"] = df.groupby("athlete_id")["EventDate"].diff().dt.days

        # Age at race date (if athlete_yob is available)
        if "athlete_yob" in df.columns:
            df["age_at_race"] = df["EventDate"].dt.year - df["athlete_yob"]

        logger.info("Time-based features created successfully")
        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode categorical features.

        Args:
            df: DataFrame with categorical columns

        Returns:
            DataFrame with one-hot encoded features
        """
        logger.info("Encoding categorical features...")
        # One-hot encoding for categorical features
        df = pd.get_dummies(
            df,
            columns=["race_type", "distance", "event_mode"],
            prefix=["rt", "dist", "mode"],
        )

        logger.info("Categorical encoding completed")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.

        Args:
            df: Raw DataFrame from data extraction

        Returns:
            DataFrame with all features engineered
        """
        df = self.clean_initial_data(df)
        df = self.process_event_specifications(df)
        df = self.create_event_mode_feature(df)
        df = self.handle_dnf_flags(df)
        df = self.create_rolling_features(df)
        df = self.create_time_features(df)
        df = self.encode_categorical_features(df)        # Ensure all column names are strings (handle pandas Index objects)
        df.columns = [str(col) for col in df.columns]

        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df


def engineer_features_for_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to engineer features for a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer()
    return engineer.engineer_features(df)


if __name__ == "__main__":
    # For testing - would normally use real data
    logger.info("Feature engineering module loaded successfully")
    print("Feature engineering module ready for use")
