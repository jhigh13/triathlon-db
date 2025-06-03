"""
Label Generation Module for Triathlon ML Pipeline

This module creates target labels for the next-race prediction model by shifting
athlete finish times and positions to create "next_time_sec" and "next_position" targets.

Based on the exploration in notebooks/model_pipeline.ipynb
"""

import pandas as pd
import logging
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generates target labels for next-race prediction."""

    def __init__(self):
        """Initialize LabelGenerator."""
        pass

    def create_next_race_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create next-race finish time and position labels using athlete-grouped shifts.

        Args:
            df: DataFrame with athlete_id, EventDate, TotalTime_sec, Position columns

        Returns:
            DataFrame with next_time_sec and next_position columns added
        """
        logger.info("Creating next-race labels...")

        # Ensure data is sorted by athlete and event date
        df = df.sort_values(["athlete_id", "EventDate"])

        # Create next-race labels using groupby shift
        df["next_time_sec"] = df.groupby("athlete_id")["TotalTime_sec"].shift(-1)
        df["next_position"] = df.groupby("athlete_id")["Position"].shift(-1)

        # Optional: Create human-readable next time for inspection
        df["next_time"] = pd.to_timedelta(df["next_time_sec"], unit="s").astype(str)

        logger.info(
            f"Added next-race labels. Rows with valid next race: {df['next_time_sec'].notna().sum()}"
        )

        return df

    def filter_for_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to keep only rows that have a next race (valid labels).

        Args:
            df: DataFrame with next_time_sec and next_position columns
              Returns:
            DataFrame with only rows having valid next-race labels
        """
        logger.info("Filtering for modeling dataset...")

        initial_rows = len(df)

        # Drop rows without a next race (final race per athlete)
        df_model = df.dropna(subset=["next_time_sec", "next_position"]).reset_index(
            drop=True
        )

        final_rows = len(df_model)

        if initial_rows > 0:
            retention_rate = final_rows / initial_rows
            logger.info(
                f"Filtered from {initial_rows} to {final_rows} rows ({retention_rate:.1%} retention)"
            )
        else:
            logger.info(f"No data to filter - starting with {initial_rows} rows")

        return df_model

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete label generation pipeline.

        Args:
            df: DataFrame with cleaned and feature-engineered triathlon data

        Returns:
            DataFrame ready for model training with next-race labels
        """
        df_with_labels = self.create_next_race_labels(df)
        df_model = self.filter_for_modeling(df_with_labels)
        return df_model


def generate_labels_for_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to generate labels for a DataFrame.

    Args:
        df: Input DataFrame with triathlon data

    Returns:
        DataFrame with next-race labels ready for modeling
    """
    generator = LabelGenerator()
    return generator.generate_labels(df)


if __name__ == "__main__":
    # For testing - create sample data
    import numpy as np

    # Create sample data with multiple athletes and races
    sample_data = {
        "athlete_id": [1, 1, 1, 2, 2, 3, 3, 3, 3],
        "EventDate": pd.to_datetime(
            [
                "2023-01-01",
                "2023-03-01",
                "2023-06-01",
                "2023-02-01",
                "2023-05-01",
                "2023-01-15",
                "2023-04-01",
                "2023-07-01",
                "2023-09-01",
            ]
        ),
        "TotalTime_sec": [3600, 3550, 3580, 3700, 3650, 3500, 3480, 3520, 3490],
        "Position": [10, 8, 12, 15, 13, 5, 4, 7, 6],
    }

    df_test = pd.DataFrame(sample_data)

    # Test label generation
    generator = LabelGenerator()
    df_result = generator.generate_labels(df_test)

    print("Sample label generation:")
    print(
        df_result[
            [
                "athlete_id",
                "EventDate",
                "TotalTime_sec",
                "next_time_sec",
                "Position",
                "next_position",
            ]
        ].head()
    )
