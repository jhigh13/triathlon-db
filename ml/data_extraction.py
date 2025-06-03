"""
Data Extraction Module for Triathlon ML Pipeline

This module handles the extraction and initial joining of data from the PostgreSQL database
tables (race_results, events, athlete) to create the base dataset for feature engineering.

Based on the exploration in notebooks/model_pipeline.ipynb
"""

import pandas as pd
from sqlalchemy import create_engine, text
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExtractor:
    """Extracts and joins triathlon data from PostgreSQL database."""

    def __init__(self, engine):
        """
        Initialize DataExtractor with database engine.

        Args:
            engine: SQLAlchemy engine for database connection
        """
        self.engine = engine

    def get_available_tables(self) -> list:
        """
        Get list of available tables in the database.

        Returns:
            List of table names
        """
        with self.engine.connect() as conn:
            tables = conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
                )
            ).fetchall()
        return [t[0] for t in tables]

    def load_raw_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load the three main tables into DataFrames.

        Returns:
            Tuple of (athletes_df, events_df, results_df)
        """
        logger.info("Loading raw tables from database...")

        # Load dimension tables
        athletes_df = pd.read_sql_table("athlete", self.engine)
        events_df = pd.read_sql_table("events", self.engine)
        results_df = pd.read_sql_table("race_results", self.engine)

        logger.info(
            f"Loaded {len(athletes_df)} athletes, {len(events_df)} events, {len(results_df)} race results"
        )

        return athletes_df, events_df, results_df

    def join_tables(
        self,
        athletes_df: pd.DataFrame,
        events_df: pd.DataFrame,
        results_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join the three tables to create the base ML dataset.

        Args:
            athletes_df: Athlete dimension table
            events_df: Events dimension table
            results_df: Race results fact table

        Returns:
            Joined DataFrame ready for cleaning and feature engineering
        """
        logger.info("Joining tables for ML dataset...")

        # Convert EventDate to datetime for feature engineering
        events_df["EventDate"] = pd.to_datetime(events_df["EventDate"])

        # Merge results with events and athletes
        ml_df = results_df.merge(events_df, on="EventID", how="left").merge(
            athletes_df, left_on="athlete_id", right_on="athlete_id", how="left"
        )

        logger.info(f"Joined dataset shape: {ml_df.shape}")

        return ml_df

    def extract_base_dataset(self) -> pd.DataFrame:
        """
        Complete extraction pipeline: load tables and join them.

        Returns:
            Base ML dataset ready for cleaning
        """
        athletes_df, events_df, results_df = self.load_raw_tables()
        ml_df = self.join_tables(athletes_df, events_df, results_df)
        return ml_df


def extract_data_from_engine(engine) -> pd.DataFrame:
    """
    Convenience function to extract data using an existing engine.

    Args:
        engine: SQLAlchemy engine

    Returns:
        Base ML dataset
    """
    extractor = DataExtractor(engine)
    return extractor.extract_base_dataset()


if __name__ == "__main__":
    # For testing - import engine from existing config
    from Data_Import.database import get_engine

    engine = get_engine(echo=False)
    extractor = DataExtractor(engine)

    # Test the extraction
    df = extractor.extract_base_dataset()
    print(f"Extracted dataset with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
