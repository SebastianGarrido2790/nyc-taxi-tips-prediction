"""
Data Transformation Component.

This module handles the cleaning, imputation, and temporal splitting of the dataset.
It implements the FTI (Feature-Training-Inference) architecture by ensuring
the data is devoid of outliers and missing values before feature engineering.
"""

import sys
import polars as pl
from src.entity.config_entity import DataTransformationConfig
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataTransformation:
    """
    Transforms data: Cleaning, Imputation, Dropping, and Temporal Splitting.
    """

    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation component.

        Args:
            config (DataTransformationConfig): Configuration for data transformation.
        """
        self.config = config

    def _clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies cleaning, imputation, and filtering rules.

        Rules:
        - Imputation:
          - airport_fee, congestion_surcharge -> 0 (if null)
          - passenger_count -> 1 (if null or 0)
          - RatecodeID -> 99 (if null)
        - Dropping:
          - store_and_fwd_flag (irrelevant)
        - Filtering:
          - total_amount >= 0 (Remove refunds)
          - total_amount <= 1000 (Remove outliers)
          - 0.5 < trip_distance < 100 (Remove invalid distances)

        Args:
            df (pl.DataFrame): Input DataFrame.

        Returns:
            pl.DataFrame: Cleaned DataFrame.
        """
        logger.info("Starting Data Cleaning (Imputation & Filtering)...")
        initial_rows = df.height

        # 1. Dropping Columns
        if "store_and_fwd_flag" in df.columns:
            df = df.drop("store_and_fwd_flag")
            logger.info("Dropped 'store_and_fwd_flag' column.")

        # 2. Imputation
        df = df.with_columns(
            pl.col("airport_fee").fill_null(0.0),
            pl.col("congestion_surcharge").fill_null(0.0),
            pl.col("passenger_count")
            .fill_null(1)
            .replace(0, 1),  # Replace 0 with 1 as well
            pl.col("RatecodeID").fill_null(99),
        )
        logger.info("Applied basic imputation rules.")

        # 3. Filtering
        # Construct integer masks for filtering
        # Refund Check: total_amount >= 0
        # Outlier Check: total_amount <= 1000
        # Distance Check: 0.5 < trip_distance < 100

        filter_mask = (
            (pl.col("total_amount") >= 0)
            & (pl.col("total_amount") <= 1000)
            & (pl.col("trip_distance") > 0.5)
            & (pl.col("trip_distance") < 100)
        )

        df_cleaned = df.filter(filter_mask)
        final_rows = df_cleaned.height

        dropped_rows = initial_rows - final_rows
        percentage_dropped = (dropped_rows / initial_rows) * 100

        logger.info(
            f"Filtering complete. Dropped {dropped_rows} rows ({percentage_dropped:.2f}%). "
            f"Final shape: {df_cleaned.shape}"
        )

        return df_cleaned

    def initiate_data_transformation(self) -> None:
        """
        Executes the data transformation pipeline:
        1. Loads enriched data.
        2. Cleans and filters data.
        3. Splits data temporally (Jan-May: Train, June: Test).
        4. Saves train and test sets as Parquet.

        Raises:
            CustomException: If transformation fails.
        """
        try:
            logger.info("Loading data for transformation...")
            if not self.config.data_path.exists():
                raise FileNotFoundError(
                    f"Data file not found at: {self.config.data_path}"
                )

            df = pl.read_parquet(self.config.data_path)
            logger.info(f"Loaded raw data: {df.shape}")

            # 1. Clean Data
            df = self._clean_data(df)

            # 2. Temporal Splitting
            # Train: Jan - May 2023
            # Test: June 2023
            # We need to parse tpep_pickup_datetime to extract month

            # Ensure datetime type using precise format (e.g., "08/16/2023 05:24:41 PM")
            df = df.with_columns(
                pl.col("tpep_pickup_datetime").str.strptime(
                    pl.Datetime, "%m/%d/%Y %I:%M:%S %p", strict=False
                )
            )

            # Drop rows where datetime parsing failed (NaTs)
            initial_rows = df.height
            df = df.drop_nulls(subset=["tpep_pickup_datetime"])
            final_rows = df.height
            if initial_rows != final_rows:
                logger.warning(
                    f"Dropped {initial_rows - final_rows} rows due to datetime parsing failures."
                )

            logger.info("Splitting data temporally (Train: Jan-May, Test: Jun)...")

            train_df = df.filter(
                (pl.col("tpep_pickup_datetime").dt.month().is_between(1, 5))
            )
            test_df = df.filter((pl.col("tpep_pickup_datetime").dt.month() == 6))

            logger.info(f"Train Set Shape: {train_df.shape}")
            logger.info(f"Test Set Shape: {test_df.shape}")

            # Simple validation to ensure we have data
            if train_df.is_empty():
                logger.warning(
                    "Train set is empty! Check filtering logic or date range."
                )
            if test_df.is_empty():
                logger.warning(
                    "Test set is empty! Check filtering logic or date range."
                )

            # 3. Save Artifacts
            train_path = self.config.root_dir / "train.parquet"
            test_path = self.config.root_dir / "test.parquet"

            logger.info(f"Saving train data to {train_path}")
            train_df.write_parquet(train_path)

            logger.info(f"Saving test data to {test_path}")
            test_df.write_parquet(test_path)

            logger.info("Data Transformation completed successfully.")

        except Exception as e:
            raise CustomException(e, sys) from e
