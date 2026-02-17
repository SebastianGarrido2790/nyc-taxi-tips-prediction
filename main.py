"""
Main entry point for the NYC Taxi Tips Prediction pipeline.

This script allows for manual orchestration of the pipeline stages for debugging
and development purposes.

Usage:
    uv run python main.py
"""

import sys
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.pipeline.stage_04_feature_engineering import FeatureEngineeringTrainingPipeline
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="main.py")

try:
    STAGE_NAME = "Data Ingestion"
    logger.info(f"🚀 Stage: {STAGE_NAME} started 🚀")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f"✅ Stage: {STAGE_NAME} completed ✅")

    STAGE_NAME = "Data Validation"
    logger.info(f"🚀 Stage: {STAGE_NAME} started 🚀")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f"✅ Stage: {STAGE_NAME} completed ✅")

    STAGE_NAME = "Data Transformation"
    logger.info(f"🚀 Stage: {STAGE_NAME} started 🚀")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f"✅ Stage: {STAGE_NAME} completed ✅")

    STAGE_NAME = "Feature Engineering"
    logger.info(f"🚀 Stage: {STAGE_NAME} started 🚀")
    feature_engineering = FeatureEngineeringTrainingPipeline()
    feature_engineering.main()
    logger.info(f"✅ Stage: {STAGE_NAME} completed ✅")


except Exception as e:
    raise CustomException(e, sys) from e
