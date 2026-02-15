"""
Main entry point for the NYC Taxi Tips Prediction pipeline.

This script allows for manual orchestration of the pipeline stages for debugging
and development purposes.

Usage:
    uv run python main.py
"""

import sys

from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.utils.exception import CustomException
from src.utils.logger import get_logger

STAGE_NAME = "Data Ingestion"
logger = get_logger(__name__, headline="main.py")

try:
    logger.info(f"🚀 Stage: {STAGE_NAME} started 🚀")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f"✅ Stage: {STAGE_NAME} completed ✅")
except Exception as e:
    raise CustomException(e, sys) from e
