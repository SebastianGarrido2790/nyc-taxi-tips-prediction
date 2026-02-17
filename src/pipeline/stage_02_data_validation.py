"""
Data Validation Pipeline Stage (Conductor).

This module orchestrates the Data Validation process.
"""

import sys
from src.components.data_validation import DataValidation
from src.config.configuration import ConfigurationManager
from src.utils.exception import CustomException
from src.utils.logger import get_logger

STAGE_NAME = "Data Validation"
logger = get_logger(__name__)


class DataValidationTrainingPipeline:
    """
    Orchestrates the data validation stage.
    """

    def __init__(self):
        """
        Initializes the pipeline stage.
        """
        self.config = ConfigurationManager()

    def main(self):
        """
        Executes the main orchestration logic.
        """
        try:
            config = self.config.get_data_validation_config()
            data_validation = DataValidation(config=config)
            data_validation.validate_all_columns()

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        logger.info(f"🚀 Stage: {STAGE_NAME} started 🚀")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f"✅ Stage: {STAGE_NAME} completed ✅")
    except Exception as e:
        logger.exception(e)
        raise e
