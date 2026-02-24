"""
Pipeline stage for model training and benchmarking.

This stage loads the processed data, trains multiple candidate models,
compares them using MLflow, and saves the best model.
"""

import sys
from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer
from src.utils.exception import CustomException
from src.utils.logger import get_logger

STAGE_NAME = "Model Training"
logger = get_logger(__name__)


class ModelTrainerPipeline:
    """
    Pipeline stage for Model Training.
    """

    def __init__(self):
        """
        Initializes the pipeline stage.
        """
        self.config = ConfigurationManager()

    def main(self):
        """
        Executes the model training stage.
        """
        try:
            train_config = self.config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=train_config)
            model_trainer.train_and_register()

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        logger.info(f"Stage {STAGE_NAME} started")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f"Stage {STAGE_NAME} completed")
    except Exception as e:
        logger.exception(e)
        raise e
