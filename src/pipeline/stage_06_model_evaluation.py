"""
Stage 06: Model Evaluation

This module orchestrates the model evaluation phase of the pipeline.
It handles evaluating the registered/champion model against the hold-out test set
and running an inference simulation.
"""

import sys
from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation
from src.components.predict_model import PredictModel
from src.utils.logger import get_logger, log_spacer
from src.utils.exception import CustomException

logger = get_logger(__name__, headline="Stage: Model Evaluation")

STAGE_NAME = "Model Evaluation"


class ModelEvaluationPipeline:
    """
    Pipeline for executing Model Evaluation and Inference Simulation.
    """

    def __init__(self):
        """Initializes the ModelEvaluation Pipeline."""
        pass

    def main(self):
        """
        Executes the main functionality of the Model Evaluation pipeline stage.
        """
        try:
            config = ConfigurationManager()
            model_eval_config = config.get_model_evaluation_config()

            # Step 1: Evaluate model against Test Dataset
            logger.info("Starting Model Evaluation on Test set...")
            model_eval = ModelEvaluation(config=model_eval_config)
            model_eval.evaluate()

            log_spacer()

            # Step 2: Inference simulation
            logger.info("Running Batch Inference Simulation...")
            predict_model = PredictModel(config=model_eval_config)
            predict_model.perform_inference()

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        pipeline = ModelEvaluationPipeline()
        pipeline.main()
        logger.info(
            f">>>>>> Stage: {STAGE_NAME} completed! <<<<<<\n\n======================="
        )
    except Exception as e:
        logger.exception("Exception occurred during execution:")
        raise CustomException(e, sys) from e
