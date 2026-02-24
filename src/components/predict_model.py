"""
Predict Model component for the NYC Taxi Tips Prediction pipeline.

This module simulates an inference pipeline by loading a batch of data,
generating predictions using the champion model, and saving the results.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

from src.entity.config_entity import ModelEvaluationConfig
from src.utils.common import create_directories
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="Model Inference")


class PredictModel:
    """
    Simulates batch inference using the champion model and fresh data.
    """

    def __init__(self, config: ModelEvaluationConfig):
        """
        Initializes the PredictModel with the evaluation configuration.

        Args:
            config (ModelEvaluationConfig): Configuration for model path and data.
        """
        self.config = config

    def perform_inference(
        self,
        predictions_dir: str = "artifacts/predictions",
        output_filename: str = "inference_results.csv",
    ) -> None:
        """
        Loads the test data (as fresh batch), loads the model, generates predictions,
        and saves the output predictions to a CSV file.
        """
        try:
            # Ensure predictions directory exists
            create_directories([Path(predictions_dir)])

            # Load dataset representing the new batch of data
            # Here, we use the test set as our fresh data for simulation.
            logger.info(
                f"Loading incoming batch data from {self.config.test_data_path}"
            )
            batch_data = pd.read_parquet(self.config.test_data_path)

            # Extract features (drop tip_amount as it would not be available in real inference)
            if "tip_amount" in batch_data.columns:
                X_batch = batch_data.drop(["tip_amount"], axis=1)
            else:
                X_batch = batch_data.copy()

            # Process features for prediction (keep only numeric)
            X_batch_processed = X_batch.select_dtypes(include=["number"])

            # Load the model directly from local system
            # Note: Production systems might load from MLflow Model Registry like:
            # import mlflow.pyfunc
            # model_name = "nyc-taxi-tips-champion"
            # model_version = "latest"
            # model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
            model = joblib.load(self.config.model_path)
            logger.info(f"Loaded model from {self.config.model_path}")

            # Predict
            predictions = model.predict(X_batch_processed)

            # Persist results
            # We join predictions with an identifier like VendorID or tpep_pickup_datetime
            results_df = pd.DataFrame()

            # Since VendorID might be categorical/encoded, we include what's available
            # Let's concatenate the predictions directly
            # Fallback to index if no distinct IDs are present to pass along
            results_df["predicted_tip"] = predictions
            if "VendorID" in batch_data.columns:
                results_df["VendorID"] = batch_data["VendorID"].values

            output_path = Path(predictions_dir) / output_filename
            results_df.to_csv(output_path, index=False)

            logger.info(f"Inference complete! Results saved to {output_path}")

        except Exception as e:
            raise CustomException(e, sys) from e
