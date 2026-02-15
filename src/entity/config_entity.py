"""
Data entity definitions for project configuration.

This module defines the Pydantic models (data contracts) used throughout the pipeline.
Using Pydantic ensures that all configuration values are validated for type and
presence before the pipeline begins execution, preventing runtime attribute errors.
"""

from pathlib import Path
from pydantic import BaseModel


class DataIngestionConfig(BaseModel):
    """
    Configuration for the data ingestion stage.

    Attributes:
        root_dir (Path): Directory where ingestion artifacts are stored.
        source_URL (str): The remote URL to download the raw dataset.
        local_data_file (Path): Path where the downloaded file will be saved.
        unzip_dir (Path): Directory where data will be extracted/prepared.
    """

    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


class DataValidationConfig(BaseModel):
    """
    Configuration for the data validation stage.

    Attributes:
        root_dir (Path): Directory for validation reports and status files.
        STATUS_FILE (str): Path to the text file indicating if validation passed.
        unzip_dir (Path): Path to the data file to be validated.
    """

    root_dir: Path
    STATUS_FILE: str
    unzip_dir: Path


class DataTransformationConfig(BaseModel):
    """
    Configuration for the data transformation stage.

    Attributes:
        root_dir (Path): Directory for transformed feature storage.
        data_path (Path): Path to the input data for transformation.
    """

    root_dir: Path
    data_path: Path


class ModelTrainerConfig(BaseModel):
    """
    Configuration for the model training stage, including hyperparameters.

    Attributes:
        root_dir (Path): Directory where the trained model artifact is saved.
        train_data_path (Path): Path to the training dataset.
        test_data_path (Path): Path to the testing dataset.
        model_name (str): Filename for the saved model (e.g., model.joblib).
        n_estimators (int): Number of boosting rounds for XGBoost.
        max_depth (int): Maximum tree depth for XGBoost.
        learning_rate (float): Step size shrinkage used in update to prevent
            overfitting.
        subsample (float): Fraction of observations to be randomly sampled per tree.
        colsample_bytree (float): Fraction of columns to be randomly sampled per tree.
        objective (str): Learning task objective (e.g., reg:squarederror).
        random_state (int): Seed for reproducibility.
    """

    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    objective: str
    random_state: int


class ModelEvaluationConfig(BaseModel):
    """
    Configuration for the model evaluation stage and MLflow tracking.

    Attributes:
        root_dir (Path): Directory for evaluation metrics and artifacts.
        test_data_path (Path): Path to the test dataset for evaluation.
        model_path (Path): Path to the trained model artifact.
        all_params (dict): Dictionary of all hyperparameters logged to MLflow.
        metric_file_name (Path): Path to the JSON file where metrics are saved.
        mlflow_uri (str): URI for the MLflow tracking server.
    """

    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    mlflow_uri: str
