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
        source_data_path (Path): Path to the raw distilled dataset.
        taxi_zones_path (Path): Path to the taxi zones CSV.
        output_data_path (Path): Path where the enriched Parquet file will be saved.
    """

    root_dir: Path
    source_data_path: Path
    taxi_zones_path: Path
    output_data_path: Path
    all_schema: dict


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
    all_schema: dict


class DataTransformationConfig(BaseModel):
    """
    Configuration for the data transformation stage.

    Attributes:
        root_dir (Path): Directory for transformed feature storage.
        data_path (Path): Path to the input data for transformation.
    """

    root_dir: Path
    data_path: Path


class FeatureEngineeringConfig(BaseModel):
    """
    Configuration for the feature engineering stage.

    Attributes:
        root_dir (Path): Directory where engineered features are saved.
        data_path (Path): Path to the cleaned input data.
    """

    root_dir: Path
    data_path: Path


class ModelTrainerConfig(BaseModel):
    """
    Configuration for the model training stage, including candidates.

    Attributes:
        root_dir (Path): Directory where model artifacts are saved.
        train_data_path (Path): Path to the training dataset.
        model_name (str): Filename for the saved champion model.
        all_params (dict): Dictionary containing hyperparameters for all candidate models.
        mlflow_uri (str): URI for the MLflow tracking server.
    """

    root_dir: Path
    train_data_path: Path
    val_data_path: Path
    model_name: str
    all_params: dict
    mlflow_uri: str
    subsample_fraction: float
    selection_metrics: dict


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
