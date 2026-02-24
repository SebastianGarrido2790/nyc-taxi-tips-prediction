# NYC Taxi Tip Prediction System

A production-grade MLOps system to predict NYC Yellow Taxi tip amounts using the **FTI (Feature, Training, Inference)** pattern.

##  Overview

This system decouples data engineering, model development, and model serving into distinct, independently operational pipelines:

1.  **Feature Pipeline**: Transforms raw taxi logs into a versioned Feature Store.
2.  **Training Pipeline**: Consumes features to train an XGBoost regressor with temporal splitting.
3.  **Inference Pipeline**: Performs batch predictions on new data.

## 🛠 Tech Stack

- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **Orchestration**: [DVC](https://dvc.org/)
- **Data**: Pandas / Polars / Parquet
- **Model**: XGBoost
- **Tracking**: MLflow
- **Validation**: Pydantic

## 🔨 Setup

```bash
# Install dependencies
uv sync --all-extras

# Initialize DVC (if not done)
uv run dvc pull
```

## 📈 Usage

```bash
# Run the full pipeline
uv run dvc repro
```
