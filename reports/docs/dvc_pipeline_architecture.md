# DVC Pipeline Architecture Report

## 1. Overview
The **NYC Taxi Tips Prediction** pipeline is orchestrated by Data Version Control (DVC).
It defines a Directed Acyclic Graph (DAG) ensuring that:
1.  **Reproducibility**: Every artifact is tied to a specific Git commit and code version.
2.  **Efficiency**: Stages are re-run `ONLY` if their dependencies (code or data) change.
3.  **Traceability**: The lineage of every model is fully auditable.

## 2. Pipeline DAG
The pipeline consists of 4 primary stages (currently implemented):

```mermaid
flowchart TD
    subgraph Raw_Data [External Data]
        A[Raw Text Data]
        B[Taxi Zones CSV]
    end

    subgraph Stage_01 [1. Data Ingestion]
        A & B --> C((Run Ingestion))
        C --> D[Enriched Trip Data Parquet]
    end

    subgraph Stage_02 [2. Data Validation]
        D --> E((Run Validation))
        E --> F[Validation Status Report]
    end

    subgraph Stage_03 [3. Data Transformation]
        D --> G((Run Transformation))
        G --> H[Cleaned Trip Data Parquet]
    end

    subgraph Stage_04 [4. Feature Engineering]
        H --> I((Run Feature Eng))
        I --> J[Train Set]
        I --> K[Validation Set]
        I --> L[Test Set]
    end

    style C fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
```

## 3. Detailed Stage Breakdown

### 3.1 Data Ingestion (`stage_01_data_ingestion`)
*   **Input**: Raw Distilled Trip Records (`.txt`), Taxi Zones (`.csv`).
*   **Logic**: Joins trip data with zone names, handles basic schema enforcement.
*   **Output**: `artifacts/data_ingestion/enriched_trip_data.parquet`.
*   **Dependency**: `src/components/data_ingestion.py`.

### 3.2 Data Validation (`stage_02_data_validation`)
*   **Input**: Enriched Data (`.parquet`).
*   **Logic**: Validates schema against `config/schema.yaml`. Checks for critical column existence.
*   **Output**: `artifacts/data_validation/status.txt`.
*   **Dependency**: `src/components/data_validation.py`.
*   **Note**: This stage is a "Gatekeeper". If it fails, the pipeline halts.

### 3.3 Data Transformation (`stage_03_data_transformation`)
*   **Input**: Enriched Data (`.parquet`).
*   **Logic**:
    *   **Imputation**: Fills nulls in `airport_fee`, `congestion_surcharge`, `passenger_count`.
    *   **Filtering**: Prunes negative fares, invalid distances, and outliers.
*   **Output**: `artifacts/data_transformation/cleaned_trip_data.parquet`.
*   **Dependency**: `src/components/data_transformation.py`.

### 3.4 Feature Engineering (`stage_04_feature_engineering`)
*   **Input**: Cleaned Data (`.parquet`).
*   **Logic**:
    *   **Cyclical Encoding**: Transforms Hour/Day/Month into Sin/Cos pairs.
    *   **Temporal Splitting**: Splits data by month (Jan-Aug/Sept-Oct/Nov-Dec).
*   **Outputs**:
    *   `artifacts/feature_engineering/train.parquet`
    *   `artifacts/feature_engineering/val.parquet`
    *   `artifacts/feature_engineering/test.parquet`
*   **Dependency**: `src/components/feature_engineering.py`.

## 4. Execution
To reproduce the entire pipeline (or only what changed):
```bash
uv run dvc repro
```

To visualize the DAG in terminal:
```bash
uv run dvc dag
```
