## Project Executive Summary

### 1. Executive Overview

This project involves the end-to-end development of a production-grade Machine Learning system designed to predict tip amounts for NYC Yellow Taxi rides. By analyzing a dataset of **5 million records**, the system identifies key drivers of tipping behavior (e.g., trip duration, location, pickup time) to provide actionable revenue optimization insights for fleet managers.

Unlike standard exploratory projects, this system implements the **FTI (Feature, Training, Inference) Pattern**, strictly decoupling data engineering from model development. This ensures the architecture is scalable, reproducible, and ready for batch production environments.

---

### 2. System Architecture & Design

The solution is architected into three isolated pipelines to ensure modularity and prevent technical debt.

* **1. Feature Pipeline (Data Engineering)**
    * **Objective:** Transform raw, messy logs into a pristine "Feature Store."
    * **Key Logic:**
        * **Ingestion:** Handles 5M+ rows using chunking strategies to manage memory.
        * **Cleaning:** Implements domain-specific rules: filling `Airport_fee` with 0, setting `passenger_count` to 1 (default), and filtering anomalies (negative fares, trips > 100 miles).
        * **Engineering:** Converts timestamps into cyclical temporal features (Hour, Day, Month) to capture seasonality.
    * **Output:** Versioned Parquet files stored in `artifacts\data_transformation`.

* **2. Training Pipeline (Model Development)**
    * **Objective:** Produce a high-performance predictive model without data leakage.
    * **Key Logic:**
        * **Splitting:** Utilizes **Temporal Splitting** (e.g., Train: Jan-May, Test: June) rather than random shuffling to respect the time-series nature of the data.
        * **Algorithm:** Trains an **XGBoost Regressor**, optimized for tabular data performance.
    * **Evaluation:** Benchmarks against a "Mean Absolute Error" (MAE) baseline and optimizes for Mean Squared Error (MSE).
    * **Output:** A versioned model artifact saved to the **Model Registry**.

* **3. Inference Pipeline (Model Serving)**
    * **Objective:** Generate predictions on new data batches.
    * **Key Logic:** Simulates a production batch job by loading recent trip data, fetching the latest model artifact, and generating tip predictions.
    * **Output:** A predictions dataset stored in `artifacts/predictions/` for downstream analytics.

---

### 3. Task List per Pipeline

Here is the step-by-step implementation plan, mapping to the "FTI" (Feature, Training, Inference) pattern.

#### 1. The Prep Station: Feature Pipeline (Data Engineering)

**Goal:** Turn raw, messy logs into a clean, query-ready Feature Store.

* **1.1 Setup & Config:** Initialize `dvc` and configure `src/config/configuration.py` to manage paths for `raw` vs `processed` data.
* **1.2 Ingestion (The Heavy Lift):** Write a chunking mechanism (or use Polars) in `src\components\feature_engineering.py` to load the 5M row CSV without crashing RAM.
* **1.3 Cleaning - Imputation:** Implement logic to fill `NaN`s:
    * `Airport_fee` & `congestion_surcharge`  0.
    * `passenger_count`  1.
    * `RatecodeID`  99 (Unknown).
* **1.4 Cleaning - Filtering:**
    * Remove "Refunds" (negative `total_amount`).
    * Filter unrealistic trips: `0.5 miles < distance < 100 miles`.
    * Cap outliers: `total_amount < $1000`.
* **1.5 Feature Engineering:** Transform `tpep_pickup_datetime` into cyclical features:
    * `pickup_hour` (0-23)
    * `day_of_week` (0-6)
    * `trip_duration_minutes` (calculated from dropoff - pickup).
* **1.6 Storage:** Save the final dataframe as a compressed **Parquet** file in `artifacts\data_transformation`.

#### 2. The Chef's Lab: Training Pipeline (Model Development)

**Goal:** Create a mathematical representation of tipping behavior.

* **2.1 Temporal Splitting:** In `src/components/train_model.py`, with the time-based split implemented (e.g., Train on Jan-May data, Validate on June data). **Strictly avoid random shuffling.**
* **2.2 Baseline creation:** Calculate the Mean Absolute Error (MAE) of a "dumb" model (always predicting the mean tip) to set a performance floor.
* **2.3 Model Training:** Initialize and train an **XGBoost Regressor**  on the training set.
* **2.4 Hyperparameter Tuning:** Configure `params.yaml` to control `max_depth`, `learning_rate`, and `n_estimators`.
* **2.5 Evaluation:** Calculate RMSE and R-squared on the validation set.
* **2.6 Interpretability:** Generate and save a "Feature Importance" plot to explain *why* the model makes certain predictions.
* **2.7 Model Registry:** Save the trained model object (e.g., `xgb_model.json`) to the `artifacts/model_registry/` directory.

#### 3. The Dinner Rush: Inference Pipeline (Serving)

**Goal:** Serve predictions on new data using the frozen model.

* **3.1 Batch Loader:** Create `src/components/predict_model.py` to load a "fresh" batch of data (simulating the latest taxi rides).
* **3.2 Model Loading:** Load the saved XGBoost artifact from the `models/` directory.
* **3.3 Prediction:** Run the model on the fresh data to generate `predicted_tip`.
* **3.4 Persistence:** Join predictions with `VendorID` and save the results to `artifacts/predictions/inference_results.csv` for analysis.

---

### 4. Technology Stack

The project utilizes a "Boring Technology" stack—prioritizing reliability and industry standards over experimental tools.

| Component | Tool | Justification |
| --- | --- | --- |
| **Language** | **Python 3.10+** | Standard for ML and Data Engineering. |
| **Dependency Mgmt** | **UV** | High-performance replacement for Pip/Poetry; ensures deterministic environments. |
| **Orchestration** | **DVC (Data Version Control)** | Manages the pipeline DAG and versions large datasets (5M rows) that Git cannot handle. |
| **Processing** | **Pandas / Polars** | Robust tabular data manipulation. Polars is reserved as a fallback for high-memory operations. |
| **Modeling** | **XGBoost** | The industry standard for gradient boosting on structured data. |

---

### 5. Data Strategy & Governance

* **Dataset:** 2023 NYC Yellow Taxi Trip Data (~5,000,000 rows).
* **Quality Gates:**
    * **Imputation:** Strict handling of `NaN` values (e.g., `RatecodeID` -> 99).
    * **Outlier Removal:** Caps `total_amount` at $1,000 to prevent skewing from data entry errors.
    * **Leakage Prevention:** Strict separation of training and testing periods ensures the model never "sees" the future.

---

### 6. Expected Deliverables

1. **Source Code Repository:** A clean, modular codebase structured according to the `src/` layout (components, pipeline, entity).
2. **Reproducible Pipeline:** A `dvc.yaml` configuration allowing any engineer to reproduce results via `dvc repro`.
3. **Model Artifact:** A trained, versioned XGBoost model.
4. **Insight Report:** A "Feature Importance" analysis ranking the factors that most strongly influence tipping behavior.
5. **UI Dashboard:** A Streamlit dashboard to visualize the data and model predictions.

---

### 7. Risk Assessment

* **Memory Constraints:** Processing 5M rows may trigger OOM errors. *Mitigation:* Implementation of chunked processing or switching to Polars.
* **Concept Drift:** Tipping patterns may change seasonally. *Mitigation:* The temporal splitting strategy specifically validates the model's ability to generalize across months.

---

### 8. Project Analogy: The "Professional Kitchen"

To understand this MLOps system, imagine running a high-end restaurant kitchen.

* **The Raw Ingredients (Data):** The 5 million raw taxi logs are the vegetables and meats delivered in bulk—messy, unwashed, and mixed with occasional spoiled items (anomalies).
* **The Prep Station (Feature Pipeline):** This is where the *Sous Chefs* work. They wash, peel, chop, and organize ingredients into "mise en place" containers (The Feature Store). They don't cook; they just ensure everything is consistent and ready for the main line.
* **The Chef de Cuisine (Training Pipeline):** This is the Head Chef experimenting with recipes. They take the prepped ingredients, adjust heat levels and spices (hyperparameters), and taste-test until they perfect the "Signature Dish" (The Model). Once perfected, the recipe is written down and locked in a safe (Model Registry).
* **The Line Cooks (Inference Pipeline):** During the dinner rush (Production), these cooks don't invent new recipes. They grab the "Signature Dish" recipe and the prepped ingredients to serve plates to customers (Predictions) as fast as possible.
