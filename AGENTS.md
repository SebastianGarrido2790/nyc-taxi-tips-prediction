# MLOps Development Guidelines

This document serves as the "Constitution" for our project, ensuring that any developer (human or AI) maintains the high MLOps standards we have set.

---

## 1. Project Philosophy
**"We build systems, not just models."**
This is a production-grade software engineering project. We move away from "notebook thinking" (hidden state, manual execution) to "pipeline thinking" (explicit state, automated orchestration).

* **Reproducibility is King:** If I cannot run `dvc repro` and get the exact same result bit-for-bit, the feature is not complete.
* **Configuration over Code:** Hardcoded paths and "magic numbers" are strictly forbidden. All variables live in `config/`.
* **Fail Fast:** We validate data schemas and config integrity at the start of every pipeline.

---

## 2. Technology Stack & Tools
We stick to "Boring Technology" to ensure reliability.

* **Dependency Management:** `uv` (Fast, deterministic).
    * Run `uv sync` to install.
    * Run `uv add <package>` to install new libraries.
    * **Never** use `pip install` directly; always update `pyproject.toml`.
* **Data Versioning:** `DVC` (Data Version Control).
    * Large files (`data/`, `models/`) are tracked by DVC.
    * Only `.dvc` placeholder files are committed to Git.
* **Orchestration:** `dvc.yaml`.
    * This file defines the Directed Acyclic Graph (DAG) of our pipeline.
* **Processing:** `Pandas` (with strict type casting) or `Polars` (if memory constraints arise).

---

## 3. Coding Standards

### 3.1 Style & Structure
* **Type Hinting:** Mandatory for all function signatures.
    * *Bad:* `def process(df):`
    * *Good:* `def process_data(df: pd.DataFrame) -> pd.DataFrame:`
* **Docstrings:** Google Style is required. Explain *Arguments*, *Returns*, and *Raises*.
* **Logging:** Never use `print()`. Use the custom logger:
    ```python
    from src.utils.logger import logger
    logger.info("Ingesting data from source...")
    ```
* **Error Handling:** Use custom exceptions from `src/utils/exception.py` to wrap critical failures with context.

### 3.2 The "No Notebooks in Prod" Rule
* Jupyter Notebooks (`notebooks/`) are for **exploration only**.
* Once a concept is proven, it must be refactored into modular Python functions in `src/components/`.
* Do not import code *from* `.ipynb` files.

---

## 4. The FTI (Feature-Training-Inference) Architecture

### 4.1 Feature Pipeline (`src/features/`)
* **Objective:** Transform raw, messy logs into a pristine Feature Store.
* **Strict Rules:**
    * **Memory Safety:** The dataset has 5M rows. Use chunking or `polars` if standard pandas operations spike memory.
    * **Imputation:** * `Airport_fee` & `congestion_surcharge` $\to$ 0.
        * `passenger_count` $\to$ 1 (Default).
        * `RatecodeID` $\to$ 99 (Unknown).
    * **Filtering:** * Drop refunds (negative `total_amount`).
        * Trips must be: `0.5 miles < distance < 100 miles`.
        * Cap `total_amount` at $1,000.
    * **Output:** Save as compressed Parquet (Snappy).

### 4.2 Training Pipeline (`src/models/`)
* **Objective:** Train an XGBoost Regressor without data leakage.
* **Strict Rules:**
    * **Temporal Splitting:** **CRITICAL.** Do NOT use `train_test_split(shuffle=True)`.
        * *Strategy:* Train on Jan-May, Validate on June.
        * *Why:* To prevent "Look-ahead Bias."
    * **Registry:** Models must be saved to `models/model.bst` (or `.json`) and versioned via DVC.
    * **Baseline:** Always compute MAE of a "dummy" mean-prediction model first.

### 4.3 Inference Pipeline (`src/models/predict_model.py`)
* **Objective:** Batch inference on new data.
* **Strict Rules:**
    * Must handle cases where new data might be missing non-critical columns (graceful degradation).
    * Output results to `artifacts/predictions/` with a timestamp.

---

## 5. Configuration Management
We decouple code from configuration.

* **`config.yaml`:** Stores file paths (`raw_data_path`, `model_path`).
* **`params.yaml`:** Stores hyperparameters (`learning_rate`, `max_depth`).
* **`schema.yaml`:** Stores data schemas and validation rules (e.g., data types, required columns).
* **Access Pattern:**
    Do not read YAML files directly in components. Use the Configuration Manager:
    ```python
    # src/config/configuration.py
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    ```

---

## 6. Git & DVC Workflow
1.  **Modify Code:** Edit `src/`.
2.  **Run Pipeline:** `dvc repro` (Re-runs only changed stages).
3.  **Commit Code:** `git add .` (Includes `dvc.lock` and `dvc.yaml`).
4.  **Commit Data:** `dvc push` (If using remote storage) or ensure local cache is updated.

**Files to IGNORE in Git:**
* `data/*` (except `.dvc` files)
* `models/*` (except `.dvc` files)
* `__pycache__`
* `.ipynb_checkpoints`
* `artifacts/*` (except `.dvc` files)
* `logs/*` (except `.dvc` files)
* `venv/*` (except `.dvc` files)
* `ruff_cache/*` (except `.dvc` files)
* `.env`

---

## 7. Testing & Validation Strategy
**"Code compiles $\neq$ System works."**
In MLOps, we test three things: The Code, The Data, and The Model.

### 7.1 Unit Testing (`tests/unit/`)
* **Scope:** Test individual functions in `src/`, not the entire pipeline.
* **Philosophy:** "Test logic, not libraries." Do not write a test to prove `pandas.read_csv` works. Test that *your* cleaning function actually drops negative values.
* **Key Tests:**
    * `test_clean_data`: Pass a dummy dataframe with negative fares and `NaNs`. Assert they are removed/filled correctly.
    * `test_feature_engineering`: Pass a known date. Assert the cyclical features (sin/cos or integer representation) are calculated correctly.
* **Tool:** `pytest`.

### 7.2 Integration Testing (`tests/integration/`)
* **Scope:** Test the interactions between pipeline stages.
* **Method:** Run the full `dvc repro` DAG on a **tiny subsample** (e.g., 100 rows) of the data.
* **Success Criteria:** The pipeline must run from Ingestion $\to$ Inference without crashing and produce a valid `predictions.csv`.

### 7.3 Data Validation (The "Guardrails")
Before any data enters the Feature Store or Model, it must pass these gates:
* **Schema Validation:** Ensure raw data matches the expected schema (Column names, Data types).
* **Null Checks:** Critical columns (e.g., `trip_distance`) must not contain `NaN` after cleaning.
* **Range Checks:**
    * `passenger_count`: Must be $\ge$ 0.
    * `trip_distance`: Must be $>$ 0.
    * `total_amount`: Must be $>$ 0.
* **Tool:** Simple assertions in `src/features/build_features.py` or `pandera` decorators.

### 7.4 Model Validation (The "Performance Gate")
Just because a model trains doesn't mean it's good.
* **Baseline Check:** The trained model's RMSE *must* be lower than the Baseline Model (Mean Absolute Error). If not, the training pipeline should **fail**.
* **Shape Check:** Ensure the output vector length matches the input rows ($N_{pred} == N_{input}$).
* **Overfitting Check:** If Training Accuracy $\gg$ Validation Accuracy, raise a warning.
