# NYC Taxi Tips Predictor: App Architecture & Functionality

## 1. Executive Summary

The `app.py` Streamlit application serves as the interactive frontend (Phase 5) of the NYC Taxi Tip Prediction System. Operating strictly as a UI layer, it bridges the complex machine learning pipelines (Feature, Training, Inference - FTI) and business stakeholders (like Fleet Managers). The app allows users to visually explore the model's performance, understand what drives tip amounts, and simulate real-time rides, delegating all heavy machine learning computations to a robust FastAPI microservice backend.

## 2. Architecture & The FTI Pattern Integration

Following strict **Agentic MLOps Rules**, the system perfectly executes the **FTI (Feature, Training, Inference)** architecture by decoupling the Model Serving pipeline completely from the frontend UI interface.

### The FastAPI Serving Layer (`src/api/predict_api.py`)
Instead of running prediction computations on the fly in the frontend, the Streamlit app acts as a client to an independent FastAPI microservice.
*   **Model Registry & Lifespan:** The FastAPI server utilizes `lifespan` event handlers to securely load the trained champion model (`model.joblib`) from `artifacts/model_trainer/` into a global memory dictionary precisely once upon startup. This guarantees rapid, sub-millisecond inference times on subsequent API calls without costly repetitive I/O operations.
*   **Strict Pydantic Contracts:** All communication between the Streamlit UI and the FastAPI backend is validated via strict Pydantic schemas (`PredictRequest`, `PredictResponse` in `src/entity/api_entity.py`), enforcing rigid type safety before any code touches the ML model.
*   **Custom Exception Handling:** The backend throws standard HTTP `503 Service Unavailable` or `500 Server Errors` when the model registry fails, preventing silent failures and displaying clean exception traces.

### Artifact Consumption
To ensure the dashboard loads quickly and interactively, it utilizes Streamlit's caching mechanisms (`@st.cache_data`) to parse pre-computed outputs from the orchestrated DVC pipelines:
*   **Model Evaluation Metrics:** Loads `artifacts/model_evaluation/metrics.json` to extract performance markers like Test MAE, MSE, and R².
*   **Batch Predictions:** Loads `artifacts/predictions/inference_results.csv` to display the offline batch inference outputs.
*   **Training Parameters:** Loads `config/params.yaml` to extract the metric weights used during automated champion selection.

## 3. Orchestrator Launchpad (`launch_app.bat`)
To ensure smooth boots of this decoupled architecture, the system relies on a central windows batch orchestrator (`launch_app.bat`).
Mimicking enterprise Agentic System tooling, this script guarantees reliable startups via sequential checks:
1.  **Dependency Synchronization:** Runs `uv sync` to ensure local environments match the `pyproject.toml` lockfile exactly.
2.  **API Background Execution:** Spins up the Uvicorn/FastAPI server (`src.api.predict_api:app --reload`) inside a separate explicitly minimized Windows terminal, maintaining visual cleanliness.
3.  **Frontend Execution:** Safely sleeps for a 5-second warmup period before starting the Streamlit interface on `localhost:8501`.

## 4. User Interface & Core Functionality

The dashboard features a persistent sidebar for navigation and is divided into two primary pages. 

### Page 1: Dashboard & Evaluation
Dedicated to high-level analytics, model transparency, and offline batch prediction reviews.

*   **Test MAE (Mean Absolute Error):** Represents the average absolute difference between the predicted tip and the actual tip in dollars. A lower value indicates higher precision and reliability in the model's dollar-amount predictions.
*   **Test R² (Variance Explained):** Indicates the proportion of the variance in the tip amount that is predictable from the trip characteristics. A value closer to 1.0 (or > 0.90) signifies an excellently performing model covering edge cases effectively.
*   **Test MSE (Mean Squared Error):** Represents the average of the squares of the errors. It heavily penalizes larger errors, ensuring the model deals effectively with anomalies and outliers, minimizing extreme deviations.
*   **Champion Selection Weights:** An informative banner displaying the exact metric weights (e.g., MAE, MSE, R2) used by the automated backend pipeline to evaluate, rank, and select this specific champion model from all candidate models.

*   **Model Version Context:** The UI continuously pings the FastAPI `/health` endpoint to confirm server availability and dynamically extract and display the name of the active ML model loaded in the registry (e.g., *serving (Ridge model) predictions*).
*   **Dynamic Sliders & Interactive Sampling:** To maintain high performance without crashing the browser, users are equipped with interactive sliders mapped to dataframe sampling logic:
    *   **Feature Importance Chart:** Users dial the slider to configure the exact number of top drivers they wish to investigate (e.g., Top 3 to Top 20 features). This visual requests data via a dedicated `/feature-importance` FastAPI endpoint.
    *   **Distribution Histogram:** Users can adjust the sample size (up to 10,000) of the latest batch inferences rendered in the distribution plot.
    *   **Inferences Ledger:** A slider adjusts the random sample generated from the offline dataset for granular visual review. The ledger seamlessly maps Vendor IDs 1 and 2 to human-readable strings ("Creative Mobile Technologies", "VeriFone Inc.").
*   **Test Metrics & Weights:** Read-only metric cards visually report Test MAE, MSE, and R², alongside the specific weight distribution that configured the algorithm.

### Page 2: Interactive Prediction
This page allows users to simulate taxi rides and query the champion model's REST API in real-time. It is designed heavily around a frictionless, dynamically editable batch-input experience.

*   **Simulate a Ride (Header & Controls):** 
    *   The page introduces the predictive capabilities, explaining that users can either predict a single ride or add multiple rows to process batch inferences.
    *   A prominent **"🔄 Reset Details"** button gracefully flushes the Streamlit session state, immediately clearing all user edits and resetting the interface to its clean, default single-row state.

*   **Interactive Trip Details (`st.data_editor`):** Instead of using a long, vertical form of static input fields, users manage ride features via an interactive, horizontal spreadsheet-like table.
    *   **Input Features:** The table exposes ten critical trip characteristics for editing, including Distance (miles), Total Fare ($), Passengers, Rate Code, and fine-grained charges like Airport Fee and Congestion Surcharge, alongside the temporal data (Hour, Day, Month).
    *   **Batch Capability:** Users can swiftly append new rows via the `+` icon at the bottom of the table to simulate multiple different scenarios simultaneously.
    *   **Rich Column Configuration:** Strict constraints (min, max, step, format) are enforced natively via Streamlit's `column_config` API, ensuring that inputs are sanitized before they ever reach the backend Pydantic validation schemas. Helpful tooltips describe each column upon hover.
    *   **Native Deletion:** Invalid rows are effortlessly deleted by selecting the built-in left-hand checkboxes and clicking the native toolbar's trash (`🗑️`) icon.

*   **Execution & API Routing:** Upon clicking the **"Predict Tip(s) 🔮"** button, the UI serializes the entire dataframe into a JSON payload and executes a single HTTP `requests.post()` call to the `/predict` FastAPI endpoint. This delegates all data transformations (like cyclical time encoding) and model processing to the backend microservice.

*   **Comprehensive Output Architecture:** Once the FastAPI server returns the prediction array, the UI dynamically constructs a multi-layered results view:
    *   **Success Banner:** A clear visual indicator (e.g., `Output For 2 Ride(s)`) confirms successful execution.
    *   **Batch Prediction Averages:** High-level analytical `st.metric` cards automatically calculate and visualize the arithmetic mean of both the expected dollar amount and the Tip Percentage across the entire submitted batch of rides.
    *   **Stylized Output Frame:** The individual predictions are appended to the inputs and displayed in a finalized table. This dataframe utilizes Pandas `Styler` object properties to apply a bold gold background specifically to the prediction outcome columns (`Expected Tip` and `Tip %`), creating quick visual isolation of the model's insights. The dataframe indices are hidden for a cleaner, dashboard-grade user experience.

## 5. Design & Aesthetics

Following strict guidelines for high-quality, professional tooling, the UI employs **custom CSS** to inject a modern, dark-mode aesthetic. 

*   Uses a tailored `#0E1117` base with `#FAFAFA` text for high contrast.
*   Incorporates subtle accent colors like gold (`#FFD700`) for headers and buttons, alongside green (`#00FF7F`) for positive metrics, bypassing Streamlit's default generic look to deliver a premium user experience.
