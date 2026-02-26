"""
NYC Taxi Tips Predictor & Analyst Dashboard.

This Streamlit application serves as the interactive frontend (Phase 5) for the NYC Taxi Tip
Prediction System. It loads evaluation metrics, batch predictions, and the trained Champion Model
from the local artifacts to visualize performance and allow real-time interactive predictions.

Usage:
    1. Ensure the FastAPI backend is running:
        uv run uvicorn src.api.predict_api:app --reload (Backend API)
    2. Run the Streamlit app:
        uv run streamlit run app.py (Frontend UI)
"""

import os
import streamlit as st
import pandas as pd
import json
import requests
import plotly.express as px
import yaml
from pathlib import Path

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Set page config
st.set_page_config(
    page_title="NYC Taxi Tips Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for aesthetics (Antigravity Rule 3: Visual Excellence)
st.markdown(
    """
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #FFD700;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric boxes */
    div[data-testid="stMetricValue"] {
        color: #00FF7F;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1c24;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #FFD700;
        color: #000;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FFC125;
        transform: scale(1.02);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Constants for paths
ARTIFACTS_DIR = Path("artifacts")
METRICS_PATH = ARTIFACTS_DIR / "model_evaluation" / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions" / "inference_results.csv"
MODEL_DIR = ARTIFACTS_DIR / "model_trainer"
PARAMS_PATH = Path("config") / "params.yaml"


# --- Cached Data Loading Functions ---
@st.cache_data
def load_metrics():
    """
    Loads the evaluation metrics (JSON) from the local artifacts directory.
    Returns None if the file does not exist.
    """
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_predictions():
    """
    Loads the simulated batch inference results (CSV) from the local artifacts directory.
    Returns None if the file does not exist.
    """
    if not PREDICTIONS_PATH.exists():
        return None
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data(ttl=60)
def check_api_health():
    """Checks if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return True, data.get("model_version", "unknown")
        return False, "unknown"
    except requests.exceptions.RequestException:
        return False, "unknown"


@st.cache_data
def load_params():
    """
    Loads the parameters (YAML) to extract selection metrics.
    Returns None if the file does not exist.
    """
    if not PARAMS_PATH.exists():
        return None
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


# --- Load Artifacts ---
metrics = load_metrics()
predictions_df = load_predictions()
api_is_healthy, model_name = check_api_health()
params = load_params()

# --- Main App Layout ---
st.title("🚕 NYC Taxi Tips Predictor & Analyst")
st.markdown(
    f"*A production-ready FTI architecture serving ({model_name} model) predictions via FastAPI.*"
)

if metrics is None or predictions_df is None:
    st.error(
        "⚠️ Initial artifacts not found. Please run the DVC pipeline (`uv run dvc repro`) first."
    )
    st.stop()

if not api_is_healthy:
    st.warning(
        "⚠️ FastAPI Backend is not reachable. Ensure the server is running (`uv run uvicorn src.api.predict_api:app`). Real-time predictions will fail."
    )

# --- Sidebar ---
st.sidebar.image(
    "reports/figures/nyc_taxi_logo.jpg",
    width=150,
)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate", ["📊 Dashboard & Evaluation", "⚡ Interactive Prediction"]
)

# --- PAGE 1: Dashboard & Evaluation ---
if page == "📊 Dashboard & Evaluation":
    st.header("Model Performance & Batch Inferences")

    # Metrics Row
    col1, col2, col3 = st.columns(3)
    val_mae = metrics.get("test_mae", 0.0)
    val_mse = metrics.get("test_mse", 0.0)
    val_r2 = metrics.get("test_r2", 0.0)

    col1.metric(
        "Test MAE (Mean Absolute Error)",
        f"${val_mae:.2f}",
        delta="- Error indicates high precision",
        delta_color="normal",
    )
    col2.metric(
        "Test R² (Variance Explained)",
        f"{val_r2:.4f}",
        delta="> 0.90 is excellent",
        delta_color="normal",
    )
    col3.metric("Test MSE (Mean Squared Error)", f"{val_mse:.4f}")

    if params and "Training" in params and "selection_metrics" in params["Training"]:
        selection_metrics = params["Training"]["selection_metrics"]
        weights_str = "   |   ".join(
            [f"**{k.upper()}**: {v}" for k, v in selection_metrics.items()]
        )
        st.info(
            f"**🏆 Champion Selection Weights:**   {weights_str}  *(Models are scored and ranked automatically based on these parameters)*",
            icon="⚖️",
        )
        st.caption(
            "💡 *Note: A weight of 1.0 is always considered the 'best' performance for that specific metric.*"
        )

    st.markdown("---")

    # Feature Importance and Predictions Plot Row
    col_feat, col_pred = st.columns([1, 1])

    with col_feat:
        st.subheader("Feature Importance")
        st.markdown("What drives tipping behavior?")

        # Fetch Feature Importance from FastAPI Backend
        feature_names, importances = None, None
        try:
            res = requests.get(f"{API_URL}/feature-importance", timeout=2)
            if res.status_code == 200:
                data = res.json()
                feature_names = data.get("features")
                importances = data.get("importances")
        except requests.exceptions.RequestException:
            pass

        if importances is not None and feature_names is not None:
            feat_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importances}
            )

            # Normalize the importance to sum to 100 (Relative Percentage)
            total_importance = feat_df["Importance"].sum()
            if total_importance > 0:
                feat_df["Importance"] = (
                    (feat_df["Importance"] / total_importance) * 100
                ).round(4)

            num_features = st.slider(
                "Number of Top Features to Display",
                min_value=3,
                max_value=20,
                value=10,
                step=1,
            )

            feat_df = feat_df.sort_values(by="Importance", ascending=True).tail(
                num_features
            )  # Top N

            fig = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Top {num_features} Drivers of Taxi Tips",
                color="Importance",
                color_continuous_scale="Viridis",
                template="plotly_dark",
                labels={"Importance": "Relative Importance (%)"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "💡 *Importance is normalized as a percentage of the total predictive power across all features.*"
            )
        else:
            st.info(
                "Feature importance not supported for the current champion model type."
            )

    with col_pred:
        st.subheader("Latest Batch Predictions")

        sample_size_limit = (
            min(10000, len(predictions_df)) if len(predictions_df) > 0 else 100
        )
        num_batch_sample = st.slider(
            "Distribution Sample Size",
            min_value=100,
            max_value=sample_size_limit,
            value=min(5000, len(predictions_df)),
            step=100,
        )

        st.markdown(
            f"Displaying a sample of {num_batch_sample:,} out of {len(predictions_df):,} recent inferences."
        )

        # Quick distribution plot of predictions
        if "predicted_tip" in predictions_df.columns:
            fig2 = px.histogram(
                predictions_df.sample(num_batch_sample)
                if num_batch_sample <= len(predictions_df)
                else predictions_df,
                x="predicted_tip",
                nbins=50,
                title="Distribution of Predicted Tips (Sample)",
                color_discrete_sequence=["#FFD700"],
                template="plotly_dark",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Show actual data with enhanced visual context
            st.markdown("##### 🧾 Inferences Ledger")

            ledger_sample_limit = (
                min(1000, len(predictions_df)) if len(predictions_df) > 0 else 10
            )
            num_ledger_sample = st.slider(
                "Ledger Random Sample",
                min_value=10,
                max_value=ledger_sample_limit,
                value=min(100, len(predictions_df)),
                step=10,
            )

            disp_df = (
                predictions_df.sample(num_ledger_sample).copy()
                if num_ledger_sample <= len(predictions_df)
                else predictions_df.copy()
            )
            if "VendorID" in disp_df.columns:
                # Map real NYC taxi service providers for better business context
                vendor_map = {1: "🚗 Creative Mobile", 2: "🚕 VeriFone Inc"}
                disp_df["Vendor"] = disp_df["VendorID"].map(
                    lambda x: vendor_map.get(x, str(x))
                )

                # Reorder columns to put Vendor first
                cols = ["Vendor", "predicted_tip"] + [
                    c
                    for c in disp_df.columns
                    if c not in ["Vendor", "predicted_tip", "VendorID"]
                ]
                disp_df = disp_df[cols]

            st.dataframe(
                disp_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "predicted_tip": st.column_config.NumberColumn(
                        "Predicted Tip",
                        help="Model's estimated tip amount in USD",
                        format="$ %.2f",
                    ),
                    "Vendor": st.column_config.TextColumn(
                        "Taxi Provider", help="Technology Service Provider"
                    ),
                },
            )
            st.caption(
                "💡 *Vendor IDs from the TLC data are mapped to their respective service providers:*\n"
                "* **1:** Creative Mobile Technologies, LLC\n"
                "* **2:** VeriFone Inc."
            )

# --- PAGE 2: Interactive Prediction ---
elif page == "⚡ Interactive Prediction":
    col_head, col_btn = st.columns([5, 1])
    with col_head:
        st.header("Simulate a Ride")
        st.markdown(
            "Input ride characteristics below to get a real-time tip prediction from the model. *You can add or specify multiple rows to run batch predictions!*"
        )
    with col_btn:
        st.write("")  # Adjust vertical alignment
        if st.button("🔄 Reset Details", use_container_width=True):
            if "input_df" in st.session_state:
                del st.session_state["input_df"]
            if "editor_key" in st.session_state:
                del st.session_state["editor_key"]
            st.rerun()

    # Base characteristics layout replaced with an editable dataframe using column_config
    if "input_df" not in st.session_state:
        st.session_state.input_df = pd.DataFrame(
            [
                {
                    "trip_distance": 2.5,
                    "total_amount": 15.0,
                    "passenger_count": 1,
                    "ratecode_id": 1,
                    "airport_fee": 0.0,
                    "congestion_surcharge": 2.5,
                    "tolls_amount": 0.0,
                    "hour": 12,
                    "day": 15,
                    "month": 1,
                }
            ]
        )

    with st.form("prediction_form"):
        st.subheader("Trip Details (Editable)")
        st.caption(
            "Double-click any cell to adjust features. Use the '+' below the table to add more rides."
        )

        edited_df = st.data_editor(
            st.session_state.input_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "trip_distance": st.column_config.NumberColumn(
                    "Distance (miles)",
                    min_value=0.1,
                    max_value=100.0,
                    step=0.1,
                    format="%.1f",
                    help="Total distance of the ride in miles.",
                    default=2.5,
                    required=True,
                ),
                "total_amount": st.column_config.NumberColumn(
                    "Total Fare ($)",
                    min_value=1.0,
                    max_value=500.0,
                    step=0.5,
                    format="$ %.2f",
                    help="Total amount charged to the passenger, excluding tip.",
                    default=15.0,
                ),
                "passenger_count": st.column_config.NumberColumn(
                    "Passengers",
                    min_value=1,
                    max_value=6,
                    step=1,
                    help="Number of passengers in the vehicle.",
                    default=1,
                ),
                "ratecode_id": st.column_config.SelectboxColumn(
                    "Rate Code",
                    options=[1, 2, 3, 4, 5, 6, 99],
                    help="Final rate code in effect (e.g., 1 for Standard, 2 for JFK).",
                    default=1,
                ),
                "airport_fee": st.column_config.NumberColumn(
                    "Airport Fee ($)",
                    min_value=0.0,
                    max_value=5.0,
                    step=1.25,
                    format="$ %.2f",
                    help="Additional fee for airport trips.",
                    default=0.0,
                ),
                "congestion_surcharge": st.column_config.NumberColumn(
                    "Congestion ($)",
                    min_value=0.0,
                    max_value=5.0,
                    step=2.5,
                    format="$ %.2f",
                    help="Surcharge for entering the Manhattan congestion zone.",
                    default=2.5,
                ),
                "tolls_amount": st.column_config.NumberColumn(
                    "Tolls ($)",
                    min_value=0.0,
                    max_value=50.0,
                    step=0.5,
                    format="$ %.2f",
                    help="Total amount of all tolls paid in trip.",
                    default=0.0,
                ),
                "hour": st.column_config.NumberColumn(
                    "Pickup Hour",
                    min_value=0,
                    max_value=23,
                    step=1,
                    help="Hour of the day when the meter was engaged (0-23).",
                    default=12,
                ),
                "day": st.column_config.NumberColumn(
                    "Day of Month",
                    min_value=1,
                    max_value=31,
                    step=1,
                    help="Day of the month when the trip started.",
                    default=15,
                ),
                "month": st.column_config.NumberColumn(
                    "Month",
                    min_value=1,
                    max_value=12,
                    step=1,
                    help="Month of the year (1-12).",
                    default=1,
                ),
            },
            key="editor_key",
        )

        submitted = st.form_submit_button("Predict Tip(s) 🔮")

    if submitted:
        # Save dataframe to session state so it persists across renders
        st.session_state.input_df = edited_df.reset_index(drop=True)
        if len(edited_df) == 0:
            st.warning("Please add at least one ride to predict.")
        else:
            try:
                with st.spinner("Routing prediction requests to FastAPI Backend..."):
                    # Prepare the payload using the Pydantic schema structure
                    payload = edited_df.to_dict(orient="records")

                    response = requests.post(
                        f"{API_URL}/predict", json=payload, timeout=5
                    )
                    response.raise_for_status()

                    predictions_data = response.json()
                    preds = [item["predicted_tip"] for item in predictions_data]

                st.success(f"### Output For {len(preds)} Ride(s)")

                # Combine input config and predictions to show explicit results
                results_df = edited_df.copy()
                results_df["predicted_tip"] = preds
                results_df["tip_percentage"] = (
                    results_df["predicted_tip"] / results_df["total_amount"]
                ) * 100

                # Calculate averages for the batch
                avg_tip = results_df["predicted_tip"].mean()
                avg_pct = results_df["tip_percentage"].mean()

                st.markdown("##### 📊 Batch Prediction Averages")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Average Expected Tip", f"$ {avg_tip:.2f}")
                with col_m2:
                    st.metric("Average Tip Percentage", f"{avg_pct:.1f} %")

                st.write("")  # spacing
                disp_cols = [
                    "total_amount",
                    "trip_distance",
                    "passenger_count",
                    "predicted_tip",
                    "tip_percentage",
                ]

                # Highlight prediction outcome columns using pandas Styler
                styled_df = (
                    results_df[disp_cols]
                    .style.set_properties(
                        subset=["predicted_tip", "tip_percentage"],
                        **{
                            "background-color": "rgba(255, 215, 0, 0.15)",
                            "color": "#FFD700",
                            "font-weight": "bold",
                        },
                    )
                    .hide(axis="index")
                )

                # Display Results beautifully using column_config
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "total_amount": st.column_config.NumberColumn(
                            "Total Fare Baseline", format="$ %.2f"
                        ),
                        "trip_distance": st.column_config.NumberColumn(
                            "Distance (miles)", format="%.1f"
                        ),
                        "passenger_count": st.column_config.NumberColumn("Passengers"),
                        "predicted_tip": st.column_config.NumberColumn(
                            "Expected Tip", format="$ %.2f"
                        ),
                        "tip_percentage": st.column_config.NumberColumn(
                            "Tip %", format="%.1f %%"
                        ),
                    },
                )

            except Exception as e:
                st.error(
                    f"Prediction failed. Ensure model input shapes match. Error: {e}"
                )
