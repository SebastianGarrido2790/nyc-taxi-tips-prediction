"""
NYC Taxi Tips Predictor & Analyst Dashboard.

This Streamlit application serves as the interactive frontend (Phase 5) for the
NYC Taxi Tip Prediction System. It loads evaluation metrics, batch predictions,
and the trained Champion Model (XGBoost) artifact to visualize performance
and allow real-time interactive predictions.

Usage:
    uv run streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import joblib
import plotly.express as px
import yaml
from pathlib import Path

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


@st.cache_resource
def load_model():
    """
    Loads the Champion Model (Joblib binary) from the local artifacts directory.
    Returns (model, model_name) or (None, None) if the model cannot be found.
    """
    if not MODEL_DIR.exists():
        return None, None

    model_files = list(MODEL_DIR.glob("*.joblib"))
    if not model_files:
        return None, None

    model_path = model_files[0]
    return joblib.load(model_path), model_path.stem


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
model, model_name = load_model()
params = load_params()

# --- Main App Layout ---
st.title("🚕 NYC Taxi Tips Predictor & Analyst")
st.markdown(
    f"*A production-ready FTI architecture serving ({model_name} model) predictions.*"
)

if model is None or metrics is None or predictions_df is None:
    st.error(
        "⚠️ Initial artifacts not found. Please run the DVC pipeline (`uv run dvc repro`) first."
    )
    st.stop()

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

    st.markdown("---")

    # Feature Importance and Predictions Plot Row
    col_feat, col_pred = st.columns([1, 1])

    with col_feat:
        st.subheader("Feature Importance")
        st.markdown("What drives tipping behavior?")

        # Extract feature importance safely
        current_model = model
        feature_names = None
        importances = None

        # Try to get from XGBoost or Random Forest
        if hasattr(current_model, "feature_importances_") and hasattr(
            current_model, "feature_names_in_"
        ):
            importances = current_model.feature_importances_
            feature_names = current_model.feature_names_in_
        elif hasattr(current_model, "get_booster"):  # direct XGBoost
            booster = current_model.get_booster()
            importance_map = booster.get_score(importance_type="gain")
            feature_names = list(importance_map.keys())
            importances = list(importance_map.values())

        if importances is not None and feature_names is not None:
            feat_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importances}
            )
            feat_df = feat_df.sort_values(by="Importance", ascending=True).tail(
                10
            )  # Top 10

            fig = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 10 Drivers of Taxi Tips",
                color="Importance",
                color_continuous_scale="Viridis",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Feature importance not supported for the current champion model type."
            )

    with col_pred:
        st.subheader("Latest Batch Predictions")
        st.markdown(f"Displaying a sample of {len(predictions_df)} recent inferences.")

        # Quick distribution plot of predictions
        if "predicted_tip" in predictions_df.columns:
            fig2 = px.histogram(
                predictions_df.sample(min(1000, len(predictions_df))),
                x="predicted_tip",
                nbins=50,
                title="Distribution of Predicted Tips (Sample)",
                color_discrete_sequence=["#FFD700"],
                template="plotly_dark",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Show actual data
            st.dataframe(predictions_df.head(100), use_container_width=True)

# --- PAGE 2: Interactive Prediction ---
elif page == "⚡ Interactive Prediction":
    st.header("Simulate a Ride")
    st.markdown(
        "Input ride characteristics below to get a real-time tip prediction from the model."
    )

    with st.form("prediction_form"):
        st.subheader("Trip Details")
        col1, col2 = st.columns(2)

        with col1:
            trip_distance = st.number_input(
                "Trip Distance (miles)",
                min_value=0.1,
                max_value=100.0,
                value=2.5,
                step=0.1,
            )
            total_amount = st.number_input(
                "Total Fare Amount ($)",
                min_value=1.0,
                max_value=500.0,
                value=15.0,
                step=0.5,
            )
            passenger_count = st.number_input(
                "Passenger Count", min_value=1, max_value=6, value=1, step=1
            )
            ratecode_id = st.selectbox(
                "Rate Code",
                options=[1, 2, 3, 4, 5, 6, 99],
                format_func=lambda x: f"Code {x}",
            )

        with col2:
            airport_fee = st.number_input(
                "Airport Fee ($)", min_value=0.0, max_value=5.0, value=0.0, step=1.25
            )
            congestion_surcharge = st.number_input(
                "Congestion Surcharge ($)",
                min_value=0.0,
                max_value=5.0,
                value=2.5,
                step=2.5,
            )
            tolls_amount = st.number_input(
                "Tolls Amount ($)", min_value=0.0, max_value=50.0, value=0.0, step=0.5
            )
            hour = st.slider("Pickup Hour", 0, 23, 12)
            day = st.slider("Day of Month", 1, 31, 15)
            month = st.slider("Month", 1, 12, 1)

        submitted = st.form_submit_button("Predict Tip 🔮")

    if submitted:
        # 1. We need to construct the input exactly as the model expects it.
        # This requires mimicking the feature engineering step.
        # We need the cyclical features:
        import math

        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        day_sin = math.sin(2 * math.pi * day / 31)
        day_cos = math.cos(2 * math.pi * day / 31)
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)

        # Create input df. Note: The exact columns depend on what was outputted by feature_engineering
        # Let's read the feature names from the model if possible, or build a safe dict.
        input_dict = {
            "VendorID": 1.0,  # dummy
            "passenger_count": float(passenger_count),
            "trip_distance": float(trip_distance),
            "RatecodeID": float(ratecode_id),
            "PULocationID": 132.0,  # dummy JFK
            "DOLocationID": 236.0,  # dummy Upper East Side
            "payment_type": 1.0,  # dummy Credit Card
            "fare_amount": total_amount
            - airport_fee
            - congestion_surcharge
            - tolls_amount,  # Approx
            "extra": 0.0,  # dummy
            "mta_tax": 0.5,  # dummy
            "tolls_amount": float(tolls_amount),
            "improvement_surcharge": 0.3,  # dummy
            "total_amount": float(total_amount),
            "congestion_surcharge": float(congestion_surcharge),
            "Airport_fee": float(airport_fee),
            "pickup_hour_sin": hour_sin,
            "pickup_hour_cos": hour_cos,
            "pickup_day_sin": day_sin,
            "pickup_day_cos": day_cos,
            "pickup_month_sin": month_sin,
            "pickup_month_cos": month_cos,
        }

        input_df = pd.DataFrame([input_dict])

        # Ensure exact columns
        if hasattr(model, "feature_names_in_"):
            expected_cols = model.feature_names_in_
            # Add missing cols with 0
            for col in expected_cols:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            # Order and filter
            input_df = input_df[expected_cols]

        try:
            with st.spinner("Calculating..."):
                pred = model.predict(input_df)[0]

            st.success(f"### Expected Tip: **${pred:.2f}**")

            # Additional context
            st.info(
                f"That's a **{(pred / total_amount) * 100:.1f}%** tip on the ${total_amount:.2f} total."
            )
        except Exception as e:
            st.error(f"Prediction failed. Ensure model input shapes match. Error: {e}")
