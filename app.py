"""
NYC Taxi Tips Predictor & Analyst Dashboard.

This Streamlit application serves as the interactive frontend (Phase 5) for the NYC Taxi Tip
Prediction System. It loads evaluation metrics, batch predictions, and the trained Champion Model
from the local artifacts to visualize performance and allow real-time interactive predictions.

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
from src.utils.model_utils import get_feature_importances

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
        st.caption(
            "💡 *Note: A weight of 1.0 is always considered the 'best' performance for that specific metric.*"
        )

    st.markdown("---")

    # Feature Importance and Predictions Plot Row
    col_feat, col_pred = st.columns([1, 1])

    with col_feat:
        st.subheader("Feature Importance")
        st.markdown("What drives tipping behavior?")

        # Extract feature importance safely from dedicated utility
        feature_names, importances = get_feature_importances(model)

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
        sample_size = min(5000, len(predictions_df))
        st.markdown(
            f"Displaying a sample of {sample_size:,} out of {len(predictions_df):,} recent inferences."
        )

        # Quick distribution plot of predictions
        if "predicted_tip" in predictions_df.columns:
            fig2 = px.histogram(
                predictions_df.sample(sample_size),
                x="predicted_tip",
                nbins=50,
                title="Distribution of Predicted Tips (Sample)",
                color_discrete_sequence=["#FFD700"],
                template="plotly_dark",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Show actual data with enhanced visual context
            st.markdown("##### 🧾 Inferences Ledger (Random 100 Sample)")

            disp_df = predictions_df.sample(100).copy()
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
        import math

        if len(edited_df) == 0:
            st.warning("Please add at least one ride to predict.")
        else:
            predictions_list = []

            for _, row in edited_df.iterrows():
                hour = row["hour"]
                day = row["day"]
                month = row["month"]

                # Cyclical features
                hour_sin = math.sin(2 * math.pi * hour / 24)
                hour_cos = math.cos(2 * math.pi * hour / 24)
                day_sin = math.sin(2 * math.pi * day / 31)
                day_cos = math.cos(2 * math.pi * day / 31)
                month_sin = math.sin(2 * math.pi * month / 12)
                month_cos = math.cos(2 * math.pi * month / 12)

                # Create input dict for model mapped to expected feature engineering features
                input_dict = {
                    "VendorID": 1.0,
                    "passenger_count": float(row["passenger_count"]),
                    "trip_distance": float(row["trip_distance"]),
                    "RatecodeID": float(row["ratecode_id"]),
                    "PULocationID": 132.0,
                    "DOLocationID": 236.0,
                    "payment_type": 1.0,
                    "fare_amount": row["total_amount"]
                    - row["airport_fee"]
                    - row["congestion_surcharge"]
                    - row["tolls_amount"],
                    "extra": 0.0,
                    "mta_tax": 0.5,
                    "tolls_amount": float(row["tolls_amount"]),
                    "improvement_surcharge": 0.3,
                    "total_amount": float(row["total_amount"]),
                    "congestion_surcharge": float(row["congestion_surcharge"]),
                    "Airport_fee": float(row["airport_fee"]),
                    "pickup_hour_sin": hour_sin,
                    "pickup_hour_cos": hour_cos,
                    "pickup_day_sin": day_sin,
                    "pickup_day_cos": day_cos,
                    "pickup_month_sin": month_sin,
                    "pickup_month_cos": month_cos,
                }
                predictions_list.append(input_dict)

            model_input_df = pd.DataFrame(predictions_list)

            # Ensure exact columns
            if hasattr(model, "feature_names_in_"):
                expected_cols = model.feature_names_in_
                for col in expected_cols:
                    if col not in model_input_df.columns:
                        model_input_df[col] = 0.0
                model_input_df = model_input_df[expected_cols]

            try:
                with st.spinner("Calculating..."):
                    preds = model.predict(model_input_df)

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
