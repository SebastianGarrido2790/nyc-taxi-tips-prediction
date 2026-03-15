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

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yaml

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
    with open(METRICS_PATH) as f:
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
    with open(PARAMS_PATH) as f:
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
page = st.sidebar.radio("Navigate", ["📊 Dashboard & Evaluation", "⚡ Interactive Prediction"])

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
            feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

            # Normalize the importance to sum to 100 (Relative Percentage)
            total_importance = feat_df["Importance"].sum()
            if total_importance > 0:
                feat_df["Importance"] = ((feat_df["Importance"] / total_importance) * 100).round(4)

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
            st.info("Feature importance not supported for the current champion model type.")

    with col_pred:
        st.subheader("Latest Batch Predictions")

        sample_size_limit = min(10000, len(predictions_df)) if len(predictions_df) > 0 else 100
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

            ledger_sample_limit = min(1000, len(predictions_df)) if len(predictions_df) > 0 else 10
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
                disp_df["Vendor"] = disp_df["VendorID"].map(lambda x: vendor_map.get(x, str(x)))

                # Reorder columns to put Vendor first
                cols = ["Vendor", "predicted_tip"] + [
                    c for c in disp_df.columns if c not in ["Vendor", "predicted_tip", "VendorID"]
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

# --- PAGE 2: Agentic Chat UI ---
elif page == "⚡ Interactive Prediction":
    from dotenv import load_dotenv

    from src.agents.taxi_analyst_agent import AgentConfigError, get_taxi_analyst_agent

    load_dotenv()  # Ensure .env is loaded (idempotent; safe to call multiple times)

    st.header("🤖 Agentic Taxi Analyst")
    st.markdown(
        "Chat naturally with the **Agentic Taxi Analyst** — describe your ride and get an "
        "ML-powered tip prediction, or ask any NYC taxi question."
    )

    # --- Session State Initialisation ---
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict] = []

    # Lazy-load the compiled LangGraph agent once per browser session
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = get_taxi_analyst_agent()
        except AgentConfigError as cfg_err:
            st.session_state.agent = None
            st.session_state.agent_error = str(cfg_err)

    # --- API Key Missing Banner ---
    if st.session_state.get("agent") is None:
        st.error(
            f"⚠️ **Agent not initialised:** {st.session_state.get('agent_error', 'Unknown configuration error.')}  \n"
            "Open `.env`, set `GOOGLE_API_KEY=AIza<your-key>`, and restart the app.",
            icon="🔑",
        )
        st.stop()

    # --- Toolbar ---
    col_title, col_clear = st.columns([5, 1])
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    # --- Chat History ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat Input ---
    user_input = st.chat_input(
        "Describe your ride or ask a question… e.g. 'Predict a tip for a 5-mile JFK trip at 3 PM, $25 fare, 2 passengers.'"
    )

    if user_input:
        # Append and render the user's message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # --- Agent Invocation ---
        with st.chat_message("assistant"):
            with st.spinner("Analysing your request…"):
                try:
                    # Construct the full conversation history for LangGraph's state
                    langchain_history = []
                    for m in st.session_state.messages:
                        if m["role"] == "user":
                            langchain_history.append(("human", m["content"]))
                        elif m["role"] == "assistant":
                            langchain_history.append(("ai", m["content"]))

                    result = st.session_state.agent.invoke({"messages": langchain_history})

                    # The last message in the graph output is the assistant's reply
                    msg = result["messages"][-1]
                    raw_content = msg.content

                    if isinstance(raw_content, list):
                        # Extract and join all 'text' parts (handles Gemini's block-based format)
                        assistant_reply = "".join(
                            [
                                part.get("text", "")
                                for part in raw_content
                                if isinstance(part, dict) and part.get("type") == "text"
                            ]
                        )
                    else:
                        assistant_reply = str(raw_content)
                except Exception as agent_err:
                    err_name = type(agent_err).__name__
                    if (
                        "RateLimitError" in err_name
                        or "quota" in str(agent_err).lower()
                        or "RESOURCE_EXHAUSTED" in str(agent_err)
                    ):
                        assistant_reply = (
                            f"⚠️ **Brain Error (Google Quota):**\n\n"
                            f"`{err_name}: {agent_err}`\n\n"
                            "It looks like your **Google API Key** has hit a rate limit. "
                            "Please check your usage at [Google AI Studio](https://aistudio.google.com/app/apikey)."
                        )
                    elif "ConnectionError" in err_name or "localhost:8000" in str(agent_err):
                        assistant_reply = (
                            f"⚠️ **Brawn Error (FastAPI Offline):**\n\n"
                            f"`{err_name}: {agent_err}`\n\n"
                            "The analyst can't reach the tip prediction model. Please ensure the **FastAPI backend** "
                            "is running (`uv run uvicorn src.api.predict_api:app --reload`)."
                        )
                    else:
                        assistant_reply = (
                            f"⚠️ **The analyst encountered an unexpected error:**\n\n"
                            f"`{err_name}: {agent_err}`\n\n"
                            "Check your `.env` configuration or ensure all backend services are active."
                        )

            st.markdown(assistant_reply)

        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
