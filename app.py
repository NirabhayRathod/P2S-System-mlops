# Imports
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
load_dotenv()
import mlflow
import mlflow.pyfunc

# Helper: Required environment vars
def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value

# MLflow + DagsHub Configuration
try:
    DAGSHUB_USER = require_env("DAGSHUB_USER")
    DAGSHUB_TOKEN = require_env("DAGSHUB_TOKEN")
    DAGSHUB_MLFLOW_URI = require_env("MLFLOW_TRACKING_URI")
except RuntimeError as e:
    st.error(f"❌ Configuration Error: {e}")
    st.stop()

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

# Streamlit Page Config
st.set_page_config(
    page_title="P2S Earthquake Warning System",
    page_icon="🌍",
    layout="wide"
)

# ========================
# CUSTOM CSS FOR BETTER UI
# ========================
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #0b1120;
    }
    /* Cards */
    .info-card {
        background-color: #0f172a;
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid #1e293b;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5);
    }
    /* Alert cards */
    .alert-danger {
        background: linear-gradient(135deg, #7f1a1a, #991b1b);
        border-left: 8px solid #ef4444;
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);
    }
    .alert-success {
        background: linear-gradient(135deg, #064e3b, #047857);
        border-left: 8px solid #10b981;
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
    }
    .big-number {
        font-size: 3.5rem;
        font-weight: 800;
        color: #facc15;
        margin: 0.5rem 0;
    }
    .warning-title {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        letter-spacing: 2px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    hr {
        border-color: #1e293b;
    }
    .stButton button {
        background: linear-gradient(90deg, #dc2626, #ef4444);
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 0.6rem 1.5rem;
        border-radius: 40px;
        border: none;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.02);
        background: linear-gradient(90deg, #b91c1c, #dc2626);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="info-card">
    <h1 style="color:#f1f5f9;">🌍 P2S Earthquake Early Warning System</h1>
    <p style="color:#94a3b8; font-size:1.1rem;">
        AI‑powered real‑time earthquake detection with early warning capability
    </p>
    <div>
        <span style="display:inline-block; background:#1e293b; color:#38bdf8; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin-right:6px;">Machine Learning</span>
        <span style="display:inline-block; background:#1e293b; color:#38bdf8; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin-right:6px;">MLOps</span>
        <span style="display:inline-block; background:#1e293b; color:#38bdf8; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin-right:6px;">MLflow</span>
        <span style="display:inline-block; background:#1e293b; color:#38bdf8; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin-right:6px;">Airflow</span>
        <span style="display:inline-block; background:#1e293b; color:#38bdf8; padding:4px 12px; border-radius:20px; font-size:0.8rem; margin-right:6px;">Docker</span>
        <span style="display:inline-block; background:#1e293b; color:#38bdf8; padding:4px 12px; border-radius:20px; font-size:0.8rem;">Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================
# PLACEHOLDER FOR PREDICTION RESULTS (will be filled on button click)
# ==============================================
prediction_placeholder = st.container()

# ==============================================
# HOW THIS SYSTEM WORKS (always visible)
# ==============================================
st.header("🧠 How This System Works")
st.markdown("""
<div class="info-card">
<b>1. Seismic Signal Capture</b><br>
Raw vibration signals are collected from seismic sensors.<br><br>

<b>2. Feature Engineering</b><br>
Six engineered features are extracted including amplitude, noise level, PGA, and SNR.<br><br>

<b>3. P-wave Detection</b><br>
A classification model detects early P-waves indicating earthquake onset.<br><br>

<b>4. S-wave Arrival Prediction</b><br>
If detected, a regression model predicts remaining time before destructive S-waves arrive.<br><br>

<b>5. Early Warning</b><br>
The system provides a <b>5–10 second advance warning</b> enabling immediate safety actions.
</div>
""", unsafe_allow_html=True)

# Sidebar Inputs
with st.sidebar:
    st.header("📡 Seismic Sensor Inputs")
    sensor_reading = st.number_input("Sensor Reading", -1000.0, 1000.0, 0.45)
    noise_level = st.slider("Noise Level", 0.0, 1.0, 0.28)
    rolling_avg = st.number_input("Rolling Average", -100.0, 100.0, 3.07)
    reading_diff = st.number_input("Reading Difference", -10.0, 10.0, 0.24)
    pga = st.slider("PGA (Peak Ground Acceleration)", 0.0, 1.0, 0.33)
    snr = st.number_input("SNR (Signal-to-Noise Ratio)", -50.0, 50.0, 16.44)
    threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.8, 0.05)
    st.divider()
    predict_btn = st.button("🚨 PREDICT EARTHQUAKE", use_container_width=True)

    # Load models
    @st.cache_resource(show_spinner=False)
    def load_models():
        try:
            pwave_model = mlflow.pyfunc.load_model("models:/P2S_PWAVE_MODEL/Production")
            swave_model = mlflow.pyfunc.load_model("models:/P2S_SWAVE_MODEL/Production")
            return pwave_model, swave_model
        except Exception as e:
            raise RuntimeError(str(e))

    try:
        pwave_model, swave_model = load_models()
        st.success("✅ Models ready (Production)")
        model_ready = True
    except Exception as e:
        st.error("❌ Model loading failed")
        st.code(str(e))
        model_ready = False

# ==============================================
# PREDICTION LOGIC - updates the placeholder above
# ==============================================
if predict_btn and model_ready:
    features = np.array([[
        sensor_reading, noise_level, rolling_avg,
        reading_diff, pga, snr
    ]])
    pwave_prob = float(pwave_model.predict(features)[0])
    earthquake_detected = pwave_prob > threshold

    # Clear the placeholder and write new results
    with prediction_placeholder:
        st.markdown("---")
        st.header("📊 Real‑time Prediction Results")
        
        if earthquake_detected:
            swave_arrival = float(swave_model.predict(features)[0])
            st.markdown(f"""
            <div class="alert-danger">
                <div class="warning-title">⚠️ EARTHQUAKE DETECTED ⚠️</div>
                <div style="font-size:1.3rem; margin-top:0.5rem;">S‑wave will arrive in</div>
                <div class="big-number">{swave_arrival:.1f} seconds</div>
                <div style="margin-top:0.8rem;">🔔 Take cover immediately! 🔔</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-success">
                <div class="warning-title" style="color:#a7f3d0;">✅ NO EARTHQUAKE DETECTED</div>
                <div style="font-size:1.2rem; margin-top:0.5rem;">Ground is stable</div>
                <div style="margin-top:0.5rem;">P‑wave probability: {pwave_prob:.2%} (below threshold)</div>
            </div>
            """, unsafe_allow_html=True)

        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="P‑wave Probability", value=f"{pwave_prob:.2%}", delta=None)
        with col2:
            st.metric(label="Alert Threshold", value=f"{threshold:.0%}")
        with col3:
            if earthquake_detected:
                st.metric(label="S‑wave arrival (seconds)", value=f"{swave_arrival:.1f} sec", delta="WARNING")
            else:
                st.metric(label="S‑wave arrival", value="N/A")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pwave_prob * 100,
            title={"text": "P‑wave Probability (%)", "font": {"color": "white"}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': "white"},
                'bar': {'color': "#ef4444"},
                'steps': [
                    {'range': [0, threshold*100], 'color': "#14532d"},
                    {'range': [threshold*100, 100], 'color': "#7f1a1a"}
                ],
                'threshold': {
                    'line': {'color': "#facc15", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold*100
                }
            }
        ))
        fig_gauge.update_layout(height=300, paper_bgcolor="#0f172a", font={'color': 'white'})
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Input vs Prediction insights
        with st.expander("🔍 Seismic Signature Analysis"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Input Features**")
                input_df = pd.DataFrame({
                    "Feature": ["Sensor Reading", "Noise Level", "Rolling Avg", "Reading Diff", "PGA", "SNR"],
                    "Value": [sensor_reading, noise_level, rolling_avg, reading_diff, pga, snr]
                })
                st.dataframe(input_df, hide_index=True, use_container_width=True)
            with col_b:
                st.write("**Model Interpretation**")
                if earthquake_detected:
                    st.markdown("• High P‑wave probability suggests **impending S‑wave**")
                    st.markdown("• PGA and SNR values indicate significant ground motion")
                else:
                    st.markdown("• P‑wave probability below safety threshold")
                    st.markdown("• No immediate action required")
else:
    # If no prediction yet, show a placeholder message in the container
    with prediction_placeholder:
        st.info("👈 Enter sensor readings in the sidebar and click **PREDICT EARTHQUAKE** to see real‑time warnings.")

# MLOps & Architecture Section
st.markdown("---")
st.header("⚙️ MLOps Pipeline & Deployment")
with st.expander("📦 View full MLOps architecture"):
    st.markdown("""
    - **Data Versioning**: DVC + DagsHub  
    - **Experiment Tracking**: MLflow  
    - **Orchestration**: Apache Airflow (scheduled retraining)  
    - **Model Registry**: MLflow Model Registry (Production stage)  
    - **Deployment**: Docker + Streamlit on Render  
    - **CI/CD**: GitHub Actions  
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align:center">
    <h4>P2S Earthquake Early Warning System</h4>
    <p><strong>Developer:</strong> Nirabhay Singh Rathod</p>
    <p><strong>Contact:</strong> nirbhay105633016@gmail.com</p>
    <p><strong>MLOps Stack:</strong> Git • DVC • MLflow • Airflow • Docker • Streamlit</p>
</div>
""", unsafe_allow_html=True)