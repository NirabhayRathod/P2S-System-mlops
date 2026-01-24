
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

# Auth for DagsHub MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# Set MLflow Tracking URI
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

# Streamlit Page Config

st.set_page_config(
    page_title="P2S Earthquake Warning System",
    page_icon="🌍",
    layout="wide"
)

# Global Styling

st.markdown("""
<style>
.main { background-color: #0e1117; }
h1, h2, h3 { color: #f1f5f9; }
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    background-color: #1f2937;
    color: #38bdf8;
    font-size: 0.8rem;
    margin-right: 6px;
}
.info-card {
    background-color: #020617;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #1e293b;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Header

st.markdown("""
<div class="info-card">
    <h1>🌍 P2S Earthquake Early Warning System</h1>
    <p style="color:#94a3b8;">
        AI-powered real-time earthquake detection with early warning capability
    </p>
    <div>
        <span class="badge">Machine Learning</span>
        <span class="badge">MLOps</span>
        <span class="badge">MLflow</span>
        <span class="badge">Airflow</span>
        <span class="badge">Docker</span>
        <span class="badge">Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "**Dual ML Models:** P-wave detection (classification) + S-wave arrival prediction (regression)"
)
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


# Load Models from MLflow Registry

@st.cache_resource(show_spinner=True)
def load_models():
    """
    Load Production models from MLflow Model Registry.
    This is the ONLY correct approach for production inference.
    """
    try:
        pwave_model = mlflow.pyfunc.load_model(
            "models:/P2S_PWAVE_MODEL/Production"
        )
        swave_model = mlflow.pyfunc.load_model(
            "models:/P2S_SWAVE_MODEL/Production"
        )
        return pwave_model, swave_model
    except Exception as e:
        raise RuntimeError(str(e))

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

    # Load models safely
    try:
        pwave_model, swave_model = load_models()
        st.success("✅ Models loaded from MLflow (Production)")
        model_ready = True
    except Exception as e:
        st.error("❌ Model loading failed")
        st.code(str(e))
        model_ready = False


# How the System Works




# Prediction Section

st.header("📡 Live Earthquake Prediction Dashboard")

if predict_btn and model_ready:

    features = np.array([[
        sensor_reading,
        noise_level,
        rolling_avg,
        reading_diff,
        pga,
        snr
    ]])

    pwave_probability = float(pwave_model.predict(features)[0])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("P-wave Probability", f"{pwave_probability:.2%}")

    with col2:
        if pwave_probability > threshold:
            st.error("🔴 EARTHQUAKE DETECTED")
        else:
            st.success("🟢 NO EARTHQUAKE")

    with col3:
        if pwave_probability > threshold:
            swave_arrival = float(swave_model.predict(features)[0])
            st.metric("S-wave Arrival", f"{swave_arrival:.1f} seconds")
        else:
            st.metric("S-wave Arrival", "N/A")

    fig = go.Figure(go.Scatterpolar(
        r=[pwave_probability, abs(sensor_reading) / 10, pga, min(abs(snr) / 50, 1)],
        theta=["P-wave Prob", "Amplitude", "PGA", "SNR"],
        fill="toself"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Seismic Signature Pattern"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Enter sensor readings and click **PREDICT EARTHQUAKE**")

# MLOps & Architecture

st.header("⚙️ MLOps Pipeline & Deployment")

st.markdown("""
<div class="info-card">
<b>Training</b><br>
• Data versioned using DVC + DagsHub<br>
• Multiple ML models trained & evaluated<br>
• Best models registered in MLflow<br><br>

<b>Orchestration</b><br>
• Apache Airflow schedules retraining pipelines<br>
• Automated model promotion to Production<br><br>

<b>Deployment</b><br>
• Streamlit app loads <b>Production models</b> from MLflow<br>
• Dockerized inference service<br>
• CI/CD using GitHub Actions
</div>
""", unsafe_allow_html=True)


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
