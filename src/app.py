import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from config import MODELS_DIR, FEATURE_COLUMN

st.set_page_config(page_title="NASA Engine Diagnostic", layout="wide")

st.title("🚀 NASA Turbofan Health Monitor")
st.markdown("#### *AI Prognostics System based on C-MAPSS Dataset*")
st.divider()

@st.cache_resource
def load_assets():
    model = joblib.load(MODELS_DIR / "engine_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    return model, scaler

model, scaler = load_assets()

# 1. Sidebar Panel (Top 5 Sensors from your Importance list)
st.sidebar.header("🕹️ Critical Telemetry")
important_sensors = ['sensor_11', 'sensor_4', 'sensor_9', 'sensor_12', 'sensor_14']

inputs = {}
# Default values (averages from NASA data) for all sensors
for sensor in FEATURE_COLUMN:
    inputs[sensor] = 500.0 if sensor in ['sensor_4', 'sensor_11'] else 15.0

# Sliders for the most important ones
for sensor in important_sensors:
    inputs[sensor] = st.sidebar.slider(f"Active {sensor.replace('_', ' ').title()}", 0.0, 1000.0 if sensor in ['sensor_4', 'sensor_11', 'sensor_9'] else 100.0, float(inputs[sensor]))

# NASA Engineering Feature
inputs['Temp_Diff'] = inputs['sensor_12'] - inputs['sensor_14']

# 2. Prediction Logic
input_df = pd.DataFrame([inputs])[FEATURE_COLUMN]
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]
confidence = model.predict_proba(scaled_input)[0]

# 3. UI Display
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Engine Status")
    if prediction == 1:
        st.error("## 🚨 CRITICAL: FAILURE IMMINENT")
        st.write(f"Estimated Probability of Failure: **{confidence[1]*100:.1f}%**")
    else:
        st.success("## ✅ SYSTEM NORMAL")
        st.write(f"Health Confidence: **{confidence[0]*100:.1f}%**")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence[prediction] * 100,
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "#ef4444" if prediction == 1 else "#10b981"}}
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Sensor Footprint")
    # Show how the current inputs compare to failure thresholds
    st.line_chart(input_df.T)
    st.info(f"💡 Note: {important_sensors[0]} is your most critical predictor (61% weight).")

st.caption("NASA Jet Engine Maintenance AI | Developed by Vaibhav Arya")
