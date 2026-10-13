
import streamlit as st
from pathlib import Path
import joblib
import traceback

st.set_page_config(page_title="RouteWise Model Test")
st.title("RouteWise Traffic Model Test")

model_path = Path(__file__).parent / "models" / "traffic_prediction_model.pkl"

st.write("Checking model file...")

if not model_path.exists():
    st.error(f"Model not found: {model_path}")
    st.stop()

st.write(f"Model size: {model_path.stat().st_size / (1024 * 1024):.1f} MB")
st.write("Attempting to load model...")

try:
    model = joblib.load(model_path)
    st.success("Traffic model loaded successfully.")
    st.write(f"Loaded model type: {type(model).__name__}")
except Exception as e:
    st.error(f"{type(e).__name__}: {e}")
    st.code(traceback.format_exc())
