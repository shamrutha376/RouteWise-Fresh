
from pathlib import Path

import streamlit as st
import joblib

st.set_page_config(page_title="RouteWise Diagnostic")

st.title("RouteWise Startup Diagnostic")

model_path = Path(__file__).parent / "models" / "traffic_prediction_model.pkl"

st.write("Checking traffic model file...")

if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    st.stop()

st.write(f"Model file size: {model_path.stat().st_size / (1024 * 1024):.1f} MB")

st.write("Loading traffic model...")

try:
    model = joblib.load(model_path)
    st.success("Traffic model loaded successfully.")
    st.write(f"Model type: {type(model).__name__}")
except Exception as exc:
    st.error(f"{type(exc).__name__}: {exc}")
    st.exception(exc)
