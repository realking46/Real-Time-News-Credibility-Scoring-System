from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"

st.set_page_config(
    page_title="News Credibility Scoring System",
    page_icon="📰",
    layout="centered",
)

st.title("📰 Real-Time News Credibility Scoring System")
st.write("HuggingFace-ready Streamlit deployment version.")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please include models/baseline_model.joblib.")
        return None
    return joblib.load(MODEL_PATH)


def risk_level(score: float) -> str:
    if score >= 70:
        return "Low"
    if score >= 40:
        return "Medium"
    return "High"


def format_prediction(label_id: int) -> dict:
    credibility_score = 80 if label_id == 1 else 25
    label = "real" if label_id == 1 else "fake"

    return {
        "prediction_label": label,
        "credibility_score": credibility_score,
        "risk_level": risk_level(credibility_score),
    }


model = load_model()

text = st.text_area(
    "Paste news article text",
    height=220,
    placeholder="Paste article or claim here...",
)

if st.button("Check Credibility"):
    if not text.strip():
        st.warning("Please enter news text.")
    elif model is None:
        st.error("Model is not available.")
    else:
        prediction = model.predict(pd.Series([text]))
        result = format_prediction(int(prediction[0]))

        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Credibility Score", result["credibility_score"])
        col2.metric("Risk Level", result["risk_level"])
        col3.metric("Prediction", result["prediction_label"])

        if result["risk_level"] == "Low":
            st.success("This article appears relatively credible.")
        elif result["risk_level"] == "Medium":
            st.warning("This article has moderate credibility risk.")
        else:
            st.error("This article has high credibility risk.")