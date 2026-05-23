from pathlib import Path
import joblib
import mlflow
import pandas as pd
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()

MODEL_NAME = "news_credibility_model"
LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"


def load_latest_model():
    if LOCAL_MODEL_PATH.exists():
        return joblib.load(LOCAL_MODEL_PATH)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    )
    model_uri = f"models:/{MODEL_NAME}/latest"
    return mlflow.pyfunc.load_model(model_uri)


def risk_level(score: float) -> str:
    if score >= 70:
        return "Low"
    if score >= 40:
        return "Medium"
    return "High"


def format_prediction(label_id: int, confidence: float | None = None) -> dict:
    label = "real" if label_id == 1 else "fake"

    if confidence is None:
        credibility_score = 80 if label_id == 1 else 25
    else:
        if label_id == 1:
            credibility_score = int(confidence * 100)
        else:
            credibility_score = int((1 - confidence) * 100)

    return {
        "prediction_label": label,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "credibility_score": credibility_score,
        "risk_level": risk_level(credibility_score),
    }


def predict_credibility(text: str) -> dict:
    model = load_latest_model()
    input_df = pd.Series([text])

    prediction = model.predict(input_df)
    label_id = int(prediction[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(max(probabilities))

    return format_prediction(label_id, confidence)


def predict_with_model(model, text: str) -> dict:
    input_df = pd.Series([text])
    prediction = model.predict(input_df)
    return format_prediction(int(prediction[0]))