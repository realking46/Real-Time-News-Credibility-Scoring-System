from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.predict import predict_credibility


app = FastAPI(
    title="Real-Time News Credibility API",
    description="Predicts credibility score and risk level for news text.",
    version="0.1.0",
)


class NewsInput(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "message": "News Credibility API is running",
        "endpoint": "/predict",
    }


@app.post("/predict/")
def predict(input_data: NewsInput):
    result = predict_credibility(input_data.text)
    return result