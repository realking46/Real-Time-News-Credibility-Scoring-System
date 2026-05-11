from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "bert_news_model"


def risk_level(score: float) -> str:
    if score >= 70:
        return "Low"
    if score >= 40:
        return "Medium"
    return "High"


def load_bert_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"BERT model not found at {MODEL_DIR}. "
            "Run python -m src.training.train_bert first."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    return tokenizer, model


def predict_bert(text: str) -> dict:
    tokenizer, model = load_bert_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(torch.max(probs).item())

    label = "real" if predicted_class == 1 else "fake"

    if label == "real":
        credibility_score = int(60 + confidence * 40)
    else:
        credibility_score = int((1 - confidence) * 40)

    return {
        "model": "bert-base-uncased",
        "prediction_label": label,
        "confidence": round(confidence, 4),
        "credibility_score": credibility_score,
        "risk_level": risk_level(credibility_score),
    }


if __name__ == "__main__":
    sample = "The government confirmed the new policy in an official statement."
    print(predict_bert(sample))