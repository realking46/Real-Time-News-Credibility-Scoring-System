from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = PROJECT_ROOT / "data" / "feature_store" / "news_features.parquet"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "bert_news_model"

MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()
EXPERIMENT_NAME = "news_credibility_bert_experiment"

MODEL_NAME = "bert-base-uncased"

MAX_SAMPLES = 2000
MAX_LENGTH = 128


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
    }


def main() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_PATH}. "
            "Run src/features/build_features.py first."
        )

    df = pd.read_parquet(FEATURE_PATH)
    df = df[["text", "label_id"]].dropna()
    df = df[df["text"].str.len() > 20]

    # Keep BERT training lightweight for CPU/local demo
    if len(df) > MAX_SAMPLES:
        df = df.sample(MAX_SAMPLES, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label_id"].astype(int),
        test_size=0.2,
        random_state=42,
        stratify=df["label_id"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = NewsDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    eval_dataset = NewsDataset(X_test, y_test, tokenizer, MAX_LENGTH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "bert_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir=str(PROJECT_ROOT / "logs" / "bert"),
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="bert_base_uncased_experiment"):
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("max_samples", MAX_SAMPLES)
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("epochs", 1)
        mlflow.log_param("batch_size", 8)
        mlflow.log_param("learning_rate", 2e-5)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        metrics = trainer.evaluate()

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key.replace("eval_", ""), value)

        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        trainer.save_model(str(MODEL_OUTPUT_DIR))
        tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

        mlflow.log_artifacts(str(MODEL_OUTPUT_DIR), artifact_path="bert_model")

        print("BERT training completed.")
        print(metrics)
        print(f"Saved BERT model to: {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()