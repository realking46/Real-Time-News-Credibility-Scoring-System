"""
training_pipeline/train.py
───────────────────────────
Reads the latest feature Parquet, fine-tunes a BERT regression head,
and logs everything to MLflow.

Model architecture:
    bert-base-uncased  →  dropout  →  Linear(768 → 1)  →  sigmoid × 100
    Output: credibility score 0–100

Usage:
    python -m training_pipeline.train
    python -m training_pipeline.train --epochs 5 --batch-size 16 --no-bert-cols
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from feature_pipeline.store import load_all_features
from training_pipeline.evaluate import compute_metrics, plot_predictions

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME   = os.getenv("MLFLOW_MODEL_NAME", "news-credibility-scorer")
BERT_MODEL_NAME     = "bert-base-uncased"
RANDOM_SEED         = 42


# ── Dataset ───────────────────────────────────────────────────────────────────

class NewsDataset(Dataset):
    """
    PyTorch Dataset that tokenizes article text on-the-fly.
    Falls back to pre-computed BERT embeddings if available.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[float],
        tokenizer,
        max_length: int = 256,
    ):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx] / 100.0, dtype=torch.float32),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class CredibilityScorer(nn.Module):
    """
    BERT + regression head.
    Output is sigmoid(linear) * 100 → score in [0, 100].
    """

    def __init__(self, bert_model_name: str = BERT_MODEL_NAME, dropout: float = 0.3):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]   # [CLS]
        dropped   = self.dropout(cls_token)
        logit     = self.head(dropped).squeeze(-1)
        score     = torch.sigmoid(logit)                 # 0–1
        return score                                     # caller multiplies by 100 if needed


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)          # 0–1

        optimiser.zero_grad()
        preds = model(input_ids, attention_mask)            # 0–1
        loss  = criterion(preds, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            preds = model(input_ids, attention_mask)
            loss  = criterion(preds, labels)
            total_loss += loss.item()

            all_preds.extend((preds.cpu().numpy() * 100).tolist())
            all_labels.extend((labels.cpu().numpy() * 100).tolist())

    return (
        total_loss / len(loader),
        np.array(all_preds),
        np.array(all_labels),
    )


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame) -> tuple[list[str], list[float]]:
    """
    Filter to labelled rows, combine title + text, return (texts, labels).
    """
    labelled = df.dropna(subset=["credibility_score"]).copy()
    log.info(f"Labelled rows for training: {len(labelled)} / {len(df)}")

    if len(labelled) < 10:
        raise ValueError(
            "Not enough labelled data. Run backfill.py first: "
            "python -m feature_pipeline.backfill --max 500 --no-bert"
        )

    texts  = (labelled["title"].fillna("") + " " + labelled["text"].fillna("")).tolist()
    labels = labelled["credibility_score"].tolist()
    return texts, labels


# ── Main training function ────────────────────────────────────────────────────

def run_training(
    epochs: int      = 3,
    batch_size: int  = 8,
    lr: float        = 2e-5,
    max_length: int  = 256,
    val_split: float = 0.15,
    test_split: float = 0.10,
) -> str:
    """
    Full training run. Returns the MLflow run ID.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info("Loading features from feature store…")
    df = load_all_features()
    texts, labels = prepare_data(df)

    # Train / val / test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels,
        test_size=(val_split + test_split),
        random_state=RANDOM_SEED,
    )
    val_ratio = val_split / (val_split + test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        random_state=RANDOM_SEED,
    )
    log.info(f"Split → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # ── Tokenizer + datasets ───────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    train_ds = NewsDataset(X_train, y_train, tokenizer, max_length)
    val_ds   = NewsDataset(X_val,   y_val,   tokenizer, max_length)
    test_ds  = NewsDataset(X_test,  y_test,  tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # ── Model, optimiser, loss ────────────────────────────────────────────────
    model     = CredibilityScorer().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("news-credibility")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log.info(f"MLflow run ID: {run_id}")

        # Log hyperparameters
        mlflow.log_params({
            "epochs":       epochs,
            "batch_size":   batch_size,
            "lr":           lr,
            "max_length":   max_length,
            "bert_model":   BERT_MODEL_NAME,
            "train_size":   len(X_train),
            "val_size":     len(X_val),
            "test_size":    len(X_test),
        })

        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimiser, criterion, device)
            val_loss, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)

            val_metrics = compute_metrics(val_labels, val_preds, prefix="val")

            log.info(
                f"Epoch {epoch}/{epochs} — "
                f"train_loss: {train_loss:.4f}  "
                f"val_loss: {val_loss:.4f}  "
                f"val_mae: {val_metrics['val_mae']:.2f}"
            )

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, **val_metrics},
                step=epoch,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                log.info(f"  ↑ New best model (val_loss={val_loss:.4f})")

        # ── Evaluate on test set ───────────────────────────────────────────────
        model.load_state_dict(best_model_state)
        model.to(device)
        _, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)
        test_metrics = compute_metrics(test_labels, test_preds, prefix="test")

        log.info(f"Test metrics: {test_metrics}")
        mlflow.log_metrics(test_metrics)

        # ── Save prediction plot as artifact ───────────────────────────────────
        plot_path = plot_predictions(test_labels, test_preds)
        mlflow.log_artifact(str(plot_path))

        # ── Log model ─────────────────────────────────────────────────────────
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )
        log.info(f"Model logged to MLflow registry as '{MLFLOW_MODEL_NAME}'.")

    return run_id


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train the news credibility scorer.")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--max-length",  type=int,   default=256)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_id = run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )
    log.info(f"Training complete. Run ID: {run_id}")
