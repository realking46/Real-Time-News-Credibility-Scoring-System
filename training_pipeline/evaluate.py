"""
training_pipeline/evaluate.py
──────────────────────────────
Evaluation utilities shared by train.py and register.py.

Public API:
    compute_metrics(y_true, y_pred, prefix)  → dict of metrics
    score_to_risk_label(score)               → "Low" | "Medium" | "High"
    plot_predictions(y_true, y_pred)         → Path to saved PNG
"""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ── Risk label thresholds ──────────────────────────────────────────────────────
# Score 0–100: lower = less credible = higher risk
RISK_THRESHOLDS = {
    "High":   (0,  40),   # 0–39   → High risk
    "Medium": (40, 70),   # 40–69  → Medium risk
    "Low":    (70, 101),  # 70–100 → Low risk
}


def score_to_risk_label(score: float) -> str:
    """
    Convert a 0–100 credibility score to a risk label.

    Low    → credible (score 70–100)
    Medium → uncertain (score 40–69)
    High   → likely unreliable (score 0–39)
    """
    score = max(0.0, min(100.0, float(score)))
    if score >= 70:
        return "Low"
    elif score >= 40:
        return "Medium"
    else:
        return "High"


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute regression metrics for credibility score predictions.

    Returns
    -------
    dict with keys (prefixed if prefix given):
        mae   — Mean Absolute Error
        rmse  — Root Mean Squared Error
        r2    — R² coefficient of determination
        risk_accuracy — % of articles whose risk label matches ground truth
    """
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Risk label accuracy
    true_labels = [score_to_risk_label(s) for s in y_true]
    pred_labels = [score_to_risk_label(s) for s in y_pred]
    risk_acc = float(np.mean([t == p for t, p in zip(true_labels, pred_labels)]))

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}mae":           round(mae, 4),
        f"{p}rmse":          round(rmse, 4),
        f"{p}r2":            round(r2, 4),
        f"{p}risk_accuracy": round(risk_acc, 4),
    }


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str = "/tmp",
) -> Path:
    """
    Scatter plot of predicted vs actual credibility scores.
    Saved as a PNG and returned as a Path (for MLflow artifact logging).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend (safe in Docker/CI)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))

        ax.scatter(y_true, y_pred, alpha=0.45, s=20, color="#3B82F6", label="Predictions")

        # Perfect prediction line
        lo, hi = 0, 100
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect")

        ax.set_xlabel("Actual Credibility Score", fontsize=12)
        ax.set_ylabel("Predicted Credibility Score", fontsize=12)
        ax.set_title("Predicted vs Actual Credibility Score", fontsize=13)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.legend()
        ax.grid(alpha=0.3)

        path = Path(save_dir) / "predictions_scatter.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Prediction plot saved: {path}")
        return path

    except Exception as exc:
        log.warning(f"Could not generate prediction plot: {exc}")
        # Return a dummy path so MLflow doesn't crash
        path = Path(save_dir) / "predictions_scatter.png"
        path.touch()
        return path


# ── CLI (quick sanity check) ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate predictions with random noise
    rng = np.random.default_rng(42)
    y_true = rng.uniform(0, 100, size=200)
    y_pred = np.clip(y_true + rng.normal(0, 10, size=200), 0, 100)

    metrics = compute_metrics(y_true, y_pred, prefix="test")
    print("\nSample metrics (random noise baseline):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nRisk label examples:")
    for score in [5, 35, 55, 75, 95]:
        print(f"  score={score:3d}  →  {score_to_risk_label(score)} risk")
