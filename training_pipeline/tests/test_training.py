"""
tests/test_training.py
───────────────────────
Unit tests for the training pipeline.
All tests avoid actual BERT / MLflow calls to stay fast in CI.
"""

import numpy as np
import pytest

from training_pipeline.evaluate import compute_metrics, score_to_risk_label


# ── score_to_risk_label ───────────────────────────────────────────────────────

@pytest.mark.parametrize("score,expected", [
    (0,   "High"),
    (20,  "High"),
    (39,  "High"),
    (40,  "Medium"),
    (55,  "Medium"),
    (69,  "Medium"),
    (70,  "Low"),
    (85,  "Low"),
    (100, "Low"),
])
def test_score_to_risk_label(score, expected):
    assert score_to_risk_label(score) == expected


def test_score_to_risk_label_clamps_below_zero():
    assert score_to_risk_label(-10) == "High"


def test_score_to_risk_label_clamps_above_100():
    assert score_to_risk_label(110) == "Low"


# ── compute_metrics ───────────────────────────────────────────────────────────

def test_compute_metrics_perfect_predictions():
    y = np.array([10.0, 50.0, 90.0])
    metrics = compute_metrics(y, y, prefix="val")
    assert metrics["val_mae"]  == 0.0
    assert metrics["val_rmse"] == 0.0
    assert metrics["val_r2"]   == 1.0
    assert metrics["val_risk_accuracy"] == 1.0


def test_compute_metrics_returns_all_keys():
    y_true = np.array([30.0, 60.0, 80.0])
    y_pred = np.array([35.0, 55.0, 75.0])
    metrics = compute_metrics(y_true, y_pred, prefix="test")
    for key in ["test_mae", "test_rmse", "test_r2", "test_risk_accuracy"]:
        assert key in metrics


def test_compute_metrics_no_prefix():
    y = np.array([50.0, 60.0, 70.0])
    metrics = compute_metrics(y, y)
    assert "mae" in metrics
    assert "rmse" in metrics


def test_compute_metrics_mae_positive():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([100.0, 0.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mae"] == 100.0


def test_compute_metrics_risk_accuracy_partial():
    # true: High(10), Low(80)  pred: Medium(50), Low(80)
    y_true = np.array([10.0, 80.0])
    y_pred = np.array([50.0, 80.0])
    metrics = compute_metrics(y_true, y_pred)
    # 1 out of 2 correct = 0.5
    assert metrics["risk_accuracy"] == pytest.approx(0.5)


def test_compute_metrics_r2_constant_target():
    # When all true values are the same, ss_tot = 0 → r2 = 0 (not NaN)
    y_true = np.array([50.0, 50.0, 50.0])
    y_pred = np.array([48.0, 52.0, 50.0])
    metrics = compute_metrics(y_true, y_pred)
    assert isinstance(metrics["r2"], float)
    assert not np.isnan(metrics["r2"])
