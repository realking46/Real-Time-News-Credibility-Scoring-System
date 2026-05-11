from pathlib import Path
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "live_news_predictions.csv"
MONITORING_DIR = PROJECT_ROOT / "reports" / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}. "
            "Run live prediction pipeline first."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    total_articles = len(df)
    risk_distribution = df["risk_level"].value_counts().to_dict()
    label_distribution = df["prediction_label"].value_counts().to_dict()
    avg_score = df["credibility_score"].mean()

    report = f"""
# Live News Prediction Monitoring Report

Generated at: {datetime.now().isoformat()}

## Summary

Total articles scored: {total_articles}

Average credibility score: {avg_score:.2f}

## Risk Level Distribution

{risk_distribution}

## Prediction Label Distribution

{label_distribution}

## Notes

This report monitors the output distribution of the live inference pipeline.
It helps detect prediction drift, such as the model suddenly predicting too many articles as fake or real.

For the final project report, this supports the monitoring requirement together with MLflow tracking.
"""

    report_path = MONITORING_DIR / "prediction_monitoring_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("Monitoring report created.")
    print(f"Saved: {report_path}")
    print(report)


if __name__ == "__main__":
    main()