from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = PROJECT_ROOT / "data" / "feature_store" / "news_features.parquet"
CURRENT_PATH = PROJECT_ROOT / "data" / "feature_store" / "live_news_features.parquet"

REPORT_DIR = PROJECT_ROOT / "reports" / "evidently"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "data_drift_report.html"


DRIFT_COLUMNS = [
    "text_length",
    "word_count",
    "sentence_count",
    "avg_word_length",
    "exclamation_count",
    "uppercase_ratio",
    "punctuation_ratio",
    "sensational_word_count",
    "title_length",
    "has_url",
]


def main() -> None:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference feature file not found: {REFERENCE_PATH}"
        )

    if not CURRENT_PATH.exists():
        raise FileNotFoundError(
            f"Current live feature file not found: {CURRENT_PATH}"
        )

    reference_df = pd.read_parquet(REFERENCE_PATH)
    current_df = pd.read_parquet(CURRENT_PATH)

    available_cols = [
        col for col in DRIFT_COLUMNS
        if col in reference_df.columns and col in current_df.columns
    ]

    if not available_cols:
        raise RuntimeError("No common drift columns found.")

    reference_data = reference_df[available_cols].dropna()
    current_data = current_df[available_cols].dropna()

    report = Report([
        DataDriftPreset()
    ])

    evaluation = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    evaluation.save_html(str(REPORT_PATH))

    print("Evidently drift report created.")
    print(f"Saved: {REPORT_PATH}")
    print(f"Columns checked: {available_cols}")


if __name__ == "__main__":
    main()