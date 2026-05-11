from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.inference.predict import load_latest_model, predict_with_model


LIVE_FEATURE_PATH = PROJECT_ROOT / "data" / "feature_store" / "live_news_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "live_news_predictions.parquet"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "live_news_predictions.csv"


def main() -> None:
    if not LIVE_FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Live feature file not found: {LIVE_FEATURE_PATH}. "
            "Run rss_ingest.py and build_live_features.py first."
        )

    df = pd.read_parquet(LIVE_FEATURE_PATH)

    results = []
    model = load_latest_model()

    for _, row in df.iterrows():
        prediction = predict_with_model(model, row["text"])

        results.append({
            "article_id": row.get("article_id", ""),
            "title": row.get("title", ""),
            "url": row.get("url", ""),
            "source_name": row.get("source_name", ""),
            "published_at": row.get("published_at", ""),
            "prediction_label": prediction["prediction_label"],
            "credibility_score": prediction["credibility_score"],
            "risk_level": prediction["risk_level"],
        })

    output_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(OUTPUT_PATH, index=False)
    output_df.to_csv(OUTPUT_CSV, index=False)

    print("Live news prediction completed.")
    print(f"Rows scored: {len(output_df)}")
    print(f"Saved: {OUTPUT_PATH}")
    print(output_df[["title", "credibility_score", "risk_level"]].head())


if __name__ == "__main__":
    main()