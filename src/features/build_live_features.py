from pathlib import Path
import pandas as pd

from src.features.build_features import build_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LIVE_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "live" / "live_news_latest.parquet"
FEATURE_STORE_DIR = PROJECT_ROOT / "data" / "feature_store"
FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

LIVE_FEATURE_PATH = FEATURE_STORE_DIR / "live_news_features.parquet"
LIVE_FEATURE_CSV = FEATURE_STORE_DIR / "live_news_features.csv"


def main() -> None:
    if not LIVE_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Live data not found: {LIVE_INPUT_PATH}. "
            "Run src/ingestion/rss_ingest.py first."
        )

    print("Loading live RSS data...")
    df = pd.read_parquet(LIVE_INPUT_PATH)

    print(f"Live rows: {len(df)}")

    print("Building live features...")
    features = build_features(df)

    features.to_parquet(LIVE_FEATURE_PATH, index=False)
    features.to_csv(LIVE_FEATURE_CSV, index=False)

    print("Live feature pipeline completed.")
    print(f"Saved parquet: {LIVE_FEATURE_PATH}")
    print(f"Saved csv: {LIVE_FEATURE_CSV}")
    print(f"Rows: {len(features)}")


if __name__ == "__main__":
    main()