from pathlib import Path
import re
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "combined_news_dataset.parquet"
FEATURE_STORE_DIR = PROJECT_ROOT / "data" / "feature_store"
FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)


SENSATIONAL_WORDS = [
    "breaking", "shocking", "exclusive", "urgent", "viral",
    "exposed", "secret", "scandal", "unbelievable", "alert"
]


def count_sensational_words(text: str) -> int:
    text = str(text).lower()
    return sum(1 for word in SENSATIONAL_WORDS if word in text)


def count_exclamation_marks(text: str) -> int:
    return str(text).count("!")


def uppercase_ratio(text: str) -> float:
    text = str(text)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    uppercase = [c for c in letters if c.isupper()]
    return len(uppercase) / len(letters)


def punctuation_ratio(text: str) -> float:
    text = str(text)
    if len(text) == 0:
        return 0.0
    punct = re.findall(r"[^\w\s]", text)
    return len(punct) / len(text)


def word_count(text: str) -> int:
    return len(str(text).split())


def sentence_count(text: str) -> int:
    sentences = re.split(r"[.!?]+", str(text))
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def avg_word_length(text: str) -> float:
    words = str(text).split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()

    features["text_length"] = features["text"].astype(str).str.len()
    features["word_count"] = features["text"].apply(word_count)
    features["sentence_count"] = features["text"].apply(sentence_count)
    features["avg_word_length"] = features["text"].apply(avg_word_length)
    features["exclamation_count"] = features["text"].apply(count_exclamation_marks)
    features["uppercase_ratio"] = features["text"].apply(uppercase_ratio)
    features["punctuation_ratio"] = features["text"].apply(punctuation_ratio)
    features["sensational_word_count"] = features["text"].apply(count_sensational_words)

    features["title_length"] = features["title"].astype(str).str.len()
    features["has_url"] = features["url"].astype(str).str.len().gt(0).astype(int)

    return features


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}. "
            "Run src/ingestion/load_static_data.py first."
        )

    print("Loading combined dataset...")
    df = pd.read_parquet(INPUT_PATH)

    print(f"Input rows: {len(df)}")

    print("Building features...")
    features = build_features(df)

    output_path = FEATURE_STORE_DIR / "news_features.parquet"
    csv_path = FEATURE_STORE_DIR / "news_features.csv"

    features.to_parquet(output_path, index=False)
    features.to_csv(csv_path, index=False)

    print("Feature pipeline completed.")
    print(f"Rows: {len(features)}")
    print(f"Columns: {list(features.columns)}")
    print(f"Saved parquet: {output_path}")
    print(f"Saved csv: {csv_path}")


if __name__ == "__main__":
    main()