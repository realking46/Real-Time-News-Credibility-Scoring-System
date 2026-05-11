from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LIAR_DIR = PROJECT_ROOT / "data" / "raw" / "liar"
FAKENEWSNET_DIR = PROJECT_ROOT / "data" / "raw" / "fakenewsnet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
    "context",
]


def map_liar_label(label: str) -> str:
    real_labels = {"true", "mostly-true", "half-true"}
    fake_labels = {"barely-true", "false", "pants-fire"}

    label = str(label).strip().lower()

    if label in real_labels:
        return "real"
    if label in fake_labels:
        return "fake"

    return "unknown"


def load_liar_split(filename: str, split: str) -> pd.DataFrame:
    path = LIAR_DIR / filename

    if not path.exists():
        print(f"[WARNING] Missing LIAR file: {path}")
        return pd.DataFrame()

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=LIAR_COLUMNS,
        quoting=3,
        on_bad_lines="skip",
    )

    output = pd.DataFrame()
    output["text"] = df["statement"].astype(str)
    output["title"] = ""
    output["label"] = df["label"].apply(map_liar_label)
    output["source_dataset"] = "liar"
    output["split"] = split
    output["source_name"] = df["speaker"].astype(str)
    output["date"] = ""
    output["url"] = ""
    output["metadata"] = df[
        ["subject", "speaker_job", "state", "party", "context"]
    ].astype(str).agg(" | ".join, axis=1)

    return output


def load_liar_dataset() -> pd.DataFrame:
    parts = [
        load_liar_split("train.tsv", "train"),
        load_liar_split("valid.tsv", "valid"),
        load_liar_split("test.tsv", "test"),
    ]

    parts = [p for p in parts if not p.empty]

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def load_fakenewsnet_file(filename: str, label: str) -> pd.DataFrame:
    path = FAKENEWSNET_DIR / filename

    if not path.exists():
        print(f"[WARNING] Missing FakeNewsNet file: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, on_bad_lines="skip")

    output = pd.DataFrame()

    if "text" in df.columns:
        output["text"] = df["text"].astype(str)
    elif "content" in df.columns:
        output["text"] = df["content"].astype(str)
    else:
        raise ValueError(f"No text/content column found in {filename}")

    if "title" in df.columns:
        output["title"] = df["title"].astype(str)
    else:
        output["title"] = ""

    output["label"] = label
    output["source_dataset"] = "fakenewsnet"
    output["split"] = "train"

    if "source" in df.columns:
        output["source_name"] = df["source"].astype(str)
    else:
        output["source_name"] = ""

    if "publish_date" in df.columns:
        output["date"] = df["publish_date"].astype(str)
    elif "date" in df.columns:
        output["date"] = df["date"].astype(str)
    else:
        output["date"] = ""

    if "url" in df.columns:
        output["url"] = df["url"].astype(str)
    else:
        output["url"] = ""

    output["metadata"] = ""

    return output


def load_fakenewsnet_dataset() -> pd.DataFrame:
    files = [
        ("PolitiFact_fake_news_content.csv", "fake"),
        ("PolitiFact_real_news_content.csv", "real"),
        ("GossipCop_fake_news_content.csv", "fake"),
        ("GossipCop_real_news_content.csv", "real"),
    ]

    parts = []

    for filename, label in files:
        try:
            part = load_fakenewsnet_file(filename, label)
            if not part.empty:
                parts.append(part)
        except Exception as e:
            print(f"[WARNING] Could not load {filename}: {e}")

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    df = df[df["label"].isin(["real", "fake"])]
    df = df[df["text"].str.len() > 20]

    df = df.drop_duplicates(subset=["text"])
    df = df.reset_index(drop=True)

    df["label_id"] = df["label"].map({"fake": 0, "real": 1})

    return df


def main() -> None:
    print("Loading LIAR dataset...")
    liar_df = load_liar_dataset()
    print(f"LIAR rows: {len(liar_df)}")

    print("Loading FakeNewsNet dataset...")
    fakenewsnet_df = load_fakenewsnet_dataset()
    print(f"FakeNewsNet rows: {len(fakenewsnet_df)}")

    datasets = [df for df in [liar_df, fakenewsnet_df] if not df.empty]

    if not datasets:
        raise RuntimeError("No datasets found. Check data/raw folders.")

    combined = pd.concat(datasets, ignore_index=True)
    combined = clean_dataset(combined)

    output_path = OUTPUT_DIR / "combined_news_dataset.parquet"
    csv_path = OUTPUT_DIR / "combined_news_dataset.csv"

    combined.to_parquet(output_path, index=False)
    combined.to_csv(csv_path, index=False)

    print("Combined dataset created successfully.")
    print(f"Rows: {len(combined)}")
    print(f"Labels:\n{combined['label'].value_counts()}")
    print(f"Saved parquet: {output_path}")
    print(f"Saved csv: {csv_path}")


if __name__ == "__main__":
    main()