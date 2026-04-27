"""
feature_pipeline/store.py
──────────────────────────
Saves feature DataFrames to the feature store as versioned Parquet files.

Layout on disk:
    feature_store/
        features_v1_20260101_120000.parquet
        features_v1_20260102_120000.parquet
        ...
        latest.parquet    ← symlink / copy of the most recent file

Public API:
    save_features(df)        → saves timestamped Parquet + updates latest
    load_latest_features()   → loads latest.parquet as a DataFrame
    load_all_features()      → concatenates ALL Parquet files in the store
    list_versions()          → returns sorted list of (filename, timestamp, rows)
"""

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

FEATURE_STORE_PATH = Path(os.getenv("FEATURE_STORE_PATH", "./feature_store"))
VERSION_PREFIX = "features_v1"
LATEST_FILE = "latest.parquet"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_store() -> Path:
    FEATURE_STORE_PATH.mkdir(parents=True, exist_ok=True)
    return FEATURE_STORE_PATH


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_features(df: pd.DataFrame) -> Path:
    """
    Persist a feature DataFrame as a timestamped Parquet file and
    overwrite latest.parquet with the same data.

    Returns the path of the newly written versioned file.
    """
    if df is None or df.empty:
        raise ValueError("Cannot save an empty or None DataFrame.")

    store = _ensure_store()
    ts = _timestamp()
    versioned_name = f"{VERSION_PREFIX}_{ts}.parquet"
    versioned_path = store / versioned_name
    latest_path = store / LATEST_FILE

    df.to_parquet(versioned_path, index=False, compression="snappy")
    log.info(f"Saved {len(df)} rows → {versioned_path}")

    # Overwrite latest
    shutil.copy2(versioned_path, latest_path)
    log.info(f"Updated latest → {latest_path}")

    return versioned_path


# ── Load ──────────────────────────────────────────────────────────────────────

def load_latest_features() -> pd.DataFrame:
    """Load the most recently saved feature file."""
    store = _ensure_store()
    latest_path = store / LATEST_FILE

    if not latest_path.exists():
        raise FileNotFoundError(
            f"No latest.parquet found in {store}. "
            "Run the feature pipeline or backfill first."
        )

    df = pd.read_parquet(latest_path)
    log.info(f"Loaded latest features: {df.shape} from {latest_path}")
    return df


def load_all_features() -> pd.DataFrame:
    """
    Concatenate ALL versioned Parquet files in the store.
    Useful for training on the full historical dataset.
    Deduplicates by URL (non-empty URLs only).
    """
    store = _ensure_store()
    parquet_files = sorted(store.glob(f"{VERSION_PREFIX}_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No versioned Parquet files found in {store}.")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    combined = pd.concat(dfs, ignore_index=True)

    # Deduplicate by URL
    has_url = combined["url"] != ""
    no_url = combined[~has_url]
    with_url = combined[has_url].drop_duplicates(subset=["url"], keep="last")
    combined = pd.concat([with_url, no_url], ignore_index=True)

    log.info(f"Loaded all features: {combined.shape} from {len(parquet_files)} files.")
    return combined


# ── Inspect ───────────────────────────────────────────────────────────────────

def list_versions() -> list[dict]:
    """Return metadata for every versioned Parquet in the store."""
    store = _ensure_store()
    results = []

    for path in sorted(store.glob(f"{VERSION_PREFIX}_*.parquet")):
        try:
            df = pd.read_parquet(path, columns=["title"])  # cheap — only one col
            rows = len(df)
        except Exception:
            rows = -1

        stat = path.stat()
        results.append(
            {
                "filename": path.name,
                "rows": rows,
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    versions = list_versions()
    if versions:
        print(f"\nFeature store at: {FEATURE_STORE_PATH.resolve()}")
        print(f"{'Filename':<45} {'Rows':>6}  {'MB':>6}  {'Modified'}")
        print("-" * 80)
        for v in versions:
            print(f"{v['filename']:<45} {v['rows']:>6}  {v['size_mb']:>6}  {v['modified']}")
    else:
        print(f"Feature store is empty: {FEATURE_STORE_PATH.resolve()}")
