"""
feature_pipeline/backfill.py
─────────────────────────────
One-time script that loads the full LIAR dataset (train + validation + test),
computes features, and saves them to the feature store.

Run this ONCE before kicking off the training pipeline.

Usage:
    python -m feature_pipeline.backfill
    python -m feature_pipeline.backfill --no-bert   # skip BERT (faster for testing)
    python -m feature_pipeline.backfill --max 200   # limit rows per split
"""

import argparse
import logging
import sys

import pandas as pd

from feature_pipeline.features import compute_features
from feature_pipeline.ingest import fetch_liar
from feature_pipeline.store import list_versions, save_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LIAR_SPLITS = ["train", "validation", "test"]


def run_backfill(max_rows_per_split: int = 1000, include_bert: bool = True) -> None:
    """
    Load all LIAR splits, compute features, and save to the feature store.
    """
    log.info("=" * 60)
    log.info("Starting LIAR backfill…")
    log.info(f"  max_rows_per_split = {max_rows_per_split}")
    log.info(f"  include_bert       = {include_bert}")
    log.info("=" * 60)

    all_articles: list[dict] = []

    for split in LIAR_SPLITS:
        log.info(f"Fetching LIAR split: {split}")
        articles = fetch_liar(split=split, max_rows=max_rows_per_split)
        log.info(f"  → {len(articles)} articles from '{split}'")
        all_articles.extend(articles)

    if not all_articles:
        log.error("No articles fetched. Check your internet connection and datasets install.")
        sys.exit(1)

    log.info(f"Total articles to process: {len(all_articles)}")
    log.info("Computing features (this may take a while with BERT enabled)…")

    df: pd.DataFrame = compute_features(all_articles, include_bert=include_bert)

    # Drop rows with no credibility_score (shouldn't happen for LIAR, but safety check)
    before = len(df)
    df = df.dropna(subset=["credibility_score"])
    after = len(df)
    if before != after:
        log.warning(f"Dropped {before - after} rows with missing credibility_score.")

    log.info(f"Feature DataFrame ready: {df.shape}")
    log.info(f"  credibility_score stats:\n{df['credibility_score'].describe().to_string()}")

    saved_path = save_features(df)
    log.info(f"Backfill complete! Saved to: {saved_path}")

    # Summary
    versions = list_versions()
    log.info(f"Feature store now has {len(versions)} versioned file(s).")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill feature store with LIAR dataset.")
    parser.add_argument(
        "--max",
        type=int,
        default=1000,
        metavar="N",
        help="Max rows per LIAR split (default: 1000).",
    )
    parser.add_argument(
        "--no-bert",
        action="store_true",
        help="Skip BERT embedding computation (much faster, use for testing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_backfill(
        max_rows_per_split=args.max,
        include_bert=not args.no_bert,
    )
