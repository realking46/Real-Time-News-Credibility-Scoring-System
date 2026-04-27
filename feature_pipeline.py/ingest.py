"""
feature_pipeline/ingest.py
──────────────────────────
Fetches raw news articles from three sources:
  1. NewsAPI        (live articles, requires API key)
  2. RSS feeds      (BBC, Reuters — no key needed)
  3. LIAR dataset   (local Kaggle TSV files)

Returns a list of dicts with a unified schema:
  {
    "title":      str,
    "text":       str,
    "source":     str,
    "url":        str,
    "published":  str (ISO 8601),
    "label":      float | None   # credibility 0–100, None if unknown
  }

LIAR dataset setup:
  Download from https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset
  Place the files in:  data/liar/train.tsv
                       data/liar/valid.tsv
                       data/liar/test.tsv
  Or set LIAR_DATA_DIR=/your/path in .env
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"

RSS_FEEDS = {
    "bbc":     "http://feeds.bbci.co.uk/news/rss.xml",
    "reuters": "https://feeds.reuters.com/reuters/topNews",
    "ap":      "https://rsshub.app/apnews/topics/apf-topnews",
}

# LIAR label -> credibility score (0-100)
LIAR_LABEL_MAP = {
    "pants-fire":  0,
    "false":       15,
    "barely-true": 30,
    "half-true":   50,
    "mostly-true": 70,
    "true":        90,
}

# Kaggle TSV has NO header row - column names in order
_LIAR_COL_NAMES = [
    "id", "label", "statement", "subject",
    "speaker", "job", "state", "party",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count", "context",
]

# Map split name -> possible filenames
_LIAR_FILE_MAP = {
    "train":      ["train.tsv"],
    "validation": ["valid.tsv", "val.tsv", "validation.tsv"],
    "test":       ["test.tsv"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(text: str) -> str:
    return " ".join(text.split()) if text else ""


# ── Source 1: NewsAPI ─────────────────────────────────────────────────────────

def fetch_newsapi(page_size: int = 20) -> list:
    if not NEWS_API_KEY:
        log.warning("NEWS_API_KEY not set -- skipping NewsAPI fetch.")
        return []

    params = {"apiKey": NEWS_API_KEY, "language": "en", "pageSize": page_size}
    try:
        resp = requests.get(NEWS_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        log.info(f"NewsAPI: fetched {len(articles)} articles.")
    except requests.RequestException as exc:
        log.error(f"NewsAPI request failed: {exc}")
        return []

    results = []
    for art in articles:
        text = _clean_text(
            (art.get("description") or "") + " " + (art.get("content") or "")
        )
        if not text.strip():
            continue
        results.append({
            "title":     _clean_text(art.get("title") or ""),
            "text":      text,
            "source":    art.get("source", {}).get("name", "newsapi"),
            "url":       art.get("url") or "",
            "published": art.get("publishedAt") or _now_iso(),
            "label":     None,
        })
    return results


# ── Source 2: RSS Feeds ───────────────────────────────────────────────────────

def fetch_rss(max_per_feed: int = 10) -> list:
    results = []
    for source_name, url in RSS_FEEDS.items():
        try:
            feed    = feedparser.parse(url)
            entries = feed.entries[:max_per_feed]
            log.info(f"RSS [{source_name}]: fetched {len(entries)} entries.")
        except Exception as exc:
            log.error(f"RSS [{source_name}] failed: {exc}")
            continue

        for entry in entries:
            text = _clean_text(
                getattr(entry, "summary", "") or getattr(entry, "description", "")
            )
            if not text:
                continue
            published = _now_iso()
            if hasattr(entry, "published"):
                published = entry.published
            results.append({
                "title":     _clean_text(getattr(entry, "title", "")),
                "text":      text,
                "source":    source_name,
                "url":       getattr(entry, "link", ""),
                "published": published,
                "label":     None,
            })
    return results


# ── Source 3: LIAR Dataset (local Kaggle TSV) ─────────────────────────────────

def fetch_liar(split="train", max_rows=500, data_dir=None) -> list:
    """
    Load LIAR dataset from local Kaggle TSV files.

    Parameters
    ----------
    split    : 'train' | 'validation' | 'test'
    max_rows : max rows to return
    data_dir : folder with train.tsv / valid.tsv / test.tsv
               Default: LIAR_DATA_DIR env var, else ./data/liar/
    """
    import pandas as pd

    if data_dir is None:
        data_dir = os.getenv("LIAR_DATA_DIR", "./data/liar")

    base       = Path(data_dir)
    candidates = _LIAR_FILE_MAP.get(split, [f"{split}.tsv"])
    tsv_path   = None

    for fname in candidates:
        candidate = base / fname
        if candidate.exists():
            tsv_path = candidate
            break

    if tsv_path is None:
        log.error(
            f"LIAR TSV not found for split='{split}' in '{base.resolve()}'.\n"
            f"  Tried: {candidates}\n"
            f"  Download from https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset\n"
            f"  Place files in: {base.resolve()}/"
        )
        return []

    try:
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            header=None,
            names=_LIAR_COL_NAMES,
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception as exc:
        log.error(f"Failed to read {tsv_path}: {exc}")
        return []

    df = df.head(max_rows)
    results = []

    for _, row in df.iterrows():
        label_str = str(row.get("label", "")).strip().lower()
        score     = LIAR_LABEL_MAP.get(label_str)

        if score is None:
            log.debug(f"Unrecognised label '{label_str}' -- skipping.")
            continue

        text = _clean_text(str(row.get("statement", "")))
        if not text:
            continue

        results.append({
            "title":     text[:120],
            "text":      text,
            "source":    str(row.get("speaker", "liar-dataset")),
            "url":       "",
            "published": _now_iso(),
            "label":     float(score),
        })

    log.info(f"LIAR [{split}]: loaded {len(results)} labelled rows from {tsv_path}.")
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_all(use_newsapi=True, use_rss=True, use_liar=False, liar_split="train") -> list:
    """Run all enabled sources and return a deduplicated combined list."""
    articles = []

    if use_newsapi:
        articles += fetch_newsapi()
    if use_rss:
        articles += fetch_rss()
    if use_liar:
        articles += fetch_liar(split=liar_split)

    seen_urls = set()
    unique = []
    for art in articles:
        url = art["url"]
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        unique.append(art)

    log.info(f"ingest_all: {len(unique)} unique articles (from {len(articles)} total).")
    return unique


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    articles = ingest_all(use_newsapi=True, use_rss=True, use_liar=False)
    for a in articles[:3]:
        print(f"\n[{a['source']}] {a['title'][:80]}")
        print(f"  text:  {a['text'][:100]}...")
        print(f"  label: {a['label']}")
