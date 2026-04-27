"""
feature_pipeline/ingest.py
──────────────────────────
Fetches raw news articles from three sources:
  1. NewsAPI        (live articles, requires API key)
  2. RSS feeds      (BBC, Reuters — no key needed)
  3. LIAR dataset   (static, labeled — via HuggingFace datasets)

Returns a list of dicts with a unified schema:
  {
    "title":      str,
    "text":       str,
    "source":     str,
    "url":        str,
    "published":  str (ISO 8601),
    "label":      float | None   # credibility 0–100, None if unknown
  }
"""

import logging
import os
from datetime import datetime, timezone

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
    "bbc": "http://feeds.bbci.co.uk/news/rss.xml",
    "reuters": "https://feeds.reuters.com/reuters/topNews",
    "ap": "https://rsshub.app/apnews/topics/apf-topnews",
}

# LIAR dataset label mapping → credibility score (0–100)
# Original labels: pants-fire, false, barely-true, half-true, mostly-true, true
LIAR_LABEL_MAP = {
    "pants-fire": 0,
    "false": 15,
    "barely-true": 30,
    "half-true": 50,
    "mostly-true": 70,
    "true": 90,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(text: str) -> str:
    """Strip excessive whitespace."""
    return " ".join(text.split()) if text else ""


# ── Source 1: NewsAPI ─────────────────────────────────────────────────────────

def fetch_newsapi(page_size: int = 20) -> list[dict]:
    """
    Fetch top headlines from NewsAPI.
    Returns empty list (with a warning) if the key is missing or the call fails.
    """
    if not NEWS_API_KEY:
        log.warning("NEWS_API_KEY not set — skipping NewsAPI fetch.")
        return []

    params = {
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "pageSize": page_size,
    }

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
        text = _clean_text((art.get("description") or "") + " " + (art.get("content") or ""))
        if not text.strip():
            continue
        results.append(
            {
                "title": _clean_text(art.get("title") or ""),
                "text": text,
                "source": art.get("source", {}).get("name", "newsapi"),
                "url": art.get("url") or "",
                "published": art.get("publishedAt") or _now_iso(),
                "label": None,  # live articles have no ground-truth label
            }
        )
    return results


# ── Source 2: RSS Feeds ───────────────────────────────────────────────────────

def fetch_rss(max_per_feed: int = 10) -> list[dict]:
    """Parse a set of RSS feeds and return articles in the unified schema."""
    results = []

    for source_name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            entries = feed.entries[:max_per_feed]
            log.info(f"RSS [{source_name}]: fetched {len(entries)} entries.")
        except Exception as exc:
            log.error(f"RSS [{source_name}] failed: {exc}")
            continue

        for entry in entries:
            text = _clean_text(
                getattr(entry, "summary", "")
                or getattr(entry, "description", "")
            )
            if not text:
                continue

            published = _now_iso()
            if hasattr(entry, "published"):
                published = entry.published  # raw string; normalised later

            results.append(
                {
                    "title": _clean_text(getattr(entry, "title", "")),
                    "text": text,
                    "source": source_name,
                    "url": getattr(entry, "link", ""),
                    "published": published,
                    "label": None,
                }
            )

    return results


# ── Source 3: LIAR Dataset (HuggingFace) ─────────────────────────────────────

def fetch_liar(split: str = "train", max_rows: int = 500) -> list[dict]:
    """
    Load the LIAR dataset from HuggingFace.
    Converts its 6-class label to a 0–100 credibility score.

    split: 'train' | 'validation' | 'test'
    """
    try:
        from datasets import load_dataset  # heavy import — only when needed
    except ImportError:
        log.error("HuggingFace `datasets` not installed. Run: pip install datasets")
        return []

    try:
        dataset = load_dataset("liar", split=split, trust_remote_code=True)
    except Exception as exc:
        log.error(f"Failed to load LIAR dataset: {exc}")
        return []

    results = []
    for row in dataset.select(range(min(max_rows, len(dataset)))):
        label_str = row.get("label", "")
        score = LIAR_LABEL_MAP.get(label_str)   # None if unrecognised
        text = _clean_text(row.get("statement", ""))
        if not text:
            continue

        results.append(
            {
                "title": text[:120],          # LIAR has no title — use truncated statement
                "text": text,
                "source": row.get("speaker", "liar-dataset"),
                "url": "",
                "published": _now_iso(),
                "label": float(score) if score is not None else None,
            }
        )

    log.info(f"LIAR [{split}]: loaded {len(results)} rows.")
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_all(
    use_newsapi: bool = True,
    use_rss: bool = True,
    use_liar: bool = False,       # False by default — use backfill.py for bulk load
    liar_split: str = "train",
) -> list[dict]:
    """
    Run all enabled sources and return a combined deduplicated list.
    Deduplication is URL-based (empty URLs are kept as-is).
    """
    articles: list[dict] = []

    if use_newsapi:
        articles += fetch_newsapi()
    if use_rss:
        articles += fetch_rss()
    if use_liar:
        articles += fetch_liar(split=liar_split)

    # Deduplicate by URL (keep first occurrence)
    seen_urls: set[str] = set()
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


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    articles = ingest_all(use_newsapi=True, use_rss=True, use_liar=False)
    for a in articles[:3]:
        print(f"\n[{a['source']}] {a['title'][:80]}")
        print(f"  text preview: {a['text'][:120]}...")
        print(f"  label: {a['label']}")
