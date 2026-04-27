"""
feature_pipeline/features.py
─────────────────────────────
Transforms raw articles (from ingest.py) into a feature DataFrame.

Features computed per article:
  Text-based
    • bert_emb_0 … bert_emb_767   768-dim [CLS] embedding from bert-base-uncased

  Engineered
    • word_count                  total words in article text
    • sensational_word_count      count of hype/clickbait keywords
    • has_sensational_words       binary flag
    • source_reliability_score    0.0–1.0 lookup by domain name
    • hours_since_published       recency signal (negative = future/bad parse → 0)

  Label
    • credibility_score           0–100 float (None for unlabelled live articles)

Usage:
    from feature_pipeline.features import compute_features
    df = compute_features(articles)   # articles = output of ingest_all()
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Sensational keyword list ───────────────────────────────────────────────────

SENSATIONAL_WORDS = {
    "breaking", "exclusive", "shocking", "bombshell", "urgent", "alert",
    "unprecedented", "unbelievable", "scandal", "exposed", "leaked",
    "secret", "conspiracy", "miracle", "disaster", "crisis", "warning",
    "danger", "outrage", "backlash", "viral", "clickbait", "hoax",
}

# ── Source reliability lookup ──────────────────────────────────────────────────
# Score 0.0 (unreliable) → 1.0 (highly reliable)
# Unlisted sources get a default of 0.5

SOURCE_RELIABILITY: dict[str, float] = {
    # Highly reliable
    "bbc": 0.95, "reuters": 0.95, "ap": 0.93, "apnews": 0.93,
    "associated press": 0.93, "bbc news": 0.95,
    "the guardian": 0.88, "new york times": 0.87, "nytimes": 0.87,
    "washington post": 0.86, "the economist": 0.90, "bloomberg": 0.88,
    "financial times": 0.89, "npr": 0.87, "pbs": 0.86,
    # Moderate
    "cnn": 0.72, "foxnews": 0.60, "fox news": 0.60,
    "daily mail": 0.55, "the sun": 0.50, "new york post": 0.58,
    "breitbart": 0.35, "buzzfeed": 0.60,
    # Known unreliable / satire / fake
    "infowars": 0.05, "naturalnews": 0.05, "theonion": 0.10,
    "beforeitsnews": 0.05, "worldnewsdailyreport": 0.05,
    # LIAR dataset speakers — map to neutral
    "liar-dataset": 0.50,
}

_DEFAULT_RELIABILITY = 0.50


# ── BERT embedding (lazy-loaded) ──────────────────────────────────────────────

_tokenizer = None
_model = None


def _load_bert():
    """Load BERT tokenizer + model once; reuse across calls."""
    global _tokenizer, _model
    if _tokenizer is None:
        log.info("Loading bert-base-uncased (first call — may take a moment)…")
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        _model = AutoModel.from_pretrained("bert-base-uncased")
        _model.eval()
        log.info("BERT loaded.")
    return _tokenizer, _model


def _bert_embedding(text: str, max_length: int = 512) -> np.ndarray:
    """
    Returns a 768-dim numpy array — the [CLS] token embedding.
    Falls back to zeros on error.
    """
    import torch

    try:
        tokenizer, model = _load_bert()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        with torch.no_grad():
            outputs = model(**inputs)
        cls_vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return cls_vector.astype(np.float32)
    except Exception as exc:
        log.error(f"BERT embedding failed: {exc}")
        return np.zeros(768, dtype=np.float32)


# ── Engineered feature helpers ────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def _sensational_word_count(text: str) -> int:
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    return len(tokens & SENSATIONAL_WORDS)


def _source_reliability(source: str) -> float:
    key = source.lower().strip()
    # Try exact match first, then partial match
    if key in SOURCE_RELIABILITY:
        return SOURCE_RELIABILITY[key]
    for known, score in SOURCE_RELIABILITY.items():
        if known in key or key in known:
            return score
    return _DEFAULT_RELIABILITY


def _hours_since_published(published: str) -> float:
    """Parse ISO-8601 or RFC-2822 published string; return hours since then."""
    if not published:
        return 0.0
    try:
        # Try ISO 8601
        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError:
        try:
            # Try RFC-2822 (common in RSS)
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(published)
        except Exception:
            return 0.0

    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta_hours = (now - dt).total_seconds() / 3600
    return max(0.0, round(delta_hours, 2))


# ── Main function ─────────────────────────────────────────────────────────────

def compute_features(
    articles: list[dict],
    include_bert: bool = True,
) -> pd.DataFrame:
    """
    Transform a list of article dicts into a feature DataFrame.

    Parameters
    ----------
    articles     : output of ingest_all()
    include_bert : set False to skip BERT (faster for testing)

    Returns
    -------
    pd.DataFrame with columns:
        title, text, source, url, published,
        word_count, sensational_word_count, has_sensational_words,
        source_reliability_score, hours_since_published,
        credibility_score,
        bert_emb_0 … bert_emb_767  (if include_bert=True)
    """
    if not articles:
        log.warning("compute_features received an empty article list.")
        return pd.DataFrame()

    rows = []
    for idx, art in enumerate(articles):
        text = art.get("text", "")
        source = art.get("source", "")

        wc = _word_count(text)
        swc = _sensational_word_count(text)
        reliability = _source_reliability(source)
        hours = _hours_since_published(art.get("published", ""))
        label: Optional[float] = art.get("label")

        row: dict = {
            "title": art.get("title", ""),
            "text": text,
            "source": source,
            "url": art.get("url", ""),
            "published": art.get("published", ""),
            "word_count": wc,
            "sensational_word_count": swc,
            "has_sensational_words": int(swc > 0),
            "source_reliability_score": reliability,
            "hours_since_published": hours,
            "credibility_score": label,
        }

        if include_bert:
            emb = _bert_embedding(text)
            for i, val in enumerate(emb):
                row[f"bert_emb_{i}"] = float(val)

        rows.append(row)

        if (idx + 1) % 10 == 0:
            log.info(f"  Processed {idx + 1}/{len(articles)} articles…")

    df = pd.DataFrame(rows)
    log.info(f"compute_features: produced DataFrame with shape {df.shape}.")
    return df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from feature_pipeline.ingest import ingest_all

    articles = ingest_all(use_newsapi=False, use_rss=True, use_liar=False)
    df = compute_features(articles[:5], include_bert=True)
    print(df[["title", "word_count", "sensational_word_count", "source_reliability_score"]].to_string())
