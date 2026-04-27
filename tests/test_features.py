"""
tests/test_features.py
───────────────────────
Unit tests for the feature pipeline.
BERT is disabled in all tests (include_bert=False) so CI stays fast.
"""

import pandas as pd
import pytest

from feature_pipeline.features import (
    _hours_since_published,
    _sensational_word_count,
    _source_reliability,
    _word_count,
    compute_features,
)
from feature_pipeline.ingest import fetch_liar, fetch_rss, ingest_all
from feature_pipeline.store import list_versions, load_latest_features, save_features


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_ARTICLES = [
    {
        "title": "Breaking: Shocking scandal exposed!",
        "text": "This is a breaking exclusive bombshell story about a shocking scandal.",
        "source": "bbc",
        "url": "https://example.com/1",
        "published": "2026-01-01T12:00:00+00:00",
        "label": 80.0,
    },
    {
        "title": "Normal news story",
        "text": "Scientists discovered a new species of deep-sea fish.",
        "source": "reuters",
        "url": "https://example.com/2",
        "published": "2026-01-01T10:00:00+00:00",
        "label": 90.0,
    },
    {
        "title": "Unlabelled live article",
        "text": "Stock markets rose today amid positive earnings reports.",
        "source": "bloomberg",
        "url": "https://example.com/3",
        "published": "2026-01-02T08:00:00+00:00",
        "label": None,
    },
]


# ── Engineered feature unit tests ─────────────────────────────────────────────

def test_word_count_basic():
    assert _word_count("hello world foo") == 3


def test_word_count_empty():
    assert _word_count("") == 0


def test_sensational_word_count_detects_keywords():
    text = "This is a breaking exclusive bombshell story"
    assert _sensational_word_count(text) >= 3


def test_sensational_word_count_clean_text():
    text = "Scientists made a new discovery in marine biology"
    assert _sensational_word_count(text) == 0


def test_source_reliability_known():
    assert _source_reliability("bbc") == 0.95
    assert _source_reliability("reuters") == 0.95
    assert _source_reliability("infowars") == 0.05


def test_source_reliability_unknown_returns_default():
    score = _source_reliability("some-random-blog.com")
    assert 0.0 <= score <= 1.0


def test_hours_since_published_iso():
    # A date far in the past should give a positive number of hours
    hours = _hours_since_published("2020-01-01T00:00:00+00:00")
    assert hours > 0


def test_hours_since_published_empty():
    assert _hours_since_published("") == 0.0


def test_hours_since_published_bad_string():
    # Should not raise — returns 0.0
    assert _hours_since_published("not-a-date") == 0.0


# ── compute_features tests ────────────────────────────────────────────────────

def test_compute_features_returns_dataframe():
    df = compute_features(SAMPLE_ARTICLES, include_bert=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_compute_features_expected_columns():
    df = compute_features(SAMPLE_ARTICLES, include_bert=False)
    expected = [
        "title", "text", "source", "url", "published",
        "word_count", "sensational_word_count", "has_sensational_words",
        "source_reliability_score", "hours_since_published", "credibility_score",
    ]
    for col in expected:
        assert col in df.columns, f"Missing column: {col}"


def test_compute_features_no_bert_columns():
    df = compute_features(SAMPLE_ARTICLES, include_bert=False)
    bert_cols = [c for c in df.columns if c.startswith("bert_emb_")]
    assert len(bert_cols) == 0


def test_compute_features_sensational_flags():
    df = compute_features(SAMPLE_ARTICLES, include_bert=False)
    # First article has sensational words
    assert df.loc[0, "sensational_word_count"] > 0
    assert df.loc[0, "has_sensational_words"] == 1
    # Second article does not
    assert df.loc[1, "sensational_word_count"] == 0
    assert df.loc[1, "has_sensational_words"] == 0


def test_compute_features_preserves_label():
    df = compute_features(SAMPLE_ARTICLES, include_bert=False)
    assert df.loc[0, "credibility_score"] == 80.0
    assert df.loc[1, "credibility_score"] == 90.0
    assert pd.isna(df.loc[2, "credibility_score"])


def test_compute_features_empty_list():
    df = compute_features([], include_bert=False)
    assert df.empty


# ── store.py tests ────────────────────────────────────────────────────────────

def test_save_and_load_features(tmp_path, monkeypatch):
    """Save a DataFrame to a temp directory and reload it."""
    import feature_pipeline.store as store_module

    monkeypatch.setattr(store_module, "FEATURE_STORE_PATH", tmp_path)

    df_in = compute_features(SAMPLE_ARTICLES, include_bert=False)
    save_features(df_in)

    df_out = load_latest_features()
    assert len(df_out) == len(df_in)
    assert list(df_out.columns) == list(df_in.columns)


def test_list_versions(tmp_path, monkeypatch):
    import feature_pipeline.store as store_module

    monkeypatch.setattr(store_module, "FEATURE_STORE_PATH", tmp_path)

    df = compute_features(SAMPLE_ARTICLES, include_bert=False)
    save_features(df)

    versions = list_versions()
    assert len(versions) == 1
    assert versions[0]["rows"] == 3


def test_save_empty_dataframe_raises(tmp_path, monkeypatch):
    import feature_pipeline.store as store_module

    monkeypatch.setattr(store_module, "FEATURE_STORE_PATH", tmp_path)

    with pytest.raises(ValueError):
        save_features(pd.DataFrame())


# ── ingest.py tests (no network calls) ───────────────────────────────────────

def test_ingest_all_no_sources():
    """With all sources disabled, result should be empty."""
    articles = ingest_all(use_newsapi=False, use_rss=False, use_liar=False)
    assert articles == []


def test_ingest_all_deduplication():
    """Duplicate URLs should be removed."""
    from feature_pipeline.ingest import ingest_all as _ingest

    # Monkey-patch fetch functions to return duplicates
    import feature_pipeline.ingest as ingest_module

    dup_articles = [
        {"title": "A", "text": "text a", "source": "bbc", "url": "https://x.com/1",
         "published": "", "label": None},
        {"title": "A", "text": "text a", "source": "bbc", "url": "https://x.com/1",
         "published": "", "label": None},
    ]

    original_newsapi = ingest_module.fetch_newsapi
    original_rss = ingest_module.fetch_rss

    ingest_module.fetch_newsapi = lambda **kw: dup_articles
    ingest_module.fetch_rss = lambda **kw: []

    try:
        result = ingest_module.ingest_all(use_newsapi=True, use_rss=False, use_liar=False)
        assert len(result) == 1
    finally:
        ingest_module.fetch_newsapi = original_newsapi
        ingest_module.fetch_rss = original_rss
