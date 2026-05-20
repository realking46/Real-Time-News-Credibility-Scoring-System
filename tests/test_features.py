from src.features.build_features import (
    word_count,
    sentence_count,
    uppercase_ratio,
    punctuation_ratio,
    count_sensational_words,
)


def test_word_count():
    text = "This is a test sentence"
    assert word_count(text) == 5


def test_sentence_count():
    text = "This is sentence one. This is sentence two!"
    assert sentence_count(text) == 2


def test_uppercase_ratio():
    text = "ABCdef"
    assert uppercase_ratio(text) == 0.5


def test_punctuation_ratio():
    text = "Hello!!!"
    assert punctuation_ratio(text) > 0


def test_sensational_words():
    text = "Breaking shocking news exposed today"
    assert count_sensational_words(text) >= 3