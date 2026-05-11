from src.ingestion.rss_ingest import make_article_id


def test_make_article_id_consistent():
    id1 = make_article_id("https://example.com/article", "Test Title")
    id2 = make_article_id("https://example.com/article", "Test Title")

    assert id1 == id2


def test_make_article_id_different_for_different_titles():
    id1 = make_article_id("https://example.com/article", "Title One")
    id2 = make_article_id("https://example.com/article", "Title Two")

    assert id1 != id2