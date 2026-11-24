"""Tests for URL/content-hash duplicate detection."""

from __future__ import annotations

from app.ingestion.deduplication import DeduplicationIndex
from app.schemas.article import Article


def _article(
    title: str,
    *,
    url: str = "",
    article_id: str = "",
    source_type: str = "mock",
    description: str = "",
) -> Article:
    return Article(
        article_id=article_id,
        title=title,
        description=description,
        url=url,
        source_type=source_type,  # type: ignore[arg-type]
    )


def test_url_duplicate_is_detected_after_normalization() -> None:
    index = DeduplicationIndex()
    index.add(_article("First copy", url="https://www.example.com/story/?utm_source=rss"))

    duplicate = index.check(
        _article("Different headline", url="https://example.com/story#section")
    )
    assert duplicate is not None
    assert duplicate.reason == "url"


def test_content_hash_duplicate_across_sources() -> None:
    mock_article = _article(
        "Dockworkers strike at west coast terminal",
        url="https://example.com/atmograph/mock/strike",
        source_type="mock",
    )
    rss_article = _article(
        "DOCKWORKERS   strike at west coast terminal",
        url="https://news.example.net/other-link",
        source_type="rss",
    )

    index = DeduplicationIndex.from_articles([mock_article])
    duplicate = index.check(rss_article)
    assert duplicate is not None
    assert duplicate.reason == "content_hash"


def test_article_id_duplicate_is_detected() -> None:
    index = DeduplicationIndex()
    index.add(_article("One", article_id="mock-001"))

    duplicate = index.check(_article("Two", article_id="mock-001"))
    assert duplicate is not None
    assert duplicate.reason == "article_id"


def test_new_articles_are_not_duplicates() -> None:
    index = DeduplicationIndex()
    index.add(_article("First", url="https://example.com/one"))

    assert index.check(_article("Second", url="https://example.com/two")) is None


def test_index_reports_number_of_content_hashes() -> None:
    index = DeduplicationIndex.from_articles(
        [
            _article("First", url="https://example.com/one"),
            _article("Second", url="https://example.com/two"),
            _article("FIRST", url="https://example.com/three"),
        ]
    )
    assert len(index) == 2
