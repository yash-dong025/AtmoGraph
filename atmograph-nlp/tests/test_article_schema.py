"""Tests for the normalized article schema and its helpers."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.article import (
    Article,
    compute_content_hash,
    normalize_url,
    strip_html,
)


def test_valid_article_is_normalized() -> None:
    article = Article(
        title="  Port   strike <b>ends</b>  ",
        description="<p>Workers &amp; management reach an agreement.</p>",
        source="  Test   Feed ",
        url="https://Example.com/news/1/?utm_source=newsletter&b=2&a=1#top",
        published_at="2026-11-01T10:00:00+02:00",
        author="  Jane   Doe ",
        source_type="rss",
    )

    assert article.title == "Port strike ends"
    assert article.description == "Workers & management reach an agreement."
    assert article.source == "Test Feed"
    assert article.url == "https://example.com/news/1?a=1&b=2"
    assert article.author == "Jane Doe"
    assert article.published_at == datetime(2026, 11, 1, 8, 0, tzinfo=timezone.utc)
    assert article.source_type == "rss"
    assert article.article_id.startswith("art_")
    assert article.content_hash == compute_content_hash(article.title, article.description)
    assert article.ingested_at.tzinfo is not None


def test_missing_optional_fields_receive_safe_defaults() -> None:
    article = Article(title="Only a headline", source_type="mock")

    assert article.description == ""
    assert article.source == "unknown"
    assert article.url == ""
    assert article.published_at is None
    assert article.author is None
    assert article.article_id
    assert article.content_hash


def test_explicit_article_id_is_preserved() -> None:
    article = Article(article_id=" mock-001 ", title="Headline", source_type="mock")
    assert article.article_id == "mock-001"


@pytest.mark.parametrize("title", ["", "   ", None])
def test_empty_title_is_rejected(title: object) -> None:
    with pytest.raises(ValidationError):
        Article(title=title, source_type="mock")


def test_missing_source_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Article(title="Headline")


def test_unknown_source_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Article(title="Headline", source_type="api")  # type: ignore[arg-type]


def test_unparseable_date_becomes_none() -> None:
    article = Article(title="Headline", source_type="mock", published_at="not-a-real-date")
    assert article.published_at is None


def test_naive_datetime_is_treated_as_utc() -> None:
    article = Article(
        title="Headline",
        source_type="mock",
        published_at=datetime(2026, 5, 4, 12, 30),
    )
    assert article.published_at == datetime(2026, 5, 4, 12, 30, tzinfo=timezone.utc)


def test_struct_time_is_supported() -> None:
    parsed = time.strptime("2026-11-02 08:30:00", "%Y-%m-%d %H:%M:%S")
    article = Article(title="Headline", source_type="rss", published_at=parsed)
    assert article.published_at == datetime(2026, 11, 2, 8, 30, tzinfo=timezone.utc)


def test_invalid_url_is_dropped_but_article_is_kept() -> None:
    article = Article(title="Headline", source_type="rss", url="javascript:alert(1)")
    assert article.url == ""


def test_content_hash_ignores_case_and_whitespace() -> None:
    first = Article(
        title="Port  Strike",
        description="Dockworkers   walk out.",
        source_type="mock",
    )
    second = Article(
        title="  port strike ",
        description="DOCKWORKERS walk out.",
        url="https://different.example.com/story",
        source_type="rss",
    )
    assert first.content_hash == second.content_hash


def test_url_normalization_handles_common_variants() -> None:
    assert normalize_url("HTTPS://WWW.Example.COM/Path/?utm_medium=x#frag") == "https://example.com/Path"
    assert normalize_url("https://example.com/") == "https://example.com/"
    assert normalize_url("https://example.com:443/a/") == "https://example.com/a"
    assert normalize_url("ftp://example.com/a") == ""
    assert normalize_url(None) == ""


def test_strip_html_removes_tags_and_scripts() -> None:
    assert strip_html("<p>Hello <b>world</b></p><script>bad()</script>") == "Hello world"


def test_from_raw_truncates_plain_text_description() -> None:
    article = Article.from_raw(
        {"title": "Headline", "description": "<p>" + ("x" * 50) + "</p>"},
        source_type="mock",
        max_description_chars=10,
    )
    assert article.description == "x" * 10
