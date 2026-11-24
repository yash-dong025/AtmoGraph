"""RSS ingestion tests using mocked HTTP responses (no live internet)."""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlsplit

import httpx
import pytest

from app.config import Settings
from app.ingestion.feed_config import get_configured_feeds, parse_feed_entry
from app.ingestion.rss_ingestor import RssIngestor
from app.schemas.ingestion import FeedConfig

RSS_OK = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Logistics Feed</title>
    <link>https://feeds.test/</link>
    <description>Test feed</description>
    <item>
      <title>Port workers strike at major terminal</title>
      <link>https://news.example.com/port-strike</link>
      <description><![CDATA[<p>Dockworkers walked out &amp; halted berthing.</p>]]></description>
      <author>jane.doe@example.com (Jane Doe)</author>
      <pubDate>Mon, 02 Nov 2026 08:30:00 GMT</pubDate>
    </item>
    <item>
      <title>Rail backlog grows after signal failure</title>
      <link>https://news.example.com/rail-backlog</link>
      <description>Intermodal terminals report growing container stacks.</description>
      <pubDate>Tue, 03 Nov 2026 09:15:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

RSS_MINIMAL = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Minimal Feed</title>
    <item><title>Only a title is present</title></item>
  </channel>
</rss>
"""

RSS_ONE_INVALID = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Mixed Feed</title>
    <item><description>No headline here</description></item>
    <item><title>Valid headline</title><link>https://news.example.com/valid</link></item>
  </channel>
</rss>
"""


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"environment": "testing", "rss_feeds": ""}
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/ok":
        return httpx.Response(200, content=RSS_OK.encode("utf-8"), headers={"content-type": "application/rss+xml"})
    if path == "/minimal":
        return httpx.Response(200, content=RSS_MINIMAL.encode("utf-8"))
    if path == "/mixed":
        return httpx.Response(200, content=RSS_ONE_INVALID.encode("utf-8"))
    if path == "/missing":
        return httpx.Response(404, text="not found")
    if path == "/empty":
        return httpx.Response(200, content=b"")
    if path == "/malformed":
        return httpx.Response(200, content=b"<html><body>Not a feed</body></html>")
    if path == "/timeout":
        raise httpx.TimeoutException("simulated timeout")
    return httpx.Response(500, text="unexpected")


def _feed(path: str, name: str = "Test Feed") -> FeedConfig:
    return FeedConfig(name=name, url=f"https://feeds.test{path}", category="logistics")


def _ingestor(feeds: list[FeedConfig], settings: Settings | None = None) -> RssIngestor:
    return RssIngestor(
        settings=settings or _settings(),
        feeds=feeds,
        transport=httpx.MockTransport(_handler),
    )


async def test_rss_feed_is_parsed_and_normalized() -> None:
    results = await _ingestor([_feed("/ok")]).fetch_all()

    assert len(results) == 1
    result = results[0]
    assert result.success is True
    assert result.error is None
    assert result.fetched == 2
    assert result.invalid_articles == 0
    assert len(result.articles) == 2

    first = result.articles[0]
    assert first.title == "Port workers strike at major terminal"
    assert first.description == "Dockworkers walked out & halted berthing."
    assert first.source == "Test Feed"
    assert first.url == "https://news.example.com/port-strike"
    assert first.source_type == "rss"
    assert "Jane Doe" in (first.author or "")
    assert first.published_at == datetime(2026, 11, 2, 8, 30, tzinfo=timezone.utc)


async def test_missing_optional_rss_fields_are_accepted() -> None:
    result = (await _ingestor([_feed("/minimal")]).fetch_all())[0]

    assert result.success is True
    assert result.fetched == 1
    assert len(result.articles) == 1
    article = result.articles[0]
    assert article.title == "Only a title is present"
    assert article.description == ""
    assert article.url == ""
    assert article.author is None
    assert article.published_at is None


async def test_invalid_rss_entry_is_counted() -> None:
    result = (await _ingestor([_feed("/mixed")]).fetch_all())[0]

    assert result.success is True
    assert result.fetched == 2
    assert result.invalid_articles == 1
    assert len(result.articles) == 1
    assert result.articles[0].title == "Valid headline"


async def test_one_failed_feed_does_not_abort_the_others() -> None:
    results = await _ingestor([_feed("/ok", "Good Feed"), _feed("/missing", "Missing Feed")]).fetch_all()

    assert len(results) == 2
    good = next(result for result in results if result.feed.name == "Good Feed")
    bad = next(result for result in results if result.feed.name == "Missing Feed")
    assert good.success is True and len(good.articles) == 2
    assert bad.success is False
    assert bad.error and "404" in bad.error


async def test_timeout_is_handled() -> None:
    result = (await _ingestor([_feed("/timeout")]).fetch_all())[0]

    assert result.success is False
    assert result.error and "timed out" in result.error.lower()


async def test_empty_response_body_is_handled() -> None:
    result = (await _ingestor([_feed("/empty")]).fetch_all())[0]

    assert result.success is False
    assert result.error == "Empty response body"


async def test_malformed_feed_is_handled() -> None:
    result = (await _ingestor([_feed("/malformed")]).fetch_all())[0]

    assert result.success is False
    assert result.error and "Malformed" in result.error


async def test_max_items_per_feed_is_applied() -> None:
    settings = _settings(rss_max_items_per_feed=1)
    result = (await _ingestor([_feed("/ok")], settings=settings).fetch_all())[0]

    assert result.fetched == 1
    assert len(result.articles) == 1


async def test_no_configured_feeds_returns_empty_list() -> None:
    assert await _ingestor([]).fetch_all() == []


def test_feed_configuration_supports_names_and_catalog_categories() -> None:
    settings = _settings(
        rss_feeds=(
            "Custom Feed|https://custom.example.com/rss,"
            "https://gcaptain.com/feed/,"
            "https://gcaptain.com/feed"
        )
    )
    feeds = get_configured_feeds(settings)

    assert [feed.name for feed in feeds] == ["Custom Feed", "gCaptain"]
    assert feeds[0].category == "general"
    assert feeds[1].category == "shipping"


def test_feed_configuration_strips_embedded_credentials() -> None:
    feed = parse_feed_entry("https://user:secret@example.com/feed.xml")
    assert feed is not None
    parts = urlsplit(feed.url)
    assert parts.username is None
    assert parts.password is None
    assert "secret" not in feed.url


def test_feed_configuration_redacts_sensitive_query_values() -> None:
    feed = parse_feed_entry("https://example.com/feed.xml?api_key=topsecret&lang=en")
    assert feed is not None
    assert "topsecret" not in feed.url
    assert "api_key=%2A%2A%2A" in feed.url
    assert "lang=en" in feed.url


def test_invalid_feed_entry_is_ignored() -> None:
    assert parse_feed_entry("not a url") is None
    assert parse_feed_entry("ftp://example.com/feed") is None
