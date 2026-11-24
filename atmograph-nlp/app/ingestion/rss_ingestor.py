"""RSS ingestion built on ``httpx`` (fetch/timeout) and ``feedparser`` (parse).

Each feed is fetched independently and concurrently.  One bad feed can never
abort the batch: the failure is captured in :class:`RssFeedResult`, logged and
counted in the ingestion statistics.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import feedparser
import httpx
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.ingestion.feed_config import get_configured_feeds
from app.schemas.article import Article
from app.schemas.ingestion import FeedConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)

_ACCEPT_HEADER = (
    "application/rss+xml, application/atom+xml, application/xml, "
    "text/xml;q=0.9, */*;q=0.8"
)


@dataclass(slots=True)
class RssFeedResult:
    """Outcome of fetching one feed."""

    feed: FeedConfig
    success: bool
    articles: list[Article] = field(default_factory=list)
    fetched: int = 0
    invalid_articles: int = 0
    error: str | None = None


class RssIngestor:
    """Fetch and parse configured RSS feeds without depending on live internet in tests."""

    def __init__(
        self,
        settings: Settings | None = None,
        feeds: list[FeedConfig] | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._feeds = feeds if feeds is not None else get_configured_feeds(self._settings)
        self._transport = transport

    @property
    def feeds(self) -> list[FeedConfig]:
        return list(self._feeds)

    async def fetch_all(self) -> list[RssFeedResult]:
        """Fetch every configured feed concurrently and return per-feed results."""
        if not self._feeds:
            logger.info("No RSS feeds are configured; nothing to ingest")
            return []

        timeout = httpx.Timeout(self._settings.rss_timeout_seconds)
        headers = {
            "User-Agent": self._settings.news_user_agent,
            "Accept": _ACCEPT_HEADER,
        }
        async with httpx.AsyncClient(
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
            transport=self._transport,
        ) as client:
            outcomes = await asyncio.gather(
                *(self._fetch_one(client, feed) for feed in self._feeds),
                return_exceptions=True,
            )

        results: list[RssFeedResult] = []
        for feed, outcome in zip(self._feeds, outcomes, strict=True):
            if isinstance(outcome, RssFeedResult):
                results.append(outcome)
                continue
            # Defensive: _fetch_one handles expected failures itself.
            message = f"Unexpected feed error: {type(outcome).__name__}: {outcome}"
            logger.error("RSS feed '%s' failed: %s", feed.name, message)
            results.append(RssFeedResult(feed=feed, success=False, error=message))
        return results

    async def _fetch_one(self, client: httpx.AsyncClient, feed: FeedConfig) -> RssFeedResult:
        logger.info("Fetching RSS feed '%s' (%s)", feed.name, feed.url)
        try:
            response = await client.get(feed.url)
            response.raise_for_status()
        except httpx.TimeoutException:
            message = f"Request timed out after {self._settings.rss_timeout_seconds:g}s"
            logger.warning("RSS feed '%s' failed: %s", feed.name, message)
            return RssFeedResult(feed=feed, success=False, error=message)
        except httpx.HTTPStatusError as exc:
            message = f"HTTP {exc.response.status_code} from {feed.url}"
            logger.warning("RSS feed '%s' failed: %s", feed.name, message)
            return RssFeedResult(feed=feed, success=False, error=message)
        except httpx.HTTPError as exc:
            message = f"{type(exc).__name__}: {exc}"
            logger.warning("RSS feed '%s' failed: %s", feed.name, message)
            return RssFeedResult(feed=feed, success=False, error=message)

        if not response.content:
            message = "Empty response body"
            logger.warning("RSS feed '%s' failed: %s", feed.name, message)
            return RssFeedResult(feed=feed, success=False, error=message)

        try:
            parsed = feedparser.parse(response.content)
        except Exception as exc:  # feedparser is forgiving, but never trust a parser
            message = f"Could not parse feed: {type(exc).__name__}: {exc}"
            logger.warning("RSS feed '%s' failed: %s", feed.name, message)
            return RssFeedResult(feed=feed, success=False, error=message)

        entries = list(parsed.entries)
        if not entries:
            bozo_error = getattr(parsed, "bozo_exception", None)
            if getattr(parsed, "bozo", False):
                message = f"Malformed RSS response: {bozo_error or 'parser error'}"[:500]
            else:
                # Some HTML error pages are accepted by feedparser as an empty
                # document without setting the bozo flag.  Treat both as a
                # failed feed so statistics stay meaningful.
                message = "Malformed or empty RSS response: feed contained no entries"
            logger.warning("RSS feed '%s' failed: %s", feed.name, message)
            return RssFeedResult(feed=feed, success=False, error=message)

        if getattr(parsed, "bozo", False):
            logger.warning(
                "RSS feed '%s' reported a parse warning but yielded %d entries: %s",
                feed.name,
                len(entries),
                getattr(parsed, "bozo_exception", "unknown warning"),
            )

        limited_entries = entries[: self._settings.rss_max_items_per_feed]
        result = RssFeedResult(feed=feed, success=True, fetched=len(limited_entries))

        for position, entry in enumerate(limited_entries):
            try:
                result.articles.append(self._entry_to_article(entry, feed))
            except (ValidationError, ValueError, TypeError) as exc:
                result.invalid_articles += 1
                logger.warning(
                    "Invalid entry %d in RSS feed '%s' skipped: %s",
                    position,
                    feed.name,
                    exc,
                )

        logger.info(
            "RSS feed '%s' ingested: %d fetched, %d valid, %d invalid",
            feed.name,
            result.fetched,
            len(result.articles),
            result.invalid_articles,
        )
        return result

    def _entry_to_article(self, entry: Any, feed: FeedConfig) -> Article:
        link = _first_text(entry.get("link"))
        if not link:
            candidate_id = _first_text(entry.get("id"))
            link = candidate_id if candidate_id.startswith(("http://", "https://")) else ""

        author = _first_text(entry.get("author")) or _first_text(entry.get("dc_creator"))
        author_detail = entry.get("author_detail")
        if not author and isinstance(author_detail, dict):
            author = _first_text(author_detail.get("name"))

        raw = {
            "title": entry.get("title"),
            "description": _entry_description(entry),
            "source": feed.name,
            "url": link,
            "published_at": (
                entry.get("published_parsed")
                or entry.get("updated_parsed")
                or entry.get("published")
                or entry.get("updated")
            ),
            "author": author or None,
        }
        return Article.from_raw(
            raw,
            source_type="rss",
            source=feed.name,
            max_description_chars=self._settings.article_max_chars,
        )


def _first_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _entry_description(entry: Any) -> str:
    for key in ("summary", "description"):
        value = entry.get(key)
        if value:
            return str(value)
    content = entry.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("value"):
                return str(item["value"])
    return ""


__all__ = ["RssFeedResult", "RssIngestor"]
