"""Configurable catalog of public RSS feeds used by the ingestion engine.

Feed URLs come from ``ATMOGRAPH_RSS_FEEDS``.  Both plain URLs and
``Name|URL`` entries are accepted.  Names/categories for the default public
feeds are defined here; unknown URLs get a name derived from their hostname.

No API keys, paid services or private credentials are used.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.config import Settings, get_settings
from app.schemas.article import normalize_url
from app.schemas.ingestion import FeedConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_FEEDS: tuple[FeedConfig, ...] = (
    FeedConfig(
        name="BBC Business",
        url="https://feeds.bbci.co.uk/news/business/rss.xml",
        category="trade",
    ),
    FeedConfig(
        name="BBC World",
        url="https://feeds.bbci.co.uk/news/world/rss.xml",
        category="natural_disaster",
    ),
    FeedConfig(
        name="Supply Chain Dive",
        url="https://www.supplychaindive.com/feeds/news/",
        category="logistics",
    ),
    FeedConfig(
        name="Manufacturing Dive",
        url="https://www.manufacturingdive.com/feeds/news/",
        category="manufacturing",
    ),
    FeedConfig(
        name="gCaptain",
        url="https://gcaptain.com/feed/",
        category="shipping",
    ),
    FeedConfig(
        name="The Loadstar",
        url="https://theloadstar.com/feed/",
        category="logistics",
    ),
    FeedConfig(
        name="Splash 247",
        url="https://splash247.com/feed/",
        category="shipping",
    ),
    FeedConfig(
        name="Container News",
        url="https://container-news.com/feed/",
        category="port_congestion",
    ),
    FeedConfig(
        name="FreightWaves",
        url="https://www.freightwaves.com/feed",
        category="transportation",
    ),
)

_CATALOG: dict[str, FeedConfig] = {normalize_url(feed.url): feed for feed in DEFAULT_FEEDS}
_SENSITIVE_QUERY_MARKERS = ("api_key", "apikey", "auth", "password", "secret", "signature", "token")


def _sanitize_url(url: str) -> str:
    """Remove embedded credentials so they can never be exposed by the API.

    User-info is dropped entirely.  Query values for obviously sensitive keys
    are redacted.  Credentialed/paid feeds are outside this module's scope, so
    a redacted feed simply fails gracefully during ingestion.
    """
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return ""
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return ""
    host = parts.hostname
    try:
        port = parts.port
    except ValueError:
        return ""

    if port:
        host = f"{host}:{port}"

    query_pairs = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        lowered = key.lower()
        if any(marker in lowered for marker in _SENSITIVE_QUERY_MARKERS):
            value = "***"
        query_pairs.append((key, value))

    return urlunsplit(
        (parts.scheme.lower(), host, parts.path, urlencode(query_pairs), "")
    )


def _name_from_url(url: str) -> str:
    try:
        host = urlsplit(url).hostname or "rss feed"
    except ValueError:
        host = "rss feed"
    return host.removeprefix("www.")


def parse_feed_entry(entry: str) -> FeedConfig | None:
    """Parse one ``ATMOGRAPH_RSS_FEEDS`` item into a :class:`FeedConfig`."""
    raw = entry.strip()
    if not raw:
        return None

    explicit_name: str | None = None
    if "|" in raw:
        explicit_name, raw = (part.strip() for part in raw.split("|", 1))
        explicit_name = explicit_name or None

    url = _sanitize_url(raw)
    if not url:
        logger.warning("Ignoring invalid RSS feed entry: %r", entry)
        return None

    catalog_entry = _CATALOG.get(normalize_url(url))
    return FeedConfig(
        name=explicit_name or (catalog_entry.name if catalog_entry else _name_from_url(url)),
        url=url,
        category=catalog_entry.category if catalog_entry else "general",
    )


def get_configured_feeds(settings: Settings | None = None) -> list[FeedConfig]:
    """Return deduplicated feeds from configuration, preserving configured order."""
    settings = settings or get_settings()
    feeds: list[FeedConfig] = []
    seen: set[str] = set()
    for entry in settings.rss_feed_list:
        feed = parse_feed_entry(entry)
        if feed is None:
            continue
        key = normalize_url(feed.url)
        if key in seen:
            logger.warning("Ignoring duplicate RSS feed URL: %s", feed.url)
            continue
        seen.add(key)
        feeds.append(feed)
    return feeds


def get_default_feeds() -> list[FeedConfig]:
    """Return a fresh copy of the built-in public feed catalog."""
    return [feed.model_copy() for feed in DEFAULT_FEEDS]


__all__ = [
    "DEFAULT_FEEDS",
    "get_configured_feeds",
    "get_default_feeds",
    "parse_feed_entry",
]
