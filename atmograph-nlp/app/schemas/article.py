"""Canonical article schema used by the Phase 2 news ingestion pipeline.

The schema is deliberately source-agnostic: RSS entries and mock records are
both normalized into :class:`Article`.  Normalization happens in validators so
every article that reaches storage has consistent whitespace, a UTC timestamp,
a stable content hash and a stable identifier.

Nothing in this module performs NLP, entity extraction or disruption
classification; those belong to later phases.
"""

from __future__ import annotations

import hashlib
import html
import re
from datetime import date, datetime, time, timezone
from html.parser import HTMLParser
from time import struct_time
from typing import Any, Literal, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dateutil import parser as date_parser
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.utils.helpers import utc_now

SourceType = Literal["mock", "rss"]

_WHITESPACE_RE = re.compile(r"\s+")
_TRACKING_QUERY_KEYS = {
    "_ga",
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "yclid",
}
_DEFAULT_PORTS = {"http": 80, "https": 443}
_MAX_URL_LENGTH = 2048


class _HTMLTextExtractor(HTMLParser):
    """Tiny HTML-to-text converter with no third-party dependency."""

    _SKIP_TAGS = {"script", "style"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    @property
    def text(self) -> str:
        return " ".join(self._parts)


def normalize_whitespace(value: str | None) -> str:
    """Collapse every run of whitespace (including newlines) to one space."""
    if value is None:
        return ""
    return _WHITESPACE_RE.sub(" ", str(value)).strip()


def strip_html(value: str | None) -> str:
    """Return plain text from an HTML fragment (RSS summaries often contain HTML)."""
    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(text)
        extractor.close()
        cleaned = extractor.text
    except Exception:  # pragma: no cover - HTMLParser is very forgiving
        cleaned = re.sub(r"<[^>]+>", " ", text)
    return normalize_whitespace(html.unescape(cleaned))


def normalize_url(value: str | None) -> str:
    """Normalize an article URL for duplicate detection.

    Rules: lowercase scheme/host, drop ``www.``, remove default ports, strip
    fragments and tracking query parameters, sort the remaining query string
    and remove a trailing slash.  Invalid or non-HTTP(S) URLs become ``""`` so
    an article with a broken link is still ingestible.
    """
    raw = normalize_whitespace(value)
    if not raw:
        return ""

    try:
        parts = urlsplit(raw)
        scheme = parts.scheme.lower()
        if scheme not in {"http", "https"} or not parts.hostname:
            return ""
        host = parts.hostname.lower()
        if host.startswith("www."):
            host = host[4:]
        try:
            port = parts.port
        except ValueError:
            return ""
        if port is not None and port != _DEFAULT_PORTS[scheme]:
            host = f"{host}:{port}"
    except ValueError:
        return ""

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in _TRACKING_QUERY_KEYS
    ]
    query = urlencode(sorted(query_pairs))

    normalized = urlunsplit((scheme, host, path, query, ""))
    return normalized[:_MAX_URL_LENGTH]


def normalize_datetime(value: Any) -> datetime | None:
    """Convert common feed/mock date representations to a UTC datetime.

    ``datetime``, ``date``, ``time.struct_time`` (what feedparser returns),
    epoch numbers and ISO/RFC date strings are supported.  Naive values are
    treated as UTC.  Unparseable values return ``None`` instead of raising so a
    single bad feed entry cannot fail the whole batch.
    """
    if value is None or value == "":
        return None

    parsed: datetime | None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    elif isinstance(value, struct_time):
        parsed = datetime(*value[:6], tzinfo=timezone.utc)
    elif isinstance(value, time):
        parsed = datetime.combine(datetime.now(timezone.utc).date(), value, tzinfo=timezone.utc)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str):
        try:
            parsed = date_parser.parse(value)
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def compute_content_hash(title: str, description: str) -> str:
    """Stable SHA-256 of case-folded, whitespace-normalized title + description."""
    payload = f"{normalize_whitespace(title).casefold()}\n{normalize_whitespace(description).casefold()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Article(BaseModel):
    """A normalized news article stored by the ingestion pipeline."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=False)

    article_id: str = Field(
        default="",
        max_length=128,
        description="Stable identifier; supplied IDs are preserved, otherwise derived from the content hash.",
    )
    title: str = Field(min_length=1, max_length=1_000, description="Plain-text headline.")
    description: str = Field(default="", max_length=200_000, description="Plain-text summary or body.")
    source: str = Field(default="unknown", max_length=200, description="Feed or outlet name.")
    url: str = Field(default="", max_length=_MAX_URL_LENGTH, description="Canonical article URL, if available.")
    published_at: datetime | None = Field(default=None, description="Publication time normalized to UTC.")
    author: str | None = Field(default=None, max_length=500, description="Author or byline, if available.")
    source_type: SourceType = Field(description="Where the record came from: ``mock`` or ``rss``.")
    content_hash: str = Field(
        default="",
        max_length=64,
        description="SHA-256 of normalized title + description; used for duplicate detection.",
    )
    ingested_at: datetime = Field(default_factory=utc_now, description="UTC time the article entered the pipeline.")

    @field_validator("title", mode="before")
    @classmethod
    def _clean_title(cls, value: Any) -> str:
        cleaned = strip_html(value) if value is not None else ""
        if not cleaned:
            raise ValueError("title must be a non-empty string")
        return cleaned

    @field_validator("description", mode="before")
    @classmethod
    def _clean_description(cls, value: Any) -> str:
        return strip_html(value) if value is not None else ""

    @field_validator("source", mode="before")
    @classmethod
    def _clean_source(cls, value: Any) -> str:
        cleaned = strip_html(value) if value is not None else ""
        return cleaned or "unknown"

    @field_validator("author", mode="before")
    @classmethod
    def _clean_author(cls, value: Any) -> str | None:
        cleaned = strip_html(value) if value is not None else ""
        return cleaned or None

    @field_validator("url", mode="before")
    @classmethod
    def _clean_url(cls, value: Any) -> str:
        return normalize_url(value) if value is not None else ""

    @field_validator("published_at", mode="before")
    @classmethod
    def _clean_published_at(cls, value: Any) -> datetime | None:
        return normalize_datetime(value)

    @field_validator("ingested_at", mode="before")
    @classmethod
    def _clean_ingested_at(cls, value: Any) -> datetime:
        return normalize_datetime(value) or utc_now()

    @field_validator("article_id", mode="before")
    @classmethod
    def _clean_article_id(cls, value: Any) -> str:
        return normalize_whitespace(value) if value is not None else ""

    @model_validator(mode="after")
    def _finalize_identifiers(self) -> "Article":
        """Always derive the hash from current content and fill a missing ID."""
        content_hash = compute_content_hash(self.title, self.description)
        object.__setattr__(self, "content_hash", content_hash)
        if not self.article_id:
            object.__setattr__(self, "article_id", f"art_{content_hash[:16]}")
        return self

    @classmethod
    def from_raw(
        cls,
        data: Mapping[str, Any],
        *,
        source_type: SourceType | None = None,
        source: str | None = None,
        max_description_chars: int | None = None,
    ) -> "Article":
        """Build an article from an RSS entry or mock JSON record.

        ``source_type``/``source`` override the raw record when supplied.  The
        description is truncated after HTML removal so the configured
        ``article_max_chars`` limit applies to stored plain text.
        """
        payload: dict[str, Any] = dict(data)
        if source_type is not None:
            payload["source_type"] = source_type
        if source is not None:
            payload["source"] = source
        if max_description_chars and payload.get("description") is not None:
            payload["description"] = strip_html(str(payload["description"]))[:max_description_chars]
        return cls.model_validate(payload)


__all__ = [
    "Article",
    "SourceType",
    "compute_content_hash",
    "normalize_datetime",
    "normalize_url",
    "normalize_whitespace",
    "strip_html",
]
