"""Duplicate detection for normalized articles.

Three independent keys are checked, in order:

1. ``article_id`` - catches records with an explicitly supplied stable ID.
2. normalized ``url`` - catches the same link syndicated by different feeds.
3. ``content_hash`` - catches the same headline/summary even when the URL
   differs (for example an RSS copy and a mock record).

The index is intentionally in-memory and deterministic; storage applies the
same rules again as a second line of defence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.schemas.article import Article, normalize_url


@dataclass(frozen=True, slots=True)
class DuplicateMatch:
    """Why an incoming article was considered a duplicate."""

    reason: str
    key: str


class DeduplicationIndex:
    """Set-based index of article IDs, normalized URLs and content hashes."""

    __slots__ = ("_ids", "_urls", "_hashes")

    def __init__(self) -> None:
        self._ids: set[str] = set()
        self._urls: set[str] = set()
        self._hashes: set[str] = set()

    @classmethod
    def from_articles(cls, articles: Iterable[Article]) -> "DeduplicationIndex":
        index = cls()
        for article in articles:
            index.add(article)
        return index

    def check(self, article: Article) -> DuplicateMatch | None:
        """Return the matching key, or ``None`` when the article is new.

        URL and content-hash checks run first because generated article IDs are
        themselves derived from the content hash.  This keeps the reported
        duplicate reason meaningful for cross-feed and cross-source matches.
        """
        url = normalize_url(article.url)
        if url and url in self._urls:
            return DuplicateMatch("url", url)
        if article.content_hash and article.content_hash in self._hashes:
            return DuplicateMatch("content_hash", article.content_hash)
        if article.article_id and article.article_id in self._ids:
            return DuplicateMatch("article_id", article.article_id)
        return None

    def add(self, article: Article) -> None:
        if article.article_id:
            self._ids.add(article.article_id)
        url = normalize_url(article.url)
        if url:
            self._urls.add(url)
        if article.content_hash:
            self._hashes.add(article.content_hash)

    def __len__(self) -> int:
        """Number of distinct content hashes currently indexed."""
        return len(self._hashes)


__all__ = ["DeduplicationIndex", "DuplicateMatch"]
