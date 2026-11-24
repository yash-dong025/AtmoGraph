"""Modular local article storage.

Phase 2 persists articles to a JSON array.  The :class:`ArticleStorage`
interface keeps route/ingestion code independent of the concrete backend so a
database implementation can be added in a later phase without touching the API.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from app.ingestion.deduplication import DeduplicationIndex
from app.schemas.article import Article
from app.utils.logger import get_logger

logger = get_logger(__name__)


class StorageError(RuntimeError):
    """Raised when the article store cannot be read or written safely."""


@dataclass(frozen=True, slots=True)
class StorageWriteResult:
    """Outcome of one ``add_articles`` call."""

    saved: int
    duplicates_skipped: int
    total_records: int


class ArticleStorage(ABC):
    """Backend-agnostic article store."""

    @abstractmethod
    def load(self) -> list[Article]:
        """Return all valid stored articles."""

    @abstractmethod
    def add_articles(self, articles: Sequence[Article]) -> StorageWriteResult:
        """Persist new articles, skipping duplicates."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of valid stored articles."""


class JsonArticleStorage(ArticleStorage):
    """Thread-safe, atomic JSON-file implementation of :class:`ArticleStorage`.

    Invalid JSON is never overwritten: the caller receives :class:`StorageError`
    and the original file stays untouched.  Empty files are treated as an empty
    store.  Invalid individual records already present in the file are logged
    and skipped, but are preserved verbatim when new records are appended.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    # -- public API --------------------------------------------------------
    def load(self) -> list[Article]:
        with self._lock:
            raw_records = self._read_raw()
            articles: list[Article] = []
            for position, record in enumerate(raw_records):
                if not isinstance(record, dict):
                    logger.warning(
                        "Skipping non-object record at %s[%d] while loading articles",
                        self._path,
                        position,
                    )
                    continue
                try:
                    articles.append(Article.model_validate(record))
                except ValidationError as exc:
                    logger.warning(
                        "Skipping invalid stored article at %s[%d]: %s",
                        self._path,
                        position,
                        exc.errors()[0].get("msg", "validation error"),
                    )
            return articles

    def add_articles(self, articles: Sequence[Article]) -> StorageWriteResult:
        with self._lock:
            raw_records = self._read_raw()
            existing = self._valid_articles(raw_records)
            index = DeduplicationIndex.from_articles(existing)

            new_records: list[dict[str, Any]] = []
            skipped = 0
            for article in articles:
                if index.check(article) is not None:
                    skipped += 1
                    continue
                index.add(article)
                new_records.append(article.model_dump(mode="json"))

            if new_records:
                raw_records.extend(new_records)
                self._write_raw(raw_records)

            return StorageWriteResult(
                saved=len(new_records),
                duplicates_skipped=skipped,
                total_records=len(raw_records),
            )

    def count(self) -> int:
        return len(self.load())

    # -- internals ---------------------------------------------------------
    def _valid_articles(self, raw_records: Sequence[Any]) -> list[Article]:
        articles: list[Article] = []
        for record in raw_records:
            if not isinstance(record, dict):
                continue
            try:
                articles.append(Article.model_validate(record))
            except ValidationError:
                continue
        return articles

    def _read_raw(self) -> list[Any]:
        if not self._path.exists():
            self._write_raw([])
            return []

        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StorageError(f"Cannot read article store '{self._path.name}': {exc}") from exc

        if not text.strip():
            # An empty file contains no data to lose; treat it as an empty store.
            return []

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StorageError(
                f"Article store '{self._path.name}' contains invalid JSON "
                f"(line {exc.lineno}, column {exc.colno}); the file was left unchanged."
            ) from exc

        if isinstance(payload, dict) and isinstance(payload.get("articles"), list):
            # Forward-compatible read of a future wrapper format.
            payload = payload["articles"]

        if not isinstance(payload, list):
            raise StorageError(
                f"Article store '{self._path.name}' must contain a JSON array; "
                "the file was left unchanged."
            )
        return payload

    def _write_raw(self, records: Sequence[Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(f"Cannot create article store directory: {exc}") from exc

        file_descriptor: int | None = None
        temp_path: str | None = None
        try:
            file_descriptor, temp_path = tempfile.mkstemp(
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                dir=str(self._path.parent),
            )
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
                file_descriptor = None  # ownership moved to the file object
                json.dump(records, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self._path)
            temp_path = None
        except OSError as exc:
            raise StorageError(f"Cannot write article store '{self._path.name}': {exc}") from exc
        finally:
            if file_descriptor is not None:
                try:
                    os.close(file_descriptor)
                except OSError:
                    pass
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


__all__ = [
    "ArticleStorage",
    "JsonArticleStorage",
    "StorageError",
    "StorageWriteResult",
]
