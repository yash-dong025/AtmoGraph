"""Separate, atomic storage for Phase 3 entity annotations.

Annotations are deliberately kept out of ``data/articles.json`` so the Phase 2
ingestion contract stays stable and annotations can be regenerated without
touching source articles.  The interface mirrors ``ArticleStorage`` so a
database backend can replace the JSON implementation later.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from app.schemas.nlp import ArticleAnnotation
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AnnotationStorageError(RuntimeError):
    """Raised when the annotation store cannot be read or written safely."""


class EntityAnnotationStorage(ABC):
    """Backend-agnostic store for per-article entity annotations."""

    @abstractmethod
    def load(self) -> list[ArticleAnnotation]:
        """Return every valid annotation record."""

    @abstractmethod
    def get(self, article_id: str) -> ArticleAnnotation | None:
        """Return the annotation for one article, or ``None``."""

    @abstractmethod
    def upsert(self, annotation: ArticleAnnotation) -> ArticleAnnotation:
        """Insert or replace the annotation for its article ID."""

    @abstractmethod
    def delete(self, article_id: str) -> bool:
        """Remove an annotation; return whether a record was removed."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of valid annotation records."""


class JsonEntityAnnotationStorage(EntityAnnotationStorage):
    """Thread-safe, atomic JSON implementation of :class:`EntityAnnotationStorage`.

    The file is a JSON array of :class:`ArticleAnnotation` objects.  Invalid
    JSON is never overwritten; invalid individual records are skipped on read
    and preserved verbatim when another article is upserted.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    # -- public API --------------------------------------------------------
    def load(self) -> list[ArticleAnnotation]:
        with self._lock:
            raw_records = self._read_raw()
            annotations: list[ArticleAnnotation] = []
            for position, record in enumerate(raw_records):
                if not isinstance(record, dict):
                    logger.warning(
                        "Skipping non-object annotation at %s[%d]",
                        self._path,
                        position,
                    )
                    continue
                try:
                    annotations.append(ArticleAnnotation.model_validate(record))
                except ValidationError as exc:
                    logger.warning(
                        "Skipping invalid annotation at %s[%d]: %s",
                        self._path,
                        position,
                        exc.errors()[0].get("msg", "validation error"),
                    )
            return annotations

    def get(self, article_id: str) -> ArticleAnnotation | None:
        for annotation in self.load():
            if annotation.article_id == article_id:
                return annotation
        return None

    def upsert(self, annotation: ArticleAnnotation) -> ArticleAnnotation:
        with self._lock:
            raw_records = self._read_raw()
            replacement = annotation.model_dump(mode="json")
            updated: list[Any] = []
            inserted = False
            for record in raw_records:
                if isinstance(record, dict) and record.get("article_id") == annotation.article_id:
                    if not inserted:
                        updated.append(replacement)
                        inserted = True
                    continue
                updated.append(record)
            if not inserted:
                updated.append(replacement)
            self._write_raw(updated)
            return annotation

    def delete(self, article_id: str) -> bool:
        with self._lock:
            raw_records = self._read_raw()
            remaining = [
                record
                for record in raw_records
                if not (isinstance(record, dict) and record.get("article_id") == article_id)
            ]
            if len(remaining) == len(raw_records):
                return False
            self._write_raw(remaining)
            return True

    def count(self) -> int:
        return len(self.load())

    # -- internals ---------------------------------------------------------
    def _read_raw(self) -> list[Any]:
        if not self._path.exists():
            return []

        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AnnotationStorageError(
                f"Cannot read annotation store '{self._path.name}': {exc}"
            ) from exc

        if not text.strip():
            return []

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AnnotationStorageError(
                f"Annotation store '{self._path.name}' contains invalid JSON "
                f"(line {exc.lineno}, column {exc.colno}); the file was left unchanged."
            ) from exc

        if isinstance(payload, dict):
            for key in ("annotations", "articles"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break

        if not isinstance(payload, list):
            raise AnnotationStorageError(
                f"Annotation store '{self._path.name}' must contain a JSON array; "
                "the file was left unchanged."
            )
        return payload

    def _write_raw(self, records: Sequence[Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AnnotationStorageError(
                f"Cannot create annotation store directory: {exc}"
            ) from exc

        file_descriptor: int | None = None
        temp_path: str | None = None
        try:
            file_descriptor, temp_path = tempfile.mkstemp(
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                dir=str(self._path.parent),
            )
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
                file_descriptor = None
                json.dump(records, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self._path)
            temp_path = None
        except OSError as exc:
            raise AnnotationStorageError(
                f"Cannot write annotation store '{self._path.name}': {exc}"
            ) from exc
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
    "AnnotationStorageError",
    "EntityAnnotationStorage",
    "JsonEntityAnnotationStorage",
]
