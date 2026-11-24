"""Loader for the bundled mock news dataset.

The mock dataset is a normal JSON array of article-shaped records.  Records are
validated with the same :class:`~app.schemas.article.Article` model used by RSS,
so the downstream pipeline cannot tell the two sources apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.schemas.article import Article
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class MockLoadResult:
    """Validated articles plus counters describing the load."""

    articles: list[Article] = field(default_factory=list)
    total_fetched: int = 0
    invalid_articles: int = 0
    errors: list[str] = field(default_factory=list)


class MockIngestor:
    """Read and validate ``data/mock_news.json``."""

    def __init__(self, settings: Settings | None = None, path: Path | str | None = None) -> None:
        self._settings = settings or get_settings()
        self._path = Path(path) if path is not None else self._settings.mock_news_path

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> MockLoadResult:
        if not self._path.exists():
            message = f"Mock dataset not found: {self._path.name}"
            logger.error(message)
            return MockLoadResult(errors=[message])

        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            message = f"Cannot read mock dataset '{self._path.name}': {exc}"
            logger.error(message)
            return MockLoadResult(errors=[message])

        if not text.strip():
            logger.warning("Mock dataset '%s' is empty", self._path.name)
            return MockLoadResult()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            message = (
                f"Mock dataset '{self._path.name}' contains invalid JSON "
                f"(line {exc.lineno}, column {exc.colno})"
            )
            logger.error(message)
            return MockLoadResult(errors=[message])

        if isinstance(payload, dict) and isinstance(payload.get("articles"), list):
            payload = payload["articles"]
        if not isinstance(payload, list):
            message = f"Mock dataset '{self._path.name}' must contain a JSON array"
            logger.error(message)
            return MockLoadResult(errors=[message])

        result = MockLoadResult(total_fetched=len(payload))
        for position, record in enumerate(payload):
            if not isinstance(record, dict):
                result.invalid_articles += 1
                logger.warning("Mock record at index %d is not a JSON object; skipped", position)
                continue
            try:
                article = Article.from_raw(
                    record,
                    source_type="mock",
                    max_description_chars=self._settings.article_max_chars,
                )
            except (ValidationError, ValueError) as exc:
                result.invalid_articles += 1
                logger.warning("Invalid mock record at index %d skipped: %s", position, exc)
                continue
            result.articles.append(article)

        if result.invalid_articles:
            result.errors.append(
                f"{self._path.name}: {result.invalid_articles} invalid record(s) skipped"
            )

        logger.info(
            "Mock dataset '%s' loaded: %d fetched, %d valid, %d invalid",
            self._path.name,
            result.total_fetched,
            len(result.articles),
            result.invalid_articles,
        )
        return result


__all__ = ["MockIngestor", "MockLoadResult"]
