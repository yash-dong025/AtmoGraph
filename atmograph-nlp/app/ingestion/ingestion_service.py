"""Orchestration for Phase 2 news ingestion.

The service is the only place that knows how to combine sources, deduplicate
against stored articles and persist results.  FastAPI routes stay thin, and a
future database backend only needs to implement ``ArticleStorage``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

from app.config import Settings, get_settings
from app.ingestion.deduplication import DeduplicationIndex
from app.ingestion.mock_ingestor import MockIngestor
from app.ingestion.rss_ingestor import RssIngestor
from app.schemas.article import Article, SourceType
from app.schemas.ingestion import (
    FeedConfig,
    IngestionResponse,
    IngestionSource,
    IngestionStats,
    IngestionStatus,
    NewsStatsResponse,
)
from app.storage.json_storage import ArticleStorage, JsonArticleStorage
from app.utils.helpers import utc_now
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SourceBatch:
    """Normalized articles and counters produced by one source adapter."""

    articles: list[Article] = field(default_factory=list)
    total_fetched: int = 0
    invalid_articles: int = 0
    feeds_successful: int = 0
    feeds_failed: int = 0
    errors: list[str] = field(default_factory=list)


class IngestionService:
    """Combine mock/RSS sources, deduplicate and persist articles."""

    def __init__(
        self,
        storage: ArticleStorage,
        settings: Settings | None = None,
        mock_ingestor: MockIngestor | None = None,
        rss_ingestor: RssIngestor | None = None,
    ) -> None:
        self._storage = storage
        self._settings = settings or get_settings()
        self._mock_ingestor = mock_ingestor or MockIngestor(self._settings)
        self._rss_ingestor = rss_ingestor or RssIngestor(self._settings)
        self._lock = asyncio.Lock()
        self._last_run: IngestionStats | None = None
        self._last_run_at: datetime | None = None
        self._last_run_source: IngestionSource | None = None

    # -- ingestion ---------------------------------------------------------
    async def ingest_mock(self) -> IngestionResponse:
        batch = self._load_mock_batch()
        return await self._persist("mock", batch)

    async def ingest_rss(self) -> IngestionResponse:
        batch = await self._load_rss_batch()
        return await self._persist("rss", batch)

    async def ingest_all(self) -> IngestionResponse:
        mock_batch = self._load_mock_batch()
        rss_batch = await self._load_rss_batch()
        return await self._persist("all", _combine_batches(mock_batch, rss_batch))

    # -- queries -----------------------------------------------------------
    def list_articles(
        self,
        *,
        limit: int,
        offset: int,
        source_type: SourceType | None = None,
    ) -> tuple[list[Article], int]:
        """Return a page of articles and the total count before pagination."""
        articles = self._storage.load()
        if source_type is not None:
            articles = [article for article in articles if article.source_type == source_type]
        total = len(articles)
        return articles[offset : offset + limit], total

    def get_stats(self) -> NewsStatsResponse:
        """Storage totals plus the most recent in-process ingestion counters."""
        articles = self._storage.load()
        last = self._last_run
        return NewsStatsResponse(
            total_articles=len(articles),
            rss_articles=sum(1 for article in articles if article.source_type == "rss"),
            mock_articles=sum(1 for article in articles if article.source_type == "mock"),
            duplicates_skipped=last.duplicates_skipped if last else 0,
            invalid_articles=last.invalid_articles if last else 0,
            feeds_successful=last.feeds_successful if last else 0,
            feeds_failed=last.feeds_failed if last else 0,
            last_ingestion_source=self._last_run_source,
            last_ingestion_at=self._last_run_at,
        )

    def list_feeds(self) -> list[FeedConfig]:
        return self._rss_ingestor.feeds

    # -- internals ---------------------------------------------------------
    def _load_mock_batch(self) -> SourceBatch:
        result = self._mock_ingestor.load()
        return SourceBatch(
            articles=result.articles,
            total_fetched=result.total_fetched,
            invalid_articles=result.invalid_articles,
            errors=list(result.errors),
        )

    async def _load_rss_batch(self) -> SourceBatch:
        results = await self._rss_ingestor.fetch_all()
        batch = SourceBatch()
        for result in results:
            batch.articles.extend(result.articles)
            batch.total_fetched += result.fetched
            batch.invalid_articles += result.invalid_articles
            if result.success:
                batch.feeds_successful += 1
            else:
                batch.feeds_failed += 1
                if result.error:
                    batch.errors.append(f"{result.feed.name}: {result.error}")
        return batch

    async def _persist(self, source: IngestionSource, batch: SourceBatch) -> IngestionResponse:
        started_at = utc_now()
        async with self._lock:
            existing = self._storage.load()
            index = DeduplicationIndex.from_articles(existing)

            new_articles: list[Article] = []
            duplicates = 0
            for article in batch.articles:
                if index.check(article) is not None:
                    duplicates += 1
                    continue
                index.add(article)
                new_articles.append(article)

            write_result = self._storage.add_articles(new_articles)
            duplicates += write_result.duplicates_skipped

            status = _determine_status(batch)
            stats = IngestionStats(
                status=status,
                total_fetched=batch.total_fetched,
                total_saved=write_result.saved,
                duplicates_skipped=duplicates,
                invalid_articles=batch.invalid_articles,
                feeds_successful=batch.feeds_successful,
                feeds_failed=batch.feeds_failed,
            )
            finished_at = utc_now()

            self._last_run = stats
            self._last_run_at = finished_at
            self._last_run_source = source

            logger.info(
                "Ingestion run source=%s status=%s fetched=%d saved=%d duplicates=%d "
                "invalid=%d feeds_ok=%d feeds_failed=%d",
                source,
                stats.status,
                stats.total_fetched,
                stats.total_saved,
                stats.duplicates_skipped,
                stats.invalid_articles,
                stats.feeds_successful,
                stats.feeds_failed,
            )
            if batch.errors:
                logger.warning(
                    "Ingestion run source=%s completed with errors: %s",
                    source,
                    "; ".join(batch.errors),
                )

            return IngestionResponse(
                **stats.model_dump(),
                source=source,
                errors=list(batch.errors),
                started_at=started_at,
                finished_at=finished_at,
            )


def _combine_batches(*batches: SourceBatch) -> SourceBatch:
    combined = SourceBatch()
    for batch in batches:
        combined.articles.extend(batch.articles)
        combined.total_fetched += batch.total_fetched
        combined.invalid_articles += batch.invalid_articles
        combined.feeds_successful += batch.feeds_successful
        combined.feeds_failed += batch.feeds_failed
        combined.errors.extend(batch.errors)
    return combined


def _determine_status(batch: SourceBatch) -> IngestionStatus:
    """Predictable run status used by tests and dashboards."""
    if batch.total_fetched == 0:
        return "failed" if batch.errors else "success"
    if batch.invalid_articles == batch.total_fetched:
        return "failed"
    if batch.feeds_failed or batch.invalid_articles:
        return "partial"
    return "success"


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    """Process-wide service used by FastAPI dependency injection."""
    settings = get_settings()
    return IngestionService(
        storage=JsonArticleStorage(settings.articles_store_path),
        settings=settings,
        mock_ingestor=MockIngestor(settings),
        rss_ingestor=RssIngestor(settings),
    )


__all__ = ["IngestionService", "SourceBatch", "get_ingestion_service"]
