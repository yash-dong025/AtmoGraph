"""Phase 2 news ingestion and article listing endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.ingestion.ingestion_service import IngestionService, get_ingestion_service
from app.schemas.article import SourceType
from app.schemas.ingestion import (
    ErrorResponse,
    FeedListResponse,
    IngestionResponse,
    NewsListResponse,
    NewsStatsResponse,
)
from app.storage.json_storage import StorageError

router = APIRouter(prefix="/news", tags=["news"])

_STORAGE_ERROR_RESPONSE = {
    500: {
        "model": ErrorResponse,
        "description": "The local article store is unreadable or contains invalid JSON.",
    }
}


def _storage_failure(exc: StorageError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get(
    "/stats",
    response_model=NewsStatsResponse,
    summary="Get news storage and ingestion statistics",
    description=(
        "Returns total stored articles, a breakdown by source type, and the duplicate/invalid/feed "
        "counters from the most recent ingestion run in this process. Persisted counters survive "
        "restarts; per-run counters reset when the service restarts."
    ),
    responses=_STORAGE_ERROR_RESPONSE,
)
def news_stats(
    service: IngestionService = Depends(get_ingestion_service),
) -> NewsStatsResponse:
    try:
        return service.get_stats()
    except StorageError as exc:
        raise _storage_failure(exc) from exc


@router.get(
    "/feeds",
    response_model=FeedListResponse,
    summary="List configured RSS feeds",
    description=(
        "Returns the feed names, public URLs and editorial categories currently configured through "
        "`ATMOGRAPH_RSS_FEEDS`. No credentials or secret values are returned."
    ),
)
def news_feeds(
    service: IngestionService = Depends(get_ingestion_service),
) -> FeedListResponse:
    feeds = service.list_feeds()
    return FeedListResponse(count=len(feeds), feeds=feeds)


@router.post(
    "/ingest/mock",
    response_model=IngestionResponse,
    summary="Ingest the bundled mock news dataset",
    description=(
        "Loads `data/mock_news.json`, validates every record, removes duplicates already in storage, "
        "and appends new articles. Safe to call repeatedly: a second call reports the same records "
        "as duplicates and saves nothing."
    ),
    responses=_STORAGE_ERROR_RESPONSE,
)
async def ingest_mock(
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionResponse:
    try:
        return await service.ingest_mock()
    except StorageError as exc:
        raise _storage_failure(exc) from exc


@router.post(
    "/ingest/rss",
    response_model=IngestionResponse,
    summary="Ingest configured RSS feeds",
    description=(
        "Fetches every configured feed concurrently with a request timeout. A failed or malformed "
        "feed is logged and counted without aborting the remaining feeds. Articles are validated, "
        "deduplicated and appended to the local store."
    ),
    responses=_STORAGE_ERROR_RESPONSE,
)
async def ingest_rss(
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionResponse:
    try:
        return await service.ingest_rss()
    except StorageError as exc:
        raise _storage_failure(exc) from exc


@router.post(
    "/ingest/all",
    response_model=IngestionResponse,
    summary="Ingest mock and RSS articles together",
    description=(
        "Runs both the mock dataset loader and the RSS fetch, then persists the combined, "
        "deduplicated result. Statistics include feed success/failure counts and duplicate totals "
        "across both sources."
    ),
    responses=_STORAGE_ERROR_RESPONSE,
)
async def ingest_all(
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionResponse:
    try:
        return await service.ingest_all()
    except StorageError as exc:
        raise _storage_failure(exc) from exc


@router.get(
    "",
    response_model=NewsListResponse,
    summary="List stored news articles",
    description=(
        "Returns normalized articles in storage (ingestion) order. Use `limit` and `offset` for "
        "pagination and `source_type` to filter mock or RSS records."
    ),
    responses=_STORAGE_ERROR_RESPONSE,
)
def list_news(
    limit: int = Query(default=20, ge=1, le=200, description="Maximum articles to return."),
    offset: int = Query(default=0, ge=0, description="Number of matching articles to skip."),
    source_type: SourceType | None = Query(
        default=None,
        description="Optional filter: `mock` or `rss`.",
    ),
    service: IngestionService = Depends(get_ingestion_service),
) -> NewsListResponse:
    try:
        items, total = service.list_articles(limit=limit, offset=offset, source_type=source_type)
    except StorageError as exc:
        raise _storage_failure(exc) from exc
    return NewsListResponse(items=items, total=total, limit=limit, offset=offset)


__all__ = ["router"]
