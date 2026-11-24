"""Request-independent response models for the Phase 2 news API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.article import Article

IngestionSource = Literal["mock", "rss", "all"]
IngestionStatus = Literal["success", "partial", "failed"]


class FeedConfig(BaseModel):
    """One configured RSS feed (no credentials are ever stored here)."""

    name: str = Field(description="Human-readable feed name.")
    url: str = Field(description="Public RSS/Atom URL.")
    category: str = Field(default="general", description="Editorial category used for orientation only.")


class FeedListResponse(BaseModel):
    """Payload returned by ``GET /api/news/feeds``."""

    count: int = Field(ge=0, description="Number of configured feeds.")
    feeds: list[FeedConfig] = Field(description="Configured feed names and public URLs.")


class IngestionStats(BaseModel):
    """Counters describing one ingestion run."""

    status: IngestionStatus = Field(description="``success``, ``partial`` or ``failed``.")
    total_fetched: int = Field(ge=0, description="Raw records/entries read from the source(s).")
    total_saved: int = Field(ge=0, description="New articles written to storage.")
    duplicates_skipped: int = Field(ge=0, description="Records skipped because they already existed.")
    invalid_articles: int = Field(ge=0, description="Records rejected by schema validation.")
    feeds_successful: int = Field(ge=0, description="RSS feeds fetched and parsed successfully.")
    feeds_failed: int = Field(ge=0, description="RSS feeds that could not be fetched or parsed.")


class IngestionResponse(IngestionStats):
    """Full response for the ingest endpoints, including per-run context."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "source": "mock",
                "total_fetched": 22,
                "total_saved": 22,
                "duplicates_skipped": 0,
                "invalid_articles": 0,
                "feeds_successful": 0,
                "feeds_failed": 0,
                "errors": [],
                "started_at": "2026-11-22T10:00:00Z",
                "finished_at": "2026-11-22T10:00:01Z",
            }
        }
    )

    source: IngestionSource = Field(description="Source that was ingested.")
    errors: list[str] = Field(default_factory=list, description="Non-fatal feed/load errors, if any.")
    started_at: datetime = Field(description="UTC start time of the run.")
    finished_at: datetime = Field(description="UTC end time of the run.")


class NewsListResponse(BaseModel):
    """Paginated article listing."""

    items: list[Article] = Field(description="Articles in storage order.")
    total: int = Field(ge=0, description="Total matching articles before pagination.")
    limit: int = Field(ge=1, description="Applied page size.")
    offset: int = Field(ge=0, description="Applied offset.")


class NewsStatsResponse(BaseModel):
    """Storage counts plus counters from the most recent run in this process."""

    total_articles: int = Field(ge=0)
    rss_articles: int = Field(ge=0)
    mock_articles: int = Field(ge=0)
    duplicates_skipped: int = Field(ge=0, description="Duplicates skipped by the most recent run.")
    invalid_articles: int = Field(ge=0, description="Invalid records rejected by the most recent run.")
    feeds_successful: int = Field(ge=0, description="Successful feeds in the most recent run.")
    feeds_failed: int = Field(ge=0, description="Failed feeds in the most recent run.")
    last_ingestion_source: IngestionSource | None = Field(
        default=None, description="Source of the most recent run, or ``null`` if none has run."
    )
    last_ingestion_at: datetime | None = Field(
        default=None, description="UTC finish time of the most recent run."
    )


class ErrorResponse(BaseModel):
    """Standard FastAPI error payload."""

    detail: str


__all__ = [
    "ErrorResponse",
    "FeedConfig",
    "FeedListResponse",
    "IngestionResponse",
    "IngestionSource",
    "IngestionStats",
    "IngestionStatus",
    "NewsListResponse",
    "NewsStatsResponse",
]
