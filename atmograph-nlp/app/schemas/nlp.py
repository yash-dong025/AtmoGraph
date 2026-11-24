"""Pydantic models for Phase 3 entity extraction and annotations."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.nlp.taxonomy import EntityType
from app.schemas.article import SourceType

SourceField = Literal["title", "description"]
ExtractionMethod = Literal["spacy", "rule"]
BatchArticleStatus = Literal["extracted", "skipped", "not_found", "failed"]


class EntityAnnotation(BaseModel):
    """One entity mention with provenance and exact character offsets."""

    model_config = ConfigDict(extra="forbid")

    article_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=2_000, description="Original surface text.")
    normalized_name: str = Field(
        min_length=1, max_length=2_000, description="Cleaned, alias-resolved display name."
    )
    entity_type: EntityType = Field(description="Supply-chain entity taxonomy type.")
    source_field: SourceField = Field(description="Article field the mention came from.")
    start_char: int = Field(ge=0, description="Inclusive start offset in the source field.")
    end_char: int = Field(ge=0, description="Exclusive end offset in the source field.")
    extraction_method: ExtractionMethod = Field(
        description="``spacy`` for statistical NER, ``rule`` for a Matcher/PhraseMatcher hit."
    )
    raw_label: str | None = Field(
        default=None, description="Original spaCy or rule label, kept for debugging."
    )

    @model_validator(mode="after")
    def _validate_offsets(self) -> "EntityAnnotation":
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self


class ArticleAnnotation(BaseModel):
    """All entities extracted from one article."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "article_id": "mock-001",
                "title": "Typhoon Halts Container Operations at Kaohsiung Terminal",
                "source_type": "mock",
                "model_name": "en_core_web_sm",
                "extracted_at": "2026-11-24T10:00:00Z",
                "entity_count": 2,
                "entities": [
                    {
                        "article_id": "mock-001",
                        "text": "Kaohsiung Terminal",
                        "normalized_name": "Kaohsiung Terminal",
                        "entity_type": "FACILITY",
                        "source_field": "title",
                        "start_char": 38,
                        "end_char": 56,
                        "extraction_method": "rule",
                        "raw_label": "ATMO_FACILITY",
                    }
                ],
            }
        },
    )

    article_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=1_000)
    source_type: SourceType
    model_name: str = Field(min_length=1, max_length=200)
    extracted_at: datetime = Field(description="UTC time this annotation was produced.")
    entity_count: int = Field(ge=0)
    entities: list[EntityAnnotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_count(self) -> "ArticleAnnotation":
        if self.entity_count != len(self.entities):
            raise ValueError("entity_count must equal len(entities)")
        return self


class BatchExtractRequest(BaseModel):
    """Optional filters for ``POST /api/nlp/extract/batch``."""

    article_ids: list[str] | None = Field(
        default=None,
        max_length=500,
        description="Explicit article IDs. If omitted, articles are selected from storage.",
    )
    limit: int = Field(default=20, ge=1, le=500, description="Maximum articles to process.")
    offset: int = Field(default=0, ge=0, description="Articles to skip when selecting from storage.")
    source_type: SourceType | None = Field(
        default=None, description="Optional mock/rss filter when selecting from storage."
    )
    overwrite: bool = Field(
        default=True, description="Re-extract and replace annotations that already exist."
    )


class BatchArticleResult(BaseModel):
    """Outcome for one article in a batch extraction run."""

    article_id: str
    status: BatchArticleStatus
    entity_count: int = Field(default=0, ge=0)
    annotation: ArticleAnnotation | None = None
    error: str | None = None


class BatchExtractionResponse(BaseModel):
    """Aggregate response for a batch extraction run."""

    requested: int = Field(ge=0)
    processed: int = Field(ge=0, description="Articles extracted successfully.")
    skipped: int = Field(ge=0, description="Articles with existing annotations and overwrite=false.")
    not_found: int = Field(ge=0, description="Requested IDs that are not in article storage.")
    failed: int = Field(ge=0, description="Articles that could not be processed.")
    results: list[BatchArticleResult] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime


class EntityTypeCount(BaseModel):
    """Number of entities of one taxonomy type."""

    entity_type: EntityType
    count: int = Field(ge=0)


class NlpStatsResponse(BaseModel):
    """Aggregate annotation statistics across the local store."""

    total_articles: int = Field(ge=0, description="Articles currently in news storage.")
    annotated_articles: int = Field(ge=0, description="Articles with a stored annotation record.")
    articles_with_entities: int = Field(ge=0)
    total_entities: int = Field(ge=0)
    entity_counts: list[EntityTypeCount] = Field(default_factory=list)
    average_entities_per_article: float = Field(ge=0.0)
    model_name: str
    last_extracted_at: datetime | None = None


class NlpHealthResponse(BaseModel):
    """Health of the NLP model and annotation store."""

    status: Literal["ok", "degraded"]
    model_name: str
    model_available: bool
    model_loaded: bool
    model_version: str | None = None
    pipeline_components: list[str] = Field(default_factory=list)
    pattern_count: int = Field(ge=0, description="Configured phrase/token rule patterns.")
    annotation_store_path: str
    annotation_store_readable: bool
    timestamp: datetime


__all__ = [
    "ArticleAnnotation",
    "BatchArticleResult",
    "BatchArticleStatus",
    "BatchExtractRequest",
    "BatchExtractionResponse",
    "EntityAnnotation",
    "EntityTypeCount",
    "ExtractionMethod",
    "NlpHealthResponse",
    "NlpStatsResponse",
    "SourceField",
]
