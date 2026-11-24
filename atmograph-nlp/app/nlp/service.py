"""Orchestration for Phase 3 entity extraction.

The service owns the article/annotation stores and the extractor.  Routes stay
thin, the extractor stays stateless, and the lazy extractor initialisation lets
the health endpoint report a missing model instead of failing at import time.
"""

from __future__ import annotations

import threading
from functools import lru_cache

from app.config import Settings, get_settings
from app.nlp.extractor import ArticleEntityExtractor
from app.nlp.model_loader import (
    NlpModelError,
    get_loaded_model,
    is_model_available,
    is_model_loaded,
    model_metadata,
)
from app.nlp.patterns import build_patterns
from app.nlp.taxonomy import ENTITY_TYPES
from app.schemas.article import Article
from app.schemas.nlp import (
    ArticleAnnotation,
    BatchArticleResult,
    BatchExtractRequest,
    BatchExtractionResponse,
    EntityTypeCount,
    NlpHealthResponse,
    NlpStatsResponse,
)
from app.storage.annotation_storage import (
    AnnotationStorageError,
    EntityAnnotationStorage,
    JsonEntityAnnotationStorage,
)
from app.storage.json_storage import ArticleStorage, JsonArticleStorage
from app.utils.helpers import utc_now
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ArticleNotFoundError(LookupError):
    """Raised when an article ID is not present in article storage."""

    def __init__(self, article_id: str) -> None:
        self.article_id = article_id
        super().__init__(f"Article '{article_id}' was not found in the article store.")


class AnnotationNotFoundError(LookupError):
    """Raised when an article has no stored annotation yet."""

    def __init__(self, article_id: str) -> None:
        self.article_id = article_id
        super().__init__(
            f"No entity annotation exists for article '{article_id}'. "
            f"Run POST /api/nlp/extract/{article_id} first."
        )


class BatchLimitError(ValueError):
    """Raised when a batch request exceeds the configured maximum size."""


class NlpService:
    """Extract, persist and query entity annotations for stored articles."""

    def __init__(
        self,
        article_storage: ArticleStorage,
        annotation_storage: EntityAnnotationStorage,
        settings: Settings | None = None,
        extractor: ArticleEntityExtractor | None = None,
    ) -> None:
        self._article_storage = article_storage
        self._annotation_storage = annotation_storage
        self._settings = settings or get_settings()
        self._extractor = extractor
        self._extractor_lock = threading.RLock()

    # -- extraction --------------------------------------------------------
    def extract_article(self, article_id: str, *, overwrite: bool = True) -> ArticleAnnotation:
        """Extract and persist annotations for one stored article."""
        article = self._find_article(article_id)
        if article is None:
            raise ArticleNotFoundError(article_id)
        annotation, _ = self._extract_article_object(article, overwrite=overwrite)
        return annotation

    def extract_batch(self, request: BatchExtractRequest | None = None) -> BatchExtractionResponse:
        """Extract annotations for explicit IDs or a page selected from storage."""
        payload = request or BatchExtractRequest()
        started_at = utc_now()

        if payload.article_ids is None and payload.limit > self._settings.max_batch_size:
            raise BatchLimitError(
                f"Batch limit {payload.limit} exceeds the configured maximum of "
                f"{self._settings.max_batch_size}."
            )

        results: list[BatchArticleResult] = []
        if payload.article_ids is not None:
            article_ids = _unique_ids(payload.article_ids)
            if len(article_ids) > self._settings.max_batch_size:
                raise BatchLimitError(
                    f"Batch size {len(article_ids)} exceeds the configured maximum of "
                    f"{self._settings.max_batch_size}."
                )
            for article_id in article_ids:
                results.append(self._extract_by_id(article_id, overwrite=payload.overwrite))
        else:
            articles = self._article_storage.load()
            if payload.source_type is not None:
                articles = [
                    article for article in articles if article.source_type == payload.source_type
                ]
            selected = articles[payload.offset : payload.offset + payload.limit]
            for article in selected:
                results.append(
                    self._extract_object_result(article, overwrite=payload.overwrite)
                )

        processed = sum(1 for result in results if result.status == "extracted")
        skipped = sum(1 for result in results if result.status == "skipped")
        not_found = sum(1 for result in results if result.status == "not_found")
        failed = sum(1 for result in results if result.status == "failed")
        finished_at = utc_now()

        logger.info(
            "Entity extraction batch requested=%d extracted=%d skipped=%d not_found=%d failed=%d",
            len(results),
            processed,
            skipped,
            not_found,
            failed,
        )

        return BatchExtractionResponse(
            requested=len(results),
            processed=processed,
            skipped=skipped,
            not_found=not_found,
            failed=failed,
            results=results,
            started_at=started_at,
            finished_at=finished_at,
        )

    # -- queries -----------------------------------------------------------
    def get_annotation(self, article_id: str) -> ArticleAnnotation:
        """Return stored annotations, raising a typed error when absent."""
        if self._find_article(article_id) is None:
            raise ArticleNotFoundError(article_id)
        annotation = self._annotation_storage.get(article_id)
        if annotation is None:
            raise AnnotationNotFoundError(article_id)
        return annotation

    def stats(self) -> NlpStatsResponse:
        """Aggregate storage totals and entity-type counts."""
        articles = self._article_storage.load()
        annotations = self._annotation_storage.load()

        counts: dict[str, int] = {entity_type: 0 for entity_type in ENTITY_TYPES}
        for annotation in annotations:
            for entity in annotation.entities:
                counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1

        total_entities = sum(counts.values())
        annotated_articles = len(annotations)
        last_extracted_at = max(
            (annotation.extracted_at for annotation in annotations),
            default=None,
        )

        return NlpStatsResponse(
            total_articles=len(articles),
            annotated_articles=annotated_articles,
            articles_with_entities=sum(1 for annotation in annotations if annotation.entities),
            total_entities=total_entities,
            entity_counts=[
                EntityTypeCount(entity_type=entity_type, count=counts.get(entity_type, 0))
                for entity_type in ENTITY_TYPES
            ],
            average_entities_per_article=(
                total_entities / annotated_articles if annotated_articles else 0.0
            ),
            model_name=self._model_name,
            last_extracted_at=last_extracted_at,
        )

    def health(self) -> NlpHealthResponse:
        """Report model availability and annotation-store readability."""
        model_name = self._settings.spacy_model
        model_available = is_model_available(model_name)
        model_loaded = is_model_loaded(model_name)
        metadata = model_metadata(model_name) if model_loaded else None

        pipeline = get_loaded_model(model_name) if model_loaded else None
        pipeline_components = list(pipeline.pipe_names) if pipeline is not None else []

        try:
            if self._extractor is not None:
                pattern_count = self._extractor.pattern_count
            else:
                pattern_count = len(build_patterns(self._settings.nlp_patterns_path))
            patterns_ok = True
        except Exception as exc:  # malformed optional pattern file
            logger.warning("Cannot load NLP patterns for health check: %s", exc)
            pattern_count = 0
            patterns_ok = False

        try:
            self._annotation_storage.load()
            store_readable = True
        except AnnotationStorageError as exc:
            logger.warning("Annotation store is not readable: %s", exc)
            store_readable = False

        healthy = model_available and store_readable and patterns_ok
        return NlpHealthResponse(
            status="ok" if healthy else "degraded",
            model_name=model_name,
            model_available=model_available,
            model_loaded=model_loaded,
            model_version=str(metadata.get("version")) if metadata else None,
            pipeline_components=pipeline_components,
            pattern_count=pattern_count,
            annotation_store_path=str(self._annotation_storage.path)
            if isinstance(self._annotation_storage, JsonEntityAnnotationStorage)
            else type(self._annotation_storage).__name__,
            annotation_store_readable=store_readable,
            timestamp=utc_now(),
        )

    # -- internals ---------------------------------------------------------
    @property
    def _model_name(self) -> str:
        return self._extractor.model_name if self._extractor else self._settings.spacy_model

    def _get_extractor(self) -> ArticleEntityExtractor:
        with self._extractor_lock:
            if self._extractor is None:
                self._extractor = ArticleEntityExtractor(self._settings)
            return self._extractor

    def _find_article(self, article_id: str) -> Article | None:
        for article in self._article_storage.load():
            if article.article_id == article_id:
                return article
        return None

    def _extract_article_object(
        self,
        article: Article,
        *,
        overwrite: bool,
    ) -> tuple[ArticleAnnotation, str]:
        existing = self._annotation_storage.get(article.article_id)
        if existing is not None and not overwrite:
            return existing, "skipped"

        extractor = self._get_extractor()
        entities = extractor.extract(article)
        annotation = ArticleAnnotation(
            article_id=article.article_id,
            title=article.title,
            source_type=article.source_type,
            model_name=extractor.model_name,
            extracted_at=utc_now(),
            entity_count=len(entities),
            entities=entities,
        )
        self._annotation_storage.upsert(annotation)
        logger.info(
            "Extracted %d entities from article %s", annotation.entity_count, article.article_id
        )
        return annotation, "extracted"

    def _extract_by_id(self, article_id: str, *, overwrite: bool) -> BatchArticleResult:
        try:
            annotation, status = self._extract_article_object(
                self._require_article(article_id),
                overwrite=overwrite,
            )
        except ArticleNotFoundError:
            return BatchArticleResult(article_id=article_id, status="not_found")
        except (AnnotationStorageError, NlpModelError):
            raise
        except Exception as exc:  # pragma: no cover - defensive per-article isolation
            logger.exception("Entity extraction failed for article %s", article_id)
            return BatchArticleResult(
                article_id=article_id,
                status="failed",
                error=str(exc),
            )
        return BatchArticleResult(
            article_id=article_id,
            status=status,
            entity_count=annotation.entity_count,
            annotation=annotation,
        )

    def _extract_object_result(
        self,
        article: Article,
        *,
        overwrite: bool,
    ) -> BatchArticleResult:
        try:
            annotation, status = self._extract_article_object(article, overwrite=overwrite)
        except (AnnotationStorageError, NlpModelError):
            raise
        except Exception as exc:  # pragma: no cover - defensive per-article isolation
            logger.exception("Entity extraction failed for article %s", article.article_id)
            return BatchArticleResult(
                article_id=article.article_id,
                status="failed",
                error=str(exc),
            )
        return BatchArticleResult(
            article_id=article.article_id,
            status=status,
            entity_count=annotation.entity_count,
            annotation=annotation,
        )

    def _require_article(self, article_id: str) -> Article:
        article = self._find_article(article_id)
        if article is None:
            raise ArticleNotFoundError(article_id)
        return article


def _unique_ids(article_ids: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in article_ids:
        article_id = raw.strip()
        if not article_id:
            raise ValueError("article_ids cannot contain empty strings")
        if article_id in seen:
            continue
        seen.add(article_id)
        cleaned.append(article_id)
    return cleaned


@lru_cache(maxsize=1)
def get_nlp_service() -> NlpService:
    """Process-wide service used by FastAPI dependency injection."""
    settings = get_settings()
    return NlpService(
        article_storage=JsonArticleStorage(settings.articles_store_path),
        annotation_storage=JsonEntityAnnotationStorage(settings.entity_annotations_path),
        settings=settings,
    )


__all__ = [
    "AnnotationNotFoundError",
    "ArticleNotFoundError",
    "BatchLimitError",
    "NlpService",
    "get_nlp_service",
]
