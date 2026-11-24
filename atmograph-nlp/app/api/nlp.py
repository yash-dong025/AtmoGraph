"""Phase 3 entity extraction and annotation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status

from app.nlp.model_loader import NlpModelError
from app.nlp.patterns import PatternConfigError
from app.nlp.service import (
    AnnotationNotFoundError,
    ArticleNotFoundError,
    NlpService,
    get_nlp_service,
)
from app.schemas.ingestion import ErrorResponse
from app.schemas.nlp import (
    ArticleAnnotation,
    BatchExtractRequest,
    BatchExtractionResponse,
    NlpHealthResponse,
    NlpStatsResponse,
)
from app.storage.annotation_storage import AnnotationStorageError
from app.storage.json_storage import StorageError

router = APIRouter(prefix="/nlp", tags=["nlp"])

_STORAGE_ERRORS = {
    500: {
        "model": ErrorResponse,
        "description": "The article or annotation store is unreadable or contains invalid JSON.",
    }
}
_EXTRACTION_ERRORS = {
    404: {"model": ErrorResponse, "description": "The article does not exist."},
    500: {
        "model": ErrorResponse,
        "description": "The article or annotation store is unreadable or contains invalid JSON.",
    },
    503: {
        "model": ErrorResponse,
        "description": "The spaCy model or configured entity patterns are unavailable.",
    },
}
# Batch requests report missing articles per item instead of failing the request.
_BATCH_ERRORS = {
    422: {
        "model": ErrorResponse,
        "description": "The batch is larger than the configured limit or contains an empty ID.",
    },
    500: _EXTRACTION_ERRORS[500],
    503: _EXTRACTION_ERRORS[503],
}


def _not_found(exc: LookupError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


def _storage_failure(exc: Exception) -> HTTPException:
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


def _nlp_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=str(exc),
    )


@router.get(
    "/health",
    response_model=NlpHealthResponse,
    summary="Check NLP model and annotation-store health",
    description=(
        "Reports whether the configured spaCy model package is installed, whether it has been "
        "loaded in this process, the configured rule-pattern count, and whether the annotation "
        "store is readable. The endpoint does not load the model just to answer the check."
    ),
)
def nlp_health(service: NlpService = Depends(get_nlp_service)) -> NlpHealthResponse:
    return service.health()


@router.get(
    "/stats",
    response_model=NlpStatsResponse,
    summary="Get entity-annotation statistics",
    description=(
        "Returns article totals, annotation coverage, total entity mentions, and counts for every "
        "taxonomy type. Counts include zero values so dashboards can render a stable set of keys."
    ),
    responses=_STORAGE_ERRORS,
)
def nlp_stats(service: NlpService = Depends(get_nlp_service)) -> NlpStatsResponse:
    try:
        return service.stats()
    except (StorageError, AnnotationStorageError) as exc:
        raise _storage_failure(exc) from exc


@router.post(
    "/extract/batch",
    response_model=BatchExtractionResponse,
    summary="Extract entities for several articles",
    description=(
        "Supply explicit `article_ids`, or omit them to select a page from article storage using "
        "`limit`, `offset` and `source_type`. Explicit IDs take precedence over pagination. "
        "The request size is capped by `ATMOGRAPH_MAX_BATCH_SIZE` (default 50). Existing "
        "annotations are replaced unless `overwrite` is false."
    ),
    responses=_BATCH_ERRORS,
)
def extract_batch(
    payload: BatchExtractRequest | None = Body(default=None),
    service: NlpService = Depends(get_nlp_service),
) -> BatchExtractionResponse:
    try:
        return service.extract_batch(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except (StorageError, AnnotationStorageError) as exc:
        raise _storage_failure(exc) from exc
    except (NlpModelError, PatternConfigError) as exc:
        raise _nlp_unavailable(exc) from exc


@router.post(
    "/extract/{article_id}",
    response_model=ArticleAnnotation,
    summary="Extract entities for one stored article",
    description=(
        "Runs spaCy NER plus the configured supply-chain rules, normalizes entity names, and "
        "upserts the annotation in `data/entity_annotations.json`. Calling the endpoint again "
        "replaces the previous annotation. Use `overwrite=false` to return an existing annotation "
        "unchanged."
    ),
    responses=_EXTRACTION_ERRORS,
)
def extract_article(
    article_id: str = Path(min_length=1, max_length=128, description="Stored article ID."),
    overwrite: bool = Query(
        default=True,
        description="Replace an existing annotation. Set false to reuse it.",
    ),
    service: NlpService = Depends(get_nlp_service),
) -> ArticleAnnotation:
    try:
        return service.extract_article(article_id, overwrite=overwrite)
    except ArticleNotFoundError as exc:
        raise _not_found(exc) from exc
    except (StorageError, AnnotationStorageError) as exc:
        raise _storage_failure(exc) from exc
    except (NlpModelError, PatternConfigError) as exc:
        raise _nlp_unavailable(exc) from exc


@router.get(
    "/entities/{article_id}",
    response_model=ArticleAnnotation,
    summary="Get stored entities for one article",
    description=(
        "Returns the saved annotation for an article. Returns 404 when the article itself does "
        "not exist or when it has not been extracted yet; call `POST /api/nlp/extract/{article_id}` "
        "first."
    ),
    responses={
        404: {
            "model": ErrorResponse,
            "description": "The article or its annotation does not exist.",
        },
        **_STORAGE_ERRORS,
    },
)
def get_entities(
    article_id: str = Path(min_length=1, max_length=128, description="Stored article ID."),
    service: NlpService = Depends(get_nlp_service),
) -> ArticleAnnotation:
    try:
        return service.get_annotation(article_id)
    except (ArticleNotFoundError, AnnotationNotFoundError) as exc:
        raise _not_found(exc) from exc
    except (StorageError, AnnotationStorageError) as exc:
        raise _storage_failure(exc) from exc


__all__ = ["router"]
