"""AtmoGraph NLP & News Intelligence API.

Phase 2 adds news ingestion, article listing and ingestion statistics.  NLP,
disruption classification and graph integration endpoints are added in later
phases.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from importlib import metadata
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.news import router as news_router
from app.config import get_settings
from app.schemas.health import ComponentStatus, FeatureFlags, HealthResponse
from app.utils.helpers import utc_now
from app.utils.logger import configure_logging, get_logger

settings = get_settings()
configure_logging(settings)
logger = get_logger(__name__)

# (distribution name, required for the service to be considered healthy)
_DEPENDENCIES: tuple[tuple[str, bool], ...] = (
    ("fastapi", True),
    ("pydantic", True),
    ("spacy", True),
    ("feedparser", True),
    ("httpx", True),
    ("transformers", False),
    ("torch", False),
    ("rapidfuzz", False),
)


def _check_component(name: str, required: bool) -> ComponentStatus:
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        return ComponentStatus(name=name, available=False, required=required)
    return ComponentStatus(name=name, available=True, required=required, version=version)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.environment)
    logger.info("Interactive API docs are served at /docs")
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "NLP & News Intelligence module for AtmoGraph: news ingestion, entity "
        "extraction, disruption classification, risk scoring and graph/GNN payloads."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(news_router, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
def service_info() -> dict[str, str]:
    """Human-friendly entry point that points at the interactive docs."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }


@app.get(
    f"{settings.api_prefix}/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Check service status",
)
def health() -> HealthResponse:
    """Report dependency availability and active feature flags.

    The endpoint deliberately avoids importing heavy libraries so it stays
    fast enough for container/orchestrator probes.
    """
    components = [_check_component(name, required) for name, required in _DEPENDENCIES]
    components.append(
        _check_component(settings.spacy_model, required=False)
    )

    healthy = all(component.available for component in components if component.required)
    status = "ok" if healthy else "degraded"
    if not healthy:
        logger.warning(
            "Health check degraded; missing required components: %s",
            ", ".join(c.name for c in components if c.required and not c.available),
        )

    return HealthResponse(
        status=status,
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        timestamp=utc_now(),
        components=components,
        features=FeatureFlags(
            transformer_classifier=settings.enable_transformer,
            graph_sync=settings.graph_sync_enabled,
            spacy_model=settings.spacy_model,
        ),
    )
