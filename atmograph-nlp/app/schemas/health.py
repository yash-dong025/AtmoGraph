"""Response models for service health and readiness checks."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ComponentStatus(BaseModel):
    """Availability of one Python dependency or model package."""

    name: str
    available: bool
    required: bool
    version: str | None = None


class FeatureFlags(BaseModel):
    """Configuration flags that explain how the service is running."""

    transformer_classifier: bool
    graph_sync: bool
    spacy_model: str


class HealthResponse(BaseModel):
    """Payload returned by ``GET /api/health``."""

    status: Literal["ok", "degraded"]
    service: str
    version: str
    environment: str
    timestamp: datetime = Field(description="UTC timestamp of the check")
    components: list[ComponentStatus]
    features: FeatureFlags
