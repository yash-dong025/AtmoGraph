from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class DisruptionKind(str, Enum):
    flood = "FLOOD"
    strike = "STRIKE"
    port_closure = "PORT_CLOSURE"
    fire = "FIRE"
    earthquake = "EARTHQUAKE"
    transport_delay = "TRANSPORT_DELAY"
    other = "OTHER"


class DisruptionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=100)
    entity_id: str
    kind: DisruptionKind
    severity: str = Field(pattern=r"^(LOW|MEDIUM|HIGH|CRITICAL)$")
    description: str = Field(min_length=1, max_length=2000)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    risk_score: float = Field(ge=0, le=100)


class PredictionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=100)
    entity_id: str
    horizon_days: int = Field(ge=1, le=365)
    predicted_delay_days: float = Field(ge=0)
    risk_score: float = Field(ge=0, le=100)
    model_name: str = Field(min_length=1, max_length=100)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
