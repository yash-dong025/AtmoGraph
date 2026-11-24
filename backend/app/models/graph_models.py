from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class EntityKind(str, Enum):
    supplier = "SUPPLIER"
    manufacturer = "MANUFACTURER"
    port = "PORT"
    warehouse = "WAREHOUSE"
    retailer = "RETAILER"


class RelationKind(str, Enum):
    supplies_to = "SUPPLIES_TO"
    ships_to = "SHIPS_TO"
    distributes_to = "DISTRIBUTES_TO"


class EntityCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=200)
    kind: EntityKind
    country: str = Field(min_length=2, max_length=100)
    city: str | None = Field(default=None, max_length=100)
    risk_score: float = Field(default=0, ge=0, le=100)


class EntityUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=200)
    country: str | None = Field(default=None, min_length=2, max_length=100)
    city: str | None = Field(default=None, max_length=100)
    risk_score: float | None = Field(default=None, ge=0, le=100)


class RelationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    target_id: str
    kind: RelationKind
    transit_days: float = Field(default=1, ge=0)
