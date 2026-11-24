import pytest
from pydantic import ValidationError
from app.models.graph_models import EntityCreate, RelationCreate


def test_entity_rejects_invalid_risk():
    with pytest.raises(ValidationError):
        EntityCreate(id="node", name="Port", kind="PORT", country="India", risk_score=101)


def test_rejects_unknown_relation():
    with pytest.raises(ValidationError):
        RelationCreate(source_id="a", target_id="b", kind="INJECTED")
