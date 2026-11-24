from fastapi import APIRouter, HTTPException, Query, Response
from app.models.graph_models import EntityCreate, EntityUpdate, EntityKind, RelationCreate, RelationKind
from app.repositories import graph_repository as repo
from app.services.graph_service import add_entity, add_relation, ensure_entity

router = APIRouter(prefix="/api/v1", tags=["Graph CRUD"])


@router.post("/entities", status_code=201)
def create_entity(payload: EntityCreate):
    return add_entity(payload.model_dump(mode="json"))


@router.get("/entities")
def list_entities(kind: EntityKind | None = None, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    return [row["entity"] for row in repo.list_entities(kind.value if kind else None, limit, offset)]


@router.get("/entities/{entity_id}")
def get_entity(entity_id: str):
    return ensure_entity(entity_id)


@router.patch("/entities/{entity_id}")
def update_entity(entity_id: str, payload: EntityUpdate):
    ensure_entity(entity_id)
    changes = payload.model_dump(exclude_unset=True, mode="json")
    if not changes:
        raise HTTPException(400, detail="No fields supplied")
    if any(changes.get(field) is None for field in ("name", "country", "risk_score") if field in changes):
        raise HTTPException(422, detail="name, country and risk_score cannot be null")
    return repo.update_entity(entity_id, changes)


@router.delete("/entities/{entity_id}", status_code=204)
def delete_entity(entity_id: str):
    if not repo.delete_entity(entity_id):
        raise HTTPException(404, detail="Entity not found")
    return Response(status_code=204)


@router.post("/relationships", status_code=201)
def create_relationship(payload: RelationCreate):
    return add_relation(payload.model_dump(mode="json"))


@router.delete("/relationships", status_code=204)
def delete_relationship(source_id: str, target_id: str, kind: RelationKind):
    if not repo.delete_relation(source_id, target_id, kind.value):
        raise HTTPException(404, detail="Relationship not found")
    return Response(status_code=204)
