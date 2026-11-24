from fastapi import APIRouter, Query
from app.repositories import graph_repository as repo
from app.services.graph_service import ensure_entity

router = APIRouter(prefix="/api/v1/graph", tags=["Graph Explorer"])


@router.get("")
def get_graph(limit: int = Query(500, ge=1, le=2000)):
    return repo.graph_export(limit)


@router.get("/downstream/{entity_id}")
def get_downstream(entity_id: str, max_hops: int = Query(3, ge=1, le=6), limit: int = Query(100, ge=1, le=500)):
    ensure_entity(entity_id)
    return {"source_id": entity_id, "max_hops": max_hops, "downstream": repo.downstream(entity_id, max_hops, limit)}
