"""Business validations for graph write operations."""
from fastapi import HTTPException
from neo4j.exceptions import ConstraintError
from app.repositories import graph_repository as repo


def ensure_entity(entity_id: str):
    entity = repo.get_entity(entity_id)
    if entity is None:
        raise HTTPException(404, detail=f"Entity '{entity_id}' not found")
    return entity


def add_entity(data: dict):
    if repo.get_entity(data["id"]):
        raise HTTPException(409, detail="Entity ID already exists")
    try:
        return repo.create_entity(data)
    except ConstraintError as exc:
        raise HTTPException(409, detail="Entity ID already exists") from exc


def add_relation(data: dict):
    if data["source_id"] == data["target_id"]:
        raise HTTPException(400, detail="Self-links are not allowed")
    ensure_entity(data["source_id"])
    ensure_entity(data["target_id"])
    return repo.create_relation(data)
