from fastapi import APIRouter, HTTPException
from neo4j.exceptions import Neo4jError
from app.database.neo4j_connection import db

router = APIRouter(tags=["Health"])


@router.get("/")
def home():
    return {"project": "AtmoGraph", "module": "Member 1", "status": "running"}


@router.get("/health")
def health():
    try:
        db.verify_connection()
        return {"status": "healthy", "database": "connected"}
    except (Neo4jError, OSError, RuntimeError):
        raise HTTPException(status_code=503, detail="Neo4j unavailable; check Neo4j Desktop and .env")
