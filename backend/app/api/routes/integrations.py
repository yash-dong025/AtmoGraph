from fastapi import APIRouter, HTTPException
from app.models.integration_models import DisruptionCreate, PredictionCreate
from app.repositories import graph_repository as repo
from app.services.graph_service import ensure_entity

router = APIRouter(prefix="/api/v1", tags=["Team Integrations"])


@router.post("/disruptions", status_code=201)
def post_disruption(payload: DisruptionCreate):
    ensure_entity(payload.entity_id)
    rows = repo.record_disruption(payload.model_dump(mode="json"))
    return rows[0] if rows else {"detail": "Already recorded"}


@router.post("/predictions", status_code=201)
def post_prediction(payload: PredictionCreate):
    ensure_entity(payload.entity_id)
    rows = repo.record_prediction(payload.model_dump(mode="json"))
    return rows[0] if rows else {"detail": "Already recorded"}


@router.get("/predictions/{entity_id}")
def get_predictions(entity_id: str):
    ensure_entity(entity_id)
    return [row["prediction"] for row in repo.predictions(entity_id)]
