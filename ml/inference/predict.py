"""Prediction output contract for backend integration.

This module does not generate predictions. It validates and formats a score that
must come from a trained model. Member 1 can use this contract in FastAPI once a
model artifact and a validated label definition exist.

Important: the current repository has no verified company-level risk labels.
``prediction_status`` must remain ``experimental`` until a labelled evaluation is
completed and approved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

PredictionType = Literal["disruption_risk", "shipment_delay", "link_prediction"]
PredictionStatus = Literal["experimental", "validated"]


@dataclass(frozen=True)
class PredictionRecord:
    company_id: str
    risk_score: float
    risk_level: str
    prediction_type: PredictionType = "disruption_risk"
    model_version: str = "gcn_v1"
    prediction_status: PredictionStatus = "experimental"

    def validate(self) -> None:
        if not self.company_id or not isinstance(self.company_id, str):
            raise ValueError("company_id must be a non-empty string")
        if not 0.0 <= float(self.risk_score) <= 1.0:
            raise ValueError("risk_score must be in [0, 1]")
        if self.risk_level not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("risk_level must be LOW, MEDIUM, or HIGH")
        if not self.model_version:
            raise ValueError("model_version is required")
        if self.prediction_status not in {"experimental", "validated"}:
            raise ValueError("prediction_status must be experimental or validated")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def risk_level_from_score(score: float) -> str:
    if not 0.0 <= float(score) <= 1.0:
        raise ValueError("score must be in [0, 1]")
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


def build_prediction_record(
    company_id: str,
    risk_score: float,
    model_version: str,
    prediction_type: PredictionType = "disruption_risk",
    prediction_status: PredictionStatus = "experimental",
) -> dict[str, Any]:
    """Build a validated output record from a real model score.

    The caller is responsible for producing ``risk_score`` from a trained model.
    This function only formats and validates the record.
    """
    record = PredictionRecord(
        company_id=company_id,
        risk_score=float(risk_score),
        risk_level=risk_level_from_score(float(risk_score)),
        prediction_type=prediction_type,
        model_version=model_version,
        prediction_status=prediction_status,
    )
    return record.to_dict()
