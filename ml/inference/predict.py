"""Inference output contracts for AtmoGraph.

The supported task is directed supplier/customer link prediction. The score is
an uncalibrated model score, not a verified probability and not a disruption-risk
score. Disruption-risk prediction is a separate unsupported task until verified
company-level labels and a dedicated model exist.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

LinkPredictionTask = Literal["link_prediction"]
LinkPredictionStatus = Literal["experimental"]
LinkScoreType = Literal["model_score"]

DEFAULT_WARNINGS = (
    "This is a directed supplier/customer link score, not a disruption-risk prediction.",
    "The link score is uncalibrated and must not be interpreted as a verified probability.",
    "No verified company-level risk labels exist in the current datasets.",
)

SCORE_SEMANTICS = (
    "Uncalibrated GraphSAGE score for whether a directed supplier/customer link "
    "is likely to exist. Not a verified probability and not a disruption-risk score."
)


class CheckpointUnavailableError(FileNotFoundError):
    """Raised when a link-prediction checkpoint is missing or unreadable."""


class RiskPredictionUnsupportedError(NotImplementedError):
    """Raised when disruption-risk prediction is requested without labels/model."""


@dataclass(frozen=True)
class LinkPredictionRecord:
    source_id: str
    target_id: str
    link_score: float
    source: dict[str, Any]
    target: dict[str, Any]
    model: dict[str, Any]
    task: LinkPredictionTask = "link_prediction"
    score_type: LinkScoreType = "model_score"
    score_semantics: str = SCORE_SEMANTICS
    probability_calibrated: bool = False
    prediction_status: LinkPredictionStatus = "experimental"
    warnings: tuple[str, ...] = field(default=DEFAULT_WARNINGS)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError("target_id must be a non-empty string")
        if self.source_id == self.target_id:
            raise ValueError("source_id and target_id must be different")
        score = float(self.link_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("link_score must be a finite value in [0, 1]")
        if self.task != "link_prediction":
            raise ValueError("task must be link_prediction")
        if self.score_type != "model_score":
            raise ValueError("score_type must be model_score")
        if self.prediction_status != "experimental":
            raise ValueError("prediction_status must remain experimental")
        if self.probability_calibrated:
            raise ValueError("link scores are not calibrated probabilities")
        if not isinstance(self.source, dict) or not self.source:
            raise ValueError("source metadata is required")
        if not isinstance(self.target, dict) or not self.target:
            raise ValueError("target metadata is required")
        if not isinstance(self.model, dict) or not self.model:
            raise ValueError("model metadata is required")
        if not self.model.get("name"):
            raise ValueError("model.name is required")
        if not self.model.get("version"):
            raise ValueError("model.version is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def ensure_checkpoint_available(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    if not path.exists():
        raise CheckpointUnavailableError(
            f"Link-prediction checkpoint is missing: {path}. No score was generated."
        )
    if not path.is_file():
        raise CheckpointUnavailableError(
            f"Link-prediction checkpoint is not a file: {path}. No score was generated."
        )
    return path


def load_link_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    path = ensure_checkpoint_available(checkpoint_path)
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Link-prediction checkpoint must be a dictionary")
    required = {"model_state_dict", "feature_names", "model_config"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Link-prediction checkpoint is missing keys: {missing}")
    return checkpoint


def build_link_prediction_record(
    source_id: str,
    target_id: str,
    link_score: float,
    source: dict[str, Any],
    target: dict[str, Any],
    model_name: str,
    model_version: str,
    model_type: str = "GraphSAGE",
    dataset_version: str | None = None,
    protocol: str | None = None,
    feature_config: str | None = None,
    checkpoint_path: str | None = None,
    warnings: tuple[str, ...] = DEFAULT_WARNINGS,
) -> dict[str, Any]:
    source_metadata = dict(source)
    target_metadata = dict(target)
    source_metadata.setdefault("id", source_id)
    target_metadata.setdefault("id", target_id)
    record = LinkPredictionRecord(
        source_id=source_id,
        target_id=target_id,
        link_score=float(link_score),
        source=source_metadata,
        target=target_metadata,
        model={
            "name": model_name,
            "type": model_type,
            "version": model_version,
            "dataset_version": dataset_version,
            "protocol": protocol,
            "feature_config": feature_config,
            "checkpoint_path": checkpoint_path,
        },
        warnings=tuple(warnings),
    )
    return record.to_dict()


def build_risk_prediction_record(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise RiskPredictionUnsupportedError(
        "Disruption-risk prediction is a separate unsupported task. "
        "It requires verified company-level labels and a dedicated model."
    )
