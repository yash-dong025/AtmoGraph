import json

import pytest

from ml.inference.predict import (
    CheckpointUnavailableError,
    RiskPredictionUnsupportedError,
    build_link_prediction_record,
    build_risk_prediction_record,
    ensure_checkpoint_available,
)


def _record() -> dict:
    return build_link_prediction_record(
        source_id="icsplc:1",
        target_id="sampoq:2",
        link_score=0.81,
        source={"source_dataset": "ic_splc"},
        target={"source_dataset": "sampoq"},
        model_name="GraphSAGE",
        model_version="graphsage_link_prediction",
        dataset_version="atmo_graph_v1",
        protocol="original_random_split",
        feature_config="c1_current_136",
        checkpoint_path="ml/experiments/graphsage_link_prediction/checkpoint.pt",
    )


def test_link_prediction_contract_has_no_risk_fields():
    record = _record()
    assert record["task"] == "link_prediction"
    assert record["score_type"] == "model_score"
    assert record["probability_calibrated"] is False
    assert record["link_score"] == pytest.approx(0.81)
    assert record["source"]["id"] == "icsplc:1"
    assert record["model"]["name"] == "GraphSAGE"
    assert "risk_score" not in record
    assert "risk_level" not in record
    assert "risk_score" not in json.dumps(record)
    assert "risk_level" not in json.dumps(record)


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf")])
def test_link_score_must_be_finite_and_bounded(score):
    with pytest.raises(ValueError):
        build_link_prediction_record(
            source_id="a",
            target_id="b",
            link_score=score,
            source={"source_dataset": "sampoq"},
            target={"source_dataset": "sampoq"},
            model_name="GraphSAGE",
            model_version="v1",
        )


def test_source_and_target_must_differ():
    with pytest.raises(ValueError):
        build_link_prediction_record(
            source_id="a",
            target_id="a",
            link_score=0.5,
            source={"source_dataset": "sampoq"},
            target={"source_dataset": "sampoq"},
            model_name="GraphSAGE",
            model_version="v1",
        )


def test_risk_prediction_is_explicitly_unsupported():
    with pytest.raises(RiskPredictionUnsupportedError):
        build_risk_prediction_record(company_id="a", risk_score=0.8)


def test_missing_checkpoint_has_clear_error(tmp_path):
    with pytest.raises(CheckpointUnavailableError, match="No score was generated"):
        ensure_checkpoint_available(tmp_path / "missing.pt")
