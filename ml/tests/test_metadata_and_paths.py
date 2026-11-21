import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str) -> dict:
    return json.loads((PROJECT_ROOT / relative).read_text())


def test_completed_experiment_statuses_are_recorded_with_history():
    config = _load("ml/experiments/graphsage_link_prediction/config.json")
    baseline = _load("ml/experiments/graphsage_link_prediction/baseline_metrics.json")
    assert config["status"] == "completed"
    assert config["previous_status"] == "planned"
    comparison = baseline["comparison"]["graphsage_link_prediction"]
    assert comparison["status"] == "completed"
    assert comparison["previous_status"] == "not_trained"


def test_experiment_metadata_paths_are_repository_relative():
    experiment = _load("ml/experiments/graphsage_link_prediction/experiment.json")
    run_summary = _load("ml/experiments/graphsage_link_prediction/run_summary.json")
    manifest = _load("ml/experiments/graphsage_phase2/config/run_manifest.json")
    assert not experiment["checkpoint_path"].startswith("/")
    assert not run_summary["command"].startswith("/")
    assert not manifest["original_experiment_dir"].startswith("/")
    assert not str(experiment["checkpoint_path"]).startswith(str(PROJECT_ROOT))
    assert not str(manifest["original_experiment_dir"]).startswith(str(PROJECT_ROOT))
