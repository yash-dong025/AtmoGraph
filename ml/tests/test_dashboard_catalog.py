import json
from pathlib import Path

from ml.reports.dashboard_data import build_experiment_catalog


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_catalog_separates_original_and_phase2_dimensions(tmp_path):
    original_dir = tmp_path / "original"
    _write_json(
        original_dir / "experiment.json",
        {
            "experiment_id": "original",
            "task": "directed_link_prediction",
            "status": "completed",
            "dataset_version": "v1",
        },
    )
    _write_json(
        original_dir / "config.json",
        {"model": "GraphSAGE", "model_class": "AtmoGraphSAGE", "status": "completed"},
    )
    _write_json(
        original_dir / "metrics.json",
        {
            "test": {
                "roc_auc": 0.67,
                "average_precision": 0.61,
                "precision_at_100": 0.6,
                "threshold": 0.12,
            }
        },
    )
    summary_path = tmp_path / "phase2" / "summary.json"
    _write_json(
        summary_path,
        {
            "experiment": "graphsage_phase2",
            "status": "completed",
            "configs": {"c1_current_136": {"label": "Current 136"}},
            "protocols": {"A": {"label": "Existing split"}},
            "comparison": [
                {
                    "protocol": "A",
                    "model_type": "gnn",
                    "model_name": "c1_current_136",
                    "source": "pooled",
                    "roc_auc": 0.64,
                    "average_precision": 0.58,
                },
                {
                    "protocol": "A",
                    "model_type": "baseline",
                    "model_name": "directed_preferential_attachment",
                    "source": "pooled",
                    "roc_auc": 0.63,
                    "average_precision": 0.69,
                },
            ],
        },
    )

    catalog = build_experiment_catalog(original_dir, summary_path)

    assert catalog["catalog_version"] == 1
    assert catalog["phase2_available"] is True
    assert catalog["experiment_count"] == 3
    assert catalog["experiments"][0]["experiment_type"] == "original"
    assert catalog["experiments"][0]["protocol"] == "original_random_split"
    assert catalog["experiments"][0]["metrics"]["average_precision"] == 0.61
    phase2 = catalog["experiments"][1]
    assert phase2["experiment_type"] == "phase2"
    assert phase2["model_type"] == "GraphSAGE"
    assert phase2["feature_config"] == "c1_current_136"
    assert phase2["dataset_source"] == "pooled"
    assert phase2["protocol_label"] == "Existing split"
    assert phase2["metrics"]["average_precision"] == 0.58
    assert catalog["experiments"][2]["model_type"] == "structural_baseline"
    assert catalog["filters"]["dataset_source"] == ["pooled"]


def test_catalog_degrades_gracefully_without_phase2_summary(tmp_path):
    original_dir = tmp_path / "original"
    _write_json(
        original_dir / "experiment.json",
        {"experiment_id": "original", "status": "completed"},
    )
    catalog = build_experiment_catalog(original_dir, tmp_path / "missing.json")
    assert catalog["phase2_available"] is False
    assert catalog["experiment_count"] == 1
    assert catalog["experiments"][0]["experiment_type"] == "original"
