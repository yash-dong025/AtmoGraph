"""Scikit-learn baseline feasibility checks for shipment delay tasks.

This is a baseline, not a production model. The delay labels have important
caveats:
- SynDelay is synthetic and its 3-class meanings are undocumented.
- Smart Logistics is a 1,000-row local file with unknown provenance.

The baseline deliberately uses the existing leakage-safe logistics features and
the existing reproducible delay splits. It does not invent risk labels or claim
real-world predictive capability.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASETS_DIR = PROJECT_ROOT / "datasets"
FEATURES_PATH = DATASETS_DIR / "processed" / "logistics" / "logistics_features.csv"
LABELS_PATH = DATASETS_DIR / "training" / "delay_labels.csv"
SPLITS_PATH = DATASETS_DIR / "training" / "dataset_splits.json"
REPORTS_DIR = DATASETS_DIR / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
REPORT_MD = REPORTS_DIR / "ml_baseline_report.md"
ARTIFACT_JSON = ARTIFACTS_DIR / "baseline_metrics.json"

TASKS = ["delay_classification_binary", "delay_classification_3class"]
DROP_COLUMNS = ["shipment_id", "source_dataset", "is_synthetic", "data_origin", "target_value"]


def load_task(task: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    task_labels = labels.loc[labels["task"] == task, ["shipment_id", "target_value"]].copy()
    if task_labels.empty:
        raise ValueError(f"No labels found for task {task}")

    merged = features.merge(task_labels, on="shipment_id", how="inner", validate="one_to_one")
    if len(merged) != len(task_labels):
        raise ValueError(
            f"Feature/label mismatch for {task}: {len(merged)} merged vs {len(task_labels)} labels"
        )

    y = merged["target_value"].astype(int)
    drop = [col for col in DROP_COLUMNS if col in merged.columns]
    X = merged.drop(columns=drop)
    keep = [
        col
        for col in X.columns
        if X[col].notna().any() and X[col].nunique(dropna=True) > 1
    ]
    X = X[keep]
    if X.empty:
        raise ValueError(f"No usable features for task {task}")
    return X, y, merged["shipment_id"]


def split_masks(shipment_ids: pd.Series, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(SPLITS_PATH, "r", encoding="utf-8") as fh:
        splits = json.load(fh)
    task_splits = splits.get("delay_splits", {}).get(task)
    if not task_splits:
        raise ValueError(f"Missing delay split for task {task}")
    masks = []
    for split_name in ("train", "val", "test"):
        split_ids = set(task_splits.get(split_name, []))
        masks.append(shipment_ids.isin(split_ids).to_numpy(dtype=bool))
    if not all(mask.any() for mask in masks):
        raise ValueError(f"One or more {task} splits are empty")
    if (masks[0] & masks[1]).any() or (masks[0] & masks[2]).any() or (masks[1] & masks[2]).any():
        raise ValueError(f"{task} splits overlap")
    return tuple(masks)  # type: ignore[return-value]


def evaluate_task(task: str) -> dict[str, Any]:
    X, y, shipment_ids = load_task(task)
    train_mask, val_mask, test_mask = split_masks(shipment_ids, task)

    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42,
        early_stopping=False,
    )
    model.fit(X.loc[train_mask], y.loc[train_mask])

    predictions = model.predict(X.loc[test_mask])
    probabilities = model.predict_proba(X.loc[test_mask])
    y_test = y.loc[test_mask]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    labels = sorted(y.unique().tolist())
    metrics: dict[str, Any] = {
        "task": task,
        "rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "features": list(X.columns),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "split_sizes": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "model": "HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, random_state=42)",
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_macro_precision": float(precision),
        "test_macro_recall": float(recall),
        "test_macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "confusion_matrix_labels": [int(v) for v in labels],
        "classification_report": report,
        "caveat": (
            "SynDelay is synthetic with undocumented class semantics; Smart Logistics provenance is unknown. "
            "These metrics are baseline feasibility checks, not real-world performance claims."
        ),
    }
    try:
        if len(labels) == 2:
            metrics["test_roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))
        else:
            metrics["test_roc_auc_ovr_macro"] = float(
                roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro")
            )
    except Exception as exc:
        metrics["test_roc_auc_error"] = str(exc)
    return metrics


def render_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# AtmoGraph ML Baseline Report",
        "",
        "Generated by `ml/models/baseline_model.py` using the existing leakage-safe logistics features and reproducible delay splits.",
        "",
        "This is a feasibility baseline. It is **not** evidence of real-world delay-prediction capability.",
        "",
        "## Results",
        "",
        "| Task | Rows | Features | Test accuracy | Macro precision | Macro recall | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| `{result['task']}` | {result['rows']:,} | {result['feature_count']} | "
            f"{result['test_accuracy']:.4f} | {result['test_macro_precision']:.4f} | "
            f"{result['test_macro_recall']:.4f} | {result['test_macro_f1']:.4f} |"
        )
    lines += ["", "## Task details", ""]
    for result in results:
        lines += [
            f"### `{result['task']}`",
            "",
            f"- Class distribution: `{result['class_distribution']}`",
            f"- Split sizes: `{result['split_sizes']}`",
            f"- Feature count: `{result['feature_count']}`",
            f"- Features used: `{result['features']}`",
            f"- Confusion matrix labels: `{result['confusion_matrix_labels']}`",
            f"- Confusion matrix: `{result['confusion_matrix']}`",
            f"- Test ROC-AUC: `{result.get('test_roc_auc', result.get('test_roc_auc_ovr_macro', 'not available'))}`",
            f"- Caveat: {result['caveat']}",
            "",
        ]
    lines += [
        "## Feature gaps",
        "",
        "- `logistics_features.csv` exposes 10 usable features for the 1,000-row Smart Logistics task.",
        "- For SynDelay, `logistics_features.csv` exposes only `latitude` and `longitude`; the other raw SynDelay fields were not carried into the processed feature table.",
        "- Therefore the SynDelay 3-class result is a pipeline-gap check, not a credible delay model. Week 2 must add a SynDelay-specific feature builder or extend the existing dataset pipeline before drawing modelling conclusions.",
        "- The 3-class baseline predicts the majority class for nearly every row, which is expected with only two geographic features and severe class imbalance.",
        "",
    ]
    lines += [
        "## Interpretation",
        "",
        "- The binary Smart Logistics task has only 1,000 rows; metrics are unstable and the data provenance is unknown.",
        "- The SynDelay 3-class task is synthetic and class meanings are undocumented; do not describe classes as low/medium/high delay.",
        "- The baseline uses numeric/categorical code features directly; a Week 2 model should compare categorical embeddings and GNN models against this baseline.",
        "- No company-level risk model was trained because no verified risk labels exist.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run shipment-delay scikit-learn baselines")
    parser.add_argument("--task", choices=TASKS, action="append", default=None)
    args = parser.parse_args()
    tasks = args.task or TASKS

    results = [evaluate_task(task) for task in tasks]
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value))
        fh.write("\n")
    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(results))
        fh.write("\n")
    print(
        json.dumps(
            {
                "tasks": [result["task"] for result in results],
                "report": str(REPORT_MD.relative_to(PROJECT_ROOT)),
                "artifact": str(ARTIFACT_JSON.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
