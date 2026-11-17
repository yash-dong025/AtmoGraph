"""Train and evaluate a leakage-controlled GraphSAGE link-prediction model.

This script is intentionally not run by the dashboard. It writes the experiment
artifacts only after a real training run completes. Graph-derived node features
are excluded, message passing uses train edges only, and validation/test
positives are never used for optimization or threshold selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.gcn_model import build_gnn
from ml.preprocessing.graph_preprocessing import build_pyg_data
from ml.training.evaluate_link_prediction_baseline import (
    binary_metrics,
    curve_payload,
    sample_negative_pairs,
    score_distribution,
    threshold_from_validation,
)

EXPERIMENT_DIR = PROJECT_ROOT / "ml" / "experiments" / "graphsage_link_prediction"
EXCLUDED_FEATURES = {
    "log1p_degree",
    "log1p_in_degree",
    "log1p_out_degree",
    "log1p_raw_degree",
    "log1p_component_size",
    "first_edge_age",
    "last_edge_age",
    "edge_year_span",
    "has_edge_year",
    "is_isolated",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def select_features(data: object, excluded: set[str]) -> tuple[torch.Tensor, list[str]]:
    selected_names = [name for name in data.feature_names if name not in excluded]
    selected_indices = [
        index for index, name in enumerate(data.feature_names) if name not in excluded
    ]
    if len(selected_indices) != len(selected_names):
        raise ValueError("Feature selection is inconsistent")
    if not selected_indices:
        raise ValueError("No node features remain after leakage exclusions")
    return torch.tensor(selected_indices, dtype=torch.long), selected_names


def pairs_from_mask(edge_index: torch.Tensor, mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    selected = edge_index[:, mask].cpu().numpy()
    return selected[0].astype(np.int64), selected[1].astype(np.int64)


def limiting_indices(
    sources: np.ndarray, targets: np.ndarray, limit: int | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(sources), dtype=np.int64)
    if limit is None or limit >= len(sources):
        return sources, targets, indices
    indices = np.sort(rng.choice(len(sources), size=limit, replace=False))
    return sources[indices], targets[indices], indices


def make_samples(
    positive_sources: np.ndarray,
    positive_targets: np.ndarray,
    negative_ratio: float,
    num_nodes: int,
    observed_keys: set[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    negative_count = max(1, int(round(len(positive_sources) * negative_ratio)))
    negative_sources, negative_targets = sample_negative_pairs(
        rng, negative_count, num_nodes, observed_keys
    )
    sources = np.concatenate([positive_sources, negative_sources])
    targets = np.concatenate([positive_targets, negative_targets])
    labels = np.concatenate(
        [
            np.ones(len(positive_sources), dtype=np.int64),
            np.zeros(negative_count, dtype=np.int64),
        ]
    )
    return sources, targets, labels


def decode_pairs(
    model: nn.Module,
    x: torch.Tensor,
    message_edge_index: torch.Tensor,
    sources: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(x, message_edge_index)
        pair_index = torch.tensor(
            np.stack([sources, targets], axis=0), dtype=torch.long, device=device
        )
        logits = model.decode(embeddings, pair_index)
    return logits.detach().cpu().numpy()


def train_epoch(
    model: nn.Module,
    x: torch.Tensor,
    message_edge_index: torch.Tensor,
    positive_sources: np.ndarray,
    positive_targets: np.ndarray,
    negative_ratio: float,
    num_nodes: int,
    observed_keys: set[int],
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad()
    sources, targets, labels = make_samples(
        positive_sources,
        positive_targets,
        negative_ratio,
        num_nodes,
        observed_keys,
        rng,
    )
    embeddings = model.encode(x, message_edge_index)
    pair_index = torch.tensor(
        np.stack([sources, targets], axis=0), dtype=torch.long, device=device
    )
    logits = model.decode(embeddings, pair_index)
    target = torch.tensor(labels, dtype=torch.float32, device=device)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def evaluate_epoch(
    model: nn.Module,
    x: torch.Tensor,
    message_edge_index: torch.Tensor,
    validation_sources: np.ndarray,
    validation_targets: np.ndarray,
    validation_labels: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    logits = decode_pairs(
        model,
        x,
        message_edge_index,
        validation_sources,
        validation_targets,
        device,
    )
    scores = 1.0 / (1.0 + np.exp(-logits))
    target = torch.tensor(validation_labels, dtype=torch.float32)
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    loss = F.binary_cross_entropy_with_logits(logits_tensor, target)
    return {
        "validation_loss": float(loss),
        "validation_roc_auc": float(roc_auc_score(validation_labels, scores)),
        "validation_average_precision": float(
            average_precision_score(validation_labels, scores)
        ),
    }


def build_metrics_payload(
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, object]:
    threshold_info = threshold_from_validation(validation_labels, validation_scores)
    validation = binary_metrics(
        validation_labels, validation_scores, threshold_info["threshold"]
    )
    test = binary_metrics(test_labels, test_scores, threshold_info["threshold"])
    test["curves"] = curve_payload(test_labels, test_scores)
    test["score_distribution"] = score_distribution(
        test_scores[test_labels == 1],
        test_scores[test_labels == 0],
        use_log1p=False,
    )
    return {
        "threshold_selection": {
            "source": "validation",
            "criterion": "maximum F1 among thresholds that predict at least one negative",
            "selected_threshold": threshold_info["threshold"],
            "validation_f1_at_selection": threshold_info["f1"],
            "non_trivial_threshold": threshold_info["non_trivial"],
        },
        "validation": validation,
        "test": test,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--out-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--negatives-per-positive", type=float, default=1.0)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-edges", type=int)
    parser.add_argument("--max-val-edges", type=int)
    parser.add_argument("--max-test-edges", type=int)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    if checkpoint_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing checkpoint: {checkpoint_path}. "
            "Pass --overwrite to replace it."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available")
    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config = load_json(output_dir / "config.json", {})
    dataset_metadata = load_json(output_dir / "dataset_metadata.json", {})
    dataset_version = dataset_metadata.get("version", "unknown")
    if config:
        excluded_features = set(config.get("features", {}).get("excluded_features", EXCLUDED_FEATURES))
    else:
        excluded_features = set(EXCLUDED_FEATURES)

    data = build_pyg_data()
    feature_indices, selected_feature_names = select_features(data, excluded_features)
    x = data.x[:, feature_indices].contiguous().to(device)
    message_edge_index = data.edge_index[:, data.train_edge_mask].contiguous().to(device)
    num_nodes = int(data.num_nodes)
    observed_keys = set(
        (
            data.edge_index[0].to(torch.int64) * num_nodes
            + data.edge_index[1].to(torch.int64)
        ).tolist()
    )

    rng = np.random.default_rng(args.seed)
    train_sources, train_targets = pairs_from_mask(data.edge_index, data.train_edge_mask)
    validation_sources, validation_targets = pairs_from_mask(
        data.edge_index, data.val_edge_mask
    )
    test_sources, test_targets = pairs_from_mask(data.edge_index, data.test_edge_mask)

    train_sources, train_targets, _ = limiting_indices(
        train_sources, train_targets, args.max_train_edges, rng
    )
    validation_sources, validation_targets, _ = limiting_indices(
        validation_sources, validation_targets, args.max_val_edges, rng
    )
    test_sources, test_targets, test_indices = limiting_indices(
        test_sources, test_targets, args.max_test_edges, rng
    )

    validation_samples = make_samples(
        validation_sources,
        validation_targets,
        1.0,
        num_nodes,
        observed_keys,
        rng,
    )
    test_samples = make_samples(
        test_sources,
        test_targets,
        1.0,
        num_nodes,
        observed_keys,
        rng,
    )
    validation_labels = validation_samples[2]
    test_labels = test_samples[2]

    summary = {
        "num_nodes": num_nodes,
        "num_edges": int(data.num_edges),
        "train_edges_used": int(len(train_sources)),
        "validation_positives": int(len(validation_sources)),
        "test_positives": int(len(test_sources)),
        "validation_negatives": int((validation_labels == 0).sum()),
        "test_negatives": int((test_labels == 0).sum()),
        "selected_feature_dim": int(len(selected_feature_names)),
        "excluded_feature_count": int(len(excluded_features)),
        "message_passing_edges": int(message_edge_index.shape[1]),
        "device": str(device),
        "subset_run": any(
            value is not None
            for value in (args.max_train_edges, args.max_val_edges, args.max_test_edges)
        ),
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    model = build_gnn(
        "sage",
        in_channels=int(len(selected_feature_names)),
        hidden_channels=args.hidden,
        out_channels=args.out_channels,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_ap = -float("inf")
    best_epoch = 0
    patience_counter = 0
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_loss = train_epoch(
            model,
            x,
            message_edge_index,
            train_sources,
            train_targets,
            args.negatives_per_positive,
            num_nodes,
            observed_keys,
            rng,
            optimizer,
            device,
        )
        validation = evaluate_epoch(
            model,
            x,
            message_edge_index,
            validation_samples[0],
            validation_samples[1],
            validation_labels,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **validation,
            "seconds": time.time() - epoch_started,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={validation['validation_loss']:.4f} "
            f"val_auc={validation['validation_roc_auc']:.4f} "
            f"val_ap={validation['validation_average_precision']:.4f}"
        )

        if validation["validation_average_precision"] > best_validation_ap:
            best_validation_ap = validation["validation_average_precision"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state")
    model.load_state_dict(best_state)
    model.eval()

    validation_logits = decode_pairs(
        model,
        x,
        message_edge_index,
        validation_samples[0],
        validation_samples[1],
        device,
    )
    test_logits = decode_pairs(
        model,
        x,
        message_edge_index,
        test_samples[0],
        test_samples[1],
        device,
    )
    validation_scores = 1.0 / (1.0 + np.exp(-validation_logits))
    test_scores = 1.0 / (1.0 + np.exp(-test_logits))
    metrics_payload = build_metrics_payload(
        validation_labels,
        validation_scores,
        test_labels,
        test_scores,
    )

    baseline = load_json(output_dir / "baseline_metrics.json", {})
    baseline_comparison = {}
    if baseline:
        selected_baseline = baseline.get("baseline", {})
        baseline_comparison = {
            "baseline_name": baseline.get("selected_baseline"),
            "baseline_test": {
                key: selected_baseline.get("test", {}).get(key)
                for key in ("roc_auc", "average_precision", "precision", "recall", "f1", "threshold")
            },
            "graphsage_test": {
                key: metrics_payload["test"].get(key)
                for key in ("roc_auc", "average_precision", "precision", "recall", "f1", "threshold")
            },
        }

    model_config = {
        "model": "GraphSAGE",
        "model_class": "AtmoGraphSAGE",
        "in_channels": int(len(selected_feature_names)),
        "hidden_channels": args.hidden,
        "out_channels": args.out_channels,
        "dropout": args.dropout,
        "layers": 2,
        "decoder": "dot_product",
    }
    metrics = {
        "experiment": "graphsage_link_prediction",
        "status": "completed",
        "generated_at": utc_now(),
        "seed": args.seed,
        "dataset_version": dataset_version,
        "model": model_config,
        "features": {
            "selected_feature_names": selected_feature_names,
            "selected_feature_dim": len(selected_feature_names),
            "excluded_features": sorted(excluded_features),
        },
        "training": {
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_average_precision": best_validation_ap,
            "optimizer": "Adam",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "loss": "BCEWithLogitsLoss",
            "negatives_per_positive": args.negatives_per_positive,
            "device": str(device),
        },
        "protocol": {
            "message_passing_edges": "train split only",
            "positive_edges": "train positives for optimization; validation/test positives for evaluation only",
            "negative_sampling": "uniform directed non-edges; every observed directed edge excluded",
            "threshold_selection": "validation-only maximum F1 among non-trivial thresholds",
            "test_protocol": "single evaluation after early stopping on validation",
            "not_a_risk_model": True,
        },
        **metrics_payload,
        "baseline_comparison": baseline_comparison,
    }

    history_payload = {
        "experiment": "graphsage_link_prediction",
        "generated_at": utc_now(),
        "history": history,
    }
    run_summary = {
        "experiment": "graphsage_link_prediction",
        "generated_at": utc_now(),
        "duration_seconds": time.time() - started_at,
        "command": " ".join(sys.argv),
        "summary": summary,
        "model": model_config,
        "best_epoch": best_epoch,
        "best_validation_average_precision": best_validation_ap,
        "dataset_version": dataset_version,
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "sage",
            "model_class": "AtmoGraphSAGE",
            "model_config": model_config,
            "feature_names": selected_feature_names,
            "excluded_features": sorted(excluded_features),
            "dataset_version": dataset_version,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "threshold": metrics_payload["test"]["threshold"],
        },
        checkpoint_path,
    )
    write_json(output_dir / "history.json", history_payload)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run_summary.json", run_summary)

    index_to_node = data.node_ids
    positive_count = len(test_sources)
    positive_edge_ids = np.asarray(data.edge_ids, dtype=object)[
        data.test_edge_mask.cpu().numpy()
    ][test_indices]
    negative_count = len(test_samples[0]) - positive_count
    predictions = {
        "edge_id": list(positive_edge_ids)
        + [f"sampled_negative_{index:06d}" for index in range(negative_count)],
        "source_index": test_samples[0],
        "target_index": test_samples[1],
        "source_company_id": [index_to_node[int(index)] for index in test_samples[0]],
        "target_company_id": [index_to_node[int(index)] for index in test_samples[1]],
        "label": test_labels,
        "score": test_scores,
        "logit": test_logits,
    }
    pd.DataFrame(predictions).to_csv(output_dir / "test_predictions.csv", index=False)

    experiment = load_json(output_dir / "experiment.json", {})
    experiment.update(
        {
            "experiment_id": "graphsage_link_prediction",
            "task": "directed_link_prediction",
            "status": "completed",
            "updated_at": utc_now(),
            "dataset_version": dataset_version,
            "random_seed": args.seed,
            "checkpoint_exists": True,
            "checkpoint_path": str(checkpoint_path),
            "gnn_results_available": True,
            "missing_artifacts": [],
            "available_artifacts": [
                "checkpoint.pt",
                "history.json",
                "metrics.json",
                "test_predictions.csv",
                "run_summary.json",
                "config.json",
                "experiment.json",
                "baseline_metrics.json",
                "baseline_test_predictions.csv",
                "dataset_metadata.json",
            ],
            "best_epoch": best_epoch,
        }
    )
    write_json(output_dir / "experiment.json", experiment)

    print(
        json.dumps(
            {
                "status": "completed",
                "checkpoint": str(checkpoint_path),
                "best_epoch": best_epoch,
                "test_roc_auc": metrics_payload["test"]["roc_auc"],
                "test_average_precision": metrics_payload["test"]["average_precision"],
                "test_f1": metrics_payload["test"]["f1"],
                "threshold": metrics_payload["test"]["threshold"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
