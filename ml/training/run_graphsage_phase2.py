from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.gcn_model import build_gnn
from ml.models.phase2_graphsage import AtmoGraphSAGESourceCat
from ml.training.phase2_utils import (
    CONFIG_SPECS,
    CURRENT_FEATURE_NAMES,
    EMBEDDING_NAMES,
    FLAG_NAMES,
    ORIGINAL_EXPERIMENT_DIR,
    SOURCE_ORDER,
    STRUCTURAL_FEATURE_NAMES,
    PairSet,
    Phase2Data,
    all_observed_keys,
    apply_platt,
    apply_temperature,
    build_evaluation_pairs,
    build_protocols,
    build_structural_adjacency,
    build_structural_features,
    calibration_report,
    compare_rows_from_metrics,
    evaluate_predictions,
    fit_platt,
    fit_temperature,
    json_default,
    load_json,
    load_phase2_data,
    precision_at_k,
    predictions_frame,
    score_structural_baselines,
    sigmoid,
    utc_now,
    write_json,
    write_reliability_svg,
)
from ml.training.evaluate_link_prediction_baseline import sample_negative_pairs

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ml" / "experiments" / "graphsage_phase2"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "ml" / "reports" / "graphsage_phase2_experiment_report.md"
PHASE2_EXPERIMENT_NAME = "graphsage_phase2"


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_value(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return ""


def split_reproduction_view(reproduction: dict[str, Any]) -> dict[str, Any]:
    view = {key: value for key, value in reproduction.items() if key != "split"}
    view["split_assignments_file"] = "config/protocol_splits.csv"
    return view


def protocol_view(protocol: dict[str, Any]) -> dict[str, Any]:
    view = {
        key: copy.deepcopy(value)
        for key, value in protocol.items()
        if key not in {"split", "train_indices", "val_indices", "test_indices"}
    }
    reverse_control = view.get("reverse_control")
    if isinstance(reverse_control, dict) and "removed_train_edge_ids" in reverse_control:
        removed = reverse_control.pop("removed_train_edge_ids")
        reverse_control["removed_train_edge_ids_count"] = int(len(removed))
        reverse_control["removed_train_edge_ids_file"] = (
            "config/reverse_control_removals.json"
        )
        view["reverse_control"] = reverse_control
    return view


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def portable_command(command: str) -> str:
    return command.replace(f"{PROJECT_ROOT}/", "")


def run_manifest(args: argparse.Namespace, data: Phase2Data, protocols: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": PHASE2_EXPERIMENT_NAME,
        "generated_at": utc_now(),
        "command": portable_command(" ".join(sys.argv)),
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": git_value(["rev-parse", "HEAD"]),
            "status_short": git_value(["status", "--short"]).splitlines(),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": args.device,
        },
        "dataset": {
            "nodes": int(data.num_nodes),
            "edges": int(len(data.edge_df)),
            "node_sources": {
                source: int((data.node_sources == source).sum()) for source in SOURCE_ORDER
            },
            "edge_sources": {
                source: int((data.edge_df["source_dataset"] == source).sum())
                for source in SOURCE_ORDER
            },
            "feature_dim": int(len(data.feature_names)),
            "current_feature_dim": int(len(CURRENT_FEATURE_NAMES)),
            "categorical_cardinalities": data.cat_cardinalities,
            "embedding_available": int(data.emb_mask.sum()),
            "split_reproduction": split_reproduction_view(data.split_reproduction),
        },
        "protocols": {
            name: protocol_view(protocol) for name, protocol in protocols.items()
        },
        "configs": CONFIG_SPECS,
        "original_experiment_dir": portable_path(ORIGINAL_EXPERIMENT_DIR),
        "original_artifacts_read_only": True,
        "notes": [
            "Original graphsage_link_prediction artifacts are never overwritten.",
            "All new outputs are written under ml/experiments/graphsage_phase2/.",
            "Full protocol split assignments are stored in config/protocol_splits.csv.",
            "Paths are written repository-relative for portability.",
        ],
    }


def make_training_samples(
    data: Phase2Data,
    positive_indices: np.ndarray,
    negative_ratio: float,
    observed_keys: set[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positive_sources = data.edge_index[0, positive_indices].astype(np.int64)
    positive_targets = data.edge_index[1, positive_indices].astype(np.int64)
    negative_count = max(1, int(round(len(positive_sources) * negative_ratio)))
    negative_sources, negative_targets = sample_negative_pairs(
        rng,
        negative_count,
        data.num_nodes,
        observed_keys,
        positive_sources=positive_sources,
        positive_targets=positive_targets,
        node_sources=data.node_sources,
    )
    sources = np.concatenate([positive_sources, negative_sources]).astype(np.int64)
    targets = np.concatenate([positive_targets, negative_targets]).astype(np.int64)
    labels = np.concatenate(
        [
            np.ones(len(positive_sources), dtype=np.int64),
            np.zeros(len(negative_sources), dtype=np.int64),
        ]
    )
    source_labels = data.node_sources[sources]
    return sources, targets, labels, source_labels


def encode_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    x_cat: torch.Tensor,
    source_id: torch.Tensor,
    use_categorical: bool,
) -> torch.Tensor:
    if use_categorical:
        return model.encode(x, edge_index, x_cat, source_id)
    return model.encode(x, edge_index)


def decode_pairs(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    x_cat: torch.Tensor,
    source_id: torch.Tensor,
    use_categorical: bool,
    sources: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        embeddings = encode_model(
            model, x, edge_index, x_cat, source_id, use_categorical
        )
        pair_index = torch.tensor(
            np.stack([sources, targets], axis=0), dtype=torch.long, device=device
        )
        logits = model.decode(embeddings, pair_index)
    return logits.detach().cpu().numpy()


def train_epoch(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    x_cat: torch.Tensor,
    source_id: torch.Tensor,
    use_categorical: bool,
    positive_indices: np.ndarray,
    negative_ratio: float,
    observed_keys: set[int],
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer,
    data: Phase2Data,
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad()
    sources, targets, labels, _ = make_training_samples(
        data, positive_indices, negative_ratio, observed_keys, rng
    )
    embeddings = encode_model(
        model, x, edge_index, x_cat, source_id, use_categorical
    )
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
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    x_cat: torch.Tensor,
    source_id: torch.Tensor,
    use_categorical: bool,
    pairs: PairSet,
    device: torch.device,
) -> dict[str, float]:
    logits = decode_pairs(
        model,
        x,
        edge_index,
        x_cat,
        source_id,
        use_categorical,
        pairs.sources,
        pairs.targets,
        device,
    )
    scores = sigmoid(logits)
    target = torch.tensor(pairs.labels, dtype=torch.float32)
    loss = F.binary_cross_entropy_with_logits(
        torch.tensor(logits, dtype=torch.float32), target
    )
    return {
        "validation_loss": float(loss),
        "validation_roc_auc": float(roc_auc_score(pairs.labels, scores)),
        "validation_average_precision": float(
            average_precision_score(pairs.labels, scores)
        ),
    }


def model_metadata(
    config_name: str,
    spec: dict[str, Any],
    in_channels: int,
    data: Phase2Data,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = {
        "config": config_name,
        "label": spec["label"],
        "description": spec["description"],
        "model": "AtmoGraphSAGESourceCat" if spec["categorical"] else "AtmoGraphSAGE",
        "in_channels": int(in_channels),
        "hidden_channels": int(args.hidden),
        "out_channels": int(args.out_channels),
        "dropout": float(args.dropout),
        "layers": 2,
        "decoder": "dot_product",
        "categorical": bool(spec["categorical"]),
        "structural": bool(spec["structural"]),
    }
    if spec["categorical"]:
        metadata["categorical_dim_per_field"] = int(args.categorical_dim)
        metadata["categorical_cardinalities"] = data.cat_cardinalities
        metadata["categorical_source_specific"] = True
        metadata["categorical_missing_index"] = 0
    return metadata


def build_run_features(
    data: Phase2Data,
    config_name: str,
    spec: dict[str, Any],
    structural_features: dict[str, Any] | None,
) -> tuple[np.ndarray, list[str], dict[str, Any] | None]:
    if spec["base"] == "current":
        base_x = data.current_x
        feature_names = list(CURRENT_FEATURE_NAMES)
    elif spec["base"] == "flags":
        base_x = data.flags_x
        feature_names = list(FLAG_NAMES)
    else:
        raise ValueError(f"Unknown feature base: {spec['base']}")
    structural_metadata = None
    if spec["structural"]:
        if structural_features is None:
            raise ValueError("Structural features are required but were not built")
        scaled, structural_metadata = structural_features
        base_x = np.concatenate([base_x, scaled], axis=1).astype(np.float32)
        feature_names = feature_names + list(STRUCTURAL_FEATURE_NAMES)
    return base_x, feature_names, structural_metadata


def build_model_for_run(
    config_name: str,
    spec: dict[str, Any],
    in_channels: int,
    data: Phase2Data,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if spec["categorical"]:
        model = AtmoGraphSAGESourceCat(
            in_channels=in_channels,
            cardinalities=data.cat_cardinalities,
            source_count=len(SOURCE_ORDER),
            categorical_dim=args.categorical_dim,
            hidden_channels=args.hidden,
            out_channels=args.out_channels,
            dropout=args.dropout,
        )
    else:
        model = build_gnn(
            "sage",
            in_channels=in_channels,
            hidden_channels=args.hidden,
            out_channels=args.out_channels,
            dropout=args.dropout,
        )
    metadata = model_metadata(config_name, spec, in_channels, data, args)
    return model, metadata


def train_and_evaluate_config(
    data: Phase2Data,
    protocol: dict[str, Any],
    pair_sets: dict[str, PairSet],
    config_name: str,
    spec: dict[str, Any],
    structural_features: dict[str, Any] | None,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    set_seed(args.seed)
    protocol_name = protocol["name"]
    metrics_path = output_dir / "metrics" / protocol_name / f"{config_name}.json"
    if metrics_path.exists() and not args.overwrite:
        logging.info("Skipping existing metrics for %s/%s", protocol_name, config_name)
        return load_json(metrics_path, {})

    x_np, feature_names, structural_metadata = build_run_features(
        data, config_name, spec, structural_features
    )
    use_categorical = bool(spec["categorical"])
    device = torch.device(args.device)
    x = torch.from_numpy(x_np).float().to(device)
    x_cat = torch.from_numpy(data.x_cat).long().to(device)
    source_id = torch.from_numpy(data.source_ids).long().to(device)
    train_edge_index = torch.from_numpy(
        data.edge_index[:, protocol["train_indices"]]
    ).long().to(device)
    positive_indices = protocol["train_indices"]
    observed_keys = all_observed_keys(data, include_reverse=True)

    model, model_config = build_model_for_run(
        config_name, spec, int(x.shape[1]), data, args
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
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
            train_edge_index,
            x_cat,
            source_id,
            use_categorical,
            positive_indices,
            args.negatives_per_positive,
            observed_keys,
            rng,
            optimizer,
            data,
            device,
        )
        validation = evaluate_epoch(
            model,
            x,
            train_edge_index,
            x_cat,
            source_id,
            use_categorical,
            pair_sets["val"],
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **validation,
            "seconds": time.time() - epoch_started,
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            logging.info(
                "protocol=%s config=%s epoch=%03d train_loss=%.4f val_loss=%.4f val_auc=%.4f val_ap=%.4f",
                protocol_name,
                config_name,
                epoch,
                train_loss,
                validation["validation_loss"],
                validation["validation_roc_auc"],
                validation["validation_average_precision"],
            )
        if validation["validation_average_precision"] > best_validation_ap:
            best_validation_ap = float(validation["validation_average_precision"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info(
                    "protocol=%s config=%s early stopping at epoch %d",
                    protocol_name,
                    config_name,
                    epoch,
                )
                break

    if best_state is None:
        raise RuntimeError("Training produced no best state")
    model.load_state_dict(best_state)
    model.eval()
    validation_logits = decode_pairs(
        model,
        x,
        train_edge_index,
        x_cat,
        source_id,
        use_categorical,
        pair_sets["val"].sources,
        pair_sets["val"].targets,
        device,
    )
    test_logits = decode_pairs(
        model,
        x,
        train_edge_index,
        x_cat,
        source_id,
        use_categorical,
        pair_sets["test"].sources,
        pair_sets["test"].targets,
        device,
    )
    validation_scores = sigmoid(validation_logits)
    test_scores = sigmoid(test_logits)
    metrics = evaluate_predictions(
        pair_sets["val"].labels,
        validation_scores,
        pair_sets["val"].source_labels,
        validation_logits,
        pair_sets["test"].labels,
        test_scores,
        pair_sets["test"].source_labels,
        test_logits,
    )
    metrics.update(
        {
            "experiment": PHASE2_EXPERIMENT_NAME,
            "generated_at": utc_now(),
            "status": "completed",
            "protocol": protocol_name,
            "protocol_label": protocol["label"],
            "model_type": "gnn",
            "model_name": config_name,
            "config": config_name,
            "config_label": spec["label"],
            "config_description": spec["description"],
            "model_config": model_config,
            "feature_names": feature_names,
            "feature_dim": int(x.shape[1]),
            "structural": structural_metadata,
            "categorical": (
                {
                    "cardinalities": data.cat_cardinalities,
                    "source_specific": True,
                    "embedding_dim_per_field": int(args.categorical_dim),
                    "missing_index": 0,
                }
                if use_categorical
                else None
            ),
            "training": {
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
                "best_validation_average_precision": best_validation_ap,
                "optimizer": "Adam",
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "loss": "BCEWithLogitsLoss",
                "negatives_per_positive": args.negatives_per_positive,
                "seed": args.seed,
                "device": str(device),
                "duration_seconds": time.time() - started_at,
            },
            "counts": {
                "train_edges": int(len(protocol["train_indices"])),
                "val_positives": pair_sets["val"].positive_count,
                "val_negatives": pair_sets["val"].negative_count,
                "test_positives": pair_sets["test"].positive_count,
                "test_negatives": pair_sets["test"].negative_count,
                "val_source_counts": pair_sets["val"].source_counts,
                "test_source_counts": pair_sets["test"].source_counts,
            },
            "protocol_stats": {
                "counts": protocol["counts"],
                "source_counts": protocol["source_counts"],
                "reverse_leakage": protocol["reverse_leakage"],
                "reverse_control": protocol["reverse_control"],
                "temporal": protocol["temporal"],
            },
        }
    )

    checkpoint_path = (
        output_dir / "checkpoints" / protocol_name / f"{config_name}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "sage_cat" if use_categorical else "sage",
            "model_config": model_config,
            "feature_names": feature_names,
            "categorical_cardinalities": data.cat_cardinalities if use_categorical else None,
            "categorical_dim": int(args.categorical_dim) if use_categorical else None,
            "source_order": list(SOURCE_ORDER),
            "structural_feature_names": (
                list(STRUCTURAL_FEATURE_NAMES) if spec["structural"] else None
            ),
            "structural_scaler_mean": (
                structural_metadata.get("scaler_mean")
                if structural_metadata is not None
                else None
            ),
            "structural_scaler_scale": (
                structural_metadata.get("scaler_scale")
                if structural_metadata is not None
                else None
            ),
            "protocol": protocol_name,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "threshold": metrics["threshold_selection"]["selected_threshold"],
        },
        checkpoint_path,
    )

    write_json(metrics_path, metrics)
    write_json(
        output_dir / "metrics" / protocol_name / f"{config_name}_history.json",
        {"experiment": PHASE2_EXPERIMENT_NAME, "protocol": protocol_name, "config": config_name, "history": history},
    )
    predictions_frame(
        data, pair_sets["val"], validation_scores, validation_logits
    ).to_csv(
        output_dir / "predictions" / protocol_name / f"{config_name}_val.csv",
        index=False,
    )
    predictions_frame(
        data, pair_sets["test"], test_scores, test_logits
    ).to_csv(
        output_dir / "predictions" / protocol_name / f"{config_name}_test.csv",
        index=False,
    )
    write_reliability_svg(
        output_dir / "plots" / f"{protocol_name}_{config_name}_reliability.svg",
        metrics["calibration"],
        f"{protocol_name} {config_name} calibrated reliability",
    )
    logging.info(
        "Completed protocol=%s config=%s test_auc=%.4f test_ap=%.4f test_f1=%.4f",
        protocol_name,
        config_name,
        metrics["test"]["roc_auc"],
        metrics["test"]["average_precision"],
        metrics["test"]["f1"],
    )
    return metrics


BASELINE_LABELS = {
    "directed_preferential_attachment": "Directed Preferential Attachment",
    "undirected_preferential_attachment": "Undirected Preferential Attachment",
    "directed_common_neighbors": "Directed Common Neighbors",
    "undirected_common_neighbors": "Undirected Common Neighbors",
}
BASELINE_FORMULAS = {
    "directed_preferential_attachment": "out_degree(source) * in_degree(target) on the protocol train graph",
    "undirected_preferential_attachment": "undirected_degree(source) * undirected_degree(target) on the protocol train graph",
    "directed_common_neighbors": "|out_neighbors(source) intersection in_neighbors(target)| on the protocol train graph",
    "undirected_common_neighbors": "|undirected_neighbors(source) intersection undirected_neighbors(target)| on the protocol train graph",
}
BASELINE_DIRECTION = {
    "directed_preferential_attachment": "Directed: source out-degree and target in-degree",
    "undirected_preferential_attachment": "Undirected: unique undirected neighbours",
    "directed_common_neighbors": "Directed: directed paths source -> w -> target",
    "undirected_common_neighbors": "Undirected: shared unique neighbours",
}


def run_structural_baselines(
    data: Phase2Data,
    protocol: dict[str, Any],
    pair_sets: dict[str, PairSet],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    protocol_name = protocol["name"]
    adjacency = build_structural_adjacency(data, protocol)
    validation_scores = score_structural_baselines(
        adjacency, pair_sets["val"].sources, pair_sets["val"].targets
    )
    test_scores = score_structural_baselines(
        adjacency, pair_sets["test"].sources, pair_sets["test"].targets
    )
    results: dict[str, dict[str, Any]] = {}
    for baseline_name in BASELINE_LABELS:
        metrics_path = output_dir / "metrics" / protocol_name / f"{baseline_name}.json"
        if metrics_path.exists() and not args.overwrite:
            logging.info(
                "Skipping existing metrics for %s/%s", protocol_name, baseline_name
            )
            results[baseline_name] = load_json(metrics_path, {})
            continue
        metrics = evaluate_predictions(
            pair_sets["val"].labels,
            validation_scores[baseline_name],
            pair_sets["val"].source_labels,
            None,
            pair_sets["test"].labels,
            test_scores[baseline_name],
            pair_sets["test"].source_labels,
            None,
        )
        metrics.update(
            {
                "experiment": PHASE2_EXPERIMENT_NAME,
                "generated_at": utc_now(),
                "status": "completed",
                "protocol": protocol_name,
                "protocol_label": protocol["label"],
                "model_type": "baseline",
                "model_name": baseline_name,
                "baseline_label": BASELINE_LABELS[baseline_name],
                "formula": BASELINE_FORMULAS[baseline_name],
                "direction_treatment": BASELINE_DIRECTION[baseline_name],
                "counts": {
                    "train_edges": int(len(protocol["train_indices"])),
                    "val_positives": pair_sets["val"].positive_count,
                    "val_negatives": pair_sets["val"].negative_count,
                    "test_positives": pair_sets["test"].positive_count,
                    "test_negatives": pair_sets["test"].negative_count,
                    "val_source_counts": pair_sets["val"].source_counts,
                    "test_source_counts": pair_sets["test"].source_counts,
                },
                "protocol_stats": {
                    "counts": protocol["counts"],
                    "source_counts": protocol["source_counts"],
                    "reverse_leakage": protocol["reverse_leakage"],
                    "reverse_control": protocol["reverse_control"],
                    "temporal": protocol["temporal"],
                },
            }
        )
        write_json(metrics_path, metrics)
        predictions_frame(data, pair_sets["val"], validation_scores[baseline_name]).to_csv(
            output_dir / "predictions" / protocol_name / f"{baseline_name}_val.csv",
            index=False,
        )
        predictions_frame(data, pair_sets["test"], test_scores[baseline_name]).to_csv(
            output_dir / "predictions" / protocol_name / f"{baseline_name}_test.csv",
            index=False,
        )
        write_reliability_svg(
            output_dir / "plots" / f"{protocol_name}_{baseline_name}_reliability.svg",
            metrics["calibration"],
            f"{protocol_name} {BASELINE_LABELS[baseline_name]} calibrated reliability",
        )
        results[baseline_name] = metrics
        logging.info(
            "Baseline %s protocol=%s test_auc=%.4f test_ap=%.4f test_f1=%.4f",
            baseline_name,
            protocol_name,
            metrics["test"]["roc_auc"],
            metrics["test"]["average_precision"],
            metrics["test"]["f1"],
        )
    return results


def run_original_checkpoint_reference(
    data: Phase2Data,
    protocol: dict[str, Any],
    pair_sets: dict[str, PairSet],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if protocol["name"] != "A":
        return None
    checkpoint_path = ORIGINAL_EXPERIMENT_DIR / "checkpoint.pt"
    if not checkpoint_path.exists():
        logging.warning("Original checkpoint is missing: %s", checkpoint_path)
        return None
    metrics_path = (
        output_dir / "metrics" / protocol["name"] / "original_graphsage_reference.json"
    )
    if metrics_path.exists() and not args.overwrite:
        logging.info("Skipping existing original checkpoint reference")
        return load_json(metrics_path, {})
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_features = list(checkpoint["feature_names"])
    name_to_index = {name: index for index, name in enumerate(data.feature_names)}
    missing = [name for name in checkpoint_features if name not in name_to_index]
    if missing:
        raise ValueError(f"Checkpoint features missing from current data: {missing}")
    feature_indices = [name_to_index[name] for name in checkpoint_features]
    x_np = data.x[:, feature_indices].astype(np.float32)
    device = torch.device(args.device)
    x = torch.from_numpy(x_np).float().to(device)
    train_edge_index = torch.from_numpy(
        data.edge_index[:, protocol["train_indices"]]
    ).long().to(device)
    model_config = checkpoint["model_config"]
    model = build_gnn(
        "sage",
        in_channels=int(model_config["in_channels"]),
        hidden_channels=int(model_config["hidden_channels"]),
        out_channels=int(model_config["out_channels"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    validation_logits = decode_pairs(
        model,
        x,
        train_edge_index,
        torch.empty((0, 0), dtype=torch.long, device=device),
        torch.empty((0,), dtype=torch.long, device=device),
        False,
        pair_sets["val"].sources,
        pair_sets["val"].targets,
        device,
    )
    test_logits = decode_pairs(
        model,
        x,
        train_edge_index,
        torch.empty((0, 0), dtype=torch.long, device=device),
        torch.empty((0,), dtype=torch.long, device=device),
        False,
        pair_sets["test"].sources,
        pair_sets["test"].targets,
        device,
    )
    validation_scores = sigmoid(validation_logits)
    test_scores = sigmoid(test_logits)
    metrics = evaluate_predictions(
        pair_sets["val"].labels,
        validation_scores,
        pair_sets["val"].source_labels,
        validation_logits,
        pair_sets["test"].labels,
        test_scores,
        pair_sets["test"].source_labels,
        test_logits,
    )
    metrics.update(
        {
            "experiment": PHASE2_EXPERIMENT_NAME,
            "generated_at": utc_now(),
            "status": "completed",
            "protocol": protocol["name"],
            "protocol_label": protocol["label"],
            "model_type": "original_checkpoint_reference",
            "model_name": "original_graphsage_reference",
            "config": "c1_current_136",
            "config_label": "Original GraphSAGE checkpoint under Phase 2 Protocol A pairs",
            "model_config": model_config,
            "feature_names": checkpoint_features,
            "feature_dim": int(len(checkpoint_features)),
            "checkpoint_path": portable_path(checkpoint_path),
            "checkpoint_best_epoch": checkpoint.get("best_epoch"),
            "checkpoint_threshold": checkpoint.get("threshold"),
            "counts": {
                "train_edges": int(len(protocol["train_indices"])),
                "val_positives": pair_sets["val"].positive_count,
                "val_negatives": pair_sets["val"].negative_count,
                "test_positives": pair_sets["test"].positive_count,
                "test_negatives": pair_sets["test"].negative_count,
                "val_source_counts": pair_sets["val"].source_counts,
                "test_source_counts": pair_sets["test"].source_counts,
            },
            "protocol_stats": {
                "counts": protocol["counts"],
                "source_counts": protocol["source_counts"],
                "reverse_leakage": protocol["reverse_leakage"],
                "reverse_control": protocol["reverse_control"],
                "temporal": protocol["temporal"],
            },
        }
    )
    write_json(metrics_path, metrics)
    predictions_frame(data, pair_sets["val"], validation_scores, validation_logits).to_csv(
        output_dir / "predictions" / protocol["name"] / "original_graphsage_reference_val.csv",
        index=False,
    )
    predictions_frame(data, pair_sets["test"], test_scores, test_logits).to_csv(
        output_dir / "predictions" / protocol["name"] / "original_graphsage_reference_test.csv",
        index=False,
    )
    write_reliability_svg(
        output_dir / "plots" / f"{protocol['name']}_original_graphsage_reference_reliability.svg",
        metrics["calibration"],
        "A original checkpoint calibrated reliability",
    )
    logging.info(
        "Original checkpoint reference under Protocol A: test_auc=%.4f test_ap=%.4f test_f1=%.4f",
        metrics["test"]["roc_auc"],
        metrics["test"]["average_precision"],
        metrics["test"]["f1"],
    )
    return metrics


def original_artifact_record(
    output_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    metrics_path = ORIGINAL_EXPERIMENT_DIR / "metrics.json"
    predictions_path = ORIGINAL_EXPERIMENT_DIR / "test_predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        return None, None
    original_metrics = load_json(metrics_path, {})
    test = copy.deepcopy(original_metrics["test"])
    predictions = pd.read_csv(predictions_path)
    labels = predictions["label"].to_numpy(dtype=np.int64)
    scores = predictions["score"].to_numpy(dtype=np.float64)
    test["precision_at_100"] = precision_at_k(labels, scores, 100)
    test["precision_at_500"] = precision_at_k(labels, scores, 500)
    for source, payload in test.get("per_source", {}).items():
        mask = predictions["source"].astype(str).to_numpy() == source
        payload["precision_at_100"] = precision_at_k(labels[mask], scores[mask], 100)
        payload["precision_at_500"] = precision_at_k(labels[mask], scores[mask], 500)
    metrics = {
        "test": test,
        "calibration": {"test": calibration_report(labels, scores)},
    }
    summary = {
        "source": "ml/experiments/graphsage_link_prediction/metrics.json",
        "test_predictions": "ml/experiments/graphsage_link_prediction/test_predictions.csv",
        "reported_threshold": original_metrics["test"]["threshold"],
        "reported_test": {
            key: original_metrics["test"].get(key)
            for key in (
                "roc_auc",
                "average_precision",
                "precision",
                "recall",
                "f1",
                "threshold",
            )
        },
        "recomputed_test_precision_at_100": test["precision_at_100"],
        "recomputed_test_precision_at_500": test["precision_at_500"],
        "test_calibration_raw": calibration_report(labels, scores),
        "note": (
            "Reported original metrics use the original negative-sample realization; "
            "they are not directly pair-identical to Phase 2 Protocol A."
        ),
    }
    write_json(output_dir / "metrics" / "A" / "original_artifact_metrics.json", summary)
    record = {
        "protocol": "A",
        "model_type": "original_artifact",
        "model_name": "original_graphsage_reported",
        "metrics": metrics,
    }
    return record, summary


def feature_audit(data: Phase2Data) -> dict[str, Any]:
    raw_nodes = pd.read_csv(
        PROJECT_ROOT / "datasets" / "processed" / "graph" / "node_features.csv",
        low_memory=False,
    )
    raw_edges = pd.read_csv(
        PROJECT_ROOT / "datasets" / "processed" / "graph" / "edges.csv",
        dtype={"edge_id": str},
        low_memory=False,
    )
    node_rows = []
    for feature_name in data.feature_names:
        if feature_name.startswith("icsplc_emb_"):
            node_rows.append(
                {
                    "name": feature_name,
                    "type": "embedding",
                    "source": "datasets/training/node_embeddings.parquet",
                    "missing_rate": float(1.0 - data.emb_mask.mean()),
                    "cardinality": "continuous",
                    "n_unique": None,
                    "varies": True,
                    "available_at_prediction": "yes for nodes with an embedding; zero-filled otherwise",
                    "leakage_risk": "low",
                    "included_in_configs": ["c1_current_136", "c3_current_struct", "c4_current_cat", "c5_current_struct_cat"],
                }
            )
            continue
        raw_column = feature_name
        derived = False
        if feature_name == "has_edge_year":
            missing_rate = float(
                (
                    raw_nodes["first_edge_year"].isna()
                    | raw_nodes["last_edge_year"].isna()
                ).mean()
            )
            values = (
                raw_nodes["first_edge_year"].notna()
                & raw_nodes["last_edge_year"].notna()
            ).astype(int)
            derived = True
            raw_column = "first_edge_year,last_edge_year"
        elif feature_name in {"first_edge_age", "last_edge_age", "edge_year_span"}:
            missing_rate = float(
                (
                    raw_nodes["first_edge_year"].isna()
                    | raw_nodes["last_edge_year"].isna()
                ).mean()
            )
            values = pd.Series([0])
            derived = True
            raw_column = "first_edge_year,last_edge_year"
        elif feature_name in raw_nodes.columns:
            values = raw_nodes[feature_name]
            if feature_name == "is_isolated":
                missing_rate = 0.0
            else:
                missing_rate = float(values.isna().mean())
        else:
            values = pd.Series([0])
            missing_rate = 0.0
        if feature_name in FLAG_NAMES:
            feature_type = "binary_flag"
            cardinality: object = 2
            included = [
                "c1_current_136",
                "c2_flags8",
                "c3_current_struct",
                "c4_current_cat",
                "c5_current_struct_cat",
            ]
            available = "yes for existing nodes"
            leakage = "low"
        elif feature_name in {
            "log1p_degree",
            "log1p_in_degree",
            "log1p_out_degree",
            "log1p_raw_degree",
            "log1p_component_size",
            "first_edge_age",
            "last_edge_age",
            "edge_year_span",
        }:
            feature_type = "full_graph_structural"
            cardinality = "continuous"
            included = []
            available = "not used by Phase 2 configs"
            leakage = "high if full-graph derived and used for link prediction"
        elif feature_name == "has_edge_year":
            feature_type = "derived_binary"
            cardinality = 2
            included = []
            available = "yes at prediction time"
            leakage = "medium if year is future information"
        else:
            feature_type = "derived_numeric"
            cardinality = "continuous"
            included = []
            available = "unknown"
            leakage = "unknown"
        n_unique = None
        if isinstance(values, pd.Series):
            try:
                n_unique = int(values.nunique(dropna=True))
            except Exception:
                n_unique = None
        node_rows.append(
            {
                "name": feature_name,
                "type": feature_type,
                "source": (
                    "datasets/processed/graph/node_features.csv"
                    if not derived
                    else "derived from node_features.csv"
                ),
                "raw_columns": raw_column,
                "missing_rate": missing_rate,
                "cardinality": cardinality,
                "n_unique": n_unique,
                "varies": bool(n_unique is not None and n_unique > 1),
                "available_at_prediction": available,
                "leakage_risk": leakage,
                "included_in_configs": included,
            }
        )

    categorical_rows = []
    for index, column in enumerate(["country_code", "industry_code", "product_category_code"]):
        values = raw_nodes[column]
        categorical_rows.append(
            {
                "name": column,
                "type": "categorical_index",
                "source": "datasets/processed/graph/node_features.csv",
                "missing_value": -1,
                "missing_rate": float((values == -1).mean()),
                "cardinality": int(data.cat_cardinalities[index]),
                "n_unique_including_missing": int(values.nunique()),
                "available_at_prediction": "yes if the code is known; missing maps to index 0",
                "leakage_risk": "low",
                "included_in_configs": ["c4_current_cat", "c5_current_struct_cat"],
                "source_specific_embeddings": True,
                "reverse_mapping_performed": False,
            }
        )

    structural_rows = [
        {
            "name": name,
            "type": "train_only_structural",
            "source": "protocol training graph",
            "missing_rate": 0.0,
            "cardinality": "continuous",
            "varies": True,
            "available_at_prediction": "transductive only; not valid for entirely unseen nodes",
            "leakage_risk": "low when computed from the protocol training graph only",
            "included_in_configs": ["c3_current_struct", "c5_current_struct_cat"],
        }
        for name in STRUCTURAL_FEATURE_NAMES
    ]

    edge_rows = []
    for feature_name in data.edge_feature_names:
        if feature_name in {"first_year", "last_year", "edge_year_span"}:
            missing_rate = float(raw_edges["first_year"].isna().mean())
        elif feature_name == "has_edge_year":
            missing_rate = 0.0
        else:
            missing_rate = 0.0
        edge_rows.append(
            {
                "name": feature_name,
                "type": "edge_feature",
                "source": "datasets/processed/graph/edges.csv / build_edge_features",
                "missing_rate": missing_rate,
                "cardinality": "continuous or one-hot",
                "available_at_prediction": "only on observed training edges; not for an unseen candidate edge",
                "leakage_risk": "high if used as a scoring feature; safe for train-edge message passing",
                "included_in_configs": [],
                "config_6_status": "not implemented; requires a separate edge-aware architecture",
            }
        )

    source_coverage = {}
    for source in SOURCE_ORDER:
        mask = data.node_sources == source
        source_coverage[source] = {
            "nodes": int(mask.sum()),
            "unique_current_136_rows": int(np.unique(data.current_x[mask], axis=0).shape[0]),
            "unique_flag_rows": int(np.unique(data.flags_x[mask], axis=0).shape[0]),
            "embedding_coverage": float(data.emb_mask[mask].mean()),
            "categorical_coverage": {
                column: float((data.x_cat[mask, index] > 0).mean())
                for index, column in enumerate(
                    ["country_code", "industry_code", "product_category_code"]
                )
            },
        }
    return {
        "node_features": node_rows,
        "categorical_features": categorical_rows,
        "structural_features": structural_rows,
        "edge_features": edge_rows,
        "source_coverage": source_coverage,
        "notes": [
            "The current 136-dimensional model input is 8 static flags plus 128 IC-SPLC embedding dimensions.",
            "Full-graph structural features are excluded from every Phase 2 model.",
            "Structural ablation features are recomputed from each protocol training graph only.",
            "Categorical features use source-specific embedding tables; anonymized codes are not reverse-mapped.",
            "Edge features are audited but not used in configs 1-5; config 6 is not implemented.",
        ],
    }


def comparison_outputs(
    metric_records: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in metric_records:
        rows.extend(
            compare_rows_from_metrics(
                record["metrics"],
                protocol=record["protocol"],
                model_type=record["model_type"],
                model_name=record["model_name"],
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "reports" / "comparison_long.csv", index=False)
    if not frame.empty:
        pooled = frame[frame["source"] == "pooled"].copy()
        pooled.to_csv(output_dir / "reports" / "comparison_pooled.csv", index=False)
        by_source = frame[frame["source"] != "pooled"].copy()
        by_source.to_csv(output_dir / "reports" / "comparison_by_source.csv", index=False)
        best: dict[str, Any] = {}
        for (protocol, source), group in frame.groupby(["protocol", "source"]):
            valid = group.dropna(subset=["average_precision"])
            p500_group = group.dropna(subset=["precision_at_500"])
            best[f"{protocol}:{source}"] = {
                "by_average_precision": (
                    valid.sort_values(
                        ["average_precision", "precision_at_500"], ascending=False
                    ).iloc[0][["model_type", "model_name", "average_precision", "precision_at_500", "roc_auc"]].to_dict()
                    if len(valid)
                    else None
                ),
                "by_precision_at_500": (
                    p500_group.sort_values(
                        ["precision_at_500", "average_precision"], ascending=False
                    ).iloc[0][["model_type", "model_name", "average_precision", "precision_at_500", "roc_auc"]].to_dict()
                    if len(p500_group)
                    else None
                ),
            }
    else:
        best = {}
    return rows, best


def write_summary(
    output_dir: Path,
    data: Phase2Data,
    protocols: dict[str, Any],
    pair_sets: dict[str, dict[str, PairSet]],
    feature_audit_payload: dict[str, Any],
    metric_records: list[dict[str, Any]],
    original_artifact_summary: dict[str, Any] | None,
    best: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows, _ = comparison_outputs(metric_records, output_dir)
    summary = {
        "experiment": PHASE2_EXPERIMENT_NAME,
        "generated_at": utc_now(),
        "status": "completed",
        "seed": args.seed,
        "configs": CONFIG_SPECS,
        "protocols": {
            name: {
                **protocol_view(protocol),
                "eval_pairs": {
                    "val": {
                        "positives": pair_sets[name]["val"].positive_count,
                        "negatives": pair_sets[name]["val"].negative_count,
                        "source_counts": pair_sets[name]["val"].source_counts,
                    },
                    "test": {
                        "positives": pair_sets[name]["test"].positive_count,
                        "negatives": pair_sets[name]["test"].negative_count,
                        "source_counts": pair_sets[name]["test"].source_counts,
                    },
                },
            }
            for name, protocol in protocols.items()
        },
        "split_reproduction": split_reproduction_view(data.split_reproduction),
        "feature_audit": {
            "source_coverage": feature_audit_payload["source_coverage"],
            "categorical_features": feature_audit_payload["categorical_features"],
            "structural_features": feature_audit_payload["structural_features"],
            "notes": feature_audit_payload["notes"],
        },
        "comparison": rows,
        "best": best,
        "original_artifact": original_artifact_summary,
        "limitations": [
            "Protocol A uses the original random split; exact negative pairs are regenerated with a fixed seed, so they are not identical to the original negative realization.",
            "Protocol B groups reverse pairs and may change source-specific test counts slightly.",
            "Protocol C keeps the original evaluation pairs and removes reverse-of-evaluation edges from training, isolating reverse leakage.",
            "Protocol D covers IC-SPLC only because AnonNet has no edge years and a pooled temporal split is not comparable.",
            "Config 6 edge-aware modeling is not implemented; edge features are audited only.",
            "Test sets are balanced 50/50, so precision and AP are not production prevalence estimates.",
            "No verified company-level risk labels exist; this is link prediction, not disruption-risk prediction.",
            "No cross-source entity resolution exists; the three source graphs are disconnected.",
            "Component and degree features are transductive and not valid for entirely unseen nodes.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AtmoGraph GraphSAGE Phase 2 protocol and feature ablations")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--protocols", type=str, default="A,B,C,D")
    parser.add_argument("--configs", type=str, default=",".join(CONFIG_SPECS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--out-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--negatives-per-positive", type=float, default=1.0)
    parser.add_argument("--categorical-dim", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("config", "checkpoints", "predictions", "metrics", "reports", "plots", "logs"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    for protocol_name in args.protocols.split(","):
        (output_dir / "metrics" / protocol_name.strip()).mkdir(
            parents=True, exist_ok=True
        )
        (output_dir / "predictions" / protocol_name.strip()).mkdir(
            parents=True, exist_ok=True
        )
        (output_dir / "checkpoints" / protocol_name.strip()).mkdir(
            parents=True, exist_ok=True
        )
    configure_logging(output_dir / "logs" / "run.log")

    protocols_requested = parse_list(args.protocols)
    configs_requested = parse_list(args.configs)
    for protocol_name in protocols_requested:
        if protocol_name not in {"A", "B", "C", "D"}:
            raise SystemExit(f"Unknown protocol: {protocol_name}")
    for config_name in configs_requested:
        if config_name not in CONFIG_SPECS:
            raise SystemExit(f"Unknown config: {config_name}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but not available")
    set_seed(args.seed)

    logging.info("Loading Phase 2 data")
    data = load_phase2_data(seed=args.seed)
    logging.info(
        "Loaded nodes=%d edges=%d current_dim=%d cat=%s",
        data.num_nodes,
        len(data.edge_df),
        data.current_x.shape[1],
        data.cat_cardinalities,
    )
    logging.info(
        "Original split reproduction: match=%s rate=%.6f",
        data.split_reproduction["matches_original"],
        data.split_reproduction["match_rate"],
    )

    protocols = build_protocols(data, seed=args.seed)
    pair_sets = build_evaluation_pairs(data, protocols, seed=args.seed)
    feature_audit_payload = feature_audit(data)
    manifest = run_manifest(args, data, protocols)
    write_json(output_dir / "config" / "run_manifest.json", manifest)
    write_json(output_dir / "config" / "feature_audit.json", feature_audit_payload)
    write_json(
        output_dir / "config" / "protocols.json",
        {name: protocol_view(protocol) for name, protocol in protocols.items()},
    )
    protocol_split_frame = pd.DataFrame(
        {
            "edge_id": data.edge_ids,
            "source_dataset": data.edge_df["source_dataset"].to_numpy(),
        }
    )
    for name, protocol in protocols.items():
        protocol_split_frame[name] = protocol["split"]
    protocol_split_frame.to_csv(
        output_dir / "config" / "protocol_splits.csv", index=False
    )
    reverse_removals = {
        name: {
            "removed_train_edges": protocol["reverse_control"]["removed_train_edges"],
            "removed_train_edge_ids": protocol["reverse_control"][
                "removed_train_edge_ids"
            ],
        }
        for name, protocol in protocols.items()
        if protocol["reverse_control"]
        and protocol["reverse_control"].get("removed_train_edge_ids")
    }
    write_json(
        output_dir / "config" / "reverse_control_removals.json", reverse_removals
    )
    pd.DataFrame(feature_audit_payload["node_features"]).to_csv(
        output_dir / "config" / "node_feature_audit.csv", index=False
    )
    pd.DataFrame(feature_audit_payload["edge_features"]).to_csv(
        output_dir / "config" / "edge_feature_audit.csv", index=False
    )

    if args.build_only:
        logging.info("Build-only mode complete")
        return 0

    structural_cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for protocol_name in protocols_requested:
        protocol = protocols[protocol_name]
        scaler_mask = data.node_train_mask.copy()
        if protocol["source_filter"] is not None:
            scaler_mask &= data.node_sources == protocol["source_filter"]
        structural_cache[protocol_name] = build_structural_features(
            data, protocol, scaler_mask=scaler_mask
        )
        logging.info(
            "Structural features protocol=%s edges=%d components=%d",
            protocol_name,
            structural_cache[protocol_name][1]["source_graph_edges"],
            structural_cache[protocol_name][1]["component_count"],
        )

    metric_records: list[dict[str, Any]] = []
    for protocol_name in protocols_requested:
        protocol = protocols[protocol_name]
        logging.info("Protocol %s: %s", protocol_name, protocol["label"])
        if not args.skip_baselines:
            baseline_results = run_structural_baselines(
                data, protocol, pair_sets[protocol_name], output_dir, args
            )
            for baseline_name, metrics in baseline_results.items():
                metric_records.append(
                    {
                        "protocol": protocol_name,
                        "model_type": "baseline",
                        "model_name": baseline_name,
                        "metrics": metrics,
                    }
                )
        if protocol_name == "A":
            reference_metrics = run_original_checkpoint_reference(
                data, protocol, pair_sets[protocol_name], output_dir, args
            )
            if reference_metrics:
                metric_records.append(
                    {
                        "protocol": protocol_name,
                        "model_type": "original_checkpoint_reference",
                        "model_name": "original_graphsage_reference",
                        "metrics": reference_metrics,
                    }
                )
        for config_name in configs_requested:
            if protocol_name == "D" and config_name not in CONFIG_SPECS:
                continue
            metrics = train_and_evaluate_config(
                data,
                protocol,
                pair_sets[protocol_name],
                config_name,
                CONFIG_SPECS[config_name],
                structural_cache[protocol_name],
                output_dir,
                args,
            )
            metric_records.append(
                {
                    "protocol": protocol_name,
                    "model_type": "gnn",
                    "model_name": config_name,
                    "metrics": metrics,
                }
            )

    original_record, original_summary = original_artifact_record(output_dir)
    if original_record is not None:
        metric_records.append(original_record)

    rows, best = comparison_outputs(metric_records, output_dir)
    summary = write_summary(
        output_dir,
        data,
        protocols,
        pair_sets,
        feature_audit_payload,
        metric_records,
        original_summary,
        best,
        args,
    )
    logging.info("Wrote summary with %d comparison rows", len(rows))
    logging.info("Phase 2 experiment complete: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
