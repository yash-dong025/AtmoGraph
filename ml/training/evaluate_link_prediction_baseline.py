"""Evaluate structural link-prediction baselines on the AtmoGraph edge splits.

The baseline is deliberately simple and leakage-safe:

- message/score graph uses only train edges,
- validation/test positives come from their own split,
- negatives are source-matched directed non-edges and exclude every observed edge,
- per-source metrics are reported alongside the pooled result,
- the decision threshold is selected on validation and applied once to test.

This is a link-prediction baseline, not a disruption-risk model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
TRAINING_DIR = DATASETS_DIR / "training"
PROCESSED_DIR = DATASETS_DIR / "processed"
EXPERIMENT_DIR = PROJECT_ROOT / "ml" / "experiments" / "graphsage_link_prediction"

EDGE_INDEX_PATH = TRAINING_DIR / "edge_index.csv"
NODE_INDEX_PATH = TRAINING_DIR / "node_index_mapping.csv"
NODES_PATH = PROCESSED_DIR / "graph" / "nodes.csv"
EDGES_PATH = PROCESSED_DIR / "graph" / "edges.csv"
NODE_FEATURES_PATH = PROCESSED_DIR / "graph" / "node_features.csv"
SPLITS_PATH = TRAINING_DIR / "dataset_splits.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.feature_engineering import build_edge_features, build_node_features


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def load_node_source_labels() -> np.ndarray:
    node_index = pd.read_csv(NODE_INDEX_PATH, dtype={"node_id": str})
    node_index = node_index.sort_values("node_index").reset_index(drop=True)
    nodes = pd.read_csv(
        NODES_PATH,
        dtype={"company_id": str, "source_dataset": str},
        usecols=["company_id", "source_dataset"],
        low_memory=False,
    )
    source_by_id = (
        nodes.drop_duplicates("company_id")
        .set_index("company_id")["source_dataset"]
        .fillna("unknown")
        .astype(str)
        .to_dict()
    )
    labels = node_index["node_id"].map(source_by_id).fillna("unknown").astype(str).to_numpy()
    if len(labels) != len(node_index):
        raise ValueError("Could not map every node to a source dataset")
    return labels


def build_adjacency(
    source_index: np.ndarray, target_index: np.ndarray, num_nodes: int
) -> tuple[list[set[int]], np.ndarray]:
    adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
    for src, dst in zip(source_index.tolist(), target_index.tolist()):
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    degrees = np.fromiter((len(row) for row in adjacency), dtype=np.int64, count=num_nodes)
    return adjacency, degrees


def _source_groups(node_sources: np.ndarray) -> dict[str, np.ndarray]:
    groups: dict[str, np.ndarray] = {}
    for source in np.unique(node_sources):
        groups[str(source)] = np.flatnonzero(node_sources == source).astype(np.int64)
    return groups


def _sample_group_pairs(
    rng: np.random.Generator,
    source_pool: np.ndarray,
    target_pool: np.ndarray,
    count: int,
    num_nodes: int,
    observed_keys: set[int],
    chosen: set[int],
) -> tuple[list[int], list[int]]:
    if count <= 0:
        return [], []
    if len(source_pool) == 0 or len(target_pool) == 0:
        raise ValueError("A source group has no candidate nodes for negative sampling")
    if len(source_pool) == 1 and len(target_pool) == 1 and source_pool[0] == target_pool[0]:
        raise ValueError("A source group has too few nodes for source-matched negatives")

    sources: list[int] = []
    targets: list[int] = []
    attempts = 0
    while len(sources) < count:
        remaining = count - len(sources)
        batch = max(1024, remaining * 2)
        batch_sources = rng.choice(source_pool, size=batch, replace=True)
        batch_targets = rng.choice(target_pool, size=batch, replace=True)
        for src, dst in zip(batch_sources.tolist(), batch_targets.tolist()):
            if src == dst:
                continue
            key = src * num_nodes + dst
            if key in observed_keys or key in chosen:
                continue
            chosen.add(key)
            sources.append(src)
            targets.append(dst)
            if len(sources) >= count:
                break
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Could not sample enough source-matched negative pairs")
    return sources, targets


def sample_negative_pairs(
    rng: np.random.Generator,
    count: int,
    num_nodes: int,
    observed_keys: set[int],
    positive_sources: np.ndarray | None = None,
    positive_targets: np.ndarray | None = None,
    node_sources: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample directed non-edges while preserving the source-pair distribution."""
    if count <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if positive_sources is None or positive_targets is None or node_sources is None:
        raise ValueError(
            "source-matched negative sampling requires positive pairs and node sources"
        )
    if len(positive_sources) != len(positive_targets):
        raise ValueError("positive_sources and positive_targets must have equal length")
    if len(positive_sources) == 0:
        raise ValueError("Cannot sample negatives without positive pairs")

    node_sources = np.asarray(node_sources, dtype=object)
    groups = _source_groups(node_sources)
    source_groups: dict[tuple[str, str], list[int]] = {}
    for position, (src, dst) in enumerate(zip(positive_sources, positive_targets)):
        key = (str(node_sources[int(src)]), str(node_sources[int(dst)]))
        source_groups.setdefault(key, []).append(position)

    keys = list(source_groups)
    raw_allocations = [
        count * len(source_groups[key]) / len(positive_sources) for key in keys
    ]
    allocations = [int(np.floor(value)) for value in raw_allocations]
    remainder = count - sum(allocations)
    ranked = sorted(
        range(len(keys)),
        key=lambda index: raw_allocations[index] - allocations[index],
        reverse=True,
    )
    for index in ranked[:remainder]:
        allocations[index] += 1

    chosen: set[int] = set()
    negative_sources: list[int] = []
    negative_targets: list[int] = []
    for key, allocation in zip(keys, allocations):
        if allocation <= 0:
            continue
        group_sources, group_targets = _sample_group_pairs(
            rng,
            groups[key[0]],
            groups[key[1]],
            allocation,
            num_nodes,
            observed_keys,
            chosen,
        )
        negative_sources.extend(group_sources)
        negative_targets.extend(group_targets)

    return (
        np.asarray(negative_sources, dtype=np.int64),
        np.asarray(negative_targets, dtype=np.int64),
    )


def common_neighbor_scores(
    adjacency: list[set[int]], sources: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    scores = np.empty(len(sources), dtype=np.float64)
    for index, (src, dst) in enumerate(zip(sources.tolist(), targets.tolist())):
        scores[index] = len(adjacency[src].intersection(adjacency[dst]))
    return scores


def preferential_attachment_scores(
    degrees: np.ndarray, sources: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    return degrees[sources].astype(np.float64) * degrees[targets].astype(np.float64)


def threshold_from_validation(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return {"threshold": 0.5, "f1": 0.0, "non_trivial": False}
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    minimum_score = float(scores.min())
    non_trivial = thresholds > minimum_score
    if not bool(non_trivial.any()):
        non_trivial = np.ones_like(thresholds, dtype=bool)
    candidate_indices = np.flatnonzero(non_trivial)
    best_index = int(candidate_indices[np.argmax(f1[candidate_indices])])
    return {
        "threshold": float(thresholds[best_index]),
        "f1": float(f1[best_index]),
        "non_trivial": bool(thresholds[best_index] > minimum_score),
    }


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).astype(int).tolist()
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "confusion_matrix": matrix,
        "confusion_matrix_labels": ["negative", "positive"],
        "counts": {
            "positive": int(labels.sum()),
            "negative": int((labels == 0).sum()),
            "true_positive": int(true_positive),
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
        },
    }


def metrics_by_source(
    labels: np.ndarray,
    scores: np.ndarray,
    source_labels: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, object]]:
    """Return the same evaluate metrics for each source dataset."""
    source_labels = np.asarray(source_labels, dtype=object)
    result: dict[str, dict[str, object]] = {}
    for source in sorted(set(source_labels.tolist())):
        mask = source_labels == source
        source_labels_subset = labels[mask]
        if len(source_labels_subset) == 0:
            continue
        source_scores = scores[mask]
        positives = int(source_labels_subset.sum())
        negatives = int(len(source_labels_subset) - positives)
        if positives == 0 or negatives == 0:
            result[str(source)] = {
                "roc_auc": None,
                "average_precision": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "threshold": float(threshold),
                "counts": {"positive": positives, "negative": negatives},
                "note": "ROC-AUC/AP require both positive and negative examples",
            }
        else:
            result[str(source)] = binary_metrics(
                source_labels_subset, source_scores, threshold
            )
    return result


def downsample_curve(x_values: np.ndarray, y_values: np.ndarray, limit: int = 240) -> tuple[list[float], list[float]]:
    if len(x_values) <= limit:
        return [float(value) for value in x_values], [float(value) for value in y_values]
    indices = np.linspace(0, len(x_values) - 1, limit).round().astype(int)
    indices = np.unique(indices)
    return [float(x_values[i]) for i in indices], [float(y_values[i]) for i in indices]


def curve_payload(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    fpr_values, tpr_values = downsample_curve(fpr, tpr)
    recall_values, precision_values = downsample_curve(recall[::-1], precision[::-1])
    return {
        "roc": {"fpr": fpr_values, "tpr": tpr_values},
        "pr": {"recall": recall_values, "precision": precision_values},
    }


def score_distribution(
    positive_scores: np.ndarray, negative_scores: np.ndarray, use_log1p: bool
) -> dict[str, object]:
    if use_log1p:
        positive_values = np.log1p(positive_scores)
        negative_values = np.log1p(negative_scores)
    else:
        positive_values = positive_scores.astype(np.float64)
        negative_values = negative_scores.astype(np.float64)

    all_values = np.concatenate([positive_values, negative_values])
    minimum = float(all_values.min())
    maximum = float(all_values.max())
    if maximum - minimum <= 1e-12:
        bin_edges = np.asarray([minimum - 0.5, maximum + 0.5], dtype=np.float64)
    elif (
        not use_log1p
        and np.allclose(all_values, np.round(all_values))
        and maximum <= 12
    ):
        bin_edges = np.arange(-0.5, maximum + 1.5, 1.0)
    else:
        bin_edges = np.linspace(minimum, maximum + 1e-9, 21)

    positive_counts, _ = np.histogram(positive_values, bins=bin_edges)
    negative_counts, _ = np.histogram(negative_values, bins=bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return {
        "x_transform": "log1p" if use_log1p else "none",
        "bin_edges": [float(value) for value in bin_edges],
        "bin_centers": [float(value) for value in centers],
        "positive_counts": [int(value) for value in positive_counts],
        "negative_counts": [int(value) for value in negative_counts],
        "positive_stats": {
            "min": float(positive_values.min()),
            "max": float(positive_values.max()),
            "mean": float(positive_values.mean()),
        },
        "negative_stats": {
            "min": float(negative_values.min()),
            "max": float(negative_values.max()),
            "mean": float(negative_values.mean()),
        },
    }


def evaluate_baseline(
    name: str,
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    validation_sources: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_sources: np.ndarray,
    use_log1p_distribution: bool,
    include_curves: bool,
) -> dict[str, object]:
    threshold_info = threshold_from_validation(validation_labels, validation_scores)
    validation = binary_metrics(validation_labels, validation_scores, threshold_info["threshold"])
    test = binary_metrics(test_labels, test_scores, threshold_info["threshold"])
    validation["per_source"] = metrics_by_source(
        validation_labels, validation_scores, validation_sources, threshold_info["threshold"]
    )
    test["per_source"] = metrics_by_source(
        test_labels, test_scores, test_sources, threshold_info["threshold"]
    )
    if include_curves:
        test["curves"] = curve_payload(test_labels, test_scores)
        test["score_distribution"] = score_distribution(
            test_scores[test_labels == 1],
            test_scores[test_labels == 0],
            use_log1p_distribution,
        )
    return {
        "name": name,
        "validation": validation,
        "test": test,
        "threshold_selection": {
            "source": "validation",
            "criterion": "maximum F1 among thresholds that predict at least one negative",
            "selected_threshold": threshold_info["threshold"],
            "validation_f1_at_selection": threshold_info["f1"],
            "non_trivial_threshold": threshold_info["non_trivial"],
        },
    }


def build_dataset_metadata(
    edge_index: pd.DataFrame,
    node_index: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    node_feature_dim: int,
    raw_node_feature_dim: int,
    edge_feature_dim: int,
    seed: int,
) -> dict[str, object]:
    splits = json.loads(SPLITS_PATH.read_text())
    edge_splits = {
        name: int((edge_index["split"] == name).sum()) for name in ("train", "val", "test")
    }
    node_splits = {
        name: len(splits["node_splits"][name]) for name in ("train", "val", "test")
    }
    source_counts = nodes["source_dataset"].fillna("unknown").value_counts().to_dict()
    relationship_counts = edges["relationship_type"].fillna("unknown").value_counts().to_dict()
    files = {
        "nodes": file_record(NODES_PATH),
        "edges": file_record(EDGES_PATH),
        "edge_index": file_record(EDGE_INDEX_PATH),
        "node_index_mapping": file_record(NODE_INDEX_PATH),
        "node_features": file_record(NODE_FEATURES_PATH),
    }
    short_digest = hashlib.sha256(
        "".join(str(files[key]["sha256"]) for key in ("nodes", "edges", "edge_index")).encode()
    ).hexdigest()[:12]
    return {
        "version": f"atmo_graph_v1_{short_digest}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "node_count": int(len(node_index)),
        "edge_count": int(len(edge_index)),
        "node_feature_dim": int(node_feature_dim),
        "raw_node_feature_columns": int(raw_node_feature_dim),
        "edge_feature_dim": int(edge_feature_dim),
        "node_split_sizes": node_splits,
        "edge_split_sizes": edge_splits,
        "source_counts": {str(key): int(value) for key, value in source_counts.items()},
        "relationship_counts": {
            str(key): int(value) for key, value in relationship_counts.items()
        },
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_index = pd.read_csv(EDGE_INDEX_PATH, dtype={"edge_id": str})
    node_index = pd.read_csv(NODE_INDEX_PATH, dtype={"node_id": str})
    nodes = pd.read_csv(NODES_PATH, dtype={"company_id": str}, low_memory=False)
    edges = pd.read_csv(EDGES_PATH, dtype={"edge_id": str})
    node_features = pd.read_csv(NODE_FEATURES_PATH, nrows=1)

    num_nodes = int(node_index["node_index"].max()) + 1
    if len(node_index) != num_nodes:
        raise ValueError("node_index_mapping.csv is not contiguous")

    source_indices = edge_index["source_index"].to_numpy(dtype=np.int64)
    target_indices = edge_index["target_index"].to_numpy(dtype=np.int64)
    split_values = edge_index["split"].to_numpy()
    observed_keys = set(
        (source_indices.astype(np.int64) * num_nodes + target_indices.astype(np.int64)).tolist()
    )
    if len(observed_keys) != len(edge_index):
        raise ValueError("edge_index.csv contains duplicate directed pairs")

    train_mask = split_values == "train"
    adjacency, train_degrees = build_adjacency(
        source_indices[train_mask], target_indices[train_mask], num_nodes
    )

    node_sources = load_node_source_labels()
    if len(node_sources) != num_nodes:
        raise ValueError("node source labels are not aligned with node_index_mapping.csv")

    rng = np.random.default_rng(args.seed)
    split_pairs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in ("val", "test"):
        mask = split_values == split
        positive_sources = source_indices[mask]
        positive_targets = target_indices[mask]
        negative_sources, negative_targets = sample_negative_pairs(
            rng,
            len(positive_sources),
            num_nodes,
            observed_keys,
            positive_sources=positive_sources,
            positive_targets=positive_targets,
            node_sources=node_sources,
        )
        samples_sources = np.concatenate([positive_sources, negative_sources])
        samples_targets = np.concatenate([positive_targets, negative_targets])
        labels = np.concatenate(
            [np.ones(len(positive_sources), dtype=np.int64), np.zeros(len(negative_sources), dtype=np.int64)]
        )
        source_labels = node_sources[samples_sources]
        split_pairs[split] = (samples_sources, samples_targets, labels, source_labels)

    validation_sources, validation_targets, validation_labels, validation_source_labels = split_pairs["val"]
    test_sources, test_targets, test_labels, test_source_labels = split_pairs["test"]
    cross_source_negatives = int(
        np.sum(
            test_source_labels[test_labels == 0]
            != node_sources[test_targets[test_labels == 0]]
        )
    )

    score_functions = {
        "common_neighbors": lambda sources, targets: common_neighbor_scores(
            adjacency, sources, targets
        ),
        "preferential_attachment": lambda sources, targets: preferential_attachment_scores(
            train_degrees, sources, targets
        ),
    }

    evaluated: dict[str, dict[str, object]] = {}
    validation_aucs: dict[str, float] = {}
    validation_aps: dict[str, float] = {}
    for name, score_function in score_functions.items():
        validation_scores = score_function(validation_sources, validation_targets)
        test_scores = score_function(test_sources, test_targets)
        result = evaluate_baseline(
            name,
            validation_scores,
            validation_labels,
            validation_source_labels,
            test_scores,
            test_labels,
            test_source_labels,
            use_log1p_distribution=name == "preferential_attachment",
            include_curves=True,
        )
        evaluated[name] = result
        validation_aucs[name] = float(result["validation"]["roc_auc"])
        validation_aps[name] = float(result["validation"]["average_precision"])

    selected_name = max(
        validation_aps,
        key=lambda name: (validation_aps[name], validation_aucs[name]),
    )
    selected = evaluated[selected_name]
    alternatives = [
        {
            "name": name,
            "validation": result["validation"],
            "test": {
                key: value
                for key, value in result["test"].items()
                if key not in {"curves", "score_distribution"}
            },
            "threshold_selection": result["threshold_selection"],
        }
        for name, result in evaluated.items()
        if name != selected_name
    ]

    node_feature_dim = int(build_node_features().x.shape[1])
    raw_node_feature_dim = int(node_features.shape[1] - 1)
    edge_feature_dim = int(build_edge_features().edge_attr.shape[1])
    dataset_metadata = build_dataset_metadata(
        edge_index,
        node_index,
        nodes,
        edges,
        node_feature_dim,
        raw_node_feature_dim,
        edge_feature_dim,
        args.seed,
    )
    protocol = {
        "task": "directed link prediction",
        "train_graph": "structural scores use only train-split edges",
        "positives": "validation/test positives come from their own split",
        "negatives": "equal number per positive, sampled from directed non-edges while matching the positive source-pair distribution",
        "negative_exclusion": "all observed directed edges are excluded",
        "cross_source_negatives": cross_source_negatives,
        "source_matched_negatives": True,
        "per_source_metrics": "reported for validation and test; threshold is selected on pooled validation and reused for every source",
        "threshold": "selected on validation by maximum F1 among non-trivial thresholds, then applied once to test",
        "threshold_note": "the trivial all-positive threshold is excluded when a higher threshold exists",
        "common_neighbors_graph": "train edges treated as undirected for common-neighbor counting",
        "preferential_attachment_graph": "undirected train degree product",
        "not_a_risk_model": "link prediction is not disruption-risk prediction",
        "negative_sampling_seed": args.seed,
    }

    payload = {
        "experiment": "graphsage_link_prediction",
        "artifact": "structural_link_prediction_baseline",
        "status": "baseline_complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dataset_version": dataset_metadata["version"],
        "dataset": dataset_metadata,
        "protocol": protocol,
        "selected_baseline": selected_name,
        "selection_criterion": "highest validation average precision, then validation ROC-AUC",
        "validation_aucs": validation_aucs,
        "validation_average_precisions": validation_aps,
        "negative_sampling": {
            "scheme": "source-matched directed non-edges",
            "source_pair_distribution": "matches positive edges",
            "cross_source_negatives": cross_source_negatives,
            "observed_edges_excluded": len(observed_keys),
        },
        "baseline": selected,
        "alternatives": alternatives,
        "comparison": {
            "graphsage_link_prediction": {
                "status": "not_trained",
                "reason": "No trained GraphSAGE checkpoint exists in this experiment directory.",
            },
            "baseline_is_not_gnn": True,
        },
    }

    metrics_path = output_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")

    metadata_path = output_dir / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(dataset_metadata, indent=2) + "\n")

    selected_test_scores = score_functions[selected_name](test_sources, test_targets)
    split_mask = split_values == "test"
    positive_rows = pd.DataFrame(
        {
            "edge_id": edge_index.loc[split_mask, "edge_id"].to_numpy(),
            "source_index": source_indices[split_mask],
            "target_index": target_indices[split_mask],
            "source_company_id": edge_index.loc[split_mask, "source_company_id"].to_numpy(),
            "target_company_id": edge_index.loc[split_mask, "target_company_id"].to_numpy(),
            "source": test_source_labels[: int(split_mask.sum())],
            "label": 1,
            "score": selected_test_scores[: int(split_mask.sum())],
        }
    )
    positive_count = int(split_mask.sum())
    index_to_node = node_index.set_index("node_index")["node_id"]
    negative_rows = pd.DataFrame(
        {
            "edge_id": [f"sampled_negative_{index:06d}" for index in range(len(test_sources) - positive_count)],
            "source_index": test_sources[positive_count:],
            "target_index": test_targets[positive_count:],
            "source_company_id": index_to_node.loc[test_sources[positive_count:]].to_numpy(),
            "target_company_id": index_to_node.loc[test_targets[positive_count:]].to_numpy(),
            "source": test_source_labels[positive_count:],
            "label": 0,
            "score": selected_test_scores[positive_count:],
        }
    )
    predictions = pd.concat([positive_rows, negative_rows], ignore_index=True)
    predictions.insert(0, "baseline", selected_name)
    predictions.to_csv(output_dir / "baseline_test_predictions.csv", index=False)

    print(json.dumps({
        "selected_baseline": selected_name,
        "selection_criterion": "highest validation average precision, then validation ROC-AUC",
        "validation_aucs": validation_aucs,
        "validation_average_precisions": validation_aps,
        "cross_source_negatives": cross_source_negatives,
        "test_roc_auc": selected["test"]["roc_auc"],
        "test_average_precision": selected["test"]["average_precision"],
        "test_f1": selected["test"]["f1"],
        "test_per_source": selected["test"]["per_source"],
        "threshold": selected["test"]["threshold"],
        "metrics_path": str(metrics_path),
        "predictions_path": str(output_dir / "baseline_test_predictions.csv"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
