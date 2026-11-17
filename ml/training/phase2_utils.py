from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.feature_engineering import build_edge_features, build_node_features
from ml.training.evaluate_link_prediction_baseline import (
    binary_metrics,
    curve_payload,
    load_node_source_labels,
    metrics_by_source,
    sample_negative_pairs,
    threshold_from_validation,
)

DATASETS_DIR = PROJECT_ROOT / "datasets"
TRAINING_DIR = DATASETS_DIR / "training"
PROCESSED_DIR = DATASETS_DIR / "processed"
EDGE_INDEX_PATH = TRAINING_DIR / "edge_index.csv"
EDGES_PATH = PROCESSED_DIR / "graph" / "edges.csv"
NODE_FEATURES_PATH = PROCESSED_DIR / "graph" / "node_features.csv"
ORIGINAL_EXPERIMENT_DIR = PROJECT_ROOT / "ml" / "experiments" / "graphsage_link_prediction"

SEED = 42
SOURCE_ORDER = ("sampoq", "ic_splc", "anonnet")
FLAG_NAMES = [
    "source_sampoq",
    "source_ic_splc",
    "source_anonnet",
    "has_country",
    "has_industry",
    "has_product_category",
    "embedding_available",
    "is_anonymised",
]
EMBEDDING_NAMES = [f"icsplc_emb_{index}" for index in range(128)]
CURRENT_FEATURE_NAMES = FLAG_NAMES + EMBEDDING_NAMES
STRUCTURAL_FEATURE_NAMES = [
    "train_in_degree",
    "train_out_degree",
    "train_undirected_degree",
    "train_directed_degree",
    "log1p_train_in_degree",
    "log1p_train_out_degree",
    "log1p_train_undirected_degree",
    "log1p_train_directed_degree",
    "log1p_train_component_size",
    "train_is_isolated",
]
CONFIG_SPECS = {
    "c1_current_136": {
        "label": "Current 136-dimensional input",
        "base": "current",
        "structural": False,
        "categorical": False,
        "description": "8 static flags plus 128 IC-SPLC embedding dimensions.",
    },
    "c2_flags8": {
        "label": "Static flags only",
        "base": "flags",
        "structural": False,
        "categorical": False,
        "description": "The 8 static flags, without embeddings.",
    },
    "c3_current_struct": {
        "label": "Current plus train-only structural",
        "base": "current",
        "structural": True,
        "categorical": False,
        "description": "Current 136 inputs plus train-graph structural features.",
    },
    "c4_current_cat": {
        "label": "Current plus categorical embeddings",
        "base": "current",
        "structural": False,
        "categorical": True,
        "description": "Current 136 inputs plus source-specific categorical embeddings.",
    },
    "c5_current_struct_cat": {
        "label": "Current plus structural and categorical",
        "base": "current",
        "structural": True,
        "categorical": True,
        "description": "Current inputs plus train-only structural and source-specific categorical embeddings.",
    },
}


@dataclass
class Phase2Data:
    node_ids: list[str]
    num_nodes: int
    x: np.ndarray
    feature_names: list[str]
    x_cat: np.ndarray
    cat_cardinalities: list[int]
    emb_mask: np.ndarray
    node_sources: np.ndarray
    source_ids: np.ndarray
    node_train_mask: np.ndarray
    edge_df: pd.DataFrame
    edge_index: np.ndarray
    edge_ids: list[str]
    edge_attr: np.ndarray
    edge_feature_names: list[str]
    current_x: np.ndarray
    flags_x: np.ndarray
    current_indices: np.ndarray
    flag_indices: np.ndarray
    original_split: np.ndarray
    split_reproduction: dict[str, Any]


@dataclass
class PairSet:
    sources: np.ndarray
    targets: np.ndarray
    labels: np.ndarray
    source_labels: np.ndarray
    pair_ids: list[str]
    positive_count: int
    negative_count: int
    source_counts: dict[str, dict[str, int]]


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_phase2_data(seed: int = SEED) -> Phase2Data:
    node_bundle = build_node_features()
    edge_bundle = build_edge_features()

    edge_index_frame = pd.read_csv(
        EDGE_INDEX_PATH,
        dtype={"edge_id": str, "source_company_id": str, "target_company_id": str},
    )
    edges_frame = pd.read_csv(
        EDGES_PATH,
        dtype={"edge_id": str, "source_company_id": str, "target_company_id": str},
        usecols=[
            "edge_id",
            "source_company_id",
            "target_company_id",
            "source_datasets",
            "first_year",
            "last_year",
            "relationship_type",
        ],
    )
    edge_df = edge_index_frame.merge(
        edges_frame,
        on="edge_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_edge"),
    )
    if edge_df["source_datasets"].isna().any():
        raise ValueError("Some edge IDs are missing from edges.csv")
    edge_df = edge_df.set_index("edge_id").reindex(edge_bundle.edge_ids).reset_index()
    if edge_df["edge_id"].tolist() != list(edge_bundle.edge_ids):
        raise ValueError("edge_index.csv ordering is inconsistent with build_edge_features")
    if not np.array_equal(
        edge_df[["source_index", "target_index"]].to_numpy(dtype=np.int64),
        edge_bundle.edge_index.T,
    ):
        raise ValueError("edge_index.csv indices are inconsistent with build_edge_features")

    edge_df = edge_df.rename(columns={"source_datasets": "source_dataset"})
    edge_df["source_dataset"] = edge_df["source_dataset"].astype(str)
    invalid_sources = sorted(set(edge_df["source_dataset"]) - set(SOURCE_ORDER))
    if invalid_sources:
        raise ValueError(f"Unknown edge source datasets: {invalid_sources}")

    node_sources = load_node_source_labels()
    if len(node_sources) != len(node_bundle.node_ids):
        raise ValueError("Node source labels are not aligned with node features")
    source_to_id = {source: index for index, source in enumerate(SOURCE_ORDER)}
    source_ids = np.array(
        [source_to_id.get(str(source), -1) for source in node_sources], dtype=np.int64
    )
    if int(source_ids.min()) < 0:
        raise ValueError("Unknown node source label")

    name_to_index = {name: index for index, name in enumerate(node_bundle.feature_names)}
    missing_current = [name for name in CURRENT_FEATURE_NAMES if name not in name_to_index]
    missing_flags = [name for name in FLAG_NAMES if name not in name_to_index]
    if missing_current or missing_flags:
        raise ValueError(
            f"Missing required features: current={missing_current}, flags={missing_flags}"
        )
    current_indices = np.array(
        [name_to_index[name] for name in CURRENT_FEATURE_NAMES], dtype=np.int64
    )
    flag_indices = np.array([name_to_index[name] for name in FLAG_NAMES], dtype=np.int64)
    current_x = node_bundle.x[:, current_indices].astype(np.float32)
    flags_x = node_bundle.x[:, flag_indices].astype(np.float32)

    original_split = reproduce_original_split(edge_df, seed=seed)

    return Phase2Data(
        node_ids=node_bundle.node_ids,
        num_nodes=len(node_bundle.node_ids),
        x=node_bundle.x,
        feature_names=node_bundle.feature_names,
        x_cat=node_bundle.x_cat.astype(np.int64),
        cat_cardinalities=node_bundle.cat_cardinalities,
        emb_mask=node_bundle.emb_mask,
        node_sources=np.asarray(node_sources, dtype=object),
        source_ids=source_ids,
        node_train_mask=node_bundle.train_mask,
        edge_df=edge_df,
        edge_index=edge_bundle.edge_index.astype(np.int64),
        edge_ids=list(edge_bundle.edge_ids),
        edge_attr=edge_bundle.edge_attr,
        edge_feature_names=list(edge_bundle.feature_names),
        current_x=current_x,
        flags_x=flags_x,
        current_indices=current_indices,
        flag_indices=flag_indices,
        original_split=original_split["split"],
        split_reproduction=original_split,
    )


def reproduce_original_split(edge_df: pd.DataFrame, seed: int = SEED) -> dict[str, Any]:
    n_edges = len(edge_df)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_edges)
    n_train = int(round(n_edges * 0.80))
    n_val = int(round(n_edges * 0.10))
    split = np.empty(n_edges, dtype=object)
    split[permutation[:n_train]] = "train"
    split[permutation[n_train : n_train + n_val]] = "val"
    split[permutation[n_train + n_val :]] = "test"
    original = edge_df["split"].to_numpy()
    matches = split == original
    counts = {name: int((split == name).sum()) for name in ("train", "val", "test")}
    source_counts = (
        edge_df.assign(_generated_split=split)
        .groupby(["source_dataset", "_generated_split"])
        .size()
        .unstack(fill_value=0)
        .to_dict()
    )
    return {
        "split": split,
        "seed": seed,
        "matches_original": bool(matches.all()),
        "match_rate": float(matches.mean()),
        "mismatch_count": int((~matches).sum()),
        "counts": counts,
        "source_counts": {
            str(source): {str(key): int(value) for key, value in values.items()}
            for source, values in source_counts.items()
        },
        "generator": "scripts/build_graph_dataset.py:335-346",
        "note": "The original random edge split is reproducible with seed 42 on edge_index.csv ordering.",
    }


def _reverse_keys(source_index: np.ndarray, target_index: np.ndarray, num_nodes: int) -> np.ndarray:
    return target_index.astype(np.int64) * num_nodes + source_index.astype(np.int64)


def _forward_keys(source_index: np.ndarray, target_index: np.ndarray, num_nodes: int) -> np.ndarray:
    return source_index.astype(np.int64) * num_nodes + target_index.astype(np.int64)


def build_reverse_aware_split(edge_df: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    split = np.full(len(edge_df), "excluded", dtype=object)
    source_values = edge_df["source_dataset"].to_numpy()
    source_index = edge_df["source_index"].to_numpy(dtype=np.int64)
    target_index = edge_df["target_index"].to_numpy(dtype=np.int64)
    for source_offset, source in enumerate(SOURCE_ORDER):
        source_edge_indices = np.flatnonzero(source_values == source)
        if len(source_edge_indices) == 0:
            continue
        groups: dict[tuple[int, int], list[int]] = {}
        for edge_index in source_edge_indices.tolist():
            left = int(source_index[edge_index])
            right = int(target_index[edge_index])
            key = (min(left, right), max(left, right))
            groups.setdefault(key, []).append(edge_index)
        group_keys = sorted(groups)
        rng = np.random.default_rng(seed + 100 * (source_offset + 1))
        rng.shuffle(group_keys)
        target_train = int(round(len(source_edge_indices) * 0.80))
        target_val = int(round(len(source_edge_indices) * 0.10))
        train_count = 0
        val_count = 0
        for group_key in group_keys:
            group = groups[group_key]
            if train_count < target_train:
                assigned = "train"
                train_count += len(group)
            elif val_count < target_val:
                assigned = "val"
                val_count += len(group)
            else:
                assigned = "test"
            for edge_index in group:
                split[edge_index] = assigned
    return split


def apply_reverse_control(
    split: np.ndarray,
    edge_df: pd.DataFrame,
    num_nodes: int,
    evaluation_splits: tuple[str, ...] = ("val", "test"),
) -> tuple[np.ndarray, dict[str, Any]]:
    controlled = split.copy()
    train_indices = np.flatnonzero(split == "train")
    evaluation_indices = np.flatnonzero(np.isin(split, list(evaluation_splits)))
    source_index = edge_df["source_index"].to_numpy(dtype=np.int64)
    target_index = edge_df["target_index"].to_numpy(dtype=np.int64)
    evaluation_reverse_keys = set(
        _reverse_keys(
            source_index[evaluation_indices],
            target_index[evaluation_indices],
            num_nodes,
        ).tolist()
    )
    train_forward_keys = _forward_keys(
        source_index[train_indices], target_index[train_indices], num_nodes
    )
    removed_mask = np.array(
        [int(key) in evaluation_reverse_keys for key in train_forward_keys.tolist()],
        dtype=bool,
    )
    removed_indices = train_indices[removed_mask]
    controlled[removed_indices] = "excluded"
    return controlled, {
        "removed_train_edges": int(len(removed_indices)),
        "removed_train_edge_ids": [str(edge_df.iloc[index]["edge_id"]) for index in removed_indices],
        "evaluation_reverse_keys": int(len(evaluation_reverse_keys)),
        "rule": "Remove a training edge when its reverse is a validation or test positive.",
    }


def build_temporal_split(edge_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    split = np.full(len(edge_df), "excluded", dtype=object)
    source_values = edge_df["source_dataset"].to_numpy()
    years = pd.to_numeric(edge_df["first_year"], errors="coerce").to_numpy()
    ic_mask = source_values == "ic_splc"
    if not np.isfinite(years[ic_mask]).all():
        raise ValueError("IC-SPLC temporal split requires complete edge years")
    split[ic_mask & (years <= 2023)] = "train"
    split[ic_mask & (years == 2024)] = "val"
    split[ic_mask & (years == 2025)] = "test"
    return split, {
        "source": "ic_splc",
        "train_years": "<= 2023",
        "validation_year": "2024",
        "test_year": "2025",
        "reason_for_source_limit": (
            "AnonNet has no edge years. SAMPOQ years are filing-derived and 92% of its edges "
            "are from 2026, so a pooled temporal split is not scientifically comparable across sources."
        ),
        "year_coverage": {
            "ic_splc": int(np.isfinite(years[ic_mask]).sum()),
            "anonnet": int(np.isfinite(years[source_values == 'anonnet']).sum()),
            "sampoq": int(np.isfinite(years[source_values == 'sampoq']).sum()),
        },
    }


def _split_counts_by_source(
    edge_df: pd.DataFrame, split: np.ndarray
) -> dict[str, dict[str, int]]:
    frame = edge_df[["source_dataset"]].copy()
    frame["split"] = split
    result: dict[str, dict[str, int]] = {}
    for source in SOURCE_ORDER:
        source_frame = frame[frame["source_dataset"] == source]
        result[source] = {
            name: int((source_frame["split"] == name).sum())
            for name in ("train", "val", "test", "excluded")
        }
    return result


def reverse_leakage_report(
    edge_df: pd.DataFrame,
    split: np.ndarray,
    num_nodes: int,
) -> dict[str, Any]:
    source_index = edge_df["source_index"].to_numpy(dtype=np.int64)
    target_index = edge_df["target_index"].to_numpy(dtype=np.int64)
    source_values = edge_df["source_dataset"].to_numpy()
    train_keys = set(
        _forward_keys(
            source_index[split == "train"], target_index[split == "train"], num_nodes
        ).tolist()
    )
    result: dict[str, Any] = {}
    for evaluation_split in ("val", "test"):
        result[evaluation_split] = {}
        evaluation_indices = np.flatnonzero(split == evaluation_split)
        for source in SOURCE_ORDER:
            source_indices = evaluation_indices[source_values[evaluation_indices] == source]
            if len(source_indices) == 0:
                result[evaluation_split][source] = {
                    "positives": 0,
                    "reverse_in_train": 0,
                    "share": 0.0,
                }
                continue
            reverse_keys = _reverse_keys(
                source_index[source_indices], target_index[source_indices], num_nodes
            )
            reverse_count = int(sum(int(key) in train_keys for key in reverse_keys.tolist()))
            result[evaluation_split][source] = {
                "positives": int(len(source_indices)),
                "reverse_in_train": reverse_count,
                "share": float(reverse_count / len(source_indices)),
            }
    return result


def build_protocols(data: Phase2Data, seed: int = SEED) -> dict[str, dict[str, Any]]:
    protocols: dict[str, dict[str, Any]] = {}

    a_split = data.original_split.copy()
    protocols["A"] = {
        "name": "A",
        "label": "Existing random/transductive split",
        "split": a_split,
        "description": "Exact original 80/10/10 random edge split, reproduced with seed 42.",
        "source_filter": None,
        "reverse_control": None,
        "temporal": None,
    }

    b_split = build_reverse_aware_split(data.edge_df, seed=seed)
    protocols["B"] = {
        "name": "B",
        "label": "Reverse-edge-aware split",
        "split": b_split,
        "description": (
            "Reverse-direction edges are grouped and assigned to the same split. "
            "No test positive has its reverse in training."
        ),
        "source_filter": None,
        "reverse_control": {
            "method": "group reverse pairs",
            "removed_train_edges": 0,
        },
        "temporal": None,
    }

    c_split, c_reverse_control = apply_reverse_control(
        a_split, data.edge_df, data.num_nodes, evaluation_splits=("val", "test")
    )
    protocols["C"] = {
        "name": "C",
        "label": "Edge-disjoint reverse control on the original split",
        "split": c_split,
        "description": (
            "Original split and original evaluation positives, but training edges whose "
            "reverse is a validation/test positive are excluded from message passing and optimization."
        ),
        "source_filter": None,
        "reverse_control": c_reverse_control,
        "temporal": None,
    }

    d_split_raw, temporal_info = build_temporal_split(data.edge_df)
    d_split, d_reverse_control = apply_reverse_control(
        d_split_raw, data.edge_df, data.num_nodes, evaluation_splits=("val", "test")
    )
    protocols["D"] = {
        "name": "D",
        "label": "Temporal evaluation (IC-SPLC only)",
        "split": d_split,
        "description": (
            "Chronological IC-SPLC split: train <=2023, validation 2024, test 2025. "
            "Reverse counterparts of validation/test positives are removed from training."
        ),
        "source_filter": "ic_splc",
        "reverse_control": d_reverse_control,
        "temporal": temporal_info,
    }

    for protocol in protocols.values():
        split = protocol["split"]
        protocol["train_indices"] = np.flatnonzero(split == "train")
        protocol["val_indices"] = np.flatnonzero(split == "val")
        protocol["test_indices"] = np.flatnonzero(split == "test")
        protocol["counts"] = {
            name: int((split == name).sum())
            for name in ("train", "val", "test", "excluded")
        }
        protocol["source_counts"] = _split_counts_by_source(data.edge_df, split)
        protocol["reverse_leakage"] = reverse_leakage_report(
            data.edge_df, split, data.num_nodes
        )
        if protocol["source_filter"] is not None:
            source_mask = data.edge_df["source_dataset"].to_numpy() == protocol["source_filter"]
            protocol["counts"]["train"] = int(
                ((split == "train") & source_mask).sum()
            )
            protocol["counts"]["val"] = int(((split == "val") & source_mask).sum())
            protocol["counts"]["test"] = int(((split == "test") & source_mask).sum())
    return protocols


def all_observed_keys(data: Phase2Data, include_reverse: bool = True) -> set[int]:
    forward = _forward_keys(
        data.edge_index[0], data.edge_index[1], data.num_nodes
    )
    keys = set(int(key) for key in forward.tolist())
    if include_reverse:
        keys.update(
            int(key)
            for key in _reverse_keys(
                data.edge_index[0], data.edge_index[1], data.num_nodes
            ).tolist()
        )
    return keys


def _pair_source_counts(
    source_labels: np.ndarray, labels: np.ndarray
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for source in SOURCE_ORDER:
        mask = source_labels == source
        if not mask.any():
            result[source] = {"positive": 0, "negative": 0, "total": 0}
            continue
        source_labels_subset = labels[mask]
        result[source] = {
            "positive": int((source_labels_subset == 1).sum()),
            "negative": int((source_labels_subset == 0).sum()),
            "total": int(mask.sum()),
        }
    return result


def build_pair_set(
    data: Phase2Data,
    protocol: dict[str, Any],
    split_name: str,
    observed_keys: set[int],
    rng: np.random.Generator,
) -> PairSet:
    if split_name == "val":
        evaluation_indices = protocol["val_indices"]
    elif split_name == "test":
        evaluation_indices = protocol["test_indices"]
    else:
        raise ValueError(f"Unknown split name: {split_name}")
    if protocol["source_filter"] is not None:
        source_values = data.edge_df["source_dataset"].to_numpy()
        evaluation_indices = evaluation_indices[
            source_values[evaluation_indices] == protocol["source_filter"]
        ]
    if len(evaluation_indices) == 0:
        raise ValueError(f"Protocol {protocol['name']} has no {split_name} positives")
    source_index = data.edge_df["source_index"].to_numpy(dtype=np.int64)
    target_index = data.edge_df["target_index"].to_numpy(dtype=np.int64)
    positive_sources = source_index[evaluation_indices]
    positive_targets = target_index[evaluation_indices]
    negative_sources, negative_targets = sample_negative_pairs(
        rng,
        len(positive_sources),
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
    pair_ids = [str(value) for value in data.edge_df.iloc[evaluation_indices]["edge_id"].tolist()]
    pair_ids.extend(
        f"sampled_negative_{index:06d}" for index in range(len(negative_sources))
    )
    source_counts = _pair_source_counts(source_labels, labels)
    return PairSet(
        sources=sources,
        targets=targets,
        labels=labels,
        source_labels=source_labels,
        pair_ids=pair_ids,
        positive_count=int((labels == 1).sum()),
        negative_count=int((labels == 0).sum()),
        source_counts=source_counts,
    )


def build_evaluation_pairs(
    data: Phase2Data,
    protocols: dict[str, dict[str, Any]],
    seed: int = SEED,
) -> dict[str, dict[str, PairSet]]:
    observed_keys = all_observed_keys(data, include_reverse=True)
    result: dict[str, dict[str, PairSet]] = {}
    for offset, protocol_name in enumerate(("A", "B", "D")):
        protocol = protocols[protocol_name]
        rng = np.random.default_rng(seed + 1000 + offset)
        result[protocol_name] = {
            "val": build_pair_set(data, protocol, "val", observed_keys, rng),
            "test": build_pair_set(data, protocol, "test", observed_keys, rng),
        }
    result["C"] = {
        "val": copy.deepcopy(result["A"]["val"]),
        "test": copy.deepcopy(result["A"]["test"]),
    }
    return result


def build_structural_features(
    data: Phase2Data,
    protocol: dict[str, Any],
    scaler_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_indices = protocol["train_indices"]
    if len(train_indices) == 0:
        raise ValueError("Cannot build structural features without training edges")
    source_index = data.edge_index[0, train_indices].astype(np.int64)
    target_index = data.edge_index[1, train_indices].astype(np.int64)
    num_nodes = data.num_nodes
    in_degree = np.bincount(target_index, minlength=num_nodes).astype(np.float64)
    out_degree = np.bincount(source_index, minlength=num_nodes).astype(np.float64)
    directed_degree = in_degree + out_degree
    low = np.minimum(source_index, target_index)
    high = np.maximum(source_index, target_index)
    undirected_keys = np.unique(low * num_nodes + high)
    undirected_left = (undirected_keys // num_nodes).astype(np.int64)
    undirected_right = (undirected_keys % num_nodes).astype(np.int64)
    undirected_degree = np.bincount(
        np.concatenate([undirected_left, undirected_right]), minlength=num_nodes
    ).astype(np.float64)
    graph = coo_matrix(
        (
            np.ones(len(undirected_left), dtype=np.int8),
            (undirected_left, undirected_right),
        ),
        shape=(num_nodes, num_nodes),
    )
    component_count, component_labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(component_labels, minlength=component_count).astype(np.float64)
    component_size = component_sizes[component_labels]

    raw = np.column_stack(
        [
            in_degree,
            out_degree,
            undirected_degree,
            directed_degree,
            np.log1p(in_degree),
            np.log1p(out_degree),
            np.log1p(undirected_degree),
            np.log1p(directed_degree),
            np.log1p(component_size),
            (undirected_degree == 0).astype(np.float64),
        ]
    ).astype(np.float64)
    if scaler_mask is None:
        scaler_mask = data.node_train_mask
    if not scaler_mask.any():
        scaler_mask = np.ones(num_nodes, dtype=bool)
    scaler = StandardScaler()
    scaler.fit(raw[scaler_mask])
    scaled = scaler.transform(raw).astype(np.float32)
    metadata = {
        "feature_names": STRUCTURAL_FEATURE_NAMES,
        "source_graph_edges": int(len(train_indices)),
        "component_count": int(component_count),
        "isolated_train_nodes": int((undirected_degree == 0).sum()),
        "scaler_mask": "node_train_mask" if scaler_mask is data.node_train_mask else "custom",
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "leakage_control": (
            "Degrees and components are recomputed from the protocol training graph only. "
            "They are transductive features and are not valid for entirely unseen nodes."
        ),
    }
    return scaled, metadata


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    return float(
        -np.mean(
            labels * np.log(probabilities)
            + (1.0 - labels) * np.log(1.0 - probabilities)
        )
    )


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0 or k <= 0:
        return None
    k = min(int(k), len(labels))
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    accumulated_positive = 0.0
    accumulated_count = 0
    index = 0
    while index < len(sorted_scores) and accumulated_count < k:
        boundary = index + 1
        while boundary < len(sorted_scores) and sorted_scores[boundary] == sorted_scores[index]:
            boundary += 1
        group_count = boundary - index
        group_positive = float(sorted_labels[index:boundary].sum())
        remaining = k - accumulated_count
        if group_count <= remaining:
            accumulated_positive += group_positive
            accumulated_count += group_count
        else:
            accumulated_positive += remaining * (group_positive / group_count)
            accumulated_count = k
        index = boundary
    return float(accumulated_positive / k)


def _augment_metrics(
    metrics: dict[str, Any],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    metrics["precision_at_100"] = precision_at_k(labels, scores, 100)
    metrics["precision_at_500"] = precision_at_k(labels, scores, 500)
    metrics["threshold"] = float(threshold)
    return metrics


def _per_source_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    source_labels: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    result = metrics_by_source(labels, scores, source_labels, threshold)
    for source in sorted(result):
        mask = np.asarray(source_labels, dtype=object) == source
        result[source] = _augment_metrics(
            result[source], labels[mask], scores[mask], threshold
        )
    return result


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        probability = float(labels.mean())
        return {
            "method": "constant",
            "temperature": 1.0,
            "constant_probability": probability,
        }

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        return _log_loss(labels, sigmoid(logits / temperature))

    result = minimize_scalar(
        objective,
        bounds=(-10.0, 10.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    temperature = float(math.exp(float(result.x)))
    return {
        "method": "temperature_scaling",
        "temperature": temperature,
        "validation_log_loss": float(result.fun),
        "success": bool(result.success),
    }


def apply_temperature(logits: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    if calibration.get("method") == "constant":
        return np.full(len(logits), float(calibration["constant_probability"]))
    return sigmoid(np.asarray(logits, dtype=np.float64) / float(calibration["temperature"]))


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) < 2 or float(scores.std()) <= 1e-12:
        return {
            "method": "constant",
            "constant_probability": float(labels.mean()),
        }
    scaler = StandardScaler()
    scaled = scaler.fit_transform(scores)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000)
    model.fit(scaled, labels)
    return {
        "method": "platt_scaling",
        "scaler_mean": float(scaler.mean_[0]),
        "scaler_scale": float(scaler.scale_[0]),
        "coef": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def apply_platt(scores: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if calibration.get("method") == "constant":
        return np.full(len(scores), float(calibration["constant_probability"]))
    scaled = (scores - float(calibration["scaler_mean"])) / float(
        calibration["scaler_scale"]
    )
    logits = float(calibration["coef"]) * scaled + float(calibration["intercept"])
    return sigmoid(logits)


def calibration_report(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.clip(np.digitize(probabilities, bins) - 1, 0, 9)
    rows = []
    ece = 0.0
    for bin_index in range(10):
        mask = bin_indices == bin_index
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": bin_index,
                    "lower": float(bins[bin_index]),
                    "upper": float(bins[bin_index + 1]),
                    "count": 0,
                    "mean_confidence": None,
                    "empirical_positive_rate": None,
                    "gap": None,
                }
            )
            continue
        mean_confidence = float(probabilities[mask].mean())
        empirical = float(labels[mask].mean())
        gap = abs(empirical - mean_confidence)
        ece += (count / len(labels)) * gap
        rows.append(
            {
                "bin": bin_index,
                "lower": float(bins[bin_index]),
                "upper": float(bins[bin_index + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "empirical_positive_rate": empirical,
                "gap": gap,
            }
        )
    return {
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": _log_loss(labels, probabilities),
        "ece_10_bins": float(ece),
        "reliability": rows,
        "positive_count": int(labels.sum()),
        "negative_count": int((labels == 0).sum()),
        "clipping": "probabilities clipped to [1e-15, 1-1e-15] for log loss",
    }


def evaluate_predictions(
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    validation_source_labels: np.ndarray,
    validation_logits: np.ndarray | None,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    test_source_labels: np.ndarray,
    test_logits: np.ndarray | None,
    include_curves: bool = True,
) -> dict[str, Any]:
    threshold_info = threshold_from_validation(validation_labels, validation_scores)
    threshold = float(threshold_info["threshold"])
    validation = _augment_metrics(
        binary_metrics(validation_labels, validation_scores, threshold),
        validation_labels,
        validation_scores,
        threshold,
    )
    test = _augment_metrics(
        binary_metrics(test_labels, test_scores, threshold),
        test_labels,
        test_scores,
        threshold,
    )
    validation["per_source"] = _per_source_metrics(
        validation_labels, validation_scores, validation_source_labels, threshold
    )
    test["per_source"] = _per_source_metrics(
        test_labels, test_scores, test_source_labels, threshold
    )
    if include_curves:
        test["curves"] = curve_payload(test_labels, test_scores)

    if validation_logits is not None and test_logits is not None:
        calibration = fit_temperature(validation_logits, validation_labels)
        calibration["score_type"] = "logit_temperature_scaling"
        validation_probabilities = apply_temperature(validation_logits, calibration)
        test_probabilities = apply_temperature(test_logits, calibration)
    else:
        calibration = fit_platt(validation_scores, validation_labels)
        calibration["score_type"] = "score_platt_scaling"
        validation_probabilities = apply_platt(validation_scores, calibration)
        test_probabilities = apply_platt(test_scores, calibration)

    calibration["validation"] = calibration_report(
        validation_labels, validation_probabilities
    )
    calibration["test"] = calibration_report(test_labels, test_probabilities)
    calibration["ranking_unchanged"] = True
    return {
        "threshold_selection": {
            "source": "validation",
            "criterion": "maximum F1 among non-trivial thresholds",
            "selected_threshold": threshold,
            "validation_f1_at_selection": float(threshold_info["f1"]),
            "non_trivial_threshold": bool(threshold_info["non_trivial"]),
        },
        "validation": validation,
        "test": test,
        "calibration": calibration,
        "precision_at_k_definition": (
            "Expected precision at K with random tie-breaking inside the boundary score group."
        ),
    }


def build_structural_adjacency(
    data: Phase2Data, protocol: dict[str, Any]
) -> dict[str, Any]:
    train_indices = protocol["train_indices"]
    source_index = data.edge_index[0, train_indices].astype(np.int64)
    target_index = data.edge_index[1, train_indices].astype(np.int64)
    num_nodes = data.num_nodes
    out_adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
    in_adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
    undirected_adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
    for source, target in zip(source_index.tolist(), target_index.tolist()):
        out_adjacency[source].add(target)
        in_adjacency[target].add(source)
        undirected_adjacency[source].add(target)
        undirected_adjacency[target].add(source)
    out_degree = np.fromiter(
        (len(row) for row in out_adjacency), dtype=np.int64, count=num_nodes
    )
    in_degree = np.fromiter(
        (len(row) for row in in_adjacency), dtype=np.int64, count=num_nodes
    )
    undirected_degree = np.fromiter(
        (len(row) for row in undirected_adjacency), dtype=np.int64, count=num_nodes
    )
    return {
        "out_adjacency": out_adjacency,
        "in_adjacency": in_adjacency,
        "undirected_adjacency": undirected_adjacency,
        "out_degree": out_degree,
        "in_degree": in_degree,
        "undirected_degree": undirected_degree,
        "train_edge_count": int(len(train_indices)),
    }


def score_structural_baselines(
    adjacency: dict[str, Any], sources: np.ndarray, targets: np.ndarray
) -> dict[str, np.ndarray]:
    sources = np.asarray(sources, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    out_degree = adjacency["out_degree"]
    in_degree = adjacency["in_degree"]
    undirected_degree = adjacency["undirected_degree"]
    out_adjacency = adjacency["out_adjacency"]
    in_adjacency = adjacency["in_adjacency"]
    undirected_adjacency = adjacency["undirected_adjacency"]
    directed_pa = (
        out_degree[sources].astype(np.float64) * in_degree[targets].astype(np.float64)
    )
    undirected_pa = (
        undirected_degree[sources].astype(np.float64)
        * undirected_degree[targets].astype(np.float64)
    )
    directed_cn = np.fromiter(
        (
            len(out_adjacency[int(source)].intersection(in_adjacency[int(target)]))
            for source, target in zip(sources.tolist(), targets.tolist())
        ),
        dtype=np.float64,
        count=len(sources),
    )
    undirected_cn = np.fromiter(
        (
            len(
                undirected_adjacency[int(source)].intersection(
                    undirected_adjacency[int(target)]
                )
            )
            for source, target in zip(sources.tolist(), targets.tolist())
        ),
        dtype=np.float64,
        count=len(sources),
    )
    return {
        "directed_preferential_attachment": directed_pa,
        "undirected_preferential_attachment": undirected_pa,
        "directed_common_neighbors": directed_cn,
        "undirected_common_neighbors": undirected_cn,
    }


def predictions_frame(
    data: Phase2Data,
    pair_set: PairSet,
    scores: np.ndarray,
    logits: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "pair_id": pair_set.pair_ids,
            "source_index": pair_set.sources,
            "target_index": pair_set.targets,
            "source_company_id": [data.node_ids[int(index)] for index in pair_set.sources],
            "target_company_id": [data.node_ids[int(index)] for index in pair_set.targets],
            "source": pair_set.source_labels,
            "label": pair_set.labels,
            "score": np.asarray(scores, dtype=np.float64),
        }
    )
    if logits is not None:
        frame["logit"] = np.asarray(logits, dtype=np.float64)
    return frame


def compare_rows_from_metrics(
    metrics: dict[str, Any],
    protocol: str,
    model_type: str,
    model_name: str,
) -> list[dict[str, Any]]:
    rows = []

    def add_row(source: str, payload: dict[str, Any]) -> None:
        rows.append(
            {
                "protocol": protocol,
                "model_type": model_type,
                "model_name": model_name,
                "source": source,
                "roc_auc": payload.get("roc_auc"),
                "average_precision": payload.get("average_precision"),
                "precision_at_100": payload.get("precision_at_100"),
                "precision_at_500": payload.get("precision_at_500"),
                "precision": payload.get("precision"),
                "recall": payload.get("recall"),
                "f1": payload.get("f1"),
                "threshold": payload.get("threshold"),
                "positive_count": (payload.get("counts") or {}).get("positive"),
                "negative_count": (payload.get("counts") or {}).get("negative"),
                "brier": None,
                "log_loss": None,
                "ece_10_bins": None,
            }
        )

    add_row("pooled", metrics["test"])
    for source, payload in metrics["test"].get("per_source", {}).items():
        add_row(str(source), payload)
    calibration = metrics.get("calibration", {})
    test_calibration = calibration.get("test", {})
    rows[0]["brier"] = test_calibration.get("brier")
    rows[0]["log_loss"] = test_calibration.get("log_loss")
    rows[0]["ece_10_bins"] = test_calibration.get("ece_10_bins")
    return rows


def write_reliability_svg(path: Path, calibration: dict[str, Any], title: str) -> None:
    rows = calibration.get("test", {}).get("reliability", [])
    width = 520
    height = 380
    left = 64
    top = 36
    plot_width = 400
    plot_height = 280
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-family="Arial" font-size="14">{title}</text>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top}" stroke="#999" stroke-width="1" stroke-dasharray="4 4"/>',
        f'<text x="{left+plot_width/2}" y="{height-16}" text-anchor="middle" font-family="Arial" font-size="12">Mean predicted probability</text>',
        f'<text x="18" y="{top+plot_height/2}" text-anchor="middle" font-family="Arial" font-size="12" transform="rotate(-90 18 {top+plot_height/2})">Empirical positive rate</text>',
        f'<text x="{left-8}" y="{top+plot_height+4}" text-anchor="end" font-family="Arial" font-size="10">0</text>',
        f'<text x="{left-8}" y="{top+4}" text-anchor="end" font-family="Arial" font-size="10">1</text>',
        f'<text x="{left}" y="{top+plot_height+18}" text-anchor="middle" font-family="Arial" font-size="10">0</text>',
        f'<text x="{left+plot_width}" y="{top+plot_height+18}" text-anchor="middle" font-family="Arial" font-size="10">1</text>',
    ]
    for row in rows:
        if row["count"] == 0:
            continue
        x = left + float(row["mean_confidence"]) * plot_width
        y = top + (1.0 - float(row["empirical_positive_rate"])) * plot_height
        radius = 3.0 + 5.0 * math.sqrt(row["count"] / max(1, calibration["test"]["positive_count"] + calibration["test"]["negative_count"]))
        pieces.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="#1f77b4" fill-opacity="0.75" stroke="#0b3d66" stroke-width="0.8"/>'
        )
    pieces.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pieces) + "\n")
