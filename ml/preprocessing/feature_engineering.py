"""Feature engineering for the AtmoGraph supply-chain graph.

Design goals
------------
- Consume the existing processed graph files without modifying them.
- Preserve node and edge identifiers outside the model feature matrices.
- Avoid target leakage.
- Fit scalers only on the training split from ``datasets/training/dataset_splits.json``.
- Keep categorical codes separate so a GNN can learn embeddings for them.
- Zero-fill missing IC-SPLC embeddings and expose an explicit mask.

The graph is transductive in this project: all nodes are known at inference time.
Fitting the scaler on training nodes is therefore a leakage-control choice, not an
inductive-generalisation claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
PROCESSED_DIR = DATASETS_DIR / "processed"
TRAINING_DIR = DATASETS_DIR / "training"

NODE_FEATURES_PATH = PROCESSED_DIR / "graph" / "node_features.csv"
NODE_INDEX_PATH = TRAINING_DIR / "node_index_mapping.csv"
NODE_EMBEDDINGS_PATH = TRAINING_DIR / "node_embeddings.parquet"
GRAPH_EDGES_PATH = PROCESSED_DIR / "graph" / "edges.csv"
EDGE_INDEX_PATH = TRAINING_DIR / "edge_index.csv"
SPLITS_PATH = TRAINING_DIR / "dataset_splits.json"

REFERENCE_YEAR = 2026

ID_COLUMNS = ["node_id", "node_type", "component_id"]
CATEGORICAL_COLUMNS = ["country_code", "industry_code", "product_category_code"]
BINARY_COLUMNS = [
    "is_isolated",
    "source_sampoq",
    "source_ic_splc",
    "source_anonnet",
    "has_country",
    "has_industry",
    "has_product_category",
    "embedding_available",
    "is_anonymised",
]
LOG_CONTINUOUS_COLUMNS = [
    "degree",
    "in_degree",
    "out_degree",
    "raw_degree",
    "component_size",
]
DERIVED_CONTINUOUS_COLUMNS = [
    "first_edge_age",
    "last_edge_age",
    "edge_year_span",
]
RELATIONSHIP_TYPES = [
    "supplies_product_to",
    "supply_relationship_anonymised",
    "supplier_to_customer",
]
SOURCE_DATASETS = ["sampoq", "ic_splc", "anonnet"]


@dataclass
class NodeFeatureBundle:
    node_ids: list[str]
    x: np.ndarray
    x_num: np.ndarray
    x_cat: np.ndarray
    x_emb: np.ndarray
    emb_mask: np.ndarray
    feature_names: list[str]
    cat_cardinalities: list[int]
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    scaler: StandardScaler
    medians: dict[str, float]


@dataclass
class EdgeFeatureBundle:
    edge_ids: list[str]
    edge_index: np.ndarray
    edge_attr: np.ndarray
    feature_names: list[str]
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    scaler: StandardScaler
    medians: dict[str, float]


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_node_index(path: Path = NODE_INDEX_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"node_id": str})
    if frame["node_id"].duplicated().any():
        raise ValueError("node_index_mapping.csv contains duplicate node_id values")
    frame["node_index"] = pd.to_numeric(frame["node_index"], errors="raise").astype(int)
    frame = frame.sort_values("node_index").reset_index(drop=True)
    if frame["node_index"].tolist() != list(range(len(frame))):
        raise ValueError("node_index_mapping.csv is not contiguous from 0..N-1")
    return frame


def load_node_frame(path: Path = NODE_FEATURES_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(ID_COLUMNS + CATEGORICAL_COLUMNS + BINARY_COLUMNS + LOG_CONTINUOUS_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"node_features.csv is missing required columns: {missing}")
    return frame


def load_embeddings(
    node_ids: list[str], path: Path = NODE_EMBEDDINGS_PATH
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(path)
    if "node_id" not in frame.columns or "embedding_available" not in frame.columns:
        raise ValueError("node_embeddings.parquet must contain node_id and embedding_available")
    emb_cols = [c for c in frame.columns if c.startswith("emb_")]
    if len(emb_cols) != 128:
        raise ValueError(f"Expected 128 embedding dimensions, found {len(emb_cols)}")
    frame = frame.set_index("node_id")
    frame = frame.reindex(node_ids)
    values = frame[emb_cols].fillna(0.0).to_numpy(dtype=np.float32)
    mask = frame["embedding_available"].fillna(False).astype(bool).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Embedding matrix contains non-finite values")
    return values, mask


def split_masks(
    node_ids: list[str], splits_path: Path = SPLITS_PATH
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits = _read_json(splits_path).get("node_splits", {})
    masks = []
    for split_name in ("train", "val", "test"):
        split_ids = set(splits.get(split_name, []))
        masks.append(np.array([node_id in split_ids for node_id in node_ids], dtype=bool))
    if not any(mask.any() for mask in masks):
        raise ValueError("No node split IDs matched the graph node IDs")
    return tuple(masks)  # type: ignore[return-value]


def _median_fill_and_scale(
    frame: pd.DataFrame, train_mask: np.ndarray
) -> tuple[np.ndarray, StandardScaler, dict[str, float]]:
    medians: dict[str, float] = {}
    filled = frame.copy()
    for col in frame.columns:
        train_values = frame.loc[train_mask, col]
        median = float(train_values.median()) if train_values.notna().any() else 0.0
        if not np.isfinite(median):
            median = 0.0
        medians[col] = median
        filled[col] = frame[col].fillna(median)
    scaler = StandardScaler()
    if train_mask.any():
        scaler.fit(filled.loc[train_mask])
    else:
        scaler.fit(filled)
    values = scaler.transform(filled).astype(np.float32)
    return values, scaler, medians


def build_node_features(
    node_features_path: Path = NODE_FEATURES_PATH,
    node_index_path: Path = NODE_INDEX_PATH,
    embeddings_path: Path = NODE_EMBEDDINGS_PATH,
    splits_path: Path = SPLITS_PATH,
) -> NodeFeatureBundle:
    node_index = load_node_index(node_index_path)
    node_ids = node_index["node_id"].tolist()

    frame = load_node_frame(node_features_path)
    if frame["node_id"].duplicated().any():
        raise ValueError("node_features.csv contains duplicate node_id values")
    frame = frame.set_index("node_id").reindex(node_ids)
    if frame.isna().all(axis=1).any():
        raise ValueError("node_features.csv does not cover every node in node_index_mapping.csv")

    has_edge_year = frame["first_edge_year"].notna() & frame["last_edge_year"].notna()
    frame["has_edge_year"] = has_edge_year.astype(int)
    frame["first_edge_age"] = REFERENCE_YEAR - frame["first_edge_year"]
    frame["last_edge_age"] = REFERENCE_YEAR - frame["last_edge_year"]
    frame["edge_year_span"] = (frame["last_edge_year"] - frame["first_edge_year"]).clip(lower=0)

    continuous = pd.DataFrame(index=frame.index)
    for col in LOG_CONTINUOUS_COLUMNS:
        continuous[f"log1p_{col}"] = np.log1p(frame[col].clip(lower=0).astype(float))
    for col in DERIVED_CONTINUOUS_COLUMNS:
        continuous[col] = frame[col].astype(float)

    train_mask, val_mask, test_mask = split_masks(node_ids, splits_path)
    continuous_values, scaler, medians = _median_fill_and_scale(continuous, train_mask)

    binary = frame[BINARY_COLUMNS + ["has_edge_year"]].fillna(0).astype(np.float32)
    binary_values = binary.to_numpy(dtype=np.float32)

    x_num = np.concatenate([continuous_values, binary_values], axis=1).astype(np.float32)
    num_feature_names = list(continuous.columns) + list(binary.columns)

    cat_raw = frame[CATEGORICAL_COLUMNS].fillna(-1).astype(np.int64)
    x_cat = (cat_raw + 1).to_numpy(dtype=np.int64)
    cat_cardinalities = [int(x_cat[:, i].max()) + 1 for i in range(x_cat.shape[1])]

    x_emb, emb_mask = load_embeddings(node_ids, embeddings_path)
    x = np.concatenate([x_num, x_emb], axis=1).astype(np.float32)
    feature_names = num_feature_names + [f"icsplc_emb_{i}" for i in range(x_emb.shape[1])]

    if not np.isfinite(x).all():
        raise ValueError("Node feature matrix contains non-finite values")

    return NodeFeatureBundle(
        node_ids=node_ids,
        x=x,
        x_num=x_num,
        x_cat=x_cat,
        x_emb=x_emb,
        emb_mask=emb_mask,
        feature_names=feature_names,
        cat_cardinalities=cat_cardinalities,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        scaler=scaler,
        medians=medians,
    )


def build_edge_features(
    edges_path: Path = GRAPH_EDGES_PATH,
    edge_index_path: Path = EDGE_INDEX_PATH,
) -> EdgeFeatureBundle:
    edges = pd.read_csv(edges_path, dtype={"edge_id": str})
    edge_index = pd.read_csv(edge_index_path, dtype={"edge_id": str})
    if edges["edge_id"].duplicated().any():
        raise ValueError("edges.csv contains duplicate edge_id values")
    if edge_index["edge_id"].duplicated().any():
        raise ValueError("edge_index.csv contains duplicate edge_id values")

    merged = edge_index.merge(
        edges,
        on="edge_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_edge"),
    )
    if merged["source_company_id"].isna().any():
        raise ValueError("edge_index.csv contains edge_id values missing from edges.csv")

    edge_ids = merged["edge_id"].astype(str).tolist()
    source_index = pd.to_numeric(merged["source_index"], errors="raise").astype(np.int64)
    target_index = pd.to_numeric(merged["target_index"], errors="raise").astype(np.int64)
    edge_index_array = np.vstack([source_index.to_numpy(), target_index.to_numpy()])

    first_year = pd.to_numeric(merged["first_year"], errors="coerce")
    last_year = pd.to_numeric(merged["last_year"], errors="coerce")
    has_year = (first_year.notna() & last_year.notna()).astype(int)
    year_span = (last_year - first_year).clip(lower=0)

    continuous = pd.DataFrame(
        {
            "log1p_edge_count": np.log1p(merged["edge_count"].clip(lower=0).astype(float)),
            "confidence": merged["confidence"].astype(float),
            "first_year": first_year,
            "last_year": last_year,
            "edge_year_span": year_span,
        },
        index=merged.index,
    )

    train_mask = merged["split"].eq("train").to_numpy(dtype=bool)
    val_mask = merged["split"].eq("val").to_numpy(dtype=bool)
    test_mask = merged["split"].eq("test").to_numpy(dtype=bool)
    if not (train_mask | val_mask | test_mask).all():
        raise ValueError("edge_index.csv contains an unrecognised split value")

    continuous_values, scaler, medians = _median_fill_and_scale(continuous, train_mask)

    relationship = pd.Categorical(merged["relationship_type"], categories=RELATIONSHIP_TYPES)
    relationship_onehot = pd.get_dummies(relationship, prefix="rel", dtype=np.float32)
    relationship_onehot = relationship_onehot.reindex(
        columns=[f"rel_{value}" for value in RELATIONSHIP_TYPES], fill_value=0.0
    )

    source_values = merged["source_datasets"].fillna("").astype(str)
    source_onehot = pd.DataFrame(
        {
            f"src_{source}": source_values.map(
                lambda value, source=source: float(source in value.split(";"))
            )
            for source in SOURCE_DATASETS
        },
        index=merged.index,
    )

    binary = pd.concat(
        [has_year.rename("has_edge_year").astype(np.float32), relationship_onehot, source_onehot],
        axis=1,
    )
    edge_attr = np.concatenate(
        [continuous_values, binary.to_numpy(dtype=np.float32)], axis=1
    ).astype(np.float32)
    feature_names = list(continuous.columns) + list(binary.columns)

    if not np.isfinite(edge_attr).all():
        raise ValueError("Edge feature matrix contains non-finite values")

    return EdgeFeatureBundle(
        edge_ids=edge_ids,
        edge_index=edge_index_array,
        edge_attr=edge_attr,
        feature_names=feature_names,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        scaler=scaler,
        medians=medians,
    )


def feature_summary() -> dict[str, Any]:
    """Return a compact machine-readable feature summary without building tensors."""
    nodes = build_node_features()
    edges = build_edge_features()
    return {
        "node_count": len(nodes.node_ids),
        "node_feature_dim": int(nodes.x.shape[1]),
        "node_numeric_dim": int(nodes.x_num.shape[1]),
        "node_embedding_dim": int(nodes.x_emb.shape[1]),
        "node_categorical_dim": int(nodes.x_cat.shape[1]),
        "node_categorical_cardinalities": nodes.cat_cardinalities,
        "node_embedding_available": int(nodes.emb_mask.sum()),
        "node_feature_names": nodes.feature_names,
        "node_splits": {
            "train": int(nodes.train_mask.sum()),
            "val": int(nodes.val_mask.sum()),
            "test": int(nodes.test_mask.sum()),
        },
        "edge_count": len(edges.edge_ids),
        "edge_feature_dim": int(edges.edge_attr.shape[1]),
        "edge_feature_names": edges.feature_names,
        "edge_splits": {
            "train": int(edges.train_mask.sum()),
            "val": int(edges.val_mask.sum()),
            "test": int(edges.test_mask.sum()),
        },
    }
