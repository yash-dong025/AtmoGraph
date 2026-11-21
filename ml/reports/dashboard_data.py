"""Data access and analytics for the AtmoGraph dashboard.

All graph responses are computed from the existing processed CSV artifacts. The
module never invents nodes, edges, labels, or model metrics. GraphSAGE inference
is available only when a real checkpoint and metrics exist.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.inference.predict import build_link_prediction_record
from ml.reports.category_normalization import aggregate_categories, classify_category

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
TRAINING_DIR = DATASETS_DIR / "training"
PROCESSED_DIR = DATASETS_DIR / "processed"

NODES_PATH = PROCESSED_DIR / "graph" / "nodes.csv"
EDGES_PATH = PROCESSED_DIR / "graph" / "edges.csv"
EDGE_INDEX_PATH = TRAINING_DIR / "edge_index.csv"
NODE_INDEX_PATH = TRAINING_DIR / "node_index_mapping.csv"
NODE_FEATURES_PATH = PROCESSED_DIR / "graph" / "node_features.csv"

EXPERIMENT_DIR = PROJECT_ROOT / "ml" / "experiments" / "graphsage_link_prediction"
EXPERIMENT_PATH = EXPERIMENT_DIR / "experiment.json"
CONFIG_PATH = EXPERIMENT_DIR / "config.json"
BASELINE_METRICS_PATH = EXPERIMENT_DIR / "baseline_metrics.json"
BASELINE_PREDICTIONS_PATH = EXPERIMENT_DIR / "baseline_test_predictions.csv"
DATASET_METADATA_PATH = EXPERIMENT_DIR / "dataset_metadata.json"
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoint.pt"
HISTORY_PATH = EXPERIMENT_DIR / "history.json"
METRICS_PATH = EXPERIMENT_DIR / "metrics.json"
TEST_PREDICTIONS_PATH = EXPERIMENT_DIR / "test_predictions.csv"
RUN_SUMMARY_PATH = EXPERIMENT_DIR / "run_summary.json"

PHASE2_DIR = PROJECT_ROOT / "ml" / "experiments" / "graphsage_phase2"
PHASE2_SUMMARY_PATH = PHASE2_DIR / "summary.json"
EXPERIMENT_CATALOG_VERSION = 1
CATALOG_METRIC_FIELDS = (
    "roc_auc",
    "average_precision",
    "precision_at_100",
    "precision_at_500",
    "precision",
    "recall",
    "f1",
    "threshold",
    "positive_count",
    "negative_count",
    "brier",
    "log_loss",
    "ece_10_bins",
)

DEFAULT_SEED = "sampoq:ticker:TSM"
DEFAULT_MAX_NODES = 250
MAX_MAX_NODES = 500
MAX_HOPS = 3

SOURCE_LABELS = {
    "sampoq": "SAMPOQ named companies",
    "ic_splc": "IC-SPLC anonymised entities",
    "anonnet": "AnonNet anonymised entities",
}

RELATIONSHIP_LABELS = {
    "supplies_product_to": "supplies_product_to",
    "supply_relationship_anonymised": "supply_relationship_anonymised",
    "supplier_to_customer": "supplier_to_customer",
}


class PredictionUnavailable(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_text(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = str(value).strip()
    return text if text else None


def clean_int(value: Any, default: int = 0) -> int:
    if is_missing(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def clean_int_or_none(value: Any) -> int | None:
    if is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def artifact_record(path: Path, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _catalog_metrics(values: dict[str, Any]) -> dict[str, Any]:
    return {key: values.get(key) for key in CATALOG_METRIC_FIELDS}


def _original_catalog_entry(experiment_dir: Path) -> dict[str, Any] | None:
    experiment = read_json(experiment_dir / "experiment.json", {}) or {}
    config = read_json(experiment_dir / "config.json", {}) or {}
    metrics_payload = read_json(experiment_dir / "metrics.json", {}) or {}
    if not experiment and not config:
        return None
    experiment_id = str(experiment.get("experiment_id") or experiment_dir.name)
    test_metrics = metrics_payload.get("test", {}) if isinstance(metrics_payload, dict) else {}
    return {
        "experiment_id": experiment_id,
        "experiment_type": "original",
        "parent_experiment": None,
        "task": str(experiment.get("task") or config.get("task") or "directed_link_prediction"),
        "status": str(experiment.get("status") or config.get("status") or "unknown"),
        "model_type": str(config.get("model") or "GraphSAGE"),
        "model_name": str(config.get("model_class") or "AtmoGraphSAGE"),
        "model_version": experiment_id,
        "protocol": "original_random_split",
        "protocol_label": "Original 80/10/10 random edge split",
        "feature_config": "c1_current_136",
        "feature_config_label": "8 static flags plus 128 IC-SPLC embeddings",
        "dataset_source": "pooled",
        "dataset_version": experiment.get("dataset_version"),
        "metrics": _catalog_metrics(test_metrics if isinstance(test_metrics, dict) else {}),
        "source_artifact": _display_path(experiment_dir / "metrics.json"),
    }


def _phase2_model_type(model_type: str) -> str:
    return {
        "gnn": "GraphSAGE",
        "baseline": "structural_baseline",
        "original_artifact": "original_artifact",
        "original_checkpoint_reference": "original_checkpoint_reference",
    }.get(model_type, model_type)


def _phase2_catalog_entry(
    row: dict[str, Any],
    summary: dict[str, Any],
    summary_path: Path,
) -> dict[str, Any]:
    protocol = str(row.get("protocol", ""))
    source = str(row.get("source", ""))
    model_name = str(row.get("model_name", ""))
    raw_model_type = str(row.get("model_type", ""))
    configs = summary.get("configs", {}) if isinstance(summary.get("configs"), dict) else {}
    protocols = summary.get("protocols", {}) if isinstance(summary.get("protocols"), dict) else {}
    config = configs.get(model_name, {}) if isinstance(configs.get(model_name), dict) else {}
    protocol_payload = protocols.get(protocol, {}) if isinstance(protocols.get(protocol), dict) else {}
    original = summary.get("original_artifact", {})
    if raw_model_type == "gnn":
        feature_config = model_name
        model_version = str(summary.get("experiment") or "graphsage_phase2")
    elif raw_model_type in {"original_artifact", "original_checkpoint_reference"}:
        feature_config = "c1_current_136"
        model_version = "graphsage_link_prediction"
    else:
        feature_config = "not_applicable"
        model_version = "structural_baseline"
    return {
        "experiment_id": f"phase2:{protocol}:{source}:{model_name}",
        "experiment_type": "phase2",
        "parent_experiment": str(summary.get("experiment") or "graphsage_phase2"),
        "task": "directed_link_prediction",
        "status": str(summary.get("status") or "unknown"),
        "model_type": _phase2_model_type(raw_model_type),
        "model_name": model_name,
        "model_version": model_version,
        "protocol": protocol,
        "protocol_label": str(protocol_payload.get("label") or protocol),
        "feature_config": feature_config,
        "feature_config_label": str(
            config.get("label")
            or ("8 static flags plus 128 IC-SPLC embeddings" if feature_config == "c1_current_136" else "not applicable")
        ),
        "dataset_source": source,
        "dataset_version": original.get("dataset_version") if isinstance(original, dict) else None,
        "metrics": _catalog_metrics(row),
        "source_artifact": _display_path(summary_path),
    }


def build_experiment_catalog(
    original_experiment_dir: Path = EXPERIMENT_DIR,
    phase2_summary_path: Path = PHASE2_SUMMARY_PATH,
) -> dict[str, Any]:
    experiments: list[dict[str, Any]] = []
    original_entry = _original_catalog_entry(original_experiment_dir)
    if original_entry is not None:
        experiments.append(original_entry)
    summary = read_json(phase2_summary_path, {}) or {}
    rows = summary.get("comparison", []) if isinstance(summary, dict) else []
    if isinstance(rows, list):
        experiments.extend(
            _phase2_catalog_entry(row, summary, phase2_summary_path)
            for row in rows
            if isinstance(row, dict)
        )
    filter_fields = (
        "experiment_type",
        "protocol",
        "feature_config",
        "dataset_source",
        "model_type",
    )
    filters = {
        field: sorted(
            {
                str(experiment[field])
                for experiment in experiments
                if experiment.get(field) not in (None, "")
            }
        )
        for field in filter_fields
    }
    return {
        "catalog_version": EXPERIMENT_CATALOG_VERSION,
        "contract": {
            "purpose": "Backward-compatible experiment catalog for dashboard consumers.",
            "required_fields": list(experiments[0]) if experiments else [],
            "additive": True,
        },
        "phase2_available": bool(rows),
        "experiment_count": len(experiments),
        "experiments": experiments,
        "filters": filters,
        "source_artifacts": {
            "original": _display_path(original_experiment_dir),
            "phase2": _display_path(phase2_summary_path),
        },
        "warnings": [
            "The catalog is additive; existing experiment, GNN, and baseline fields are unchanged.",
            "Phase 2 rows are per-protocol, per-source, per-model comparison records.",
            "The original experiment is listed separately from Phase 2 experiments.",
            "No dashboard frontend consumes the catalog yet.",
        ],
    }


class DashboardData:
    def __init__(self) -> None:
        self._inference_lock = threading.Lock()
        self._inference: dict[str, Any] | None = None
        self._load_nodes()
        self._load_edges()
        self._build_adjacency()
        self._build_search_index()
        self.dataset_metadata = read_json(DATASET_METADATA_PATH, {}) or {}
        self.analytics = self._build_analytics()
        self.experiment_catalog = build_experiment_catalog()
        self.experiment = self._build_experiment()

    def _load_nodes(self) -> None:
        node_index = pd.read_csv(NODE_INDEX_PATH, dtype={"node_id": str})
        node_index = node_index.sort_values("node_index").reset_index(drop=True)
        if node_index["node_index"].tolist() != list(range(len(node_index))):
            raise ValueError("node_index_mapping.csv is not contiguous")

        nodes = pd.read_csv(
            NODES_PATH,
            dtype={
                "company_id": str,
                "company_name": str,
                "ticker": str,
                "country": str,
                "industry": str,
                "product_category": str,
                "source_dataset": str,
                "name_is_anonymised": "boolean",
                "node_type": str,
                "component_id": str,
            },
            low_memory=False,
        )
        nodes = nodes.set_index("company_id").reindex(node_index["node_id"])
        if nodes.index.has_duplicates:
            raise ValueError("nodes.csv contains duplicate company_id values")
        if nodes["source_dataset"].isna().any():
            raise ValueError("nodes.csv does not cover every node in node_index_mapping.csv")

        self.num_nodes = int(len(node_index))
        self.node_ids = node_index["node_id"].tolist()
        self.node_id_to_index = {node_id: index for index, node_id in enumerate(self.node_ids)}
        self.node_names = nodes["company_name"].fillna(nodes.index.to_series()).astype(str).tolist()
        self.node_sources = nodes["source_dataset"].fillna("unknown").astype(str).tolist()
        self.node_degrees = nodes["degree"].fillna(0).astype(int).to_numpy()
        self.node_in_degrees = nodes["in_degree"].fillna(0).astype(int).to_numpy()
        self.node_out_degrees = nodes["out_degree"].fillna(0).astype(int).to_numpy()
        self.node_raw_degrees = nodes["raw_degree"].fillna(0).astype(int).to_numpy()
        self.node_components = nodes["component_id"].fillna("unknown").astype(str).tolist()
        self.node_component_sizes = nodes["component_size"].fillna(0).astype(int).to_numpy()
        self.node_industries = nodes["industry"].map(clean_text).tolist()
        self.node_countries = nodes["country"].map(clean_text).tolist()
        self.node_tickers = nodes["ticker"].map(clean_text).tolist()
        self.node_categories = nodes["product_category"].map(clean_text).tolist()
        self.node_id_methods = nodes["id_method"].map(clean_text).tolist()
        self.node_mention_counts = nodes["mention_count"].fillna(0).astype(int).to_numpy()
        self.node_types = nodes["node_type"].fillna("unknown").astype(str).tolist()
        self.node_is_anonymised = nodes["name_is_anonymised"].fillna(False).astype(bool).tolist()
        self.node_embedding_available = (
            nodes["embedding_available"].fillna(0).astype(bool).tolist()
        )

    def _load_edges(self) -> None:
        edge_index = pd.read_csv(
            EDGE_INDEX_PATH,
            dtype={
                "edge_id": str,
                "source_company_id": str,
                "target_company_id": str,
                "relationship_type": str,
            },
            low_memory=False,
        )
        edges = pd.read_csv(
            EDGES_PATH,
            dtype={"edge_id": str, "source_company_id": str, "target_company_id": str},
            low_memory=False,
        )
        if edge_index["edge_id"].duplicated().any():
            raise ValueError("edge_index.csv contains duplicate edge_id values")
        attributes = edges[
            ["edge_id", "confidence", "first_year", "last_year", "source_datasets"]
        ]
        merged = edge_index.merge(attributes, on="edge_id", how="left", validate="one_to_one")

        self.edge_ids = merged["edge_id"].tolist()
        self.edge_src = merged["source_index"].to_numpy(dtype=np.int64)
        self.edge_dst = merged["target_index"].to_numpy(dtype=np.int64)
        self.edge_relationships = merged["relationship_type"].fillna("unknown").astype(str).tolist()
        self.edge_counts = merged["edge_count"].fillna(1).astype(int).to_numpy()
        self.edge_splits = merged["split"].fillna("unknown").astype(str).tolist()
        self.edge_confidences = merged["confidence"].tolist()
        self.edge_first_years = merged["first_year"].tolist()
        self.edge_last_years = merged["last_year"].tolist()
        self.edge_source_datasets = merged["source_datasets"].fillna("").astype(str).tolist()
        self.num_edges = int(len(merged))

    def _build_adjacency(self) -> None:
        self.out_edges: list[list[int]] = [[] for _ in range(self.num_nodes)]
        self.in_edges: list[list[int]] = [[] for _ in range(self.num_nodes)]
        for edge_index, (source, target) in enumerate(zip(self.edge_src.tolist(), self.edge_dst.tolist())):
            self.out_edges[source].append(edge_index)
            self.in_edges[target].append(edge_index)

    def _build_search_index(self) -> None:
        self.search_ids = [node_id.lower() for node_id in self.node_ids]
        self.search_names = [name.lower() for name in self.node_names]

    def node_summary(self, index: int) -> dict[str, Any]:
        return {
            "id": self.node_ids[index],
            "label": self.node_names[index],
            "source": self.node_sources[index],
            "degree": int(self.node_degrees[index]),
            "in_degree": int(self.node_in_degrees[index]),
            "out_degree": int(self.node_out_degrees[index]),
            "component_id": self.node_components[index],
            "component_size": int(self.node_component_sizes[index]),
            "is_anonymised": bool(self.node_is_anonymised[index]),
        }

    def node_payload(self, index: int) -> dict[str, Any]:
        payload = self.node_summary(index)
        source = self.node_sources[index]
        country = classify_category(self.node_countries[index], "country", source)
        industry = classify_category(self.node_industries[index], "industry", source)
        product_category = classify_category(
            self.node_categories[index], "product_category", source
        )
        payload.update(
            {
                "raw_degree": int(self.node_raw_degrees[index]),
                "industry": clean_text(self.node_industries[index]),
                "country": clean_text(self.node_countries[index]),
                "ticker": clean_text(self.node_tickers[index]),
                "product_category": clean_text(self.node_categories[index]),
                "country_display": country["normalized_value"],
                "country_category_type": country["category_type"],
                "industry_display": industry["normalized_value"],
                "industry_category_type": industry["category_type"],
                "product_category_display": product_category["normalized_value"],
                "product_category_type": product_category["category_type"],
                "id_method": clean_text(self.node_id_methods[index]),
                "mention_count": int(self.node_mention_counts[index]),
                "node_type": self.node_types[index],
                "embedding_available": bool(self.node_embedding_available[index]),
            }
        )
        return payload

    def edge_payload(self, index: int) -> dict[str, Any]:
        return {
            "id": self.edge_ids[index],
            "source": self.node_ids[int(self.edge_src[index])],
            "target": self.node_ids[int(self.edge_dst[index])],
            "relationship": self.edge_relationships[index],
            "confidence": clean_float(self.edge_confidences[index]),
            "edge_count": int(self.edge_counts[index]),
            "split": self.edge_splits[index],
            "source_datasets": self.edge_source_datasets[index],
            "first_year": clean_int_or_none(self.edge_first_years[index]),
            "last_year": clean_int_or_none(self.edge_last_years[index]),
        }

    def node_summary_by_id(self, node_id: str) -> dict[str, Any]:
        if node_id not in self.node_id_to_index:
            raise KeyError(f"Unknown node id: {node_id}")
        return self.node_summary(self.node_id_to_index[node_id])

    def search(self, query: str, limit: int = 25) -> list[dict[str, Any]]:
        query = query.strip().lower()
        if not query:
            return []
        limit = max(1, min(limit, 100))
        matches: list[int] = []
        for index, (node_id, name) in enumerate(zip(self.search_ids, self.search_names)):
            if query in node_id or query in name:
                matches.append(index)
        matches.sort(key=lambda index: (-int(self.node_degrees[index]), self.node_ids[index]))
        return [self.node_summary(index) for index in matches[:limit]]

    def subgraph(
        self,
        seed_id: str,
        hops: int,
        max_nodes: int,
        sources: list[str],
        relationships: list[str],
        split: str = "all",
        min_degree: int = 0,
    ) -> dict[str, Any]:
        if seed_id not in self.node_id_to_index:
            raise KeyError(f"Unknown node id: {seed_id}")
        hops = max(1, min(int(hops), MAX_HOPS))
        max_nodes = max(1, min(int(max_nodes), MAX_MAX_NODES))
        seed_index = self.node_id_to_index[seed_id]
        source_filter = set(sources)
        relationship_filter = set(relationships)
        split_filter = split if split in {"train", "val", "test"} else "all"
        min_degree_value = max(0, int(min_degree))

        def node_allowed(index: int) -> bool:
            return (
                (not source_filter or self.node_sources[index] in source_filter)
                and int(self.node_degrees[index]) >= min_degree_value
            )

        def edge_allowed(index: int) -> bool:
            return (
                (not relationship_filter or self.edge_relationships[index] in relationship_filter)
                and (split_filter == "all" or self.edge_splits[index] == split_filter)
                and node_allowed(int(self.edge_src[index]))
                and node_allowed(int(self.edge_dst[index]))
            )

        selected: dict[int, int] = {}
        frontier: list[int] = []
        truncated = False
        if node_allowed(seed_index):
            selected[seed_index] = 0
            frontier = [seed_index]

        for depth in range(1, hops + 1):
            candidates: set[int] = set()
            for node_index in frontier:
                for edge_index in self.out_edges[node_index]:
                    if edge_allowed(edge_index):
                        candidates.add(int(self.edge_dst[edge_index]))
                for edge_index in self.in_edges[node_index]:
                    if edge_allowed(edge_index):
                        candidates.add(int(self.edge_src[edge_index]))
            candidates.difference_update(selected)
            ordered = sorted(
                candidates,
                key=lambda index: (-int(self.node_degrees[index]), self.node_ids[index]),
            )
            remaining = max_nodes - len(selected)
            if remaining <= 0:
                truncated = True
                break
            if len(ordered) > remaining:
                truncated = True
            added = ordered[:remaining]
            for node_index in added:
                selected[node_index] = depth
            frontier = added
            if len(selected) >= max_nodes:
                truncated = len(ordered) > remaining
                break

        displayed_node_indices = list(selected)
        displayed_node_set = set(displayed_node_indices)
        displayed_edge_indices: list[int] = []
        for node_index in displayed_node_indices:
            for edge_index in self.out_edges[node_index]:
                if int(self.edge_dst[edge_index]) in displayed_node_set and edge_allowed(edge_index):
                    displayed_edge_indices.append(edge_index)

        nodes = []
        for index in sorted(displayed_node_indices, key=lambda item: (selected[item], self.node_ids[item])):
            payload = self.node_payload(index)
            payload["depth"] = selected[index]
            nodes.append(payload)

        return {
            "dataset_version": self.dataset_metadata.get("version"),
            "seed": self.node_payload(seed_index),
            "params": {
                "seed": seed_id,
                "hops": hops,
                "max_nodes": max_nodes,
                "sources": sorted(source_filter),
                "relationships": sorted(relationship_filter),
                "split": split_filter,
                "min_degree": min_degree_value,
            },
            "full_graph": {"nodes": self.num_nodes, "edges": self.num_edges},
            "neighborhood": {
                "nodes": len(selected),
                "edges": len(displayed_edge_indices),
                "max_nodes": max_nodes,
                "cap_truncated": bool(truncated),
            },
            "displayed": {
                "nodes": len(displayed_node_indices),
                "edges": len(displayed_edge_indices),
            },
            "seed_in_displayed": seed_index in displayed_node_set,
            "filters": {
                "sources": sorted(source_filter),
                "relationships": sorted(relationship_filter),
                "split": split_filter,
                "min_degree": min_degree_value,
            },
            "nodes": nodes,
            "edges": [self.edge_payload(index) for index in displayed_edge_indices],
            "notes": [
                "The canvas is a bounded, reproducible ego-network, not the full graph.",
                "Neighbors are added breadth-first, ordered by degree descending and then node id.",
                "Source, min-degree, relationship, and split filters are applied during traversal.",
                "Only nodes and edges matching every active filter are returned.",
            ],
        }

    def node_detail(
        self,
        node_id: str,
        sources: list[str] | None = None,
        relationships: list[str] | None = None,
        split: str = "all",
        min_degree: int = 0,
    ) -> dict[str, Any]:
        if node_id not in self.node_id_to_index:
            raise KeyError(f"Unknown node id: {node_id}")
        index = self.node_id_to_index[node_id]
        source_filter = set(sources or [])
        relationship_filter = set(relationships or [])
        split_filter = split if split in {"train", "val", "test"} else "all"
        min_degree_value = max(0, int(min_degree))

        def node_allowed(node_index: int) -> bool:
            return (
                (not source_filter or self.node_sources[node_index] in source_filter)
                and int(self.node_degrees[node_index]) >= min_degree_value
            )

        def edge_allowed(edge_index: int) -> bool:
            return (
                (not relationship_filter or self.edge_relationships[edge_index] in relationship_filter)
                and (split_filter == "all" or self.edge_splits[edge_index] == split_filter)
                and node_allowed(int(self.edge_src[edge_index]))
                and node_allowed(int(self.edge_dst[edge_index]))
            )

        incoming = []
        for edge_index in self.in_edges[index]:
            if not edge_allowed(edge_index):
                continue
            payload = self.edge_payload(edge_index)
            payload["direction"] = "incoming"
            payload["neighbor"] = self.node_summary(int(self.edge_src[edge_index]))
            incoming.append(payload)
        outgoing = []
        for edge_index in self.out_edges[index]:
            if not edge_allowed(edge_index):
                continue
            payload = self.edge_payload(edge_index)
            payload["direction"] = "outgoing"
            payload["neighbor"] = self.node_summary(int(self.edge_dst[edge_index]))
            outgoing.append(payload)
        return {
            "node": self.node_payload(index),
            "node_in_filtered_view": node_allowed(index),
            "incoming_count": len(incoming),
            "outgoing_count": len(outgoing),
            "full_incoming_count": len(self.in_edges[index]),
            "full_outgoing_count": len(self.out_edges[index]),
            "filters": {
                "sources": sorted(source_filter),
                "relationships": sorted(relationship_filter),
                "split": split_filter,
                "min_degree": min_degree_value,
            },
            "incoming": incoming,
            "outgoing": outgoing,
            "notes": [
                "Incoming/outgoing counts include only relationships matching the active explorer filters.",
                "full_incoming_count and full_outgoing_count include every incident edge in the processed graph.",
            ],
        }

    def _build_analytics(self) -> dict[str, Any]:
        degrees = self.node_degrees.astype(np.int64)
        component_frame = pd.DataFrame(
            {
                "component_id": self.node_components,
                "component_size": self.node_component_sizes,
                "source": self.node_sources,
            }
        ).drop_duplicates(subset=["component_id"])
        component_sizes = component_frame["component_size"].to_numpy(dtype=np.int64)

        degree_bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, 250, 500, np.inf]
        degree_labels = ["0", "1", "2", "3-4", "5-9", "10-19", "20-49", "50-99", "100-249", "250-499", "500+"]
        degree_counts, _ = np.histogram(degrees, bins=degree_bins)

        component_bins = [1, 2, 3, 6, 11, 51, 101, 501, 1001, 5001, np.inf]
        component_labels = ["1", "2", "3-5", "6-10", "11-50", "51-100", "101-500", "501-1000", "1001-5000", "5001+"]
        component_counts, _ = np.histogram(component_sizes, bins=component_bins)

        source_counts = pd.Series(self.node_sources).value_counts().to_dict()
        relationship_counts = pd.Series(self.edge_relationships).value_counts().to_dict()
        split_counts = pd.Series(self.edge_splits).value_counts().to_dict()
        node_type_counts = pd.Series(self.node_types).value_counts().to_dict()

        country_quality = aggregate_categories(
            self.node_countries,
            self.node_sources,
            "country",
            top_n=15,
        )
        industry_quality = aggregate_categories(
            self.node_industries,
            self.node_sources,
            "industry",
            top_n=15,
        )
        product_quality = aggregate_categories(
            self.node_categories,
            self.node_sources,
            "product_category",
            top_n=15,
        )

        top_degree_indices = np.argsort(-degrees, kind="stable")[:20]
        top_degree_nodes = [self.node_payload(int(index)) for index in top_degree_indices]

        top_component_ids = (
            component_frame.sort_values("component_size", ascending=False)["component_id"].head(12).tolist()
        )
        nodes_frame = pd.DataFrame(
            {
                "company_id": self.node_ids,
                "company_name": self.node_names,
                "source": self.node_sources,
                "degree": self.node_degrees,
                "component_id": self.node_components,
                "component_size": self.node_component_sizes,
            }
        )
        top_components = []
        for component_id in top_component_ids:
            subset = nodes_frame[nodes_frame["component_id"] == component_id]
            if subset.empty:
                continue
            source_mix = subset["source"].value_counts().to_dict()
            sample_nodes = subset.sort_values(
                ["degree", "company_id"], ascending=[False, True]
            ).head(8)
            top_components.append(
                {
                    "component_id": component_id,
                    "size": int(subset["component_size"].iloc[0]),
                    "source_mix": {str(key): int(value) for key, value in source_mix.items()},
                    "sample_nodes": [
                        {
                            "id": str(row["company_id"]),
                            "label": str(row["company_name"]),
                            "source": str(row["source"]),
                            "degree": int(row["degree"]),
                        }
                        for _, row in sample_nodes.iterrows()
                    ],
                }
            )

        embeddable = int(sum(1 for value in self.node_embedding_available if value))
        anonymised = int(sum(1 for value in self.node_is_anonymised if value))
        isolated = int(np.sum(degrees == 0))
        degree_stats = {
            "min": int(degrees.min()),
            "max": int(degrees.max()),
            "mean": float(degrees.mean()),
            "median": float(np.median(degrees)),
            "p90": float(np.percentile(degrees, 90)),
            "p99": float(np.percentile(degrees, 99)),
        }
        component_stats = {
            "count": int(len(component_sizes)),
            "largest": int(component_sizes.max()) if len(component_sizes) else 0,
            "singletons": int(np.sum(component_sizes == 1)),
            "isolated_nodes": isolated,
        }

        metadata = self.dataset_metadata or {}
        return {
            "generated_at": utc_now(),
            "dataset_version": metadata.get("version", "unknown"),
            "counts": {
                "nodes": self.num_nodes,
                "edges": self.num_edges,
                "node_feature_dim": metadata.get("node_feature_dim", 146),
                "edge_feature_dim": metadata.get("edge_feature_dim", 12),
                "edge_splits": {
                    key: int(split_counts.get(key, 0)) for key in ("train", "val", "test")
                },
                "node_splits": metadata.get("node_split_sizes", {}),
                "sources": {str(key): int(value) for key, value in source_counts.items()},
                "relationships": {
                    str(key): int(value) for key, value in relationship_counts.items()
                },
                "node_types": {str(key): int(value) for key, value in node_type_counts.items()},
                "anonymised_nodes": anonymised,
                "named_nodes": self.num_nodes - anonymised,
                "embedding_available": embeddable,
                "embedding_coverage": float(embeddable / self.num_nodes) if self.num_nodes else 0.0,
                "isolated_nodes": isolated,
            },
            "degree_histogram": {
                "labels": degree_labels,
                "counts": [int(value) for value in degree_counts],
            },
            "component_histogram": {
                "labels": component_labels,
                "counts": [int(value) for value in component_counts],
            },
            "degree_stats": degree_stats,
            "component_stats": component_stats,
            "top_degree_nodes": top_degree_nodes,
            "top_components": top_components,
            "top_countries": {
                str(entry["raw_value"]): int(entry["count"])
                for entry in country_quality["top_verified"]
            },
            "top_industries": {
                str(entry["raw_value"]): int(entry["count"])
                for entry in industry_quality["top_verified"]
            },
            "category_quality": {
                "country": country_quality,
                "industry": industry_quality,
                "product_category": product_quality,
            },
            "source_labels": SOURCE_LABELS,
            "relationship_labels": RELATIONSHIP_LABELS,
        }

    def _build_experiment(self) -> dict[str, Any]:
        experiment = read_json(EXPERIMENT_PATH, {}) or {}
        config = read_json(CONFIG_PATH, {}) or {}
        baseline = read_json(BASELINE_METRICS_PATH, None)
        dataset_metadata = read_json(DATASET_METADATA_PATH, {}) or {}
        metrics = read_json(METRICS_PATH, None)
        history = read_json(HISTORY_PATH, None)

        checkpoint_exists = CHECKPOINT_PATH.exists()
        metrics_exists = METRICS_PATH.exists()
        history_exists = HISTORY_PATH.exists()
        predictions_exists = TEST_PREDICTIONS_PATH.exists()
        run_summary_exists = RUN_SUMMARY_PATH.exists()

        gnn_available = bool(checkpoint_exists and metrics_exists)
        if gnn_available:
            gnn_status = "available"
            gnn_reason = "A trained checkpoint and GNN metrics are present."
        elif checkpoint_exists:
            gnn_status = "incomplete"
            gnn_reason = "A checkpoint exists but metrics.json is missing."
        else:
            gnn_status = "not_trained"
            gnn_reason = (
                "No trained GraphSAGE checkpoint exists. Training and evaluation must be "
                "shown as unavailable."
            )

        artifacts = [
            artifact_record(CONFIG_PATH, "config.json"),
            artifact_record(EXPERIMENT_PATH, "experiment.json"),
            artifact_record(DATASET_METADATA_PATH, "dataset_metadata.json"),
            artifact_record(BASELINE_METRICS_PATH, "baseline_metrics.json"),
            artifact_record(BASELINE_PREDICTIONS_PATH, "baseline_test_predictions.csv"),
            artifact_record(CHECKPOINT_PATH, "checkpoint.pt"),
            artifact_record(HISTORY_PATH, "history.json"),
            artifact_record(METRICS_PATH, "metrics.json"),
            artifact_record(TEST_PREDICTIONS_PATH, "test_predictions.csv"),
            artifact_record(RUN_SUMMARY_PATH, "run_summary.json"),
        ]

        return {
            "experiment": experiment,
            "config": config,
            "dataset_metadata": dataset_metadata,
            "baseline": baseline,
            "baseline_is_gnn": False,
            "baseline_task": "structural_link_prediction",
            "gnn": {
                "available": gnn_available,
                "status": gnn_status,
                "reason": gnn_reason,
                "checkpoint_exists": checkpoint_exists,
                "metrics_exists": metrics_exists,
                "history_exists": history_exists,
                "predictions_exists": predictions_exists,
                "run_summary_exists": run_summary_exists,
                "metrics": metrics if gnn_available else None,
                "history": history if gnn_available else None,
                "checkpoint_path": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
                "threshold": (
                    metrics.get("test", {}).get("threshold")
                    if metrics_exists and metrics
                    else None
                ),
            },
            "artifacts": artifacts,
            "catalog": self.experiment_catalog,
            "warnings": [
                "The structural baseline is not a GNN and is not a disruption-risk model.",
                "No verified company-level risk labels are available.",
                "Link prediction scores whether a supplier/customer edge is likely to exist.",
            ],
            "generated_at": utc_now(),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "generated_at": utc_now(),
            "dataset": {
                "version": self.dataset_metadata.get("version", "unknown"),
                "node_count": self.num_nodes,
                "edge_count": self.num_edges,
                "node_feature_dim": self.dataset_metadata.get("node_feature_dim", 146),
                "edge_feature_dim": self.dataset_metadata.get("edge_feature_dim", 12),
                "edge_splits": self.dataset_metadata.get("edge_split_sizes", {}),
                "node_splits": self.dataset_metadata.get("node_split_sizes", {}),
            },
            "experiment": {
                "status": self.experiment["experiment"].get("status", "unknown"),
                "checkpoint_exists": self.experiment["gnn"]["checkpoint_exists"],
                "gnn_available": self.experiment["gnn"]["available"],
                "gnn_reason": self.experiment["gnn"]["reason"],
                "baseline_available": self.experiment["baseline"] is not None,
                "catalog_version": self.experiment_catalog["catalog_version"],
                "catalog_experiments": self.experiment_catalog["experiment_count"],
                "phase2_catalog_available": self.experiment_catalog["phase2_available"],
            },
            "warnings": self.experiment["warnings"],
        }

    def _load_inference_model(self) -> dict[str, Any]:
        if not CHECKPOINT_PATH.exists():
            raise PredictionUnavailable(
                "No trained GraphSAGE checkpoint exists. Inference is disabled."
            )
        if not METRICS_PATH.exists():
            raise PredictionUnavailable(
                "A checkpoint exists but metrics.json is missing, so the run is incomplete."
            )
        if self._inference is not None:
            return self._inference
        with self._inference_lock:
            if self._inference is not None:
                return self._inference
            import torch

            from ml.models.gcn_model import build_gnn
            from ml.preprocessing.graph_preprocessing import build_pyg_data

            checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            feature_names = list(checkpoint["feature_names"])
            model_config = dict(checkpoint.get("model_config", {}))
            data = build_pyg_data()
            feature_lookup = {name: index for index, name in enumerate(data.feature_names)}
            missing = [name for name in feature_names if name not in feature_lookup]
            if missing:
                raise PredictionUnavailable(
                    f"Checkpoint feature names are not present in the current graph: {missing[:5]}"
                )
            feature_indices = torch.tensor(
                [feature_lookup[name] for name in feature_names], dtype=torch.long
            )
            model = build_gnn(
                "sage",
                in_channels=len(feature_names),
                hidden_channels=int(model_config.get("hidden_channels", 64)),
                out_channels=int(model_config.get("out_channels", 32)),
                dropout=float(model_config.get("dropout", 0.2)),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self._inference = {
                "model": model,
                "x": data.x[:, feature_indices].contiguous(),
                "message_edge_index": data.edge_index[:, data.train_edge_mask].contiguous(),
                "checkpoint": checkpoint,
                "model_config": model_config,
            }
            return self._inference

    def predict_pair(self, source_id: str, target_id: str) -> dict[str, Any]:
        if source_id not in self.node_id_to_index:
            raise KeyError(f"Unknown source node id: {source_id}")
        if target_id not in self.node_id_to_index:
            raise KeyError(f"Unknown target node id: {target_id}")
        if source_id == target_id:
            raise ValueError("Source and target must be different nodes")
        inference = self._load_inference_model()
        import torch

        source_index = self.node_id_to_index[source_id]
        target_index = self.node_id_to_index[target_id]
        with torch.no_grad():
            embeddings = inference["model"].encode(
                inference["x"], inference["message_edge_index"]
            )
            pair_index = torch.tensor(
                [[source_index], [target_index]], dtype=torch.long
            )
            logit = inference["model"].decode(embeddings, pair_index)
            score = torch.sigmoid(logit)
        threshold = inference["checkpoint"].get("threshold")
        score_value = float(score.item())
        logit_value = float(logit.item())
        source_payload = self.node_payload(source_index)
        target_payload = self.node_payload(target_index)
        model_payload = {
            "name": "GraphSAGE",
            "model_class": "AtmoGraphSAGE",
            "checkpoint_path": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "dataset_version": inference["checkpoint"].get("dataset_version"),
            "seed": inference["checkpoint"].get("seed"),
            "best_epoch": inference["checkpoint"].get("best_epoch"),
            "feature_count": len(inference["checkpoint"].get("feature_names", [])),
        }
        link_record = build_link_prediction_record(
            source_id=source_id,
            target_id=target_id,
            link_score=score_value,
            source=source_payload,
            target=target_payload,
            model_name="GraphSAGE",
            model_version="graphsage_link_prediction",
            model_type="AtmoGraphSAGE",
            dataset_version=inference["checkpoint"].get("dataset_version"),
            protocol="original_random_split",
            feature_config="c1_current_136",
            checkpoint_path=str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        )
        return {
            "available": True,
            "source": source_payload,
            "target": target_payload,
            "score": score_value,
            "link_score": score_value,
            "task": link_record["task"],
            "score_type": link_record["score_type"],
            "probability_calibrated": False,
            "logit": logit_value,
            "threshold": float(threshold) if threshold is not None else None,
            "above_threshold": bool(threshold is not None and score_value >= float(threshold)),
            "model": model_payload,
            "link_prediction_record": link_record,
            "disclaimer": (
                "This is a link-prediction score for whether a directed supplier/customer "
                "edge is likely to exist. It is not a disruption-risk score."
            ),
        }
