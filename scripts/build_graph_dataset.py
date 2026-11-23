from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from common import (
    PROCESSED_DIR,
    RAW_DIR,
    TRAINING_DIR,
    clean_text,
    ensure_dirs,
    read_json,
    setup_logging,
    stable_id,
    utc_now,
    write_csv,
    write_json,
)

COMPANY_FILES = [
    PROCESSED_DIR / "companies" / "companies_sampoq.csv",
    PROCESSED_DIR / "companies" / "companies_ic_splc.csv",
    PROCESSED_DIR / "companies" / "companies_anonnet.csv",
]
EDGE_FILES = [
    PROCESSED_DIR / "relationships" / "supply_chain_edges_sampoq.csv",
    PROCESSED_DIR / "relationships" / "supply_chain_edges_ic_splc.csv",
    PROCESSED_DIR / "relationships" / "supply_chain_edges_anonnet.csv",
]
EMBEDDINGS = RAW_DIR / "ic_splc" / "extracted_csv" / "ic_splc_embeddings.parquet"
DELAY_LABELS = TRAINING_DIR / "delay_labels.csv"

OUT_COMPANIES = PROCESSED_DIR / "companies" / "companies.csv"
OUT_EDGES = PROCESSED_DIR / "relationships" / "supply_chain_edges.csv"
OUT_EDGES_PARQUET = PROCESSED_DIR / "relationships" / "supply_chain_edges.parquet"
GRAPH_DIR = PROCESSED_DIR / "graph"
OUT_NODES = GRAPH_DIR / "nodes.csv"
OUT_GRAPH_EDGES = GRAPH_DIR / "edges.csv"
OUT_NODE_FEATURES = GRAPH_DIR / "node_features.csv"
OUT_TRAIN_NODE_FEATURES = TRAINING_DIR / "node_features.csv"
OUT_GRAPH_META = GRAPH_DIR / "graph_metadata.json"

OUT_NODE_INDEX = TRAINING_DIR / "node_index_mapping.csv"
OUT_EDGE_INDEX = TRAINING_DIR / "edge_index.csv"
OUT_RISK = TRAINING_DIR / "risk_labels.csv"
OUT_EMBEDDINGS = TRAINING_DIR / "node_embeddings.parquet"
OUT_SPLITS = TRAINING_DIR / "dataset_splits.json"
OUT_TRAIN_META = TRAINING_DIR / "training_metadata.json"
OUT_CATEGORIES = TRAINING_DIR / "category_mappings.json"

SEED = 42
STANDARD_COMPANY = [
    "company_id",
    "company_name",
    "original_company_name",
    "ticker",
    "country",
    "industry",
    "product_category",
    "source_dataset",
    "source_record_id",
    "name_is_anonymised",
    "id_method",
]
STANDARD_EDGE = [
    "edge_id",
    "source_company_id",
    "target_company_id",
    "relationship_type",
    "product_category",
    "year",
    "filing_date",
    "source_dataset",
    "source_record_id",
    "confidence",
    "relationship_direction",
]


def first_nonempty(series: pd.Series) -> str:
    for value in series:
        text = clean_text(value)
        if text:
            return text
    return ""


def join_unique(series: pd.Series) -> str:
    return ";".join(sorted({clean_text(v) for v in series if clean_text(v)}))


def min_year(series: pd.Series) -> Any:
    values = [v for v in series.dropna().tolist() if v != ""]
    return int(min(values)) if values else ""


def max_year(series: pd.Series) -> Any:
    values = [v for v in series.dropna().tolist() if v != ""]
    return int(max(values)) if values else ""


def load_frames(paths: Iterable[Path], standard: list[str], key: str) -> pd.DataFrame:
    frames = []
    extra_order: list[str] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Missing processed file: {path}")
        df = pd.read_csv(path, dtype=str).fillna("")
        for col in df.columns:
            if col not in standard and col not in extra_order:
                extra_order.append(col)
        frames.append(df)
    all_cols = standard + extra_order
    combined = pd.concat([df.reindex(columns=all_cols, fill_value="") for df in frames], ignore_index=True)
    if combined.duplicated(key).any():
        combined = combined.groupby(key, sort=False, as_index=False).agg(first_nonempty)
    return combined[all_cols]


def code_map(values: pd.Series) -> dict[str, int]:
    unique = sorted({clean_text(v) for v in values if clean_text(v)})
    return {v: i for i, v in enumerate(unique)}


def split_counts(n: int) -> tuple[int, int, int]:
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    n_test = n - n_train - n_val
    if n >= 3:
        n_train = max(n_train, 1)
        n_val = max(n_val, 1)
        n_test = max(n_test, 1)
        if n_train + n_val + n_test > n:
            n_train = n - 2
            n_val = 1
            n_test = 1
    return n_train, n_val, n_test


def stratified_ids(df: pd.DataFrame, id_col: str, strat_col: str | None, rng: np.random.Generator) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    groups = [("__all__", df)] if not strat_col else list(df.groupby(strat_col, dropna=False, sort=True))
    for _, group in groups:
        ids = sorted(group[id_col].astype(str).tolist())
        rng.shuffle(ids)
        n_train, n_val, _ = split_counts(len(ids))
        out["train"].extend(ids[:n_train])
        out["val"].extend(ids[n_train : n_train + n_val])
        out["test"].extend(ids[n_train + n_val :])
    for key in out:
        out[key] = sorted(out[key])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build combined graph and training datasets")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    ensure_dirs()
    logger = setup_logging("build_graph_dataset")
    rng = np.random.default_rng(args.seed)

    companies = load_frames(COMPANY_FILES, STANDARD_COMPANY, "company_id")
    source_edge_rows = sum(len(pd.read_csv(path, usecols=["edge_id"])) for path in EDGE_FILES)
    edges = load_frames(EDGE_FILES, STANDARD_EDGE, "edge_id")
    duplicate_edge_ids = source_edge_rows - len(edges)
    logger.info("Loaded %d companies and %d edges (%d duplicate edge_id rows merged)", len(companies), len(edges), duplicate_edge_ids)

    write_csv(OUT_COMPANIES, companies.to_dict("records"), list(companies.columns))
    edges.to_parquet(OUT_EDGES_PARQUET, index=False, compression="zstd")
    write_csv(OUT_EDGES, edges.to_dict("records"), list(edges.columns))

    node_set = set(companies["company_id"])
    missing_endpoint = ~edges["source_company_id"].isin(node_set) | ~edges["target_company_id"].isin(node_set)
    n_missing_endpoint = int(missing_endpoint.sum())
    edges = edges.loc[~missing_endpoint].copy()
    self_loop = edges["source_company_id"] == edges["target_company_id"]
    n_self_loops = int(self_loop.sum())
    edges = edges.loc[~self_loop].copy()
    edges["year_num"] = pd.to_numeric(edges["year"], errors="coerce")

    dup_key = ["source_company_id", "target_company_id", "relationship_type"]
    duplicate_groups = int(edges.duplicated(subset=dup_key).sum())
    graph_edges = (
        edges.groupby(dup_key, as_index=False, sort=False)
        .agg(
            edge_count=("edge_id", "size"),
            source_datasets=("source_dataset", join_unique),
            first_year=("year_num", min_year),
            last_year=("year_num", max_year),
            confidence=("confidence", "max"),
        )
        .reset_index(drop=True)
    )
    graph_edges["edge_id"] = [
        stable_id("graph_edge", f"{s}|{t}|{r}")
        for s, t, r in zip(graph_edges["source_company_id"], graph_edges["target_company_id"], graph_edges["relationship_type"])
    ]
    graph_edges = graph_edges[
        [
            "edge_id",
            "source_company_id",
            "target_company_id",
            "relationship_type",
            "edge_count",
            "source_datasets",
            "first_year",
            "last_year",
            "confidence",
        ]
    ].sort_values("edge_id").reset_index(drop=True)
    logger.info("Graph edges after deduplication: %d (duplicate groups %d)", len(graph_edges), duplicate_groups)

    out_deg = graph_edges.groupby("source_company_id").size()
    in_deg = graph_edges.groupby("target_company_id").size()
    raw_out = edges.groupby("source_company_id").size()
    raw_in = edges.groupby("target_company_id").size()
    first_src = graph_edges.groupby("source_company_id")["first_year"].apply(min_year)
    first_tgt = graph_edges.groupby("target_company_id")["first_year"].apply(min_year)
    last_src = graph_edges.groupby("source_company_id")["last_year"].apply(max_year)
    last_tgt = graph_edges.groupby("target_company_id")["last_year"].apply(max_year)

    def year_combine(a: pd.Series, b: pd.Series, fn) -> dict[str, Any]:
        values: dict[str, list[Any]] = defaultdict(list)
        for mapping in (a, b):
            for k, v in mapping.items():
                if v != "" and v is not None:
                    values[k].append(v)
        return {k: fn(v) for k, v in values.items()}

    first_edge_year = year_combine(first_src, first_tgt, min)
    last_edge_year = year_combine(last_src, last_tgt, max)

    adj: dict[str, list[str]] = defaultdict(list)
    for s, t in zip(graph_edges["source_company_id"], graph_edges["target_company_id"]):
        adj[s].append(t)
        adj[t].append(s)
    component: dict[str, int] = {}
    component_sizes: list[int] = []
    for node in sorted(companies["company_id"]):
        if node in component:
            continue
        cid = len(component_sizes)
        stack = [node]
        component[node] = cid
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in adj.get(u, []):
                if v not in component:
                    component[v] = cid
                    stack.append(v)
        component_sizes.append(size)

    companies = companies.copy()
    companies["node_type"] = "company"
    companies["degree"] = [int(in_deg.get(n, 0) + out_deg.get(n, 0)) for n in companies["company_id"]]
    companies["in_degree"] = [int(in_deg.get(n, 0)) for n in companies["company_id"]]
    companies["out_degree"] = [int(out_deg.get(n, 0)) for n in companies["company_id"]]
    companies["raw_degree"] = [int(raw_in.get(n, 0) + raw_out.get(n, 0)) for n in companies["company_id"]]
    companies["component_id"] = [component[n] for n in companies["company_id"]]
    companies["component_size"] = [component_sizes[component[n]] for n in companies["company_id"]]

    write_csv(OUT_NODES, companies.to_dict("records"), list(companies.columns))
    write_csv(OUT_GRAPH_EDGES, graph_edges.to_dict("records"), list(graph_edges.columns))

    country_map = code_map(companies["country"])
    industry_map = code_map(companies["industry"])
    product_map = code_map(companies["product_category"])
    categories = {
        "country": country_map,
        "industry": industry_map,
        "product_category": product_map,
        "node_type": {"company": 0},
        "notes": "Missing values are encoded as -1. These are categorical encodings, not learned embeddings.",
    }
    write_json(OUT_CATEGORIES, categories)

    emb_available: set[str] = set()
    emb_dim = 128
    emb_frame = None
    if EMBEDDINGS.exists():
        emb_frame = pd.read_parquet(EMBEDDINGS)
        emb_frame["node_id"] = emb_frame["node_id"].astype(str).map(lambda x: f"icsplc:{x}")
        emb_frame = emb_frame[emb_frame["node_id"].isin(node_set)].copy()
        emb_available = set(emb_frame["node_id"])
        emb_cols = [c for c in emb_frame.columns if c.startswith("emb_")]
        emb_dim = len(emb_cols)

    features = pd.DataFrame(
        {
            "node_id": companies["company_id"],
            "node_type": "company",
            "node_type_code": 0,
            "degree": companies["degree"],
            "in_degree": companies["in_degree"],
            "out_degree": companies["out_degree"],
            "raw_degree": companies["raw_degree"],
            "component_id": companies["component_id"],
            "component_size": companies["component_size"],
            "is_isolated": (companies["degree"] == 0).astype(int),
            "source_sampoq": companies["source_dataset"].eq("sampoq").astype(int),
            "source_ic_splc": companies["source_dataset"].eq("ic_splc").astype(int),
            "source_anonnet": companies["source_dataset"].eq("anonnet").astype(int),
            "country_code": companies["country"].map(lambda v: country_map.get(clean_text(v), -1)),
            "industry_code": companies["industry"].map(lambda v: industry_map.get(clean_text(v), -1)),
            "product_category_code": companies["product_category"].map(lambda v: product_map.get(clean_text(v), -1)),
            "has_country": companies["country"].map(lambda v: int(bool(clean_text(v)))),
            "has_industry": companies["industry"].map(lambda v: int(bool(clean_text(v)))),
            "has_product_category": companies["product_category"].map(lambda v: int(bool(clean_text(v)))),
            "embedding_available": companies["company_id"].isin(emb_available).astype(int),
            "first_edge_year": companies["company_id"].map(lambda n: first_edge_year.get(n, "")),
            "last_edge_year": companies["company_id"].map(lambda n: last_edge_year.get(n, "")),
            "is_anonymised": companies["name_is_anonymised"].map(lambda v: int(str(v).lower() == "true")),
        }
    )
    write_csv(OUT_NODE_FEATURES, features.to_dict("records"), list(features.columns))
    write_csv(OUT_TRAIN_NODE_FEATURES, features.to_dict("records"), list(features.columns))

    node_ids_sorted = sorted(node_set)
    node_index = {nid: i for i, nid in enumerate(node_ids_sorted)}
    write_csv(
        OUT_NODE_INDEX,
        ({"node_id": nid, "node_index": node_index[nid]} for nid in node_ids_sorted),
        ["node_id", "node_index"],
    )

    edge_index = graph_edges.copy()
    edge_index["source_index"] = edge_index["source_company_id"].map(node_index)
    edge_index["target_index"] = edge_index["target_company_id"].map(node_index)
    n_edges = len(edge_index)
    perm = rng.permutation(n_edges)
    n_train = int(round(n_edges * 0.80))
    n_val = int(round(n_edges * 0.10))
    split = np.empty(n_edges, dtype=object)
    split[perm[:n_train]] = "train"
    split[perm[n_train : n_train + n_val]] = "val"
    split[perm[n_train + n_val :]] = "test"
    edge_index["split"] = split
    edge_index = edge_index[
        [
            "edge_id",
            "source_index",
            "target_index",
            "source_company_id",
            "target_company_id",
            "relationship_type",
            "edge_count",
            "split",
        ]
    ]
    write_csv(OUT_EDGE_INDEX, edge_index.to_dict("records"), list(edge_index.columns))

    risk_rows = (
        {
            "node_id": nid,
            "risk_label": "",
            "label_status": "not_available",
            "label_source": "none",
            "label_confidence": "",
            "notes": "No verified company-level risk labels exist in the acquired sources. Do not fabricate.",
        }
        for nid in node_ids_sorted
    )
    write_csv(
        OUT_RISK,
        risk_rows,
        ["node_id", "risk_label", "label_status", "label_source", "label_confidence", "notes"],
    )

    if emb_frame is not None and len(emb_frame):
        emb_cols = [c for c in emb_frame.columns if c.startswith("emb_")]
        mat = np.zeros((len(node_ids_sorted), len(emb_cols)), dtype=np.float32)
        mask = np.zeros(len(node_ids_sorted), dtype=bool)
        idx = np.array([node_index[n] for n in emb_frame["node_id"]], dtype=np.int64)
        mat[idx] = emb_frame[emb_cols].to_numpy(dtype=np.float32)
        mask[idx] = True
        out = pd.DataFrame(mat, columns=[f"emb_{i}" for i in range(len(emb_cols))])
        out.insert(0, "embedding_available", mask)
        out.insert(0, "node_index", np.arange(len(node_ids_sorted)))
        out.insert(0, "node_id", node_ids_sorted)
        out.to_parquet(OUT_EMBEDDINGS, index=False, compression="zstd")
    else:
        write_json(
            TRAINING_DIR / "embedding_metadata.json",
            {"available": False, "reason": "IC-SPLC embeddings not found", "dim": 0},
        )

    node_splits = stratified_ids(companies, "company_id", "source_dataset", rng)
    edge_splits = {k: sorted(edge_index.loc[edge_index["split"] == k, "edge_id"].tolist()) for k in ["train", "val", "test"]}
    delay_splits: dict[str, dict[str, list[str]]] = {}
    if DELAY_LABELS.exists():
        labels = pd.read_csv(DELAY_LABELS, dtype=str).fillna("")
        for task, group in labels.groupby("task", sort=True):
            delay_splits[task] = stratified_ids(group, "shipment_id", "target_value", rng)
    splits = {
        "seed": args.seed,
        "generated_at": utc_now(),
        "node_splits": node_splits,
        "edge_splits": edge_splits,
        "delay_splits": delay_splits,
        "notes": "Splits are reproducible with the recorded seed. Edge splits are random 80/10/10. Node and delay splits are stratified 70/15/15.",
    }
    write_json(OUT_SPLITS, splits)

    deg = features["degree"]
    degree_stats = {
        "min": int(deg.min()),
        "max": int(deg.max()),
        "mean": round(float(deg.mean()), 3),
        "median": float(deg.median()),
        "p90": float(deg.quantile(0.90)),
        "p99": float(deg.quantile(0.99)),
        "zero_degree_nodes": int((deg == 0).sum()),
    }
    source_counts = companies["source_dataset"].value_counts().to_dict()
    edge_source_counts = edges["source_dataset"].value_counts().to_dict()
    relationship_counts = graph_edges["relationship_type"].value_counts().to_dict()
    years = pd.to_numeric(edges["year"], errors="coerce").dropna()
    year_range = [int(years.min()), int(years.max())] if len(years) else None
    n_components = len(component_sizes)
    largest = max(component_sizes) if component_sizes else 0
    isolated = int((features["degree"] == 0).sum())
    meta = {
        "generated_at": utc_now(),
        "seed": args.seed,
        "node_count": int(len(companies)),
        "source_edge_rows": int(source_edge_rows),
        "duplicate_edge_ids_merged": int(duplicate_edge_ids),
        "raw_edge_count": int(len(edges)),
        "graph_edge_count": int(len(graph_edges)),
        "duplicate_edge_groups": duplicate_groups,
        "self_loops_removed": n_self_loops,
        "missing_endpoint_edges_removed": n_missing_endpoint,
        "node_source_counts": {k: int(v) for k, v in source_counts.items()},
        "edge_source_counts": {k: int(v) for k, v in edge_source_counts.items()},
        "relationship_type_counts": {k: int(v) for k, v in relationship_counts.items()},
        "node_type_counts": {"company": int(len(companies))},
        "isolated_nodes": isolated,
        "weakly_connected_components": n_components,
        "largest_component_size": int(largest),
        "largest_component_share": round(largest / len(companies), 4) if len(companies) else 0,
        "degree_stats": degree_stats,
        "year_range": year_range,
        "embedding": {
            "dim": int(emb_dim),
            "nodes_with_embedding": int(len(emb_available)),
            "coverage": round(len(emb_available) / len(companies), 4) if len(companies) else 0,
            "source": "IC-SPLC 128-d node embeddings; other nodes are zero-filled with an explicit mask in training/node_embeddings.parquet",
        },
        "direction_notes": {
            "sampoq": "Supplier -> Customer (derived from filing relationship type)",
            "ic_splc": "Supplier -> Customer (per dataset card; LLM-extracted)",
            "anonnet": "As provided (source -> target); direction not formally documented",
        },
        "label_availability": {
            "company_risk": "not available",
            "shipment_delay": "available for 156,488 shipments (SynDelay synthetic 3-class, Smart Logistics binary)",
        },
        "known_limitations": [
            "Company identities are not cross-source resolved; IC-SPLC and AnonNet entities are anonymised.",
            "No verified company-level risk labels exist in the acquired sources.",
            "AnonNet direction is not formally documented.",
            "SynDelay label semantics are undocumented; do not interpret classes as delay severity.",
            "Restricted CC BY-NC disruption monitoring data is excluded from this graph and from training.",
            "GDELT records are unverified news mentions, not confirmed disruptions.",
        ],
        "license_notes": {
            "sampoq": "CC BY 4.0",
            "ic_splc": "CC BY 4.0",
            "anonnet": "No LICENSE file; rights unspecified",
            "syndelay": "No LICENSE file; rights unspecified",
            "smart_logistics": "Provenance unverified",
            "disruption_monitoring": "CC BY-NC 4.0; excluded from main graph",
            "gdelt": "GDELT public data; check GDELT terms",
        },
    }
    write_json(OUT_GRAPH_META, meta)

    training_meta = {
        "generated_at": utc_now(),
        "seed": args.seed,
        "tasks": {
            "link_prediction": {
                "edge_index": str(OUT_EDGE_INDEX.relative_to(TRAINING_DIR.parent.parent)),
                "edge_count": int(len(edge_index)),
                "splits": {k: len(v) for k, v in edge_splits.items()},
            },
            "node_risk_prediction": {
                "labels": str(OUT_RISK.relative_to(TRAINING_DIR.parent.parent)),
                "label_status": "not_available",
                "do_not_train": True,
                "reason": "No verified company-level risk labels were found.",
            },
            "shipment_delay_classification": {
                "labels": str(DELAY_LABELS.relative_to(TRAINING_DIR.parent.parent)),
                "rows": int(sum(len(v) for v in delay_splits.values())),
                "tasks": {k: {s: len(v) for s, v in task_splits.items()} for k, task_splits in delay_splits.items()},
                "label_semantics": "SynDelay classes 0/1/2 are undocumented; Smart Logistics binary is direct.",
            },
        },
        "features": {
            "node_features": str(OUT_TRAIN_NODE_FEATURES.relative_to(TRAINING_DIR.parent.parent)),
            "graph_node_features": str(OUT_NODE_FEATURES.relative_to(TRAINING_DIR.parent.parent)),
            "columns": list(features.columns),
            "category_mappings": str(OUT_CATEGORIES.relative_to(TRAINING_DIR.parent.parent)),
            "embeddings": str(OUT_EMBEDDINGS.relative_to(TRAINING_DIR.parent.parent)) if OUT_EMBEDDINGS.exists() else None,
        },
        "node_index": str(OUT_NODE_INDEX.relative_to(TRAINING_DIR.parent.parent)),
        "splits": str(OUT_SPLITS.relative_to(TRAINING_DIR.parent.parent)),
        "known_limitations": meta["known_limitations"],
    }
    write_json(OUT_TRAIN_META, training_meta)

    summary = {
        "nodes": int(len(companies)),
        "raw_edges": int(len(edges)),
        "graph_edges": int(len(graph_edges)),
        "isolated_nodes": isolated,
        "components": n_components,
        "largest_component": int(largest),
        "embedding_coverage": meta["embedding"]["coverage"],
        "edge_splits": {k: len(v) for k, v in edge_splits.items()},
        "node_splits": {k: len(v) for k, v in node_splits.items()},
        "delay_splits": {k: {s: len(v) for s, v in t.items()} for k, t in delay_splits.items()},
    }
    logger.info("Graph built: %s", json.dumps(summary))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
