from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    EVALUATION_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    TRAINING_DIR,
    ensure_dirs,
    read_json,
    rel,
    setup_logging,
    utc_now,
    write_json,
)

OUT = REPORTS_DIR / "validation_results.json"

REQUIRED = [
    PROCESSED_DIR / "companies" / "companies.csv",
    PROCESSED_DIR / "relationships" / "supply_chain_edges.csv",
    PROCESSED_DIR / "graph" / "nodes.csv",
    PROCESSED_DIR / "graph" / "edges.csv",
    PROCESSED_DIR / "graph" / "node_features.csv",
    PROCESSED_DIR / "graph" / "graph_metadata.json",
    PROCESSED_DIR / "logistics" / "shipment_delays.csv",
    PROCESSED_DIR / "logistics" / "logistics_features.csv",
    PROCESSED_DIR / "disruptions" / "news_articles.csv",
    PROCESSED_DIR / "disruptions" / "disruption_events.csv",
    TRAINING_DIR / "node_features.csv",
    TRAINING_DIR / "node_index_mapping.csv",
    TRAINING_DIR / "edge_index.csv",
    TRAINING_DIR / "risk_labels.csv",
    TRAINING_DIR / "delay_labels.csv",
    TRAINING_DIR / "dataset_splits.json",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate processed datasets and training splits")
    args = parser.parse_args()
    ensure_dirs()
    logger = setup_logging("validate_datasets")
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, details: Any = None) -> None:
        checks.append({"name": name, "status": "pass" if ok else "fail", "details": details})
        if not ok:
            logger.error("FAILED: %s (%s)", name, details)

    missing = [rel(p) for p in REQUIRED if not p.exists()]
    check("required_files_exist", not missing, missing)

    companies = pd.read_csv(PROCESSED_DIR / "companies" / "companies.csv", dtype=str).fillna("")
    graph_edges = pd.read_csv(PROCESSED_DIR / "graph" / "edges.csv", dtype=str).fillna("")
    raw_edges = pd.read_csv(PROCESSED_DIR / "relationships" / "supply_chain_edges.csv", dtype=str).fillna("")
    features = pd.read_csv(PROCESSED_DIR / "graph" / "node_features.csv", dtype=str).fillna("")
    node_index = pd.read_csv(TRAINING_DIR / "node_index_mapping.csv", dtype=str).fillna("")
    edge_index = pd.read_csv(TRAINING_DIR / "edge_index.csv", dtype=str).fillna("")
    risk = pd.read_csv(TRAINING_DIR / "risk_labels.csv", dtype=str).fillna("")
    delays = pd.read_csv(TRAINING_DIR / "delay_labels.csv", dtype=str).fillna("")
    splits = read_json(TRAINING_DIR / "dataset_splits.json", default={})
    meta = read_json(PROCESSED_DIR / "graph" / "graph_metadata.json", default={})

    node_ids = set(companies["company_id"])
    check("company_ids_unique", not companies["company_id"].duplicated().any(), int(companies["company_id"].duplicated().sum()))
    check("graph_edge_endpoints_exist", graph_edges["source_company_id"].isin(node_ids).all() and graph_edges["target_company_id"].isin(node_ids).all())
    check("graph_has_no_self_loops", not (graph_edges["source_company_id"] == graph_edges["target_company_id"]).any())
    check("graph_edge_ids_unique", not graph_edges["edge_id"].duplicated().any(), int(graph_edges["edge_id"].duplicated().sum()))
    check("raw_edge_ids_unique_after_merge", not raw_edges["edge_id"].duplicated().any(), int(raw_edges["edge_id"].duplicated().sum()))

    check("node_index_covers_all_nodes", set(node_index["node_id"]) == node_ids, {"mapped": len(node_index), "nodes": len(node_ids)})
    check("node_index_unique", not node_index["node_id"].duplicated().any())
    idx_values = pd.to_numeric(node_index["node_index"], errors="coerce")
    check("node_index_contiguous", idx_values.notna().all() and sorted(idx_values.astype(int).tolist()) == list(range(len(node_index))))

    check("node_features_align", set(features["node_id"]) == node_ids, {"features": len(features), "nodes": len(node_ids)})
    train_features = pd.read_csv(TRAINING_DIR / "node_features.csv", dtype=str).fillna("")
    check("training_node_features_align", set(train_features["node_id"]) == node_ids and list(train_features.columns) == list(features.columns), {"rows": len(train_features), "columns_match": list(train_features.columns) == list(features.columns)})
    check("edge_index_align", set(edge_index["edge_id"]) == set(graph_edges["edge_id"]), {"edge_index": len(edge_index), "edges": len(graph_edges)})
    src = pd.to_numeric(edge_index["source_index"], errors="coerce")
    tgt = pd.to_numeric(edge_index["target_index"], errors="coerce")
    n = len(node_index)
    check("edge_index_in_range", src.notna().all() and tgt.notna().all() and int(src.min()) >= 0 and int(src.max()) < n and int(tgt.min()) >= 0 and int(tgt.max()) < n)
    check("edge_index_no_self_loops", not (src == tgt).any(), int((src == tgt).sum()))
    id_to_idx = dict(zip(node_index["node_id"], pd.to_numeric(node_index["node_index"], errors="coerce").astype(int)))
    sample = edge_index.head(2000)
    mapping_ok = all(
        id_to_idx.get(r["source_company_id"]) == int(r["source_index"]) and id_to_idx.get(r["target_company_id"]) == int(r["target_index"])
        for _, r in sample.iterrows()
    )
    check("edge_index_mapping_sample", mapping_ok, "first 2000 edges")

    check("risk_labels_not_fabricated", (risk["label_status"] == "not_available").all() and (risk["risk_label"] == "").all(), risk["label_status"].value_counts().to_dict())
    check("risk_labels_cover_nodes", set(risk["node_id"]) == node_ids, {"risk_rows": len(risk), "nodes": len(node_ids)})

    task_values = {task: set(group["target_value"]) for task, group in delays.groupby("task")}
    task_values_detail = {k: sorted(v) for k, v in task_values.items()}
    check("delay_task_names", set(task_values) == {"delay_classification_3class", "delay_classification_binary"}, task_values_detail)
    check("delay_3class_values", task_values.get("delay_classification_3class", set()) <= {"0", "1", "2"}, sorted(task_values.get("delay_classification_3class", [])))
    check("delay_binary_values", task_values.get("delay_classification_binary", set()) <= {"0", "1"}, sorted(task_values.get("delay_classification_binary", [])))
    check("delay_label_ids_unique_per_task", not delays.duplicated(["task", "shipment_id"]).any(), int(delays.duplicated(["task", "shipment_id"]).sum()))

    node_splits = splits.get("node_splits", {})
    node_split_sets = {k: set(v) for k, v in node_splits.items()}
    node_union = set().union(*node_split_sets.values()) if node_split_sets else set()
    node_disjoint = all(not (node_split_sets[a] & node_split_sets[b]) for a in node_split_sets for b in node_split_sets if a < b)
    check("node_splits_cover_all", node_union == node_ids, {"covered": len(node_union), "nodes": len(node_ids)})
    check("node_splits_disjoint", node_disjoint)

    edge_splits = splits.get("edge_splits", {})
    edge_split_sets = {k: set(v) for k, v in edge_splits.items()}
    edge_union = set().union(*edge_split_sets.values()) if edge_split_sets else set()
    edge_disjoint = all(not (edge_split_sets[a] & edge_split_sets[b]) for a in edge_split_sets for b in edge_split_sets if a < b)
    check("edge_splits_cover_all", edge_union == set(graph_edges["edge_id"]), {"covered": len(edge_union), "edges": len(graph_edges)})
    check("edge_splits_disjoint", edge_disjoint)

    delay_splits = splits.get("delay_splits", {})
    delay_ok = True
    delay_details = {}
    for task, group in delays.groupby("task"):
        expected = set(group["shipment_id"])
        task_splits = {k: set(v) for k, v in delay_splits.get(task, {}).items()}
        union = set().union(*task_splits.values()) if task_splits else set()
        disjoint = all(not (task_splits[a] & task_splits[b]) for a in task_splits for b in task_splits if a < b)
        delay_details[task] = {"covered": len(union), "expected": len(expected), "disjoint": disjoint}
        delay_ok = delay_ok and union == expected and disjoint
    check("delay_splits_cover_and_disjoint", delay_ok, delay_details)

    emb_path = TRAINING_DIR / "node_embeddings.parquet"
    if emb_path.exists():
        emb = pd.read_parquet(emb_path, columns=["node_id", "node_index", "embedding_available"])
        check("embedding_rows_align", set(emb["node_id"]) == node_ids, {"rows": len(emb), "nodes": len(node_ids)})
        check("embedding_mask_count", int(emb["embedding_available"].sum()) == int(meta.get("embedding", {}).get("nodes_with_embedding", -1)), int(emb["embedding_available"].sum()))
    else:
        check("embedding_file_exists", False, rel(emb_path))

    restricted_companies = PROCESSED_DIR / "restricted" / "disruption_monitoring" / "companies.csv"
    if restricted_companies.exists():
        restricted_ids = set(pd.read_csv(restricted_companies, usecols=["company_id"], dtype=str)["company_id"])
        check("restricted_excluded_from_graph", not (restricted_ids & node_ids), {"overlap": len(restricted_ids & node_ids)})
    check("no_restricted_source_in_graph", not companies["source_dataset"].str.contains("restricted|disruption_monitoring", case=False, regex=True).any())
    check("no_restricted_edge_sources", not graph_edges["source_datasets"].str.contains("restricted|disruption_monitoring", case=False, regex=True).any())

    events = pd.read_csv(PROCESSED_DIR / "disruptions" / "disruption_events.csv", dtype=str).fillna("")
    check("gdelt_events_unverified", (events["verification_status"] == "unverified_news_mention").all(), events["verification_status"].value_counts().to_dict())
    check("gdelt_no_fabricated_company", (events["affected_company"] == "").all(), int((events["affected_company"] != "").sum()))
    check("gdelt_source_urls_present", (events["source_url"] != "").all())

    logistics_features = pd.read_csv(PROCESSED_DIR / "logistics" / "logistics_features.csv", dtype=str)
    leakage = {"shipment_status", "logistics_delay", "logistics_delay_reason", "timestamp"} & set(logistics_features.columns)
    check("no_target_leakage_in_logistics_features", not leakage, sorted(leakage))
    graph_feature_cols = set(features.columns)
    check("no_label_columns_in_graph_features", not ({"risk_label", "delay_class", "target_value"} & graph_feature_cols), sorted({"risk_label", "delay_class", "target_value"} & graph_feature_cols))

    check("sampoq_confidence", set(raw_edges.loc[raw_edges["source_dataset"] == "sampoq", "confidence"]) <= {"0.9"})
    check("ic_splc_confidence", set(raw_edges.loc[raw_edges["source_dataset"] == "ic_splc", "confidence"]) <= {"0.7"})
    check("anonnet_confidence", set(raw_edges.loc[raw_edges["source_dataset"] == "anonnet", "confidence"]) <= {"0.8"})

    eval_meta = read_json(EVALUATION_DIR / "evaluation_metadata.json", default={})
    check("restricted_eval_not_in_main_graph", eval_meta.get("included_in_main_graph") is False, eval_meta.get("included_in_main_graph"))

    failed = [c for c in checks if c["status"] == "fail"]
    result = {
        "generated_at": utc_now(),
        "status": "pass" if not failed else "fail",
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "checks": checks,
        "notes": [
            "A pass on label availability does not mean risk labels exist; risk labels are explicitly marked not_available.",
            "GDELT events are unverified news mentions, not confirmed disruptions.",
            "Restricted CC BY-NC data is validated only for exclusion from the main graph.",
        ],
    }
    write_json(OUT, result)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "failed": result["failed"], "output": rel(OUT)}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
