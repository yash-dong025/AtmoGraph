from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    DATASETS_DIR,
    EVALUATION_DIR,
    METADATA_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    REPORTS_DIR,
    TRAINING_DIR,
    ensure_dirs,
    rel,
    setup_logging,
    utc_now,
    write_json,
)

OUT = REPORTS_DIR / "dataset_inventory.json"

TABLES: list[tuple[str, Path, str | None, bool]] = [
    ("companies_sampoq", PROCESSED_DIR / "companies" / "companies_sampoq.csv", "company_id", False),
    ("companies_ic_splc", PROCESSED_DIR / "companies" / "companies_ic_splc.csv", "company_id", False),
    ("companies_anonnet", PROCESSED_DIR / "companies" / "companies_anonnet.csv", "company_id", False),
    ("companies_combined", PROCESSED_DIR / "companies" / "companies.csv", "company_id", False),
    ("edges_sampoq", PROCESSED_DIR / "relationships" / "supply_chain_edges_sampoq.csv", "edge_id", False),
    ("edges_ic_splc", PROCESSED_DIR / "relationships" / "supply_chain_edges_ic_splc.csv", "edge_id", False),
    ("edges_anonnet", PROCESSED_DIR / "relationships" / "supply_chain_edges_anonnet.csv", "edge_id", False),
    ("edges_combined", PROCESSED_DIR / "relationships" / "supply_chain_edges.csv", "edge_id", False),
    ("graph_nodes", PROCESSED_DIR / "graph" / "nodes.csv", "company_id", False),
    ("graph_edges", PROCESSED_DIR / "graph" / "edges.csv", "edge_id", False),
    ("graph_node_features", PROCESSED_DIR / "graph" / "node_features.csv", "node_id", False),
    ("training_node_features", TRAINING_DIR / "node_features.csv", "node_id", False),
    ("shipment_delays", PROCESSED_DIR / "logistics" / "shipment_delays.csv", "shipment_id", False),
    ("logistics_features", PROCESSED_DIR / "logistics" / "logistics_features.csv", "shipment_id", False),
    ("news_articles", PROCESSED_DIR / "disruptions" / "news_articles.csv", "article_id", False),
    ("disruption_events", PROCESSED_DIR / "disruptions" / "disruption_events.csv", "disruption_id", False),
    ("restricted_companies", PROCESSED_DIR / "restricted" / "disruption_monitoring" / "companies.csv", "company_id", True),
    ("restricted_edges", PROCESSED_DIR / "restricted" / "disruption_monitoring" / "edges.csv", "edge_id", True),
    ("delay_labels", TRAINING_DIR / "delay_labels.csv", None, False),
    ("risk_labels", TRAINING_DIR / "risk_labels.csv", "node_id", False),
    ("node_index", TRAINING_DIR / "node_index_mapping.csv", "node_id", False),
    ("edge_index", TRAINING_DIR / "edge_index.csv", "edge_id", False),
]

CATEGORICAL = [
    "source_dataset",
    "relationship_type",
    "verification_status",
    "disruption_type",
    "collection_method",
    "task",
    "target_value",
    "label_status",
    "split",
    "is_synthetic",
    "embedding_available",
    "is_isolated",
]

RAW_ASSETS = [
    ("sampoq_csv", RAW_DIR / "sampoq" / "supply-chain.csv"),
    ("ic_splc_zip", RAW_DIR / "ic_splc" / "txh3rrsh89-v2.zip"),
    ("ic_splc_dump", RAW_DIR / "ic_splc" / "extracted" / "IC-SPLC-v2 A Supply Chain Structure Neo4j Data" / "IC-SPLC.dump"),
    ("ic_splc_nodes", RAW_DIR / "ic_splc" / "extracted_csv" / "ic_splc_nodes.csv"),
    ("ic_splc_edges", RAW_DIR / "ic_splc" / "extracted_csv" / "ic_splc_edges.csv"),
    ("ic_splc_embeddings", RAW_DIR / "ic_splc" / "extracted_csv" / "ic_splc_embeddings.parquet"),
    ("syndelay_csv", RAW_DIR / "syndelay" / "repo" / "data" / "syndelay_v1.csv"),
    ("smart_logistics_csv", RAW_DIR / "existing_smart_logistics" / "smart_logistics_dataset.csv"),
    ("anonnet_nodes", RAW_DIR / "anonnet" / "repo" / "mim2025_network_challenge" / "nodes_anon.csv"),
    ("anonnet_edges", RAW_DIR / "anonnet" / "repo" / "mim2025_network_challenge" / "edges_anon.csv"),
    ("gdelt_collection_metadata", RAW_DIR / "gdelt" / "collection_metadata.json"),
    ("restricted_kg", RAW_DIR / "disruption_monitoring" / "restricted" / "repo" / "dataset" / "supplychainKG_clean.csv"),
    ("restricted_scenarios", RAW_DIR / "disruption_monitoring" / "restricted" / "repo" / "evaluation" / "scenarios" / "multi_company_scenarios.json"),
    ("restricted_ground_truth_index", RAW_DIR / "disruption_monitoring" / "restricted" / "repo" / "evaluation" / "ground_truth_final" / "index.json"),
    ("restricted_benchmark", RAW_DIR / "disruption_monitoring" / "restricted" / "repo" / "evaluation" / "benchmark_v2" / "items" / "test" / "test_set.json"),
]


def inspect_table(name: str, path: Path, key: str | None, restricted: bool) -> dict[str, Any]:
    if not path.exists():
        return {"name": name, "path": rel(path), "status": "missing", "restricted": restricted}
    info: dict[str, Any] = {
        "name": name,
        "path": rel(path),
        "format": path.suffix.lstrip(".").lower(),
        "bytes": path.stat().st_size,
        "restricted": restricted,
    }
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        info["status"] = "unreadable"
        info["error"] = str(exc)
        return info
    info["status"] = "ok"
    info["rows"] = int(len(df))
    info["columns"] = list(df.columns)
    info["column_count"] = int(len(df.columns))
    missing = (df.replace("", pd.NA).isna().sum()).sort_values(ascending=False)
    info["missing_top"] = {str(k): int(v) for k, v in missing.head(8).items() if v > 0}
    if key and key in df.columns:
        info["key"] = key
        info["duplicate_key_rows"] = int(df[key].duplicated().sum())
    for col in CATEGORICAL:
        if col in df.columns and df[col].nunique(dropna=False) <= 25:
            info[f"value_counts_{col}"] = {str(k): int(v) for k, v in df[col].value_counts(dropna=False).head(25).items()}
    return info


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect acquired and processed datasets")
    args = parser.parse_args()
    ensure_dirs()
    logger = setup_logging("inspect_datasets")
    tables = [inspect_table(*spec) for spec in TABLES]
    raw_assets = []
    for name, path in RAW_ASSETS:
        raw_assets.append(
            {
                "name": name,
                "path": rel(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    directories = []
    for name, path in [
        ("raw", RAW_DIR),
        ("processed", PROCESSED_DIR),
        ("training", TRAINING_DIR),
        ("evaluation", EVALUATION_DIR),
        ("metadata", METADATA_DIR),
        ("reports", REPORTS_DIR),
        ("gdelt_api", RAW_DIR / "gdelt" / "api"),
        ("gdelt_gkg", RAW_DIR / "gdelt" / "gkg"),
        ("gdelt_gkg_filtered", RAW_DIR / "gdelt" / "gkg_filtered"),
        ("restricted_repo", RAW_DIR / "disruption_monitoring" / "restricted" / "repo"),
    ]:
        directories.append({"name": name, "path": rel(path), "bytes": dir_size(path)})
    inventory = {
        "generated_at": utc_now(),
        "datasets_dir": rel(DATASETS_DIR),
        "tables": tables,
        "raw_assets": raw_assets,
        "directories": directories,
        "notes": [
            "Row counts are physical rows in the processed files.",
            "Duplicate key rows are counted on the listed key column after preprocessing.",
            "Restricted tables are kept outside the main graph and training pipeline.",
        ],
    }
    write_json(OUT, inventory)
    ok = [t for t in tables if t.get("status") == "ok"]
    print(json.dumps({"tables_ok": len(ok), "tables_total": len(tables), "output": rel(OUT)}, indent=2))
    for t in tables:
        if t.get("status") != "ok":
            logger.warning("Table not available: %s (%s)", t["name"], t.get("status"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
