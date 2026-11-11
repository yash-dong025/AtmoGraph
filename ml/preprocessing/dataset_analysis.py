"""Profile the existing AtmoGraph datasets from an ML perspective.

Writes:
- ``datasets/reports/ml_dataset_analysis.json``
- ``datasets/reports/ml_dataset_analysis.md``

The script is read-only with respect to source datasets. It does not download
data, modify raw files, or use restricted CC BY-NC material.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASETS_DIR = PROJECT_ROOT / "datasets"
REPORTS_DIR = DATASETS_DIR / "reports"
OUT_JSON = REPORTS_DIR / "ml_dataset_analysis.json"
OUT_MD = REPORTS_DIR / "ml_dataset_analysis.md"

DATASET_SPECS = [
    {
        "name": "sampoq_raw",
        "path": DATASETS_DIR / "raw" / "sampoq" / "supply-chain.csv",
        "key": None,
        "role": "Named US-listed supplier/customer relationships",
        "license": "CC BY 4.0",
    },
    {
        "name": "ic_splc_nodes_raw",
        "path": DATASETS_DIR / "raw" / "ic_splc" / "extracted_csv" / "ic_splc_nodes.csv",
        "key": "id",
        "role": "Anonymised supply-chain nodes",
        "license": "CC BY 4.0",
    },
    {
        "name": "ic_splc_edges_raw",
        "path": DATASETS_DIR / "raw" / "ic_splc" / "extracted_csv" / "ic_splc_edges.csv",
        "key": None,
        "role": "Anonymised supplier -> customer edges",
        "license": "CC BY 4.0",
    },
    {
        "name": "ic_splc_embeddings_raw",
        "path": DATASETS_DIR / "raw" / "ic_splc" / "extracted_csv" / "ic_splc_embeddings.parquet",
        "key": "node_id",
        "role": "128-d IC-SPLC node embeddings",
        "license": "CC BY 4.0",
    },
    {
        "name": "anonnet_nodes_raw",
        "path": DATASETS_DIR / "raw" / "anonnet" / "repo" / "mim2025_network_challenge" / "nodes_anon.csv",
        "key": "id",
        "role": "Anonymised supply-network nodes",
        "license": "No LICENSE file; rights unspecified",
    },
    {
        "name": "anonnet_edges_raw",
        "path": DATASETS_DIR / "raw" / "anonnet" / "repo" / "mim2025_network_challenge" / "edges_anon.csv",
        "key": None,
        "role": "Anonymised supply-network edges",
        "license": "No LICENSE file; rights unspecified",
    },
    {
        "name": "syndelay_raw",
        "path": DATASETS_DIR / "raw" / "syndelay" / "repo" / "data" / "syndelay_v1.csv",
        "key": None,
        "role": "Synthetic shipment delay records",
        "license": "No LICENSE file; rights unspecified",
    },
    {
        "name": "smart_logistics_raw",
        "path": DATASETS_DIR / "raw" / "existing_smart_logistics" / "smart_logistics_dataset.csv",
        "key": None,
        "role": "Locally supplied shipment records; provenance unknown",
        "license": "Not specified",
    },
    {
        "name": "graph_nodes",
        "path": DATASETS_DIR / "processed" / "graph" / "nodes.csv",
        "key": "company_id",
        "role": "Combined graph nodes",
        "license": "Mixed: CC BY 4.0 + rights-unspecified anonymised sources",
    },
    {
        "name": "graph_edges",
        "path": DATASETS_DIR / "processed" / "graph" / "edges.csv",
        "key": "edge_id",
        "role": "Deduplicated directed graph edges",
        "license": "Mixed: CC BY 4.0 + rights-unspecified anonymised sources",
    },
    {
        "name": "graph_node_features",
        "path": DATASETS_DIR / "processed" / "graph" / "node_features.csv",
        "key": "node_id",
        "role": "Model-ready node features",
        "license": "Derived from graph sources",
    },
    {
        "name": "training_node_embeddings",
        "path": DATASETS_DIR / "training" / "node_embeddings.parquet",
        "key": "node_id",
        "role": "Aligned 128-d embeddings with mask",
        "license": "CC BY 4.0",
    },
    {
        "name": "training_edge_index",
        "path": DATASETS_DIR / "training" / "edge_index.csv",
        "key": "edge_id",
        "role": "PyG-ready edge indices with splits",
        "license": "Derived from graph sources",
    },
    {
        "name": "training_delay_labels",
        "path": DATASETS_DIR / "training" / "delay_labels.csv",
        "key": None,
        "role": "Shipment delay targets",
        "license": "Mixed; SynDelay rights unspecified; Smart Logistics provenance unknown",
    },
    {
        "name": "training_risk_labels",
        "path": DATASETS_DIR / "training" / "risk_labels.csv",
        "key": "node_id",
        "role": "Company risk labels; currently unavailable",
        "license": "Not applicable; no labels",
    },
    {
        "name": "logistics_features",
        "path": DATASETS_DIR / "processed" / "logistics" / "logistics_features.csv",
        "key": "shipment_id",
        "role": "Leakage-safe shipment features",
        "license": "Mixed; see source caveats",
    },
    {
        "name": "gdelt_articles",
        "path": DATASETS_DIR / "processed" / "disruptions" / "news_articles.csv",
        "key": "article_id",
        "role": "Unverified news metadata",
        "license": "GDELT 2.0 terms",
    },
    {
        "name": "gdelt_events",
        "path": DATASETS_DIR / "processed" / "disruptions" / "disruption_events.csv",
        "key": "disruption_id",
        "role": "Unverified disruption mentions",
        "license": "GDELT 2.0 terms",
    },
]


def profile_dataset(spec: dict[str, Any]) -> dict[str, Any]:
    path: Path = spec["path"]
    result: dict[str, Any] = {
        "name": spec["name"],
        "path": str(path.relative_to(PROJECT_ROOT)),
        "role": spec["role"],
        "license": spec["license"],
        "exists": path.exists(),
    }
    if not path.exists():
        result["status"] = "missing"
        return result
    result["bytes"] = path.stat().st_size
    try:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        result["status"] = "unreadable"
        result["error"] = str(exc)
        return result

    result.update(
        {
            "status": "ok",
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "column_names": list(frame.columns),
            "numeric_columns": list(frame.select_dtypes(include="number").columns),
            "categorical_columns": list(frame.select_dtypes(exclude="number").columns),
            "missing_total": int(frame.isna().sum().sum()),
            "missing_top": {
                str(k): int(v)
                for k, v in frame.isna().sum().sort_values(ascending=False).head(8).items()
                if v
            },
            "duplicate_full_rows": int(frame.duplicated().sum()),
        }
    )
    key = spec.get("key")
    if key and key in frame.columns:
        result["key"] = key
        result["duplicate_key_rows"] = int(frame[key].duplicated().sum())
    return result


def value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).items()}


def target_summary() -> dict[str, Any]:
    delay_path = DATASETS_DIR / "training" / "delay_labels.csv"
    risk_path = DATASETS_DIR / "training" / "risk_labels.csv"
    summary: dict[str, Any] = {}
    if delay_path.exists():
        delay = pd.read_csv(delay_path)
        summary["delay_labels"] = {
            str(task): {str(k): int(v) for k, v in group["target_value"].value_counts().sort_index().items()}
            for task, group in delay.groupby("task")
        }
    if risk_path.exists():
        risk = pd.read_csv(risk_path)
        summary["risk_labels"] = value_counts(risk, "label_status")
    return summary


def graph_summary() -> dict[str, Any]:
    meta_path = DATASETS_DIR / "processed" / "graph" / "graph_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return {
        key: meta.get(key)
        for key in (
            "node_count",
            "raw_edge_count",
            "graph_edge_count",
            "node_source_counts",
            "edge_source_counts",
            "relationship_type_counts",
            "isolated_nodes",
            "weakly_connected_components",
            "largest_component_size",
            "largest_component_share",
            "degree_stats",
            "embedding",
            "label_availability",
            "known_limitations",
        )
    }


def split_summary() -> dict[str, Any]:
    path = DATASETS_DIR / "training" / "dataset_splits.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        splits = json.load(fh)
    return {
        "node_splits": {k: len(v) for k, v in splits.get("node_splits", {}).items()},
        "edge_splits": {k: len(v) for k, v in splits.get("edge_splits", {}).items()},
        "delay_splits": {
            task: {k: len(v) for k, v in task_splits.items()}
            for task, task_splits in splits.get("delay_splits", {}).items()
        },
    }


def feature_summary() -> dict[str, Any]:
    try:
        from ml.preprocessing.feature_engineering import feature_summary as build_summary

        return build_summary()
    except Exception as exc:
        return {"error": str(exc)}


def logistics_feature_coverage() -> dict[str, Any]:
    path = DATASETS_DIR / "processed" / "logistics" / "logistics_features.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    coverage: dict[str, Any] = {}
    for source, group in frame.groupby("source_dataset"):
        usable = [col for col in group.columns if group[col].notna().any()]
        coverage[str(source)] = {
            "rows": int(len(group)),
            "usable_columns": usable,
            "fully_null_columns": [col for col in group.columns if col not in usable],
        }
    return coverage


def build_report() -> dict[str, Any]:
    profiles = [profile_dataset(spec) for spec in DATASET_SPECS]
    by_name = {item["name"]: item for item in profiles}
    report: dict[str, Any] = {
        "generated_by": "ml/preprocessing/dataset_analysis.py",
        "scope": "ML-perspective inspection of existing AtmoGraph datasets",
        "restricted_material": "Excluded from this report; CC BY-NC disruption-monitoring assets are not used for main ML training.",
        "datasets": profiles,
        "graph": graph_summary(),
        "splits": split_summary(),
        "targets": target_summary(),
        "features": feature_summary(),
        "logistics_feature_coverage": logistics_feature_coverage(),
        "ml_readiness": {
            "graph_link_prediction": "ready",
            "node_risk_prediction": "blocked: no verified company-level risk labels",
            "shipment_delay_binary": "ready with caveats: 1,000 local rows, provenance unknown",
            "shipment_delay_3class": "ready with caveats: synthetic SynDelay, class semantics undocumented",
            "news_disruption_signal": "weak: GDELT mentions are unverified and only 8 days",
            "restricted_evaluation": "excluded: CC BY-NC 4.0, permission required",
        },
        "leakage_notes": [
            "Shipment features exclude shipment_status, logistics_delay, logistics_delay_reason, and timestamp.",
            "Graph features exclude risk_label, delay_class, and target_value.",
            "Edge features exclude edge_id, source/target IDs, and split labels.",
            "Scalers are fit on the training split only.",
            "SynDelay classes 0/1/2 are not interpreted as low/medium/high severity.",
            "No company risk labels are fabricated.",
        ],
        "caveats": [
            "The graph is fragmented: 23,198 weakly connected components and only 40.4% of nodes in the largest component.",
            "Only 24.0% of graph nodes have IC-SPLC embeddings; the rest are zero-filled with a mask.",
            "IC-SPLC and AnonNet IDs are anonymised and must not be merged with named companies.",
            "GDELT records are news mentions, not confirmed disruptions.",
            "The existing logistics feature table only exposes latitude/longitude for SynDelay; the 3-class baseline is a pipeline-gap check, not a credible SynDelay model.",
            "SynDelay is synthetic; Smart Logistics provenance is unknown.",
            "AnonNet and SynDelay have no LICENSE file; confirm rights before redistribution.",
        ],
        "source_files": {
            name: by_name[name].get("path")
            for name in (
                "graph_nodes",
                "graph_edges",
                "graph_node_features",
                "training_node_embeddings",
                "training_edge_index",
                "training_delay_labels",
                "training_risk_labels",
                "logistics_features",
                "gdelt_articles",
                "gdelt_events",
            )
            if name in by_name
        },
    }
    return report


def _table_row(item: dict[str, Any]) -> str:
    rows = item.get("rows", "—")
    cols = item.get("columns", "—")
    missing = item.get("missing_total", "—")
    duplicates = item.get("duplicate_full_rows", "—")
    return f"| `{item['name']}` | {rows} | {cols} | {missing} | {duplicates} | {item['license']} |"


def render_markdown(report: dict[str, Any]) -> str:
    graph = report.get("graph", {})
    features = report.get("features", {})
    lines = [
        "# AtmoGraph ML Dataset Analysis",
        "",
        "Generated by `ml/preprocessing/dataset_analysis.py`.",
        "",
        "This report profiles the existing dataset package from the ML/GNN perspective. It does not download data, modify raw files, or use restricted CC BY-NC material for main training.",
        "",
        "## ML readiness",
        "",
        "| Task | Status |",
        "|---|---|",
    ]
    for task, status in report["ml_readiness"].items():
        lines.append(f"| `{task}` | {status} |")

    lines += [
        "",
        "## Graph snapshot",
        "",
        f"- Nodes: **{graph.get('node_count', '—'):,}**" if isinstance(graph.get("node_count"), int) else "- Nodes: —",
        f"- Unique directed edges: **{graph.get('graph_edge_count', '—'):,}**" if isinstance(graph.get("graph_edge_count"), int) else "- Unique directed edges: —",
        f"- Weakly connected components: **{graph.get('weakly_connected_components', '—'):,}**" if isinstance(graph.get("weakly_connected_components"), int) else "- Weakly connected components: —",
        f"- Largest component: **{graph.get('largest_component_size', '—'):,}**" if isinstance(graph.get("largest_component_size"), int) else "- Largest component: —",
        f"- Isolated nodes: **{graph.get('isolated_nodes', '—'):,}**" if isinstance(graph.get("isolated_nodes"), int) else "- Isolated nodes: —",
        "",
        "## Dataset profiles",
        "",
        "| Dataset | Rows | Columns | Missing cells | Full duplicate rows | Licence |",
        "|---|---:|---:|---:|---:|---|",
    ]
    lines += [_table_row(item) for item in report["datasets"]]

    lines += [
        "",
        "## Targets",
        "",
        "```json",
        json.dumps(report.get("targets", {}), indent=2),
        "```",
        "",
        "## Logistics feature coverage",
        "",
        "```json",
        json.dumps(report.get("logistics_feature_coverage", {}), indent=2),
        "```",
        "",
        "## Feature summary",
        "",
        "```json",
        json.dumps(features, indent=2)[:12000],
        "```",
        "",
        "## Splits",
        "",
        "```json",
        json.dumps(report.get("splits", {}), indent=2),
        "```",
        "",
        "## Leakage controls",
        "",
    ]
    lines += [f"- {note}" for note in report["leakage_notes"]]
    lines += ["", "## Caveats", ""]
    lines += [f"- {note}" for note in report["caveats"]]
    lines += [
        "",
        "## Restricted material",
        "",
        report["restricted_material"],
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
        fh.write("\n")
    print(
        json.dumps(
            {
                "datasets_profiled": len(report["datasets"]),
                "graph_nodes": report.get("graph", {}).get("node_count"),
                "graph_edges": report.get("graph", {}).get("graph_edge_count"),
                "node_feature_dim": report.get("features", {}).get("node_feature_dim"),
                "edge_feature_dim": report.get("features", {}).get("edge_feature_dim"),
                "output_json": str(OUT_JSON.relative_to(PROJECT_ROOT)),
                "output_markdown": str(OUT_MD.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
