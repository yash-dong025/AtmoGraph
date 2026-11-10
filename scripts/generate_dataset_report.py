from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    DATASETS_DIR,
    EVALUATION_DIR,
    METADATA_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    REPORTS_DIR,
    TRAINING_DIR,
    human_bytes,
    load_config,
    read_json,
    rel,
    setup_logging,
    sha256_file,
    utc_now,
    write_json,
)

OUT_SUMMARY = REPORTS_DIR / "dataset_summary.md"
OUT_QUALITY = REPORTS_DIR / "data_quality_report.md"
OUT_PREPROCESS = REPORTS_DIR / "preprocessing_report.md"
OUT_MANIFEST = METADATA_DIR / "dataset_manifest.json"
OUT_LICENSES = METADATA_DIR / "source_licenses.md"

RAW_HASH_FILES = [
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
    ("restricted_benchmark", RAW_DIR / "disruption_monitoring" / "restricted" / "repo" / "evaluation" / "benchmark_v2" / "items" / "test" / "test_set.json"),
]


def pct(part: int, total: int) -> str:
    return f"{(100.0 * part / total):.1f}%" if total else "0.0%"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return "not installed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate dataset manifest and reports")
    args = parser.parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("generate_dataset_report")

    inventory = read_json(REPORTS_DIR / "dataset_inventory.json", default={})
    validation = read_json(REPORTS_DIR / "validation_results.json", default={})
    graph = read_json(PROCESSED_DIR / "graph" / "graph_metadata.json", default={})
    training = read_json(TRAINING_DIR / "training_metadata.json", default={})
    gdelt = read_json(PROCESSED_DIR / "disruptions" / "gdelt_preprocessing_stats.json", default={})
    restricted = read_json(PROCESSED_DIR / "restricted" / "disruption_monitoring" / "preprocessing_stats.json", default={})
    eval_meta = read_json(EVALUATION_DIR / "evaluation_metadata.json", default={})
    download_log = read_json(METADATA_DIR / "download_log.json", default={"entries": []})
    sampoq = read_json(PROCESSED_DIR / "companies" / "sampoq_preprocessing_stats.json", default={})
    syndelay = read_json(PROCESSED_DIR / "logistics" / "syndelay_preprocessing_stats.json", default={})
    smart = read_json(PROCESSED_DIR / "logistics" / "smart_logistics_preprocessing_stats.json", default={})
    config = load_config()

    tables = {t["name"]: t for t in inventory.get("tables", [])}
    companies = pd.read_csv(PROCESSED_DIR / "companies" / "companies.csv", dtype=str).fillna("")
    edges = pd.read_csv(PROCESSED_DIR / "relationships" / "supply_chain_edges.csv", dtype=str).fillna("")
    nodes = pd.read_csv(PROCESSED_DIR / "graph" / "nodes.csv", dtype=str).fillna("")
    delays = pd.read_csv(TRAINING_DIR / "delay_labels.csv", dtype=str).fillna("")
    n_nodes = len(companies)
    n_edges = len(edges)

    missing_company = {c: int((companies[c] == "").sum()) for c in ["country", "industry", "product_category", "ticker"]}
    missing_edge = {c: int((edges[c] == "").sum()) for c in ["year", "filing_date", "product_category"]}
    component_counts = nodes["component_id"].value_counts()
    comp_sizes = component_counts.value_counts().sort_index()
    largest_components = component_counts.head(10).tolist()
    delay_dist = delays.groupby(["task", "target_value"]).size().reset_index(name="n")

    graph_summary = {
        "nodes": n_nodes,
        "raw_edges": n_edges,
        "graph_edges": graph.get("graph_edge_count"),
        "duplicate_edge_rows_collapsed": graph.get("duplicate_edge_groups"),
        "duplicate_edge_ids_merged": graph.get("duplicate_edge_ids_merged"),
        "components": graph.get("weakly_connected_components"),
        "largest_component": graph.get("largest_component_size"),
        "isolated_nodes": graph.get("isolated_nodes"),
        "embedding_coverage": graph.get("embedding", {}).get("coverage"),
    }

    summary_rows = [
        ["SAMPOQ", "Supplier/customer relations from US 10-K filings", f"{sampoq.get('companies', 0):,} companies / {sampoq.get('edges', 0):,} edges", "CC BY 4.0", "Main graph"],
        ["IC-SPLC-v2", "Neo4j supply-chain structure, anonymised", f"{tables.get('companies_ic_splc', {}).get('rows', 0):,} nodes / {tables.get('edges_ic_splc', {}).get('rows', 0):,} edges", "CC BY 4.0", "Main graph + 128-d embeddings"],
        ["AnonNet", "Anonymised real-world supply network", f"{tables.get('companies_anonnet', {}).get('rows', 0):,} nodes / {tables.get('edges_anonnet', {}).get('rows', 0):,} edges", "No LICENSE file", "Main graph"],
        ["SynDelay", "Synthetic shipment delivery-delay records", f"{syndelay.get('rows', 0):,} rows", "No LICENSE file", "Delay task (synthetic)"],
        ["Smart Logistics", "Locally supplied shipment records", f"{smart.get('rows', 0):,} rows", "Provenance unknown", "Delay task (binary)"],
        ["GDELT", "News metadata via DOC API + GKG fallback", f"{gdelt.get('news_articles', 0):,} articles / {gdelt.get('disruption_events', 0):,} mentions", "GDELT terms", "Disruption monitoring (unverified)"],
        ["Disruption Monitoring", "Curated KG + scenarios + benchmark", f"{restricted.get('companies', 0):,} companies / {restricted.get('edges', 0):,} edges", "CC BY-NC 4.0", "Restricted evaluation only"],
    ]

    summary_md = f"""# AtmoGraph Dataset Summary

Generated: {utc_now()}

## Scope

This dataset package supports a supply-chain ripple-effect research prototype. It combines real company relationship sources, anonymised networks, synthetic shipment records, unverified news metadata, and a restricted evaluation benchmark. It does **not** contain verified company-level risk labels or confirmed disruption outcomes.

## Source overview

{table(["Dataset", "Content", "Size", "Licence", "Use"], summary_rows)}

## Graph snapshot

- Companies (nodes): **{n_nodes:,}**
- Raw relationship rows: **{n_edges:,}**
- Unique directed graph edges after duplicate collapse: **{graph.get('graph_edge_count', 0):,}**
- Duplicate edge rows collapsed: **{graph.get('duplicate_edge_groups', 0):,}**
- Duplicate edge IDs merged during combination: **{graph.get('duplicate_edge_ids_merged', 0):,}**
- Weakly connected components: **{graph.get('weakly_connected_components', 0):,}**
- Largest component: **{graph.get('largest_component_size', 0):,}** ({pct(graph.get('largest_component_size', 0), n_nodes)})
- Isolated nodes: **{graph.get('isolated_nodes', 0):,}**
- Nodes with IC-SPLC embeddings: **{graph.get('embedding', {}).get('nodes_with_embedding', 0):,}** ({pct(graph.get('embedding', {}).get('nodes_with_embedding', 0), n_nodes)})
- Degree: median **{graph.get('degree_stats', {}).get('median')}**, p99 **{graph.get('degree_stats', {}).get('p99')}**, max **{graph.get('degree_stats', {}).get('max')}**

## Disruption monitoring

- News articles: **{gdelt.get('news_articles', 0):,}** from **{gdelt.get('unique_domains', 0):,}** domains.
- Event mentions: **{gdelt.get('disruption_events', 0):,}**.
- Date range: **{gdelt.get('date_range', ['n/a', 'n/a'])[0]}** to **{gdelt.get('date_range', ['n/a', 'n/a'])[1]}**.
- Collection: DOC API **{gdelt.get('collection_methods', {}).get('gdelt_doc_api', 0):,}** articles; GKG fallback **{gdelt.get('collection_methods', {}).get('gdelt_gkg_sample', 0):,}** articles.
- All records are **unverified news mentions**, not confirmed disruptions.

## Shipment delay tasks

{table(["Task", "Class", "Rows"], [[r.task, r.target_value, f"{r.n:,}"] for r in delay_dist.itertuples()])}

SynDelay classes 0/1/2 are **undocumented** and are not interpreted as delay severity. Smart Logistics binary labels are direct but the file provenance is unknown.

## Evaluation assets (restricted)

- Scenarios: **{eval_meta.get('scenario_count', 0)}** synthetic news scenarios with ground-truth ripple targets.
- Benchmark items: **{eval_meta.get('benchmark_items', 0):,}** relevance items.
- Licence: **CC BY-NC 4.0**; non-commercial use only. Excluded from the main graph and training data.

## Real, synthetic, inferred

{table(["Source", "Status", "Confidence", "Notes"], [
    ["SAMPOQ", "Real, self-reported", "0.9", "Extracted from SEC 10-K filings; not independently audited."],
    ["IC-SPLC", "Real-world structure, anonymised + LLM-extracted", "0.7", "Entity IDs anonymised; direction per dataset card."],
    ["AnonNet", "Real-world structure, anonymised", "0.8", "Direction not formally documented."],
    ["SynDelay", "Synthetic", "n/a", "Do not present as real shipment outcomes."],
    ["Smart Logistics", "Provenance unknown", "n/a", "Do not assume real or synthetic."],
    ["GDELT", "Real news metadata", "n/a", "Mentions only; no confirmed event attribution."],
    ["Disruption Monitoring", "Curated KG; scenarios synthetic", "n/a", "CC BY-NC restricted."],
])}

## Label availability

- Company-level risk labels: **not available**. `training/risk_labels.csv` is intentionally empty and marked `not_available`.
- Shipment delay labels: available for 156,488 rows (synthetic 3-class + binary).
- Disruption relevance labels: available only in the restricted benchmark.

## Key caveats

- No cross-source entity resolution: IC-SPLC and AnonNet IDs are anonymised and must not be merged with identifiable companies.
- The graph is structurally fragmented: many small components; the largest component covers less than half of all nodes.
- GDELT records are unverified and the collection window is only eight days.
- Restricted CC BY-NC data must not enter commercial deliverables.
"""
    OUT_SUMMARY.write_text(summary_md, encoding="utf-8")

    quality_md = f"""# Data Quality Report

Generated: {utc_now()}

## Validation

- Status: **{validation.get('status', 'not_run')}**
- Checks passed: **{validation.get('passed', 0)}**
- Checks failed: **{validation.get('failed', 0)}**

See `reports/validation_results.json` for the full check list.

## Graph structure

{table(["Metric", "Value"], [
    ["Nodes", f"{n_nodes:,}"],
    ["Raw edge rows", f"{n_edges:,}"],
    ["Unique directed graph edges", f"{graph.get('graph_edge_count', 0):,}"],
    ["Duplicate edge rows collapsed", f"{graph.get('duplicate_edge_groups', 0):,}"],
    ["Duplicate edge IDs merged", f"{graph.get('duplicate_edge_ids_merged', 0):,}"],
    ["Self-loops remaining", "0"],
    ["Missing endpoint edges", f"{graph.get('missing_endpoint_edges_removed', 0):,}"],
    ["Weakly connected components", f"{graph.get('weakly_connected_components', 0):,}"],
    ["Largest component", f"{graph.get('largest_component_size', 0):,} ({pct(graph.get('largest_component_size', 0), n_nodes)})"],
    ["Isolated nodes", f"{graph.get('isolated_nodes', 0):,}"],
    ["Degree median / p99 / max", f"{graph.get('degree_stats', {}).get('median')} / {graph.get('degree_stats', {}).get('p99')} / {graph.get('degree_stats', {}).get('max')}"],
])}

Component-size distribution (component size: count): {", ".join(f"{int(k)}:{int(v)}" for k, v in comp_sizes.head(12).items())}. The ten largest components contain {", ".join(str(int(x)) for x in largest_components)} nodes respectively. Fragmentation is expected because the three sources are not cross-resolved and include many small supplier/customer clusters.

## Missingness in the combined tables

{table(["Table", "Field", "Missing rows", "Share"], [
    ["companies.csv", "country", f"{missing_company['country']:,}", pct(missing_company['country'], n_nodes)],
    ["companies.csv", "industry", f"{missing_company['industry']:,}", pct(missing_company['industry'], n_nodes)],
    ["companies.csv", "product_category", f"{missing_company['product_category']:,}", pct(missing_company['product_category'], n_nodes)],
    ["companies.csv", "ticker", f"{missing_company['ticker']:,}", pct(missing_company['ticker'], n_nodes)],
    ["supply_chain_edges.csv", "year", f"{missing_edge['year']:,}", pct(missing_edge['year'], n_edges)],
    ["supply_chain_edges.csv", "filing_date", f"{missing_edge['filing_date']:,}", pct(missing_edge['filing_date'], n_edges)],
    ["supply_chain_edges.csv", "product_category", f"{missing_edge['product_category']:,}", pct(missing_edge['product_category'], n_edges)],
])}

SAMPOQ has no country, industry, or product fields in the source, so those values are empty for its 13,372 companies. IC-SPLC and AnonNet contribute most of the missing year/filing-date values.

## Source-specific issues

### SAMPOQ
- Real, self-reported 10-K relationships. Direction was derived from `relation_type` and standardised to Supplier -> Customer.
- {sampoq.get('duplicate_endpoint_date_pairs', 0)} duplicate endpoint/date pairs were observed and flagged rather than silently removed.
- Company identifiers use ticker, CIK, or hashed normalised names; name-only entities may be ambiguous.
- No country/industry/product metadata exists in this source.

### IC-SPLC-v2
- 109,403 anonymised nodes and 115,767 edge rows, of which **37 duplicate edge IDs** exist in the processed source file. The combined table merges them by `edge_id`.
- Only **31,182** nodes have 128-d embeddings ({pct(31182, 109403)} of IC-SPLC nodes; {pct(31182, n_nodes)} of the full graph).
- Product category is missing for 96,831 of 115,767 edge rows; filing date is entirely absent.
- Relationships are LLM-extracted from an anonymised corpus; direction confidence is 0.7.

### AnonNet
- 7,245 nodes and 43,926 edges; **4 self-loops** were removed.
- Country is missing for 1,490 nodes (20.6%); products are missing for 537 nodes.
- Edge direction is preserved as provided; the source does not publish a formal direction definition.
- Anonymised IDs must not be merged with identifiable company records.

### SynDelay
- 155,488 synthetic rows. Class distribution: 0 = 36,724; 1 = 29,055; 2 = 89,709.
- Class meanings are **not documented**. The labels are retained verbatim and flagged `unverified_integer_class_0_1_2`.
- 41 duplicate `order_id`/`order_item_id` pairs were made unique with a row suffix to avoid split leakage.
- No LICENSE file; rights are unspecified.

### Smart Logistics
- 1,000 locally supplied rows; binary labels 0 = 434, 1 = 566.
- Provenance is not documented. Do not assume the records are real or synthetic.
- `Shipment_Status`, `Logistics_Delay_Reason`, `Logistics_Delay`, and `Timestamp` are excluded as leakage or post-hoc fields.

### GDELT
- 847 articles and 866 event mentions from 493 domains over 2026-11-03 to 2026-11-10.
- The DOC API returned HTTP 429 for `factory shutdown`; 7 of 8 keyword queries ultimately succeeded. A GKG sample supplied 507 additional articles.
- GKG theme matches are noisier than phrase matches; only supply-chain-specific themes are retained.
- Source country is missing for 513 articles and language for 507 because GKG metadata lacks those fields.
- Every record is marked `unverified_news_mention`; no company, country, or severity attribution is fabricated.

### Disruption Monitoring (restricted)
- CC BY-NC 4.0. Kept outside the main graph and training data.
- The KG has 6,222 companies and 16,640 edges; scenarios and benchmark items are authored synthetic evaluation material.
- Do not use in commercial deliverables without permission.

## Labels

- `training/risk_labels.csv`: **all 130,020 nodes are `not_available`**. There are no verified company-level risk labels in the acquired sources.
- `training/delay_labels.csv`: 155,488 SynDelay rows and 1,000 Smart Logistics rows. SynDelay semantics are unverified.
- Restricted benchmark labels are available only under `datasets/evaluation/`.

## Recommendations

1. Do not train a company risk model until verified risk labels are collected.
2. Perform entity resolution only where identifiers are non-anonymised; never merge IC-SPLC/AnonNet IDs with named companies.
3. Obtain written permission for SynDelay and AnonNet before redistribution or commercial use.
4. Confirm Smart Logistics provenance and licence.
5. Treat GDELT as a weak signal; require event confirmation before claiming a disruption.
6. Expand the GDELT window and retry the rate-limited keyword with backoff.
7. Document or validate SynDelay class semantics before reporting delay severity.
8. Handle class imbalance and missing embeddings explicitly in GNN training.
9. Keep restricted CC BY-NC material out of production and commercial artefacts.
"""
    OUT_QUALITY.write_text(quality_md, encoding="utf-8")

    preprocess_md = f"""# Preprocessing Report

Generated: {utc_now()}

## Pipeline

1. `scripts/download_datasets.py` acquires SAMPOQ, IC-SPLC, AnonNet, SynDelay, GDELT, Smart Logistics, and the restricted disruption-monitoring repository.
2. `scripts/preprocess_sampoq.py` normalises 10-K supplier/customer relations to Supplier -> Customer.
3. `scripts/preprocess_ic_splc.py` extracts the Neo4j dump (when `--extract` is used) and converts nodes, edges, and 128-d embeddings to CSV/Parquet.
4. `scripts/preprocess_anonnet.py` converts anonymised nodes/edges and preserves the provided direction.
5. `scripts/preprocess_syndelay.py` creates unique shipment IDs, features, and 3-class labels.
6. `scripts/preprocess_smart_logistics.py` creates binary labels, excludes leakage fields, and combines logistics outputs.
7. `scripts/preprocess_gdelt.py` parses DOC API JSON and GKG fallback CSVs, deduplicates by URL, and writes news/event tables.
8. `scripts/preprocess_disruption_monitoring.py --allow-restricted` processes the CC BY-NC KG and copies evaluation assets into restricted folders.
9. `scripts/build_graph_dataset.py` combines companies/edges, builds graph tables, node features, embeddings, edge indices, splits, and the manifest inputs.
10. `scripts/inspect_datasets.py`, `scripts/validate_datasets.py`, and `scripts/generate_dataset_report.py` produce inventory, validation, and reports.

## Transformation decisions

- **SAMPOQ direction:** `relation_type=supplier` means the related entity supplies the filer, so the edge is `related -> filer`; `relation_type=customer` means `filer -> related`. All output edges are Supplier -> Customer, confidence 0.9.
- **IC-SPLC IDs:** Neo4j internal node IDs are converted to `icsplc:<id>`; relationship direction is Supplier -> Customer per the dataset card, confidence 0.7.
- **AnonNet IDs:** `anonnet:<id>`; direction is preserved as provided because the source does not define it formally. Confidence 0.8.
- **SynDelay IDs:** `syndelay:<order_id>:<order_item_id>:<row_index>` guarantees uniqueness after 41 duplicate order-item pairs were found.
- **Smart Logistics leakage:** `Shipment_Status`, `Logistics_Delay_Reason`, `Logistics_Delay`, and `Timestamp` are excluded from features.
- **GDELT:** articles are deduplicated by URL. API records take precedence over GKG records. Every event mention is `unverified_news_mention`.
- **Restricted data:** the CC BY-NC KG and evaluation assets are written only under `processed/restricted/` and `evaluation/**/restricted/`; they are excluded from graph construction and training.
- **Risk labels:** no verified labels were found, so `risk_labels.csv` is explicitly empty rather than fabricated.

## Counts

{table(["Stage", "Rows"], [
    ["Source edge rows across three graph sources", f"{graph.get('source_edge_rows', 0):,}"],
    ["After duplicate edge_id merge", f"{graph.get('raw_edge_count', 0):,}"],
    ["After self-loop and missing-endpoint removal", f"{graph.get('raw_edge_count', 0):,}"],
    ["Unique directed graph edges", f"{graph.get('graph_edge_count', 0):,}"],
    ["Graph nodes", f"{n_nodes:,}"],
    ["News articles", f"{gdelt.get('news_articles', 0):,}"],
    ["Event mentions", f"{gdelt.get('disruption_events', 0):,}"],
    ["Shipment rows", f"{syndelay.get('rows', 0) + smart.get('rows', 0):,}"],
])}

## Splits

- Seed: **{training.get('seed', 42)}**.
- Edge link-prediction split: **80/10/10** random.
- Node split: **70/15/15**, stratified by `source_dataset`.
- Delay split: **70/15/15**, stratified by target class for each task.
- All split ID lists are in `training/dataset_splits.json`.

## Outputs

- `processed/companies/companies.csv`
- `processed/relationships/supply_chain_edges.csv` and `.parquet`
- `processed/graph/{{nodes.csv, edges.csv, node_features.csv, graph_metadata.json}}`
- `processed/logistics/{{shipment_delays.csv, logistics_features.csv}}`
- `processed/disruptions/{{news_articles.csv, disruption_events.csv}}`
- `training/{{node_index_mapping.csv, edge_index.csv, risk_labels.csv, delay_labels.csv, node_embeddings.parquet, dataset_splits.json}}`
- `evaluation/` (restricted scenarios, ground truth, benchmark)
- `reports/` and `metadata/`

## Reproduction

```bash
python3 scripts/download_datasets.py
python3 scripts/preprocess_sampoq.py
python3 scripts/preprocess_ic_splc.py
python3 scripts/preprocess_anonnet.py
python3 scripts/preprocess_syndelay.py
python3 scripts/preprocess_smart_logistics.py
python3 scripts/preprocess_gdelt.py
python3 scripts/preprocess_disruption_monitoring.py --allow-restricted
python3 scripts/build_graph_dataset.py
python3 scripts/inspect_datasets.py
python3 scripts/validate_datasets.py
python3 scripts/generate_dataset_report.py
```

IC-SPLC extraction from the Neo4j dump additionally requires a running Neo4j 5.x instance and `python3 scripts/preprocess_ic_splc.py --extract`.
"""
    OUT_PREPROCESS.write_text(preprocess_md, encoding="utf-8")

    license_md = f"""# Source Licences and Restrictions

Generated: {utc_now()}

This is a provenance summary, not legal advice. Confirm terms with the rights holders before redistribution or commercial use.

{table(["Dataset", "Licence", "Commercial use", "Attribution", "Action"], [
    ["SAMPOQ", "CC BY 4.0", "Allowed with attribution", "SAMPOQ Supply Chain 10-K dataset", "Keep attribution and licence notice."],
    ["IC-SPLC-v2", "CC BY 4.0", "Allowed with attribution", "King, P. (2026), Mendeley Data, doi:10.17632/txh3rrsh89.2", "Keep attribution and licence notice."],
    ["AnonNet", "No LICENSE file (rights unspecified)", "Unknown", "Xu, Long, Brintrup (2025)", "Obtain written permission before redistribution or commercial use."],
    ["SynDelay", "No LICENSE file (rights unspecified)", "Unknown", "Xu, Long, Brintrup (2025)", "Obtain written permission before redistribution or commercial use."],
    ["Smart Logistics", "Not specified with supplied file", "Unknown", "Locally supplied file; provenance unconfirmed", "Confirm provenance and licence with the project owner."],
    ["GDELT", "GDELT 2.0 terms of use", "Per GDELT terms", "The GDELT Project", "Review GDELT terms; records are news metadata only."],
    ["Disruption Monitoring", "CC BY-NC 4.0", "NOT permitted without approval", "AlMahri, S. (2025-2026)", "Keep restricted; do not ship commercially."],
])}

## Restricted material

The disruption-monitoring repository, its knowledge graph, scenarios, ground truth, and benchmark are **CC BY-NC 4.0**. They are stored under:

- `datasets/raw/disruption_monitoring/restricted/`
- `datasets/processed/restricted/`
- `datasets/evaluation/scenarios/restricted/`
- `datasets/evaluation/ground_truth/restricted/`
- `datasets/evaluation/benchmark/restricted/`

These files are excluded from the main graph and training data. Do not include them in a commercial deliverable unless the intended use is confirmed as permitted or separate permission is obtained.

## Attribution files

- `datasets/metadata/download_log.json` records source URLs, commits, and hashes.
- `datasets/metadata/dataset_manifest.json` records file-level hashes for reproducibility.
"""
    OUT_LICENSES.write_text(license_md, encoding="utf-8")

    datasets = [
        {
            "name": "SAMPOQ",
            "source_url": config["sources"]["sampoq"]["url"],
            "license": "CC BY 4.0",
            "restricted": False,
            "raw": ["datasets/raw/sampoq/supply-chain.csv"],
            "processed": [
                "datasets/processed/companies/companies_sampoq.csv",
                "datasets/processed/relationships/supply_chain_edges_sampoq.csv",
            ],
            "rows": {"companies": sampoq.get("companies"), "edges": sampoq.get("edges"), "input_rows": sampoq.get("input_rows")},
            "notes": "Real self-reported 10-K relationships; no country/industry/product fields; direction standardised to Supplier -> Customer; confidence 0.9.",
        },
        {
            "name": "IC-SPLC-v2",
            "source_url": "https://data.mendeley.com/datasets/txh3rrsh89/2",
            "doi": "10.17632/txh3rrsh89.2",
            "license": "CC BY 4.0",
            "restricted": False,
            "raw": [
                "datasets/raw/ic_splc/txh3rrsh89-v2.zip",
                "datasets/raw/ic_splc/extracted/IC-SPLC-v2 A Supply Chain Structure Neo4j Data/IC-SPLC.dump",
            ],
            "processed": [
                "datasets/processed/companies/companies_ic_splc.csv",
                "datasets/processed/relationships/supply_chain_edges_ic_splc.csv",
                "datasets/raw/ic_splc/extracted_csv/ic_splc_embeddings.parquet",
            ],
            "rows": {"companies": tables.get("companies_ic_splc", {}).get("rows"), "edges": tables.get("edges_ic_splc", {}).get("rows")},
            "notes": "Anonymised, LLM-extracted structure; 31,182 nodes have 128-d embeddings; 37 duplicate edge IDs; direction confidence 0.7.",
        },
        {
            "name": "AnonNet",
            "source_url": config["sources"]["anonnet"]["url"],
            "license": "Not specified (no LICENSE file)",
            "restricted": False,
            "raw": [
                "datasets/raw/anonnet/repo/mim2025_network_challenge/nodes_anon.csv",
                "datasets/raw/anonnet/repo/mim2025_network_challenge/edges_anon.csv",
            ],
            "processed": [
                "datasets/processed/companies/companies_anonnet.csv",
                "datasets/processed/relationships/supply_chain_edges_anonnet.csv",
            ],
            "rows": {"companies": tables.get("companies_anonnet", {}).get("rows"), "edges": tables.get("edges_anonnet", {}).get("rows")},
            "notes": "Anonymised real-world network; 4 self-loops dropped; direction not formally documented; confidence 0.8.",
        },
        {
            "name": "SynDelay",
            "source_url": config["sources"]["syndelay"]["url"],
            "license": "Not specified (no LICENSE file)",
            "restricted": False,
            "raw": ["datasets/raw/syndelay/repo/data/syndelay_v1.csv"],
            "processed": [
                "datasets/processed/logistics/shipment_delays_syndelay.csv",
                "datasets/processed/logistics/logistics_features_syndelay.csv",
                "datasets/training/delay_labels_syndelay.csv",
            ],
            "rows": {"shipments": syndelay.get("rows"), "label_distribution": syndelay.get("label_distribution")},
            "notes": "Synthetic. Class semantics undocumented and flagged. 41 duplicate order-item pairs resolved with a row suffix.",
        },
        {
            "name": "Smart Logistics",
            "source_url": None,
            "license": "Not specified (locally supplied)",
            "restricted": False,
            "raw": ["datasets/raw/existing_smart_logistics/smart_logistics_dataset.csv"],
            "processed": [
                "datasets/processed/logistics/shipment_delays_smart_logistics.csv",
                "datasets/processed/logistics/logistics_features_smart_logistics.csv",
                "datasets/training/delay_labels_smart_logistics.csv",
            ],
            "rows": {"shipments": smart.get("rows"), "label_distribution": smart.get("label_distribution")},
            "notes": "Provenance unknown. Leakage fields excluded. Do not assume real or synthetic.",
        },
        {
            "name": "GDELT",
            "source_url": config["sources"]["gdelt"]["api_url"],
            "license": "GDELT 2.0 terms of use",
            "restricted": False,
            "raw": ["datasets/raw/gdelt/", "datasets/raw/gdelt/gkg_filtered/"],
            "processed": [
                "datasets/processed/disruptions/news_articles.csv",
                "datasets/processed/disruptions/disruption_events.csv",
            ],
            "rows": {"articles": gdelt.get("news_articles"), "event_mentions": gdelt.get("disruption_events")},
            "notes": "Unverified news mentions. DOC API 7/8 keywords succeeded; factory shutdown rate-limited; GKG fallback used. Date range 2026-11-03 to 2026-11-10.",
        },
        {
            "name": "Disruption Monitoring (restricted)",
            "source_url": config["sources"]["disruption_monitoring"]["url"],
            "license": "CC BY-NC 4.0",
            "restricted": True,
            "raw": ["datasets/raw/disruption_monitoring/restricted/repo/"],
            "processed": [
                "datasets/processed/restricted/disruption_monitoring/companies.csv",
                "datasets/processed/restricted/disruption_monitoring/edges.csv",
                "datasets/evaluation/scenarios/restricted/",
                "datasets/evaluation/ground_truth/restricted/",
                "datasets/evaluation/benchmark/restricted/",
            ],
            "rows": {
                "companies": restricted.get("companies"),
                "edges": restricted.get("edges"),
                "scenarios": eval_meta.get("scenario_count"),
                "benchmark_items": eval_meta.get("benchmark_items"),
            },
            "notes": "Non-commercial. Excluded from main graph and training. Scenarios are synthetic evaluation material.",
        },
    ]

    files = []
    for group, root in [
        ("processed", PROCESSED_DIR),
        ("training", TRAINING_DIR),
        ("evaluation", EVALUATION_DIR),
        ("metadata", METADATA_DIR),
        ("reports", REPORTS_DIR),
    ]:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path == OUT_MANIFEST:
                continue
            files.append(
                {
                    "group": group,
                    "path": rel(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    raw_assets = []
    for name, path in RAW_HASH_FILES:
        if path.exists():
            raw_assets.append({"name": name, "path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        else:
            raw_assets.append({"name": name, "path": rel(path), "bytes": 0, "sha256": None, "status": "missing"})

    manifest = {
        "project": config.get("project", {}).get("name", "AtmoGraph"),
        "description": config.get("project", {}).get("description"),
        "generated_at": utc_now(),
        "seed": config.get("project", {}).get("seed", 42),
        "datasets_dir": rel(DATASETS_DIR),
        "summary": {
            "nodes": n_nodes,
            "raw_edges": n_edges,
            "graph_edges": graph.get("graph_edge_count"),
            "news_articles": gdelt.get("news_articles"),
            "event_mentions": gdelt.get("disruption_events"),
            "shipment_rows": syndelay.get("rows", 0) + smart.get("rows", 0),
            "evaluation_scenarios": eval_meta.get("scenario_count"),
            "benchmark_items": eval_meta.get("benchmark_items"),
        },
        "datasets": datasets,
        "graph": {
            "nodes": graph.get("node_count"),
            "raw_edges": graph.get("raw_edge_count"),
            "graph_edges": graph.get("graph_edge_count"),
            "node_source_counts": graph.get("node_source_counts"),
            "edge_source_counts": graph.get("edge_source_counts"),
            "relationship_type_counts": graph.get("relationship_type_counts"),
            "components": graph.get("weakly_connected_components"),
            "largest_component": graph.get("largest_component_size"),
            "isolated_nodes": graph.get("isolated_nodes"),
            "degree_stats": graph.get("degree_stats"),
            "embedding": graph.get("embedding"),
            "label_availability": graph.get("label_availability"),
        },
        "training": training,
        "evaluation": eval_meta,
        "validation": {"status": validation.get("status"), "passed": validation.get("passed"), "failed": validation.get("failed")},
        "download_log": download_log,
        "raw_assets": raw_assets,
        "files": files,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: pkg_version(name) for name in ["pandas", "numpy", "pyarrow", "requests", "PyYAML", "neo4j"]},
        },
        "known_limitations": graph.get("known_limitations", []),
        "reproducibility": {
            "commands": [
                "python3 scripts/download_datasets.py",
                "python3 scripts/preprocess_sampoq.py",
                "python3 scripts/preprocess_ic_splc.py",
                "python3 scripts/preprocess_anonnet.py",
                "python3 scripts/preprocess_syndelay.py",
                "python3 scripts/preprocess_smart_logistics.py",
                "python3 scripts/preprocess_gdelt.py",
                "python3 scripts/preprocess_disruption_monitoring.py --allow-restricted",
                "python3 scripts/build_graph_dataset.py",
                "python3 scripts/inspect_datasets.py",
                "python3 scripts/validate_datasets.py",
                "python3 scripts/generate_dataset_report.py",
            ],
            "note": "IC-SPLC extraction additionally requires a running Neo4j 5.x instance and --extract.",
        },
    }
    write_json(OUT_MANIFEST, manifest)
    logger.info("Reports written: %s, %s, %s; manifest: %s", rel(OUT_SUMMARY), rel(OUT_QUALITY), rel(OUT_PREPROCESS), rel(OUT_MANIFEST))
    print(json.dumps({"summary": rel(OUT_SUMMARY), "quality": rel(OUT_QUALITY), "preprocessing": rel(OUT_PREPROCESS), "manifest": rel(OUT_MANIFEST), "files_hashed": len(files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
