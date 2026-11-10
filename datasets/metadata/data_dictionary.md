# Data Dictionary

Generated for the AtmoGraph dataset package. All identifiers are strings unless stated otherwise. Empty fields mean "not provided by the source"; they are not imputed.

## `processed/companies/companies.csv`

One row per company/entity across SAMPOQ, IC-SPLC, and AnonNet.

| Column | Type | Description |
|---|---|---|
| `company_id` | string | Stable namespaced ID: `sampoq:<id>`, `icsplc:<neo4j_id>`, or `anonnet:<id>`. |
| `company_name` | string | Display name. Anonymised entities use `AnonNet entity <id>` or an IC-SPLC placeholder. |
| `original_company_name` | string | Source name when available; null for anonymised sources. |
| `ticker` | string | Ticker when available (mostly SAMPOQ). |
| `country` | string | Country when provided. Missing for all SAMPOQ rows and some IC-SPLC/AnonNet rows. |
| `industry` | string | Primary industry when provided. |
| `product_category` | string | Primary product/category when provided. |
| `source_dataset` | string | `sampoq`, `ic_splc`, or `anonnet`. |
| `source_record_id` | string | Original source identifier or row reference. |
| `name_is_anonymised` | boolean-like | `True` for IC-SPLC/AnonNet entities. |
| `id_method` | string | How the ID was derived (`ticker`, `cik`, `normalised_name_hash`, `source_anonymised_id`). |
| `cik` | string | SEC CIK for SAMPOQ entities when resolved. |
| `name_variants` | string | SAMPOQ name variants separated by a double pipe. |
| `mention_count` | integer | Number of SAMPOQ source rows mentioning the entity. |
| `first_seen_filing_date` / `last_seen_filing_date` | date | SAMPOQ filing-date range for the entity. |
| `industry_2nd`, `category_2nd`, `category_3rd` | string | IC-SPLC secondary classifications. |
| `other_possible_industry_2`, `other_possible_category_3rd` | string | IC-SPLC alternative labels. |
| `embedding_available` | boolean-like | Whether a 128-d IC-SPLC embedding exists. |
| `products` | string | AnonNet product list (comma-separated). |
| `product_count` | integer | Number of AnonNet products. |

## `processed/relationships/supply_chain_edges.csv`

All relationship rows before graph-level duplicate collapse.

| Column | Type | Description |
|---|---|---|
| `edge_id` | string | Stable edge ID. |
| `source_company_id` | string | Supplier-side company ID. |
| `target_company_id` | string | Customer-side company ID. |
| `relationship_type` | string | `supplier_to_customer`, `supplies_product_to`, or `supply_relationship_anonymised`. |
| `product_category` | string | Product/category for the relationship when available. |
| `year` | integer | Relationship year when available. |
| `filing_date` | date | Filing date for SAMPOQ edges. |
| `source_dataset` | string | Source dataset. |
| `source_record_id` | string | Source row/relationship identifier. |
| `confidence` | float | 0.9 SAMPOQ, 0.8 AnonNet, 0.7 IC-SPLC. Reflects extraction confidence, not verified ground truth. |
| `relationship_direction` | string | Direction statement; see source notes. |
| `source_relation_type` | string | SAMPOQ original `supplier`/`customer` label. |
| `source_ticker`, `related_ticker` | string | SAMPOQ tickers. |
| `filing_accession` | string | SAMPOQ SEC accession number. |
| `duplicate_relationship` | boolean-like | SAMPOQ duplicate endpoint/date flag. |

## `processed/graph/nodes.csv`

Company table plus graph structural fields.

Additional columns: `node_type` (`company`), `degree`, `in_degree`, `out_degree`, `raw_degree`, `component_id`, `component_size`.

## `processed/graph/edges.csv`

Deduplicated directed graph edges.

| Column | Type | Description |
|---|---|---|
| `edge_id` | string | Graph edge ID derived from source, target, and relationship type. |
| `source_company_id` | string | Supplier-side node. |
| `target_company_id` | string | Customer-side node. |
| `relationship_type` | string | Relationship type. |
| `edge_count` | integer | Number of raw rows collapsed into this edge. |
| `source_datasets` | string | Semicolon-separated source datasets contributing to the edge. |
| `first_year` / `last_year` | integer | Observed year range. |
| `confidence` | float | Maximum confidence among contributing rows. |

## `processed/graph/node_features.csv`

Model-ready node features. No labels are included.

| Column | Description |
|---|---|
| `node_id` | Company ID. |
| `node_type`, `node_type_code` | Entity type; currently always `company` / `0`. |
| `degree`, `in_degree`, `out_degree` | Degree on unique graph edges. |
| `raw_degree` | Degree counting raw relationship rows. |
| `component_id`, `component_size` | Weakly connected component assignment and size. |
| `is_isolated` | 1 if degree is zero. |
| `source_sampoq`, `source_ic_splc`, `source_anonnet` | One-hot source flags. |
| `country_code`, `industry_code`, `product_category_code` | Integer category codes; `-1` = missing. Mappings in `training/category_mappings.json`. |
| `has_country`, `has_industry`, `has_product_category` | Missingness indicators. |
| `embedding_available` | 1 if a 128-d IC-SPLC embedding exists. |
| `first_edge_year`, `last_edge_year` | Observed edge-year range. |
| `is_anonymised` | 1 for anonymised entities. |

## `training/node_features.csv`

Identical to `processed/graph/node_features.csv`; provided at the training path expected by model code.

## `training/node_index_mapping.csv`

`node_id` -> contiguous `node_index` (0..N-1), sorted by `node_id`.

## `training/edge_index.csv`

PyTorch-Geometric-ready directed edges.

| Column | Description |
|---|---|
| `edge_id` | Graph edge ID. |
| `source_index`, `target_index` | Integer node indices. |
| `source_company_id`, `target_company_id` | Original node IDs. |
| `relationship_type` | Relationship type. |
| `edge_count` | Raw-row multiplicity. |
| `split` | `train`, `val`, or `test` (random 80/10/10). |

## `training/risk_labels.csv`

Company risk labels. **Currently unavailable.**

| Column | Description |
|---|---|
| `node_id` | Company ID. |
| `risk_label` | Empty. |
| `label_status` | Always `not_available`. |
| `label_source` | Always `none`. |
| `label_confidence` | Empty. |
| `notes` | Explains that no verified labels exist and must not be fabricated. |

## `training/delay_labels.csv`

Shipment-level classification targets.

| Column | Description |
|---|---|
| `shipment_id` | Unique shipment ID. |
| `task` | `delay_classification_3class` (SynDelay) or `delay_classification_binary` (Smart Logistics). |
| `target_name` | `delay_class` or `logistics_delay`. |
| `target_value` | Class value: 0/1/2 for SynDelay, 0/1 for Smart Logistics. |
| `target_type` | `classification`. |
| `is_synthetic` | 1 for SynDelay; empty for Smart Logistics. |
| `label_semantics` | `unverified_integer_class_0_1_2` or `binary_Logistics_Delay_0_1`. |
| `source_dataset` | `syndelay` or `smart_logistics`. |

**Warning:** SynDelay class meanings are undocumented. Do not describe classes as low/medium/high delay without external validation.

## `training/node_embeddings.parquet`

Full node-by-embedding matrix with explicit mask.

| Column | Description |
|---|---|
| `node_id` | Company ID. |
| `node_index` | Integer index. |
| `embedding_available` | Boolean mask. |
| `emb_0` .. `emb_127` | 128-d IC-SPLC embedding. Missing rows are zero-filled; use the mask. |

## `training/dataset_splits.json`

Reproducible split ID lists.

```json
{
  "seed": 42,
  "node_splits": {"train": [...], "val": [...], "test": [...]},
  "edge_splits": {"train": [...], "val": [...], "test": [...]},
  "delay_splits": {
    "delay_classification_3class": {"train": [...], "val": [...], "test": [...]},
    "delay_classification_binary": {"train": [...], "val": [...], "test": [...]}
  }
}
```

Node splits are stratified by `source_dataset`; delay splits are stratified by target class.

## `processed/logistics/shipment_delays.csv`

Combined shipment records.

| Column | Description |
|---|---|
| `shipment_id` | Unique shipment ID. |
| `asset_id` | Product/asset identifier. |
| `timestamp` | Shipment timestamp (not for advance prediction). |
| `latitude`, `longitude` | Location. |
| `shipment_status` | Source status. Excluded from features. |
| `traffic_status`, `waiting_time` | Smart Logistics fields; empty for SynDelay. |
| `logistics_delay` | Binary target for Smart Logistics; empty for SynDelay. |
| `logistics_delay_reason` | Post-hoc reason; excluded from features. |
| `source_dataset` | `syndelay` or `smart_logistics`. |
| `is_synthetic` | 1 for SynDelay. |
| `data_origin` | `synthetic` or `local_supplied_unverified`. |
| `delay_class` | SynDelay 0/1/2 class. |
| `label_semantics` | Label interpretation warning. |

## `processed/logistics/logistics_features.csv`

Model-ready shipment features. Leakage columns (`shipment_status`, `logistics_delay`, `logistics_delay_reason`, `timestamp`) are excluded.

Columns: `shipment_id`, `source_dataset`, `latitude`, `longitude`, `waiting_time`, `inventory_level`, `temperature`, `humidity`, `asset_utilization`, `demand_forecast`, `traffic_status_code`, `asset_id_code`, `is_synthetic`, `data_origin`.

## `processed/disruptions/news_articles.csv`

Deduplicated news metadata.

| Column | Description |
|---|---|
| `article_id` | Stable ID derived from URL. |
| `title` | Article title. |
| `url` | Article URL. |
| `domain` | Publisher domain. |
| `seendate` | ISO timestamp when available. |
| `language` | Language when available. |
| `source_country` | Publication country when available. |
| `query_keywords` | Semicolon-separated matched keywords. |
| `collection_method` | `gdelt_doc_api` or `gdelt_gkg_sample`. |
| `source_dataset` | `gdelt`. |
| `collected_at` | Collection timestamp. |

## `processed/disruptions/disruption_events.csv`

Keyword/theme-level event mentions. **Not confirmed disruptions.**

| Column | Description |
|---|---|
| `disruption_id` | Stable mention ID. |
| `event_title` | Article title. |
| `event_description` | Empty; no full text is stored. |
| `disruption_type` | Keyword/theme-derived type. |
| `affected_company` | Empty; no company attribution is fabricated. |
| `affected_country`, `affected_region` | Empty; not inferred. |
| `event_date` | Article date. |
| `source_url` | Article URL. |
| `source_dataset` | `gdelt`. |
| `extraction_method` | `doc_api_keyword_query` or `gkg_keyword_theme_rule`. |
| `confidence` | 0.4 API keyword, 0.35 GKG phrase, 0.2 GKG theme. |
| `verification_status` | Always `unverified_news_mention`. |
| `publication_country` | Publisher country when available. |
| `mentioned_locations`, `matched_keywords` | GKG locations/keywords. |
| `article_id` | Foreign key to `news_articles.csv`. |

## Restricted tables

`processed/restricted/disruption_monitoring/companies.csv` and `edges.csv` mirror the main company/edge schemas but use `disruption:<slug>_<hash>` IDs and `source_dataset=disruption_monitoring_restricted`. They are **excluded** from the main graph and training data. Licence: CC BY-NC 4.0.

## Evaluation assets

`evaluation/evaluation_metadata.json` describes the restricted scenarios, ground truth, and benchmark:

- `scenario_count`: 30 synthetic news scenarios.
- `scenario_ids`: scenario identifiers such as `TSLA_SC001`.
- `ground_truth_files`: ground-truth scenario files.
- `benchmark_items`: 1,117 relevance items.
- `license`: CC BY-NC 4.0.
- `included_in_main_graph`: `false`.

## Manifest and provenance

- `metadata/download_log.json`: acquisition log with URLs, commits, hashes, and status.
- `metadata/dataset_manifest.json`: file-level hashes, source licences, row counts, environment, and reproduction commands.
- `metadata/source_licenses.md`: licence summary and restricted-material notice.
- `reports/dataset_inventory.json`: per-table row counts, columns, missingness, and duplicate keys.
- `reports/validation_results.json`: 40 automated integrity checks.
