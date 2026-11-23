# Preprocessing Report

Generated: 2026-11-10T05:42:28Z

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

| Stage | Rows |
|---|---|
| Source edge rows across three graph sources | 175,791 |
| After duplicate edge_id merge | 175,754 |
| After self-loop and missing-endpoint removal | 175,754 |
| Unique directed graph edges | 166,690 |
| Graph nodes | 130,020 |
| News articles | 847 |
| Event mentions | 866 |
| Shipment rows | 156,488 |

## Splits

- Seed: **42**.
- Edge link-prediction split: **80/10/10** random.
- Node split: **70/15/15**, stratified by `source_dataset`.
- Delay split: **70/15/15**, stratified by target class for each task.
- All split ID lists are in `training/dataset_splits.json`.

## Outputs

- `processed/companies/companies.csv`
- `processed/relationships/supply_chain_edges.csv` and `.parquet`
- `processed/graph/{nodes.csv, edges.csv, node_features.csv, graph_metadata.json}`
- `processed/logistics/{shipment_delays.csv, logistics_features.csv}`
- `processed/disruptions/{news_articles.csv, disruption_events.csv}`
- `training/{node_index_mapping.csv, edge_index.csv, risk_labels.csv, delay_labels.csv, node_embeddings.parquet, dataset_splits.json}`
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
