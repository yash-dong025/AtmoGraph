# Phase 4A Dataset Investigation

**Date:** 2026-11-21  
**Repository:** `/Users/yash/Desktop/Infotact`  
**Branch:** `Yash` (tracking `origin/Yash`)  
**Last commit at investigation start:** `607f43b Update project documentation and add Phase 3 cleanup report`  
**Dataset version:** `atmo_graph_v1_015d0c924af0`  
**Task under audit:** directed supplier-to-customer link prediction  
**Scope:** Goal A / Phase 4A read-only investigation. This report does not train, retrain, preprocess, re-split, regenerate, modify, delete, or commit any dataset, checkpoint, metric, prediction, model, dashboard, or experiment artifact.

**Important boundary:** This report is about link prediction. It is not about company disruption risk, shipment delay, causal propagation, or financial distress. No verified company-level disruption labels exist in the acquired data.

---

## 1. Executive Summary

### Verdict

Goal A is complete with documented caveats.

The feature collapse is real, but it is not one bug. It is the combined result of:

1. a deliberately narrow 136-dimensional model contract,
2. source-level missing attributes in SAMPOQ,
3. an engineering failure to use available AnonNet categorical values,
4. partial and non-random IC-SPLC embedding coverage,
5. unused categorical and edge features,
6. source heterogeneity that makes a pooled headline misleading.

The original GraphSAGE checkpoint uses exactly:

- 8 static flags:
  - `source_sampoq`
  - `source_ic_splc`
  - `source_anonnet`
  - `has_country`
  - `has_industry`
  - `has_product_category`
  - `embedding_available`
  - `is_anonymised`
- 128 IC-SPLC embedding dimensions: `icsplc_emb_0` through `icsplc_emb_127`.

The full feature builder produces 146 dimensions: 18 numeric plus 128 embeddings. It also produces 3 categorical fields and 12 edge features. The model does not consume the categorical matrix or any edge feature.

### The three biggest findings

1. **SAMPOQ cannot be fixed by feature engineering alone.**  
   The raw SAMPOQ table has 9 columns and none contains country, industry, or product category. The preprocessor correctly writes empty values rather than inventing them. All 13,372 SAMPOQ nodes therefore have the same 136-dimensional model vector. The source lacks the discriminative attributes.

2. **AnonNet has useful attributes, but the model ignores them.**  
   Country exists for 79.43% of AnonNet nodes, product category for 92.59%, and product lists for 92.59%. The feature builder creates integer category codes, but the 136-dimensional model uses only presence flags. AnonNet collapses to 4 unique 136-dimensional rows. This is a model-feature-selection failure, not a source-data failure.

3. **IC-SPLC embeddings are genuinely partial and their missingness is not random.**  
   Only 31,182 of 109,403 IC-SPLC nodes have embeddings. Alignment is correct: no unmatched IDs, no false availability flags, and no unflagged rows. But embedding coverage rises sharply with degree: 23.33% at degree 1, 95.72% at degree 6-10, and 98.63% at degree 11+. Missing embeddings are zero-filled with an explicit mask; they are not evidence of a join bug.

### Recovery priority

**AnonNet is the first controlled feature-recovery target.** It has the richest unused attributes, requires no new data acquisition, and has a strong structural baseline: directed preferential attachment reached AnonNet AP 0.905, 0.908, and 0.904 under Phase 2 protocols A/B/C. The clean comparison is an AnonNet-only categorical model against the 4-row flags-only model and directed PA.

**SAMPOQ is primarily a data-acquisition problem.** Its one-row collapse cannot be repaired by adding architecture complexity. If defensible country/industry/product data cannot be acquired or linked, SAMPOQ should not drive pooled headline claims.

**IC-SPLC is the best temporal candidate, but not a clean category-recovery target yet.** Its edge years cover 2013-2025, while SAMPOQ is 92.0% 2026. Its categorical codes are anonymized and sparse; its 128-d embeddings are the main distinguishing signal, but their provenance is not fully established.

### What must not be claimed

- The current results do not show that GraphSAGE is unsuitable for supply-chain link prediction.
- The scores are link scores, not calibrated link probabilities.
- The scores are not disruption-risk scores.
- The balanced evaluation does not estimate production precision at realistic prevalence.
- The single-seed results do not establish statistically reliable improvement.
- Anonymized categories cannot be interpreted as named industries, countries, or products.

### Current operational result

The original artifact reports test ROC-AUC `0.6727`, AP `0.6094`, precision `0.5889`, recall `0.8924`, and F1 `0.7096`. Directed preferential attachment has lower AUC (`0.5226`) but higher AP (`0.6983`) and far better top-of-ranking precision. Those are different objectives. The original scores are severely overconfident: Brier `0.3354`, ECE `0.3253` at 10 bins.

---

## 2. Investigation Scope

### In scope

- SAMPOQ raw and processed nodes and edges.
- AnonNet raw and processed nodes and edges.
- IC-SPLC-v2 raw and processed nodes and edges, including embeddings.
- Combined graph construction, category coding, feature engineering, split construction, and model input selection.
- Original GraphSAGE experiment artifacts and Phase 2 ablations.
- Phase 3 inference, metadata, dashboard-catalog, and test compatibility.
- Existing reports, README files, data dictionary, manifest, tests, and generated artifacts.
- Evidence-driven root causes, leakage risks, recovery opportunities, and a gated Goal B roadmap.

### Out of scope

- Training, retraining, or hyperparameter tuning.
- Preprocessing or dataset regeneration.
- Feature implementation.
- Split modification.
- Checkpoint or prediction generation.
- Dashboard functionality changes.
- Entity resolution or external enrichment.
- Risk, delay, or causal modelling.
- Goal B implementation.

### Method

- Read raw and processed tables directly.
- Recompute counts, missingness, uniqueness, embedding alignment, and feature-collapse statistics.
- Trace each source through preprocessing, graph construction, feature engineering, `Data` construction, and model selection.
- Inspect original and Phase 2 experiment metrics and reports.
- Re-run the synthetic Phase 3 test suite with bytecode and pytest cache writes disabled where possible.
- Check Git state, tracked files, ignored artifacts, and repository-relative path handling.

No table, feature, split, model, or checkpoint was written. The only file intentionally updated is this report.

---

## 3. Repository and Artifact Inventory

### 3.1 Active code path

| Component | Path | Status | Purpose |
|---|---|---|---|
| Source preprocessing | `scripts/preprocess_sampoq.py` | Active | SAMPOQ nodes/edges |
| Source preprocessing | `scripts/preprocess_anonnet.py` | Active | AnonNet nodes/edges |
| Source preprocessing | `scripts/preprocess_ic_splc.py` | Active | IC-SPLC nodes/edges/embeddings |
| Graph build | `scripts/build_graph_dataset.py` | Active | Combined graph, categories, splits |
| Feature engineering | `ml/preprocessing/feature_engineering.py` | Active | 146-d node matrix, categories, embeddings, edge features |
| PyG assembly | `ml/preprocessing/graph_preprocessing.py` | Active | `Data` object with unused `x_cat`, `x_emb`, `emb_mask`, `edge_attr` |
| Training | `ml/training/train_graphsage_link_prediction.py` | Active | 136-d feature selection, GraphSAGE training |
| Evaluation | `ml/training/evaluate_link_prediction_baseline.py` | Active | Baselines, threshold, metrics |
| Phase 2 evaluation | `ml/training/phase2_utils.py` | Active | Protocols, calibration, P@K |
| Inference | `ml/inference/predict.py` | Active, Phase 3 | Link-prediction contract only |
| Dashboard data | `ml/reports/dashboard_data.py` | Active, Phase 3 | Experiment catalog and prediction records |
| Tests | `ml/tests/` | Active, Phase 3 | 19 synthetic/integration tests |

### 3.2 Data artifacts

| Artifact | Rows / shape | Notes |
|---|---:|---|
| `datasets/raw/sampoq/supply-chain.csv` | 16,102 x 9 | No country/industry/product |
| `datasets/raw/anonnet/repo/mim2025_network_challenge/nodes_anon.csv` | 7,245 | `location`, `products` |
| `datasets/raw/anonnet/repo/mim2025_network_challenge/edges_anon.csv` | 43,926 | No year |
| `datasets/raw/ic_splc/extracted_csv/ic_splc_nodes.csv` | 109,403 | Includes embeddings |
| `datasets/raw/ic_splc/extracted_csv/ic_splc_edges.csv` | 115,796 | 29 self-loops, 37 exact duplicates |
| `datasets/raw/ic_splc/extracted_csv/ic_splc_embeddings.parquet` | 31,182 x 128 | Only IC-SPLC embeddings |
| `datasets/processed/graph/nodes.csv` | 130,020 | Combined nodes |
| `datasets/processed/graph/edges.csv` | 166,690 | Deduplicated directed graph edges |
| `datasets/processed/graph/node_features.csv` | 130,020 x 23 | Numeric/flags/categories; no embeddings |
| `datasets/training/node_features.csv` | 130,020 x 23 | Byte-identical copy of processed graph node features |
| `datasets/training/node_embeddings.parquet` | 130,020 x 130 | 128-d block + IDs/mask |
| `datasets/training/node_index_mapping.csv` | 130,020 | Contiguous 0..130,019 |
| `datasets/training/edge_index.csv` | 166,690 | Random 80/10/10 split |
| `datasets/training/dataset_splits.json` | Node/edge/delay | Reproducible ID lists |
| `datasets/training/training_metadata.json` | Metadata | Feature, edge, split, label references |
| `datasets/metadata/dataset_manifest.json` | 91 listed files | Counts and SHA-256 hashes |
| `datasets/metadata/download_log.json` | Acquisition log | Contains machine-specific absolute paths |

The combined graph metadata records:

- `node_count`: 130,020
- `source_edge_rows`: 175,791
- `duplicate_edge_ids_merged`: 37
- `raw_edge_count`: 175,754
- `graph_edge_count`: 166,690
- `duplicate_edge_groups`: 9,064

The 9,064 graph-edge groups are a real information-loss step: multiple source rows are collapsed to one `(source, target, relationship)` edge, retaining only `edge_count`, `source_datasets`, `first_year`, `last_year`, and `confidence`.

### 3.3 Experiment and report artifacts

| Artifact | Status | Notes |
|---|---|---|
| `ml/experiments/graphsage_link_prediction/` | Active historical baseline | Checkpoint, metrics, predictions, config, history |
| `ml/experiments/graphsage_link_prediction/checkpoint.pt` | Ignored/untracked | 94,965 bytes, hash `7c1abf...e4de6` |
| `ml/experiments/graphsage_link_prediction/metrics.json` | Tracked | No calibration fields |
| `ml/experiments/graphsage_link_prediction/test_predictions.csv` | Tracked | 33,338 pairs |
| `ml/experiments/graphsage_phase2/` | Active Phase 2 | 204 files, including 40 plots and a log |
| `ml/experiments/graphsage_phase2/summary.json` | Tracked | 134 comparison rows |
| `ml/experiments/graphsage_phase2/logs/run.log` | Ignored | Contains absolute path |
| `ml/experiments/graphsage_phase2/metrics/A/original_graphsage_reference.json` | Tracked historical | Contains absolute path |
| `ml/reports/graphsage_diagnostic_audit.md` | Historical report | Original diagnostics |
| `ml/reports/graphsage_phase2_experiment_report.md` | Historical report | Protocols and ablations |
| `ml/reports/phase3_foundation_cleanup_report.md` | Historical report | Phase 3 changes and tests |
| `ml/reports/category_quality_report.md` | Generated report | Category provenance and coverage |
| `ml/reports/atmograph_dashboard.html` | Active UI | Dashboard |
| `ml/reports/week1_showcase.html` | Historical | Preserved, not runtime entry point |
| `datasets/reports/ml_dataset_analysis.*` | Historical analysis | Referenced by Week 1 showcase |
| `datasets/reports/data_quality_report.md` | Data quality report | Overlaps category report |

Repository counts:

- Tracked files under `ml/`: 155.
- Tracked files under `ml/experiments/`: 119.
- Files in `ml/experiments/graphsage_phase2/`: 204 (including ignored logs and plots).
- Phase 2 reliability/ranking plots: 40.
- Dataset manifest inventory: 91 files.

### 3.4 Duplicates, stale artifacts, and unversioned artifacts

| Item | Finding | Treatment |
|---|---|---|
| Node feature copies | `datasets/processed/graph/node_features.csv` and `datasets/training/node_features.csv` are byte-identical, SHA-256 `c2149400eeaf66070a4ebbeb06612d7451d509cae2a66f56ac586395c7a51ac3` | Intentional or redundant copy; do not silently diverge |
| Checkpoint | `ml/experiments/graphsage_link_prediction/checkpoint.pt` is ignored by `ml/.gitignore` (`*.pt`) | Required artifact is present but not versioned |
| Phase 2 logs | Ignored by root `*.log` rule | Useful historical evidence, not tracked |
| Raw data | Ignored by root `datasets/raw/` rule | Licences still apply |
| Phase 4A draft | This report existed as an untracked 11-section draft | Replaced by this 16-section report |
| Week 1 showcase | Historical-only HTML | Preserved; not part of runtime path |
| Dataset and ML reports | Several overlap on counts and quality | Useful, but documentation drift is present |
| Dataset manifest/download log | Generated metadata contains absolute acquisition paths | Not runtime paths; do not treat as portable metadata |

The checkpoint metadata was verified safely:

- `feature_names`: 136.
- `model_config`: `in_channels=136`, hidden 64, out 32, 2 layers, dropout 0.2, dot-product decoder.
- `threshold`: `0.11992187052965164`.
- `best_epoch`: `164`.
- `dataset_version`: `atmo_graph_v1_015d0c924af0`.
- No optimizer state, scheduler state, or RNG state.

---

## 4. Historical Problem Verification

### 4.1 Historical source and feature claims

| ID | Historical claim | Status | Evidence |
|---|---|---|---|
| A1 | The original model uses only 136 features: 8 flags + 128 embeddings | Confirmed | Checkpoint `feature_names`; `config.json`; `metrics.json` |
| A2 | SAMPOQ collapses to one unique 136-d row | Confirmed | Direct 136-d uniqueness calculation: 1 row for 13,372 nodes |
| A3 | AnonNet collapses to four unique 136-d rows | Confirmed | Direct calculation: 4 rows for 7,245 nodes |
| A4 | Only about 28.5% of IC-SPLC nodes have embeddings | Confirmed | 31,182 / 109,403 = 28.51% |
| A5 | IC-SPLC has more unique 136-d rows than embeddings | Confirmed | 31,187 unique rows vs 31,182 embeddings; flag patterns add 5 |
| A6 | Categorical codes are unused by the 136-d model | Confirmed | `x_cat` is built and stored; training selects `x` only |
| A7 | All 12 edge features are unused | Confirmed | `edge_attr` exists; no model consumes it |
| A8 | Full-graph structural features were excluded without a train-only replacement | Confirmed | `EXCLUDED_FEATURES`; Phase 2 c3/c5 are train-only but transductive |
| A9 | Source heterogeneity makes pooled metrics misleading | Confirmed | SAMPOQ AP ~0.50; AnonNet directed PA AP ~0.905; IC-SPLC differs by protocol |
| A10 | The graph is fragmented | Confirmed | 23,198 components; largest 52,473 nodes; 16 isolated nodes |

Additional verification:

- SAMPOQ raw has no country, industry, or product category columns.
- SAMPOQ preprocessor hardcodes `country=None`, `industry=None`, and `product_category=None`.
- AnonNet preprocessor preserves country and the first product, and preserves the full product list.
- AnonNet has no industry and no time field.
- IC-SPLC country, industry, and product values are anonymized frequency-ranked codes, not verified names.

### 4.2 Metric implementation and evaluation claims

| ID | Check | Status | Evidence |
|---|---|---|---|
| B1 | ROC-AUC, AP, precision, recall, F1, threshold, and confusion matrix are implemented | Confirmed | `binary_metrics()` in `ml/training/evaluate_link_prediction_baseline.py:254-270` |
| B2 | Threshold is selected on validation only | Confirmed | `threshold_from_validation()` at `:231-252`; Phase 2 uses validation-only fitting |
| B3 | Per-source metrics are implemented | Confirmed | `metrics_by_source()` at `:282`; Phase 2 per-source reporting |
| B4 | Precision@K is implemented | Confirmed | `precision_at_k()` in `ml/training/phase2_utils.py:756`; expected tie-breaking |
| B5 | Brier, log loss, ECE, and reliability bins are implemented for Phase 2 | Confirmed | `calibration_report()` at `phase2_utils.py:880-922`; 40 reliability SVGs |
| B6 | The original artifact has calibration metrics | Not present | `metrics.json` has no calibration block; Brier 0.3354/ECE 0.3253 are derived in Phase 2 |
| B7 | Multiple seeds and confidence intervals exist | Not present | Seed 42 only; Phase 2 explicitly states no CIs |
| B8 | The original AP is exactly reproducible | Partially confirmed | Reported AP 0.6094; checkpoint reference on Phase 2 pairs 0.5817; c1 retrain 0.5761. Negative realization differs |

Metric interpretation is confirmed:

- GraphSAGE has higher AUC but lower AP than directed PA.
- PA and CN dominate top-of-ranking precision.
- The original model is severely overconfident.
- Balanced 1:1 evaluation is not a production prevalence estimate.
- Pooled metrics mix source regimes and must not hide per-source results.

### 4.3 Phase 3 compatibility checks

| ID | Check | Status | Evidence |
|---|---|---|---|
| C1 | Link-prediction contract exists and is explicit | Confirmed | `ml/inference/predict.py` |
| C2 | Risk prediction is explicitly unsupported | Confirmed | `build_risk_prediction_record()` raises `RiskPredictionUnsupportedError` |
| C3 | Missing checkpoint has a clear error | Confirmed | `CheckpointUnavailableError` |
| C4 | Inference record has no risk fields | Confirmed | Tests assert no `risk_score`/`risk_level` |
| C5 | Dashboard catalog is additive and phase-aware | Confirmed | 135 entries: 1 original + 134 Phase 2 rows |
| C6 | Targeted metadata paths are repository-relative | Confirmed | `experiment.json`, `run_summary.json`, Phase 2 manifest |
| C7 | Existing test suite passes | Confirmed | 19 passed on current rerun |
| C8 | No retraining occurred | Confirmed | No model command run; artifact mtimes/hashes unchanged |
| C9 | Git has no unintended tracked changes | Confirmed | Only this report is untracked |
| C10 | Documentation path portability is complete | Not confirmed | Absolute paths remain in docs/log/reference and dataset metadata |
| C11 | Documentation category cardinalities are correct | Not confirmed | `ml/README.md` swaps country and industry cardinalities |
| C12 | Documentation duplicate-edge count is correct | Not confirmed | `ml/README.md:111` says 37 duplicate triples were removed; they remain in processed data |

### 4.4 Confirmed documentation and compatibility inconsistencies

1. `ml/README.md:135-137` says:
   - industry: 282 categories
   - country: 23 categories
   - product: 617 categories

   Actual values are:
   - country: 282
   - industry: 23
   - product: 617

   This is a documentation error, not a data or model error.

2. `ml/README.md:111` says IC-SPLC raw edges have “29 self-loops and 37 duplicate triples removed.”  
   Actual:
   - raw edges: 115,796
   - self-loops: 29
   - processed edges: 115,767 (29 removed)
   - exact duplicate rows remaining in processed: 37
   - unique edge IDs in processed: 115,730
   - final IC-SPLC graph edges: 106,683

   The 37 duplicates are merged later at graph level, not removed by preprocessing.

3. Absolute paths remain in:
   - `ml/experiments/graphsage_phase2/metrics/A/original_graphsage_reference.json:1610`
   - `ml/experiments/graphsage_phase2/logs/run.log:296`
   - `ml/reports/graphsage_diagnostic_audit.md:223`
   - `ml/README.md:343`
   - `datasets/metadata/download_log.json` and `datasets/metadata/dataset_manifest.json` (acquisition metadata)

   Phase 3 intentionally fixed targeted experiment metadata and future writers, not every historical or generated path.

4. The report `ml/reports/category_quality_report.md` is dated `2026-11-13`, older than the latest Phase 3 documentation. Its findings still agree with the direct data checks.

---

## 5. SAMPOQ Investigation

### 5.1 Raw source inventory

| Field | Value |
|---|---:|
| Rows | 16,102 |
| Columns | 9 |
| Exact duplicate rows | 0 |
| Unique source names | 2,662 |
| Unique source tickers | 2,662 |
| Unique source CIKs | 2,662 |
| Unique related names | 11,587 |
| Unique related tickers | 1,075 |
| Customer-role records | 10,270 |
| Supplier-role records | 5,832 |
| Date range | 2004-03-29 to 2026-10-27 |

Raw columns do not include country, industry, or product category. The source is a supply-chain relationship table extracted from 10-K filings, not a company-attribute table.

`related_ticker` is missing for 12,554 of 16,102 records (77.97%). The source relation roles are unbalanced toward customer records.

### 5.2 Processed inventory

| Field | Value |
|---|---:|
| Nodes | 13,372 |
| Edges | 16,102 |
| Unique tickers | 3,373 (25.22%) |
| Unique CIKs | 2,662 (19.91%) |
| Country coverage | 0% |
| Industry coverage | 0% |
| Product coverage | 0% |
| Edge year coverage | 100% |
| Filing date coverage | 100% |
| Edge year range | 2004-2026 |
| Edges in 2026 | 14,815 (92.0%) |
| Duplicate-relationship flags | 4 |
| Relationship type | `supplier_to_customer` for all edges |
| Direction | `Supplier -> Customer` for all edges |
| Confidence | 0.9 for all edges |

Node ID methods:

- name: 9,999
- ticker: 3,372
- name-resolved-to-ticker: 1

### 5.3 What is genuinely missing

The source has no defensible country, industry, or product category. This is not a feature-engineering loss. The preprocessor writes `None` rather than guessing, which is the correct behavior.

Consequences:

- Every SAMPOQ node has the same country/industry/product absence pattern.
- No SAMPOQ node has an IC-SPLC embedding.
- The 8 flags are identical within SAMPOQ.
- The 136-d model input is identical for all 13,372 SAMPOQ nodes.
- GraphSAGE cannot separate SAMPOQ endpoints except through graph position and source flags.

The original SAMPOQ results are consistent with this: GNN AP is ~0.50, while directed PA reaches 0.641. The model is not learning SAMPOQ node identity; it is mostly seeing a constant feature block.

### 5.4 What is available but currently unused

- Ticker presence and value.
- CIK presence and value.
- Company names and name variants.
- Mention count.
- First/last filing dates.
- Edge year and filing date.
- Supplier/customer source role.
- Related ticker presence.
- Duplicate-relationship flag.
- Repeated-edge counts.
- Train-window structural features.

These are identity-adjacent or temporal, not generalizable company attributes. Ticker and CIK can support future entity linkage if allowed, but they do not explain supply relationships by themselves. Name fields are high-cardinality and require entity resolution before modelling.

### 5.5 Information lost at graph construction

The processed SAMPOQ edge table contains:

- `year`
- `filing_date`
- `source_relation_type`
- `source_ticker`
- `related_ticker`
- `filing_accession`
- `relationship_direction`
- `source_record_id`
- `product_category`

The combined graph keeps only:

- `edge_id`
- `source_company_id`
- `target_company_id`
- `relationship_type`
- `edge_count`
- `source_datasets`
- `first_year`
- `last_year`
- `confidence`

This means the graph loses filing-level dates, source role, ticks, accession, and record provenance. Some of these are correctly excluded from candidate scoring because they describe the observed target edge. They remain useful for auditing, temporal splits, and descriptive analysis.

### 5.6 Risk and leakage notes

- Edge year and filing date must not be used as candidate-edge features; they describe the target relationship.
- Full-graph degree or component size can leak split structure.
- Train-window structural features are transductive and do not solve unseen-node behaviour.
- Ticker/CIK must not be used to merge anonymized sources.
- SAMPOQ’s 2026 concentration makes a temporal split possible but leaves little history and may reflect acquisition timing rather than a stable process.

### 5.7 SAMPOQ root-cause questions

1. Does the raw SAMPOQ source contain any country, industry, or product fields in another release?
2. Are the 2,662 unique source tickers/CIKs stable entity keys across filings?
3. Why do name variants exist for every entity, and do they refer to the same legal company?
4. Is `mention_count` a filing count, relationship count, or extraction count?
5. Are the 14,815 2026 edges genuine relationships or an artifact of the acquisition cutoff?
6. Is `filing_date` the relationship disclosure date or the dataset ingestion date?
7. Why are only 4 duplicate-relationship flags present, and do they correspond to exact duplicate rows?
8. Does `source_relation_type` always imply the direction assigned by the preprocessor?
9. Can `related_ticker` support relationship-specific company linkage without violating privacy or licensing?
10. Is external enrichment permitted and licensed, and would it be defensible under the project’s scope?

---

## 6. AnonNet Investigation

### 6.1 Raw and processed inventory

| Field | Value |
|---|---:|
| Raw nodes | 7,245 |
| Raw edges | 43,926 |
| Processed nodes | 7,245 |
| Processed edges | 43,922 |
| Raw self-loops | 4 |
| Exact duplicate edge pairs | 0 |
| Country coverage | 79.43% (5,755 nodes) |
| Country distinct cleaned values | 102 |
| Country distinct raw strings | 105 |
| Product-category coverage | 92.59% (6,708 nodes) |
| Primary product distinct values | 600 |
| Full product-list distinct values | 3,706 |
| Product-count non-null | 6,708 |
| Product-count blanks | 537 |
| Product-count range | 1 to 69 |
| Product-count median | 2 |
| Nodes with one product | 3,158 |
| Nodes with multiple products | 3,550 |
| Industry coverage | 0% |
| Edge product coverage | 97.02% (42,615 edges) |
| Edge product distinct values | 501 |
| Edge year coverage | 0% |
| Edge relationship type | `supply_relationship_anonymised` for all edges |
| Node ticker coverage | 0% |

The corrected product-count maximum is 69, not 36. The source product lists are comma-separated and can be long.

### 6.2 What the preprocessor does

- Sets `country` from `location`.
- Splits `products` on commas.
- Sets `product_category` to the first product.
- Preserves the full `products` string.
- Sets `product_count` to the number of products.
- Sets `industry` to `None`.
- Sets edge `product_category` to the source node’s first product.
- Sets edge `year` to `None`.
- Preserves direction as provided and warns that it is likely supplier-to-customer but not formally verified.

### 6.3 What the model does with it

The feature builder creates:

- `country_code`
- `product_category_code`
- `has_country`
- `has_product_category`
- `embedding_available` (always false for AnonNet)
- `is_anonymised`

The 136-d model uses only the presence flags, source flags, and embedding block. The actual country and product codes are not used. AnonNet therefore has 4 unique 136-d rows from combinations of country-present and product-present flags.

Phase 2 configs c4/c5 do use source-specific categorical embeddings. Those are the right controlled experiments for recovery; c1 is not evidence that categorical values are useless.

### 6.4 Direction and provenance

- Direction is documented as “as provided (source -> target), likely Supplier -> Customer”; it is not formally verified.
- No LICENSE file is present.
- Entities are anonymized.
- No cross-source entity resolution exists.
- The edge product is a node-level proxy, not a verified transaction product.
- No time field exists, so temporal validation is impossible for AnonNet alone.

### 6.5 Information loss

- The first product is used as the primary category; the full list is preserved but not modelled.
- Industry is absent at the source.
- Edge product is inferred from the source node, not the relationship.
- Four self-loop edges are removed.
- No year means no temporal features or temporal split.
- The graph drops the full product list after feature construction.

### 6.6 AnonNet root-cause questions

1. Is the source-to-target direction a verified supplier-to-customer relation?
2. Are country strings normalized consistently, or are the 105 raw strings hiding duplicates?
3. Are product lists delimiter-safe, and do commas appear inside product names?
4. Why do 537 nodes lack product lists and product counts?
5. Is a 69-product list a true multi-product firm or a data-quality artifact?
6. Is the edge product defined as the source firm’s first product, or was it supplied per relationship?
7. Can any external time signal be legally and defensibly joined to AnonNet?
8. Are anonymized entity IDs stable across snapshots or only within this release?
9. Is redistribution or commercial use permitted without a LICENSE file?
10. Are the four self-loops meaningful or data-entry errors?

---

## 7. IC-SPLC Investigation

### 7.1 Raw and processed inventory

| Field | Raw / processed value |
|---|---:|
| Raw nodes | 109,403 |
| Raw edges | 115,796 |
| Raw self-loops | 29 |
| Raw exact duplicate rows | 37 |
| Processed nodes | 109,403 |
| Processed edges | 115,767 |
| Processed self-loops | 0 |
| Processed exact duplicate rows | 37 |
| Processed unique edge IDs/triples | 115,730 |
| Final IC-SPLC graph edges | 106,683 |
| Edge year range | 2013-2025 |
| Edge product coverage | 16.36% (18,936) |
| Edge product distinct values | 16 |
| Country coverage | 22.96% (25,122) |
| Country distinct values | 179 |
| Industry coverage | 9.66% (10,565) |
| Industry distinct values | 22 |
| Product-category coverage | 1.69% (1,845) |
| Product-category distinct values | 16 |
| Secondary industry coverage | 9.26% |
| Secondary category coverage | 1.68% |
| Third category coverage | 1.67% |
| Other possible industry coverage | 5.74% |
| Other possible third category coverage | 1.61% |
| `selected` coverage | 0% |
| Embeddings | 31,182 x 128 |
| Embedding coverage | 28.51% of IC-SPLC nodes |

Count reconciliation:

- 115,796 raw edge rows
- minus 29 self-loops
- = 115,767 processed rows
- 37 exact duplicate rows remain
- 115,730 unique edge IDs/triples
- graph construction merges duplicates and multi-year records to 106,683 edges

The 37 duplicate edges are not unexplained data loss. They are duplicate IDs in the processed source file and are merged at graph level.

### 7.2 Category semantics

IC-SPLC categories are anonymized frequency-ranked codes:

- country examples: `C`, `B`, `D`, `E`
- industry examples: `A`, `C`, `B`, `D`
- product examples: `A`, `B`, `C`

The publisher does not provide a mapping to real countries, industries, or products. The code `A` in industry is not the same semantic category as `A` in country or product. Source-scoped embeddings are required if categories are used.

### 7.3 Embedding alignment

Direct checks found:

- Raw embedding flags: 31,182 available / 31,182 unavailable.
- Embedding parquet rows: 31,182.
- Unique embedding node IDs: 31,182.
- Unmatched IDs: 0.
- Flagged rows missing from parquet: 0.
- Parquet rows without a flag: 0.
- Processed availability flags: 31,182.
- Training mask: 31,182.
- Missing embeddings are zero-filled, not random vectors.
- All embeddings are finite.
- No all-zero embedding rows among available rows.
- Per-dimension standard deviation: 0.0339 to 0.0747.

The alignment is correct. The coverage gap is real source coverage, not an ID or join bug.

### 7.4 Embedding missingness by degree

| Degree bin | IC-SPLC nodes | With embeddings | Coverage |
|---|---:|---:|---:|
| 0 | 16 | 8 | 50.00% |
| 1 | 87,800 | 20,488 | 23.33% |
| 2 | 11,093 | 3,817 | 34.41% |
| 3-5 | 7,098 | 3,574 | 50.35% |
| 6-10 | 1,868 | 1,788 | 95.72% |
| 11+ | 1,528 | 1,507 | 98.63% |

This is the key statistical risk: embedding availability is strongly associated with graph degree. A model can learn “has embedding” as a popularity/degree proxy. It must be evaluated within degree strata and against degree-aware baselines.

### 7.5 Information loss

- 29 self-loop rows are removed.
- 37 duplicate rows persist until graph grouping.
- `selected` is blank for all nodes.
- Category labels have no semantic mapping.
- Product coverage is too low for a category-only model.
- Edge year is available but is target-edge metadata for a future-link task.
- The graph drops exact per-year records, retaining only first/last year.
- Embedding provenance is undocumented in this repository.

### 7.6 IC-SPLC root-cause questions

1. Which model, version, corpus, and objective produced the 128-d embeddings?
2. Are the embeddings normalized, centered, or raw hidden states?
3. Why are only 28.5% of IC-SPLC nodes embedded?
4. Is embedding availability selected by degree, source record completeness, or model confidence?
5. Are country/industry/category codes frequency ranks, hashes, or publisher-assigned labels?
6. Can the anonymization mapping be obtained under the dataset licence?
7. Are 2013-2025 edge years disclosure years, transaction years, or extraction years?
8. Why do 37 exact duplicate edge rows exist, and are they intentional multi-source corroboration?
9. What does the blank `selected` field mean?
10. Is the edge product a verified transaction product or a source-node proxy?

---

## 8. End-to-End Feature Pipeline Trace

### 8.1 Trace table

| Stage | Input | Transformation | Output | Information lost | Evidence |
|---|---|---|---|---|---|
| SAMPOQ preprocessing | 16,102 x 9 raw rows | Entity resolution; role-to-direction mapping; hardcoded absent attributes | 13,372 nodes; 16,102 edges | No attributes to lose; filing-level role retained only on processed edges | `scripts/preprocess_sampoq.py:144-175`, `:191-204` |
| AnonNet preprocessing | 7,245 nodes; 43,926 edges | Preserve country; split products; first product as category; drop self-loops | 7,245 nodes; 43,922 edges | Industry absent; edge product is source-node proxy; full list not modelled | `scripts/preprocess_anonnet.py:71-115` |
| IC-SPLC preprocessing | 109,403 nodes; 115,796 edges; 31,182 embeddings | Map fields; preserve codes/year; drop 29 self-loops | 109,403 nodes; 115,767 edges | No name mapping; 37 duplicate rows remain; `selected` blank | `scripts/preprocess_ic_splc.py:189-230` |
| Combined graph | 175,791 source edge rows | Merge 37 duplicate edge IDs; group by source/target/relationship | 175,754 raw grouped rows; 166,690 graph edges | 9,064 groups lose per-record dates, filing data, product, direction | `scripts/build_graph_dataset.py:191-214`, `:437-455` |
| Category coding | Combined company table | Source value -> integer code; missing -> -1, then 0 after offset | 282/23/617 categories including missing | Anonymized code systems mixed unless source-scoped | `scripts/build_graph_dataset.py:312-318` |
| Node features | `node_features.csv` + embeddings | Train-only median/scale; 18 numeric; 3 categorical; 128 embeddings; mask | `x` 146-d; `x_cat` 3-d; `x_emb` 128-d; `emb_mask`; edge features 12-d | Category semantics and embeddings remain separate; edge time is full-graph | `ml/preprocessing/feature_engineering.py:182-243` |
| PyG assembly | Feature bundles; edge index | Build `Data` with `x`, `x_cat`, `x_emb`, `emb_mask`, `edge_attr`, masks | PyG `Data` | Nothing at assembly, but downstream use is narrow | `ml/preprocessing/graph_preprocessing.py:49-59` |
| Original model selection | `Data` | Exclude 10 full-graph/time features; select remaining names | 136-d `x` | Categorical codes, edge features, and 10 structural/time fields excluded | `ml/training/train_graphsage_link_prediction.py:44-56`, `:83-94`, `:320-321` |
| Split construction | 166,690 edges | Seeded random 80/10/10 on edge ordering | 133,352 / 16,669 / 16,669 | No temporal cutoff; reverse edges can cross splits | `scripts/build_graph_dataset.py:342-346` |
| Evaluation | Score pairs | Validation-only threshold; balanced source-matched negatives; metrics | AUC/AP/P/F1/curves; Phase 2 calibration/P@K | Original lacks calibration; no CIs; balanced prevalence | `ml/training/evaluate_link_prediction_baseline.py:231-270` |

### 8.2 Exact 136-dimensional contract

Selected columns, in stored order:

1. `source_sampoq`
2. `source_ic_splc`
3. `source_anonnet`
4. `has_country`
5. `has_industry`
6. `has_product_category`
7. `embedding_available`
8. `is_anonymised`
9. `icsplc_emb_0`
...
136. `icsplc_emb_127`

Excluded features:

- `log1p_degree`
- `log1p_in_degree`
- `log1p_out_degree`
- `log1p_raw_degree`
- `log1p_component_size`
- `first_edge_age`
- `last_edge_age`
- `edge_year_span`
- `has_edge_year`
- `is_isolated`

The exclusion rationale is leakage control for full-graph structural/time information. The flaw is that the replacement is missing: there are no train-window structural features and no source-scoped categorical values in the original contract.

### 8.3 Feature collapse, corrected

| Source | Nodes | Unique 136-d rows | Embedding coverage |
|---|---:|---:|---:|
| SAMPOQ | 13,372 | 1 | 0% |
| AnonNet | 7,245 | 4 | 0% |
| IC-SPLC | 109,403 | 31,187 | 28.51% |
| Combined | 130,020 | 31,192 | 23.98% |

Why IC-SPLC has 31,187 unique rows but only 31,182 embeddings:

- The flags add combinations even when embeddings are absent.
- Thirteen distinct flag patterns exist for IC-SPLC.
- The extra five rows are flag-pattern distinctions, not additional embedding vectors.

### 8.4 Feature ordering and reindexing

`datasets/processed/graph/node_features.csv` is not sorted by `node_id`. The feature builder:

1. loads `node_index_mapping.csv`,
2. sorts by `node_index`,
3. reindexes node features to that order,
4. reindexes embeddings to the same order,
5. zero-fills missing embeddings,
6. builds masks from the availability flag.

A naive direct comparison between the raw CSV order and the embedding order is invalid. After reindexing, the mask matches exactly and unavailable embeddings are zero. There is no order bug.

### 8.5 What reaches the model

The model consumes:

- `x` only, selected to 136 columns,
- `edge_index` and train edge mask for message passing,
- train/validation/test edge masks,
- no `x_cat`,
- no `edge_attr`,
- no `x_emb` separately (embeddings are inside `x`),
- no `emb_mask` directly (availability is a flag inside `x`).

The full feature architecture is therefore not the original model. Any claim about “the 146-feature model” must distinguish it from the 136-d checkpoint.

### 8.6 Pipeline conclusion

The collapse is layered:

1. **Source limitation:** SAMPOQ has no useful attributes.
2. **Feature-selection failure:** AnonNet categories are available but ignored.
3. **Coverage limitation:** IC-SPLC embeddings cover a minority and are degree-correlated.
4. **Structural-feature gap:** full-graph features were excluded without a valid train-only replacement.
5. **Evaluation limitation:** pooled balanced metrics obscure per-source behavior.

No single architecture change can fix all five.

---

## 9. Feature Recovery Opportunities

### 9.1 Opportunity table

| Dataset | Candidate | Raw availability | Currently used? | Missingness | Leakage / validity | Potential value |
|---|---|---|---|---|---|---|
| SAMPOQ | Country | 0% | No | 100% | Cannot recover from source | Low unless new data |
| SAMPOQ | Industry | 0% | No | 100% | Cannot recover from source | Low unless new data |
| SAMPOQ | Product | 0% | No | 100% | Cannot recover from source | Low unless new data |
| SAMPOQ | Ticker / CIK | 25.22% / 19.91% | No | High | Identity-adjacent; not a general attribute | Medium for linkage, low standalone |
| SAMPOQ | Names / variants | 100% | No | High cardinality | Entity-resolution risk | Low/medium for linkage |
| SAMPOQ | Filing dates / mention count | 100% | No | None | Temporal leakage if full-history | Medium with cutoff |
| SAMPOQ | Edge year / role | 100% | No | None | Target-edge leakage if scored | Temporal split or descriptive only |
| AnonNet | Country | 79.43% | No in 136; yes in c4/c5 | 20.57% | Source-scoped vocabulary required | High |
| AnonNet | Primary product | 92.59% | No in 136; yes in c4/c5 | 7.41% | Source-scoped; 600 values | High |
| AnonNet | Full product list / count | 92.59% | No | 7.41% | Multi-label; 3,706 lists | High |
| AnonNet | Edge product | 97.02% | No | 2.98% | Source-node proxy, not transaction product | Low / descriptive |
| IC-SPLC | 128-d embeddings | 28.51% | Yes in c1 | 71.49% | Provenance unknown; degree-correlated | High with mask and degree controls |
| IC-SPLC | Country code | 22.96% | No in 136; yes in c4/c5 | 77.04% | Anonymized, source-scoped | Medium/low |
| IC-SPLC | Industry code | 9.66% | No in 136; yes in c4/c5 | 90.34% | Anonymized, 22 codes | Low/medium |
| IC-SPLC | Product code | 1.69% | No in 136; yes in c4/c5 | 98.31% | Anonymized, 16 codes | Low |
| IC-SPLC | Edge year | 100% | No | None | Temporal cutoff only; target leakage otherwise | High for temporal tests |
| All | Train-window structure | Derived | c3/c5 only | Varies | Transductive; not for unseen nodes | Medium, with strict controls |
| All | Cross-source entity resolution | None | No | N/A | Unauthorized/unsafe without mapping | Not recommended now |

### 9.2 What can be recovered without new data

- AnonNet country, primary product, full product list, and product count.
- AnonNet source-scoped categorical embeddings.
- IC-SPLC categorical codes with missing tokens, subject to semantic caveats.
- IC-SPLC embedding mask and embedding-availability interactions.
- IC-SPLC temporal cutoffs from edge year.
- SAMPOQ ticker/CIK presence, filing recency, mention count, repeat-edge counts, and train-window structure.
- Source-aware routing and source-scoped category tables.
- Better per-source baselines and evaluation.

### 9.3 What should not be recovered by guessing

- Do not infer SAMPOQ country from names or tickers without a validated mapping.
- Do not infer industry/product from embedding neighbourhoods and present it as source data.
- Do not merge AnonNet or IC-SPLC entities with SAMPOQ companies.
- Do not treat confidence `0.9/0.8/0.7` as calibrated probabilities.
- Do not treat AnonNet edge product as a verified transaction product.
- Do not treat IC-SPLC anonymized codes as named categories.
- Do not use target-edge year, filing date, product, confidence, or count as candidate-scoring features.

---

## 10. Leakage and Evaluation Risks

| Risk | Evidence | Consequence | Mitigation |
|---|---|---|---|
| Reverse-edge leakage | Protocol A: 9.98% AnonNet, 4.91% SAMPOQ, 0.45% IC-SPLC test positives have reverse train edges | Baselines and GNNs can exploit reversed structure | Use Protocol B or C for clean random-split claims |
| Temporal leakage | SAMPOQ 92.0% 2026; IC 2013-2025; AnonNet no time | Random split mixes future and past; future-link claims invalid | Use cutoff-based splits per source; never use target year as feature |
| Target-edge leakage | Edge year, filing date, product, confidence, count describe observed edge | Inflated metrics; undeployable model | Exclude target-edge metadata from candidate scoring |
| Full-graph structural leakage | Degree/component computed over all edges | Test structure enters features | Recompute from train graph only; still transductive |
| Embedding-coverage leakage | Embedding availability rises with degree | Model may learn popularity shortcut | Stratify by degree; compare no-embedding and degree baselines |
| Source shortcut | Source flags and missingness patterns | Pooled model can learn source identity rather than link propensity | Source-specific models; source-balanced loss; per-source reporting |
| Cross-source entity leakage | No entity resolution; namespaced IDs | Unauthorized merging could contaminate sources | Keep namespaces; no external entity matching without approval |
| Balanced evaluation | 1:1 source-matched negatives | Precision/F1/Brier/ECE are not production estimates | Report prevalence-matched or all-non-edge metrics |
| Threshold optimism | Threshold selected on validation | Test metrics depend on one validation realization | Keep validation-only selection; bootstrap uncertainty |
| Score saturation | 18,895 test scores >= 0.99; 11,101 exactly 1.0 | Brier/log-loss/ECE severe; probabilities unusable | Calibrate on validation; report ECE and reliability; do not call scores probabilities |
| Single seed | Phase 2 seed 42 only | No reliable improvement claims | Multiple seeds and confidence intervals |
| Label absence | `risk_labels.csv` all `not_available` | Risk model would be fabricated | Keep link prediction separate; no risk claims |
| Count drift | Processed 115,767 vs raw 115,796; 37 duplicates remain | Documentation confusion | Reconcile counts in every release |
| Artifact drift | Checkpoint ignored; duplicate node features; historical paths | Reproducibility and audit risk | Hash, version, and path checks before training |

Phase 2 Protocol A/B/C results are close for the primary configurations, which supports the conclusion that reverse leakage is not the main driver of the collapse. It remains a validity problem for any clean claim.

Evaluation recommendations:

- Use B/C for random-split claims.
- Use D only for IC-SPLC temporal evidence.
- Report source-specific AP, AUC, P@K, Brier, ECE, and CIs.
- Report P@K alongside AP/AUC because PA and degree-heavy models can win the top of the ranking while losing overall discrimination.
- Do not use balanced precision as a production estimate.
- Do not use a single seed as evidence of improvement.

---

## 11. Data Quality Test Plan

The current suite has 19 tests, but it does not enforce most source-level data-quality invariants. The following 20 tests should become Goal B gates. “Blocks training” means the training pipeline should fail closed when the assertion fails.

| ID | Test | Purpose | Expected | Failure signal | Location | Blocks training? |
|---|---|---|---|---|---|---|
| DQ01 | Manifest/hash verification | Detect silent data mutation | Every required artifact matches manifest SHA-256 | Missing/mismatched hash | `datasets/metadata/dataset_manifest.json` | Yes |
| DQ02 | Source row-count reconciliation | Ensure documented counts match files | SAMPOQ 16,102; AnonNet 43,926; IC raw 115,796; processed 115,767 | Any unexplained difference | Source and processed tables | Yes |
| DQ03 | Edge endpoint integrity | Prevent dangling edges | Every endpoint resolves to a node | Orphan endpoint | Graph build | Yes |
| DQ04 | Self-loop and duplicate handling | Make removal/merge explicit | 29 IC self-loops removed; 4 AnonNet removed; 37 duplicate IDs merged at graph level | Unexplained self-loop/duplicate | Preprocessing and graph build | Yes |
| DQ05 | Graph aggregation conservation | Prove no unexplained edge loss | Source rows - duplicates - controlled drops = graph edges | Conservation failure | `graph_metadata.json` | Yes |
| DQ06 | Node ID uniqueness and namespacing | Prevent accidental merges | 130,020 unique IDs; no cross-source collision | Duplicate or merged namespace | Node index | Yes |
| DQ07 | 136/146 contract integrity | Catch silent feature drift | 8 flags + 128 embeddings = 136; 18 numeric + 128 = 146 | Dimension or name mismatch | Feature builder/checkpoint | Yes |
| DQ08 | Per-source unique-row floors | Catch feature collapse | SAMPOQ >= 2; AnonNet > 4; IC unique >= embeddings | Regrowth to constant rows | Feature audit | Yes |
| DQ09 | Category cardinality and missing token | Keep categories and missingness explicit | 282/23/617 including missing; missing maps to token 0 | Cardinality drift or missing treated as real code | `feature_engineering.py` | Yes |
| DQ10 | Source-scoped category semantics | Prevent code-system conflation | No shared vocabulary across sources | Same code reused across source without scope | Category mappings | Yes |
| DQ11 | SAMPOQ absent-attribute provenance | Do not invent source fields | Country/industry/product remain null unless new source provided | Non-null SAMPOQ values without provenance | SAMPOQ preprocess | Yes |
| DQ12 | AnonNet product preservation | Ensure full list/count survive | 6,708 non-null; count range 1-69; blank 537 | Truncated list or changed count | AnonNet preprocess | Yes |
| DQ13 | Embedding alignment | Prove ID-level correctness | 31,182 embeddings; 0 unmatched; mask 31,182 | Any unmatched/mismatched row | `load_embeddings()` | Yes |
| DQ14 | Embedding mask/zero consistency | Prevent fake vectors | Every unavailable row is zero; every available row is finite and non-zero | Non-zero unmasked or zero masked | Feature builder | Yes |
| DQ15 | Split integrity | Prevent node/edge contamination | Disjoint train/val/test; reproducible counts | Overlap or drift | `dataset_splits.json` | Yes |
| DQ16 | Reverse-edge leakage | Quantify and control reverse support | Report 9.98/4.91/0.45; B/C zero | Reverse train edge in test | Phase 2 protocol builder | Yes for clean claims |
| DQ17 | Temporal cutoff leakage | Prevent future information | All features use data before cutoff | Any future edge/date in training | Temporal pipeline | Yes |
| DQ18 | Target-edge feature exclusion | Prevent label-side leakage | No year/product/count/confidence/filing in candidate scoring | Target metadata appears in feature vector | Model feature contract | Yes |
| DQ19 | Negative-sampling validity | Keep pairs real, unobserved, source-matched | No observed edges, no duplicates, no cross-source negatives | Invalid negative | Sampler | Yes |
| DQ20 | Evaluation reporting and uncertainty | Prevent overclaiming | Per-source AP/AUC/P@K/Brier/ECE; seed count and CIs reported | Pooled-only or single-seed claim | Evaluation pipeline | Yes for any reported result |

Minimum Phase 4B gates:

1. DQ01-DQ07, DQ13-DQ15, DQ18-DQ19 must pass before any model comparison.
2. DQ08-DQ12 must pass before feature-recovery claims.
3. DQ16-DQ17 must pass before random-split or temporal claims.
4. DQ20 must pass before any improvement is called reliable.
5. A failed gate must block training, not merely emit a warning.

---

## 12. Source-Specific Recommendations

### 12.1 SAMPOQ

**Recommendation:** Treat SAMPOQ as a separate temporal/identity model and a data-acquisition decision, not as a feature-engineering problem.

- Preserve ticker/CIK presence as identity-adjacent fields.
- Use filing recency, mention count, repeat-edge count, and train-window structure only with a cutoff.
- Never use edge year/filing date/product/role as candidate-scoring features.
- Compare against directed PA on SAMPOQ only.
- If SAMPOQ remains near chance, document the source limitation and exclude it from pooled headline claims.
- Do not infer country, industry, or product from names.
- If enrichment is required, obtain a licensed, documented mapping and audit its coverage before use.

**Root-cause questions:** see Section 5.7.

### 12.2 AnonNet

**Recommendation:** Make AnonNet the first controlled recovery target.

- Add source-scoped country and product embeddings.
- Add missing tokens separate from category indices.
- Add product count, multi-product indicator, and rare-product grouping.
- Preserve the full product list; do not reduce it to the first product in all models.
- Evaluate against the AnonNet flags-only model, directed PA, undirected PA, and common neighbours.
- Use Protocol B or C; do not use a pooled result to judge AnonNet.
- Use the edge product only as a proxy/descriptive field, not as a verified transaction feature.
- Do not assume a GNN win: directed PA is a strong AnonNet baseline.

**Root-cause questions:** see Section 6.6.

### 12.3 IC-SPLC

**Recommendation:** Use IC-SPLC for temporal and embedding-mask experiments, not for a category-only model.

- Treat embedding availability as a first-class feature.
- Evaluate within degree strata because coverage is degree-correlated.
- Add source-scoped categorical embeddings with missing tokens.
- Report category coverage by split.
- Use the 2013-2023/2024/2025 cutoff for future-link evidence.
- Do not use edge year or edge product as candidate-scoring features.
- Do not interpret anonymized codes as named industries/countries/products.
- Verify embedding provenance before relying on it as a general signal.
- Compare against c1, c4, c5, directed PA, and undirected PA.

**Root-cause questions:** see Section 7.6.

### 12.4 Pooled model

**Recommendation:** Keep pooled results secondary until each source is stable.

- Use source-specific categorical vocabularies and source embeddings.
- Use source-balanced loss and per-source evaluation heads.
- Keep source-matched negative sampling.
- Do not use full-graph structural features unless recomputed from the training graph and shown valid for the intended transductive task.
- Report per-source results before pooled results.
- Never let an easy AnonNet or IC-SPLC regime hide SAMPOQ collapse.
- Add multiple seeds and confidence intervals before any improvement claim.

### 12.5 What should not be done

- Do not add architecture complexity to compensate for absent SAMPOQ attributes.
- Do not use edge target metadata as features.
- Do not calibrate on test.
- Do not tune on test.
- Do not merge anonymized sources.
- Do not present link scores as risk probabilities.
- Do not generate risk labels from link scores.
- Do not treat a single seed as reliable evidence.

---

## 13. Goal B Implementation Roadmap

### 13.1 Priority table

| Priority | Task | Why | Complexity | Dependency | Completion gate |
|---:|---|---|---|---|---|
| 0 | Add data-quality and feature-contract gates DQ01-DQ20 as blocking checks | Prevent silent drift before changing anything | Low | Existing tests and manifests | All gates pass on current artifacts |
| 1 | AnonNet source-scoped categorical recovery | Highest unused attribute value; no new data | Medium | DQ08-DQ12 | Beats AnonNet flags-only and is compared with directed PA under B/C |
| 2 | IC-SPLC temporal + mask model | Best temporal source; explicit degree/missingness risk | Medium-high | DQ13-DQ17 | Reproducible temporal result; degree-stratified reporting |
| 3 | SAMPOQ temporal/identity baseline + data decision | Determine whether the source is usable at all | Medium | DQ11, DQ17 | Separate SAMPOQ result and documented acquisition decision |
| 4 | Source-aware pooled model | Only after per-source stability | High | Priorities 1-3 | No source collapse; source-balanced evaluation |
| 5 | Calibration and multi-seed/CIs | Current probabilities and single-seed claims are unsafe | Medium | Priorities 1-4 | Brier/ECE/reliability + repeated-seed variance |
| 6 | Edge-aware model (config 6) | Only if candidate-time edge features are valid | High | A valid non-leaking edge design | No target-edge metadata; no unavailable features |
| 7 | Documentation and provenance cleanup | Known count/cardinality/path drift | Low | None | Docs agree with data and tests |

### 13.2 First Goal B task

**First task:** implement the blocking DQ01-DQ20 gate, then run the AnonNet-only categorical recovery experiment.

Why this order:

- The report found documentation/count drift and no data-quality gates.
- Changing features before adding gates would make any improvement unverifiable.
- AnonNet is the cleanest controlled test because the attributes already exist and the baseline is strong.

Do not start Goal B implementation in this task.

---

## 14. Unresolved Questions

1. Is the intended production task future-link prediction with a cutoff, or transductive completion of a fixed graph?
2. What is the real candidate distribution at prediction time, and what prevalence should evaluation match?
3. Is external enrichment of SAMPOQ permitted by licence and project scope?
4. Can a defensible SAMPOQ industry/country/product source be acquired?
5. Is AnonNet direction formally supplier-to-customer, or only source-to-target?
6. Which model and corpus generated the IC-SPLC embeddings, and are they stable across releases?
7. Can IC-SPLC anonymized category mappings be obtained under the licence?
8. Are 2026 SAMPOQ edges genuine or an acquisition-window artifact?
9. Are anonymized entity IDs stable across snapshots?
10. Is AnonNet redistributable or commercially usable without a LICENSE file?
11. Should SAMPOQ be excluded from pooled headline metrics if its attributes remain absent?
12. What confidence interval and seed protocol is required before an improvement claim is accepted?
13. Are the two node-feature copies intentionally redundant or an artifact of generation?
14. Should historical absolute paths in logs/reports be corrected, or preserved as historical evidence?
15. Should the 37 duplicate IC-SPLC edge rows be deduplicated at preprocessing or retained for corroboration counting?

---

## 15. Final Verdict

### 15.1 Goal A verdict

**Goal A complete with documented caveats.**

The investigation answered the required questions:

- What data exists: confirmed by direct counts and schemas.
- What is corrupted or missing: SAMPOQ attributes are absent; AnonNet industry/time absent; IC categories sparse/anonymized; embeddings partial.
- Where information is lost: preprocessing, graph aggregation, feature selection, split construction, and evaluation each lose or exclude information.
- Why previous experiments behaved as they did: source limitation plus narrow model contract plus unused attributes.
- Whether Phase 3 remains compatible: yes at runtime and tests; documentation/path/count caveats remain.

### 15.2 Root-cause verdict

The collapse is **not** a single preprocessing bug.

- **SAMPOQ:** source limitation. No country/industry/product fields exist. One unique 136-d row is the correct consequence of the current source and model contract.
- **AnonNet:** engineering/feature-selection failure. Country and product values exist in processed data, but the 136-d model ignores them. Four unique rows are avoidable.
- **IC-SPLC:** partial source coverage plus a modelling opportunity. Embedding alignment is correct; coverage is 28.5% and strongly degree-correlated; categories are sparse and anonymized.
- **Combined graph:** information loss and source heterogeneity. Graph grouping collapses 175,754 grouped rows to 166,690 edges and drops useful provenance.
- **Evaluation:** the original metrics are real but limited: balanced pairs, single seed, no calibration, and pooled aggregation that hides source failure.

### 15.3 What is safe to conclude

- Link prediction is the only supported ML task.
- SAMPOQ is not usable as a discriminative node-feature source in its current form.
- AnonNet categorical recovery is the highest-value controlled next experiment.
- IC-SPLC is the best temporal candidate but needs provenance and degree-aware evaluation.
- The current GraphSAGE results are not evidence that GraphSAGE is intrinsically unsuitable.
- PA/CN are serious baselines and must remain in every experiment.
- The current scores are not calibrated probabilities and not risk scores.

### 15.4 What is not safe to conclude

- That feature recovery will definitely improve AP.
- That categorical embeddings are sufficient without baselines.
- That pooled metrics prove a source-level improvement.
- That a single seed establishes reliability.
- That anonymized categories have known semantics.
- That link scores can be interpreted as disruption risk.

### 15.5 Recommendation

Proceed to Goal B only after accepting the DQ gate and the source-specific evaluation plan. Start with AnonNet. Keep SAMPOQ as a separate data decision. Treat IC-SPLC as a temporal/mask experiment. Do not begin Goal B in this task.

---

## 16. File Inventory and Change Verification

### 16.1 Change summary

| Item | Result |
|---|---|
| Files created | This report only |
| Files modified | None tracked; existing untracked Phase 4A draft replaced |
| Files deleted | None |
| Datasets modified | None |
| Splits modified | None |
| Features modified | None |
| Models trained | None |
| Checkpoints written | None |
| Metrics/predictions regenerated | None |
| Dashboard changed | None |
| Dependencies installed/changed | None |
| Commit/push | None |

### 16.2 Git state

Before this report update:

```text
?? ml/reports/phase4a_dataset_investigation.md
```

- Branch: `Yash`
- Upstream: `origin/Yash`
- Last commit: `607f43b Update project documentation and add Phase 3 cleanup report`
- No tracked modifications.
- Only the Phase 4A report was untracked.

After this report update, the expected tracked state is unchanged: only `ml/reports/phase4a_dataset_investigation.md` is untracked.

`git diff --check` was clean. No commit or push was performed.

### 16.3 Validation

- Phase 3 test suite: **19 passed**.
- Command used: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q`.
- Test coverage confirmed: inference contract, checkpoint loading, feature/split contract, evaluation behavior, dashboard catalog, metadata/path checks.
- Test gaps confirmed: no source-level coverage thresholds, no Doc/cardinality check, no reverse-leakage enforcement, no temporal gate, no target-edge gate, no multi-seed/CI gate.

### 16.4 Preserved artifact evidence

| Artifact | SHA-256 / size | Status |
|---|---|---|
| `ml/experiments/graphsage_link_prediction/checkpoint.pt` | `7c1abf5688cd60ffc30875f264106ca1195a014c806c90425ffd267520ee4de6`; 94,965 bytes | Unchanged; ignored/untracked |
| `ml/experiments/graphsage_link_prediction/metrics.json` | `58246204dcb2f984f84315236cc2736438ad76a914c7fce4c8c9cb17e2877d0b` | Unchanged |
| `ml/experiments/graphsage_link_prediction/test_predictions.csv` | `1f518eaf934233d73e0c516f67c43eb302780887ae7239cfc03a7c18b2eba22b` | Unchanged |
| `ml/experiments/graphsage_phase2/summary.json` | `9e982f7bd0ab6730cbd85c454deeedeb76eebcea6e741ac357d795ab5dd3feda` | Unchanged |
| `datasets/processed/graph/node_features.csv` | `c2149400eeaf66070a4ebbeb06612d7451d509cae2a66f56ac586395c7a51ac3` | Unchanged |
| `datasets/training/node_features.csv` | `c2149400eeaf66070a4ebbeb06612d7451d509cae2a66f56ac586395c7a51ac3` | Unchanged; identical copy |

The checkpoint mtime is `2026-11-17 09:57:53`. The metrics mtime is the same. The Phase 2 summary mtime is `2026-11-17 11:44:56`. This investigation did not write them.

### 16.5 Files inspected

Code and tests:

- `scripts/preprocess_sampoq.py`
- `scripts/preprocess_anonnet.py`
- `scripts/preprocess_ic_splc.py`
- `scripts/build_graph_dataset.py`
- `ml/preprocessing/feature_engineering.py`
- `ml/preprocessing/graph_preprocessing.py`
- `ml/training/train_graphsage_link_prediction.py`
- `ml/training/evaluate_link_prediction_baseline.py`
- `ml/training/phase2_utils.py`
- `ml/inference/predict.py`
- `ml/reports/dashboard_data.py`
- `ml/tests/*.py`
- `pytest.ini`

Data and metadata:

- `datasets/raw/sampoq/supply-chain.csv`
- `datasets/raw/anonnet/repo/mim2025_network_challenge/nodes_anon.csv`
- `datasets/raw/anonnet/repo/mim2025_network_challenge/edges_anon.csv`
- `datasets/raw/ic_splc/extracted_csv/ic_splc_nodes.csv`
- `datasets/raw/ic_splc/extracted_csv/ic_splc_edges.csv`
- `datasets/raw/ic_splc/extracted_csv/ic_splc_embeddings.parquet`
- `datasets/processed/companies/companies_sampoq.csv`
- `datasets/processed/companies/companies_anonnet.csv`
- `datasets/processed/companies/companies_ic_splc.csv`
- `datasets/processed/relationships/supply_chain_edges_sampoq.csv`
- `datasets/processed/relationships/supply_chain_edges_anonnet.csv`
- `datasets/processed/relationships/supply_chain_edges_ic_splc.csv`
- `datasets/processed/graph/nodes.csv`
- `datasets/processed/graph/edges.csv`
- `datasets/processed/graph/node_features.csv`
- `datasets/processed/graph/graph_metadata.json`
- `datasets/training/node_features.csv`
- `datasets/training/node_embeddings.parquet`
- `datasets/training/node_index_mapping.csv`
- `datasets/training/edge_index.csv`
- `datasets/training/dataset_splits.json`
- `datasets/training/training_metadata.json`
- `datasets/metadata/dataset_manifest.json`
- `datasets/metadata/download_log.json`
- `datasets/metadata/data_dictionary.md`

Experiments and reports:

- `ml/experiments/graphsage_link_prediction/checkpoint.pt`
- `ml/experiments/graphsage_link_prediction/metrics.json`
- `ml/experiments/graphsage_link_prediction/config.json`
- `ml/experiments/graphsage_link_prediction/experiment.json`
- `ml/experiments/graphsage_link_prediction/run_summary.json`
- `ml/experiments/graphsage_link_prediction/test_predictions.csv`
- `ml/experiments/graphsage_phase2/summary.json`
- `ml/experiments/graphsage_phase2/config/*`
- `ml/experiments/graphsage_phase2/plots/*`
- `ml/reports/graphsage_diagnostic_audit.md`
- `ml/reports/graphsage_phase2_experiment_report.md`
- `ml/reports/phase3_foundation_cleanup_report.md`
- `ml/reports/category_quality_report.md`
- `ml/reports/category_quality_report.json`
- `README.md`
- `README_DATASETS.md`
- `ml/README.md`

### 16.6 Report hash

The SHA-256 of this report after final write is recorded in the task completion response to avoid a self-referential write.
