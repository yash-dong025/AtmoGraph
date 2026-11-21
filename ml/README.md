# AtmoGraph ML

ML/GNN research layer for the AtmoGraph supply-chain graph. It covers dataset analysis, graph preprocessing, link-prediction baselines, GraphSAGE experiments, a local dashboard, and contract tests.

This is a research prototype. It is **not** a production GNN, it does not claim real-world risk accuracy, and it does not replace the existing dataset pipeline, backend, frontend, or NLP work.

## Scope and status

- Scope: dataset analysis, graph preprocessing, PyG loading, link-prediction baselines, GraphSAGE experiments, output contracts, and documentation.
- Existing `config/`, `scripts/`, and `datasets/` outputs remain the source of truth.
- Work is additive under `ml/`, with a local dashboard and reports under `ml/reports/`.
- Phase 1 diagnostic audit: complete.
- Original GraphSAGE link-prediction experiment: complete and preserved.
- Phase 2 protocol and feature ablations: complete and preserved.
- Phase 3 foundation cleanup: contracts, metadata, portability, tests, and documentation.
- No company-level risk model was trained because no verified risk labels exist.
- No fabricated data, labels, predictions, or metrics are used.
- No commit or push is performed automatically.

## Repository architecture inspection

- `origin/main` and `origin/Yash` contain `config/`, `datasets/`, and `scripts/`.
- `origin/Aiman`, `origin/Supraja`, and `origin/Vidhi` contain no backend, frontend, NLP, or ML service directories at inspection time.
- No pre-existing `backend/`, `frontend/`, `nlp/`, or `ml/` implementation was found.
- `README.md` documents the project; `README_DATASETS.md` documents the dataset pipeline.
- Integration approach: add `ml/` at the repository root and reuse existing processed datasets rather than duplicating or restructuring them.

## ML folder layout

```text
ml/
├── __init__.py
├── .gitignore
├── README.md
├── requirements-ml.txt
├── requirements-test.txt
├── experiments/               # original and Phase 2 artifacts
├── inference/
│   ├── __init__.py
│   └── predict.py             # link-prediction output contract
├── models/
│   ├── __init__.py
│   ├── baseline_model.py      # scikit-learn delay baselines
│   ├── gcn_model.py           # GCN/GraphSAGE/GAT prototypes
│   └── phase2_graphsage.py    # source-aware Phase 2 GraphSAGE
├── preprocessing/
│   ├── __init__.py
│   ├── dataset_analysis.py    # dataset/report generator
│   ├── feature_engineering.py # node/edge feature construction
│   └── graph_preprocessing.py # PyG Data builder and smoke test
├── reports/                   # dashboard, catalog, audits, reports
├── tests/                     # synthetic ML contract and integrity tests
└── training/                  # original and Phase 2 training/evaluation
```

Generated reports:

- `datasets/reports/ml_dataset_analysis.md`
- `datasets/reports/ml_dataset_analysis.json`
- `datasets/reports/ml_baseline_report.md`

## Environment and dependencies

- Python 3.13.15, macOS arm64.
- Local virtual environment: `.venv/` (ignored by Git).
- Installed: `torch 2.15.0`, `torch-geometric 2.8.1`, `pandas 3.1.1`, `numpy 2.5.4`, `scikit-learn 1.9.1`, `pyarrow 26.0.1`.
- Tests: `pytest 8.4.2`.
- PyTorch MPS is available; training and deterministic preprocessing checks used CPU.
- Existing `requirements-data.txt` was not modified.

Install ML dependencies:

```bash
python -m venv .venv
uv pip install --python .venv/bin/python -r ml/requirements-ml.txt
uv pip install --python .venv/bin/python -r ml/requirements-test.txt
```

If `uv` is unavailable, run `.venv/bin/python -m ensurepip --upgrade` once and use `.venv/bin/python -m pip` with the same requirements files.

## Data used and exclusions

Main ML assets:

- Combined supply-chain graph: `datasets/processed/graph/`
- PyG-ready node/edge splits: `datasets/training/`
- Shipment delay data: `datasets/processed/logistics/` and `datasets/training/delay_labels.csv`
- GDELT news mentions: `datasets/processed/disruptions/`

Excluded:

- Restricted CC BY-NC disruption-monitoring assets are not used for main training.
- No full GDELT archive download.
- No fabricated labels or inferred company risk scores.
- Anonymised IC-SPLC and AnonNet IDs are not merged with named companies.

## Dataset analysis findings

Graph:

- **130,020 nodes**
- **166,690 unique directed edges**
- **23,198 weakly connected components**
- Largest component: **52,473 nodes (40.4%)**
- **16 isolated nodes**
- IC-SPLC embeddings: **31,182 nodes (24.0%)**

Source tables:

- SAMPOQ: 16,102 rows × 9 columns, CC BY 4.0.
- IC-SPLC: 109,403 nodes and 115,796 raw edges; 29 self-loops and 37 duplicate triples removed.
- AnonNet: 7,245 nodes and 43,926 edges; 4 self-loops removed.
- SynDelay: 155,488 rows × 41 columns; synthetic; class semantics undocumented.
- Smart Logistics: 1,000 rows × 16 columns; provenance unknown.
- GDELT: 847 articles / 866 unverified mentions; bounded keyword/date sample only.

Key caveats:

- The graph is fragmented; message passing is weak across small components.
- Only 24.0% of nodes have embeddings; the rest are zero-filled with a mask.
- GDELT records are news mentions, not confirmed disruptions.
- SynDelay and AnonNet have no LICENSE file.
- Smart Logistics licence and provenance are unknown.
- No verified company-level risk labels exist.

## Feature identification

Node features (`feature_engineering.py`):

- 18 numeric features:
  - log1p degree, in-degree, out-degree, raw degree, and component size
  - first-edge age, last-edge age, and edge-year span
  - binary source flags, country/industry/product-category presence, embedding availability, anonymised flag, and edge-year availability
- 3 categorical features:
  - industry (282 categories)
  - country (23 categories)
  - product category (617 categories)
- 128 IC-SPLC embedding dimensions, zero-filled when unavailable, with `embedding_available` mask.
- Full engineered model input: **146 dimensions** (18 numeric + 128 embeddings).
- The original GraphSAGE checkpoint uses only **136 dimensions**: 8 static flags plus 128 embeddings. It excludes graph-derived structural features and does not use the 3 categorical codes or 12 edge features.
- Phase 2 evaluates the 136-d contract, flags-only, train-only structural features, source-specific categorical embeddings, and combined configurations.

Edge features:

- 12 dimensions:
  - log1p edge count, confidence, first year, last year, edge-year span
  - has-edge-year flag
  - relationship type one-hot (3)
  - source dataset one-hot (3)

Leakage controls:

- Drop `risk_label`, `delay_class`, `target_value`, IDs, and split labels.
- Fit scalers on the training split only.
- Keep categorical codes as indices and add `+1` before embedding lookup; index `0` is reserved for padding.
- Do not interpret SynDelay classes 0/1/2 as low/medium/high severity.

Important feature gap:

- `logistics_features.csv` exposes 10 usable features for the 1,000-row Smart Logistics task.
- For SynDelay, it exposes only `latitude` and `longitude`; the other raw SynDelay fields were not carried into the processed feature table.
- The 3-class baseline is therefore a pipeline-gap check, not a credible delay model. Week 2 must add a SynDelay-specific feature builder or extend the existing dataset pipeline.

## Graph representation and preprocessing

`graph_preprocessing.py` builds a PyG `Data` object with:

- `x`: `[130020, 146]` float features
- `edge_index`: `[2, 166690]` directed supplier → customer edges
- `edge_attr`: `[166690, 12]`
- `x_cat`: categorical indices for embeddings
- `x_emb`: aligned 128-d embeddings
- `emb_mask`: embedding availability mask
- `train_edge_mask`, `val_edge_mask`, `test_edge_mask`
- `node_train_mask`, `node_val_mask`, `node_test_mask`
- `node_ids`, `edge_ids`, and feature metadata

Validation smoke test passed:

```text
node_count: 130020
edge_count: 166690
node_feature_dim: 146
edge_feature_dim: 12
node_features_finite: true
edge_features_finite: true
edge_index_in_range: true
no_self_loops: true
edge_ids_unique: true
node_ids_unique: true
embedding_coverage: 0.2398
```

The smoke test also ran a GCN forward pass and produced a `[130020, 16]` embedding and `[100]` sample scores on CPU. PyTorch emits a non-writable NumPy warning during tensor conversion; preprocessing completed successfully.

## GNN architecture recommendation

Recommended:

1. **GraphSAGE as the primary model** for supplier → customer link prediction.
   - Inductive and suitable for a fragmented graph.
   - Supports neighbour sampling and mini-batching.
   - Works with node features, embeddings, and edge attributes.
2. **GCN as the baseline**.
   - Simple and fast.
   - Useful for measuring whether the added complexity of GraphSAGE is justified.
3. **GAT as a comparison model**.
   - Attention can weight important neighbours.
   - Heavier and more prone to overfitting on small or fragmented components.

Prototypes exist in `ml/models/gcn_model.py`: `AtmoGCN`, `AtmoGraphSAGE`, `AtmoGAT`, and `build_gnn()`. A single experimental GraphSAGE link-prediction run has now been trained under the source-matched protocol in `ml/experiments/graphsage_link_prediction/`; this is not a production risk model.

## Training and evaluation methodology

Existing splits from `datasets/training/dataset_splits.json`:

- Node splits: train 91,014 / val 19,503 / test 19,503.
- Edge splits: train 133,352 / val 16,669 / test 16,669.
- SynDelay 3-class: train 108,841 / val 23,323 / test 23,324.
- Smart Logistics binary: train 700 / val 150 / test 150.

Implemented link prediction:

- Train on train edges, tune on validation edges, evaluate once on test edges.
- Sample **source-matched** directed non-edges; every observed edge is excluded,
  and the positive source-pair distribution is preserved.
- Report pooled and per-source ROC-AUC, average precision, precision, recall, F1,
  confusion matrices, and PR/ROC curves.
- Compare GraphSAGE against common-neighbor and preferential-attachment baselines
  on the same seeded validation/test pairs.
- Results are experimental link prediction, not disruption-risk prediction.

Planned delay modelling:

- Use leakage-safe features only.
- Report accuracy, macro F1, per-class recall, and confusion matrices.
- Compare the scikit-learn baseline against an MLP or graph-aware model only after the SynDelay feature gap is fixed.

Company risk modelling:

- Blocked until verified company-level labels exist.
- Do not use GDELT mentions as ground-truth risk labels.
- Do not report risk accuracy or risk levels from an untrained model.

## Baseline feasibility results

From `ml/models/baseline_model.py`:

| Task | Rows | Features | Test accuracy | Macro F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Smart Logistics binary | 1,000 | 10 | 0.7133 | 0.7133 | 0.7645 |
| SynDelay 3-class | 155,488 | 2 | 0.5769 | 0.2441 | 0.5064 |

Interpretation:

- The Smart Logistics baseline is a small-sample feasibility check; provenance is unknown, so the metric is not a real-world claim.
- The SynDelay baseline uses only latitude/longitude and predicts the majority class for almost every row. It is a pipeline-gap check, not a usable delay model.
- No company risk baseline was run because there are no verified risk labels.

## Link-prediction output contract

`ml/inference/predict.py` defines a link-prediction record, not a disruption-risk record:

```json
{
  "task": "link_prediction",
  "source_id": "icsplc:123",
  "target_id": "sampoq:0000320193",
  "link_score": 0.81,
  "score_type": "model_score",
  "score_semantics": "Uncalibrated GraphSAGE score ...",
  "probability_calibrated": false,
  "prediction_status": "experimental",
  "source": { "id": "icsplc:123", "source_dataset": "ic_splc" },
  "target": { "id": "sampoq:0000320193", "source_dataset": "sampoq" },
  "model": {
    "name": "GraphSAGE",
    "type": "AtmoGraphSAGE",
    "version": "graphsage_link_prediction",
    "dataset_version": "atmo_graph_v1_015d0c924af0",
    "protocol": "original_random_split",
    "feature_config": "c1_current_136",
    "checkpoint_path": "ml/experiments/graphsage_link_prediction/checkpoint.pt"
  },
  "warnings": [
    "This is a directed supplier/customer link score, not a disruption-risk prediction.",
    "The link score is uncalibrated and must not be interpreted as a verified probability."
  ]
}
```

Rules:

- `task` must be `link_prediction`.
- `link_score` must be a finite model score in `[0, 1]`; it is not a calibrated probability.
- `source_id` and `target_id` must be different, and source/model metadata is required.
- `prediction_status` remains `experimental`.
- No `risk_score`, `risk_level`, `LOW`/`MEDIUM`/`HIGH`, or `disruption_risk` fields are part of this contract.
- `build_risk_prediction_record()` raises `RiskPredictionUnsupportedError`; disruption risk requires verified labels and a dedicated model.
- Missing checkpoints raise `CheckpointUnavailableError` with an explicit no-score message.

The local dashboard endpoint `POST /api/predict` keeps its existing `score`, `logit`, and `threshold` fields for compatibility and adds `link_score`, `task`, `score_type`, `probability_calibrated`, and `link_prediction_record`.

## Known limitations and blockers

- No verified company-level risk labels; risk prediction cannot be trained or claimed.
- The graph is fragmented: 23,198 weakly connected components; 40.4% of nodes are in the largest component.
- Embedding coverage is only 24.0%; zero-filled embeddings may limit performance.
- IC-SPLC and AnonNet IDs are anonymised and cannot be merged with named companies.
- SynDelay is synthetic and its class meanings are undocumented.
- The existing SynDelay feature table carries only latitude/longitude.
- Smart Logistics has 1,000 rows and unknown provenance.
- GDELT mentions are unverified and cover only a short date range.
- AnonNet and SynDelay lack a LICENSE file; confirm rights before redistribution.
- Restricted CC BY-NC disruption data requires permission and remains excluded.
- A local dashboard exists under `ml/reports/`; it is not a production backend/frontend.
- The original GraphSAGE run and the Phase 2 protocol/feature suite are experimental link-prediction results; no production GNN exists.
- SAMPOQ has one unique 136-d feature row and AnonNet has four; their embeddings are collapsed.
- IC-SPLC embedding coverage is 28.5% in the original 136-d contract, so most nodes are zero-filled.
- Link scores are uncalibrated; original-checkpoint ECE is 0.3253 and Brier score is 0.3354.
- Reverse-edge leakage remains in Protocol A and is controlled in B/C; it is not fully removable from historical data without changing the original split.

## Deliverables checklist

- [x] Repository and architecture inspection
- [x] ML folder integration without restructuring existing work
- [x] Dataset analysis report
- [x] Feature identification and leakage controls
- [x] Graph preprocessing prototype
- [x] PyG loading and forward-pass smoke test
- [x] GNN architecture recommendation
- [x] Baseline feasibility report
- [x] ML documentation
- [x] Original GraphSAGE link-prediction training and evaluation
- [x] Phase 2 protocol and feature ablations
- [x] Link-prediction inference contract and synthetic test suite
- [ ] Production GNN training and evaluation
- [ ] Disruption-risk model and verified labels

## Commands

```bash
cd /Users/yash/Desktop/Infotact

.venv/bin/python -m pytest
.venv/bin/python -m compileall -q ml
.venv/bin/python ml/preprocessing/dataset_analysis.py
.venv/bin/python ml/preprocessing/graph_preprocessing.py --smoke-test
.venv/bin/python ml/models/baseline_model.py
```

Do not retrain or overwrite experiment artifacts merely to run tests. Tests use synthetic fixtures and real split-integrity checks.

## Git hygiene

Do not commit `.venv/`, `datasets/raw/`, or generated `__pycache__` directories; these are ignored. Review `git status --short` before staging. No commit or push is performed automatically.
