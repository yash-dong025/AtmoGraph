# AtmoGraph ML / GNN Foundation

Week 1 ML/GNN foundation for the AtmoGraph supply-chain risk project.

This is an inspection and prototyping layer. It is **not** a production GNN, it does not claim real-world risk accuracy, and it does not replace the existing dataset pipeline, backend, frontend, or NLP work.

## Scope and status

- Member 3 scope: ML/GNN foundation, dataset analysis, graph preprocessing, PyG loading test, baseline feasibility, and documentation.
- Existing `config/`, `scripts/`, and `datasets/` outputs remain the source of truth.
- No tracked files were modified; this is additive work under `ml/` plus reports under `datasets/reports/`.
- No branch change, commit, or push was performed.
- No company-level risk model was trained because no verified risk labels exist.
- No fabricated data, labels, predictions, or metrics are used.

## Repository architecture inspection

- `origin/main` and `origin/Yash` contain `config/`, `datasets/`, and `scripts/`.
- `origin/Aiman`, `origin/Supraja`, and `origin/Vidhi` contain no backend, frontend, NLP, or ML service directories at inspection time.
- No pre-existing `backend/`, `frontend/`, `nlp/`, or `ml/` implementation was found.
- `README.md` is only `# AtmoGraph`; `README_DATASETS.md` documents the dataset pipeline.
- Integration approach: add `ml/` at the repository root and reuse existing processed datasets rather than duplicating or restructuring them.

## ML folder layout

```text
ml/
├── __init__.py
├── .gitignore
├── README.md
├── requirements-ml.txt
├── artifacts/                 # ignored; local metrics
├── saved_models/              # ignored; model weights
├── notebooks/                 # optional local notebooks
├── inference/
│   ├── __init__.py
│   └── predict.py             # prediction output contract
├── models/
│   ├── __init__.py
│   ├── baseline_model.py      # scikit-learn delay baselines
│   └── gcn_model.py           # GCN/GraphSAGE/GAT prototypes
└── preprocessing/
    ├── __init__.py
    ├── dataset_analysis.py    # dataset/report generator
    ├── feature_engineering.py # node/edge feature construction
    └── graph_preprocessing.py # PyG Data builder and smoke test
```

Generated reports:

- `datasets/reports/ml_dataset_analysis.md`
- `datasets/reports/ml_dataset_analysis.json`
- `datasets/reports/ml_baseline_report.md`

## Environment and dependencies

- Python 3.13.15, macOS arm64.
- Local virtual environment: `.venv/` (ignored by Git).
- Installed: `torch 2.15.0`, `torch-geometric 2.8.1`, `pandas 3.1.1`, `numpy 2.5.4`, `scikit-learn 1.9.1`, `pyarrow 26.0.1`.
- PyTorch MPS is available; the smoke test used CPU for deterministic preprocessing checks.
- Existing `requirements-data.txt` was not modified.

Install ML dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r ml/requirements-ml.txt
```

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
- Total model input: **146 dimensions** (18 numeric + 128 embeddings).

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

Prototypes exist in `ml/models/gcn_model.py`: `AtmoGCN`, `AtmoGraphSAGE`, `AtmoGAT`, and `build_gnn()`. No GNN training metrics are claimed yet; only the forward-pass smoke test has been run.

## Training and evaluation methodology

Existing splits from `datasets/training/dataset_splits.json`:

- Node splits: train 91,014 / val 19,503 / test 19,503.
- Edge splits: train 133,352 / val 16,669 / test 16,669.
- SynDelay 3-class: train 108,841 / val 23,323 / test 23,324.
- Smart Logistics binary: train 700 / val 150 / test 150.

Planned link prediction:

- Train on train edges, tune on validation edges, evaluate once on test edges.
- Sample negative edges with the same split discipline.
- Report AUC, average precision, Hits@K, and MRR.
- Compare GCN, GraphSAGE, and GAT under the same feature set and splits.

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

## Prediction output contract

`ml/inference/predict.py` defines the backend-facing record:

```json
{
  "company_id": "sampoq:0000320193",
  "risk_score": 0.82,
  "risk_level": "HIGH",
  "prediction_type": "disruption_risk",
  "model_version": "gcn_v1",
  "prediction_status": "experimental"
}
```

Rules:

- `risk_score` must be in `[0, 1]`.
- `risk_level` must be `LOW`, `MEDIUM`, or `HIGH`.
- `prediction_status` must remain `experimental` until a labelled evaluation is completed and approved.
- The caller must supply a score from a trained model; the contract does not generate predictions.

Suggested backend integration, after Member 1 confirms the API shape:

```text
POST /predict
request:  { "company_id": "sampoq:0000320193" }
response: { "company_id": "...", "risk_score": 0.82, "risk_level": "HIGH",
            "prediction_type": "disruption_risk", "model_version": "gcn_v1",
            "prediction_status": "experimental" }
```

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
- No backend, frontend, or NLP implementation exists to integrate with yet.
- No production GNN has been trained or evaluated.
- The current branch is ahead of `origin/Yash` by one commit; no commit or push was performed for this work.

## Week 1 deliverables checklist

- [x] Repository and architecture inspection
- [x] ML folder integration without restructuring existing work
- [x] Dataset analysis report
- [x] Feature identification and leakage controls
- [x] Graph preprocessing prototype
- [x] PyG loading and forward-pass smoke test
- [x] GNN architecture recommendation
- [x] Baseline feasibility report
- [x] ML documentation
- [ ] Production GNN training and evaluation (out of Week 1 scope)

## Commands

```bash
cd /Users/yash/Desktop/Infotact

.venv/bin/python -m compileall -q ml
.venv/bin/python ml/preprocessing/dataset_analysis.py
.venv/bin/python ml/preprocessing/graph_preprocessing.py --smoke-test
.venv/bin/python ml/models/baseline_model.py
```

## Suggested Git commands

Do not commit `.venv/` or `datasets/raw/`; both are already ignored.

```bash
git status --short
git add ml DATASET_INSPECTION.md DATASET_DOWNLOAD_PLAN.md \
  datasets/reports/ml_dataset_analysis.md \
  datasets/reports/ml_dataset_analysis.json \
  datasets/reports/ml_baseline_report.md
git commit -m "Add ML/GNN Week 1 foundation"
```

No commit or push was executed automatically.
