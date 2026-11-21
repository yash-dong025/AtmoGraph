# AtmoGraph

## Project Overview

AtmoGraph is a supply-chain graph and machine-learning research prototype. It combines company/entity data from several sources into a directed supplier-to-customer graph and studies link prediction: estimating whether a directed supplier/customer edge is likely to exist.

The project is not a production risk system. The current datasets do not contain verified company-level disruption labels, so the repository does **not** provide disruption-risk probabilities, risk levels, or validated shipment-delay predictions.

## Architecture

```text
Raw/source datasets
        |
        v
scripts/ + config/  ->  datasets/processed/ and datasets/training/
        |
        v
ml/preprocessing/   ->  node features, edge features, PyG graph
        |
        v
ml/models/ + ml/training/  ->  GraphSAGE experiments and structural baselines
        |
        v
ml/experiments/     ->  checkpoints, metrics, predictions, reports
        |
        v
ml/reports/         ->  local dashboard and experiment catalog
```

The graph is directed. Edge direction is documented as supplier -> customer for IC-SPLC. AnonNet direction is represented as provided but is not independently verified. Link prediction scores a directed pair; it does not infer a real-world disruption probability.

## Dataset Sources

| Source | Role | Notes |
|---|---|---|
| SAMPOQ | Named company records | 16,102 rows; CC BY 4.0 |
| IC-SPLC | Supplier/customer relationships and embeddings | 109,403 nodes; 128-d embeddings for 31,182 nodes |
| AnonNet | Anonymised supply relationships | 7,245 nodes; direction and provenance require care |
| SynDelay | Synthetic shipment-delay tabular data | 155,488 rows; class semantics undocumented |
| Smart Logistics | Small logistics tabular sample | 1,000 rows; provenance unknown |
| GDELT | News mentions | 847 articles / 866 unverified mentions |

The combined graph contains **130,020 nodes** and **166,690 directed edges**. Edge splits are 133,352 train / 16,669 validation / 16,669 test. Node splits are 91,014 / 19,503 / 19,503.

Restricted CC BY-NC disruption data is excluded. Anonymised IC-SPLC and AnonNet IDs are not merged with named companies.

## ML Pipeline

1. `scripts/build_graph_dataset.py` builds the processed graph, node index, edge index, and reproducible splits.
2. `ml/preprocessing/feature_engineering.py` builds node and edge features with training-only scaling and explicit embedding masks.
3. `ml/preprocessing/graph_preprocessing.py` validates the graph and builds a PyTorch Geometric `Data` object.
4. `ml/models/` contains GCN, GraphSAGE, and GAT prototypes plus tabular baselines.
5. `ml/training/` contains the original GraphSAGE run and the Phase 2 protocol/feature-ablation suite.
6. `ml/inference/predict.py` defines the link-prediction output contract.
7. `ml/reports/` contains the local dashboard and an additive experiment catalog.

Important feature distinction:

- The full engineered node matrix is **146 dimensions** (18 numeric + 128 IC-SPLC embeddings).
- The original 136-d GraphSAGE baseline uses **8 static flags + 128 embeddings**.
- Phase 2 tests flags-only, structural, categorical, and combined configurations.
- All 12 edge features and categorical codes are not used by the original 136-d checkpoint.

## How to Run Experiments

Use the existing virtual environment:

```bash
uv pip install --python .venv/bin/python -r ml/requirements-ml.txt
uv pip install --python .venv/bin/python -r ml/requirements-test.txt
```

If `uv` is unavailable, run `.venv/bin/python -m ensurepip --upgrade` once and use `.venv/bin/python -m pip` with the same requirements files.

Run the ML tests:

```bash
.venv/bin/python -m pytest
```

Run the original link-prediction experiment only when explicitly intending to retrain:

```bash
.venv/bin/python -m ml.training.train_graphsage_link_prediction --overwrite
```

Run the Phase 2 protocol/feature suite only when explicitly intending to regenerate those outputs:

```bash
.venv/bin/python -m ml.training.run_graphsage_phase2 --output-dir ml/experiments/graphsage_phase2 --overwrite
```

Start the local dashboard:

```bash
.venv/bin/python ml/reports/serve_dashboard.py
```

Do not retrain, overwrite, or delete experiment artifacts merely to run tests. The test suite uses small synthetic fixtures.

## Results

Original GraphSAGE checkpoint on the original test split:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6727 |
| Average precision | 0.6094 |
| Precision | 0.5889 |
| Recall | 0.8924 |
| F1 | 0.7096 |

Phase 2 key findings:

- Under Protocols A/B/C, directed preferential attachment achieves pooled AP of about **0.697-0.699**.
- The best GraphSAGE configuration (`c5_current_struct_cat`) reaches pooled AP of about **0.656-0.659**.
- Under the IC-SPLC-only Protocol D, `c3_current_struct` and `c5_current_struct_cat` reach pooled AP of about **0.650-0.651**, compared with about **0.555-0.559** for preferential attachment.
- Removing reverse-edge leakage does not collapse performance in B/C.
- Original-checkpoint calibration is poor: Brier score `0.3354`, ECE `0.3253`, with weak probability calibration.
- SAMPOQ and AnonNet have collapsed 136-d feature vectors; IC-SPLC embedding coverage is only 28.5% in the original 136-d contract.

These are link-prediction results, not disruption-risk results. See `ml/reports/graphsage_phase2_experiment_report.md` and `ml/reports/graphsage_diagnostic_audit.md` for details.

## Limitations

- No verified company-level risk labels; disruption-risk prediction is unsupported.
- No cross-source entity resolution; anonymised and named identities remain separate.
- Link scores are uncalibrated model scores, not probabilities.
- The graph is fragmented; many nodes have weak neighbourhood signal.
- SAMPOQ and AnonNet feature vectors collapse to very few unique rows.
- Directional semantics are unverified for AnonNet.
- The original 136-d model ignores edge attributes and categorical codes.
- The dashboard is local research tooling, not a production API or frontend.
- Citation, licence, and provenance gaps remain for several datasets.

## Repository Structure

```text
config/                     dataset build configuration
datasets/                   raw, processed, and training artifacts
scripts/                    dataset build and validation scripts
ml/
├── experiments/            original and Phase 2 artifacts
├── inference/              link-prediction output contract
├── models/                 GCN, GraphSAGE, GAT, tabular baselines
├── preprocessing/          feature engineering and PyG builder
├── reports/                local dashboard, catalog, audits, reports
├── tests/                  synthetic ML contract and integrity tests
└── training/               original and Phase 2 training/evaluation code
pytest.ini                  test discovery configuration
```

See `ml/README.md` for ML-specific details and commands.

## License and Compliance

There is no top-level project LICENSE file in this repository. Dataset licences and provenance differ by source and must be reviewed before redistribution or external use. SAMPOQ is documented as CC BY 4.0. AnonNet and SynDelay have no LICENSE file in the acquired data. Smart Logistics provenance is unknown. Restricted CC BY-NC disruption data is excluded. GDELT records are unverified news mentions, not confirmed disruptions.

## Current Status

- Dataset pipeline: complete and reproducible for the existing graph and splits.
- Original GraphSAGE link-prediction experiment: completed and preserved.
- Phase 2 protocol and feature ablations: completed and preserved.
- Inference contract: link prediction only; disruption risk remains explicitly unsupported.
- Dashboard: local research tooling with an additive experiment catalog.
- Tests: dedicated synthetic ML suite under `ml/tests/`.
- Production readiness: not production-ready; no verified risk labels, no entity resolution, and no validated production model.
- No commit or push is performed by the project commands.
