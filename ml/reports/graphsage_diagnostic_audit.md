# AtmoGraph GraphSAGE Diagnostic Audit

**Date:** 2026-11-17  
**Scope:** Audit only. No source code, model, dataset, split, checkpoint, dashboard, or Git state was modified. The only file added is this report.  
**Experiment:** `ml/experiments/graphsage_link_prediction/`  
**Dataset version:** `atmo_graph_v1_015d0c924af0`

This audit evaluates **directed link prediction**. It does not evaluate company disruption risk, shipment delay, or disruption propagation. No verified company-level risk labels exist in the acquired data.

---

## 1. Executive Summary

### Verdict

The GraphSAGE experiment is **partially correct**.

The core training and evaluation plumbing is mostly sound:

- The model returns raw logits.
- `BCEWithLogitsLoss` receives raw logits.
- The sigmoid is applied once for evaluation.
- Labels have the correct shape and positive/negative meaning.
- The validation threshold is selected on validation data only and reproduced exactly.
- Message passing uses train-split edges only.
- Negative samples are source-matched, exclude observed directed edges, and contain no duplicates.
- GraphSAGE and the structural baselines use identical test pairs.
- The reported ranking metrics and confusion matrix match the artifacts.

However, the experiment is **not trustworthy as evidence that GraphSAGE is a good link predictor for all three sources**. The dominant reason is not a loss-function bug. It is the feature contract:

- The model uses **136 features**: 8 static flags plus 128 IC-SPLC embeddings.
- It silently ignores the actual categorical codes (`country_code`, `industry_code`, `product_category_code`) and all 12 edge features.
- It excludes all degree/structural features, without replacing them with train-only structural features.
- Every SAMPOQ node has the **same 136-dimensional feature vector**.
- AnonNet has only **4 unique 136-dimensional feature rows**.
- Only 31,182 IC-SPLC nodes (28.5%) have embeddings; the remaining 78,221 have zero-filled embeddings and a small number of flag patterns.

The second major issue is **severe overconfidence**:

- Test logits range from `-21.24` to `+25.93`.
- 18,895 / 33,338 test scores are `>= 0.99`.
- 11,101 scores are exactly `1.0`, including 4,444 negative pairs.
- Test Brier score is `0.3354`.
- Test expected calibration error (ECE, 10 equal-width bins) is `0.3253`.
- Test log loss is `6.1535` using `sklearn.metrics.log_loss` on the saved float64 scores. This value is sensitive to clipping for exact `1.0` predictions; it is a saturation diagnostic, not a stable official metric.

The third major issue is **metric interpretation**. GraphSAGE has higher ROC-AUC than Preferential Attachment but lower Average Precision, and much worse precision at the top of the ranking:

- GraphSAGE: AUC `0.6727`, AP `0.6094`.
- Preferential Attachment: AUC `0.5226`, AP `0.6983`.
- GraphSAGE precision@100 is `0.58`; PA precision@100 is `1.00`.
- GraphSAGE precision@500 is `0.61`; PA precision@500 is `0.998`.

For a candidate-ranking use case, the current GraphSAGE model is **not better than PA overall**, and PA/CN are much better at the very top of the ranking. For balanced discrimination, GraphSAGE is better. These are different objectives and must not be conflated.

Finally, the random edge split has a **reverse-edge leakage limitation**:

- 566 test positive edges have their reverse direction in the train split.
- This is 10.0% of AnonNet test positives, 4.9% of SAMPOQ test positives, and 0.45% of IC-SPLC test positives.
- PA and CN treat train edges as undirected, so these reverse-train positives are structurally supported by the baseline graph.
- The current split is transductive and random; it is not a temporal/future-edge split.

### Highest-impact findings

1. **Feature starvation:** the model sees only 8 non-embedding flags; SAMPOQ nodes are indistinguishable; AnonNet has 4 unique feature rows.
2. **Overconfidence:** unbounded dot-product logits, no calibration, many exact `1.0` predictions, including negatives.
3. **Ranking-metric mismatch:** better AUC but worse AP and much worse precision@K than PA/CN.
4. **Split/protocol leakage:** reverse-direction edges cross train/test; baselines are undirected; split is random rather than temporal.
5. **Source heterogeneity:** one pooled model is not appropriate for AnonNet, IC-SPLC, and SAMPOQ in their current feature regimes.

---

## 2. Current Experiment Reference Results

### 2.1 Overall test performance

Values verified from `metrics.json` and `test_predictions.csv`.

| Model | ROC-AUC | AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| GraphSAGE | 0.6727 | 0.6094 | 0.5889 | 0.8924 | 0.7096 |
| Preferential Attachment | 0.5226 | 0.6983 | 0.7606 | 0.4324 | 0.5513 |
| Common Neighbors | 0.6129 | 0.6122 | 0.9491 | 0.2373 | 0.3796 |

GraphSAGE uses the validation-selected threshold `0.11992187052965164`. PA uses validation-selected threshold `12.0`; CN uses `1.0`.

### 2.2 GraphSAGE confusion matrix at threshold 0.1199

Verified from `metrics.json`:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 6,286 | 10,383 |
| Actual positive | 1,794 | 14,875 |

The model is optimized for recall at this threshold. It produces 10,383 false positives and 1,794 false negatives. Precision is 58.9%, recall is 89.2%.

### 2.3 Test-set composition

- Total pairs: 33,338.
- Positive edges: 16,669.
- Negative pairs: 16,669.
- Cross-source negative pairs: 0.
- Positive/negative pairs are source-balanced:

| Source | Positives | Negatives | Total rows |
|---|---:|---:|---:|
| AnonNet | 4,400 | 4,400 | 8,800 |
| IC-SPLC | 10,659 | 10,659 | 21,318 |
| SAMPOQ | 1,610 | 1,610 | 3,220 |

The test set is 50/50 positive/negative by construction. This makes the random AP baseline `0.5`. It does **not** represent a realistic candidate pool, where positives are usually much rarer. Pooled precision and AP therefore should not be interpreted as real-world precision.

### 2.4 Per-source GraphSAGE and best baseline

Verified from `metrics.json` and `baseline_metrics.json`.

| Source | GraphSAGE AUC | GraphSAGE AP | GraphSAGE F1 | Best baseline | Best baseline AP |
|---|---:|---:|---:|---|---:|
| AnonNet | 0.6267 | 0.5895 | 0.6708 | Preferential Attachment | 0.9146 |
| IC-SPLC | 0.7353 | 0.6511 | 0.7435 | Preferential Attachment | 0.6289 |
| SAMPOQ | 0.5320 | 0.5121 | 0.5999 | Preferential Attachment | 0.6438 |

The validation-selected baseline is Preferential Attachment, selected by highest validation AP (`0.70275` vs CN `0.61773`), then validation ROC-AUC. CN is the alternative baseline.

### 2.5 Derived calibration diagnostics

These values are **not** fields in `metrics.json`; they were computed from the saved `test_predictions.csv` during this audit. No artifact was changed.

| Diagnostic | GraphSAGE test | GraphSAGE validation |
|---|---:|---:|
| Brier score | 0.3354 | 0.3297 |
| Log loss (`sklearn`, float64) | 6.1535 | 5.9710 |
| ECE, 10 equal-width bins | 0.3253 | 0.3191 |
| Scores `>= 0.99` | 18,895 | 18,795 |
| Scores `<= 0.01` | 6,155 | 6,118 |
| Scores exactly `1.0` | 11,101 | 10,933 |
| Negative pairs exactly `1.0` | 4,444 | 4,263 |
| Positive pairs exactly `1.0` | 6,657 | 6,670 |

The log-loss value depends on how exact `1.0` predictions are clipped. With `sklearn.metrics.log_loss` on float64 scores, it is `6.1535`. With a fixed `1e-7` clip it is approximately `3.476`. The qualitative conclusion, severe overconfidence, does not depend on the clipping choice.

---

## 3. Repository and Artifact Inventory

### 3.1 Components

| Component | File | Purpose | Status |
|---|---|---|---|
| GraphSAGE/GCN/GAT definitions | `ml/models/gcn_model.py` | Model classes and decoder | Implemented; GraphSAGE in use |
| GraphSAGE training | `ml/training/train_graphsage_link_prediction.py` | Full-batch training, sampling, evaluation, artifact writing | Implemented; used for checkpoint |
| Structural baselines | `ml/training/evaluate_link_prediction_baseline.py` | PA/CN scoring, negative sampling, metrics, thresholding | Implemented; used for baseline artifacts |
| Feature construction | `ml/preprocessing/feature_engineering.py` | Node/edge feature matrices, scaler, embeddings | Implemented; categorical/edge outputs unused by model |
| PyG graph builder | `ml/preprocessing/graph_preprocessing.py` | Builds `Data` with `x`, `edge_index`, `edge_attr`, `x_cat`, `x_emb`, masks | Implemented; only `x` and `edge_index` used in training |
| Inference contract | `ml/inference/predict.py` | Formats a risk score record | Contract only; not a link-prediction model |
| Dashboard data assembly | `ml/reports/dashboard_data.py` | Reads experiment artifacts and serves metrics | Reads artifacts; does not recompute metrics |
| Delay baseline | `ml/models/baseline_model.py` | HistGradientBoosting for shipment-delay tasks | Unrelated to link prediction |
| Delay baseline artifact | `ml/artifacts/baseline_metrics.json` | Shipment-delay baseline metrics | Unrelated to GraphSAGE link prediction |
| Split artifact | `datasets/training/dataset_splits.json` | Reproducible node/edge/delay split ID lists | Present; generator not in repository |
| Graph edge split | `datasets/training/edge_index.csv` | Directed edges with train/val/test split | Present |
| Node index | `datasets/training/node_index_mapping.csv` | Stable node ordering | Present |
| Node features | `datasets/processed/graph/node_features.csv` | 146-column feature source | Present |
| Node embeddings | `datasets/training/node_embeddings.parquet` | 128-d IC-SPLC embeddings plus mask | Present |
| Edges | `datasets/processed/graph/edges.csv` | Edge metadata and features | Present |

### 3.2 Exact experiment artifacts used

Checkpoint:

- `ml/experiments/graphsage_link_prediction/checkpoint.pt`  
  Size: 94,965 bytes. Contains `model_state_dict`, model config, selected feature names, excluded features, seed, best epoch, and threshold. It does **not** contain optimizer state, scheduler state, or RNG state.

Reports and predictions:

- `ml/experiments/graphsage_link_prediction/metrics.json`  
  Official pooled and per-source metrics, threshold selection, test curves, score distribution.
- `ml/experiments/graphsage_link_prediction/history.json`  
  184 training epochs; best validation AP at epoch 164.
- `ml/experiments/graphsage_link_prediction/test_predictions.csv`  
  33,338 rows with pair indices, company IDs, source, label, score, and raw logit.
- `ml/experiments/graphsage_link_prediction/baseline_metrics.json`  
  PA/CN validation and test metrics, per-source metrics, threshold selection, protocol.
- `ml/experiments/graphsage_link_prediction/baseline_test_predictions.csv`  
  33,338 rows; same pair ordering as GraphSAGE.
- `ml/experiments/graphsage_link_prediction/run_summary.json`
- `ml/experiments/graphsage_link_prediction/config.json`
- `ml/experiments/graphsage_link_prediction/dataset_metadata.json`
- `ml/experiments/graphsage_link_prediction/experiment.json`

### 3.3 Missing artifacts

- No saved **validation predictions**. Validation metrics exist, but validation scores/logits were not persisted. For this audit, validation samples were reconstructed read-only from the checkpoint and the documented seed/sampler; the validation threshold reproduced exactly.
- No calibration artifact.
- No split-generation script. Only the split artifact and the data-dictionary description exist.
- No automated test suite for the GNN, sampler, metrics, or dashboard.
- No optimizer/RNG state in the checkpoint, so training is not resumable.
- No per-source validation prediction files.

### 3.4 Data integrity

Current dataset files match the SHA-256 values recorded in `dataset_metadata.json`:

- `datasets/processed/graph/nodes.csv`
- `datasets/processed/graph/edges.csv`
- `datasets/training/edge_index.csv`
- `datasets/training/node_index_mapping.csv`
- `datasets/processed/graph/node_features.csv`

Environment used for diagnostics:

- Python 3.13.15
- torch 2.15.0
- torch-geometric 2.8.1
- scikit-learn 1.9.1
- NumPy 2.5.4
- pandas 3.1.1

Training command recorded in `run_summary.json`:

```text
/Users/yash/Desktop/Infotact/ml/training/train_graphsage_link_prediction.py --overwrite
```

Recorded duration: 53.9 seconds on CPU.

---

## 4. Graph and Split Integrity Findings

### 4.1 Dataset counts

| Quantity | Value |
|---|---:|
| Nodes | 130,020 |
| Directed edges | 166,690 |
| Train edges | 133,352 |
| Validation edges | 16,669 |
| Test edges | 16,669 |
| Cross-source edges | 0 |
| Duplicate directed pairs | 0 |
| Self-loops | 0 |
| Node splits | 91,014 / 19,503 / 19,503 |
| Edge splits | 80% / 10% / 10% |
| Weakly connected components | 23,198 |
| Largest component share | 40.36% |

Per-source graph statistics:

| Source | Nodes | Edges | Avg. undirected degree | Median degree | Density (directed) | Components |
|---|---:|---:|---:|---:|---:|---:|
| AnonNet | 7,245 | 43,922 | 11.34 | 3 | 0.000837 | 1 |
| IC-SPLC | 109,403 | 106,683 | 1.94 | 1 | 0.0000089 | 22,354 |
| SAMPOQ | 13,372 | 16,085 | 2.33 | 1 | 0.0000900 | 843 |

The graph is highly fragmented. IC-SPLC has 16 isolated nodes in the full graph and 18,178 nodes isolated in the train graph. Large parts of the train graph contain no usable neighborhood for a positive test edge.

### 4.2 Split integrity

Verified:

- Exact train/validation/test edge overlap: 0.
- Node split overlap: 0.
- Node split union: 130,020, all nodes assigned exactly once.
- Node splits are source-stratified.
- `dataset_splits.json` edge ID lists exactly match `edge_index.csv` split assignments.
- No self-loops.
- No exact duplicate directed pairs.
- No cross-source edges.

Edge splits are random 80/10/10, as documented in `datasets/metadata/data_dictionary.md:102-113`. Node splits are stratified 70/15/15. There is no temporal split, no future-edge holdout, and no edge-disjoint-by-entity split.

### 4.3 Directed reverse-edge structure

The graph contains reverse-direction pairs. A→B and B→A are treated as distinct edges, but this matters for leakage and for undirected baselines.

| Quantity | Count |
|---|---:|
| Directed pairs with a reverse present in full graph | 7,366 |
| Train edges with internal reverse | 4,720 |
| Validation edges with internal reverse | 60 |
| Test edges with internal reverse | 72 |
| Validation positives whose reverse is in train | 619 |
| Test positives whose reverse is in train | 566 |

Test positives with reverse in train, by source:

| Source | Reverse-in-train positives | Share of source positives |
|---|---:|---:|
| AnonNet | 439 / 4,400 | 9.98% |
| IC-SPLC | 48 / 10,659 | 0.45% |
| SAMPOQ | 79 / 1,610 | 4.91% |

This is not exact-positive leakage, because the reverse direction is a different directed edge. It is still a protocol limitation:

- PA and CN treat train edges as **undirected** (`ml/training/evaluate_link_prediction_baseline.py:91-99`, `216-228`). A test edge B→A with train edge A→B is therefore structurally supported by the baseline.
- The protocol even documents this: `common_neighbors_graph` is "train edges treated as undirected" and PA is "undirected train degree product" (`evaluate_link_prediction_baseline.py:603-618`).
- The current split is random and transductive. Reverse-direction and near-duplicate relationships can cross split boundaries.

A diagnostic sensitivity check that removed reverse-in-train positives (while keeping all negatives) changed pooled AP as follows. This is not a corrected official metric because it changes the positive/negative prevalence; it is only a sensitivity check.

| Model | Full test AP | AP excluding reverse-in-train positives |
|---|---:|---:|
| PA | 0.6983 | 0.6829 |
| CN | 0.6122 | 0.5987 |
| GraphSAGE | 0.6094 | 0.5989 |

AnonNet GraphSAGE AP fell from `0.5895` to `0.5646` under this sensitivity check. The effect is real but not large enough to explain the overall model weaknesses. It should still be fixed in the protocol.

### 4.4 Negative sampling

Verified from the sampler and prediction files:

- Validation and test each have 16,669 positives and 16,669 negatives.
- Negatives are source-matched: 0 cross-source negatives.
- Negative source counts exactly match positive source counts per source.
- No negative pair is an observed directed edge.
- No duplicate negative pairs.
- No negative pair appears twice in either direction within the negative set.
- Only 3 test negatives are the reverse of a train edge; this is negligible for the negative set.

Classification: **negative sampling is correct with respect to exact directed observed edges.** The unresolved issue is reverse-edge treatment for positives and for undirected baselines, not the exclusion rule itself.

### 4.5 Message-passing leakage

`ml/training/train_graphsage_link_prediction.py:308-311`:

```python
data = build_pyg_data()
feature_indices, selected_feature_names = select_features(data, excluded_features)
x = data.x[:, feature_indices].contiguous().to(device)
message_edge_index = data.edge_index[:, data.train_edge_mask].contiguous().to(device)
```

The model encodes with `message_edge_index`, which is restricted to `train_edge_mask`. Validation and test edges are never passed to message passing. No exact positive-edge leakage into message passing was found.

Graph-derived node features (`degree`, `in_degree`, `out_degree`, `component_size`, edge-year fields, `is_isolated`) are excluded from the model. That is the correct leakage-control decision, but it also removes all structural signal. The right replacement is to compute train-only structural features, not to remove structural information entirely.

### 4.6 Is the split appropriate?

For **transductive link prediction on the existing graph**, a random edge split is defensible: all nodes are known, and the task is to recover held-out edges. It is not appropriate for forecasting future supply-chain links or for evaluating generalization to new entities.

For the project's likely applied use, the following limitations matter:

- No temporal split. The dataset spans 2004-2026, but the split is random.
- Reverse-direction edges can cross splits.
- AnonNet direction is not formally documented (`datasets/processed/graph/graph_metadata.json`), so directed evaluation semantics are uncertain for that source.
- Structural baselines are undirected while the model is directed, so the comparison is not apples-to-apples.
- The test set is 50/50 positive/negative, not a realistic candidate distribution.

Classification: **appropriate only as a transductive interpolation benchmark; not sufficient for future-link or production claims.**

---

## 5. Architecture and Feature Audit

### 5.1 Architecture

From `ml/models/gcn_model.py` and checkpoint config:

| Property | Value |
|---|---|
| Model | `AtmoGraphSAGE` |
| Layers | 2 |
| Convolution | `SAGEConv`, default mean aggregation |
| Input dim | 136 |
| Hidden dim | 64 |
| Output embedding dim | 32 |
| Activation | ReLU after first layer only |
| Normalization | LayerNorm after each convolution |
| Dropout | 0.2 after first activation |
| Decoder | Raw dot product `(z[u] * z[v]).sum(-1)` |
| Parameters | 21,792 |
| Checkpoint size | 94,965 bytes |
| Optimizer | Adam, lr 1e-3, weight decay 1e-5 |
| Loss | `BCEWithLogitsLoss` |
| Batch | Full-batch graph encode; one negative set per epoch |
| Early stopping | Validation AP, patience 20 |
| Best epoch | 164 of 184 |

The encoder is `_TwoLayerEncoder` (`gcn_model.py:35-63`). The decoder is lines 59-60:

```python
def decode(self, z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
```

There is no decoder bias, no embedding normalization before the dot product, no learned logit scale, and no temperature parameter.

### 5.2 Feature dimensions

Verified:

- Total node feature matrix: `146` columns.
- Total edge feature matrix: `12` columns.
- GraphSAGE input: `136` columns.
- Excluded node columns: `10`.
- Used non-embedding columns: `8`.
- Used embedding columns: `128`.

The 8 used non-embedding features are:

| Feature | What it encodes |
|---|---|
| `source_sampoq` | Source indicator |
| `source_ic_splc` | Source indicator |
| `source_anonnet` | Source indicator |
| `has_country` | Whether a country is present |
| `has_industry` | Whether an industry is present |
| `has_product_category` | Whether a product category is present |
| `embedding_available` | Whether an IC-SPLC embedding exists |
| `is_anonymised` | Anonymization flag |

Feature construction is in `ml/preprocessing/feature_engineering.py:204-225`. The model selection is in `train_graphsage_link_prediction.py:72-81`. The model only receives `x` (`train_graphsage_link_prediction.py:308-311`).

The excluded 10 are:

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

They are excluded because they were computed from the full graph/edge set and could leak validation/test structure. That reasoning is correct. The problem is that no train-only structural replacement is supplied.

### 5.3 Features that are built but ignored

The preprocessing layer provides additional information that never reaches the model:

- **Categorical indices:** `x_cat` with cardinalities `[282, 23, 617]` for country, industry, and product category (`feature_engineering.py:219-221`). These are stored in PyG `Data` (`graph_preprocessing.py:53`) but never passed to `build_gnn` or used by `AtmoGraphSAGE`.
- **Edge features:** 12 columns including relationship type, confidence, first/last year, edge-year span, edge count, and source indicators (`feature_engineering.py:278-321`). Stored as `edge_attr` (`graph_preprocessing.py:52`) but never used.
- **Embedding mask:** `emb_mask` is stored but not used by the model; the model sees zero-filled embeddings without an explicit mask input.
- **Raw categorical values:** `country_code`, `industry_code`, `product_category_code` are not passed as embeddings or one-hot features.

Search evidence: `x_cat` and `edge_attr` appear only in preprocessing/building code, not in model or training code. `ml/models/gcn_model.py` accepts only `x` and `edge_index`.

### 5.4 Feature collapse by source

This is the most important feature-audit result. The 136-dimensional model input has very little variation within sources.

| Source | Nodes | Unique 136-d model feature rows | Unique 8-flag patterns | Nodes with embeddings |
|---|---:|---:|---:|---:|
| AnonNet | 7,245 | 4 | 4 | 0 |
| IC-SPLC | 109,403 | 31,187 | 13 | 31,182 |
| SAMPOQ | 13,372 | 1 | 1 | 0 |

Consequences:

- **SAMPOQ:** every node has the identical model input. The model cannot distinguish two SAMPOQ companies by attributes. Any learned difference must come from graph structure through two mean-aggregation layers. With sparse degree and no structural features, this is close to a constant/degree-only model. That is consistent with AUC `0.5320` and AP `0.5121`, close to random.
- **AnonNet:** only 4 unique feature rows (country-present × product-present combinations). Actual country and product values are ignored. The model cannot use the 103 countries or 601 product codes present in AnonNet. Structural degree is also excluded. PA, which directly uses degree, has AP `0.9146`; GraphSAGE has `0.5895`.
- **IC-SPLC:** 31,182 nodes have 128-d embeddings, but 78,221 nodes have zero-filled embeddings. 77,891 IC-SPLC nodes share the same 8 flags and zero embeddings. The source still benefits from the embedding subset, which is consistent with GraphSAGE AP `0.6511` vs PA `0.6289`. But most nodes remain attribute-identical to each other.

### 5.5 Feature coverage

| Source | Has country | Has industry | Has product | Has embedding | Anonymized |
|---|---:|---:|---:|---:|---:|
| AnonNet | 79.4% | 0.0% | 92.6% | 0.0% | 100% |
| IC-SPLC | 23.0% | 9.7% | 1.7% | 28.5% | 100% |
| SAMPOQ | 0.0% | 0.0% | 0.0% | 0.0% | 0% |

The actual categorical codes are available for many nodes even where the `has_*` flag is 0? No: a missing code has `-1`, so the flag reflects presence. The key point is that the model only sees presence flags, not the code values.

### 5.6 Embeddings

- 31,182 / 130,020 nodes have embeddings (24.0%).
- All available embeddings have unit L2 norm (median 1.0, max 1.0000004).
- Missing embeddings are zero-filled and flagged by `embedding_available`.
- Embeddings exist only for IC-SPLC.
- The model uses a zero vector for AnonNet and SAMPOQ; it does not use the mask as an input feature, although `embedding_available` is included as a flag.

Unit-normalized input embeddings are not the cause of large final logits. The final learned embeddings have norm around `5.0`, and the dot product is unbounded.

### 5.7 Source information

Source information is used directly through source-indicator features. The model is therefore source-aware, not source-blind. Because negatives are source-matched, the source flag is constant across positive and negative pairs within a source, so it does not directly separate labels within a source. It can still let the shared model specialize its weights by source.

### 5.8 Classification

| Finding | Classification |
|---|---|
| Only 8 non-embedding features enter the model | Confirmed implementation gap |
| `x_cat` and `edge_attr` are built but unused | Confirmed implementation gap |
| All structural features are removed without train-only replacements | Confirmed design gap |
| SAMPOQ has one unique model feature row | Confirmed data/feature defect |
| AnonNet has four unique model feature rows | Confirmed data/feature defect |
| Feature scaler is fit on node-train split only | No issue found |
| Missing embeddings are zero-filled with an explicit mask in preprocessing | Correct preprocessing; mask unused by model |
| Source information is indirectly/directly used | Confirmed; source flags are inputs |

---

## 6. Loss, Logits, and Probability Audit

### 6.1 What the model returns

`AtmoGraphSAGE` returns **raw logits**. `decode` is a dot product with no sigmoid. There is no final sigmoid inside the model.

Trace:

1. `AtmoGraphSAGE.encode` (`gcn_model.py:50-57`) returns 32-dimensional embeddings.
2. `_TwoLayerEncoder.decode` (`gcn_model.py:59-60`) returns `(z[src] * z[dst]).sum(-1)`.
3. `train_epoch` (`train_graphsage_link_prediction.py:148-182`) calls `model.decode` and passes the result to `F.binary_cross_entropy_with_logits` at line 179.
4. `evaluate_epoch` (`train_graphsage_link_prediction.py:185-212`) calls `decode_pairs`, applies sigmoid once at line 202, then computes ranking metrics.
5. Final evaluation (`train_graphsage_link_prediction.py:468-485`) applies sigmoid once.

### 6.2 Loss audit

| Check | Result | Evidence |
|---|---|---|
| Model outputs raw logits | Yes | `gcn_model.py:59-60` |
| Loss expects logits | `BCEWithLogitsLoss` / `binary_cross_entropy_with_logits` | `train_graphsage_link_prediction.py:179`, `205` |
| Double sigmoid in training | No | No sigmoid before loss |
| Double sigmoid in evaluation | No | Sigmoid applied once at `202`, `484-485` |
| Labels are positive=1, negative=0 | Yes | `make_samples`, `train_graphsage_link_prediction.py:120-125` |
| Labels are float32 for loss | Yes | Lines 178, 203 |
| Labels have shape `[N]` | Yes | One label per pair |
| Metrics use probabilities | Yes | Scores from sigmoid |
| Log loss in official metrics | No | Not present in `metrics.json` |

**No issue found** in the logits/loss/probability plumbing. The overconfidence is not caused by passing probabilities into `BCEWithLogitsLoss` or by applying sigmoid twice.

### 6.3 Logit distribution

| Statistic | Validation | Test |
|---|---:|---:|
| Min logit | -21.2381 | -21.2381 |
| Max logit | 26.0921 | 25.9332 |
| Mean logit | 8.3528 | 8.4647 |
| Std logit | 12.7888 | 12.8125 |
| Positive mean logit | 12.0700 | 12.0458 |
| Negative mean logit | 4.6356 | 4.8836 |
| Positive logits `>= 20` | 35.4% | 35.2% |
| Negative logits `>= 20` | 23.8% | 24.7% |

The model is not merely ranking; it is producing extremely large magnitudes. A logit of 20 corresponds to a probability of approximately `0.999999998`.

### 6.4 Initialization and training trajectory

A read-only reconstruction with the documented seed and architecture shows that the **untrained** model already has a positive bias on the validation pairs:

- Initial validation logits: min `8.93`, max `32.00`, mean `26.14`, std `5.88`.
- Initial pre-training validation loss: `12.8895`.
- Epoch 1 train loss: `10.6384`; validation loss: `11.8337`.

Training reduces the loss but does not produce calibrated probabilities:

| Epoch | Train loss | Validation loss | Validation AUC | Validation AP |
|---:|---:|---:|---:|---:|
| 1 | 10.6384 | 11.8337 | 0.5125 | 0.5092 |
| 10 | 5.0680 | 7.6793 | 0.5670 | 0.5511 |
| 25 | 3.1553 | 6.0176 | 0.6369 | 0.5830 |
| 50 | 2.2297 | 6.1051 | 0.6354 | 0.5879 |
| 100 | 1.6919 | 5.3353 | 0.6616 | 0.6057 |
| 150 | 1.4434 | 4.6446 | 0.6761 | 0.6142 |
| 164 (best AP) | 1.3958 | 4.4403 | 0.6807 | 0.6171 |
| 184 (final) | 1.3340 | 4.3562 | 0.6771 | 0.6144 |

The best checkpoint is selected by validation AP, not by validation loss or calibration. Validation AP peaks at epoch 164 while validation loss continues to decrease slowly. The model is not classically overfitting in the training-loss sense; it is overfitting the ranking objective while producing increasingly confident scores.

### 6.5 Calibration diagnostics

Using validation-only fitting was not performed in this audit. Diagnostics on existing artifacts:

| Diagnostic | Value |
|---|---:|
| Test Brier | 0.3354 |
| Test log loss (float64, sklearn) | 6.1535 |
| Test ECE (10 bins) | 0.3253 |
| Validation Brier | 0.3297 |
| Validation log loss (float64) | 5.9710 |
| Validation ECE (10 bins) | 0.3191 |

Reliability at the extremes, test set:

| Bin | Count | Mean confidence | Empirical positive rate | Gap |
|---|---:|---:|---:|---:|
| 0.0-0.1 | 7,918 | 0.0101 | 0.2209 | 0.2108 |
| 0.1-0.2 | 614 | 0.1440 | 0.3974 | 0.2534 |
| 0.8-0.9 | 430 | 0.8563 | 0.4953 | 0.3609 |
| 0.9-1.0 | 22,348 | 0.9932 | 0.6072 | 0.3860 |

The top bin contains 67.0% of all test pairs; the model says approximately 99.3% positive, but only 60.7% are positive. The bottom bin is also overconfident in the negative direction: 1.0% predicted probability, 22.1% actual positive rate. This is severe miscalibration in both directions.

---

## 7. Overconfidence Investigation

### 7.1 Confirmed cause

The primary confirmed cause is **model/decoder scale**, not a loss bug:

- The decoder is an unbounded dot product (`gcn_model.py:59-60`).
- Final embeddings have norm around `5.0` (median 5.018, mean 5.027, max 5.219).
- A 32-dimensional dot product of high-norm embeddings can easily reach ±20 to ±26.
- There is no embedding normalization, decoder bias, logit scale, temperature, or probability calibration.
- Weight decay is only `1e-5`; there is no gradient clipping.
- The untrained model already has mean validation logit `26.14`, so training starts in a saturated regime.
- The loss is unweighted BCE on a 50/50 negative sample; it is not responsible for double-sigmoid saturation.

This is classified as a **confirmed model-design/optimization cause**, not a confirmed code bug. The code is doing what it says; the architecture is producing uncalibrated, overconfident scores.

### 7.2 Ruled out

| Hypothesis | Classification | Evidence |
|---|---|---|
| Probabilities passed to `BCEWithLogitsLoss` | Ruled out | Raw logits at `train_graphsage_link_prediction.py:177-179` |
| Double sigmoid in evaluation | Ruled out | One sigmoid at `202`, `484-485` |
| Wrong positive class | Ruled out | Positive=1; metrics and confusion matrix consistent |
| Labels wrong dtype/shape | Ruled out | int64 then float32, shape `[N]` |
| Threshold selected on test | Ruled out | `threshold_from_validation`; reproduced exactly |
| Metrics computed on wrong scores | Ruled out | PA recomputation matches baseline CSV exactly; GraphSAGE logits reproduce CSV within `9.6e-7` |
| Exact positive-edge leakage into message passing | Ruled out | Only `train_edge_mask` passed to encoder |
| Negative samples contain observed edges | Ruled out | 0 observed negatives |
| Cross-source negatives | Ruled out | 0 |

### 7.3 Remaining hypotheses

- The initial positive bias may be partly caused by SAGEConv/LayerNorm initialization and the shared positive direction of initial embeddings. This is a likely contributor but not fully isolated.
- Full-batch training with one negative sample per positive per epoch may encourage extreme scores, but this is not proven to be causal.
- No early stopping on calibration or log loss allows the model to continue increasing confidence after AP peaks.
- Source flags may encourage source-specific embedding offsets, but source-matched negatives prevent source flags from directly separating labels.

---

## 8. Evaluation Metric Audit

### 8.1 Metric implementations

From `ml/training/evaluate_link_prediction_baseline.py`:

- `binary_metrics` (`254-279`): `roc_auc_score`, `average_precision_score`, `precision_recall_fscore_support(average="binary")`, `confusion_matrix(labels, predictions, labels=[0,1])`.
- `metrics_by_source` (`282-314`): same metrics on source subsets.
- `threshold_from_validation` (`231-251`): maximum F1 over thresholds from `precision_recall_curve`, excluding the trivial all-positive threshold.
- `curve_payload` (`325-333`): ROC and PR curves.
- `score_distribution` (`336-379`): histograms and min/max/mean.

The metric code is correct for binary link prediction:

- Positive class is 1.
- Scores are probabilities from sigmoid or structural scores.
- Higher score means more likely positive.
- Confusion matrix labels are `["negative", "positive"]`.
- Threshold predictions use `scores >= threshold`.
- AP and ROC-AUC use the same labels and scores.
- Per-source aggregation is on the correct subset.

**No issue found** in metric arithmetic.

### 8.2 Threshold audit

- Threshold `0.11992187052965164` was selected on validation by maximum F1 among non-trivial thresholds.
- Reproduced exactly from the reconstructed validation sample.
- Checkpoint stores the same threshold.
- Test was not used to choose the threshold.
- Per-source metrics reuse the pooled validation threshold (`metrics.json` protocol), which is defensible but not source-optimal.
- The validation threshold yields test precision `0.5889`, recall `0.8924`, F1 `0.7096`.
- PA threshold `12.0` yields precision `0.7606`, recall `0.4324`, F1 `0.5513`.
- CN threshold `1.0` yields precision `0.9491`, recall `0.2373`, F1 `0.3796`.

The threshold is valid, but **maximum F1 is not necessarily the right objective** for candidate ranking or manual review. A high-recall threshold with 10,383 false positives may be operationally expensive. A precision-at-K or cost-weighted threshold should be selected on validation for the intended use.

### 8.3 Identical test pairs

Verified:

- `test_predictions.csv` and `baseline_test_predictions.csv` have identical `source_index`, `target_index`, `source_company_id`, `target_company_id`, `source`, and `label` ordering.
- GraphSAGE test logits reproduce the saved CSV within `9.6e-7`.
- PA scores recomputed from the train graph match `baseline_test_predictions.csv` exactly (`max diff 0.0`).

**No issue found** in pair alignment.

### 8.4 Why GraphSAGE has higher AUC but lower AP

This is not a contradiction. It reflects different ranking objectives and tie behavior.

- **PA** uses degree product. It has only 2,466 unique scores and many ties. 52.3% of positives have score 0, and 25.6% of negatives have score 0. High-degree positive edges get very large scores; the top of the PA ranking is extremely pure. But many high-degree negative pairs also get large scores, so the global ROC-AUC is near chance.
- **CN** has only 51 unique scores. 76.3% of positives and 98.7% of negatives score 0. At threshold 1, it has very high precision but low recall. Its AP is close to GraphSAGE, but its top-K ranking is much purer.
- **GraphSAGE** produces continuous scores and better global separation, so AUC is higher. But its top-ranked candidates are less pure than PA/CN, so AP is lower.

Precision at K (pooled test, K = number of top-ranked pairs):

| Model | P@100 | P@500 | P@1000 | P@5000 |
|---|---:|---:|---:|---:|
| PA | 1.000 | 0.998 | 0.993 | 0.945 |
| CN | 1.000 | 0.990 | 0.991 | 0.791 |
| GraphSAGE | 0.580 | 0.610 | 0.586 | 0.577 |

Per-source precision at top 1%:

| Source | PA | CN | GraphSAGE |
|---|---:|---:|---:|
| AnonNet | 1.000 | 1.000 | 0.784 |
| IC-SPLC | 1.000 | 1.000 | 0.451 |
| SAMPOQ | 0.969 | 0.938 | 0.563 |

Practical implication:

- If the intended use is **balanced discrimination**, GraphSAGE is the best of the three.
- If the intended use is **top-K candidate generation**, PA/CN are better in this experiment, especially at the very top.
- If the intended use is **probability output for a decision threshold**, none of the models is calibrated; GraphSAGE is especially overconfident.

### 8.5 Test composition effect

The test set is 50/50 positive/negative. This has two consequences:

1. Random AP is `0.5`, not the real-world positive prevalence.
2. Precision at a fixed threshold is inflated relative to a candidate pool with mostly non-edges.

The reported AP and precision are therefore **conditional on the balanced evaluation set**. They should not be presented as production precision.

### 8.6 Metric gaps

The official `metrics.json` does not include:

- Log loss.
- Brier score.
- ECE / reliability.
- Precision@K, recall@K, MRR, NDCG.
- Source-specific thresholds.
- Calibration curves.

For a ranking model, AP is appropriate, but top-K metrics are necessary to understand operational usefulness. For a probability-output model, calibration metrics are necessary.

---

## 9. Source-wise Performance Analysis

### 9.1 Source comparison

| Source | Nodes | Edges | Density | Train-isolated nodes | Test positives with an isolated train endpoint | Unique model feature rows | Best baseline AP | GraphSAGE AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AnonNet | 7,245 | 43,922 | 0.000837 | 437 | 5.5% | 4 | 0.9146 | 0.5895 |
| IC-SPLC | 109,403 | 106,683 | 0.0000089 | 18,178 | 70.3% | 31,187 | 0.6289 | 0.6511 |
| SAMPOQ | 13,372 | 16,085 | 0.0000900 | 1,994 | 61.9% | 1 | 0.6438 | 0.5121 |

The "isolated train endpoint" column is the share of positive test edges where at least one endpoint has degree 0 in the training graph. This is important: for 70.3% of IC-SPLC positives and 61.9% of SAMPOQ positives, message passing has no train neighborhood at one endpoint.

### 9.2 AnonNet

**Graph:** 7,245 nodes, 43,922 edges, average undirected degree 11.34, median 3, one weak component in the full graph. This is the densest and most structurally informative source.

**Features:** 0% embeddings, 79.4% country presence, 92.6% product presence, 0% industry presence. Only 4 unique 136-d feature rows. Actual country and product values are ignored. The graph is dense enough that degree is a strong signal.

**Why PA is much better:** PA AP `0.9146`; CN AP `0.8047`; GraphSAGE AP `0.5895`. PA directly uses undirected train degree, and AnonNet is dense. GraphSAGE uses directed mean aggregation, only two layers, and no degree or categorical features. It cannot represent the actual country/product code values that might explain linkage. PA top-1% precision is 1.000 vs GraphSAGE 0.784.

**Interpretation:** AnonNet is a structural/degree-dominated source in the current feature regime. A GNN could potentially use train-only degree and categorical embeddings, but the current model does not.

### 9.3 IC-SPLC

**Graph:** 109,403 nodes, 106,683 edges, average undirected degree 1.94, median 1, 22,354 components, 18,178 train-isolated nodes. Highly fragmented and sparse.

**Features:** 28.5% embeddings, 23.0% country, 9.7% industry, 1.7% product. 77,891 nodes share the same 8 flags with zero embeddings. The embedding subset is the main source of node-level differentiation.

**Why GraphSAGE is better than PA:** GraphSAGE AP `0.6511` vs PA `0.6289`; AUC `0.7353` vs PA `0.3979`. The embeddings likely carry useful supplier/product information for the 28.5% covered nodes. PA is near random in AUC because degree product is a poor global ranker in this sparse graph.

**Caveat:** 70.3% of IC-SPLC test positives have at least one train-isolated endpoint. For those pairs, graph propagation cannot use a neighborhood at that node; the model is relying on embeddings/flags. PA's AP is surprisingly high despite AUC below 0.5 because its top-ranked pairs are pure, not because its global ranking is good.

**Interpretation:** IC-SPLC is the strongest case for GraphSAGE, but the advantage is concentrated in the embedding-covered subset. Coverage expansion or better features is the likely improvement path.

### 9.4 SAMPOQ

**Graph:** 13,372 nodes, 16,085 edges, average undirected degree 2.33, median 1, 843 components, 1,994 train-isolated nodes.

**Features:** all 13,372 nodes have the exact same 136-dimensional input. No country, no industry, no product, no embeddings, no anonymization flag. Only the source flag is present, and it is constant within the source.

**Why GraphSAGE is near random:** AUC `0.5320`, AP `0.5121`. With identical inputs and only two mean-aggregation layers on a sparse graph, the model has almost no node-level signal. It can only learn degree/neighborhood summaries, and 61.9% of positive test pairs have a train-isolated endpoint.

**Why PA AP is high while AUC is low:** PA AP `0.6438`, AUC `0.4441`. PA has 197 unique scores; positive degree products concentrate at the top for some high-degree pairs, but negative high-degree pairs also score high. Top-1% precision is 0.969, but global ranking is below chance.

**Interpretation:** SAMPOQ should not be modeled with the current features. Either add real attributes (country/industry/product/embedding coverage), use a source-specific structural model, or report that SAMPOQ link prediction is not supported.

### 9.5 Is one combined model suitable?

Evidence says **not in the current form**:

- The three sources have different graph density, feature coverage, and direction semantics.
- GraphSAGE wins on IC-SPLC, PA wins on AnonNet, and SAMPOQ is near random.
- A single shared 136-d input produces one constant vector for all SAMPOQ nodes and four patterns for all AnonNet nodes.
- Pooled metrics hide these failures.

A pooled model can still be a valid experiment if it uses source-specific features and per-source evaluation. But the current pooled model should not be presented as a single model that works across all sources. Source-specific models or source-conditioned feature branches are worth testing.

---

## 10. Calibration Assessment

### 10.1 Current state

- No calibration code exists in the repository.
- No Brier score, log loss, ECE, or reliability diagram is reported in `metrics.json`.
- Validation predictions are not persisted.
- The checkpoint stores only model weights and the threshold, not calibration parameters.
- Temperature scaling / isotonic regression is not implemented.

Calibration is **appropriate and necessary** because the model outputs scores used as probabilities. The current scores are not probabilities in any practical sense.

### 10.2 Recommended diagnostics

Use validation data only for fitting calibration parameters:

- **Reliability diagram** with equal-width or equal-frequency bins.
- **Expected Calibration Error (ECE)** and maximum calibration error.
- **Brier score** and **log loss** on validation and test.
- **Confidence histogram**, split by positive/negative and source.
- **Positive vs negative score distributions**, not just histograms over all pairs.

### 10.3 Temperature scaling

Temperature scaling is a reasonable first calibration method:

- Fit a single temperature `T > 0` on validation logits/labels by minimizing BCE.
- Apply `sigmoid(logit / T)` to test.
- It preserves ranking: AUC and AP should be unchanged.
- It does not fix ranking weaknesses; it only changes probability scale.
- It must not be fit on test labels.

Do not claim that calibration improves AP or precision@K. It may improve log loss, Brier, and ECE. It may change threshold-based precision/recall if the threshold is re-selected after calibration, but ranking remains the same.

### 10.4 Calibration by source

Per-source test diagnostics (derived, not official):

| Source | Brier | Log loss | ECE |
|---|---:|---:|---:|
| AnonNet | 0.4137 | 8.9188 | 0.4079 |
| IC-SPLC | 0.2851 | 4.5205 | 0.2747 |
| SAMPOQ | 0.4542 | 9.4071 | 0.4511 |

A single global temperature may not be enough because source miscalibration differs. Source-specific temperatures would need validation data per source and should be reported separately. However, source-specific calibration must not be used to hide ranking failures.

---

## 11. Confirmed Bugs

This section separates **confirmed code/artifact defects** from **protocol limitations**. The audit found no confirmed loss/logit bug.

### 11.1 Confirmed artifact defect: stale experiment status

**Files:**

- `ml/experiments/graphsage_link_prediction/config.json`
- `ml/experiments/graphsage_link_prediction/baseline_metrics.json`

**Evidence:**

- `config.json` has `"status": "planned"`.
- `baseline_metrics.json` has:
  - `comparison.graphsage_link_prediction.status = "not_trained"`
  - reason: `"No trained GraphSAGE checkpoint exists in this experiment directory."`
- `metrics.json` and `experiment.json` correctly say `"status": "completed"` and `"gnn_results_available": true`.

**Impact:** Dashboard/consumers that read `config.json` or the baseline comparison may incorrectly report the run as unfinished. This is a metadata consistency bug, not a model-performance bug.

**Correction:** Update `config.json.status` after successful training, or derive status from `experiment.json`. Remove/replace the stale `comparison` block in `baseline_metrics.json`, or regenerate the baseline after training.

### 11.2 Confirmed implementation gap: feature contract mismatch

**Files/functions:**

- `ml/preprocessing/feature_engineering.py:204-225`
- `ml/preprocessing/graph_preprocessing.py:49-68`
- `ml/training/train_graphsage_link_prediction.py:72-81`, `308-311`
- `ml/models/gcn_model.py:79-89`

**Evidence:**

- `feature_engineering.py` builds 146 node features, including `x_cat` and `x_emb`.
- `graph_preprocessing.py` stores `x_cat`, `x_emb`, `emb_mask`, `edge_attr`.
- `train_graphsage_link_prediction.py` selects only `data.x` columns and passes only `x` and `edge_index` to `build_gnn`.
- `build_gnn`/`AtmoGraphSAGE` never accepts `x_cat`, `x_emb`, `emb_mask`, or `edge_attr`.
- Actual categorical codes are present with cardinalities `[282, 23, 617]` but unused.
- 12 edge features are built but unused.

**Impact:** The model is starved of node attributes and edge information. This is the main explanation for SAMPOQ near-random performance and AnonNet underperformance.

**Correction:** Either:
- add categorical embeddings and edge-feature support to the model, or
- explicitly document that this is a flags-plus-IC-SPLC-embeddings baseline and stop claiming it uses the full feature set.

### 11.3 Confirmed feature-collapse defect

**Files/data:**

- `datasets/processed/graph/node_features.csv`
- `datasets/training/node_embeddings.parquet`
- `ml/training/train_graphsage_link_prediction.py:72-81`

**Evidence:**

- SAMPOQ: 13,372 nodes, 1 unique 136-d feature row.
- AnonNet: 7,245 nodes, 4 unique 136-d feature rows.
- IC-SPLC: 109,403 nodes, 31,187 unique rows, but 77,891 nodes share the same 8 flags with zero embeddings.

**Impact:** A node-attribute GNN cannot distinguish most SAMPOQ/AnonNet nodes. Performance is then determined almost entirely by graph structure and the decoder.

**Correction:** Add categorical embeddings/one-hot codes and train-only structural features; verify unique-row counts per source before training.

### 11.4 Confirmed protocol limitation: reverse-edge cross-split leakage

**Files/functions:**

- `ml/training/evaluate_link_prediction_baseline.py:91-99` (undirected adjacency)
- `ml/training/evaluate_link_prediction_baseline.py:216-228` (CN/PA)
- `ml/training/evaluate_link_prediction_baseline.py:603-618` (protocol notes)

**Evidence:**

- 566 test positive edges have their reverse in train.
- AnonNet: 439 / 4,400 = 9.98%.
- SAMPOQ: 79 / 1,610 = 4.91%.
- IC-SPLC: 48 / 10,659 = 0.45%.
- PA/CN treat train edges as undirected, so a reverse train edge supports a test positive.

**Impact:** Random-split baseline results are slightly optimistic, especially AnonNet. GraphSAGE is less directly affected because its message passing is directed, but the evaluation comparison is not clean.

**Correction:** Exclude reverse-edge components from split leakage, use an edge-disjoint split, or use a temporal split. Report a sensitivity analysis. Do not present the current random-split numbers as future-link performance.

### 11.5 Confirmed evaluation gap: no calibration or probability metrics

**Files:**

- `ml/training/train_graphsage_link_prediction.py:215-256`
- `ml/experiments/graphsage_link_prediction/metrics.json`

**Evidence:**

- `metrics.json` contains AUC, AP, precision, recall, F1, confusion matrix, curves, and score distribution.
- It does not contain log loss, Brier, ECE, or reliability data.
- No calibration parameters or validation predictions are saved.

**Impact:** The reported metrics cannot support probability-based decisions. The model is severely miscalibrated, but the official report does not show it.

**Correction:** Add Brier, log loss, ECE, reliability, and confidence histograms. Save validation scores/logits. Fit calibration on validation only, evaluate on test, and keep ranking and calibration objectives separate.

### 11.6 Confirmed design cause: unbounded decoder / overconfidence

**File/function:**

- `ml/models/gcn_model.py:59-60`
- `ml/training/train_graphsage_link_prediction.py:148-182`, `185-212`

**Evidence:**

- Decoder is a raw dot product with no normalization or bias.
- Final embedding norms are approximately 5.0.
- Test logits reach ±26.
- 4,444 negative test pairs are exactly `1.0`.
- ECE is `0.3253`; top-bin confidence `0.9932` vs empirical positive rate `0.6072`.

**Impact:** Scores cannot be interpreted as probabilities. Threshold-based decisions are fragile.

**Correction:** Consider normalized/cosine decoding, a learned logit scale, decoder bias, gradient clipping, stronger weight decay, and validation-only temperature scaling. Do not rely on calibration to fix ranking.

### 11.7 Confirmed inference-contract risk

**File:** `ml/inference/predict.py:17-27`

**Evidence:** The default `PredictionType` is `"disruption_risk"`, the field is `risk_score`, and `risk_level` maps scores to LOW/MEDIUM/HIGH. The GraphSAGE checkpoint is a link-prediction model.

**Impact:** If link scores are routed through this contract, they will be mislabeled as company disruption risk. That violates the project's stated constraint that link prediction is not disruption-risk prediction.

**Correction:** Add a distinct link-prediction record type or require `prediction_type="link_prediction"` explicitly. Do not use `risk_level` for link scores without a validated risk-label mapping.

---

## 12. Potential Issues Requiring Further Investigation

### 12.1 Evaluation protocol

- **Temporal split is missing.** Edges span 2004-2026. A time-based split would test future-link prediction and would be more defensible for applied use.
- **Undirected baselines vs directed model.** PA/CN use undirected train edges; GraphSAGE uses directed message passing. This is not a clean comparison. A directed PA/CN variant or a directed baseline should be reported.
- **Reverse-edge leakage.** 566 test positives have a train reverse. The sensitivity check reduces PA AP by about 0.015 and CN/GraphSAGE AP by about 0.013. Fix the split or report the limitation prominently.
- **50/50 test composition.** AP and precision are not realistic production estimates. Add a candidate-pool evaluation with realistic prevalence.
- **Per-source threshold.** The pooled validation threshold is reused for every source. Per-source thresholds selected on validation would be more appropriate if source-specific deployment is intended.
- **No precision@K in official metrics.** Add top-K metrics for ranking use.

### 12.2 Model and training

- **No train-only structural features.** Degree, in/out-degree, component size, and temporal features were excluded because they were full-graph-derived. Recompute them from the train graph only.
- **Categorical values ignored.** Country, industry, and product codes are available and could be embedded safely.
- **Edge features ignored.** Relationship type, confidence, edge count, and years are available.
- **Source-specific models not tested.** The evidence suggests different sources need different treatment.
- **No gradient clipping.** Large logits and no clipping may contribute to saturation.
- **Checkpoint selection only on AP.** Calibration and validation loss are not part of model selection.
- **No calibration fit.** Temperature scaling or isotonic regression should be tested on validation.
- **No optimizer/RNG state in checkpoint.** Runs cannot be resumed exactly.
- **Early stopping on AP with patience 20.** Best AP at epoch 164; final epoch 184. This is valid, but the model continues to become more confident after AP peaks.

### 12.3 Reproducibility and artifact gaps

- No split-generation code in the repository. Only `dataset_splits.json` and the data-dictionary description are available.
- No validation predictions saved.
- No calibration artifact.
- No automated tests.
- `config.json` and `baseline_metrics.json` contain stale status metadata.
- `ml/artifacts/baseline_metrics.json` is a shipment-delay baseline, not a structural link-prediction baseline; do not confuse it with `ml/experiments/graphsage_link_prediction/baseline_metrics.json`.
- No browser automation was available for dashboard verification. This audit did not modify dashboard code.

### 12.4 Data and semantics

- AnonNet direction is not formally documented. The meaning of A→B vs B→A is uncertain.
- IC-SPLC and AnonNet entities are anonymized; no official code-to-name mapping is available.
- No cross-source entity resolution; the graph remains three disconnected source networks.
- No verified company-level risk labels.
- The IC-SPLC embeddings are source-provided and only cover 28.5% of IC-SPLC nodes.
- The graph has 23,198 weakly connected components; many nodes have no train neighborhood.

---

## 13. Prioritized Improvement Plan

The plan is ordered by expected scientific value and dependency. Do not tune on test. Use validation for all selection and calibration.

| Priority | Experiment | Hypothesis | Required changes | Success metric | Risk |
|---:|---|---|---|---|---|
| 1 | Protocol control: reverse-edge / edge-disjoint evaluation | Current conclusions may be slightly inflated by reverse-edge crossovers and undirected baselines | Add reverse-edge-aware split or exclude reverse components; add directed PA/CN; report sensitivity | AP/AUC/precision@K on corrected split; report deltas vs current | Fewer positives; results may not be directly comparable; split redesign needed |
| 2 | Feature-contract ablation | The 8-flag/128-embedding input is the main bottleneck, especially for SAMPOQ/AnonNet | Compare current 136-d input vs 8 flags vs +train-only structural vs +categorical vs +both | Per-source AP, ROC-AUC, precision@K; validation-selected settings | Leakage if structural features not train-only; overfitting on small sources |
| 3 | Add train-only structural features | Degree and component signals are strong, especially for AnonNet | Compute degree/in-out/component features from train graph only; add to node input | AnonNet/SAMPOQ AP and precision@K vs PA/CN | Full-graph leaks if computed incorrectly; distribution shift |
| 4 | Add categorical embeddings | Country/industry/product values can distinguish nodes where flags cannot | Learn embeddings for `x_cat` (282/23/617); handle missing codes; do not fabricate mappings | Unique-row coverage; AnonNet/SAMPOQ AP; calibration | Small categories overfit; anonymized codes have unknown semantics |
| 5 | Add edge features | Relationship type, confidence, and year may explain links | Use `edge_dim` in SAGEConv or a GINE/GATv2 edge-aware layer | AP and calibration, especially IC-SPLC/SAMPOQ | Temporal leakage if year is used; sparse edge feature coverage |
| 6 | Source-specific vs pooled models | One shared model may be inappropriate given source heterogeneity | Train per-source models; compare to pooled with source flags | Per-source AP/precision@K; pooled macro average | Small source data; more artifacts; harder deployment |
| 7 | Decoder alternatives | Unbounded dot product causes saturation | Compare normalized dot/cosine, learned scale, MLP decoder, decoder bias | AP + Brier/ECE/log loss; no ranking regression | Normalization may reduce expressiveness; calibration not ranking |
| 8 | Regularization/optimization | Overconfidence may be reduced with better optimization | Weight decay, gradient clipping, logit scale, early stopping on AP+loss, LR schedule | Validation AP, Brier, ECE; test only once | Underfitting; hyperparameter search cost |
| 9 | Ranking metrics and threshold selection | Max-F1 threshold is not aligned with candidate ranking | Add precision@K, recall@K, MRR/NDCG; select threshold by validation cost/precision target | Top-K precision/recall; false-positive budget | Requires a declared operational objective |
| 10 | Calibration | Probabilities are unusable | Fit temperature scaling or isotonic on validation only; save calibration artifact | Brier, log loss, ECE, reliability; ranking unchanged | Source-specific miscalibration; calibration cannot fix ranking |
| 11 | Reproducibility hardening | Current checkpoint/results are hard to resume and audit | Save optimizer/RNG, validation predictions, calibration params, and config status | Re-run reproducibility; artifact consistency tests | Engineering time; storage |
| 12 | Regression tests | Silent feature/metric regressions are possible | Add tests for sampler, label alignment, threshold, metrics, feature dimensions, checkpoint load | CI passes; invariants enforced | Test maintenance |

### Recommended immediate order

1. Fix the evaluation protocol and report the sensitivity.
2. Fix the feature contract (categorical + train-only structural features).
3. Re-run source-wise, with AP and precision@K.
4. Add calibration and probability metrics.
5. Only then compare decoder/aggregation variants.

Do not start with architecture tuning. The current model is not receiving the information needed to make the task learnable for AnonNet/SAMPOQ.

---

## 14. Recommended Next Experiments

### 14.1 Single most valuable next experiment

**Source-wise feature-contract ablation with train-only structural features and categorical embeddings.**

Why this first:

- The audit proves the current model input is nearly constant within SAMPOQ and has only four patterns within AnonNet.
- The model ignores 282 country codes, 23 industry codes, 617 product codes, and all 12 edge features.
- The model ignores structural degree, which is the strongest signal for AnonNet.
- The current results cannot distinguish "GraphSAGE is weak" from "the GNN was given almost no features."

Experiment arms:

1. Current 136-d input (8 flags + IC-SPLC embeddings).
2. 8 flags only (no embeddings).
3. Current input + train-only degree/in-degree/out-degree/component features.
4. Current input + categorical embeddings for country/industry/product.
5. Current input + both structural and categorical features.
6. Optional: add edge features with an edge-aware layer.

Evaluation:

- Per-source validation and test AP, ROC-AUC, and precision@K.
- Pooled metrics only as secondary.
- Calibration diagnostics for any model intended to output probabilities.
- Fixed seed and fixed split for comparability.
- No test-set tuning.

Success criteria:

- At least one source-specific model beats the best structural baseline on validation AP and precision@K, or the experiment clearly shows that the source lacks learnable signal under the available features.
- SAMPOQ is no longer modeled by a single constant feature vector.
- AnonNet gains measurable signal from categorical/structural features.

Risk:

- Train-only structural features must be computed without validation/test edges.
- Categorical codes may have weak or noisy semantics, especially anonymized IC-SPLC codes.
- Small sources may overfit.
- If the protocol still has reverse-edge leakage, improvements will be slightly optimistic; run the protocol-control evaluation in parallel.

### 14.2 Second experiment

**Protocol-corrected re-evaluation of the existing checkpoint and baselines** with reverse-edge exposure removed or with an edge-disjoint/temporal split. This is cheap and determines whether the current conclusions survive a cleaner protocol.

### 14.3 Third experiment

**Calibration-only evaluation** on the existing checkpoint: fit temperature scaling on validation, report test Brier/log loss/ECE, confirm AUC/AP unchanged. This is a low-risk way to make the probability output usable while feature work proceeds.

---

## 15. Reproducibility and Limitations

### 15.1 Reproducibility

Verified:

- Dataset files match recorded SHA-256 hashes.
- Split artifact lists match `edge_index.csv`.
- GraphSAGE test pairs match baseline test pairs exactly.
- GraphSAGE test logits reproduce `test_predictions.csv` within `9.6e-7`.
- PA scores reproduce `baseline_test_predictions.csv` exactly.
- Validation threshold reproduces exactly from the checkpoint and documented sampler.
- Seed is 42.
- Environment versions are recorded above.

The training run is not fully resumable because the checkpoint lacks optimizer, scheduler, and RNG state. It is reproducible only by rerunning the full training script with the same seed and data.

### 15.2 Audit limitations

- No retraining was performed.
- No model architecture, data, split, checkpoint, or dashboard code was modified.
- Validation predictions were not persisted; they were reconstructed read-only for diagnostics. The threshold reproduced exactly, but this is not a substitute for a saved validation artifact.
- The split-generation code was not found, so the exact random split procedure could not be inspected. Only the data-dictionary description (random 80/10/10) and the artifact were available.
- No automated tests exist; diagnostics were ad hoc read-only scripts.
- No browser automation was available for dashboard verification.
- Log-loss values depend on clipping for exact `1.0` predictions. The audit reports the sklearn float64 value and notes the sensitivity.
- Calibration was not fitted or implemented; only diagnostics were computed.
- No verified risk labels exist. No disruption-risk, delay, or propagation claim can be made.
- No cross-source entity resolution exists. The three source graphs are disconnected.
- AnonNet direction semantics are undocumented.
- The test set is balanced 50/50 and random; it is not a temporal or production candidate-pool evaluation.

### 15.3 Unknowns

- Why the untrained model starts with mean validation logit `26.14` is not fully isolated. Initialization and LayerNorm/SAGEConv interactions are plausible contributors; this audit confirms the effect but does not reduce it to a single line.
- The exact reason PA AP is high while PA AUC is below 0.5 is tie/discrete-score behavior; the audit demonstrates the score distributions and top-K purity, but does not derive a closed-form explanation.
- The quality and semantics of the IC-SPLC 128-d embeddings are unknown beyond coverage and unit norm.
- The exact AnonNet direction semantics are unknown.
- Whether source-specific models will beat a pooled model with better features is unknown; it is a hypothesis for the next experiment.

---

## Appendix A. Key File and Line References

| Topic | File | Lines |
|---|---|---:|
| GraphSAGE encoder/decoder | `ml/models/gcn_model.py` | 35-63 |
| Dot-product decoder | `ml/models/gcn_model.py` | 59-60 |
| GraphSAGE definition | `ml/models/gcn_model.py` | 79-89 |
| Feature selection | `ml/training/train_graphsage_link_prediction.py` | 72-81 |
| Pair sampling wrapper | `ml/training/train_graphsage_link_prediction.py` | 99-127 |
| Decode pairs | `ml/training/train_graphsage_link_prediction.py` | 130-145 |
| BCEWithLogits | `ml/training/train_graphsage_link_prediction.py` | 179 |
| Evaluation sigmoid | `ml/training/train_graphsage_link_prediction.py` | 202, 484-485 |
| Message-passing mask | `ml/training/train_graphsage_link_prediction.py` | 308-311 |
| Validation/test sampling | `ml/training/train_graphsage_link_prediction.py` | 323-357 |
| Threshold selection | `ml/training/train_graphsage_link_prediction.py` | 223-227 |
| Checkpoint save | `ml/training/train_graphsage_link_prediction.py` | 596-610 |
| Node feature construction | `ml/preprocessing/feature_engineering.py` | 204-225 |
| Categorical indices | `ml/preprocessing/feature_engineering.py` | 219-221 |
| Edge feature construction | `ml/preprocessing/feature_engineering.py` | 247-336 |
| PyG data construction | `ml/preprocessing/graph_preprocessing.py` | 49-68 |
| Undirected adjacency | `ml/training/evaluate_link_prediction_baseline.py` | 91-99 |
| Negative sampler | `ml/training/evaluate_link_prediction_baseline.py` | 150-213 |
| PA/CN scoring | `ml/training/evaluate_link_prediction_baseline.py` | 216-228 |
| Validation threshold | `ml/training/evaluate_link_prediction_baseline.py` | 231-251 |
| Binary metrics | `ml/training/evaluate_link_prediction_baseline.py` | 254-279 |
| Per-source metrics | `ml/training/evaluate_link_prediction_baseline.py` | 282-314 |
| Baseline protocol notes | `ml/training/evaluate_link_prediction_baseline.py` | 603-618 |
| Baseline prediction output | `ml/training/evaluate_link_prediction_baseline.py` | 656-686 |
| Random edge split description | `datasets/metadata/data_dictionary.md` | 102-113 |
| Split artifact description | `datasets/metadata/data_dictionary.md` | 156-172 |
| Risk inference contract | `ml/inference/predict.py` | 17-27 |

---

## Appendix B. Audit Result Classification Summary

| Finding | Classification |
|---|---|
| Raw logits passed to BCEWithLogitsLoss | No issue found |
| Double sigmoid | No issue found |
| Label shape/dtype/meaning | No issue found |
| Threshold selected on validation only | No issue found |
| Identical GraphSAGE/baseline test pairs | No issue found |
| Message passing uses train edges only | No issue found |
| Negative samples exclude observed directed edges | No issue found |
| Cross-source negatives | No issue found |
| Self-loops / duplicate directed pairs | No issue found |
| Exact split overlap | No issue found |
| Categorical/edge features built but unused | Confirmed implementation gap |
| SAMPOQ single unique feature row | Confirmed feature-collapse defect |
| AnonNet four unique feature rows | Confirmed feature-collapse defect |
| Unbounded dot-product overconfidence | Confirmed design cause |
| No calibration / probability metrics | Confirmed evaluation gap |
| Reverse-edge cross-split leakage | Confirmed protocol limitation |
| Random split, no temporal holdout | Confirmed protocol limitation |
| Stale config/status metadata | Confirmed artifact defect |
| No validation predictions saved | Reproducibility gap |
| No optimizer/RNG in checkpoint | Reproducibility gap |
| No split-generation code | Unknown / reproducibility gap |
| Inference contract defaults to risk | Confirmed contract risk |
| Undocumented AnonNet direction | Confirmed data limitation |
