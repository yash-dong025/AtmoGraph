# AtmoGraph GraphSAGE Phase 2 Experiment Report

Generated from `ml/experiments/graphsage_phase2/summary.json`.

- Status: completed
- Seed: 42
- Device: CPU
- Protocol set: A, B, C, D
- Feature configurations: c1, c2, c3, c4, c5
- Original experiment directory: `ml/experiments/graphsage_link_prediction/` (read-only, not overwritten)

## 1. Executive summary

Phase 2 completed 20 controlled GNN runs (four protocols by five configurations), four directed/undirected structural baselines per protocol, an original-checkpoint reference, and an original-artifact comparison. The main result is that **the simple preferential-attachment baselines remain stronger than every GNN configuration on pooled average precision under protocols A, B, and C**. Under the IC-SPLC temporal protocol D, the GNNs are stronger than the structural baselines, but this comparison is limited to one source and one chronological split.

The original 136-dimensional feature contract was confirmed: the model input is **8 static flags plus 128 IC-SPLC embedding dimensions**, not the continuous/full-graph features that appear first in the current feature builder. The original checkpoint can only be reproduced by reordering columns to its stored `feature_names`.

Key findings:

1. **Reverse leakage is real but did not explain the original result.** Protocol A has reverse-of-validation/test edges in training for 9.98% of AnonNet test positives, 4.91% of SAMPOQ test positives, and 0.45% of IC-SPLC test positives. Protocols B and C remove this leakage and produce results close to A. For c1, pooled test AP is 0.576 (A), 0.573 (B), and 0.577 (C). This does not support reverse leakage as the main driver of GNN performance, but B/C remain the cleaner protocols for final claims.
2. **The 128-dimensional embeddings are not consistently beneficial.** c2 uses only the 8 static flags. Its pooled AP is 0.597 in A versus 0.576 for c1, but it is lower than c1 in B, C, and D. The embeddings improve some rankings but are not a robust source of predictive value.
3. **Categorical embeddings improve ranking AUC** in A/B/C: c4 reaches pooled AUC 0.685, 0.678, and 0.684 versus 0.640, 0.633, and 0.638 for c1. Pooled AP improves modestly, from 0.576 to 0.610 in A. In D, c4 is worse than c1 (AUC 0.641 versus 0.689).
4. **Train-only structural features raise AP and top-K precision but lower AUC and calibration quality.** c3/c5 reach pooled AP 0.650-0.659 in A, but AUC is only 0.592-0.596 and ECE rises to 0.129-0.166. These features are transductive and likely encode degree/popularity shortcuts. They also hurt SAMPOQ badly.
5. **AnonNet is easy for structural baselines, SAMPOQ is close to random for the GNNs.** Directed PA reaches AnonNet AP 0.905, 0.908, and 0.904 in A/B/C. SAMPOQ GNN AP stays near 0.50, while its c3/c5 AUC falls to 0.36-0.42. SAMPOQ has only one unique 136-dimensional feature row, so the current features cannot distinguish its nodes.
6. **The original reported metric was not reproduced exactly.** The original artifact reports test AP 0.6094 and AUC 0.6727. Re-scoring the original checkpoint on Phase 2 Protocol A pairs gives AP 0.5817 and AUC 0.6643 because the negative-sample realization differs. c1 retraining gives AP 0.5761 and AUC 0.6399; c4 gives AP 0.6101 and AUC 0.6850. These differences are single-seed and are not evidence of a statistically reliable improvement.
7. **Calibration remains imperfect.** Temperature scaling reduces the original artifact's severe overconfidence (ECE 0.325) to ECE 0.069-0.074 for c1 in A/B/C and 0.071 for the checkpoint reference. Structural configurations remain less calibrated: c5 ECE is 0.135-0.166 across A/B/C/D and c3 is 0.064-0.133.

## 2. Data, splits, and protocols

Data size: 130,020 nodes and 166,690 edges. Edge sources: IC-SPLC 106,683; AnonNet 43,922; SAMPOQ 16,085. Node sources: IC-SPLC 109,403; SAMPOQ 13,372; AnonNet 7,245.

The original random 80/10/10 edge split was reproduced exactly from `scripts/build_graph_dataset.py:335-346` with seed 42 on `edge_index.csv` ordering: match rate 1.000000, mismatch count 0.

| Protocol | Train | Validation | Test | Excluded | Source scope | Reverse control | Test positives with reverse in train |
|---|---:|---:|---:|---:|---|---|---:|
| A | 133,352 | 16,669 | 16,669 | 0 | All | Original split, no control | AnonNet 9.98%, SAMPOQ 4.91%, IC-SPLC 0.45% |
| B | 133,352 | 16,668 | 16,670 | 0 | All | Reverse pairs grouped into the same split | 0% |
| C | 132,167 | 16,669 | 16,669 | 1,185 | All | 1,185 training edges removed when their reverse is a val/test positive | 0% |
| D | 75,332 | 16,761 | 14,514 | 60,083 | IC-SPLC only | 76 training edges removed for reverse controls | 0% |

Protocol D uses `first_year <= 2023` for training, 2024 for validation, and 2025 for test. It is IC-SPLC-only because AnonNet has no edge years and 92% of SAMPOQ edges are from 2026, which makes a pooled temporal split non-comparable.

Evaluation pairs are balanced: each validation/test positive is paired with one source-matched sampled negative. This is useful for controlled ranking comparisons but does not estimate production precision at real prevalence. Precision@100 and Precision@500 use expected precision with random tie-breaking inside the boundary score group.

Direction handling: IC-SPLC is documented supplier -> customer. AnonNet is "as provided (source -> target), likely Supplier -> Customer" but the direction is not formally verified. Reversing AnonNet edges is therefore not treated as a harmless duplicate operation.

## 3. Feature contract audit

The current model input has 136 dimensions:

- 8 static flags: `source_sampoq`, `source_ic_splc`, `source_anonnet`, `has_country`, `has_industry`, `has_product_category`, `embedding_available`, `is_anonymised`.
- 128 IC-SPLC embedding dimensions: `icsplc_emb_0` through `icsplc_emb_127`.

The current `build_node_features()` output has 146 dimensions because it starts with scaled continuous features and then appends the flags and embeddings. The original checkpoint does not contain those continuous features. The original checkpoint reference in Phase 2 reorders columns using the checkpoint's stored `feature_names`; Phase 2 configs c1 and c3 explicitly select flags + embeddings by name.

Categorical fields were unused by the original model:

| Field | Cardinality including missing | Missing rate | Phase 2 use |
|---|---:|---:|---|
| `country_code` | 282 | 76.25% | Source-specific embeddings in c4/c5 |
| `industry_code` | 23 | 91.87% | Source-specific embeddings in c4/c5 |
| `product_category_code` | 617 | 93.42% | Source-specific embeddings in c4/c5 |

Missing categories map to index 0. Anonymized codes were not reverse-mapped to names. Source-specific embedding tables are used because the same integer code does not have the same meaning across the three source datasets.

Source feature collapse:

| Source | Nodes | Unique 136-d rows | Embedding coverage |
|---|---:|---:|---:|
| SAMPOQ | 13,372 | 1 | 0.0% |
| IC-SPLC | 109,403 | 31,187 | 28.5% |
| AnonNet | 7,245 | 4 | 0.0% |

All 12 edge features are audited but unused in configs 1-5. They are unavailable for an unseen candidate edge, so using them as scoring features would leak or be operationally invalid. Config 6, the edge-aware architecture, was deliberately not implemented.

Full-graph structural features are excluded from every config. c3/c5 use structural features recomputed only from each protocol's training graph, but those features remain transductive and are not valid for entirely unseen nodes.

## 4. Results

### 4.1 Pooled test results

#### Protocol A: original random split

| Model | Type | AUC | AP | P@100 | P@500 | F1 |
|---|---|---:|---:|---:|---:|---:|
| directed_preferential_attachment | baseline | 0.636 | 0.697 | 1.000 | 0.998 | 0.549 |
| undirected_preferential_attachment | baseline | 0.522 | 0.697 | 1.000 | 0.994 | 0.550 |
| undirected_common_neighbors | baseline | 0.612 | 0.612 | 0.998 | 0.995 | 0.379 |
| directed_common_neighbors | baseline | 0.572 | 0.572 | 0.990 | 0.998 | 0.256 |
| c5_current_struct_cat | GNN | 0.596 | 0.659 | 0.896 | 0.892 | 0.667 |
| c3_current_struct | GNN | 0.592 | 0.650 | 0.793 | 0.871 | 0.667 |
| c4_current_cat | GNN | 0.685 | 0.610 | 0.400 | 0.494 | 0.708 |
| c2_flags8 | GNN | 0.663 | 0.597 | 0.613 | 0.692 | 0.715 |
| c1_current_136 | GNN | 0.640 | 0.576 | 0.365 | 0.427 | 0.706 |
| original_graphsage_reference | checkpoint reference | 0.664 | 0.582 | 0.069 | 0.124 | 0.709 |
| original_graphsage_reported | original artifact | 0.673 | 0.609 | 0.600 | 0.600 | 0.710 |

#### Protocol B: reverse-aware grouped split

| Model | Type | AUC | AP | P@100 | P@500 | F1 |
|---|---|---:|---:|---:|---:|---:|
| directed_preferential_attachment | baseline | 0.636 | 0.699 | 1.000 | 0.998 | 0.550 |
| undirected_preferential_attachment | baseline | 0.517 | 0.698 | 1.000 | 1.000 | 0.551 |
| undirected_common_neighbors | baseline | 0.614 | 0.614 | 1.000 | 0.996 | 0.382 |
| directed_common_neighbors | baseline | 0.575 | 0.575 | 1.000 | 0.997 | 0.264 |
| c5_current_struct_cat | GNN | 0.592 | 0.656 | 0.913 | 0.910 | 0.667 |
| c3_current_struct | GNN | 0.585 | 0.637 | 0.469 | 0.821 | 0.670 |
| c4_current_cat | GNN | 0.678 | 0.604 | 0.290 | 0.414 | 0.704 |
| c1_current_136 | GNN | 0.633 | 0.573 | 0.588 | 0.440 | 0.704 |
| c2_flags8 | GNN | 0.617 | 0.562 | 0.415 | 0.416 | 0.697 |

#### Protocol C: original split with reverse-controlled training

| Model | Type | AUC | AP | P@100 | P@500 | F1 |
|---|---|---:|---:|---:|---:|---:|
| directed_preferential_attachment | baseline | 0.637 | 0.697 | 1.000 | 0.998 | 0.550 |
| undirected_preferential_attachment | baseline | 0.519 | 0.694 | 1.000 | 0.994 | 0.546 |
| undirected_common_neighbors | baseline | 0.607 | 0.607 | 0.990 | 0.994 | 0.366 |
| directed_common_neighbors | baseline | 0.569 | 0.568 | 0.990 | 0.998 | 0.244 |
| c5_current_struct_cat | GNN | 0.590 | 0.656 | 0.897 | 0.892 | 0.667 |
| c3_current_struct | GNN | 0.587 | 0.647 | 0.775 | 0.868 | 0.667 |
| c4_current_cat | GNN | 0.684 | 0.609 | 0.420 | 0.435 | 0.707 |
| c1_current_136 | GNN | 0.638 | 0.577 | 0.445 | 0.448 | 0.705 |
| c2_flags8 | GNN | 0.621 | 0.568 | 0.373 | 0.429 | 0.700 |

#### Protocol D: IC-SPLC temporal split

| Model | Type | AUC | AP | P@100 | P@500 | F1 |
|---|---|---:|---:|---:|---:|---:|
| c3_current_struct | GNN | 0.589 | 0.651 | 0.788 | 0.819 | 0.667 |
| c5_current_struct_cat | GNN | 0.578 | 0.650 | 0.870 | 0.836 | 0.667 |
| c1_current_136 | GNN | 0.689 | 0.611 | 0.077 | 0.287 | 0.747 |
| c4_current_cat | GNN | 0.641 | 0.588 | 0.644 | 0.332 | 0.713 |
| c2_flags8 | GNN | 0.623 | 0.574 | 0.013 | 0.242 | 0.737 |
| undirected_preferential_attachment | baseline | 0.348 | 0.559 | 1.000 | 0.998 | 0.214 |
| directed_preferential_attachment | baseline | 0.500 | 0.555 | 1.000 | 0.998 | 0.196 |
| undirected_common_neighbors | baseline | 0.525 | 0.525 | 1.000 | 0.991 | 0.097 |
| directed_common_neighbors | baseline | 0.512 | 0.512 | 0.996 | 0.860 | 0.049 |

### 4.2 Per-source average precision

Selected models only; full per-source results are in `ml/experiments/graphsage_phase2/reports/comparison_by_source.csv`.

| Protocol | Source | c1 | c4 | c5 | Directed PA |
|---|---|---:|---:|---:|---:|
| A | AnonNet | 0.575 | 0.767 | 0.830 | 0.905 |
| A | IC-SPLC | 0.606 | 0.607 | 0.629 | 0.627 |
| A | SAMPOQ | 0.502 | 0.509 | 0.467 | 0.641 |
| B | AnonNet | 0.575 | 0.760 | 0.833 | 0.908 |
| B | IC-SPLC | 0.604 | 0.605 | 0.630 | 0.630 |
| B | SAMPOQ | 0.496 | 0.496 | 0.453 | 0.627 |
| C | AnonNet | 0.579 | 0.760 | 0.824 | 0.904 |
| C | IC-SPLC | 0.605 | 0.608 | 0.629 | 0.627 |
| C | SAMPOQ | 0.494 | 0.495 | 0.457 | 0.640 |
| D | IC-SPLC | 0.611 | 0.588 | 0.650 | 0.555 |

The source split is not cosmetic. AnonNet dominates the structural-baseline result, IC-SPLC drives the GNN ranking AUC, and SAMPOQ is nearly unlearnable with the current feature contract. Pooled averages mix these regimes and should not be read as a single deployment estimate.

### 4.3 Calibration

Pooled test calibration after validation-only temperature/Platt scaling:

| Protocol | Model | Brier | Log loss | ECE (10 bins) |
|---|---|---:|---:|---:|
| A | original_graphsage_reported | 0.335 | 5.953 | 0.325 |
| A | original_graphsage_reference | 0.235 | 0.663 | 0.071 |
| A | c1_current_136 | 0.238 | 0.670 | 0.074 |
| A | c2_flags8 | 0.235 | 0.662 | 0.071 |
| A | c3_current_struct | 0.237 | 0.666 | 0.129 |
| A | c4_current_cat | 0.235 | 0.663 | 0.119 |
| A | c5_current_struct_cat | 0.239 | 0.670 | 0.166 |
| B | c1_current_136 | 0.239 | 0.672 | 0.073 |
| B | c4_current_cat | 0.236 | 0.666 | 0.114 |
| B | c5_current_struct_cat | 0.239 | 0.671 | 0.135 |
| C | c1_current_136 | 0.239 | 0.670 | 0.069 |
| C | c4_current_cat | 0.235 | 0.663 | 0.117 |
| C | c5_current_struct_cat | 0.239 | 0.671 | 0.164 |
| D | c1_current_136 | 0.231 | 0.656 | 0.129 |
| D | c4_current_cat | 0.239 | 0.671 | 0.063 |
| D | c5_current_struct_cat | 0.241 | 0.675 | 0.162 |

The original artifact is severely overconfident and has very high log loss. Temperature scaling materially improves the new runs, but structural configurations remain poorly calibrated. ECE is computed on the balanced evaluation mix, not at production prevalence.

### 4.4 Original experiment comparison

| Quantity | Value |
|---|---:|
| Original reported test AUC | 0.6727 |
| Original reported test AP | 0.6094 |
| Original artifact test P@100 (recomputed) | 0.5997 |
| Original checkpoint reference test AUC | 0.6643 |
| Original checkpoint reference test AP | 0.5817 |
| Original checkpoint reference test P@100 | 0.0691 |
| c1 retrained test AUC | 0.6399 |
| c1 retrained test AP | 0.5761 |
| c4 retrained test AUC | 0.6850 |
| c4 retrained test AP | 0.6101 |

The original checkpoint reference is not pair-identical to the original artifact because Phase 2 regenerates negatives with a fixed seed. The low checkpoint-reference P@100 reflects score saturation and large tie groups at the top of the ranking; it is not directly comparable to the original artifact's P@100. The original artifact itself also contains saturated scores, but fewer exactly tied top scores under its own negative realization.

No claim of improvement over the original experiment is made. The c4 AP is numerically close to the original reported AP, but the evaluation pairs, negative realization, and seed are not identical, and no confidence intervals were computed.

## 5. Interpretation

1. **The original model's input was narrower than the current feature builder implies.** The 136-dimensional contract is flags plus embeddings. Any comparison that feeds the continuous/full-graph features into the original architecture is not the original experiment.
2. **Reverse-edge leakage is not the whole story.** A, B, and C produce similar rankings. The AnonNet leakage share is large enough to require B/C for clean claims, but removing it does not rescue or collapse the model.
3. **The embeddings carry limited signal.** c1 is not consistently better than c2. The 128-dimensional block may encode identity or source structure rather than generalizable supply-chain propensity.
4. **Categorical information is the most defensible feature addition so far.** c4 improves AUC in every non-temporal protocol, but its AP gain is modest and it does not beat PA on pooled AP.
5. **Structural features are not a clean win.** c3/c5 improve AP and top-K precision while reducing AUC and worsening calibration. They are train-graph degree/component features and are transductive. On SAMPOQ they degrade AUC from roughly 0.53 to 0.36-0.42, which is consistent with source-specific shortcut behavior rather than transferable structure.
6. **PA is a serious baseline, not a straw man.** Directed PA reaches pooled AP 0.697, 0.699, and 0.697 in A/B/C, versus 0.659, 0.656, and 0.656 for the best GNN. Under D, the GNNs lead because the temporal shift weakens the degree-based baseline.
7. **SAMPOQ needs a separate feature strategy.** With one unique 136-dimensional row, no node-level GNN can distinguish SAMPOQ nodes. Pooled metrics should not be presented without per-source results.
8. **Top-K precision alone is misleading.** Structural baselines and degree-heavy GNNs can place many positives at the top while having weak overall ranking AUC. Precision@K is tie-aware and reported alongside AP/AUC for this reason.
9. **Config 6 was not attempted.** Edge features are unavailable for candidate scoring and require a separate edge-aware design. Leaving it unimplemented is preferable to inventing an invalid scoring path.
10. **This remains link prediction.** No verified company-level disruption-risk labels were used, and link scores are not risk probabilities.

## 6. Limitations

- Single seed per configuration; no confidence intervals or repeated-seed variance estimates.
- Fixed hyperparameters; no per-configuration tuning beyond early stopping on validation AP.
- Balanced 50/50 evaluation negatives; precision, F1, Brier, and ECE are not production-prevalence estimates.
- Protocol B slightly changes validation/test counts because reverse pairs are grouped.
- Protocol C removes 1,185 training edges; this is a controlled intervention, not the original training distribution.
- Protocol D is IC-SPLC-only and has different source composition from A/B/C.
- AnonNet direction is not formally verified; reverse controls assume source/target indices are meaningful enough to test reversal sensitivity.
- The three source graphs are disconnected; no cross-source entity resolution exists.
- SAMPOQ has only one unique current feature row, so its GNN results mostly reflect graph position and source flags.
- Structural features are transductive and invalid for entirely unseen nodes.
- The original artifact negative-sample realization cannot be exactly recovered from the Phase 2 sampler.
- Categorical codes are anonymized; no official mapping or semantic validation was performed.

## 7. Recommendations

1. Use Protocol B or C as the primary random-split evidence, and Protocol D only for IC-SPLC temporal evidence. Do not pool D with A/B/C.
2. Keep directed and undirected PA/CN baselines in every experiment. A GNN result that does not beat PA on pooled and per-source AP is not yet a useful replacement.
3. Treat c1 and c4 as the strongest ranking models by AUC. Treat c5/c3 as top-K candidates only after controlling for degree shortcuts, calibration, and SAMPOQ degradation.
4. Do not use c2 as the default just because it is simple; it is competitive in A but weaker in B/C/D. The embedding ablation should be repeated across seeds before concluding that embeddings help.
5. Fix SAMPOQ or exclude it from pooled headline claims. Its current node features are effectively constant.
6. Add multiple seeds, bootstrap confidence intervals, and prevalence-matched or all-non-edge evaluation before making deployment claims.
7. Implement config 6 only if an edge-aware model can score candidate edges without using unavailable edge metadata. Otherwise keep edge features out of the scoring path.
8. Preserve explicit feature-name mapping for any checkpoint. The original checkpoint is not compatible with the current feature-builder column order without reordering.
9. Report calibration and top-K precision together with AP/AUC. The original artifact's high F1 hides severe overconfidence and poor P@100 under the Phase 2 ranking.
10. Revisit the 136-dimensional embedding block: test source-specific embeddings, embedding availability interactions, and no-embedding baselines under B/C and D before treating it as a core feature contract.

## 8. Reproducibility and artifacts

Command used:

```bash
.venv/bin/python ml/training/run_graphsage_phase2.py \
  --output-dir ml/experiments/graphsage_phase2 \
  --protocols A,B,C,D \
  --configs c1_current_136,c2_flags8,c3_current_struct,c4_current_cat,c5_current_struct_cat \
  --epochs 200 --patience 20 --overwrite
```

Primary artifacts:

| Artifact | Path |
|---|---|
| Machine-readable summary | `ml/experiments/graphsage_phase2/summary.json` |
| Protocol definitions and counts | `ml/experiments/graphsage_phase2/config/protocols.json` |
| Full protocol split assignments | `ml/experiments/graphsage_phase2/config/protocol_splits.csv` |
| Reverse-control removal IDs | `ml/experiments/graphsage_phase2/config/reverse_control_removals.json` |
| Run manifest | `ml/experiments/graphsage_phase2/config/run_manifest.json` |
| Feature audit JSON | `ml/experiments/graphsage_phase2/config/feature_audit.json` |
| Node feature audit CSV | `ml/experiments/graphsage_phase2/config/node_feature_audit.csv` |
| Edge feature audit CSV | `ml/experiments/graphsage_phase2/config/edge_feature_audit.csv` |
| Pooled comparison | `ml/experiments/graphsage_phase2/reports/comparison_pooled.csv` |
| By-source comparison | `ml/experiments/graphsage_phase2/reports/comparison_by_source.csv` |
| Long comparison | `ml/experiments/graphsage_phase2/reports/comparison_long.csv` |
| Pooled AP chart | `ml/experiments/graphsage_phase2/plots/pooled_average_precision.svg` |
| Pooled AUC chart | `ml/experiments/graphsage_phase2/plots/pooled_roc_auc.svg` |
| Pooled P@500 chart | `ml/experiments/graphsage_phase2/plots/pooled_precision_at_500.svg` |
| Reliability diagrams | `ml/experiments/graphsage_phase2/plots/<protocol>_<model>_reliability.svg` |
| Per-run metrics | `ml/experiments/graphsage_phase2/metrics/<protocol>/<model>.json` |
| Per-run predictions | `ml/experiments/graphsage_phase2/predictions/<protocol>/<model>_{val,test}.csv` |
| Per-run checkpoints | `ml/experiments/graphsage_phase2/checkpoints/<protocol>/<model>.pt` |
| Run log | `ml/experiments/graphsage_phase2/logs/run.log` |

Original artifacts were only read. `ml/experiments/graphsage_link_prediction/checkpoint.pt`, `metrics.json`, and `test_predictions.csv` retain their pre-run modification times (09:57) and were not overwritten by Phase 2.

No commit or push was performed. The working tree contains pre-existing uncommitted changes outside this experiment plus the new Phase 2 files.
