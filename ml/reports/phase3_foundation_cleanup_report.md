# Phase 3 Foundation Cleanup Report

## 1. Summary

Phase 3 completed the engineering cleanup of the ML foundation without retraining models, changing the graph, changing splits, or regenerating old results.

Completed work:

- Replaced the risk-oriented inference contract with a distinct directed link-prediction contract.
- Corrected stale experiment metadata and preserved previous status values.
- Removed machine-specific absolute paths from the targeted experiment metadata.
- Updated the future training writers so new outputs use repository-relative paths.
- Added a backward-compatible dashboard experiment catalog.
- Added a dedicated synthetic ML test suite and test configuration.
- Updated the root and ML READMEs.
- Verified preserved artifacts and the final file inventory.

Validation:

- `.venv/bin/python -m pytest` -> **19 passed**.
- Dashboard catalog smoke test -> **135 catalog entries** (1 original + 134 Phase 2 comparison rows).
- Dashboard inference smoke test -> `task=link_prediction`, `score_type=model_score`, `probability_calibrated=false`.
- `git diff --check` -> clean.
- No checkpoint, metric, prediction, or Phase 2 summary result was regenerated.

No commit or push was performed.

## 2. Inference-Contract Changes

File: `ml/inference/predict.py`

The old contract exposed `PredictionRecord`, `risk_score`, `risk_level`, `risk_level_from_score()`, and a default `prediction_type="disruption_risk"`. That was removed.

New contract:

- `LinkPredictionRecord`.
- `build_link_prediction_record()`.
- `ensure_checkpoint_available()`.
- `load_link_checkpoint()`.
- `RiskPredictionUnsupportedError`.
- `CheckpointUnavailableError`.

The new record requires:

- `task="link_prediction"`.
- `link_score` as a finite `[0, 1]` **model score**.
- `score_type="model_score"`.
- `probability_calibrated=false`.
- distinct `source_id` and `target_id`.
- source metadata, target metadata, and model metadata.
- `prediction_status="experimental"`.

The record does not contain `risk_score`, `risk_level`, `LOW`/`MEDIUM`/`HIGH`, or a `disruption_risk` task. Its warnings explicitly state that the score is not a verified probability and not a disruption-risk prediction.

Missing checkpoints now raise `CheckpointUnavailableError` with an explicit no-score message. `build_risk_prediction_record()` always raises `RiskPredictionUnsupportedError` because no verified risk labels or dedicated risk model exist.

Integration:

- `ml/reports/dashboard_data.py` now builds `link_prediction_record` for `predict_pair()`.
- Existing `score`, `logit`, and `threshold` fields remain for compatibility.
- New additive fields: `link_score`, `task`, `score_type`, `probability_calibrated`, and `link_prediction_record`.

Consumer check:

- No repository code consumer of the old `PredictionRecord`, `build_prediction_record()`, or `risk_level_from_score()` was found in `ml/` or `scripts/`.
- Only documentation referenced the old contract.
- External consumers, if any, must migrate to `build_link_prediction_record()` and source/target pair inputs.

## 3. Metadata Corrections

Verified before editing: `checkpoint.pt`, `metrics.json`, `experiment.json`, and `run_summary.json` all show a completed run.

| File | Old value | New value | Evidence |
|---|---|---|---|
| `ml/experiments/graphsage_link_prediction/config.json` | `status: planned` | `status: completed` | checkpoint, metrics, experiment |
| `ml/experiments/graphsage_link_prediction/baseline_metrics.json` | GraphSAGE `status: not_trained` | `status: completed` | checkpoint, metrics, experiment |

Historical values were preserved:

- `config.json` now contains `previous_status: "planned"` and a `metadata_correction` object.
- `baseline_metrics.json` now contains `previous_status: "not_trained"` and a `correction_note`.

The baseline metrics themselves were not recalculated. The git diff for `baseline_metrics.json` is metadata-only.

## 4. Documentation Changes

Root `README.md` was expanded into 10 sections:

1. Project overview.
2. Architecture.
3. Dataset sources.
4. ML pipeline.
5. How to run experiments.
6. Results.
7. Limitations.
8. Repository structure.
9. License and compliance.
10. Current status.

`ml/README.md` was updated to:

- Remove stale Week 1-only/branch-ahead claims.
- Document the original and Phase 2/3 status.
- Distinguish the full 146-d feature matrix from the original 136-d checkpoint contract.
- Replace the old risk contract with the link-prediction contract.
- Document the additive dashboard catalog and test commands.
- Add current limitations: feature collapse, uncalibrated scores, embedding coverage, and unsupported risk prediction.

Both READMEs explicitly distinguish:

- directed link prediction,
- disruption risk,
- shipment delay.

## 5. Portability Changes

Corrected existing metadata:

- `ml/experiments/graphsage_link_prediction/experiment.json`: absolute checkpoint path -> repository-relative path.
- `ml/experiments/graphsage_link_prediction/run_summary.json`: absolute command -> repository-relative module command.
- `ml/experiments/graphsage_phase2/config/run_manifest.json`: absolute original experiment directory -> repository-relative path.

Prevented future regressions:

- `ml/training/train_graphsage_link_prediction.py` now writes portable checkpoint paths and commands.
- `ml/training/run_graphsage_phase2.py` now writes portable manifest paths, commands, and checkpoint references.

No training outputs were regenerated. The existing absolute path in `ml/experiments/graphsage_phase2/metrics/A/original_graphsage_reference.json` was intentionally left unchanged to avoid rewriting a historical output; the generator now writes relative paths for future runs.

## 6. Tests and Results

Created test configuration:

- `pytest.ini`.

Created test dependency file:

- `ml/requirements-test.txt`.

Created test suite:

- `ml/tests/test_inference_contract.py`.
- `ml/tests/test_checkpoint_loading.py`.
- `ml/tests/test_features_and_splits.py`.
- `ml/tests/test_evaluation.py`.
- `ml/tests/test_dashboard_catalog.py`.
- `ml/tests/test_metadata_and_paths.py`.

Coverage:

- Link-prediction contract fields and forbidden risk fields.
- Score bounds and finite checks.
- Source/target validation.
- Explicit unsupported risk behavior.
- Missing-checkpoint behavior.
- Checkpoint metadata loading and required keys.
- Node/edge feature dimensions, encodings, masks, and training-only fill.
- Real split disjointness and metadata count consistency.
- Validation-only threshold selection and test evaluation.
- Single-class source handling without fabricated AUC/AP.
- Catalog separation of original vs Phase 2, protocol, feature config, source, model type, and metrics.
- Metadata status corrections and portable path checks.

Result:

```text
19 passed
```

No full experiment was run by the test suite. Tests use small synthetic fixtures plus real split-integrity metadata.

## 7. Files Modified

- `README.md`
- `ml/README.md`
- `ml/inference/__init__.py`
- `ml/inference/predict.py`
- `ml/reports/dashboard_data.py`
- `ml/experiments/graphsage_link_prediction/config.json`
- `ml/experiments/graphsage_link_prediction/baseline_metrics.json`
- `ml/experiments/graphsage_link_prediction/experiment.json`
- `ml/experiments/graphsage_link_prediction/run_summary.json`
- `ml/experiments/graphsage_phase2/config/run_manifest.json`
- `ml/training/train_graphsage_link_prediction.py`
- `ml/training/run_graphsage_phase2.py`

## 8. Files Created

- `pytest.ini`
- `ml/requirements-test.txt`
- `ml/tests/test_inference_contract.py`
- `ml/tests/test_checkpoint_loading.py`
- `ml/tests/test_features_and_splits.py`
- `ml/tests/test_evaluation.py`
- `ml/tests/test_dashboard_catalog.py`
- `ml/tests/test_metadata_and_paths.py`
- `ml/reports/phase3_foundation_cleanup_report.md`

## 9. Compatibility Issues

- The old inference classes/functions were removed. This is a breaking change for external code that imported `PredictionRecord`, `build_prediction_record()`, or `risk_level_from_score()`. No in-repo consumer was found.
- Migration path: use `build_link_prediction_record()` with `source_id`, `target_id`, source/target metadata, and model metadata. Use `build_risk_prediction_record()` only to receive the explicit unsupported-task error.
- `/api/experiment` now includes an additive `catalog` key. Existing keys are unchanged. The catalog currently contains 135 entries and increases payload size.
- No frontend file was modified. The frontend may ignore the catalog until it is explicitly wired.
- The dashboard `POST /api/predict` retains existing fields and adds link-prediction fields; existing consumers should continue to work.
- Some historical artifacts still contain absolute paths. Only the targeted metadata and future writers were corrected.

## 10. Remaining Technical Debt

- No verified company-level disruption labels; disruption-risk prediction remains unsupported.
- No cross-source entity resolution; anonymised and named companies remain separate.
- Link scores are uncalibrated and should not be presented as probabilities.
- Original-checkpoint calibration is poor: Brier `0.3354`, ECE `0.3253`.
- SAMPOQ has one unique 136-d feature row; AnonNet has four.
- IC-SPLC embedding coverage is 28.5% in the original 136-d contract.
- The original model ignores 12 edge features and 3 categorical fields.
- Phase 2 Protocol A retains reverse-edge leakage; B/C control it but do not recover unobserved data.
- The dashboard remains local research tooling, not a production API/frontend.
- One historical absolute path remains in a Phase 2 metrics file.
- Dataset licensing and provenance gaps remain for AnonNet, SynDelay, Smart Logistics, and GDELT.
- No commit or push was performed.
