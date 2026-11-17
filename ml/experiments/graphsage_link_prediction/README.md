# GraphSAGE Link-Prediction Experiment

Status: **trained once, source-matched evaluation complete**.

This directory contains the real GraphSAGE checkpoint and evaluation metrics for
directed supplier/customer link prediction, plus the leakage-controlled structural
baselines evaluated on the same validation/test pairs.

## Artifacts

- `config.json`: GraphSAGE architecture and evaluation protocol.
- `dataset_metadata.json`: graph counts, split sizes, feature dimensions, and file hashes.
- `baseline_metrics.json`: real common-neighbor / preferential-attachment metrics.
- `baseline_test_predictions.csv`: baseline test scores for positives and sampled negatives.
- `checkpoint.pt`: trained GraphSAGE weights.
- `history.json`: per-epoch training and validation history.
- `metrics.json`: pooled and per-source test metrics.
- `test_predictions.csv`: GraphSAGE test scores for positives and sampled negatives.
- `run_summary.json`: command, seed, runtime, and run configuration.
- `experiment.json`: dashboard-facing status and artifact inventory.

## Evaluation protocol

- Message passing uses **train-split edges only**.
- Validation/test positives are used only for evaluation; no test edges train the model.
- Negative pairs are **source-matched directed non-edges**: the positive source-pair
  distribution is preserved and every observed directed edge is excluded.
- GraphSAGE and both structural baselines use the same seeded validation/test pairs.
- Metrics are reported overall and per source dataset.
- The decision threshold is selected on pooled validation by maximum F1 among
  non-trivial thresholds, then reused for every source.
- The structural baseline is selected by validation average precision, then ROC-AUC.
- Link prediction is not disruption-risk prediction. No verified risk labels exist.

## Run training

```bash
.venv/bin/python -m ml.training.train_graphsage_link_prediction
```

The training script refuses to overwrite an existing checkpoint unless
`--overwrite` is supplied.
