# GraphSAGE Link-Prediction Experiment

Status: **not trained**.

This directory contains the real experiment setup, dataset metadata, and a
leakage-controlled structural link-prediction baseline. It does **not** contain a
trained GraphSAGE checkpoint or GNN evaluation metrics.

## Available now

- `config.json`: planned GraphSAGE architecture and evaluation protocol.
- `dataset_metadata.json`: graph counts, split sizes, feature dimensions, and file hashes.
- `baseline_metrics.json`: actual common-neighbor / preferential-attachment baseline metrics.
- `baseline_test_predictions.csv`: actual baseline test scores for real positive edges and sampled negatives.

## Not available yet

The following files are produced only by a successful training run:

- `checkpoint.pt`
- `history.json`
- `metrics.json`
- `test_predictions.csv`
- `run_summary.json`

Until those files exist, the dashboard must show GNN training and evaluation as
unavailable. The PyG smoke test is not a training run.

## Run training

```bash
.venv/bin/python -m ml.training.train_graphsage_link_prediction
```

The training script refuses to overwrite an existing checkpoint unless
`--overwrite` is supplied.

## Leakage controls

- Graph message passing uses **train-split edges only**.
- Validation and test positives are used only for evaluation.
- Negative pairs are sampled from directed non-edges, excluding every observed edge.
- Degree, component-size, and edge-year features are excluded from node features.
- The decision threshold is selected on validation and applied once to test.

## Scope

This is a link-prediction experiment: it scores whether a directed supplier/customer
edge is likely to exist. It is not a disruption-risk predictor. No verified
company-level risk labels are available in the current datasets.
