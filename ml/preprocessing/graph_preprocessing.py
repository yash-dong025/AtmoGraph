"""Build and validate a PyTorch Geometric graph for AtmoGraph.

The existing graph pipeline already defines:
- node = company/entity,
- directed edge = supplier -> customer,
- namespaced node IDs such as ``sampoq:...`` and ``icsplc:...``,
- a stable ``node_index_mapping.csv``,
- an ``edge_index.csv`` with train/val/test edge splits.

This module converts those existing artifacts into a PyG ``Data`` object. It
does not rebuild the graph and does not change the source datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.gcn_model import build_gnn
from ml.preprocessing.feature_engineering import build_edge_features, build_node_features


def build_pyg_data() -> Data:
    nodes = build_node_features()
    edges = build_edge_features()

    num_nodes = len(nodes.node_ids)
    if edges.edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edges.edge_index.min() < 0 or edges.edge_index.max() >= num_nodes:
        raise ValueError("edge_index contains out-of-range node indices")
    if (edges.edge_index[0] == edges.edge_index[1]).any():
        raise ValueError("edge_index contains self-loops")
    if len(set(edges.edge_ids)) != len(edges.edge_ids):
        raise ValueError("edge_ids are not unique")
    if len(set(nodes.node_ids)) != len(nodes.node_ids):
        raise ValueError("node_ids are not unique")

    data = Data(
        x=torch.from_numpy(nodes.x).float(),
        edge_index=torch.from_numpy(edges.edge_index).long(),
        edge_attr=torch.from_numpy(edges.edge_attr).float(),
        x_cat=torch.from_numpy(nodes.x_cat).long(),
        x_emb=torch.from_numpy(nodes.x_emb).float(),
        emb_mask=torch.from_numpy(nodes.emb_mask).bool(),
        train_edge_mask=torch.from_numpy(edges.train_mask).bool(),
        val_edge_mask=torch.from_numpy(edges.val_mask).bool(),
        test_edge_mask=torch.from_numpy(edges.test_mask).bool(),
        node_train_mask=torch.from_numpy(nodes.train_mask).bool(),
        node_val_mask=torch.from_numpy(nodes.val_mask).bool(),
        node_test_mask=torch.from_numpy(nodes.test_mask).bool(),
        num_nodes=num_nodes,
    )
    data.node_ids = nodes.node_ids
    data.edge_ids = edges.edge_ids
    data.cat_cardinalities = nodes.cat_cardinalities
    data.feature_names = nodes.feature_names
    data.edge_feature_names = edges.feature_names
    return data


def validate_pyg_data(data: Data) -> dict[str, object]:
    checks = {
        "node_count": int(data.num_nodes),
        "edge_count": int(data.num_edges),
        "node_feature_dim": int(data.x.shape[1]),
        "edge_feature_dim": int(data.edge_attr.shape[1]),
        "node_features_finite": bool(torch.isfinite(data.x).all()),
        "edge_features_finite": bool(torch.isfinite(data.edge_attr).all()),
        "edge_index_in_range": bool(
            int(data.edge_index.min()) >= 0 and int(data.edge_index.max()) < data.num_nodes
        ),
        "no_self_loops": bool(not (data.edge_index[0] == data.edge_index[1]).any()),
        "edge_ids_unique": bool(len(set(data.edge_ids)) == len(data.edge_ids)),
        "node_ids_unique": bool(len(set(data.node_ids)) == len(data.node_ids)),
        "node_split_sizes": {
            "train": int(data.node_train_mask.sum()),
            "val": int(data.node_val_mask.sum()),
            "test": int(data.node_test_mask.sum()),
        },
        "edge_split_sizes": {
            "train": int(data.train_edge_mask.sum()),
            "val": int(data.val_edge_mask.sum()),
            "test": int(data.test_edge_mask.sum()),
        },
        "embedding_coverage": float(data.emb_mask.float().mean()),
        "categorical_cardinalities": list(data.cat_cardinalities),
    }
    failed = [
        key
        for key in (
            "node_features_finite",
            "edge_features_finite",
            "edge_index_in_range",
            "no_self_loops",
            "edge_ids_unique",
            "node_ids_unique",
        )
        if not checks[key]
    ]
    if failed:
        raise ValueError(f"PyG graph validation failed: {failed}")
    return checks


def smoke_test(data: Data, model_name: str = "gcn") -> dict[str, object]:
    model = build_gnn(
        model_name,
        in_channels=int(data.x.shape[1]),
        hidden_channels=32,
        out_channels=16,
    )
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        sample_count = min(100, int(data.num_edges))
        sample_edges = data.edge_index[:, :sample_count]
        scores = model.decode(z, sample_edges)
    if not torch.isfinite(z).all() or not torch.isfinite(scores).all():
        raise ValueError("GNN smoke test produced non-finite values")
    return {
        "model": model_name,
        "embedding_shape": list(z.shape),
        "sample_score_shape": list(scores.shape),
        "embedding_dim": int(z.shape[1]),
        "device": str(z.device),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the AtmoGraph PyG Data object")
    parser.add_argument("--smoke-test", action="store_true", help="Run a forward pass")
    parser.add_argument("--model", choices=["gcn", "sage", "gat"], default="gcn")
    parser.add_argument("--save", type=Path, default=None, help="Optional .pt output path")
    args = parser.parse_args()

    data = build_pyg_data()
    summary = validate_pyg_data(data)
    if args.smoke_test:
        summary["smoke_test"] = smoke_test(data, model_name=args.model)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, args.save)
        summary["saved_to"] = str(args.save)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
