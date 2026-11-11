"""GNN prototypes for AtmoGraph.

Architecture recommendation for Week 2
---------------------------------------
Use **GraphSAGE** as the primary candidate, with GCN as the simple baseline and
GAT as a comparison model.

Why GraphSAGE first:
- The graph is fragmented (23,198 weakly connected components) and many nodes
  have low degree. GraphSAGE aggregates neighbourhood information without
  requiring the full graph Laplacian.
- It supports inductive-style neighbourhood aggregation if new entities are
  added later.
- It is less sensitive than GAT to noisy attention weights on very small
  neighbourhoods.

GCN is the baseline because it is simple, fast, and well understood. GAT is
worth testing because supplier/customer edges have different confidence and
relationship types, but attention can overfit on low-degree nodes and is more
expensive.

These are prototypes, not trained production models. There are no verified
company-level risk labels in the current datasets, so a risk head must not be
trained or reported as a real predictor.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class _TwoLayerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1: nn.Module
        self.conv2: nn.Module
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(out_channels)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        return x

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x, edge_index), edge_index)


class AtmoGCN(_TwoLayerEncoder):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(in_channels, hidden_channels, out_channels, dropout)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)


class AtmoGraphSAGE(_TwoLayerEncoder):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(in_channels, hidden_channels, out_channels, dropout)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)


class AtmoGAT(_TwoLayerEncoder):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        dropout: float = 0.2,
        heads: int = 4,
    ) -> None:
        super().__init__(in_channels, hidden_channels, out_channels, dropout)
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
        )
        self.conv2 = GATConv(
            hidden_channels,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )


def build_gnn(
    name: str,
    in_channels: int,
    hidden_channels: int = 64,
    out_channels: int = 32,
    dropout: float = 0.2,
) -> _TwoLayerEncoder:
    name = name.lower()
    if name == "gcn":
        return AtmoGCN(in_channels, hidden_channels, out_channels, dropout)
    if name in {"sage", "graphsage"}:
        return AtmoGraphSAGE(in_channels, hidden_channels, out_channels, dropout)
    if name == "gat":
        return AtmoGAT(in_channels, hidden_channels, out_channels, dropout)
    raise ValueError(f"Unknown GNN model: {name}. Choose gcn, sage, or gat.")
