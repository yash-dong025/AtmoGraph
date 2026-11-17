from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


class SourceAwareCategoricalEmbeddings(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        source_count: int = 3,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if not cardinalities:
            raise ValueError("cardinalities must be non-empty")
        if source_count < 1:
            raise ValueError("source_count must be positive")
        self.cardinalities = [int(value) for value in cardinalities]
        self.source_count = int(source_count)
        self.embedding_dim = int(embedding_dim)
        self.tables = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Embedding(cardinality, embedding_dim) for cardinality in self.cardinalities]
                )
                for _ in range(self.source_count)
            ]
        )
        self.output_dim = len(self.cardinalities) * self.embedding_dim

    def forward(self, x_cat: torch.Tensor, source_id: torch.Tensor) -> torch.Tensor:
        if x_cat.ndim != 2:
            raise ValueError("x_cat must have shape [N, fields]")
        if source_id.ndim != 1 or source_id.shape[0] != x_cat.shape[0]:
            raise ValueError("source_id must have shape [N]")
        if x_cat.shape[1] != len(self.cardinalities):
            raise ValueError("x_cat field count does not match cardinalities")
        if int(source_id.min()) < 0 or int(source_id.max()) >= self.source_count:
            raise ValueError("source_id contains an out-of-range value")
        for field_index, cardinality in enumerate(self.cardinalities):
            column = x_cat[:, field_index]
            if int(column.min()) < 0 or int(column.max()) >= cardinality:
                raise ValueError("x_cat contains an out-of-range category index")

        one_hot = F.one_hot(source_id, num_classes=self.source_count).to(x_cat.dtype)
        parts = []
        for field_index in range(len(self.cardinalities)):
            candidates = torch.stack(
                [
                    self.tables[source_index][field_index](x_cat[:, field_index])
                    for source_index in range(self.source_count)
                ],
                dim=1,
            )
            parts.append((candidates * one_hot.unsqueeze(-1)).sum(dim=1))
        return torch.cat(parts, dim=-1)


class AtmoGraphSAGESourceCat(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cardinalities: list[int],
        source_count: int = 3,
        categorical_dim: int = 16,
        hidden_channels: int = 64,
        out_channels: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.categorical = SourceAwareCategoricalEmbeddings(
            cardinalities=cardinalities,
            source_count=source_count,
            embedding_dim=categorical_dim,
        )
        encoded_channels = int(in_channels) + self.categorical.output_dim
        self.conv1 = SAGEConv(encoded_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        x_cat: torch.Tensor,
        source_id: torch.Tensor,
    ) -> torch.Tensor:
        categorical = self.categorical(x_cat, source_id)
        x = torch.cat([x, categorical], dim=-1)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        return x

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
