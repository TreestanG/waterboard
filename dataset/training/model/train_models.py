"""Models: gradient-boosting baseline, 1D CNN detector, 1D CNN localizer."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn


def window_features(X: np.ndarray) -> np.ndarray:
    """[N, C, W] -> [N, 5*C]: per sensor min, max, mean, last value, slope."""
    t = np.arange(X.shape[2], dtype=np.float32)
    t -= t.mean()
    slope = (X * t).sum(axis=2) / (t ** 2).sum()
    return np.concatenate([X.min(axis=2), X.max(axis=2), X.mean(axis=2), X[:, :, -1], slope], axis=1)


class GBBaseline:
    """Binary detection on window summary features. The number to beat."""

    def __init__(self, seed: int = 0):
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBBaseline":
        self.clf.fit(window_features(X), y)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(window_features(X))[:, 1]


class Backbone(nn.Module):
    """Three conv blocks over [B, n_sensors, W], then global average + max pooling."""

    out_dim = 128

    def __init__(self, n_sensors: int, dropout: float):
        super().__init__()

        def block(i, o, k, pool):
            layers = [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU()]
            return nn.Sequential(*layers, nn.MaxPool1d(2)) if pool else nn.Sequential(*layers)

        self.conv = nn.Sequential(block(n_sensors, 32, 7, True), block(32, 64, 5, True), block(64, 64, 3, False))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv(x)
        return self.drop(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))


class CNNDetector(nn.Module):
    """One logit per window: is a leak active at the last step?"""

    def __init__(self, n_sensors: int, dropout: float = 0.2):
        super().__init__()
        self.backbone = Backbone(n_sensors, dropout)
        self.head = nn.Linear(Backbone.out_dim, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


class CNNLocalizer(nn.Module):
    """Same backbone, one logit per node. Tied to one network: the head does not transfer."""

    def __init__(self, n_sensors: int, n_nodes: int, dropout: float = 0.2):
        super().__init__()
        self.backbone = Backbone(n_sensors, dropout)
        self.head = nn.Linear(Backbone.out_dim, n_nodes)

    def forward(self, x):
        return self.head(self.backbone(x))


class LSTMGNNDetector(nn.Module):
    """Shared LSTM per node -> static node features -> GAT layers over the pipe graph.

    forward(x, graph):
      x      [B, N, W, F]  per-node windows. F = pressure (0 where no sensor), sensor_present, [flow]
      graph  PyG Data for ONE network (train_graph.load_graph): x [N, S] static features,
             edge_index [2, E], edge_attr [E, edge_dim]
    returns (node_logits [B, N], det_logit [B]).

    Every weight is shared across nodes, so the same model runs on any network. There is
    no per-network head.
    """

    def __init__(self, in_features: int = 2, static_features: int = 3, edge_dim: int = 2, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.2, gnn_layers: int = 2, pool: str = "max", heads: int = 4):
        super().__init__()
        from torch_geometric.nn import GATConv  # lazy: the other models do not need PyG

        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by the number of attention heads")
        if pool not in ("max", "attn"):
            raise ValueError("pool must be 'max' or 'attn'")
        self.lstm = nn.LSTM(in_features, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_size + static_features, hidden_size)
        self.convs = nn.ModuleList(GATConv(hidden_size, hidden_size // heads, heads=heads, edge_dim=edge_dim, dropout=dropout) for _ in range(gnn_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(gnn_layers))
        self.drop = nn.Dropout(dropout)
        self.node_head = nn.Linear(hidden_size, 1)
        self.attn = nn.Linear(hidden_size, 1) if pool == "attn" else None

    def forward(self, x, graph):
        B, N, W, F_ = x.shape
        if N != graph.num_nodes:
            raise ValueError(f"x has {N} nodes but the graph has {graph.num_nodes}")
        _, (h, _) = self.lstm(x.reshape(B * N, W, F_))  # one sequence per (batch, node)
        h = torch.cat([self.drop(h[-1]), graph.x.repeat(B, 1)], dim=1)
        h = torch.relu(self.proj(h))
        # the batch is B disjoint copies of the same graph
        offsets = torch.arange(B, device=x.device).repeat_interleave(graph.edge_index.shape[1]) * N
        edge_index = graph.edge_index.repeat(1, B) + offsets
        edge_attr = graph.edge_attr.repeat(B, 1)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.drop(torch.relu(norm(conv(h, edge_index, edge_attr))))  # residual
        node_logits = self.node_head(h).view(B, N)
        if self.attn is None:
            det = node_logits.amax(dim=1)
        else:
            det = (torch.softmax(self.attn(h).view(B, N), dim=1) * node_logits).sum(dim=1)
        return node_logits, det


def build_model(cfg, n_sensors: int, n_nodes: int):
    if cfg.model == "gb":
        return GBBaseline(cfg.seed)
    if cfg.model == "cnn":
        return CNNDetector(n_sensors, cfg.dropout)
    if cfg.model == "cnn_loc":
        return CNNLocalizer(n_sensors, n_nodes, cfg.dropout)
    if cfg.model == "lstm_gnn":  # n_sensors / n_nodes unused: the model is network-agnostic
        return LSTMGNNDetector(in_features=3 if cfg.use_flow else 2, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
                               dropout=cfg.dropout, gnn_layers=cfg.gnn_layers, pool=cfg.graph_pool)
    raise ValueError(f"unknown model {cfg.model!r} (use gb, cnn, cnn_loc or lstm_gnn)")
