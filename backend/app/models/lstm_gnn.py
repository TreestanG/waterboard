"""The trained LSTM + GNN detector from models/lstm_gnn_full_h128.

One LSTM (shared by every node) reads a 24 h window of pressure residuals per node, then
two graph-attention layers pass messages along the pipes. The output is one leak logit per
node; the detection score is the largest of them. No weight is tied to a network, so the
same checkpoint runs on any .inp.

Inputs are prepared the way training prepared them (dataset/training/model):
  - 30-minute samples, 48-step (24 h) windows
  - residual = measured pressure minus its rolling median over the past 7 days
    (so the model sees the daily swing plus any drop; it does NOT use the hydraulic
    model's expected pressures)
  - z-scored with the pooled pressure mean/std of the training networks
  - graph features: elevation (z-scored), base demand and pipe length/diameter
    (each divided by the network mean), sensor-present flag

This checkpoint was trained with use_flow = false: it reads pressure sensors only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from .base import DetectionEvent, ModelInput, ModelOutput

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "lstm_gnn_full_h128"
STEP_H = 0.5  # training sampling interval
TOP_K = 5


class LSTMGNNDetector(nn.Module):
    """Same module as dataset/training/model/train_models.py (kept in sync by hand so the
    API doesn't import the training code and its dependencies)."""

    def __init__(self, in_features=2, static_features=3, edge_dim=2, hidden_size=64, num_layers=1,
                 dropout=0.2, gnn_layers=2, pool="max", heads=4):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.lstm = nn.LSTM(in_features, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_size + static_features, hidden_size)
        self.convs = nn.ModuleList(GATConv(hidden_size, hidden_size // heads, heads=heads, edge_dim=edge_dim, dropout=dropout) for _ in range(gnn_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(gnn_layers))
        self.drop = nn.Dropout(dropout)
        self.node_head = nn.Linear(hidden_size, 1)
        self.attn = nn.Linear(hidden_size, 1) if pool == "attn" else None

    def forward(self, x, graph):
        B, N, W, F_ = x.shape
        _, (h, _) = self.lstm(x.reshape(B * N, W, F_))
        h = torch.cat([self.drop(h[-1]), graph.x.repeat(B, 1)], dim=1)
        h = torch.relu(self.proj(h))
        offsets = torch.arange(B).repeat_interleave(graph.edge_index.shape[1]) * N
        edge_index = graph.edge_index.repeat(1, B) + offsets
        edge_attr = graph.edge_attr.repeat(B, 1)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.drop(torch.relu(norm(conv(h, edge_index, edge_attr))))
        node_logits = self.node_head(h).view(B, N)
        if self.attn is None:
            det = node_logits.amax(dim=1)
        else:
            det = (torch.softmax(self.attn(h).view(B, N), dim=1) * node_logits).sum(dim=1)
        return node_logits, det


def build_graph(g: dict, sensors: list[str]) -> SimpleNamespace:
    """Static graph tensors, as train_graph.load_graph builds them from network.json."""
    idx = {n: i for i, n in enumerate(g["node_ids"])}
    elev = np.asarray(g["elevation"], dtype=np.float64)
    demand = np.asarray(g["base_demand"], dtype=np.float64)
    mask = np.zeros(len(idx))
    mask[[idx[s] for s in sensors]] = 1.0
    links = [l for l in g["links"] if l["start"] in idx and l["end"] in idx]
    known_len = [l["length"] for l in links if l["length"] is not None]
    known_d = [l["diameter"] for l in links if l["diameter"] is not None]
    # pumps and valves have no length/diameter: they get the network mean (1.0 once divided)
    length = np.array([l["length"] if l["length"] is not None else np.mean(known_len) for l in links], dtype=np.float64)
    diam = np.array([l["diameter"] if l["diameter"] is not None else np.mean(known_d) for l in links], dtype=np.float64)
    ends = np.array([(idx[l["start"]], idx[l["end"]]) for l in links], dtype=np.int64).T
    elev_z = (elev - elev.mean()) / (elev.std() if elev.std() > 1e-8 else 1.0)
    demand_rel = demand / (demand.mean() if abs(demand.mean()) > 1e-8 else 1.0)
    return SimpleNamespace(
        x=torch.tensor(np.stack([elev_z, demand_rel, mask], axis=1), dtype=torch.float32),
        edge_index=torch.from_numpy(np.concatenate([ends, ends[::-1]], axis=1)),
        edge_attr=torch.tensor(np.tile(np.stack([length / max(length.mean(), 1e-8), diam / max(diam.mean(), 1e-8)], axis=1), (2, 1)), dtype=torch.float32),
        num_nodes=len(idx),
        node_index=idx,
    )


class LstmGnnModel:
    id = "lstm_gnn"
    name = "LSTM + GNN (trained, pressure only)"
    description = (
        "Trained on LeakDB Net1 and Hanoi plus simulated Net1 and Net3 leaks. Reads 24 h windows of pressure "
        "at 30-minute steps, so it gives no score for the first 24 h. Pressure sensors only; flow is not used."
    )

    def __init__(self) -> None:
        self._model: LSTMGNNDetector | None = None
        metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
        self.cfg = json.loads((MODEL_DIR / "config.json").read_text())
        self.threshold = float(metrics["threshold"])
        self.window = int(round(self.cfg["window_hours"] / STEP_H))
        self.baseline_steps = int(round(self.cfg["baseline_hours"] / STEP_H))
        pooled = json.loads((MODEL_DIR / "norm_stats.json").read_text())["_pooled"]["p"]
        self.mean, self.std = float(pooled["mean"]), float(pooled["std"])

    def _load(self) -> LSTMGNNDetector:
        if self._model is None:
            ck = torch.load(MODEL_DIR / "best.pt", map_location="cpu", weights_only=True)
            c = self.cfg
            model = LSTMGNNDetector(in_features=3 if c["use_flow"] else 2, hidden_size=c["hidden_size"], num_layers=c["num_layers"],
                                    dropout=c["dropout"], gnn_layers=c["gnn_layers"], pool=c["graph_pool"])
            model.load_state_dict(ck["state_dict"])
            self._model = model.eval()
        return self._model

    def run(self, inputs: ModelInput) -> ModelOutput:
        if inputs.graph is None:
            raise ValueError("The LSTM + GNN model needs the pipe graph.")
        model = self._load()
        graph = build_graph(inputs.graph, inputs.sensors)
        times = np.asarray(inputs.times_h, dtype=float)
        notes: list[str] = []

        # 30-minute grid, as in training
        grid = np.arange(0.0, times[-1] + 1e-9, STEP_H)
        x = np.stack([np.interp(grid, times, inputs.measured[:, c]) for c in range(inputs.measured.shape[1])], axis=1)
        resid = (pd.DataFrame(x) - pd.DataFrame(x).rolling(self.baseline_steps, min_periods=1).median()).to_numpy()
        z = torch.tensor((resid - self.mean) / self.std, dtype=torch.float32)

        W = self.window
        ends = np.arange(W - 1, len(grid))
        score_grid = np.full(len(grid), np.nan)
        node_prob = np.zeros((len(grid), len(inputs.junctions)), dtype=np.float32)
        j_idx = torch.tensor([graph.node_index[j] for j in inputs.junctions])
        if len(ends) == 0:
            notes.append("The simulation is shorter than the model's 24 h window, so it produced no score. Simulate at least 48 h.")
        else:
            p_node = torch.tensor([graph.node_index[s] for s in inputs.sensors])
            bs = max(8, 20000 // graph.num_nodes)
            with torch.no_grad():
                for i in range(0, len(ends), bs):
                    e = torch.from_numpy(ends[i : i + bs])
                    win = z[e[:, None] + torch.arange(-W + 1, 1)]  # [B, W, C]
                    xb = torch.zeros(len(e), graph.num_nodes, W, 2)
                    xb[:, p_node, :, 0] = win.permute(0, 2, 1)
                    xb[:, p_node, :, 1] = 1.0
                    node_logits, det = model(xb, graph)
                    score_grid[ends[i : i + bs]] = torch.sigmoid(det).numpy()
                    # where the leak is, given there is one: softmax over junctions (the training loss)
                    node_prob[ends[i : i + bs]] = torch.softmax(node_logits[:, j_idx], dim=1).numpy()

        # Alarm on each rising edge of score >= threshold.
        events: list[DetectionEvent] = []
        above = np.nan_to_num(score_grid, nan=0.0) >= self.threshold
        jxy = inputs.junction_xy
        for k in np.flatnonzero(above & ~np.concatenate([[False], above[:-1]])):
            order = np.argsort(-node_prob[k])[:TOP_K]
            best = int(order[0])
            events.append(DetectionEvent(
                time_h=float(grid[k]), node=inputs.junctions[best], x=float(jxy[best, 0]), y=float(jxy[best, 1]),
                candidates=[{"node": inputs.junctions[int(j)], "probability": round(float(node_prob[k, j]), 4),
                             "x": float(jxy[j, 0]), "y": float(jxy[j, 1])} for j in order],
            ))

        if len(ends) and above[ends].all():
            notes.append(
                "The model scored every window above its alarm threshold, including before any leak could have started, "
                "so this result says nothing about the leaks. This checkpoint did the same on simulated Net3 in its own "
                "validation (it only separated leaks on the LeakDB Net1 and Hanoi data)."
            )

        # Back onto the simulation's own time axis: each step shows the latest 30-minute score.
        at = np.clip(np.floor((times + 1e-9) / STEP_H).astype(int), 0, len(grid) - 1)
        score = [None if np.isnan(v) else round(float(v), 4) for v in score_grid[at]]
        if len(ends) and len(times) > 1 and abs((times[1] - times[0]) - STEP_H) > 1e-6:
            notes.append(f"Readings were resampled from {round((times[1] - times[0]) * 60)}-minute to 30-minute steps, the interval the model was trained on.")
        return ModelOutput(score=score, threshold=self.threshold, score_label="Leak probability", events=events, notes=notes)
