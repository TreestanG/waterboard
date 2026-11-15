"""Pipe graph for a dataset: network.json -> PyG Data + sensor mask.

Nothing is invented. Any field network.json lacks is listed in `missing` (for the
step 1 owner) and replaced by a neutral placeholder that is recorded in
`data.placeholders`. Placeholders are for shape testing only, never for training.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch_geometric.data import Data

from train_windows import _nid

NODE_KEYS = {"elevation": ("elevation", "elevation_m", "elev"), "base_demand": ("base_demand", "base_demand_m3s", "demand")}
LINK_KEYS = {"length": ("length", "length_m"), "diameter": ("diameter", "diameter_m", "diam")}
END_KEYS = (("start", "end"), ("start_node", "end_node"), ("from", "to"), ("node1", "node2"))
PLACEHOLDER = {"elevation": 0.0, "base_demand": 0.0, "length": 1.0, "diameter": 1.0}


def _get(d: dict, keys):
    for k in keys:
        if d.get(k) is not None:
            return float(d[k])
    return None


def _as_dicts(raw, id_keys=("id", "name")) -> list:
    """Accept {id: {...}}, [{"id": ...}, ...] or ["id", ...]."""
    if isinstance(raw, dict):
        return [{**(v if isinstance(v, dict) else {}), "id": _nid(k)} for k, v in raw.items()]
    out = []
    for n in raw:
        if isinstance(n, dict):
            nid = next((n[k] for k in id_keys if k in n), None)
            out.append({**n, "id": _nid(nid) if nid is not None else None})
        else:
            out.append({"id": _nid(n)})
    return out


def load_graph(ds_dir, pressure_sensors=None):
    """Returns (data, sensor_mask, missing).

    data: PyG Data with
      x          [N, 3]  elevation (z-scored within the network), base demand (divided by the
                         network mean), sensor_present (0/1)
      edge_index [2, 2P] every pipe in both directions
      edge_attr  [2P, 2] pipe length and diameter, each divided by the network mean
      pos        [N, 2]  coordinates (only if every node has them)
      node_ids, link_ids, link_start (node index of each pipe's start), placeholders
    sensor_mask: bool [N], True where the node has a pressure sensor
    missing: list of strings describing what network.json lacks
    """
    ds_dir = Path(ds_dir)
    path = ds_dir / "network.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: the graph model needs step 1's network.json")
    raw = json.loads(path.read_text())
    nodes, links = _as_dicts(raw.get("nodes", [])), _as_dicts(raw.get("links", raw.get("pipes", [])))
    if not nodes or not links:
        raise ValueError(f"{path}: needs non-empty 'nodes' and 'links' lists, there is no graph without them")
    node_ids = [n["id"] for n in nodes]
    idx = {n: i for i, n in enumerate(node_ids)}
    N = len(nodes)
    missing, placeholders = [], []

    def column(items, name, keys, what):
        vals = [_get(it, keys) for it in items]
        # pumps and valves have no length or diameter: give them the network mean (1.0 once
        # divided by the mean), which is a neutral edge feature, not a missing field
        known = [v for v in vals if v is not None]
        if what == "links" and known:
            vals = [float(np.mean(known)) if v is None and it.get("type", "Pipe") != "Pipe" else v for it, v in zip(items, vals)]
        n_missing = sum(v is None for v in vals)
        if n_missing:
            missing.append(f"{what}[].{name}: missing on {n_missing} of {len(items)}")
            placeholders.append(name)
            vals = [PLACEHOLDER[name] if v is None else v for v in vals]
        return np.array(vals, dtype=np.float64)

    elev = column(nodes, "elevation", NODE_KEYS["elevation"], "nodes")
    demand = column(nodes, "base_demand", NODE_KEYS["base_demand"], "nodes")

    ends, kept = [], []
    for l in links:
        pair = next(((l[a], l[b]) for a, b in END_KEYS if a in l and b in l), None)
        if pair is None or _nid(pair[0]) not in idx or _nid(pair[1]) not in idx:
            continue
        ends.append((idx[_nid(pair[0])], idx[_nid(pair[1])]))
        kept.append(l)
    if len(kept) < len(links):
        missing.append(f"links[]: {len(links) - len(kept)} of {len(links)} have no start/end node id that matches nodes[]")
    if not kept:
        raise ValueError(f"{path}: no link has usable start/end node ids")
    length = column(kept, "length", LINK_KEYS["length"], "links")
    diam = column(kept, "diameter", LINK_KEYS["diameter"], "links")

    xy = []
    for n in nodes:
        c = n.get("coordinates", n.get("coords"))
        if c is None and "x" in n and "y" in n:
            c = (n["x"], n["y"])
        xy.append(c)
    if any(c is None for c in xy):
        missing.append(f"nodes[].x/y (coordinates): missing on {sum(c is None for c in xy)} of {N}")

    # ---- sensor mask ---------------------------------------------------------------------------
    if pressure_sensors is None:
        pressure_sensors = raw.get("pressure_sensors")
    if pressure_sensors is None and isinstance(raw.get("sensors"), dict):  # step 1 layout
        pressure_sensors = raw["sensors"].get("pressure")
    if pressure_sensors is None:  # fall back to the p_<node> columns of the first scenario
        first = next(iter(sorted(ds_dir.glob("*.parquet"))), None)
        if first is not None:
            pressure_sensors = [c[2:] for c in pq.read_schema(first).names if c.startswith("p_")]
            missing.append("pressure_sensors: not in network.json, taken from the p_<node> parquet columns")
    if pressure_sensors is None:
        missing.append("pressure_sensors: not in network.json and no parquet file to read them from")
        placeholders.append("sensor_present")
        pressure_sensors = []
    names = [_nid(s)[2:] if _nid(s).startswith("p_") else _nid(s) for s in pressure_sensors]
    unknown = [s for s in names if s not in idx]
    if unknown:
        missing.append(f"pressure sensors not found in nodes[]: {unknown[:5]}{'...' if len(unknown) > 5 else ''}")
    mask = np.zeros(N, dtype=bool)
    mask[[idx[s] for s in names if s in idx]] = True

    # ---- scale-free features, so one model can see several networks ----------------------------
    elev_z = (elev - elev.mean()) / (elev.std() if elev.std() > 1e-8 else 1.0)
    demand_rel = demand / (demand.mean() if abs(demand.mean()) > 1e-8 else 1.0)
    length_rel, diam_rel = length / max(length.mean(), 1e-8), diam / max(diam.mean(), 1e-8)

    ends = np.array(ends, dtype=np.int64).T  # [2, P]
    data = Data(
        x=torch.tensor(np.stack([elev_z, demand_rel, mask.astype(np.float64)], axis=1), dtype=torch.float32),
        edge_index=torch.from_numpy(np.concatenate([ends, ends[::-1]], axis=1)),
        edge_attr=torch.tensor(np.tile(np.stack([length_rel, diam_rel], axis=1), (2, 1)), dtype=torch.float32),
        num_nodes=N,
    )
    if all(c is not None for c in xy):
        data.pos = torch.tensor(xy, dtype=torch.float32)
    data.node_ids = node_ids
    data.link_ids = [l["id"] for l in kept]
    data.link_start = torch.from_numpy(ends[0].copy())
    data.placeholders = placeholders
    if placeholders:
        warnings.warn(f"{ds_dir.name}: PLACEHOLDER values used for {placeholders}. Testing only, do not train on this graph.")
    return data, torch.from_numpy(mask), missing


def node_inputs(X, sensors: list, graph, use_flow: bool = False) -> torch.Tensor:
    """Windows [B, C, W] from train_windows (one channel per sensor column) -> [B, N, W, F].

    F = (pressure, sensor_present[, flow]). Pressure is the normalised residual, 0 where the
    node has no sensor. With use_flow, each q_<link> series is placed on the link's start node.
    """
    X = torch.as_tensor(X)
    B, _, W = X.shape
    node_idx = {n: i for i, n in enumerate(graph.node_ids)}
    link_idx = {l: i for i, l in enumerate(graph.link_ids)}
    out = torch.zeros(B, graph.num_nodes, W, 3 if use_flow else 2, dtype=torch.float32)
    for c, name in enumerate(sensors):
        kind, key = name.split("_", 1)
        if kind == "p" and key in node_idx:
            out[:, node_idx[key], :, 0] = X[:, c]
            out[:, node_idx[key], :, 1] = 1.0
        elif kind == "q" and use_flow and key in link_idx:
            out[:, int(graph.link_start[link_idx[key]]), :, 2] = X[:, c]
    return out
