"""Fake scenarios on small random pipe graphs, for testing the graph model's plumbing.

Writes two networks with different node counts to train_runs/fake_graph_data/
(never to processed/). Same parquet schema as train_fake_data.py, but network.json
also carries what the graph model needs: node elevation, base_demand and x/y, and
pipe length and diameter. A leak lowers pressure most at the sensors fewest pipes away.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_config import RUNS_DIR
from train_fake_data import STEP_S, _hops, _scenario

FAKE_GRAPH_DIR = RUNS_DIR / "fake_graph_data"


def _random_network(rng, n_nodes: int):
    xy = rng.uniform(0, 1000, (n_nodes, 2))
    nodes = [{"id": f"J{i + 1}", "x": round(float(x), 1), "y": round(float(y), 1),
              "elevation": round(float(rng.uniform(0, 30)), 2), "base_demand": round(float(rng.uniform(0.5, 5)), 3)}
             for i, (x, y) in enumerate(xy)]
    pairs = set()
    for i in range(1, n_nodes):  # spanning tree: join each node to its nearest earlier node
        j = int(np.argmin(np.linalg.norm(xy[:i] - xy[i], axis=1)))
        pairs.add((j, i))
    while len(pairs) < n_nodes - 1 + max(2, n_nodes // 4):  # a few loops
        a, b = sorted(rng.choice(n_nodes, 2, replace=False).tolist())
        pairs.add((a, b))
    links = [{"id": f"P{k + 1}", "start": nodes[a]["id"], "end": nodes[b]["id"],
              "length": round(float(np.linalg.norm(xy[a] - xy[b])), 1), "diameter": float(rng.choice([100, 150, 200, 300]))}
             for k, (a, b) in enumerate(sorted(pairs))]
    p_idx = np.sort(rng.choice(n_nodes, max(3, int(0.4 * n_nodes)), replace=False))  # sparse sensors
    q_idx = np.sort(rng.choice(len(links), 2, replace=False))
    return nodes, links, p_idx, q_idx


def write_dataset(out: Path, name: str, n_nodes: int, n_scenarios: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    nodes, links, p_idx, q_idx = _random_network(rng, n_nodes)
    hops, idx = _hops(nodes, links)
    elev = np.array([n["elevation"] for n in nodes]) - 15  # pressure offset per node
    amp = rng.uniform(2, 4, n_nodes)
    ids = []
    for k in range(n_scenarios):
        sid = f"scn_{k:03d}"
        _scenario(rng, nodes, links, p_idx, q_idx, hops, idx, elev, amp, leak=k % 10 < 7).to_parquet(d / f"{sid}.parquet")
        ids.append(sid)
    a, b = round(0.7 * n_scenarios), round(0.85 * n_scenarios)
    (d / "splits.json").write_text(json.dumps({"train": ids[:a], "val": ids[a:b], "test": ids[b:]}, indent=2))
    (d / "network.json").write_text(json.dumps({
        "name": name, "fake": True, "sampling_interval_s": STEP_S,
        "units": {"pressure": "m", "flow": "m3/h", "elevation": "m", "base_demand": "m3/h", "length": "m", "diameter": "mm"},
        "nodes": nodes, "links": links,
        "pressure_sensors": [nodes[i]["id"] for i in p_idx],
        "flow_sensors": [links[j]["id"] for j in q_idx],
    }, indent=2))
    return d


def generate(out: Path = FAKE_GRAPH_DIR, n_scenarios: int = 10, seed: int = 0) -> Path:
    write_dataset(out, "fake_graph_a", 15, n_scenarios, seed)
    write_dataset(out, "fake_graph_b", 24, n_scenarios, seed + 1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FAKE_GRAPH_DIR))
    ap.add_argument("--n_scenarios", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("wrote", generate(Path(a.out), a.n_scenarios, a.seed))
