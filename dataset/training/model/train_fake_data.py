"""Fake data in the step 1 schema, for testing the plumbing before real data exists.

Writes to train_runs/fake_data/ (never to processed/):
  fake_net/    50 scenarios, split by scenario into train/val/test
  fake_net_b/  12 scenarios on a different layout, used only as a held-out network

Each scenario: 10 sensors (8 pressure, 2 flow), 5-minute steps, 3 days, a daily
sinusoid plus noise, and in leak scenarios a pressure dip that is largest at the
sensors closest to the leak node.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from train_config import FAKE_DATA_DIR

STEP_S = 300
N_STEPS = 3 * 24 * 12
ROWS, COLS = 3, 4  # 12 nodes on a grid


def _network(rng):
    nodes, links = [], []
    for r in range(ROWS):
        for c in range(COLS):
            x, y = c * 200 + rng.normal(0, 20), r * 200 + rng.normal(0, 20)
            nodes.append({"id": f"N{r * COLS + c + 1}", "x": round(float(x), 1), "y": round(float(y), 1)})
    for r in range(ROWS):
        for c in range(COLS):
            i = r * COLS + c
            if c + 1 < COLS:
                links.append((i, i + 1))
            if r + 1 < ROWS:
                links.append((i, i + COLS))
    links = [{"id": f"L{k + 1}", "start": nodes[a]["id"], "end": nodes[b]["id"]} for k, (a, b) in enumerate(links)]
    p_idx = np.sort(rng.choice(len(nodes), 8, replace=False))
    q_idx = np.sort(rng.choice(len(links), 2, replace=False))
    return nodes, links, p_idx, q_idx


def _hops(nodes, links):
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    adj = [[] for _ in nodes]
    for l in links:
        a, b = idx[l["start"]], idx[l["end"]]
        adj[a].append(b)
        adj[b].append(a)
    d = np.full((len(nodes), len(nodes)), 99)
    for s in range(len(nodes)):
        d[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if d[s, v] == 99:
                    d[s, v] = d[s, u] + 1
                    q.append(v)
    return d, idx


def _scenario(rng, nodes, links, p_idx, q_idx, hops, idx, elev, amp, leak: bool):
    hours = np.arange(N_STEPS) * STEP_S / 3600
    scale = rng.uniform(0.8, 1.2)  # demand level differs per scenario
    daily = np.sin(2 * np.pi * hours / 24 + rng.normal(0, 0.3)) + 0.3 * np.sin(4 * np.pi * hours / 24)
    cols = {"timestamp": pd.date_range("2024-01-01", periods=N_STEPS, freq=f"{STEP_S}s")}

    profile = np.zeros(N_STEPS)
    node, size, ltype = None, None, None
    if leak:
        node = int(rng.integers(len(nodes)))
        size = float(rng.uniform(0.5, 2.0))  # pressure drop at the leak node, m
        start = int(rng.uniform(30, 64) * 12)
        if rng.random() < 0.5:
            ltype = "abrupt"
            profile[start:] = 1.0
        else:
            ltype = "incipient"
            ramp = int(rng.uniform(6, 12) * 12)
            profile[start:] = np.minimum(1.0, np.arange(N_STEPS - start) / ramp)

    for i in p_idx:
        p = 50 + elev[i] - scale * amp[i] * daily + rng.normal(0, 0.15, N_STEPS)
        if leak:
            p -= size * np.exp(-hops[node, i] / 1.5) * profile
        cols[f"p_{nodes[i]['id']}"] = p
    for j in q_idx:
        q = 100 * scale * (1 + 0.3 * daily) + rng.normal(0, 1.0, N_STEPS)
        if leak:
            q += 2 * size * np.exp(-hops[node, idx[links[j]["start"]]] / 2) * profile
        cols[f"q_{links[j]['id']}"] = q

    active = (profile > 0).astype(int)
    cols["leak_active"] = active
    cols["leak_node"] = np.where(active == 1, nodes[node]["id"] if leak else None, None)
    cols["leak_size"] = np.where(active == 1, size if leak else np.nan, np.nan).astype(float)
    cols["leak_type"] = np.where(active == 1, ltype, None)
    return pd.DataFrame(cols)


def _write_dataset(out: Path, name: str, n: int, seed: int, heldout: bool) -> None:
    rng = np.random.default_rng(seed)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    nodes, links, p_idx, q_idx = _network(rng)
    hops, idx = _hops(nodes, links)
    elev = rng.uniform(-5, 5, len(nodes))
    amp = rng.uniform(2, 4, len(nodes))

    is_leak = np.arange(n) % 10 < 7  # 70% leak scenarios
    ids = {True: [], False: []}
    for k in range(n):
        sid = f"scn_{k:03d}"
        _scenario(rng, nodes, links, p_idx, q_idx, hops, idx, elev, amp, bool(is_leak[k])).to_parquet(d / f"{sid}.parquet")
        ids[bool(is_leak[k])].append(sid)

    splits = {"train": [], "val": [], "test": []}
    if heldout:
        splits["test"] = sorted(ids[True] + ids[False])
    else:  # stratified 70/15/15 by scenario
        for group in ids.values():
            group = list(rng.permutation(group))
            a, b = round(0.7 * len(group)), round(0.85 * len(group))
            splits["train"] += group[:a]
            splits["val"] += group[a:b]
            splits["test"] += group[b:]
    (d / "splits.json").write_text(json.dumps({k: sorted(map(str, v)) for k, v in splits.items()}, indent=2))
    (d / "network.json").write_text(json.dumps({
        "name": name, "fake": True, "sampling_interval_s": STEP_S,
        "units": {"pressure": "m", "flow": "m3/h", "leak_size": "m pressure drop at leak node (fake)"},
        "nodes": nodes, "links": links,
        "pressure_sensors": [nodes[i]["id"] for i in p_idx],
        "flow_sensors": [links[j]["id"] for j in q_idx],
    }, indent=2))


def generate(out: Path = FAKE_DATA_DIR, n_scenarios: int = 50, seed: int = 0) -> Path:
    _write_dataset(out, "fake_net", n_scenarios, seed, heldout=False)
    _write_dataset(out, "fake_net_b", 12, seed + 1000, heldout=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FAKE_DATA_DIR))
    ap.add_argument("--n_scenarios", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("wrote", generate(Path(a.out), a.n_scenarios, a.seed))
