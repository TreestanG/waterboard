"""Unified parquet scenarios -> sliding windows as numpy arrays (step 2, read-only on step 1).

Per scenario: regular time grid -> forward-fill small gaps -> subtract a rolling
median baseline -> normalise with train-only stats -> windows of shape [C, W].
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class WindowSet:
    X: np.ndarray  # [N, C, W] float32
    y: np.ndarray  # [N] 0/1, leak active at the last step of the window
    node: np.ndarray  # [N] node index of the leak at the last step, -1 if none/unknown
    scen: np.ndarray  # [N] index into self.scenarios
    end_idx: np.ndarray  # [N] step index of the last step of the window
    scenarios: list  # dicts: dataset, id, step_s, n_steps, events [(start, end)]

    def __len__(self):
        return len(self.y)

    def subset(self, mask) -> "WindowSet":
        return WindowSet(self.X[mask], self.y[mask], self.node[mask], self.scen[mask], self.end_idx[mask], self.scenarios)


def _nid(v) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        v = int(v)
    return str(v)


def load_network(ds_dir: Path) -> dict:
    """Tolerant reader for step 1's network.json. Returns nodes, coords, links, step_s."""
    p = ds_dir / "network.json"
    raw = json.loads(p.read_text()) if p.exists() else {}
    nodes, coords = [], {}
    rn = raw.get("nodes", [])
    items = rn.items() if isinstance(rn, dict) else [(None, n) for n in rn]
    for key, n in items:
        if isinstance(n, dict):
            nid = _nid(key if key is not None else n.get("id", n.get("name")))
            xy = n.get("coordinates", n.get("coords"))
            if xy is None and "x" in n and "y" in n:
                xy = (n["x"], n["y"])
            if xy is not None:
                coords[nid] = (float(xy[0]), float(xy[1]))
        else:
            nid = _nid(key if key is not None else n)
        nodes.append(nid)
    links = []
    rl = raw.get("links", [])
    for l in (rl.values() if isinstance(rl, dict) else rl):
        if isinstance(l, dict):
            for a, b in (("start", "end"), ("start_node", "end_node"), ("from", "to"), ("node1", "node2")):
                if a in l and b in l:
                    links.append((_nid(l[a]), _nid(l[b])))
                    break
    step_s = None
    for k in ("sampling_interval_s", "sampling_interval_seconds", "sampling_interval", "step_seconds"):
        if k in raw:
            v = raw[k]
            step_s = float(v) if isinstance(v, (int, float)) else pd.Timedelta(v).total_seconds()
            break
    return {"nodes": nodes, "coords": coords, "links": links, "step_s": step_s}


def stable_split(ds: str, sid: str, seed: int = 0) -> str:
    """train / val / test (70/15/15) from a hash of the dataset and scenario id.

    Each scenario's part depends only on its own name, so adding scenarios later
    never moves an existing one to another part.
    """
    h = hashlib.sha256(f"{seed}:{ds}:{sid}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2 ** 64
    return "train" if u < 0.70 else "val" if u < 0.85 else "test"


def read_splits(root: Path, ds: str, ids: list, seed: int) -> dict:
    """Step 1's splits.json (per dataset folder, or one at the root). Falls back to stable_split."""
    known = set(ids)
    for p in (root / ds / "splits.json", root / "splits.json"):
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        s = s.get(ds, s)
        if "train" in s:
            out = {}
            for k in ("train", "val", "test"):
                v = s.get(k, [])
                v = v.get(ds, []) if isinstance(v, dict) else v
                out[k] = [str(i) for i in v if str(i) in known]
            if any(out.values()):
                return out
    out = {"train": [], "val": [], "test": []}
    for sid in sorted(ids):
        out[stable_split(ds, sid, seed)].append(sid)
    return out


def _prep_scenario(path: Path, sensors: list, node_index: dict, step_s: float, baseline_steps: int, max_gap: int):
    """One parquet -> (residual [T, C] with NaN where unusable, active [T], node [T])."""
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    grid = pd.date_range(df.index[0], df.index[-1], freq=pd.Timedelta(seconds=step_s))
    if not df.index.equals(grid):
        df = df.reindex(grid, method="nearest", tolerance=pd.Timedelta(seconds=step_s / 2))
    x = df.reindex(columns=sensors).astype("float64").ffill(limit=max_gap)
    resid = (x - x.rolling(baseline_steps, min_periods=1).median()).to_numpy()
    active = pd.to_numeric(df["leak_active"], errors="coerce").ffill().fillna(0).to_numpy().astype(np.int64)
    if "leak_node" in df:
        node = np.array([node_index.get(_nid(v), -1) if pd.notna(v) else -1 for v in df["leak_node"]], dtype=np.int64)
    else:
        node = np.full(len(df), -1, dtype=np.int64)
    node[active == 0] = -1
    return resid, active, node


def prep_scenario_job(args):
    """_prep_scenario for a process pool; residuals come back as float32."""
    resid, active, node = _prep_scenario(*args)
    return resid.astype(np.float32), active, node


def _events(active: np.ndarray) -> list:
    d = np.diff(np.concatenate([[0], active, [0]]))
    return [(int(s), int(e) - 1) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def _sensor_kind(name: str) -> str:
    return name.split("_", 1)[0]


def make_datasets(cfg) -> dict:
    """Returns {"train","val","test": WindowSet, "holdout": {ds: WindowSet}, "meta": {...}}."""
    root = Path(cfg.data_root)
    all_ds = list(cfg.datasets) + [d for d in cfg.holdout_datasets if d not in cfg.datasets]

    # ---- pass 1: discover sensors, networks, splits -------------------------------------------
    info = {}
    for ds in all_ds:
        files = {p.stem: p for p in sorted((root / ds).glob("*.parquet"))}
        if not files:
            raise FileNotFoundError(f"no parquet scenarios in {root / ds}")
        first = pd.read_parquet(next(iter(files.values())))
        sensors = [c for c in first.columns if c.startswith(("p_", "q_"))]
        if cfg.sensor_subset != "all":
            sensors = [c for c in sensors if c in set(cfg.sensor_subset)]
        if not sensors:
            raise ValueError(f"{ds}: no sensor columns left after sensor_subset")
        net = load_network(root / ds)
        if net["step_s"] is None:
            net["step_s"] = pd.to_datetime(first["timestamp"]).diff().median().total_seconds()
        if not net["nodes"]:  # no node list from step 1: fall back to what we can see
            seen = {c[2:] for c in sensors if c.startswith("p_")}
            for p in files.values():
                seen |= {_nid(v) for v in pd.read_parquet(p, columns=["leak_node"])["leak_node"].dropna().unique()}
            net["nodes"] = sorted(seen)
        if ds in cfg.datasets:
            splits = read_splits(root, ds, list(files), cfg.seed)
        else:
            splits = {"train": [], "val": [], "test": list(files)}
        if cfg.max_scenarios:
            splits = {k: v[: cfg.max_scenarios] for k, v in splits.items()}
        info[ds] = {"files": files, "sensors": sensors, "net": net, "splits": splits}

    step0 = info[all_ds[0]]["net"]["step_s"]
    W = cfg.window_len or int(round(cfg.window_hours * 3600 / step0))
    S = cfg.stride or max(1, int(round(cfg.stride_hours * 3600 / step0)))
    for ds in all_ds:
        if info[ds]["net"]["step_s"] != step0:
            warnings.warn(f"{ds} samples every {info[ds]['net']['step_s']}s but windows are {W} steps of {step0}s")
    C = max(len(info[ds]["sensors"]) for ds in all_ds)  # smaller networks are zero-padded

    # ---- pass 2: load scenarios, baseline-subtract ---------------------------------------------
    loaded = {}  # (ds, split) -> list of (sid, resid, active, node)
    for ds in all_ds:
        i = info[ds]
        node_index = {n: k for k, n in enumerate(i["net"]["nodes"])}
        base = max(1, int(round(cfg.baseline_hours * 3600 / i["net"]["step_s"])))
        for split, ids in i["splits"].items():
            loaded[ds, split] = [(sid, *_prep_scenario(i["files"][sid], i["sensors"], node_index, i["net"]["step_s"], base, cfg.max_gap_steps)) for sid in ids]

    # ---- normalisation stats from train scenarios only -----------------------------------------
    stats, pooled = {}, {}
    for ds in cfg.datasets:
        tr = [r for _, r, _, _ in loaded[ds, "train"]]
        if not tr:
            raise ValueError(f"{ds}: no train scenarios")
        cat = np.concatenate(tr)
        mean, std = np.nanmean(cat, axis=0), np.nanstd(cat, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        stats[ds] = {s: {"mean": float(m), "std": float(v)} for s, m, v in zip(info[ds]["sensors"], mean, std)}
        for s, m, v in zip(info[ds]["sensors"], mean, std):
            pooled.setdefault(_sensor_kind(s), []).append((m, v))
    # unseen networks have no train stats: use the average over train sensors of the same kind
    stats["_pooled"] = {k: {"mean": float(np.mean([m for m, _ in v])), "std": float(np.mean([s for _, s in v]))} for k, v in pooled.items()}

    def norm_arrays(ds):
        out = []
        for s in info[ds]["sensors"]:
            st = stats.get(ds, {}).get(s) or stats["_pooled"].get(_sensor_kind(s)) or {"mean": 0.0, "std": 1.0}
            out.append((st["mean"], st["std"]))
        a = np.array(out)
        return a[:, 0], a[:, 1]

    # ---- pass 3: windows ----------------------------------------------------------------------
    def build(pairs) -> WindowSet:
        Xs, ys, ns, sc, en, scenarios = [], [], [], [], [], []
        for ds, split in pairs:
            mean, std = norm_arrays(ds)
            for sid, resid, active, node in loaded[ds, split]:
                scenarios.append({"dataset": ds, "id": sid, "step_s": info[ds]["net"]["step_s"], "n_steps": len(active), "events": _events(active)})
                if len(active) < W:
                    continue
                z = ((resid - mean) / std).astype(np.float32)
                v = sliding_window_view(z, W, axis=0)[::S]  # [n, c, W]
                ends = np.arange(W - 1, len(active), S)
                ok = ~np.isnan(v).any(axis=(1, 2))  # drop windows that still contain a gap
                x = np.zeros((int(ok.sum()), C, W), dtype=np.float32)
                x[:, : v.shape[1]] = v[ok]
                Xs.append(x)
                ys.append(active[ends][ok])
                ns.append(node[ends][ok])
                en.append(ends[ok])
                sc.append(np.full(int(ok.sum()), len(scenarios) - 1))
        if not Xs:
            return WindowSet(np.zeros((0, C, W), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), scenarios)
        return WindowSet(np.concatenate(Xs), np.concatenate(ys), np.concatenate(ns), np.concatenate(sc), np.concatenate(en), scenarios)

    out = {k: build([(ds, k) for ds in cfg.datasets]) for k in ("train", "val", "test")}
    out["holdout"] = {ds: build([(ds, "test")]) for ds in cfg.holdout_datasets}
    out["meta"] = {
        "n_channels": C, "window_len": W, "stride": S, "norm_stats": stats,
        "networks": {ds: info[ds]["net"] for ds in all_ds},
        "sensors": {ds: info[ds]["sensors"] for ds in all_ds},
    }
    return out
