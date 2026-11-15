"""Unified scenario format shared by every dataset.

One parquet per scenario: dataset/training/processed/<dataset>/<scenario_id>.parquet
One metadata file per dataset: dataset/training/processed/<dataset>/network.json

Columns
    timestamp         datetime64[ns]
    p_<node>          pressure at a sensor node, m
    q_<link>          flow at a sensor link, m3/h
    demand_<node>     metered demand, m3/h (optional)
    level_<tank>      tank level, m (optional)
    leak_active       0/1
    leak_node         node id of the dominant active leak, null when no leak
    leak_link         pipe id of the dominant active leak, null when unknown or no leak
    leak_size         leak diameter in m at this step (ramps up for incipient leaks), null when no leak
    leak_type         "abrupt" / "incipient" / null
    leak_count        number of leaks active at this step

"Dominant" means the largest current diameter when several leaks overlap. The full list of
leaks per scenario is kept in network.json under scenarios.<id>.leaks.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "dataset" / "training"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

UNITS = {"pressure": "m", "flow": "m3/h", "demand": "m3/h", "level": "m", "leak_size": "m (diameter)"}
LABEL_COLUMNS = ["leak_active", "leak_node", "leak_link", "leak_size", "leak_type", "leak_count"]
SENSOR_PREFIXES = ("p_", "q_", "demand_", "level_")


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(SENSOR_PREFIXES)]


def leak_labels(n_steps: int, leaks: list[dict]) -> pd.DataFrame:
    """Build the label columns from a list of leaks.

    Each leak: {node, link, type, start_idx, end_idx (exclusive), peak_idx, diameter}.
    Incipient leaks grow linearly in diameter from start_idx to peak_idx (this is how the
    LeakDB / BattLeDIM generator defines them), abrupt leaks are at full diameter at once.
    """
    size = np.zeros(n_steps, dtype=np.float32)
    count = np.zeros(n_steps, dtype=np.int8)
    node = np.full(n_steps, None, dtype=object)
    link = np.full(n_steps, None, dtype=object)
    kind = np.full(n_steps, None, dtype=object)
    for leak in leaks:
        s, e = max(leak["start_idx"], 0), min(leak["end_idx"], n_steps)
        if e <= s:
            continue
        d = np.full(e - s, leak["diameter"], dtype=np.float32)
        ramp = min(leak.get("peak_idx", s), e) - s
        if leak["type"] == "incipient" and ramp > 0:
            d[:ramp] = leak["diameter"] * np.arange(1, ramp + 1) / (ramp + 1)
        count[s:e] += 1
        bigger = d > size[s:e]
        idx = np.arange(s, e)[bigger]
        size[idx] = d[bigger]
        node[idx] = leak.get("node")
        link[idx] = leak.get("link")
        kind[idx] = leak["type"]
    active = count > 0
    return pd.DataFrame({
        "leak_active": active.astype(np.int8),
        "leak_node": node,
        "leak_link": link,
        "leak_size": np.where(active, size, np.nan).astype(np.float32),
        "leak_type": kind,
        "leak_count": count,
    })


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce column order and dtypes of the unified format."""
    sensors = sensor_columns(df)
    df = df[["timestamp"] + sensors + LABEL_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df[sensors] = df[sensors].astype(np.float32)
    df["leak_active"] = df["leak_active"].astype(np.int8)
    df["leak_count"] = df["leak_count"].astype(np.int8)
    df["leak_size"] = df["leak_size"].astype(np.float32)
    for c in ("leak_node", "leak_link", "leak_type"):
        df[c] = df[c].astype("string")
    return df.reset_index(drop=True)


def write_scenario(df: pd.DataFrame, dataset: str, scenario_id: str) -> Path:
    out = PROCESSED_DIR / dataset / f"{scenario_id}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    finalize(df).to_parquet(out, index=False)
    return out


def rel(path) -> str:
    """Path relative to the repo root, with forward slashes, for metadata."""
    return Path(path).resolve().relative_to(REPO_DIR).as_posix()


def network_topology(inp_path) -> dict:
    """Nodes and links (with coordinates) read from an .inp file."""
    import wntr

    wn = wntr.network.WaterNetworkModel(str(inp_path))
    nodes = {}
    for name, n in wn.nodes():
        x, y = n.coordinates
        nodes[name] = {"type": n.node_type, "x": float(x), "y": float(y),
                       "elevation_m": float(getattr(n, "elevation", 0.0) or 0.0),
                       "base_demand_m3s": float(getattr(n, "base_demand", 0.0) or 0.0)}
    links = {}
    for name, l in wn.links():
        links[name] = {"type": l.link_type, "start": l.start_node_name, "end": l.end_node_name}
        if l.link_type == "Pipe":
            links[name]["length_m"] = float(l.length)
            links[name]["diameter_m"] = float(l.diameter)
    return {"nodes": nodes, "links": links}


def write_network_json(dataset: str, inp_path, sampling_interval_s: int, sensors: dict,
                       scenarios: dict, source: str, notes: list[str] = ()) -> Path:
    meta = {
        "dataset": dataset,
        "source": source,
        "inp_path": rel(inp_path),
        "sampling_interval_s": int(sampling_interval_s),
        "units": UNITS,
        "sensors": sensors,
        "scenarios": scenarios,
        "notes": list(notes),
        **network_topology(inp_path),
    }
    out = PROCESSED_DIR / dataset / "network.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=1))
    return out


def refresh_topology(dataset: str) -> Path:
    """Rewrite nodes and links of an existing network.json from its .inp (scenarios untouched)."""
    out = PROCESSED_DIR / dataset / "network.json"
    meta = json.loads(out.read_text())
    meta.update(network_topology(REPO_DIR / meta["inp_path"]))
    out.write_text(json.dumps(meta, indent=1))
    return out


class Dataset:
    """Processed dataset: metadata plus lazily loaded scenario frames."""

    def __init__(self, name: str):
        self.name = name
        self.dir = PROCESSED_DIR / name
        meta_file = self.dir / "network.json"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"{meta_file} not found. Build it first: python -m src.build {name}")
        self.meta = json.loads(meta_file.read_text())
        self.scenario_ids = sorted(p.stem for p in self.dir.glob("*.parquet"))

    def load(self, scenario_id: str) -> pd.DataFrame:
        return pd.read_parquet(self.dir / f"{scenario_id}.parquet")

    def __len__(self) -> int:
        return len(self.scenario_ids)

    def __iter__(self):
        for sid in self.scenario_ids:
            yield sid, self.load(sid)

    def __repr__(self) -> str:
        return f"Dataset({self.name!r}, {len(self)} scenarios)"


def load_dataset(name: str) -> Dataset:
    """Load any processed dataset (leakdb_net1, leakdb_hanoi, ltown, sim_<network>, ...)."""
    return Dataset(name)
