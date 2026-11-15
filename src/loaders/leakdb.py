"""LeakDB (Net1 / Hanoi) raw scenarios -> unified format.

Raw layout per scenario (verified on the downloaded data):
    Scenario-<i>/Pressures/Node_<id>.csv   m, columns Index,Value
    Scenario-<i>/Flows/Link_<id>.csv       m3/h
    Scenario-<i>/Demands/Node_<id>.csv     m3/h
    Scenario-<i>/Labels.csv                0/1 per step
    Scenario-<i>/Timestamps.csv            30-min steps, one year (17520 rows)
    Scenario-<i>/Leaks/Leak_<node>_info.csv  node, diameter (m), type, start/end/peak (1-based steps)
    Scenario-<i>/<Net>_CMH_Scenario-<i>.inp
The Leaks folder is empty for no-leak scenarios.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd

from ..schema import RAW_DIR, leak_labels, rel, write_network_json, write_scenario

SOURCE = "LeakDB, KIOS Research (https://github.com/KIOS-Research/LeakDB), EUPL-1.2"
NOTES = [
    "Every node and link is a sensor in LeakDB, far more than a real utility has. Subsample for realism.",
    "leak_active comes from Labels.csv. leak_node / leak_size / leak_type come from Leaks/*_info.csv.",
    "leak_size is the nominal leak diameter; for incipient leaks it is a linear ramp reconstructed from "
    "start and peak time, not a measured value.",
    "Each scenario has its own .inp with perturbed pipe parameters; topology is identical across scenarios.",
]


def _values(folder: Path, prefix: str) -> dict:
    out = {}
    for f in sorted(folder.glob("*.csv"), key=lambda p: p.stem):
        out[prefix + f.stem.split("_", 1)[1]] = pd.read_csv(f, usecols=["Value"])["Value"].to_numpy()
    return out


def _read_leaks(folder: Path) -> list[dict]:
    leaks = []
    for f in sorted(folder.glob("*_info.csv")):
        info = pd.read_csv(f, sep=r"\s*,\s*", engine="python", index_col=0)["Value"]
        kind = str(info["Leak Type"]).strip().lower()
        # 1-based in the file. Checked against Labels.csv: incipient leaks are labelled from
        # step Start-1 (0-based), abrupt leaks one step later.
        start = int(info["Leak Start"]) - (1 if kind == "incipient" else 0)
        leaks.append({
            "node": str(info["Leak Node"]),
            "link": None,
            "type": kind,
            "diameter": float(info["Leak Diameter"]),
            "start_idx": start,
            "end_idx": int(info["Leak End"]),  # 1-based inclusive end == 0-based exclusive end
            "peak_idx": max(int(info["Peak Time"]) - 1, start),
        })
    return leaks


def convert_scenario(folder: Path, dataset: str) -> tuple[str, dict]:
    """Convert one raw scenario folder. Returns (scenario_id, metadata entry)."""
    scenario_id = f"scenario_{int(folder.name.split('-')[1]):04d}"
    cols = {"timestamp": pd.to_datetime(pd.read_csv(folder / "Timestamps.csv")["Timestamp"])}
    cols |= _values(folder / "Pressures", "p_")
    cols |= _values(folder / "Flows", "q_")
    cols |= _values(folder / "Demands", "demand_")
    df = pd.DataFrame(cols)

    leaks = _read_leaks(folder / "Leaks")
    labels = leak_labels(len(df), leaks)
    labels["leak_active"] = pd.read_csv(folder / "Labels.csv")["Label"].to_numpy().astype("int8")
    write_scenario(pd.concat([df, labels], axis=1), dataset, scenario_id)
    return scenario_id, {"inp_path": rel(next(folder.glob("*.inp"))), "leaks": leaks,
                         "raw": rel(folder)}


def build(net: str, limit: int | None = None, workers: int | None = None) -> None:
    """net: 'net1' or 'hanoi'. Converts every downloaded scenario (or the first `limit`), one process per core."""
    dataset = f"leakdb_{net}"
    folders = sorted((RAW_DIR / dataset).glob("Scenario-*/"), key=lambda p: int(p.name.split("-")[1]))
    folders = [f for f in folders if (f / "Labels.csv").exists()][:limit]
    if not folders:
        raise FileNotFoundError(f"No raw scenarios in {RAW_DIR / dataset}. Run: python -m src.download {dataset}")

    scenarios = {}
    with ProcessPoolExecutor(workers or os.cpu_count()) as pool:
        for i, (sid, entry) in enumerate(pool.map(partial(convert_scenario, dataset=dataset), folders, chunksize=4), 1):
            scenarios[sid] = entry
            if i % 50 == 0 or i == len(folders):
                print(f"{dataset}: {i}/{len(folders)}", flush=True)

    first = folders[0]
    ids = lambda sub: sorted(f.stem.split("_", 1)[1] for f in (first / sub).glob("*.csv"))
    write_network_json(
        dataset, next(first.glob("*.inp")), 1800,
        sensors={"pressure": ids("Pressures"), "flow": ids("Flows"), "demand": ids("Demands"), "level": []},
        scenarios=scenarios, source=SOURCE, notes=NOTES)
