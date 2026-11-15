"""L-Town / BattLeDIM raw data -> unified format. Two scenarios: "2018" and "2019", both test-only.

Raw files (Zenodo record 4017659, CC-BY-4.0), all ';' separated with decimal commas, 5-min steps:
    <year>_SCADA_Pressures.csv   33 nodes, m
    <year>_SCADA_Flows.csv       3 links, m3/h
    <year>_SCADA_Levels.csv      1 tank, m
    <year>_SCADA_Demands.csv     82 AMRs, L/h (converted to m3/h here)
    <year>_Leakages.csv          leak outflow per leaky pipe, m3/h (ground truth, both years are public)
    dataset_configuration.yaml   per leak: pipe, start, end, diameter (m), type, peak time

Labels: leaks are located on pipes, not nodes. leak_link is the pipe id, leak_node is the pipe's
start node. leak_active is 1 whenever any pipe in <year>_Leakages.csv has outflow > 0.
"""
import pandas as pd
import yaml

from ..schema import RAW_DIR, leak_labels, rel, write_network_json, write_scenario

DATASET = "ltown"
RAW = RAW_DIR / DATASET
SOURCE = "BattLeDIM 2020, KIOS Research (https://zenodo.org/records/4017659), CC-BY-4.0"
STEP = pd.Timedelta(minutes=5)
NOTES = [
    "Test / inference only: neither year is used for training (enforced in src/splits.py). "
    "2018 is BattLeDIM's historical set, 2019 its evaluation set.",
    "Leak start times, diameters and types come from dataset_configuration.yaml, which was published "
    "after the competition. 2018_Fixed_Leakages_Report.txt (what a utility would actually know) only "
    "gives repair times of 10 of the 14 leaks in 2018.",
    "Several small background leaks are never repaired (p257, p427, p654, p810 run from 2018 to the end "
    "of 2019), so leak_active is 1 for most of the data. Use leak_count / leak_size / the per-leak list "
    "in scenarios.<year>.leaks to target new or large leaks instead of leak_active alone.",
    "leak_node is the start node of the leaking pipe (leak_link). The data itself is simulated on "
    "L-TOWN_Real.inp, which differs from the nominal L-TOWN.inp referenced here.",
    "Demand columns are AMR readings converted from L/h to m3/h.",
]


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / name, sep=";", decimal=",", index_col=0, parse_dates=True)


def _leaks(config: dict, pipes: dict, year: int, index: pd.DatetimeIndex) -> list[dict]:
    leaks = []
    for row in config["leakages"]:
        if not isinstance(row, str):
            continue  # the first list item is a comment that yaml parses as None
        link, start, end, diameter, kind, peak = [v.strip() for v in row.split(",")]
        start, end, peak = pd.Timestamp(start), pd.Timestamp(end), pd.Timestamp(peak)
        if end < index[0] or start > index[-1]:
            continue
        leaks.append({
            "node": pipes[link]["start"], "link": link, "type": kind, "diameter": float(diameter),
            # negative start_idx = already leaking when the year begins
            "start_idx": int((start - index[0]) / STEP),
            "end_idx": int((end - index[0]) / STEP) + 1,
            "peak_idx": int((peak - index[0]) / STEP),
            "start_time": str(start), "end_time": str(end), "peak_time": str(peak),
        })
    return leaks


def _leak_labels_with_history(n: int, leaks: list[dict]) -> pd.DataFrame:
    """leak_labels, but leaks that began in a previous year keep their true ramp position."""
    offset = -min([l["start_idx"] for l in leaks] + [0])
    shifted = [l | {k: l[k] + offset for k in ("start_idx", "end_idx", "peak_idx")} for l in leaks]
    return leak_labels(n + offset, shifted).iloc[offset:].reset_index(drop=True)


def build() -> None:
    if not (RAW / "2018_SCADA_Pressures.csv").exists():
        raise FileNotFoundError(f"No raw data in {RAW}. Run: python -m src.download ltown")
    config = yaml.safe_load((RAW / "dataset_configuration.yaml").read_text())
    inp = RAW / "L-TOWN.inp"

    import wntr
    wn = wntr.network.WaterNetworkModel(str(inp))
    pipes = {name: {"start": l.start_node_name} for name, l in wn.links()}

    scenarios = {}
    for year in (2018, 2019):
        p = _read(f"{year}_SCADA_Pressures.csv").add_prefix("p_")
        q = _read(f"{year}_SCADA_Flows.csv").add_prefix("q_")
        lvl = _read(f"{year}_SCADA_Levels.csv").add_prefix("level_")
        dem = (_read(f"{year}_SCADA_Demands.csv") / 1000.0).add_prefix("demand_")  # L/h -> m3/h
        df = pd.concat([p, q, lvl, dem], axis=1)
        df.index.name = "timestamp"

        leaks = _leaks(config, pipes, year, df.index)
        labels = _leak_labels_with_history(len(df), leaks)
        outflow = _read(f"{year}_Leakages.csv").reindex(df.index)
        labels["leak_active"] = (outflow > 0).any(axis=1).to_numpy().astype("int8")

        write_scenario(pd.concat([df.reset_index(), labels], axis=1), DATASET, str(year))
        scenarios[str(year)] = {"inp_path": rel(inp), "leaks": leaks, "split": "test"}
        print(f"ltown: {year} done ({len(df)} rows)", flush=True)

    write_network_json(
        DATASET, inp, 300,
        sensors={"pressure": config["pressure_sensors"], "flow": config["flow_sensors"],
                 "demand": config["amrs"], "level": config["level_sensors"]},
        scenarios=scenarios, source=SOURCE, notes=NOTES)
