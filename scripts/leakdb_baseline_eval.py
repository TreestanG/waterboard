"""Score the residual baseline on raw LeakDB Hanoi scenarios.

Usage (from backend/):  PYTHONPATH=. uv run python ../scripts/leakdb_baseline_eval.py ../dataset/training/raw/leakdb_hanoi out.csv

Expected (no-leak) pressures come from simulating each scenario's own .inp, which has that
scenario's demands and no leak (it reproduces LeakDB's no-leak pressures to 0.001 m).
LeakDB pressures have no sensor noise, so Gaussian noise (0.05 m, as in the app) is added.
"""
import csv, os, sys, tempfile, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, wntr

from app.models import ModelInput
from app.models.baseline import ResidualBaseline, THRESHOLD
from app.simulate import choose_sensors

raw, out_path = Path(sys.argv[1]), sys.argv[2]
NOISE, STEP_H = 0.05, 0.5
model = ResidualBaseline()
rows = []
for folder in sorted(raw.glob("Scenario-*"), key=lambda p: int(p.name.split("-")[1])):
    wn = wntr.network.WaterNetworkModel(str(next(folder.glob("*.inp"))))
    with tempfile.TemporaryDirectory() as tmp:
        res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(tmp, "s"))
    junctions = list(wn.junction_name_list)
    xy = {j: tuple(map(float, wn.get_node(j).coordinates)) for j in junctions}
    label = pd.read_csv(folder / "Labels.csv")["Label"].to_numpy()
    T = len(label)
    measured_all = np.stack([pd.read_csv(folder / f"Pressures/Node_{j}.csv")["Value"].to_numpy() for j in junctions], axis=1)
    expected_all = res.node["pressure"][junctions].to_numpy()[:T]
    leaks = []
    for f in sorted((folder / "Leaks").glob("*_info.csv")):
        info = pd.read_csv(f, sep=r"\s*,\s*", engine="python", index_col=0)["Value"]
        leaks.append({"node": str(info["Leak Node"]), "diam_cm": 100 * float(info["Leak Diameter"]), "type": str(info["Leak Type"]).strip(),
                      "start": int(info["Leak Start"]) - 1, "end": int(info["Leak End"])})
    leaks.sort(key=lambda l: l["start"])
    for config, count in (("all 31 junctions", len(junctions)), ("4 sensors", 4)):
        sensors = junctions if count == len(junctions) else choose_sensors(junctions, xy, count)
        idx = [junctions.index(s) for s in sensors]
        rng = np.random.default_rng(int(folder.name.split("-")[1]))
        measured = measured_all[:, idx] + rng.normal(0, NOISE, (T, len(idx)))
        out = model.run(ModelInput(times_h=list(np.arange(T) * STEP_H), sensors=sensors, sensor_xy=np.array([xy[s] for s in sensors]),
                                   measured=measured, expected=expected_all[:, idx], sensor_sigma_m=NOISE,
                                   junctions=junctions, junction_xy=np.array([xy[j] for j in junctions])))
        score = np.array(out.score)
        alarm_steps = [int(round(e.time_h / STEP_H)) for e in out.events]
        used = set()
        det = []
        for l in leaks:  # first unused alarm while this leak is open
            m = next((k for k, a in enumerate(alarm_steps) if k not in used and l["start"] <= a < l["end"]), None)
            if m is not None:
                used.add(m)
            det.append(None if m is None else (alarm_steps[m] - l["start"]) * STEP_H)
        extra = [a for k, a in enumerate(alarm_steps) if k not in used]
        above = score >= THRESHOLD
        rows.append({"scenario": folder.name, "sensors": config, "n_leaks": len(leaks),
                     "leaks": "; ".join(f"node {l['node']} {l['diam_cm']:.1f}cm {l['type']}" for l in leaks),
                     "detected": sum(d is not None for d in det), "delays_h": "; ".join("missed" if d is None else f"{d:g}" for d in det),
                     "repeat_alarms_during_leak": sum(bool(label[a]) for a in extra), "false_alarms_no_leak": sum(not label[a] for a in extra),
                     "step_tp": int((above & (label == 1)).sum()), "step_fp": int((above & (label == 0)).sum()),
                     "step_fn": int((~above & (label == 1)).sum()), "step_tn": int((~above & (label == 0)).sum())})
        print(rows[-1], flush=True)
with open(out_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

d = pd.DataFrame(rows)
f1 = lambda tp, fp, fn: (tp / (tp + fp) if tp + fp else float("nan"), tp / (tp + fn) if tp + fn else float("nan"), 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"))
for cfg, g in d.groupby("sensors"):
    tp, fn = g.detected.sum(), g.n_leaks.sum() - g.detected.sum()
    print(f"\n{cfg}: {len(g)} scenarios, {g.n_leaks.sum()} leaks, {(g.n_leaks == 0).sum()} no-leak scenarios")
    print("  event P/R/F1, repeats ignored      ", [round(v, 3) for v in f1(tp, g.false_alarms_no_leak.sum(), fn)], "false alarms", g.false_alarms_no_leak.sum())
    print("  event P/R/F1, repeats counted false", [round(v, 3) for v in f1(tp, g.false_alarms_no_leak.sum() + g.repeat_alarms_during_leak.sum(), fn)], "repeats", g.repeat_alarms_during_leak.sum())
    print("  per-step P/R/F1 (score >= threshold)", [round(v, 3) for v in f1(g.step_tp.sum(), g.step_fp.sum(), g.step_fn.sum())])
