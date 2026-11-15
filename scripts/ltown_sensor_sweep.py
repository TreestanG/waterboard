"""Baseline detector on L-TOWN: leak size x number of pressure sensors.

Usage (from backend/):  PYTHONPATH=. uv run python ../scripts/ltown_sensor_sweep.py L-TOWN.inp out.csv
Each leak is simulated once (72 h, 30-min steps, leak at 36 h); the same run is then read
through 33 (L-TOWN's real sensor sites), 66, 130 and 260 sensors. Extra sensors are added to
the real 33 by farthest-point sampling, i.e. spread as evenly as possible.
"""
import copy, csv, dataclasses, math, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, wntr

import app.simulate as S
from app.evaluate import evaluate

inp_path, out_path = sys.argv[1], sys.argv[2]
SIZES = list(range(1, 30, 2))
COUNTS = [33, 66, 130, 260]
PIPES = ["p575", "p462", "p244", "p279", "p767"]  # the five from the first sweep
HOURS, STEP_MIN, START_H = 72.0, 30.0, 36.0

WN = wntr.network.WaterNetworkModel(inp_path)
real, sec = [], None
with open(inp_path, errors="replace") as fh:
    for line in fh:
        s = line.strip()
        if s.startswith("["):
            sec = s.upper()
            if sec == "[PIPES]":
                break
        elif sec == "[JUNCTIONS]" and "PRESSURE SENSOR" in s.upper() and not s.startswith(";"):
            real.append(s.split()[0])
rng = np.random.default_rng(1)
PIPES += [str(p) for p in rng.choice([p for p in WN.pipe_name_list if p not in PIPES], 5, replace=False)]

junctions = list(WN.junction_name_list)
pts = np.array([WN.get_node(j).coordinates for j in junctions], dtype=float)
order = [junctions.index(s) for s in real]
dist = np.min(np.linalg.norm(pts[:, None] - pts[order][None], axis=2), axis=1)
while len(order) < max(COUNTS):
    nxt = int(np.argmax(dist)); order.append(nxt)
    dist = np.minimum(dist, np.linalg.norm(pts - pts[nxt], axis=1))
sensor_sets = {c: [junctions[i] for i in order[:c]] for c in COUNTS}

S._load = lambda _t: copy.deepcopy(WN)
S.choose_sensors = lambda j, xy, count, seed=0: list(j)  # measure everywhere, subset later
_orig, _base = S._run, {}
def _run(wn):
    leaky = any(n.endswith(S.SPLIT_SUFFIX) for n in wn.node_name_list)
    if not leaky and "r" in _base:
        return _base["r"]
    r = _orig(wn)
    if not leaky:
        _base["r"] = r
    return r
S._run = _run

rows = []
def one(area, pipe):
    sim = S.simulate("", [] if area is None else [S.LeakSpec("a", pipe, area, START_H)], duration_h=HOURS, step_min=STEP_MIN)
    after = np.array(sim.times_h) >= START_H
    for c, sensors in sensor_sets.items():
        idx = [sim.junctions.index(s) for s in sensors]
        r = evaluate(dataclasses.replace(sim, sensors=sensors, measured=sim.measured[:, idx]), "baseline")
        row = {"area_cm2": area or 0, "pipe": pipe or "", "sensors": c, "alarms_total": len(r["events"]), "unmatched_alarms": len(r["falseAlarms"])}
        if area:
            d_pipe = WN.get_link(pipe).diameter * 100
            pl = r["perLeak"][0]
            drop = (sim.pressure_baseline - sim.pressure_leak)[after][:, idx].mean(axis=0)
            row.update(hole_diam_cm=round(2 * math.sqrt(area / math.pi), 2), pipe_diam_cm=round(d_pipe, 1),
                       hole_vs_pipe_area_pct=round(100 * area / (math.pi * (d_pipe / 2) ** 2), 1),
                       leak_lps=round(float(np.mean(np.array(sim.leaks[0]["flowLps"])[after])), 2),
                       max_sensor_drop_m=round(float(drop.max()), 3),
                       detected=pl["detected"], delay_h=pl["delayHours"], loc_error=pl["locationError"])
        rows.append(row)
    with open(out_path, "w", newline="") as fh:
        keys = list(dict.fromkeys(k for r_ in rows for k in r_))
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

one(None, None)
print("no-leak alarms by sensor count:", {r["sensors"]: r["alarms_total"] for r in rows}, flush=True)
print("pipes", [(p, round(WN.get_link(p).diameter * 1000)) for p in PIPES], flush=True)
for area in SIZES:
    for pipe in PIPES:
        t = time.time()
        try:
            one(area, pipe)
        except Exception as e:
            print(f"FAILED {area} {pipe}: {e}", flush=True)
        print(f"{area} cm2 on {pipe}: {time.time() - t:.0f}s", flush=True)
