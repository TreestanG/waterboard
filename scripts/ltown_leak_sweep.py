"""Run the detection models on L-TOWN for a range of leak sizes.

Usage (from backend/):  PYTHONPATH=. uv run python ../scripts/ltown_leak_sweep.py L-TOWN.inp out.csv
Leak size is the hole AREA in cm2 (as in the app). Also reported: the equivalent round-hole
diameter and the hole size relative to the pipe it is on.
"""
import copy, csv, math, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

import app.simulate as S
from app.evaluate import evaluate
from app.models import REGISTRY

inp_path, out_path = sys.argv[1], sys.argv[2]
SIZES = [float(a) for a in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(1, 30, 2))
N_PIPES, HOURS, STEP_MIN, START_H = 5, 72.0, 30.0, 36.0

t0 = time.time()
import wntr
WN = wntr.network.WaterNetworkModel(inp_path)
print(f"parsed in {time.time() - t0:.0f}s", flush=True)

# real L-TOWN pressure sensors: tagged ';...PRESSURE SENSOR' in [JUNCTIONS]
sensors, sec = [], None
with open(inp_path, errors="replace") as fh:
    for line in fh:
        s = line.strip()
        if s.startswith("["):
            sec = s.upper()
            if sec == "[PIPES]":
                break
        elif sec == "[JUNCTIONS]" and "PRESSURE SENSOR" in s.upper() and not s.startswith(";"):
            sensors.append(s.split()[0])
print(len(sensors), "pressure sensors", flush=True)

# reuse the parsed network and the no-leak run instead of redoing them for every leak
S._load = lambda _text: copy.deepcopy(WN)
S.choose_sensors = lambda junctions, xy, count, seed=0: sensors
_orig_run, _base = S._run, {}
def _run(wn):
    leaky = any(n.endswith(S.SPLIT_SUFFIX) for n in wn.node_name_list)
    if not leaky and "r" in _base:
        return _base["r"]
    t = time.time(); r = _orig_run(wn); print(f"  hydraulic run {time.time() - t:.0f}s", flush=True)
    if not leaky:
        _base["r"] = r
    return r
S._run = _run

rng = np.random.default_rng(0)
pipes = [str(p) for p in rng.choice(WN.pipe_name_list, N_PIPES, replace=False)]
print("pipes", [(p, round(WN.get_link(p).diameter * 1000)) for p in pipes], flush=True)

def stats(r, lo, hi):
    sc = np.array([np.nan if v is None else v for v in r["score"]], dtype=float); th = np.array(r["timesH"])
    v = sc[(th >= lo) & (th < hi) & ~np.isnan(sc)]
    return (float(v.mean()), float(v.max())) if len(v) else (float("nan"), float("nan"))

rows = []
def one(area, pipe):
    leaks = [] if area is None else [S.LeakSpec("a", pipe, area, START_H)]
    sim = S.simulate("", leaks, duration_h=HOURS, step_min=STEP_MIN)
    row = {"area_cm2": area or 0, "pipe": pipe or "", "reached_h": sim.times_h[-1]}
    if area:
        d_pipe = WN.get_link(pipe).diameter * 100  # cm
        d_hole = 2 * math.sqrt(area / math.pi)
        idx = [sim.junctions.index(s) for s in sim.sensors]
        after = np.array(sim.times_h) >= START_H
        drop = (sim.pressure_baseline - sim.pressure_leak)[after][:, idx]
        row.update(hole_diam_cm=round(d_hole, 2), pipe_diam_cm=round(d_pipe, 1),
                   hole_vs_pipe_diam_pct=round(100 * d_hole / d_pipe, 1),
                   hole_vs_pipe_area_pct=round(100 * area / (math.pi * (d_pipe / 2) ** 2), 1),
                   leak_lps=round(float(np.mean(np.array(sim.leaks[0]["flowLps"])[after])), 2),
                   max_sensor_drop_m=round(float(drop.mean(axis=0).max()), 3))
    for mid in ("lstm_gnn", "baseline"):
        r = evaluate(sim, mid)
        pre, post = stats(r, 24, START_H), stats(r, START_H, HOURS + 1)
        row.update({f"{mid}_pre_mean": round(pre[0], 3), f"{mid}_pre_max": round(pre[1], 3),
                    f"{mid}_post_mean": round(post[0], 3), f"{mid}_post_max": round(post[1], 3),
                    f"{mid}_threshold": round(r["threshold"], 3), f"{mid}_false_alarms": len(r["falseAlarms"])})
        if area:
            pl = r["perLeak"][0]
            row.update({f"{mid}_detected": pl["detected"], f"{mid}_delay_h": pl["delayHours"], f"{mid}_loc_error": pl["locationError"]})
    rows.append(row)
    print(row, flush=True)
    keys = sorted({k for r_ in rows for k in r_}, key=lambda k: (not k.startswith(("area", "pipe", "hole", "leak", "max", "reached")), k))
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

one(None, None)
for area in SIZES:
    for pipe in pipes:
        t = time.time()
        try:
            one(area, pipe)
        except Exception as e:
            print(f"FAILED area={area} pipe={pipe}: {e}", flush=True)
        print(f"  {area} cm2 on {pipe}: {time.time() - t:.0f}s", flush=True)
