"""Experiment: CUSUM detection + signature-matching localization on L-TOWN, against the
threshold/centroid baseline.

Usage (from backend/):
  PYTHONPATH=. uv run python ../scripts/physics_experiment.py L-TOWN.inp signatures
  PYTHONPATH=. uv run python ../scripts/physics_experiment.py L-TOWN.inp run

Every run is 96 h at 30-minute steps. Hours 0-48 are leak-free history used for calibration; the
leak starts at hour 60. Sensor noise is 0.05 m. Two conditions:
  perfect      the "measured" network has exactly the model's demands
  demand_err   every junction's demand is off by a random factor (sd 10%, plus a network-wide
               factor, sd 3%) that the hydraulic model does not know about
"""
import csv, math, os, sys, time, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
import numpy as np, wntr, networkx as nx

import app.simulate as S
from app.models import ModelInput
from app.models.baseline import ResidualBaseline
from app.models.physics import cusum_detect, rank_candidates

INP, STAGE = sys.argv[1], sys.argv[2]
CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
OUT = Path(__file__).resolve().parents[1] / "data" / "physics_experiment.csv"
HOURS, STEP_MIN, START_H, CALIB_DAYS, NOISE = 96.0, 30.0, 60.0, 2, 0.05
SPD = int(24 * 60 / STEP_MIN)
SIZES = [0.5, 1, 2, 3, 5, 9, 15, 29]
N_PIPES, N_NOLEAK = 15, 12
SIG_AREA_CM2, SIG_HOURS, SIG_STEP_MIN = 5.0, 24.0, 60.0
LOC_EXTRA_STEPS = 12  # use up to 6 h of readings after the alarm to localize
WORKERS = 8  # each worker holds its own copy of the network (~1.3 GB for L-TOWN)

WN = None
def init():
    global WN
    WN = wntr.network.WaterNetworkModel(INP)

def configure(wn, hours, step_min):
    S._configure(wn, hours, step_min)

def real_sensors():
    out, sec = [], None
    with open(INP, errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("["):
                sec = s.upper()
                if sec == "[PIPES]":
                    break
            elif sec == "[JUNCTIONS]" and "PRESSURE SENSOR" in s.upper() and not s.startswith(";"):
                out.append(s.split()[0])
    return out

# ---------------------------------------------------------------- signatures
def sig_base(_):
    configure(WN, SIG_HOURS, SIG_STEP_MIN)
    r, _w = S._run(WN); WN.reset_initial_values()
    return r.node["pressure"][WN.junction_name_list].to_numpy()

def sig_one(j):
    configure(WN, SIG_HOURS, SIG_STEP_MIN)
    node = WN.get_node(j)
    try:
        node.add_leak(WN, area=SIG_AREA_CM2 / 1e4, start_time=0)
        r, _w = S._run(WN)
        p = r.node["pressure"][WN.junction_name_list].to_numpy()
    except Exception as e:
        p = None
    finally:
        node.remove_leak(WN); WN.reset_initial_values()
    return j, p

def stage_signatures():
    init()
    junctions = list(WN.junction_name_list)
    t = time.time()
    with Pool(WORKERS, initializer=init) as pool:
        base = pool.map(sig_base, [0])[0]
        sig = np.full((len(junctions), len(junctions)), np.nan, dtype=np.float32)
        for n, (j, p) in enumerate(pool.imap_unordered(sig_one, junctions, chunksize=4), 1):
            if p is not None and len(p) == len(base):
                sig[junctions.index(j)] = (base - p).mean(axis=0)
            if n % 100 == 0:
                print(f"signatures {n}/{len(junctions)}  {time.time() - t:.0f}s", flush=True)
    np.savez_compressed(CACHE / "ltown_signatures.npz", sig=sig, junctions=np.array(junctions))
    print(f"done in {time.time() - t:.0f}s; {int(np.isnan(sig[:, 0]).sum())} junctions without a signature", flush=True)

# ---------------------------------------------------------------- test runs
G = {}
def init_run():
    init()
    z = np.load(CACHE / "ltown_signatures.npz")
    G["sig"], G["junctions"] = z["sig"], list(z["junctions"])
    G["base"] = np.load(CACHE / "ltown_base96.npy")
    G["xy"] = np.array([WN.get_node(j).coordinates for j in G["junctions"]], dtype=float)
    G["graph"] = nx.Graph(WN.to_graph().to_undirected())
    real = real_sensors()
    order = [G["junctions"].index(s) for s in real]
    dist = np.min(np.linalg.norm(G["xy"][:, None] - G["xy"][order][None], axis=2), axis=1)
    while len(order) < 130:
        nxt = int(np.argmax(dist)); order.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(G["xy"] - G["xy"][nxt], axis=1))
    G["sets"] = {33: order[:33], 130: order[:130]}

def run_one(job):
    cond, area, pipe, seed = job
    rng = np.random.default_rng(seed)
    configure(WN, HOURS, STEP_MIN)
    saved = []
    if cond == "demand_err":
        g = 1 + rng.normal(0, 0.03)
        for _, j in WN.junctions():
            f = max(0.5, g * (1 + rng.normal(0, 0.10)))
            for d in j.demand_timeseries_list:
                saved.append((d, d.base_value)); d.base_value = d.base_value * f
    try:
        if area:
            wn = wntr.morph.split_pipe(WN, pipe, pipe + "__B", pipe + "__leak")
            wn.get_node(pipe + "__leak").add_leak(wn, area=area / 1e4, start_time=int(START_H * 3600))
        else:
            wn = WN
        r, _w = S._run(wn)
        if not area:
            WN.reset_initial_values()
    finally:
        for d, v in saved:
            d.base_value = v
    p = r.node["pressure"][G["junctions"]].to_numpy()
    T = len(G["base"])
    if len(p) < T:
        return [{"condition": cond, "area_cm2": area, "pipe": pipe, "failed": True}]
    start = int(START_H * 60 / STEP_MIN)
    truth = None
    if area:
        link = WN.get_link(pipe)
        ends = [n for n in (link.start_node_name, link.end_node_name) if n in G["junctions"]]
        truth = {"xy": np.array(wn.get_node(pipe + "__leak").coordinates, dtype=float), "ends": ends,
                 "lps": float(r.node["leak_demand"][pipe + "__leak"].to_numpy()[start:].mean() * 1000), "d_pipe": link.diameter * 100}
    noise = rng.normal(0, NOISE, p.shape)
    rows = []
    for count, idx in G["sets"].items():
        sensors = [G["junctions"][i] for i in idx]
        measured, expected = p[:, idx] + noise[:, idx], G["base"][:, idx]
        drop = expected - measured
        row = {"condition": cond, "area_cm2": area or 0, "pipe": pipe or "", "seed": seed, "sensors": count}
        if truth:
            row.update(hole_diam_cm=round(2 * math.sqrt(area / math.pi), 2), hole_vs_pipe_area_pct=round(100 * area / (math.pi * (truth["d_pipe"] / 2) ** 2), 2),
                       leak_lps=round(truth["lps"], 2), max_sensor_drop_m=round(float((G["base"] - p)[start:, idx].mean(axis=0).max()), 3))

        def loc_metrics(prefix, node_i, ranking=None):
            if not truth:
                return
            row[prefix + "_err_units"] = round(float(np.linalg.norm(G["xy"][node_i] - truth["xy"])), 1)
            row[prefix + "_err_pipes"] = min(nx.shortest_path_length(G["graph"], G["junctions"][node_i], e) for e in truth["ends"])
            if ranking is not None:
                pos = {int(j): r_ for r_, j in enumerate(ranking)}
                row[prefix + "_true_rank"] = 1 + min(pos[G["junctions"].index(e)] for e in truth["ends"])

        # --- CUSUM + signature matching
        c = cusum_detect(drop, SPD, CALIB_DAYS)
        a = c["alarm"]
        row["cusum_false_alarm"] = a is not None and a < start
        row["cusum_detected"] = bool(truth) and a is not None and a >= start
        if a is not None:
            row["cusum_alarm_h"] = a * STEP_MIN / 60
            if row["cusum_detected"]:
                row["cusum_delay_h"] = (a - start) * STEP_MIN / 60
                obs = c["excess"][c["onset"] : min(a + LOC_EXTRA_STEPS, T - 1) + 1].mean(axis=0)
                ranking = rank_candidates(obs, G["sig"][:, idx], c["sigma"])
                loc_metrics("sig", int(ranking[0]), ranking)
        # --- existing baseline (threshold on RMS residual, centroid localization)
        b = ResidualBaseline().run(ModelInput(times_h=list(np.arange(T) * STEP_MIN / 60), sensors=sensors, sensor_xy=G["xy"][idx],
                                             measured=measured, expected=expected, sensor_sigma_m=NOISE,
                                             junctions=G["junctions"], junction_xy=G["xy"]))
        ev = [(int(round(e.time_h * 60 / STEP_MIN)), e.node) for e in b.events]
        row["base_false_alarm"] = any(t < start for t, _ in ev)
        hit = next(((t, n) for t, n in ev if t >= start), None) if truth else None
        row["base_detected"] = hit is not None
        if hit:
            row["base_delay_h"] = (hit[0] - start) * STEP_MIN / 60
            loc_metrics("base", G["junctions"].index(hit[1]))
        rows.append(row)
    return rows

def stage_run():
    init()
    configure(WN, HOURS, STEP_MIN)
    r, _w = S._run(WN); WN.reset_initial_values()
    np.save(CACHE / "ltown_base96.npy", r.node["pressure"][WN.junction_name_list].to_numpy())
    rng = np.random.default_rng(7)
    pipes = [str(p) for p in rng.choice(WN.pipe_name_list, N_PIPES, replace=False)]
    jobs, seed = [], 1000
    for cond in ("perfect", "demand_err"):
        for k in range(N_NOLEAK):
            seed += 1; jobs.append((cond, None, None, seed))
        for area in SIZES:
            for pipe in pipes:
                seed += 1; jobs.append((cond, area, pipe, seed))
    print(len(jobs), "runs on", WORKERS, "workers; pipes", pipes, flush=True)
    rows, t = [], time.time()
    with Pool(WORKERS, initializer=init_run) as pool:
        for n, out in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows += out
            if n % 20 == 0:
                print(f"{n}/{len(jobs)}  {time.time() - t:.0f}s", flush=True)
    keys = list(dict.fromkeys(k for r_ in rows for k in r_))
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote", OUT, flush=True)

if __name__ == "__main__":
    stage_signatures() if STAGE == "signatures" else stage_run()
