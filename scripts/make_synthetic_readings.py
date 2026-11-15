"""Build a synthetic readings.csv for an EPANET model.

Sensor locations are taken from the trailing `;...SENSOR` comments in the .inp
(L-TOWN tags them this way). The no-leak model is simulated with WNTR/EPANET and
Gaussian sensor noise is added, so the values are SIMULATED, not measured.

Usage: python make_synthetic_readings.py network.inp out.csv [--start 2019-01-01]
Columns: timestamp, then one per sensor named by its node ID (pressure, m)
or link ID (flow, model flow units).
"""
import argparse
import os
import tempfile

import numpy as np
import pandas as pd
import wntr

NODE_SECTIONS = {"[JUNCTIONS]"}
LINK_SECTIONS = {"[PIPES]", "[PUMPS]", "[VALVES]"}


def tagged_sensors(inp_path):
    """Return (pressure_node_ids, flow_link_ids) from ';... SENSOR' comments."""
    pressure, flow = [], []
    section = None
    with open(inp_path, errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("["):
                section = s.upper()
                if section == "[DEMANDS]":  # everything we need is above this
                    break
                continue
            if not s or s.startswith(";") or ";" not in s:
                continue
            body, comment = s.split(";", 1)
            comment = comment.upper()
            if not body.split():
                continue
            ident = body.split()[0]
            if section in NODE_SECTIONS and "PRESSURE SENSOR" in comment:
                pressure.append(ident)
            elif section in LINK_SECTIONS and "FLOW SENSOR" in comment:
                flow.append(ident)
    return pressure, flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--start", default="2019-01-01 00:00:00")
    ap.add_argument("--pressure-noise", type=float, default=0.05, help="sigma, m")
    ap.add_argument("--flow-noise", type=float, default=0.01, help="sigma, fraction of reading")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pressure_ids, flow_ids = tagged_sensors(args.inp)
    if not pressure_ids and not flow_ids:
        raise SystemExit("No ';PRESSURE SENSOR' / ';FLOW SENSOR' tags found in the .inp")
    print(f"{len(pressure_ids)} pressure sensors, {len(flow_ids)} flow sensors")

    wn = wntr.network.WaterNetworkModel(args.inp)
    with tempfile.TemporaryDirectory() as tmp:
        res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(tmp, "sim"))

    p = res.node["pressure"][pressure_ids].astype(float)  # m
    q = res.link["flowrate"][flow_ids].astype(float)  # m^3/s (WNTR is SI)
    # back to the model's own flow units so the CSV matches what EPANET shows
    units = wn.options.hydraulic.inpfile_units.upper()
    per_m3s = {"CMH": 3600.0, "CMD": 86400.0, "LPS": 1000.0, "LPM": 60000.0,
               "MLD": 86.4, "CMS": 1.0}.get(units)
    if per_m3s is None:
        raise SystemExit(f"Unhandled flow units {units}")
    q = q * per_m3s

    rng = np.random.default_rng(args.seed)
    p = p + rng.normal(0, args.pressure_noise, p.shape)
    q = q * (1 + rng.normal(0, args.flow_noise, q.shape))

    df = pd.concat([p.round(2), q.round(2)], axis=1)
    df.insert(0, "timestamp",
              (pd.Timestamp(args.start) + pd.to_timedelta(df.index, unit="s"))
              .strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows x {df.shape[1] - 1} sensors, flow in {units}")


if __name__ == "__main__":
    main()
