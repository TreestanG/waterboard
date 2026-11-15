"""Sanity checks on processed datasets.

Usage:
    python -m src.checks leakdb_net1 leakdb_hanoi ltown sim_net3
    python -m src.checks --compare sim_net1 leakdb_net1

Prints a summary per dataset and saves one plot per dataset to notebooks/figures/.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .schema import REPO_DIR, load_dataset, sensor_columns

FIG_DIR = REPO_DIR / "notebooks" / "figures"
DAY = pd.Timedelta(days=1)


def check_scenario(df: pd.DataFrame, step_s: int) -> dict:
    gaps = df["timestamp"].diff().dropna().dt.total_seconds() != step_s
    return {
        "rows": len(df),
        "nan_sensor_values": int(df[sensor_columns(df)].isna().sum().sum()),
        "timestamp_gaps": int(gaps.sum()),
        "leak_fraction": float(df["leak_active"].mean()),
        "leak_steps": int(df["leak_active"].sum()),
        # leak_active (from the dataset's own labels) vs leak windows rebuilt from the leak info
        "label_mismatch_steps": int((df["leak_active"] != (df["leak_count"] > 0)).sum()),
    }


def leak_windows(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    a = df["leak_active"].to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, a, 0]))
    t = df["timestamp"].to_numpy()
    return [(pd.Timestamp(t[s]), pd.Timestamp(t[e - 1])) for s, e in zip(edges[::2], edges[1::2])]


def plot_leak(ds, scenario_id: str, leak: dict, out_name: str):
    """Pressure at the most affected sensor around one leak, leak window shaded."""
    df = ds.load(scenario_id)
    t0 = df["timestamp"].iloc[0]
    step = pd.Timedelta(seconds=ds.meta["sampling_interval_s"])
    start, end, peak = (t0 + leak[k] * step for k in ("start_idx", "end_idx", "peak_idx"))
    view = df[(df["timestamp"] >= start - 3 * DAY) & (df["timestamp"] <= min(end, peak + 2 * DAY) + 2 * DAY)]
    before = view[view["timestamp"] < start]
    after = view[(view["timestamp"] >= peak) & (view["timestamp"] < min(end, peak + 2 * DAY))]
    p_cols = [c for c in df.columns if c.startswith("p_")]
    drop = (before[p_cols].mean() - after[p_cols].mean()).sort_values(ascending=False)
    col = drop.index[0]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(view["timestamp"], view[col], lw=0.8, label=f"{col} (m)")
    ax.plot(view["timestamp"], view[col].rolling(int(DAY / step), min_periods=1).mean(), lw=1.5,
            label="24 h rolling mean")
    ax.axvspan(start, min(end, view["timestamp"].iloc[-1]), color="red", alpha=0.15, label="leak active")
    where = leak.get("link") or leak["node"]
    ax.set_title(f"{ds.name} / {scenario_id}: {leak['type']} leak at {where}, "
                 f"d={leak['diameter'] * 1000:.0f} mm, mean drop {drop.iloc[0]:.2f} m")
    ax.set_ylabel("pressure (m)")
    ax.legend(loc="lower left", fontsize=8)
    fig.autofmt_xdate()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out, float(drop.iloc[0])


def check_dataset(name: str, plot: bool = True) -> pd.DataFrame:
    ds = load_dataset(name)
    step_s = ds.meta["sampling_interval_s"]
    rows = {sid: check_scenario(df, step_s) for sid, df in ds}
    report = pd.DataFrame.from_dict(rows, orient="index")

    s = ds.meta["sensors"]
    print(f"\n== {name}: {len(ds)} scenarios, step {step_s} s, sensors: {len(s['pressure'])} pressure, "
          f"{len(s['flow'])} flow, {len(s['demand'])} demand, {len(s['level'])} level")
    print(f"   scenarios with no leak: {(report.leak_steps == 0).mean():.0%}")
    print(f"   leak-active fraction: mean {report.leak_fraction.mean():.3f}, max {report.leak_fraction.max():.3f}")
    print(f"   NaN sensor values: {report.nan_sensor_values.sum()}, timestamp gaps: {report.timestamp_gaps.sum()}, "
          f"label mismatch steps: {report.label_mismatch_steps.sum()}")
    types = pd.Series([l["type"] for sc in ds.meta["scenarios"].values() for l in sc["leaks"]]).value_counts()
    print(f"   leaks by type: {types.to_dict()}")

    if plot:
        # largest abrupt leak if there is one (clearest drop), else the largest leak
        leaks = [(sid, l) for sid, sc in ds.meta["scenarios"].items() for l in sc["leaks"]
                 if sid in rows and l["start_idx"] > 0]
        if leaks:
            abrupt = [x for x in leaks if x[1]["type"] == "abrupt"]
            sid, leak = max(abrupt or leaks, key=lambda x: x[1]["diameter"])
            out, drop = plot_leak(ds, sid, leak, f"sanity_{name}.png")
            print(f"   plot: {out.relative_to(REPO_DIR).as_posix()} (pressure drop {drop:.2f} m)")
    return report


def compare(sim_name: str, real_name: str) -> pd.DataFrame:
    """Compare per-sensor statistics of a WNTR-simulated dataset with a real one on the same network."""
    def stats(name):
        frames = [df.loc[df.leak_active == 0, sensor_columns(df)] for _, df in load_dataset(name)]
        allv = pd.concat(frames)
        return allv.mean(), allv.std()

    sm, ss = stats(sim_name)
    rm, rs = stats(real_name)
    common = [c for c in sm.index if c in rm.index]
    out = pd.DataFrame({"sim_mean": sm[common], "real_mean": rm[common],
                        "sim_std": ss[common], "real_std": rs[common]})
    out["mean_diff_%"] = 100 * (out.sim_mean - out.real_mean) / out.real_mean.abs().clip(lower=1e-6)
    print(f"\n== {sim_name} vs {real_name} (no-leak steps, {len(common)} common sensors)")
    print(out.round(2).to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--compare", nargs=2, metavar=("SIM", "REAL"))
    args = ap.parse_args()
    for name in args.datasets:
        check_dataset(name)
    if args.compare:
        compare(*args.compare)
