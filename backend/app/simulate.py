"""Hydraulic simulation of planted leaks with WNTR.

Each leak is placed at the midpoint of its pipe (the pipe is split there and the leak is
attached to the new junction), opens at its start time and optionally closes at its end
(repair) time. The same network is also simulated without leaks to give the pressures the
sensors *should* read. Sensor "measurements" are the leak-run pressures at the sensor nodes
plus Gaussian noise, which is what a detection model gets to see.
"""

from __future__ import annotations

import math
import re
import tempfile
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import wntr

# Defaults added when an .inp has no [OPTIONS] section (WNTR needs flow units to read it).
DEFAULT_OPTIONS = "[OPTIONS]\nUnits LPS\nHeadloss H-W\n\n"

SPLIT_SUFFIX = "__leak"


class SimulationError(ValueError):
    """Raised for inputs the simulator can't run (bad pipe IDs, unconvergent network, ...)."""


@dataclass
class LeakSpec:
    id: str
    link_id: str
    area_cm2: float
    start_hour: float
    end_hour: float | None = None


@dataclass
class SimulationResult:
    """Everything the API returns, plus the arrays the test endpoint needs later."""

    sim_id: str
    times_h: list[float]
    requested_hours: float
    junctions: list[str]
    node_xy: dict[str, tuple[float, float]]
    sensors: list[str]
    noise_m: float
    # Arrays are time x node (junction order) or time x link (link order).
    pressure_baseline: np.ndarray
    pressure_leak: np.ndarray
    measured: np.ndarray  # time x sensor
    links: list[str]
    flow_baseline: np.ndarray
    flow_leak: np.ndarray
    leaks: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    graph: dict | None = None  # see models.base.ModelInput.graph


def _load(inp_text: str) -> wntr.network.WaterNetworkModel:
    if not re.search(r"^\s*\[OPTIONS\]", inp_text, re.M | re.I):
        if re.search(r"^\s*\[END\]", inp_text, re.M | re.I):
            inp_text = re.sub(r"^\s*\[END\]", DEFAULT_OPTIONS + "[END]", inp_text, count=1, flags=re.M | re.I)
        else:
            inp_text += "\n" + DEFAULT_OPTIONS
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "network.inp"
        path.write_text(inp_text)
        try:
            return wntr.network.WaterNetworkModel(str(path))
        except Exception as e:  # WNTR raises a variety of parse errors
            raise SimulationError(f"Couldn't read the .inp file: {e}") from e


def _configure(wn: wntr.network.WaterNetworkModel, duration_h: float, step_min: float) -> None:
    wn.options.time.duration = int(round(duration_h * 3600))
    wn.options.time.hydraulic_timestep = int(round(step_min * 60))
    wn.options.time.report_timestep = int(round(step_min * 60))
    # Pressure-dependent demand: realistic when leaks pull pressure down.
    wn.options.hydraulic.demand_model = "PDD"


def _run(wn: wntr.network.WaterNetworkModel) -> tuple[object, str | None]:
    """Run WNTR. Returns (results, warning). WNTR stops early (with a warning, not an error)
    when the hydraulics stop converging, so that warning is captured and passed on."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            results = wntr.sim.WNTRSimulator(wn).run_sim()
        except Exception as e:
            raise SimulationError(f"The hydraulic simulation failed: {e}") from e
    message = next((str(w.message) for w in caught if "converge" in str(w.message).lower()), None)
    return results, message


def choose_sensors(junctions: list[str], xy: dict[str, tuple[float, float]], count: int, seed: int = 0) -> list[str]:
    """Spread `count` sensors over the network by farthest-point sampling on coordinates."""
    if not junctions:
        return []
    count = max(1, min(count, len(junctions)))
    pts = np.array([xy[j] for j in junctions], dtype=float)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(junctions)))]
    dist = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    while len(chosen) < count:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[nxt], axis=1))
    return [junctions[i] for i in chosen]


def simulate(
    inp_text: str,
    leaks: list[LeakSpec],
    duration_h: float = 24.0,
    step_min: float = 15.0,
    sensor_count: int | None = None,
    noise_m: float = 0.05,
    seed: int = 7,
) -> SimulationResult:
    if duration_h <= 0 or duration_h > 24 * 14:
        raise SimulationError("Duration must be between 0 and 14 days.")
    if step_min <= 0 or step_min > 60 * 6:
        raise SimulationError("Time step must be between 1 minute and 6 hours.")

    base = _load(inp_text)
    _configure(base, duration_h, step_min)
    junctions = list(base.junction_name_list)
    links = list(base.link_name_list)
    node_xy = {n: tuple(map(float, base.get_node(n).coordinates)) for n in base.node_name_list}

    # Validate leaks against the original network before splitting anything.
    pipe_names = set(base.pipe_name_list)
    for leak in leaks:
        if leak.link_id not in pipe_names:
            raise SimulationError(f"Pipe {leak.link_id} isn't in this network (leaks can only be on pipes).")
        if leak.end_hour is not None and leak.end_hour <= leak.start_hour:
            raise SimulationError(f"Leak on {leak.link_id} ends before it starts.")
    if len({leak.link_id for leak in leaks}) != len(leaks):
        raise SimulationError("Only one leak per pipe is supported.")

    graph = {
        "node_ids": list(base.node_name_list),
        "elevation": [float(getattr(n, "elevation", 0.0) or 0.0) for _, n in base.nodes()],
        "base_demand": [float(getattr(n, "base_demand", 0.0) or 0.0) for _, n in base.nodes()],
        "links": [
            {
                "start": l.start_node_name,
                "end": l.end_node_name,
                "length": float(l.length) if l.link_type == "Pipe" else None,
                "diameter": float(l.diameter) if l.link_type == "Pipe" else None,
            }
            for _, l in base.links()
        ],
    }

    baseline, base_warning = _run(base)

    wn = _load(inp_text)
    _configure(wn, duration_h, step_min)
    leak_nodes: dict[str, str] = {}
    for leak in leaks:
        node_name = f"{leak.link_id}{SPLIT_SUFFIX}"
        wn = wntr.morph.split_pipe(wn, leak.link_id, f"{leak.link_id}{SPLIT_SUFFIX}B", node_name)
        wn.get_node(node_name).add_leak(
            wn,
            area=leak.area_cm2 / 1e4,
            start_time=int(round(leak.start_hour * 3600)),
            end_time=None if leak.end_hour is None else int(round(leak.end_hour * 3600)),
        )
        leak_nodes[leak.id] = node_name
    leaky, leak_warning = _run(wn)

    # If either run stopped early, compare only the time steps both reached.
    common = baseline.node["pressure"].index.intersection(leaky.node["pressure"].index)
    if len(common) == 0:
        raise SimulationError("The hydraulic simulation didn't produce any results (it failed at the first time step).")
    times_s = common.to_numpy()
    p_base = baseline.node["pressure"].loc[common, junctions].to_numpy()
    p_leak = leaky.node["pressure"].loc[common, junctions].to_numpy()
    f_base = baseline.link["flowrate"].loc[common, links].to_numpy()
    f_leak = leaky.link["flowrate"].loc[common, links].to_numpy()

    notes: list[str] = []
    reached_h = float(times_s[-1]) / 3600.0
    if reached_h < duration_h - 1e-6:
        which = "without leaks" if base_warning and not leak_warning else "with leaks" if leak_warning and not base_warning else "in both runs"
        notes.append(
            f"WNTR stopped at {reached_h:g} h of {duration_h:g} h because the hydraulics stopped converging ({which}). "
            "This usually means part of the network ran out of pressure or a tank emptied. Results end there."
        )

    sensor_count = sensor_count or max(4, round(0.05 * len(junctions)))
    sensors = choose_sensors(junctions, node_xy, sensor_count, seed)
    idx = [junctions.index(s) for s in sensors]
    rng = np.random.default_rng(seed)
    measured = p_leak[:, idx] + rng.normal(0.0, noise_m, size=(len(times_s), len(idx)))

    leak_out = []
    for leak in leaks:
        node = leak_nodes[leak.id]
        flow = leaky.node["leak_demand"].loc[common, node].to_numpy() * 1000.0  # m3/s -> L/s
        x, y = map(float, wn.get_node(node).coordinates)
        # Pressure at the leak while it's open. A leak can't discharge where pressure is <= 0,
        # so the UI uses this to explain a leak that produced no flow.
        p = leaky.node["pressure"].loc[common, node].to_numpy()
        hours = times_s / 3600.0
        open_mask = (hours >= leak.start_hour) & ((hours < leak.end_hour) if leak.end_hour is not None else True)
        mean_p = float(np.mean(p[open_mask])) if open_mask.any() else None
        # The pipe is split at the leak: the original ID is now the half from the start node to
        # the leak, and "<id>__leakB" runs from the leak to the end node. Report the water
        # arriving at the leak from each end (positive = toward the leak).
        pipe = base.get_link(leak.link_id)
        from_start = leaky.link["flowrate"].loc[common, leak.link_id].to_numpy() * 1000.0
        from_end = -leaky.link["flowrate"].loc[common, f"{leak.link_id}{SPLIT_SUFFIX}B"].to_numpy() * 1000.0
        leak_out.append(
            {
                "id": leak.id,
                "linkId": leak.link_id,
                "startHour": leak.start_hour,
                "endHour": leak.end_hour,
                "areaCm2": leak.area_cm2,
                "x": x,
                "y": y,
                "flowLps": [round(float(v), 4) for v in flow],
                "meanPressureM": None if mean_p is None else round(mean_p, 2),
                "startNode": pipe.start_node_name,
                "endNode": pipe.end_node_name,
                "inflowFromStartLps": [round(float(v), 4) for v in from_start],
                "inflowFromEndLps": [round(float(v), 4) for v in from_end],
            }
        )

    return SimulationResult(
        sim_id=uuid.uuid4().hex,
        times_h=[round(float(t) / 3600.0, 4) for t in times_s],
        requested_hours=duration_h,
        junctions=junctions,
        node_xy=node_xy,
        sensors=sensors,
        noise_m=noise_m,
        pressure_baseline=p_base,
        pressure_leak=p_leak,
        measured=measured,
        links=links,
        flow_baseline=f_base,
        flow_leak=f_leak,
        leaks=leak_out,
        warnings=notes,
        graph=graph,
    )


def _round(a: np.ndarray, digits: int) -> list:
    return np.round(np.nan_to_num(a, nan=0.0), digits).tolist()


def to_json(r: SimulationResult) -> dict:
    """Column-oriented payload: per node / link series, so the UI can chart any element."""
    return {
        "simId": r.sim_id,
        "timesH": r.times_h,
        "requestedHours": r.requested_hours,
        "warnings": r.warnings,
        "sensors": r.sensors,
        "noiseM": r.noise_m,
        "units": {"pressure": "m", "flow": "L/s", "time": "h"},
        "pressure": {
            j: {"baseline": _round(r.pressure_baseline[:, i], 3), "leak": _round(r.pressure_leak[:, i], 3)}
            for i, j in enumerate(r.junctions)
        },
        "measured": {s: _round(r.measured[:, i], 3) for i, s in enumerate(r.sensors)},
        "flow": {
            l: {"baseline": _round(r.flow_baseline[:, i] * 1000.0, 3), "leak": _round(r.flow_leak[:, i] * 1000.0, 3)}
            for i, l in enumerate(r.links)
        },
        "leaks": r.leaks,
        "totalLeakLps": [round(sum(l["flowLps"][t] for l in r.leaks), 4) for t in range(len(r.times_h))],
    }


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
