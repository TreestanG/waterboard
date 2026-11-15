"""Simulate leak scenarios for any .inp with WNTR and write them in the unified format.

Used for networks that ship with no data (Net3, Anytown, ...) and to add scenarios to L-Town.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import wntr
from wntr.network import LinkStatus
from wntr.network.controls import Control, ControlAction, SimTimeCondition

from ..schema import leak_labels, rel, write_network_json, write_scenario

DISCHARGE_COEFF = 0.75
NOTES = [
    "Simulated with WNTR (WNTRSimulator, pressure-driven demand). Leaks are idealized orifices at "
    "junctions; incipient leaks grow linearly in diameter from start to peak time.",
    "Only the listed sensor nodes/links are recorded. Gaussian sensor noise is added "
    "(see scenarios.<id>.params).",
]


def default_sensors(wn, n_pressure: int | None = None) -> tuple[list[str], list[str]]:
    """Evenly spread pressure sensors over the junction list, flow meters on links at sources."""
    junctions = wn.junction_name_list
    n = n_pressure or max(3, round(0.1 * len(junctions)))
    nodes = [junctions[i] for i in np.linspace(0, len(junctions) - 1, min(n, len(junctions))).astype(int)]
    sources = set(wn.reservoir_name_list) | set(wn.tank_name_list)
    links = [name for name, l in wn.links()
             if l.start_node_name in sources or l.end_node_name in sources]
    return sorted(set(nodes), key=junctions.index), links


def _randomize_demands(wn, n_pattern_steps: int, noise: float, rng) -> None:
    """Each junction keeps its own daily pattern, scaled per node and perturbed per step."""
    day = int(86400 / wn.options.time.pattern_timestep)
    diurnal = 1 + 0.4 * np.sin(2 * np.pi * (np.arange(day) / day - 0.3))  # for junctions with no pattern
    scenario_scale = rng.uniform(0.9, 1.1)
    default = wn.options.hydraulic.pattern
    for name, j in wn.junctions():
        demands = [(d.base_value, d.pattern_name or default) for d in j.demand_timeseries_list]
        j.demand_timeseries_list.clear()
        for k, (base, pat) in enumerate(demands):
            if base == 0:
                continue
            mult = np.asarray(wn.get_pattern(pat).multipliers) if pat in wn.pattern_name_list else diurnal
            mult = np.resize(mult, n_pattern_steps)
            mult = mult * scenario_scale * rng.uniform(0.85, 1.15) * (1 + rng.normal(0, noise, n_pattern_steps))
            pat_name = f"sim_{name}_{k}"
            wn.add_pattern(pat_name, np.clip(mult, 0, None).tolist())
            j.add_demand(base, pat_name)


def _pump_speed_controls_to_status(wn) -> None:
    """Rewrite controls that switch a pump with speed 0 / 1 as status closed / open.

    Some .inp files (LeakDB's Net1) switch pumps via the speed setting, which WNTRSimulator
    does not support ("Pump speeds other than 1.0 are not yet supported").
    """
    for name in list(wn.control_name_list):
        control = wn.get_control(name)
        actions = list(control._then_actions)
        if not any(a._attribute == "base_speed" and a._value in (0.0, 1.0) for a in actions):
            continue
        new_actions = [
            ControlAction(a._target_obj, "status", LinkStatus.Open if a._value else LinkStatus.Closed)
            if a._attribute == "base_speed" else a for a in actions]
        wn.remove_control(name)
        wn.add_control(name, Control(control._condition, new_actions, priority=control._priority))


def _add_leak(wn, leak: dict, step_s: int) -> None:
    j = wn.get_node(leak["node"])
    area = lambda d: np.pi * (d / 2) ** 2
    start, end, peak = (leak[k] * step_s for k in ("start_idx", "end_idx", "peak_idx"))
    n_ramp = leak["peak_idx"] - leak["start_idx"]
    if leak["type"] == "abrupt" or n_ramp <= 0:
        j.add_leak(wn, area=area(leak["diameter"]), discharge_coeff=DISCHARGE_COEFF,
                   start_time=start, end_time=end)
        return
    # same ramp as schema.leak_labels: diameter * i / (n_ramp + 1) for i = 1..n_ramp, then full size
    diam = [leak["diameter"] * i / (n_ramp + 1) for i in range(1, n_ramp + 1)] + [leak["diameter"]]
    j.add_leak(wn, area=area(diam[0]), discharge_coeff=DISCHARGE_COEFF, start_time=start, end_time=end)
    for i, d in enumerate(diam[1:], 1):
        action = ControlAction(j, "leak_area", area(d))
        action._private_attribute = "_leak_area"  # leak_area has no public setter in WNTR
        wn.add_control(f"leak_ramp_{leak['node']}_{i}",
                       Control(SimTimeCondition(wn, "=", start + i * step_s), action))


def simulate(inp_path, n_scenarios: int, sensor_nodes: list[str] | None = None,
             sensor_links: list[str] | None = None, dataset: str | None = None,
             duration_days: float = 7, timestep_min: int = 30, leak_fraction: float = 0.75,
             incipient_fraction: float = 0.5, leak_diameter_m: tuple = (0.01, 0.08),
             demand_noise: float = 0.05, pressure_noise_m: float = 0.02, flow_noise_rel: float = 0.01,
             required_pressure_m: float = 10.0, start: str = "2020-01-06", seed: int = 0,
             first_id: int = 1) -> str:
    """Write n_scenarios unified parquet files for the network in inp_path. Returns the dataset name.

    About (1 - leak_fraction) of the scenarios have no leak, so the model sees normal operation.
    """
    inp_path = Path(inp_path)
    dataset = dataset or f"sim_{inp_path.stem.lower().replace('-', '')}"
    rng = np.random.default_rng(seed)
    step_s = timestep_min * 60
    n_steps = int(duration_days * 86400 / step_s)

    base = wntr.network.WaterNetworkModel(str(inp_path))
    auto_nodes, auto_links = default_sensors(base)
    sensor_nodes = list(sensor_nodes or auto_nodes)
    sensor_links = list(sensor_links if sensor_links is not None else auto_links)

    scenarios = {}
    i = failures = 0
    while i < n_scenarios:
        wn = wntr.network.WaterNetworkModel(str(inp_path))
        wn.options.time.duration = (n_steps - 1) * step_s
        wn.options.time.hydraulic_timestep = step_s
        wn.options.time.report_timestep = step_s
        wn.options.hydraulic.demand_model = "PDD"
        wn.options.hydraulic.required_pressure = required_pressure_m
        wn.options.hydraulic.minimum_pressure = 0.0
        n_pattern_steps = int(np.ceil(duration_days * 86400 / wn.options.time.pattern_timestep))
        _pump_speed_controls_to_status(wn)
        _randomize_demands(wn, n_pattern_steps, demand_noise, rng)

        leaks = []
        if rng.random() < leak_fraction:
            kind = "incipient" if rng.random() < incipient_fraction else "abrupt"
            s = int(rng.integers(n_steps // 8, n_steps * 3 // 4))
            # half the leaks run to the end of the scenario, the rest get repaired
            e = n_steps if rng.random() < 0.5 else int(rng.integers(s + max(2, n_steps // 20), n_steps + 1))
            peak = s if kind == "abrupt" else int(rng.integers(s + 1, max(s + 2, e)))
            leaks.append({"node": str(rng.choice(wn.junction_name_list)), "link": None, "type": kind,
                          "diameter": float(rng.uniform(*leak_diameter_m)),
                          "start_idx": s, "end_idx": e, "peak_idx": peak})
            _add_leak(wn, leaks[0], step_s)

        try:
            res = wntr.sim.WNTRSimulator(wn).run_sim()
        except Exception as err:  # non-convergence on an extreme draw: redraw
            failures += 1
            if failures >= 5:
                raise RuntimeError(f"{dataset}: 5 simulations failed in a row, last error: {err}") from err
            print(f"{dataset}: redraw after simulation failure ({err})", flush=True)
            continue
        p = res.node["pressure"][sensor_nodes].to_numpy()
        q = res.link["flowrate"][sensor_links].to_numpy() * 3600.0  # m3/s -> m3/h
        if len(p) != n_steps or not np.isfinite(p).all():
            failures += 1
            if failures >= 5:
                raise RuntimeError(f"{dataset}: 5 simulations in a row gave incomplete or non-finite results")
            continue
        failures = 0
        p = p + rng.normal(0, pressure_noise_m, p.shape)
        q = q * (1 + rng.normal(0, flow_noise_rel, q.shape))

        df = pd.DataFrame({"timestamp": pd.date_range(start, periods=n_steps, freq=f"{timestep_min}min")})
        df = pd.concat([df, pd.DataFrame(p, columns=[f"p_{n}" for n in sensor_nodes]),
                        pd.DataFrame(q, columns=[f"q_{l}" for l in sensor_links]),
                        leak_labels(n_steps, leaks)], axis=1)
        sid = f"scenario_{first_id + i:04d}"
        write_scenario(df, dataset, sid)
        scenarios[sid] = {"inp_path": rel(inp_path), "leaks": leaks}
        i += 1
        print(f"{dataset}: {i}/{n_scenarios}", flush=True)

    params = {"duration_days": duration_days, "leak_fraction": leak_fraction,
              "incipient_fraction": incipient_fraction, "leak_diameter_m": list(leak_diameter_m),
              "demand_noise": demand_noise, "pressure_noise_m": pressure_noise_m,
              "flow_noise_rel": flow_noise_rel, "required_pressure_m": required_pressure_m,
              "discharge_coeff": DISCHARGE_COEFF, "seed": seed}
    write_network_json(dataset, inp_path, step_s,
                       sensors={"pressure": sensor_nodes, "flow": sensor_links, "demand": [], "level": []},
                       scenarios=scenarios, source=f"WNTR simulation of {inp_path.name}",
                       notes=NOTES + [f"simulation params: {params}"])
    return dataset
