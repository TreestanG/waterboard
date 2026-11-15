"""Run a detection model on a stored simulation and score it against the planted leaks."""

from __future__ import annotations

import numpy as np

from .models import REGISTRY, ModelInput
from .simulate import SimulationResult, distance


def model_input(sim: SimulationResult) -> ModelInput:
    j_index = {j: i for i, j in enumerate(sim.junctions)}
    idx = [j_index[s] for s in sim.sensors]
    return ModelInput(
        times_h=sim.times_h,
        sensors=sim.sensors,
        sensor_xy=np.array([sim.node_xy[s] for s in sim.sensors], dtype=float),
        measured=sim.measured,
        expected=sim.pressure_baseline[:, idx],
        sensor_sigma_m=sim.noise_m,
        junctions=sim.junctions,
        junction_xy=np.array([sim.node_xy[j] for j in sim.junctions], dtype=float),
        graph=sim.graph,
    )


def evaluate(sim: SimulationResult, model_id: str) -> dict:
    if model_id not in REGISTRY:
        raise KeyError(model_id)
    model = REGISTRY[model_id]
    out = model.run(model_input(sim))

    # Match each planted leak (in start order) to the first unused alarm at or after its start.
    leaks = sorted(sim.leaks, key=lambda l: l["startHour"])
    used: set[int] = set()
    rows = []
    for leak in leaks:
        match = next(
            (i for i, e in enumerate(out.events) if i not in used and e.time_h >= leak["startHour"] - 1e-9),
            None,
        )
        row = {
            "leakId": leak["id"],
            "linkId": leak["linkId"],
            "startHour": leak["startHour"],
            "peakLps": round(max(leak["flowLps"]) if leak["flowLps"] else 0.0, 3),
            "detected": match is not None,
            "detectedHour": None,
            "delayHours": None,
            "predictedNode": None,
            "locationError": None,
        }
        if match is not None:
            used.add(match)
            e = out.events[match]
            row.update(
                detectedHour=e.time_h,
                delayHours=round(e.time_h - leak["startHour"], 3),
                predictedNode=e.node,
                locationError=round(distance((e.x, e.y), (leak["x"], leak["y"])), 2),
            )
        rows.append(row)

    false_alarms = [
        {"timeHour": e.time_h, "node": e.node} for i, e in enumerate(out.events) if i not in used
    ]
    detected = [r for r in rows if r["detected"]]
    return {
        "model": {"id": model.id, "name": model.name, "description": model.description},
        "timesH": sim.times_h,
        "score": out.score,
        "threshold": out.threshold,
        "scoreLabel": out.score_label,
        "notes": out.notes,
        "events": [{"timeHour": e.time_h, "node": e.node, "x": e.x, "y": e.y, "candidates": e.candidates} for e in out.events],
        "groundTruth": {
            "totalLeakLps": [round(sum(l["flowLps"][t] for l in sim.leaks), 4) for t in range(len(sim.times_h))],
            "leaks": [{"id": l["id"], "linkId": l["linkId"], "startHour": l["startHour"], "endHour": l["endHour"]} for l in leaks],
        },
        "perLeak": rows,
        "falseAlarms": false_alarms,
        "summary": {
            "leaks": len(rows),
            "detected": len(detected),
            "falseAlarms": len(false_alarms),
            "meanDelayHours": round(float(np.mean([r["delayHours"] for r in detected])), 3) if detected else None,
            "meanLocationError": round(float(np.mean([r["locationError"] for r in detected])), 2) if detected else None,
        },
    }
