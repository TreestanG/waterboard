"""Waterboard API: hydraulic simulation of planted leaks and model evaluation.

Run with:  uv run uvicorn app.main:app --port 8000   (from backend/)
The Vite dev server proxies /api here.
"""

from __future__ import annotations

from collections import OrderedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .evaluate import evaluate
from .models import REGISTRY
from .simulate import LeakSpec, SimulationError, SimulationResult, simulate, to_json

app = FastAPI(title="Waterboard API")

# Recent simulations, kept in memory so a model can be tested without resending everything.
# Lost on restart; the UI re-runs the simulation if a test can't find it.
MAX_CACHED = 20
_cache: "OrderedDict[str, SimulationResult]" = OrderedDict()


class LeakIn(BaseModel):
    id: str
    linkId: str
    areaCm2: float = Field(gt=0)
    startHour: float = Field(ge=0)
    endHour: float | None = None


class SimulateIn(BaseModel):
    inpText: str
    leaks: list[LeakIn] = []
    durationHours: float = 24
    timestepMinutes: float = 15
    sensorCount: int | None = None
    noiseM: float = Field(default=0.05, ge=0)


class TestIn(BaseModel):
    simId: str
    model: str = "baseline"


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/models")
def models() -> list[dict]:
    return [{"id": m.id, "name": m.name, "description": m.description} for m in REGISTRY.values()]


@app.post("/api/simulate")
def run_simulation(body: SimulateIn) -> dict:
    try:
        result = simulate(
            body.inpText,
            [LeakSpec(l.id, l.linkId, l.areaCm2, l.startHour, l.endHour) for l in body.leaks],
            duration_h=body.durationHours,
            step_min=body.timestepMinutes,
            sensor_count=body.sensorCount,
            noise_m=body.noiseM,
        )
    except SimulationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    _cache[result.sim_id] = result
    while len(_cache) > MAX_CACHED:
        _cache.popitem(last=False)
    return to_json(result)


@app.post("/api/test")
def run_test(body: TestIn) -> dict:
    sim = _cache.get(body.simId)
    if sim is None:
        raise HTTPException(status_code=404, detail="That simulation is no longer on the server. Run the simulation again.")
    try:
        return evaluate(sim, body.model)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Unknown model: {body.model}") from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
