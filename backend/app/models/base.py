"""The contract every detection model implements.

A model only sees what a real deployment would: the sensor IDs and positions, the network
geometry, the noisy sensor measurements over time, and the pressures the hydraulic model
expects with no leaks. It never sees the planted leaks; those are only used for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class ModelInput:
    times_h: list[float]
    sensors: list[str]
    sensor_xy: np.ndarray  # sensor x 2
    measured: np.ndarray  # time x sensor, metres of pressure head
    expected: np.ndarray  # time x sensor, no-leak pressures from the hydraulic model
    sensor_sigma_m: float  # sensor accuracy (instrument spec)
    junctions: list[str]
    junction_xy: np.ndarray  # junction x 2
    # Pipe graph for models that use it: node_ids, elevation, base_demand (all nodes, incl. tanks
    # and reservoirs) and links [{start, end, length, diameter}] (length/diameter None for pumps/valves).
    graph: dict | None = None


@dataclass
class DetectionEvent:
    """A raised alarm: when, and where the model thinks the leak is."""

    time_h: float
    node: str
    x: float
    y: float
    # Optional ranked alternatives: [{node, probability, x, y}], most likely first.
    candidates: list[dict] = field(default_factory=list)


@dataclass
class ModelOutput:
    score: list[float | None]  # one value per time step, in the model's own units; None = no score yet
    threshold: float  # alarm level on the same scale as `score`
    score_label: str
    events: list[DetectionEvent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # plain-language caveats about this run


class LeakModel(Protocol):
    id: str
    name: str
    description: str

    def run(self, inputs: ModelInput) -> ModelOutput: ...
