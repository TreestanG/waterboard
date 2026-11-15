"""Placeholder detector so the test pipeline runs end to end before the real model exists.

Detection: the RMS pressure drop across sensors (expected minus measured), in units of sensor
noise. An alarm is raised when it first crosses the threshold, and again whenever it jumps by
a further threshold-sized step (a new leak adds its own drop on top).

Localization: the residual-weighted centroid of the sensors that dropped since the previous
alarm, snapped to the nearest junction. Deliberately simple; expect coarse locations.
"""

from __future__ import annotations

import numpy as np

from .base import DetectionEvent, ModelInput, ModelOutput

THRESHOLD = 4.0  # alarm level, in sensor-noise standard deviations
CONFIRM_STEPS = 2  # consecutive steps above the level before alarming


class ResidualBaseline:
    id = "baseline"
    name = "Residual baseline (placeholder)"
    description = "RMS pressure-drop threshold with residual-weighted localization. Stand-in until the real model is ready."

    def run(self, inputs: ModelInput) -> ModelOutput:
        residual = inputs.expected - inputs.measured  # positive = pressure lower than expected
        sigma = max(inputs.sensor_sigma_m, 1e-6)
        score = np.sqrt(np.mean(np.square(residual), axis=1)) / sigma

        events: list[DetectionEvent] = []
        level = 0.0  # score at the last alarm
        ref = np.zeros(residual.shape[1])  # residuals at the last alarm
        streak = 0
        for t, s in enumerate(score):
            if s - level > THRESHOLD:
                streak += 1
            else:
                streak = 0
            if streak < CONFIRM_STEPS:
                continue
            delta = np.clip(residual[t] - ref, 0.0, None)
            w = delta**2
            if w.sum() <= 0:
                w = np.ones_like(delta)
            cx, cy = (inputs.sensor_xy * w[:, None]).sum(axis=0) / w.sum()
            j = int(np.argmin(np.hypot(inputs.junction_xy[:, 0] - cx, inputs.junction_xy[:, 1] - cy)))
            events.append(
                DetectionEvent(
                    time_h=inputs.times_h[t],
                    node=inputs.junctions[j],
                    x=float(inputs.junction_xy[j, 0]),
                    y=float(inputs.junction_xy[j, 1]),
                )
            )
            level, ref, streak = s, residual[t].copy(), 0

        return ModelOutput(
            score=[round(float(v), 3) for v in score],
            threshold=THRESHOLD,
            score_label="Pressure-drop score (× sensor noise)",
            events=events,
        )
