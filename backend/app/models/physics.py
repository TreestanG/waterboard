"""Physics-based detection and localization (no training).

Detection: one-sided CUSUM per sensor on the pressure drop (expected minus measured), after
removing each sensor's usual residual at that time of day, learned from leak-free history.
Localization: signature matching. `signatures[j, s]` is the simulated pressure drop at sensor
s for a leak at candidate junction j; candidates are ranked by cosine similarity with the
observed drop.
"""

from __future__ import annotations

import numpy as np

CUSUM_K = 0.5  # slack, in standard deviations: drifts smaller than this are ignored
CUSUM_H = 12.0  # alarm level. For unit Gaussian noise the in-control run length is ~3e5 steps per sensor.


def calibrate(drop: np.ndarray, steps_per_day: int, n_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Usual drop per sensor by time of day, and its spread, from the first n_days (assumed leak-free).

    drop: [T, S]. Returns (profile [steps_per_day, S], sigma [S])."""
    days = drop[: steps_per_day * n_days].reshape(n_days, steps_per_day, -1)
    profile = days.mean(axis=0)
    # day-to-day spread at the same time of day: sensor noise plus whatever the reference gets wrong differently each day
    sigma = (days - profile).std(axis=(0, 1)) * np.sqrt(n_days / max(n_days - 1, 1))
    return profile, np.maximum(sigma, 1e-6)


def cusum_detect(drop: np.ndarray, steps_per_day: int, calib_days: int = 2, k: float = CUSUM_K, h: float = CUSUM_H) -> dict:
    """First alarm after the calibration period. Returns score [T] (max CUSUM over sensors, NaN during
    calibration), alarm step (or None), estimated onset step, and the calibrated drop `excess` [T, S]."""
    T, S = drop.shape
    profile, sigma = calibrate(drop, steps_per_day, calib_days)
    excess = drop - profile[np.arange(T) % steps_per_day]
    z = excess / (sigma * np.sqrt(1 + 1 / calib_days))  # the profile is itself an average of calib_days noisy days
    c = np.zeros(S)
    last_zero = np.full(S, steps_per_day * calib_days)
    score = np.full(T, np.nan)
    alarm = onset = None
    for t in range(steps_per_day * calib_days, T):
        c = np.maximum(0.0, c + z[t] - k)
        last_zero[c == 0] = t + 1
        score[t] = c.max()
        if alarm is None and score[t] > h:
            alarm, onset = t, int(last_zero[int(c.argmax())])
    return {"score": score, "alarm": alarm, "onset": onset, "excess": excess, "sigma": sigma}


def rank_candidates(observed: np.ndarray, signatures: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Candidate indices, best match first. observed [S], signatures [J, S] (NaN rows = no signature)."""
    o = observed / sigma
    s = signatures / sigma
    sim = (s @ o) / (np.linalg.norm(s, axis=1) * np.linalg.norm(o) + 1e-12)
    sim = np.where(np.isfinite(sim), sim, -np.inf)
    return np.argsort(-sim)
