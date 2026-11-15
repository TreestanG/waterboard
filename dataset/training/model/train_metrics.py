"""Detection and localization metrics."""
from __future__ import annotations

from collections import deque

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

MONTH_S = 30 * 86400


def _f(v):
    """JSON-safe float: NaN/inf -> None."""
    v = float(v)
    return v if np.isfinite(v) else None


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    if len(y) == 0 or y.min() == y.max():
        return 0.5
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def detection_metrics(ws, score: np.ndarray, thr: float, stride: int) -> dict:
    """ws: WindowSet, score: probability per window. Event metrics use each scenario's own time axis."""
    y = ws.y
    alarm = score >= thr
    tp, fp, fn = int((alarm & (y == 1)).sum()), int((alarm & (y == 0)).sum()), int((~alarm & (y == 1)).sum())
    out = {
        "n_windows": int(len(y)),
        "leak_fraction": _f(y.mean()) if len(y) else None,
        "pr_auc": _f(average_precision_score(y, score)) if len(y) and 0 < y.sum() < len(y) else None,
        "threshold": float(thr),
        "precision": _f(tp / (tp + fp)) if tp + fp else None,
        "recall": _f(tp / (tp + fn)) if tp + fn else None,
        "f1": _f(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else None,
    }

    delays, n_events, n_missed = [], 0, 0
    false_alarms, noleak_s, n_noleak = 0, 0.0, 0
    for k, sc in enumerate(ws.scenarios):
        m = ws.scen == k
        ends, a = ws.end_idx[m], alarm[m]
        order = np.argsort(ends)
        ends, a = ends[order], a[order]
        if not sc["events"]:
            n_noleak += 1
            noleak_s += len(ends) * stride * sc["step_s"]
            false_alarms += int((a & ~np.concatenate([[False], a[:-1]])).sum())  # rising edges
            continue
        for start, end in sc["events"]:
            inside = (ends >= start) & (ends <= end)
            if not inside.any():
                continue  # no window ends during this leak, cannot be scored
            n_events += 1
            hit = ends[inside][a[inside]]
            if len(hit):
                delays.append((hit[0] - start) * sc["step_s"] / 3600)
            else:
                n_missed += 1
    out.update({
        "n_leak_events": n_events,
        "missed_leak_fraction": _f(n_missed / n_events) if n_events else None,
        "delay_hours_median": _f(np.median(delays)) if delays else None,
        "delay_hours_p90": _f(np.percentile(delays, 90)) if delays else None,
        "n_noleak_scenarios": n_noleak,
        "false_alarms_per_month": _f(false_alarms / (noleak_s / MONTH_S)) if noleak_s else None,
    })
    return out


def graph_distances(net: dict):
    """Hop-count matrix over the pipe graph and straight-line metres (None if unavailable)."""
    nodes = net["nodes"]
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    hops = None
    if net["links"]:
        adj = [[] for _ in nodes]
        for a, b in net["links"]:
            if a in idx and b in idx:
                adj[idx[a]].append(idx[b])
                adj[idx[b]].append(idx[a])
        hops = np.full((n, n), np.nan)
        for s in range(n):
            hops[s, s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if np.isnan(hops[s, v]):
                        hops[s, v] = hops[s, u] + 1
                        q.append(v)
    meters = None
    if all(nid in net["coords"] for nid in nodes) and n:
        xy = np.array([net["coords"][nid] for nid in nodes])
        meters = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    return hops, meters


def localization_metrics(logits: np.ndarray, node: np.ndarray, hops, meters) -> dict:
    if len(node) == 0:
        return {"n_windows": 0}
    order = np.argsort(-logits, axis=1)
    pred = order[:, 0]
    n_nodes = logits.shape[1]
    out = {
        "n_windows": int(len(node)),
        "n_nodes": int(n_nodes),
        "top1": _f((pred == node).mean()),
        "top5": _f((order[:, :5] == node[:, None]).any(axis=1).mean()),
        "chance_top1": 1.0 / n_nodes,
        "chance_top5": min(1.0, 5.0 / n_nodes),
    }
    if hops is not None:
        out["error_pipes_mean"] = _f(np.nanmean(hops[pred, node]))
        out["chance_error_pipes_mean"] = _f(np.nanmean(hops[:, node]))  # uniform random guess
    if meters is not None:
        out["error_m_mean"] = _f(meters[pred, node].mean())  # straight line, not pipe length
        out["chance_error_m_mean"] = _f(meters[:, node].mean())
    return out
