"""Training loop for lstm_gnn: detection + leak-node localization on the pipe graph.

Windows are never materialised. Each dataset keeps its normalised residuals as one
[sum of steps, sensors] tensor on the training device and a window is a slice of it,
so memory grows with the number of time steps, not with the number of windows.
Every batch comes from one network, because the model takes one graph per call.
"""
from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from train_graph import load_graph
from train_loop import _det_loss, _plot, set_seed
from train_metrics import best_f1_threshold, detection_metrics, graph_distances, localization_metrics
from train_models import build_model
from train_windows import _events, _sensor_kind, load_network, prep_scenario_job, read_splits

SPLITS = ("train", "val", "test")


def _map(jobs: list, workers: int) -> list:
    workers = workers or min(os.cpu_count() or 1, 16)
    if workers <= 1 or len(jobs) < 8:
        return [prep_scenario_job(j) for j in jobs]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(prep_scenario_job, jobs, chunksize=max(1, len(jobs) // (workers * 8))))


def load_data(cfg, dev) -> tuple[dict, dict]:
    """Returns ({dataset: namespace}, norm_stats)."""
    root = Path(cfg.data_root)
    use = SPLITS if cfg.eval_test else ("train", "val")  # test scenarios are not even read without eval_test
    all_ds = list(cfg.datasets) + [d for d in cfg.holdout_datasets if d not in cfg.datasets and cfg.eval_test]
    data, jobs, owner = {}, [], []
    for ds in all_ds:
        files = {p.stem: p for p in sorted((root / ds).glob("*.parquet"))}
        if not files:
            raise FileNotFoundError(f"no parquet scenarios in {root / ds}")
        graph, _, missing = load_graph(root / ds)
        if graph.placeholders:
            raise ValueError(f"{ds}: network.json lacks {missing}; refusing to train on placeholder graph features")
        net = load_network(root / ds)
        if net["nodes"] != graph.node_ids:
            raise ValueError(f"{ds}: label node order and graph node order differ")
        first = next(iter(files.values()))
        step_s = net["step_s"] or pd.read_parquet(first, columns=["timestamp"])["timestamp"].diff().median().total_seconds()
        node_idx = {n: i for i, n in enumerate(graph.node_ids)}
        link_idx = {l: i for i, l in enumerate(graph.link_ids)}
        cols = pq.read_schema(first).names
        if cfg.sensor_subset != "all":
            cols = [c for c in cols if c in set(cfg.sensor_subset)]
        p_cols = [c for c in cols if c.startswith("p_") and c[2:] in node_idx]
        q_cols = [c for c in cols if c.startswith("q_") and c[2:] in link_idx] if cfg.use_flow else []
        if not p_cols:
            raise ValueError(f"{ds}: no pressure sensor column matches a graph node")
        if ds in cfg.datasets:
            splits = read_splits(root, ds, list(files), cfg.seed)
        else:
            splits = {"train": [], "val": [], "test": sorted(files)}
        if cfg.max_scenarios:
            splits = {k: v[: cfg.max_scenarios] for k, v in splits.items()}
        W = cfg.window_len or int(round(cfg.window_hours * 3600 / step_s))
        S = cfg.stride or max(1, int(round(cfg.stride_hours * 3600 / step_s)))
        base = max(1, int(round(cfg.baseline_hours * 3600 / step_s)))
        for split in use:
            for sid in splits[split]:
                jobs.append((files[sid], p_cols + q_cols, node_idx, step_s, base, cfg.max_gap_steps))
                owner.append((ds, split, sid))
        data[ds] = SimpleNamespace(
            name=ds, graph=graph.to(dev), net=net, step_s=step_s, W=W, S=S, sensors=p_cols + q_cols, n_p=len(p_cols), splits=splits,
            p_node=torch.tensor([node_idx[c[2:]] for c in p_cols], device=dev),
            q_node=torch.tensor([int(graph.link_start[link_idx[c[2:]]]) for c in q_cols], dtype=torch.long, device=dev),
            win=torch.arange(-W + 1, 1, device=dev), raw={s: [] for s in use})

    t = time.time()
    for (ds, split, sid), (resid, active, node) in zip(owner, _map(jobs, cfg.workers)):
        data[ds].raw[split].append((sid, resid, active, node))
    print(f"loaded {len(jobs)} scenarios in {time.time() - t:.0f} s")

    # ---- normalisation: train scenarios only; unseen networks use the average per sensor kind ----
    stats, pooled = {}, {}
    for ds in cfg.datasets:
        tr = [r for _, r, _, _ in data[ds].raw["train"]]
        if not tr:
            raise ValueError(f"{ds}: no train scenarios")
        n = sum((~np.isnan(r)).sum(axis=0) for r in tr)
        mean = sum(np.nansum(r, axis=0, dtype=np.float64) for r in tr) / np.maximum(n, 1)
        var = sum(np.nansum((r - mean) ** 2, axis=0, dtype=np.float64) for r in tr) / np.maximum(n, 1)
        std = np.sqrt(var)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        stats[ds] = {s: {"mean": float(m), "std": float(v)} for s, m, v in zip(data[ds].sensors, mean, std)}
        for s, m, v in zip(data[ds].sensors, mean, std):
            pooled.setdefault(_sensor_kind(s), []).append((m, v))
    stats["_pooled"] = {k: {"mean": float(np.mean([m for m, _ in v])), "std": float(np.mean([s for _, s in v]))} for k, v in pooled.items()}

    for ds, d in data.items():
        st = [stats.get(ds, {}).get(s) or stats["_pooled"].get(_sensor_kind(s)) or {"mean": 0.0, "std": 1.0} for s in d.sensors]
        mean = np.array([x["mean"] for x in st], dtype=np.float32)
        std = np.array([x["std"] for x in st], dtype=np.float32)
        chunks, offset, d.parts = [], 0, {}
        for split in use:
            pos, ys, ns, sc, en, scenarios = [], [], [], [], [], []
            for sid, resid, active, node in d.raw[split]:
                scenarios.append({"dataset": ds, "id": sid, "step_s": d.step_s, "n_steps": len(active), "events": _events(active)})
                z = (resid - mean) / std
                if len(active) >= d.W:
                    gaps = np.concatenate([[0], np.cumsum(np.isnan(z).any(axis=1))])
                    ends = np.arange(d.W - 1, len(active), d.S)
                    ends = ends[gaps[ends + 1] - gaps[ends + 1 - d.W] == 0]  # drop windows that contain a gap
                    pos.append(offset + ends)
                    ys.append(active[ends])
                    ns.append(node[ends])
                    en.append(ends)
                    sc.append(np.full(len(ends), len(scenarios) - 1))
                chunks.append(np.nan_to_num(z).astype(np.float32))
                offset += len(z)
            cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.int64)
            ws = SimpleNamespace(y=cat(ys), node=cat(ns), scen=cat(sc), end_idx=cat(en), scenarios=scenarios)
            d.parts[split] = SimpleNamespace(ws=ws, pos=torch.from_numpy(cat(pos)).to(dev), y=torch.from_numpy(ws.y).float().to(dev),
                                             node=torch.from_numpy(ws.node).to(dev))
        d.Z = torch.from_numpy(np.concatenate(chunks)).to(dev)
        del d.raw
    return data, stats


def batch_x(d, pos, use_flow: bool) -> torch.Tensor:
    """Window end positions [B] -> per-node input [B, N, W, F] (see train_graph.node_inputs)."""
    z = d.Z[pos[:, None] + d.win]  # [B, W, C]
    x = torch.zeros(len(pos), d.graph.num_nodes, d.W, 3 if use_flow else 2, device=z.device)
    x[:, d.p_node, :, 0] = z[:, :, : d.n_p].permute(0, 2, 1)
    x[:, d.p_node, :, 1] = 1.0
    if use_flow and len(d.q_node):
        x[:, d.q_node, :, 2] = z[:, :, d.n_p :].permute(0, 2, 1)
    return x


@torch.no_grad()
def predict(model, d, part, use_flow: bool):
    """Detection probability [n] and node logits [n, N] for every window of one part."""
    model.eval()
    bs = max(32, 30000 // d.graph.num_nodes)  # the LSTM runs once per node per window
    probs, logits = [], []
    for i in range(0, len(part.pos), bs):
        node_logits, det = model(batch_x(d, part.pos[i : i + bs], use_flow), d.graph)
        probs.append(torch.sigmoid(det).cpu())
        logits.append(node_logits.cpu())
    if not probs:
        return np.zeros(0, np.float32), np.zeros((0, d.graph.num_nodes), np.float32)
    return torch.cat(probs).numpy(), torch.cat(logits).numpy()


def _pool(wss: list):
    """Several window sets as one, for pooled detection metrics."""
    scenarios, scen = [], []
    for ws in wss:
        scen.append(ws.scen + len(scenarios))
        scenarios += ws.scenarios
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.int64)
    return SimpleNamespace(y=cat([w.y for w in wss]), node=cat([w.node for w in wss]), scen=cat(scen),
                           end_idx=cat([w.end_idx for w in wss]), scenarios=scenarios)


def _pr_auc(y, score):
    return float(average_precision_score(y, score)) if len(y) and 0 < y.sum() < len(y) else None


def train_graph(cfg) -> dict:
    set_seed(cfg.seed)
    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, stats = load_data(cfg, dev)
    (run_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    (run_dir / "splits.json").write_text(json.dumps({ds: d.splits for ds, d in data.items()}, indent=1))
    for ds, d in data.items():
        print(f"{ds}: {d.graph.num_nodes} nodes, {d.n_p} pressure sensors, W={d.W} S={d.S}, windows "
              + "  ".join(f"{s} {len(d.parts[s].pos)} ({len(d.splits[s])} scenarios)" for s in d.parts))

    n_train = sum(len(data[ds].parts["train"].pos) for ds in cfg.datasets)
    n_pos = max(1, int(sum(data[ds].parts["train"].ws.y.sum() for ds in cfg.datasets)))
    if n_train == 0:
        raise ValueError("no training windows")
    pos_weight = torch.tensor(cfg.pos_weight or (n_train - n_pos) / n_pos, device=dev, dtype=torch.float32)

    model = build_model(cfg, 0, 0).to(dev)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    t_start, n_epochs, sched = time.time(), cfg.epochs, None
    print(f"device {dev}, {sum(p.numel() for p in model.parameters()):,} parameters, {n_train} train windows, pos_weight {pos_weight.item():.2f}")

    def evaluate(split: str, names: list) -> dict:
        return {ds: predict(model, data[ds], data[ds].parts[split], cfg.use_flow) for ds in names}

    def top1(split: str, out: dict):
        hit = [(out[ds][1].argmax(axis=1) == data[ds].parts[split].ws.node)[(data[ds].parts[split].ws.y == 1) & (data[ds].parts[split].ws.node >= 0)] for ds in out]
        hit = np.concatenate(hit) if hit else np.zeros(0)
        return float(hit.mean()) if len(hit) else None

    best, bad, history, epoch = -1.0, 0, [], -1
    while epoch + 1 < n_epochs:
        epoch += 1
        t0 = time.time()
        model.train()
        batches = []
        for ds in cfg.datasets:
            perm = torch.randperm(len(data[ds].parts["train"].pos), device=dev)
            # the LSTM runs once per node per window: cap nodes x windows so large networks fit on the GPU
            bs = min(cfg.batch_size, max(8, 8192 // data[ds].graph.num_nodes))
            batches += [(ds, perm[i : i + bs]) for i in range(0, len(perm), bs)]
        random.shuffle(batches)
        tot_det, tot_loc, n = 0.0, 0.0, 0
        probe = max(1, min(300, len(batches) - 1))
        for k, (ds, idx) in enumerate(batches):
            if sched is None and k == probe:  # set the number of epochs from the measured speed
                if cfg.time_budget_hours > 0:
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    epoch_s = 1.1 * (time.time() - t0) / probe * len(batches)  # 10% for validation
                    left = cfg.time_budget_hours * 3600 - (t0 - t_start) - epoch_s  # one epoch's worth kept for the final evaluation
                    n_epochs = max(1, min(cfg.epochs, int(left / epoch_s)))
                    print(f"time budget {cfg.time_budget_hours} h: about {epoch_s / 60:.1f} min per epoch, running {n_epochs} epochs", flush=True)
                sched = CosineAnnealingLR(opt, T_max=n_epochs)
            d, part = data[ds], data[ds].parts["train"]
            y, node = part.y[idx], part.node[idx]
            node_logits, det = model(batch_x(d, part.pos[idx], cfg.use_flow), d.graph)
            det_loss = _det_loss(det, y, cfg, pos_weight)
            leak = (y == 1) & (node >= 0)
            loc_loss = F.cross_entropy(node_logits[leak], node[leak]) if leak.any() else det_loss.new_zeros(())
            loss = det_loss + cfg.loc_weight * loc_loss
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot_det, tot_loc, n = tot_det + det_loss.item() * len(idx), tot_loc + loc_loss.item() * len(idx), n + len(idx)
        if sched is None:
            sched = CosineAnnealingLR(opt, T_max=n_epochs)
        sched.step()
        t_train = time.time() - t0

        out = evaluate("val", cfg.datasets)
        y_val = np.concatenate([data[ds].parts["val"].ws.y for ds in cfg.datasets])
        score = _pr_auc(y_val, np.concatenate([out[ds][0] for ds in cfg.datasets])) or 0.0
        row = {"epoch": epoch + 1, "train_loss": (tot_det + cfg.loc_weight * tot_loc) / n, "train_det_loss": tot_det / n, "train_loc_loss": tot_loc / n,
               "val_score": score, "val_top1": top1("val", out),
               "val_pr_auc_by_dataset": {ds: _pr_auc(data[ds].parts["val"].ws.y, out[ds][0]) for ds in cfg.datasets},
               "train_seconds": round(t_train, 1), "val_seconds": round(time.time() - t0 - t_train, 1)}
        history.append(row)
        print(f"epoch {epoch + 1:3d}  det_loss {row['train_det_loss']:.4f}  loc_loss {row['train_loc_loss']:.4f}  val_pr_auc {score:.4f}"
              f"  val_top1 {row['val_top1'] if row['val_top1'] is None else round(row['val_top1'], 4)}  train {t_train:.0f} s  val {row['val_seconds']:.0f} s", flush=True)
        if score > best:
            best, bad = score, 0
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "epoch": epoch + 1, "val_score": score,
                        "sensors": {ds: d.sensors for ds, d in data.items()}}, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    best_epoch = torch.load(run_dir / "best.pt", map_location=dev)
    model.load_state_dict(best_epoch["state_dict"])

    # ---- final evaluation with the best checkpoint ----------------------------------------------
    val_out = evaluate("val", cfg.datasets)
    val_ws = _pool([data[ds].parts["val"].ws for ds in cfg.datasets])
    val_score = np.concatenate([val_out[ds][0] for ds in cfg.datasets])
    thr = cfg.threshold or best_f1_threshold(val_ws.y, val_score)

    def report(split: str, names: list, out: dict) -> dict:
        res = {}
        if len(names) > 1:
            pooled = _pool([data[ds].parts[split].ws for ds in names])
            res["all"] = {"detection": detection_metrics(pooled, np.concatenate([out[ds][0] for ds in names]), thr, 1)}
            for k in ("delay_hours_median", "delay_hours_p90", "false_alarms_per_month"):  # stride differs per dataset
                res["all"]["detection"].pop(k, None)
        for ds in names:
            d, ws = data[ds], data[ds].parts[split].ws
            prob, logits = out[ds]
            leak = (ws.y == 1) & (ws.node >= 0)
            hops, meters = graph_distances(d.net)
            res[ds] = {"n_scenarios": len(ws.scenarios), "detection": detection_metrics(ws, prob, thr, d.S),
                       "localization": localization_metrics(logits[leak], ws.node[leak], hops, meters)}
        return res

    metrics = {"model": cfg.model, "task": "detection + leak-node localization", "datasets": cfg.datasets, "holdout_datasets": cfg.holdout_datasets,
               "epochs_run": len(history), "epochs_planned": n_epochs, "hidden_size": cfg.hidden_size, "best_epoch": best_epoch["epoch"], "threshold": float(thr), "threshold_source": "best F1 on pooled validation",
               "window_len": {ds: d.W for ds, d in data.items()}, "stride": {ds: d.S for ds, d in data.items()},
               "n_scenarios": {ds: {s: len(d.splits[s]) for s in d.parts} for ds, d in data.items()},
               "history": history,
               "val": report("val", cfg.datasets, val_out)}
    if cfg.eval_test:
        metrics["test_seen_network"] = report("test", cfg.datasets, evaluate("test", cfg.datasets))
        metrics["test_heldout_network"] = report("test", cfg.holdout_datasets, evaluate("test", cfg.holdout_datasets)) if cfg.holdout_datasets else {}
    else:
        metrics["test"] = "not evaluated (pass --eval_test)"
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- one row per validation window ----------------------------------------------------------
    frames = []
    for ds in cfg.datasets:
        d, ws = data[ds], data[ds].parts["val"].ws
        prob, logits = val_out[ds]
        ids = np.array(d.graph.node_ids, dtype=object)
        pred = logits.argmax(axis=1) if len(logits) else np.zeros(0, np.int64)
        hops, _ = graph_distances(d.net)
        known = ws.node >= 0
        err = np.full(len(ws.y), np.nan)
        if hops is not None and known.any():
            err[known] = hops[pred[known], ws.node[known]]
        frames.append(pd.DataFrame({
            "dataset": ds, "scenario_id": np.array([s["id"] for s in ws.scenarios], dtype=object)[ws.scen] if len(ws.y) else [],
            "end_idx": ws.end_idx, "hours_from_start": ws.end_idx * d.step_s / 3600, "leak_active": ws.y.astype(np.int8),
            "score": prob.astype(np.float32), "alarm": prob >= thr,
            "leak_node": np.where(known, ids[np.maximum(ws.node, 0)], None), "pred_node": ids[pred] if len(pred) else [],
            "pred_error_pipes": err}))
    pd.concat(frames, ignore_index=True).to_parquet(run_dir / "val_predictions.parquet", index=False)
    _plot(run_dir, history, "det", val_ws, val_score)
    return metrics
