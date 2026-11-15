"""Training and validation loop for detection (gb, cnn) and localization (cnn_loc)."""
from __future__ import annotations

import json
import pickle
import random
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from train_config import RUNS_DIR
from train_metrics import best_f1_threshold, detection_metrics, graph_distances, localization_metrics
from train_models import build_model
from train_windows import make_datasets


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, X: np.ndarray, task: str, batch_size: int = 1024) -> np.ndarray:
    """Detection: probability per window. Localization: node logits per window."""
    if not isinstance(model, torch.nn.Module):
        return model.predict_score(X) if len(X) else np.zeros(0)
    model.eval()
    dev = next(model.parameters()).device
    out = [model(torch.from_numpy(X[i : i + batch_size]).to(dev)).cpu() for i in range(0, len(X), batch_size)]
    if not out:
        return np.zeros(0)
    out = torch.cat(out)
    return (torch.sigmoid(out) if task == "det" else out).numpy()


def _det_loss(logit, y, cfg, pos_weight):
    if cfg.loss == "focal":
        bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        p_t = torch.exp(-bce)
        alpha = torch.where(y == 1, pos_weight / (1 + pos_weight), 1 / (1 + pos_weight))
        return (alpha * (1 - p_t) ** cfg.focal_gamma * bce).mean()
    return F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)


def _val_score(model, ws, task) -> float:
    if len(ws) == 0:
        return 0.0
    out = predict(model, ws.X, task)
    if task == "det":
        return float(average_precision_score(ws.y, out)) if 0 < ws.y.sum() < len(ws) else 0.0
    return float((out.argmax(axis=1) == ws.node).mean())


def _fit_torch(cfg, model, train, val, task, soft, run_dir, ckpt_extra) -> list:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    target = train.y.astype(np.float32) if task == "det" else train.node
    loader = DataLoader(TensorDataset(torch.from_numpy(train.X), torch.from_numpy(target)), batch_size=cfg.batch_size, shuffle=True)
    n_pos = max(1, int(train.y.sum()))
    pos_weight = torch.tensor(cfg.pos_weight or (len(train) - n_pos) / n_pos, device=dev, dtype=torch.float32)
    soft = None if soft is None else torch.from_numpy(soft).float().to(dev)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)
    best, bad, history = -1.0, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            if task == "det":
                loss = _det_loss(out, y, cfg, pos_weight)
            elif soft is not None:
                loss = -(soft[y] * F.log_softmax(out, dim=1)).sum(dim=1).mean()
            else:
                loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(x), n + len(x)
        sched.step()
        score = _val_score(model, val, task)
        history.append({"epoch": epoch + 1, "train_loss": total / n, "val_score": score})
        print(f"epoch {epoch + 1:3d}  train_loss {total / n:.4f}  val_{'pr_auc' if task == 'det' else 'top1'} {score:.4f}")
        if score > best:
            best, bad = score, 0
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "epoch": epoch + 1, "val_score": score, **ckpt_extra}, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=dev)["state_dict"])
    return history


def _plot(run_dir, history, task, val, val_out):
    panels = (1 if history else 0) + (1 if task == "det" else 0)
    if not panels:
        return
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4), squeeze=False)
    axes = list(axes[0])
    if history:
        ax = axes.pop(0)
        ep = [h["epoch"] for h in history]
        ax.plot(ep, [h["train_loss"] for h in history], label="train loss")
        ax.plot(ep, [h["val_score"] for h in history], label="val PR-AUC" if task == "det" else "val top-1")
        ax.set_xlabel("epoch")
        ax.legend()
    if task == "det" and len(val.y) and 0 < val.y.sum() < len(val.y):
        p, r, _ = precision_recall_curve(val.y, val_out)
        axes[0].plot(r, p)
        axes[0].axhline(val.y.mean(), ls="--", c="gray", label="chance")
        axes[0].set_xlabel("recall")
        axes[0].set_ylabel("precision")
        axes[0].set_title("validation PR curve")
        axes[0].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "val_curves.png", dpi=120)
    plt.close(fig)


def train(cfg) -> dict:
    if cfg.model == "lstm_gnn":
        from train_graph_loop import train_graph  # needs PyG, the other models do not
        return train_graph(cfg)
    set_seed(cfg.seed)
    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")

    data = make_datasets(cfg)
    meta = data["meta"]
    (run_dir / "norm_stats.json").write_text(json.dumps(meta["norm_stats"], indent=2))
    task = "loc" if cfg.model == "cnn_loc" else "det"
    train_ws, val_ws, test_ws = data["train"], data["val"], data["test"]

    n_nodes, hops, meters, soft = 0, None, None, None
    if task == "loc":
        if len(cfg.datasets) != 1:
            raise ValueError("cnn_loc is tied to one network: pass exactly one dataset")
        net = meta["networks"][cfg.datasets[0]]
        n_nodes = len(net["nodes"])
        hops, meters = graph_distances(net)
        keep = lambda ws: ws.subset((ws.y == 1) & (ws.node >= 0))  # leak-active windows only
        train_ws, val_ws, test_ws = keep(train_ws), keep(val_ws), keep(test_ws)
        if cfg.loc_soft_tau > 0 and hops is not None:
            w = np.exp(-np.nan_to_num(hops, nan=1e9) / cfg.loc_soft_tau)
            soft = w / w.sum(axis=1, keepdims=True)
    if len(train_ws) == 0:
        raise ValueError("no training windows")
    print(f"windows: train {len(train_ws)}  val {len(val_ws)}  test {len(test_ws)}  shape [C={meta['n_channels']}, W={meta['window_len']}]")

    model = build_model(cfg, meta["n_channels"], n_nodes)
    history = []
    if cfg.model == "gb":
        model.fit(train_ws.X, train_ws.y)
        with open(run_dir / "best.pkl", "wb") as f:
            pickle.dump(model, f)
    else:
        if cfg.init_from:
            src = torch.load(RUNS_DIR / cfg.init_from / "best.pt", map_location="cpu")["state_dict"]
            model.load_state_dict({k: v for k, v in src.items() if k.startswith("backbone.")}, strict=False)
        extra = {"n_channels": meta["n_channels"], "n_nodes": n_nodes, "sensors": meta["sensors"],
                 "nodes": meta["networks"][cfg.datasets[0]]["nodes"] if task == "loc" else None}
        history = _fit_torch(cfg, model, train_ws, val_ws, task, soft, run_dir, extra)

    # ---- final evaluation: seen network and held-out network are never mixed -------------------
    val_out = predict(model, val_ws.X, task)
    metrics = {"model": cfg.model, "task": task, "datasets": cfg.datasets, "window_len": meta["window_len"],
               "stride": meta["stride"], "n_channels": meta["n_channels"], "epochs_run": len(history)}
    if task == "det":
        thr = cfg.threshold or best_f1_threshold(val_ws.y, val_out)
        ev = lambda ws: detection_metrics(ws, predict(model, ws.X, task), thr, meta["stride"])
        metrics["val"] = ev(val_ws)
        if cfg.eval_test:
            metrics["test_seen_network"] = ev(test_ws)
            metrics["test_heldout_network"] = {ds: ev(ws) for ds, ws in data["holdout"].items()}
    else:
        ev = lambda ws: localization_metrics(predict(model, ws.X, task), ws.node, hops, meters)
        metrics["val"] = ev(val_ws)
        if cfg.eval_test:
            metrics["test_seen_network"] = ev(test_ws)
        metrics["note"] = f"node head is tied to network {cfg.datasets[0]!r}; it cannot be evaluated on another network"
    if not cfg.eval_test:
        metrics["test"] = "not evaluated (pass --eval_test)"
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _plot(run_dir, history, task, val_ws, val_out)
    return metrics
