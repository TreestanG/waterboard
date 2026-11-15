"""Verification only, no training: shapes, gradient flow, and two graphs of different size.

  python train_verify_graph.py
"""
from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from train_config import FAKE_DATA_DIR, TrainConfig
from train_fake_graph import FAKE_GRAPH_DIR, generate
from train_graph import load_graph, node_inputs
from train_models import LSTMGNNDetector, build_model
from train_windows import make_datasets

B = 8


def fake_batch(ds: str, cfg: TrainConfig):
    """Real plumbing: fake parquet -> train_windows -> per-node tensor on the pipe graph."""
    c = TrainConfig(**{**cfg.__dict__, "datasets": [ds], "data_root": str(FAKE_GRAPH_DIR)})
    data = make_datasets(c)
    ws = data["train"]
    graph, mask, missing = load_graph(FAKE_GRAPH_DIR / ds)
    assert not missing and not graph.placeholders, missing
    assert data["meta"]["networks"][ds]["nodes"] == graph.node_ids  # label node index == graph node index
    pick = torch.cat([torch.nonzero(torch.from_numpy(ws.y) == 1)[: B // 2, 0], torch.nonzero(torch.from_numpy(ws.y) == 0)[: B // 2, 0]])
    x = node_inputs(ws.X[pick.numpy()], data["meta"]["sensors"][ds], graph, use_flow=cfg.use_flow)
    return x, graph, mask, torch.from_numpy(ws.y[pick.numpy()]).float(), torch.from_numpy(ws.node[pick.numpy()])


def main() -> None:
    torch.manual_seed(0)
    generate()
    for pool, use_flow, num_layers, gnn_layers in (("max", False, 1, 2), ("attn", True, 2, 3)):
        cfg = TrainConfig(model="lstm_gnn", graph_pool=pool, use_flow=use_flow, num_layers=num_layers, gnn_layers=gnn_layers)
        model = build_model(cfg, n_sensors=0, n_nodes=0)
        assert isinstance(model, LSTMGNNDetector)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n== pool={pool} use_flow={use_flow} lstm_layers={num_layers} gnn_layers={gnn_layers}  ({n_params:,} parameters)")

        for ds in ("fake_graph_a", "fake_graph_b"):  # same model object, different node counts
            x, graph, mask, y, node = fake_batch(ds, cfg)
            N, W, Fdim = graph.num_nodes, x.shape[2], 3 if use_flow else 2
            assert x.shape == (B, N, W, Fdim), x.shape
            assert (x[:, ~mask, :, 0] == 0).all() and (x[:, ~mask, :, 1] == 0).all() and (x[:, mask, :, 1] == 1).all()

            model.train()
            model.zero_grad()
            node_logits, det = model(x, graph)
            assert node_logits.shape == (B, N), node_logits.shape
            assert det.shape == (B,), det.shape
            assert torch.isfinite(node_logits).all() and torch.isfinite(det).all()

            # one backward pass: detection BCE + node cross-entropy on leak-active windows
            leak = (y == 1) & (node >= 0)
            loss = F.binary_cross_entropy_with_logits(det, y) + F.cross_entropy(node_logits[leak], node[leak])
            loss.backward()
            dead = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
            assert not dead, f"no gradient reached: {dead}"
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            print(f"{ds}: x {tuple(x.shape)}  sensors {int(mask.sum())}/{N}  edges {graph.edge_index.shape[1]}"
                  f"  -> node logits {tuple(node_logits.shape)}, detection logit {tuple(det.shape)}"
                  f"  loss {loss.item():.3f}  grad norm {norm.item():.3f} (all {sum(1 for _ in model.parameters())} tensors have gradients)")

    # network.json without the graph fields: report what is missing, placeholders are labelled
    old = FAKE_DATA_DIR / "fake_net"
    if (old / "network.json").exists():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            graph, mask, missing = load_graph(old)
        print(f"\nmissing-field report for {old.name} (written before the graph fields existed):")
        for m in missing:
            print("  -", m)
        print("  placeholders:", graph.placeholders, "|", w[0].message if w else "")
        assert set(graph.placeholders) == {"elevation", "base_demand", "length", "diameter"}
    print("\nall checks passed (no training was run)")


if __name__ == "__main__":
    main()
