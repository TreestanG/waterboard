"""All settings for a training run, in one dataclass (step 2)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
RUNS_DIR = MODEL_DIR / "train_runs"
FAKE_DATA_DIR = RUNS_DIR / "fake_data"
DEFAULT_DATA_ROOT = MODEL_DIR.parent / "processed"  # step 1 output, read-only


@dataclass
class TrainConfig:
    # data
    datasets: list = field(default_factory=lambda: ["fake_net"])  # trained on (split by scenario)
    holdout_datasets: list = field(default_factory=list)  # whole networks, evaluation only
    data_root: str = str(DEFAULT_DATA_ROOT)
    sensor_subset: object = "all"  # "all" or a list of column names, e.g. ["p_N3", "q_L2"]
    max_scenarios: int = 0  # 0 = all; otherwise cap per dataset and split (memory guard)

    # windowing. window_len / stride are in steps; 0 = derive from the *_hours fields
    # using the sampling interval of the first dataset.
    window_len: int = 0
    stride: int = 0
    window_hours: float = 24.0
    stride_hours: float = 1.0
    baseline_hours: float = 168.0  # rolling-median baseline length
    max_gap_steps: int = 6  # forward-fill at most this many missing steps

    # model
    model: str = "cnn"  # gb | cnn | cnn_loc | lstm_gnn
    init_from: str = ""  # cnn_loc only: detector run name to copy the backbone from
    dropout: float = 0.2
    # lstm_gnn only
    hidden_size: int = 64  # LSTM hidden size and GNN width
    num_layers: int = 1  # LSTM layers (1-2)
    gnn_layers: int = 2  # message-passing layers (2-3)
    graph_pool: str = "max"  # detection logit from node logits: max | attn
    use_flow: bool = False  # add a flow channel to the per-node input
    grad_clip: float = 1.0  # max gradient norm, 0 = off
    loc_weight: float = 1.0  # weight of the leak-node loss next to the detection loss
    workers: int = 0  # processes for loading scenarios, 0 = one per core (max 16)

    # optimisation
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 40
    batch_size: int = 256
    patience: int = 5
    loss: str = "bce"  # bce | focal
    focal_gamma: float = 2.0
    pos_weight: float = 0.0  # 0 = n_negative / n_positive of the train windows
    loc_soft_tau: float = 0.0  # cnn_loc: >0 spreads the label over neighbours, exp(-hops/tau)
    threshold: float = 0.0  # 0 = pick the best-F1 threshold on validation

    # the test split and held-out networks are only loaded and scored when this is set
    eval_test: bool = False
    time_budget_hours: float = 0.0  # lstm_gnn: >0 lowers `epochs` so the whole run fits in this many hours

    seed: int = 0
    run_name: str = "run"

    def run_dir(self) -> Path:
        return RUNS_DIR / self.run_name

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        return cls(**json.loads(Path(path).read_text()))
