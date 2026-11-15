"""Command-line entry point.

  python train_run.py --fake --model gb  --run_name gb_fake
  python train_run.py --fake --model cnn --run_name cnn_fake
  python train_run.py --model gb  --datasets leakdb_net1 --run_name gb_net1_v1
  python train_run.py --model cnn --datasets leakdb_net1 ltown --holdout_datasets leakdb_hanoi --run_name cnn_v1
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields

from train_config import FAKE_DATA_DIR, TrainConfig
from train_loop import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="use (and generate if missing) the fake data in train_runs/fake_data")
    for f in fields(TrainConfig):
        default = f.default_factory() if callable(getattr(f, "default_factory", None)) else f.default
        if f.name in ("datasets", "holdout_datasets"):
            ap.add_argument(f"--{f.name}", nargs="*", default=None)
        elif f.name == "sensor_subset":
            ap.add_argument("--sensor_subset", nargs="+", default=None, help='"all" or a list of sensor columns')
        elif isinstance(default, bool):
            ap.add_argument(f"--{f.name}", type=lambda v: v.lower() in ("1", "true", "yes"), nargs="?", const=True, default=None)
        else:
            ap.add_argument(f"--{f.name}", type=type(default), default=None)
    args = vars(ap.parse_args())

    fake = args.pop("fake")
    kw = {k: v for k, v in args.items() if v is not None}
    if kw.get("sensor_subset") == ["all"]:
        kw["sensor_subset"] = "all"
    if fake:
        kw.setdefault("data_root", str(FAKE_DATA_DIR))
        kw.setdefault("datasets", ["fake_net"])
        if kw["data_root"] == str(FAKE_DATA_DIR) and not (FAKE_DATA_DIR / "fake_net").exists():
            from train_fake_data import generate
            generate()
    cfg = TrainConfig(**kw)
    metrics = train(cfg)
    print(json.dumps(metrics, indent=2))
    print("outputs in", cfg.run_dir())


if __name__ == "__main__":
    main()
