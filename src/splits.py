"""Train / val / test splits. Always by whole scenario, with whole networks held out.

A split is a dict of lists of (dataset, scenario_id) pairs:
    train   scenarios of the training networks
    val     held-out scenarios of the training networks
    test    every scenario of the held-out networks (generalization test)

L-Town is test / inference only: it is never allowed in train or val.
"""
import numpy as np

from .schema import load_dataset

TEST_ONLY = {"ltown"}


def make_split(train_datasets: list[str], test_datasets: list[str], val_fraction: float = 0.15,
               seed: int = 0) -> dict[str, list[tuple[str, str]]]:
    overlap = set(train_datasets) & set(test_datasets)
    if overlap:
        raise ValueError(f"Datasets in both train and test: {sorted(overlap)}")
    forbidden = set(train_datasets) & TEST_ONLY
    if forbidden:
        raise ValueError(f"{sorted(forbidden)} is test/inference only and cannot be used for training")

    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for name in train_datasets:
        ids = list(load_dataset(name).scenario_ids)
        rng.shuffle(ids)
        n_val = int(round(val_fraction * len(ids))) if len(ids) > 1 else 0
        split["val"] += [(name, s) for s in sorted(ids[:n_val])]
        split["train"] += [(name, s) for s in sorted(ids[n_val:])]
    for name in test_datasets:
        split["test"] += [(name, s) for s in load_dataset(name).scenario_ids]

    check_split(split)
    return split


def check_split(split: dict) -> None:
    """No scenario in two parts, no held-out network in train/val, no test-only data in train/val."""
    seen = {}
    for part, items in split.items():
        for item in items:
            if item in seen:
                raise AssertionError(f"{item} is in both {seen[item]} and {part}")
            seen[item] = part
    fit_networks = {d for d, _ in split["train"] + split["val"]}
    held_out = {d for d, _ in split["test"]}
    if fit_networks & held_out:
        raise AssertionError(f"Held-out networks also used for fitting: {sorted(fit_networks & held_out)}")
    if fit_networks & TEST_ONLY:
        raise AssertionError(f"Test-only datasets used for fitting: {sorted(fit_networks & TEST_ONLY)}")


def load_split_part(items: list[tuple[str, str]]):
    """Yield (dataset, scenario_id, DataFrame) for one part of a split."""
    cache = {}
    for name, sid in items:
        ds = cache.setdefault(name, load_dataset(name))
        yield name, sid, ds.load(sid)
