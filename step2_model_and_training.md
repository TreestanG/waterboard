# Step 2: Model and training loop

Scope: everything in this file lives inside `dataset/training/model/`. It only **reads** the unified parquet files that step 1 produces. It does not edit any step 1 file.

## 0. Avoiding overlap with step 1

Step 1 owns these names. Do not create or edit them:
`leakdb.py`, `ltown.py`, `wntr_sim.py`, `schema.py`, `splits.py`, `checks.py`, plus the folders `raw/` and `processed/`.

Step 2 owns only these files, all with the `train_` prefix, all inside `dataset/training/model/`:

```
dataset/training/model/
  train_config.py      # all settings in one dataclass
  train_windows.py     # parquet -> sliding windows -> torch/numpy arrays
  train_models.py      # the models (baseline + neural)
  train_loop.py        # training and validation loop
  train_metrics.py     # detection and localization metrics
  train_run.py         # command-line entry point
  train_fake_data.py   # fake data generator for testing before real data exists
  train_runs/          # outputs (one subfolder per run)
```

## 1. Contract with step 1 (read-only)

Input: `dataset/training/processed/<dataset>/<scenario_id>.parquet`

Expected columns (as defined in step 1):
- `timestamp`
- `p_<node>` for pressure sensors, `q_<link>` for flow sensors
- `leak_active` (0/1), `leak_node`, `leak_size`, `leak_type`

Also read `network.json` per dataset for the node list and sampling interval.

Splits: step 1's `splits.py` should write a `splits.json` of the form `{"train": [...], "val": [...], "test": [...]}` listing scenario ids per dataset. Step 2 only reads this file. **Check this with step 1's owner.** Until it exists, `train_fake_data.py` writes its own fake `splits.json` to the fake data folder.

## 2. Windowing (`train_windows.py`)

For each scenario, slide a window over the time series.
- Window length `W` (start with 24 hours of steps, tune later).
- Stride `S` (start with 1 hour of steps).
- Window input: array of shape `[W, n_sensors]`.
- Window label (detection): 1 if `leak_active` is 1 for the last step of the window, else 0.
- Window label (localization): `leak_node` of the last step (only used when the label is 1).

Preprocessing, applied per scenario so no information leaks across scenarios:
1. Subtract a per-sensor baseline (rolling median over a long window) so the model sees changes, not absolute levels.
2. Normalize each sensor with mean and std computed from the **train** scenarios only. Save these stats to the run folder.
3. Fill small gaps by forward fill. Drop windows with long gaps.

Keep the first version simple. One scenario file in, windows out, no cross-scenario caching tricks.

## 3. Models (`train_models.py`)

Build in this order. Do not skip the baseline.

1. **`GBBaseline`**: gradient boosting (scikit-learn `HistGradientBoostingClassifier`) on summary features of each window: per sensor min, max, mean, last value, slope. Binary detection only. This is the number to beat.
2. **`CNNDetector`**: small 1D CNN over `[n_sensors, W]`. 2-3 conv blocks, global pooling, linear head, one logit for detection.
3. **`CNNLocalizer`** (second stage): same backbone, second head that outputs one logit per node in the network. Only trained on windows where a leak is active. The head size depends on the network, so this model is tied to one network. Say so in results.

Leave GNN and sequence models (GRU) as optional extensions after the baselines work.

Keep model construction in a function `build_model(cfg, n_sensors, n_nodes)` so the loop does not care which model it is.

## 4. Training loop (`train_loop.py`)

Detection:
- Loss: binary cross-entropy with `pos_weight` set from the train leak fraction, or focal loss as a switch in the config.
- Optimizer: AdamW, learning rate 1e-3, cosine decay.
- Batch size 256 (adjust to memory).
- Early stopping on validation **PR-AUC** (not loss, not accuracy), patience about 5 epochs.
- Save best checkpoint and config to `train_runs/<run_name>/`.

Localization:
- Loss: cross-entropy over nodes, computed only on leak-active windows.
- Optional: soft labels that give some probability to neighbor nodes, because exact location is hard and the target is a map.

Skeleton:

```python
def train(cfg):
    set_seed(cfg.seed)
    train_ds, val_ds = make_datasets(cfg)          # train_windows.py
    model = build_model(cfg, ...)                  # train_models.py
    opt = AdamW(model.parameters(), lr=cfg.lr)
    best = -1
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            loss = criterion(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        score = evaluate(model, val_loader)        # train_metrics.py
        if score["pr_auc"] > best:
            best = score["pr_auc"]
            save_checkpoint(model, cfg, score)
        if no_improvement(patience=cfg.patience):
            break
```

## 5. Metrics (`train_metrics.py`)

Detection (computed on held-out scenarios):
- PR-AUC and F1 at a chosen threshold
- Detection delay: time from true leak start to first alarm (median and 90th percentile)
- False alarms per month on no-leak scenarios
- Missed leaks (fraction of leak scenarios never flagged)

Localization:
- Top-1 and top-5 accuracy over nodes
- Localization error: hydraulic or graph distance (in pipes, and in meters if coordinates exist) between the predicted node and the true one

Report results separately for: (a) held-out scenarios on a seen network, (b) held-out whole network. Never mix them in one number.

## 6. Config (`train_config.py`)

One dataclass so every run is reproducible. Fields to include:
`datasets`, `window_len`, `stride`, `sensor_subset` (all or a list, so you can test sparse sensors), `model` (`gb`, `cnn`, `cnn_loc`), `lr`, `epochs`, `batch_size`, `pos_weight`, `loss` (`bce` or `focal`), `seed`, `run_name`, `data_root`.

Save the config next to every checkpoint.

## 7. Entry point (`train_run.py`)

```
python train_run.py --model gb  --datasets leakdb_net1 --run_name gb_net1_v1
python train_run.py --model cnn --datasets leakdb_net1 ltown --run_name cnn_v1
```

Outputs in `train_runs/<run_name>/`: `config.json`, `norm_stats.json`, `best.pt` (or `best.pkl` for gb), `metrics.json`, `val_curves.png`.

## 8. Test before the real data exists (`train_fake_data.py`)

Generate 50 fake scenarios in the step 1 schema in a **separate** folder (`dataset/training/model/train_runs/fake_data/`), never in `processed/`:
- 10 sensors, 5-minute steps, 3 days.
- Daily sinusoid on every sensor plus noise.
- In leak scenarios, a pressure dip at the sensors near the leak starting at a random time.
- Fake `splits.json`.

Acceptance test: the baseline gets clearly above chance on fake data, and the CNN is at least as good. If not, the bug is in the plumbing, not the data.

## 9. Done when

- [ ] Fake-data run finishes end to end and writes all outputs.
- [ ] Baseline and CNN both train on fake data and beat chance.
- [ ] Same code runs on real step 1 files with only a config change.
- [ ] Held-out-network evaluation is reported separately.
- [ ] Results table in the README with detection and localization numbers and the baseline for comparison.

## 10. Risks to watch

- **Label timing:** if leak start times in the data are approximate, delay metrics will be noisy.
- **Leakage across windows:** overlapping windows from the same scenario must stay in the same split.
- **Localization across networks:** the node head does not transfer. Do not claim it does.
- **Sensor realism:** models trained on every-node sensors will not work with a realistic sparse set. Use `sensor_subset` early.
- **Simulation gap:** all of this is trained on simulated leaks. Report L-Town 2018 as the closest-to-real check.
