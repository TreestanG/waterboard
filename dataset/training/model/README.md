# Step 2: model and training

Everything here is self-contained and only reads the step 1 parquet files. See `step2_model_and_training.md` in the repo root for the plan.

## Run

```
python train_run.py --fake --model gb  --holdout_datasets fake_net_b --run_name gb_fake
python train_run.py --fake --model cnn --holdout_datasets fake_net_b --run_name cnn_fake
python train_run.py --fake --model cnn_loc --init_from cnn_fake --run_name cnn_loc_fake
```

`--fake` generates `train_runs/fake_data/` if it is missing and points `data_root` at it. On real step 1 data, drop `--fake` and name the datasets:

```
python train_run.py --model cnn --datasets leakdb_net1 --holdout_datasets leakdb_hanoi --run_name cnn_v1
```

Each run writes `config.json`, `norm_stats.json`, `best.pt` (`best.pkl` for gb), `metrics.json` and `val_curves.png` to `train_runs/<run_name>/`.

LSTM + GNN on the pipe graph (detection and leak-node localization, one model for every network):

```
python train_run.py --model lstm_gnn --datasets leakdb_net1 leakdb_hanoi sim_net1 sim_net3 --epochs 10 --patience 10 --run_name lstm_gnn_test_v1
```

This path keeps residuals in memory and slices windows per batch, so it scales with time steps, not windows. It also writes `splits.json` and `val_predictions.parquet` (one row per validation window).

**Splits** come from a hash of the dataset and scenario id (`train_windows.stable_split`, 70/15/15), unless a `splits.json` is present. A scenario's part never changes when more scenarios are added. Changing `--seed` changes the split.

## Results so far: fake data only

No real step 1 output existed when this was written (`processed/` was empty), so these numbers only show that the plumbing works. They say nothing about real leaks.

Detection, 24 h windows, 1 h stride, threshold = best F1 on validation. Test sets are tiny (7 and 12 scenarios).

| Model | Test set | PR-AUC | Chance | F1 | Median delay (h) | P90 delay (h) | False alarms / month | Missed leaks |
|---|---|---|---|---|---|---|---|---|
| GB baseline | (a) held-out scenarios, seen network `fake_net` | 0.70 | 0.37 | 0.75 | 2.7 | 6.1 | 14.7 | 0 of 5 |
| CNN detector | (a) held-out scenarios, seen network `fake_net` | 1.00 | 0.37 | 0.98 | 0.8 | 1.9 | 0.0 | 0 of 5 |
| GB baseline | (b) held-out network `fake_net_b` | 0.64 | 0.41 | 0.57 | 0.6 | 5.3 | 29.4 | 1 of 9 |
| CNN detector | (b) held-out network `fake_net_b` | 0.64 | 0.41 | 0.59 | 0.5 | 0.8 | 14.7 | 0 of 9 |

Localization (`cnn_loc`, backbone initialised from `cnn_fake`), leak-active windows of held-out scenarios on `fake_net` (12 nodes, 128 windows from 5 leaks):

| Metric | CNN localizer | Uniform random guess |
|---|---|---|
| Top-1 | 0.68 | 0.08 |
| Top-5 | 0.76 | 0.42 |
| Mean error, pipes | 0.38 | 2.15 |
| Mean error, m (straight line) | 67 | 353 |

The node head is tied to one network. It is not evaluated on `fake_net_b` and does not transfer.

## Things to know

- **Held-out network is close to chance.** Sensor channels have no shared meaning across networks, so neither model transfers. Networks with fewer sensors are zero-padded to the widest one.
- **Normalisation on an unseen network** uses the average train mean/std per sensor kind (`p_`, `q_`), since it has no train scenarios of its own.
- **Rolling-median baseline absorbs long leaks.** Once a leak has lasted longer than half of `baseline_hours` (default 168), the baseline moves to the leak level and the residual returns to zero while the label stays 1. Expect this to matter on L-Town.
- **Window length is in steps**, derived from the first dataset's sampling interval. Mixing 30-minute LeakDB and 5-minute L-Town in one run gives windows of different durations; a warning is printed.
- **Memory:** all windows are held in RAM. Use `--max_scenarios` on the full LeakDB sets.
- **Contract assumptions not yet checked against step 1:** `splits.json` is read from `<data_root>/<dataset>/splits.json` or `<data_root>/splits.json` (falls back to a seeded 70/15/15 split by scenario with a warning); `network.json` is read for `nodes`, `links` and `sampling_interval_s`, with fallbacks if they are missing.
- **Distance in metres** is straight-line between node coordinates, not pipe length.
