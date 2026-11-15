# Step 1: Data pipeline (get and process leak datasets)

Goal: one function that loads any supported dataset and returns data in the same format, so every later step (training, dashboard) is dataset-agnostic.

Time budget: about 3-4 hours of the 20.

---

## 1. Which datasets, and what each one gives you

| Dataset | Network | Has sensor readings? | Has leak labels? | What you must do |
|---|---|---|---|---|
| **LeakDB (Net1, Hanoi)** | Tiny (Net1: 10 nodes, Hanoi: 32) | Yes. Pressure, demand and flow for every node/link, 30-min steps | Yes. Labels.csv per scenario, plus leak location and type | Download and parse. No simulation needed |
| **L-Town / BattLeDIM** | 782 junctions, 905 pipes | Yes. 2018 and 2019 SCADA, 5-min steps (33 pressure, 3 flow, 1 tank level, 82 AMR) | 2018: yes (leak report with repair times). 2019: evaluation set, check whether answers are published | Download and parse. Run WNTR for extra leak scenarios |
| **Net3 (EPANET example)** | Small/medium | **No** | **No** | **Run WNTR** to simulate leaks and readings |
| **Other benchmark networks** (Anytown, Balerma, Fossolo, Modena, Kentucky set, etc.) | Varies | **No** | **No** | **Run WNTR** to simulate leaks and readings |

Rule of thumb: if a dataset is only an `.inp` file, it has no leaks and no readings. Everything has to be generated with WNTR.

Notes to verify when you download (details recalled from docs, not tested):
- Exact folder/file names inside LeakDB scenarios.
- Which LeakDB sensors exist (it gives every node and link, which is far more than a real utility has).
- Whether L-Town 2019 leak answers are public.
- Licenses for each dataset.

---

## 2. Setup

```
pip install wntr epyt-flow pandas numpy pyarrow scikit-learn
```

- **WNTR**: loads `.inp` files and runs hydraulic simulations with leaks.
- **EPyT-Flow**: can load LeakDB scenarios (original data or as editable scenario configs) and simulate.
- **WaterBenchmarkHub** (optional): has loaders for LeakDB and others.

Suggested repo layout:

Everything dataset-related goes inside the existing `dataset/training/` folder. Create these subfolders there and leave the rest of the project alone:

```
dataset/
  training/
    raw/            # untouched downloads
      leakdb_net1/
      leakdb_hanoi/
      ltown/
      networks/     # .inp files (Net3, etc.)
    processed/      # unified parquet files
```

Code lives outside the dataset folder:

```
src/
  loaders/
    leakdb.py
    ltown.py
    wntr_sim.py   # simulate leaks for any .inp
  schema.py
  splits.py
  checks.py
notebooks/
```

---

## 3. Unified format (the main design decision)

Every dataset becomes the same thing. One file per scenario:

`dataset/training/processed/<dataset>/<scenario_id>.parquet`

Columns:
- `timestamp`
- `p_<node>`: pressure at each sensor node
- `q_<link>`: flow at each sensor link
- `demand_<node>`: only where metered (optional)
- `leak_active`: 0/1 for each time step
- `leak_node`: node id (or null when no leak)
- `leak_size`: size of the leak (or null). Keep the units noted
- `leak_type`: `abrupt` / `incipient` / null

Plus one metadata file per dataset: `network.json` (nodes, links, coordinates, which are sensors, sampling interval, units) and the `.inp` path.

Keep units consistent (pick m for pressure, m3/h for flow) and record conversions.

---

## 4. Tasks

### 4a. LeakDB loader
1. Download Net1 and Hanoi (the full set is large: roughly 8 GB for Net1 and 25 GB for Hanoi. Start with about 100 scenarios each).
2. For each scenario, read pressures, flows, demands, Labels.csv, and the Leaks info.
3. Convert to the unified schema. Fill `leak_node`, `leak_size`, `leak_type` from the leak info, aligned to the 30-minute steps.
4. Keep the scenario's `.inp` path in metadata.
5. Check: label counts per scenario, fraction of scenarios with no leak, timestamps are contiguous.

### 4b. L-Town loader
1. Download the BattLeDIM data: `L-Town.inp`, 2018 SCADA, 2019 SCADA, 2018 leak report.
2. Parse the SCADA files (5-min steps) into the unified schema. Pressure sensors are the 33 given nodes.
3. Build 2018 labels from the leak report. Needs care: the report gives repair times, so check what it says about leak start times and sizes before assuming anything.
4. L-Town is test/inference only. Neither year goes into training (2018 is the historical set, 2019 the evaluation set).
5. Check: sensor count is 33 pressure, 3 flow, and no unexplained gaps.

### 4c. WNTR simulator (for networks with no data, and to add data to L-Town)
Function: `simulate(inp_path, n_scenarios, sensor_nodes, ...) -> unified parquet files`.

For each scenario:
1. Load the `.inp` with WNTR.
2. Randomize demands (base demand times a daily pattern plus noise).
3. Pick a leak node at random. Pick a leak size and a start time. Choose abrupt or incipient.
4. Run the simulation with pressure-driven demand.
5. Record pressure and flow at the chosen sensor nodes only (so it matches a realistic utility), add sensor noise.
6. Write the unified file with labels.

Always also generate no-leak scenarios (the model needs to learn what normal looks like).

### 4d. Splits (do this now, it prevents mistakes later)
- Split by **scenario**, never by random rows. A scenario's rows are highly correlated.
- Keep **whole networks** held out for the generalization test (for example train on Net1 + Net3, test on Hanoi).
- L-Town is never in train or val. Both years are test/inference only.

### 4e. Sanity checks (`checks.py`)
- Plot a pressure sensor with the leak window shaded. A visible pressure drop should appear after the leak starts.
- Check the leak-active fraction. Expect it to be small.
- Check no NaNs or timestamp gaps.
- Compare your WNTR output against a real LeakDB scenario on the same network to confirm it looks similar.

---

## 5. Done when

- [ ] `load_dataset("leakdb_net1")`, `load_dataset("leakdb_hanoi")`, and `load_dataset("ltown")` return the unified format.
- [ ] `simulate("Net3.inp", ...)` produces files in the same format.
- [ ] Split function holds out whole networks and whole scenarios.
- [ ] Sanity plots look right for one scenario per dataset.
- [ ] Short README listing sources, licenses, and units.

## 6. Known limitations to write down now

- LeakDB networks are tiny and not realistic city scale.
- L-Town is a virtual network, and its data is simulated.
- WNTR leaks are idealized. Real leaks are messier.
- Simulated sensors are cleaner than real ones. Add noise and test the sensitivity to it.
