# Waterboard

Leak detection and localization for municipal water networks. The user uploads a
network model and sensor readings, and the app shows where a leak is most likely,
as ranked search zones on a 3D city and pipe-network view. Users can also plant
leaks on any pipe and watch the system try to find them.

This is a ~20-hour project. Scope is deliberately narrow: one real network
(L-TOWN) plus portability to any network that has an EPANET model and pressure
sensors. It is **not** a system proven to work on arbitrary real cities, and the
UI and docs must not claim that.

## How it works

The approach is physics-informed: a hydraulic model, not a trained network.

1. **Load the network.** Parse the uploaded `.inp` (EPANET format) with WNTR to

   get nodes, pipes, coordinates and demands.
2. **Simulate the baseline.** Run the model with no leaks to get the pressures

   the sensors *should* read.
3. **Compute residuals.** Residual = measured − expected, per sensor per timestep.

   A leak shows up as a drop in residuals.
4. **Detect.** Flag when residuals drift beyond noise (CUSUM or a threshold on

   normalized residuals). This stage is network-agnostic.
5. **Localize by signature matching.** For each candidate node, simulate a leak

   there and record the pressure change at each sensor (its "signature"). Score
    each node by cosine similarity between its signature and the observed residual
    vector, then softmax into a probability map. No per-city training is needed,
    so this transfers to any uploaded network.
6. **Group into search zones.** Cluster high-probability nodes into contiguous

   zones ("Zone A: ~40 pipes, 1.2 km") that a crew can act on.

Out of scope unless the core is done early: a PINN, a GNN trained across
networks, and the pipe break-risk prior (age/material from Toronto or Kitchener
open data).

## Inputs

- `network.inp`: EPANET model of the network.
- `readings.csv`: sensor time series. The first column is `timestamp`. Every other
column is one sensor, **named by the node ID (pressure) or link ID (flow) it
sits on in the .inp**. That is how sensor locations are determined; there is no
separate locations file.
- Fallback for headers that don't match an .inp ID: list the unmatched columns and
let the user assign each one by clicking a node, or accept an optional
`sensor_name,node_id` mapping CSV.
- **.inp only (no readings):** offer the detectability map, sensor placement
suggestions, and click-to-plant-a-leak (the app simulates the "measurements").

## Data

- **L-TOWN / BattLeDIM 2020**: the preloaded demo network and real benchmark sensor
data. Check which parts are measured and which are simulated, and label the
data accordingly.
- **Other EPANET benchmarks** (C-Town, Net3, Modena): for showing portability.
Their sensors and leaks are synthetic (we place the sensors and simulate a hidden
leak plus noise), so results on them show the code is portable, not real-world
accuracy.
- No city publishes live SCADA data. "Live" in the dashboard means **replaying a
historical period**, and the UI labels it "replay".

## UX flow

### 1. Landing
- Light theme only (no dark mode).
- A clean white screen titled "Upload .inp file" with a large white drop zone.
Below it are a "Try the sample network" link and a list of recent networks.
- After upload, the whole background becomes the rendered 3D network. A floating
panel (top left) holds the network name, a network switcher, stats, the selected
element, files (replace .inp, attach .csv) and delete. There is no timeline.

### 2. Upload
- Drop zone for `network.inp` (required) and `readings.csv` (optional).
- Sensors are matched automatically from CSV headers. Show the matched count and
list unmatched columns with specific errors ("column `n54` doesn't match any
node"). Unmatched sensors can be assigned by clicking a node later.
- Downloadable sample CSV next to the drop zone.
- After upload the backend parses the .inp, runs the baseline simulation, and
computes the leak signatures in the background (show progress; this runs once per
network and is cached).

### 3. City view (3D, above ground)
- **Georeferenced .inp** (coordinates are lat/lon or a known projection): place the
network on a real basemap with 3D building extrusions from OpenStreetMap.
- **Not georeferenced** (arbitrary x/y, common in benchmark files): generate a
procedural city that matches the network. Streets follow pipe links, buildings
are placed near junctions, and building height or density scales with the
node's base demand. Label it "generated layout" so it isn't read as a real place.
- Overlay: sensor markers, and the current leak probability as a glow over the
city, so the risk is visible before diving in.

### 4. Transition to pipe view
- Clicking the city (or a "Show pipes" toggle) animates into the network: buildings
fade to translucent or drop away, the camera tilts down, and the pipes appear as
3D tubes below ground.
- Pipe thickness = diameter. Color = pressure or leak probability (toggle).
Sensors are highlighted nodes. Shaded blind spots show where sensors can't see a
leak.
- A toggle returns to the city view.

### 5. Focus on a pipe or node
- Hover highlights; click selects and the camera focuses on it.
- The side panel switches to that element: ID, diameter, length, material (if in the
.inp), current pressure or flow, and its leak probability.
- **Add leak** in the panel: size (small / medium / large, mapped to an emitter
coefficient or leak area), start time, and optional growth over time. Multiple
leaks are allowed.

### 6. Simulate and detect
- **Run** sends the planted leaks to the backend. WNTR simulates the network with
those leaks, and the result becomes the "measured" readings (plus sensor noise).
- Detection and localization then run on those readings as if they were real, and
the map updates with the probability heatmap and ranked search zones.
- Show a score against the planted leak: did detection fire, how long it took, and
whether the leak's pipe fell inside the top zone (and how far away the top
prediction was).
- Uploaded real readings go through the same detection path, without a planted
truth to score against.

### 7. Search zones (all modes)
- No timeline or time scrubber (removed by design).
- Ranked search zones in the panel ("Zone A: ~40 pipes, 1.2 km, most likely").
Clicking a zone flies the camera to it.

## Synthetic city generation

For cities with no .inp available. **This produces a plausible synthetic twin,
not the city's real network.** Leaks "found" on it are simulated, and the UI must
label it that way.

- No public source gives a city's real pipe layout, so it is approximated:
  - **Pipes:** the street network from OpenStreetMap (via OSMnx), since mains
mostly run under streets.
  - **Demands:** building footprints from OSM, weighted by population density.
  - **Elevations:** an open elevation model (e.g. SRTM).
  - **Sources:** reservoir and tank placement by heuristic (high ground, network
edges).
  - **Diameters:** sized from downstream demand.
- City statistics (population, per-capita water use, typical pipe material and age)
set the generation parameters. Bright Data scraping can collect these from public
sources. OSM data itself comes from its free APIs and needs no scraping.
- WNTR builds the model and writes it out as a normal `.inp`, so it then goes through
the same pipeline as an upload.
- Build this last: it's the highest-risk feature and isn't needed for the core
demo.

## Known limitations (state these honestly in the UI and write-up)

- **Coarse localization:** with a few dozen sensors you get an area, not a pipe.
- **Small leaks are invisible** below the pressure noise floor.
- **Model calibration:** an out-of-date `.inp` (wrong roughness, demands, valve
states) produces residuals that look like leaks. Output quality depends on the
uploaded model.
- **Sim-to-real gap:** signatures come from simulation. Report performance on real  
L-TOWN data separately from synthetic tests.

## Evaluation

- Held-out time periods on L-TOWN, scored against the BattLeDIM leak labels
(detection delay, localization distance or zone hit rate).
- If time allows, a second benchmark network with synthetic leaks.
- Always report real-data and synthetic results separately.

## Running the app

- Frontend: `npm --prefix frontend run dev` (Vite, port 5173). Proxies `/api` to the backend.
- Backend: `uv --directory backend run uvicorn app.main:app --port 8000` (FastAPI + WNTR 1.5,
  Python 3.12 managed by uv). Both are also in `.claude/launch.json`.
- The sample network is EPA Net3 (`frontend/public/samples/Net3.inp`): it has pumps, tanks and
  controls, so WNTR converges over a full week. Hand-made test grids without pumps tend to drain
  their tanks and stop converging.

## Simulation and model testing

- **Planting leaks:** each leak has a pipe, a size (hole area: 1 / 5 / 20 cm²), a break time
  and an optional repair time. Stored with the network in the browser.
- **`POST /api/simulate`** (`backend/app/simulate.py`): splits each leaking pipe at its midpoint,
  adds a WNTR leak there, and runs WNTRSimulator with pressure-driven demand, with and without
  the leaks. Sensors are spread automatically (farthest-point sampling, ~5% of junctions);
  "measured" = leak-run pressure + Gaussian noise (0.05 m). Returns pressures, flows, measured
  sensor series and ground-truth leak outflow per step. If WNTR stops converging, results are cut
  at the last good step and a plain-language warning is returned.
- **`POST /api/test`** (`backend/app/evaluate.py`): runs a detection model on a cached simulation
  and scores it against the planted leaks: detection delay, location error (distance from the
  predicted junction to the leak, in network coordinate units) and false alarms.
- **Models** live in `backend/app/models/` and implement `LeakModel.run(ModelInput) -> ModelOutput`
  (score per step, threshold, alarm events with a predicted junction). A model only sees sensor
  IDs/positions, noisy measurements, expected no-leak pressures and sensor accuracy, never the
  planted leaks. Register new models in `models/__init__.py`; the UI lists them automatically.
- `baseline` is a **placeholder** (RMS pressure-drop threshold + residual-weighted centroid). Its
  localization is coarse by design; the signature-matching model replaces it.
- Simulation results are cached in backend memory (last 20) and in frontend memory; both are lost
  on restart or reload.
