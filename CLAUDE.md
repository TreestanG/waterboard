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

### 1. Dashboard (landing)
- Opens on the L-TOWN demo already loaded in the city view, replay running. Never
land on an empty upload form.
- Sidebar: network list (L-TOWN demo, uploaded networks, generated cities), an
**Upload network** button, and a **Generate from a city** button (see Synthetic
city generation).

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

### 7. Timeline and alerts (all modes)
- Time scrubber under the 3D view; the pipe colors and heatmap follow it.
- Event feed ("possible leak started 14 Mar, growing"). Clicking an event jumps the
scrubber and camera to it.
- Ranked search zones panel ("Zone A: ~40 pipes, 1.2 km, most likely"). Clicking a
zone flies the camera to it.

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
