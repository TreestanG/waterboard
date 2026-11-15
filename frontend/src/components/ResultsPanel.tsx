import { useState } from "react";
import type { ModelInfo, SimResult, TestResult } from "../api";
import type { Selection } from "./NetworkMap";
import LineChart, { type ChartSeries } from "./LineChart";
import { formatStartHour } from "../leaks";

type Props = {
  sim: SimResult;
  stale: boolean;
  selection: Selection;
  models: ModelInfo[];
  test: TestResult | null;
  testRunning: boolean;
  testError: string | null;
  onRunTest: (modelId: string) => void;
  onClose: () => void;
};

/** Distinct, readable line colors for per-leak series. */
const LEAK_COLORS = ["#ec265e", "#f26e42", "#7c5cff", "#12a8b6", "#22a565", "#b7791f"];
const BASE = "#8a93a0";
const LEAK_RUN = "#12161c";
const MEASURED = "#12a8b6";

function hours(h: number | null | undefined): string {
  return h == null ? "—" : h < 1 ? `${Math.round(h * 60)} min` : `${h.toFixed(h % 1 ? 2 : 0)} h`;
}

export default function ResultsPanel(props: Props) {
  const { sim, selection, test } = props;
  const [model, setModel] = useState(props.models[0]?.id ?? "baseline");
  const step = sim.timesH.length > 1 ? sim.timesH[1] - sim.timesH[0] : 0;
  const duration = sim.timesH[sim.timesH.length - 1] ?? 0;
  const dryLeaks = sim.leaks.filter((l) => Math.max(0, ...l.flowLps) <= 0);
  const notStarted = (l: (typeof sim.leaks)[number]) => l.startHour > duration;
  const leakStarts = sim.leaks.map((l) => ({ at: l.startHour, label: "Leak start", color: "rgba(236,38,94,0.35)" }));

  // Detail chart: the selected pipe's flow or node's pressure; otherwise the first sensor.
  let detailTitle = "";
  let detailSeries: ChartSeries[] = [];
  let detailUnit = "";
  const selectedLeak = selection?.type === "link" ? sim.leaks.find((l) => l.linkId === selection.link.id) : undefined;
  if (selectedLeak) {
    // A leaking pipe is split at the leak, so show what leaves through the hole and what arrives
    // from each end, rather than the flow in one half of the pipe.
    detailTitle = `Leak on pipe ${selectedLeak.linkId}`;
    detailUnit = "Flow (L/s)";
    detailSeries = [
      { label: "Leak outflow", values: selectedLeak.flowLps, color: "#ec265e", step: true },
      ...(selectedLeak.inflowFromStartLps
        ? [
            { label: `In from ${selectedLeak.startNode} end`, values: selectedLeak.inflowFromStartLps, color: LEAK_RUN },
            { label: `In from ${selectedLeak.endNode} end`, values: selectedLeak.inflowFromEndLps, color: "#7c5cff" },
          ]
        : []),
      ...(sim.flow[selectedLeak.linkId]
        ? [{ label: "Pipe flow without leaks", values: sim.flow[selectedLeak.linkId].baseline, color: BASE, dashed: true }]
        : []),
    ];
  } else if (selection?.type === "link" && sim.flow[selection.link.id]) {
    const f = sim.flow[selection.link.id];
    detailTitle = `Flow in ${selection.link.kind} ${selection.link.id}`;
    detailUnit = "Flow (L/s)";
    detailSeries = [
      { label: "No leaks", values: f.baseline, color: BASE, dashed: true },
      { label: "With leaks", values: f.leak, color: LEAK_RUN },
    ];
  } else {
    const node = selection?.type === "node" && sim.pressure[selection.node.id] ? selection.node.id : sim.sensors[0];
    const p = node ? sim.pressure[node] : undefined;
    if (node && p) {
      const isSensor = sim.sensors.includes(node);
      detailTitle = `Pressure at ${isSensor ? "sensor" : "junction"} ${node}`;
      detailUnit = "Pressure (m)";
      detailSeries = [
        { label: "Expected (no leaks)", values: p.baseline, color: BASE, dashed: true },
        { label: "Actual (with leaks)", values: p.leak, color: LEAK_RUN },
        ...(isSensor ? [{ label: "Measured (noisy)", values: sim.measured[node], color: MEASURED }] : []),
      ];
    }
  }

  return (
    <aside className="results">
      <div className="results-head">
        <div>
          <div className="results-eyebrow">Simulation results</div>
          <div className="results-title">
            {sim.leaks.length} leak{sim.leaks.length === 1 ? "" : "s"} · {hours(duration)}
            {sim.requestedHours != null && duration < sim.requestedHours - 1e-6 && <span className="muted"> of {hours(sim.requestedHours)}</span>}
          </div>
        </div>
        <button className="round-btn sm" onClick={props.onClose} aria-label="Close results">
          <svg viewBox="0 0 24 24" aria-hidden>
            <path d="M6 6l12 12M18 6 6 18" />
          </svg>
        </button>
      </div>

      <div className="results-body">
        <div className="metrics">
          <div className="metric">
            <span className="metric-label">Time step</span>
            <span className="metric-value">{Math.round(step * 60)}m</span>
          </div>
          <div className="metric">
            <span className="metric-label">Sensors</span>
            <span className="metric-value">{sim.sensors.length}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Peak loss</span>
            <span className="metric-value">{Math.max(0, ...sim.totalLeakLps).toFixed(1)}</span>
          </div>
        </div>
        <p className="hint">
          WNTR, pressure-driven demand. Sensors are spread automatically; measurements add ±{sim.noiseM} m noise. Peak loss in L/s.
        </p>

        {(sim.warnings ?? []).map((w) => (
          <p key={w} className="notice">
            {w}
          </p>
        ))}
        {props.stale && <p className="notice">Leaks or settings changed since this run. Run the simulation again to update.</p>}
        {dryLeaks.map((l) => (
          <p key={l.id} className="notice">
            {notStarted(l)
              ? `The leak on pipe ${l.linkId} breaks at ${formatStartHour(l.startHour)}, after the results end, so it has no data.`
              : `The leak on pipe ${l.linkId} produced no flow: pressure there averaged ${l.meanPressureM == null ? "—" : `${l.meanPressureM} m`} while it was open, and water can't escape below 0 m.`}
          </p>
        ))}

        <section>
          <h3>Leak flow (ground truth)</h3>
          {/* One small chart per leak, each on its own scale, so a 2 L/s leak isn't flattened
              against a 45 L/s one. */}
          <div className="leak-charts">
            {sim.leaks.map((l, i) => (
              <LineChart
                key={l.id}
                times={sim.timesH}
                leftLabel={`Outflow (L/s) · own scale`}
                height={96}
                markers={[{ at: l.startHour, label: "Break", color: "rgba(236,38,94,0.35)" }, ...(l.endHour != null ? [{ at: l.endHour, label: "Repair", color: "rgba(34,165,101,0.5)" }] : [])]}
                series={[
                  {
                    label: `${l.linkId} · ${formatStartHour(l.startHour)}${l.endHour != null ? `–${formatStartHour(l.endHour)}` : ""} · peak ${Math.max(0, ...l.flowLps).toFixed(1)} L/s`,
                    values: l.flowLps,
                    color: LEAK_COLORS[i % LEAK_COLORS.length],
                    step: true,
                  },
                ]}
              />
            ))}
          </div>
        </section>

        {detailSeries.length > 0 && (
          <section>
            <h3>{detailTitle}</h3>
            <LineChart times={sim.timesH} leftLabel={detailUnit} series={detailSeries} markers={leakStarts} />
            <p className="hint">Select a pipe or junction on the map to chart it.</p>
          </section>
        )}

        <section className="test-section">
          <h3>Test a detection model</h3>
          <div className="test-row">
            <select className="pill-select" value={model} onChange={(e) => setModel(e.target.value)} aria-label="Model">
              {(props.models.length ? props.models : [{ id: "baseline", name: "Residual baseline (placeholder)", description: "" }]).map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
            <button className="btn btn-primary" disabled={props.testRunning || props.stale} onClick={() => props.onRunTest(model)}>
              {props.testRunning ? "Running…" : "Run test model"}
            </button>
          </div>
          {props.testError && <p className="error">{props.testError}</p>}

          {test && (
            <>
              {(test.notes ?? []).map((n) => (
                <p key={n} className="notice">
                  {n}
                </p>
              ))}
              <div className="metrics">
                <div className="metric">
                  <span className="metric-label">Detected</span>
                  <span className="metric-value">
                    {test.summary.detected}/{test.summary.leaks}
                  </span>
                </div>
                <div className="metric">
                  <span className="metric-label">Mean delay</span>
                  <span className="metric-value">{hours(test.summary.meanDelayHours)}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">False alarms</span>
                  <span className="metric-value">{test.summary.falseAlarms}</span>
                </div>
              </div>

              <LineChart
                times={test.timesH}
                leftLabel="True leak outflow (L/s)"
                rightLabel={test.scoreLabel}
                rightThreshold={{ value: test.threshold, label: "Alarm threshold" }}
                markers={[
                  ...leakStarts,
                  ...test.events.map((e) => ({ at: e.timeHour, label: "Alarm", color: "rgba(18,22,28,0.55)" })),
                ]}
                series={[
                  { label: "Ground truth (total)", values: test.groundTruth.totalLeakLps, color: "#ec265e", step: true },
                  { label: "Model score", values: test.score, color: "#12a8b6", right: true },
                ]}
                height={180}
              />
              <p className="hint">
                Pink: actual leak flow. Teal: model score (right axis); the dashed line is its alarm threshold. Gray lines mark alarms.
              </p>

              <table className="result-table">
                <thead>
                  <tr>
                    <th>Pipe</th>
                    <th>Starts</th>
                    <th>Detected</th>
                    <th>Delay</th>
                    <th>Predicted</th>
                    <th>Location error</th>
                  </tr>
                </thead>
                <tbody>
                  {test.perLeak.map((r) => (
                    <tr key={r.leakId}>
                      <td className="mono">{r.linkId}</td>
                      <td>{formatStartHour(r.startHour)}</td>
                      <td>{r.detected ? formatStartHour(r.detectedHour ?? 0) : <span className="miss">Missed</span>}</td>
                      <td>{hours(r.delayHours)}</td>
                      <td className="mono">{r.predictedNode ?? "—"}</td>
                      <td>{r.locationError == null ? "—" : `${Math.round(r.locationError)} u`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {test.events.some((e) => e.candidates?.length) && (
                <table className="result-table">
                  <thead>
                    <tr>
                      <th>Alarm</th>
                      <th>Most likely junctions (share of location probability)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {test.events.map((e, i) => (
                      <tr key={i}>
                        <td>{formatStartHour(e.timeHour)}</td>
                        <td className="mono">
                          {(e.candidates ?? []).map((c) => `${c.node} ${Math.round(c.probability * 100)}%`).join(" · ") || e.node}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <p className="hint">
                {test.model.description} Location error is the distance from the model's predicted junction to the leak, in the
                network's coordinate units (u).
              </p>
            </>
          )}
        </section>
      </div>
    </aside>
  );
}
