import { useRef, useState } from "react";
import type { Leak, LeakSize, Network } from "../types";
import type { SimSettings } from "../api";
import type { FileState } from "../App";
import type { Selection } from "./NetworkMap";
import { formatBytes } from "../files";
import { LEAK_SIZES, formatStartHour } from "../leaks";
import LeakEditor from "./LeakEditor";

/** Leak start-time range when the .inp doesn't set a simulation duration. */
const DEFAULT_DURATION_HOURS = 24;

type Props = {
  network: Network;
  file: FileState;
  networks: Network[];
  selection: Selection;
  onClearSelection: () => void;
  onSwitch: (id: string) => void;
  onNewUpload: () => void;
  onRename: (name: string) => void;
  onReplaceInp: (file: File) => Promise<string | null>;
  onAttachCsv: (file: File) => Promise<string | null>;
  onRemoveCsv: () => void;
  onDelete: () => void;
  onSaveLeak: (linkId: string, size: LeakSize, startHour: number, endHour: number | null) => void;
  onRemoveLeak: (leakId: string) => void;
  onFocusLink: (linkId: string) => void;
  simSettings: SimSettings;
  onSimSettings: (s: SimSettings) => void;
  onRunSimulation: () => void;
  simRunning: boolean;
  simError: string | null;
  hasResults: boolean;
  resultsOpen: boolean;
  onShowResults: () => void;
};

/** Three small squares, filled one/two/three for small/medium/large. */
function SizePips({ size }: { size: LeakSize }) {
  return (
    <span className={`pips ${size}`} aria-hidden>
      <i />
      <i />
      <i />
    </span>
  );
}

/** Leak summary for the selected pipe, switching to the editor when adding or editing. */
function LeakControls({
  leak,
  maxStartHour,
  onSave,
  onRemove,
}: {
  leak: Leak | null;
  maxStartHour: number;
  onSave: (size: LeakSize, startHour: number, endHour: number | null) => void;
  onRemove: (leakId: string) => void;
}) {
  const [editing, setEditing] = useState(false);

  if (editing) {
    return (
      <LeakEditor
        initial={leak}
        maxStartHour={maxStartHour}
        onCancel={() => setEditing(false)}
        onSave={(size, startHour, endHour) => {
          onSave(size, startHour, endHour);
          setEditing(false);
        }}
      />
    );
  }
  if (!leak) {
    return (
      <button className="btn btn-block btn-primary" onClick={() => setEditing(true)}>
        Add leak here
      </button>
    );
  }
  return (
    <div className="leak-card">
      <div className="leak-card-title">
        <span className="leak-dot" aria-hidden />
        {LEAK_SIZES[leak.size].label} leak
        <span className="spacer" />
        <span className="muted">
          {formatStartHour(leak.startHour)}
          {leak.endHour != null ? `–${formatStartHour(leak.endHour)}` : " onward"}
        </span>
      </div>
      <SizePips size={leak.size} />
      <div className="leak-card-actions">
        <button className="btn btn-sm" onClick={() => setEditing(true)}>
          Edit
        </button>
        <button className="btn btn-sm btn-danger" onClick={() => onRemove(leak.id)}>
          Remove
        </button>
      </div>
    </div>
  );
}

/** 782 → "782", 12500 → "12.5k". */
function compact(n: number): string {
  return n >= 10_000 ? `${(n / 1000).toFixed(1).replace(/\.0$/, "")}k` : String(n);
}

function fmt(n: number | null, unit = ""): string {
  if (n === null) return "—";
  return `${Number.isInteger(n) ? n : n.toFixed(2)}${unit}`;
}

export default function FloatingPanel(props: Props) {
  const { network, file, networks, selection } = props;
  const [collapsed, setCollapsed] = useState(false);
  const [editingName, setEditingName] = useState(false);
  const [draftName, setDraftName] = useState(network.name);
  const [error, setError] = useState<string | null>(null);
  const inpRef = useRef<HTMLInputElement>(null);
  const csvRef = useRef<HTMLInputElement>(null);

  function commitName() {
    const name = draftName.trim();
    if (name && name !== network.name) props.onRename(name);
    else setDraftName(network.name);
    setEditingName(false);
  }

  const s = network.stats;
  // Leak times are limited by the chosen simulation length.
  const duration = props.simSettings.durationHours || DEFAULT_DURATION_HOURS;
  const maxStartHour = Math.max(Math.ceil(duration) - 1, 0);

  return (
    <aside className={`panel ${collapsed ? "collapsed" : ""}`}>
      <div className="panel-head">
        {editingName ? (
          <input
            className="name-input"
            value={draftName}
            autoFocus
            onChange={(e) => setDraftName(e.target.value)}
            onBlur={commitName}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitName();
              if (e.key === "Escape") {
                setDraftName(network.name);
                setEditingName(false);
              }
            }}
          />
        ) : (
          <button
            className="name-btn"
            title="Rename"
            onClick={() => {
              setDraftName(network.name);
              setEditingName(true);
            }}
          >
            {network.name}
          </button>
        )}
        <button
          className="round-btn"
          onClick={() => setCollapsed((c) => !c)}
          aria-label={collapsed ? "Expand panel" : "Collapse panel"}
        >
          <svg viewBox="0 0 24 24" aria-hidden className={collapsed ? "" : "flip"}>
            <path d="m6 9 6 6 6-6" />
          </svg>
        </button>
      </div>

      {!collapsed && (
        <div className="panel-body">
          <div className="pill-row">
            <select
              className="pill-select"
              value={network.id}
              onChange={(e) => (e.target.value === "__new" ? props.onNewUpload() : props.onSwitch(e.target.value))}
              aria-label="Switch network"
            >
              {networks.map((n) => (
                <option key={n.id} value={n.id}>
                  {n.name}
                </option>
              ))}
              <option value="__new">+ Upload another .inp…</option>
            </select>
          </div>

          {s && (
            <div className="metrics">
              <div className="metric">
                <span className="metric-label">Junctions</span>
                <span className="metric-value">{compact(s.junctions)}</span>
              </div>
              <div className="metric">
                <span className="metric-label">Pipes</span>
                <span className="metric-value">{compact(s.pipes)}</span>
              </div>
              <div className="metric">
                <span className="metric-label">Sources</span>
                <span className="metric-value">{s.reservoirs + s.tanks}</span>
              </div>
            </div>
          )}

          {selection && (
            <section className="selection">
              <div className="section-head">
                <div className="selection-title">
                  <span className="chip">{selection.type === "node" ? selection.node.kind : selection.link.kind}</span>
                  <span className="selection-id">{selection.type === "node" ? selection.node.id : selection.link.id}</span>
                </div>
                <button className="round-btn sm" onClick={props.onClearSelection} aria-label="Clear selection">
                  <svg viewBox="0 0 24 24" aria-hidden>
                    <path d="M6 6l12 12M18 6 6 18" />
                  </svg>
                </button>
              </div>
              <dl className="props">
                {selection.type === "node" ? (
                  <>
                    <div><dt>{selection.node.kind === "reservoir" ? "Head" : "Elevation"}</dt><dd>{fmt(selection.node.elevation)}</dd></div>
                    {selection.node.kind === "junction" && (
                      <div><dt>Base demand</dt><dd>{fmt(selection.node.baseDemand)}</dd></div>
                    )}
                  </>
                ) : (
                  <>
                    <div><dt>From → to</dt><dd className="mono">{selection.link.from} → {selection.link.to}</dd></div>
                    <div><dt>Length</dt><dd>{fmt(selection.link.length)}</dd></div>
                    <div><dt>Diameter</dt><dd>{fmt(selection.link.diameter)}</dd></div>
                  </>
                )}
              </dl>
              {selection.type === "node" && <p className="hint">Select a pipe to add a leak.</p>}
              {selection.type === "link" && selection.link.kind !== "pipe" && (
                <p className="hint">Leaks can only be added to pipes.</p>
              )}
              {selection.type === "link" && selection.link.kind === "pipe" && (
                <LeakControls
                  key={selection.link.id}
                  leak={network.leaks.find((l) => l.linkId === selection.link.id) ?? null}
                  maxStartHour={maxStartHour}
                  onSave={(size, startHour, endHour) => props.onSaveLeak(selection.link.id, size, startHour, endHour)}
                  onRemove={props.onRemoveLeak}
                />
              )}
            </section>
          )}
          {!selection && <p className="hint">Click a pipe to inspect it or add a leak.</p>}

          {network.leaks.length > 0 && (
            <section>
              <h3>
                Planted leaks <span className="count">{network.leaks.length}</span>
              </h3>
              <ul className="leak-list">
                {[...network.leaks]
                  .sort((a, b) => a.startHour - b.startHour)
                  .map((l) => (
                    <li key={l.id}>
                      <button
                        className={`leak-item ${selection?.type === "link" && selection.link.id === l.linkId ? "active" : ""}`}
                        onClick={() => props.onFocusLink(l.linkId)}
                      >
                        <span className="leak-item-row">
                          <span className="leak-item-name">
                            Pipe <span className="mono">{l.linkId}</span>
                          </span>
                          <span className="leak-item-value">
                            {formatStartHour(l.startHour)}
                            {l.endHour != null ? `–${formatStartHour(l.endHour)}` : ""}
                          </span>
                        </span>
                        <span className="leak-item-row">
                          <SizePips size={l.size} />
                          <span className="leak-item-size">{LEAK_SIZES[l.size].label}</span>
                        </span>
                      </button>
                    </li>
                  ))}
              </ul>
            </section>
          )}

          <section>
            <h3>Simulation</h3>
            <div className="sim-settings">
              <label className="sim-field">
                <span>Duration</span>
                <select
                  className="pill-select"
                  value={props.simSettings.durationHours}
                  onChange={(e) => props.onSimSettings({ ...props.simSettings, durationHours: Number(e.target.value) })}
                >
                  {[6, 12, 24, 48, 72, 168].map((h) => (
                    <option key={h} value={h}>
                      {h < 48 ? `${h} hours` : `${h / 24} days`}
                    </option>
                  ))}
                </select>
              </label>
              <label className="sim-field">
                <span>Time step</span>
                <select
                  className="pill-select"
                  value={props.simSettings.timestepMinutes}
                  onChange={(e) => props.onSimSettings({ ...props.simSettings, timestepMinutes: Number(e.target.value) })}
                >
                  {[5, 15, 30, 60].map((m) => (
                    <option key={m} value={m}>
                      {m} min
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <button
              className="btn btn-block btn-primary"
              disabled={props.simRunning || file.status !== "ready"}
              onClick={props.onRunSimulation}
            >
              {props.simRunning ? "Simulating…" : network.leaks.length ? "Run simulation" : "Run simulation (no leaks)"}
            </button>
            {props.simError && <p className="error">{props.simError}</p>}
            {props.hasResults && !props.resultsOpen && (
              <button className="link-btn" onClick={props.onShowResults}>
                Show results
              </button>
            )}
            <p className="hint">
              WNTR simulates pressure and flow from the .inp, with and without the leaks. Breaks, repairs and sizes come from
              the leaks above.
            </p>
          </section>

          <section>
            <h3>Files</h3>
            <div className="file-row">
              <div className="file-info">
                <span className="file-name">{network.inpFileName}</span>
                <span className="muted">{formatBytes(network.inpSize)}</span>
              </div>
              <button className="btn btn-sm" onClick={() => inpRef.current?.click()}>
                Replace
              </button>
              <input
                ref={inpRef}
                type="file"
                accept=".inp"
                hidden
                onChange={async (e) => {
                  const f = e.target.files?.[0];
                  e.target.value = "";
                  if (f) setError(await props.onReplaceInp(f));
                }}
              />
            </div>
            <div className="file-row">
              <div className="file-info">
                <span className="file-name">{network.csvFileName ?? "No sensor readings"}</span>
                <span className="muted">{network.csvFileName ? formatBytes(network.csvSize) : "timestamp + one column per sensor ID"}</span>
              </div>
              {network.csvFileName ? (
                <button className="btn btn-sm" onClick={props.onRemoveCsv}>
                  Remove
                </button>
              ) : (
                <button className="btn btn-sm" onClick={() => csvRef.current?.click()}>
                  Attach .csv
                </button>
              )}
              <input
                ref={csvRef}
                type="file"
                accept=".csv"
                hidden
                onChange={async (e) => {
                  const f = e.target.files?.[0];
                  e.target.value = "";
                  if (f) setError(await props.onAttachCsv(f));
                }}
              />
            </div>
            {error && <p className="error">{error}</p>}
            {file.status === "missing" && (
              <p className="error">The .inp file isn't saved in this browser. Replace it to view the network again.</p>
            )}
            {file.status === "invalid" && <p className="error">{file.error}</p>}
            {file.status === "ready" && file.parsed.missingCoords > 0 && (
              <p className="hint">{file.parsed.missingCoords} nodes have no coordinates and aren't drawn.</p>
            )}
          </section>

          <div className="panel-foot">
            <button className="link-btn" onClick={props.onNewUpload}>
              Upload another
            </button>
            <button className="link-btn danger" onClick={props.onDelete}>
              Delete network
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}
