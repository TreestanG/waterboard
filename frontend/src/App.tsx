import { useEffect, useState } from "react";
import type { Leak, Network } from "./types";
import * as store from "./store";
import { parseInp, type ParsedNetwork } from "./inp";
import { readCsvFile, readInpFile } from "./files";
import UploadScreen from "./components/UploadScreen";
import NetworkMap, { type Selection } from "./components/NetworkMap";
import FloatingPanel from "./components/FloatingPanel";
import ConfirmDialog from "./components/ConfirmDialog";
import ResultsPanel from "./components/ResultsPanel";
import { listModels, simulate, testModel, type ModelInfo, type SimResult, type SimSettings, type TestResult } from "./api";

const PANEL_INSET = 360;

export type FileState =
  | { status: "loading" }
  | { status: "ready"; parsed: ParsedNetwork; text: string }
  | { status: "missing" }
  | { status: "invalid"; error: string };

export default function App() {
  const [networks, setNetworks] = useState<Network[]>(() => store.listNetworks());
  /** null shows the upload screen. */
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [file, setFile] = useState<FileState>({ status: "loading" });
  /** Bumped when the current network's .inp is replaced, to reload it. */
  const [fileVersion, setFileVersion] = useState(0);
  const [selection, setSelection] = useState<Selection>(null);
  const [pendingDelete, setPendingDelete] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const current = networks.find((n) => n.id === currentId) ?? null;

  // Simulation state, per network, kept in memory for this session.
  const [settingsById, setSettingsById] = useState<Record<string, SimSettings>>({});
  const [runsById, setRunsById] = useState<Record<string, { result: SimResult; key: string }>>({});
  const [testsById, setTestsById] = useState<Record<string, TestResult | null>>({});
  const [simRunning, setSimRunning] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [testRunning, setTestRunning] = useState(false);
  const [testError, setTestError] = useState<string | null>(null);
  const [resultsOpen, setResultsOpen] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);

  useEffect(() => {
    listModels().then(setModels);
  }, []);

  useEffect(() => {
    if (!currentId) return;
    let cancelled = false;
    setFile({ status: "loading" });
    store.loadInp(currentId).then((text) => {
      if (cancelled) return;
      if (text === null) return setFile({ status: "missing" });
      const r = parseInp(text);
      setFile(r.ok ? { status: "ready", parsed: r.network, text } : { status: "invalid", error: r.error });
    });
    return () => {
      cancelled = true;
    };
  }, [currentId, fileVersion]);

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 4000);
    return () => clearTimeout(t);
  }, [notice]);

  function refresh(warning: string | null, message?: string) {
    setNetworks(store.listNetworks());
    if (warning || message) setNotice(warning ?? message ?? null);
  }

  function open(id: string | null) {
    setCurrentId(id);
    setSelection(null);
    setSimError(null);
    setTestError(null);
    setResultsOpen(id !== null && id in runsById);
  }

  async function handleUpload(f: File): Promise<string | null> {
    const r = await readInpFile(f);
    if (!r.ok) return r.error;
    const { network, warning } = await store.createNetwork(
      {
        ...r.value.fields,
        name: f.name.replace(/\.inp$/i, ""),
        description: r.value.title ?? "",
        csvFileName: null,
        csvSize: null,
      },
      r.value.text,
    );
    refresh(warning);
    open(network.id);
    return null;
  }

  /** Loads the bundled EPA Net3 benchmark (pumps, tanks, controls; converges in WNTR). */
  async function handleSample() {
    const existing = networks.find((n) => n.source === "demo" && n.inpFileName === "Net3.inp");
    if (existing) return open(existing.id);
    let text: string;
    try {
      const res = await fetch("/samples/Net3.inp");
      if (!res.ok) throw new Error(String(res.status));
      text = await res.text();
    } catch {
      setNotice("Couldn't load the sample network.");
      return;
    }
    const r = parseInp(text);
    if (!r.ok) {
      setNotice(r.error);
      return;
    }
    const { network, warning } = await store.createNetwork(
      {
        name: "Net3 (EPA sample)",
        description: "EPANET example network 3: 92 junctions, 2 pumps, 3 tanks",
        inpFileName: "Net3.inp",
        inpSize: text.length,
        stats: r.network.stats,
        csvFileName: null,
        csvSize: null,
      },
      text,
      "demo",
    );
    refresh(warning);
    open(network.id);
  }

  /** Duration choices in the panel; the .inp's own duration is used when it's one of them. */
  const DURATIONS = [6, 12, 24, 48, 72, 168];
  function settingsFor(n: Network): SimSettings {
    const saved = settingsById[n.id];
    if (saved) return saved;
    const inpHours = file.status === "ready" ? file.parsed.durationHours : null;
    return { durationHours: inpHours && DURATIONS.includes(inpHours) ? inpHours : 24, timestepMinutes: 15 };
  }
  /** What a run depends on; a run whose key differs from the current one is stale. */
  function runKey(n: Network, settings: SimSettings): string {
    const leaks = [...n.leaks]
      .sort((a, b) => a.linkId.localeCompare(b.linkId))
      .map((l) => [l.linkId, l.size, l.startHour, l.endHour ?? null]);
    return JSON.stringify({ leaks, settings, fileVersion });
  }

  async function runSimulation(n: Network) {
    if (file.status !== "ready") return;
    const settings = settingsFor(n);
    setSimRunning(true);
    setSimError(null);
    try {
      const result = await simulate(file.text, n.leaks, settings);
      setRunsById((r) => ({ ...r, [n.id]: { result, key: runKey(n, settings) } }));
      setTestsById((t) => ({ ...t, [n.id]: null }));
      setTestError(null);
      setResultsOpen(true);
    } catch (e) {
      setSimError(e instanceof Error ? e.message : String(e));
    } finally {
      setSimRunning(false);
    }
  }

  async function runTest(n: Network, modelId: string) {
    const run = runsById[n.id];
    if (!run) return;
    setTestRunning(true);
    setTestError(null);
    try {
      const result = await testModel(run.result.simId, modelId);
      setTestsById((t) => ({ ...t, [n.id]: result }));
    } catch (e) {
      setTestError(e instanceof Error ? e.message : String(e));
    } finally {
      setTestRunning(false);
    }
  }

  if (!current) {
    return (
      <>
        <UploadScreen recent={networks} onFile={handleUpload} onSample={handleSample} onOpen={open} />
        {notice && <div className="toast" role="status">{notice}</div>}
      </>
    );
  }

  const settings = settingsFor(current);
  const run = runsById[current.id];
  const test = testsById[current.id] ?? null;
  const stale = !!run && run.key !== runKey(current, settings);

  return (
    <div className="workspace">
      {file.status === "ready" ? (
        <NetworkMap
          network={file.parsed}
          leaks={current.leaks}
          sensors={run && resultsOpen ? run.result.sensors : []}
          predictions={test && resultsOpen ? test.events : []}
          selection={selection}
          onSelect={setSelection}
          inset={{ left: PANEL_INSET, bottom: 0 }}
        />
      ) : (
        <div className="map map-empty">{file.status === "loading" ? "Loading network…" : "Network can't be drawn."}</div>
      )}

      <FloatingPanel
        key={current.id}
        network={current}
        file={file}
        networks={networks}
        selection={selection}
        onClearSelection={() => setSelection(null)}
        onSwitch={open}
        onNewUpload={() => open(null)}
        onRename={async (name) => refresh((await store.updateNetwork(current.id, { name })).warning)}
        onReplaceInp={async (f) => {
          const r = await readInpFile(f);
          if (!r.ok) return r.error;
          setSelection(null);
          // Keep only leaks whose pipe still exists in the new file.
          const parsed = parseInp(r.value.text);
          const pipeIds = new Set(parsed.ok ? parsed.network.links.filter((l) => l.kind === "pipe").map((l) => l.id) : []);
          const leaks = current.leaks.filter((l) => pipeIds.has(l.linkId));
          const dropped = current.leaks.length - leaks.length;
          const { warning } = await store.updateNetwork(current.id, { ...r.value.fields, leaks }, r.value.text);
          refresh(
            warning,
            dropped > 0
              ? `Replaced with ${f.name}. Removed ${dropped} leak${dropped === 1 ? "" : "s"} on pipes that no longer exist.`
              : `Replaced with ${f.name}.`,
          );
          setFileVersion((v) => v + 1);
          return null;
        }}
        onAttachCsv={async (f) => {
          const r = readCsvFile(f);
          if (!r.ok) return r.error;
          refresh((await store.updateNetwork(current.id, r.value)).warning, `Attached ${f.name}.`);
          return null;
        }}
        onRemoveCsv={async () =>
          refresh((await store.updateNetwork(current.id, { csvFileName: null, csvSize: null })).warning)
        }
        onDelete={() => setPendingDelete(true)}
        onSaveLeak={async (linkId, size, startHour, endHour) => {
          const existing = current.leaks.find((l) => l.linkId === linkId);
          const leak: Leak = {
            id: existing?.id ?? crypto.randomUUID(),
            linkId,
            size,
            startHour,
            endHour,
            createdAt: existing?.createdAt ?? new Date().toISOString(),
          };
          refresh((await store.upsertLeak(current.id, leak)).warning, existing ? "Leak updated." : `Leak added to pipe ${linkId}.`);
        }}
        onRemoveLeak={async (leakId) => refresh((await store.removeLeak(current.id, leakId)).warning, "Leak removed.")}
        onFocusLink={(linkId) => {
          const link = file.status === "ready" ? file.parsed.links.find((l) => l.id === linkId) : undefined;
          if (link) setSelection({ type: "link", link });
        }}
        simSettings={settings}
        onSimSettings={(s) => setSettingsById((all) => ({ ...all, [current.id]: s }))}
        onRunSimulation={() => runSimulation(current)}
        simRunning={simRunning}
        simError={simError}
        hasResults={!!run}
        resultsOpen={resultsOpen}
        onShowResults={() => setResultsOpen(true)}
      />

      {run && resultsOpen && (
        <ResultsPanel
          sim={run.result}
          stale={stale}
          selection={selection}
          models={models}
          test={test}
          testRunning={testRunning}
          testError={testError}
          onRunTest={(modelId) => runTest(current, modelId)}
          onClose={() => setResultsOpen(false)}
        />
      )}

      {pendingDelete && (
        <ConfirmDialog
          title={`Delete “${current.name}”?`}
          body="This removes the network and its attached files from this browser. It can't be undone."
          confirmLabel="Delete"
          onCancel={() => setPendingDelete(false)}
          onConfirm={async () => {
            const name = current.name;
            const { warning } = await store.deleteNetwork(current.id);
            refresh(warning, `Deleted “${name}”.`);
            setPendingDelete(false);
            open(null);
          }}
        />
      )}
      {notice && <div className="toast" role="status">{notice}</div>}
    </div>
  );
}
