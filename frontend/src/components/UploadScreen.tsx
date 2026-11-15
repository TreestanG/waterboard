import { useRef, useState, type DragEvent } from "react";
import type { Network } from "../types";

type Props = {
  recent: Network[];
  onFile: (file: File) => Promise<string | null>;
  onSample: () => void;
  onOpen: (id: string) => void;
};

export default function UploadScreen({ recent, onFile, onSample, onOpen }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handle(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setError(await onFile(file));
    setBusy(false);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    handle(e.dataTransfer.files[0]);
  }

  return (
    <div className="upload-screen">
      <div className="upload-brand">
        <span className="logo" aria-hidden />
        Waterboard
      </div>
      <div className="upload-center">
        <h1 className="upload-title">Upload .inp file</h1>
        <p className="upload-sub">Drop an EPANET network model to render it in 3D and start finding leaks.</p>

        <div
          className={`upload-drop ${dragging ? "dragging" : ""} ${error ? "has-error" : ""}`}
          role="button"
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".inp"
            hidden
            onChange={(e) => {
              handle(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
          <span className="upload-icon-wrap" aria-hidden>
            <svg className="upload-icon" viewBox="0 0 24 24">
              <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
            </svg>
          </span>
          <div className="upload-drop-title">{busy ? "Reading…" : "Drag a file here or click to browse"}</div>
          <div className="upload-drop-hint">.inp files only</div>
        </div>

        {error && <p className="upload-error">{error}</p>}

        <button className="link-btn" onClick={onSample}>
          No file? Try the sample network
        </button>

        {recent.length > 0 && (
          <div className="recent">
            <div className="recent-label">Recent</div>
            {recent.slice(0, 5).map((n) => (
              <button key={n.id} className="recent-item" onClick={() => onOpen(n.id)}>
                <span>{n.name}</span>
                <span className="muted">{n.stats ? `${n.stats.junctions} junctions` : ""}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
