import { useState } from "react";
import type { Leak, LeakSize } from "../types";
import { LEAK_SIZES, LEAK_SIZE_ORDER, formatStartHour } from "../leaks";

type Props = {
  initial: Leak | null;
  /** Latest allowed start hour (simulation length minus one hour). */
  maxStartHour: number;
  onSave: (size: LeakSize, startHour: number, endHour: number | null) => void;
  onCancel: () => void;
};

export default function LeakEditor({ initial, maxStartHour, onSave, onCancel }: Props) {
  const [size, setSize] = useState<LeakSize>(initial?.size ?? "medium");
  const [startHour, setStartHour] = useState(Math.min(initial?.startHour ?? 0, maxStartHour));
  const [repaired, setRepaired] = useState(initial?.endHour != null);
  const [endHour, setEndHour] = useState(initial?.endHour ?? Math.min(startHour + 6, maxStartHour + 1));
  const maxEnd = maxStartHour + 1;
  const effectiveEnd = Math.max(endHour, startHour + 1);

  return (
    <div className="leak-editor">
      <div className="field-label">Size</div>
      <div className="segmented" role="radiogroup" aria-label="Leak size">
        {LEAK_SIZE_ORDER.map((s) => (
          <button
            key={s}
            type="button"
            role="radio"
            aria-checked={size === s}
            className={size === s ? "on" : ""}
            onClick={() => setSize(s)}
          >
            {LEAK_SIZES[s].label}
          </button>
        ))}
      </div>
      <p className="hint">{LEAK_SIZES[size].hint}</p>

      <div className="field-label">
        Breaks at <span className="field-value">{formatStartHour(startHour)}</span>
      </div>
      <input
        className="range"
        type="range"
        min={0}
        max={maxStartHour}
        step={1}
        value={startHour}
        onChange={(e) => setStartHour(Number(e.target.value))}
        aria-label="Leak start time"
      />

      <label className="check-row">
        <input type="checkbox" checked={repaired} onChange={(e) => setRepaired(e.target.checked)} />
        Repaired during the simulation
      </label>
      {repaired && (
        <>
          <div className="field-label">
            Repaired at <span className="field-value">{formatStartHour(effectiveEnd)}</span>
          </div>
          <input
            className="range"
            type="range"
            min={startHour + 1}
            max={maxEnd}
            step={1}
            value={effectiveEnd}
            onChange={(e) => setEndHour(Number(e.target.value))}
            aria-label="Leak repair time"
          />
        </>
      )}
      <p className="hint">Placed at the middle of the pipe. Hours from the start of the simulation.</p>

      <div className="leak-editor-actions">
        <button className="btn btn-sm" onClick={onCancel}>
          Cancel
        </button>
        <button
          className="btn btn-sm btn-primary"
          onClick={() => onSave(size, startHour, repaired ? Math.min(effectiveEnd, maxEnd) : null)}
        >
          {initial ? "Save leak" : "Add leak"}
        </button>
      </div>
    </div>
  );
}
