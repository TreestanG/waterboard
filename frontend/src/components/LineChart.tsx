import { useMemo, useState } from "react";

export type ChartSeries = {
  label: string;
  /** null = no value at that step (drawn as a gap). */
  values: (number | null)[];
  color: string;
  dashed?: boolean;
  /** Plot against the right-hand axis (its own scale). */
  right?: boolean;
  /** Draw as a step line (values hold until the next sample). */
  step?: boolean;
};

type Props = {
  times: number[];
  series: ChartSeries[];
  leftLabel: string;
  rightLabel?: string;
  /** Vertical markers, e.g. leak starts or alarms. */
  markers?: { at: number; label: string; color: string }[];
  /** Horizontal reference line on the right axis (e.g. a detection threshold). */
  rightThreshold?: { value: number; label: string };
  height?: number;
};

const W = 360;
const PAD = { l: 38, r: 38, t: 10, b: 24 };

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  const n = v / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
}

function niceMin(v: number, max: number): number {
  if (v >= 0) return 0;
  return -niceMax(-v > max ? -v : -v);
}

function fmt(v: number): string {
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1).replace(/\.0$/, "") : v.toFixed(2).replace(/0$/, "").replace(/\.0$/, "");
}

/** Small dependency-free SVG line chart with a hover readout, sized for the side panel. */
export default function LineChart({ times, series, leftLabel, rightLabel, markers = [], rightThreshold, height = 160 }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const H = height;
  const plotW = W - PAD.l - PAD.r;
  const plotH = H - PAD.t - PAD.b;

  const scales = useMemo(() => {
    const range = (right: boolean) => {
      const vals = series.filter((s) => !!s.right === right).flatMap((s) => s.values).filter((v): v is number => v != null);
      if (right && rightThreshold) vals.push(rightThreshold.value);
      if (!vals.length) return { min: 0, max: 1 };
      let min = Math.min(...vals), max = Math.max(...vals);
      if (min >= 0 && max > 0 && min / max > 0.6) {
        // Narrow range far from zero (e.g. pressures): zoom in instead of starting at 0.
        const pad = (max - min) * 0.15 || Math.abs(max) * 0.05 || 1;
        return { min: min - pad, max: max + pad };
      }
      max = niceMax(Math.max(max, 0));
      min = niceMin(Math.min(min, 0), max);
      return { min, max: max === min ? min + 1 : max };
    };
    return { left: range(false), right: range(true) };
  }, [series, rightThreshold]);

  const t0 = times[0] ?? 0;
  const t1 = times[times.length - 1] ?? 1;
  const x = (t: number) => PAD.l + ((t - t0) / (t1 - t0 || 1)) * plotW;
  const y = (v: number, right: boolean) => {
    const s = right ? scales.right : scales.left;
    return PAD.t + (1 - (v - s.min) / (s.max - s.min || 1)) * plotH;
  };

  const path = (s: ChartSeries) =>
    s.values
      .map((v, i) => {
        if (v == null) return "";
        const px = x(times[i]), py = y(v, !!s.right);
        if (i === 0 || s.values[i - 1] == null) return `M${px},${py}`;
        return s.step ? `H${px}V${py}` : `L${px},${py}`;
      })
      .join("");

  const ticks = [0, 0.5, 1];
  const hasRight = series.some((s) => s.right);
  const xTicks = 4;

  return (
    <div className="chart">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${leftLabel}${rightLabel ? ` and ${rightLabel}` : ""} over time`}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const px = ((e.clientX - r.left) / r.width) * W;
          const t = t0 + ((px - PAD.l) / plotW) * (t1 - t0);
          let best = 0;
          times.forEach((tt, i) => {
            if (Math.abs(tt - t) < Math.abs(times[best] - t)) best = i;
          });
          setHover(px >= PAD.l && px <= W - PAD.r ? best : null);
        }}
      >
        {ticks.map((k) => {
          const yy = PAD.t + (1 - k) * plotH;
          const lv = scales.left.min + k * (scales.left.max - scales.left.min);
          const rv = scales.right.min + k * (scales.right.max - scales.right.min);
          return (
            <g key={k}>
              <line x1={PAD.l} x2={W - PAD.r} y1={yy} y2={yy} className="chart-grid" />
              <text x={PAD.l - 6} y={yy + 3} className="chart-tick" textAnchor="end">{fmt(lv)}</text>
              {hasRight && <text x={W - PAD.r + 6} y={yy + 3} className="chart-tick">{fmt(rv)}</text>}
            </g>
          );
        })}
        {Array.from({ length: xTicks + 1 }, (_, i) => {
          const t = t0 + ((t1 - t0) * i) / xTicks;
          return (
            <text key={i} x={x(t)} y={H - 6} className="chart-tick" textAnchor="middle">
              {fmt(t)}h
            </text>
          );
        })}
        {rightThreshold && (
          <g>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(rightThreshold.value, true)} y2={y(rightThreshold.value, true)} className="chart-threshold" />
          </g>
        )}
        {markers.map((m, i) => (
          <line key={i} x1={x(m.at)} x2={x(m.at)} y1={PAD.t} y2={PAD.t + plotH} stroke={m.color} className="chart-marker" />
        ))}
        {series.map((s) => (
          <path key={s.label} d={path(s)} fill="none" stroke={s.color} strokeWidth={1.8} strokeDasharray={s.dashed ? "4 3" : undefined} />
        ))}
        {hover !== null && (
          <line x1={x(times[hover])} x2={x(times[hover])} y1={PAD.t} y2={PAD.t + plotH} className="chart-cursor" />
        )}
      </svg>
      <div className="chart-legend">
        {series.map((s) => (
          <span key={s.label}>
            <i style={{ background: s.color, opacity: s.dashed ? 0.6 : 1 }} />
            {s.label}
            {hover !== null && <b>{s.values[hover] == null ? "—" : fmt(s.values[hover] as number)}</b>}
          </span>
        ))}
        {hover !== null && <span className="muted">at {fmt(times[hover])}h</span>}
      </div>
      <div className="chart-axes muted">
        <span>{leftLabel}</span>
        {rightLabel && <span>{rightLabel}</span>}
      </div>
    </div>
  );
}
