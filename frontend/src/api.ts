import type { Leak } from "./types";
import { LEAK_SIZES } from "./leaks";

/** Client for the simulation API (backend/, proxied at /api by the Vite dev server). */

export type SimSettings = { durationHours: number; timestepMinutes: number };

export type SimLeak = {
  id: string;
  linkId: string;
  startHour: number;
  endHour: number | null;
  areaCm2: number;
  x: number;
  y: number;
  /** Ground-truth leak outflow per time step, L/s. */
  flowLps: number[];
  /** Mean pressure at the leak while open, m. A leak can't discharge at <= 0. */
  meanPressureM: number | null;
  startNode: string;
  endNode: string;
  /** Water arriving at the leak from each end of the pipe, L/s (sums to flowLps). */
  inflowFromStartLps: number[];
  inflowFromEndLps: number[];
};

export type SimResult = {
  simId: string;
  timesH: number[];
  /** Duration asked for; timesH can end earlier if WNTR stopped converging. */
  requestedHours: number;
  /** Plain-language problems with the run (e.g. stopped early). */
  warnings: string[];
  sensors: string[];
  noiseM: number;
  pressure: Record<string, { baseline: number[]; leak: number[] }>;
  measured: Record<string, number[]>;
  flow: Record<string, { baseline: number[]; leak: number[] }>;
  leaks: SimLeak[];
  totalLeakLps: number[];
};

export type TestResult = {
  model: { id: string; name: string; description: string };
  timesH: number[];
  /** One value per time step; null where the model has no score yet (e.g. before its first full window). */
  score: (number | null)[];
  threshold: number;
  scoreLabel: string;
  /** Plain-language caveats from the model about this run. */
  notes?: string[];
  events: { timeHour: number; node: string; x: number; y: number; candidates?: { node: string; probability: number; x: number; y: number }[] }[];
  groundTruth: { totalLeakLps: number[]; leaks: { id: string; linkId: string; startHour: number; endHour: number | null }[] };
  perLeak: {
    leakId: string;
    linkId: string;
    startHour: number;
    peakLps: number;
    detected: boolean;
    detectedHour: number | null;
    delayHours: number | null;
    predictedNode: string | null;
    locationError: number | null;
  }[];
  falseAlarms: { timeHour: number; node: string }[];
  summary: { leaks: number; detected: number; falseAlarms: number; meanDelayHours: number | null; meanLocationError: number | null };
};

export type ModelInfo = { id: string; name: string; description: string };

async function post<T>(path: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch {
    throw new Error("Couldn't reach the simulation server. Is the backend running?");
  }
  return parse<T>(res);
}

async function parse<T>(res: Response): Promise<T> {
  if (res.ok) return (await res.json()) as T;
  let detail = `${res.status} ${res.statusText}`;
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") detail = data.detail;
  } catch {
    // Non-JSON error (e.g. proxy can't reach the backend).
  }
  if (res.status === 500 || res.status === 502 || res.status === 504) {
    detail = "Couldn't reach the simulation server. Is the backend running?";
  }
  throw new Error(detail);
}

export function simulate(inpText: string, leaks: Leak[], settings: SimSettings): Promise<SimResult> {
  return post<SimResult>("/api/simulate", {
    inpText,
    durationHours: settings.durationHours,
    timestepMinutes: settings.timestepMinutes,
    leaks: leaks.map((l) => ({
      id: l.id,
      linkId: l.linkId,
      areaCm2: LEAK_SIZES[l.size].areaCm2,
      startHour: l.startHour,
      endHour: l.endHour ?? null,
    })),
  });
}

export function testModel(simId: string, model: string): Promise<TestResult> {
  return post<TestResult>("/api/test", { simId, model });
}

export async function listModels(): Promise<ModelInfo[]> {
  try {
    return await parse<ModelInfo[]>(await fetch("/api/models"));
  } catch {
    return [];
  }
}
