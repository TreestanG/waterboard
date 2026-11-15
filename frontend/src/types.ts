export type NetworkSource = "demo" | "uploaded" | "generated";

export type InpStats = {
  junctions: number;
  pipes: number;
  reservoirs: number;
  tanks: number;
  pumps: number;
  valves: number;
};

export type LeakSize = "small" | "medium" | "large";

/**
 * A leak the user planted on a pipe. The simulator places it at the pipe's midpoint
 * (by splitting the pipe), since position along a pipe is below what detection can resolve.
 */
export type Leak = {
  id: string;
  linkId: string;
  size: LeakSize;
  /** Hours after the start of the simulation. */
  startHour: number;
  /** Hours after the start when the leak is repaired; null = still leaking at the end. */
  endHour?: number | null;
  createdAt: string;
};

/** Network metadata. The .inp text itself is stored separately (see blobs.ts). */
export type Network = {
  id: string;
  name: string;
  description: string;
  source: NetworkSource;
  inpFileName: string | null;
  inpSize: number | null;
  csvFileName: string | null;
  csvSize: number | null;
  stats: InpStats | null;
  leaks: Leak[];
  createdAt: string;
  updatedAt: string;
};

export type NetworkInput = Pick<
  Network,
  "name" | "description" | "inpFileName" | "inpSize" | "csvFileName" | "csvSize" | "stats"
> & { leaks?: Leak[] };
