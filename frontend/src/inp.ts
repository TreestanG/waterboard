import type { InpStats } from "./types";

export type NodeKind = "junction" | "reservoir" | "tank";
export type LinkKind = "pipe" | "pump" | "valve";

export type NetNode = {
  id: string;
  kind: NodeKind;
  x: number;
  y: number;
  /** Ground elevation for junctions and tanks, total head for reservoirs. */
  elevation: number;
  baseDemand: number | null;
};

export type NetLink = {
  id: string;
  kind: LinkKind;
  from: string;
  to: string;
  length: number | null;
  diameter: number | null;
  /** Start node, any intermediate vertices, end node. */
  path: [number, number][];
};

export type ParsedNetwork = {
  title: string | null;
  stats: InpStats;
  nodes: NetNode[];
  links: NetLink[];
  /** Nodes listed in the file that have no [COORDINATES] entry. */
  missingCoords: number;
  /** Simulation length from [TIMES] DURATION, or null if unset or a single snapshot. */
  durationHours: number | null;
};

/** Parses an EPANET time value: "72", "24:00", "1.5 HOURS", "90 MIN", "2 DAYS". */
function parseHours(tokens: string[]): number | null {
  const [value, unit = "HOURS"] = tokens;
  if (!value) return null;
  let hours: number;
  if (value.includes(":")) {
    const [h, m = "0"] = value.split(":");
    hours = Number(h) + Number(m) / 60;
  } else {
    hours = Number(value);
    const u = unit.toUpperCase();
    if (u.startsWith("SEC")) hours /= 3600;
    else if (u.startsWith("MIN")) hours /= 60;
    else if (u.startsWith("DAY")) hours *= 24;
  }
  return Number.isFinite(hours) && hours > 0 ? hours : null;
}

export type InpParseResult = { ok: true; network: ParsedNetwork } | { ok: false; error: string };

type Sections = Map<string, string[][]>;

function splitSections(text: string): Sections {
  const sections: Sections = new Map();
  let current: string[][] | null = null;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.split(";")[0].trim();
    if (!line) continue;
    const header = line.match(/^\[([A-Z_]+)\]$/i);
    if (header) {
      const name = header[1].toUpperCase();
      current = sections.get(name) ?? [];
      sections.set(name, current);
      continue;
    }
    current?.push(line.split(/\s+/));
  }
  return sections;
}

const num = (s: string | undefined): number | null => {
  if (s === undefined) return null;
  const v = Number(s);
  return Number.isFinite(v) ? v : null;
};

export function parseInp(text: string): InpParseResult {
  const s = splitSections(text);
  const rows = (name: string) => s.get(name) ?? [];

  if (!s.has("JUNCTIONS")) {
    return { ok: false, error: "This file has no [JUNCTIONS] section, so it doesn't look like an EPANET .inp file." };
  }
  if (!s.has("PIPES")) return { ok: false, error: "This file has no [PIPES] section." };
  if (rows("RESERVOIRS").length + rows("TANKS").length === 0) {
    return { ok: false, error: "The network has no reservoirs or tanks, so there's no water source to simulate from." };
  }

  const coords = new Map<string, [number, number]>();
  for (const [id, x, y] of rows("COORDINATES")) {
    const nx = num(x);
    const ny = num(y);
    if (nx !== null && ny !== null) coords.set(id, [nx, ny]);
  }
  if (coords.size === 0) {
    return { ok: false, error: "This file has no [COORDINATES] section, so the network can't be drawn." };
  }

  const vertices = new Map<string, [number, number][]>();
  for (const [id, x, y] of rows("VERTICES")) {
    const nx = num(x);
    const ny = num(y);
    if (nx === null || ny === null) continue;
    const list = vertices.get(id) ?? [];
    list.push([nx, ny]);
    vertices.set(id, list);
  }

  const nodes: NetNode[] = [];
  let missingCoords = 0;
  const addNode = (id: string, kind: NodeKind, elevation: number | null, baseDemand: number | null) => {
    const c = coords.get(id);
    if (!c) {
      missingCoords += 1;
      return;
    }
    nodes.push({ id, kind, x: c[0], y: c[1], elevation: elevation ?? 0, baseDemand });
  };
  for (const r of rows("JUNCTIONS")) addNode(r[0], "junction", num(r[1]), num(r[2]));
  for (const r of rows("RESERVOIRS")) addNode(r[0], "reservoir", num(r[1]), null);
  for (const r of rows("TANKS")) addNode(r[0], "tank", num(r[1]), null);

  const links: NetLink[] = [];
  const addLink = (r: string[], kind: LinkKind, length: number | null, diameter: number | null) => {
    const [id, from, to] = r;
    const a = coords.get(from);
    const b = coords.get(to);
    if (!a || !b) return;
    links.push({ id, kind, from, to, length, diameter, path: [a, ...(vertices.get(id) ?? []), b] });
  };
  for (const r of rows("PIPES")) addLink(r, "pipe", num(r[3]), num(r[4]));
  for (const r of rows("PUMPS")) addLink(r, "pump", null, null);
  for (const r of rows("VALVES")) addLink(r, "valve", null, num(r[3]));

  const title = rows("TITLE")[0]?.join(" ") ?? null;
  const durationRow = rows("TIMES").find((r) => r[0].toUpperCase() === "DURATION");
  const durationHours = durationRow ? parseHours(durationRow.slice(1)) : null;

  return {
    ok: true,
    network: {
      title,
      stats: {
        junctions: rows("JUNCTIONS").length,
        pipes: rows("PIPES").length,
        reservoirs: rows("RESERVOIRS").length,
        tanks: rows("TANKS").length,
        pumps: rows("PUMPS").length,
        valves: rows("VALVES").length,
      },
      nodes,
      links,
      missingCoords,
      durationHours,
    },
  };
}
