import { useMemo, useState, useEffect, useLayoutEffect, useRef } from "react";
import DeckGL from "@deck.gl/react";
import {
  AmbientLight,
  DirectionalLight,
  LightingEffect,
  LinearInterpolator,
  OrbitView,
  type OrbitViewState,
  type PickingInfo,
} from "@deck.gl/core";
import { PathLayer, ScatterplotLayer } from "@deck.gl/layers";
import { SimpleMeshLayer } from "@deck.gl/mesh-layers";
import { Geometry, IcoSphereGeometry, SphereGeometry, TruncatedConeGeometry } from "@luma.gl/engine";
import type { NetLink, NetNode, ParsedNetwork } from "../inp";
import type { Leak } from "../types";
import { LEAK_SIZES } from "../leaks";
import FlowLayer from "./FlowLayer";

export type Selection = { type: "node"; node: NetNode } | { type: "link"; link: NetLink } | null;

type Props = {
  network: ParsedNetwork;
  leaks: Leak[];
  /** Junction IDs with simulated pressure sensors (shown while results are open). */
  sensors?: string[];
  /** A detection model's alarms: where it thinks leaks are. */
  predictions?: { timeHour: number; node: string }[];
  selection: Selection;
  onSelect: (s: Selection) => void;
  /** Screen space the floating UI covers, so the network is fitted into what's left. */
  inset: { left: number; bottom: number };
};

type P3 = [number, number, number];
/**
 * `main` marks the larger pipes (top half of the diameter range), drawn as trunk mains.
 * `t` is the link's position in the diameter range (0-1); `radius` is its 3D tube radius in world units.
 */
type DrawLink = { link: NetLink; path: P3[]; ground: P3[]; width: number; main: boolean; t: number; radius: number };
type DrawNode = { node: NetNode; position: P3 };
type DrawValve = { link: NetLink; position: P3; yaw: number };

/**
 * Light flow-map palette: a muted, slightly tinted network on a pale canvas, with water
 * drawn in saturated orange (mains) and teal (branches). Leaks are pink-red so they stand
 * apart from both water colors; selection is near-black.
 */
const COLORS = {
  pump: [176, 140, 92] as const,
  shadow: [20, 30, 40, 12] as const,
  /** Teal stands out against dark iron pipes and the green terrain. */
  selected: [18, 168, 182] as const,
  leak: [236, 38, 94] as const,
  leakPipe: [236, 38, 94] as const,
  /** Industrial-model gray: one near-uniform matte mid-gray for pipe, mains and fittings, like a clay render. */
  tube: [152, 154, 156] as const,
  groundDot: [178, 186, 196] as const,
  trunk: [150, 132, 114] as const,
  conifer: [118, 162, 134] as const,
  broadleaf: [156, 188, 138] as const,
  tubeMain: [146, 148, 150] as const,
  /** Socket collars and junction fittings: cast iron, a shade darker than the pipe body. */
  fitting: [140, 142, 144] as const,
  waterMain: [242, 110, 66] as const,
  waterBranch: [18, 168, 182] as const,
};

/** Water draws over the pipes it sits on without hiding behind them or occluding markers. */
const FLOW_PARAMETERS = {
  depthCompare: "less-equal",
  depthWriteEnabled: false,
} as const;

/**
 * Animated water dashes along the pipes. Off for now; the direction is only estimated until
 * the hydraulic simulation provides real flows. Set to true to bring the animation back.
 */
const WATER_ANIMATION = false;

/**
 * Opacity (0-255) of pipes, joints and the tank/reservoir/valve shells. 255 = solid iron;
 * lower values make them see-through (and they then stop writing depth).
 */
const PIPE_ALPHA = 255;

/** Height of the elevation relief, as a fraction of the network's horizontal extent. */
const RELIEF = 0.12;

function buildGeometry(net: ParsedNetwork) {
  const xs = net.nodes.map((n) => n.x);
  const ys = net.nodes.map((n) => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const extentX = Math.max(maxX - minX, 1e-6);
  const extentY = Math.max(maxY - minY, 1e-6);
  const extent = Math.max(extentX, extentY);

  const elevs = net.nodes.filter((n) => n.kind !== "reservoir").map((n) => n.elevation);
  const minZ = elevs.length ? Math.min(...elevs) : 0;
  const maxZ = elevs.length ? Math.max(...elevs) : 0;
  const zScale = maxZ > minZ ? (RELIEF * extent) / (maxZ - minZ) : 0;

  const zById = new Map<string, number>();
  for (const n of net.nodes) if (n.kind !== "reservoir") zById.set(n.id, (n.elevation - minZ) * zScale);
  // A reservoir's "elevation" in an .inp is its water head, not ground level, so it
  // would float above the network. Sit it on the ground of the node it feeds instead.
  for (const n of net.nodes) {
    if (n.kind !== "reservoir") continue;
    const link = net.links.find((l) => l.from === n.id || l.to === n.id);
    const other = link && (link.from === n.id ? link.to : link.from);
    zById.set(n.id, other ? (zById.get(other) ?? 0) : 0);
  }
  const z = (n: NetNode) => zById.get(n.id) ?? 0;

  const byId = new Map(net.nodes.map((n) => [n.id, n]));
  const nodes: DrawNode[] = net.nodes.map((n) => ({ node: n, position: [n.x - cx, n.y - cy, z(n)] }));

  const diams = net.links.map((l) => l.diameter).filter((d): d is number => d !== null && d > 0);
  const minD = diams.length ? Math.min(...diams) : 0;
  const maxD = diams.length ? Math.max(...diams) : 0;

  const links: DrawLink[] = net.links.map((l) => {
    const za = z(byId.get(l.from)!);
    const zb = z(byId.get(l.to)!);
    const last = l.path.length - 1;
    const path = l.path.map(([x, y], i): P3 => [x - cx, y - cy, za + ((zb - za) * i) / Math.max(last, 1)]);
    const ground = path.map(([x, y]): P3 => [x, y, 0]);
    const t = l.diameter && maxD > minD ? (l.diameter - minD) / (maxD - minD) : 0.3;
    return { link: l, path, ground, width: l.kind === "pipe" ? 2.4 + 5 * t : 5, main: l.kind === "pipe" && t >= 0.5, t, radius: 0 };
  });

  const valves: DrawValve[] = links
    .filter((d) => d.link.kind === "valve")
    .map((d) => {
      const a = d.path[0];
      const b = d.path[d.path.length - 1];
      return {
        link: d.link,
        position: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2],
        yaw: (Math.atan2(b[1] - a[1], b[0] - a[0]) * 180) / Math.PI,
      };
    });

  const unit = extent * 0.011;
  const groundDots = buildGroundDots(extent);
  const terrain = buildTerrain(nodes, unit, extent);
  const trees = buildTrees(links, nodes, unit, extent, terrain.height);
  for (const d of links) d.radius = unit * (d.link.kind === "pipe" ? 0.16 + 0.34 * d.t : 0.3);

  return {
    nodes,
    links,
    valves,
    flows: buildFlows(net, links),
    ...buildTubes(links, nodes),
    groundDots,
    trees,
    terrainMesh: terrain.mesh,
    extentX,
    extentY,
    height: (maxZ - minZ) * zScale,
    /** World size of one model unit, so models stay a consistent size relative to the network. */
    unit,
  };
}

/** A link oriented in its (estimated) flow direction, with distance-from-source at each vertex. */
type DrawFlow = { link: DrawLink; path: P3[]; distances: number[]; width: number };

const segLength = (a: P3, b: P3) => Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);

/**
 * Estimated flow direction for the animation, until the hydraulic simulation provides
 * real flows: water moves away from reservoirs and tanks along the shortest pipe route
 * (Dijkstra over pipe length). Links unreachable from any source get no animation.
 */
function buildFlows(net: ParsedNetwork, links: DrawLink[]): DrawFlow[] {
  const lengths = links.map((d) => d.path.slice(1).reduce((s, p, i) => s + segLength(d.path[i], p), 0));
  const adjacency = new Map<string, { to: string; w: number }[]>();
  links.forEach((d, i) => {
    const { from, to } = d.link;
    (adjacency.get(from) ?? adjacency.set(from, []).get(from)!).push({ to, w: lengths[i] });
    (adjacency.get(to) ?? adjacency.set(to, []).get(to)!).push({ to: from, w: lengths[i] });
  });

  // Dijkstra with a binary heap of [distance, nodeId].
  const dist = new Map<string, number>();
  const heap: [number, string][] = [];
  const push = (item: [number, string]) => {
    heap.push(item);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (heap[p][0] <= heap[i][0]) break;
      [heap[p], heap[i]] = [heap[i], heap[p]];
      i = p;
    }
  };
  const pop = (): [number, string] => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
        if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
        if (m === i) break;
        [heap[m], heap[i]] = [heap[i], heap[m]];
        i = m;
      }
    }
    return top;
  };
  for (const n of net.nodes) {
    if (n.kind === "junction") continue;
    dist.set(n.id, 0);
    push([0, n.id]);
  }
  while (heap.length) {
    const [d, id] = pop();
    if (d > (dist.get(id) ?? Infinity)) continue;
    for (const e of adjacency.get(id) ?? []) {
      const nd = d + e.w;
      if (nd < (dist.get(e.to) ?? Infinity)) {
        dist.set(e.to, nd);
        push([nd, e.to]);
      }
    }
  }

  return links.flatMap((d, i) => {
    const da = dist.get(d.link.from);
    const db = dist.get(d.link.to);
    if (da === undefined || db === undefined || lengths[i] === 0) return [];
    const forward = da <= db;
    const path = forward ? d.path : [...d.path].reverse();
    let acc = forward ? da : db;
    const distances = path.map((p, j) => (j === 0 ? acc : (acc += segLength(path[j - 1], p))));
    return [{ link: d, path, distances, width: Math.max(d.width * 0.4, 1) }];
  });
}

/** A dot on the ground plane; `alpha` fades the grid out toward its edges. */
type GroundDot = { position: P3; alpha: number };

/**
 * Evenly spaced dots on the ground plane, lying flat so perspective shrinks and packs the
 * far ones together, which shows the map's depth and tilt. Covers well past the network
 * and fades out radially so the grid has no hard edge.
 */
function buildGroundDots(extent: number): GroundDot[] {
  const spacing = extent / 36;
  const reach = extent * 1.6;
  const n = Math.ceil(reach / spacing);
  const dots: GroundDot[] = [];
  for (let i = -n; i <= n; i++) {
    for (let j = -n; j <= n; j++) {
      const x = i * spacing;
      const y = j * spacing;
      const r = Math.hypot(x, y) / reach;
      if (r > 1) continue;
      // Full strength across the network, easing to zero at the edge.
      const alpha = r < 0.45 ? 1 : 1 - (r - 0.45) / 0.55;
      dots.push({ position: [x, y, -extent * 0.0005], alpha: alpha * alpha });
    }
  }
  return dots;
}

/**
 * Terrain: a smooth height field interpolated from node elevations (inverse-distance weighting
 * over nearby nodes), dropped a little below the pipes so they always sit visibly on top. Away
 * from the network it eases down and blends into the background, so it reads as land the
 * network sits in rather than a slab with edges.
 */
function buildTerrain(nodes: DrawNode[], unit: number, extent: number) {
  const pts = nodes.filter((n) => n.node.kind !== "reservoir");
  const src = pts.length ? pts : nodes;
  const sink = unit * 0.75; // how far below the pipes the ground sits
  const cell = extent / 30;
  const key = (i: number, j: number) => `${i},${j}`;
  const grid = new Map<string, DrawNode[]>();
  for (const n of src) {
    const k = key(Math.floor(n.position[0] / cell), Math.floor(n.position[1] / cell));
    (grid.get(k) ?? grid.set(k, []).get(k)!).push(n);
  }
  const zs = src.map((n) => n.position[2]);
  const zMin = zs.length ? Math.min(...zs) : 0;
  const zMax = zs.length ? Math.max(...zs) : 0;
  // Level the land settles to away from the network: just below the lowest pipe, so the
  // outer slope is gentle rather than a cliff.
  const base = zMin - sink * 1.4;
  const falloff = extent * 0.18;

  /** Returns [height, distance to nearest node]. */
  const sample = (x: number, y: number): [number, number] => {
    const ci = Math.floor(x / cell), cj = Math.floor(y / cell);
    let wsum = 0, zsum = 0, nearest = Infinity, found = 0;
    for (let r = 0; r < 40 && found < 6; r++) {
      for (let i = ci - r; i <= ci + r; i++) for (let j = cj - r; j <= cj + r; j++) {
        if (Math.max(Math.abs(i - ci), Math.abs(j - cj)) !== r) continue;
        for (const n of grid.get(key(i, j)) ?? []) {
          const d = Math.hypot(x - n.position[0], y - n.position[1]);
          const w = 1 / Math.max(d * d, 1e-6);
          wsum += w;
          zsum += w * n.position[2];
          nearest = Math.min(nearest, d);
          found++;
        }
      }
    }
    if (!found) return [base, Infinity];
    const idw = zsum / wsum - sink;
    const t = Math.min(1, nearest / falloff);
    const ease = t * t * (3 - 2 * t);
    return [idw + (base - idw) * ease, nearest];
  };

  // Height-field mesh over the network's area plus a generous margin.
  const xs = nodes.map((n) => n.position[0]);
  const ys = nodes.map((n) => n.position[1]);
  const pad = extent * 0.45;
  const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad;
  const y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
  const N = 140; // vertices per side
  const positions = new Float32Array(N * N * 3);
  const colors = new Float32Array(N * N * 3);
  const fades = new Float32Array(N * N * 2);
  const heights = new Float32Array(N * N);
  const low = [0.84, 0.89, 0.82], high = [0.75, 0.83, 0.71];
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const x = x0 + ((x1 - x0) * i) / (N - 1);
      const y = y0 + ((y1 - y0) * j) / (N - 1);
      const [z, nearest] = sample(x, y);
      const v = j * N + i;
      positions.set([x, y, z], v * 3);
      heights[v] = z;
      // Greener on high ground; fully opaque near the network, fading to transparent away from it.
      const h = zMax > zMin ? Math.max(0, Math.min(1, (z + sink - zMin) / (zMax - zMin))) : 0.5;
      const f = Math.min(1, Math.max(0, (nearest - falloff * 0.35) / (falloff * 0.9)));
      for (let c = 0; c < 3; c++) colors[v * 3 + c] = low[c] + (high[c] - low[c]) * h;
      fades[v * 2] = 1 - f * f * (3 - 2 * f);
    }
  }
  // Normals from central differences.
  const normals = new Float32Array(N * N * 3);
  const dx = (x1 - x0) / (N - 1), dy = (y1 - y0) / (N - 1);
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const hL = heights[j * N + Math.max(i - 1, 0)], hR = heights[j * N + Math.min(i + 1, N - 1)];
      const hD = heights[Math.max(j - 1, 0) * N + i], hU = heights[Math.min(j + 1, N - 1) * N + i];
      const nx = -(hR - hL) / (2 * dx), ny = -(hU - hD) / (2 * dy);
      const len = Math.hypot(nx, ny, 1);
      normals.set([nx / len, ny / len, 1 / len], (j * N + i) * 3);
    }
  }
  const indices = new Uint32Array((N - 1) * (N - 1) * 6);
  let k = 0;
  for (let j = 0; j < N - 1; j++) {
    for (let i = 0; i < N - 1; i++) {
      const a = j * N + i, b = a + 1, c = a + N, d = c + 1;
      indices.set([a, b, d, a, d, c], k);
      k += 6;
    }
  }
  const mesh = new Geometry({
    topology: "triangle-list",
    attributes: {
      POSITION: { size: 3, value: positions },
      NORMAL: { size: 3, value: normals },
      COLOR_0: { size: 3, value: colors },
      TEXCOORD_0: { size: 2, value: fades },
    },
    indices: { size: 1, value: indices },
  });
  return { mesh, height: (x: number, y: number) => sample(x, y)[0] };
}


/** A procedurally placed tree. `kind` picks a conifer (cone) or a broadleaf (sphere) canopy. */
type Tree = { position: P3; height: number; radius: number; kind: "conifer" | "broadleaf"; shade: number };

/** Deterministic pseudo-random in [0, 1) from two numbers, so trees don't move between renders. */
function hash2(x: number, y: number): number {
  const h = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return h - Math.floor(h);
}

/**
 * Scatters trees in clumps across the network's area, keeping a clear corridor around every
 * pipe and node so trees never sit on or hide the network. Trees stand on the terrain surface.
 */
function buildTrees(
  links: DrawLink[],
  nodes: DrawNode[],
  unit: number,
  extent: number,
  groundHeight: (x: number, y: number) => number,
): Tree[] {
  if (nodes.length === 0) return [];
  const clearance = unit * 2.6;
  const cell = clearance * 2;
  const key = (i: number, j: number) => `${i},${j}`;

  // Spatial hash of pipe segments (2D) for fast clearance checks.
  const segGrid = new Map<string, [P3, P3][]>();
  for (const d of links) {
    for (let k = 1; k < d.path.length; k++) {
      const a = d.path[k - 1], b = d.path[k];
      const i0 = Math.floor(Math.min(a[0], b[0]) / cell), i1 = Math.floor(Math.max(a[0], b[0]) / cell);
      const j0 = Math.floor(Math.min(a[1], b[1]) / cell), j1 = Math.floor(Math.max(a[1], b[1]) / cell);
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
        const list = segGrid.get(key(i, j)) ?? segGrid.set(key(i, j), []).get(key(i, j))!;
        list.push([a, b]);
      }
    }
  }
  // Spatial hash of nodes for clearance and terrain height.
  const nodeGrid = new Map<string, DrawNode[]>();
  for (const n of nodes) {
    const k = key(Math.floor(n.position[0] / cell), Math.floor(n.position[1] / cell));
    (nodeGrid.get(k) ?? nodeGrid.set(k, []).get(k)!).push(n);
  }
  const segDist = (x: number, y: number, a: P3, b: P3) => {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const len2 = dx * dx + dy * dy || 1e-9;
    const t = Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / len2));
    return Math.hypot(x - (a[0] + t * dx), y - (a[1] + t * dy));
  };
  const clearOfNetwork = (x: number, y: number) => {
    const ci = Math.floor(x / cell), cj = Math.floor(y / cell);
    for (let i = ci - 1; i <= ci + 1; i++) for (let j = cj - 1; j <= cj + 1; j++) {
      for (const [a, b] of segGrid.get(key(i, j)) ?? []) if (segDist(x, y, a, b) < clearance) return false;
      for (const n of nodeGrid.get(key(i, j)) ?? []) if (Math.hypot(x - n.position[0], y - n.position[1]) < clearance * 1.6) return false;
    }
    return true;
  };
  const xs = nodes.map((n) => n.position[0]);
  const ys = nodes.map((n) => n.position[1]);
  const pad = extent * 0.12;
  const minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad;
  const minY = Math.min(...ys) - pad, maxY = Math.max(...ys) + pad;
  const step = extent / 55;
  const trees: Tree[] = [];
  for (let x = minX; x <= maxX; x += step) {
    for (let y = minY; y <= maxY; y += step) {
      // Jitter within the cell, and keep only where a smooth "forest" field is high, so trees clump.
      const jx = x + (hash2(x, y) - 0.5) * step * 0.5;
      const jy = y + (hash2(y, x) - 0.5) * step * 0.5;
      const field = Math.sin(jx / (extent * 0.09)) * Math.cos(jy / (extent * 0.11)) + 0.6 * Math.sin((jx + jy) / (extent * 0.05));
      if (field < 0.35 || hash2(jx * 0.7, jy * 1.3) > 0.8) continue;
      if (!clearOfNetwork(jx, jy)) continue;
      const r = hash2(jx * 3.1, jy * 2.7);
      trees.push({
        position: [jx, jy, groundHeight(jx, jy)],
        height: unit * (1.6 + 1.4 * r),
        radius: unit * (0.38 + 0.3 * r),
        kind: hash2(jx * 5.3, jy * 0.9) > 0.45 ? "conifer" : "broadleaf",
        shade: 0.85 + 0.3 * hash2(jx * 1.9, jy * 4.1),
      });
    }
  }
  return trees;
}

/** One straight piece of a 3D pipe: a unit cylinder placed by `position` and shaped by `matrix`. */
/** `collar` marks flanges and hubs (drawn in the fitting color). */
type TubeSegment = { target: DrawLink; position: P3; matrix: number[]; collar?: boolean };
/** A sphere at a junction, sized to the widest pipe meeting there, so tube ends join smoothly. */
type TubeJoint = { target: DrawNode; position: P3; radius: number };

/**
 * Column-major transform that maps the unit cylinder (radius 1, height 1, along Z, centered
 * on the origin) onto the segment a→b with the given radius.
 */
function segmentMatrix(a: P3, b: P3, radius: number): number[] {
  const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const len = Math.hypot(d[0], d[1], d[2]) || 1e-6;
  const w = [d[0] / len, d[1] / len, d[2] / len];
  // Any vector not parallel to w, to build an orthonormal basis (u, v, w).
  const helper = Math.abs(w[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  let u = [helper[1] * w[2] - helper[2] * w[1], helper[2] * w[0] - helper[0] * w[2], helper[0] * w[1] - helper[1] * w[0]];
  const ul = Math.hypot(u[0], u[1], u[2]);
  u = [u[0] / ul, u[1] / ul, u[2] / ul];
  const v = [w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2], w[0] * u[1] - w[1] * u[0]];
  return [
    u[0] * radius, u[1] * radius, u[2] * radius, 0,
    v[0] * radius, v[1] * radius, v[2] * radius, 0,
    d[0], d[1], d[2], 0,
    0, 0, 0, 1,
  ];
}

function buildTubes(links: DrawLink[], nodes: DrawNode[]) {
  const tubes: TubeSegment[] = [];
  const jointRadius = new Map<string, number>();
  for (const d of links) {
    for (let i = 1; i < d.path.length; i++) {
      const a = d.path[i - 1];
      const b = d.path[i];
      tubes.push({ target: d, position: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2], matrix: segmentMatrix(a, b, d.radius) });
    }
    // Flanged joints, as on industrial pipe models: a short hub then a bolted pair of wide,
    // thin flange discs near each end, plus a mid-span flange pair on long runs.
    const r = d.radius;
    /** Hub + two flange discs centred `at` distance `t` along `dir` from `origin`. */
    const flangePair = (origin: P3, dir: P3, t: number, withHub: boolean) => {
      const at = (k: number): P3 => [origin[0] + dir[0] * k, origin[1] + dir[1] * k, origin[2] + dir[2] * k];
      const disc = r * 0.2;
      const gap = r * 0.05;
      for (const offset of [-(gap / 2 + disc / 2), gap / 2 + disc / 2]) {
        const c = t + offset;
        tubes.push({ target: d, collar: true, position: at(c), matrix: segmentMatrix(at(c - disc / 2), at(c + disc / 2), r * 1.72) });
      }
      if (withHub) {
        const h0 = t + gap / 2 + disc, h1 = h0 + r * 0.75;
        tubes.push({ target: d, collar: true, position: at((h0 + h1) / 2), matrix: segmentMatrix(at(h0), at(h1), r * 1.2) });
      }
    };
    const ends: [P3, P3][] = [
      [d.path[0], d.path[1]],
      [d.path[d.path.length - 1], d.path[d.path.length - 2]],
    ];
    for (const [node, toward] of ends) {
      const dx = toward[0] - node[0], dy = toward[1] - node[1], dz = toward[2] - node[2];
      const segLen = Math.hypot(dx, dy, dz);
      if (segLen < r * 8) continue; // too short to show cleanly
      flangePair(node, [dx / segLen, dy / segLen, dz / segLen], r * 1.7, true);
    }
    if (d.path.length === 2) {
      const [a0, a1] = d.path;
      const len = Math.hypot(a1[0] - a0[0], a1[1] - a0[1], a1[2] - a0[2]);
      if (len > r * 26) flangePair(a0, [(a1[0] - a0[0]) / len, (a1[1] - a0[1]) / len, (a1[2] - a0[2]) / len], len / 2, false);
    }
    for (const id of [d.link.from, d.link.to]) jointRadius.set(id, Math.max(jointRadius.get(id) ?? 0, d.radius));
  }
  const joints: TubeJoint[] = nodes
    .filter((n) => n.node.kind === "junction" && jointRadius.has(n.node.id))
    // A sphere the size of the widest pipe rounds each corner into a smooth bend.
    .map((n) => ({ target: n, position: n.position, radius: jointRadius.get(n.node.id)! * 1.02 }));
  return { tubes, joints };
}

/**
 * SimpleMeshLayer whose mesh carries per-vertex opacity in TEXCOORD_0.x, so the terrain can
 * fade to fully transparent at its edges instead of ending in a visible slab.
 */
class FadingMeshLayer<D> extends SimpleMeshLayer<D> {
  static layerName = "FadingMeshLayer";
  getShaders() {
    const shaders = super.getShaders();
    shaders.inject = { ...(shaders.inject ?? {}), "fs:DECKGL_FILTER_COLOR": "color.a *= geometry.uv.x;\n" };
    return shaders;
  }
}

// Unit-sized meshes, centered on the origin with Z up. Instances are scaled by `unit`.
const CYLINDER = new TruncatedConeGeometry({ bottomRadius: 1, topRadius: 1, height: 1, nradial: 40, topCap: true, bottomCap: true, verticalAxis: "z" });
const CONE = new TruncatedConeGeometry({ bottomRadius: 1, topRadius: 0, height: 1, nradial: 28, bottomCap: true, verticalAxis: "z" });
const SPHERE = new SphereGeometry({ radius: 1, nlat: 20, nlong: 32 });
/** Faceted sphere for tree canopies (low-poly look). */
const ICOSPHERE = new IcoSphereGeometry({ iterations: 1 });
/** Slightly glossy, like coated pipe. */
/** Matte painted steel: soft diffuse shading, almost no specular, like a studio clay render. */
const PIPE_MATERIAL = { ambient: 0.55, diffuse: 0.62, shininess: 10, specularColor: [36, 36, 36] as [number, number, number] };

const MATERIAL = { ambient: 0.45, diffuse: 0.6, shininess: 24, specularColor: [255, 255, 255] as [number, number, number] };
/** Matte foliage and bark: almost no specular, so the key light shades rather than whites them out. */
const FOLIAGE_MATERIAL = { ambient: 0.55, diffuse: 0.6, shininess: 8, specularColor: [28, 30, 26] as [number, number, number] };

type Rgb = [number, number, number];
const shade = (c: readonly number[], k: number): Rgb => c.map((v) => Math.round(Math.min(255, v * k))) as Rgb;

/**
 * A leak on the map: where water bursts out of the pipe (`position`, the top of the pipe at its
 * midpoint), the jet height and the puddle radius, both scaled by leak size.
 */
type LeakMarker = { marker: true; leak: Leak; link: DrawLink; position: P3; jetHeight: number; puddleRadius: number; index: number };

/** Jet height and puddle radius, in model units, by leak size. */
const LEAK_JET: Record<Leak["size"], { height: number; puddle: number }> = {
  small: { height: 3, puddle: 1.6 },
  medium: { height: 6.5, puddle: 2.6 },
  large: { height: 11, puddle: 3.8 },
};
/** Droplets per jet. */
const JET_PARTICLES = 460;
/** Seconds for the tallest (large) jet's water to rise and fall back. All jets share the same gravity. */
const JET_LARGE_FLIGHT_SECONDS = 1.9;

/**
 * The solid core of a jet: a tapered water column from the pipe up most of the jet height,
 * flickering slightly in height so it doesn't look static.
 */
type JetColumn = { marker: LeakMarker; position: P3; scale: P3 };

function buildJetColumns(markers: LeakMarker[], unit: number, t: number): JetColumn[] {
  return markers.map((m) => {
    const flicker = 0.92 + 0.08 * Math.sin(t * 9 + m.index * 1.7) * Math.sin(t * 5.3 + m.index);
    const h = m.jetHeight * 0.82 * flicker;
    const r = Math.max(m.link.radius * 0.75, unit * 0.18) * (0.8 + 0.4 * (m.jetHeight / (LEAK_JET.large.height * unit)));
    return { marker: m, position: [m.position[0], m.position[1], m.position[2] + h / 2], scale: [r, r, h] };
  });
}

/** Cone flaring outward as it rises, like water fanning out of a burst hydrant. */
const JET_CONE = new TruncatedConeGeometry({ bottomRadius: 0.55, topRadius: 1, height: 1, nradial: 24, verticalAxis: "z" });

/** One droplet of a leak's water jet, at the current animation time. */
type Droplet = { marker: LeakMarker; position: P3; radius: number; color: [number, number, number, number] };

/**
 * Fire-hydrant style spray: each droplet follows a ballistic arc out of the pipe and back down,
 * looping forever with its own phase so the jet looks continuous. Mostly near-vertical (a dense
 * column) with some wider-angle spray. Deterministic per droplet, so the jet is stable over time.
 */
function buildSpray(markers: LeakMarker[], unit: number, t: number): Droplet[] {
  const largeH = LEAK_JET.large.height * unit;
  const halfFlight = JET_LARGE_FLIGHT_SECONDS / 2;
  const gravity = (2 * largeH) / (halfFlight * halfFlight);
  const out: Droplet[] = [];
  for (const m of markers) {
    const v0 = Math.sqrt(2 * gravity * m.jetHeight);
    for (let i = 0; i < JET_PARTICLES; i++) {
      const h = (k: number) => hash2(i * 12.9898 + k * 78.233, m.index * 37.719 + k * 4.581);
      const azimuth = h(1) * Math.PI * 2;
      // Squaring biases toward vertical: a tight column with a skirt of wider spray.
      const tilt = h(2) * h(2) * 0.42;
      const speed = v0 * (0.72 + 0.28 * h(3));
      const vz = speed * Math.cos(tilt);
      const vh = speed * Math.sin(tilt);
      const flight = (2 * vz) / gravity;
      const age = (t + h(4) * flight) % flight;
      const life = age / flight;
      const [x, y, z] = m.position;
      const fade = Math.min(1, life / 0.06) * (1 - life * life * life);
      // White foam near the top of the arc and in the outer spray; blue water in the column.
      const height = (vz * age - 0.5 * gravity * age * age) / Math.max(m.jetHeight, 1e-6);
      const mist = h(5) > 0.78 || (height > 0.8 && h(7) > 0.45) || tilt > 0.3;
      out.push({
        marker: m,
        position: [x + Math.cos(azimuth) * vh * age, y + Math.sin(azimuth) * vh * age, z + vz * age - 0.5 * gravity * age * age],
        // Droplets spread and grow a little as they fly.
        radius: unit * (0.07 + 0.11 * h(6)) * (1 + life * 0.9),
        color: mist
          ? [236, 246, 252, Math.round(210 * fade)]
          : [88, 166, 226, Math.round(235 * fade)],
      });
    }
  }
  return out;
}

/** Point halfway along a path by length, so markers sit on curved pipes too. */
function pathMidpoint(path: P3[]): P3 {
  const seg = path.slice(1).map((p, i) => Math.hypot(p[0] - path[i][0], p[1] - path[i][1], p[2] - path[i][2]));
  let remaining = seg.reduce((a, b) => a + b, 0) / 2;
  for (let i = 0; i < seg.length; i++) {
    if (remaining <= seg[i] || i === seg.length - 1) {
      const t = seg[i] > 0 ? Math.min(remaining / seg[i], 1) : 0;
      const a = path[i], b = path[i + 1];
      return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    }
    remaining -= seg[i];
  }
  return path[0];
}

function buildLeakMarkers(g: ReturnType<typeof buildGeometry>, leaks: Leak[]): LeakMarker[] {
  const byId = new Map(g.links.map((d) => [d.link.id, d]));
  return leaks.flatMap((leak, i) => {
    const link = byId.get(leak.linkId);
    if (!link) return [];
    const [x, y, z] = pathMidpoint(link.path);
    const size = LEAK_JET[leak.size];
    return [
      {
        marker: true as const,
        leak,
        link,
        // Water bursts from the top of the pipe.
        position: [x, y, z + link.radius] as P3,
        jetHeight: size.height * g.unit,
        puddleRadius: size.puddle * g.unit,
        index: i,
      },
    ];
  });
}

/** One placed copy of a mesh. `target` is what a click on it selects. */
type ModelPart = { target: DrawNode | DrawValve; position: P3; scale: P3; yaw: number };

function isSelected(t: ModelPart["target"], nodeId: string | null, linkId: string | null) {
  return "node" in t ? t.node.id === nodeId : t.link.id === linkId;
}

function partLayer(
  id: string,
  mesh: TruncatedConeGeometry | IcoSphereGeometry,
  data: ModelPart[],
  color: Rgb,
  selNode: string | null,
  selLink: string | null,
  /** Pipe styling: glossy pipe material, and a shell alpha below 255 to go translucent like the pipes. */
  pipeStyle?: { alpha: number },
) {
  const alpha = pipeStyle?.alpha ?? 255;
  return new SimpleMeshLayer<ModelPart>({
    id,
    data,
    mesh,
    material: pipeStyle ? PIPE_MATERIAL : MATERIAL,
    getPosition: (d) => d.position,
    getScale: (d) => d.scale,
    getOrientation: (d) => [0, d.yaw, 0],
    getColor: (d) =>
      isSelected(d.target, selNode, selLink) ? [...COLORS.selected, 255] as [number, number, number, number] : [...color, alpha] as [number, number, number, number],
    pickable: true,
    autoHighlight: true,
    highlightColor: [18, 168, 182, 255],
    // Translucent shells don't write depth (so the water inside shows) and skip inner faces.
    parameters: alpha < 255 ? { depthWriteEnabled: false, cullMode: "back" } : { cullMode: "back" },
    updateTriggers: { getColor: [selNode, selLink, alpha] },
  });
}

/** A model part placed with a full transform (for parts that follow a pipe's 3D direction). */
type AlignedPart = { target: DrawValve; position: P3; matrix: number[] };

function alignedPartLayer(id: string, data: AlignedPart[], color: Rgb, selNode: string | null, selLink: string | null, alpha: number) {
  return new SimpleMeshLayer<AlignedPart>({
    id,
    data,
    mesh: CYLINDER,
    material: PIPE_MATERIAL,
    getPosition: (d) => d.position,
    getTransformMatrix: (d) => d.matrix,
    getColor: (d) =>
      isSelected(d.target, selNode, selLink)
        ? ([...COLORS.selected, 255] as [number, number, number, number])
        : ([...color, alpha] as [number, number, number, number]),
    pickable: true,
    autoHighlight: true,
    highlightColor: [18, 168, 182, 255],
    parameters: alpha < 255 ? { depthWriteEnabled: false, cullMode: "back" } : { cullMode: "back" },
    updateTriggers: { getColor: [selNode, selLink, alpha] },
  });
}

/** 3D models for tanks (water tower), reservoirs (open basin) and valves (body + handwheel). */
/** `shellAlpha` matches the pipes: below 255 while water is animating so tanks and reservoirs show their water. */
function modelLayers(g: ReturnType<typeof buildGeometry>, selNode: string | null, selLink: string | null, shellAlpha: number) {
  const u = g.unit;
  const tanks = g.nodes.filter((n) => n.node.kind === "tank");
  const reservoirs = g.nodes.filter((n) => n.node.kind === "reservoir");

  // Tank: column from the ground plane up to the tank's elevation, then the tank body and a conical roof.
  const BODY_R = 2.2 * u, BODY_H = 2.4 * u, ROOF_H = 1.1 * u;
  const tankLegs: ModelPart[] = tanks.map((t) => {
    const [x, y, z] = t.position;
    const h = Math.max(z, 0.5 * u);
    return { target: t, position: [x, y, h / 2 - (h - z)], scale: [0.4 * u, 0.4 * u, h], yaw: 0 };
  });
  const tankBodies: ModelPart[] = tanks.map((t) => {
    const [x, y, z] = t.position;
    return { target: t, position: [x, y, z + BODY_H / 2], scale: [BODY_R, BODY_R, BODY_H], yaw: 0 };
  });
  const tankWater: ModelPart[] = tanks.map((t) => {
    const [x, y, z] = t.position;
    const h = BODY_H * 0.6;
    return { target: t, position: [x, y, z + h / 2 + 0.02 * u], scale: [BODY_R * 0.86, BODY_R * 0.86, h], yaw: 0 };
  });
  const tankRoofs: ModelPart[] = tanks.map((t) => {
    const [x, y, z] = t.position;
    return { target: t, position: [x, y, z + BODY_H + ROOF_H / 2], scale: [BODY_R * 1.1, BODY_R * 1.1, ROOF_H], yaw: 0 };
  });

  // Reservoir: a wide, low basin with a water surface just inside the rim.
  const BASIN_R = 3.6 * u, BASIN_H = 1.2 * u;
  const basins: ModelPart[] = reservoirs.map((r) => {
    const [x, y, z] = r.position;
    return { target: r, position: [x, y, z + BASIN_H / 2], scale: [BASIN_R, BASIN_R, BASIN_H], yaw: 0 };
  });
  const water: ModelPart[] = reservoirs.map((r) => {
    const [x, y, z] = r.position;
    // Clearly above the basin's solid top (not a hair above it), so the two never z-fight and
    // flip between gray and blue at different zoom levels.
    return { target: r, position: [x, y, z + BASIN_H + 0.18 * u], scale: [BASIN_R * 0.86, BASIN_R * 0.86, 0.12 * u], yaw: 0 };
  });

  // Valve: an in-line fitting aligned to the pipe in 3D (follows its slope, not just its
  // heading): a collar body with a flange at each end, and a stem with a handwheel on top.
  const linkById = new Map(g.links.map((d) => [d.link.id, d]));
  const valveBodies: AlignedPart[] = [];
  const valveFlanges: AlignedPart[] = [];
  const valveStems: ModelPart[] = [];
  const valveWheels: AlignedPart[] = [];
  const valveHubs: ModelPart[] = [];
  for (const v of g.valves) {
    const d = linkById.get(v.link.id);
    if (!d) continue;
    const a = d.path[0], b = d.path[d.path.length - 1];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) || 1;
    const dir: P3 = [(b[0] - a[0]) / len, (b[1] - a[1]) / len, (b[2] - a[2]) / len];
    const at = (t: number): P3 => [v.position[0] + dir[0] * t, v.position[1] + dir[1] * t, v.position[2] + dir[2] * t];
    const bodyR = Math.max(d.radius * 1.9, u * 0.42);
    const bodyL = Math.min(u * 1.3, len * 0.8);
    valveBodies.push({ target: v, position: v.position, matrix: segmentMatrix(at(-bodyL / 2), at(bodyL / 2), bodyR) });
    for (const end of [-1, 1]) {
      const c = at((end * bodyL) / 2);
      const half = u * 0.07;
      const p0: P3 = [c[0] - dir[0] * half, c[1] - dir[1] * half, c[2] - dir[2] * half];
      const p1: P3 = [c[0] + dir[0] * half, c[1] + dir[1] * half, c[2] + dir[2] * half];
      valveFlanges.push({ target: v, position: c, matrix: segmentMatrix(p0, p1, bodyR * 1.22) });
    }
    const top = v.position[2] + bodyR;
    const stemH = u * 0.7;
    valveStems.push({ target: v, position: [v.position[0], v.position[1], top + stemH / 2], scale: [u * 0.07, u * 0.07, stemH], yaw: 0 });
    // Spoked handwheel: a rim of short segments around a circle, five spokes and a hub.
    const wz = top + stemH;
    const R = u * 0.5;
    const rimSegments = 20;
    const rim = (k: number): P3 => {
      const a = (k / rimSegments) * Math.PI * 2;
      return [v.position[0] + Math.cos(a) * R, v.position[1] + Math.sin(a) * R, wz];
    };
    for (let k = 0; k < rimSegments; k++) {
      const a0 = rim(k), a1 = rim(k + 1);
      valveWheels.push({ target: v, position: [(a0[0] + a1[0]) / 2, (a0[1] + a1[1]) / 2, wz], matrix: segmentMatrix(a0, a1, u * 0.05) });
    }
    const centre: P3 = [v.position[0], v.position[1], wz];
    for (let k = 0; k < 5; k++) {
      const a = (k / 5) * Math.PI * 2 + 0.3;
      const end: P3 = [centre[0] + Math.cos(a) * R, centre[1] + Math.sin(a) * R, wz];
      valveWheels.push({ target: v, position: [(centre[0] + end[0]) / 2, (centre[1] + end[1]) / 2, wz], matrix: segmentMatrix(centre, end, u * 0.03) });
    }
    valveHubs.push({ target: v, position: centre, scale: [u * 0.1, u * 0.1, u * 0.1], yaw: 0 });
  }

  return [
    // Tanks and reservoirs share the pipes' material and color. Water goes first (opaque),
    // then the translucent shells over it.
    partLayer("tank-water", CYLINDER, tankWater, [...COLORS.waterBranch] as Rgb, selNode, selLink, { alpha: 255 }),
    partLayer("reservoir-water", CYLINDER, water, [...COLORS.waterBranch] as Rgb, selNode, selLink, { alpha: 255 }),
    partLayer("tank-legs", CYLINDER, tankLegs, shade(COLORS.tube, 0.85), selNode, selLink, { alpha: 255 }),
    partLayer("tank-bodies", CYLINDER, tankBodies, [...COLORS.tube] as Rgb, selNode, selLink, { alpha: shellAlpha }),
    partLayer("tank-roofs", CONE, tankRoofs, shade(COLORS.tube, 0.92), selNode, selLink, { alpha: shellAlpha }),
    partLayer("reservoir-basins", CYLINDER, basins, [...COLORS.tube] as Rgb, selNode, selLink, { alpha: shellAlpha }),
    // Valves in the same style: pipe-colored, glossy, the body translucent with the pipes.
    alignedPartLayer("valve-flanges", valveFlanges, shade(COLORS.tube, 0.82), selNode, selLink, 255),
    partLayer("valve-stems", CYLINDER, valveStems, shade(COLORS.tube, 0.78), selNode, selLink, { alpha: 255 }),
    alignedPartLayer("valve-wheels", valveWheels, shade(COLORS.tube, 0.9), selNode, selLink, 255),
    partLayer("valve-hubs", CYLINDER, valveHubs, shade(COLORS.tube, 0.85), selNode, selLink, { alpha: 255 }),
    alignedPartLayer("valve-bodies", valveBodies, [...COLORS.tube] as Rgb, selNode, selLink, shellAlpha),
  ];
}

/** Matches the CSS breakpoint where the panel moves from the left side to the top. */
const NARROW = 640;

function effectiveInset(wideInset: Props["inset"]): Props["inset"] {
  return window.innerWidth <= NARROW ? { left: 0, bottom: wideInset.bottom } : wideInset;
}

/**
 * Camera target that puts `center` in the middle of the free screen area (right of
 * the panel) rather than the middle of the window.
 */
function offsetTarget(center: P3, scale: number, rotationOrbit: number, inset: Props["inset"]): P3 {
  const a = (rotationOrbit * Math.PI) / 180;
  const shiftRight = inset.left / 2 / scale;
  const shiftUp = inset.bottom / 2 / scale;
  return [
    center[0] - shiftRight * Math.cos(a) - shiftUp * Math.sin(a) * 0.5,
    center[1] + shiftRight * Math.sin(a) - shiftUp * Math.cos(a) * 0.5,
    center[2],
  ];
}

/** View state that centers and zooms on a clicked pipe or node, keeping the current camera angle. */
/**
 * Size of the map's own container (not the window), so fitting stays correct with an app
 * header and status bar around the map. Updated by the component as the container resizes.
 */
let viewport = { w: window.innerWidth, h: window.innerHeight };

function focusViewState(
  g: ReturnType<typeof buildGeometry>,
  obj: DrawLink | DrawNode | DrawValve,
  from: OrbitViewState,
  wideInset: Props["inset"],
): OrbitViewState {
  const inset = effectiveInset(wideInset);
  const freeW = Math.max(viewport.w - inset.left, 200);
  let center: P3;
  let span: number;
  if ("path" in obj) {
    const a = obj.path[0];
    const b = obj.path[obj.path.length - 1];
    center = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
    span = Math.max(Math.hypot(b[0] - a[0], b[1] - a[1]), g.unit * 6);
  } else {
    center = obj.position;
    span = g.unit * 14;
  }
  // The focused element fills about 40% of the free width.
  const scale = (0.4 * freeW) / span;
  const zoom = Math.min(Math.log2(scale), from.maxZoom ?? Infinity);
  const rotationOrbit = from.rotationOrbit ?? 0;
  return {
    ...from,
    target: offsetTarget(center, 2 ** zoom, rotationOrbit, inset),
    zoom,
    transitionDuration: 700,
    transitionInterpolator: new LinearInterpolator(["target", "zoom"]),
  };
}

function fitViewState(g: ReturnType<typeof buildGeometry>, wideInset: Props["inset"]): OrbitViewState {
  const inset = effectiveInset(wideInset);
  const w = Math.max(viewport.w - inset.left - 48, 200);
  const h = Math.max(viewport.h - inset.bottom - 48, 200);
  // A tilted view foreshortens y, so it can use a little more zoom than a flat fit.
  const scale = Math.min(w / g.extentX, (h / g.extentY) * 1.25) * 0.85;
  const rotationOrbit = -15;
  // Move the camera target so the network centers in the space right of the panel.
  // Screen-right in world space, for a camera orbiting around Z.
  return {
    target: offsetTarget([0, 0, g.height / 2], scale, rotationOrbit, inset),
    zoom: Math.log2(scale),
    rotationX: 50,
    rotationOrbit,
    minZoom: Math.log2(scale) - 4,
    maxZoom: Math.log2(scale) + 8,
  };
}

/**
 * Neutral studio lighting: soft ambient, a white key light from high above and to one side, and a
 * weaker white fill from the opposite side, so round pipes shade evenly without color casts.
 */
const LIGHTING = new LightingEffect({
  ambient: new AmbientLight({ color: [255, 255, 255], intensity: 0.66 }),
  key: new DirectionalLight({ color: [255, 255, 255], intensity: 1.05, direction: [-3, -5, -8] }),
  fill: new DirectionalLight({ color: [255, 255, 255], intensity: 0.38, direction: [4, 3, -3] }),
});

// Near plane: small enough that close-ups aren't sliced open, large enough to keep depth precision
// (too small and nearly coplanar surfaces flicker when zoomed out).
const VIEW = new OrbitView({ id: "orbit", orbitAxis: "Z", near: 0.04 });

export default function NetworkMap({ network, leaks, sensors = [], predictions = [], selection, onSelect, inset }: Props) {
  const geom = useMemo(() => buildGeometry(network), [network]);
  const leakMarkers = useMemo(() => buildLeakMarkers(geom, leaks), [geom, leaks]);
  const nodeById = useMemo(() => new Map(geom.nodes.map((d) => [d.node.id, d])), [geom]);
  const sensorMarkers = useMemo(
    () => sensors.flatMap((id) => (nodeById.has(id) ? [{ kind: "sensor" as const, id, node: nodeById.get(id)! }] : [])),
    [sensors, nodeById],
  );
  const predictionMarkers = useMemo(
    () =>
      predictions.flatMap((p) =>
        nodeById.has(p.node) ? [{ kind: "prediction" as const, id: p.node, timeHour: p.timeHour, node: nodeById.get(p.node)! }] : [],
      ),
    [predictions, nodeById],
  );
  const leakLinkIds = useMemo(() => new Set(leaks.map((l) => l.linkId)), [leaks]);

  // Water animation clock. Off for now (WATER_ANIMATION); the pipes stay translucent regardless.
  const [flowOn] = useState(() => WATER_ANIMATION && !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);
  const [flowTime, setFlowTime] = useState(0);
  const flowSpacing = geom.unit * 4;
  useEffect(() => {
    if (!flowOn) return;
    let frame = 0;
    const start = performance.now();
    // One dash spacing per second.
    const tick = (now: number) => {
      setFlowTime(((now - start) / 1000) * flowSpacing);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [flowOn, flowSpacing]);

  // Landscape (terrain + trees). Remembered per browser; storage may be unavailable.
  const [landscapeOn, setLandscapeOn] = useState(() => {
    try {
      return localStorage.getItem("waterboard.landscape") !== "off";
    } catch {
      return true;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem("waterboard.landscape", landscapeOn ? "on" : "off");
    } catch {
      // Not saved; the toggle still works for this session.
    }
  }, [landscapeOn]);
  const conifers = useMemo(() => geom.trees.filter((t) => t.kind === "conifer"), [geom]);
  const broadleafs = useMemo(() => geom.trees.filter((t) => t.kind === "broadleaf"), [geom]);

  // Leak spray clock, in seconds. Runs only while there are leaks; frozen mid-spray for reduced motion.
  const [leakTime, setLeakTime] = useState(0.7);
  const hasLeaks = leakMarkers.length > 0;
  useEffect(() => {
    if (!hasLeaks || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    let frame = 0;
    const tick = (now: number) => {
      setLeakTime(now / 1000);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [hasLeaks]);
  const spray = useMemo(() => buildSpray(leakMarkers, geom.unit, leakTime), [leakMarkers, geom.unit, leakTime]);
  const jetColumns = useMemo(() => buildJetColumns(leakMarkers, geom.unit, leakTime), [leakMarkers, geom.unit, leakTime]);
  const mapRef = useRef<HTMLDivElement>(null);
  // Measure the container before the first fit, and keep the measurement current.
  useLayoutEffect(() => {
    const el = mapRef.current;
    if (!el) return;
    const measure = () => {
      viewport = { w: el.clientWidth || window.innerWidth, h: el.clientHeight || window.innerHeight };
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  const [initialViewState, setInitialViewState] = useState(() => fitViewState(geom, inset));
  /** Latest camera, so a fly-to starts from wherever the user has moved it. */
  const viewRef = useRef<OrbitViewState>(initialViewState);

  useEffect(() => {
    const fitted = fitViewState(geom, inset);
    viewRef.current = fitted;
    setInitialViewState(fitted);
    // Refit only when the network changes, not when the panel resizes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geom]);

  const selectedLinkId = selection?.type === "link" ? selection.link.id : null;
  const selectedNodeId = selection?.type === "node" ? selection.node.id : null;

  // Fly to whatever gets selected, whether by clicking the map or picking from the leak list.
  const selectionKey = selectedLinkId ? `link:${selectedLinkId}` : selectedNodeId ? `node:${selectedNodeId}` : null;
  useEffect(() => {
    if (!selectionKey) return;
    const target = selectedLinkId
      ? geom.links.find((d) => d.link.id === selectedLinkId)
      : geom.nodes.find((d) => d.node.id === selectedNodeId);
    if (!target) return;
    // deck.gl only reports view changes from user input, so record programmatic moves here too.
    const focused = focusViewState(geom, target, viewRef.current, inset);
    viewRef.current = focused;
    setInitialViewState(focused);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectionKey, geom]);

  const layers = [
    // Terrain under the network. One mesh, vertex-colored.
    new FadingMeshLayer<{ id: string }>({
      id: "terrain",
      data: [{ id: "terrain" }],
      visible: landscapeOn,
      mesh: geom.terrainMesh,
      material: { ambient: 0.62, diffuse: 0.48, shininess: 6, specularColor: [40, 40, 40] },
      getPosition: [0, 0, 0],
      getColor: [255, 255, 255],
    }),
    new ScatterplotLayer<GroundDot>({
      id: "ground-dots",
      data: geom.groundDots,
      getPosition: (d) => d.position,
      // World-sized and lying flat, so dots shrink with distance like the ground they mark.
      getRadius: geom.unit * 0.22,
      radiusUnits: "common",
      radiusMinPixels: 0.6,
      billboard: false,
      getFillColor: (d) => [...COLORS.groundDot, Math.round(255 * d.alpha)] as [number, number, number, number],
      // Depth-tested against the terrain, so hills hide the dots behind and under them.
      parameters: { depthWriteEnabled: false },
    }),
    new PathLayer<DrawLink>({
      id: "pipe-shadows",
      data: geom.links,
      getPath: (d) => d.ground,
      getColor: COLORS.shadow as unknown as [number, number, number, number],
      getWidth: (d) => d.width,
      widthUnits: "pixels",
      capRounded: true,
    }),
    // Trees: trunk + canopy per tree, matte, not pickable so clicks go to the network.
    ...(() => {
      const canopyColor = (base: readonly number[]) => (d: Tree) => shade(base, d.shade);
      const trunkTop = (d: Tree) => d.position[2] + d.height * 0.35;
      return [
        new SimpleMeshLayer<Tree>({
          id: "tree-trunks",
          visible: landscapeOn,
          data: geom.trees,
          mesh: CYLINDER,
          material: FOLIAGE_MATERIAL,
          getPosition: (d) => [d.position[0], d.position[1], d.position[2] + d.height * 0.175],
          getScale: (d) => [d.radius * 0.16, d.radius * 0.16, d.height * 0.35],
          getColor: [...COLORS.trunk] as Rgb,
        }),
        new SimpleMeshLayer<Tree>({
          id: "tree-conifers",
          visible: landscapeOn,
          data: conifers,
          mesh: CONE,
          material: FOLIAGE_MATERIAL,
          getPosition: (d) => [d.position[0], d.position[1], trunkTop(d) + d.height * 0.325],
          getScale: (d) => [d.radius, d.radius, d.height * 0.65],
          getColor: canopyColor(COLORS.conifer),
        }),
        new SimpleMeshLayer<Tree>({
          id: "tree-broadleafs",
          visible: landscapeOn,
          data: broadleafs,
          mesh: ICOSPHERE,
          material: FOLIAGE_MATERIAL,
          getPosition: (d) => [d.position[0], d.position[1], trunkTop(d) + d.radius * 0.95],
          getScale: (d) => [d.radius * 1.1, d.radius * 1.1, d.radius * 1.15],
          getColor: canopyColor(COLORS.broadleaf),
        }),
      ];
    })(),
    // 3D pipe model: matte gray cylinders per segment, bolted flanges near pipe ends, smooth bends at junctions.
    ...(() => {
      const alpha = PIPE_ALPHA;
      const tubeColor = (link: DrawLink, collar = false): [number, number, number, number] => [
        ...(link.link.id === selectedLinkId
          ? COLORS.selected
          : leakLinkIds.has(link.link.id)
            ? COLORS.leakPipe
            : link.link.kind === "pump"
              ? COLORS.pump
              : collar
                ? COLORS.fitting
                : link.main
                  ? COLORS.tubeMain
                  : COLORS.tube),
        leakLinkIds.has(link.link.id) || link.link.id === selectedLinkId ? 255 : alpha,
      ] as [number, number, number, number];
      // Solid pipes write depth; translucent ones don't. Inner walls are always culled.
      const parameters = { depthWriteEnabled: alpha === 255, cullMode: "back" } as const;
      return [
        new SimpleMeshLayer<TubeSegment>({
          id: "pipe-tubes",
          data: geom.tubes,
          mesh: CYLINDER,
          material: PIPE_MATERIAL,
          getPosition: (d) => d.position,
          getTransformMatrix: (d) => d.matrix,
          getColor: (d) => tubeColor(d.target, d.collar),
          pickable: true,
          autoHighlight: true,
          highlightColor: [18, 168, 182, 220],
          parameters,
          updateTriggers: { getColor: [selectedLinkId, leakLinkIds, alpha] },
        }),
        new SimpleMeshLayer<TubeJoint>({
          id: "pipe-joints",
          data: geom.joints,
          mesh: SPHERE,
          material: PIPE_MATERIAL,
          getPosition: (d) => d.position,
          getScale: (d) => [d.radius, d.radius, d.radius],
          getColor: (d) =>
            [...(d.target.node.id === selectedNodeId ? COLORS.selected : COLORS.tube), d.target.node.id === selectedNodeId ? 255 : alpha] as [
              number,
              number,
              number,
              number,
            ],
          pickable: true,
          autoHighlight: true,
          highlightColor: [18, 168, 182, 220],
          parameters,
          updateTriggers: { getColor: [selectedNodeId, alpha] },
        }),
      ];
    })(),
    // Two passes of the same moving dashes: a wide faint halo, then a bright core.
    ...(["glow", "core"] as const).map(
      (pass) =>
        new FlowLayer<DrawFlow>({
          id: `water-flow-${pass}`,
          data: geom.flows,
          visible: flowOn,
          getPath: (d) => d.path,
          getDistances: (d) => d.distances,
          getColor: (d) =>
            [...(d.link.main ? COLORS.waterMain : COLORS.waterBranch), pass === "glow" ? 38 : 240] as [
              number,
              number,
              number,
              number,
            ],
          getWidth: (d) => (pass === "glow" ? d.width * 1.8 : d.width),
          widthUnits: "pixels",
          widthMinPixels: 1,
          time: flowTime,
          spacing: flowSpacing,
          dashRatio: 0.45,
          parameters: FLOW_PARAMETERS,
        }),
    ),
    ...modelLayers(geom, selectedNodeId, selectedLinkId, PIPE_ALPHA),
    // Leaks: a puddle spreading under the pipe and a jet of water bursting out of it, like an
    // open fire hydrant. Jet height and puddle size scale with the leak size.
    new ScatterplotLayer<LeakMarker>({
      id: "leak-puddles",
      data: leakMarkers,
      // On the ground just under the pipe.
      getPosition: (d) => [d.position[0], d.position[1], d.position[2] - d.link.radius * 2 - geom.unit * 0.5],
      getRadius: (d) => d.puddleRadius,
      getFillColor: [96, 168, 222, 92],
      getLineColor: [96, 168, 222, 150],
      stroked: true,
      lineWidthUnits: "pixels",
      getLineWidth: 1,
      billboard: false,
      pickable: true,
      parameters: { depthWriteEnabled: false },
    }),
    new SimpleMeshLayer<JetColumn>({
      id: "leak-jet-columns",
      data: jetColumns,
      mesh: JET_CONE,
      material: { ambient: 0.75, diffuse: 0.45, shininess: 90, specularColor: [255, 255, 255] },
      getPosition: (d) => d.position,
      getScale: (d) => d.scale,
      getColor: [120, 188, 236, 150],
      pickable: true,
      parameters: { depthWriteEnabled: false },
    }),
    // Simulated sensors (teal) and model-predicted leak locations (rings), floating just above the network.
    new ScatterplotLayer<(typeof sensorMarkers)[number]>({
      id: "sensors",
      data: sensorMarkers,
      getPosition: (d) => [d.node.position[0], d.node.position[1], d.node.position[2] + geom.unit * 0.9],
      getRadius: 6,
      radiusUnits: "pixels",
      getFillColor: [18, 168, 182],
      getLineColor: [255, 255, 255],
      stroked: true,
      lineWidthUnits: "pixels",
      getLineWidth: 2,
      billboard: true,
      pickable: true,
      parameters: { depthCompare: "always" },
    }),
    new ScatterplotLayer<(typeof predictionMarkers)[number]>({
      id: "predictions",
      data: predictionMarkers,
      getPosition: (d) => [d.node.position[0], d.node.position[1], d.node.position[2] + geom.unit * 0.9],
      getRadius: 13,
      radiusUnits: "pixels",
      filled: false,
      stroked: true,
      getLineColor: [124, 92, 255],
      lineWidthUnits: "pixels",
      getLineWidth: 3,
      billboard: true,
      pickable: true,
      parameters: { depthCompare: "always" },
    }),
    new ScatterplotLayer<Droplet>({
      id: "leak-spray",
      data: spray,
      getPosition: (d) => d.position,
      getRadius: (d) => d.radius,
      getFillColor: (d) => d.color,
      radiusUnits: "common",
      radiusMinPixels: 1.2,
      billboard: true,
      pickable: true,
      parameters: { depthWriteEnabled: false },
    }),
  ];

  function handleClick(info: PickingInfo) {
    // Sensor and prediction markers select the junction they sit on.
    const marker = info.object as { kind?: string; node?: DrawNode } | undefined;
    if (marker?.kind && marker.node) return onSelect({ type: "node", node: marker.node.node });
    const raw = info.object as DrawLink | DrawNode | ModelPart | TubeSegment | TubeJoint | LeakMarker | Droplet | JetColumn | undefined;
    const picked = raw && "marker" in raw && typeof raw.marker === "object" ? raw.marker : (raw as Exclude<typeof raw, Droplet | JetColumn>);
    if (!picked) return onSelect(null);
    const obj = "target" in picked ? picked.target : "marker" in picked ? picked.link : picked;
    if ("link" in obj) onSelect({ type: "link", link: obj.link });
    else onSelect({ type: "node", node: obj.node });
  }

  return (
    <div className="map" ref={mapRef}>
      <DeckGL
        views={VIEW}
        effects={[LIGHTING]}
        initialViewState={initialViewState}
        // Google Maps style: drag pans; Cmd/Ctrl/Alt/Shift + drag (or right-drag) rotates and tilts.
        controller={{ inertia: 300, dragMode: "pan" }}
        layers={layers}
        // Thin pipes are 1-3 px wide; give clicks and hovers some slack.
        pickingRadius={6}
        onClick={handleClick}
        onViewStateChange={({ viewState }) => {
          viewRef.current = viewState as OrbitViewState;
        }}
        getCursor={({ isHovering, isDragging }) => (isDragging ? "grabbing" : isHovering ? "pointer" : "grab")}
        getTooltip={({ object }) => {
          const raw = object as DrawLink | DrawNode | ModelPart | TubeSegment | TubeJoint | LeakMarker | Droplet | JetColumn | undefined;
          const p = raw && "marker" in raw && typeof raw.marker === "object" ? raw.marker : (raw as Exclude<typeof raw, Droplet | JetColumn>);
          if (!p) return null;
          if (p && "kind" in p) {
            const m = p as unknown as { kind: "sensor" | "prediction"; id: string; timeHour?: number };
            return {
              text: m.kind === "sensor" ? `Pressure sensor at ${m.id}` : `Model alarm at ${m.timeHour?.toFixed(2)} h: leak near ${m.id}`,
              className: "map-tooltip",
            };
          }
          if ("marker" in p) return { text: `${LEAK_SIZES[p.leak.size].label} leak on pipe ${p.leak.linkId}`, className: "map-tooltip" };
          const o = "target" in p ? p.target : p;
          const label = "link" in o ? `${o.link.kind} ${o.link.id}` : `${o.node.kind} ${o.node.id}`;
          return { text: label, className: "map-tooltip" };
        }}
      />
      <div className="map-controls">
        <button
          className={`map-btn ${landscapeOn ? "on" : ""}`}
          onClick={() => setLandscapeOn((v) => !v)}
          aria-pressed={landscapeOn}
          title="Show terrain and trees around the network"
        >
          Landscape
        </button>
        <button
          className="map-btn"
          onClick={() => {
            const fitted: OrbitViewState = {
              ...fitViewState(geom, inset),
              transitionDuration: 600,
              transitionInterpolator: new LinearInterpolator(["target", "zoom", "rotationX", "rotationOrbit"]),
            };
            viewRef.current = fitted;
            setInitialViewState(fitted);
          }}
        >
          Reset view
        </button>
      </div>
    </div>
  );
}
