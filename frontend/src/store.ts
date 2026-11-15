import type { Leak, Network, NetworkInput } from "./types";
import { deleteInp, getInp, putInp } from "./blobs";

/**
 * Browser-only network store. Metadata lives in localStorage; .inp text lives in
 * IndexedDB (blobs.ts). Every function here maps to a future API call, so
 * components never touch storage directly.
 */

const KEY = "waterboard.networks.v3";
const LEGACY_KEY = "waterboard.networks.v2";

type LegacyNetwork = Network & { inpText?: string | null };

function load(): Network[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const list = JSON.parse(raw) as Network[];
      // Networks saved before leaks existed have no `leaks` field.
      if (Array.isArray(list)) return list.map((n) => ({ ...n, leaks: n.leaks ?? [] }));
    }
    // Migrate v2, which kept .inp text inline in localStorage.
    const legacy = localStorage.getItem(LEGACY_KEY);
    if (legacy) {
      const list = JSON.parse(legacy) as LegacyNetwork[];
      const migrated = list.map(({ inpText, ...n }) => {
        if (inpText) void putInp(n.id, inpText);
        return { ...n, leaks: n.leaks ?? [] };
      });
      localStorage.setItem(KEY, JSON.stringify(migrated));
      localStorage.removeItem(LEGACY_KEY);
      return migrated;
    }
  } catch {
    // Storage unavailable or corrupt: start empty.
  }
  return [];
}

let networks: Network[] = load();

function save(): string | null {
  try {
    localStorage.setItem(KEY, JSON.stringify(networks));
    return null;
  } catch {
    return "Couldn't save to browser storage. Changes will be lost when you reload.";
  }
}

/** Newest first. */
export function listNetworks(): Network[] {
  return [...networks].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export function getNetwork(id: string): Network | undefined {
  return networks.find((n) => n.id === id);
}

/** The network's .inp text, or null if it isn't stored in this browser. */
export function loadInp(id: string): Promise<string | null> {
  return getInp(id);
}

export async function createNetwork(
  input: NetworkInput,
  inpText: string,
  source: Network["source"] = "uploaded",
): Promise<{ network: Network; warning: string | null }> {
  const now = new Date().toISOString();
  const network: Network = {
    ...input,
    leaks: input.leaks ?? [],
    id: crypto.randomUUID(),
    source,
    createdAt: now,
    updatedAt: now,
  };
  const blobWarning = await putInp(network.id, inpText);
  networks = [...networks, network];
  return { network, warning: save() ?? blobWarning };
}

export async function updateNetwork(
  id: string,
  patch: Partial<NetworkInput>,
  inpText?: string,
): Promise<{ network: Network; warning: string | null }> {
  const existing = getNetwork(id);
  if (!existing) throw new Error(`Network ${id} not found`);
  const blobWarning = inpText !== undefined ? await putInp(id, inpText) : null;
  const network: Network = { ...existing, ...patch, updatedAt: new Date().toISOString() };
  networks = networks.map((n) => (n.id === id ? network : n));
  return { network, warning: save() ?? blobWarning };
}

/** Adds the leak, or replaces the existing leak on the same pipe (one leak per pipe). */
export function upsertLeak(networkId: string, leak: Leak): Promise<{ network: Network; warning: string | null }> {
  const existing = getNetwork(networkId);
  if (!existing) throw new Error(`Network ${networkId} not found`);
  const leaks = [...existing.leaks.filter((l) => l.linkId !== leak.linkId), leak];
  return updateNetwork(networkId, { leaks });
}

export function removeLeak(networkId: string, leakId: string): Promise<{ network: Network; warning: string | null }> {
  const existing = getNetwork(networkId);
  if (!existing) throw new Error(`Network ${networkId} not found`);
  return updateNetwork(networkId, { leaks: existing.leaks.filter((l) => l.id !== leakId) });
}

export async function deleteNetwork(id: string): Promise<{ warning: string | null }> {
  networks = networks.filter((n) => n.id !== id);
  await deleteInp(id);
  return { warning: save() };
}
