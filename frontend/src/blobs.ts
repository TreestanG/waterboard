/**
 * Large file contents (.inp text) live in IndexedDB, which allows hundreds of MB,
 * instead of localStorage (~5 MB total). An in-memory cache keeps the current
 * session working even if IndexedDB is unavailable (private mode, blocked storage).
 */

const DB_NAME = "waterboard";
const STORE = "inp";
const memory = new Map<string, string>();

let dbPromise: Promise<IDBDatabase> | null = null;

function db(): Promise<IDBDatabase> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      try {
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = () => req.result.createObjectStore(STORE);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      } catch (e) {
        reject(e);
      }
    });
    dbPromise.catch(() => (dbPromise = null));
  }
  return dbPromise;
}

function run<T>(mode: IDBTransactionMode, op: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return db().then(
    (d) =>
      new Promise<T>((resolve, reject) => {
        const req = op(d.transaction(STORE, mode).objectStore(STORE));
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      }),
  );
}

/** Resolves to an error message when the text couldn't be persisted (it's still cached for this session). */
export async function putInp(id: string, text: string): Promise<string | null> {
  memory.set(id, text);
  try {
    await run("readwrite", (s) => s.put(text, id));
    return null;
  } catch {
    return "Couldn't save the file in this browser. It will be gone after you reload.";
  }
}

export async function getInp(id: string): Promise<string | null> {
  const cached = memory.get(id);
  if (cached !== undefined) return cached;
  try {
    const text = await run<string | undefined>("readonly", (s) => s.get(id));
    if (text !== undefined) memory.set(id, text);
    return text ?? null;
  } catch {
    return null;
  }
}

export async function deleteInp(id: string): Promise<void> {
  memory.delete(id);
  try {
    await run("readwrite", (s) => s.delete(id));
  } catch {
    // Nothing to clean up if storage is unavailable.
  }
}
