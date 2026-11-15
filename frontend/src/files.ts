import type { NetworkInput } from "./types";
import { parseInp } from "./inp";

export type ReadResult<T> = { ok: true; value: T } | { ok: false; error: string };

/** Reads and validates an uploaded .inp file into the fields a Network stores. */
export async function readInpFile(
  file: File,
): Promise<
  ReadResult<{ fields: Pick<NetworkInput, "inpFileName" | "inpSize" | "stats">; text: string; title: string | null }>
> {
  if (!file.name.toLowerCase().endsWith(".inp")) {
    return { ok: false, error: `“${file.name}” isn't an .inp file.` };
  }
  const text = await file.text();
  const parsed = parseInp(text);
  if (!parsed.ok) return parsed;
  return {
    ok: true,
    value: {
      fields: { inpFileName: file.name, inpSize: file.size, stats: parsed.network.stats },
      text,
      title: parsed.network.title,
    },
  };
}

export function readCsvFile(file: File): ReadResult<Pick<NetworkInput, "csvFileName" | "csvSize">> {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    return { ok: false, error: `“${file.name}” isn't a .csv file.` };
  }
  return { ok: true, value: { csvFileName: file.name, csvSize: file.size } };
}

export function formatBytes(n: number | null): string {
  if (n == null) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
