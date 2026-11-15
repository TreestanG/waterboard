import type { LeakSize } from "./types";

/**
 * Leak sizes offered in the UI. `areaCm2` is the hole area the backend will pass to
 * WNTR's leak model; flow depends on local pressure, so the UI describes size, not flow.
 */
export const LEAK_SIZES: Record<LeakSize, { label: string; areaCm2: number; hint: string }> = {
  small: { label: "Small", areaCm2: 1, hint: "Seeping joint or pinhole, ~1 cm² opening" },
  medium: { label: "Medium", areaCm2: 5, hint: "Cracked pipe, ~5 cm² opening" },
  large: { label: "Large", areaCm2: 20, hint: "Burst main, ~20 cm² opening" },
};

export const LEAK_SIZE_ORDER: LeakSize[] = ["small", "medium", "large"];

/** Hours from the simulation start as clock time: 6 → "06:00", 4.25 → "04:15", 30 → "Day 2, 06:00". */
export function formatStartHour(h: number): string {
  const totalMin = Math.round(h * 60);
  const day = Math.floor(totalMin / (24 * 60));
  const hh = String(Math.floor((totalMin % (24 * 60)) / 60)).padStart(2, "0");
  const mm = String(totalMin % 60).padStart(2, "0");
  return day > 0 ? `Day ${day + 1}, ${hh}:${mm}` : `${hh}:${mm}`;
}
