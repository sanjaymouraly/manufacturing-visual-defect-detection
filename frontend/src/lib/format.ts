export const pct = (v: number | null | undefined, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`);
/** Probability display that never claims certainty: Platt-calibrated outputs saturate at the extremes. */
export const prob = (v: number | null | undefined) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v > 0.999 ? ">99.9%" : v < 0.001 ? "<0.1%" : `${(v * 100).toFixed(1)}%`;
export const num = (v: number | null | undefined, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(d));
export const int = (v: number | null | undefined) => (v === null || v === undefined ? "—" : v.toLocaleString());
export const ms = (v: number | null | undefined) => (v === null || v === undefined ? "—" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(0)} ms`);
export const dt = (s: string | null | undefined) => {
  if (!s) return "—";
  const d = new Date(s.endsWith("Z") || s.includes("+") ? s : `${s}Z`);
  return d.toLocaleString(undefined, { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" });
};
export const ago = (s: string | null | undefined) => {
  if (!s) return "—";
  const d = new Date(s.endsWith("Z") || s.includes("+") ? s : `${s}Z`).getTime();
  const sec = Math.round((Date.now() - d) / 1000);
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.round(sec / 60)} min ago`;
  if (sec < 86400) return `${Math.round(sec / 3600)} h ago`;
  return `${Math.round(sec / 86400)} d ago`;
};
export const ci = (v: number[] | null | undefined, d = 3) => (v && v.length === 2 ? `[${v[0].toFixed(d)}–${v[1].toFixed(d)}]` : "");
export const titleCase = (s: string) => s.replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
