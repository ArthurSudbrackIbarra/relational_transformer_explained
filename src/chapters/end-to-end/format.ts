/** Number formatting for the "start to end" page (display only; real minus signs). */

/** Fixed digits with a real minus; `parenNeg` wraps negatives in parentheses for products. */
export function fmtNum(v: number, digits = 3, parenNeg = false): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'NaN' : v > 0 ? '+∞' : '−∞';
  const s = Math.abs(v).toFixed(digits);
  if (v < 0 && Number(s) !== 0) return parenNeg ? `(−${s})` : `−${s}`;
  return s;
}

/** "+0.214" / "−0.031". */
export function fmtSigned(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return fmtNum(v, digits);
  const s = Math.abs(v).toFixed(digits);
  return (v < 0 && Number(s) !== 0 ? '−' : '+') + s;
}

/** Integer with thousands separators. */
export function int(n: number): string {
  return n.toLocaleString('en-US');
}

/** 0.4506 → "45.1%". */
export function pct(p: number, digits = 1): string {
  return `${(p * 100).toFixed(digits)}%`;
}

/** "1 row" / "3 rows". */
export function plural(n: number, word: string, pluralWord?: string): string {
  return `${int(n)} ${n === 1 ? word : (pluralWord ?? `${word}s`)}`;
}

/** "customer" from "customers". */
export function singular(table: string): string {
  if (table.endsWith('ies')) return `${table.slice(0, -3)}y`;
  if (table.endsWith('s')) return table.slice(0, -1);
  return table;
}

const SUBS = '₀₁₂₃₄₅₆₇₈₉';
/** 12 → "₁₂" (for plain-text names like Q₁). */
export function sub(n: number): string {
  return String(n).replace(/\d/g, (d) => SUBS[Number(d)]);
}
