/**
 * Small, pure helpers for the attention chapter. Nothing here adds model semantics: every number the
 * chapter shows comes from the engine's forward-pass capture (`ForwardResult.blocks[b].attn[l]`,
 * a mirror of rt/model.py:20-47 and :82-86). The helpers only slice, group and label it.
 */
import { tokenValueText } from '@/components';
import type { Token } from '@/engine/types';

// ─────────────────────────────────────────────────────────────────────────────
// Names and labels (display only)
// ─────────────────────────────────────────────────────────────────────────────

/** Shortest unique prefix per table name, e.g. ShopDB → c, o, p, r. */
export function tablePrefixes(names: readonly string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const n of names) {
    let len = 1;
    while (len < n.length && names.some((m) => m !== n && m.startsWith(n.slice(0, len)))) len++;
    out[n] = n.slice(0, len);
  }
  return out;
}

/** Full row name: "customers[35]" or "customer-churn:Val[32]". */
export function rowName(t: Token): string {
  if (t.isPadding) return 'padding';
  if (t.tableType && t.tableType !== 'Db') return `${t.tableName}:${t.tableType}[${t.rowPos}]`;
  return `${t.tableName}[${t.rowPos}]`;
}

/** Short row tag: "c35", "o70", "val32", "train258". */
export function rowTag(t: Token, prefixes: Record<string, string>): string {
  if (t.isPadding) return 'pad';
  if (t.tableType && t.tableType !== 'Db') return `${t.tableType.toLowerCase()}${t.rowPos}`;
  return `${prefixes[t.tableName] ?? t.tableName[0]}${t.rowPos}`;
}

/** Short token label for axes and lists: "c35·age". */
export function tokenLabel(t: Token, prefixes: Record<string, string>): string {
  if (t.isPadding) return `pad #${t.pos}`;
  return `${rowTag(t, prefixes)}·${t.colName}`;
}

/** The value a token shows (raw value, "?" for the target). Display only. */
export function valueOf(t: Token): string {
  return tokenValueText(t);
}

/** "customer" from "customers" (preset labels only). */
export function singular(table: string): string {
  if (table.endsWith('ies')) return `${table.slice(0, -3)}y`;
  if (table.endsWith('s')) return table.slice(0, -1);
  return table;
}

export function article(word: string): string {
  return /^[aeiou]/i.test(word) ? 'an' : 'a';
}

// ─────────────────────────────────────────────────────────────────────────────
// Keys grouped by table (for the score / weight bar charts)
// ─────────────────────────────────────────────────────────────────────────────

export interface KeyGroup {
  /** Table name (task splits share one table, like table_name_idx). */
  name: string;
  slot: number;
  isTask: boolean;
  /** Token positions in context order. */
  positions: number[];
}

/** Group the non-padding tokens by table, in order of first appearance; padding listed separately. */
export function groupKeys(tokens: readonly Token[], slots: Record<string, number>): { groups: KeyGroup[]; order: number[]; padding: number[] } {
  const groups: KeyGroup[] = [];
  const byName = new Map<string, KeyGroup>();
  const padding: number[] = [];
  for (const t of tokens) {
    if (t.isPadding) {
      padding.push(t.pos);
      continue;
    }
    let g = byName.get(t.tableName);
    if (!g) {
      g = { name: t.tableName, slot: slots[t.tableName] ?? 0, isTask: !!t.tableType && t.tableType !== 'Db', positions: [] };
      byName.set(t.tableName, g);
      groups.push(g);
    }
    g.positions.push(t.pos);
  }
  return { groups, order: groups.flatMap((g) => g.positions), padding };
}

// ─────────────────────────────────────────────────────────────────────────────
// Vector helpers (Float32Array views)
// ─────────────────────────────────────────────────────────────────────────────

/** Row r of a row-major [rows × n] tensor (zero-copy view). */
export function rowOf(a: Float32Array, r: number, n: number): Float32Array {
  return a.subarray(r * n, (r + 1) * n);
}

export function norm2(v: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

export function maxAbsOf(...vs: ArrayLike<number>[]): number {
  let m = 0;
  for (const v of vs) for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i]));
  return m;
}

/** Positions with the largest values of `w` among `positions`, descending. */
export function topK(w: ArrayLike<number>, positions: readonly number[], k: number): number[] {
  return [...positions].sort((a, b) => w[b] - w[a] || a - b).slice(0, k);
}

/** Integer with thousands separators. */
export const int = (n: number) => n.toLocaleString('en-US');

/** Signed number with a real minus sign. */
export function sgn(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return v > 0 ? '+∞' : v < 0 ? '−∞' : 'NaN';
  const s = Math.abs(v).toFixed(digits);
  return (v < 0 ? '−' : '') + s;
}

/** TeX-safe signed number (uses "-" so KaTeX renders a proper minus). */
export function texNum(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return v > 0 ? '+\\infty' : '-\\infty';
  return v.toFixed(digits);
}

/** Percentage with adaptive precision (small weights keep two significant digits). */
export function pctw(p: number): string {
  if (!Number.isFinite(p)) return '–';
  if (p === 0) return '0%';
  const x = p * 100;
  if (x >= 10) return `${x.toFixed(1)}%`;
  if (x >= 1) return `${x.toFixed(1)}%`;
  if (x >= 0.1) return `${x.toFixed(2)}%`;
  return `${x.toPrecision(2)}%`;
}

/** Fixed decimals without trailing zeros: trimNum(0.25, 3) = "0.25", trimNum(0.1768, 3) = "0.177". */
export function trimNum(v: number, digits = 3): string {
  return v.toFixed(digits).replace(/\.?0+$/, '');
}
