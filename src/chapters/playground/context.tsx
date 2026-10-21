/**
 * One live playground state per page: computed once here and shared by every figure of the
 * chapter (so the forward passes are not repeated per figure), plus small naming helpers.
 */
import { createContext, useContext, useEffect, type ReactNode } from 'react';
import { formatDate } from '@/components';
import type { PreprocessedDB, RawDatabase } from '@/engine/types';
import { usePlay } from './state';
import { usePlayground, type PlayLive } from './usePlayground';

const Ctx = createContext<PlayLive | null>(null);

export function PlayProvider({ children }: { children: ReactNode }) {
  const live = usePlayground();
  const pushHistory = usePlay((s) => s.pushHistory);
  const { cur, sig, chips } = live;
  // the history is compared with ONE baseline: a new seed, task, L, w, RNG seed or checkpoint restarts it
  const histScope = `${live.scope}|${live.globalL}|${live.w}|${live.rngSeed}|${live.baseCkpt}`;
  // record a run once the switches have settled (dragging the L slider must not log every step)
  useEffect(() => {
    if (!cur) return;
    const t = window.setTimeout(() => pushHistory({ scope: histScope, sig, chips, value: cur.display }), 650);
    return () => window.clearTimeout(t);
  }, [cur, sig, histScope, chips, pushHistory]);
  return <Ctx.Provider value={live}>{children}</Ctx.Provider>;
}

export function usePlayLive(): PlayLive {
  const v = useContext(Ctx);
  if (!v) throw new Error('usePlayLive outside <PlayProvider>');
  return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Names (display only)
// ─────────────────────────────────────────────────────────────────────────────

/** Singular noun of a table's rows, e.g. customers → customer. */
export function rowNoun(table: string): string {
  if (table === 'customer-churn' || table === 'product-sales') return `${table} row`;
  return table.endsWith('s') ? table.slice(0, -1) : table;
}

/** "Anna Müller", "Wireless Mouse", "review 12", … for a DB row. */
export function entityName(db: RawDatabase, table: string, rowPos: number): string {
  const t = db.tables.find((x) => x.schema.name === table);
  const r = t?.rows[rowPos];
  if (r && typeof r.name === 'string') return r.name;
  return `${rowNoun(table)} ${rowPos}`;
}

/** Name of a node: DB rows by their entity name; task rows by their entity's name. */
export function nodeTitle(db: RawDatabase, pdb: PreprocessedDB, node: number): { title: string; sub: string } {
  const n = pdb.nodes[node];
  if (!n) return { title: `#${node}`, sub: '' };
  if (n.tableType === 'Db') return { title: entityName(db, n.tableName, n.rowPos), sub: `${rowNoun(n.tableName)} ${n.rowPos}` };
  const parent = n.f2pNbrIdxs[0];
  const p = parent !== undefined ? pdb.nodes[parent] : undefined;
  const date = n.ts !== null ? formatDate(n.ts * 1000) : '';
  if (!p) return { title: `${n.tableName} ${n.tableType}[${n.rowPos}]`, sub: date };
  return { title: entityName(db, p.tableName, p.rowPos), sub: `${rowNoun(p.tableName)} ${p.rowPos}${date ? ` · ${date}` : ''}` };
}

/** "1" / "0" for booleans, numbers compactly. */
export function labelText(v: unknown): string {
  if (v === true) return '1';
  if (v === false) return '0';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(2);
  if (v === null || v === undefined) return '∅';
  return String(v);
}
