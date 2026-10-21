/**
 * Pure helpers for chapter 2 ("The relational database"). Everything here is
 * derived from the live RawDatabase / PreprocessedDB; nothing is hard-coded.
 *
 * Links follow spec §1.2-1.3: every non-null FK value of a child row points at
 * the parent row whose position equals that value (pre.rs:452-459, 477).
 * Null FKs produce no link (pre.rs:462-464).
 */
import type { PreprocessedDB, RawDatabase, RawTable, RawValue, TaskDef } from '@/engine/types';
import { formatDate, formatRaw, truncate } from '@/components/viz/format';

// ─────────────────────────────────────────────────────────────────────────────
// Row references
// ─────────────────────────────────────────────────────────────────────────────

export interface RowRef {
  table: string;
  row: number;
}

export const rowKey = (r: RowRef) => `${r.table}:${r.row}`;
export const sameRow = (a: RowRef | null | undefined, b: RowRef | null | undefined) =>
  !!a && !!b && a.table === b.table && a.row === b.row;
export const rowLabel = (r: RowRef) => `${r.table}[${r.row}]`;

export function parseRowKey(k: string): RowRef {
  const i = k.lastIndexOf(':');
  return { table: k.slice(0, i), row: Number(k.slice(i + 1)) };
}

// ─────────────────────────────────────────────────────────────────────────────
// Link index (F→P parents and P→F children of every DB row)
// ─────────────────────────────────────────────────────────────────────────────

export interface ParentLink {
  fkCol: string;
  parentTable: string;
  /** null when the FK cell is null (no link, pre.rs:462-464). */
  parentRow: number | null;
}

export interface ChildGroup {
  childTable: string;
  fkCol: string;
  /** Child row positions, ascending (ShopDB numbers orders/reviews chronologically). */
  rows: number[];
}

export interface LinkIndex {
  db: RawDatabase;
  table(name: string): RawTable | undefined;
  parentsOf(ref: RowRef): ParentLink[];
  childrenOf(ref: RowRef): ChildGroup[];
  /** Total number of child rows (all groups). */
  childCount(ref: RowRef): number;
  /** Tables that reference `table` through at least one FK column. */
  childTablesOf(table: string): { childTable: string; fkCol: string }[];
}

const indexCache = new WeakMap<RawDatabase, LinkIndex>();

export function buildLinkIndex(db: RawDatabase): LinkIndex {
  const hit = indexCache.get(db);
  if (hit) return hit;
  const tables = new Map(db.tables.map((t) => [t.schema.name, t]));
  // parentTable -> list of { childTable, fkCol, byParent: rows[] per parent row }
  const children = new Map<string, { childTable: string; fkCol: string; byParent: number[][] }[]>();
  for (const t of db.tables) {
    for (const [fkCol, parent] of Object.entries(t.schema.fkeys)) {
      const p = tables.get(parent);
      if (!p) continue;
      const byParent: number[][] = Array.from({ length: p.rows.length }, () => []);
      t.rows.forEach((r, i) => {
        const v = r[fkCol];
        if (typeof v === 'number' && Number.isInteger(v) && v >= 0 && v < byParent.length) byParent[v].push(i);
      });
      if (!children.has(parent)) children.set(parent, []);
      children.get(parent)!.push({ childTable: t.schema.name, fkCol, byParent });
    }
  }
  const idx: LinkIndex = {
    db,
    table: (name) => tables.get(name),
    parentsOf: (ref) => {
      const t = tables.get(ref.table);
      if (!t) return [];
      const rec = t.rows[ref.row];
      return Object.entries(t.schema.fkeys).map(([fkCol, parentTable]) => {
        const v = rec?.[fkCol];
        return { fkCol, parentTable, parentRow: typeof v === 'number' && Number.isFinite(v) ? v : null };
      });
    },
    childrenOf: (ref) =>
      (children.get(ref.table) ?? []).map((g) => ({ childTable: g.childTable, fkCol: g.fkCol, rows: g.byParent[ref.row] ?? [] })),
    childCount: (ref) => (children.get(ref.table) ?? []).reduce((s, g) => s + (g.byParent[ref.row]?.length ?? 0), 0),
    childTablesOf: (table) => (children.get(table) ?? []).map((g) => ({ childTable: g.childTable, fkCol: g.fkCol })),
  };
  indexCache.set(db, idx);
  return idx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Row descriptions (display only)
// ─────────────────────────────────────────────────────────────────────────────

/** A short human caption for a row: a name-like text column, else its time column, else the first feature value. */
export function rowCaption(db: RawDatabase, ref: RowRef, max = 28): string {
  const t = db.tables.find((x) => x.schema.name === ref.table);
  const rec = t?.rows[ref.row];
  if (!t || !rec) return '';
  const s = t.schema;
  const named = s.columns.find((c) => c.dtype === 'str' && /name|title/.test(c.name) && rec[c.name] != null);
  if (named) return truncate(String(rec[named.name]), max);
  if (s.timeCol && rec[s.timeCol] != null) return formatDate(rec[s.timeCol] as number, 'never');
  const feat = s.columns.find((c) => c.name !== s.pkey && !(c.name in s.fkeys) && rec[c.name] != null);
  return feat ? truncate(formatRaw(rec[feat.name], feat.dtype), max) : '';
}

/** Raw time-column value (epoch ms) of a row, or null. */
export function rowTime(db: RawDatabase, ref: RowRef): number | null {
  const t = db.tables.find((x) => x.schema.name === ref.table);
  const tc = t?.schema.timeCol;
  if (!t || !tc) return null;
  const v = t.rows[ref.row]?.[tc];
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

export const isMissing = (v: RawValue | undefined) => v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v));

export type ColumnKind = 'pk' | 'fk' | 'feature';

export function columnKind(t: RawTable, col: string): ColumnKind {
  return col === t.schema.pkey ? 'pk' : col in t.schema.fkeys ? 'fk' : 'feature';
}

/** Missing-value counts per column of a table. */
export function nullCounts(t: RawTable): Record<string, number> {
  const out: Record<string, number> = {};
  for (const c of t.schema.columns) out[c.name] = t.rows.reduce((n, r) => n + (isMissing(r[c.name]) ? 1 : 0), 0);
  return out;
}

/** Cells of a row split by what RT does with them (spec §1.2, §3.5). */
export function classifyRow(t: RawTable, row: number) {
  const rec = t.rows[row] ?? {};
  const tokens: string[] = [];
  const links: string[] = [];
  const pk: string[] = [];
  const missing: string[] = [];
  for (const c of t.schema.columns) {
    const k = columnKind(t, c.name);
    const miss = isMissing(rec[c.name]);
    if (k === 'pk') pk.push(c.name);
    else if (k === 'fk') (miss ? missing : links).push(c.name);
    else (miss ? missing : tokens).push(c.name);
  }
  return { tokens, links, pk, missing };
}

// ─────────────────────────────────────────────────────────────────────────────
// Running example
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The row this chapter follows by default: for a forecast seed (a task row), the entity it
 * points to (its single FK, spec §2.1); for an autocomplete seed (a DB row, spec §2.4), the
 * seed row itself.
 */
export function defaultFocus(pdb: PreprocessedDB, db: RawDatabase, seedNode: number): RowRef {
  const seed = seedNode >= 0 ? pdb.nodes[seedNode] : undefined;
  if (seed) {
    if (seed.isTaskNode && seed.f2pNbrIdxs.length > 0) {
      const e = pdb.nodes[seed.f2pNbrIdxs[0]];
      if (e) return { table: e.tableName, row: e.rowPos };
    }
    if (seed.tableType === 'Db') return { table: seed.tableName, row: seed.rowPos };
  }
  return { table: db.tables[0].schema.name, row: 0 };
}

/**
 * The row whose time-stamped children the timeline shows: the focus row if it has children in
 * a table with a time column, else its first parent that has some.
 */
export function timelineAnchor(idx: LinkIndex, focus: RowRef): RowRef | null {
  const hasTimedChildren = (r: RowRef) =>
    idx.childrenOf(r).some((g) => g.rows.length > 0 && !!idx.table(g.childTable)?.schema.timeCol);
  if (hasTimedChildren(focus)) return focus;
  for (const p of idx.parentsOf(focus)) {
    if (p.parentRow === null) continue;
    const r = { table: p.parentTable, row: p.parentRow };
    if (hasTimedChildren(r)) return r;
  }
  return null;
}

/** Seconds → epoch ms (node.ts is whole seconds, spec §1.4). */
export const tsToMs = (ts: number | null) => (ts === null ? null : ts * 1000);

/** The seed's timestamp in ms (null if the seed row has none). */
export function seedTimeMs(pdb: PreprocessedDB, seedNode: number): number | null {
  const s = seedNode >= 0 ? pdb.nodes[seedNode] : undefined;
  return s ? tsToMs(s.ts) : null;
}

/** Rows of each DB table that carry a sampler timestamp (pre.rs:481-493). */
export function timestampCoverage(pdb: PreprocessedDB, db: RawDatabase) {
  return db.tables.map((t) => {
    const info = pdb.tableInfo[`${t.schema.name}:Db`];
    let withTs = 0;
    if (info) for (let i = 0; i < info.numNodes; i++) if (pdb.nodes[info.nodeIdxOffset + i]?.ts !== null) withTs++;
    return {
      table: t.schema.name,
      timeCol: t.schema.timeCol,
      nFk: Object.keys(t.schema.fkeys).length,
      rows: t.rows.length,
      withTs,
    };
  });
}

/** For a forecast task whose entity table is `focus.table`: the seed node of that entity in `seeds`, if any. */
export function seedForEntity(pdb: PreprocessedDB, task: TaskDef, seeds: number[], focus: RowRef): number | null {
  if (task.kind !== 'forecast' || task.entityTable !== focus.table) return null;
  const info = pdb.tableInfo[`${focus.table}:Db`];
  if (!info) return null;
  const node = info.nodeIdxOffset + focus.row;
  for (const s of seeds) if (pdb.nodes[s]?.f2pNbrIdxs[0] === node) return s;
  return null;
}

/**
 * Up to `cap` children for a small drawing, taken round-robin across the child groups so every
 * child table appears (display only; RT's own width bound is a different, random rule, spec §4.7).
 */
export function childrenForDisplay(groups: ChildGroup[], cap: number): RowRef[] {
  const out: RowRef[] = [];
  for (let i = 0; out.length < cap; i++) {
    let any = false;
    for (const g of groups) {
      if (i < g.rows.length) {
        any = true;
        if (out.length < cap) out.push({ table: g.childTable, row: g.rows[i] });
      }
    }
    if (!any) break;
  }
  return out;
}
