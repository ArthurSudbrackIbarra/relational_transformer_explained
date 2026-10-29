/**
 * Pure helpers for chapter 4 ("From rows to cells"). They only READ the engine's
 * pre.rs mirror (src/engine/preprocess.ts) and the raw database; nothing here
 * adds preprocessing semantics. Where a number is re-derived for display (fate
 * counts, histograms), the stats shown always come from `pdb.colStats` /
 * `pdb.datetimeStats`, i.e. from the engine itself.
 */
import type { Cell, ColumnSchema, ColumnStats, PreprocessedDB, RawDatabase, RawTable, RawValue, TableType } from '@/engine/types';

// ─────────────────────────────────────────────────────────────────────────────
// Table files (one parquet file of pre.rs each)
// ─────────────────────────────────────────────────────────────────────────────

export interface TableFile {
  /** table_info.json key, e.g. "orders:Db", "customer-churn:Val". */
  key: string;
  tableName: string;
  tableType: TableType;
  /** Path relative to the dataset dir, as pre.rs globs it (pre.rs:79-92). */
  path: string;
  offset: number;
  numNodes: number;
  table: RawTable;
  isTask: boolean;
}

/** The RawTable behind a table_info key. */
export function rawTableOf(db: RawDatabase, tableName: string, tableType: TableType): RawTable | null {
  if (tableType === 'Db') return db.tables.find((t) => t.schema.name === tableName) ?? null;
  const task = db.tasks.find((t) => t.kind === 'forecast' && t.name === tableName);
  return task?.tables?.[tableType] ?? null;
}

/** Every table file of the DB in node (= glob) order, from the engine's `tableInfo`. */
export function tableFiles(db: RawDatabase, pdb: PreprocessedDB): TableFile[] {
  const out: TableFile[] = [];
  for (const [key, info] of Object.entries(pdb.tableInfo)) {
    const i = key.lastIndexOf(':');
    const tableName = key.slice(0, i);
    const tableType = key.slice(i + 1) as TableType;
    const table = rawTableOf(db, tableName, tableType);
    if (!table) continue;
    const isTask = tableType !== 'Db';
    out.push({
      key,
      tableName,
      tableType,
      path: isTask ? `tasks/${tableName}/${tableType.toLowerCase()}.parquet` : `db/${tableName}.parquet`,
      offset: info.nodeIdxOffset,
      numNodes: info.numNodes,
      table,
      isTask,
    });
  }
  return out.sort((a, b) => a.offset - b.offset);
}

/** Short display label: "orders" or "customer-churn · Val". */
export function fileLabel(f: Pick<TableFile, 'tableName' | 'tableType'>): string {
  return f.tableType === 'Db' ? f.tableName : `${f.tableName} · ${f.tableType}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Column roles and the fate of every raw value (pre.rs:439-611)
// ─────────────────────────────────────────────────────────────────────────────

export type ColRole = 'pk' | 'fk' | 'feature';

export function colRole(table: RawTable, col: string): ColRole {
  if (col === table.schema.pkey) return 'pk';
  if (Object.prototype.hasOwnProperty.call(table.schema.fkeys, col)) return 'fk';
  return 'feature';
}

export function isMissing(v: RawValue | undefined): boolean {
  return v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v));
}

export type Fate =
  | { kind: 'pk'; col: ColumnSchema; value: RawValue }
  | { kind: 'link'; col: ColumnSchema; value: number; parentTable: string; parentNode: number }
  | { kind: 'null-fk'; col: ColumnSchema; parentTable: string }
  | { kind: 'missing'; col: ColumnSchema; value: RawValue }
  | { kind: 'cell'; col: ColumnSchema; cell: Cell; value: RawValue };

/** What pre.rs makes of each column of one row, in DataFrame column order. */
export function rowFates(pdb: PreprocessedDB, f: TableFile, row: number): Fate[] {
  const rec = f.table.rows[row] ?? {};
  const node = pdb.nodes[f.offset + row];
  return f.table.schema.columns.map((col): Fate => {
    const v = rec[col.name] ?? null;
    const role = colRole(f.table, col.name);
    if (role === 'pk') return { kind: 'pk', col, value: v };
    if (role === 'fk') {
      const parentTable = f.table.schema.fkeys[col.name];
      if (v === null) return { kind: 'null-fk', col, parentTable };
      const pOff = pdb.tableInfo[`${parentTable}:Db`]?.nodeIdxOffset ?? 0;
      return { kind: 'link', col, value: v as number, parentTable, parentNode: pOff + (v as number) };
    }
    if (isMissing(v)) return { kind: 'missing', col, value: v };
    const cell = node?.cells.find((c) => c.colName === col.name);
    if (!cell) return { kind: 'missing', col, value: v };
    return { kind: 'cell', col, cell, value: v };
  });
}

export interface FateCounts {
  raw: number;
  cell: number;
  link: number;
  pk: number;
  missing: number;
  /** cells per sem type code 0..3 */
  bySem: [number, number, number, number];
}

/** Counts over a whole table file (pk / link / missing from the raw rows, cells from the engine's nodes). */
export function fateCounts(pdb: PreprocessedDB, f: TableFile): FateCounts {
  const out: FateCounts = { raw: 0, cell: 0, link: 0, pk: 0, missing: 0, bySem: [0, 0, 0, 0] };
  const cols = f.table.schema.columns;
  for (let r = 0; r < f.table.rows.length; r++) {
    const rec = f.table.rows[r];
    for (const c of cols) {
      out.raw++;
      const role = colRole(f.table, c.name);
      const v = rec[c.name] ?? null;
      if (role === 'pk') out.pk++;
      else if (role === 'fk') {
        if (v === null) out.missing++;
        else out.link++;
      } else if (isMissing(v)) out.missing++;
    }
    const node = pdb.nodes[f.offset + r];
    for (const cell of node?.cells ?? []) {
      out.cell++;
      out.bySem[cell.sem]++;
    }
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Node timestamps (pre.rs:481-493, spec §1.4)
// ─────────────────────────────────────────────────────────────────────────────

/** Why a node has (or lacks) a timestamp, in words. */
export function tsReason(f: TableFile, row: number, ts: number | null): string {
  const s = f.table.schema;
  if (ts !== null) return `whole seconds of ${s.timeCol}`;
  if (!s.timeCol) return `${f.tableName} has no time column`;
  const fks = Object.keys(s.fkeys);
  const rec = f.table.rows[row] ?? {};
  if (fks.length === 0) return `${f.tableName} has a time column but no foreign key, and pre.rs sets the timestamp while it reads a foreign key`;
  if (fks.every((k) => rec[k] === null || rec[k] === undefined)) return 'every foreign key of this row is null, and pre.rs sets the timestamp while it reads a foreign key';
  return `the ${s.timeCol} value of this row is missing`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Column statistics (pre.rs:316-396)
// ─────────────────────────────────────────────────────────────────────────────

export interface StatColumn {
  /** Unique id "<key>/<col>". */
  id: string;
  file: TableFile;
  col: ColumnSchema;
  /** Non-null (and non-NaN) values of the file whose rows FED the stats. */
  values: number[];
  /** Nulls / NaNs dropped. */
  dropped: number;
  /** The stats the engine actually uses (pdb.colStats). */
  stats: ColumnStats;
}

/**
 * Number (`kind = 'number'`) or boolean (`kind = 'boolean'`) feature columns whose stats are
 * computed from their own rows: every DB table and the Train split of each task (Val/Test copy Train,
 * pre.rs:383-396). Boolean values are returned as 0/1.
 */
export function statColumns(files: TableFile[], pdb: PreprocessedDB, kind: 'number' | 'boolean'): StatColumn[] {
  const out: StatColumn[] = [];
  for (const f of files) {
    if (f.tableType === 'Val' || f.tableType === 'Test') continue;
    for (const col of f.table.schema.columns) {
      if (colRole(f.table, col.name) !== 'feature') continue;
      const isNum = col.dtype === 'int' || col.dtype === 'float';
      if (kind === 'number' ? !isNum : col.dtype !== 'bool') continue;
      const stats = pdb.colStats[f.key]?.[col.name];
      if (!stats) continue;
      const values: number[] = [];
      let dropped = 0;
      for (const r of f.table.rows) {
        const v = r[col.name];
        if (isMissing(v)) dropped++;
        else values.push(kind === 'boolean' ? (v ? 1 : 0) : (v as number));
      }
      out.push({ id: `${f.key}/${col.name}`, file: f, col, values, dropped, stats });
    }
  }
  return out;
}

export interface DatetimeColumn {
  id: string;
  file: TableFile;
  col: ColumnSchema;
  /** epoch ms, non-null */
  values: number[];
}

/** Every datetime column of every file (all of them feed the ONE global pair, pre.rs:342-366). */
export function datetimeColumns(files: TableFile[]): DatetimeColumn[] {
  const out: DatetimeColumn[] = [];
  for (const f of files) {
    for (const col of f.table.schema.columns) {
      if (col.dtype !== 'datetime') continue;
      const values: number[] = [];
      for (const r of f.table.rows) {
        const v = r[col.name];
        if (!isMissing(v)) values.push(v as number);
      }
      out.push({ id: `${f.key}/${col.name}`, file: f, col, values });
    }
  }
  return out;
}

/**
 * Boolean stats for k trues among n non-null values, exactly as pre.rs:320-329 computes them:
 * μ = k/n (0 if n = 0), σ = sample std (ddof = 1) or 0 when undefined, NO zero guard.
 * Sample variance of 0/1 values = n·μ·(1−μ)/(n−1).
 */
export function booleanStats(n: number, k: number): { mean: number; std: number; zTrue: number; zFalse: number } {
  const mean = n > 0 ? k / n : 0;
  const std = n > 1 ? Math.sqrt((n * mean * (1 - mean)) / (n - 1)) : 0;
  return { mean, std, zTrue: (1 - mean) / std, zFalse: (0 - mean) / std };
}

/** Share of k trues among n as a chip label: "25%", or "25.5%" when it is not a whole percent at one decimal. */
export function sharePct(k: number, n: number): string {
  const p = (100 * k) / Math.max(1, n);
  return Math.abs(p - Math.round(p)) < 0.05 ? `${Math.round(p)}%` : `${p.toFixed(1)}%`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Misc
// ─────────────────────────────────────────────────────────────────────────────

/** Integer-valued data with few distinct values gets one bin per integer. */
export function isSmallInteger(values: number[]): boolean {
  if (values.length === 0) return false;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (!Number.isInteger(v)) return false;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return hi - lo <= 30;
}

/** Epoch ms → "2025-03-01" (UTC). */
export function isoDay(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

/** Epoch seconds → "2025-03-01 00:00:00 UTC". */
export function isoSeconds(s: number): string {
  return `${new Date(s * 1000).toISOString().slice(0, 19).replace('T', ' ')} UTC`;
}

export const DAY_MS = 86_400_000;
