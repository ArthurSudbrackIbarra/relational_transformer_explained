/**
 * Data helpers for chapter 3 ("tasks"). Everything here READS the live engine output
 * (RawDatabase, TaskDef task tables, TASK_SPECS, PreprocessedDB); nothing re-implements RT.
 *
 * The one recomputation is didactic: `windowEvents` re-derives a forecast label from the raw DB
 * rows of the window (t, t+Δ] (spec §2.2), so the figures can show WHICH rows produced it. Every
 * figure that uses it cross-checks the result against the engine's task table (`labelMatches`).
 */
import { DAY_MS, TASK_SPECS, isDbName, type ForecastSpec } from '@/engine/db';
import type { RawDatabase, RawValue, Split, TaskDef } from '@/engine/types';

export const SPLITS: Split[] = ['Train', 'Val', 'Test'];

/** The TASK_SPECS entry behind a forecast TaskDef (label sources, aggregation, activity rule). */
export function forecastSpecOf(task: TaskDef): ForecastSpec | undefined {
  if (task.kind !== 'forecast' || !isDbName(task.dbName)) return undefined;
  return TASK_SPECS[task.dbName].forecast.find((s) => s.name === task.name);
}

/** One raw DB row that counts as an event of an entity (e.g. one order of a customer). */
export interface EntityEvent {
  /** Event time, epoch ms. */
  t: number;
  /** Value accumulated by 'sum' (1 for count-style tasks). */
  v: number;
  table: string;
  /** Row position in `table`. */
  row: number;
}

function eventsFrom(db: RawDatabase, sources: ForecastSpec['label']['sources'], entity: number): EntityEvent[] {
  const out: EntityEvent[] = [];
  for (const src of sources) {
    const table = db.tables.find((t) => t.schema.name === src.table);
    if (!table) continue;
    table.rows.forEach((row, pos) => {
      if (src.where && !src.where(row)) return;
      const t = row[src.timeCol];
      if (t === null || t === undefined) return;
      const e = src.entityFk === null ? pos : row[src.entityFk];
      if (e !== entity) return;
      out.push({ t: t as number, v: src.value ? src.value(row) : 1, table: src.table, row: pos });
    });
  }
  return out.sort((a, b) => a.t - b.t || a.row - b.row);
}

/** The rows that define an entity's label (e.g. a customer's orders), sorted by time. */
export function labelEvents(db: RawDatabase, spec: ForecastSpec, entity: number): EntityEvent[] {
  return eventsFrom(db, spec.label.sources, entity);
}

/** First activity time of an entity (the generator's "active at t" rule: ≥ 1 activity row at or before t). */
export function firstActivity(db: RawDatabase, spec: ForecastSpec, entity: number): number {
  const ev = eventsFrom(db, spec.activity, entity);
  return ev.length ? ev[0].t : Infinity;
}

/** Events in the half-open window (t, t + Δ] (spec §2.2). */
export function windowEvents(events: readonly EntityEvent[], t: number, deltaMs: number): EntityEvent[] {
  return events.filter((e) => e.t > t && e.t <= t + deltaMs);
}

/** Aggregate a window exactly as the task spec says ('none' = churn-style, 'any', 'count', 'sum'). */
export function aggregateWindow(spec: ForecastSpec, inWindow: readonly EntityEvent[]): boolean | number {
  const n = inWindow.length;
  switch (spec.label.agg) {
    case 'none':
      return n === 0;
    case 'any':
      return n > 0;
    case 'count':
      return n;
    case 'sum':
      return inWindow.reduce((s, e) => s + e.v, 0);
  }
}

export const deltaMsOf = (task: TaskDef) => (task.timedeltaDays ?? 30) * DAY_MS;

/** Sorted distinct timestamps of each split of a forecast task table. */
export function splitTimestamps(task: TaskDef): Record<Split, number[]> {
  const out = { Train: [], Val: [], Test: [] } as Record<Split, number[]>;
  if (!task.tables) return out;
  for (const s of SPLITS) out[s] = [...new Set(task.tables[s].rows.map((r) => r.timestamp as number))].sort((a, b) => a - b);
  return out;
}

export interface StampInfo {
  t: number;
  split: Split;
  /** Index within its split. */
  k: number;
}

/** Every timestamp of the task, chronologically, with its split. */
export function allStamps(task: TaskDef): StampInfo[] {
  const st = splitTimestamps(task);
  return SPLITS.flatMap((s) => st[s].map((t, k) => ({ t, split: s, k }))).sort((a, b) => a.t - b.t);
}

/** One task-table row of an entity, as stored by the engine. */
export interface EntityTaskRow {
  split: Split;
  /** Row position within the split's task table. */
  rowPos: number;
  t: number;
  label: RawValue;
}

/** The entity's rows in every split of the task table (engine output), chronologically. */
export function entityTaskRows(task: TaskDef, entity: number): EntityTaskRow[] {
  if (!task.tables || !task.entityCol) return [];
  const out: EntityTaskRow[] = [];
  for (const s of SPLITS) {
    task.tables[s].rows.forEach((r, pos) => {
      if (r[task.entityCol!] === entity) out.push({ split: s, rowPos: pos, t: r.timestamp as number, label: r[task.targetCol] });
    });
  }
  return out.sort((a, b) => a.t - b.t);
}

/** Per-timestamp summary of the whole task table: rows and (binary) positives / (regression) mean. */
export interface StampSummary extends StampInfo {
  rows: number;
  positives: number;
  mean: number;
}

export function stampSummaries(task: TaskDef): StampSummary[] {
  return allStamps(task).map((s) => {
    const rows = task.tables![s.split].rows.filter((r) => r.timestamp === s.t);
    const vals = rows.map((r) => r[task.targetCol]);
    const positives = vals.filter((v) => v === true).length;
    const nums = vals.filter((v): v is number => typeof v === 'number');
    const mean = nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : NaN;
    return { ...s, rows: rows.length, positives, mean };
  });
}

/** Strict equality of a recomputed label and the engine's stored label. */
export function labelMatches(a: boolean | number, b: RawValue): boolean {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) < 1e-9;
  return a === b;
}

/** "2024-03-06" style date (UTC). */
export function isoDay(ms: number): string {
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())}`;
}

/** "2024-03-06 00:00" style date and time (UTC). */
export function isoMinute(ms: number): string {
  return new Date(ms).toISOString().slice(0, 16).replace('T', ' ');
}

/**
 * The entity's row to show as "the" example row of a forecast task: its Val row, else its Test
 * row (an entity that becomes active after the val cutoff), else null (no row in any split).
 * Never invents a row (the activity rule is monotone: no Val row ⇒ no Train row either).
 */
export function exampleTaskRow(task: TaskDef, entity: number): EntityTaskRow | null {
  const rows = entityTaskRows(task, entity);
  return rows.find((r) => r.split === 'Val') ?? rows.find((r) => r.split === 'Test') ?? rows[rows.length - 1] ?? null;
}

/** "Mar 6" style short date (UTC). */
export function shortDay(ms: number, withYear = false): string {
  const d = new Date(ms);
  const m = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][d.getUTCMonth()];
  return `${m} ${d.getUTCDate()}${withYear ? `, ${d.getUTCFullYear()}` : ''}`;
}

/** Human label of a boolean task label. */
export function boolText(v: RawValue): string {
  return v === true ? 'true' : v === false ? 'false' : String(v);
}

/** Display name of an entity row (customer "Anna Müller", product "Wireless Mouse", …). */
export function entityName(db: RawDatabase, table: string, row: number): string {
  const t = db.tables.find((x) => x.schema.name === table);
  const r = t?.rows[row];
  if (!r) return `${table}[${row}]`;
  const name = r.name ?? r.display_name ?? r.title;
  return typeof name === 'string' ? name : `${table}[${row}]`;
}
