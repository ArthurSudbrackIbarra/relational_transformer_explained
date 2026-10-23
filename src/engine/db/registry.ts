/**
 * Database registry (BUILD_PLAN §2): deterministic, cached access to the four dummy databases,
 * their tasks, and the table color slots used by every view (BUILD_PLAN §3.5, §5).
 */
import type { RawDatabase, RawTable, RawValue, Split, TaskDef } from '../types';
import { generateForum } from './generators/forum';
import { generateRacing } from './generators/racing';
import { generateSchool } from './generators/school';
import { generateShop } from './generators/shop';
import type { GeneratedDb } from './generators/types';
import { DB_NAMES, TASK_SPECS, buildTasks, type DbName } from './tasks';

export { DB_NAMES, type DbName };

export type DbRole = 'target' | 'pretrain';

export interface DbInfo {
  name: DbName;
  title: string;
  role: DbRole;
  description: string;
}

/** Static metadata (available without generating the data). ShopDB is the held-out target. */
const DB_META: Record<DbName, Omit<DbInfo, 'name'>> = {
  shop: {
    title: 'ShopDB (e-commerce)',
    role: 'target',
    description:
      'A small online shop: customers, products, orders and reviews from January 2024 to June 2025. ' +
      'It is the target database: the model never sees it during pretraining.',
  },
  school: {
    title: 'SchoolDB (university)',
    role: 'pretrain',
    description:
      'A university: students, courses and assignment submissions over three terms (January 2024 to June 2025). ' +
      'Used for pretraining.',
  },
  racing: {
    title: 'RacingDB (motor racing)',
    role: 'pretrain',
    description:
      'A fictional motor-racing championship: drivers, teams, a race every two weeks and per-race results ' +
      '(January 2024 to June 2025). Used for pretraining.',
  },
  forum: {
    title: 'ForumDB (Q&A forum)',
    role: 'pretrain',
    description:
      'A programming question-and-answer forum: users, questions and votes (January 2024 to June 2025). ' +
      'Used for pretraining.',
  },
};

const GENERATORS: Record<DbName, () => GeneratedDb> = {
  shop: () => generateShop(),
  school: () => generateSchool(),
  racing: () => generateRacing(),
  forum: () => generateForum(),
};

export function isDbName(name: string): name is DbName {
  return (DB_NAMES as readonly string[]).includes(name);
}

/**
 * Global time range of a DB, like RelBench `Database.min_timestamp / max_timestamp`: the min/max
 * over the time columns of the DB tables that declare one (task tables excluded).
 */
function timeRange(tables: RawTable[]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const t of tables) {
    const tc = t.schema.timeCol;
    if (tc === null) continue;
    for (const row of t.rows) {
      const v = row[tc];
      if (v === null) continue;
      if ((v as number) < min) min = v as number;
      if ((v as number) > max) max = v as number;
    }
  }
  return { min, max };
}

/** Generate a database from scratch (uncached). Same output on every call. */
export function buildDatabase(name: DbName): RawDatabase {
  const gen = GENERATORS[name]();
  const { min, max } = timeRange(gen.tables);
  return {
    name,
    title: DB_META[name].title,
    description: DB_META[name].description,
    tables: gen.tables,
    tasks: buildTasks(name, gen.tables, min, max),
    minTimestamp: min,
    maxTimestamp: max,
  };
}

const cache = new Map<DbName, RawDatabase>();

/**
 * The database `name`, generated once and cached. Deterministic (same data in browser and Node).
 * The returned object is shared: treat it as immutable and clone before editing.
 */
export function getDatabase(name: DbName): RawDatabase {
  let db = cache.get(name);
  if (!db) {
    if (!isDbName(name)) throw new Error(`unknown database "${String(name)}"`);
    db = buildDatabase(name);
    cache.set(name, db);
  }
  return db;
}

/** The four databases in display order: shop (target) first, then the pretraining DBs. */
export function listDatabases(): DbInfo[] {
  return DB_NAMES.map((name) => ({ name, ...DB_META[name] }));
}

/** Look up a task by id ("shop/customer-churn", "shop/reviews.rating", …). */
export function getTask(taskId: string): TaskDef {
  const dbName = taskId.split('/')[0];
  if (!isDbName(dbName)) throw new Error(`unknown task "${taskId}"`);
  const task = getDatabase(dbName).tasks.find((t) => t.id === taskId);
  if (!task) throw new Error(`unknown task "${taskId}"`);
  return task;
}

// ─────────────────────────────────────────────────────────────────────────────
// Table color slots (BUILD_PLAN §3.5)
// ─────────────────────────────────────────────────────────────────────────────

/** Every task table, in every DB and split, uses categorical slot 5 (magenta). */
export const TASK_TABLE_SLOT = 5;

/** DB tables get slots in schema order; ShopDB's reviews take violet (7) per the plan. */
const DB_TABLE_SLOTS: Record<DbName, Record<string, number>> = {
  shop: { customers: 1, products: 2, orders: 3, reviews: 7 },
  school: { students: 1, courses: 2, submissions: 3 },
  racing: { drivers: 1, teams: 2, races: 3, results: 4 },
  forum: { users: 1, posts: 2, votes: 3 },
};

/**
 * Categorical color slot (1..8) of a table. Accepts a plain table name ("orders",
 * "customer-churn") or a table_info key ("customer-churn:Val", "orders:Db").
 * Throws for a table that does not belong to the DB.
 */
export function tableColorSlot(dbName: string, tableName: string): number {
  if (!isDbName(dbName)) throw new Error(`unknown database "${dbName}"`);
  const name = tableName.replace(/:(Db|Train|Val|Test)$/, '');
  const slot = DB_TABLE_SLOTS[dbName][name];
  if (slot !== undefined) return slot;
  if (TASK_SPECS[dbName].forecast.some((s) => s.name === name)) return TASK_TABLE_SLOT;
  throw new Error(`table "${tableName}" is not part of database "${dbName}"`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Showcase entity for the running example
// ─────────────────────────────────────────────────────────────────────────────

export interface ShowcaseEntity {
  taskId: string;
  entityTable: string;
  /** Row position of the entity in its DB table (= its PK value). */
  entityRow: number;
  /** Row positions of this entity's rows in each split of the task table (ascending time). */
  rows: Record<Split, number[]>;
  /** Labels of those rows, aligned with `rows`. */
  labels: Record<Split, RawValue[]>;
}

/**
 * Pick the running-example entity of a binary forecast task (default: ShopDB `customer-churn`):
 * an entity with a long, mixed label history, i.e. several Train rows with both label values
 * and present in both Val and Test. Ties break on the lowest id. Deterministic.
 * Use `rows.Val[0]` (+ the Val split's node offset) as the default seed.
 */
export function pickShowcaseEntity(db: RawDatabase, taskId?: string): ShowcaseEntity {
  const task = taskId
    ? db.tasks.find((t) => t.id === taskId)
    : db.tasks.find((t) => t.kind === 'forecast' && t.targetType === 'binary');
  if (!task || task.kind !== 'forecast' || !task.tables || !task.entityCol || !task.entityTable) {
    throw new Error(`pickShowcaseEntity: no binary forecast task ${taskId ?? ''} in ${db.name}`);
  }
  const entityCol = task.entityCol;
  const entityTable = db.tables.find((t) => t.schema.name === task.entityTable)!;
  const perEntity = (split: Split) => {
    const m = new Map<number, number[]>();
    task.tables![split].rows.forEach((r, pos) => {
      const e = r[entityCol] as number;
      if (!m.has(e)) m.set(e, []);
      m.get(e)!.push(pos);
    });
    return m;
  };
  const byTrain = perEntity('Train');
  const byVal = perEntity('Val');
  const byTest = perEntity('Test');

  // how many child rows reference the entity (a richer neighborhood makes a better example)
  const childCount = new Array<number>(entityTable.rows.length).fill(0);
  for (const t of db.tables) {
    for (const [fk, parent] of Object.entries(t.schema.fkeys)) {
      if (parent !== task.entityTable) continue;
      for (const r of t.rows) if (r[fk] !== null) childCount[r[fk] as number]++;
    }
  }
  const nonNullCells = (e: number) => Object.values(entityTable.rows[e]).filter((v) => v !== null).length;

  let best: { e: number; score: number } | null = null;
  for (let e = 0; e < entityTable.rows.length; e++) {
    const train = byTrain.get(e) ?? [];
    if (!byVal.has(e) || !byTest.has(e) || train.length < 6) continue;
    const labels = train.map((p) => task.tables!.Train.rows[p][task.targetCol]);
    const nTrue = labels.filter((v) => v === true).length;
    const nFalse = labels.length - nTrue;
    if (nTrue === 0 || nFalse === 0) continue;
    let switches = 0;
    for (let i = 1; i < labels.length; i++) if (labels[i] !== labels[i - 1]) switches++;
    const score =
      3 * Math.min(nTrue, nFalse) +
      2 * switches +
      train.length +
      Math.min(childCount[e], 40) / 10 +
      (nonNullCells(e) === entityTable.schema.columns.length ? 2 : 0);
    if (!best || score > best.score) best = { e, score };
  }
  if (!best) throw new Error(`pickShowcaseEntity: no suitable entity in ${task.id}`);

  const e = best.e;
  const rows: Record<Split, number[]> = { Train: byTrain.get(e)!, Val: byVal.get(e)!, Test: byTest.get(e)! };
  const lab = (s: Split) => rows[s].map((p) => task.tables![s].rows[p][task.targetCol]);
  return {
    taskId: task.id,
    entityTable: task.entityTable,
    entityRow: e,
    rows,
    labels: { Train: lab('Train'), Val: lab('Val'), Test: lab('Test') },
  };
}
