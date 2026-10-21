/**
 * The pretraining item list of our leave-one-DB-out run, rebuilt live with the engine:
 * every task of SchoolDB, RacingDB and ForumDB, its qualifying seed rows (forecast: the Train split
 * of the task table; autocomplete: every row of the DB table that has the target cell,
 * data.py:77-81, fly.rs:287-296), concatenated in task order into ONE list with no balancing
 * (main.py:123-136, spec §4.14, §9.1). The task order is the exporter's (python/README.md), read
 * from dataset.json when present.
 *
 * Built once per page load and deferred to after the first paint (~0.2 s for the three DBs).
 */
import { useEffect, useState } from 'react';
import { getDatabase, tableColorSlot, type DbName } from '@/engine/db/registry';
import { getPreprocessed, getSeeds } from '@/engine/pipeline';
import type { PreprocessedDB, RawDatabase, TaskDef } from '@/engine/types';
import { DB_SLOT } from './data';

export const PRETRAIN_DBS: DbName[] = ['school', 'racing', 'forum'];

export interface PreTask {
  idx: number;
  id: string;
  dbName: DbName;
  short: string;
  task: TaskDef;
  db: RawDatabase;
  pdb: PreprocessedDB;
  seeds: number[];
  slot: number;
}

export interface PretrainSet {
  tasks: PreTask[];
  /** Item k = (taskOf[k], seedOf[k]) in list order (task by task, node order). */
  taskOf: Int32Array;
  seedOf: Int32Array;
  N: number;
  /** Per DB: tableName → categorical slot, for TokenStrip. */
  tableSlots: Record<string, Record<string, number>>;
  dbs: { name: DbName; title: string; tables: number; rows: number; tasks: PreTask[] }[];
}

/** Exporter order: forecast clf, forecast reg, autocomplete clf, autocomplete reg; DBs in DB_NAMES order. */
function groupOf(t: TaskDef): number {
  return (t.kind === 'forecast' ? 0 : 2) + (t.targetType === 'binary' ? 0 : 1);
}

let cached: PretrainSet | null = null;

export function buildPretrainSet(order?: string[]): PretrainSet {
  if (cached) return cached;
  const all: { task: TaskDef; db: RawDatabase; pdb: PreprocessedDB; dbName: DbName }[] = [];
  const tableSlots: Record<string, Record<string, number>> = {};
  const dbs: PretrainSet['dbs'] = [];
  for (const dbName of PRETRAIN_DBS) {
    const db = getDatabase(dbName);
    const pdb = getPreprocessed(db);
    const slots: Record<string, number> = {};
    for (const t of db.tables) slots[t.schema.name] = tableColorSlot(db.name, t.schema.name);
    for (const t of db.tasks) if (t.kind === 'forecast') slots[t.name] = tableColorSlot(db.name, t.name);
    tableSlots[dbName] = slots;
    for (const task of db.tasks) all.push({ task, db, pdb, dbName });
    dbs.push({ name: dbName, title: db.title, tables: db.tables.length, rows: db.tables.reduce((s, t) => s + t.rows.length, 0), tasks: [] });
  }
  const rank = (t: TaskDef) => {
    const i = order?.indexOf(t.id) ?? -1;
    return i >= 0 ? i : 100 + groupOf(t) * 10 + PRETRAIN_DBS.indexOf(t.dbName as DbName);
  };
  all.sort((a, b) => rank(a.task) - rank(b.task));
  const tasks: PreTask[] = all.map((a, idx) => ({
    idx,
    id: a.task.id,
    dbName: a.dbName,
    short: a.task.id.split('/')[1],
    task: a.task,
    db: a.db,
    pdb: a.pdb,
    seeds: getSeeds(a.pdb, a.task, 'Train'),
    slot: DB_SLOT[a.dbName],
  }));
  for (const t of tasks) dbs.find((d) => d.name === t.dbName)!.tasks.push(t);
  const N = tasks.reduce((s, t) => s + t.seeds.length, 0);
  const taskOf = new Int32Array(N);
  const seedOf = new Int32Array(N);
  let k = 0;
  for (const t of tasks)
    for (const s of t.seeds) {
      taskOf[k] = t.idx;
      seedOf[k] = s;
      k++;
    }
  cached = { tasks, taskOf, seedOf, N, tableSlots, dbs };
  return cached;
}

/** The pretraining set, built after the first paint. null until ready. */
export function usePretrainSet(order?: string[]): PretrainSet | null {
  const [set, setSet] = useState<PretrainSet | null>(cached);
  const key = order?.join('|') ?? '';
  useEffect(() => {
    if (cached) {
      setSet(cached);
      return;
    }
    const id = window.setTimeout(() => setSet(buildPretrainSet(order)), 30);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return set;
}
