/**
 * Data invariants of the dummy databases (spec §1, §2, §3.3, §4.16, §12; BUILD_PLAN §3).
 */
import { describe, expect, it } from 'vitest';
import type { RawDatabase, RawTable, RawValue, Split, TaskDef } from '../types';
import {
  DATA_END,
  DATA_START,
  DAY_MS,
  DB_NAMES,
  VAL_TIMESTAMP,
  TASK_TABLE_SLOT,
  buildDatabase,
  columnPhrase,
  getDatabase,
  getTask,
  listDatabases,
  pickShowcaseEntity,
  tableColorSlot,
} from './index';
import embMeta from '../../../public/data/embeddings/minilm-l12-v2.json';

const SPLITS: Split[] = ['Train', 'Val', 'Test'];

/** Every table of a DB: DB tables plus every split of every forecast task table. */
function allTables(db: RawDatabase): { label: string; table: RawTable }[] {
  const out = db.tables.map((t) => ({ label: `${t.schema.name}:Db`, table: t }));
  for (const task of db.tasks) {
    if (task.kind !== 'forecast') continue;
    for (const s of SPLITS) out.push({ label: `${task.name}:${s}`, table: task.tables![s] });
  }
  return out;
}

function forecastTasks(db: RawDatabase): TaskDef[] {
  return db.tasks.filter((t) => t.kind === 'forecast');
}

function dbTable(db: RawDatabase, name: string): RawTable {
  const t = db.tables.find((x) => x.schema.name === name);
  if (!t) throw new Error(`no table ${name}`);
  return t;
}

describe.each(DB_NAMES)('database %s', (name) => {
  const db = getDatabase(name);

  it('has well-formed schemas and rows keyed in column order', () => {
    for (const { label, table } of allTables(db)) {
      const cols = table.schema.columns.map((c) => c.name);
      expect(table.rows.length, label).toBeGreaterThan(0);
      for (const row of table.rows) expect(Object.keys(row), label).toEqual(cols);
      if (table.schema.timeCol !== null) {
        expect(table.schema.columns.find((c) => c.name === table.schema.timeCol)?.dtype, label).toBe('datetime');
      }
      for (const c of table.schema.columns) expect(c.description, `${label}.${c.name}`).toBeTruthy();
    }
  });

  it('PK = row position (spec §1.2)', () => {
    for (const t of db.tables) {
      expect(t.schema.pkey, t.schema.name).not.toBeNull();
      t.rows.forEach((r, i) => expect(r[t.schema.pkey!]).toBe(i));
    }
    for (const task of forecastTasks(db)) for (const s of SPLITS) expect(task.tables![s].schema.pkey).toBeNull();
  });

  it('FK values are valid parent row positions or null; ≤5 FKs per table (spec §1.2, §4.16)', () => {
    for (const { label, table } of allTables(db)) {
      const fks = Object.entries(table.schema.fkeys);
      expect(fks.length, label).toBeLessThanOrEqual(5);
      for (const [fk, parentName] of fks) {
        const n = dbTable(db, parentName).rows.length; // parents are always DB tables (spec §1.3)
        for (const r of table.rows) {
          const v = r[fk];
          if (v === null) continue;
          expect(Number.isInteger(v), `${label}.${fk}`).toBe(true);
          expect(v as number).toBeGreaterThanOrEqual(0);
          expect(v as number).toBeLessThan(n);
        }
      }
    }
  });

  it('values match dtypes; no NaN/inf (missing = null)', () => {
    for (const { label, table } of allTables(db)) {
      for (const c of table.schema.columns) {
        for (const r of table.rows) {
          const v = r[c.name];
          if (v === null) continue;
          if (c.dtype === 'bool') expect(typeof v, `${label}.${c.name}`).toBe('boolean');
          else if (c.dtype === 'str') expect(typeof v, `${label}.${c.name}`).toBe('string');
          else {
            expect(typeof v, `${label}.${c.name}`).toBe('number');
            expect(Number.isFinite(v as number), `${label}.${c.name}`).toBe(true);
            if (c.dtype !== 'float') expect(Number.isInteger(v), `${label}.${c.name}`).toBe(true);
          }
        }
      }
    }
  });

  it('has NO constant boolean column in any table or task split (spec §3.3 NaN hazard)', () => {
    for (const { label, table } of allTables(db)) {
      for (const c of table.schema.columns) {
        if (c.dtype !== 'bool') continue;
        const vals = table.rows.map((r) => r[c.name]).filter((v) => v !== null);
        expect(vals.includes(true), `${label}.${c.name} has a true`).toBe(true);
        expect(vals.includes(false), `${label}.${c.name} has a false`).toBe(true);
      }
    }
  });

  it('contains some nulls (missing values produce no token, spec §3.5)', () => {
    const nulls = db.tables.reduce((acc, t) => acc + t.rows.reduce((a, r) => a + Object.values(r).filter((v) => v === null).length, 0), 0);
    expect(nulls).toBeGreaterThan(0);
  });

  it('keeps every datetime within 2024-01-01 .. 2025-06-30 (and so within i32 seconds, spec §1.4)', () => {
    for (const { label, table } of allTables(db)) {
      for (const c of table.schema.columns) {
        if (c.dtype !== 'datetime') continue;
        for (const r of table.rows) {
          const v = r[c.name];
          if (v === null) continue;
          expect(v as number, `${label}.${c.name}`).toBeGreaterThanOrEqual(DATA_START);
          expect(v as number, `${label}.${c.name}`).toBeLessThanOrEqual(DATA_END);
        }
      }
    }
    expect(db.minTimestamp).toBeGreaterThanOrEqual(DATA_START);
    expect(db.maxTimestamp).toBeLessThanOrEqual(DATA_END);
  });

  it('builds forecast task tables RelBench-style (spec §2.1-2.2)', () => {
    for (const task of forecastTasks(db)) {
      expect(task.id).toBe(`${db.name}/${task.name}`);
      const delta = task.timedeltaDays! * DAY_MS;
      for (const s of SPLITS) {
        const schema = task.tables![s].schema;
        expect(schema.name).toBe(task.name);
        expect(schema.pkey).toBeNull();
        expect(schema.fkeys).toEqual({ [task.entityCol!]: task.entityTable });
        expect(schema.timeCol).toBe('timestamp');
        expect(schema.columns.map((c) => [c.name, c.dtype])).toEqual([
          ['timestamp', 'datetime'],
          [task.entityCol, 'int'],
          [task.targetCol, task.targetType === 'binary' ? 'bool' : 'float'],
        ]);
      }
      // RelBench: test − val ≥ Δ, val/test windows inside the DB
      expect(task.testTimestamp! - task.valTimestamp!).toBeGreaterThanOrEqual(delta);
      expect(task.testTimestamp! + delta).toBeLessThanOrEqual(db.maxTimestamp);
      // train timestamps = val − kΔ, k = 1.., while ≥ dbMin + Δ (BUILD_PLAN §3)
      const train = [...new Set(task.tables!.Train.rows.map((r) => r.timestamp as number))].sort((a, b) => a - b);
      const expected: number[] = [];
      for (let k = 1; task.valTimestamp! - k * delta >= db.minTimestamp + delta; k++) expected.unshift(task.valTimestamp! - k * delta);
      expect(train).toEqual(expected);
      expect(train.length).toBeGreaterThanOrEqual(3);
      for (const r of task.tables!.Val.rows) expect(r.timestamp).toBe(task.valTimestamp);
      for (const r of task.tables!.Test.rows) expect(r.timestamp).toBe(task.testTimestamp);
    }
  });

  it('every split of every forecast task is non-empty; binary splits contain both classes', () => {
    for (const task of forecastTasks(db)) {
      for (const s of SPLITS) {
        const labels = task.tables![s].rows.map((r) => r[task.targetCol]);
        expect(labels.length, `${task.id} ${s}`).toBeGreaterThan(0);
        expect(labels.every((v) => v !== null), `${task.id} ${s} labels non-null`).toBe(true);
        if (task.targetType === 'binary') {
          expect(labels.includes(true), `${task.id} ${s} has positives`).toBe(true);
          expect(labels.includes(false), `${task.id} ${s} has negatives`).toBe(true);
        }
      }
    }
  });

  it('defines autocomplete tasks on existing columns with non-null seeds', () => {
    for (const task of db.tasks.filter((t) => t.kind === 'autocomplete')) {
      expect(task.id).toBe(`${db.name}/${task.table}.${task.targetCol}`);
      expect(task.name).toBe(task.table);
      const t = dbTable(db, task.table!);
      const names = t.schema.columns.map((c) => c.name);
      expect(names).toContain(task.targetCol);
      for (const c of task.columnsToDrop) expect(names).toContain(c);
      const dtype = t.schema.columns.find((c) => c.name === task.targetCol)!.dtype;
      expect(task.targetType === 'binary' ? ['bool'] : ['int', 'float']).toContain(dtype);
      expect(t.rows.filter((r) => r[task.targetCol] !== null).length).toBeGreaterThan(50);
    }
  });

  it('is deterministic: two independent builds are deep-equal; getDatabase is cached', () => {
    expect(buildDatabase(name)).toEqual(db);
    expect(getDatabase(name)).toBe(db);
  });

  it('gives every DB table a distinct color slot (1..8, never the task slot) and task tables slot 5', () => {
    const slots = db.tables.map((t) => tableColorSlot(name, t.schema.name));
    expect(new Set(slots).size).toBe(slots.length);
    for (const s of slots) {
      expect(s).toBeGreaterThanOrEqual(1);
      expect(s).toBeLessThanOrEqual(8);
      expect(s).not.toBe(TASK_TABLE_SLOT);
    }
    for (const task of forecastTasks(db)) {
      expect(tableColorSlot(name, task.name)).toBe(TASK_TABLE_SLOT);
      expect(tableColorSlot(name, `${task.name}:Val`)).toBe(TASK_TABLE_SLOT);
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Label correctness: independent brute-force recomputation (BUILD_PLAN §3 definitions)
// ─────────────────────────────────────────────────────────────────────────────

type Row = Record<string, RawValue>;
interface Recompute {
  /** rows that make entity e active (time ≤ t) */
  activity: (db: RawDatabase, e: number) => number[];
  /** label of entity e for the window (t0, t1] */
  label: (db: RawDatabase, e: number, t0: number, t1: number) => boolean | number;
}
const inWin = (t: RawValue, t0: number, t1: number) => (t as number) > t0 && (t as number) <= t1;
const rowsOf = (db: RawDatabase, table: string) => dbTable(db, table).rows;
const times = (rows: Row[], fk: string, e: number, tc: string) => rows.filter((r) => r[fk] === e).map((r) => r[tc] as number);

const RECOMPUTE: Record<string, Recompute> = {
  'shop/customer-churn': {
    activity: (db, e) => times(rowsOf(db, 'orders'), 'customer_id', e, 'order_time'),
    label: (db, e, t0, t1) => !rowsOf(db, 'orders').some((o) => o.customer_id === e && inWin(o.order_time, t0, t1)),
  },
  'shop/product-sales': {
    activity: (db, e) => times(rowsOf(db, 'orders'), 'product_id', e, 'order_time'),
    label: (db, e, t0, t1) =>
      rowsOf(db, 'orders')
        .filter((o) => o.product_id === e && inWin(o.order_time, t0, t1))
        .reduce((a, o) => a + (o.quantity as number), 0),
  },
  'school/student-dropout': {
    activity: (db, e) => times(rowsOf(db, 'submissions'), 'student_id', e, 'submitted_at'),
    label: (db, e, t0, t1) => !rowsOf(db, 'submissions').some((s) => s.student_id === e && inWin(s.submitted_at, t0, t1)),
  },
  'school/course-submissions': {
    activity: (db, e) => times(rowsOf(db, 'submissions'), 'course_id', e, 'submitted_at'),
    label: (db, e, t0, t1) => rowsOf(db, 'submissions').filter((s) => s.course_id === e && inWin(s.submitted_at, t0, t1)).length,
  },
  'racing/driver-dnf': {
    activity: (db, e) => times(rowsOf(db, 'results'), 'driver_id', e, 'date'),
    label: (db, e, t0, t1) => rowsOf(db, 'results').some((r) => r.driver_id === e && r.dnf === true && inWin(r.date, t0, t1)),
  },
  'racing/driver-points': {
    activity: (db, e) => times(rowsOf(db, 'results'), 'driver_id', e, 'date'),
    label: (db, e, t0, t1) =>
      rowsOf(db, 'results')
        .filter((r) => r.driver_id === e && inWin(r.date, t0, t1))
        .reduce((a, r) => a + (r.points as number), 0),
  },
  'forum/user-engagement': {
    activity: (db, e) => [...times(rowsOf(db, 'posts'), 'owner_id', e, 'created'), ...times(rowsOf(db, 'votes'), 'user_id', e, 'created')],
    label: (db, e, t0, t1) =>
      rowsOf(db, 'posts').some((p) => p.owner_id === e && inWin(p.created, t0, t1)) ||
      rowsOf(db, 'votes').some((v) => v.user_id === e && inWin(v.created, t0, t1)),
  },
  'forum/post-votes': {
    activity: (db, e) => [rowsOf(db, 'posts')[e].created as number],
    label: (db, e, t0, t1) => rowsOf(db, 'votes').filter((v) => v.post_id === e && v.is_upvote === true && inWin(v.created, t0, t1)).length,
  },
};

describe('forecast labels and the active-entity filter (recomputed by brute force)', () => {
  const tasks = DB_NAMES.flatMap((n) => forecastTasks(getDatabase(n)));

  it('covers every forecast task', () => {
    expect(tasks.map((t) => t.id).sort()).toEqual(Object.keys(RECOMPUTE).sort());
  });

  it.each(tasks.map((t) => [t.id, t] as const))('%s', (_id, task) => {
    const db = getDatabase(task.dbName as (typeof DB_NAMES)[number]);
    const rc = RECOMPUTE[task.id];
    const delta = task.timedeltaDays! * DAY_MS;
    const nEntities = dbTable(db, task.entityTable!).rows.length;
    const firstActive = Array.from({ length: nEntities }, (_, e) => Math.min(...rc.activity(db, e)));
    for (const s of SPLITS) {
      const rows = task.tables![s].rows;
      // active filter: exactly the entities with ≥1 activity row at or before t, ascending ids
      const stamps = [...new Set(rows.map((r) => r.timestamp as number))];
      for (const t of stamps) {
        const got = rows.filter((r) => r.timestamp === t).map((r) => r[task.entityCol!] as number);
        const want = firstActive.map((f, e) => (f <= t ? e : -1)).filter((e) => e >= 0);
        expect(got, `${task.id} ${s} @${new Date(t).toISOString()}`).toEqual(want);
      }
      // label spot-check: every 5th row (and all Val/Test rows)
      rows.forEach((r, i) => {
        if (s === 'Train' && i % 5 !== 0) return;
        const t = r.timestamp as number;
        const want = rc.label(db, r[task.entityCol!] as number, t, t + delta);
        expect(r[task.targetCol], `${task.id} ${s} row ${i}`).toBe(want);
      });
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Registry API
// ─────────────────────────────────────────────────────────────────────────────

describe('registry', () => {
  it('lists the four DBs with shop as the held-out target', () => {
    const list = listDatabases();
    expect(list.map((d) => d.name)).toEqual(['shop', 'school', 'racing', 'forum']);
    expect(list.map((d) => d.role)).toEqual(['target', 'pretrain', 'pretrain', 'pretrain']);
    for (const d of list) expect(d.title).toBe(getDatabase(d.name).title);
  });

  it('uses the plan color slots for ShopDB (BUILD_PLAN §3.5)', () => {
    expect(tableColorSlot('shop', 'customers')).toBe(1);
    expect(tableColorSlot('shop', 'products')).toBe(2);
    expect(tableColorSlot('shop', 'orders')).toBe(3);
    expect(tableColorSlot('shop', 'reviews')).toBe(7);
    expect(tableColorSlot('shop', 'customer-churn')).toBe(5);
    expect(tableColorSlot('shop', 'product-sales:Train')).toBe(5);
    expect(tableColorSlot('shop', 'orders:Db')).toBe(3);
    expect(() => tableColorSlot('shop', 'students')).toThrow();
  });

  it('has the ShopDB sizes and tasks of BUILD_PLAN §3.1', () => {
    const db = getDatabase('shop');
    expect(db.tables.map((t) => t.schema.name)).toEqual(['customers', 'products', 'orders', 'reviews']);
    expect(dbTable(db, 'customers').rows.length).toBe(80);
    expect(dbTable(db, 'products').rows.length).toBe(30);
    expect(dbTable(db, 'orders').rows.length).toBeGreaterThan(1000);
    expect(dbTable(db, 'orders').rows.length).toBeLessThan(1400);
    expect(dbTable(db, 'reviews').rows.length).toBeGreaterThan(250);
    expect(dbTable(db, 'reviews').rows.length).toBeLessThan(350);
    expect(db.tasks.map((t) => t.id)).toEqual([
      'shop/customer-churn',
      'shop/product-sales',
      'shop/reviews.rating',
      'shop/reviews.verified',
    ]);
    expect(getTask('shop/reviews.rating').columnsToDrop).toEqual(['review_text']);
    expect(getTask('racing/results.position').columnsToDrop).toEqual(['points', 'dnf']);
    expect(getTask('shop/customer-churn').valTimestamp).toBe(Date.UTC(2025, 2, 1));
    expect(getTask('shop/customer-churn').testTimestamp).toBe(Date.UTC(2025, 3, 15));
  });

  it('picks a showcase customer with a long, mixed churn history present in Val', () => {
    const db = getDatabase('shop');
    const sc = pickShowcaseEntity(db);
    expect(sc.taskId).toBe('shop/customer-churn');
    expect(sc.entityTable).toBe('customers');
    expect(sc.rows.Train.length).toBeGreaterThanOrEqual(6);
    expect(sc.labels.Train).toContain(true);
    expect(sc.labels.Train).toContain(false);
    expect(sc.rows.Val.length).toBe(1);
    const task = getTask('shop/customer-churn');
    for (const s of SPLITS) for (const p of sc.rows[s]) expect(task.tables![s].rows[p].customer_id).toBe(sc.entityRow);
    expect(pickShowcaseEntity(db)).toEqual(sc); // deterministic
  });

  it('builds column phrases exactly as "<col> of <table>" (spec §3.6)', () => {
    expect(columnPhrase('churn', 'customer-churn')).toBe('churn of customer-churn');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// ShopDB is frozen: the held-out target DB must not change when the pretraining DBs are tuned
// ─────────────────────────────────────────────────────────────────────────────

/**
 * SHA-256 of JSON.stringify(getDatabase('shop')) (tables, task tables of every split, metadata).
 * Recorded before the pretraining DBs were re-tuned for label persistence; the zero-shot results,
 * the parity fixtures and every chapter's running example depend on ShopDB staying byte-identical.
 */
const SHOP_SHA256 = '2d976108405eea67982186c7a51f6851f0db9de134eee8dc416a078a0f1df47f';

async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

describe('ShopDB is unchanged', () => {
  it('hashes to the recorded SHA-256 (deep equality with the previous ShopDB)', async () => {
    expect(await sha256Hex(JSON.stringify(getDatabase('shop')))).toBe(SHOP_SHA256);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Self-label persistence of the pretraining tasks (BUILD_PLAN §3: "make entity behavior persistent";
// spec §11.2: the target entity's past labels drive zero-shot transfer)
// ─────────────────────────────────────────────────────────────────────────────

interface LabelRow {
  t: number;
  e: number;
  y: number;
  split: Split;
}

function labelRows(task: TaskDef): LabelRow[] {
  return SPLITS.flatMap((split) =>
    task.tables![split].rows.map((r) => ({
      t: r.timestamp as number,
      e: r[task.entityCol!] as number,
      y: typeof r[task.targetCol] === 'boolean' ? (r[task.targetCol] ? 1 : 0) : (r[task.targetCol] as number),
      split,
    })),
  );
}

function pearson(a: number[], b: number[]): number {
  const n = a.length;
  const ma = a.reduce((s, x) => s + x, 0) / n;
  const mb = b.reduce((s, x) => s + x, 0) / n;
  let ab = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < n; i++) {
    ab += (a[i] - ma) * (b[i] - mb);
    aa += (a[i] - ma) ** 2;
    bb += (b[i] - mb) ** 2;
  }
  return ab / Math.sqrt(aa * bb);
}

function r2(y: number[], p: number[]): number {
  const m = y.reduce((s, x) => s + x, 0) / y.length;
  let sse = 0;
  let sst = 0;
  for (let i = 0; i < y.length; i++) {
    sse += (y[i] - p[i]) ** 2;
    sst += (y[i] - m) ** 2;
  }
  return 1 - sse / sst;
}

/** AUROC with average ranks for ties (= sklearn roc_auc_score for binary labels). */
function auroc(y: number[], s: number[]): number {
  const idx = s.map((_, i) => i).sort((a, b) => s[a] - s[b]);
  const ranks = new Array<number>(s.length);
  for (let i = 0; i < idx.length; ) {
    let j = i;
    while (j + 1 < idx.length && s[idx[j + 1]] === s[idx[i]]) j++;
    for (let k = i; k <= j; k++) ranks[idx[k]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  const nPos = y.filter((v) => v > 0).length;
  let rankSum = 0;
  for (let i = 0; i < y.length; i++) if (y[i] > 0) rankSum += ranks[i];
  return (rankSum - (nPos * (nPos + 1)) / 2) / (nPos * (y.length - nPos));
}

/**
 * Window-to-window label correlation (Pearson over (entity, consecutive Train timestamps) pairs) and
 * the EntityMean baseline on the Train split (paper App. I: mean of the entity's earlier labels of
 * the task table, fallback = Train mean; scripts/export-contexts.ts computes the same thing per seed).
 */
function persistence(task: TaskDef): { lag1: number; entityMeanTrain: number } {
  const rows = labelRows(task);
  const train = rows.filter((r) => r.split === 'Train');
  const trainMean = train.reduce((s, r) => s + r.y, 0) / train.length;
  const byEntity = new Map<number, LabelRow[]>();
  for (const r of rows) {
    if (!byEntity.has(r.e)) byEntity.set(r.e, []);
    byEntity.get(r.e)!.push(r);
  }
  const stamps = [...new Set(train.map((r) => r.t))].sort((a, b) => a - b);
  const prev: number[] = [];
  const next: number[] = [];
  for (let k = 1; k < stamps.length; k++) {
    for (const r of train) {
      if (r.t !== stamps[k]) continue;
      const p = byEntity.get(r.e)!.find((x) => x.t === stamps[k - 1]);
      if (p) {
        prev.push(p.y);
        next.push(r.y);
      }
    }
  }
  const pred = train.map((r) => {
    const past = byEntity.get(r.e)!.filter((x) => x.t < r.t);
    return past.length ? past.reduce((s, x) => s + x.y, 0) / past.length : trainMean;
  });
  const y = train.map((r) => r.y);
  return { lag1: pearson(prev, next), entityMeanTrain: task.targetType === 'binary' ? auroc(y, pred) : r2(y, pred) };
}

describe('pretraining tasks have persistent self labels (spec §11.2)', () => {
  const tasks = DB_NAMES.filter((n) => n !== 'shop').flatMap((n) => forecastTasks(getDatabase(n)));

  it.each(tasks.map((t) => [t.id, t] as const))('%s: window-to-window correlation ≥ 0.4, EntityMean beats the Train mean', (_id, task) => {
    const { lag1, entityMeanTrain } = persistence(task);
    expect(lag1).toBeGreaterThanOrEqual(0.4);
    // regression: EntityMean R² on Train > 0.3; binary: EntityMean AUROC on Train > 0.6
    expect(entityMeanTrain).toBeGreaterThan(task.targetType === 'binary' ? 0.6 : 0.3);
  });

  it('RacingDB uses 30-day windows and the shared cutoffs (RelBench rel-f1 driver-dnf: "next 1 month")', () => {
    for (const id of ['racing/driver-dnf', 'racing/driver-points']) {
      const t = getTask(id);
      expect(t.timedeltaDays).toBe(30);
      expect(t.valTimestamp).toBe(VAL_TIMESTAMP);
      expect(t.testTimestamp).toBe(getTask('shop/customer-churn').testTimestamp);
    }
  });
});

describe('precomputed embeddings (public/data/embeddings)', () => {
  it('cover every table name, column phrase and string value of every DB (spec §3.6)', () => {
    // If this fails, a generator changed: re-run `npx tsx scripts/export-strings.ts` and
    // `python/.venv/bin/python python/embed.py`.
    const have = new Set(embMeta.strings);
    expect(embMeta.dim).toBe(384);
    expect(embMeta.count).toBe(embMeta.strings.length);
    const missing: string[] = [];
    for (const n of DB_NAMES) {
      for (const { table } of allTables(getDatabase(n))) {
        const s = table.schema;
        if (!have.has(s.name)) missing.push(s.name);
        for (const c of s.columns) {
          if (!have.has(columnPhrase(c.name, s.name))) missing.push(columnPhrase(c.name, s.name));
          if (c.dtype !== 'str') continue;
          for (const r of table.rows) if (typeof r[c.name] === 'string' && !have.has(r[c.name] as string)) missing.push(r[c.name] as string);
        }
      }
    }
    expect([...new Set(missing)]).toEqual([]);
  });
});
