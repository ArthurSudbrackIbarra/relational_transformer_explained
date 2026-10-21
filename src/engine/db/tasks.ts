/**
 * RelBench-style predictive tasks for the dummy databases (spec §2; BUILD_PLAN §3).
 *
 * FORECAST tasks get a task TABLE with three splits (spec §2.1-2.2), built like RelBench
 * `BaseTask._get_table` + `make_table` (relbench/base/task_base.py:102-142 @ v1.1.0, local copy
 * research/specchecks/relbench_task_base.py):
 *   - schema: pkey = null, ONE FK `<entityCol> → <entityTable>`, time column `timestamp`,
 *     columns in SELECT order [timestamp (datetime), <entityCol> (int), <target> (bool|float)]
 *     (RelBench f1 driver-dnf: date, driverId, did_not_finish; spec §2.1);
 *   - train timestamps: val − kΔ, k = 1, 2, … (BUILD_PLAN §3: while ≥ dbMin + Δ, so every train
 *     window has at least Δ of history; RelBench itself goes back to dbMin);
 *   - val split: the single timestamp valTimestamp; test split: testTimestamp
 *     (num_eval_timestamps = 1, RelBench default);
 *   - rows at time t: every entity "active" at t, i.e. with ≥ 1 activity row at or before t
 *     (RelBench SQL uses `<= timestamp`);
 *   - label aggregates the window (t, t + Δ] (task_base.py docstring, spec §2.2).
 *   - RelBench sanity checks are enforced: test − val ≥ Δ (task_base.py:73-80), val + Δ and
 *     test + Δ ≤ dbMax (task_base.py:116-130), ≥ 3 train timestamps (task_base.py:144-148).
 *     Because test − val ≥ Δ, RelBench's `get_db(upto_test_timestamp=True)` truncation for
 *     train/val labels is a no-op here, so labels are computed on the full DB.
 *   Test labels are kept in the table, as in the RelBench parquet cache that pre.rs reads (spec §2.2).
 *
 * AUTOCOMPLETE tasks have no task table: the target is an existing DB column, seeds are all DB
 * rows with a non-null target (split ignored, spec §2.4), and `columnsToDrop` lists leakage
 * columns removed on the fly by the sampler (spec §4.9; PE/rt/tasks.py:30-88).
 */
import type { RawTable, RawValue, Split, TaskDef } from '../types';
import { DAY_MS, TEST_TIMESTAMP, VAL_TIMESTAMP, col, isoDate, makeTable } from './util';

export const DB_NAMES = ['shop', 'school', 'racing', 'forum'] as const;
export type DbName = (typeof DB_NAMES)[number];

/** Rows of a DB table that count as events of an entity. */
export interface EventSource {
  table: string;
  /** FK column that points at the entity, or null when each row IS an entity (row position). */
  entityFk: string | null;
  /** Datetime column giving the event time (the table's time column). */
  timeCol: string;
  /** Optional row filter (e.g. only DNF results, only up-votes). */
  where?: (row: Record<string, RawValue>) => boolean;
  /** Value accumulated by agg 'sum' (default 1 per row). */
  value?: (row: Record<string, RawValue>) => number;
}

/**
 * Label aggregation over the window (t, t+Δ]:
 *  - 'none'  → binary: true iff the entity has NO event in the window (churn-style);
 *  - 'any'   → binary: true iff it has at least one event;
 *  - 'count' → regression: number of events;
 *  - 'sum'   → regression: sum of `value` over events.
 */
export type LabelAgg = 'none' | 'any' | 'count' | 'sum';

export interface ForecastSpec {
  name: string;
  targetCol: string;
  targetType: 'binary' | 'regression';
  entityTable: string;
  entityCol: string;
  timedeltaDays: number;
  valTimestamp: number;
  testTimestamp: number;
  description: string;
  targetDescription: string;
  /** Rows that make an entity "active" (included in the task table) once one exists at time ≤ t. */
  activity: EventSource[];
  label: { agg: LabelAgg; sources: EventSource[] };
}

export interface AutocompleteSpec {
  table: string;
  targetCol: string;
  targetType: 'binary' | 'regression';
  columnsToDrop: string[];
  description: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Task specifications per DB (BUILD_PLAN §3.1-3.4)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * RacingDB used Δ = 60 days with val 2025-02-14 in an earlier version of the data; it now uses the
 * shared cutoffs and Δ = 30 days like RelBench rel-f1 driver-dnf / driver-top3 (timedelta 30 days;
 * races here run every two weeks, so a window holds about two races). Kept as an export for callers
 * that referenced it; equal to VAL_TIMESTAMP.
 */
export const RACING_VAL_TIMESTAMP = VAL_TIMESTAMP;

const orderEv = (fk: string): EventSource => ({ table: 'orders', entityFk: fk, timeCol: 'order_time' });
const subEv = (fk: string): EventSource => ({ table: 'submissions', entityFk: fk, timeCol: 'submitted_at' });
const resultEv: EventSource = { table: 'results', entityFk: 'driver_id', timeCol: 'date' };

export const TASK_SPECS: Record<DbName, { forecast: ForecastSpec[]; autocomplete: AutocompleteSpec[] }> = {
  shop: {
    forecast: [
      {
        name: 'customer-churn',
        targetCol: 'churn',
        targetType: 'binary',
        entityTable: 'customers',
        entityCol: 'customer_id',
        timedeltaDays: 30,
        valTimestamp: VAL_TIMESTAMP,
        testTimestamp: TEST_TIMESTAMP,
        description: 'Will the customer place no order in the next 30 days? (churn = true means no order in (t, t+30d]).',
        targetDescription: 'True iff the customer places no order in (timestamp, timestamp + 30 days].',
        activity: [orderEv('customer_id')],
        label: { agg: 'none', sources: [orderEv('customer_id')] },
      },
      {
        name: 'product-sales',
        targetCol: 'sales',
        targetType: 'regression',
        entityTable: 'products',
        entityCol: 'product_id',
        timedeltaDays: 30,
        valTimestamp: VAL_TIMESTAMP,
        testTimestamp: TEST_TIMESTAMP,
        description: 'How many units of the product will be ordered in the next 30 days?',
        targetDescription: 'Total quantity ordered in (timestamp, timestamp + 30 days].',
        activity: [orderEv('product_id')],
        label: { agg: 'sum', sources: [{ ...orderEv('product_id'), value: (r) => r.quantity as number }] },
      },
    ],
    autocomplete: [
      {
        table: 'reviews',
        targetCol: 'rating',
        targetType: 'regression',
        columnsToDrop: ['review_text'],
        description: 'Fill in a review’s star rating. The review text would leak it, so it is dropped.',
      },
      {
        table: 'reviews',
        targetCol: 'verified',
        targetType: 'binary',
        columnsToDrop: [],
        description: 'Is this review a verified purchase?',
      },
    ],
  },
  school: {
    forecast: [
      {
        name: 'student-dropout',
        targetCol: 'dropout',
        targetType: 'binary',
        entityTable: 'students',
        entityCol: 'student_id',
        timedeltaDays: 30,
        valTimestamp: VAL_TIMESTAMP,
        testTimestamp: TEST_TIMESTAMP,
        description: 'Will the student submit nothing in the next 30 days?',
        targetDescription: 'True iff the student makes no submission in (timestamp, timestamp + 30 days].',
        activity: [subEv('student_id')],
        label: { agg: 'none', sources: [subEv('student_id')] },
      },
      {
        name: 'course-submissions',
        targetCol: 'num_submissions',
        targetType: 'regression',
        entityTable: 'courses',
        entityCol: 'course_id',
        timedeltaDays: 30,
        valTimestamp: VAL_TIMESTAMP,
        testTimestamp: TEST_TIMESTAMP,
        description: 'How many assignments will be submitted for the course in the next 30 days?',
        targetDescription: 'Number of submissions in (timestamp, timestamp + 30 days].',
        activity: [subEv('course_id')],
        label: { agg: 'count', sources: [subEv('course_id')] },
      },
    ],
    autocomplete: [
      {
        table: 'submissions',
        targetCol: 'grade',
        targetType: 'regression',
        columnsToDrop: [],
        description: 'Fill in the grade of a submission.',
      },
      {
        table: 'submissions',
        targetCol: 'late',
        targetType: 'binary',
        columnsToDrop: [],
        description: 'Was this submission late?',
      },
    ],
  },
  racing: {
    forecast: [
      {
        name: 'driver-dnf',
        targetCol: 'did_not_finish',
        targetType: 'binary',
        entityTable: 'drivers',
        entityCol: 'driver_id',
        timedeltaDays: 30,
        valTimestamp: RACING_VAL_TIMESTAMP,
        testTimestamp: TEST_TIMESTAMP,
        description: 'Will the driver fail to finish at least one race in the next 30 days?',
        targetDescription: 'True iff the driver has a DNF in (timestamp, timestamp + 30 days].',
        activity: [resultEv],
        label: { agg: 'any', sources: [{ ...resultEv, where: (r) => r.dnf === true }] },
      },
      {
        name: 'driver-points',
        targetCol: 'points',
        targetType: 'regression',
        entityTable: 'drivers',
        entityCol: 'driver_id',
        timedeltaDays: 30,
        valTimestamp: RACING_VAL_TIMESTAMP,
        testTimestamp: TEST_TIMESTAMP,
        description: 'How many championship points will the driver score in the next 30 days?',
        targetDescription: 'Points scored in (timestamp, timestamp + 30 days].',
        activity: [resultEv],
        label: { agg: 'sum', sources: [{ ...resultEv, value: (r) => r.points as number }] },
      },
    ],
    autocomplete: [
      {
        table: 'results',
        targetCol: 'position',
        targetType: 'regression',
        columnsToDrop: ['points', 'dnf'],
        description: 'Fill in a driver’s finishing position. Points and DNF reveal it, so they are dropped.',
      },
    ],
  },
  forum: {
    forecast: [
      {
        name: 'user-engagement',
        targetCol: 'engaged',
        targetType: 'binary',
        entityTable: 'users',
        entityCol: 'user_id',
        timedeltaDays: 30,
        valTimestamp: VAL_TIMESTAMP,
        testTimestamp: TEST_TIMESTAMP,
        description: 'Will the user post or vote in the next 30 days? (The opposite polarity of churn.)',
        targetDescription: 'True iff the user posts or votes in (timestamp, timestamp + 30 days].',
        activity: [
          { table: 'posts', entityFk: 'owner_id', timeCol: 'created' },
          { table: 'votes', entityFk: 'user_id', timeCol: 'created' },
        ],
        label: {
          agg: 'any',
          sources: [
            { table: 'posts', entityFk: 'owner_id', timeCol: 'created' },
            { table: 'votes', entityFk: 'user_id', timeCol: 'created' },
          ],
        },
      },
      {
        name: 'post-votes',
        targetCol: 'upvotes',
        targetType: 'regression',
        entityTable: 'posts',
        entityCol: 'post_id',
        timedeltaDays: 30,
        valTimestamp: VAL_TIMESTAMP,
        testTimestamp: TEST_TIMESTAMP,
        description: 'How many up-votes will the post receive in the next 30 days?',
        targetDescription: 'Number of up-votes in (timestamp, timestamp + 30 days].',
        // a post is an entity from the moment it is created (RelBench stack post-votes: CreationDate <= t)
        activity: [{ table: 'posts', entityFk: null, timeCol: 'created' }],
        label: {
          agg: 'count',
          sources: [{ table: 'votes', entityFk: 'post_id', timeCol: 'created', where: (r) => r.is_upvote === true }],
        },
      },
    ],
    autocomplete: [
      {
        table: 'votes',
        targetCol: 'is_upvote',
        targetType: 'binary',
        columnsToDrop: [],
        description: 'Is this vote an up-vote?',
      },
    ],
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

/** Per entity: sorted event times (and values, aligned). */
interface EntityEvents {
  times: number[];
  values: number[];
}

function collectEvents(tables: Map<string, RawTable>, sources: EventSource[]): Map<number, EntityEvents> {
  const raw = new Map<number, { t: number; v: number }[]>();
  for (const src of sources) {
    const table = tables.get(src.table);
    if (!table) throw new Error(`event source table ${src.table} not found`);
    table.rows.forEach((row, pos) => {
      if (src.where && !src.where(row)) return;
      const t = row[src.timeCol];
      if (t === null) return; // no time → never in a window
      const entity = src.entityFk === null ? pos : row[src.entityFk];
      if (entity === null) return; // null FK → belongs to no entity
      const v = src.value ? src.value(row) : 1;
      let list = raw.get(entity as number);
      if (!list) raw.set(entity as number, (list = []));
      list.push({ t: t as number, v });
    });
  }
  const out = new Map<number, EntityEvents>();
  for (const [e, list] of raw) {
    list.sort((a, b) => a.t - b.t);
    out.set(e, { times: list.map((x) => x.t), values: list.map((x) => x.v) });
  }
  return out;
}

/** First index i with times[i] > t (times sorted ascending). */
function upperBound(times: number[], t: number): number {
  let lo = 0;
  let hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (times[mid] <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Aggregate events in the half-open window (t0, t1]. */
function aggregate(ev: EntityEvents | undefined, t0: number, t1: number, agg: LabelAgg): boolean | number {
  let count = 0;
  let sum = 0;
  if (ev) {
    const a = upperBound(ev.times, t0); // first event > t0
    const b = upperBound(ev.times, t1); // first event > t1
    count = b - a;
    for (let i = a; i < b; i++) sum += ev.values[i];
  }
  switch (agg) {
    case 'none':
      return count === 0;
    case 'any':
      return count > 0;
    case 'count':
      return count;
    case 'sum':
      return sum;
  }
}

/** The timestamps of each split (ascending), with RelBench's sanity checks. */
export function forecastTimestamps(spec: ForecastSpec, dbMin: number, dbMax: number): Record<Split, number[]> {
  const delta = spec.timedeltaDays * DAY_MS;
  const { valTimestamp: val, testTimestamp: test } = spec;
  if (test - val < delta) {
    throw new Error(`${spec.name}: timedelta larger than test − val (RelBench task_base.py:73-80)`);
  }
  if (val + delta > dbMax) throw new Error(`${spec.name}: val + Δ beyond the DB max timestamp (task_base.py:116-121)`);
  if (test + delta > dbMax) throw new Error(`${spec.name}: test + Δ beyond the DB max timestamp (task_base.py:125-130)`);
  const train: number[] = [];
  for (let k = 1; val - k * delta >= dbMin + delta; k++) train.push(val - k * delta);
  train.reverse();
  if (train.length < 3) throw new Error(`${spec.name}: fewer than 3 train timestamps (task_base.py:144-148)`);
  return { Train: train, Val: [val], Test: [test] };
}

/** Build one forecast TaskDef (with its Train/Val/Test task tables). */
export function buildForecastTask(
  dbName: string,
  tables: Map<string, RawTable>,
  spec: ForecastSpec,
  dbMin: number,
  dbMax: number,
): TaskDef {
  const entityTable = tables.get(spec.entityTable);
  if (!entityTable) throw new Error(`${spec.name}: entity table ${spec.entityTable} not found`);
  const nEntities = entityTable.rows.length;
  const delta = spec.timedeltaDays * DAY_MS;
  const stamps = forecastTimestamps(spec, dbMin, dbMax);

  // first activity time per entity → active filter
  const activity = collectEvents(tables, spec.activity);
  const firstActive = new Array<number>(nEntities).fill(Infinity);
  for (const [e, ev] of activity) firstActive[e] = ev.times[0];
  const events = collectEvents(tables, spec.label.sources);

  const isBinary = spec.targetType === 'binary';
  const targetDtype = isBinary ? 'bool' : 'float';
  const deltaText = `${spec.timedeltaDays} days`;

  const makeSplit = (split: Split): RawTable => {
    const records: Record<string, RawValue>[] = [];
    for (const t of stamps[split]) {
      for (let e = 0; e < nEntities; e++) {
        if (!(firstActive[e] <= t)) continue; // active filter: ≥1 activity row at or before t
        records.push({
          timestamp: t,
          [spec.entityCol]: e,
          [spec.targetCol]: aggregate(events.get(e), t, t + delta, spec.label.agg),
        });
      }
    }
    return makeTable(
      {
        name: spec.name, // all three splits share the table name (one table_name_idx, spec §2.2)
        pkey: null,
        fkeys: { [spec.entityCol]: spec.entityTable },
        timeCol: 'timestamp',
        description:
          `Task table of "${spec.name}" (${split} split). One row per active ${spec.entityTable.replace(/s$/, '')} ` +
          `and prediction time; the label looks ${deltaText} ahead.`,
        columns: [
          col('timestamp', 'datetime', `Prediction time t. The label covers (t, t + ${deltaText}].`),
          col(spec.entityCol, 'int', `Foreign key → ${spec.entityTable}: the entity the row is about.`),
          col(spec.targetCol, targetDtype, spec.targetDescription),
        ],
      },
      records,
    );
  };

  return {
    name: spec.name,
    id: `${dbName}/${spec.name}`,
    dbName,
    kind: 'forecast',
    targetType: spec.targetType,
    targetCol: spec.targetCol,
    columnsToDrop: [],
    description: spec.description,
    entityTable: spec.entityTable,
    entityCol: spec.entityCol,
    timeCol: 'timestamp',
    timedeltaDays: spec.timedeltaDays,
    valTimestamp: spec.valTimestamp,
    testTimestamp: spec.testTimestamp,
    tables: { Train: makeSplit('Train'), Val: makeSplit('Val'), Test: makeSplit('Test') },
  };
}

/** Build one autocomplete TaskDef (no task table; spec §2.4). */
export function buildAutocompleteTask(dbName: string, tables: Map<string, RawTable>, spec: AutocompleteSpec): TaskDef {
  const table = tables.get(spec.table);
  if (!table) throw new Error(`autocomplete: table ${spec.table} not found`);
  const names = table.schema.columns.map((c) => c.name);
  for (const c of [spec.targetCol, ...spec.columnsToDrop]) {
    if (!names.includes(c)) throw new Error(`autocomplete ${spec.table}.${spec.targetCol}: column ${c} not found`);
  }
  const dtype = table.schema.columns.find((c) => c.name === spec.targetCol)!.dtype;
  // the decoder/loss is chosen by the target cell's sem type (model.py:238-258, spec §2.6)
  const ok = spec.targetType === 'binary' ? dtype === 'bool' : dtype === 'int' || dtype === 'float';
  if (!ok) throw new Error(`autocomplete ${spec.table}.${spec.targetCol}: dtype ${dtype} does not fit ${spec.targetType}`);
  return {
    name: spec.table,
    id: `${dbName}/${spec.table}.${spec.targetCol}`,
    dbName,
    kind: 'autocomplete',
    targetType: spec.targetType,
    targetCol: spec.targetCol,
    columnsToDrop: spec.columnsToDrop.slice(),
    description: spec.description,
    table: spec.table,
  };
}

/** All tasks of a DB: forecast tasks first, then autocomplete tasks (BUILD_PLAN §3 order). */
export function buildTasks(dbName: DbName, tables: RawTable[], dbMin: number, dbMax: number): TaskDef[] {
  const byName = new Map(tables.map((t) => [t.schema.name, t]));
  const specs = TASK_SPECS[dbName];
  return [
    ...specs.forecast.map((s) => buildForecastTask(dbName, byName, s, dbMin, dbMax)),
    ...specs.autocomplete.map((s) => buildAutocompleteTask(dbName, byName, s)),
  ];
}

/** Human-readable summary of a forecast task's timeline, e.g. for logs and UI captions. */
export function describeTimeline(task: TaskDef): string {
  if (task.kind !== 'forecast' || !task.tables) return `${task.id}: autocomplete (no timeline)`;
  const train = [...new Set(task.tables.Train.rows.map((r) => r.timestamp as number))];
  return (
    `${task.id}: Δ=${task.timedeltaDays}d, train ${isoDate(train[0])}..${isoDate(train[train.length - 1])} ` +
    `(${train.length} timestamps), val ${isoDate(task.valTimestamp!)}, test ${isoDate(task.testTimestamp!)}`
  );
}
