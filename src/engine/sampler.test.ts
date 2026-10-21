/**
 * Tests for sampler.ts (mirror of fly.rs:357-511, spec §4).
 *  1. The spec §6.3 worked example: for some RNG seed our sampler reproduces the printed token
 *     list exactly (and its masks), and every printed decision appears in the event trace.
 *  2. Invariants on hand-made DBs (truncation, leakage drop rule, other task tables, seeds).
 *  3. Invariants on randomized small DBs, including a cross-check against an independent,
 *     verbatim transcription of the spec §4.3 pseudo-code that consumes the same RNG.
 */
import { describe, expect, it } from 'vitest';
import { buildMasks, maskToAscii } from './masks';
import { preprocess } from './preprocess';
import { Rng } from './rng';
import {
  dropColNameIdxs,
  listSeeds,
  paddingToken,
  sampleContext,
  sampleParamsFor,
  seedTableKey,
  targetColNameIdx,
  targetStats,
} from './sampler';
import {
  MAX_F2P_NBRS,
  type DType,
  type PreprocessedDB,
  type RawDatabase,
  type RawTable,
  type SampleParams,
  type SampledContext,
  type Split,
  type TableSchema,
  type TaskDef,
} from './types';

const D = (s: string) => Date.parse(`${s}T00:00:00Z`);
const DAY = 86_400_000;

// ─────────────────────────────────────────────────────────────────────────────
// Toy DB of spec §6.3 (same data as preprocess.test.ts)
// ─────────────────────────────────────────────────────────────────────────────
function table(schema: Partial<TableSchema> & Pick<TableSchema, 'name' | 'columns'>, rows: RawTable['rows']): RawTable {
  return { schema: { pkey: null, fkeys: {}, timeCol: null, ...schema }, rows };
}
function churnTable(rows: [string, number, boolean | null][], name = 'customer-churn', target = 'churn'): RawTable {
  return table(
    {
      name,
      fkeys: { customer_id: 'customers' },
      timeCol: 'timestamp',
      columns: [
        { name: 'timestamp', dtype: 'datetime' },
        { name: 'customer_id', dtype: 'int' },
        { name: target, dtype: 'bool' },
      ],
    },
    rows.map(([t, c, y]) => ({ timestamp: D(t), customer_id: c, [target]: y })),
  );
}
function forecastTask(name: string, targetCol: string, tables: Partial<Record<Split, RawTable>>): TaskDef {
  return {
    name,
    id: `toy/${name}`,
    dbName: 'toy',
    kind: 'forecast',
    targetType: 'binary',
    targetCol,
    columnsToDrop: [],
    description: '',
    tables: tables as TaskDef['tables'],
  };
}
function toyDb(extraTasks: TaskDef[] = []): RawDatabase {
  const customers = table(
    {
      name: 'customers',
      pkey: 'customer_id',
      columns: [
        { name: 'customer_id', dtype: 'int' },
        { name: 'country', dtype: 'str' },
        { name: 'age', dtype: 'float' },
        { name: 'is_premium', dtype: 'bool' },
      ],
    },
    [
      { customer_id: 0, country: 'BR', age: 34, is_premium: true },
      { customer_id: 1, country: 'US', age: 51, is_premium: false },
      { customer_id: 2, country: 'BR', age: null, is_premium: false },
    ],
  );
  const products = table(
    {
      name: 'products',
      pkey: 'product_id',
      columns: [
        { name: 'product_id', dtype: 'int' },
        { name: 'category', dtype: 'str' },
        { name: 'price', dtype: 'float' },
      ],
    },
    [
      { product_id: 0, category: 'books', price: 12 },
      { product_id: 1, category: 'toys', price: 30 },
    ],
  );
  const orders = table(
    {
      name: 'orders',
      pkey: 'order_id',
      fkeys: { customer_id: 'customers', product_id: 'products' },
      timeCol: 'order_time',
      columns: [
        { name: 'order_id', dtype: 'int' },
        { name: 'customer_id', dtype: 'int' },
        { name: 'product_id', dtype: 'int' },
        { name: 'order_time', dtype: 'datetime' },
        { name: 'quantity', dtype: 'int' },
      ],
    },
    [
      { order_id: 0, customer_id: 0, product_id: 0, order_time: D('2024-01-10'), quantity: 1 },
      { order_id: 1, customer_id: 0, product_id: 1, order_time: D('2024-02-05'), quantity: 2 },
      { order_id: 2, customer_id: 1, product_id: 0, order_time: D('2024-02-20'), quantity: 1 },
      { order_id: 3, customer_id: 0, product_id: 0, order_time: D('2024-03-15'), quantity: 3 },
      { order_id: 4, customer_id: 2, product_id: 1, order_time: D('2024-01-25'), quantity: 1 },
    ],
  );
  const churn = forecastTask('customer-churn', 'churn', {
    Train: churnTable([
      ['2024-01-01', 0, false],
      ['2024-01-01', 1, true],
      ['2024-02-01', 0, false],
      ['2024-02-01', 2, true],
    ]),
    Val: churnTable([
      ['2024-03-01', 0, true],
      ['2024-03-01', 1, false],
    ]),
    Test: churnTable([
      ['2024-04-01', 0, false],
      ['2024-04-01', 2, true],
    ]),
  });
  return {
    name: 'toy',
    title: 'toy',
    description: '',
    tables: [customers, orders, products],
    tasks: [churn, ...extraTasks],
    minTimestamp: D('2024-01-01'),
    maxTimestamp: D('2024-04-01'),
  };
}

// The token list printed in spec §6.3 (seed #16, L=24, w=1): [node, column, raw display, f2p]
const SPEC_TOKENS: [number, string, string, number[]][] = [
  [16, 'timestamp', '2024-03-01', [0]],
  [16, 'churn', 'true', [0]], // [MASK] (the stored label is still the true value, spec §5.3)
  [0, 'country', 'BR', []],
  [0, 'age', '34', []],
  [0, 'is_premium', 'true', []],
  [14, 'timestamp', '2024-02-01', [0]],
  [14, 'churn', 'false', [0]],
  [4, 'order_time', '2024-02-05', [0, 9]],
  [4, 'quantity', '2', [0, 9]],
  [9, 'category', 'toys', []],
  [9, 'price', '30', []],
  [12, 'timestamp', '2024-01-01', [0]],
  [12, 'churn', 'false', [0]],
  [7, 'order_time', '2024-01-25', [2, 9]],
  [7, 'quantity', '1', [2, 9]],
  [2, 'country', 'BR', []],
  [2, 'is_premium', 'false', []],
  [15, 'timestamp', '2024-02-01', [2]],
  [15, 'churn', 'true', [2]],
];

// The masks printed in spec §6.3 (rows = query, '#' = allowed); rows 19-23 are all '.'.
const SPEC_COL = [
  '#....#.....#.....#......', '.#....#.....#.....#.....', '..#............#........', '...#....................',
  '....#...........#.......', '#....#.....#.....#......', '.#....#.....#.....#.....', '.......#.....#..........',
  '........#.....#.........', '.........#..............', '..........#.............', '#....#.....#.....#......',
  '.#....#.....#.....#.....', '.......#.....#..........', '........#.....#.........', '..#............#........',
  '....#...........#.......', '#....#.....#.....#......', '.#....#.....#.....#.....',
];
const SPEC_FEAT = [
  '#####...................', '#####...................', '..###...................', '..###...................',
  '..###...................', '..#####.................', '..#####.................', '..###..####.............',
  '..###..####.............', '.........##.............', '.........##.............', '..###......##...........',
  '..###......##...........', '.........##..####.......', '.........##..####.......', '...............##.......',
  '...............##.......', '...............####.....', '...............####.....',
];
const SPEC_NBR = [
  '........................', '........................', '##...####..##...........', '##...####..##...........',
  '##...####..##...........', '........................', '........................', '........................',
  '........................', '.......##....##.........', '.......##....##.........', '........................',
  '........................', '........................', '........................', '.............##..##.....',
  '.............##..##.....', '........................', '........................',
];
const PAD_ROWS = Array.from({ length: 5 }, () => '.'.repeat(24));

function rawDisplay(raw: unknown, colName: string): string {
  if (colName === 'timestamp' || colName === 'order_time') return new Date(raw as number).toISOString().slice(0, 10);
  return String(raw);
}

function matchesSpec(ctx: SampledContext): boolean {
  if (ctx.numReal !== SPEC_TOKENS.length) return false;
  return SPEC_TOKENS.every(([node, col], i) => ctx.tokens[i].nodeIdx === node && ctx.tokens[i].colName === col);
}

describe('sampler: spec §6.3 worked example (seed #16, L=24, w=1)', () => {
  const raw = toyDb();
  const pdb = preprocess(raw);
  const task = raw.tasks[0];
  const base = sampleParamsFor(pdb, task, 16, { L: 24, w: 1 });

  // Find an RNG seed whose draws coincide with the ones of the spec's run (≈1/16 of seeds).
  let found: SampledContext | null = null;
  for (let s = 0; s < 2000 && !found; s++) {
    const ctx = sampleContext(pdb, { ...base, rngSeed: s });
    if (matchesSpec(ctx)) found = ctx;
  }

  it('some RNG seed reproduces the printed token list exactly', () => {
    expect(found).not.toBeNull();
    const ctx = found!;
    expect(ctx.tokens).toHaveLength(24);
    SPEC_TOKENS.forEach(([node, col, rawStr, f2p], i) => {
      const t = ctx.tokens[i];
      expect(t.pos).toBe(i);
      expect(t.nodeIdx).toBe(node);
      expect(t.colName).toBe(col);
      expect(rawDisplay(t.raw, col)).toBe(rawStr);
      expect(t.f2pNbrIdxs).toEqual([...f2p, -1, -1, -1, -1, -1].slice(0, 5));
    });
    // exactly one masked token: the seed's churn cell
    expect(ctx.tokens.filter((t) => t.mask).map((t) => t.pos)).toEqual([1]);
    expect(ctx.targetPos).toBe(1);
    expect(ctx.tokens[1]).toMatchObject({ isTarget: true, mask: true, colNameIdx: base.targetColNameIdx });
    expect(ctx.tokens[1].booleanValue).toBeGreaterThan(0); // label true stays in the token
    // padding 19..23 with fly.rs defaults
    for (let p = 19; p < 24; p++) expect(ctx.tokens[p]).toEqual(paddingToken(p));
    expect(ctx.numReal).toBe(19);
  });

  it('the visit order and depth labels follow the spec narrative', () => {
    const ctx = found!;
    expect(ctx.visited.map((v) => [v.node, v.depth])).toEqual([
      [16, 0], [0, 1], [14, 2], [4, 2], [9, 3], [12, 2], [7, 4], [2, 5], [15, 6],
    ]);
    expect(ctx.tokens.slice(0, 19).map((t) => t.depth)).toEqual([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 4, 4, 5, 5, 6, 6]);
  });

  it('the event trace records every decision of the narrative', () => {
    const ev = found!.events;
    const has = (e: object) => ev.some((x) => Object.entries(e).every(([k, v]) => JSON.stringify((x as Record<string, unknown>)[k]) === JSON.stringify(v)));
    // step 2: o3 (#6) and Test #10 are future children of c0; #12, #14, #16 go to bucket 2
    const seedTs = D('2024-03-01') / 1000;
    expect(has({ kind: 'child-future', from: 0, node: 6, childTs: D('2024-03-15') / 1000, seedTs })).toBe(true);
    expect(has({ kind: 'child-future', from: 0, node: 10, childTs: D('2024-04-01') / 1000, seedTs })).toBe(true);
    for (const n of [12, 14, 16]) expect(has({ kind: 'push-task-child', from: 0, node: n, depth: 2 })).toBe(true);
    // DB kids [o0, o1] exceed w=1; o1 kept, o0 never reached
    expect(has({ kind: 'db-children-subsampled', from: 0, candidates: [3, 4], kept: [4], w: 1 })).toBe(true);
    expect(found!.visited.some((v) => v.node === 3)).toBe(false);
    // p1's DB kids [o1, o4]: o4 kept → bucket 4
    expect(has({ kind: 'db-children-subsampled', from: 9, candidates: [4, 7], kept: [7], w: 1 })).toBe(true);
    expect(has({ kind: 'push-db-child', from: 9, node: 7, depth: 4 })).toBe(true);
    // the seed itself sits in bucket 2 and is skipped as visited; c2's Test row #11 is future
    expect(has({ kind: 'skip-visited', node: 16, depth: 2 })).toBe(true);
    expect(has({ kind: 'child-future', from: 2, node: 11 })).toBe(true);
    expect(has({ kind: 'push-task-child', from: 2, node: 15, depth: 6 })).toBe(true);
    expect(ev.at(-2)).toEqual({ kind: 'frontier-empty', tokens: 19 });
    expect(ev.at(-1)).toEqual({ kind: 'pad', from: 19, to: 23 });
    expect(ev[0]).toEqual({ kind: 'pop', from: 'stack', node: 16, depth: 0 });
  });

  it('its masks equal the printed masks', () => {
    const m = buildMasks(found!.tokens);
    expect(maskToAscii(m.col, 24)).toEqual([...SPEC_COL, ...PAD_ROWS]);
    expect(maskToAscii(m.feat, 24)).toEqual([...SPEC_FEAT, ...PAD_ROWS]);
    expect(maskToAscii(m.nbr, 24)).toEqual([...SPEC_NBR, ...PAD_ROWS]);
    const fullRow = '#'.repeat(19) + '.....';
    expect(maskToAscii(m.full, 24)).toEqual([...Array(19).fill(fullRow), ...PAD_ROWS]);
  });

  it('w=2 can hit L=24 in the middle of row c1 (only its country emitted)', () => {
    let cut: SampledContext | null = null;
    for (let s = 0; s < 3000 && !cut; s++) {
      const ctx = sampleContext(pdb, { ...base, w: 2, rngSeed: s });
      if (ctx.tokens[23].nodeIdx === 1 && ctx.tokens[23].colName === 'country') cut = ctx;
    }
    expect(cut).not.toBeNull();
    const c = cut!;
    expect(c.numReal).toBe(24);
    expect(c.tokens.filter((t) => t.nodeIdx === 1).map((t) => t.colName)).toEqual(['country']);
    expect(c.events.at(-1)).toEqual({ kind: 'truncated', node: 1, pos: 23 });
    expect(c.events.some((e) => e.kind === 'pad')).toBe(false);
  });
});

describe('sampler: hand-made invariants', () => {
  const raw = toyDb();
  const pdb = preprocess(raw);
  const task = raw.tasks[0];

  it('listSeeds / targetColNameIdx / dropColNameIdxs mirror fly.rs:287-296 and data.py', () => {
    expect(listSeeds(pdb, task, 'Test')).toEqual([10, 11]);
    expect(listSeeds(pdb, task, 'Train')).toEqual([12, 13, 14, 15]);
    expect(listSeeds(pdb, task, 'Val')).toEqual([16, 17]);
    expect(targetColNameIdx(pdb, task)).toBe(pdb.columnIndex['churn of customer-churn']);
    expect(dropColNameIdxs(pdb, task)).toEqual([]);
    expect(seedTableKey(pdb, task, 'Val')).toBe('customer-churn:Val');
    expect(targetStats(pdb, task)).toMatchObject({ mean: 0.5, kind: 'boolean' });
    expect(() => targetColNameIdx(pdb, { ...task, targetCol: 'nope' })).toThrow(/not found/);
  });

  it('null labels are not seeds; autocomplete seeds ignore the split (§2.3-2.4)', () => {
    const t = toyDb();
    t.tasks[0].tables!.Val = churnTable([
      ['2024-03-01', 0, null],
      ['2024-03-01', 1, false],
    ]);
    const auto: TaskDef = {
      name: 'customers', id: 'toy/age', dbName: 'toy', kind: 'autocomplete', targetType: 'regression',
      targetCol: 'age', columnsToDrop: ['is_premium'], description: '', table: 'customers',
    };
    t.tasks.push(auto);
    const p = preprocess(t);
    expect(listSeeds(p, t.tasks[0], 'Val')).toEqual([17]);
    for (const split of ['Train', 'Val', 'Test'] as Split[]) expect(listSeeds(p, auto, split)).toEqual([0, 1]); // c2 has no age
    expect(seedTableKey(p, auto, 'Test')).toBe('customers:Db');
    expect(dropColNameIdxs(p, auto)).toEqual([p.columnIndex['is_premium of customers']]);
    expect(targetStats(p, auto)).toMatchObject({ mean: 42.5 });
  });

  it('hard stop at exactly L, even mid-row (fly.rs:502-509)', () => {
    const ctx = sampleContext(pdb, sampleParamsFor(pdb, task, 16, { L: 3, w: 16 }));
    expect(ctx.tokens.map((t) => [t.nodeIdx, t.colName])).toEqual([
      [16, 'timestamp'],
      [16, 'churn'],
      [0, 'country'],
    ]); // c0's age and is_premium are cut
    expect(ctx.numReal).toBe(3);
    expect(ctx.events.at(-1)).toEqual({ kind: 'truncated', node: 0, pos: 2 });
    // L=1: the target (2nd cell of the seed row) does not fit — degenerate, targetPos = −1
    const one = sampleContext(pdb, sampleParamsFor(pdb, task, 16, { L: 1 }));
    expect(one.tokens.map((t) => t.colName)).toEqual(['timestamp']);
    expect(one.targetPos).toBe(-1);
  });

  it('with large L every same-task row with ts ≤ seed ts is reached; future and other-task rows never', () => {
    const ltv = forecastTask('customer-ltv', 'big', {
      Train: churnTable([['2024-01-01', 0, true], ['2024-01-01', 1, false]], 'customer-ltv', 'big'),
    });
    const r = toyDb([ltv]);
    const p = preprocess(r);
    for (let s = 0; s < 20; s++) {
      const ctx = sampleContext(p, sampleParamsFor(p, r.tasks[0], 16, { L: 1000, w: 100, rngSeed: s }));
      const seen = new Set(ctx.visited.map((v) => v.node));
      const churnRows = (k: string) => {
        const i = p.tableInfo[k];
        return Array.from({ length: i.numNodes }, (_, j) => i.nodeIdxOffset + j);
      };
      for (const n of [...churnRows('customer-churn:Train'), ...churnRows('customer-churn:Val')]) expect(seen.has(n)).toBe(true);
      for (const n of [...churnRows('customer-churn:Test'), ...churnRows('customer-ltv:Train')]) expect(seen.has(n)).toBe(false);
      expect(seen.has(6)).toBe(false); // o3 is in the future of the seed
      const ltvRows = churnRows('customer-ltv:Train');
      expect(ctx.events.filter((e) => e.kind === 'child-other-task-table').map((e) => (e as { node: number }).node).sort()).toEqual(
        expect.arrayContaining([ltvRows[0]]),
      );
      expect(ctx.numReal).toBe(ctx.tokens.filter((t) => !t.isPadding).length);
    }
  });

  it('autocomplete seed: no task rows at all; leakage columns dropped for the seed and same-ts rows (None==None)', () => {
    // results(race FK, date, position target, points leak): r0 and r1 same date, r2 other date.
    const races = table({ name: 'races', pkey: 'id', columns: [{ name: 'id', dtype: 'int' }, { name: 'circuit', dtype: 'str' }] }, [{ id: 0, circuit: 'Monza' }]);
    const results = table(
      {
        name: 'results',
        pkey: 'id',
        fkeys: { race: 'races' },
        timeCol: 'date',
        columns: [
          { name: 'id', dtype: 'int' },
          { name: 'race', dtype: 'int' },
          { name: 'date', dtype: 'datetime' },
          { name: 'position', dtype: 'int' },
          { name: 'points', dtype: 'float' },
        ],
      },
      [
        { id: 0, race: 0, date: D('2024-05-01'), position: 1, points: 25 },
        { id: 1, race: 0, date: D('2024-05-01'), position: 2, points: 18 },
        { id: 2, race: 0, date: D('2024-04-01'), position: 3, points: 15 },
      ],
    );
    // items without a time column: the seed ts is None, so every None-ts row of the table drops `leak`.
    const items = table(
      { name: 'items', pkey: 'id', fkeys: { race: 'races' }, columns: [{ name: 'id', dtype: 'int' }, { name: 'race', dtype: 'int' }, { name: 'val', dtype: 'float' }, { name: 'leak', dtype: 'float' }] },
      [
        { id: 0, race: 0, val: 1, leak: 10 },
        { id: 1, race: 0, val: 2, leak: 20 },
      ],
    );
    const entity = table({ name: 'drivers', pkey: 'id', columns: [{ name: 'id', dtype: 'int' }, { name: 'n', dtype: 'str' }] }, [{ id: 0, n: 'A' }]);
    const dnf = forecastTask('races-dnf', 'y', {
      Train: table(
        { name: 'races-dnf', fkeys: { r: 'races' }, timeCol: 'ts', columns: [{ name: 'ts', dtype: 'datetime' }, { name: 'r', dtype: 'int' }, { name: 'y', dtype: 'bool' }] },
        [{ ts: D('2024-01-01'), r: 0, y: true }, { ts: D('2024-01-02'), r: 0, y: false }],
      ),
    });
    const auto: TaskDef = { name: 'results', id: 'x/pos', dbName: 'x', kind: 'autocomplete', targetType: 'regression', targetCol: 'position', columnsToDrop: ['points'], description: '', table: 'results' };
    const auto2: TaskDef = { name: 'items', id: 'x/val', dbName: 'x', kind: 'autocomplete', targetType: 'regression', targetCol: 'val', columnsToDrop: ['leak'], description: '', table: 'items' };
    const r: RawDatabase = { name: 'x', title: 'x', description: '', tables: [races, results, items, entity], tasks: [dnf, auto, auto2], minTimestamp: 0, maxTimestamp: 0 };
    const p = preprocess(r);
    const res0 = p.tableInfo['results:Db'].nodeIdxOffset;
    const ctx = sampleContext(p, sampleParamsFor(p, auto, res0, { L: 100, w: 10 }));
    const colsOf = (n: number) => ctx.tokens.filter((t) => t.nodeIdx === n).map((t) => t.colName);
    expect(colsOf(res0)).toEqual(['date', 'position']); // seed: points dropped
    expect(colsOf(res0 + 1)).toEqual(['date', 'position']); // same date: points dropped, position visible
    expect(colsOf(res0 + 2)).toEqual(['date', 'position', 'points']); // other date: visible
    expect(ctx.tokens.filter((t) => t.isTarget).map((t) => t.nodeIdx)).toEqual([res0]);
    expect(ctx.tokens.some((t) => t.tableType !== null && t.tableType !== 'Db')).toBe(false);
    expect(ctx.events.filter((e) => e.kind === 'child-other-task-table')).toHaveLength(2);
    expect(ctx.events.filter((e) => e.kind === 'drop-leakage-cell')).toHaveLength(2);

    const it0 = p.tableInfo['items:Db'].nodeIdxOffset;
    const ctx2 = sampleContext(p, sampleParamsFor(p, auto2, it0, { L: 100, w: 10 }));
    expect(p.nodes[it0].ts).toBeNull();
    const cols2 = (n: number) => ctx2.tokens.filter((t) => t.nodeIdx === n).map((t) => t.colName);
    expect(cols2(it0)).toEqual(['val']);
    expect(cols2(it0 + 1)).toEqual(['val']); // None == None → leak dropped too (fly.rs:458)
    expect(cols2(res0 + 2)).toEqual(['date', 'position', 'points']); // not in the drop list's table
  });

  it('F→P stack is LIFO: the LAST FK parent is visited first (§4.4)', () => {
    const ctx = sampleContext(pdb, { ...sampleParamsFor(pdb, { ...task, kind: 'autocomplete', table: 'orders', targetCol: 'quantity', name: 'orders' }, 4), L: 100 });
    // o1 (#4) → parents c0 (#0), p1 (#9) pushed in FK order; p1 popped first
    expect(ctx.visited.slice(0, 3).map((v) => v.node)).toEqual([4, 9, 0]);
  });

  it('deterministic for a fixed rngSeed; events can be disabled', () => {
    const p = sampleParamsFor(pdb, task, 17, { L: 24, w: 1, rngSeed: 7 });
    const a = sampleContext(pdb, p);
    const b = sampleContext(pdb, p);
    expect(b).toEqual(a);
    const c = sampleContext(pdb, p, { recordEvents: false });
    expect(c.tokens).toEqual(a.tokens);
    expect(c.events).toEqual([]);
  });

  it('a row with more than 5 parents throws only when it emits a cell (fly.rs:464)', () => {
    const p = table({ name: 'p', pkey: 'id', columns: [{ name: 'id', dtype: 'int' }, { name: 'v', dtype: 'float' }] }, [{ id: 0, v: 1 }]);
    const fks = ['a', 'b', 'c', 'd', 'e', 'f'];
    const mk = (withCell: boolean) =>
      table(
        {
          name: 'k',
          pkey: 'id',
          fkeys: Object.fromEntries(fks.map((f) => [f, 'p'])),
          columns: [{ name: 'id', dtype: 'int' }, ...fks.map((f) => ({ name: f, dtype: 'int' as DType })), ...(withCell ? [{ name: 'w', dtype: 'float' as DType }] : [])],
        },
        [{ id: 0, a: 0, b: 0, c: 0, d: 0, e: 0, f: 0, w: 1 }],
      );
    const params = (pp: PreprocessedDB): SampleParams => ({ seedNode: 0, L: 50, w: 5, targetColNameIdx: pp.columnIndex['v of p'], dropColNameIdxs: [], rngSeed: 0 });
    const ok = preprocess({ name: 'x', title: '', description: '', tables: [mk(false), p], tasks: [], minTimestamp: 0, maxTimestamp: 0 });
    expect(() => sampleContext(ok, params(ok))).not.toThrow(); // zero-cell row with 6 FKs: no panic
    const bad = preprocess({ name: 'x', title: '', description: '', tables: [mk(true), p], tasks: [], minTimestamp: 0, maxTimestamp: 0 });
    expect(() => sampleContext(bad, params(bad))).toThrow(/MAX_F2P_NBRS/);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Adversarial pins of individual fly.rs decisions (each would catch a plausible mis-port)
// ─────────────────────────────────────────────────────────────────────────────

describe('sampler: adversarial pins of fly.rs semantics', () => {
  const raw = toyDb();
  const pdb = preprocess(raw);
  const churn = raw.tasks[0];
  const auto = (table: string, targetCol: string): TaskDef => ({
    name: table, id: `toy/${table}.${targetCol}`, dbName: 'toy', kind: 'autocomplete', targetType: 'regression',
    targetCol, columnsToDrop: [], description: '', table,
  });
  const seeds = Array.from({ length: 60 }, (_, s) => s);

  it('F→P parents are never time-filtered; P→F children with null ts always pass (fly.rs:406-408, 421-424)', () => {
    const cust = table({ name: 'cust', pkey: 'id', columns: [{ name: 'id', dtype: 'int' }, { name: 'nm', dtype: 'str' }] }, [{ id: 0, nm: 'A' }]);
    const sessions = table(
      { name: 'sessions', pkey: 'id', fkeys: { cust: 'cust' }, timeCol: 'started', columns: [{ name: 'id', dtype: 'int' }, { name: 'cust', dtype: 'int' }, { name: 'started', dtype: 'datetime' }] },
      [{ id: 0, cust: 0, started: D('2024-02-01') }], // parent is in the FUTURE of the seed click
    );
    const clicks = table(
      { name: 'clicks', pkey: 'id', fkeys: { session: 'sessions' }, timeCol: 'at', columns: [{ name: 'id', dtype: 'int' }, { name: 'session', dtype: 'int' }, { name: 'at', dtype: 'datetime' }, { name: 'dur', dtype: 'float' }] },
      [
        { id: 0, session: 0, at: D('2024-01-05'), dur: 1 }, // seed
        { id: 1, session: 0, at: D('2024-03-01'), dur: 2 }, // future sibling → skipped
        { id: 2, session: 0, at: null, dur: 3 }, // null time → ts null → always passes
      ],
    );
    const r: RawDatabase = { name: 'x', title: '', description: '', tables: [cust, sessions, clicks], tasks: [], minTimestamp: 0, maxTimestamp: 0 };
    const p = preprocess(r);
    const c0 = p.tableInfo['clicks:Db'].nodeIdxOffset;
    const s0 = p.tableInfo['sessions:Db'].nodeIdxOffset;
    const cu = p.tableInfo['cust:Db'].nodeIdxOffset;
    expect(p.nodes[s0].ts! > p.nodes[c0].ts!).toBe(true);
    const ctx = sampleContext(p, sampleParamsFor(p, auto('clicks', 'dur'), c0, { L: 100, w: 10 }));
    const vis = ctx.visited.map((v) => v.node);
    expect(vis.slice(0, 3)).toEqual([c0, s0, cu]); // F→P chain followed first, despite s0.ts > seed.ts
    expect(ctx.tokens.some((t) => t.nodeIdx === s0 && t.colName === 'started')).toBe(true);
    expect(vis).toContain(c0 + 2); // null-ts child kept
    expect(vis).not.toContain(c0 + 1); // future child dropped
    expect(ctx.events).toContainEqual({ kind: 'child-future', from: s0, node: c0 + 1, childTs: p.nodes[c0 + 1].ts, seedTs: p.nodes[c0].ts });
  });

  it('a seed without a timestamp disables the temporal filter for the whole context (§4.6)', () => {
    // products have no time column → seed ts null → o3 (2024-03-15) is reachable like any other order
    const ctx = sampleContext(pdb, sampleParamsFor(pdb, auto('products', 'price'), 8, { L: 1000, w: 100 }));
    expect(pdb.nodes[8].ts).toBeNull();
    const vis = new Set(ctx.visited.map((v) => v.node));
    for (const o of [3, 5, 6]) expect(vis.has(o)).toBe(true); // o0, o2, o3: every child of p0
    expect(ctx.events.some((e) => e.kind === 'child-future')).toBe(false);
  });

  it('the width bound is applied AFTER the time filter (fly.rs:422 before :439; not the 7f891d1 order)', () => {
    // c0's DB children: o0 (01-10), o1 (02-05), o3 (03-15, future). With w = 2 only o0, o1 survive the
    // filter, so they are never subsampled. Subsampling first would lose one of them 1/3 of the time.
    for (const s of seeds) {
      const ctx = sampleContext(pdb, { ...sampleParamsFor(pdb, churn, 16, { L: 1000, w: 2 }), rngSeed: s });
      expect(ctx.events.some((e) => e.kind === 'db-children-subsampled' && e.from === 0)).toBe(false);
      const kids = ctx.events.filter((e) => e.kind === 'push-db-child' && e.from === 0).map((e) => (e as { node: number }).node);
      expect(kids).toEqual([3, 4]);
    }
  });

  it('already-visited DB children count toward w (no visited test before the sample, fly.rs:415-443)', () => {
    // seed o1 (#4): LIFO pops p1 (#9) first; p1's DB children are [o1 (visited!), o4]. With w = 1 the
    // sample keeps o1 about half of the time, and then o4 (and c2 behind it) is never reached.
    let keptVisited = 0;
    let keptNew = 0;
    for (const s of seeds) {
      const ctx = sampleContext(pdb, { ...sampleParamsFor(pdb, auto('orders', 'quantity'), 4, { L: 1000, w: 1 }), rngSeed: s });
      const ev = ctx.events.find((e) => e.kind === 'db-children-subsampled' && e.from === 9);
      expect(ev).toMatchObject({ candidates: [4, 7] });
      const kept = (ev as { kept: number[] }).kept;
      const vis = new Set(ctx.visited.map((v) => v.node));
      if (kept[0] === 4) {
        keptVisited++;
        expect(vis.has(7)).toBe(false);
        expect(ctx.events).toContainEqual({ kind: 'skip-visited', node: 4, depth: 2 });
      } else {
        keptNew++;
        expect(vis.has(7)).toBe(true);
      }
    }
    expect(keptVisited).toBeGreaterThan(10);
    expect(keptNew).toBeGreaterThan(10);
  });

  it('w = 0 removes every DB child but never a task row of the seed task (fly.rs:426-452)', () => {
    const ctx = sampleContext(pdb, sampleParamsFor(pdb, churn, 16, { L: 1000, w: 0 }));
    expect(new Set(ctx.visited.map((v) => v.node))).toEqual(new Set([16, 0, 12, 14]));
    expect(ctx.events).toContainEqual({ kind: 'db-children-subsampled', from: 0, candidates: [3, 4], kept: [], w: 0 });
  });

  it('stops the moment L tokens exist, even with a non-empty frontier (fly.rs:503-509)', () => {
    // L = 5 = seed row (2 cells) + c0 (3 cells): bucket 2 still holds #12, #14, #16, o0, o1
    const ctx = sampleContext(pdb, sampleParamsFor(pdb, churn, 16, { L: 5, w: 16 }));
    expect(ctx.tokens.map((t) => t.nodeIdx)).toEqual([16, 16, 0, 0, 0]);
    expect(ctx.events.filter((e) => e.kind === 'pop')).toHaveLength(2);
    expect(ctx.events.at(-1)).toEqual({ kind: 'truncated', node: 0, pos: 4 });
    expect(ctx.events.some((e) => e.kind === 'frontier-empty' || e.kind === 'pad')).toBe(false);
  });

  it('same-horizon rows: a Val seed sees another entity\'s Val row at the SAME timestamp, label unmasked (§2.5)', () => {
    for (const s of seeds.slice(0, 10)) {
      const ctx = sampleContext(pdb, { ...sampleParamsFor(pdb, churn, 16, { L: 1000, w: 16 }), rngSeed: s });
      const lab17 = ctx.tokens.find((t) => t.nodeIdx === 17 && t.colName === 'churn');
      expect(lab17).toBeDefined();
      expect(lab17!.mask).toBe(false);
      expect(lab17!.booleanValue).toBeLessThan(0); // c1 Val label = false, visible
      expect(ctx.tokens.filter((t) => t.mask)).toHaveLength(1);
      expect(ctx.visited.some((v) => v.node === 10 || v.node === 11)).toBe(false); // Test rows are future
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Randomized small DBs
// ─────────────────────────────────────────────────────────────────────────────

/** A random RelBench-style DB: PK = row position, ≤3 FKs (self-FKs allowed), nulls, NaNs, time columns. */
function randomDb(seed: number): RawDatabase {
  const rng = new Rng(seed * 7919 + 13);
  const nTables = 2 + rng.int(4);
  const t0 = D('2024-01-01');
  const tables: RawTable[] = [];
  const names = Array.from({ length: nTables }, (_, i) => `t${i}`);
  const sizes = names.map(() => 1 + rng.int(10));
  for (let i = 0; i < nTables; i++) {
    const cols: { name: string; dtype: DType }[] = [{ name: 'id', dtype: 'int' }];
    const fkeys: Record<string, string> = {};
    const nFk = i === 0 ? 0 : rng.int(4);
    for (let f = 0; f < nFk; f++) {
      const parent = names[rng.int(i + 1)]; // earlier table or itself
      fkeys[`fk${f}`] = parent;
      cols.push({ name: `fk${f}`, dtype: 'int' });
    }
    const hasTime = rng.bool(0.6);
    if (hasTime) cols.push({ name: 'time', dtype: 'datetime' });
    const kinds: DType[] = ['int', 'float', 'bool', 'str', 'datetime'];
    const nFeat = rng.int(4);
    for (let c = 0; c < nFeat; c++) cols.push({ name: `c${c}`, dtype: kinds[rng.int(kinds.length)] });
    const rows = Array.from({ length: sizes[i] }, (_, r) => {
      const row: Record<string, number | boolean | string | null> = { id: r };
      for (const col of cols.slice(1)) {
        if (col.name.startsWith('fk')) {
          row[col.name] = rng.bool(0.15) ? null : rng.int(sizes[names.indexOf(fkeys[col.name])]);
          continue;
        }
        if (rng.bool(0.12)) {
          row[col.name] = null;
          continue;
        }
        switch (col.dtype) {
          case 'int': row[col.name] = rng.int(5); break;
          case 'float': row[col.name] = rng.bool(0.1) ? NaN : rng.normal(); break;
          case 'bool': row[col.name] = rng.bool(0.5); break;
          case 'str': row[col.name] = ['a', 'b', 'c'][rng.int(3)]; break;
          case 'datetime': row[col.name] = t0 + rng.int(120) * DAY + (rng.bool(0.3) ? 500 : 0); break;
        }
      }
      return row;
    });
    tables.push({ schema: { name: names[i], pkey: 'id', fkeys, timeCol: hasTime ? 'time' : null, columns: cols }, rows });
  }
  // two forecast tasks on random entity tables + one autocomplete task
  const tasks: TaskDef[] = [];
  for (const tn of ['task-a', 'task-b']) {
    const ent = rng.int(nTables);
    const split = (base: number, n: number, nullable: boolean) =>
      table(
        {
          name: tn,
          fkeys: { ent: names[ent] },
          timeCol: 'ts',
          columns: [{ name: 'ts', dtype: 'datetime' }, { name: 'ent', dtype: 'int' }, { name: 'y', dtype: 'bool' }],
        },
        Array.from({ length: n }, (_, k) => ({
          ts: t0 + (base + (rng.int(3) - 1) * 10) * DAY,
          ent: rng.int(sizes[ent]),
          y: nullable && rng.bool(0.1) ? null : k % 2 === 0,
        })),
      );
    tasks.push(forecastTask(tn, 'y', { Train: split(40, 2 + rng.int(8), true), Val: split(70, 1 + rng.int(4), true), Test: split(100, 1 + rng.int(4), false) }));
  }
  const at = rng.int(nTables);
  const feat = tables[at].schema.columns.filter((c) => !c.name.startsWith('fk') && c.name !== 'id');
  if (feat.length > 0) {
    tasks.push({
      name: names[at], id: `rand/${names[at]}`, dbName: 'rand', kind: 'autocomplete', targetType: 'regression',
      targetCol: feat[0].name, columnsToDrop: feat.slice(1, 3).map((c) => c.name), description: '', table: names[at],
    });
  }
  return { name: `rand${seed}`, title: '', description: '', tables, tasks, minTimestamp: t0, maxTimestamp: t0 + 150 * DAY };
}

/** Verbatim transcription of the spec §4.3 pseudo-code (independent of sampler.ts), same RNG calls. */
function referenceSample(pdb: PreprocessedDB, p: SampleParams): { nodes: number[]; cols: number[]; visited: number[] } {
  const nodes = pdb.nodes;
  const rng = new Rng(p.rngSeed);
  const S = nodes[p.seedNode];
  const visited = new Array(nodes.length).fill(false);
  const f2pStack: [number, number][] = [[0, p.seedNode]];
  const buckets: number[][] = [];
  const out: { node: number; col: number }[] = [];
  const order: number[] = [];
  const push = (d: number, n: number) => {
    while (buckets.length <= d) buckets.push([]);
    buckets[d].push(n);
  };
  for (;;) {
    let depth: number, n: number;
    if (f2pStack.length > 0) {
      [depth, n] = f2pStack.pop()!;
    } else {
      const d = buckets.findIndex((b) => b.length > 0);
      if (d === -1) break;
      const b = buckets[d];
      const r = rng.int(b.length);
      [b[r], b[b.length - 1]] = [b[b.length - 1], b[r]];
      n = b.pop()!;
      depth = d;
    }
    if (visited[n]) continue;
    visited[n] = true;
    order.push(n);
    const node = nodes[n];
    for (const e of node.f2pEdges) f2pStack.push([depth + 1, e.node]);
    const dbKids: number[] = [];
    for (const e of pdb.p2fAdj[n]) {
      if (e.tableType !== 'Db' && e.tableNameIdx !== S.tableNameIdx) continue;
      if (e.ts !== null && S.ts !== null && e.ts > S.ts) continue;
      if (e.tableType === 'Db') {
        dbKids.push(e.node);
        continue;
      }
      push(depth + 1, e.node);
    }
    const keep = dbKids.length > p.w ? rng.sampleIndices(dbKids.length, p.w) : [...dbKids.keys()];
    for (const i of keep) push(depth + 1, dbKids[i]);
    for (const c of node.cells) {
      if (p.dropColNameIdxs.includes(c.colNameIdx) && (node.nodeIdx === p.seedNode || node.ts === S.ts)) continue;
      out.push({ node: n, col: c.colNameIdx });
      if (out.length >= p.L) return { nodes: out.map((o) => o.node), cols: out.map((o) => o.col), visited: order };
    }
  }
  return { nodes: out.map((o) => o.node), cols: out.map((o) => o.col), visited: order };
}


// Plain assertions: the randomized sweep checks ~10^6 conditions, too many for `expect`.
function check(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`invariant violated: ${msg}`);
}
function same(a: unknown, b: unknown, msg: string): void {
  if (!(Object.is(a, b) || JSON.stringify(a) === JSON.stringify(b))) {
    throw new Error(`invariant violated: ${msg}: ${JSON.stringify(a)} !== ${JSON.stringify(b)}`);
  }
}

/** Check every spec §12 sampling invariant on one context, by replaying its event trace. */
function checkInvariants(pdb: PreprocessedDB, p: SampleParams, ctx: SampledContext): void {
  const { tokens, events } = ctx;
  const seed = pdb.nodes[p.seedNode];
  const drop = new Set(p.dropColNameIdxs);
  const dropped = (n: number, col: number) => drop.has(col) && (n === p.seedNode || pdb.nodes[n].ts === seed.ts);
  const real = tokens.slice(0, ctx.numReal);

  // ── exactly L tokens, real tokens first, padding defaults (§4.11) ──
  same(tokens.length, p.L, 'exactly L tokens');
  tokens.forEach((t, i) => same(t.pos, i, 'pos'));
  for (const t of real) check(!t.isPadding, 'real tokens first');
  for (let i = ctx.numReal; i < p.L; i++) same(tokens[i], paddingToken(i), 'padding defaults');

  // ── identical to the verbatim spec §4.3 pseudo-code ──
  const ref = referenceSample(pdb, p);
  same(real.map((t) => t.nodeIdx), ref.nodes, 'token rows == spec pseudo-code');
  same(real.map((t) => t.colNameIdx), ref.cols, 'token columns == spec pseudo-code');
  same(ctx.visited.map((v) => v.node), ref.visited, 'visit order == spec pseudo-code');

  // ── the seed's cells come first (§4.12) ──
  const seedCells = seed.cells.filter((c) => !dropped(p.seedNode, c.colNameIdx));
  const k = Math.min(p.L, seedCells.length);
  for (let i = 0; i < k; i++) same(tokens[i].nodeIdx, p.seedNode, 'seed row first');
  for (let i = k; i < ctx.numReal; i++) check(tokens[i].nodeIdx !== p.seedNode, 'seed row only once');

  // ── exactly one target, masked; mask == isTarget (§4.10) ──
  const targetIdx = seedCells.findIndex((c) => c.colNameIdx === p.targetColNameIdx);
  const targets = tokens.filter((t) => t.isTarget);
  if (targetIdx >= 0 && targetIdx < p.L) {
    same(targets.length, 1, 'exactly one target');
    same(targets[0].pos, targetIdx, 'target position');
    same(ctx.targetPos, targetIdx, 'targetPos');
    check(targets[0].nodeIdx === p.seedNode && targets[0].colNameIdx === p.targetColNameIdx && targets[0].mask, 'target is the masked seed cell');
  } else {
    same(targets.length, 0, 'no target when it does not fit');
    same(ctx.targetPos, -1, 'targetPos -1');
  }
  for (const t of tokens) same(t.mask, t.isTarget, 'mask == isTarget');

  // ── token fields equal the node/cell they came from (§5.1) ──
  const depthOf = new Map(ctx.visited.map((v) => [v.node, v.depth]));
  for (const t of real) {
    const node = pdb.nodes[t.nodeIdx];
    const cell = node.cells.find((c) => c.colNameIdx === t.colNameIdx)!;
    same(t.f2pNbrIdxs, [...node.f2pNbrIdxs, -1, -1, -1, -1, -1].slice(0, MAX_F2P_NBRS), 'f2p layout');
    same(t.tableNameIdx, node.tableNameIdx, 'tableNameIdx');
    same(t.sem, cell.sem, 'sem');
    check(Object.is(t.numberValue, cell.number) && Object.is(t.booleanValue, cell.boolean) && Object.is(t.datetimeValue, cell.datetime), 'values');
    same(t.textIdx, cell.textIdx, 'textIdx');
    same(t.classValueIdx, cell.sem === 1 ? cell.textIdx : -1, 'classValueIdx');
    same(t.isTaskNode, node.isTaskNode || t.colNameIdx === p.targetColNameIdx, 'isTaskNode (fly.rs:498-499)');
    same(t.depth, depthOf.get(t.nodeIdx), 'depth');
    same([t.tableName, t.tableType, t.rowPos, t.colName, t.raw], [node.tableName, node.tableType, node.rowPos, cell.colName, cell.raw], 'display');
  }

  // ── leakage drop rule: each visited row emits exactly its non-dropped cells, in order (last may be cut) ──
  const last = ctx.numReal === p.L ? tokens[p.L - 1].nodeIdx : -1;
  for (const v of ctx.visited) {
    const expected = pdb.nodes[v.node].cells.filter((c) => !dropped(v.node, c.colNameIdx)).map((c) => c.colNameIdx);
    const got = real.filter((t) => t.nodeIdx === v.node).map((t) => t.colNameIdx);
    same(v.node === last ? expected.slice(0, got.length) : expected, got, 'row cells minus leakage');
  }

  // ── replay the trace: LIFO stack priority, smallest bucket, filters, width bound ──
  const stack: number[] = [p.seedNode]; // initial entry (0, seed) (fly.rs:366)
  const buckets: number[][] = [];
  const seen = new Set<number>();
  let cur = -1;
  let curDepth = -1;
  let i = 0;
  const truncated = events.some((e) => e.kind === 'truncated');
  while (i < events.length) {
    const e = events[i++];
    if (e.kind === 'pop') {
      if (e.from === 'stack') {
        check(stack.length > 0, 'stack pop from non-empty stack');
        same(stack.pop(), e.node, 'LIFO stack');
      } else {
        same(stack.length, 0, 'F→P rows always before any P→F row');
        const d = buckets.findIndex((b) => b && b.length > 0);
        same(e.depth, d, 'smallest non-empty depth bucket');
        const b = buckets[d];
        same(e.bucketSize, b.length, 'bucket size');
        same(b[e.draw!], e.node, 'uniform swap-remove pick');
        [b[e.draw!], b[b.length - 1]] = [b[b.length - 1], b[e.draw!]];
        b.pop();
      }
      const next = events[i++];
      if (next.kind === 'skip-visited') {
        check(seen.has(e.node), 'skip only visited rows');
        continue;
      }
      same(next, { kind: 'visit', node: e.node, depth: e.depth }, 'visit after pop');
      check(!seen.has(e.node), 'visited once');
      seen.add(e.node);
      cur = e.node;
      curDepth = e.depth;
      const node = pdb.nodes[cur];
      for (const f of node.f2pEdges) {
        same(events[i++], { kind: 'push-parent', from: cur, node: f.node, depth: curDepth + 1 }, 'all parents pushed, FK order');
        stack.push(f.node);
      }
      const dbKids: number[] = [];
      for (const c of pdb.p2fAdj[cur]) {
        if (c.tableType !== 'Db' && c.tableNameIdx !== seed.tableNameIdx) {
          same(events[i++], { kind: 'child-other-task-table', from: cur, node: c.node }, 'other task table skipped');
        } else if (c.ts !== null && seed.ts !== null && c.ts > seed.ts) {
          same(events[i++], { kind: 'child-future', from: cur, node: c.node, childTs: c.ts, seedTs: seed.ts }, 'future child skipped');
        } else if (c.tableType === 'Db') {
          dbKids.push(c.node);
        } else {
          same(events[i++], { kind: 'push-task-child', from: cur, node: c.node, depth: curDepth + 1 }, 'task child never subsampled');
          (buckets[curDepth + 1] ??= []).push(c.node);
        }
      }
      let kept = dbKids;
      if (dbKids.length > p.w) {
        const s = events[i++];
        check(s.kind === 'db-children-subsampled', 'subsample event');
        if (s.kind !== 'db-children-subsampled') return;
        same(s.candidates, dbKids, 'subsample candidates = time-filtered DB children (visited included)');
        same(s.kept.length, p.w, 'exactly w DB children kept');
        const used = new Set<number>();
        for (const kNode of s.kept) {
          const at = dbKids.findIndex((x, j) => x === kNode && !used.has(j));
          check(at >= 0, 'kept ⊆ candidates, without replacement');
          used.add(at);
        }
        kept = s.kept;
      }
      for (const kNode of kept) {
        same(events[i++], { kind: 'push-db-child', from: cur, node: kNode, depth: curDepth + 1 }, 'DB child pushed');
        (buckets[curDepth + 1] ??= []).push(kNode);
      }
      continue;
    }
    if (e.kind === 'emit' || e.kind === 'drop-leakage-cell' || e.kind === 'truncated') {
      same(e.node, cur, 'cells belong to the current row');
      continue;
    }
    if (e.kind === 'frontier-empty') {
      same(stack.length, 0, 'empty stack at the end');
      check(buckets.every((b) => !b || b.length === 0), 'empty buckets at the end');
      same(e.tokens, ctx.numReal, 'frontier-empty token count');
      continue;
    }
    if (e.kind === 'pad') {
      same(e, { kind: 'pad', from: ctx.numReal, to: p.L - 1 }, 'pad event');
      continue;
    }
    throw new Error(`unexpected event ${JSON.stringify(e)}`);
  }

  // ── no P→F row in the future of the seed; other task tables never reached ──
  for (const v of ctx.visited) {
    const n = pdb.nodes[v.node];
    if (v.node !== p.seedNode && n.isTaskNode) same(n.tableNameIdx, seed.tableNameIdx, 'only the seed task table');
    if (v.node !== p.seedNode && seed.tableType === 'Db') check(!n.isTaskNode, 'autocomplete: no task rows');
  }
  for (const e of events) {
    if (e.kind === 'push-task-child' || e.kind === 'push-db-child') {
      const ts = pdb.nodes[e.node].ts;
      check(ts === null || seed.ts === null || ts <= seed.ts, 'no P→F child with ts > seed ts');
    }
  }
  const dbPushes = new Map<number, number>();
  for (const e of events) if (e.kind === 'push-db-child') dbPushes.set(e.from, (dbPushes.get(e.from) ?? 0) + 1);
  for (const c of dbPushes.values()) check(c <= p.w, '≤ w DB children per row');

  // ── when the frontier ran dry: every parent visited, every qualifying task child visited ──
  if (!truncated) {
    for (const v of ctx.visited) {
      for (const f of pdb.nodes[v.node].f2pNbrIdxs) check(seen.has(f), 'every parent visited');
      for (const c of pdb.p2fAdj[v.node]) {
        const qualifies = c.tableType !== 'Db' && c.tableNameIdx === seed.tableNameIdx && !(c.ts !== null && seed.ts !== null && c.ts > seed.ts);
        if (qualifies) check(seen.has(c.node), 'every qualifying task child visited');
      }
    }
  }
}

describe('sampler: invariants on randomized small DBs', () => {
  it('holds for 40 random DBs × all tasks × several seeds, L and w', () => {
    let contexts = 0;
    let truncations = 0;
    let subsamples = 0;
    let midRowCuts = 0;
    let exhausted = 0;
    for (let s = 0; s < 40; s++) {
      const raw = randomDb(s);
      const pdb = preprocess(raw);
      for (const task of raw.tasks) {
        for (const split of ['Train', 'Val', 'Test'] as Split[]) {
          const seeds = listSeeds(pdb, task, split);
          for (const seedNode of seeds.slice(0, 3)) {
            for (const [L, w] of [[6, 1], [17, 2], [40, 3], [400, 100]]) {
              const p = sampleParamsFor(pdb, task, seedNode, { L, w, rngSeed: s * 31 + seedNode });
              const ctx = sampleContext(pdb, p);
              checkInvariants(pdb, p, ctx);
              same(JSON.stringify(sampleContext(pdb, p)), JSON.stringify(ctx), 'deterministic for a fixed rngSeed');
              contexts++;
              const tr = ctx.events.find((e) => e.kind === 'truncated');
              if (tr) {
                truncations++;
                const emitted = ctx.tokens.filter((t) => t.nodeIdx === tr.node).length;
                if (emitted < pdb.nodes[tr.node].cells.length) midRowCuts++;
              } else exhausted++;
              if (ctx.events.some((e) => e.kind === 'db-children-subsampled')) subsamples++;
            }
          }
          if (task.kind === 'autocomplete') break; // split ignored (§2.4)
        }
      }
    }
    // the random DBs actually exercise the interesting branches
    expect(contexts).toBeGreaterThan(400);
    expect(truncations).toBeGreaterThan(50);
    expect(exhausted).toBeGreaterThan(50);
    expect(midRowCuts).toBeGreaterThan(10);
    expect(subsamples).toBeGreaterThan(50);
  }, 30_000);
});
