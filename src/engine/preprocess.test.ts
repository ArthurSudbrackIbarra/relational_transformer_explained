/**
 * Tests for preprocess.ts (mirror of pre.rs) on the spec §6.3 toy DB and on targeted edge cases.
 * Reference numbers: spec §6.3 "Statistics", recomputed with numpy (ddof=1 sample std for
 * columns, population std for the global datetime pair).
 */
import { describe, expect, it } from 'vitest';
import { columnPhrase, nodeIdxOf, nodeLabel, preprocess, toNodeTs } from './preprocess';
import { SemType, type RawDatabase, type RawTable, type TableSchema, type TaskDef } from './types';

const D = (s: string) => Date.parse(`${s}T00:00:00Z`); // epoch ms
const DAY = 86_400_000;

// ─────────────────────────────────────────────────────────────────────────────
// The spec §6.3 toy DB as a RawDatabase (DB tables deliberately listed out of order).
// ─────────────────────────────────────────────────────────────────────────────
function churnTable(rows: [string, number, boolean | null][]): RawTable {
  return {
    schema: {
      name: 'customer-churn',
      pkey: null,
      fkeys: { customer_id: 'customers' },
      timeCol: 'timestamp',
      columns: [
        { name: 'timestamp', dtype: 'datetime' },
        { name: 'customer_id', dtype: 'int' },
        { name: 'churn', dtype: 'bool' },
      ],
    },
    rows: rows.map(([t, c, y]) => ({ timestamp: D(t), customer_id: c, churn: y })),
  };
}

function toyDb(): RawDatabase {
  const products: RawTable = {
    schema: {
      name: 'products',
      pkey: 'product_id',
      fkeys: {},
      timeCol: null,
      columns: [
        { name: 'product_id', dtype: 'int' },
        { name: 'category', dtype: 'str' },
        { name: 'price', dtype: 'float' },
      ],
    },
    rows: [
      { product_id: 0, category: 'books', price: 12 },
      { product_id: 1, category: 'toys', price: 30 },
    ],
  };
  const customers: RawTable = {
    schema: {
      name: 'customers',
      pkey: 'customer_id',
      fkeys: {},
      timeCol: null,
      columns: [
        { name: 'customer_id', dtype: 'int' },
        { name: 'country', dtype: 'str' },
        { name: 'age', dtype: 'float' },
        { name: 'is_premium', dtype: 'bool' },
      ],
    },
    rows: [
      { customer_id: 0, country: 'BR', age: 34, is_premium: true },
      { customer_id: 1, country: 'US', age: 51, is_premium: false },
      { customer_id: 2, country: 'BR', age: null, is_premium: false },
    ],
  };
  const orders: RawTable = {
    schema: {
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
    rows: [
      { order_id: 0, customer_id: 0, product_id: 0, order_time: D('2024-01-10'), quantity: 1 },
      { order_id: 1, customer_id: 0, product_id: 1, order_time: D('2024-02-05'), quantity: 2 },
      { order_id: 2, customer_id: 1, product_id: 0, order_time: D('2024-02-20'), quantity: 1 },
      { order_id: 3, customer_id: 0, product_id: 0, order_time: D('2024-03-15'), quantity: 3 },
      { order_id: 4, customer_id: 2, product_id: 1, order_time: D('2024-01-25'), quantity: 1 },
    ],
  };
  const churn: TaskDef = {
    name: 'customer-churn',
    id: 'toy/customer-churn',
    dbName: 'toy',
    kind: 'forecast',
    targetType: 'binary',
    targetCol: 'churn',
    columnsToDrop: [],
    description: 'toy churn',
    entityTable: 'customers',
    entityCol: 'customer_id',
    timeCol: 'timestamp',
    timedeltaDays: 30,
    valTimestamp: D('2024-03-01'),
    testTimestamp: D('2024-04-01'),
    tables: {
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
    },
  };
  return {
    name: 'toy',
    title: 'Toy DB (spec §6.3)',
    description: '',
    tables: [products, orders, customers],
    tasks: [churn],
    minTimestamp: D('2024-01-01'),
    maxTimestamp: D('2024-04-01'),
  };
}

/** Minimal DB builder for edge cases. */
function table(schema: Partial<TableSchema> & Pick<TableSchema, 'name' | 'columns'>, rows: RawTable['rows']): RawTable {
  return { schema: { pkey: null, fkeys: {}, timeCol: null, ...schema }, rows };
}
function db(tables: RawTable[], tasks: TaskDef[] = []): RawDatabase {
  return { name: 'edge', title: 'edge', description: '', tables, tasks, minTimestamp: 0, maxTimestamp: 0 };
}
function forecast(name: string, tables: Partial<Record<'Train' | 'Val' | 'Test', RawTable>>): TaskDef {
  return {
    name,
    id: `edge/${name}`,
    dbName: 'edge',
    kind: 'forecast',
    targetType: 'binary',
    targetCol: 'y',
    columnsToDrop: [],
    description: '',
    tables: tables as TaskDef['tables'],
  };
}

describe('preprocess: toy DB of spec §6.3', () => {
  const pdb = preprocess(toyDb());
  const txt = (s: string) => pdb.textToIdx.get(s)!;

  it('global node indexing follows glob order (§1.5)', () => {
    expect(pdb.tableInfo).toEqual({
      'customers:Db': { nodeIdxOffset: 0, numNodes: 3 },
      'orders:Db': { nodeIdxOffset: 3, numNodes: 5 },
      'products:Db': { nodeIdxOffset: 8, numNodes: 2 },
      'customer-churn:Test': { nodeIdxOffset: 10, numNodes: 2 },
      'customer-churn:Train': { nodeIdxOffset: 12, numNodes: 4 },
      'customer-churn:Val': { nodeIdxOffset: 16, numNodes: 2 },
    });
    expect(pdb.nodes).toHaveLength(18);
    pdb.nodes.forEach((n, i) => expect(n.nodeIdx).toBe(i));
    expect(nodeLabel(pdb, 16)).toBe('customer-churn:Val[0]#16');
    expect(nodeLabel(pdb, 4)).toBe('orders[1]#4');
    expect(nodeIdxOf(pdb, 'customer-churn:Train', 2)).toBe(14);
    expect(pdb.nodes[14]).toMatchObject({ tableKey: 'customer-churn:Train', tableType: 'Train', rowPos: 2, isTaskNode: true });
    expect(pdb.nodes[9]).toMatchObject({ tableKey: 'products:Db', tableType: 'Db', rowPos: 1, isTaskNode: false });
  });

  it('task splits share table_name_idx and col_name_idx (§2.2)', () => {
    const idx = new Set([10, 12, 16].map((i) => pdb.nodes[i].tableNameIdx));
    expect(idx.size).toBe(1);
    expect(pdb.nodes[10].tableNameIdx).toBe(txt('customer-churn'));
    const churnCols = new Set([10, 13, 17].map((i) => pdb.nodes[i].cells[1].colNameIdx));
    expect([...churnCols]).toEqual([pdb.columnIndex['churn of customer-churn']]);
  });

  it('FK columns become f2p ids and edges, in FK-column order (§1.2)', () => {
    const f2p = pdb.nodes.map((n) => n.f2pNbrIdxs);
    expect(f2p.slice(0, 3)).toEqual([[], [], []]);
    expect(f2p.slice(3, 8)).toEqual([[0, 8], [0, 9], [1, 8], [0, 8], [2, 9]]);
    expect(f2p.slice(8, 10)).toEqual([[], []]);
    expect(f2p.slice(10)).toEqual([[0], [2], [0], [1], [0], [2], [0], [1]]);
    // F→P edge: parent idx, parent table name idx, always Db, parent's own ts (null: no time column).
    expect(pdb.nodes[7].f2pEdges).toEqual([
      { node: 2, tableNameIdx: txt('customers'), tableType: 'Db', ts: null },
      { node: 9, tableNameIdx: txt('products'), tableType: 'Db', ts: null },
    ]);
  });

  it('P→F edges carry the CHILD idx, table, table type and timestamp (§1.3)', () => {
    const sec = (s: string) => D(s) / 1000;
    const churnIdx = txt('customer-churn');
    const ordersIdx = txt('orders');
    const byNode = (edges: typeof pdb.p2fAdj[number]) => [...edges].sort((a, b) => a.node - b.node);
    expect(byNode(pdb.p2fAdj[0])).toEqual([
      { node: 3, tableNameIdx: ordersIdx, tableType: 'Db', ts: sec('2024-01-10') },
      { node: 4, tableNameIdx: ordersIdx, tableType: 'Db', ts: sec('2024-02-05') },
      { node: 6, tableNameIdx: ordersIdx, tableType: 'Db', ts: sec('2024-03-15') },
      { node: 10, tableNameIdx: churnIdx, tableType: 'Test', ts: sec('2024-04-01') },
      { node: 12, tableNameIdx: churnIdx, tableType: 'Train', ts: sec('2024-01-01') },
      { node: 14, tableNameIdx: churnIdx, tableType: 'Train', ts: sec('2024-02-01') },
      { node: 16, tableNameIdx: churnIdx, tableType: 'Val', ts: sec('2024-03-01') },
    ]);
    expect(byNode(pdb.p2fAdj[8]).map((e) => e.node)).toEqual([3, 5, 6]);
    expect(byNode(pdb.p2fAdj[9]).map((e) => e.node)).toEqual([4, 7]);
    for (const i of [3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17]) expect(pdb.p2fAdj[i]).toEqual([]);
  });

  it('node.ts: whole seconds, only for rows with ≥1 FK, a time column and a time value (§1.4)', () => {
    for (const i of [0, 1, 2, 8, 9]) expect(pdb.nodes[i].ts).toBeNull(); // no time column
    expect(pdb.nodes[4].ts).toBe(Math.trunc(D('2024-02-05') / 1000));
    expect(pdb.nodes[16].ts).toBe(Math.trunc(D('2024-03-01') / 1000));
    expect(pdb.nodes[10].ts).toBe(Math.trunc(D('2024-04-01') / 1000));
  });

  it('cells: PK/FK/null removed, stored column order, sem types by dtype (§3.2, §3.5)', () => {
    const cols = (i: number) => pdb.nodes[i].cells.map((c) => [c.colName, c.sem]);
    expect(cols(0)).toEqual([
      ['country', SemType.Text],
      ['age', SemType.Number],
      ['is_premium', SemType.Boolean],
    ]);
    expect(cols(2)).toEqual([
      ['country', SemType.Text],
      ['is_premium', SemType.Boolean],
    ]); // c2 has no age cell
    expect(cols(4)).toEqual([
      ['order_time', SemType.DateTime],
      ['quantity', SemType.Number],
    ]);
    expect(cols(9)).toEqual([
      ['category', SemType.Text],
      ['price', SemType.Number],
    ]);
    expect(cols(16)).toEqual([
      ['timestamp', SemType.DateTime],
      ['churn', SemType.Boolean],
    ]); // a task row has exactly 2 tokens (§2.1)
    const c0 = pdb.nodes[0].cells;
    expect(c0[0]).toMatchObject({ colNameIdx: pdb.columnIndex['country of customers'], textIdx: txt('BR'), number: 0, datetime: 0, boolean: 0, raw: 'BR' });
    expect(c0[1]).toMatchObject({ textIdx: 0, raw: 34 }); // non-text cells store text index 0 (pre.rs:578)
    expect(pdb.nodes[2].cells[0].textIdx).toBe(c0[0].textIdx); // identical strings share one index
  });

  it('number stats: sample std (ddof=1) per column (§3.3)', () => {
    const age = pdb.colStats['customers:Db'].age;
    expect(age.kind).toBe('number');
    expect(age.mean).toBe(42.5);
    expect(age.std).toBeCloseTo(12.020815280171307, 12);
    expect(age.count).toBe(2);
    expect(pdb.nodes[0].cells[1].number).toBeCloseTo((34 - 42.5) / 12.020815280171307, 6);
    expect(pdb.nodes[1].cells[1].number).toBeCloseTo(0.7071067811865476, 6);
    const q = pdb.colStats['orders:Db'].quantity;
    expect(q.mean).toBeCloseTo(1.6, 12);
    expect(q.std).toBeCloseTo(Math.sqrt(0.8), 12);
    const price = pdb.colStats['products:Db'].price;
    expect(price.mean).toBe(21);
    expect(price.std).toBeCloseTo(18 / Math.SQRT2, 12);
    // PK/FK/text/datetime columns have no "used" stats
    expect(Object.keys(pdb.colStats['orders:Db'])).toEqual(['quantity']);
    expect(Object.keys(pdb.colStats['customers:Db'])).toEqual(['age', 'is_premium']);
  });

  it('boolean stats: z-scored 0/1 with sample std, true > 0 > false (§3.3)', () => {
    const p = pdb.colStats['customers:Db'].is_premium;
    expect(p.kind).toBe('boolean');
    expect(p.mean).toBeCloseTo(1 / 3, 12);
    expect(p.std).toBeCloseTo(0.5773502691896258, 12);
    expect(pdb.nodes[0].cells[2].boolean).toBeCloseTo(1.1547, 4);
    expect(pdb.nodes[1].cells[2].boolean).toBeCloseTo(-0.5774, 4);
  });

  it('churn uses Train stats in every split: ±0.8660 (§3.3)', () => {
    for (const split of ['Train', 'Val', 'Test']) {
      const s = pdb.colStats[`customer-churn:${split}`].churn;
      expect(s.mean).toBe(0.5);
      expect(s.std).toBeCloseTo(0.5773502691896257, 12);
      expect(s.count).toBe(4); // the Train count
    }
    const z = (i: number) => pdb.nodes[i].cells[1].boolean;
    expect(z(13)).toBeCloseTo(0.866, 4); // Train c1 true
    expect(z(12)).toBeCloseTo(-0.866, 4); // Train c0 false
    expect(z(16)).toBeCloseTo(0.866, 4); // Val c0 true
    expect(z(17)).toBeCloseTo(-0.866, 4); // Val c1 false
    expect(z(10)).toBeCloseTo(-0.866, 4); // Test c0 false
    expect(z(11)).toBeCloseTo(0.866, 4); // Test c2 true
  });

  it('datetime: ONE global pair over all 13 datetime cells, population std ≈ 30.08 days (§3.3)', () => {
    const { mean, std, count } = pdb.datetimeStats;
    expect(count).toBe(13);
    expect(mean).toBeCloseTo(1707762461538.4614, 2); // 2024-02-12T18:27:41.538Z
    expect(new Date(mean).toISOString().slice(0, 16)).toBe('2024-02-12T18:27');
    // pre.rs:376 uses sqrt(Σt²/n − μ²); its cancellation error is ~1e-11 relative here.
    expect(Math.abs(std / 2598892610.779996 - 1)).toBeLessThan(1e-9);
    expect(std / DAY).toBeCloseTo(30.0798, 4);
    // the same pair z-scores DB time columns and task time columns
    expect(pdb.nodes[16].cells[0].datetime).toBeCloseTo(0.5728356975441801, 6); // 2024-03-01
    expect(pdb.nodes[3].cells[0].datetime).toBeCloseTo(-1.1226556750976173, 6); // 2024-01-10
    expect(pdb.nodes[10].cells[0].datetime).toBeCloseTo(1.6034284926793902, 6); // 2024-04-01
  });

  it('values are stored as float32 like pre.rs (`as f32`)', () => {
    for (const n of pdb.nodes)
      for (const c of n.cells) {
        expect(Math.fround(c.number)).toBe(c.number);
        expect(Math.fround(c.datetime)).toBe(c.datetime);
        expect(Math.fround(c.boolean)).toBe(c.boolean);
      }
  });

  // NB: pre.rs walks its table HashMap in hash order (pre.rs:415), so the official indices are an
  // arbitrary permutation of these; within ONE table the order below (name, then per column: phrase,
  // parent names / string values) is exactly pre.rs's. Only equality of indices is ever used.
  it('text dictionary: table names, every column phrase incl. PK/FK, string values; pre.rs per-table order, tables in glob order', () => {
    expect(pdb.text).toEqual([
      'customers',
      'customer_id of customers',
      'country of customers',
      'BR',
      'US',
      'age of customers',
      'is_premium of customers',
      'orders',
      'order_id of orders',
      'customer_id of orders',
      'product_id of orders',
      'products', // interned by the FK loop (pre.rs:495-498) before the products table is reached
      'order_time of orders',
      'quantity of orders',
      'product_id of products',
      'category of products',
      'books',
      'toys',
      'price of products',
      'customer-churn',
      'timestamp of customer-churn',
      'customer_id of customer-churn',
      'churn of customer-churn',
    ]);
    pdb.text.forEach((s, i) => expect(pdb.textToIdx.get(s)).toBe(i));
    const phrases = pdb.text.filter((s) => s.includes(' of '));
    expect(Object.keys(pdb.columnIndex).sort()).toEqual([...phrases].sort());
    for (const p of phrases) expect(pdb.columnIndex[p]).toBe(txt(p));
    expect(columnPhrase('churn', 'customer-churn')).toBe('churn of customer-churn');
  });
});

describe('preprocess: edge cases mirrored from pre.rs', () => {
  const idCol = { name: 'id', dtype: 'int' as const };

  it('node.ts rules (§1.4): time column without FK → null; all-null FKs → null; null time → null; parent ts on F→P edge', () => {
    const parents = table(
      {
        name: 'events',
        pkey: 'id',
        timeCol: 'at',
        columns: [idCol, { name: 'at', dtype: 'datetime' }, { name: 'v', dtype: 'float' }],
      },
      [
        { id: 0, at: D('2024-05-01'), v: 1 },
        { id: 1, at: null, v: 2 },
      ],
    );
    const kids = table(
      {
        name: 'logs',
        pkey: 'id',
        fkeys: { ev: 'events', ev2: 'events' },
        timeCol: 'at',
        columns: [idCol, { name: 'ev', dtype: 'int' }, { name: 'ev2', dtype: 'int' }, { name: 'at', dtype: 'datetime' }, { name: 'w', dtype: 'float' }],
      },
      [
        { id: 0, ev: 0, ev2: null, at: D('2024-05-02'), w: 1 }, // FK + time → ts
        { id: 1, ev: null, ev2: null, at: D('2024-05-03'), w: 2 }, // no non-null FK → null
        { id: 2, ev: 1, ev2: 0, at: null, w: 3 }, // null time → null
      ],
    );
    const pdb = preprocess(db([parents, kids]));
    const ev = pdb.tableInfo['events:Db'].nodeIdxOffset;
    const lg = pdb.tableInfo['logs:Db'].nodeIdxOffset;
    expect(pdb.nodes[ev].ts).toBeNull(); // has a time column but no FK
    expect(pdb.nodes[lg].ts).toBe(D('2024-05-02') / 1000);
    expect(pdb.nodes[lg + 1].ts).toBeNull();
    expect(pdb.nodes[lg + 1].f2pNbrIdxs).toEqual([]);
    expect(pdb.nodes[lg + 2].ts).toBeNull();
    expect(pdb.nodes[lg + 2].f2pNbrIdxs).toEqual([ev + 1, ev]); // FK-column order, null skipped
    // F→P edges carry the parent's raw time (even though the parent node itself has ts = null)
    expect(pdb.nodes[lg + 2].f2pEdges.map((e) => e.ts)).toEqual([null, D('2024-05-01') / 1000]);
    // the null time cell produces no cell; the datetime stats skip it
    expect(pdb.nodes[lg + 2].cells.map((c) => c.colName)).toEqual(['w']);
    expect(pdb.datetimeStats.count).toBe(3);
  });

  it('FK columns are walked in DataFrame COLUMN order, not in fkeys-object key order (pre.rs:429, 444)', () => {
    const p = table({ name: 'p', pkey: 'id', columns: [idCol, { name: 'v', dtype: 'float' }] }, [{ id: 0, v: 1 }, { id: 1, v: 2 }]);
    const q = table({ name: 'q', pkey: 'id', columns: [idCol, { name: 'v', dtype: 'float' }] }, [{ id: 0, v: 3 }]);
    // fkeys lists b first, but column a comes first in the DataFrame
    const k = table(
      { name: 'k', pkey: 'id', fkeys: { b: 'p', a: 'q' }, columns: [idCol, { name: 'a', dtype: 'int' }, { name: 'b', dtype: 'int' }, { name: 'w', dtype: 'float' }] },
      [{ id: 0, a: 0, b: 1, w: 5 }],
    );
    const pdb = preprocess(db([p, q, k]));
    const kn = pdb.nodes[pdb.tableInfo['k:Db'].nodeIdxOffset];
    expect(kn.f2pNbrIdxs).toEqual([pdb.tableInfo['q:Db'].nodeIdxOffset, pdb.tableInfo['p:Db'].nodeIdxOffset + 1]);
    expect(kn.f2pEdges.map((e) => e.node)).toEqual(kn.f2pNbrIdxs);
  });

  it('a column that is both PK and FK is treated as the PK: no edge, no cell (pre.rs:439 is tested before :444)', () => {
    const base = table({ name: 'base', pkey: 'id', columns: [idCol, { name: 'v', dtype: 'float' }] }, [{ id: 0, v: 1 }]);
    const ext = table({ name: 'ext', pkey: 'id', fkeys: { id: 'base' }, columns: [idCol, { name: 'x', dtype: 'float' }] }, [{ id: 0, x: 2 }]);
    const pdb = preprocess(db([base, ext]));
    const e = pdb.nodes[pdb.tableInfo['ext:Db'].nodeIdxOffset];
    expect(e.f2pNbrIdxs).toEqual([]);
    expect(pdb.p2fAdj[pdb.tableInfo['base:Db'].nodeIdxOffset]).toEqual([]);
    expect(e.cells.map((c) => c.colName)).toEqual(['x']);
  });

  it('two FKs to the same parent row: duplicate f2p entry and two P→F edges (pre.rs:479, 529)', () => {
    const u = table({ name: 'u', pkey: 'id', columns: [idCol, { name: 'v', dtype: 'float' }] }, [{ id: 0, v: 1 }]);
    const m = table(
      { name: 'm', pkey: 'id', fkeys: { from: 'u', to: 'u' }, timeCol: 'at', columns: [idCol, { name: 'from', dtype: 'int' }, { name: 'to', dtype: 'int' }, { name: 'at', dtype: 'datetime' }] },
      [{ id: 0, from: 0, to: 0, at: D('2024-01-02') }],
    );
    const pdb = preprocess(db([m, u]));
    const mi = pdb.tableInfo['m:Db'].nodeIdxOffset;
    const ui = pdb.tableInfo['u:Db'].nodeIdxOffset;
    expect(pdb.nodes[mi].f2pNbrIdxs).toEqual([ui, ui]);
    expect(pdb.p2fAdj[ui].map((e) => e.node)).toEqual([mi, mi]);
  });

  it('DB-table stats use ALL rows, including rows after the val/test timestamps (no cutoff, spec §3.4)', () => {
    const ent = table({ name: 'e', pkey: 'id', columns: [idCol] }, [{ id: 0 }]);
    const ev = table(
      { name: 'ev', pkey: 'id', fkeys: { e: 'e' }, timeCol: 'at', columns: [idCol, { name: 'e', dtype: 'int' }, { name: 'at', dtype: 'datetime' }, { name: 'x', dtype: 'float' }] },
      [
        { id: 0, e: 0, at: D('2024-01-01'), x: 0 },
        { id: 1, e: 0, at: D('2030-01-01'), x: 100 }, // far beyond any val/test timestamp
      ],
    );
    const pdb = preprocess(db([ent, ev]));
    expect(pdb.colStats['ev:Db'].x).toMatchObject({ mean: 50, count: 2 });
  });

  it('toNodeTs mirrors `(ns / 1e9) as i32`: truncation toward zero and i32 wrap', () => {
    expect(toNodeTs(1500)).toBe(1);
    expect(toNodeTs(-1500)).toBe(-1);
    expect(toNodeTs(null)).toBeNull();
    expect(toNodeTs(NaN)).toBeNull();
    const max = Date.parse('2038-01-19T03:14:07Z');
    expect(toNodeTs(max)).toBe(2 ** 31 - 1);
    expect(toNodeTs(max + 1000)).toBe(-(2 ** 31)); // i32 overflow wraps silently (§1.4)
    expect(toNodeTs(D('2040-01-01'))).toBe(D('2040-01-01') / 1000 - 2 ** 32); // wraps negative
  });

  it('number guards: n ≤ 1 → σ=1, σ=0 → σ=1, empty → μ=0; NaN dropped and produces no cell', () => {
    const t = table(
      {
        name: 't',
        pkey: 'id',
        columns: [idCol, { name: 'one', dtype: 'float' }, { name: 'const', dtype: 'int' }, { name: 'empty', dtype: 'float' }, { name: 'nan', dtype: 'float' }],
      },
      [
        { id: 0, one: 5, const: 7, empty: null, nan: NaN },
        { id: 1, one: null, const: 7, empty: null, nan: 2 },
        { id: 2, one: null, const: 7, empty: NaN, nan: 4 },
      ],
    );
    const pdb = preprocess(db([t]));
    const s = pdb.colStats['t:Db'];
    expect(s.one).toMatchObject({ mean: 5, std: 1, count: 1 });
    expect(s.const).toMatchObject({ mean: 7, std: 1, count: 3 });
    expect(s.empty).toMatchObject({ mean: 0, std: 1, count: 0 });
    expect(s.nan).toMatchObject({ mean: 3, count: 2 });
    expect(s.nan.std).toBeCloseTo(Math.SQRT2, 12);
    expect(pdb.nodes[0].cells.map((c) => c.colName)).toEqual(['one', 'const']);
    expect(pdb.nodes[0].cells[0].number).toBe(0);
    expect(pdb.nodes[2].cells.map((c) => c.colName)).toEqual(['const', 'nan']);
  });

  it('boolean: NO zero guard — constant or single-value columns give NaN; Val reusing a constant Train column gives ±inf', () => {
    const t = table(
      { name: 't', pkey: 'id', columns: [idCol, { name: 'c', dtype: 'bool' }, { name: 's', dtype: 'bool' }] },
      [
        { id: 0, c: true, s: true },
        { id: 1, c: true, s: null },
      ],
    );
    const ent = table({ name: 'e', pkey: 'id', columns: [idCol, { name: 'x', dtype: 'float' }] }, [{ id: 0, x: 1 }]);
    const split = (ys: boolean[]) =>
      table(
        {
          name: 'tsk',
          fkeys: { e: 'e' },
          timeCol: 'ts',
          columns: [{ name: 'ts', dtype: 'datetime' }, { name: 'e', dtype: 'int' }, { name: 'y', dtype: 'bool' }],
        },
        ys.map((y, i) => ({ ts: D('2024-01-01') + i * DAY, e: 0, y })),
      );
    const pdb = preprocess(db([t, ent], [forecast('tsk', { Train: split([false, false]), Val: split([true]) })]));
    expect(pdb.colStats['t:Db'].c).toMatchObject({ mean: 1, std: 0 });
    expect(pdb.colStats['t:Db'].s).toMatchObject({ mean: 1, std: 0, count: 1 }); // std(1) undefined → 0
    const t0 = pdb.nodes[pdb.tableInfo['t:Db'].nodeIdxOffset];
    expect(t0.cells[0].boolean).toBeNaN();
    expect(t0.cells[1].boolean).toBeNaN();
    const val0 = pdb.nodes[pdb.tableInfo['tsk:Val'].nodeIdxOffset];
    expect(val0.cells[1].boolean).toBe(Infinity);
  });

  it('Val/Test reuse Train stats even when their own distribution differs', () => {
    const ent = table({ name: 'e', pkey: 'id', columns: [idCol] }, [{ id: 0 }, { id: 1 }]);
    const split = (ys: boolean[]) =>
      table(
        { name: 'tsk', fkeys: { e: 'e' }, timeCol: 'ts', columns: [{ name: 'ts', dtype: 'datetime' }, { name: 'e', dtype: 'int' }, { name: 'y', dtype: 'bool' }] },
        ys.map((y, i) => ({ ts: D('2024-01-01') + i * DAY, e: i % 2, y })),
      );
    const pdb = preprocess(
      db([ent], [forecast('tsk', { Train: split([true, false, false, false]), Val: split([true, true]), Test: split([false]) })]),
    );
    const sigma = Math.sqrt((0.75 ** 2 + 3 * 0.25 ** 2) / 3);
    for (const k of ['tsk:Train', 'tsk:Val', 'tsk:Test']) {
      expect(pdb.colStats[k].y.mean).toBe(0.25);
      expect(pdb.colStats[k].y.std).toBeCloseTo(sigma, 12);
    }
    const val0 = pdb.nodes[pdb.tableInfo['tsk:Val'].nodeIdxOffset];
    expect(val0.cells[1].boolean).toBeCloseTo(0.75 / sigma, 6); // not 0/0 from Val's own stats
  });

  it('table order is PathBuf order: "<name>.parquet" byte-wise for DB files, task dir names then test<train<val', () => {
    const mk = (name: string) => table({ name, pkey: 'id', columns: [idCol, { name: 'v', dtype: 'float' }] }, [{ id: 0, v: 1 }]);
    const tsk = (fk: string) =>
      table({ name: 'x', fkeys: { e: fk }, timeCol: null, columns: [{ name: 'e', dtype: 'int' }, { name: 'y', dtype: 'bool' }] }, [{ e: 0, y: true }]);
    // "a-b.parquet" < "a.parquet" because '-' (0x2d) < '.' (0x2e); "B" < "a" (uppercase first)
    const pdb = preprocess(db([mk('a'), mk('a-b'), mk('B'), mk('ab')], [forecast('t-x', { Val: tsk('a'), Train: tsk('a') }), forecast('t', { Test: tsk('a'), Train: tsk('a') })]));
    const order = Object.entries(pdb.tableInfo)
      .sort((p, q) => p[1].nodeIdxOffset - q[1].nodeIdxOffset)
      .map(([k]) => k);
    expect(order).toEqual(['B:Db', 'a-b:Db', 'a:Db', 'ab:Db', 't:Test', 't:Train', 't-x:Train', 't-x:Val']);
  });

  it('throws where pre.rs would panic (or silently skip)', () => {
    const ent = table({ name: 'e', pkey: 'id', columns: [idCol] }, [{ id: 0 }]);
    const kid = (fkVal: number | string) =>
      table({ name: 'k', pkey: 'id', fkeys: { e: 'e' }, columns: [idCol, { name: 'e', dtype: 'int' }] }, [{ id: 0, e: fkVal }]);
    expect(() => preprocess(db([ent, kid(1)]))).toThrow(/outside/);
    expect(() => preprocess(db([ent, kid(0.5)]))).toThrow(/integer/);
    expect(() => preprocess(db([ent, table({ name: 'k', fkeys: { e: 'e' }, columns: [{ name: 'e', dtype: 'datetime' }] }, [{ e: 0 }])]))).toThrow(/dtype int/);
    expect(() => preprocess(db([ent, table({ name: 'k', fkeys: { e: 'nope' }, columns: [{ name: 'e', dtype: 'int' }] }, [{ e: 0 }])]))).toThrow(/not a DB table/);
    expect(() => preprocess(db([table({ name: 'p', pkey: 'id', columns: [idCol] }, [{ id: 1 }])]))).toThrow(/row position/);
    expect(() => preprocess(db([table({ name: 'a-b-c', columns: [{ name: 'v', dtype: 'float' }] }, [])]))).toThrow(/two dashes/);
    expect(() => preprocess(db([table({ name: 'p', timeCol: 'v', columns: [{ name: 'v', dtype: 'float' }] }, [])]))).toThrow(/datetime/);
    const split = table({ name: 'x', fkeys: { e: 'e' }, columns: [{ name: 'e', dtype: 'int' }, { name: 'y', dtype: 'bool' }] }, [{ e: 0, y: true }]);
    expect(() => preprocess(db([ent], [forecast('tsk', { Val: split })]))).toThrow(/no Train split/);
    expect(() => preprocess(db([table({ name: 'p', columns: [{ name: 'v', dtype: 'bool' }] }, [{ v: 1 }])]))).toThrow(/expected boolean/);
  });
});
