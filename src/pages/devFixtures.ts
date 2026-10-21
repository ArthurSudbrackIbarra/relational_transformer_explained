/**
 * Fake data for the component gallery (/#/dev/components). Not used by any
 * chapter. The token list is the spec §6.3 toy context (seed = churn Val row
 * #16, L = 24, w = 1), so the masks built here can be checked against the
 * masks printed in the spec.
 */
import { Rng } from '@/engine/rng';
import { SemType, type AttnLayer, type MaskSet, type RawValue, type TableSchema, type Token } from '@/engine/types';

const DAY = 86_400_000;
const d = (s: string) => Date.parse(`${s}T00:00:00Z`);

// ── spec §6.3 statistics ────────────────────────────────────────────────────
const MU_T = Date.parse('2024-02-12T18:27:00Z');
const SIGMA_T = 30.08 * DAY;
const zT = (ms: number) => (ms - MU_T) / SIGMA_T;
const zAge = (v: number) => (v - 42.5) / 12.0208;
const zPrem = (b: boolean) => (b ? 1.1547 : -0.5774);
const zChurn = (b: boolean) => (b ? 0.866 : -0.866);
const zQty = (v: number) => (v - 1.6) / 0.8944;
const zPrice = (v: number) => (v - 21) / 12.7279;

const TBL = { customers: 0, orders: 1, products: 2, 'customer-churn': 3 } as const;
const COL: Record<string, number> = {
  'timestamp of customer-churn': 10,
  'churn of customer-churn': 11,
  'country of customers': 12,
  'age of customers': 13,
  'is_premium of customers': 14,
  'order_time of orders': 15,
  'quantity of orders': 16,
  'category of products': 17,
  'price of products': 18,
};

/** tableName → slot, ShopDB convention (BUILD_PLAN §3.5). */
export const TOY_SLOTS: Record<string, number> = { customers: 1, products: 2, orders: 3, reviews: 7, 'customer-churn': 5 };

interface Spec {
  node: number;
  table: keyof typeof TBL;
  type: Token['tableType'];
  row: number;
  col: string;
  sem: SemType;
  raw: RawValue;
  f2p: number[];
  depth: number;
  target?: boolean;
}

const SPECS: Spec[] = [
  { node: 16, table: 'customer-churn', type: 'Val', row: 0, col: 'timestamp', sem: SemType.DateTime, raw: d('2024-03-01'), f2p: [0], depth: 0 },
  { node: 16, table: 'customer-churn', type: 'Val', row: 0, col: 'churn', sem: SemType.Boolean, raw: true, f2p: [0], depth: 0, target: true },
  { node: 0, table: 'customers', type: 'Db', row: 0, col: 'country', sem: SemType.Text, raw: 'BR', f2p: [], depth: 1 },
  { node: 0, table: 'customers', type: 'Db', row: 0, col: 'age', sem: SemType.Number, raw: 34, f2p: [], depth: 1 },
  { node: 0, table: 'customers', type: 'Db', row: 0, col: 'is_premium', sem: SemType.Boolean, raw: true, f2p: [], depth: 1 },
  { node: 14, table: 'customer-churn', type: 'Train', row: 2, col: 'timestamp', sem: SemType.DateTime, raw: d('2024-02-01'), f2p: [0], depth: 2 },
  { node: 14, table: 'customer-churn', type: 'Train', row: 2, col: 'churn', sem: SemType.Boolean, raw: false, f2p: [0], depth: 2 },
  { node: 4, table: 'orders', type: 'Db', row: 1, col: 'order_time', sem: SemType.DateTime, raw: d('2024-02-05'), f2p: [0, 9], depth: 2 },
  { node: 4, table: 'orders', type: 'Db', row: 1, col: 'quantity', sem: SemType.Number, raw: 2, f2p: [0, 9], depth: 2 },
  { node: 9, table: 'products', type: 'Db', row: 1, col: 'category', sem: SemType.Text, raw: 'toys', f2p: [], depth: 3 },
  { node: 9, table: 'products', type: 'Db', row: 1, col: 'price', sem: SemType.Number, raw: 30, f2p: [], depth: 3 },
  { node: 12, table: 'customer-churn', type: 'Train', row: 0, col: 'timestamp', sem: SemType.DateTime, raw: d('2024-01-01'), f2p: [0], depth: 2 },
  { node: 12, table: 'customer-churn', type: 'Train', row: 0, col: 'churn', sem: SemType.Boolean, raw: false, f2p: [0], depth: 2 },
  { node: 7, table: 'orders', type: 'Db', row: 4, col: 'order_time', sem: SemType.DateTime, raw: d('2024-01-25'), f2p: [2, 9], depth: 4 },
  { node: 7, table: 'orders', type: 'Db', row: 4, col: 'quantity', sem: SemType.Number, raw: 1, f2p: [2, 9], depth: 4 },
  { node: 2, table: 'customers', type: 'Db', row: 2, col: 'country', sem: SemType.Text, raw: 'BR', f2p: [], depth: 5 },
  { node: 2, table: 'customers', type: 'Db', row: 2, col: 'is_premium', sem: SemType.Boolean, raw: false, f2p: [], depth: 5 },
  { node: 15, table: 'customer-churn', type: 'Train', row: 3, col: 'timestamp', sem: SemType.DateTime, raw: d('2024-02-01'), f2p: [2], depth: 6 },
  { node: 15, table: 'customer-churn', type: 'Train', row: 3, col: 'churn', sem: SemType.Boolean, raw: true, f2p: [2], depth: 6 },
];

const TEXT_IDX: Record<string, number> = { BR: 20, toys: 21 };

function scalar(s: Spec): Pick<Token, 'numberValue' | 'datetimeValue' | 'booleanValue' | 'textIdx' | 'classValueIdx'> {
  const out = { numberValue: 0, datetimeValue: 0, booleanValue: 0, textIdx: 0, classValueIdx: -1 };
  if (s.sem === SemType.DateTime) out.datetimeValue = zT(s.raw as number);
  else if (s.sem === SemType.Number) out.numberValue = s.col === 'age' ? zAge(s.raw as number) : s.col === 'quantity' ? zQty(s.raw as number) : zPrice(s.raw as number);
  else if (s.sem === SemType.Boolean) out.booleanValue = s.col === 'churn' ? zChurn(s.raw as boolean) : zPrem(s.raw as boolean);
  else {
    out.textIdx = TEXT_IDX[s.raw as string] ?? 0;
    out.classValueIdx = out.textIdx;
  }
  return out;
}

export const TOY_L = 24;

/** The spec §6.3 toy context as engine Tokens (padding included). */
export const TOY_TOKENS: Token[] = Array.from({ length: TOY_L }, (_, pos): Token => {
  const s = SPECS[pos];
  if (!s) {
    return {
      pos,
      isPadding: true,
      nodeIdx: -1,
      f2pNbrIdxs: [-1, -1, -1, -1, -1],
      tableNameIdx: 0,
      colNameIdx: 0,
      sem: SemType.Number,
      numberValue: 0,
      textIdx: 0,
      datetimeValue: 0,
      booleanValue: 0,
      mask: false,
      isTarget: false,
      isTaskNode: false,
      classValueIdx: -1,
      tableName: '',
      tableType: null,
      colName: '',
      rowPos: -1,
      raw: null,
      depth: -1,
    };
  }
  const f2p = [...s.f2p, -1, -1, -1, -1, -1].slice(0, 5);
  return {
    pos,
    isPadding: false,
    nodeIdx: s.node,
    f2pNbrIdxs: f2p,
    tableNameIdx: TBL[s.table],
    colNameIdx: COL[`${s.col} of ${s.table}`],
    sem: s.sem,
    ...scalar(s),
    mask: !!s.target,
    isTarget: !!s.target,
    isTaskNode: s.table === 'customer-churn',
    tableName: s.table,
    tableType: s.type,
    colName: s.col,
    rowPos: s.row,
    raw: s.raw,
    depth: s.depth,
  };
});

/** Masks exactly as model.py:164-191 (spec §6.2, Appendix A), for the gallery only. */
export function toyMasks(tokens: Token[]): MaskSet {
  const S = tokens.length;
  const mk = () => new Uint8Array(S * S);
  const m: MaskSet = { col: mk(), feat: mk(), nbr: mk(), full: mk() };
  for (let q = 0; q < S; q++) {
    for (let k = 0; k < S; k++) {
      const a = tokens[q];
      const b = tokens[k];
      const pad = !a.isPadding && !b.isPadding;
      if (!pad) continue;
      const i = q * S + k;
      m.full[i] = 1;
      if (a.colNameIdx === b.colNameIdx && a.tableNameIdx === b.tableNameIdx) m.col[i] = 1;
      if (a.nodeIdx === b.nodeIdx || a.f2pNbrIdxs.includes(b.nodeIdx)) m.feat[i] = 1;
      if (b.f2pNbrIdxs.includes(a.nodeIdx)) m.nbr[i] = 1;
    }
  }
  return m;
}

/** The ASCII column mask row for the target as printed in spec §6.3 (to self-check the gallery). */
export const SPEC_COLUMN_ROW_1 = '.#....#.....#.....#.....';

export function maskRowAscii(mask: Uint8Array, S: number, q: number): string {
  let s = '';
  for (let k = 0; k < S; k++) s += mask[q * S + k] ? '#' : '.';
  return s;
}

export const TOY_LABELS = TOY_TOKENS.map((t) => (t.isPadding ? 'pad' : `${t.tableName === 'customer-churn' ? 'churn' : t.tableName}.${t.colName}`));

// ── Toy entity graph (spec §6.3 DB) ─────────────────────────────────────────
export const TOY_GRAPH = (() => {
  const nodes = [
    ...[0, 1, 2].map((i) => ({ id: i, label: `c${i}`, table: 'customers', slot: 1, title: `customers[${i}]` })),
    ...[0, 1, 2, 3, 4].map((i) => ({ id: 3 + i, label: `o${i}`, table: 'orders', slot: 3, title: `orders[${i}]` })),
    ...[0, 1].map((i) => ({ id: 8 + i, label: `p${i}`, table: 'products', slot: 2, title: `products[${i}]` })),
    ...[
      [10, 'Test[0]'],
      [11, 'Test[1]'],
      [12, 'Train[0]'],
      [13, 'Train[1]'],
      [14, 'Train[2]'],
      [15, 'Train[3]'],
      [16, 'Val[0]'],
      [17, 'Val[1]'],
    ].map(([id, l]) => ({ id: id as number, label: `#${id}`, table: 'customer-churn', slot: 5, title: `customer-churn:${l}` })),
  ];
  const orders: [number, number][] = [
    [0, 0],
    [0, 1],
    [1, 0],
    [0, 0],
    [2, 1],
  ];
  const edges = [
    ...orders.flatMap(([c, p], i) => [
      { source: 3 + i, target: c },
      { source: 3 + i, target: 8 + p },
    ]),
    ...[
      [10, 0],
      [11, 2],
      [12, 0],
      [13, 1],
      [14, 0],
      [15, 2],
      [16, 0],
      [17, 1],
    ].map(([s, t]) => ({ source: s, target: t })),
  ];
  return { nodes, edges };
})();

// ── ShopDB-like schema (BUILD_PLAN §3.1) ─────────────────────────────────────
export const SHOP_SCHEMA: TableSchema[] = [
  {
    name: 'customers',
    pkey: 'customer_id',
    fkeys: {},
    timeCol: null,
    columns: [
      { name: 'customer_id', dtype: 'int' },
      { name: 'name', dtype: 'str' },
      { name: 'country', dtype: 'str' },
      { name: 'age', dtype: 'float' },
      { name: 'is_premium', dtype: 'bool' },
      { name: 'signup_date', dtype: 'datetime' },
    ],
  },
  {
    name: 'products',
    pkey: 'product_id',
    fkeys: {},
    timeCol: null,
    columns: [
      { name: 'product_id', dtype: 'int' },
      { name: 'name', dtype: 'str' },
      { name: 'category', dtype: 'str' },
      { name: 'price', dtype: 'float' },
    ],
  },
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
      { name: 'channel', dtype: 'str' },
    ],
  },
  {
    name: 'reviews',
    pkey: 'review_id',
    fkeys: { customer_id: 'customers', product_id: 'products' },
    timeCol: 'review_time',
    columns: [
      { name: 'review_id', dtype: 'int' },
      { name: 'customer_id', dtype: 'int' },
      { name: 'product_id', dtype: 'int' },
      { name: 'review_time', dtype: 'datetime' },
      { name: 'rating', dtype: 'int' },
      { name: 'review_text', dtype: 'str' },
      { name: 'verified', dtype: 'bool' },
    ],
  },
  {
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
];

// ── Fake rows for TableView ─────────────────────────────────────────────────
const COUNTRIES = ['BR', 'US', 'DE', 'JP', 'IN', 'FR'];
const NAMES = ['Ana', 'Bruno', 'Chen', 'Dara', 'Elif', 'Femi', 'Gita', 'Hugo', 'Ines', 'Jon', 'Kai', 'Lea'];

export const FAKE_CUSTOMERS: Record<string, RawValue>[] = (() => {
  const rng = new Rng(11);
  return Array.from({ length: 80 }, (_, i) => ({
    customer_id: i,
    name: `${NAMES[i % NAMES.length]} ${String.fromCharCode(65 + ((i * 7) % 26))}.`,
    country: rng.choice(COUNTRIES),
    age: rng.bool(0.1) ? null : Math.round(rng.uniform(18, 70)),
    is_premium: rng.bool(0.3),
    signup_date: d('2023-01-01') + Math.floor(rng.uniform(0, 365)) * DAY,
  }));
})();

/** z-scores of the fake customers (sample std for numbers/booleans; population std for datetimes, spec §3.3). */
export const FAKE_CUSTOMER_Z = (() => {
  const stat = (xs: number[], ddof: number) => {
    const n = xs.length;
    const mu = xs.reduce((a, b) => a + b, 0) / n;
    const v = xs.reduce((a, b) => a + (b - mu) ** 2, 0) / (n - ddof);
    return { mu, sd: Math.sqrt(v) || 1 };
  };
  const age = stat(FAKE_CUSTOMERS.filter((r) => r.age !== null).map((r) => r.age as number), 1);
  const prem = stat(FAKE_CUSTOMERS.map((r) => (r.is_premium ? 1 : 0)), 1);
  const sign = stat(FAKE_CUSTOMERS.map((r) => r.signup_date as number), 0);
  return (row: number, col: string): number | null => {
    const r = FAKE_CUSTOMERS[row];
    if (!r) return null;
    if (col === 'age') return r.age === null ? null : ((r.age as number) - age.mu) / age.sd;
    if (col === 'is_premium') return ((r.is_premium ? 1 : 0) - prem.mu) / prem.sd;
    if (col === 'signup_date') return ((r.signup_date as number) - sign.mu) / sign.sd;
    return null;
  };
})();

export const FAKE_ORDERS: Record<string, RawValue>[] = (() => {
  const rng = new Rng(5);
  const rows = Array.from({ length: 400 }, () => ({
    t: d('2024-01-01') + Math.floor(rng.uniform(0, 540)) * DAY + Math.floor(rng.uniform(8, 22)) * 3_600_000,
    c: rng.int(80),
    p: rng.int(30),
    q: 1 + rng.int(5),
    ch: rng.choice(['web', 'app', 'store']),
  })).sort((a, b) => a.t - b.t);
  return rows.map((r, i) => ({ order_id: i, customer_id: r.c, product_id: r.p, order_time: r.t, quantity: r.q, channel: r.ch }));
})();

// ── Fake numeric arrays ─────────────────────────────────────────────────────
export function fakeVector(n: number, seed: number, scale = 1): Float32Array {
  const rng = new Rng(seed);
  const v = new Float32Array(n);
  for (let i = 0; i < n; i++) v[i] = rng.normal(0, scale);
  return v;
}

/** H×S×S row-softmax "attention" with a block structure, plus a shared mask (fake). */
export function fakeAttention(S: number, H: number, seed: number, mask?: Uint8Array): Float32Array {
  const rng = new Rng(seed);
  const out = new Float32Array(H * S * S);
  for (let h = 0; h < H; h++) {
    for (let q = 0; q < S; q++) {
      let z = 0;
      const row = new Float64Array(S);
      for (let k = 0; k < S; k++) {
        if (mask && !mask[q * S + k]) continue;
        const near = Math.abs(((q * (h + 1)) % S) - k) < 6 ? 2.2 : 0;
        const e = Math.exp(rng.normal(0, 0.9) + near + (k % (8 + h * 3) === 0 ? 1.5 : 0));
        row[k] = e;
        z += e;
      }
      for (let k = 0; k < S; k++) out[(h * S + q) * S + k] = z > 0 ? row[k] / z : 0;
    }
  }
  return out;
}

/** A block-diagonal fake mask for big heatmap demos. */
export function fakeMask(S: number): Uint8Array {
  const m = new Uint8Array(S * S);
  for (let q = 0; q < S; q++) for (let k = 0; k < S; k++) m[q * S + k] = Math.floor(q / 16) === Math.floor(k / 16) || (q + k) % 23 === 0 ? 1 : 0;
  return m;
}

export const ATTN_LAYERS: AttnLayer[] = ['col', 'feat', 'nbr', 'full'];

/** Fake fine-tuning curves (illustrative only). */
export function fakeCurves() {
  const steps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
  const pre = steps.map((s, i) => ({ x: s, y: 0.7 + 0.08 * (1 - Math.exp(-i / 4)) - (i === 1 ? 0.01 : 0) }));
  const rnd = steps.map((s, i) => ({ x: s, y: 0.5 + 0.26 * (1 - Math.exp(-i / 5)) }));
  return [
    { id: 'pre', label: 'from pretrained', slot: 1, points: pre },
    { id: 'rnd', label: 'from random init', slot: 2, points: rnd },
  ];
}
