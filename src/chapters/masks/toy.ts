/**
 * The worked example of spec §6.3: a toy ShopDB, seed = customer-churn Val row #16 (c0 @ 2024-03-01),
 * L = 24, w = 1. The token list and the printed masks below are copied from the spec; the chapter
 * rebuilds the masks with the engine's `buildMasks` and checks them against these strings.
 *
 * Ids: node ids are the spec's global node indices. Table and column ids are arbitrary but consistent
 * (any interning gives the same masks, because the masks only compare ids for equality). Column ids
 * start at 4, so no real column shares id 0 with the padding default (fly.rs:65-85), just as in
 * pre.rs, where index 0 of the string table is a table name, never a column phrase (spec §3.6).
 */
import type { Shape } from '@/components';
import { SemType, type Token } from '@/engine/types';

const d = (s: string) => Date.parse(`${s}T00:00:00Z`);

/** Table ids: arbitrary but consistent. Task splits share one table id (spec §6.2, pre.rs:424-437). */
const TABLE_ID: Record<string, number> = { customers: 0, orders: 1, products: 2, 'customer-churn': 3 };
const COL_ID: Record<string, number> = {
  'timestamp of customer-churn': 4,
  'churn of customer-churn': 5,
  'country of customers': 6,
  'age of customers': 7,
  'is_premium of customers': 8,
  'order_time of orders': 9,
  'quantity of orders': 10,
  'category of products': 11,
  'price of products': 12,
};

type Row = [node: number, table: string, type: Token['tableType'], rowPos: number, col: string, sem: SemType, raw: Token['raw'], f2p: number[], depth: number];

/** Spec §6.3 "Tokens" (positions 0..18), in emission order. */
const ROWS: Row[] = [
  [16, 'customer-churn', 'Val', 0, 'timestamp', SemType.DateTime, d('2024-03-01'), [0], 0],
  [16, 'customer-churn', 'Val', 0, 'churn', SemType.Boolean, true, [0], 0],
  [0, 'customers', 'Db', 0, 'country', SemType.Text, 'BR', [], 1],
  [0, 'customers', 'Db', 0, 'age', SemType.Number, 34, [], 1],
  [0, 'customers', 'Db', 0, 'is_premium', SemType.Boolean, true, [], 1],
  [14, 'customer-churn', 'Train', 2, 'timestamp', SemType.DateTime, d('2024-02-01'), [0], 2],
  [14, 'customer-churn', 'Train', 2, 'churn', SemType.Boolean, false, [0], 2],
  [4, 'orders', 'Db', 1, 'order_time', SemType.DateTime, d('2024-02-05'), [0, 9], 2],
  [4, 'orders', 'Db', 1, 'quantity', SemType.Number, 2, [0, 9], 2],
  [9, 'products', 'Db', 1, 'category', SemType.Text, 'toys', [], 3],
  [9, 'products', 'Db', 1, 'price', SemType.Number, 30, [], 3],
  [12, 'customer-churn', 'Train', 0, 'timestamp', SemType.DateTime, d('2024-01-01'), [0], 2],
  [12, 'customer-churn', 'Train', 0, 'churn', SemType.Boolean, false, [0], 2],
  [7, 'orders', 'Db', 4, 'order_time', SemType.DateTime, d('2024-01-25'), [2, 9], 4],
  [7, 'orders', 'Db', 4, 'quantity', SemType.Number, 1, [2, 9], 4],
  [2, 'customers', 'Db', 2, 'country', SemType.Text, 'BR', [], 5],
  [2, 'customers', 'Db', 2, 'is_premium', SemType.Boolean, false, [], 5],
  [15, 'customer-churn', 'Train', 3, 'timestamp', SemType.DateTime, d('2024-02-01'), [2], 6],
  [15, 'customer-churn', 'Train', 3, 'churn', SemType.Boolean, true, [2], 6],
];

export const TOY_L = 24;
export const TOY_REAL = ROWS.length; // 19
export const TOY_TARGET = 1;

/** Rows and columns of a toy-context mask, for its ShapeTag. */
export const TOY_SHAPE: Shape = {
  rows: `one per query token of the toy context of this chapter’s worked example (a small ShopDB, L = ${TOY_L}, w = 1), not the live one: ${TOY_REAL} cells, then ${TOY_L - TOY_REAL} padding positions`,
  cols: 'one per key token, in the same order as the rows',
};

/** The 24 toy tokens (19 cells + 5 padding with the fly.rs:65-85 defaults). Values are display-only. */
export const TOY_TOKENS: Token[] = Array.from({ length: TOY_L }, (_, pos): Token => {
  const r = ROWS[pos];
  if (!r) {
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
  const [node, table, type, rowPos, col, sem, raw, f2p, depth] = r;
  const isTarget = pos === TOY_TARGET;
  return {
    pos,
    isPadding: false,
    nodeIdx: node,
    f2pNbrIdxs: [...f2p, -1, -1, -1, -1, -1].slice(0, 5),
    tableNameIdx: TABLE_ID[table],
    colNameIdx: COL_ID[`${col} of ${table}`],
    sem,
    numberValue: 0,
    textIdx: 0,
    datetimeValue: 0,
    booleanValue: 0,
    mask: isTarget,
    isTarget,
    isTaskNode: table === 'customer-churn',
    classValueIdx: -1,
    tableName: table,
    tableType: type,
    colName: col,
    rowPos,
    raw,
    depth,
  };
});

/** ShopDB color slots (BUILD_PLAN §3.5): customers 1, products 2, orders 3, task tables 5. */
export const TOY_SLOTS: Record<string, number> = { customers: 1, products: 2, orders: 3, 'customer-churn': 5 };

/** Toy row names as the spec writes them (c0, o1, p1, #16 …). */
export function toyRowName(t: Token): string {
  if (t.isPadding) return 'PAD';
  if (t.tableName === 'customer-churn') return `#${t.nodeIdx}`;
  return `${t.tableName[0]}${t.rowPos}`;
}

const PAD5 = Array.from({ length: 5 }, () => '.'.repeat(TOY_L));

/** The masks exactly as printed in spec §6.3 (rows = query, cols = key, '#' = allowed). */
export const SPEC_PRINTED: Record<'col' | 'feat' | 'nbr' | 'full', string[]> = {
  col: [
    '#....#.....#.....#......',
    '.#....#.....#.....#.....',
    '..#............#........',
    '...#....................',
    '....#...........#.......',
    '#....#.....#.....#......',
    '.#....#.....#.....#.....',
    '.......#.....#..........',
    '........#.....#.........',
    '.........#..............',
    '..........#.............',
    '#....#.....#.....#......',
    '.#....#.....#.....#.....',
    '.......#.....#..........',
    '........#.....#.........',
    '..#............#........',
    '....#...........#.......',
    '#....#.....#.....#......',
    '.#....#.....#.....#.....',
    ...PAD5,
  ],
  feat: [
    '#####...................',
    '#####...................',
    '..###...................',
    '..###...................',
    '..###...................',
    '..#####.................',
    '..#####.................',
    '..###..####.............',
    '..###..####.............',
    '.........##.............',
    '.........##.............',
    '..###......##...........',
    '..###......##...........',
    '.........##..####.......',
    '.........##..####.......',
    '...............##.......',
    '...............##.......',
    '...............####.....',
    '...............####.....',
    ...PAD5,
  ],
  nbr: [
    '........................',
    '........................',
    '##...####..##...........',
    '##...####..##...........',
    '##...####..##...........',
    '........................',
    '........................',
    '........................',
    '........................',
    '.......##....##.........',
    '.......##....##.........',
    '........................',
    '........................',
    '........................',
    '........................',
    '.............##..##.....',
    '.............##..##.....',
    '........................',
    '........................',
    ...PAD5,
  ],
  // "full: rows 0-18 = '#' on keys 0-18, '.' on 19-23; rows 19-23 all '.'"
  full: [...Array.from({ length: TOY_REAL }, () => '#'.repeat(TOY_REAL) + '.'.repeat(TOY_L - TOY_REAL)), ...PAD5],
};
