/**
 * Tests for masks.ts (mirror of model.py:164-191, spec §6.2).
 * The main test rebuilds the 19 tokens + 5 padding of the spec §6.3 token list by hand (ids are
 * arbitrary but consistent, as in pre.rs) and checks the masks against the printed matrices.
 */
import { describe, expect, it } from 'vitest';
import { allowedKeys, buildMasks, emptyQueryRows, maskDensity, maskToAscii, type MaskToken } from './masks';
import { Rng } from './rng';
import { ATTN_ORDER } from './types';

// Spec §6.3 token list: [node, table, column, f2p]. Task splits share the table name (§2.2).
const PRINTED: [number, string, string, number[]][] = [
  [16, 'customer-churn', 'timestamp', [0]],
  [16, 'customer-churn', 'churn', [0]],
  [0, 'customers', 'country', []],
  [0, 'customers', 'age', []],
  [0, 'customers', 'is_premium', []],
  [14, 'customer-churn', 'timestamp', [0]],
  [14, 'customer-churn', 'churn', [0]],
  [4, 'orders', 'order_time', [0, 9]],
  [4, 'orders', 'quantity', [0, 9]],
  [9, 'products', 'category', []],
  [9, 'products', 'price', []],
  [12, 'customer-churn', 'timestamp', [0]],
  [12, 'customer-churn', 'churn', [0]],
  [7, 'orders', 'order_time', [2, 9]],
  [7, 'orders', 'quantity', [2, 9]],
  [2, 'customers', 'country', []],
  [2, 'customers', 'is_premium', []],
  [15, 'customer-churn', 'timestamp', [2]],
  [15, 'customer-churn', 'churn', [2]],
];

function specTokens(): MaskToken[] {
  const ids = new Map<string, number>();
  const id = (s: string) => {
    if (!ids.has(s)) ids.set(s, ids.size);
    return ids.get(s)!;
  };
  const real: MaskToken[] = PRINTED.map(([node, tbl, col, f2p]) => ({
    isPadding: false,
    nodeIdx: node,
    f2pNbrIdxs: [...f2p, -1, -1, -1, -1, -1].slice(0, 5),
    tableNameIdx: id(tbl),
    colNameIdx: id(`${col} of ${tbl}`),
  }));
  // fly.rs:65-85 padding defaults
  const pad: MaskToken[] = Array.from({ length: 5 }, () => ({ isPadding: true, nodeIdx: -1, f2pNbrIdxs: [-1, -1, -1, -1, -1], tableNameIdx: 0, colNameIdx: 0 }));
  return [...real, ...pad];
}

const PAD_ROWS = Array.from({ length: 5 }, () => '.'.repeat(24));
const SPEC = {
  col: [
    '#....#.....#.....#......', '.#....#.....#.....#.....', '..#............#........', '...#....................',
    '....#...........#.......', '#....#.....#.....#......', '.#....#.....#.....#.....', '.......#.....#..........',
    '........#.....#.........', '.........#..............', '..........#.............', '#....#.....#.....#......',
    '.#....#.....#.....#.....', '.......#.....#..........', '........#.....#.........', '..#............#........',
    '....#...........#.......', '#....#.....#.....#......', '.#....#.....#.....#.....',
  ],
  feat: [
    '#####...................', '#####...................', '..###...................', '..###...................',
    '..###...................', '..#####.................', '..#####.................', '..###..####.............',
    '..###..####.............', '.........##.............', '.........##.............', '..###......##...........',
    '..###......##...........', '.........##..####.......', '.........##..####.......', '...............##.......',
    '...............##.......', '...............####.....', '...............####.....',
  ],
  nbr: [
    '........................', '........................', '##...####..##...........', '##...####..##...........',
    '##...####..##...........', '........................', '........................', '........................',
    '........................', '.......##....##.........', '.......##....##.........', '........................',
    '........................', '........................', '........................', '.............##..##.....',
    '.............##..##.....', '........................', '........................',
  ],
  full: Array.from({ length: 19 }, () => '#'.repeat(19) + '.....'),
};

describe('buildMasks on the spec §6.3 token list', () => {
  const tokens = specTokens();
  const m = buildMasks(tokens);
  const S = 24;

  for (const layer of ATTN_ORDER) {
    it(`${layer} mask equals the printed matrix exactly`, () => {
      expect(m[layer]).toBeInstanceOf(Uint8Array);
      expect(m[layer]).toHaveLength(S * S);
      expect(maskToAscii(m[layer], S)).toEqual([...SPEC[layer], ...PAD_ROWS]);
    });
  }

  it('reading the masks (spec §6.3)', () => {
    // the masked target (row 1) sees the label cells 6, 12, 18 (and itself) in column attention
    expect(allowedKeys(m.col, S, 1)).toEqual([1, 6, 12, 18]);
    // feature: own row (0,1) and parent c0 (2-4); o1 sees itself, c0 and p1
    expect(allowedKeys(m.feat, S, 1)).toEqual([0, 1, 2, 3, 4]);
    expect(allowedKeys(m.feat, S, 7)).toEqual([2, 3, 4, 7, 8, 9, 10]);
    // neighbor: c0 sees its children in context, p1 sees o1 and o4
    expect(allowedKeys(m.nbr, S, 2)).toEqual([0, 1, 5, 6, 7, 8, 11, 12]);
    expect(allowedKeys(m.nbr, S, 9)).toEqual([7, 8, 13, 14]);
    expect(emptyQueryRows(m.nbr, tokens)).toEqual([0, 1, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18]);
    for (const l of ['col', 'feat', 'full'] as const) expect(emptyQueryRows(m[l], tokens)).toEqual([]);
    expect(maskDensity(m).full).toBe(19 * 19);
  });

  it('the padding AND is essential: without it padding keys would match empty parent slots', () => {
    // a query with f2p [0,-1,...] vs a padding key (node −1): the raw membership test is true…
    expect(tokens[0].f2pNbrIdxs.includes(tokens[20].nodeIdx)).toBe(true);
    // …but the mask is 0
    expect(m.feat[0 * S + 20]).toBe(0);
  });
});

describe('buildMasks properties on random token sets', () => {
  function randomTokens(rng: Rng, S: number): MaskToken[] {
    const nNodes = 1 + rng.int(8);
    const parents = Array.from({ length: nNodes }, () => {
      const k = rng.int(4);
      return Array.from({ length: k }, () => rng.int(nNodes)); // self-FKs and duplicates allowed
    });
    const nReal = rng.int(S + 1);
    return Array.from({ length: S }, (_, i) => {
      if (i >= nReal && rng.bool(0.8)) return { isPadding: true, nodeIdx: -1, f2pNbrIdxs: [-1, -1, -1, -1, -1], tableNameIdx: 0, colNameIdx: 0 };
      const node = rng.int(nNodes);
      return {
        isPadding: false,
        nodeIdx: node,
        f2pNbrIdxs: [...parents[node], -1, -1, -1, -1, -1].slice(0, 5),
        tableNameIdx: node % 3,
        colNameIdx: rng.int(4),
      };
    });
  }

  it('match the model.py definitions, nbr is the transpose of the parent part of feat, permutation-equivariant', () => {
    const rng = new Rng(42);
    for (let trial = 0; trial < 200; trial++) {
      const S = 1 + rng.int(20);
      const t = randomTokens(rng, S);
      const m = buildMasks(t);
      for (let q = 0; q < S; q++)
        for (let k = 0; k < S; k++) {
          const pad = !t[q].isPadding && !t[k].isPadding;
          const i = q * S + k;
          expect(m.full[i]).toBe(pad ? 1 : 0);
          expect(m.col[i]).toBe(pad && t[q].colNameIdx === t[k].colNameIdx && t[q].tableNameIdx === t[k].tableNameIdx ? 1 : 0);
          const kvInF2p = t[q].f2pNbrIdxs.includes(t[k].nodeIdx);
          const qInF2p = t[k].f2pNbrIdxs.includes(t[q].nodeIdx);
          expect(m.feat[i]).toBe(pad && (t[q].nodeIdx === t[k].nodeIdx || kvInF2p) ? 1 : 0);
          expect(m.nbr[i]).toBe(pad && qInF2p ? 1 : 0);
          // nbr[q,k] = parent-part of feat[k,q]
          if (pad) expect(m.nbr[i] === 1).toBe(t[k].f2pNbrIdxs.includes(t[q].nodeIdx));
          if (pad && q === k) {
            expect(m.col[i] && m.feat[i] && m.full[i]).toBe(1); // self allowed (§6.2)
          }
        }
      // permuting tokens permutes masks (no positional information, §6.5)
      const perm = rng.shuffle(Array.from({ length: S }, (_, i) => i));
      const mp = buildMasks(perm.map((j) => t[j]));
      for (const l of ATTN_ORDER)
        for (let q = 0; q < S; q++) for (let k = 0; k < S; k++) expect(mp[l][q * S + k]).toBe(m[l][perm[q] * S + perm[k]]);
    }
  });
});
