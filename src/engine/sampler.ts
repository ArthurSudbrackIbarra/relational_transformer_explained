/**
 * Context-window sampler: a line-by-line mirror of `Sampler::seq` in the paper-era
 * `rustler/src/fly.rs:357-511` (commit 2a899c8, byte-identical to 801a32b "fixed bfs width
 * sampling"), plus the seed/target helpers of `fly.rs:287-296` and `rt/data.py:31-108`.
 *
 * Spec: research/RT_TECHNICAL_SPEC.md §4.3 (pseudo-code we follow exactly), §4.4-§4.16,
 * §2.3-§2.5 (seeds, in-context labels), §5.1 (token fields).
 *
 * The only intended difference from fly.rs is the PRNG (xoshiro128** instead of ChaCha12, see
 * rng.ts). Every draw happens at the same point of the algorithm and with the same range, so the
 * distribution over contexts is the same (§4.2: bit-exact parity is not required).
 *
 * In addition to the tokens we record a `SamplerEvent[]` trace of every decision so the UI can
 * replay the BFS step by step. Event order per loop iteration:
 *   pop → (skip-visited | visit → push-parent* → per P→F edge: child-other-task-table |
 *   child-future | push-task-child | (collected as DB child) → db-children-subsampled? →
 *   push-db-child* → per cell: drop-leakage-cell | emit → truncated?)
 * and finally frontier-empty (if the frontier ran dry) and pad (if positions remain).
 * Conventions: `truncated.pos` = position of the LAST emitted token (L−1); `pad.from`/`pad.to`
 * are the first/last padded positions (inclusive). `truncated` means "the L hard stop fired"
 * (fly.rs:503-509), NOT necessarily that a row was cut: when L is reached exactly at a row's last
 * cell nothing is cut. Whether the row was cut = fewer of its non-dropped cells emitted than it has.
 */
import { Rng, contextSeed } from './rng';
import {
  MAX_F2P_NBRS,
  SemType,
  type Cell,
  type ColumnStats,
  type GraphNode,
  type PreprocessedDB,
  type SampleParams,
  type SampledContext,
  type SamplerEvent,
  type Split,
  type TaskDef,
  type Token,
} from './types';
import { columnPhrase, tableKey } from './preprocess';

// ─────────────────────────────────────────────────────────────────────────────
// Task → seeds, target column, leakage columns  (data.py:31-108, fly.rs:287-296)
// ─────────────────────────────────────────────────────────────────────────────

/** The `table_name` data.py passes for a task: the task table (forecast) or the DB table (autocomplete). */
export function taskTableName(task: TaskDef): string {
  return task.kind === 'autocomplete' ? (task.table ?? task.name) : task.name;
}

/**
 * The table_info key whose rows are the seeds (data.py:77-81): `"<table>:Db"` if such a DB table
 * exists (autocomplete; the split is then IGNORED, §2.4), else `"<table>:<split>"` (forecast).
 */
export function seedTableKey(pdb: PreprocessedDB, task: TaskDef, split: Split): string {
  const name = taskTableName(task);
  const dbKey = tableKey(name, 'Db');
  if (dbKey in pdb.tableInfo) return dbKey;
  const key = tableKey(name, split);
  if (!(key in pdb.tableInfo)) throw new Error(`seedTableKey: no table "${key}" in ${pdb.dbName}`);
  return key;
}

/** `column_index["<target> of <table>"]` (data.py:31-43, 86). Throws like data.py if absent. */
export function targetColNameIdx(pdb: PreprocessedDB, task: TaskDef): number {
  return lookupColumn(pdb, task.targetCol, taskTableName(task));
}

/** `column_index["<col> of <table>"]` for every leakage column (data.py:89-92). `[]` for forecast tasks. */
export function dropColNameIdxs(pdb: PreprocessedDB, task: TaskDef): number[] {
  return task.columnsToDrop.map((c) => lookupColumn(pdb, c, taskTableName(task)));
}

function lookupColumn(pdb: PreprocessedDB, col: string, table: string): number {
  const phrase = columnPhrase(col, table);
  const idx = pdb.columnIndex[phrase];
  if (idx === undefined) throw new Error(`Column "${phrase}" not found in column_index for dataset ${pdb.dbName}.`);
  return idx;
}

/**
 * Seeds of a task split (spec §2.3-2.4): every row of the seed table whose cells contain the target
 * column, i.e. whose label is non-null and non-NaN (fly.rs:287-296). Returned in node order.
 */
export function listSeeds(pdb: PreprocessedDB, task: TaskDef, split: Split): number[] {
  const info = pdb.tableInfo[seedTableKey(pdb, task, split)];
  const target = targetColNameIdx(pdb, task);
  const out: number[] = [];
  for (let j = info.nodeIdxOffset; j < info.nodeIdxOffset + info.numNodes; j++) {
    if (pdb.nodes[j].cells.some((c) => c.colNameIdx === target)) out.push(j);
  }
  return out;
}

/**
 * Stats used to de-normalize the target (spec §8.4): the target column's stats in the seed table.
 * For forecast tasks these are the Train split's stats (also used by Val/Test, pre.rs:383-396);
 * for autocomplete, the DB table's stats over all rows.
 */
export function targetStats(pdb: PreprocessedDB, task: TaskDef): ColumnStats {
  const key = seedTableKey(pdb, task, 'Train');
  const s = pdb.colStats[key]?.[task.targetCol];
  if (!s) throw new Error(`targetStats: no stats for ${task.targetCol} in ${key}`);
  return s;
}

/**
 * Convenience: SampleParams for one seed of a task. L and w default to the mini setup (128, 16).
 * The PRNG seed is, like fly.rs:371-375, `epoch + seed_node_idx + seed` (defaults epoch = 0,
 * seed = 0, i.e. the fixed evaluation contexts of §8.6), unless `rngSeed` is given explicitly.
 */
export function sampleParamsFor(
  pdb: PreprocessedDB,
  task: TaskDef,
  seedNode: number,
  opts: { L?: number; w?: number; epoch?: number; seed?: number; rngSeed?: number } = {},
): SampleParams {
  return {
    seedNode,
    L: opts.L ?? 128,
    w: opts.w ?? 16,
    targetColNameIdx: targetColNameIdx(pdb, task),
    dropColNameIdxs: dropColNameIdxs(pdb, task),
    rngSeed: opts.rngSeed ?? contextSeed(opts.epoch ?? 0, seedNode, opts.seed ?? 0),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Tokens
// ─────────────────────────────────────────────────────────────────────────────

/** A padding token with the defaults of `Vecs::new` (fly.rs:65-85, spec §4.11). */
export function paddingToken(pos: number): Token {
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

/** One emitted cell (fly.rs:462-500). */
function cellToken(pos: number, node: GraphNode, cell: Cell, depth: number, isTarget: boolean, targetCol: number): Token {
  const f2p = [-1, -1, -1, -1, -1];
  node.f2pNbrIdxs.forEach((p, j) => (f2p[j] = p)); // fly.rs:465-467 (all parents, in FK-column order)
  return {
    pos,
    isPadding: false,
    nodeIdx: node.nodeIdx,
    f2pNbrIdxs: f2p,
    tableNameIdx: node.tableNameIdx,
    colNameIdx: cell.colNameIdx,
    sem: cell.sem,
    // The label stays in the token even when masked (fly.rs:482-490, spec §5.3).
    numberValue: cell.number,
    textIdx: cell.textIdx,
    datetimeValue: cell.datetime,
    booleanValue: cell.boolean,
    mask: isTarget, // fly.rs:496
    isTarget, // fly.rs:492-494
    isTaskNode: node.isTaskNode || cell.colNameIdx === targetCol, // fly.rs:498-499
    classValueIdx: cell.sem === SemType.Text ? cell.textIdx : -1, // pre.rs:563, 582, 593, 604
    tableName: node.tableName,
    tableType: node.tableType,
    colName: cell.colName,
    rowPos: node.rowPos,
    raw: cell.raw,
    depth,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// sampleContext
// ─────────────────────────────────────────────────────────────────────────────

export interface SampleOptions {
  /** Record the SamplerEvent trace (default true). Disable for bulk evaluation. */
  recordEvents?: boolean;
}

/**
 * Build one context window around `params.seedNode`, exactly as fly.rs `seq` (spec §4.3).
 * Returns exactly L tokens (padding included), the visit order and the event trace.
 * `targetPos` is −1 only in the degenerate case where the target cell is not emitted
 * (L smaller than its position inside the seed row, or the target listed as a leakage column).
 */
export function sampleContext(pdb: PreprocessedDB, params: SampleParams, opts: SampleOptions = {}): SampledContext {
  const { seedNode, L, w, targetColNameIdx: targetCol, rngSeed } = params;
  if (!Number.isInteger(L) || L < 1) throw new Error(`sampleContext: L must be an integer ≥ 1, got ${L}`);
  if (!Number.isInteger(w) || w < 0) throw new Error(`sampleContext: w must be an integer ≥ 0, got ${w}`);
  const nodes = pdb.nodes;
  if (!Number.isInteger(seedNode) || seedNode < 0 || seedNode >= nodes.length) {
    throw new Error(`sampleContext: seed node ${seedNode} out of range`);
  }

  const record = opts.recordEvents !== false;
  const events: SamplerEvent[] = [];
  const ev = (e: SamplerEvent) => {
    if (record) events.push(e);
  };
  const drop = new Set(params.dropColNameIdxs);

  const seed = nodes[seedNode]; // fly.rs:367
  const visitedFlag = new Uint8Array(nodes.length); // fly.rs:364 (whole DB)
  const f2pStack: [number, number][] = [[0, seedNode]]; // fly.rs:366 (depth, node), LIFO
  const buckets: number[][] = []; // fly.rs:368 p2f_ftr[depth]
  const rng = new Rng(rngSeed); // fly.rs:371-375 (caller passes epoch + seed_node_idx + seed)
  const pushBucket = (d: number, n: number) => {
    while (buckets.length <= d) buckets.push([]); // fly.rs:431-435, 446-450
    buckets[d].push(n);
  };

  const tokens: Token[] = [];
  const visited: SampledContext['visited'] = [];
  let targetPos = -1;
  let full = false;

  while (!full) {
    // ── select the next row (fly.rs:377-397) ──
    let depth: number;
    let n: number;
    if (f2pStack.length > 0) {
      [depth, n] = f2pStack.pop()!; // F→P rows: absolute priority, LIFO (fly.rs:378-379)
      ev({ kind: 'pop', from: 'stack', node: n, depth });
    } else {
      let d = -1;
      for (let i = 0; i < buckets.length; i++) {
        if (buckets[i].length > 0) {
          d = i; // smallest depth label with candidates (fly.rs:381-390)
          break;
        }
      }
      if (d === -1) {
        ev({ kind: 'frontier-empty', tokens: tokens.length }); // fly.rs:387-389
        break;
      }
      const b = buckets[d];
      const size = b.length;
      const r = rng.int(size); // rng.random_range(0..len) (fly.rs:391)
      const last = size - 1;
      [b[r], b[last]] = [b[last], b[r]]; // swap-remove (fly.rs:392-394)
      n = b.pop()!;
      depth = d;
      ev({ kind: 'pop', from: 'bucket', node: n, depth, bucketSize: size, draw: r });
    }

    // ── visited check at POP time (fly.rs:399-402) ──
    if (visitedFlag[n]) {
      ev({ kind: 'skip-visited', node: n, depth });
      continue;
    }
    visitedFlag[n] = 1;
    visited.push({ node: n, depth, order: visited.length });
    ev({ kind: 'visit', node: n, depth });
    const node = nodes[n];

    // ── expand the frontier BEFORE emitting cells (fly.rs:406-452, spec §4.8) ──
    // F→P: every parent, no time filter, no visited test (fly.rs:406-408).
    for (const e of node.f2pEdges) {
      f2pStack.push([depth + 1, e.node]);
      ev({ kind: 'push-parent', from: n, node: e.node, depth: depth + 1 });
    }

    // P→F (fly.rs:410-437)
    const dbKids: number[] = [];
    for (const e of pdb.p2fAdj[n]) {
      // Task-table children only from the seed's own task table, any split (fly.rs:417).
      if (e.tableNameIdx !== seed.tableNameIdx && e.tableType !== 'Db') {
        ev({ kind: 'child-other-task-table', from: n, node: e.node });
        continue;
      }
      // Temporal filter vs the SEED, only when both timestamps exist; equal kept (fly.rs:421-424, §4.6).
      if (e.ts !== null && seed.ts !== null && e.ts > seed.ts) {
        ev({ kind: 'child-future', from: n, node: e.node, childTs: e.ts, seedTs: seed.ts });
        continue;
      }
      // DB children are collected for width subsampling; already-visited ones count too (fly.rs:426-429, §4.7).
      if (e.tableType === 'Db') {
        dbKids.push(e.node);
        continue;
      }
      // Task rows (in-context labels): never subsampled (fly.rs:431-436).
      pushBucket(depth + 1, e.node);
      ev({ kind: 'push-task-child', from: n, node: e.node, depth: depth + 1 });
    }

    // Width bound w on DB children only, AFTER the time filter (fly.rs:439-452).
    let keep: number[];
    if (dbKids.length > w) {
      keep = rng.sampleIndices(dbKids.length, w); // rand::seq::index::sample (fly.rs:440)
      ev({ kind: 'db-children-subsampled', from: n, candidates: dbKids.slice(), kept: keep.map((i) => dbKids[i]), w });
    } else {
      keep = dbKids.map((_, i) => i); // fly.rs:442
    }
    for (const i of keep) {
      pushBucket(depth + 1, dbKids[i]);
      ev({ kind: 'push-db-child', from: n, node: dbKids[i], depth: depth + 1 });
    }

    // ── emit the row's cells in stored order (fly.rs:454-506) ──
    for (const c of node.cells) {
      // Leakage drop rule (fly.rs:457-461). JS `null === null` is true, like Rust `None == None`.
      if ((node.nodeIdx === seedNode && drop.has(c.colNameIdx)) || (node.ts === seed.ts && drop.has(c.colNameIdx))) {
        ev({ kind: 'drop-leakage-cell', node: n, colName: c.colName });
        continue;
      }
      // fly.rs:464 assert, reached only when this row actually emits a cell (§1.2).
      if (node.f2pNbrIdxs.length > MAX_F2P_NBRS) {
        throw new Error(`sampleContext: row #${n} has ${node.f2pNbrIdxs.length} parents > MAX_F2P_NBRS=${MAX_F2P_NBRS} (fly.rs:464 panics)`);
      }
      const isTarget = node.nodeIdx === seedNode && c.colNameIdx === targetCol; // fly.rs:492-494
      const pos = tokens.length;
      tokens.push(cellToken(pos, node, c, depth, isTarget, targetCol));
      if (isTarget) targetPos = pos;
      ev({ kind: 'emit', node: n, pos, colName: c.colName, isTarget });
      // Hard stop at exactly L tokens, possibly mid-row (fly.rs:502-509, §4.11).
      if (tokens.length >= L) {
        ev({ kind: 'truncated', node: n, pos });
        full = true;
        break;
      }
    }
  }

  // ── padding (fly.rs:65-85: the batch buffers start as padding) ──
  const numReal = tokens.length;
  if (numReal < L) ev({ kind: 'pad', from: numReal, to: L - 1 });
  for (let pos = numReal; pos < L; pos++) tokens.push(paddingToken(pos));

  return { params, tokens, numReal, visited, events, targetPos };
}
