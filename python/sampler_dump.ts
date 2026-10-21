/**
 * Dump the raw DBs + the TS engine's preprocess / sampler / mask outputs for the independent Python
 * cross-check python/sampler_crosscheck.py (a port of pre.rs + fly.rs + model.py masks written from
 * the Rust/Python sources, not from the TS code). Per task and split: the first 8 seeds plus 12 random
 * ones, each sampled at (L, w) = (128, 16), (512, 3), (64, 1) and (1024, 256).
 *
 * Usage: npx tsx python/sampler_dump.ts [out.json]   (default python/data/sampler-dump.json)
 */
import { writeFileSync } from 'node:fs';
import { getDatabase } from '../src/engine/db/index.ts';
import { preprocess } from '../src/engine/preprocess.ts';
import { listSeeds, sampleContext, sampleParamsFor } from '../src/engine/sampler.ts';
import { buildMasks } from '../src/engine/masks.ts';
import { Rng } from '../src/engine/rng.ts';
import type { RawTable } from '../src/engine/types.ts';

const out: any = {};
const dumpTable = (t: RawTable) => ({
  name: t.schema.name, pkey: t.schema.pkey, fkeys: t.schema.fkeys, timeCol: t.schema.timeCol,
  columns: t.schema.columns.map((c) => ({ name: c.name, dtype: c.dtype })),
  rows: t.rows.map((r) => t.schema.columns.map((c) => (r[c.name] === undefined ? null : r[c.name]))),
});
for (const name of ['shop', 'school', 'racing', 'forum'] as const) {
  const db = getDatabase(name);
  const pdb = preprocess(db);
  const raw = {
    tables: db.tables.map(dumpTable),
    tasks: db.tasks.map((t) => ({
      name: t.name, kind: t.kind, targetCol: t.targetCol, columnsToDrop: t.columnsToDrop, table: t.table ?? null,
      tables: t.tables ? { Train: dumpTable(t.tables.Train), Val: dumpTable(t.tables.Val), Test: dumpTable(t.tables.Test) } : null,
    })),
  };
  const pre = {
    text: pdb.text, tableInfo: pdb.tableInfo, colStats: pdb.colStats, datetimeStats: pdb.datetimeStats, columnIndex: pdb.columnIndex,
    nodes: pdb.nodes.map((n) => ({ t: n.tableKey, ts: n.ts, f2p: n.f2pNbrIdxs, fts: n.f2pEdges.map((e) => e.ts), task: n.isTaskNode, tni: n.tableNameIdx,
      cells: n.cells.map((c) => [c.colNameIdx, c.sem, c.number, c.textIdx, c.datetime, c.boolean]) })),
    p2f: pdb.p2fAdj.map((es) => es.map((e) => [e.node, e.tableNameIdx, e.tableType, e.ts])),
  };
  const ctxs: any[] = [];
  const pick = new Rng(12345);
  for (const task of db.tasks) {
    for (const split of ['Train', 'Val', 'Test'] as const) {
      const seeds = listSeeds(pdb, task, split);
      const chosen = new Set<number>(seeds.slice(0, 8));
      for (let i = 0; i < 12; i++) chosen.add(seeds[pick.int(seeds.length)]);
      for (const sd of chosen) {
        for (const [L, w, rs] of [[128, 16, -1], [512, 3, -1], [64, 1, 7], [1024, 256, 3]] as const) {
          const p = sampleParamsFor(pdb, task, sd, { L, w, ...(rs >= 0 ? { rngSeed: rs + sd } : {}) });
          const ctx = sampleContext(pdb, p, { recordEvents: false });
          const m = buildMasks(ctx.tokens);
          // compact mask digests: allowed (q,k) counts per row
          const dig = (a: Uint8Array) => { const r: number[] = []; for (let q = 0; q < L; q++) { let c = 0, h = 0; for (let k = 0; k < L; k++) if (a[q * L + k]) { c++; h = (h * 31 + k) % 1000000007; } r.push(c * 1000000007 + h); } return r; };
          ctxs.push({ task: task.name, targetCol: task.targetCol, split, seed: sd, L, w, rngSeed: p.rngSeed, target: p.targetColNameIdx, drop: p.dropColNameIdxs,
            tok: ctx.tokens.map((t) => [t.nodeIdx, t.colNameIdx, t.mask ? 1 : 0, t.isPadding ? 1 : 0, t.f2pNbrIdxs, t.tableNameIdx, t.sem, t.textIdx, t.classValueIdx, t.isTaskNode ? 1 : 0, t.numberValue, t.datetimeValue, t.booleanValue]),
            visited: ctx.visited.map((v) => [v.node, v.depth]),
            masks: { col: dig(m.col), feat: dig(m.feat), nbr: dig(m.nbr), full: dig(m.full) } });
        }
      }
      if (task.kind === 'autocomplete') break;
    }
  }
  out[name] = { raw, pre, ctxs };
  console.log(name, 'contexts', ctxs.length);
}
const OUT = process.argv[2] ?? 'python/data/sampler-dump.json';
writeFileSync(OUT, JSON.stringify(out));
console.log('→', OUT);
