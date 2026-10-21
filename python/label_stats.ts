/**
 * Label persistence of every forecast task (why zero-shot transfer can work, spec §11.2: the target
 * entity's past labels drive it). Written to python/data/label-stats.json; `train.py export` copies it
 * into public/data/training/results.json → labelPersistence.
 *
 * Per task:
 *   lag1            Pearson correlation of an entity's labels in consecutive Train windows (t_{k−1}, t_k),
 *                   pooled over all entities and Train timestamps;
 *   entityMean      EntityMean baseline per split (paper App. I: mean of the entity's earlier labels of the
 *                   same task table, any split, strictly earlier timestamp; fallback = Train mean), scored
 *                   with the task metric (AUROC for binary, R² for regression). Same definition as
 *                   scripts/export-contexts.ts, which computes it per eval seed;
 *   rows, positiveRate (binary) / mean and zero share (regression) per split.
 *
 * Usage: npx tsx python/label_stats.ts [--out python/data/label-stats.json]
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { DB_NAMES, getDatabase } from '../src/engine/db/index.ts';
import type { Split, TaskDef } from '../src/engine/types.ts';

const SPLITS: Split[] = ['Train', 'Val', 'Test'];
const outIdx = process.argv.indexOf('--out');
const OUT = resolve(outIdx >= 0 ? process.argv[outIdx + 1] : 'python/data/label-stats.json');

interface Row {
  split: Split;
  t: number;
  e: number;
  y: number;
}

const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length;

function pearson(a: number[], b: number[]): number {
  const ma = mean(a);
  const mb = mean(b);
  let ab = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i++) {
    ab += (a[i] - ma) * (b[i] - mb);
    aa += (a[i] - ma) ** 2;
    bb += (b[i] - mb) ** 2;
  }
  return ab / Math.sqrt(aa * bb);
}

function r2(y: number[], p: number[]): number {
  const m = mean(y);
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

const r4 = (x: number) => Math.round(x * 1e4) / 1e4;

function stats(task: TaskDef) {
  const rows: Row[] = SPLITS.flatMap((split) =>
    task.tables![split].rows.map((r) => {
      const v = r[task.targetCol];
      return { split, t: r.timestamp as number, e: r[task.entityCol!] as number, y: typeof v === 'boolean' ? (v ? 1 : 0) : (v as number) };
    }),
  );
  const train = rows.filter((r) => r.split === 'Train');
  const trainMean = mean(train.map((r) => r.y));
  const byEntity = new Map<number, Row[]>();
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
  const binary = task.targetType === 'binary';
  const perSplit: Record<string, unknown> = {};
  const entityMean: Record<string, number> = {};
  for (const split of SPLITS) {
    const rs = rows.filter((r) => r.split === split);
    const y = rs.map((r) => r.y);
    const pred = rs.map((r) => {
      const past = byEntity.get(r.e)!.filter((x) => x.t < r.t);
      return past.length ? mean(past.map((x) => x.y)) : trainMean;
    });
    entityMean[split] = r4(binary ? auroc(y, pred) : r2(y, pred));
    perSplit[split] = binary
      ? { rows: rs.length, positiveRate: r4(mean(y)) }
      : { rows: rs.length, mean: r4(mean(y)), zeroShare: r4(y.filter((v) => v === 0).length / y.length) };
  }
  return {
    db: task.dbName,
    targetType: task.targetType,
    metric: binary ? 'auroc' : 'r2',
    timedeltaDays: task.timedeltaDays,
    trainTimestamps: stamps.length,
    lag1: r4(pearson(prev, next)),
    lag1Pairs: prev.length,
    entityMean,
    splits: perSplit,
  };
}

const tasks: Record<string, unknown> = {};
for (const name of DB_NAMES) for (const task of getDatabase(name).tasks) if (task.kind === 'forecast') tasks[task.id] = stats(task);
const out = {
  note:
    'lag1 = Pearson correlation of an entity\'s labels in consecutive Train windows; entityMean = EntityMean baseline ' +
    '(paper App. I) per split, AUROC (binary) or R² (regression). Computed from the generated databases by python/label_stats.ts.',
  tasks,
};
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(out, null, 1));
for (const [id, s] of Object.entries(tasks)) {
  const x = s as { lag1: number; entityMean: Record<string, number> };
  console.log(`${id.padEnd(28)} lag1 ${x.lag1.toFixed(3)}  EntityMean Train ${x.entityMean.Train.toFixed(3)}  Val ${x.entityMean.Val.toFixed(3)}  Test ${x.entityMean.Test.toFixed(3)}`);
}
console.log(`→ ${OUT}`);
