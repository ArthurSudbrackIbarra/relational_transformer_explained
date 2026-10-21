/**
 * TS forward (src/engine/model/forward.ts) vs PyTorch (python/rt_model.py, itself == the official
 * PE/rt/model.py to ~1e-6) on every parity fixture in tests/fixtures, as numbers for
 * public/data/verification.json. src/engine/model/forward.test.ts asserts the same comparisons
 * (≤ 1e-4); this script only records the maxima.
 *
 *   parity-random*         synthetic context, random weights, synthetic embeddings (2 sequences)
 *   parity-trained*        real contexts rebuilt by the TS engine, trained checkpoints, real MiniLM vectors
 *
 * Usage: npx tsx python/ts_parity.ts [--out python/data/ts-parity.json]
 */
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createEmbeddingStore, loadEmbeddingStore } from '../src/engine/embeddings.ts';
import { forward } from '../src/engine/model/forward.ts';
import { maxAbsDiff } from '../src/engine/model/ops.ts';
import { parseWeights, type WeightsManifest } from '../src/engine/model/weights.ts';
import { simulate } from '../src/engine/pipeline.ts';
import { ATTN_ORDER, type RTWeights, type Token } from '../src/engine/types.ts';
import type { DbName } from '../src/engine/db/index.ts';

const outIdx = process.argv.indexOf('--out');
const OUT = resolve(outIdx >= 0 ? process.argv[outIdx + 1] : 'python/data/ts-parity.json');
const FIX = resolve('tests/fixtures');
const WEIGHTS = resolve('public/data/weights');

interface Ref {
  offset: number;
  length: number;
}
interface Expected {
  x0: Ref;
  sublayers: Record<string, Ref>;
  xFinal: Ref;
  yhat: Record<'number' | 'datetime' | 'boolean', Ref>;
  loss: number;
  prediction: number;
}

const readJson = <T>(p: string): T => JSON.parse(readFileSync(p, 'utf8')) as T;
const readF32 = (p: string) => {
  const b = readFileSync(p);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
const loadWeightsFile = (prefix: string): RTWeights => {
  const b = readFileSync(`${prefix}.bin`);
  return parseWeights(readJson<WeightsManifest>(`${prefix}.json`), b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};

function diffs(r: ReturnType<typeof forward>, e: Expected, bin: Float32Array): Record<string, number> {
  const ref = (x: Ref) => bin.subarray(x.offset, x.offset + x.length);
  const d: Record<string, number> = { x0: maxAbsDiff(r.embedding.x0, ref(e.x0)) };
  for (const [b, blk] of r.blocks.entries()) {
    for (const l of ATTN_ORDER) d[`block${b}.${l}`] = maxAbsDiff(blk.attn[l].xOut, ref(e.sublayers[`block${b}.${l}`]));
    d[`block${b}.ffn`] = maxAbsDiff(blk.ffn.xOut, ref(e.sublayers[`block${b}.ffn`]));
  }
  d.xFinal = maxAbsDiff(r.xFinal, ref(e.xFinal));
  for (const h of ['number', 'datetime', 'boolean'] as const) d[`yhat.${h}`] = maxAbsDiff(r.yhat[h], ref(e.yhat[h]));
  d.loss = Math.abs(r.loss - e.loss);
  d.prediction = Math.abs(r.prediction - e.prediction);
  return d;
}

const result: Record<string, { maxAbsDiff: number; contexts: number; checkpoint?: string; worstStage: string }> = {};

// 1. random-weight fixture
{
  const fx = readJson<{
    weights: string;
    text: string[];
    embeddings: Ref & { dim: number };
    sequences: { tokens: Token[]; expected: Expected }[];
  }>(`${FIX}/parity-random.json`);
  const bin = readF32(`${FIX}/parity-random.bin`);
  const w = loadWeightsFile(`${FIX}/${fx.weights}`);
  const store = createEmbeddingStore(fx.text, bin.slice(fx.embeddings.offset, fx.embeddings.offset + fx.embeddings.length), fx.embeddings.dim);
  let worst = 0;
  let stage = '';
  for (const seq of fx.sequences) {
    for (const [k, v] of Object.entries(diffs(forward(seq.tokens, { text: fx.text }, store, w), seq.expected, bin))) {
      if (v > worst) [worst, stage] = [v, k];
    }
  }
  result['parity-random'] = { maxAbsDiff: worst, contexts: fx.sequences.length, worstStage: stage };
}

// 2. trained fixtures: the TS engine rebuilds the context, the TS forward runs with real embeddings
const realStore = await loadEmbeddingStore();
for (const f of readdirSync(FIX).filter((n) => /^parity-trained(-[a-z]+)?\.json$/.test(n) && !n.endsWith('-weights.json'))) {
  const name = f.replace(/\.json$/, '');
  const tf = readJson<{
    dbName?: string;
    taskId: string;
    split: 'Train' | 'Val' | 'Test';
    seedNode: number;
    L: number;
    w: number;
    rngSeed: number;
    checkpoint: string;
    weights: string | null;
    targetPos: number;
    expected: Expected;
  }>(`${FIX}/${f}`);
  const bin = readF32(`${FIX}/${name}.bin`);
  const w = tf.weights ? loadWeightsFile(`${FIX}/${tf.weights}`) : loadWeightsFile(`${WEIGHTS}/${tf.checkpoint}`);
  const s = simulate({
    dbName: (tf.dbName ?? tf.taskId.split('/')[0]) as DbName,
    taskId: tf.taskId,
    split: tf.split,
    seedNode: tf.seedNode,
    L: tf.L,
    w: tf.w,
    rngSeed: tf.rngSeed,
  });
  if (s.seedNode !== tf.seedNode || s.ctx.targetPos !== tf.targetPos) throw new Error(`${name}: the TS engine did not rebuild the fixture context`);
  let worst = 0;
  let stage = '';
  for (const [k, v] of Object.entries(diffs(forward(s.ctx.tokens, s.pdb, realStore, w), tf.expected, bin))) {
    if (v > worst) [worst, stage] = [v, k];
  }
  result[name] = { maxAbsDiff: worst, contexts: 1, checkpoint: tf.checkpoint, worstStage: stage };
}

const overall = Math.max(...Object.values(result).map((r) => r.maxAbsDiff));
writeFileSync(OUT, JSON.stringify({ maxAbsDiff: overall, fixtures: result }, null, 1));
for (const [k, v] of Object.entries(result)) console.log(`${k.padEnd(24)} max |TS − PyTorch| ${v.maxAbsDiff.toExponential(2)} (${v.worstStage})`);
console.log(`overall ${overall.toExponential(2)} → ${OUT}`);
