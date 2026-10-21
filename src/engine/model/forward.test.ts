/**
 * Forward-pass tests.
 *
 * 1. Parity with PyTorch: tests/fixtures/parity-random.* (written by python/parity.py) hold a random
 *    but structurally valid B=2, S=128 batch, the weights, the dense masks of the OFFICIAL
 *    PE/rt/model.py and python/rt_model.py's activations (which match the official class to ~1e-6).
 *    The TS forward must match every activation to ≤ 1e-4.
 * 2. Model invariants from spec §12 (padding, empty neighbour rows, permutation, hidden label,
 *    ablations, lean capture, error cases, speed).
 * 3. Parity with PyTorch on TRAINED weights and REAL contexts: tests/fixtures/parity-trained*.*
 *    (python/parity.py --stage trained) hold contexts exported by scripts/export-contexts.ts (the default
 *    shop/customer-churn Val context with pretrain-churn; a shop/product-sales Val context with
 *    pretrain-sales; a forum/post-votes Val context with pretrain-final) and python/rt_model.py's
 *    activations (which match the official PE/rt/model.py class to ~1e-6). The TS engine must rebuild
 *    exactly each context and the TS forward (with the real MiniLM embeddings) must match every
 *    activation to ≤ 1e-4.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import { ATTN_ORDER, SemType, type Token } from '../types';
import { Rng } from '../rng';
import { buildMasks } from '../masks';
import { createEmbeddingStore, type EmbeddingStore } from '../embeddings';
import { MINI_CONFIG, parameterShapes } from './config';
import { forward, type RTForwardResult } from './forward';
import { maxAbsDiff } from './ops';
import { initWeights, loadWeights, parseWeights, type WeightsManifest } from './weights';
import { readFileBuffer, readFileJson } from './io';
import { loadEmbeddingStore } from '../embeddings';
import { simulate } from '../pipeline';
import type { RTWeights } from '../types';
import type { DbName } from '../db/registry';

interface Ref {
  offset: number;
  length: number;
}
interface FixtureSeq {
  description: string;
  tokens: Token[];
  numReal: number;
  targetPos: number;
  masks: Record<string, string>;
  expected: {
    x0: Ref;
    sublayers: Record<string, Ref>;
    xFinal: Ref;
    yhat: { number: Ref; datetime: Ref; boolean: Ref };
    loss: number;
    prediction: number;
    predictionHead: string;
  };
}
interface Fixture {
  meta: Record<string, unknown>;
  weights: string;
  S: number;
  text: string[];
  embeddings: Ref & { dim: number };
  sublayerOrder: string[];
  sequences: FixtureSeq[];
  batchLoss: number;
}

const TOL = 1e-4;
const FIXTURES = `${(globalThis as { process?: { cwd(): string } }).process!.cwd()}/tests/fixtures`;

let fx: Fixture;
let bin: Float32Array;
let weights: RTWeights;
let store: EmbeddingStore;
let pdb: { text: string[] };
const ref = (r: Ref) => bin.subarray(r.offset, r.offset + r.length);

function unpackMask(b64: string, S: number): Uint8Array {
  const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const out = new Uint8Array(S * S);
  for (let i = 0; i < S * S; i++) out[i] = (raw[i >> 3] >> (i & 7)) & 1;
  return out;
}

const cloneTokens = (t: Token[]) => t.map((x) => ({ ...x, f2pNbrIdxs: x.f2pNbrIdxs.slice() }));
const row = (a: Float32Array, i: number, d: number) => a.subarray(i * d, (i + 1) * d);

beforeAll(async () => {
  fx = await readFileJson<Fixture>(`${FIXTURES}/parity-random.json`);
  bin = new Float32Array(await readFileBuffer(`${FIXTURES}/parity-random.bin`));
  const manifest = await readFileJson<WeightsManifest>(`${FIXTURES}/${fx.weights}.json`);
  weights = parseWeights(manifest, await readFileBuffer(`${FIXTURES}/${fx.weights}.bin`));
  store = createEmbeddingStore(fx.text, ref(fx.embeddings).slice(), fx.embeddings.dim);
  pdb = { text: fx.text };
});

describe('parity with PyTorch (python/parity.py fixture)', () => {
  it('fixture is the official model vs port at ≤ 1e-5', () => {
    const r = fx.meta.officialVsPort as Record<string, number | string>;
    for (const [k, v] of Object.entries(r)) if (k.startsWith('maxAbsDiff') || k.startsWith('absDiff')) expect(v as number).toBeLessThanOrEqual(1e-5);
    expect(fx.sequences.length).toBe(2);
  });

  it('parameterShapes lists tensors in the OFFICIAL state_dict order (fixture = official.state_dict())', async () => {
    // parity.py exports `official.state_dict()` of PE/rt/model.py in its own order. nn.ParameterDict
    // built from a plain dict SORTS its keys (torch/nn/modules/container.py ParameterDict.update), so
    // mask_embs come out boolean, datetime, number, text — not in the model.py:144 list order.
    const manifest = await readFileJson<WeightsManifest>(`${FIXTURES}/${fx.weights}.json`);
    expect(parameterShapes(manifest.config).map(([n, s]) => `${n}:${s}`)).toEqual(
      Object.entries(manifest.tensors).map(([n, t]) => `${n}:${t.shape}`),
    );
  });

  it('buildMasks (src/engine/masks.ts) equals the official dense masks (model.py:164-191)', () => {
    for (const seq of fx.sequences) {
      const m = buildMasks(seq.tokens);
      for (const l of ATTN_ORDER) expect(maxAbsDiff(m[l], unpackMask(seq.masks[l], fx.S))).toBe(0);
    }
  });

  it('x0, every sublayer, xFinal, yhat, loss and prediction match to ≤ 1e-4', () => {
    const report: Record<string, number> = {};
    for (const [si, seq] of fx.sequences.entries()) {
      const r = forward(seq.tokens, pdb, store, weights);
      const e = seq.expected;
      const diffs: Record<string, number> = { x0: maxAbsDiff(r.embedding.x0, ref(e.x0)) };
      for (const [b, blk] of r.blocks.entries()) {
        for (const l of ATTN_ORDER) diffs[`block${b}.${l}`] = maxAbsDiff(blk.attn[l].xOut, ref(e.sublayers[`block${b}.${l}`]));
        diffs[`block${b}.ffn`] = maxAbsDiff(blk.ffn.xOut, ref(e.sublayers[`block${b}.ffn`]));
      }
      diffs.xFinal = maxAbsDiff(r.xFinal, ref(e.xFinal));
      for (const h of ['number', 'datetime', 'boolean'] as const) diffs[`yhat.${h}`] = maxAbsDiff(r.yhat[h], ref(e.yhat[h]));
      diffs.loss = Math.abs(r.loss - e.loss);
      diffs.prediction = Math.abs(r.prediction - e.prediction);
      expect(r.targetPos).toBe(seq.targetPos);
      for (const [k, v] of Object.entries(diffs)) {
        expect(v, `seq ${si} ${k}`).toBeLessThanOrEqual(TOL);
        report[k] = Math.max(report[k] ?? 0, v);
      }
    }
    console.log('[parity] max |TS − PyTorch| per stage:', JSON.stringify(report));
  });

  it('per-sequence losses recombine to the official batch loss (mean over all masked cells)', () => {
    let sum = 0;
    let n = 0;
    for (const seq of fx.sequences) {
      const r = forward(seq.tokens, pdb, store, weights);
      sum += r.lossTerms.reduce((a, t) => a + t.loss, 0);
      n += r.lossTerms.length;
    }
    expect(n).toBe(3); // number target, extra masked datetime cell, boolean target
    expect(Math.abs(sum / n - fx.batchLoss)).toBeLessThanOrEqual(TOL);
  });
});

describe('model invariants (spec §12)', () => {
  let base: RTForwardResult[];
  beforeAll(() => {
    base = fx.sequences.map((s) => forward(s.tokens, pdb, store, weights));
  });

  it('padding stays exactly 0 through every sublayer and the final norm', () => {
    const d = weights.config.dModel;
    for (const [si, seq] of fx.sequences.entries()) {
      const r = base[si];
      const pads = seq.tokens.flatMap((t, i) => (t.isPadding ? [i] : []));
      expect(pads.length).toBeGreaterThan(0);
      const tensors = [r.embedding.x0, r.xFinal, ...r.blocks.flatMap((b) => [...ATTN_ORDER.map((l) => b.attn[l].xOut), b.ffn.xOut])];
      for (const t of tensors) for (const p of pads) expect(row(t, p, d).every((v) => v === 0)).toBe(true);
    }
  });

  it('padding has no influence on real tokens (same outputs with the padding tail removed)', () => {
    const d = weights.config.dModel;
    for (const [si, seq] of fx.sequences.entries()) {
      const real = seq.tokens.filter((t) => !t.isPadding);
      const r = forward(real, pdb, store, weights);
      expect(maxAbsDiff(r.xFinal, base[si].xFinal.subarray(0, real.length * d))).toBeLessThanOrEqual(1e-6);
      expect(Math.abs(r.prediction - base[si].prediction)).toBeLessThanOrEqual(1e-6);
    }
  });

  it('a query with an empty neighbour row gets exactly 0 update (and probs row 0)', () => {
    const d = weights.config.dModel;
    const S = fx.S;
    let nEmpty = 0;
    let nNonEmpty = 0;
    for (const [si, seq] of fx.sequences.entries()) {
      const r = base[si];
      for (let q = 0; q < S; q++) {
        if (seq.tokens[q].isPadding) continue;
        let any = false;
        for (let k = 0; k < S; k++) any ||= r.masks.nbr[q * S + k] === 1;
        if (any) {
          nNonEmpty++;
          continue;
        }
        nEmpty++;
        for (const blk of r.blocks) {
          const a = blk.attn.nbr;
          expect(row(a.out, q, d).every((v) => v === 0)).toBe(true);
          expect(maxAbsDiff(row(a.xOut, q, d), row(a.xIn, q, d))).toBe(0);
          for (let h = 0; h < weights.config.nHeads; h++)
            expect(a.probs.subarray(h * S * S + q * S, h * S * S + (q + 1) * S).every((v) => v === 0)).toBe(true);
        }
      }
    }
    expect(nEmpty).toBeGreaterThan(0);
    expect(nNonEmpty).toBeGreaterThan(0);
  });

  it('permuting the tokens permutes every output; the prediction is invariant (no positional encoding)', () => {
    const d = weights.config.dModel;
    for (const [si, seq] of fx.sequences.entries()) {
      const perm = new Rng(100 + si).shuffle(Array.from({ length: fx.S }, (_, i) => i)); // new pos i ← old perm[i]
      const toks = perm.map((old, i) => ({ ...seq.tokens[old], pos: i }));
      const r = forward(toks, pdb, store, weights);
      const r0 = base[si];
      let m = 0;
      for (let i = 0; i < fx.S; i++) {
        m = Math.max(m, maxAbsDiff(row(r.xFinal, i, d), row(r0.xFinal, perm[i], d)));
        expect(Math.abs(r.yhat.boolean[i] - r0.yhat.boolean[perm[i]])).toBeLessThan(1e-5);
      }
      expect(m).toBeLessThan(1e-5);
      expect(perm[r.targetPos]).toBe(r0.targetPos);
      expect(Math.abs(r.prediction - r0.prediction)).toBeLessThan(1e-5);
      expect(Math.abs(r.loss - r0.loss)).toBeLessThan(1e-5);
    }
  });

  it('changing the hidden label of the masked target does not change the prediction (only the loss)', () => {
    for (const [si, seq] of fx.sequences.entries()) {
      const toks = cloneTokens(seq.tokens);
      const t = toks[seq.targetPos];
      if (t.sem === SemType.Number) t.numberValue = -7.25;
      else t.booleanValue = -t.booleanValue;
      const r = forward(toks, pdb, store, weights);
      expect(r.prediction).toBe(base[si].prediction); // bit-identical
      expect(maxAbsDiff(r.xFinal, base[si].xFinal)).toBe(0);
      expect(r.loss).not.toBe(base[si].loss);
    }
  });

  it('a NON-FINITE hidden label is rejected, not silently ignored (official: NaN·0 = NaN, spec §5.3)', () => {
    // model.py:216-224 multiplies EVERY type's value term by a 0/1 gate, so a NaN/inf in any value
    // slot (the masked target's hidden label included) makes x NaN in the official model; a finite
    // "unchanged" prediction here would contradict it. The engine must throw instead.
    for (const seq of fx.sequences) {
      const toks = cloneTokens(seq.tokens);
      const t = toks[seq.targetPos];
      expect(t.mask).toBe(true);
      if (t.sem === SemType.Number) t.numberValue = NaN;
      else t.booleanValue = Infinity;
      expect(() => forward(toks, pdb, store, weights)).toThrow(/non-finite/);
    }
  });

  it('a non-finite value in a slot of ANOTHER sem type is rejected too (it is gated by ×0, not skipped)', () => {
    const toks = cloneTokens(fx.sequences[1].tokens);
    const i = toks.findIndex((t) => !t.isPadding && !t.mask && t.sem === SemType.Text);
    toks[i].numberValue = NaN; // a text cell's (unused) number slot
    expect(() => forward(toks, pdb, store, weights)).toThrow(/non-finite/);
  });

  it('disabledLayers skips the sublayer in every block (residual unchanged, skipped = true)', () => {
    const seq = fx.sequences[1];
    const r = forward(seq.tokens, pdb, store, weights, { disabledLayers: ['nbr', 'full'] });
    for (const blk of r.blocks) {
      for (const l of ['nbr', 'full'] as const) {
        expect(blk.attn[l].skipped).toBe(true);
        expect(maxAbsDiff(blk.attn[l].xOut, blk.attn[l].xIn)).toBe(0);
        expect(blk.attn[l].out.every((v) => v === 0)).toBe(true);
      }
      expect(blk.attn.col.skipped).toBe(false);
      expect(blk.attn.feat.skipped).toBe(false);
      expect(maxAbsDiff(blk.attn.nbr.xIn, blk.attn.feat.xOut)).toBe(0);
      expect(maxAbsDiff(blk.ffn.xIn, blk.attn.feat.xOut)).toBe(0);
    }
    expect(r.prediction).not.toBe(base[1].prediction);
    expect(Number.isFinite(r.prediction)).toBe(true);
  });

  it('capture=false gives the same outputs without the per-layer internals', () => {
    for (const [si, seq] of fx.sequences.entries()) {
      const r = forward(seq.tokens, pdb, store, weights, { capture: false });
      expect(maxAbsDiff(r.xFinal, base[si].xFinal)).toBe(0);
      expect(r.prediction).toBe(base[si].prediction);
      expect(r.loss).toBe(base[si].loss);
      expect(r.blocks[0].attn.col.probs.length).toBe(0);
      expect(r.blocks[0].ffn.gated.length).toBe(0);
    }
  });

  it('embedding parts: x0 = colTerm + valueTerm; masked tokens use the raw mask vector', () => {
    const d = weights.config.dModel;
    const seq = fx.sequences[0];
    const e = base[0].embedding;
    for (const [i, t] of seq.tokens.entries()) {
      if (t.isPadding) continue;
      for (let j = 0; j < d; j++) expect(e.x0[i * d + j]).toBe(Math.fround(e.colTerm[i * d + j] + e.valueTerm[i * d + j]));
      if (t.mask) {
        const key = ['number', 'text', 'datetime', 'boolean'][t.sem];
        expect(maxAbsDiff(row(e.valueTerm, i, d), weights.tensors[`mask_embs.${key}`].data)).toBe(0);
        expect(row(e.valuePreNorm, i, d).every((v) => v === 0)).toBe(true);
      }
    }
  });

  it('a masked text cell throws "masking text not supported" (model.py:257)', () => {
    const toks = cloneTokens(fx.sequences[0].tokens);
    const ti = toks.findIndex((t) => !t.isPadding && t.sem === SemType.Text);
    toks[ti].mask = true;
    expect(() => forward(toks, pdb, store, weights)).toThrow(/masking text not supported/);
  });

  it('an unknown string throws instead of fabricating an embedding', () => {
    const small = createEmbeddingStore(fx.text.slice(0, 3), new Float32Array(3 * 384), 384);
    expect(() => forward(fx.sequences[0].tokens, pdb, small, weights)).toThrow(/no precomputed embedding/);
  });
});

describe('performance', () => {
  it('timing: S=128, d=64, 4 blocks (target ~50 ms in Node)', () => {
    const w = initWeights(MINI_CONFIG, 0);
    const toks = fx.sequences[0].tokens;
    for (let i = 0; i < 3; i++) forward(toks, pdb, store, w); // warm-up (JIT)
    const time = (capture: boolean) => {
      const n = 10;
      const t0 = performance.now();
      for (let i = 0; i < n; i++) forward(toks, pdb, store, w, { capture });
      return (performance.now() - t0) / n;
    };
    const full = time(true);
    const lean = time(false);
    console.log(`[perf] forward S=128 d=64 4 blocks: capture=true ${full.toFixed(1)} ms, capture=false ${lean.toFixed(1)} ms`);
    expect(full).toBeLessThan(250); // generous bound for slow CI; target ~50 ms
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Trained checkpoints + real contexts (python/parity.py --stage trained)
//    parity-trained        shop/customer-churn Val (default seed), pretrain-churn, own weights copy
//    parity-trained-sales  shop/product-sales Val, pretrain-sales (number head)
//    parity-trained-forum  forum/post-votes Val, pretrain-final (a pretraining DB)
//    The extra fixtures use the shipped public/data/weights/<checkpoint> (sha256 checked).
// ─────────────────────────────────────────────────────────────────────────────
interface TrainedFixture {
  meta: Record<string, unknown> & { officialVsPort: Record<string, number> };
  dbName?: string;
  taskId: string;
  split: 'Train' | 'Val' | 'Test';
  seedNode: number;
  L: number;
  w: number;
  rngSeed: number;
  checkpoint: string;
  checkpointSha256: string;
  /** Fixture copy of the weights (tests/fixtures/<weights>.*), or null = public/data/weights/<checkpoint>. */
  weights: string | null;
  S: number;
  targetPos: number;
  batch: {
    nodeIdxs: number[];
    f2pNbrIdxs: number[][];
    tableNameIdxs: number[];
    colNameIdxs: number[];
    semTypes: number[];
    values: number[];
    textIdxs: number[];
    flags: number[];
  };
  masks: Record<string, string>;
  sublayerOrder: string[];
  expected: FixtureSeq['expected'] & { label: number };
}

/** Names of the trained fixtures present in tests/fixtures (parity-trained*.json, weights files excluded). */
const trainedFixtures: string[] = (() => {
  try {
    const proc = (globalThis as { process?: { getBuiltinModule?(m: string): unknown } }).process;
    const fs = proc?.getBuiltinModule?.('node:fs') as { readdirSync(p: string): string[] } | undefined;
    if (!fs) return ['parity-trained'];
    return fs
      .readdirSync(FIXTURES)
      .filter((n) => /^parity-trained(-[a-z]+)?\.json$/.test(n) && !n.endsWith('-weights.json'))
      .map((n) => n.replace(/\.json$/, ''))
      .sort();
  } catch {
    return ['parity-trained'];
  }
})();

async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const d = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

it('the default trained fixture (parity-trained) is present', () => {
  expect(trainedFixtures).toContain('parity-trained');
});

describe.each(trainedFixtures)('parity with PyTorch on TRAINED weights and a real context (%s)', (name) => {
  let tf: TrainedFixture;
  let tbin: Float32Array;
  let tw: RTWeights;
  let realStore: EmbeddingStore;
  const tref = (r: Ref) => tbin.subarray(r.offset, r.offset + r.length);

  beforeAll(async () => {
    tf = await readFileJson<TrainedFixture>(`${FIXTURES}/${name}.json`);
    tbin = new Float32Array(await readFileBuffer(`${FIXTURES}/${name}.bin`));
    tw = tf.weights
      ? parseWeights(await readFileJson<WeightsManifest>(`${FIXTURES}/${tf.weights}.json`), await readFileBuffer(`${FIXTURES}/${tf.weights}.bin`))
      : await loadWeights(tf.checkpoint); // public/data/weights/<checkpoint>
    realStore = await loadEmbeddingStore(); // public/data/embeddings (real all-MiniLM-L12-v2 vectors)
  });

  const sim = () =>
    simulate({
      dbName: (tf.dbName ?? 'shop') as DbName,
      taskId: tf.taskId,
      split: tf.split,
      seedNode: tf.seedNode,
      L: tf.L,
      w: tf.w,
      rngSeed: tf.rngSeed,
    });

  it('fixture: the official model.py class equals the port on this context and on every trained checkpoint (≤ 1e-5)', () => {
    for (const [k, v] of Object.entries(tf.meta.officialVsPort)) if (k.startsWith('maxAbsDiff') || k.startsWith('absDiff')) expect(v, k).toBeLessThanOrEqual(1e-5);
    const ck = (tf.meta.officialVsPortCheckpoints as { checkpoints: Record<string, Record<string, number> | string> }).checkpoints;
    expect(Object.keys(ck).length).toBeGreaterThan(0);
    for (const [cname, r] of Object.entries(ck)) {
      if (typeof r === 'string') continue; // a checkpoint missing at fixture time
      for (const [k, v] of Object.entries(r)) if (k.startsWith('maxAbsDiff') || k.startsWith('absDiff')) expect(v, `${cname} ${k}`).toBeLessThanOrEqual(1e-5);
    }
  });

  it('the TS engine (preprocess + sampler) rebuilds exactly the exported context', () => {
    const s = sim();
    expect(s.seedNode).toBe(tf.seedNode);
    expect(s.ctx.tokens.length).toBe(tf.S);
    expect(s.ctx.targetPos).toBe(tf.targetPos);
    const b = tf.batch;
    for (const [i, t] of s.ctx.tokens.entries()) {
      expect(t.nodeIdx, `pos ${i} node`).toBe(b.nodeIdxs[i]);
      expect(t.f2pNbrIdxs, `pos ${i} f2p`).toEqual(b.f2pNbrIdxs[i]);
      expect(t.tableNameIdx, `pos ${i} table`).toBe(b.tableNameIdxs[i]);
      expect(t.colNameIdx, `pos ${i} col`).toBe(b.colNameIdxs[i]);
      expect(t.sem, `pos ${i} sem`).toBe(b.semTypes[i]);
      const own = t.sem === SemType.Number ? t.numberValue : t.sem === SemType.DateTime ? t.datetimeValue : t.sem === SemType.Boolean ? t.booleanValue : 0;
      expect(Math.fround(own), `pos ${i} value`).toBe(b.values[i]);
      expect(t.isPadding ? -1 : t.textIdx, `pos ${i} text`).toBe(b.textIdxs[i]);
      const flags = (t.mask ? 1 : 0) | (t.isTarget ? 2 : 0) | (t.isPadding ? 4 : 0) | (t.isTaskNode ? 8 : 0);
      expect(flags, `pos ${i} flags`).toBe(b.flags[i]);
    }
    // the masks of the rebuilt context equal the official dense masks of the exported batch
    const m = buildMasks(s.ctx.tokens);
    for (const l of ATTN_ORDER) expect(maxAbsDiff(m[l], unpackMask(tf.masks[l], tf.S)), l).toBe(0);
  });

  it('x0, every sublayer, xFinal, yhat, loss and prediction match PyTorch to ≤ 1e-4 (trained weights, real embeddings)', () => {
    const s = sim();
    const r = forward(s.ctx.tokens, s.pdb, realStore, tw);
    const e = tf.expected;
    const diffs: Record<string, number> = { x0: maxAbsDiff(r.embedding.x0, tref(e.x0)) };
    for (const [b, blk] of r.blocks.entries()) {
      for (const l of ATTN_ORDER) diffs[`block${b}.${l}`] = maxAbsDiff(blk.attn[l].xOut, tref(e.sublayers[`block${b}.${l}`]));
      diffs[`block${b}.ffn`] = maxAbsDiff(blk.ffn.xOut, tref(e.sublayers[`block${b}.ffn`]));
    }
    diffs.xFinal = maxAbsDiff(r.xFinal, tref(e.xFinal));
    for (const h of ['number', 'datetime', 'boolean'] as const) diffs[`yhat.${h}`] = maxAbsDiff(r.yhat[h], tref(e.yhat[h]));
    diffs.loss = Math.abs(r.loss - e.loss);
    diffs.prediction = Math.abs(r.prediction - e.prediction);
    expect(r.targetPos).toBe(tf.targetPos);
    for (const [k, v] of Object.entries(diffs)) expect(v, k).toBeLessThanOrEqual(TOL);
    // the hidden label read from the target token is the exported one
    const tgt = s.ctx.tokens[tf.targetPos];
    expect(Math.fround(e.predictionHead === 'boolean' ? tgt.booleanValue : tgt.numberValue)).toBe(e.label);
    console.log(`[${name}] max |TS − PyTorch|:`, JSON.stringify(Object.fromEntries(Object.entries(diffs).map(([k, v]) => [k, Number(v.toExponential(2))]))));
  });

  it('the weights are byte-identical to the shipped public checkpoint', async () => {
    const pubBin = await readFileBuffer(`${FIXTURES}/../../public/data/weights/${tf.checkpoint}.bin`);
    expect(await sha256Hex(pubBin)).toBe(tf.checkpointSha256);
    const pub = parseWeights(await readFileJson<WeightsManifest>(`${FIXTURES}/../../public/data/weights/${tf.checkpoint}.json`), pubBin);
    for (const [tname, t] of Object.entries(tw.tensors)) expect(maxAbsDiff(pub.tensors[tname].data, t.data), tname).toBe(0);
  });
});
