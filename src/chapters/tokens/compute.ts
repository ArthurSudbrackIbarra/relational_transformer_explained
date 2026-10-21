/**
 * Extra forward passes for the chapter's two experiments (hidden label, shuffled order) and the
 * search for a context that ends in padding. Everything runs the engine's own mirrors:
 * buildMasks (model.py:164-191), forwardWithMasks (model.py:153-267) and sampleContext
 * (fly.rs:357-511). Nothing here changes the model or the sampler; it only feeds them different
 * token arrays. The hidden-label experiment uses a literal-gate path (literalWithStoredLabel)
 * assembled from the engine's exported kernels and checked against the engine bit for bit.
 *
 * Cost: one forward ≈ 60-100 ms at L = 128, so results are cached and computed off the render
 * path (setTimeout), never in a render loop.
 */
import type { EmbeddingStore } from '@/engine/embeddings';
import { buildMasks } from '@/engine/masks';
import {
  SEM_KEYS,
  forwardWithMasks,
  linear,
  lossTerms,
  maskRows,
  maskedAttentionCore,
  meanLoss,
  rmsnorm,
  scalarValueTerm,
  silu,
  tensor,
  textEmbedder,
  type DecoderOutputs,
  type MaskRows,
  type RTForwardResult,
  type TextEmbedder,
} from '@/engine/model';
import { getContext, getMasks, makeSampleParams } from '@/engine/pipeline';
import { Rng } from '@/engine/rng';
import { sampleContext } from '@/engine/sampler';
import {
  ATTN_ORDER,
  SemType,
  type AttnLayer,
  type ForwardResult,
  type MaskSet,
  type PreprocessedDB,
  type RTWeights,
  type SampledContext,
  type Split,
  type TaskDef,
  type Token,
} from '@/engine/types';

// ─────────────────────────────────────────────────────────────────────────────
// Small identity helper for cache keys
// ─────────────────────────────────────────────────────────────────────────────

const ids = new WeakMap<object, number>();
let nextId = 1;
export function objId(o: object): number {
  let id = ids.get(o);
  if (id === undefined) ids.set(o, (id = nextId++));
  return id;
}

/** Run `fn` after the browser has painted (so a "running…" state shows first). */
export function afterPaint<T>(fn: () => T): Promise<T> {
  return new Promise((resolve, reject) => {
    const go = () => {
      try {
        resolve(fn());
      } catch (e) {
        reject(e);
      }
    };
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => setTimeout(go, 0));
    else setTimeout(go, 0);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. The hidden label: same tokens, a different value in the target's value slot
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The engine's embedTokens skips a term whose gate is 0 (identical numbers for finite values), so a
 * masked token's stored value never enters its arithmetic. To make the hidden-label experiment a
 * measurement rather than a consequence of that shortcut, this path evaluates the input embedding
 * literally as model.py:208-224 writes it: at every position, the column-name term and the value
 * term of all four types are computed from the stored slots (the hidden label included), each is
 * multiplied by its 0/1 gate, and the products are summed in the code's order with float32 rounding.
 * The resulting x₀ then goes through the blocks (model.py:226-229: col → feat → nbr → full → FFN,
 * model.py:82-86) and the decoders (model.py:238-239), built from the engine's exported kernels.
 * `literalSelfCheck` confirms that, on the sampler's own tokens, this path reproduces the engine's
 * forward pass bit for bit.
 */
export interface LiteralResult {
  /** x₀ [S·d] as the literal sum of model.py:208-224. */
  x0: Float32Array;
  xFinal: Float32Array;
  yhat: DecoderOutputs;
  prediction: number;
  loss: number;
  targetPos: number;
  ms: number;
}

const f32 = Math.fround;

/** model.py:208-224, term by term, gates included (see LiteralResult). */
function literalEmbedding(tokens: readonly Token[], textEmb: TextEmbedder, w: RTWeights): Float32Array {
  const { dModel: d, dText, eps } = w.config;
  const S = tokens.length;
  const zeros = new Float32Array(dText); // padding slots hold the zero vector Vecs::new allocated (fly.rs:73, 76)
  const normed = new Map<string, Float32Array>();
  // norm_dict[k](enc_dict[k](v)) for a 384-d input (column phrase or text embedding), cached per index.
  const term384 = (k: 'col_name' | 'text', idx: number | null) => {
    const key = `${k}|${idx ?? 'pad'}`;
    let v = normed.get(key);
    if (!v) {
      const e = idx === null ? zeros : textEmb(idx);
      const pre = linear(e, 1, dText, tensor(w, `enc_dict.${k}.weight`), d, tensor(w, `enc_dict.${k}.bias`));
      v = rmsnorm(pre, 1, d, tensor(w, `norm_dict.${k}.weight`), eps);
      normed.set(key, v);
    }
    return v;
  };
  const x0 = new Float32Array(S * d);
  const acc = new Float32Array(d);
  for (let i = 0; i < S; i++) {
    const t = tokens[i];
    const notPad = t.isPadding ? 0 : 1;
    // x = 0 + norm_dict["col_name"](enc_dict["col_name"](col_name_values)) * ~is_padding   (model.py:208-214)
    const col = term384('col_name', t.isPadding ? null : t.colNameIdx);
    for (let j = 0; j < d; j++) acc[j] = f32(0 + f32(col[j] * notPad));
    // for i, t in enumerate(["number", "text", "datetime", "boolean"])   (model.py:216-224)
    SEM_KEYS.forEach((key, s) => {
      const value =
        key === 'text'
          ? term384('text', t.isPadding ? null : t.textIdx)
          : scalarValueTerm(w, key, key === 'number' ? t.numberValue : key === 'datetime' ? t.datetimeValue : t.booleanValue).term;
      const gValue = t.sem === s && !t.mask && !t.isPadding ? 1 : 0;
      const gMask = t.sem === s && t.mask && !t.isPadding ? 1 : 0;
      const m = tensor(w, `mask_embs.${key}`);
      for (let j = 0; j < d; j++) {
        acc[j] = f32(acc[j] + f32(value[j] * gValue));
        acc[j] = f32(acc[j] + f32(m[j] * gMask));
      }
    });
    x0.set(acc, i * d);
  }
  return x0;
}

/** Blocks, final norm and decoders on a given x₀ (model.py:82-86, 226-229, 238-239). */
function blocksAndDecoders(x0: Float32Array, S: number, masks: MaskSet, w: RTWeights): { xFinal: Float32Array; yhat: DecoderOutputs } {
  const { dModel: d, nHeads, dFF, nBlocks, eps } = w.config;
  const rows = {} as Record<AttnLayer, MaskRows>;
  for (const l of ATTN_ORDER) rows[l] = maskRows(masks[l], S);
  let x = x0;
  for (let b = 0; b < nBlocks; b++) {
    for (const l of ATTN_ORDER) {
      // x = x + attns[l](norms[l](x), block_mask=block_masks[l])   (model.py:83-84; projections bias-free, model.py:25-28)
      const p = `blocks.${b}.attns.${l}`;
      const h = rmsnorm(x, S, d, tensor(w, `blocks.${b}.norms.${l}.weight`), eps);
      const q = linear(h, S, d, tensor(w, `${p}.wq.weight`), d);
      const k = linear(h, S, d, tensor(w, `${p}.wk.weight`), d);
      const v = linear(h, S, d, tensor(w, `${p}.wv.weight`), d);
      const o = linear(maskedAttentionCore(q, k, v, S, d, nHeads, rows[l], null), S, d, tensor(w, `${p}.wo.weight`), d);
      const nx = new Float32Array(S * d);
      for (let i = 0; i < S * d; i++) nx[i] = x[i] + o[i];
      x = nx;
    }
    // x = x + ffn(norms["ffn"](x)), ffn = w2(silu(w1 h) * w3 h)   (model.py:58-59, 85)
    const h = rmsnorm(x, S, d, tensor(w, `blocks.${b}.norms.ffn.weight`), eps);
    const a = linear(h, S, d, tensor(w, `blocks.${b}.ffn.w1.weight`), dFF);
    const c = linear(h, S, d, tensor(w, `blocks.${b}.ffn.w3.weight`), dFF);
    const g = new Float32Array(S * dFF);
    for (let i = 0; i < S * dFF; i++) g[i] = silu(a[i]) * c[i];
    const o = linear(g, S, dFF, tensor(w, `blocks.${b}.ffn.w2.weight`), d);
    const nx = new Float32Array(S * d);
    for (let i = 0; i < S * d; i++) nx[i] = x[i] + o[i];
    x = nx;
  }
  const xFinal = rmsnorm(x, S, d, tensor(w, 'norm_out.weight'), eps); // model.py:229
  const dec = (k: 'number' | 'datetime' | 'boolean') => linear(xFinal, S, d, tensor(w, `dec_dict.${k}.weight`), 1, tensor(w, `dec_dict.${k}.bias`));
  return { xFinal, yhat: { number: dec('number'), datetime: dec('datetime'), boolean: dec('boolean') } };
}

function literalForward(tokens: readonly Token[], masks: MaskSet, textEmb: TextEmbedder, w: RTWeights): LiteralResult {
  const t0 = performance.now();
  const S = tokens.length;
  const x0 = literalEmbedding(tokens, textEmb, w);
  const { xFinal, yhat } = blocksAndDecoders(x0, S, masks, w);
  const targetPos = tokens.findIndex((t) => t.isTarget && !t.isPadding);
  const s = targetPos >= 0 ? tokens[targetPos].sem : -1;
  const prediction =
    s === SemType.Boolean ? yhat.boolean[targetPos] : s === SemType.Number ? yhat.number[targetPos] : s === SemType.DateTime ? yhat.datetime[targetPos] : NaN;
  return { x0, xFinal, yhat, prediction, loss: meanLoss(lossTerms(tokens, yhat)), targetPos, ms: performance.now() - t0 };
}

const labelCache = new Map<string, LiteralResult>();

/**
 * The literal forward pass of `ctx` with the target token's stored label replaced by `z` (the
 * z-scored value in boolean_values for a Boolean target, number_values for a Number target,
 * datetime_values for a DateTime target). Masks are unchanged: they read only ids (model.py:165-191).
 */
export function literalWithStoredLabel(
  ctx: SampledContext,
  pdb: Pick<PreprocessedDB, 'text'>,
  store: EmbeddingStore,
  weights: RTWeights,
  z: number,
): LiteralResult {
  const key = `${objId(ctx)}|${objId(pdb)}|${objId(store)}|${objId(weights)}|${z}`;
  const hit = labelCache.get(key);
  if (hit) return hit;
  const tp = ctx.targetPos;
  const tokens: Token[] = ctx.tokens.map((t, i) => {
    if (i !== tp) return t;
    return t.sem === SemType.Boolean
      ? { ...t, booleanValue: z }
      : t.sem === SemType.Number
        ? { ...t, numberValue: z }
        : { ...t, datetimeValue: z };
  });
  const r = literalForward(tokens, getMasks(ctx), textEmbedder(pdb, store), weights);
  labelCache.set(key, r);
  if (labelCache.size > 12) labelCache.delete(labelCache.keys().next().value as string);
  return r;
}

/** Largest |difference| between the literal path and the engine on the sampler's own tokens (0 = bit for bit). */
export function literalSelfCheck(literal: LiteralResult, engine: ForwardResult): { x0: number; xFinal: number; prediction: boolean; loss: boolean } {
  const diff = (a: Float32Array, b: Float32Array) => {
    let m = a.length === b.length ? 0 : Infinity;
    for (let i = 0; i < a.length && m !== Infinity; i++) {
      const v = Math.abs(a[i] - b[i]);
      if (!(v <= m)) m = Number.isNaN(v) ? Infinity : v;
    }
    return m;
  };
  return {
    x0: diff(literal.x0, engine.embedding.x0),
    xFinal: diff(literal.xFinal, engine.xFinal),
    prediction: Object.is(literal.prediction, engine.prediction),
    loss: Object.is(literal.loss, engine.loss),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Shuffled token order
// ─────────────────────────────────────────────────────────────────────────────

/** A uniform random permutation of 0..n-1 (Fisher-Yates) from a seeded PRNG. perm[newPos] = oldPos. */
export function randomPermutation(n: number, seed: number): number[] {
  const rng = new Rng(seed);
  const p = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    [p[i], p[j]] = [p[j], p[i]];
  }
  return p;
}

export interface ShuffleResult {
  perm: number[];
  tokens: Token[];
  masks: MaskSet;
  fr: RTForwardResult;
  /** Target position in the shuffled order. */
  targetPos: number;
  /** max |x_final'[j] − x_final[perm[j]]| over every non-padding token and dimension. */
  maxEquivDiff: number;
  ms: number;
}

const shuffleCache = new Map<string, ShuffleResult>();

/**
 * Permute the tokens, rebuild the masks from the permuted ids and run the forward pass again.
 * Positions are not an input of the model (no positional encodings, spec §6.5), so only the
 * order of the arrays changes.
 */
export function forwardShuffled(
  ctx: SampledContext,
  pdb: Pick<PreprocessedDB, 'text'>,
  store: EmbeddingStore,
  weights: RTWeights,
  base: { xFinal: Float32Array; d: number },
  seed: number,
): ShuffleResult {
  const key = `${objId(ctx)}|${objId(pdb)}|${objId(store)}|${objId(weights)}|${seed}`;
  const hit = shuffleCache.get(key);
  if (hit) return hit;
  const t0 = performance.now();
  const perm = randomPermutation(ctx.tokens.length, seed);
  const tokens = perm.map((i) => ctx.tokens[i]);
  const masks = buildMasks(tokens);
  const fr = forwardWithMasks(tokens, masks, textEmbedder(pdb, store), weights, { capture: false });
  const d = base.d;
  let maxEquivDiff = 0;
  for (let j = 0; j < tokens.length; j++) {
    if (tokens[j].isPadding) continue;
    const a = j * d;
    const b = perm[j] * d;
    for (let k = 0; k < d; k++) {
      const diff = Math.abs(fr.xFinal[a + k] - base.xFinal[b + k]);
      if (diff > maxEquivDiff) maxEquivDiff = diff;
    }
  }
  const res: ShuffleResult = { perm, tokens, masks, fr, targetPos: fr.targetPos, maxEquivDiff, ms: performance.now() - t0 };
  shuffleCache.set(key, res);
  if (shuffleCache.size > 6) shuffleCache.delete(shuffleCache.keys().next().value as string);
  return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. A context that ends in padding
// ─────────────────────────────────────────────────────────────────────────────

export interface PaddedPick {
  seedNode: number;
  split: Split;
  numReal: number;
  scanned: number;
}

const padCache = new Map<string, PaddedPick | null>();

/**
 * Scan every seed of the task (Train, Val, Test; autocomplete has one seed list) at the given L, w
 * and RNG seed, and return the one whose context leaves the most padding (ties → first in scan
 * order). Chunked with setTimeout so the page stays responsive; `onProgress` gets (done, total).
 */
export function findMostPadded(
  pdb: PreprocessedDB,
  task: TaskDef,
  seedsBySplit: { split: Split; seeds: number[] }[],
  L: number,
  w: number,
  rngSeed: number,
  onProgress: (done: number, total: number) => void,
  signal: { cancelled: boolean },
): Promise<PaddedPick | null> {
  const key = `${objId(pdb)}|${task.id}|${L}|${w}|${rngSeed}`;
  if (padCache.has(key)) return Promise.resolve(padCache.get(key) ?? null);
  const jobs: { split: Split; seed: number }[] = [];
  for (const g of seedsBySplit) for (const s of g.seeds) jobs.push({ split: g.split, seed: s });
  const total = jobs.length;
  let best: PaddedPick | null = null;
  let i = 0;
  return new Promise((resolve) => {
    const step = () => {
      if (signal.cancelled) return resolve(null);
      const end = Math.min(total, i + 40);
      for (; i < end; i++) {
        const { split, seed } = jobs[i];
        const ctx = sampleContext(pdb, makeSampleParams(pdb, task, seed, L, w, rngSeed), { recordEvents: false });
        if (ctx.numReal < L && (!best || ctx.numReal < best.numReal)) best = { seedNode: seed, split, numReal: ctx.numReal, scanned: 0 };
      }
      onProgress(i, total);
      if (i < total) setTimeout(step, 0);
      else {
        const out = best ? { ...best, scanned: total } : null;
        padCache.set(key, out);
        resolve(out);
      }
    };
    setTimeout(step, 0);
  });
}

/** The cached (getContext) context of a seed with the given knobs. */
export function contextFor(pdb: PreprocessedDB, task: TaskDef, seedNode: number, L: number, w: number, rngSeed: number): SampledContext {
  return getContext(pdb, makeSampleParams(pdb, task, seedNode, L, w, rngSeed));
}
