/**
 * Extra forward passes for the chapter's two experiments (hidden label, shuffled order) and the
 * search for a context that ends in padding. Everything runs the engine's own mirrors:
 * buildMasks (model.py:164-191), forwardWithMasks (model.py:153-267) and sampleContext
 * (fly.rs:357-511). Nothing here changes the model or the sampler; it only feeds them different
 * token arrays.
 *
 * Cost: one forward ≈ 60-100 ms at L = 128, so results are cached and computed off the render
 * path (setTimeout), never in a render loop.
 */
import type { EmbeddingStore } from '@/engine/embeddings';
import { buildMasks } from '@/engine/masks';
import { forwardWithMasks, textEmbedder, type RTForwardResult } from '@/engine/model';
import { getContext, getMasks, makeSampleParams } from '@/engine/pipeline';
import { Rng } from '@/engine/rng';
import { sampleContext } from '@/engine/sampler';
import { SemType, type MaskSet, type PreprocessedDB, type RTWeights, type SampledContext, type Split, type TaskDef, type Token } from '@/engine/types';

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

const labelCache = new Map<string, RTForwardResult>();

/**
 * Forward pass of `ctx` with the target token's stored label replaced by `z` (the z-scored value in
 * boolean_values for a Boolean target, number_values for a Number target). Masks are unchanged:
 * they read only ids (model.py:165-191).
 */
export function forwardWithStoredLabel(
  ctx: SampledContext,
  pdb: Pick<PreprocessedDB, 'text'>,
  store: EmbeddingStore,
  weights: RTWeights,
  z: number,
): RTForwardResult {
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
  const r = forwardWithMasks(tokens, getMasks(ctx), textEmbedder(pdb, store), weights, { capture: false });
  labelCache.set(key, r);
  if (labelCache.size > 12) labelCache.delete(labelCache.keys().next().value as string);
  return r;
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
