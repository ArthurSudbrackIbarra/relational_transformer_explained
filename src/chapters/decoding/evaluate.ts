/**
 * Batch evaluations for the decoding chapter, run in the browser with the engine's own mirrors:
 * sampleContext (fly.rs:357-511), buildMasks (model.py:164-191) and forwardWithMasks
 * (model.py:153-267, capture off). Nothing here adds model semantics.
 *
 *  - evaluateSplit: one context and one forward pass per seed of a task split ("a single forward
 *    pass", no ensembling, spec §8.5), then AUROC on raw logits or R² in z-space (main.py:328-337).
 *    Eval contexts use epoch 0 and seed 0 (main.py:158-160, spec §8.6), i.e. the store's RNG seed.
 *  - miniBatch: per-sequence loss terms of a few Train contexts, averaged like model.py:265.
 *
 * Cost: ~60-120 ms per seed, so seeds are processed one per macrotask (setTimeout), runs are queued
 * one at a time, can be cancelled, and results are cached at module level by every input they
 * depend on (so leaving the chapter and coming back is instant).
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { buildMasks } from '@/engine/masks';
import { auroc, binaryLabel, r2 } from '@/engine/metrics';
import { forwardWithMasks, textEmbedder, type LossTerm } from '@/engine/model';
import { getSeeds, makeSampleParams } from '@/engine/pipeline';
import { Rng } from '@/engine/rng';
import { sampleContext } from '@/engine/sampler';
import type { AttnLayer, PreprocessedDB, RawValue, RTWeights, Split, TaskDef } from '@/engine/types';
import type { EmbeddingStore } from '@/engine/embeddings';
import type { Lane } from './live';

// ─────────────────────────────────────────────────────────────────────────────
// Identity keys and a serial queue
// ─────────────────────────────────────────────────────────────────────────────

const ids = new WeakMap<object, number>();
let nextId = 1;
function oid(o: object): number {
  let id = ids.get(o);
  if (id === undefined) ids.set(o, (id = nextId++));
  return id;
}

let queue: Promise<unknown> = Promise.resolve();
/** Run jobs one after another so two figures never compete for the main thread. */
function enqueue<T>(job: () => Promise<T>): Promise<T> {
  const p = queue.then(job, job);
  queue = p.catch(() => undefined);
  return p;
}

const tick = () => new Promise<void>((r) => window.setTimeout(r, 0));

// ─────────────────────────────────────────────────────────────────────────────
// Split evaluation
// ─────────────────────────────────────────────────────────────────────────────

export interface EvalItem {
  seed: number;
  /** Decoder output at the target: boolean logit or number z-score (main.py:283-288). */
  pred: number;
  /** The stored, z-scored label (boolean_values / number_values at is_targets). */
  z: number;
  /** Binary: int(z > 0) (main.py:336). */
  y01: 0 | 1;
  raw: RawValue;
  /** Entity row position (forecast) or seed row position (autocomplete), for tooltips. */
  entityRow: number;
}

export interface EvalResult {
  key: string;
  task: TaskDef;
  split: Split | 'all';
  items: EvalItem[];
  metricName: 'auroc' | 'r2';
  metric: number;
  msPerSeed: number;
}

const evalCache = new Map<string, EvalResult>();

export interface EvalInputs {
  pdb: PreprocessedDB;
  task: TaskDef;
  split: Split;
  L: number;
  w: number;
  rngSeed: number;
  store: EmbeddingStore;
  weights: RTWeights;
  disabled: readonly AttnLayer[];
}

export function evalKey(e: EvalInputs): string {
  const split = e.task.kind === 'autocomplete' ? '*' : e.split;
  return [oid(e.pdb), e.task.id, split, e.L, e.w, e.rngSeed, oid(e.store), oid(e.weights), [...e.disabled].sort().join(',')].join('|');
}

export function cachedEval(e: EvalInputs): EvalResult | undefined {
  return evalCache.get(evalKey(e));
}

async function runEval(e: EvalInputs, onProgress: (done: number, total: number) => void, cancelled: () => boolean): Promise<EvalResult | null> {
  const key = evalKey(e);
  const hit = evalCache.get(key);
  if (hit) return hit;
  const seeds = getSeeds(e.pdb, e.task, e.split);
  const embed = textEmbedder(e.pdb, e.store);
  const items: EvalItem[] = [];
  const t0 = performance.now();
  onProgress(0, seeds.length);
  for (let i = 0; i < seeds.length; i++) {
    await tick();
    if (cancelled()) return null;
    const s = seeds[i];
    const ctx = sampleContext(e.pdb, makeSampleParams(e.pdb, e.task, s, e.L, e.w, e.rngSeed));
    const fr = forwardWithMasks(ctx.tokens, buildMasks(ctx.tokens), embed, e.weights, { capture: false, disabledLayers: [...e.disabled] });
    const tok = ctx.tokens[fr.targetPos];
    const z = e.task.targetType === 'binary' ? tok.booleanValue : tok.numberValue;
    const node = e.pdb.nodes[s];
    const entityRow = e.task.kind === 'forecast' && node.f2pNbrIdxs.length ? e.pdb.nodes[node.f2pNbrIdxs[0]].rowPos : node.rowPos;
    items.push({
      seed: s,
      pred: fr.prediction,
      z,
      y01: binaryLabel(z),
      raw: tok.raw,
      entityRow,
    });
    onProgress(i + 1, seeds.length);
  }
  let metric = NaN;
  try {
    metric =
      e.task.targetType === 'binary'
        ? auroc(
            items.map((it) => it.y01),
            items.map((it) => it.pred),
          )
        : r2(
            items.map((it) => it.z),
            items.map((it) => it.pred),
          );
  } catch {
    metric = NaN; // e.g. one class only: roc_auc_score raises
  }
  const res: EvalResult = {
    key,
    task: e.task,
    split: e.task.kind === 'autocomplete' ? 'all' : e.split,
    items,
    metricName: e.task.targetType === 'binary' ? 'auroc' : 'r2',
    metric,
    msPerSeed: (performance.now() - t0) / Math.max(1, seeds.length),
  };
  evalCache.set(key, res);
  if (evalCache.size > 16) evalCache.delete(evalCache.keys().next().value as string);
  return res;
}

export type EvalStatus = 'idle' | 'waiting' | 'running' | 'done' | 'error';

export interface EvalState {
  status: EvalStatus;
  done: number;
  total: number;
  result: EvalResult | null;
  error?: string;
  start: () => void;
  cancel: () => void;
}

/** Evaluate a lane's task on `split` (all its seeds). `auto` starts as soon as the inputs are ready. */
export function useEvaluation(lane: Lane, split: Split, auto: boolean): EvalState {
  const { assets, pdb, task, L, w, rngSeed, disabled } = lane;
  const inputs: EvalInputs | null =
    assets.status === 'ready' && assets.store && assets.weights
      ? {
          pdb,
          task,
          split,
          L,
          w,
          rngSeed,
          store: assets.store,
          weights: assets.weights,
          disabled,
        }
      : null;
  const key = inputs ? evalKey(inputs) : null;
  const [state, setState] = useState<{
    key: string | null;
    status: EvalStatus;
    done: number;
    total: number;
    result: EvalResult | null;
    error?: string;
  }>(() => ({
    key,
    status: 'idle',
    done: 0,
    total: 0,
    result: inputs ? (cachedEval(inputs) ?? null) : null,
  }));
  const token = useRef(0);
  const inputsRef = useRef(inputs);
  inputsRef.current = inputs;

  const start = useCallback(() => {
    const e = inputsRef.current;
    if (!e) return;
    const k = evalKey(e);
    const my = ++token.current;
    const hit = cachedEval(e);
    if (hit) {
      setState({
        key: k,
        status: 'done',
        done: hit.items.length,
        total: hit.items.length,
        result: hit,
      });
      return;
    }
    setState({
      key: k,
      status: 'waiting',
      done: 0,
      total: getSeeds(e.pdb, e.task, e.split).length,
      result: null,
    });
    enqueue(() =>
      runEval(
        e,
        (done, total) => {
          if (token.current === my) setState((s) => ({ ...s, key: k, status: 'running', done, total }));
        },
        () => token.current !== my,
      ),
    ).then(
      (res) => {
        if (token.current !== my || !res) return;
        setState({
          key: k,
          status: 'done',
          done: res.items.length,
          total: res.items.length,
          result: res,
        });
      },
      (err: unknown) => {
        if (token.current !== my) return;
        setState((s) => ({
          ...s,
          key: k,
          status: 'error',
          error: err instanceof Error ? err.message : String(err),
        }));
      },
    );
  }, []);

  const cancel = useCallback(() => {
    token.current++;
    setState((s) => ({ ...s, status: 'idle', done: 0 }));
  }, []);

  // New inputs: show the cached result if any, else (auto) start.
  useEffect(() => {
    if (!key || !inputsRef.current) return;
    const hit = cachedEval(inputsRef.current);
    if (hit) {
      token.current++;
      setState({
        key,
        status: 'done',
        done: hit.items.length,
        total: hit.items.length,
        result: hit,
      });
    } else if (auto) {
      start();
    } else {
      token.current++;
      setState({ key, status: 'idle', done: 0, total: 0, result: null });
    }
  }, [key, auto, start]);

  // Cancel a running job when the component unmounts.
  useEffect(
    () => () => {
      token.current++;
    },
    [],
  );

  const current = state.key === key ? state : { key, status: 'idle' as EvalStatus, done: 0, total: 0, result: null };
  return { ...current, start, cancel };
}

// ─────────────────────────────────────────────────────────────────────────────
// A mini batch for the loss
// ─────────────────────────────────────────────────────────────────────────────

export interface BatchItem {
  taskLabel: string;
  task: TaskDef;
  seed: number;
  entityRow: number;
  term: LossTerm;
}

export interface BatchInputs {
  parts: { pdb: PreprocessedDB; task: TaskDef; n: number }[];
  L: number;
  w: number;
  store: EmbeddingStore;
  weights: RTWeights;
  draw: number;
}

const batchCache = new Map<string, BatchItem[]>();

function batchKey(b: BatchInputs): string {
  return [b.parts.map((p) => `${oid(p.pdb)}:${p.task.id}:${p.n}`).join(','), b.L, b.w, oid(b.store), oid(b.weights), b.draw].join('|');
}

/** `n` Train seeds per task (autocomplete: any row), drawn with a seeded shuffle. */
function drawSeeds(pdb: PreprocessedDB, task: TaskDef, n: number, draw: number): number[] {
  const seeds = getSeeds(pdb, task, 'Train').slice();
  const rng = new Rng(1000 + draw * 7919 + task.id.length);
  for (let i = seeds.length - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    [seeds[i], seeds[j]] = [seeds[j], seeds[i]];
  }
  return seeds.slice(0, n);
}

async function runBatch(b: BatchInputs, cancelled: () => boolean): Promise<BatchItem[] | null> {
  const key = batchKey(b);
  const hit = batchCache.get(key);
  if (hit) return hit;
  const out: BatchItem[] = [];
  for (const part of b.parts) {
    const embed = textEmbedder(part.pdb, b.store);
    for (const s of drawSeeds(part.pdb, part.task, part.n, b.draw)) {
      await tick();
      if (cancelled()) return null;
      // A training context: epoch 0, seed 0 (fly.rs:371-375 seeds the PRNG with epoch + node + seed).
      const ctx = sampleContext(part.pdb, makeSampleParams(part.pdb, part.task, s, b.L, b.w, 0));
      const fr = forwardWithMasks(ctx.tokens, buildMasks(ctx.tokens), embed, b.weights, { capture: false });
      const node = part.pdb.nodes[s];
      const entityRow = part.task.kind === 'forecast' && node.f2pNbrIdxs.length ? part.pdb.nodes[node.f2pNbrIdxs[0]].rowPos : node.rowPos;
      for (const term of fr.lossTerms)
        out.push({
          taskLabel: part.task.name,
          task: part.task,
          seed: s,
          entityRow,
          term,
        });
    }
  }
  batchCache.set(key, out);
  if (batchCache.size > 8) batchCache.delete(batchCache.keys().next().value as string);
  return out;
}

export function useMiniBatch(b: BatchInputs | null): {
  items: BatchItem[] | null;
  running: boolean;
} {
  const key = b ? batchKey(b) : null;
  const [state, setState] = useState<{
    key: string | null;
    items: BatchItem[] | null;
  }>({ key: null, items: null });
  const bRef = useRef(b);
  bRef.current = b;
  useEffect(() => {
    const cur = bRef.current;
    if (!key || !cur) return;
    const hit = batchCache.get(key);
    if (hit) {
      setState({ key, items: hit });
      return;
    }
    let dead = false;
    enqueue(() => runBatch(cur, () => dead)).then(
      (items) => {
        if (!dead && items) setState({ key, items });
      },
      () => undefined,
    );
    return () => {
      dead = true;
    };
  }, [key]);
  const items = state.key === key ? state.items : null;
  return { items, running: !!key && !items };
}
