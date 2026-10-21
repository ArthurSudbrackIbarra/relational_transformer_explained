/**
 * Re-run a zero-shot / fine-tuned evaluation in the browser with the engine's own mirrors:
 * one eval-style context per seed (epoch 0, seed 0: main.py:158-160, spec §8.6), one forward pass
 * (model.py:153-267, capture off), the decoder output at the target (main.py:283-288), then AUROC on
 * raw logits or R² in z-space (main.py:328-337). L and w are the training run's (128 / 16), not the
 * store's, so the result is comparable with results.json.
 *
 * Seeds run one per macrotask (setTimeout) so the page stays responsive; jobs run one at a time,
 * can be cancelled, and finished results are cached for the page's lifetime.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { loadAssets } from '@/state/hooks';
import { getDatabase } from '@/engine/db/registry';
import { buildMasks } from '@/engine/masks';
import { auroc, binaryLabel, r2 } from '@/engine/metrics';
import { forwardWithMasks, textEmbedder } from '@/engine/model';
import { findTask, getPreprocessed, getSeeds, makeSampleParams, type CheckpointName } from '@/engine/pipeline';
import { sampleContext } from '@/engine/sampler';
import type { Split } from '@/engine/types';

export interface EvalJob {
  taskId: string;
  split: Split;
  ckpt: CheckpointName;
  L: number;
  w: number;
}

export interface EvalOutcome {
  metric: number;
  n: number;
  msPerSeed: number;
  fallback: boolean;
}

const done = new Map<string, EvalOutcome>();
const keyOf = (j: EvalJob) => `${j.taskId}|${j.split}|${j.ckpt}|${j.L}|${j.w}`;
let queue: Promise<unknown> = Promise.resolve();
const tick = () => new Promise<void>((r) => window.setTimeout(r, 0));

async function run(j: EvalJob, onProgress: (d: number, n: number) => void, cancelled: () => boolean): Promise<EvalOutcome | null> {
  const hit = done.get(keyOf(j));
  if (hit) return hit;
  const assets = await loadAssets(j.ckpt);
  if (assets.status !== 'ready' || !assets.store || !assets.weights) throw new Error(assets.error ?? 'assets failed to load');
  const db = getDatabase(j.taskId.split('/')[0] as 'shop');
  const pdb = getPreprocessed(db);
  const task = findTask(db, j.taskId);
  const seeds = getSeeds(pdb, task, j.split);
  const embed = textEmbedder(pdb, assets.store);
  const ys: number[] = [];
  const preds: number[] = [];
  const t0 = performance.now();
  onProgress(0, seeds.length);
  for (let i = 0; i < seeds.length; i++) {
    await tick();
    if (cancelled()) return null;
    const ctx = sampleContext(pdb, makeSampleParams(pdb, task, seeds[i], j.L, j.w, 0));
    const fr = forwardWithMasks(ctx.tokens, buildMasks(ctx.tokens), embed, assets.weights, { capture: false });
    const tok = ctx.tokens[fr.targetPos];
    ys.push(task.targetType === 'binary' ? binaryLabel(tok.booleanValue) : tok.numberValue);
    preds.push(fr.prediction);
    onProgress(i + 1, seeds.length);
  }
  let metric = NaN;
  try {
    metric = task.targetType === 'binary' ? auroc(ys, preds) : r2(ys, preds);
  } catch {
    metric = NaN;
  }
  const out: EvalOutcome = { metric, n: seeds.length, msPerSeed: (performance.now() - t0) / Math.max(1, seeds.length), fallback: assets.fallback };
  done.set(keyOf(j), out);
  return out;
}

export interface EvalState {
  status: 'idle' | 'waiting' | 'running' | 'done' | 'error';
  done: number;
  total: number;
  result: EvalOutcome | null;
  error?: string;
  start: () => void;
  cancel: () => void;
}

export function useBrowserEval(job: EvalJob): EvalState {
  const key = keyOf(job);
  const [st, setSt] = useState<Omit<EvalState, 'start' | 'cancel'>>(() => {
    const hit = done.get(key);
    return hit ? { status: 'done', done: hit.n, total: hit.n, result: hit } : { status: 'idle', done: 0, total: 0, result: null };
  });
  const token = useRef(0);
  useEffect(() => {
    token.current++;
    const hit = done.get(key);
    setSt(hit ? { status: 'done', done: hit.n, total: hit.n, result: hit } : { status: 'idle', done: 0, total: 0, result: null });
  }, [key]);
  useEffect(
    () => () => {
      token.current++;
    },
    [],
  );
  const start = useCallback(() => {
    const my = ++token.current;
    const alive = () => token.current === my;
    setSt({ status: 'waiting', done: 0, total: 0, result: null });
    const p = queue.then(() =>
      run(
        job,
        (d, n) => alive() && setSt({ status: 'running', done: d, total: n, result: null }),
        () => !alive(),
      ).then(
        (r) => {
          if (alive() && r) setSt({ status: 'done', done: r.n, total: r.n, result: r });
        },
        (e: unknown) => alive() && setSt({ status: 'error', done: 0, total: 0, result: null, error: e instanceof Error ? e.message : String(e) }),
      ),
    );
    queue = p.catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const cancel = useCallback(() => {
    token.current++;
    setSt({ status: 'idle', done: 0, total: 0, result: null });
  }, []);
  return { ...st, start, cancel };
}
