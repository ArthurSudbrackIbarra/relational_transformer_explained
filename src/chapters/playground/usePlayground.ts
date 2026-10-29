/**
 * Live state of the playground: the global context (same seed, L, w, RNG seed and checkpoint as every
 * other chapter) and its ablated twin. Both forward passes come from the engine
 * (getForward = model.py:153-267 with the context's cached masks; ablatedForward for the twin).
 *
 *  - baseline = the course's live context and checkpoint, full model (no layer switched off);
 *  - current  = the same seed with the playground's ablations applied.
 */
import { useEffect, useMemo, useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { LAYER_INFO } from '@/components';
import { denormalize, sigmoid } from '@/engine/metrics';
import { autoCheckpoint, getForward, makeSampleParams, type CheckpointName } from '@/engine/pipeline';
import { sampleContext } from '@/engine/sampler';
import type { AttnLayer, ColumnStats, ForwardResult, PreprocessedDB, RawValue, RTWeights, SampledContext, TaskDef } from '@/engine/types';
import { ATTN_ORDER } from '@/engine/types';
import type { EmbeddingStore } from '@/engine/embeddings';
import {
  loadAssets,
  useAssets,
  useContextWindow,
  useDatabase,
  usePreprocessed,
  useSampleParams,
  useSeedNode,
  useSeeds,
  useTask,
  type AssetsState,
} from '@/state/hooks';
import { useSim } from '@/state/store';
import { ablateContext, ablatedForward, labelStats, tokenAblationKey, type AblatedContext, type TokenAblation } from './ablate';
import { usePlay } from './state';

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoints
// ─────────────────────────────────────────────────────────────────────────────

/** Embedding store + weights of any checkpoint (shares the global loader's cache). */
export function useCkptAssets(ckpt: CheckpointName): AssetsState {
  const [st, setSt] = useState<AssetsState | null>(null);
  useEffect(() => {
    let alive = true;
    loadAssets(ckpt).then((s) => {
      if (alive) setSt(s);
    });
    return () => {
      alive = false;
    };
  }, [ckpt]);
  return st && st.ckpt === ckpt ? st : { status: 'loading', store: null, weights: null, weightsKind: null, fallback: false, ckpt };
}

export interface CkptOption {
  id: 'init' | 'zero' | 'ft' | 'base';
  name: CheckpointName;
  label: string;
  /** One line for the switch. */
  short: string;
  hint: string;
}

/**
 * The checkpoints the reader can compare for a task: untrained, zero-shot, and fine-tuned (if one
 * exists). A course checkpoint chosen in the top bar that is none of these is listed too.
 */
export function ckptOptions(task: TaskDef, baseCkpt?: CheckpointName): CkptOption[] {
  const zero = autoCheckpoint(task.id).name;
  const out: CkptOption[] = [
    { id: 'init', name: 'init', label: 'Untrained', short: 'step 0: random init', hint: 'Step 0 of pretraining: random initial weights.' },
    { id: 'zero', name: zero, label: 'Zero-shot', short: 'pretrained on 3 other DBs', hint: `${zero}: pretrained on the other three databases only.` },
  ];
  const ft: CheckpointName | null = task.id === 'shop/customer-churn' ? 'finetune-churn' : task.id === 'shop/product-sales' ? 'finetune-sales' : null;
  if (ft)
    out.push({
      id: 'ft',
      name: ft,
      label: 'Fine-tuned',
      short: 'trained on ShopDB labels',
      hint: `${ft}: further trained on ShopDB’s own Train labels, so not zero-shot.`,
    });
  if (baseCkpt && !out.some((o) => o.name === baseCkpt))
    out.push({ id: 'base', name: baseCkpt, label: baseCkpt, short: 'chosen in the top bar', hint: `${baseCkpt}: the course checkpoint set in the top bar.` });
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Read-outs
// ─────────────────────────────────────────────────────────────────────────────

export interface Readout {
  /** Decoder output at the target: logit (binary) or z-score (regression), spec §8.3. */
  output: number;
  /** σ(logit) (binary) or ẑ·σ + μ in original units (regression), spec §8.4. */
  display: number;
}

export function readout(output: number, task: TaskDef, stats: ColumnStats): Readout {
  return { output, display: task.targetType === 'binary' ? sigmoid(output) : denormalize(output, stats) };
}

/** "45.1%" or "5.3" for a display value. */
export function fmtDisplay(v: number, task: TaskDef, digits = 1): string {
  if (!Number.isFinite(v)) return '—';
  return task.targetType === 'binary' ? `${(v * 100).toFixed(digits)}%` : v.toFixed(digits === 1 ? 2 : digits);
}

/** Signed difference: "+3.2 pp" or "−0.41". */
export function fmtDelta(d: number, task: TaskDef): string {
  if (!Number.isFinite(d)) return '—';
  const sign = d > 0 ? '+' : d < 0 ? '−' : '±';
  return task.targetType === 'binary' ? `${sign}${Math.abs(d * 100).toFixed(1)} pp` : `${sign}${Math.abs(d).toFixed(2)}`;
}

/** What the prediction is, in words. */
export function targetPhrase(task: TaskDef): { short: string; long: string; unit: string } {
  switch (task.id) {
    case 'shop/customer-churn':
      return { short: 'P(churn)', long: 'probability that the customer churns (no order in the next 30 days)', unit: '' };
    case 'shop/product-sales':
      return { short: 'units sold', long: 'units of the product ordered in the next 30 days', unit: 'units' };
    case 'shop/reviews.rating':
      return { short: 'rating', long: 'star rating of the review', unit: 'stars' };
    case 'shop/reviews.verified':
      return { short: 'P(verified)', long: 'probability that the review is a verified purchase', unit: '' };
    default:
      return task.targetType === 'binary'
        ? { short: `P(${task.targetCol})`, long: `probability that ${task.targetCol} is true`, unit: '' }
        : { short: task.targetCol, long: task.targetCol, unit: '' };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ablated forward cache
// ─────────────────────────────────────────────────────────────────────────────

const ids = new WeakMap<object, number>();
let nextId = 1;
export function oid(o: object): number {
  let id = ids.get(o);
  if (id === undefined) ids.set(o, (id = nextId++));
  return id;
}

/**
 * Contexts at the playground's own L, sampled with the engine's sampler but kept out of the shared
 * pipeline cache (dragging the L slider must not evict the course's live context).
 */
const localCtx = new Map<string, SampledContext>();
function localContext(pdb: PreprocessedDB, task: TaskDef, seedNode: number, L: number, w: number, rngSeed: number): SampledContext {
  const key = [oid(pdb), task.id, seedNode, L, w, rngSeed].join('|');
  let c = localCtx.get(key);
  if (!c) {
    c = sampleContext(pdb, makeSampleParams(pdb, task, seedNode, L, w, rngSeed), { recordEvents: false });
    localCtx.set(key, c);
    while (localCtx.size > 6) localCtx.delete(localCtx.keys().next().value as string);
  }
  return c;
}

const fwdCache = new Map<string, ForwardResult>();
const FWD_MAX = 3;

/** Captured forward of an ablated context, cached by everything it depends on. */
function cachedAblatedForward(
  key: string,
  abl: AblatedContext,
  pdb: PreprocessedDB,
  store: EmbeddingStore,
  weights: RTWeights,
  disabled: readonly AttnLayer[],
): ForwardResult {
  const hit = fwdCache.get(key);
  if (hit) {
    fwdCache.delete(key);
    fwdCache.set(key, hit);
    return hit;
  }
  const fr = ablatedForward(abl, pdb, store, weights, disabled, true);
  fwdCache.set(key, fr);
  while (fwdCache.size > FWD_MAX) fwdCache.delete(fwdCache.keys().next().value as string);
  return fr;
}

/** `value`, but only after it has stopped changing for `ms` (a first value is returned at once). */
function useSettled<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    if (Object.is(value, settled)) return;
    const t = window.setTimeout(() => setSettled(value), ms);
    return () => window.clearTimeout(t);
  }, [value, settled, ms]);
  return settled;
}

// ─────────────────────────────────────────────────────────────────────────────
// The hook
// ─────────────────────────────────────────────────────────────────────────────

export interface PlayLive {
  pdb: PreprocessedDB;
  db: ReturnType<typeof useDatabase>;
  task: TaskDef;
  seeds: number[];
  seedNode: number;
  /** "<taskId>|<seed>" */
  scope: string;
  /** The course's live context (global L). */
  ctx: SampledContext;
  globalL: number;
  w: number;
  rngSeed: number;
  /** The context at the playground's L (=== ctx when L is not changed). */
  ablCtx: SampledContext;
  effL: number;
  baseAssets: AssetsState;
  ablAssets: AssetsState;
  baseCkpt: CheckpointName;
  ablCkpt: CheckpointName;
  tokenAb: TokenAblation;
  abl: AblatedContext;
  disabled: AttnLayer[];
  baseFr: ForwardResult | null;
  ablFr: ForwardResult | null;
  base: Readout | null;
  cur: Readout | null;
  stats: ColumnStats;
  /** The masked target's hidden label. */
  truth: RawValue;
  /** Any ablation active? */
  active: boolean;
  chips: string[];
  sig: string;
}

export function usePlayground(): PlayLive {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const task = useTask();
  const seeds = useSeeds();
  const seedNode = useSeedNode();
  const ctx = useContextWindow();
  const params = useSampleParams();
  const rngSeed = useSim((s) => s.rngSeed);
  const disabled = useSim(useShallow((s) => s.disabledLayers));
  const baseAssets = useAssets();
  const play = usePlay(
    useShallow((s) => ({
      removeSelf: s.removeSelf,
      removeOther: s.removeOther,
      removeWhat: s.removeWhat,
      shuffle: s.shuffle,
      shuffleSeed: s.shuffleSeed,
      L: s.L,
      ckpt: s.ckpt,
      editScope: s.editScope,
      edits: s.edits,
    })),
  );
  const scope = `${task.id}|${seedNode}`;
  const globalL = params.L;
  // the slider moves at once; the (costly) new context and forward pass follow once it rests
  const effL = useSettled(play.L ?? globalL, 180);

  const ablCtx = useMemo(
    () => (effL === globalL ? ctx : localContext(pdb, task, seedNode, effL, params.w, rngSeed)),
    [effL, globalL, ctx, pdb, task, seedNode, params.w, rngSeed],
  );

  const baseCkpt = baseAssets.ckpt;
  const ablCkpt = play.ckpt ?? baseCkpt;
  const otherAssets = useCkptAssets(ablCkpt);
  const ablAssets = ablCkpt === baseCkpt ? baseAssets : otherAssets;

  const tokenAb = useMemo<TokenAblation>(
    () => ({
      removeSelf: task.kind === 'forecast' && play.removeSelf,
      removeOther: task.kind === 'forecast' && play.removeOther,
      removeWhat: play.removeWhat,
      shuffle: play.shuffle,
      shuffleSeed: play.shuffleSeed,
      edits: new Map(play.editScope === scope ? Object.entries(play.edits).map(([k, v]) => [Number(k), v] as [number, RawValue]) : []),
    }),
    [task.kind, play.removeSelf, play.removeOther, play.removeWhat, play.shuffle, play.shuffleSeed, play.editScope, play.edits, scope],
  );
  const abl = useMemo(() => ablateContext(pdb, task, ablCtx, tokenAb), [pdb, task, ablCtx, tokenAb]);
  const stats = useMemo(() => labelStats(pdb, task), [pdb, task]);

  const baseFr = useMemo(() => {
    if (baseAssets.status !== 'ready' || !baseAssets.store || !baseAssets.weights) return null;
    return getForward(ctx, pdb, baseAssets.store, baseAssets.weights, []);
  }, [ctx, pdb, baseAssets]);

  const disabledKey = [...disabled].sort().join(',');
  const active = abl.untouched === false || effL !== globalL || ablCkpt !== baseCkpt || disabled.length > 0;
  const tokenKey = tokenAblationKey(tokenAb);
  const ablFr = useMemo(() => {
    if (ablAssets.status !== 'ready' || !ablAssets.store || !ablAssets.weights) return null;
    if (abl.untouched && ablCtx === ctx && ablCkpt === baseCkpt && disabled.length === 0) return baseFr;
    // the live context (any checkpoint / layer switch) shares the course's forward cache; other lengths stay local
    if (abl.untouched && ablCtx === ctx) return getForward(ablCtx, pdb, ablAssets.store, ablAssets.weights, disabled);
    const key = [oid(ablCtx), oid(pdb), oid(ablAssets.store), oid(ablAssets.weights), tokenKey, disabledKey].join('#');
    return cachedAblatedForward(key, abl, pdb, ablAssets.store, ablAssets.weights, disabled);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [abl, ablCtx, ctx, pdb, ablAssets, ablCkpt, baseCkpt, disabledKey, baseFr, tokenKey]);

  const base = useMemo(() => (baseFr && baseFr.targetPos >= 0 ? readout(baseFr.prediction, task, stats) : null), [baseFr, task, stats]);
  const cur = useMemo(() => (ablFr && ablFr.targetPos >= 0 ? readout(ablFr.prediction, task, stats) : null), [ablFr, task, stats]);

  const chips = useMemo(() => {
    const c: string[] = [];
    const rows = tokenAb.removeWhat === 'rows' ? ' (rows)' : '';
    if (tokenAb.removeSelf) c.push(`no self labels${rows}`);
    if (tokenAb.removeOther) c.push(`no other labels${rows}`);
    if (abl.editedPos.size > 0) c.push(`${abl.editedPos.size} label${abl.editedPos.size === 1 ? '' : 's'} edited`);
    if (abl.perm) c.push(`names shuffled #${tokenAb.shuffleSeed}`);
    if (effL !== globalL) c.push(`L = ${effL}`);
    for (const l of ATTN_ORDER) if (disabled.includes(l)) c.push(`no ${LAYER_INFO[l].name.toLowerCase()} attn.`);
    if (ablCkpt !== baseCkpt) c.push(ablCkpt);
    return c.length ? c : ['full model'];
  }, [tokenAb, abl, effL, globalL, disabled, ablCkpt, baseCkpt]);

  const sig = `${scope}#${tokenKey}#${effL}#${disabledKey}#${ablCkpt}`;

  return {
    pdb,
    db,
    task,
    seeds,
    seedNode,
    scope,
    ctx,
    globalL,
    w: params.w,
    rngSeed,
    ablCtx,
    effL,
    baseAssets,
    ablAssets,
    baseCkpt,
    ablCkpt,
    tokenAb,
    abl,
    disabled,
    baseFr,
    ablFr,
    base,
    cur,
    stats,
    truth: ctx.tokens[ctx.targetPos]?.raw ?? null,
    active,
    chips,
    sig,
  };
}
