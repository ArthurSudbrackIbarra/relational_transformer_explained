/**
 * Per-seed sweeps and the batch evaluation. Every forward pass goes through the engine
 * (sampleContext / getContext → ablateContext → ablatedForward = model.py:153-267, capture off);
 * the jobs yield to the browser between passes (jobs.ts).
 */
import type { EmbeddingStore } from '@/engine/embeddings';
import { auroc, binaryLabel, r2 } from '@/engine/metrics';
import { getContext, getSeeds, inContextLabels, makeSampleParams } from '@/engine/pipeline';
import { sampleContext } from '@/engine/sampler';
import type { AttnLayer, PreprocessedDB, RawValue, RTWeights, Split, TaskDef } from '@/engine/types';
import { ablateContext, ablatedForward, NO_TOKEN_ABLATION, type TokenAblation } from './ablate';
import { yieldToBrowser } from './jobs';

export interface SweepBase {
  pdb: PreprocessedDB;
  task: TaskDef;
  seedNode: number;
  L: number;
  w: number;
  rngSeed: number;
  store: EmbeddingStore;
  weights: RTWeights;
}

/**
 * The context of the sweep's seed at length L. The course's own L goes through the shared cache
 * (same object as every chapter); other lengths are sampled without polluting that cache.
 */
function ctxAt(b: SweepBase, L: number) {
  const params = makeSampleParams(b.pdb, b.task, b.seedNode, L, b.w, b.rngSeed);
  return L === b.L ? getContext(b.pdb, params) : sampleContext(b.pdb, params, { recordEvents: false });
}

function predict(b: SweepBase, L: number, ab: TokenAblation, disabled: readonly AttnLayer[] = []): number {
  const abl = ablateContext(b.pdb, b.task, ctxAt(b, L), ab);
  return ablatedForward(abl, b.pdb, b.store, b.weights, disabled, false).prediction;
}

type Report = (done: number, total: number) => void;
type Cancelled = () => boolean;

// ─────────────────────────────────────────────────────────────────────────────
// 1. One label at a time
// ─────────────────────────────────────────────────────────────────────────────

export type Perturb = 'flip' | 'remove' | 'zero';

export interface LabelEffect {
  node: number;
  role: 'self' | 'other';
  ts: number | null;
  raw: RawValue;
  /** Decoder output with this one label perturbed. */
  output: number;
}

export interface LabelSweep {
  base: number;
  items: LabelEffect[];
  /** All self labels removed / all other labels removed (label cells), forecast only. */
  noSelf: number | null;
  noOther: number | null;
}

export const LABEL_SWEEP_MAX = 24;

export async function runLabelSweep(b: SweepBase, perturb: Perturb, report: Report, cancelled: Cancelled): Promise<LabelSweep | null> {
  const ctx = ctxAt(b, b.L);
  const labels = inContextLabels(b.pdb, ctx, b.task);
  const all = [
    ...labels.self.map((l) => ({ ...l, role: 'self' as const })),
    ...labels.other.map((l) => ({ ...l, role: 'other' as const })),
  ]
    .sort((x, y) => (y.ts ?? 0) - (x.ts ?? 0))
    .slice(0, LABEL_SWEEP_MAX);
  const forecast = b.task.kind === 'forecast';
  const total = all.length + 1 + (forecast ? 2 : 0);
  let done = 0;
  await yieldToBrowser();
  const base = predict(b, b.L, NO_TOKEN_ABLATION);
  report(++done, total);
  const items: LabelEffect[] = [];
  for (const l of all) {
    await yieldToBrowser();
    if (cancelled()) return null;
    let output: number;
    if (perturb === 'remove') {
      // drop exactly this one label token
      const abl = ablateContext(b.pdb, b.task, ctx, NO_TOKEN_ABLATION);
      const tokens = abl.tokens.filter((t) => t.pos !== l.pos).map((t, i) => ({ ...t, pos: i }));
      output = ablatedForward({ ...abl, tokens, untouched: false }, b.pdb, b.store, b.weights, [], false).prediction;
    } else {
      const v: RawValue = perturb === 'flip' ? !(l.raw === true) : 0;
      output = predict(b, b.L, { ...NO_TOKEN_ABLATION, edits: new Map([[l.nodeIdx, v]]) });
    }
    items.push({ node: l.nodeIdx, role: l.role, ts: l.ts, raw: l.raw, output });
    report(++done, total);
  }
  let noSelf: number | null = null;
  let noOther: number | null = null;
  if (forecast) {
    await yieldToBrowser();
    if (cancelled()) return null;
    noSelf = labels.self.length ? predict(b, b.L, { ...NO_TOKEN_ABLATION, removeSelf: true }) : null;
    report(++done, total);
    await yieldToBrowser();
    if (cancelled()) return null;
    noOther = labels.other.length ? predict(b, b.L, { ...NO_TOKEN_ABLATION, removeOther: true }) : null;
    report(++done, total);
  }
  return { base, items, noSelf, noOther };
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Several name shuffles
// ─────────────────────────────────────────────────────────────────────────────

export const SHUFFLE_DRAWS = 12;

export async function runShuffleSweep(b: SweepBase, report: Report, cancelled: Cancelled): Promise<{ base: number; draws: number[] } | null> {
  await yieldToBrowser();
  const base = predict(b, b.L, NO_TOKEN_ABLATION);
  report(1, SHUFFLE_DRAWS + 1);
  const draws: number[] = [];
  for (let s = 1; s <= SHUFFLE_DRAWS; s++) {
    await yieldToBrowser();
    if (cancelled()) return null;
    draws.push(predict(b, b.L, { ...NO_TOKEN_ABLATION, shuffle: true, shuffleSeed: s }));
    report(s + 1, SHUFFLE_DRAWS + 1);
  }
  return { base, draws };
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Context length
// ─────────────────────────────────────────────────────────────────────────────

export const L_SWEEP = [16, 32, 48, 64, 96, 128, 160, 192, 256];

export interface LPoint {
  L: number;
  output: number;
  real: number;
  self: number;
  other: number;
}

export async function runLSweep(b: SweepBase, report: Report, cancelled: Cancelled): Promise<LPoint[] | null> {
  const Ls = [...new Set([...L_SWEEP, b.L])].sort((x, y) => x - y);
  const out: LPoint[] = [];
  for (let i = 0; i < Ls.length; i++) {
    await yieldToBrowser();
    if (cancelled()) return null;
    const L = Ls[i];
    const ctx = ctxAt(b, L);
    const lab = inContextLabels(b.pdb, ctx, b.task);
    out.push({ L, output: predict(b, L, NO_TOKEN_ABLATION), real: ctx.numReal, self: lab.self.length, other: lab.other.length });
    report(i + 1, Ls.length);
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Attention layers off, one at a time
// ─────────────────────────────────────────────────────────────────────────────

export interface LayerPoint {
  off: AttnLayer[];
  output: number;
}

export async function runLayerSweep(b: SweepBase, report: Report, cancelled: Cancelled): Promise<{ base: number; items: LayerPoint[] } | null> {
  const sets: AttnLayer[][] = [['col'], ['feat'], ['nbr'], ['full'], ['col', 'feat', 'nbr', 'full']];
  await yieldToBrowser();
  const base = predict(b, b.L, NO_TOKEN_ABLATION);
  report(1, sets.length + 1);
  const items: LayerPoint[] = [];
  for (let i = 0; i < sets.length; i++) {
    await yieldToBrowser();
    if (cancelled()) return null;
    items.push({ off: sets[i], output: predict(b, b.L, NO_TOKEN_ABLATION, sets[i]) });
    report(i + 2, sets.length + 1);
  }
  return { base, items };
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Batch over every seed of a split
// ─────────────────────────────────────────────────────────────────────────────

export interface BatchSide {
  L: number;
  ab: TokenAblation;
  disabled: AttnLayer[];
  weights: RTWeights;
}

export interface BatchItem {
  seed: number;
  /** Stored z-scored label (main.py:285/288). */
  z: number;
  base: number;
  abl: number;
}

export interface BatchResult {
  items: BatchItem[];
  metric: 'auroc' | 'r2';
  base: number;
  abl: number;
  msPerSeed: number;
}

/** AUROC on logits with labels int(z > 0), or R² on z-scores, exactly as main.py:328-337. */
export function metricOf(task: TaskDef, z: number[], pred: number[]): number {
  try {
    return task.targetType === 'binary'
      ? auroc(
          z.map((v) => binaryLabel(v)),
          pred,
        )
      : r2(z, pred);
  } catch {
    return NaN; // e.g. a single class: roc_auc_score raises
  }
}

export async function runBatch(
  common: Omit<SweepBase, 'seedNode' | 'L' | 'weights'> & { split: Split },
  baseSide: BatchSide,
  ablSide: BatchSide,
  report: Report,
  cancelled: Cancelled,
): Promise<BatchResult | null> {
  const seeds = getSeeds(common.pdb, common.task, common.split);
  const items: BatchItem[] = [];
  const t0 = performance.now();
  for (let i = 0; i < seeds.length; i++) {
    await yieldToBrowser();
    if (cancelled()) return null;
    const s = seeds[i];
    const run = (side: BatchSide) => {
      const ctx = sampleContext(common.pdb, makeSampleParams(common.pdb, common.task, s, side.L, common.w, common.rngSeed), { recordEvents: false });
      const abl = ablateContext(common.pdb, common.task, ctx, side.ab);
      const fr = ablatedForward(abl, common.pdb, common.store, side.weights, side.disabled, false);
      const tok = abl.tokens[fr.targetPos];
      return { pred: fr.prediction, z: common.task.targetType === 'binary' ? tok.booleanValue : tok.numberValue };
    };
    const a = run(baseSide);
    const bb = run(ablSide);
    items.push({ seed: s, z: a.z, base: a.pred, abl: bb.pred });
    report(i + 1, seeds.length);
  }
  const z = items.map((it) => it.z);
  return {
    items,
    metric: common.task.targetType === 'binary' ? 'auroc' : 'r2',
    base: metricOf(common.task, z, items.map((it) => it.base)),
    abl: metricOf(common.task, z, items.map((it) => it.abl)),
    msPerSeed: (performance.now() - t0) / Math.max(1, seeds.length),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Every switch over every seed (a Table 4 / 5 / 13 analogue)
// ─────────────────────────────────────────────────────────────────────────────

export interface TableSpec {
  id: string;
  ab: TokenAblation;
  disabled: AttnLayer[];
}

export interface TableResult {
  metric: 'auroc' | 'r2';
  n: number;
  base: number;
  /** spec id → metric over the same seeds. */
  rows: Record<string, number>;
  msPerPass: number;
}

/**
 * For every seed of the split: one context (the course's L, w, RNG seed), the unablated forward pass,
 * then one forward pass per spec. Metrics as main.py:328-337. Yields to the browser after each pass.
 */
export async function runAblationTable(
  common: Omit<SweepBase, 'seedNode'> & { split: Split },
  specs: TableSpec[],
  report: Report,
  cancelled: Cancelled,
): Promise<TableResult | null> {
  const seeds = getSeeds(common.pdb, common.task, common.split);
  const total = seeds.length * (specs.length + 1);
  const z: number[] = [];
  const base: number[] = [];
  const preds: Record<string, number[]> = Object.fromEntries(specs.map((s) => [s.id, []]));
  let done = 0;
  const t0 = performance.now();
  for (const s of seeds) {
    await yieldToBrowser();
    if (cancelled()) return null;
    const ctx = sampleContext(common.pdb, makeSampleParams(common.pdb, common.task, s, common.L, common.w, common.rngSeed), { recordEvents: false });
    const abl0 = ablateContext(common.pdb, common.task, ctx, NO_TOKEN_ABLATION);
    const fr0 = ablatedForward(abl0, common.pdb, common.store, common.weights, [], false);
    const tok = abl0.tokens[fr0.targetPos];
    z.push(common.task.targetType === 'binary' ? tok.booleanValue : tok.numberValue);
    base.push(fr0.prediction);
    report(++done, total);
    for (const sp of specs) {
      await yieldToBrowser();
      if (cancelled()) return null;
      const abl = ablateContext(common.pdb, common.task, ctx, sp.ab);
      preds[sp.id].push(ablatedForward(abl, common.pdb, common.store, common.weights, sp.disabled, false).prediction);
      report(++done, total);
    }
  }
  const rows: Record<string, number> = {};
  for (const sp of specs) rows[sp.id] = metricOf(common.task, z, preds[sp.id]);
  return {
    metric: common.task.targetType === 'binary' ? 'auroc' : 'r2',
    n: seeds.length,
    base: metricOf(common.task, z, base),
    rows,
    msPerPass: (performance.now() - t0) / Math.max(1, total),
  };
}
