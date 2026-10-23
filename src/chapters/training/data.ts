/**
 * This site's real training run, read at runtime from public/data/training/*.json (written by
 * python/train.py, python/README.md). Nothing here is hard-coded: every number the chapter shows
 * about our run comes from these files, so a re-run of the pipeline updates the page.
 *
 *   curves.json    per run: train loss, grad norm (and lr, β1 for pretraining) per step; every eval
 *   results.json   zero-shot per ShopDB task, EntityMean, fine-tuning, checkpoint provenance, config
 *   schedule.json  lr / β1 per step of our run, and the paper's 50,001-step OneCycle (sampled)
 *   dataset.json   tasks, seeds, contexts and tokens of every exported set
 *
 * Every field is optional in the types: the files come from a separate pipeline.
 */
import { useEffect, useState } from 'react';
import { loadPublicJson } from '@/engine/model/io';

// ─────────────────────────────────────────────────────────────────────────────
// File shapes
// ─────────────────────────────────────────────────────────────────────────────

export interface EvalPoint {
  step: number;
  /** "<db>/<task>/<Split>" → AUROC (binary) or R² (regression). */
  metrics: Record<string, number>;
  loss?: Record<string, number>;
}

export interface BestPick {
  step: number;
  val: number;
  test: number;
  all?: number;
}

export interface CurveRun {
  step: number[];
  loss: number[];
  gradNorm?: number[];
  lr?: number[];
  beta1?: number[];
  evals?: EvalPoint[];
  maxSteps?: number;
  completedSteps?: number;
  lrSchedule?: string;
  hp?: { lr?: number; wd?: number; max_grad_norm?: number; eval_freq?: number; eval_pow2?: boolean };
  trainSet?: string;
  best?: Record<string, BestPick>;
  wallSeconds?: number;
}

export interface Curves {
  mainPretrainRun?: string;
  note?: string;
  runs: Record<string, CurveRun>;
}

export interface ScheduleSeries {
  totalSteps: number;
  step: number[];
  lr: number[];
  beta1: number[];
}

export interface Schedule {
  note?: string;
  mini?: ScheduleSeries;
  paper?: ScheduleSeries;
}

export interface DatasetTask {
  id: string;
  kind: 'forecast' | 'autocomplete';
  targetType: 'binary' | 'regression';
  seeds: number;
}

export interface DatasetSet {
  contexts?: number;
  epochs?: number;
  itemsPerEpoch?: number;
  tasks?: string[];
  split?: string;
  tokens?: number;
  realTokens?: number;
  meanRealTokens?: number;
  meanSelfLabels?: number;
  meanOtherLabels?: number;
}

export interface Dataset {
  L?: number;
  w?: number;
  batchSize?: number;
  pretrain?: {
    dbs?: string[];
    heldOut?: string;
    numTasks?: number;
    tasks?: DatasetTask[];
    itemsPerEpoch?: number;
    exportedEpochs?: number;
    steps?: number;
    contextsSeen?: number;
    tokensSeen?: number;
    epochsSeen?: number;
  };
  paper?: { steps?: number; batch?: number; L?: number; contextsSeen?: number; tokensSeen?: number; note?: string };
  sets?: Record<string, DatasetSet>;
}

export interface SplitTriple {
  step?: number;
  val?: number;
  test?: number;
}

export interface ZeroShotEntry {
  metric?: 'auroc' | 'r2';
  kind?: 'forecast' | 'autocomplete';
  pretrainBest?: SplitTriple;
  pretrainFinal?: SplitTriple;
  untrainedInit?: SplitTriple;
  entityMean?: { val?: number; test?: number; fracSeedsWithPastLabels?: { val?: number; test?: number } } | null;
}

export interface FinetuneEntry {
  task?: string;
  init?: string;
  steps?: number;
  completedSteps?: number;
  best?: SplitTriple;
  step0?: SplitTriple;
  final?: SplitTriple;
}

export interface CheckpointInfo {
  name?: string;
  run?: string;
  step?: number;
  selection?: string;
  val?: number;
  test?: number;
}

export interface Results {
  note?: string;
  config?: {
    num_blocks?: number;
    d_model?: number;
    num_heads?: number;
    d_ff?: number;
    L?: number;
    w?: number;
    batch?: number;
    dtype?: string;
    device?: string;
    threads?: number;
    eps?: number;
  };
  pretrain?: { steps?: number; evalFreq?: number; wallSeconds?: number; stepSeconds?: number };
  zeroShot?: Record<string, ZeroShotEntry>;
  finetune?: Record<string, FinetuneEntry>;
  checkpoints?: Record<string, CheckpointInfo | number>;
  labelPersistence?: {
    tasks?: Record<string, { lag1?: number; db?: string; splits?: Record<string, { rows?: number; positiveRate?: number; mean?: number }> }>;
  };
  diagnostics?: {
    untrainedInitSeeds?: Record<string, Record<string, number>>;
    inContextLabelAblation?: Record<string, Record<string, { with?: number; withoutLabels?: number }>>;
  };
}

export interface TrainingData {
  curves: Curves | null;
  results: Results | null;
  schedule: Schedule | null;
  dataset: Dataset | null;
  status: 'loading' | 'ready' | 'error';
  error?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Loader (once per page load, shared by every figure)
// ─────────────────────────────────────────────────────────────────────────────

let cache: TrainingData | null = null;
let pending: Promise<TrainingData> | null = null;

function loadAll(): Promise<TrainingData> {
  pending ??= Promise.all([
    loadPublicJson<Curves>('data/training/curves.json'),
    loadPublicJson<Results>('data/training/results.json'),
    loadPublicJson<Schedule>('data/training/schedule.json'),
    loadPublicJson<Dataset>('data/training/dataset.json'),
  ]).then(
    ([curves, results, schedule, dataset]): TrainingData => (cache = { curves, results, schedule, dataset, status: 'ready' }),
    (e: unknown): TrainingData => {
      pending = null;
      return { curves: null, results: null, schedule: null, dataset: null, status: 'error', error: e instanceof Error ? e.message : String(e) };
    },
  );
  return pending;
}

const LOADING: TrainingData = { curves: null, results: null, schedule: null, dataset: null, status: 'loading' };

export function useTrainingData(): TrainingData {
  const [data, setData] = useState<TrainingData>(cache ?? LOADING);
  useEffect(() => {
    if (cache) return;
    let alive = true;
    loadAll().then((d) => alive && setData(d));
    return () => {
      alive = false;
    };
  }, []);
  return data;
}

// ─────────────────────────────────────────────────────────────────────────────
// Derived helpers
// ─────────────────────────────────────────────────────────────────────────────

export const SHOP_CHURN = 'shop/customer-churn';
export const SHOP_SALES = 'shop/product-sales';

/** The two ShopDB forecast tasks the paper protocol evaluates, with their metric. */
export const SHOP_FORECAST: { id: string; short: string; metric: 'auroc' | 'r2'; metricLabel: string; ft: 'churn' | 'sales' }[] = [
  { id: SHOP_CHURN, short: 'customer-churn', metric: 'auroc', metricLabel: 'AUROC', ft: 'churn' },
  { id: SHOP_SALES, short: 'product-sales', metric: 'r2', metricLabel: 'R²', ft: 'sales' },
];

export function mainRun(c: Curves | null): CurveRun | undefined {
  if (!c) return undefined;
  return c.runs[c.mainPretrainRun ?? 'pretrain'] ?? c.runs.pretrain;
}

/** Centered moving average (window w, shrinking at the ends). */
export function movingAverage(ys: readonly number[], w: number): number[] {
  const n = ys.length;
  const half = Math.floor(w / 2);
  const pre = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + ys[i];
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - half);
    const b = Math.min(n, i + half + 1);
    out[i] = (pre[b] - pre[a]) / (b - a);
  }
  return out;
}

/** Mean of a slice. */
export function mean(ys: readonly number[]): number {
  return ys.length ? ys.reduce((s, v) => s + v, 0) / ys.length : NaN;
}

/** The evaluation series of one metric key ("shop/customer-churn/Val") of a run. */
export function evalSeries(run: CurveRun | undefined, key: string): { x: number; y: number }[] {
  return (run?.evals ?? []).filter((e) => Number.isFinite(e.metrics[key])).map((e) => ({ x: e.step, y: e.metrics[key] }));
}

/**
 * Replay of main.py:377-392 on the logged evaluations: at every eval, if val > best so far, record
 * the test metric of the same step. `-inf` start, strict `>`, exactly as the code.
 */
export interface SelectionEvent {
  i: number;
  step: number;
  val: number;
  test: number;
  improved: boolean;
  bestStep: number;
  bestVal: number;
  bestTest: number;
}

export function replaySelection(run: CurveRun | undefined, task: string, valSplit = 'Val', testSplit = 'Test'): SelectionEvent[] {
  const out: SelectionEvent[] = [];
  let bestVal = -Infinity;
  let bestTest = NaN;
  let bestStep = -1;
  (run?.evals ?? []).forEach((e, i) => {
    const val = e.metrics[`${task}/${valSplit}`];
    const test = e.metrics[`${task}/${testSplit}`];
    if (!Number.isFinite(val)) return;
    const improved = val > bestVal;
    if (improved) {
      bestVal = val;
      bestTest = test;
      bestStep = e.step;
    }
    out.push({ i, step: e.step, val, test, improved, bestStep, bestVal, bestTest });
  });
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatting
// ─────────────────────────────────────────────────────────────────────────────

const nf = (d: number) => new Intl.NumberFormat('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

/** Integer with thousands separators. */
export const int = (v: number | undefined) => (v === undefined || !Number.isFinite(v) ? '–' : Math.round(v).toLocaleString('en-US'));

/** Fixed decimals with a real minus sign. */
export function fx(v: number | undefined | null, d = 3): string {
  if (v === undefined || v === null || !Number.isFinite(v)) return '–';
  const s = nf(d).format(Math.abs(v));
  return v < 0 && Number(s) !== 0 ? `−${s}` : s;
}

/** Metric in percent (AUROC and R² both ×100, like the paper's tables). */
export function pct(v: number | undefined | null, d = 1): string {
  if (v === undefined || v === null || !Number.isFinite(v)) return '–';
  return fx(v * 100, d);
}

/** Learning rates: 1e-3 → "1.0×10⁻³" style. */
export function sci(v: number | undefined, digits = 2): string {
  if (v === undefined || !Number.isFinite(v)) return '–';
  if (v === 0) return '0';
  let e = Math.floor(Math.log10(Math.abs(v)));
  let m = Number((v / 10 ** e).toFixed(digits));
  // 9.9999e-4 rounds to "10.0": renormalize to 1.0e-3
  if (Math.abs(m) >= 10) {
    e += 1;
    m = Number((v / 10 ** e).toFixed(digits));
  }
  const sup = String(e)
    .replace('-', '⁻')
    .replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(c)]);
  return `${m.toString()}×10${sup}`;
}

/** Short tokens count: 16,388,096 → "16.4 M", 13.1e9 → "13.1 B". */
export function big(v: number | undefined): string {
  if (v === undefined || !Number.isFinite(v)) return '–';
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)} B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)} M`;
  if (v >= 1e4) return `${(v / 1e3).toFixed(0)} k`;
  return int(v);
}

/** Minutes from seconds. */
export const minutes = (s: number | undefined) => (s === undefined || !Number.isFinite(s) ? '–' : `${Math.round(s / 60)} min`);

/** DB display names and categorical slots (same mapping as the semantics chapter: shop 1, school 2, racing 3, forum 4). */
export const DB_LABEL: Record<string, string> = { shop: 'ShopDB', school: 'SchoolDB', racing: 'RacingDB', forum: 'ForumDB' };
export const DB_SLOT: Record<string, number> = { shop: 1, school: 2, racing: 3, forum: 4 };
