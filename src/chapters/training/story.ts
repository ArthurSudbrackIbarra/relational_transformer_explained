/**
 * The data-dependent sentences of the chapter: every number the prose quotes about our run is
 * derived here from the training JSON files (never typed by hand), together with the direction
 * words ("above" / "below") so the text stays true if the pipeline is re-run.
 */
import { evalSeries, mainRun, mean, replaySelection, SHOP_CHURN, SHOP_SALES, type Curves, type Dataset, type Results } from './data';
import type { PretrainSet } from './pretrainSet';

export interface TaskStory {
  zsVal?: number;
  zsTest?: number;
  zsStep?: number;
  emVal?: number;
  emTest?: number;
  initVal?: number;
  initTest?: number;
  lastTest?: number;
  ftPreTest?: number;
  ftPreStep?: number;
  ftUntTest?: number;
  ftUntStep?: number;
  nVal?: number;
  nTest?: number;
  /** Largest jump of the val metric between consecutive evaluations after step 100. */
  valJump?: number;
  /** Fine-tuning from pretrained: step-0 test, min test over steps 1..8, peak test and its step, final test. */
  ft?: {
    step0?: number;
    earlyMin?: number;
    earlyMinStep?: number;
    peak?: number;
    peakStep?: number;
    final?: number;
    finalStep?: number;
    untStep0?: number;
    untPeak?: number;
    untFinal?: number;
  };
}

export interface Story {
  churn: TaskStory;
  sales: TaskStory;
  autocompleteShare?: number;
  lag1Min?: number;
  lag1Max?: number;
  tokenRatio?: number;
  initSpread?: { lo: number; hi: number; n: number };
  ftEpochs?: number;
  trainRowsChurn?: number;
}

function taskStory(results: Results | null, curves: Curves | null, id: string, ft: 'churn' | 'sales'): TaskStory {
  const z = results?.zeroShot?.[id];
  const fp = results?.finetune?.[`finetune-${ft}-pretrained`];
  const fu = results?.finetune?.[`finetune-${ft}-untrained`];
  const lp = results?.labelPersistence?.tasks?.[id]?.splits;
  const run = mainRun(curves);
  const sel = replaySelection(run, id);
  let valJump = 0;
  for (let i = 1; i < sel.length; i++) if (sel[i].step > 100) valJump = Math.max(valJump, Math.abs(sel[i].val - sel[i - 1].val));
  const pre = curves?.runs[`finetune-${ft}-pretrained`];
  const unt = curves?.runs[`finetune-${ft}-untrained`];
  const pt = evalSeries(pre, `${id}/Test`);
  const ut = evalSeries(unt, `${id}/Test`);
  let ftStory: TaskStory['ft'];
  if (pt.length) {
    const early = pt.filter((p) => p.x >= 1 && p.x <= 8);
    const em = early.reduce((a, b) => (b.y < a.y ? b : a), early[0] ?? pt[0]);
    const pk = pt.reduce((a, b) => (b.y > a.y ? b : a), pt[0]);
    ftStory = {
      step0: pt.find((p) => p.x === 0)?.y,
      earlyMin: em?.y,
      earlyMinStep: em?.x,
      peak: pk.y,
      peakStep: pk.x,
      final: pt[pt.length - 1].y,
      finalStep: pt[pt.length - 1].x,
      untStep0: ut.find((p) => p.x === 0)?.y,
      untPeak: ut.length ? Math.max(...ut.map((p) => p.y)) : undefined,
      untFinal: ut.length ? ut[ut.length - 1].y : undefined,
    };
  }
  return {
    zsVal: z?.pretrainBest?.val,
    zsTest: z?.pretrainBest?.test,
    zsStep: z?.pretrainBest?.step,
    emVal: z?.entityMean?.val,
    emTest: z?.entityMean?.test,
    initVal: z?.untrainedInit?.val,
    initTest: z?.untrainedInit?.test,
    lastTest: z?.pretrainFinal?.test,
    ftPreTest: fp?.best?.test,
    ftPreStep: fp?.best?.step,
    ftUntTest: fu?.best?.test,
    ftUntStep: fu?.best?.step,
    nVal: lp?.Val?.rows,
    nTest: lp?.Test?.rows,
    valJump: sel.length ? valJump : undefined,
    ft: ftStory,
  };
}

export function buildStory(results: Results | null, curves: Curves | null, dataset: Dataset | null, set: PretrainSet | null): Story {
  const story: Story = { churn: taskStory(results, curves, SHOP_CHURN, 'churn'), sales: taskStory(results, curves, SHOP_SALES, 'sales') };
  if (set) {
    const auto = set.tasks.filter((t) => t.task.kind === 'autocomplete').reduce((s, t) => s + t.seeds.length, 0);
    story.autocompleteShare = auto / set.N;
  }
  const lags = Object.entries(results?.labelPersistence?.tasks ?? {})
    .filter(([id, t]) => !id.startsWith('shop/') && Number.isFinite(t.lag1))
    .map(([, t]) => t.lag1 as number);
  if (lags.length) {
    story.lag1Min = Math.min(...lags);
    story.lag1Max = Math.max(...lags);
  }
  const pt = dataset?.paper?.tokensSeen;
  const mt = dataset?.pretrain?.tokensSeen;
  if (pt && mt) story.tokenRatio = pt / mt;
  const seeds = results?.diagnostics?.untrainedInitSeeds;
  if (seeds) {
    const v = Object.values(seeds)
      .map((m) => m[`${SHOP_CHURN}/Test`])
      .filter(Number.isFinite);
    if (v.length) story.initSpread = { lo: Math.min(...v), hi: Math.max(...v), n: v.length };
  }
  const fset = dataset?.sets?.['finetune/shop/customer-churn'];
  const ftRun = curves?.runs['finetune-churn-pretrained'];
  story.trainRowsChurn = fset?.itemsPerEpoch;
  if (fset?.itemsPerEpoch && ftRun?.maxSteps) story.ftEpochs = (ftRun.maxSteps * (dataset?.batchSize ?? 32)) / fset.itemsPerEpoch;
  return story;
}

/** "above" / "below" / "level with" for a − b. */
export function cmp(a: number | undefined, b: number | undefined, tol = 0.0005): string {
  if (a === undefined || b === undefined || !Number.isFinite(a) || !Number.isFinite(b)) return 'compared with';
  if (Math.abs(a - b) <= tol) return 'level with';
  return a > b ? 'above' : 'below';
}

export { mean };
