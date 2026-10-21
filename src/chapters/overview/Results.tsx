/**
 * Headline results: the paper's published numbers (Tables 1, 2, 10; copied exactly from the PDF,
 * spec §11.1) and this site's miniature on ShopDB (public/data/training/results.json, written by
 * python/train.py; never hard-coded here).
 */
import { useEffect, useState } from 'react';
import { BarChart, Cite, Segmented, Stat, StatusBadge, formatNumber } from '@/components';
import { getSeeds } from '@/engine/pipeline';
import { PAPER_CONFIG, paramCount } from '@/engine/model';
import { loadPublicJson } from '@/engine/model/io';
import { findTask, getPreprocessed } from '@/engine/pipeline';
import { getDatabase } from '@/engine/db/registry';
import { int } from './live';

// Paper Table 1 (p.8): mean zero-shot test AUROC (%) over 10 binary tasks.
const AUROC_T1: { label: string; v: number; rt?: boolean }[] = [
  { label: 'Gemma3-4B', v: 63.3 },
  { label: 'Gemma3-12B', v: 62.4 },
  { label: 'Gemma3-27B', v: 64.6 },
  { label: 'EntityMean', v: 66.7 },
  { label: 'Griffin', v: 64.0 },
  { label: 'RT', v: 69.7, rt: true },
  { label: 'RelLLM, seen', v: 63.2 },
  { label: 'Griffin, seen', v: 67.3 },
  { label: 'RT, seen', v: 71.9, rt: true },
];
// Paper Table 2 (p.8): mean zero-shot test R² (%) over 8 regression tasks (LLMs: Table 16, below −9).
const R2_T2: { label: string; v: number; rt?: boolean }[] = [
  { label: 'EntityMean', v: 19.3 },
  { label: 'Griffin', v: 8.9 },
  { label: 'RT', v: 22.7, rt: true },
  { label: 'Griffin, seen', v: 10.1 },
  { label: 'RT, seen', v: 26.4, rt: true },
];
const SUPERVISED_AUROC = 77.2; // Table 10 (p.21), fine-tuned pretrained RT

export function PaperHeadline() {
  const [metric, setMetric] = useState<'auroc' | 'r2'>('auroc');
  const rows = metric === 'auroc' ? AUROC_T1 : R2_T2;
  const share = (v: number) => `${((v / SUPERVISED_AUROC) * 100).toFixed(1)}%`;
  return (
    <>
      <div className="wide my-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label="Parameters"
          value="22.3M"
          caption={
            <>
              {int(paramCount(PAPER_CONFIG))} by module count; “about 22M” in the paper <Cite kind="paper" ref="p.6 §4.1" />
              <Cite kind="spec" ref="§7.3" />
            </>
          }
        />
        <Stat
          label="Mean zero-shot AUROC, unseen database"
          value="69.7"
          delta={69.7 - 64.6}
          deltaLabel="vs Gemma3-27B"
          emphasis
          caption={
            <>
              10 binary tasks, random = 50 <Cite kind="paper" ref="p.8 Table 1" />
            </>
          }
        />
        <Stat
          label="Of supervised AUROC, zero-shot"
          value={share(69.7)}
          caption={
            <>
              69.7 / 77.2, the fine-tuned RT <Cite kind="paper" ref="p.2 §1" />
              <Cite kind="paper" ref="p.21 Table 10" />
            </>
          }
        />
        <Stat
          label="With continued pretraining"
          value={share(71.9)}
          caption={
            <>
              71.9 / 77.2; the database was seen, the task was not <Cite kind="paper" ref="p.2 §1" />
            </>
          }
        />
      </div>
      <div className="wide card my-6 px-4 py-4 sm:px-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <Segmented
            label="Metric"
            value={metric}
            onChange={setMetric}
            options={[
              { value: 'auroc', label: 'AUROC, 10 binary tasks' },
              { value: 'r2', label: 'R², 8 regression tasks' },
            ]}
          />
        </div>
        <BarChart
          title={metric === 'auroc' ? 'Mean zero-shot test AUROC (%), paper Table 1' : 'Mean zero-shot test R² (%), paper Table 2'}
          subtitle={
            metric === 'auroc' ? (
              <>
                “seen”: the target database (not the task) was in pretraining; all others never saw it, except Gemma3, whose pretraining data is
                unknown <Cite kind="paper" ref="p.7 §4.3" />.
              </>
            ) : (
              <>
                “seen”: the target database (not the task) was in pretraining. The Gemma3 models score below −9 on most regression tasks and are left
                out <Cite kind="paper" ref="p.26 Table 16" />.
              </>
            )
          }
          categories={rows.map((r) => r.label)}
          series={[
            {
              id: metric,
              label: metric === 'auroc' ? 'AUROC (%)' : 'R² (%)',
              values: rows.map((r) => r.v),
              slot: 1,
            },
          ]}
          emphasis={rows.filter((r) => r.rt).map((r) => r.label)}
          domain={metric === 'auroc' ? [0, 80] : [0, 30]}
          refLines={metric === 'auroc' ? [{ value: 50, label: 'random = 50' }] : undefined}
          format={(v) => formatNumber(v, 1)}
          valueLabel={metric === 'auroc' ? 'AUROC (%)' : 'R² (%)'}
        />
      </div>
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// This site's miniature (results.json)
// ─────────────────────────────────────────────────────────────────────────────

export interface ZsEntry {
  metric: 'auroc' | 'r2';
  pretrainBest: { step: number; val: number; test: number };
  untrainedInit: { val: number; test: number };
  entityMean: { val: number; test: number } | null;
}
export interface ResultsJson {
  zeroShot: Record<string, ZsEntry>;
  pretrain?: { steps: number; wallSeconds?: number };
  config?: {
    batch?: number;
    dtype?: string;
    device?: string;
    attention?: { kernel?: string; maxAbsDiffForward?: number };
  };
}

let resultsPromise: Promise<ResultsJson> | null = null;

/** public/data/training/results.json (python/train.py), loaded once per page. */
export function useResultsJson(): {
  data: ResultsJson | null;
  error: string | null;
} {
  const [state, setState] = useState<{
    data: ResultsJson | null;
    error: string | null;
  }>({ data: null, error: null });
  useEffect(() => {
    let alive = true;
    if (!resultsPromise) {
      resultsPromise = loadPublicJson<ResultsJson>('data/training/results.json');
      resultsPromise.catch(() => {
        resultsPromise = null;
      });
    }
    resultsPromise.then(
      (d) => alive && setState({ data: d, error: null }),
      (e: unknown) =>
        alive &&
        setState({
          data: null,
          error: e instanceof Error ? e.message : String(e),
        }),
    );
    return () => {
      alive = false;
    };
  }, []);
  return state;
}

const MINI_TASKS = ['shop/customer-churn', 'shop/product-sales'];

export function MiniResults() {
  const { data, error: err } = useResultsJson();
  if (err) return <StatusBadge status="warning">Training results unavailable: {err}</StatusBadge>;
  if (!data) return <p className="text-[16px] text-ink-2">Loading this site’s training results…</p>;
  const db = getDatabase('shop');
  const pdb = getPreprocessed(db);
  const pct = (v: number | undefined) => (v === undefined || !Number.isFinite(v) ? '—' : formatNumber(v * 100, 1));
  return (
    <div className="wide card my-6 px-0 py-0">
      <div className="px-4 pt-3.5 pb-2 text-[16.5px] font-semibold text-ink" id="mini-results-title">
        This site’s miniature on ShopDB, zero-shot (test split, %)
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[620px] border-collapse text-[15px]" aria-labelledby="mini-results-title">
          <thead>
            <tr className="border-y border-grid bg-surface-2 text-left text-ink-2">
              <th className="px-4 py-2 font-medium">task</th>
              <th className="px-3 py-2 font-medium">metric</th>
              <th className="px-3 py-2 text-right font-medium">test rows</th>
              <th className="px-3 py-2 text-right font-semibold text-ink">mini RT, zero-shot</th>
              <th className="px-3 py-2 text-right font-medium">EntityMean</th>
              <th className="px-4 py-2 text-right font-medium">untrained RT</th>
            </tr>
          </thead>
          <tbody>
            {MINI_TASKS.map((id) => {
              const z = data.zeroShot[id];
              if (!z) return null;
              const task = findTask(db, id);
              const n = getSeeds(pdb, task, 'Test').length;
              return (
                <tr key={id} className="border-b border-grid/70 last:border-0">
                  <td className="px-4 py-2 font-mono text-ink">{task.name}</td>
                  <td className="px-3 py-2 text-ink-2">{z.metric === 'auroc' ? 'AUROC' : 'R²'}</td>
                  <td className="num px-3 py-2 text-right">{int(n)}</td>
                  <td className="num px-3 py-2 text-right font-semibold text-ink">{pct(z.pretrainBest.test)}</td>
                  <td className="num px-3 py-2 text-right">{pct(z.entityMean?.test)}</td>
                  <td className="num px-4 py-2 text-right">{pct(z.untrainedInit.test)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="!m-0 border-t border-grid px-4 py-3 text-[15px] leading-snug text-ink-2">
        Checkpoint chosen per task on ShopDB’s validation split, then scored on its test split, as in the paper <Cite kind="spec" ref="§9.1" />. One
        training run and this few test rows make the scores noisy. They show that the miniature runs the paper’s protocol, not that it reproduces the
        paper’s numbers. EntityMean winning a regression task happens in the paper too: it beats RT on item-ltv and post-votes{' '}
        <Cite kind="paper" ref="p.8 Table 2" />. Chapter 13 has the training curves.
      </p>
    </div>
  );
}
