/**
 * Smaller live figures: the column-name permutation of the current shuffle, the label encoding,
 * and this site's logged metrics per checkpoint (public/data/training/results.json).
 */
import { ArrowRight } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { DataTable, TeX, formatSigned } from '@/components';
import { loadPublicJson } from '@/engine/model/io';
import { getSeeds, inContextLabels, makeSampleParams } from '@/engine/pipeline';
import { sampleContext } from '@/engine/sampler';
import type { PreprocessedDB, SampledContext, TaskDef } from '@/engine/types';
import { useSim } from '@/state/store';
import { ablateContext } from './ablate';
import { usePlayLive } from './context';
import { useJob, useNearView, yieldToBrowser } from './jobs';
import { TABLE12_EXAMPLES } from './paper';
import { usePlay } from './state';
import { oid } from './usePlayground';

/** A context sampled with the engine's sampler, outside the shared cache (many seeds at once). */
function getContextNoCache(pdb: PreprocessedDB, task: TaskDef, seed: number, L: number, w: number, rngSeed: number): SampledContext {
  return sampleContext(pdb, makeSampleParams(pdb, task, seed, L, w, rngSeed), { recordEvents: false });
}

// ─────────────────────────────────────────────────────────────────────────────
// Names map
// ─────────────────────────────────────────────────────────────────────────────

/** Which column phrase each column of the live context reads as under the current shuffle draw. */
export function NamesMap() {
  const { pdb, task, ablCtx, tokenAb, abl } = usePlayLive();
  const on = usePlay((s) => s.shuffle);
  const set = usePlay((s) => s.set);
  const reshuffle = usePlay((s) => s.reshuffle);
  // show the current shuffle when it is on, otherwise a preview of the draw that switching it on gives
  const perm = useMemo(() => {
    if (abl.perm) return abl.perm;
    return ablateContext(pdb, task, ablCtx, { ...tokenAb, shuffle: true }).perm;
  }, [abl, pdb, task, ablCtx, tokenAb]);
  if (!perm) return null;
  const targetCol = ablCtx.params.targetColNameIdx;
  // the masked cell's column first, then alphabetical
  const rows = [...perm.entries()].sort((a, b) => (a[0] === targetCol ? -1 : b[0] === targetCol ? 1 : pdb.text[a[0]].localeCompare(pdb.text[b[0]])));
  return (
    <div className="card my-6 px-4 py-4 sm:px-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[17px] font-semibold text-ink">{on ? `Shuffle #${tokenAb.shuffleSeed}: what each column now reads as` : `Preview of shuffle #${tokenAb.shuffleSeed}`}</div>
          <p className="!m-0 mt-1 text-[15.5px] leading-snug text-ink-2">
            {rows.length} column phrases of the live context, permuted in one cycle. Left: the column a cell belongs to, which the masks use.
            Right: the phrase whose embedding its input now receives.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {!on && (
            <button type="button" onClick={() => set({ shuffle: true })} className="inline-flex h-9 items-center rounded-lg bg-accent px-3 text-[15px] font-medium text-accent-ink hover:brightness-110">
              Apply in the playground
            </button>
          )}
          <button type="button" onClick={reshuffle} className="inline-flex h-9 items-center rounded-lg border border-border-strong px-3 text-[15px] font-medium text-ink hover:bg-surface-2">
            Draw another
          </button>
        </div>
      </div>
      <div className="scroll-quiet mt-3 overflow-x-auto">
        <table className="w-full border-collapse text-[14.5px] [overflow-wrap:anywhere]">
          <caption className="sr-only">Column phrase permutation</caption>
          <thead>
            <tr className="border-b border-grid text-left text-ink-2">
              <th scope="col" className="py-1.5 pr-3 font-semibold">
                Cells of the column…
              </th>
              <th scope="col" className="w-6 py-1.5" aria-hidden="true" />
              <th scope="col" className="py-1.5 font-semibold">
                …receive the name embedding of
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([from, to]) => (
              <tr key={from} className={from === targetCol ? 'border-b border-grid/60 bg-accent-soft/50' : 'border-b border-grid/60'}>
                <td className="py-1 pr-3 font-mono text-ink">
                  <span className={from === targetCol ? 'font-semibold' : ''}>{pdb.text[from]}</span>
                  {from === targetCol && <span className="ml-2 font-sans text-[14px] text-muted">masked cell’s column</span>}
                </td>
                <td className="py-1 text-muted">
                  <ArrowRight aria-label="receives" size={14} />
                </td>
                <td className="py-1 font-mono text-ink-2">{pdb.text[to]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Label encoding
// ─────────────────────────────────────────────────────────────────────────────

/** The z-score an edited label gets, with the live Train statistics. */
export function LabelEncoding() {
  const { db, task, stats } = usePlayLive();
  // example raw values: the whole scale of a bounded autocomplete column (e.g. ratings 1-5), else a few counts
  const examples = useMemo(() => {
    if (task.kind !== 'autocomplete') return [0, 1, 5, 10];
    const rows = db.tables.find((t) => t.schema.name === task.table)?.rows ?? [];
    const vals = rows.map((r) => r[task.targetCol]).filter((v): v is number => typeof v === 'number');
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    return Number.isInteger(lo) && Number.isInteger(hi) && hi - lo <= 6 ? Array.from({ length: hi - lo + 1 }, (_, i) => lo + i) : [lo, hi];
  }, [db, task]);
  if (task.targetType === 'binary') {
    const zt = Math.fround((1 - stats.mean) / stats.std);
    const zf = Math.fround((0 - stats.mean) / stats.std);
    return (
      <div className="my-5 rounded-xl border border-border bg-surface px-4 py-3 text-[17px]">
        <TeX tex={String.raw`z = \frac{\mathbb{1}[y] - \mu}{\sigma},\quad \mu = ${stats.mean.toFixed(3)},\ \sigma = ${stats.std.toFixed(3)}`} />
        <div className="num mt-2 flex flex-wrap gap-x-6 gap-y-1 text-[16px] text-ink">
          <span>
            true → {formatSigned(zt, 3)}
          </span>
          <span>
            false → {formatSigned(zf, 3)}
          </span>
        </div>
      </div>
    );
  }
  return (
    <div className="my-5 rounded-xl border border-border bg-surface px-4 py-3 text-[17px]">
      <TeX tex={String.raw`z = \frac{y - \mu}{\sigma},\quad \mu = ${stats.mean.toFixed(3)},\ \sigma = ${stats.std.toFixed(3)}`} />
      <div className="num mt-2 flex flex-wrap gap-x-6 gap-y-1 text-[16px] text-ink">
        {examples.map((v) => (
          <span key={v}>
            {v} → {formatSigned(Math.fround((v - stats.mean) / stats.std), 3)}
          </span>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Logged metrics per checkpoint
// ─────────────────────────────────────────────────────────────────────────────

interface Split2 {
  step?: number;
  val?: number;
  test?: number;
}
interface ResultsJson {
  zeroShot?: Record<string, { metric?: string; pretrainBest?: Split2; untrainedInit?: Split2; entityMean?: { val?: number; test?: number } | null }>;
  finetune?: Record<string, { task?: string; best?: Split2 }>;
}

let cache: ResultsJson | null = null;

export function CkptMetrics() {
  const { task } = usePlayLive();
  const [d, setD] = useState<ResultsJson | null>(cache);
  const [err, setErr] = useState(false);
  useEffect(() => {
    if (cache) return;
    let alive = true;
    loadPublicJson<ResultsJson>('data/training/results.json').then(
      (x) => {
        cache = x;
        if (alive) setD(x);
      },
      () => alive && setErr(true),
    );
    return () => {
      alive = false;
    };
  }, []);
  if (err) return <p className="text-[15px] text-muted">The training log could not be loaded.</p>;
  const zs = d?.zeroShot?.[task.id];
  if (!d || !zs) return null;
  const metric = zs.metric === 'r2' ? 'R²' : 'AUROC';
  // the fine-tuned checkpoint of this site continues from the zero-shot one (run "<name>-pretrained")
  const ftEntry = Object.entries(d.finetune ?? {}).find(([k, f]) => f.task === task.id && k.endsWith('-pretrained'))?.[1];
  const f3 = (v: number | undefined) => (v === undefined ? '—' : v.toFixed(3));
  const rows: (string | number)[][] = [
    ['Untrained (init, step 0)', f3(zs.untrainedInit?.val), f3(zs.untrainedInit?.test)],
    [`Zero-shot (best val step ${zs.pretrainBest?.step ?? '?'})`, f3(zs.pretrainBest?.val), f3(zs.pretrainBest?.test)],
  ];
  if (zs.entityMean) rows.push(['EntityMean baseline', f3(zs.entityMean.val), f3(zs.entityMean.test)]);
  if (ftEntry?.best) rows.push([`Fine-tuned on ShopDB (step ${ftEntry.best.step ?? '?'})`, f3(ftEntry.best.val), f3(ftEntry.best.test)]);
  return (
    <div className="card my-6 px-4 py-4 sm:px-5">
      <div className="mb-2 text-[17px] font-semibold text-ink">
        {metric} of these checkpoints over all seeds of {task.name}, from our training log
      </div>
      <DataTable caption={`${metric} per checkpoint`} columns={['Checkpoint', task.kind === 'autocomplete' ? 'Val (even ids)' : 'Val', task.kind === 'autocomplete' ? 'Test (odd ids)' : 'Test']} rows={rows} />
      <p className="!mb-0 mt-2 text-[14.5px] text-muted">
        public/data/training/results.json, written by python/train.py. The zero-shot checkpoint is picked on Val, so its Test column is the fair
        number (spec §9.1). Small splits: expect large variance.
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// In-context label counts (Table 12 analogue)
// ─────────────────────────────────────────────────────────────────────────────

interface CountStats {
  n: number;
  self: [number, number];
  other: [number, number];
}

function meanSd(xs: number[]): [number, number] {
  const n = xs.length || 1;
  const m = xs.reduce((a, v) => a + v, 0) / n;
  const v = xs.reduce((a, x) => a + (x - m) ** 2, 0) / n;
  return [m, Math.sqrt(v)];
}

/** Self / other in-context labels per context over every seed of the live split, next to paper Table 12. */
export function LabelCounts() {
  const { pdb, task, globalL, w, rngSeed, abl, effL } = usePlayLive();
  const split = useSim((s) => s.split);
  const [ref, near] = useNearView<HTMLDivElement>();
  const forecast = task.kind === 'forecast';
  const key = near && forecast ? ['counts', oid(pdb), task.id, split, globalL, w, rngSeed].join('|') : null;
  const job = useJob<CountStats>(
    key,
    async (report, cancelled) => {
      const seeds = getSeeds(pdb, task, split);
      const self: number[] = [];
      const other: number[] = [];
      for (let i = 0; i < seeds.length; i++) {
        if (i % 20 === 0) {
          await yieldToBrowser();
          if (cancelled()) return null;
          report(i, seeds.length);
        }
        const ctx = getContextNoCache(pdb, task, seeds[i], globalL, w, rngSeed);
        const lab = inContextLabels(pdb, ctx, task);
        self.push(lab.self.length);
        other.push(lab.other.length);
      }
      return { n: seeds.length, self: meanSd(self), other: meanSd(other) };
    },
    true,
  );
  if (!forecast) return null;
  const r = job.result;
  const pm = (x: [number, number]) => `${Math.round(x[0])} ± ${Math.round(x[1])}`;
  return (
    <div ref={ref} className="card my-6 px-4 py-4 sm:px-5">
      <div className="text-[17px] font-semibold text-ink">How many labels does a context hold?</div>
      <p className="!m-0 mt-1 text-[15.5px] leading-snug text-ink-2">
        Mean ± sd per context, rounded like the paper’s Table 12. Live seed at L = {effL}: <strong className="num text-ink">{abl.labels.self.length}</strong> self,{' '}
        <strong className="num text-ink">{abl.labels.other.length}</strong> other.
      </p>
      <div className="scroll-quiet mt-3 overflow-x-auto">
        <table className="w-full min-w-[440px] border-collapse text-[15px]">
          <caption className="sr-only">In-context labels per context</caption>
          <thead>
            <tr className="border-b border-grid text-left text-ink-2">
              <th scope="col" className="py-1.5 pr-3 font-semibold">
                Task
              </th>
              <th scope="col" className="py-1.5 pr-3 text-right font-semibold">
                L
              </th>
              <th scope="col" className="py-1.5 pr-3 text-right font-semibold">
                self
              </th>
              <th scope="col" className="py-1.5 text-right font-semibold">
                other
              </th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-grid/60 bg-surface-2/70">
              <th scope="row" className="py-1.5 pr-3 text-left font-normal text-ink">
                ShopDB {task.name}, all {r ? r.n : ''} {split} seeds
              </th>
              <td className="num py-1.5 pr-3 text-right text-ink-2">{globalL}</td>
              <td className="num py-1.5 pr-3 text-right text-ink">{r ? pm(r.self) : '…'}</td>
              <td className="num py-1.5 text-right text-ink">{r ? pm(r.other) : '…'}</td>
            </tr>
            {TABLE12_EXAMPLES.map((e) => (
              <tr key={e.task} className="border-b border-grid/60">
                <th scope="row" className="py-1.5 pr-3 text-left font-normal text-ink">
                  Paper: {e.task}
                </th>
                <td className="num py-1.5 pr-3 text-right text-ink-2">1024</td>
                <td className="num py-1.5 pr-3 text-right text-ink">{e.self}</td>
                <td className="num py-1.5 text-right text-ink">{e.other}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="!mb-0 mt-2 text-[14.5px] leading-snug text-muted">
        Paper rows: a few of the 18 tasks of Table 12 (p.23). On rel-trial study-outcome no label reaches the context, and zero-shot AUROC there is
        51.8, close to random (Table 1). Table 12 dates from arXiv v1, when the public sampler still left out other entities’ task rows at the
        seed’s own timestamp; the released sampler keeps them, so other-entity counts here can be higher (spec §11.3).
      </p>
    </div>
  );
}
