/**
 * Batch mode. Two evaluations over every seed of the live split, each against the full model on the
 * same seeds, with metrics exactly as main.py:328-337 (AUROC on logits with labels int(z > 0), R² on
 * z-scores; spec §8.5):
 *  - AblationTable: each switch on its own (a Table 4 / 5 / 13 analogue), next to the paper's means;
 *  - CurrentAblation: the playground's current combination (minus seed-specific label edits).
 * Jobs are on demand, chunked one forward pass per macrotask, cancellable and cached (jobs.ts).
 */
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { ChartFrame, DataTable, LAYER_INFO, Stat } from '@/components';
import { loadPublicJson } from '@/engine/model/io';
import { autoCheckpoint, getSeeds } from '@/engine/pipeline';
import { PAPER_CONFIG, paramCount } from '@/engine/model';
import { sigmoid } from '@/engine/metrics';
import { ATTN_ORDER, type AttnLayer } from '@/engine/types';
import { useSim } from '@/state/store';
import { catVar } from '@/styles/palette';
import { NO_TOKEN_ABLATION, tokenAblationKey, type TokenAblation } from './ablate';
import { JobProgress } from './bits';
import { usePlayLive } from './context';
import { useJob } from './jobs';
import { HEADLINE, LAYER_ABLATION, paperMetric, paperNum, TABLE4, type ContextAblationRow } from './paper';
import { runAblationTable, runBatch, type BatchResult, type TableResult } from './sweeps';
import { oid } from './usePlayground';

interface MiniZs {
  zeroShot?: Record<string, { pretrainBest?: { val?: number; test?: number }; all?: { pretrainBest?: number } }>;
}
let resultsCache: MiniZs | null = null;
function useResultsJson(): MiniZs | null {
  const [d, setD] = useState<MiniZs | null>(resultsCache);
  useEffect(() => {
    if (resultsCache) return;
    let alive = true;
    loadPublicJson<MiniZs>('data/training/results.json').then(
      (x) => {
        resultsCache = x;
        if (alive) setD(x);
      },
      () => undefined,
    );
    return () => {
      alive = false;
    };
  }, []);
  return d;
}

const fmtParams = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${Math.round(n / 1e3)}k`);
const pts = (v: number) => (Number.isFinite(v) ? (v * 100).toFixed(1) : '—');
const signedPts = (d: number) => (Number.isFinite(d) ? `${d > 0 ? '+' : d < 0 ? '−' : '±'}${Math.abs(d).toFixed(1)}` : '—');

/** Measured cost of one forward pass (capture off), updated after every batch; a first guess before that. */
let msPerPass = 75;

function estimate(passes: number, L: number): string {
  const sec = Math.max(1, Math.round((passes * msPerPass * Math.max(1, (L / 128) ** 1.4)) / 1000));
  return sec < 90 ? `about ${sec} s` : `about ${Math.round(sec / 60)} min`;
}

function RunButton({ onClick, disabled, children }: { onClick: () => void; disabled?: boolean; children: ReactNode }) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className="inline-flex h-10 shrink-0 items-center rounded-lg bg-accent px-4 text-[16px] font-medium text-accent-ink hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45"
    >
      {children}
    </button>
  );
}

export function BatchMode() {
  return (
    <div className="flex min-w-0 flex-col gap-5">
      <AblationTable />
      <CurrentAblation />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Every switch, every seed: our Table 4 / 5 / 13 next to the paper's
// ─────────────────────────────────────────────────────────────────────────────

interface Spec {
  id: string;
  label: string;
  group: 'context' | 'layers';
  ab: TokenAblation;
  disabled: AttnLayer[];
  paper: ContextAblationRow;
}

function useSpecs(): Spec[] {
  const { task } = usePlayLive();
  return useMemo(() => {
    const out: Spec[] = [];
    if (task.kind === 'forecast') {
      out.push({ id: 'self', label: 'Self labels removed', group: 'context', ab: { ...NO_TOKEN_ABLATION, removeSelf: true }, disabled: [], paper: TABLE4.selfLabels });
      out.push({ id: 'other', label: 'Other-entity labels removed', group: 'context', ab: { ...NO_TOKEN_ABLATION, removeOther: true }, disabled: [], paper: TABLE4.otherLabels });
    }
    out.push({ id: 'names', label: 'Column names shuffled', group: 'context', ab: { ...NO_TOKEN_ABLATION, shuffle: true, shuffleSeed: 1 }, disabled: [], paper: TABLE4.colNames });
    for (const l of ATTN_ORDER)
      out.push({ id: l, label: `${LAYER_INFO[l].name} attention off`, group: 'layers', ab: NO_TOKEN_ABLATION, disabled: [l], paper: LAYER_ABLATION[l] });
    return out;
  }, [task.kind]);
}

/** A signed change as a small diverging bar on a shared scale, plus the number. */
function DeltaCell({ d, max }: { d: number | null; max: number }) {
  if (d === null || !Number.isFinite(d)) return <span className="num block text-right text-muted">—</span>;
  const w = Math.min(50, (Math.abs(d) / (max || 1)) * 50);
  return (
    <span className="flex items-center justify-end gap-2">
      <span className="relative hidden h-3 w-[84px] shrink-0 rounded-sm bg-surface-2 sm:block" aria-hidden="true">
        <span className="absolute -inset-y-0.5 left-1/2 w-px bg-axis" />
        <span
          className="absolute inset-y-0"
          style={{
            left: d < 0 ? `${50 - w}%` : '50%',
            width: `${w}%`,
            background: d < 0 ? 'var(--div-neg)' : 'var(--div-pos)',
            borderRadius: d < 0 ? '3px 0 0 3px' : '0 3px 3px 0',
          }}
        />
      </span>
      <span className="num w-[52px] text-right text-ink-2">{signedPts(d)}</span>
    </span>
  );
}

function AblationTable() {
  const { pdb, task, globalL, w, rngSeed, baseAssets, baseCkpt } = usePlayLive();
  const split = useSim((s) => s.split);
  const binary = task.targetType === 'binary';
  const m = paperMetric(binary);
  const specs = useSpecs();
  const seeds = useMemo(() => getSeeds(pdb, task, split), [pdb, task, split]);
  const ready = baseAssets.status === 'ready' && !!baseAssets.store && !!baseAssets.weights;
  const key = ready
    ? ['table', oid(pdb), task.id, task.kind === 'autocomplete' ? '*' : split, globalL, w, rngSeed, oid(baseAssets.store!), oid(baseAssets.weights!)].join('|')
    : null;
  const job = useJob<TableResult>(
    key,
    async (rep, cancelled) => {
      const r = await runAblationTable(
        { pdb, task, L: globalL, w, rngSeed, store: baseAssets.store!, weights: baseAssets.weights!, split },
        specs.map((s) => ({ id: s.id, ab: s.ab, disabled: s.disabled })),
        rep,
        cancelled,
      );
      if (r) msPerPass = r.msPerPass;
      return r;
    },
    false,
  );
  const r = job.result;
  const passes = seeds.length * (specs.length + 1);
  const ours = (id: string) => (r ? r.rows[id] * 100 : null);
  const oursBase = r ? r.base * 100 : null;
  const deltas = specs.flatMap((s) => {
    const o = ours(s.id);
    return [s.paper[m.key] - TABLE4.none[m.key], o !== null && oursBase !== null ? o - oursBase : 0];
  });
  const max = Math.max(1, ...deltas.map((d) => (Number.isFinite(d) ? Math.abs(d) : 0)));
  const groups: { id: Spec['group']; title: string; how: string }[] = [
    {
      id: 'context',
      title: 'Context ablations (Table 4)',
      how: 'Paper and here: applied to the sampled context at inference (p.22 App. G).',
    },
    {
      id: 'layers',
      title: binary ? 'Attention layers (Table 13)' : 'Attention layers (Table 5)',
      how: 'Paper: a new model trained without the layer. Here: a trained layer skipped at inference. Not comparable.',
    },
  ];

  return (
    <div className="card min-w-0 px-4 py-4 sm:px-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1 basis-[300px]">
          <div className="text-[17px] font-semibold text-ink">Every switch over every seed, next to the paper</div>
          <p className="!m-0 mt-1 text-[15.5px] leading-snug text-ink-2">
            {task.kind === 'autocomplete' ? `All ${seeds.length} rows of ${task.table}` : `All ${seeds.length} ${split} seeds of ${task.name}`}, checkpoint{' '}
            <span className="font-mono text-[15px]">{baseCkpt}</span>, L = {globalL}: one context per seed, the full model, then each switch on its
            own. {m.name} in %, as the paper prints it.
          </p>
        </div>
        {job.status !== 'running' && job.status !== 'waiting' && (
          <div className="flex flex-col items-start gap-1 sm:items-end">
            <RunButton onClick={job.start} disabled={!key}>
              {r ? 'Run again' : `Run ${passes.toLocaleString('en-US')} forward passes`}
            </RunButton>
            {!r && <span className="text-[14px] text-muted">{estimate(passes, globalL)}; kept for this page</span>}
          </div>
        )}
      </div>
      {(job.status === 'running' || job.status === 'waiting' || job.status === 'error') && (
        <div className="mt-3">
          <JobProgress job={job} label="forward passes" />
        </div>
      )}
      <div className="scroll-quiet mt-4 overflow-x-auto">
        <table className="w-full min-w-[520px] border-collapse text-[15px]">
          <caption className="sr-only">{m.name} with each switch: this model on ShopDB, and the paper’s mean over RelBench</caption>
          <thead>
            <tr className="text-ink-2">
              <th scope="col" rowSpan={2} className="border-b border-grid py-1.5 pr-3 text-left align-bottom font-semibold">
                Switch
              </th>
              <th scope="colgroup" colSpan={2} className="px-2 pt-1.5 text-right font-semibold">
                This model, ShopDB
              </th>
              <th scope="colgroup" colSpan={2} className="pt-1.5 pl-2 text-right font-semibold">
                Paper, RelBench mean
              </th>
            </tr>
            <tr className="border-b border-grid text-right text-[14.5px] text-ink-2">
              <th scope="col" className="px-2 py-1 font-medium">
                {m.name}
              </th>
              <th scope="col" className="px-2 py-1 font-medium">
                change
              </th>
              <th scope="col" className="px-2 py-1 font-medium">
                {m.name}
              </th>
              <th scope="col" className="py-1 pl-2 font-medium">
                change
              </th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-grid/60 bg-surface-2/70">
              <th scope="row" className="py-1.5 pr-3 text-left font-semibold text-ink">
                None (full model)
              </th>
              <td className="num px-2 py-1.5 text-right font-semibold text-ink">{oursBase !== null ? oursBase.toFixed(1) : '—'}</td>
              <td className="px-2 py-1.5 text-right text-muted">—</td>
              <td className="num px-2 py-1.5 text-right font-semibold text-ink">{paperNum(TABLE4.none[m.key])}</td>
              <td className="py-1.5 pl-2 text-right text-muted">—</td>
            </tr>
            {groups.map((g) => {
              const rows = specs.filter((s) => s.group === g.id);
              if (rows.length === 0) return null;
              return [
                <tr key={`h-${g.id}`}>
                  <td colSpan={5} className="pt-3 pb-1">
                    <span className="text-[15px] font-semibold text-ink">{g.title}</span>
                    <span className="block text-[14px] leading-snug text-muted">{g.how}</span>
                  </td>
                </tr>,
                ...rows.map((s) => {
                  const o = ours(s.id);
                  return (
                    <tr key={s.id} className="border-b border-grid/60">
                      <th scope="row" className="py-1.5 pr-3 text-left font-normal text-ink">
                        {s.label}
                      </th>
                      <td className="num px-2 py-1.5 text-right text-ink">{o !== null && Number.isFinite(o) ? o.toFixed(1) : '—'}</td>
                      <td className="px-2 py-1.5">
                        <DeltaCell d={o !== null && oursBase !== null ? o - oursBase : null} max={max} />
                      </td>
                      <td className="num px-2 py-1.5 text-right text-ink">{paperNum(s.paper[m.key])}</td>
                      <td className="py-1.5 pl-2">
                        <DeltaCell d={s.paper[m.key] - TABLE4.none[m.key]} max={max} />
                      </td>
                    </tr>
                  );
                }),
              ];
            })}
          </tbody>
        </table>
      </div>
      <p className="!mb-0 mt-3 text-[14.5px] leading-snug text-muted">
        Bars share one scale in {m.name} points; blue is lower, red is higher. The paper’s “none” is the arXiv v1 run ({paperNum(TABLE4.none.zsClf)} /{' '}
        {paperNum(TABLE4.none.zsReg)}), not the headline {paperNum(HEADLINE.zsClf)} / {paperNum(HEADLINE.zsReg)} of Tables 1 and 2 (spec §10 #25).
        Its means cover {m.tasks} and a {fmtParams(paramCount(PAPER_CONFIG))}-parameter model; ours one task of a toy database and a{' '}
        {baseAssets.weights ? fmtParams(paramCount(baseAssets.weights.config)) : 'small'}-parameter model. Compare directions, not sizes.
        {r ? ` One forward pass took ${r.msPerPass.toFixed(0)} ms in your browser.` : ''}
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// The playground's current combination over every seed
// ─────────────────────────────────────────────────────────────────────────────

function CurrentAblation() {
  const live = usePlayLive();
  const { pdb, task, globalL, w, rngSeed, baseAssets, ablAssets, effL, tokenAb, disabled, baseCkpt, ablCkpt } = live;
  const split = useSim((s) => s.split);
  const results = useResultsJson();
  const binary = task.targetType === 'binary';
  const metricName = binary ? 'AUROC' : 'R²';
  const seeds = useMemo(() => getSeeds(pdb, task, split), [pdb, task, split]);
  const batchAb = useMemo(() => ({ ...tokenAb, edits: NO_TOKEN_ABLATION.edits }), [tokenAb]);
  const nEdits = tokenAb.edits.size;
  const ablActive = tokenAblationKey(batchAb) !== tokenAblationKey(NO_TOKEN_ABLATION) || effL !== globalL || disabled.length > 0 || ablCkpt !== baseCkpt;

  const ready = baseAssets.status === 'ready' && ablAssets.status === 'ready' && !!baseAssets.store && !!baseAssets.weights && !!ablAssets.weights;
  const key = ready
    ? [
        'batch',
        oid(pdb),
        task.id,
        task.kind === 'autocomplete' ? '*' : split,
        w,
        rngSeed,
        oid(baseAssets.store!),
        `${globalL}:${oid(baseAssets.weights!)}`,
        `${effL}:${oid(ablAssets.weights!)}:${tokenAblationKey(batchAb)}:${[...disabled].sort().join(',')}`,
      ].join('|')
    : null;

  const job = useJob<BatchResult>(
    key,
    async (rep, cancelled) => {
      const r = await runBatch(
        { pdb, task, w, rngSeed, store: baseAssets.store!, split },
        { L: globalL, ab: NO_TOKEN_ABLATION, disabled: [], weights: baseAssets.weights! },
        { L: effL, ab: batchAb, disabled: [...disabled], weights: ablAssets.weights! },
        rep,
        cancelled,
      );
      if (r && effL === globalL) msPerPass = r.msPerSeed / 2;
      return r;
    },
    false,
  );
  const r = job.result;

  // the number python/train.py logged for this checkpoint (only meaningful at the course defaults)
  const logged = useMemo(() => {
    const zs = results?.zeroShot?.[task.id];
    if (!zs || globalL !== 128 || w !== 16 || rngSeed !== 0) return null;
    if (task.kind === 'autocomplete') return zs.all?.pretrainBest ?? null;
    return split === 'Val' ? (zs.pretrainBest?.val ?? null) : split === 'Test' ? (zs.pretrainBest?.test ?? null) : null;
  }, [results, task, globalL, w, rngSeed, split]);
  const loggedApplies = logged !== null && baseCkpt === autoCheckpoint(task.id).name;
  const chips = live.chips.filter((c) => !c.includes('edited'));

  return (
    <div className="card min-w-0 px-4 py-4 sm:px-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1 basis-[300px]">
          <div className="text-[17px] font-semibold text-ink">Your combination over every seed</div>
          <p className="!m-0 mt-1 text-[15.5px] leading-snug text-ink-2">
            Everything switched on in the playground at once, over{' '}
            {task.kind === 'autocomplete' ? `all ${seeds.length} rows of ${task.table}` : `all ${seeds.length} ${split} seeds`}, against the full model on
            the same seeds.{' '}
            {ablActive ? (
              <>
                Now on: <span className="text-ink">{chips.join(' · ')}</span>.
              </>
            ) : (
              <span className="text-ink">Nothing is switched on yet: turn something on in the playground above.</span>
            )}
          </p>
          {nEdits > 0 && <p className="!m-0 mt-1 text-[14.5px] text-muted">Your label edits belong to one context, so the batch leaves them out.</p>}
        </div>
        {job.status !== 'running' && job.status !== 'waiting' && (
          <div className="flex flex-col items-start gap-1 sm:items-end">
            <RunButton onClick={job.start} disabled={!key || !ablActive}>
              {r ? 'Run again' : `Run on ${seeds.length} seeds`}
            </RunButton>
            {!r && ablActive && <span className="text-[14px] text-muted">{estimate(seeds.length * 2, Math.max(effL, globalL))}</span>}
          </div>
        )}
      </div>
      {(job.status === 'running' || job.status === 'waiting' || job.status === 'error') && (
        <div className="mt-3">
          <JobProgress job={job} label="seeds" />
        </div>
      )}
      {r && (
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <Stat
            label={`Full model ${metricName}`}
            value={pts(r.base)}
            caption={loggedApplies ? `Our training log for this checkpoint: ${pts(logged!)}` : `${baseCkpt}, L = ${globalL}`}
          />
          <Stat label={`${metricName} with your changes`} value={pts(r.abl)} delta={(r.abl - r.base) * 100} deltaLabel="points" emphasis />
          <Stat label="Seeds" value={r.items.length} caption={`${r.msPerSeed.toFixed(0)} ms per seed for both passes`} />
        </div>
      )}
      {r && <Scatter r={r} binary={binary} />}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Seed-by-seed scatter
// ─────────────────────────────────────────────────────────────────────────────

function Scatter({ r, binary }: { r: BatchResult; binary: boolean }) {
  const { task, stats } = usePlayLive();
  const disp = (o: number) => (binary ? sigmoid(o) : o * stats.std + stats.mean);
  const xs = r.items.map((it) => disp(it.base));
  const ys = r.items.map((it) => disp(it.abl));
  const lo = binary ? 0 : Math.min(...xs, ...ys);
  const hi = binary ? 1 : Math.max(...xs, ...ys);
  const span = hi - lo || 1;
  const S = 300;
  const L = 52;
  const B = 50;
  const T = 10;
  const R = 12;
  const sx = (v: number) => L + ((v - lo) / span) * (S - L - R);
  const sy = (v: number) => S - B - ((v - lo) / span) * (S - B - T);
  const fmt = (v: number) => (binary ? `${Math.round(v * 100)}%` : v.toFixed(span < 3 ? 1 : 0));
  const ticks = [lo, lo + span / 2, hi];
  return (
    <ChartFrame
      className="mt-5"
      title="Seed by seed"
      subtitle="Each dot is one seed: the full model’s prediction (x) against the prediction with your changes (y). On the dashed diagonal nothing changed."
      legend={
        binary ? (
          <ul className="!m-0 flex !list-none flex-wrap gap-x-4 !p-0 text-[15px] text-ink-2" aria-label="Legend">
            <li className="!m-0 inline-flex items-center gap-1.5 !p-0">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: catVar(1) }} /> true {task.targetCol} = 0
            </li>
            <li className="!m-0 inline-flex items-center gap-1.5 !p-0">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: catVar(2) }} /> true {task.targetCol} = 1
            </li>
          </ul>
        ) : undefined
      }
      table={
        <DataTable
          caption="Per-seed predictions"
          columns={['Seed node', 'label z', 'full model', 'with your changes']}
          rows={r.items.map((it, i) => [`#${it.seed}`, it.z.toFixed(3), fmt(xs[i]), fmt(ys[i])])}
        />
      }
    >
      <svg width={S} height={S} viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Scatter of full-model against ablated predictions" className="block h-auto max-w-full">
        <line x1={sx(lo)} y1={sy(lo)} x2={sx(hi)} y2={sy(hi)} style={{ stroke: 'var(--axis)' }} strokeDasharray="4 4" />
        {ticks.map((t, i) => (
          <g key={i}>
            <text x={sx(t)} y={S - B + 18} textAnchor={i === 0 ? 'start' : i === 2 ? 'end' : 'middle'} className="num" style={{ fontSize: 13, fill: 'var(--muted)' }}>
              {fmt(t)}
            </text>
            <text x={L - 6} y={sy(t) + 4} textAnchor="end" className="num" style={{ fontSize: 13, fill: 'var(--muted)' }}>
              {fmt(t)}
            </text>
          </g>
        ))}
        <line x1={L} x2={S - R} y1={S - B} y2={S - B} style={{ stroke: 'var(--axis)' }} />
        <line x1={L} x2={L} y1={T} y2={S - B} style={{ stroke: 'var(--axis)' }} />
        <text x={(L + S - R) / 2} y={S - 8} textAnchor="middle" style={{ fontSize: 13, fill: 'var(--ink-2)' }}>
          full model
        </text>
        <text x={14} y={(T + S - B) / 2} textAnchor="middle" transform={`rotate(-90 14 ${(T + S - B) / 2})`} style={{ fontSize: 13, fill: 'var(--ink-2)' }}>
          with your changes
        </text>
        {r.items.map((it, i) => (
          <circle
            key={it.seed}
            cx={sx(xs[i])}
            cy={sy(ys[i])}
            r={4.5}
            style={{ fill: binary ? catVar(it.z > 0 ? 2 : 1) : 'var(--accent)', stroke: 'var(--surface)', opacity: 0.85 }}
            strokeWidth={1.2}
          >
            <title>
              seed #{it.seed}: {fmt(xs[i])} → {fmt(ys[i])}
            </title>
          </circle>
        ))}
      </svg>
    </ChartFrame>
  );
}
