/**
 * One-factor experiments on the live seed. Each starts from the course's live context and checkpoint
 * (full model) and changes ONE thing, so its effect is isolated. All numbers are computed in the
 * browser by the engine (sweeps.ts); nothing is precomputed.
 */
import { useEffect, useMemo, useState } from 'react';
import { ChartFrame, DataTable, LAYER_INFO, LineChart, Segmented, formatDate } from '@/components';
import type { AttnLayer } from '@/engine/types';
import { useElementSize } from '@/components/viz/useSize';
import { loadAssets } from '@/state/hooks';
import { ablateContext, ablatedForward, NO_TOKEN_ABLATION } from './ablate';
import { DeltaRows, JobProgress, useSweepBase } from './bits';
import { labelText, usePlayLive } from './context';
import { useJob, useNearView } from './jobs';
import { LABEL_SWEEP_MAX, runLabelSweep, runLayerSweep, runLSweep, runShuffleSweep, type Perturb } from './sweeps';
import { ckptOptions, fmtDelta, fmtDisplay, oid, readout, targetPhrase } from './usePlayground';

// ─────────────────────────────────────────────────────────────────────────────
// Labels, one at a time
// ─────────────────────────────────────────────────────────────────────────────

export function LabelEffects() {
  const { task, stats } = usePlayLive();
  const binary = task.targetType === 'binary';
  const [perturb, setPerturb] = useState<Perturb>(binary ? 'flip' : 'remove');
  const p: Perturb = binary ? (perturb === 'zero' ? 'flip' : perturb) : perturb === 'flip' ? 'remove' : perturb;
  const { base, key } = useSweepBase();
  const [ref, near] = useNearView<HTMLDivElement>();
  const job = useJob(key && near ? `labels|${key}|${p}` : null, (rep, cancelled) => runLabelSweep(base!, p, rep, cancelled), true);
  const r = job.result;

  const rows = useMemo(() => {
    if (!r) return [];
    const out = [];
    if (r.noSelf !== null) out.push({ key: 'noself', label: <strong>All self labels removed</strong>, output: r.noSelf, emphasis: true });
    if (r.noOther !== null) out.push({ key: 'noother', label: <strong>All other-entity labels removed</strong>, output: r.noOther, emphasis: true });
    for (const it of r.items) {
      const what = p === 'flip' ? `${labelText(it.raw)} → ${labelText(!(it.raw === true))}` : p === 'zero' ? `${labelText(it.raw)} → 0` : `${labelText(it.raw)} removed`;
      out.push({
        key: String(it.node),
        label: (
          <span className="inline-flex flex-wrap items-baseline gap-x-2">
            <span className="num">{it.ts !== null ? formatDate(it.ts * 1000) : '—'}</span>
            <span className="text-[14px] text-muted">{it.role}</span>
            <span className="num text-ink-2">{what}</span>
          </span>
        ),
        output: it.output,
      });
    }
    return out;
  }, [r, p]);

  return (
    <div ref={ref}>
      <ChartFrame
        title={`One label at a time: ${p === 'flip' ? 'flip it' : p === 'zero' ? 'set it to 0' : 'remove it'}`}
        subtitle={`Each row changes one in-context label of the live context (newest first, up to ${LABEL_SWEEP_MAX}) and reruns the model.`}
      >
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <Segmented<Perturb>
            label="How to perturb each label"
            value={p}
            onChange={setPerturb}
            options={
              binary
                ? [
                    { value: 'flip', label: 'Flip the label' },
                    { value: 'remove', label: 'Remove the label' },
                  ]
                : [
                    { value: 'remove', label: 'Remove the label' },
                    { value: 'zero', label: 'Set it to 0' },
                  ]
            }
          />
          <JobProgress job={job} label="forward passes" />
        </div>
        {r && r.items.length === 0 && <p className="text-[15px] text-muted">This context holds no in-context labels, so there is nothing to perturb.</p>}
        {r && r.items.length > 0 && <DeltaRows rows={rows} baseOutput={r.base} task={task} caption="Prediction change per perturbed label" />}
        {r && (
          <p className="!mb-0 mt-2 text-[14.5px] text-muted">
            Full context: {fmtDisplay(readout(r.base, task, stats).display, task)}. Red bars raise the prediction, blue bars lower it.
          </p>
        )}
      </ChartFrame>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Several name shuffles
// ─────────────────────────────────────────────────────────────────────────────

export function ShuffleSpread() {
  const { task, stats } = usePlayLive();
  const { base, key } = useSweepBase();
  const [ref, near] = useNearView<HTMLDivElement>();
  const job = useJob(key && near ? `shuffle|${key}` : null, (rep, c) => runShuffleSweep(base!, rep, c), true);
  const r = job.result;
  const phrase = targetPhrase(task);
  const series = useMemo(() => {
    if (!r) return null;
    const b = readout(r.base, task, stats).display;
    const d = r.draws.map((o) => readout(o, task, stats).display);
    return { b, d, lo: Math.min(b, ...d), hi: Math.max(b, ...d) };
  }, [r, task, stats]);

  return (
    <div ref={ref}>
      <ChartFrame
        title="Twelve different shuffles of the column names"
        subtitle={`${phrase.short} for the live seed under twelve random cyclic shuffles (dots), against the intact names (dashed line).`}
        table={
          series ? (
            <DataTable
              caption="Predictions under each shuffle"
              columns={['Variant', phrase.short, 'change']}
              rows={[['names intact', fmtDisplay(series.b, task), '—'], ...series.d.map((v, i) => [`shuffle #${i + 1}`, fmtDisplay(v, task), fmtDelta(v - series.b, task)])]}
            />
          ) : undefined
        }
      >
        <JobProgress job={job} label="forward passes" />
        {series && <DotStrip base={series.b} values={series.d} binary={task.targetType === 'binary'} fmt={(v) => fmtDisplay(v, task)} />}
        {series && (
          <p className="!mb-0 mt-2 text-[15px] text-ink-2">
            Mean change <span className="num font-semibold text-ink">{fmtDelta(series.d.reduce((a, v) => a + v, 0) / series.d.length - series.b, task)}</span>, range{' '}
            <span className="num">
              {fmtDelta(Math.min(...series.d) - series.b, task)} to {fmtDelta(Math.max(...series.d) - series.b, task)}
            </span>
            .
          </p>
        )}
      </ChartFrame>
    </div>
  );
}

/** Stacked dot plot of the shuffled predictions, with the intact-names prediction as a reference line. */
function DotStrip({ base, values, binary, fmt }: { base: number; values: number[]; binary: boolean; fmt: (v: number) => string }) {
  const tickFmt = (v: number) => (binary ? `${Math.round(v * 100)}%` : Number.isInteger(v) ? String(v) : v.toFixed(1));
  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(280, size.width || 600);
  const lo0 = binary ? 0 : Math.min(base, ...values);
  const hi0 = binary ? 1 : Math.max(base, ...values);
  const pad = binary ? 0 : (hi0 - lo0) * 0.12 || 1;
  const lo = lo0 - pad;
  const hi = hi0 + pad;
  const mx = 18;
  const sx = (v: number) => mx + ((v - lo) / (hi - lo || 1)) * (W - 2 * mx);
  const r = 6.5;
  const placed: { x: number; level: number; v: number; i: number }[] = [];
  values
    .map((v, i) => ({ v, i }))
    .sort((a, b) => a.v - b.v)
    .forEach(({ v, i }) => {
      const x = sx(v);
      let level = 0;
      while (placed.some((p) => p.level === level && Math.abs(p.x - x) < 2 * r + 1)) level++;
      placed.push({ x, level, v, i });
    });
  const levels = Math.max(1, ...placed.map((p) => p.level + 1));
  const top = 26;
  const axisY = top + levels * (2 * r + 2) + 6;
  const H = axisY + 26;
  const ticks = binary ? [0, 0.25, 0.5, 0.75, 1] : niceTicks(lo, hi, W < 420 ? 3 : 5);
  const bx = sx(base);
  const anchor = bx > W * 0.72 ? 'end' : bx < W * 0.28 ? 'start' : 'middle';
  return (
    <div ref={ref} className="mt-3 w-full">
      <svg width={W} height={H} role="img" aria-label={`Names intact ${fmt(base)}; shuffles from ${fmt(Math.min(...values))} to ${fmt(Math.max(...values))}`} className="block">
        {ticks.map((t) => (
          <line key={`g${t}`} x1={sx(t)} x2={sx(t)} y1={top} y2={axisY} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
        ))}
        <line x1={mx} x2={W - mx} y1={axisY} y2={axisY} style={{ stroke: 'var(--axis)' }} strokeWidth={1} />
        {ticks.map((t, i) => (
          <text
            key={`t${t}`}
            x={sx(t)}
            y={axisY + 19}
            textAnchor={i === 0 ? 'start' : i === ticks.length - 1 ? 'end' : 'middle'}
            className="num"
            style={{ fontSize: 13, fill: 'var(--muted)' }}
          >
            {tickFmt(t)}
          </text>
        ))}
        <line x1={bx} x2={bx} y1={top - 6} y2={axisY} style={{ stroke: 'var(--ink-2)' }} strokeWidth={2} strokeDasharray="4 3" />
        <text x={bx + (anchor === 'start' ? -2 : anchor === 'end' ? 2 : 0)} y={top - 10} textAnchor={anchor} style={{ fontSize: 13, fill: 'var(--ink-2)' }}>
          names intact {fmt(base)}
        </text>
        {placed.map((p) => (
          <circle key={p.i} cx={p.x} cy={axisY - r - 3 - p.level * (2 * r + 2)} r={r} style={{ fill: 'var(--accent)', stroke: 'var(--surface)' }} strokeWidth={1.5}>
            <title>
              shuffle #{p.i + 1}: {fmt(p.v)}
            </title>
          </circle>
        ))}
      </svg>
    </div>
  );
}

function niceTicks(lo: number, hi: number, n: number): number[] {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((x) => x >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Context length
// ─────────────────────────────────────────────────────────────────────────────

export function LSweepFigure() {
  const { task, stats, globalL } = usePlayLive();
  const { base, key } = useSweepBase();
  const [ref, near] = useNearView<HTMLDivElement>();
  const job = useJob(key && near ? `L|${key}` : null, (rep, c) => runLSweep(base!, rep, c), true);
  const r = job.result;
  const phrase = targetPhrase(task);
  const binary = task.targetType === 'binary';
  const forecast = task.kind === 'forecast';

  return (
    <div ref={ref}>
      <JobProgress job={job} label="context lengths" />
      {r && (
        <>
          <LineChart
            title={`${phrase.short} of the live seed as the context grows`}
            subtitle="Same seed row and sampler; only L changes. A longer context keeps every row of a shorter one and adds more."
            series={[{ id: 'pred', label: phrase.short, slot: 7, points: r.map((p) => ({ x: p.L, y: readout(p.output, task, stats).display })) }]}
            xLabel="context length L (cells)"
            yLabel={binary ? 'P (%)' : phrase.unit || phrase.short}
            formatX={(v) => String(Math.round(v))}
            formatY={(v) => (binary ? `${(v * 100).toFixed(0)}%` : v.toFixed(1))}
            yDomain={binary ? [0, 1] : undefined}
            markers={[{ x: globalL, label: `course L = ${globalL}` }]}
            height={260}
          />
          <div className="scroll-quiet mt-3 overflow-x-auto">
            <table className="w-full min-w-[520px] border-collapse text-[14.5px]">
              <caption className="sr-only">Prediction and in-context labels per context length</caption>
              <thead>
                <tr className="border-b border-grid text-ink-2">
                  <th scope="row" className="py-1 pr-2 text-left font-semibold">
                    L
                  </th>
                  {r.map((p) => (
                    <th key={p.L} scope="col" className={`num px-1.5 py-1 text-right font-semibold ${p.L === globalL ? 'text-ink' : ''}`}>
                      {p.L}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-grid/60">
                  <th scope="row" className="py-1 pr-2 text-left font-normal text-ink-2">
                    {phrase.short}
                  </th>
                  {r.map((p) => (
                    <td key={p.L} className="num px-1.5 py-1 text-right text-ink">
                      {fmtDisplay(readout(p.output, task, stats).display, task)}
                    </td>
                  ))}
                </tr>
                <tr className="border-b border-grid/60">
                  <th scope="row" className="py-1 pr-2 text-left font-normal text-ink-2">
                    {forecast ? 'self labels' : 'labels'}
                  </th>
                  {r.map((p) => (
                    <td key={p.L} className="num px-1.5 py-1 text-right text-ink-2">
                      {forecast ? p.self : p.other}
                    </td>
                  ))}
                </tr>
                {forecast && (
                  <tr>
                    <th scope="row" className="py-1 pr-2 text-left font-normal text-ink-2">
                      other labels
                    </th>
                    {r.map((p) => (
                      <td key={p.L} className="num px-1.5 py-1 text-right text-ink-2">
                        {p.other}
                      </td>
                    ))}
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Attention layers off
// ─────────────────────────────────────────────────────────────────────────────

const SET_LABEL = (off: AttnLayer[]) =>
  off.length === 4 ? 'All four attention layers off' : `${LAYER_INFO[off[0]].name} attention off`;

export function LayerEffects() {
  const { task } = usePlayLive();
  const { base, key } = useSweepBase();
  const [ref, near] = useNearView<HTMLDivElement>();
  const job = useJob(key && near ? `layers|${key}` : null, (rep, c) => runLayerSweep(base!, rep, c), true);
  const r = job.result;
  return (
    <div ref={ref}>
      <ChartFrame title="Skip one trained layer (in every block)" subtitle="The live seed, full context; each row skips one attention sublayer at inference.">
        <JobProgress job={job} label="forward passes" />
        {r && (
          <DeltaRows
            task={task}
            baseOutput={r.base}
            caption="Prediction change per switched-off layer"
            rows={r.items.map((it) => ({ key: it.off.join(','), label: SET_LABEL(it.off), output: it.output, emphasis: it.off.length === 4 }))}
          />
        )}
      </ChartFrame>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoints
// ─────────────────────────────────────────────────────────────────────────────

export function CkptCompare() {
  const live = usePlayLive();
  const { task, ctx, pdb, stats } = live;
  const opts = ckptOptions(task);
  const [preds, setPreds] = useState<Record<string, number>>({});
  const [ref, near] = useNearView<HTMLDivElement>();
  const ctxKey = `${oid(ctx)}|${oid(pdb)}`;

  useEffect(() => {
    if (!near) return;
    let alive = true;
    setPreds({});
    (async () => {
      for (const o of opts) {
        const a = await loadAssets(o.name);
        if (!alive) return;
        if (a.status !== 'ready' || !a.store || !a.weights) continue;
        await new Promise((res) => window.setTimeout(res, 0));
        if (!alive) return;
        // capture off and outside the shared forward cache: three checkpoints must not evict the live pass
        const fr = ablatedForward(ablateContext(pdb, task, ctx, NO_TOKEN_ABLATION), pdb, a.store, a.weights, [], false);
        if (!alive) return;
        setPreds((p) => ({ ...p, [o.name]: fr.prediction }));
      }
    })();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [near, ctxKey, task.id]);

  const zero = opts.find((o) => o.id === 'zero')!;
  const baseOut = preds[zero.name];
  return (
    <div ref={ref}>
      <ChartFrame title="The same context under three checkpoints" subtitle="Live seed and context; only the weights change.">
        {baseOut === undefined ? (
          <p className="text-[15px] text-muted">Loading the checkpoints…</p>
        ) : (
          <DeltaRows
            task={task}
            baseOutput={baseOut}
            caption="Prediction per checkpoint"
            rows={opts
              .filter((o) => preds[o.name] !== undefined)
              .map((o) => ({ key: o.name, label: `${o.label} (${o.name})`, output: preds[o.name], emphasis: o.id === 'zero' }))}
          />
        )}
        {baseOut !== undefined && (
          <p className="!mb-0 mt-2 text-[14.5px] text-muted">
            Bars compare with the zero-shot checkpoint ({fmtDisplay(readout(baseOut, task, stats).display, task)}).
          </p>
        )}
      </ChartFrame>
    </div>
  );
}
