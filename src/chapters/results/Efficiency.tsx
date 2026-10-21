/**
 * Learning efficiency: Fig. 3 (p.7) described on its own log-scaled x axis, with only the numbers the
 * paper prints or states, and Table 10 (p.21), supervised fine-tuning, as a per-task explorer.
 */
import { useMemo, useState } from 'react';
import { Cite, Figure, Segmented } from '@/components';
import { textWidth } from '@/components/viz/text';
import { useElementSize } from '@/components/viz/useSize';
import { FIG3_POINTS, TABLE10, taskLabel, type Cell } from './data';
import { DotPlot, type DotRow } from './DotPlot';
import { f1, isNum, MethodLegend, PaperTableView, TitleWithBadge, ChartTableFrame } from './kit';
import { METHODS, T10_METHOD } from './methods';

// ─────────────────────────────────────────────────────────────────────────────
// Fig. 3 in words
// ─────────────────────────────────────────────────────────────────────────────

interface Event {
  at: number; // steps + 1
  label: string;
  level: 0 | 1;
  anchor: 'start' | 'end';
}

const LANES = [
  {
    id: 'clf' as const,
    title: 'Binary classification',
    sub: 'mean over 5 tasks',
    pts: FIG3_POINTS.clf,
    unit: 'AUROC',
    events: [
      { at: 1, label: 'step 0: the zero-shot point', level: 0, anchor: 'start' },
      { at: 3, label: 'after 2 steps, untrained RT overtakes RDL-GNN', level: 1, anchor: 'start' },
      { at: 32769, label: '32k steps ≈ 8M examples', level: 0, anchor: 'end' },
    ] as Event[],
  },
  {
    id: 'reg' as const,
    title: 'Regression',
    sub: 'mean over 4 tasks',
    pts: FIG3_POINTS.reg,
    unit: 'R²',
    events: [
      { at: 1, label: 'step 0: the zero-shot point', level: 0, anchor: 'start' },
      { at: 101, label: 'after ≈100 steps, untrained RT overtakes RDL-GNN', level: 1, anchor: 'start' },
      { at: 32769, label: '32k steps ≈ 8M examples', level: 0, anchor: 'end' },
    ] as Event[],
  },
];

function LogStrip({ events }: { events: Event[] }) {
  const [box, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(260, size.width || 600);
  const narrow = W < 560;
  const L = 10;
  const R = W - 10;
  const lx = (s: number) => L + (Math.log10(s) / Math.log10(32769)) * (R - L);
  const levelY = (lv: number) => (lv === 0 ? 14 : 36);

  // label layout; if any label would be crossed by another event's pin or overlap a label, fall back to numbered pins
  const laid = events.map((e) => {
    const ex = lx(e.at);
    const text = e.label;
    const tw = textWidth(text, 14.5);
    const anchor: 'start' | 'end' = e.anchor === 'start' && ex - 2 + tw > R ? 'end' : e.anchor;
    const tx = anchor === 'start' ? ex - 2 : Math.max(L + tw, ex + 2);
    const x0 = anchor === 'start' ? tx : tx - tw;
    return { e, ex, text, tw, anchor, tx, x0, x1: x0 + tw, ty: levelY(e.level) };
  });
  const conflict = laid.some((a) =>
    laid.some(
      (b) =>
        a !== b &&
        // b's pin runs from b's label down to the axis: it crosses a's label if a sits lower and b's pin is inside a's span
        ((b.ty < a.ty && b.ex > a.x0 - 4 && b.ex < a.x1 + 4) || (b.ty === a.ty && !(b.x1 + 8 < a.x0 || b.x0 > a.x1 + 8))),
    ),
  );
  const numbered = narrow || conflict;
  const axisY = numbered ? 22 : 62;

  return (
    <div ref={box} className="w-full">
      <svg width={W} height={axisY + 30} role="img" aria-label={events.map((e) => e.label).join('; ')} className="block">
        <line x1={L} x2={R} y1={axisY} y2={axisY} style={{ stroke: 'var(--axis)' }} strokeWidth={1.5} />
        {[1, 10, 100, 1000, 10000].map((t, k) => (
          <g key={t} transform={`translate(${lx(t)},${axisY})`}>
            <line y2={6} style={{ stroke: 'var(--axis)' }} />
            <text y={22} textAnchor={k === 0 ? 'start' : 'middle'} style={{ fill: 'var(--muted)', fontSize: 13.5 }}>
              10
              <tspan dy="-6" style={{ fontSize: 12 }}>
                {k}
              </tspan>
            </text>
          </g>
        ))}
        {laid.map((l, i) =>
          numbered ? (
            <g key={`pin${l.e.label}`} transform={`translate(${Math.min(R - 9, Math.max(L + 9, l.ex))},${axisY - 12})`}>
              <line y1={9} y2={12} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.25} />
              <circle r={9} style={{ fill: 'var(--ink)' }} />
              <text dy="0.35em" textAnchor="middle" style={{ fill: 'var(--surface)', fontSize: 13, fontWeight: 700 }}>
                {i + 1}
              </text>
            </g>
          ) : (
            <g key={`pin${l.e.label}`}>
              <line x1={l.ex} x2={l.ex} y1={l.ty + 5} y2={axisY} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.25} />
              <circle cx={l.ex} cy={axisY} r={4.5} style={{ fill: 'var(--ink)', stroke: 'var(--surface-2)' }} strokeWidth={2} />
            </g>
          ),
        )}
        {!numbered &&
          laid.map((l) => (
            <text
              key={`lab${l.e.label}`}
              x={l.tx}
              y={l.ty}
              textAnchor={l.anchor}
              style={{
                fill: 'var(--ink)',
                fontSize: 14.5,
                stroke: 'var(--surface-2)',
                strokeWidth: 5,
                paintOrder: 'stroke',
                strokeLinejoin: 'round',
              }}
            >
              {l.text}
            </text>
          ))}
      </svg>
      <div className="text-center text-[14.5px] text-ink-2">fine-tuning steps + 1 (log scale)</div>
      {numbered && (
        <ol className="!mt-2 !mb-0 grid !list-none gap-1 !p-0 text-[15px] text-ink">
          {events.map((e, i) => (
            <li key={e.label} className="!m-0 flex items-baseline gap-2 !p-0">
              <span className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-ink text-[13px] font-bold text-surface">
                {i + 1}
              </span>
              {e.label}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export function Fig3InWords() {
  return (
    <Figure
      wide
      title={<TitleWithBadge title="Figure 3, in words: fine-tuning curves on unseen tasks" kind="paper" badge="Fig. 3 · p.7" />}
      caption={
        <>
          The paper plots test AUROC and R² against fine-tuning steps + 1 on a log axis, up to 32k steps at batch size 256, averaged over the tasks
          that do not overfit <Cite kind="paper" ref="p.7 Fig. 3" />. We do not redraw the curves: each axis carries only the numbers the paper prints
          or states. Per-task curves are in Figs. 7 and 8 <Cite kind="paper" ref="p.20-21 App. F" />.
        </>
      }
    >
      <div className="grid gap-3 lg:grid-cols-2">
        {LANES.map((ln) => (
          <div key={ln.id} className="min-w-0 rounded-xl bg-surface-2 px-4 py-3">
            <div className="text-[16px] font-semibold text-ink">
              {ln.title} <span className="font-normal text-muted">· {ln.sub}</span>
            </div>
            <div className="mt-2 text-[15px] text-ink-2">Zero-shot start, as printed on the plot (test {ln.unit}, %):</div>
            <div className="mt-1 flex flex-wrap gap-x-5 gap-y-1 text-[16px] text-ink">
              <span className="inline-flex items-center gap-1.5">
                <span aria-hidden="true" className="inline-block h-3 w-3 rounded-full" style={{ background: 'var(--cat-7)' }} />
                pretrained RT <strong className="num">{f1(ln.pts.rtPretrained)}</strong>
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span aria-hidden="true" className="inline-block h-3 w-3 rounded-full" style={{ background: 'var(--cat-2)' }} />
                pretrained Griffin <strong className="num">{f1(ln.pts.griffinPretrained)}</strong>
              </span>
            </div>
            <div className="mt-2">
              <LogStrip events={ln.events} />
            </div>
          </div>
        ))}
      </div>
      <ul className="mt-4 grid gap-x-6 gap-y-1.5 text-[16px] leading-snug text-ink md:grid-cols-2">
        <li className="flex gap-2">
          <span aria-hidden="true" className="mt-[0.55em] h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
          <span>
            “Pretrained RT is best overall, and untrained RT catches up towards the end” <Cite kind="paper" ref="p.7 Fig. 3" />.
          </span>
        </li>
        <li className="flex gap-2">
          <span aria-hidden="true" className="mt-[0.55em] h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
          <span>
            Pretrained RT reaches the second-best baseline’s performance in 10–100× fewer steps <Cite kind="paper" ref="p.3 §1" />.
          </span>
        </li>
        <li className="flex gap-2">
          <span aria-hidden="true" className="mt-[0.55em] h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
          <span>
            Fine-tuning from the pretrained checkpoint shows “a small dip in performance in the very early steps”, which the authors conjecture is a
            transition from a zero-shot, in-context-labels- and semantics-driven regime to a data-driven one <Cite kind="paper" ref="p.7 §4.2" />.
          </span>
        </li>
        <li className="flex gap-2">
          <span aria-hidden="true" className="mt-[0.55em] h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
          <span>
            Griffin also benefits from zero-shot transfer, but RT consistently outperforms it <Cite kind="paper" ref="p.7 §4.2" />.
          </span>
        </li>
      </ul>
    </Figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Table 10 explorer
// ─────────────────────────────────────────────────────────────────────────────

type Metric = 'clf' | 'reg';

const PRESETS = [
  { id: 'schema', label: 'RT vs schema-specific', cols: ['rdlGnn', 'relGnn', 'relGt', 'rtYes'] },
  { id: 'init', label: 'Pretrained vs untrained', cols: ['griffinNo', 'griffinYes', 'rtNo', 'rtYes'] },
  { id: 'all', label: 'All eight', cols: ['rdlGnn', 'relGnn', 'relGt', 'relllm', 'griffinNo', 'griffinYes', 'rtNo', 'rtYes'] },
];

export function SupervisedExplorer() {
  const [metric, setMetric] = useState<Metric>('clf');
  const [presetId, setPresetId] = useState('init');
  const t = TABLE10[metric];
  const preset = PRESETS.find((p) => p.id === presetId)!;
  const methods = preset.cols.map((c) => T10_METHOD[c]);

  const rows: DotRow[] = useMemo(
    () => [
      ...t.rows.map((r, i) => ({
        id: r.task,
        label: taskLabel(r.task),
        detail: `train set ${t.size[i]}`,
        values: Object.fromEntries(t.cols.map((c, j) => [T10_METHOD[c], r.v[j]])) as Record<string, Cell>,
      })),
      {
        id: 'mean',
        label: 'Mean (as printed)',
        mean: true,
        values: Object.fromEntries(t.cols.map((c, j) => [T10_METHOD[c], t.mean[j]])) as Record<string, Cell>,
      },
    ],
    [t],
  );

  // who has the best printed mean (computed)
  const best = t.cols
    .map((c, j) => ({ c, v: t.mean[j] }))
    .filter((x) => isNum(x.v))
    .sort((a, b) => (b.v as number) - (a.v as number));

  const table = (
    <PaperTableView
      headCols={['Dataset', 'Task', 'Train set']}
      columns={t.cols.map((c) => ({
        id: c,
        label: METHODS[T10_METHOD[c]].label,
        group: ['rdlGnn', 'relGnn', 'relGt'].includes(c) ? 'Cannot be pretrained' : 'Can be pretrained',
        method: T10_METHOD[c],
      }))}
      rows={[
        ...t.rows.map((r, i) => {
          const [db, task] = taskLabel(r.task).split(' · ');
          return { id: r.task, head: [db, task, t.size[i]], cells: r.v };
        }),
        { id: 'mean', head: ['Mean', '', ''], cells: t.mean, mean: true },
      ]}
      markMax
      caption={<>{t.caption} Rows in the paper’s order (sorted by training-set size). Underline: best in row, computed here.</>}
    />
  );

  return (
    <Figure
      wide
      title={<TitleWithBadge title="Supervised fine-tuning on the full training set" kind="paper" badge="Table 10 · p.21" />}
      controls={
        <>
          <Segmented
            label="Metric"
            value={metric}
            onChange={setMetric}
            options={[
              { value: 'clf', label: 'AUROC · 10 tasks' },
              { value: 'reg', label: 'R² · 8 tasks' },
            ]}
          />
          <Segmented label="Compare" value={presetId} onChange={setPresetId} options={PRESETS.map((p) => ({ value: p.id, label: p.label }))} />
        </>
      }
      caption={
        <>
          “Models are trained on the full training set until convergence, with checkpoint selection based on validation performance”{' '}
          <Cite kind="paper" ref="p.21 Table 10" />. Hollow markers: untrained, or a method that cannot be pretrained. Highest printed mean:{' '}
          {METHODS[T10_METHOD[best[0].c]].label} ({f1(best[0].v as number)}), then {METHODS[T10_METHOD[best[1].c]].label} ({f1(best[1].v as number)}).
          {metric === 'reg' && ' RelLLM prints “nan” for every regression task.'} Hover a row for the training-set size.
        </>
      }
    >
      <ChartTableFrame title={<MethodLegend ids={methods} />} table={table}>
        <DotPlot
          rows={rows}
          methods={methods}
          domain={metric === 'clf' ? [55, 95] : [-10, 55]}
          refLine={metric === 'reg' ? { value: 0, label: 'mean = 0' } : undefined}
          axisLabel={metric === 'clf' ? 'test AUROC (%)' : 'test R² (%)'}
          focus="rtP"
          ariaLabel={`Supervised fine-tuning ${metric === 'clf' ? 'AUROC' : 'R²'} per task, paper Table 10`}
        />
      </ChartTableFrame>
    </Figure>
  );
}
