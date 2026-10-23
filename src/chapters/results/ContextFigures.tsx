/**
 * Figures that tie the context ablations to the context itself:
 *  - SelfLabelScatter: Table 12 (self labels per 1024-cell context) against Table 11 (zero-shot change
 *    when self labels are removed). Both tables date from the paper's first version (spec §11.3, §13 #1).
 *  - LiveLabelStats: the same statistic for this site's live task, computed with our sampler over every
 *    seed of the current split (sampler only, no forward pass), plus the live seed's own counts.
 *  - MiniLabelAblation: this site's miniature, all in-context labels removed at inference time
 *    (public/data/training/results.json → diagnostics.inContextLabelAblation).
 */
import { useEffect, useMemo, useState } from 'react';
import { BarChart, Cite, cx, DataTable, Figure, ScaleNote, Segmented } from '@/components';
import { textWidth } from '@/components/viz/text';
import { useElementSize } from '@/components/viz/useSize';
import { inContextLabels } from '@/engine/pipeline';
import { sampleContext } from '@/engine/sampler';
import type { Split } from '@/engine/types';
import { getSeeds, makeSampleParams, useInContextLabels, usePreprocessed, useSeedNode, useTask } from '@/state/hooks';
import { useSim } from '@/state/store';
import { catVar } from '@/styles/palette';
import { TABLE11, TABLE12, TABLE4, TASKS, taskLabel, type TaskKey } from './data';
import { d1, f1, mean, sd, Tip, TitleWithBadge, xScale, ChartTableFrame } from './kit';
import { useMiniResults } from './useMiniResults';

// ─────────────────────────────────────────────────────────────────────────────
// Table 12 × Table 11
// ─────────────────────────────────────────────────────────────────────────────

type Metric = 'clf' | 'reg';

interface Pt {
  key: TaskKey;
  label: string;
  self: number;
  selfSd: number;
  other: number;
  none: number;
  removed: number;
  d: number;
}

function points(metric: Metric): Pt[] {
  const part = TABLE11[metric]!;
  const out: Pt[] = [];
  for (const r of TABLE12.filter((x) => x.kind === metric)) {
    const key = `${r.db.replace(/^rel-/, '')}/${r.task}` as TaskKey;
    const row = part.rows.find((x) => x.task === key);
    if (!row || !TASKS[key]) continue;
    out.push({
      key,
      label: taskLabel(key),
      self: r.self[0],
      selfSd: r.self[1],
      other: r.other[0],
      none: row.zs[0],
      removed: row.zs[2],
      d: row.zs[2] - row.zs[0],
    });
  }
  return out;
}

export function SelfLabelScatter() {
  const [metric, setMetric] = useState<Metric>('clf');
  const [box, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<TaskKey | null>(null);
  const pts = useMemo(() => points(metric), [metric]);
  const W = Math.max(300, size.width || 700);
  const H = W < 520 ? 380 : 420;
  const M = { l: 58, r: 18, t: 18, b: 52 };
  const xmax = Math.max(20, Math.ceil(Math.max(...pts.map((p) => p.self)) / 5) * 5);
  const ymin = Math.floor(Math.min(...pts.map((p) => p.d)) / 10) * 10;
  const { x, ticks: xt } = xScale([0, xmax], M.l, W - M.r, 60);
  const ymax = 5; // headroom above 0, so the zero line is not the plot's edge
  const yS = (v: number) => M.t + ((ymax - v) / (ymax - ymin)) * (H - M.t - M.b);
  const yt: number[] = [];
  const step = Math.abs(ymin) > 50 ? 20 : 10;
  for (let v = 0; v >= ymin; v -= step) yt.push(v);

  // greedy label placement: right, left, above, below; skip if all collide
  const placed = useMemo(() => {
    const boxes: { x0: number; x1: number; y0: number; y1: number }[] = pts.map((p) => ({
      x0: x(p.self) - 8,
      x1: x(p.self) + 8,
      y0: yS(p.d) - 8,
      y1: yS(p.d) + 8,
    }));
    // the zero line is an obstacle too: labels must not sit on it
    boxes.push({ x0: M.l, x1: W - M.r, y0: yS(0) - 1, y1: yS(0) + 1 });
    const out = new Map<TaskKey, { x: number; y: number; anchor: 'start' | 'end' | 'middle' }>();
    const order = [...pts].sort((a, b) => Math.abs(b.d) - Math.abs(a.d));
    for (const p of order) {
      const w = textWidth(p.label, 13.5);
      const px = x(p.self);
      const py = yS(p.d);
      const cands = [
        { x: px + 11, y: py, anchor: 'start' as const, b: { x0: px + 10, x1: px + 12 + w, y0: py - 9, y1: py + 9 } },
        { x: px - 11, y: py, anchor: 'end' as const, b: { x0: px - 12 - w, x1: px - 10, y0: py - 9, y1: py + 9 } },
        { x: px, y: py - 16, anchor: 'middle' as const, b: { x0: px - w / 2, x1: px + w / 2, y0: py - 26, y1: py - 10 } },
        { x: px, y: py + 18, anchor: 'middle' as const, b: { x0: px - w / 2, x1: px + w / 2, y0: py + 10, y1: py + 27 } },
        { x: px + 4, y: py - 16, anchor: 'start' as const, b: { x0: px + 3, x1: px + 5 + w, y0: py - 26, y1: py - 10 } },
        { x: px + 4, y: py + 18, anchor: 'start' as const, b: { x0: px + 3, x1: px + 5 + w, y0: py + 10, y1: py + 27 } },
      ];
      for (const c of cands) {
        const inside = c.b.x0 >= M.l - 4 && c.b.x1 <= W - 2 && c.b.y0 >= 0 && c.b.y1 <= H - M.b + 4;
        const hit = boxes.some((o) => !(c.b.x1 < o.x0 || c.b.x0 > o.x1 || c.b.y1 < o.y0 || c.b.y0 > o.y1));
        if (inside && !hit) {
          out.set(p.key, { x: c.x, y: c.y, anchor: c.anchor });
          boxes.push(c.b);
          break;
        }
      }
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pts, W, H]);

  const hp = pts.find((p) => p.key === hover);
  const unit = metric === 'clf' ? 'AUROC' : 'R²';
  return (
    <Figure
      wide
      title={<TitleWithBadge title="Where self labels exist, removing them hurts" kind="paper" badge="Tables 12 and 11 · p.23, p.22" />}
      controls={
        <Segmented
          label="Metric"
          value={metric}
          onChange={setMetric}
          options={[
            { value: 'clf', label: 'AUROC · 10 tasks' },
            { value: 'reg', label: 'R² · 8 tasks' },
          ]}
        />
      }
      caption={
        <>
          Each point is one task. Horizontal: how many past labels of the target entity the sampler put into a 1024-cell context, on average (the
          standard deviations are large; see the tooltip or the table) <Cite kind="paper" ref="p.23 Table 12" />. Vertical: the change in zero-shot{' '}
          {unit} when those self labels are removed <Cite kind="paper" ref="p.22 Table 11" />. Both tables date from the paper’s first version{' '}
          <Cite kind="spec" ref="§13 #1" />. The two tasks whose contexts hold no self labels at all, rel-trial study-outcome and study-adverse, do
          not change when they are removed.
        </>
      }
    >
      <ChartTableFrame
        table={
          <DataTable
            caption="Table 12 self labels and Table 11 zero-shot change"
            columns={['Task', 'self labels (mean ± sd)', 'other labels (mean)', `${unit} none`, `${unit} self removed`, 'change']}
            rows={pts.map((p) => [p.label, `${p.self} ± ${p.selfSd}`, String(p.other), f1(p.none), f1(p.removed), d1(p.d)])}
          />
        }
      >
        <div ref={box} className="relative w-full" onPointerLeave={() => setHover(null)}>
          <svg
            width={W}
            height={H}
            role="img"
            aria-label={`Self labels per context vs change in zero-shot ${unit} when they are removed`}
            className="block"
          >
            {yt.map((t) => (
              <g key={t}>
                <line
                  x1={M.l}
                  x2={W - M.r}
                  y1={yS(t)}
                  y2={yS(t)}
                  style={{ stroke: t === 0 ? 'var(--ink-2)' : 'var(--grid)' }}
                  strokeWidth={t === 0 ? 1.25 : 1}
                />
                <text x={M.l - 8} y={yS(t)} dy="0.32em" textAnchor="end" className="tabular" style={{ fill: 'var(--muted)', fontSize: 13.5 }}>
                  {t === 0 ? '0' : String(t).replace('-', '−')}
                </text>
              </g>
            ))}
            {xt.map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={M.t} y2={H - M.b} style={{ stroke: 'var(--grid)' }} />
                <text x={x(t)} y={H - M.b + 20} textAnchor="middle" className="tabular" style={{ fill: 'var(--muted)', fontSize: 13.5 }}>
                  {t}
                </text>
              </g>
            ))}
            <text x={(M.l + W - M.r) / 2} y={H - 8} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: 14.5 }}>
              {W < 520 ? 'self labels per context (Table 12)' : 'self labels per 1024-cell context (Table 12)'}
            </text>
            <text transform={`translate(15,${(M.t + H - M.b) / 2}) rotate(-90)`} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: 14.5 }}>
              change when removed (points)
            </text>
            {pts.map((p) => {
              const cx = x(p.self);
              const cy = yS(p.d);
              const lab = placed.get(p.key);
              return (
                <g
                  key={p.key}
                  tabIndex={0}
                  role="listitem"
                  aria-label={`${p.label}: ${p.self} self labels, change ${d1(p.d)}`}
                  onPointerEnter={() => setHover(p.key)}
                  onFocus={() => setHover(p.key)}
                  onBlur={() => setHover(null)}
                  className="outline-none"
                >
                  <circle cx={cx} cy={cy} r={14} style={{ fill: 'transparent' }} />
                  <circle cx={cx} cy={cy} r={hover === p.key ? 8 : 6.5} style={{ fill: catVar(1), stroke: 'var(--surface)' }} strokeWidth={2} />
                </g>
              );
            })}
            {pts.map((p) => {
              const lab = placed.get(p.key);
              if (!lab) return null;
              return (
                <text
                  key={`l${p.key}`}
                  x={lab.x}
                  y={lab.y}
                  dy="0.32em"
                  textAnchor={lab.anchor}
                  pointerEvents="none"
                  style={{
                    fill: 'var(--ink-2)',
                    fontSize: 13.5,
                    stroke: 'var(--surface)',
                    strokeWidth: 4,
                    paintOrder: 'stroke',
                    strokeLinejoin: 'round',
                  }}
                >
                  {p.label}
                </text>
              );
            })}
          </svg>
          {hp && (
            <Tip x={x(hp.self)} y={yS(hp.d) + 14} width={W} flip={x(hp.self) > W * 0.6}>
              <div className="mb-1 text-[14.5px] text-ink">{hp.label}</div>
              <div className="text-ink-2">
                self labels:{' '}
                <span className="num text-ink">
                  {hp.self} ± {hp.selfSd}
                </span>
                , other: <span className="num text-ink">{hp.other}</span>
              </div>
              <div className="text-ink-2">
                zero-shot {unit}: <span className="num text-ink">{f1(hp.none)}</span> →{' '}
                <span className="num font-semibold text-ink">{f1(hp.removed)}</span> (<span className="num">{d1(hp.d)}</span>)
              </div>
            </Tip>
          )}
        </div>
      </ChartTableFrame>
    </Figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Live: in-context label counts over every seed of the current split
// ─────────────────────────────────────────────────────────────────────────────

interface Counts {
  self: number[];
  other: number[];
}

const countsCache = new Map<string, Counts>();

export function LiveLabelStats() {
  const pdb = usePreprocessed();
  const task = useTask();
  const split = useSim((s) => s.split) as Split;
  const L = useSim((s) => s.L);
  const w = useSim((s) => s.w);
  const rngSeed = useSim((s) => s.rngSeed);
  const seedNode = useSeedNode();
  const live = useInContextLabels();
  const seeds = useMemo(() => getSeeds(pdb, task, split), [pdb, task, split]);
  const key = `${pdb.dbName}|${task.id}|${task.kind === 'autocomplete' ? '*' : split}|${L}|${w}|${rngSeed}`;
  const [counts, setCounts] = useState<Counts | null>(() => countsCache.get(key) ?? null);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const hit = countsCache.get(key);
    if (hit) {
      setCounts(hit);
      setProgress(seeds.length);
      return;
    }
    setCounts(null);
    setProgress(0);
    let i = 0;
    let timer = 0;
    const acc: Counts = { self: [], other: [] };
    const CHUNK = 6;
    const step = () => {
      const end = Math.min(seeds.length, i + CHUNK);
      for (; i < end; i++) {
        const ctx = sampleContext(pdb, makeSampleParams(pdb, task, seeds[i], L, w, rngSeed), { recordEvents: false });
        const lab = inContextLabels(pdb, ctx, task);
        acc.self.push(lab.self.length);
        acc.other.push(lab.other.length);
      }
      setProgress(i);
      if (i < seeds.length) timer = window.setTimeout(step, 0);
      else {
        countsCache.set(key, acc);
        setCounts(acc);
      }
    };
    timer = window.setTimeout(step, 30);
    return () => window.clearTimeout(timer);
  }, [key, pdb, task, seeds, L, w, rngSeed]);

  const [histBox, histSize] = useElementSize<HTMLDivElement>();
  const seedPos = seeds.indexOf(seedNode);
  const isForecast = task.kind === 'forecast';
  const hist = useMemo(() => {
    if (!counts) return null;
    const src = isForecast ? counts.self : counts.other;
    const max = Math.max(0, ...src);
    const bins = Array.from({ length: max + 1 }, (_, k) => src.filter((v) => v === k).length);
    return { cats: bins.map((_, k) => String(k)), bins };
  }, [counts, isForecast]);

  return (
    <Figure
      wide
      title={<TitleWithBadge title={`The same count on this site: ${task.id}`} kind="site" badge="Live · our sampler" />}
      caption={
        <>
          Computed in your browser with this site’s sampler (a mirror of <code>fly.rs</code>) for every {isForecast ? `${split}` : ''} seed of{' '}
          <code>{task.id}</code>, with the global settings L = {L}, w = {w}, RNG seed {rngSeed}. Self labels are visible cells of the target column
          whose task row points to the seed’s entity; every other visible label counts as other <Cite kind="spec" ref="§2.5" />. Paper Table 12 was
          sampled before the code kept same-horizon task rows, so this released-code sampler can count more other-entity labels{' '}
          <Cite kind="spec" ref="§11.3" />.
        </>
      }
    >
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="grid content-start gap-3 sm:grid-cols-2 lg:grid-cols-1">
          <div className="rounded-xl border border-border bg-surface px-4 py-3">
            <div className="text-[15.5px] font-medium text-ink-2">
              {isForecast ? 'Target-entity (self) labels per context' : 'Visible target cells per context'}
            </div>
            <div className="mt-1 text-[30px] leading-tight font-semibold text-ink">
              {counts ? `${f1(mean(isForecast ? counts.self : counts.other))}` : '…'}
              <span className="ml-1.5 text-[18px] font-medium text-ink-2">
                {counts ? `± ${f1(sd(isForecast ? counts.self : counts.other))}` : ''}
              </span>
            </div>
            <div className="text-[15px] text-muted">mean ± sd over {seeds.length} seeds</div>
          </div>
          {isForecast && (
            <div className="rounded-xl border border-border bg-surface px-4 py-3">
              <div className="text-[15.5px] font-medium text-ink-2">Other-entity labels per context</div>
              <div className="mt-1 text-[30px] leading-tight font-semibold text-ink">
                {counts ? f1(mean(counts.other)) : '…'}
                <span className="ml-1.5 text-[18px] font-medium text-ink-2">{counts ? `± ${f1(sd(counts.other))}` : ''}</span>
              </div>
              <div className="text-[15px] text-muted">mean ± sd over {seeds.length} seeds</div>
            </div>
          )}
          <div className={cx('rounded-xl border border-accent/40 bg-accent-soft px-4 py-3', !isForecast && 'sm:col-span-1')}>
            <div className="text-[15.5px] font-medium text-ink-2">The live seed{seedPos >= 0 ? `, one of the ${seeds.length}` : ''}</div>
            <div className="mt-1 text-[17px] text-ink">
              {isForecast ? (
                <>
                  <strong className="num">{live.self.length}</strong> self labels, <strong className="num">{live.other.length}</strong> other-entity
                  labels
                </>
              ) : (
                <>
                  <strong className="num">{live.other.length}</strong> visible target cells
                </>
              )}
            </div>
            <div className="text-[15px] text-muted">the context every chapter shows</div>
          </div>
        </div>
        <div ref={histBox} className="min-w-0">
          {hist ? (
            <BarChart
              title={isForecast ? 'Seeds by number of self labels in their context' : 'Seeds by number of visible target cells'}
              subtitle={`${seeds.length} ${isForecast ? split : ''} seeds of ${task.name}`}
              categories={hist.cats}
              series={[{ id: 'n', label: 'seeds', slot: 1, values: hist.bins }]}
              orientation={(histSize.width || 600) < 480 ? 'horizontal' : 'vertical'}
              height={190}
              format={(v) => String(Math.round(v))}
              valueLabel="seeds"
            />
          ) : (
            <div className="flex h-[240px] flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border-strong text-[15.5px] text-ink-2">
              Sampling contexts… {progress} / {seeds.length}
              <div className="h-1.5 w-48 overflow-hidden rounded-full bg-surface-3" aria-hidden="true">
                <div className="h-full bg-accent transition-[width]" style={{ width: `${(100 * progress) / Math.max(1, seeds.length)}%` }} />
              </div>
            </div>
          )}
        </div>
      </div>
      <ScaleNote paper="L = 1024, w = 256 (Table 12)" ours={`L = ${L}, w = ${w}`}>
        {L < 1024 ? `A context ${f1(1024 / L).replace(/\.0$/, '')}× shorter holds fewer rows, so` : 'The databases differ, so'} the counts are not
        comparable one to one; the statistic is the same.
      </ScaleNote>
    </Figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// This site's miniature: all in-context labels removed (inference-time)
// ─────────────────────────────────────────────────────────────────────────────

const T4_NONE = TABLE4.rows[0];
const T4_SELF = TABLE4.rows[2];

export function MiniLabelAblation() {
  const { data, error } = useMiniResults();
  const items = [
    { task: 'shop/customer-churn', ckpt: 'pretrain-churn', metric: 'AUROC', paper: { with: T4_NONE.zsClf, without: T4_SELF.zsClf } },
    { task: 'shop/product-sales', ckpt: 'pretrain-sales', metric: 'R²', paper: { with: T4_NONE.zsReg, without: T4_SELF.zsReg } },
  ];
  return (
    <Figure
      wide
      title={<TitleWithBadge title="Does our miniature lean on labels too?" kind="site" badge="results.json" />}
      caption={
        <>
          This site’s zero-shot checkpoints on ShopDB’s test split, with every visible cell of the target column turned into padding at inference time
          (self and other labels together). The paper removes self and other labels as separate ablations{' '}
          <Cite kind="paper" ref="p.9 §5.1" />, and its implementation is not released <Cite kind="spec" ref="§13 #3" />, so the setups differ; the paper’s Table 4 numbers are shown for direction only. Our test sets are
          tiny, so expect large variance.
        </>
      }
    >
      {error && <p className="text-[16px] text-ink-2">Could not load results.json ({error}).</p>}
      {!data && !error && <p className="text-[16px] text-ink-2">Loading this site’s results…</p>}
      {data && (
        <div className="grid gap-3 md:grid-cols-2">
          {items.map((it) => {
            const d = data.diagnostics?.inContextLabelAblation?.[it.ckpt]?.[`${it.task}/Test`];
            const n = data.labelPersistence?.tasks?.[it.task]?.splits?.Test?.rows;
            if (!d || d.with === undefined || d.withoutLabels === undefined) {
              return (
                <div key={it.task} className="rounded-xl border border-border px-4 py-3 text-[16px] text-ink-2">
                  No ablation result for {it.task} in results.json.
                </div>
              );
            }
            const a = d.with * 100;
            const b = d.withoutLabels * 100;
            return (
              <div key={it.task} className="rounded-xl border border-border bg-surface px-4 py-3.5">
                <div className="text-[15px] text-ink">{it.task}</div>
                <div className="text-[15px] text-ink-2">
                  zero-shot test {it.metric}
                  {n ? `, ${n} seeds` : ''}, checkpoint <code>{it.ckpt}</code>
                </div>
                <div className="mt-2 flex items-baseline gap-2.5">
                  <span className="num text-[28px] font-semibold text-ink">{f1(a)}</span>
                  <span className="text-[18px] text-muted">→</span>
                  <span className="num text-[28px] font-semibold text-ink">{f1(b)}</span>
                  <span className="num text-[16px] text-ink-2">({d1(b - a)})</span>
                </div>
                <div className="mt-1 text-[15px] text-ink-2">with labels → all labels removed</div>
                <div className="mt-2 border-t border-grid pt-2 text-[15px] text-muted">
                  Paper, mean over RelBench tasks, self labels removed: {f1(it.paper.with)} → {f1(it.paper.without)}{' '}
                  <Cite kind="paper" ref="p.9 Table 4" />
                </div>
              </div>
            );
          })}
        </div>
      )}
      <ScaleNote paper="22M parameters, RelBench, 10 + 8 tasks" ours="4 blocks, d = 64, ShopDB, 1 + 1 tasks" />
    </Figure>
  );
}
