/**
 * Step 5 figures: AUROC on raw logits and R² in z-space (main.py:328-337, spec §8.5), computed live
 * over every seed of a split with one context and one forward pass per seed (evaluate.ts). The
 * offline PyTorch evaluation of the same checkpoint (public/data/training/results.json, written by
 * python/train.py) is shown next to the browser's number when the settings match.
 */
import { scaleLinear } from 'd3';
import { Play, Square } from 'lucide-react';
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Button, Figure, Legend, Segmented, Slider, Stat, StatusBadge, SVG_TYPE, Switch } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { loadPublicJson } from '@/engine/model/io';
import type { Split } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { useEvaluation, type EvalItem, type EvalState } from './evaluate';
import { LaneHeader } from './LaneHeader';
import { fmt, fmtS, int, singular, type Lane } from './live';
import { bce, pairCounts, rocCurve, sigmoid } from './math';

const T = SVG_TYPE;

// ─────────────────────────────────────────────────────────────────────────────
// The offline reference numbers (results.json)
// ─────────────────────────────────────────────────────────────────────────────

interface ResultsJson {
  config?: { L?: number; w?: number };
  diagnostics?: {
    inContextLabelAblation?: Record<string, Record<string, { with?: number }>>;
  };
  extraZeroShotAndDiagnostics?: Record<string, { init?: number; final?: { value?: number } }>;
}

let resultsPromise: Promise<ResultsJson | null> | null = null;
function loadResults(): Promise<ResultsJson | null> {
  if (!resultsPromise) resultsPromise = loadPublicJson<ResultsJson>('data/training/results.json').catch(() => null);
  return resultsPromise;
}

/** The offline metric of the same checkpoint, task and split, when the context settings match. */
function useOfflineMetric(lane: Lane, split: Split): number | null {
  const [res, setRes] = useState<ResultsJson | null>(null);
  useEffect(() => {
    let alive = true;
    loadResults().then((r) => alive && setRes(r));
    return () => {
      alive = false;
    };
  }, []);
  if (!res || lane.disabled.length || lane.rngSeed !== 0) return null;
  if (res.config?.L !== lane.L || res.config?.w !== lane.w) return null;
  // results.json keys: "<task id>/<Val|Test>" for forecast tasks, "<task id>/All" for autocomplete (every row)
  const key = `${lane.task.id}/${lane.task.kind === 'autocomplete' ? 'All' : split}`;
  let v: number | undefined = res.diagnostics?.inContextLabelAblation?.[lane.ckpt]?.[key]?.with;
  if (v === undefined) {
    const extra = res.extraZeroShotAndDiagnostics?.[key];
    if (lane.ckpt === 'pretrain-final') v = extra?.final?.value;
    else if (lane.ckpt === 'init') v = extra?.init;
  }
  return typeof v === 'number' ? v : null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Runner UI
// ─────────────────────────────────────────────────────────────────────────────

function useSplitChoice(lane: Lane): [Split, (s: Split) => void] {
  const initial: Split = lane.split === 'Test' ? 'Test' : 'Val';
  const [split, setSplit] = useState<Split>(initial);
  useEffect(() => setSplit(initial), [initial]);
  return [split, setSplit];
}

function RunnerBar({ ev, lane, split }: { ev: EvalState; lane: Lane; split: Split }) {
  const n = ev.total;
  const what = lane.task.kind === 'autocomplete' ? 'rows' : `${split} seeds`;
  if (ev.status === 'done' && ev.result)
    return (
      <div className="text-[15px] text-ink-2">
        {int(ev.result.items.length)} {what} evaluated in the browser, one context and one forward pass each ({fmt(ev.result.msPerSeed, 0)} ms per
        seed).
      </div>
    );
  if (ev.status === 'running' || ev.status === 'waiting')
    return (
      <div className="flex flex-wrap items-center gap-3">
        <div
          className="h-2.5 w-[220px] max-w-full overflow-hidden rounded-full bg-surface-3"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={n}
          aria-valuenow={ev.done}
        >
          <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${n ? (100 * ev.done) / n : 0}%` }} />
        </div>
        <span className="num text-[15px] text-ink-2">{ev.status === 'waiting' ? 'queued…' : `${int(ev.done)} / ${int(n)} ${what}`}</span>
        <Button size="sm" variant="ghost" icon={<Square size={13} />} onClick={ev.cancel}>
          Stop
        </Button>
      </div>
    );
  return (
    <div className="flex flex-wrap items-center gap-3">
      <Button size="sm" icon={<Play size={14} />} onClick={ev.start} disabled={lane.assets.status !== 'ready'}>
        Evaluate all {n ? `${int(n)} ` : ''}
        {what}
      </Button>
      {ev.status === 'error' && <StatusBadge status="critical">{ev.error ?? 'evaluation failed'}</StatusBadge>}
    </div>
  );
}

function Offline({ live, offline, digits = 4 }: { live: number; offline: number | null; digits?: number }) {
  if (offline === null || !Number.isFinite(live)) return null;
  const same = Math.abs(live - offline) < 5e-4;
  return (
    <StatusBadge status={same ? 'good' : 'info'}>
      offline PyTorch evaluation (results.json): {fmt(offline, digits)}
      {same ? ', same number' : `, browser ${fmt(live, digits)}`}
    </StatusBadge>
  );
}

/** A callback ref + "has this element come near the viewport yet" (starts the evaluation lazily). */
function useAutoWhenVisible(): [(el: HTMLDivElement | null) => void, boolean] {
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    if (!el || seen) return;
    if (typeof IntersectionObserver === 'undefined') {
      setSeen(true);
      return;
    }
    const io = new IntersectionObserver(
      (es) => {
        if (es.some((e) => e.isIntersecting)) setSeen(true);
      },
      { rootMargin: '300px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [el, seen]);
  return [setEl, seen];
}

// ─────────────────────────────────────────────────────────────────────────────
// ROC / AUROC
// ─────────────────────────────────────────────────────────────────────────────

export function RocFigure({ lane }: { lane: Lane }) {
  const [split, setSplit] = useSplitChoice(lane);
  const [boxRef, visible] = useAutoWhenVisible();
  const ev = useEvaluation(lane, split, visible);
  const offline = useOfflineMetric(lane, split);
  const items = ev.result?.items ?? null;
  return (
    <Figure
      wide
      title="AUROC on the logits: how well the scores rank the two classes"
      controls={
        lane.task.kind === 'forecast' ? (
          <Segmented
            label="Split"
            value={split}
            onChange={setSplit}
            options={[
              { value: 'Val', label: 'Val' },
              { value: 'Test', label: 'Test' },
            ]}
          />
        ) : undefined
      }
      caption={
        <>
          Left: the ROC curve, true-positive rate against false-positive rate as the threshold sweeps down the logits. Right: every seed’s logit,
          split by its label. The violet threshold line and the violet dot on the curve move together. AUROC is the shaded area under the curve.
        </>
      }
    >
      <LaneHeader lane={lane} />
      <div ref={boxRef} className="mb-4">
        <RunnerBar ev={ev} lane={lane} split={split} />
      </div>
      {items && items.length > 0 ? (
        <RocBody items={items} metric={ev.result!.metric} offline={offline} lane={lane} />
      ) : (
        <div className="rounded-lg border border-dashed border-border-strong px-4 py-10 text-center text-[16px] text-ink-2">
          {ev.status === 'running' || ev.status === 'waiting'
            ? 'Running one forward pass per seed…'
            : 'The ROC curve appears once every seed is evaluated.'}
        </div>
      )}
    </Figure>
  );
}

function RocBody({ items, metric, offline, lane }: { items: EvalItem[]; metric: number; offline: number | null; lane: Lane }) {
  const y = items.map((it) => it.y01);
  const s = items.map((it) => it.pred);
  const roc = useMemo(() => rocCurve(y, s), [items]); // eslint-disable-line react-hooks/exhaustive-deps
  const pairs = useMemo(() => pairCounts(y, s), [items]); // eslint-disable-line react-hooks/exhaustive-deps
  const nPos = y.reduce<number>((a, b) => a + b, 0);
  const nNeg = y.length - nPos;
  const lo = Math.min(...s);
  const hi = Math.max(...s);
  const [thr, setThr] = useState(0);
  const [probAxis, setProbAxis] = useState(false);
  const [shift, setShift] = useState(0);
  const tp = items.filter((it) => it.y01 === 1 && it.pred >= thr).length;
  const fp = items.filter((it) => it.y01 === 0 && it.pred >= thr).length;
  const tpr = nPos ? tp / nPos : 0;
  const fpr = nNeg ? fp / nNeg : 0;
  const meanBce = (c: number) => items.reduce((a, it) => a + bce(it.pred + c, it.y01), 0) / items.length;
  const shiftedPairs = useMemo(
    () =>
      pairCounts(
        y,
        s.map((v) => v + shift),
      ),
    [items, shift],
  ); // eslint-disable-line react-hooks/exhaustive-deps
  const aurocShift = (shiftedPairs.correct + shiftedPairs.ties / 2) / Math.max(1, shiftedPairs.total);
  const ent = lane.task.entityTable ? singular(lane.task.entityTable) : `${lane.task.table ?? lane.task.name} row`;

  return (
    <div>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,360px)_minmax(0,1fr)] lg:gap-8">
        <div className="min-w-0">
          <RocChart roc={roc} point={{ fpr, tpr }} />
        </div>
        <div className="min-w-0">
          <div className="mb-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
            <Stat label={`AUROC, ${int(items.length)} seeds`} value={fmt(metric, 4)} emphasis />
            <Stat label="label 1 / label 0" value={`${int(nPos)} / ${int(nNeg)}`} />
            <Stat
              label="pairs ranked right"
              value={`${int(pairs.correct)}`}
              caption={`of ${int(pairs.total)}${pairs.ties ? `, ${int(pairs.ties)} ties` : ''}`}
            />
          </div>
          <ScoreStrip items={items} thr={thr} prob={probAxis} entity={ent} />
          <p className="!mt-2 !mb-0 text-[15.5px] leading-snug text-ink-2">
            At this threshold the model flags {int(tp)} of the {int(nPos)} label-1 seeds (true-positive rate{' '}
            <span className="num text-ink">{fmt(tpr, 2)}</span>) and {int(fp)} of the {int(nNeg)} label-0 seeds (false-positive rate{' '}
            <span className="num text-ink">{fmt(fpr, 2)}</span>): the violet dot on the curve.
          </p>
          <div className="mt-3 flex flex-wrap items-end gap-x-6 gap-y-3">
            <Slider
              label="Threshold on the logit"
              value={thr}
              min={Math.floor(lo * 10) / 10 - 0.1}
              max={Math.ceil(hi * 10) / 10 + 0.1}
              step={0.01}
              onChange={setThr}
              format={(v) => (
                <>
                  ẑ ≥ {fmtS(v, 2)} <span className="text-ink-2">(p ≥ {fmt(sigmoid(v), 2)})</span>
                </>
              )}
              className="w-full max-w-[460px]"
            />
            <Switch checked={probAxis} onChange={setProbAxis} label="Show σ(logit) instead of the logit" />
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Offline live={metric} offline={offline} />
          </div>
        </div>
      </div>

      <div className="mt-6 rounded-xl border border-border bg-surface-2/60 p-4">
        <div className="mb-1 text-[16.5px] font-semibold text-ink">Shift every logit by the same amount</div>
        <p className="!mb-3 text-[15.5px] leading-snug text-ink-2">
          Adding a constant changes every probability but not the order of the scores. AUROC only sees the order; the loss sees the values.
        </p>
        <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
          <Slider
            label="Shift c"
            value={shift}
            min={-4}
            max={4}
            step={0.1}
            onChange={setShift}
            format={(v) => fmtS(v, 1)}
            className="w-full max-w-[300px]"
          />
          <div className="flex flex-wrap gap-x-8 gap-y-2">
            <div>
              <div className="text-[15px] text-ink-2">AUROC of ẑ + c</div>
              <div className="num text-[20px] font-semibold text-ink">{fmt(aurocShift, 4)}</div>
            </div>
            <div>
              <div className="text-[15px] text-ink-2">mean BCE of ẑ + c</div>
              <div className="num text-[20px] font-semibold text-ink">
                {fmt(meanBce(shift), 4)} <span className="text-[15px] font-normal text-ink-2">(at c = 0: {fmt(meanBce(0), 4)})</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function RocChart({ roc, point }: { roc: { fpr: number; tpr: number; thr: number }[]; point: { fpr: number; tpr: number } }) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const side = Math.max(260, Math.min(360, size.width || 340));
  const m = { l: 60, r: 12, t: 12, b: 46 };
  const w = side - m.l - m.r;
  const h = side - m.t - m.b;
  const x = scaleLinear().domain([0, 1]).range([0, w]);
  const yy = scaleLinear().domain([0, 1]).range([h, 0]);
  const d = roc.map((p, i) => `${i ? 'L' : 'M'}${x(p.fpr).toFixed(1)},${yy(p.tpr).toFixed(1)}`).join('');
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const [hover, setHover] = useState<number | null>(null);
  return (
    <div ref={ref} className="min-w-0">
      <svg
        width={side}
        height={side}
        role="img"
        aria-label="ROC curve"
        className="block"
        onPointerMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const fx = x.invert(e.clientX - r.left - m.l);
          let best = 0;
          roc.forEach((p, i) => {
            if (Math.abs(p.fpr - fx) < Math.abs(roc[best].fpr - fx)) best = i;
          });
          setHover(best);
        }}
        onPointerLeave={() => setHover(null)}
      >
        <g transform={`translate(${m.l},${m.t})`}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={0} y2={h} style={{ stroke: 'var(--grid)' }} />
              <line x1={0} x2={w} y1={yy(t)} y2={yy(t)} style={{ stroke: 'var(--grid)' }} />
              <text x={x(t)} y={h + 19} textAnchor="middle" className="tabular" style={{ fill: 'var(--muted)', fontSize: T.tick }}>
                {t}
              </text>
              <text x={-7} y={yy(t)} dy="0.32em" textAnchor="end" className="tabular" style={{ fill: 'var(--muted)', fontSize: T.tick }}>
                {t}
              </text>
            </g>
          ))}
          <path d={`${d}L${x(1)},${yy(0)}L${x(0)},${yy(0)}Z`} style={{ fill: 'var(--accent-soft)' }} />
          <line x1={x(0)} y1={yy(0)} x2={x(1)} y2={yy(1)} style={{ stroke: 'var(--axis)' }} strokeDasharray="5 4" strokeWidth={1.5} />
          <text
            x={x(0.62)}
            y={yy(0.52)}
            transform={`rotate(-45 ${x(0.62)} ${yy(0.52)})`}
            textAnchor="middle"
            style={{ fill: 'var(--muted)', fontSize: T.annotation }}
          >
            random ranking: 0.5
          </text>
          <path d={d} fill="none" style={{ stroke: 'var(--ink)' }} strokeWidth={2} strokeLinejoin="round" />
          {hover !== null && (
            <circle cx={x(roc[hover].fpr)} cy={yy(roc[hover].tpr)} r={5} style={{ fill: 'var(--surface)', stroke: 'var(--ink)' }} strokeWidth={2} />
          )}
          <circle cx={x(point.fpr)} cy={yy(point.tpr)} r={7} style={{ fill: 'var(--accent)', stroke: 'var(--surface)' }} strokeWidth={2.5} />
          <text x={w / 2} y={h + 40} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: T.axisTitle }}>
            false-positive rate
          </text>
          <text transform={`translate(${-48},${h / 2}) rotate(-90)`} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: T.axisTitle }}>
            true-positive rate
          </text>
        </g>
      </svg>
      <div className="mt-1 min-h-[24px] text-[15px] text-ink-2">
        {hover !== null && Number.isFinite(roc[hover].thr) ? (
          <>
            threshold ẑ ≥ <span className="num text-ink">{fmtS(roc[hover].thr, 3)}</span>: TPR{' '}
            <span className="num text-ink">{fmt(roc[hover].tpr, 3)}</span>, FPR <span className="num text-ink">{fmt(roc[hover].fpr, 3)}</span>
          </>
        ) : (
          <span className="text-muted">Hover the curve to read a threshold.</span>
        )}
      </div>
    </div>
  );
}

/** Deterministic jitter in [−1, 1] from a seed id. */
function jitter(k: number): number {
  const v = Math.sin(k * 12.9898 + 78.233) * 43758.5453;
  return (v - Math.floor(v)) * 2 - 1;
}

function ScoreStrip({ items, thr, prob, entity }: { items: EvalItem[]; thr: number; prob: boolean; entity: string }) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const width = Math.max(280, size.width || 520);
  const labelW = 74;
  const rowH = 54;
  const top = 6;
  const H = top + rowH * 2 + 30;
  const lo = Math.min(...items.map((it) => it.pred));
  const hi = Math.max(...items.map((it) => it.pred));
  const f = (v: number) => (prob ? sigmoid(v) : v);
  const dom: [number, number] = prob ? [0, 1] : [Math.floor(lo) - 0.25, Math.ceil(hi) + 0.25];
  const x = scaleLinear()
    .domain(dom)
    .range([labelW, width - 12]);
  const ticks = x.ticks(Math.max(2, Math.floor((width - labelW) / 80)));
  const [hover, setHover] = useState<EvalItem | null>(null);
  const tx = x(f(thr));
  return (
    <div ref={ref} className="min-w-0">
      <Legend
        className="mb-1"
        items={[
          { id: '1', label: 'label 1', slot: 1, mark: 'rect' },
          { id: '0', label: 'label 0', slot: 2, mark: 'rect' },
        ]}
      />
      <svg
        width={width}
        height={H}
        role="img"
        aria-label="Logits of every evaluated seed by label"
        className="block"
        onPointerLeave={() => setHover(null)}
      >
        <rect x={tx} y={top} width={Math.max(0, width - 12 - tx)} height={rowH * 2} style={{ fill: 'var(--accent-soft)' }} />
        {[1, 0].map((lab, r) => (
          <g key={lab}>
            <line x1={labelW} x2={width - 12} y1={top + rowH * (r + 1)} y2={top + rowH * (r + 1)} style={{ stroke: 'var(--grid)' }} />
            <text x={labelW - 10} y={top + rowH * r + rowH / 2} dy="0.32em" textAnchor="end" style={{ fill: 'var(--ink)', fontSize: T.category }}>
              label {lab}
            </text>
          </g>
        ))}
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={top + rowH * 2 + 20} textAnchor="middle" className="tabular" style={{ fill: 'var(--muted)', fontSize: T.tick }}>
            {prob ? fmt(t, 2) : t === 0 ? '0' : fmtS(t, Number.isInteger(t) ? 0 : 1)}
          </text>
        ))}
        {items.map((it, i) => {
          const r = it.y01 === 1 ? 0 : 1;
          const cy = top + rowH * r + rowH / 2 + jitter(it.seed) * (rowH / 2 - 9);
          const isH = hover === it;
          return (
            <circle
              key={i}
              cx={x(f(it.pred))}
              cy={cy}
              r={isH ? 6.5 : 5}
              style={{
                fill: catVar(it.y01 === 1 ? 1 : 2),
                stroke: 'var(--surface)',
              }}
              strokeWidth={1.5}
              onPointerEnter={() => setHover(it)}
            />
          );
        })}
        <line x1={tx} x2={tx} y1={top - 2} y2={top + rowH * 2 + 2} style={{ stroke: 'var(--accent)' }} strokeWidth={2.5} />
      </svg>
      <div className="min-h-[24px] text-[15px] text-ink-2" aria-live="polite">
        {hover ? (
          <>
            {entity} {hover.entityRow}: logit <span className="num text-ink">{fmtS(hover.pred, 3)}</span> (p = {fmt(sigmoid(hover.pred), 3)}), label{' '}
            <span className="num text-ink">{hover.y01}</span>
          </>
        ) : (
          <span className="text-muted">Hover a dot to see one seed. Shaded: predicted positive at this threshold.</span>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// R²
// ─────────────────────────────────────────────────────────────────────────────

export function R2Figure({ lane }: { lane: Lane }) {
  const [split, setSplit] = useSplitChoice(lane);
  const [boxRef, visible] = useAutoWhenVisible();
  const ev = useEvaluation(lane, split, visible);
  const offline = useOfflineMetric(lane, split);
  const items = ev.result?.items ?? null;
  return (
    <Figure
      wide
      title="R² in z-space: how much of the labels’ spread the predictions explain"
      controls={
        lane.task.kind === 'forecast' ? (
          <Segmented
            label="Split"
            value={split}
            onChange={setSplit}
            options={[
              { value: 'Val', label: 'Val' },
              { value: 'Test', label: 'Test' },
            ]}
          />
        ) : undefined
      }
      caption={
        <>
          Each dot is one seed: its label against the number head’s prediction. Dots on the diagonal are perfect. The dashed line predicts the mean
          label ȳ for everyone, which scores R² = 0 by definition. Switching to original units rescales both axes by σ and leaves R² unchanged.
        </>
      }
    >
      <LaneHeader lane={lane} />
      <div ref={boxRef} className="mb-4">
        <RunnerBar ev={ev} lane={lane} split={split} />
      </div>
      {items && items.length > 1 ? (
        <R2Body items={items} metric={ev.result!.metric} offline={offline} lane={lane} />
      ) : (
        <div className="rounded-lg border border-dashed border-border-strong px-4 py-10 text-center text-[16px] text-ink-2">
          {ev.status === 'running' || ev.status === 'waiting'
            ? 'Running one forward pass per seed…'
            : 'The scatter plot appears once every seed is evaluated.'}
        </div>
      )}
    </Figure>
  );
}

function R2Body({ items, metric, offline, lane }: { items: EvalItem[]; metric: number; offline: number | null; lane: Lane }) {
  const [units, setUnits] = useState(false);
  const mu = lane.stats?.mean ?? 0;
  const sd = lane.stats?.std ?? 1;
  const tr = (z: number) => (units ? z * sd + mu : z);
  const ys = items.map((it) => tr(it.z));
  const ps = items.map((it) => tr(it.pred));
  const mean = ys.reduce((a, b) => a + b, 0) / ys.length;
  let ssRes = 0;
  let ssTot = 0;
  ys.forEach((v, i) => {
    ssRes += (v - ps[i]) ** 2;
    ssTot += (v - mean) ** 2;
  });
  const r2v = 1 - ssRes / ssTot;
  const ent = lane.task.entityTable ? singular(lane.task.entityTable) : `${lane.task.table ?? lane.task.name} row`;
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,420px)_minmax(0,1fr)] lg:gap-8">
      <div className="min-w-0">
        <Scatter ys={ys} ps={ps} mean={mean} items={items} units={units} col={lane.task.targetCol} entity={ent} />
      </div>
      <div className="min-w-0">
        <div className="mb-3">
          <Segmented
            label="Scale"
            value={units ? 'units' : 'z'}
            onChange={(v) => setUnits(v === 'units')}
            options={[
              { value: 'z', label: 'z-space (as the code)' },
              {
                value: 'units',
                label: `original units (${lane.task.targetCol})`,
              },
            ]}
          />
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <Stat label={`R², ${int(items.length)} seeds`} value={fmt(metric, 4)} emphasis caption="r2_score in z-space" />
          <Stat label="SS_res = Σ(y − ŷ)²" value={fmt(ssRes, units ? 1 : 3)} />
          <Stat label="SS_tot = Σ(y − ȳ)²" value={fmt(ssTot, units ? 1 : 3)} />
        </div>
        <p className="!mt-4 !mb-2 text-[16px] leading-relaxed text-ink-2">
          R² = 1 − SS_res / SS_tot = 1 − {fmt(ssRes, units ? 1 : 3)} / {fmt(ssTot, units ? 1 : 3)} ={' '}
          <span className="num font-semibold text-ink">{fmt(r2v, 4)}</span>
          {units ? <> (both sums are σ² = {fmt(sd * sd, 2)} times their z-space values)</> : null}.
        </p>
        <Offline live={metric} offline={offline} />
      </div>
    </div>
  );
}

function Scatter({
  ys,
  ps,
  mean,
  items,
  units,
  col,
  entity,
}: {
  ys: number[];
  ps: number[];
  mean: number;
  items: EvalItem[];
  units: boolean;
  col: string;
  entity: string;
}) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const side = Math.max(270, Math.min(420, size.width || 400));
  const m = { l: 54, r: 14, t: 12, b: 50 };
  const w = side - m.l - m.r;
  const h = side - m.t - m.b;
  const all = [...ys, ...ps];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pad = (hi - lo) * 0.06 || 1;
  const x = scaleLinear()
    .domain([lo - pad, hi + pad])
    .range([0, w])
    .nice();
  const y = scaleLinear().domain(x.domain()).range([h, 0]);
  const ticks = x.ticks(5);
  const [hover, setHover] = useState<number | null>(null);
  const f = (v: number) => (units ? fmt(v, Math.abs(v) >= 10 ? 0 : 1) : fmtS(v, 2));
  const tickFmt = (v: number) => (Number.isInteger(v) ? String(v).replace('-', '−') : fmt(v, 1));
  const [x0, x1] = x.domain();
  let readout: ReactNode = <span className="text-muted">Hover a dot to see one seed.</span>;
  if (hover !== null)
    readout = (
      <>
        {entity} {items[hover].entityRow}: label <span className="num text-ink">{f(ys[hover])}</span>, prediction{' '}
        <span className="num text-ink">{f(ps[hover])}</span>
      </>
    );
  return (
    <div ref={ref} className="min-w-0">
      <svg width={side} height={side} role="img" aria-label="Predictions against labels" className="block" onPointerLeave={() => setHover(null)}>
        <g transform={`translate(${m.l},${m.t})`}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={0} y2={h} style={{ stroke: 'var(--grid)' }} />
              <line x1={0} x2={w} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--grid)' }} />
              <text x={x(t)} y={h + 19} textAnchor="middle" className="tabular" style={{ fill: 'var(--muted)', fontSize: T.tick }}>
                {tickFmt(t)}
              </text>
              <text x={-7} y={y(t)} dy="0.32em" textAnchor="end" className="tabular" style={{ fill: 'var(--muted)', fontSize: T.tick }}>
                {tickFmt(t)}
              </text>
            </g>
          ))}
          <line x1={x(x0)} y1={y(x0)} x2={x(x1)} y2={y(x1)} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.5} />
          <text x={x(x1) - 12} y={y(x1) + 4} textAnchor="end" style={{ fill: 'var(--ink-2)', fontSize: T.annotation }}>
            ŷ = y
          </text>
          <line x1={0} x2={w} y1={y(mean)} y2={y(mean)} style={{ stroke: 'var(--axis)' }} strokeDasharray="5 4" strokeWidth={1.5} />
          <text
            x={w - 4}
            y={y(mean) - 6}
            textAnchor="end"
            style={{
              fill: 'var(--ink-2)',
              fontSize: T.annotation,
              paintOrder: 'stroke',
              stroke: 'var(--surface)',
              strokeWidth: 4,
            }}
          >
            ŷ = ȳ (R² = 0)
          </text>
          {ys.map((v, i) => (
            <circle
              key={i}
              cx={x(v)}
              cy={y(ps[i])}
              r={hover === i ? 6.5 : 5}
              style={{
                fill: 'var(--accent)',
                stroke: 'var(--surface)',
                opacity: hover === null || hover === i ? 0.9 : 0.55,
              }}
              strokeWidth={1.5}
              onPointerEnter={() => setHover(i)}
            />
          ))}
          <text x={w / 2} y={h + 42} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: T.axisTitle }}>
            label y {units ? `(${col})` : '(z)'}
          </text>
          <text transform={`translate(${-40},${h / 2}) rotate(-90)`} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: T.axisTitle }}>
            prediction ŷ {units ? `(${col})` : '(z)'}
          </text>
        </g>
      </svg>
      <div className="mt-1 min-h-[24px] text-[15px] text-ink-2" aria-live="polite">
        {readout}
      </div>
    </div>
  );
}
