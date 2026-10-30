/**
 * Step 4 figures: the loss with live numbers (model.py:250-265, spec §8.2).
 *  - BceFigure: BCEWithLogits(ẑ, 1{z > 0}) = max(ẑ,0) − ẑ·y + log(1 + e^{−|ẑ|}) at the binary lane's target.
 *  - HuberFigure: Huber_δ=1(ẑ − z) at the regression lane's target, in z-space.
 *  - BatchFigure: a mini batch of Train contexts; the loss is the plain mean over its masked cells.
 * The engine's own loss (`ForwardResult.loss`, src/engine/model/loss.ts) is shown next to the
 * hand-computed value as a check.
 */
import { Shuffle } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Button, Figure, Legend, ScaleNote, SemBadge, Slider, StatusBadge, SVG_TYPE, textWidth } from '@/components';
import { hBarPath } from '@/components/viz/chartParts';
import { useElementSize } from '@/components/viz/useSize';
import { catVar } from '@/styles/palette';
import { SemType } from '@/engine/types';
import { CurveChart, Readout } from './charts';
import { useMiniBatch, type BatchItem } from './evaluate';
import { LaneHeader } from './LaneHeader';
import { fmt, fmtS, int, singular, taskLabel, type Lane } from './live';
import { bce, huber, sigmoid } from './math';

const T = SVG_TYPE;

function Waiting() {
  return (
    <div className="rounded-lg border border-dashed border-border-strong px-4 py-8 text-center text-[16px] text-ink-2">Running the forward pass…</div>
  );
}

function EngineCheck({ mine, engine }: { mine: number; engine: number | undefined }) {
  if (engine === undefined || !Number.isFinite(engine)) return null;
  const d = Math.abs(mine - engine);
  return d < 1e-5 ? (
    <StatusBadge status="good">equals the engine’s loss for this context ({fmt(engine, 5)})</StatusBadge>
  ) : (
    <StatusBadge status="critical">
      engine loss {fmt(engine, 5)} differs by {d.toExponential(1)}
    </StatusBadge>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// BCE
// ─────────────────────────────────────────────────────────────────────────────

export function BceFigure({ lane }: { lane: Lane }) {
  const { pred, fr, task } = lane;
  const z = pred?.output ?? NaN;
  const y = pred?.label.binary ?? 0;
  const R = Number.isFinite(z) ? Math.max(5, Math.ceil(Math.abs(z) + 1.5)) : 5;
  const a = Math.max(z, 0);
  const b = z * y;
  const c = Math.log1p(Math.exp(-Math.abs(z)));
  const loss = bce(z, y);
  const viaSig = y === 1 ? -Math.log(sigmoid(z)) : -Math.log(1 - sigmoid(z));
  const yMax = Math.ceil(Math.max(bce(-R, 1), bce(R, 0)));
  return (
    <Figure
      wide
      title="Binary cross-entropy with logits, at the masked cell"
      caption={
        <>
          Each curve is the loss as a function of the logit for one value of the label. The dot is the live prediction on the curve of the actual
          label. A confident wrong logit costs about |z|; a confident right one costs almost nothing.
        </>
      }
    >
      <LaneHeader lane={lane} />
      {!pred ? (
        <Waiting />
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] lg:gap-8">
          <div className="min-w-0">
            <Legend
              className="mb-2"
              items={[
                {
                  id: 'y1',
                  label: `label 1 (${task.targetCol} = true)`,
                  slot: 1,
                  mark: 'line',
                },
                {
                  id: 'y0',
                  label: `label 0 (${task.targetCol} = false)`,
                  slot: 2,
                  mark: 'line',
                },
              ]}
            />
            <CurveChart
              title="BCE with logits for label 1 and label 0"
              series={[
                {
                  id: 'y1',
                  label: 'label 1',
                  color: catVar(1),
                  fn: (x) => bce(x, 1),
                },
                {
                  id: 'y0',
                  label: 'label 0',
                  color: catVar(2),
                  fn: (x) => bce(x, 0),
                },
              ]}
              xDomain={[-R, R]}
              yDomain={[0, yMax]}
              xLabel="logit ẑ"
              yLabel="loss"
              height={290}
              vlines={[{ x: 0, label: 'ẑ = 0: ln 2 for either label' }]}
              marks={[
                {
                  x: z,
                  y: loss,
                  kind: 'live',
                  label: `ℓ = ${fmt(loss, 3)}`,
                  guides: true,
                },
              ]}
              readout={(x) => (
                <>
                  ẑ = <span className="num text-ink">{fmtS(x, 2)}</span>: loss <span className="num text-ink">{fmt(bce(x, 1), 3)}</span> if the label
                  is 1, <span className="num text-ink">{fmt(bce(x, 0), 3)}</span> if it is 0
                </>
              )}
            />
          </div>
          <div className="min-w-0">
            <div className="rounded-lg border border-border bg-surface px-3 py-1">
              <Readout label="Logit ẑ (boolean head)" value={fmtS(z, 4)} />
              <Readout label="Target y = 1{z_label > 0}" value={String(y)} />
              <Readout label="max(ẑ, 0)" value={fmt(a, 4)} />
              <Readout label="− ẑ · y" value={fmtS(-b, 4)} />
              <Readout label="+ log(1 + e^(−|ẑ|))" value={fmtS(c, 4)} />
              <Readout
                label="Loss ℓ"
                value={fmt(loss, 4)}
                strong
                note={
                  <>
                    Same as −log {y === 1 ? 'σ(ẑ)' : '(1 − σ(ẑ))'} = {fmt(viaSig, 4)}, written in a form that never overflows.
                  </>
                }
              />
            </div>
            <div className="mt-3">
              <EngineCheck mine={loss} engine={fr?.loss} />
            </div>
          </div>
        </div>
      )}
    </Figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Huber
// ─────────────────────────────────────────────────────────────────────────────

export function HuberFigure({ lane }: { lane: Lane }) {
  const { pred, fr, ctx, stats } = lane;
  const zHat = pred?.output ?? NaN;
  const zLab = ctx.tokens[ctx.targetPos]?.numberValue ?? NaN;
  const dLive = zHat - zLab;
  const [tryD, setTryD] = useState<number | null>(null);
  const R = Number.isFinite(dLive) ? Math.max(3, Math.ceil(Math.abs(dLive) + 0.75)) : 3;
  const loss = huber(dLive);
  const yMax = Math.max(2.5, Math.ceil(huber(R) + 0.5));
  return (
    <Figure
      wide
      title="Huber loss (δ = 1), at the masked cell"
      caption={
        <>
          The solid curve is the Huber loss of the residual d = ẑ − z, both in z-space. Inside the shaded band (|d| &lt; 1) it equals ½d², the dashed
          parabola; outside it grows only linearly, |d| − ½, so its slope never exceeds 1.
        </>
      }
    >
      <LaneHeader lane={lane} />
      {!pred ? (
        <Waiting />
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] lg:gap-8">
          <div className="min-w-0">
            <Legend
              className="mb-2"
              items={[
                { id: 'h', label: 'Huber, δ = 1', slot: 7, mark: 'line' },
                {
                  id: 'q',
                  label: '½ d² (for comparison)',
                  slot: 0,
                  mark: 'line',
                },
              ]}
            />
            <CurveChart
              title="Huber loss of the z-space residual"
              series={[
                {
                  id: 'q',
                  label: '½d²',
                  color: 'var(--muted)',
                  fn: (x) => 0.5 * x * x,
                  dash: '5 4',
                  width: 1.75,
                },
                {
                  id: 'h',
                  label: 'Huber',
                  color: catVar(7),
                  fn: huber,
                  width: 2.5,
                },
              ]}
              xDomain={[-R, R]}
              yDomain={[0, yMax]}
              xLabel="residual d = ẑ − z (z-space)"
              yLabel="loss"
              height={290}
              bands={[{ x0: -1, x1: 1, label: 'quadratic zone' }]}
              marks={[
                ...(tryD !== null && Math.abs(tryD - dLive) > 0.02
                  ? [
                      {
                        x: tryD,
                        y: huber(tryD),
                        kind: 'ghost' as const,
                        label: `ℓ = ${fmt(huber(tryD), 3)}`,
                      },
                    ]
                  : []),
                {
                  x: dLive,
                  y: loss,
                  kind: 'live' as const,
                  label: `d = ${fmtS(dLive, 2)}, ℓ = ${fmt(loss, 3)}`,
                  guides: true,
                },
              ]}
              readout={(x) => (
                <>
                  d = <span className="num text-ink">{fmtS(x, 2)}</span>: Huber <span className="num text-ink">{fmt(huber(x), 3)}</span>, ½d²{' '}
                  <span className="num text-ink">{fmt(0.5 * x * x, 3)}</span>
                </>
              )}
            />
            <Slider
              label="Try another residual"
              value={tryD ?? Math.round(dLive * 20) / 20}
              min={-R}
              max={R}
              step={0.05}
              onChange={setTryD}
              format={(v) => (
                <>
                  d = {fmtS(v, 2)} <span className="text-ink-2">→ ℓ = {fmt(huber(v), 3)}</span>
                </>
              )}
              className="mt-3 w-full max-w-[420px]"
            />
          </div>
          <div className="min-w-0">
            <div className="rounded-lg border border-border bg-surface px-3 py-1">
              <Readout label="Prediction ẑ (number head)" value={fmtS(zHat, 4)} />
              <Readout label="Stored label z" value={fmtS(zLab, 4)} />
              <Readout
                label="Residual d = ẑ − z"
                value={fmtS(dLive, 4)}
                note={stats ? <>That is {fmt(Math.abs(dLive) * stats.std, 2)} units of the original column (|d| × σ).</> : undefined}
              />
              <Readout
                label={Math.abs(dLive) < 1 ? '|d| < 1, so ½ d²' : '|d| ≥ 1, so |d| − ½'}
                value={fmt(loss, 4)}
                strong
                note="F.huber_loss with its default δ = 1.0."
              />
            </div>
            <div className="mt-3">
              <EngineCheck mine={loss} engine={fr?.loss} />
            </div>
          </div>
        </div>
      )}
    </Figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// A mini batch
// ─────────────────────────────────────────────────────────────────────────────

const PER_TASK = 4;

export function BatchFigure({ binary, regression, model }: { binary: Lane; regression: Lane; model: Lane }) {
  const [draw, setDraw] = useState(0);
  const store = model.assets.store;
  const weights = model.assets.weights;
  const inputs = useMemo(
    () =>
      store && weights
        ? {
            parts: [
              { pdb: binary.pdb, task: binary.task, n: PER_TASK },
              { pdb: regression.pdb, task: regression.task, n: PER_TASK },
            ],
            L: model.L,
            w: model.w,
            store,
            weights,
            draw,
          }
        : null,
    [binary.pdb, binary.task, regression.pdb, regression.task, model.L, model.w, store, weights, draw],
  );
  const { items } = useMiniBatch(inputs);
  const mean = items && items.length ? items.reduce((s, it) => s + it.term.loss, 0) / items.length : NaN;
  const sums = useMemo(() => {
    const s = {
      boolean: { sum: 0, n: 0 },
      number: { sum: 0, n: 0 },
      datetime: { sum: 0, n: 0 },
    };
    for (const it of items ?? []) {
      s[it.term.head].sum += it.term.loss;
      s[it.term.head].n++;
    }
    return s;
  }, [items]);

  return (
    <Figure
      wide
      title={`A mini batch of ${PER_TASK * 2} training contexts: one masked cell each`}
      controls={
        <Button size="sm" icon={<Shuffle size={15} />} onClick={() => setDraw((d) => d + 1)}>
          Draw another batch
        </Button>
      }
      caption={
        <>
          {PER_TASK} Train rows of {taskLabel(binary.task)} and {PER_TASK} of {taskLabel(regression.task)}, each with its own sampled context (epoch
          0), run through one model (checkpoint <span className="font-mono">{model.ckpt}</span>). Bars are the per-cell losses; the line is their
          mean, the number the optimizer minimizes. The batch is an illustration: these exact steps were never taken in training.
        </>
      }
    >
      {!items ? (
        <div className="rounded-lg border border-dashed border-border-strong px-4 py-8 text-center text-[16px] text-ink-2">
          Sampling {PER_TASK * 2} contexts and running {PER_TASK * 2} forward passes…
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)] lg:gap-8">
          <BatchBars items={items} mean={mean} />
          <div className="min-w-0">
            <div className="rounded-lg border border-border bg-surface px-3 py-1">
              <Readout
                label={
                  <span className="inline-flex items-center gap-1.5">
                    <SemBadge sem={SemType.Boolean} size="sm" /> boolean cells: Σ BCE
                  </span>
                }
                value={`${fmt(sums.boolean.sum, 4)}  (${sums.boolean.n})`}
              />
              <Readout
                label={
                  <span className="inline-flex items-center gap-1.5">
                    <SemBadge sem={SemType.Number} size="sm" /> number cells: Σ Huber
                  </span>
                }
                value={`${fmt(sums.number.sum, 4)}  (${sums.number.n})`}
              />
              <Readout label="masks.sum(): masked cells in the batch" value={int(items.length)} note="One per context, whatever its task or type." />
              <Readout
                label="loss = (Σ BCE + Σ Huber) / masks.sum()"
                value={fmt(mean, 4)}
                strong
                note="No weight per task, per type or per class: every masked cell counts once."
              />
            </div>
            <ScaleNote
              paper="256 contexts per step (32 on each of 8 GPUs)"
              ours={`${PER_TASK * 2} contexts here, for legibility`}
              className="mt-3"
            />
          </div>
        </div>
      )}
    </Figure>
  );
}

function BatchBars({ items, mean }: { items: BatchItem[]; mean: number }) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const width = Math.max(300, size.width || 560);
  // narrow screens: the entity alone (the bar color and the legend give the task's loss type)
  const compact = width < 560;
  const labels = items.map(
    (it) => `${singular(it.task.entityTable ?? it.task.table ?? it.task.name)} ${it.entityRow}${compact ? '' : ` · ${it.task.name}`}`,
  );
  const labelW = Math.min(width * 0.52, Math.ceil(Math.max(...labels.map((l) => textWidth(l, T.category))) + 10));
  const rowH = 30;
  const top = 8;
  const H = top + items.length * rowH + 34;
  const plotW = width - labelW - 56;
  const maxL = Math.max(mean * 1.2, ...items.map((it) => it.term.loss), 0.5);
  const sx = (v: number) => labelW + (v / maxL) * plotW;
  return (
    <div ref={ref} className="min-w-0">
      <svg width={width} height={H} role="img" aria-label={`Per-cell losses of the batch, mean ${fmt(mean, 3)}`} className="block">
        <line
          x1={sx(mean)}
          x2={sx(mean)}
          y1={top - 4}
          y2={top + items.length * rowH + 4}
          style={{ stroke: 'var(--ink)' }}
          strokeWidth={2}
          strokeDasharray="5 3"
        />
        {items.map((it, i) => {
          const y = top + i * rowH;
          const slot = it.term.head === 'boolean' ? 1 : 2;
          const label = labels[i];
          const shown =
            textWidth(label, T.category) > labelW - 8
              ? label.slice(0, Math.max(6, Math.floor(((labelW - 8) / textWidth(label, T.category)) * label.length) - 1)) + '…'
              : label;
          return (
            <g key={i}>
              <title>
                {label}: {it.term.head === 'boolean' ? 'BCE' : 'Huber'} {fmt(it.term.loss, 4)} (ŷ {fmtS(it.term.yhat, 3)}, y {fmtS(it.term.y, 3)})
              </title>
              <text x={labelW - 8} y={y + rowH / 2} dy="0.32em" textAnchor="end" style={{ fill: 'var(--ink)', fontSize: T.category }}>
                {shown}
              </text>
              <path d={hBarPath(labelW, sx(it.term.loss), y + 6, rowH - 12)} style={{ fill: catVar(slot) }} />
              <text
                x={sx(it.term.loss) + 6}
                y={y + rowH / 2}
                dy="0.32em"
                className="tabular"
                style={{
                  fill: 'var(--ink-2)',
                  fontSize: T.value,
                  paintOrder: 'stroke',
                  stroke: 'var(--surface)',
                  strokeWidth: 4,
                }}
              >
                {fmt(it.term.loss, 3)}
              </text>
            </g>
          );
        })}
        <text
          x={Math.min(width - 4, sx(mean) + 6)}
          y={top + items.length * rowH + 22}
          textAnchor={sx(mean) + 120 > width ? 'end' : 'start'}
          style={{ fill: 'var(--ink)', fontSize: T.label, fontWeight: 600 }}
        >
          mean = {fmt(mean, 3)}
        </text>
      </svg>
      <Legend
        className="mt-1"
        items={[
          { id: 'b', label: 'BCE (boolean target)', slot: 1, mark: 'rect' },
          { id: 'n', label: 'Huber (number target)', slot: 2, mark: 'rect' },
        ]}
      />
    </div>
  );
}
