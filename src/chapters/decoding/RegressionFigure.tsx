/**
 * Step 3 figure: the number head's output at the target is a z-score ẑ in the target column's own
 * scale; ŷ = ẑ·σ + μ with the Train stats of that column (spec §8.4; pre.rs:569 inverted). One x-axis
 * carries both labelings (z above, original units below) because the map is affine. The histogram is
 * every Train label of the column (all rows of the table for an autocomplete target).
 */
import { scaleLinear } from 'd3';
import { useEffect, useMemo, useState } from 'react';
import { Figure, maxTextWidth, Slider, SVG_TYPE, textWidth } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { seedTableKey } from '@/engine/sampler';
import { Readout } from './charts';
import { LaneHeader } from './LaneHeader';
import { fmt, fmtS, int, rawLabelText, type Lane } from './live';

const T = SVG_TYPE;

/** Raw values of the target column in the table its stats come from. */
export function trainLabels(lane: Pick<Lane, 'pdb' | 'task'>): number[] {
  const { pdb, task } = lane;
  let key: string;
  try {
    key = seedTableKey(pdb, task, 'Train');
  } catch {
    return [];
  }
  const info = pdb.tableInfo[key];
  if (!info) return [];
  const out: number[] = [];
  for (let j = info.nodeIdxOffset; j < info.nodeIdxOffset + info.numNodes; j++) {
    const c = pdb.nodes[j].cells.find((cc) => cc.colName === task.targetCol);
    if (c && typeof c.raw === 'number') out.push(c.raw);
  }
  return out;
}

function fmtUnits(v: number): string {
  const a = Math.abs(v);
  return fmt(v, a >= 100 ? 0 : a >= 10 ? 1 : 2);
}

export function RegressionFigure({ lane }: { lane: Lane }) {
  const { pred, ctx, stats, task } = lane;
  const zHat = pred?.output ?? NaN;
  const tok = ctx.tokens[ctx.targetPos];
  const zLab = tok?.numberValue ?? NaN;
  const rawLab = typeof tok?.raw === 'number' ? tok.raw : NaN;
  const mu = stats?.mean ?? 0;
  const sd = stats?.std ?? 1;
  const yHat = pred?.value ?? zHat * sd + mu;
  const labels = useMemo(() => trainLabels(lane), [lane.pdb, lane.task]); // eslint-disable-line react-hooks/exhaustive-deps
  const zs = useMemo(() => labels.map((v) => (v - mu) / sd), [labels, mu, sd]);
  const [tryZ, setTryZ] = useState(0);
  useEffect(() => {
    if (Number.isFinite(zHat)) setTryZ(Math.round(zHat * 20) / 20);
  }, [zHat]);
  const showGhost = Number.isFinite(zHat) && Math.abs(tryZ - zHat) > 0.03;

  const [ref, size] = useElementSize<HTMLDivElement>();
  const width = Math.max(300, size.width || 600);
  // x-range: the bulk of the labels (up to the 97th percentile) plus the prediction and the label; rare
  // large labels are counted at the right edge instead of squeezing everything else.
  const finite = (v: number) => (Number.isFinite(v) ? [v] : []);
  const sortedZ = useMemo(() => [...zs].sort((a, b) => a - b), [zs]);
  const q97 = sortedZ.length ? sortedZ[Math.min(sortedZ.length - 1, Math.floor(0.97 * (sortedZ.length - 1)))] : 2;
  const lo = Math.min(-1.5, ...(sortedZ.length ? [sortedZ[0]] : []), ...finite(zHat), ...finite(zLab)) - 0.3;
  const hi = Math.max(2, q97, ...finite(zHat), ...finite(zLab)) + 0.3;
  const beyond = zs.filter((z) => z > hi).length;
  const left = 16;
  const right = 16;
  const x = scaleLinear()
    .domain([lo, hi])
    .range([left, width - right]);
  const plotW = width - left - right;

  // histogram in z-space: one bin per integer value when the visible range of an integer column is small, else bins of 0.2 σ
  const hist = useMemo(() => {
    if (!zs.length) return [];
    const visible = labels.filter((v) => (v - mu) / sd <= hi);
    const integer = labels.every((v) => Number.isInteger(v)) && visible.length > 0 && Math.max(...visible) - Math.min(...visible) <= 60;
    const bins = new Map<number, number>();
    if (integer) {
      for (const v of visible) bins.set(v, (bins.get(v) ?? 0) + 1);
      return [...bins.entries()].map(([v, n]) => ({
        z0: (v - 0.5 - mu) / sd,
        z1: (v + 0.5 - mu) / sd,
        n,
        label: `${v}`,
      }));
    }
    const step = 0.2;
    for (const z of zs) {
      if (z > hi) continue;
      const b = Math.floor(z / step);
      bins.set(b, (bins.get(b) ?? 0) + 1);
    }
    return [...bins.entries()].map(([b, n]) => ({
      z0: b * step,
      z1: (b + 1) * step,
      n,
      label: `${fmtUnits(b * step * sd + mu)} to ${fmtUnits((b + 1) * step * sd + mu)}`,
    }));
  }, [zs, labels, mu, sd, hi]);
  const maxN = Math.max(1, ...hist.map((h) => h.n));

  const topAxisY = 30;
  const histTop = 62;
  const histH = 150;
  const baseY = histTop + histH;
  const H = baseY + 64;
  const zTicks = (() => {
    for (let n = Math.max(2, Math.floor(plotW / 60)); n >= 2; n--) {
      const ts = x.ticks(n);
      if (
        ts.length *
          (maxTextWidth(
            ts.map((t) => fmtS(t, 0)),
            T.tick,
          ) +
            22) <=
        plotW
      )
        return ts;
    }
    return x.ticks(2);
  })();
  const rawScale = scaleLinear()
    .domain([lo * sd + mu, hi * sd + mu])
    .range([left, width - right]);
  const rawTicks = (() => {
    for (let n = Math.max(2, Math.floor(plotW / 60)); n >= 2; n--) {
      const ts = rawScale.ticks(n);
      if (
        ts.length *
          (maxTextWidth(
            ts.map((t) => fmt(t, Number.isInteger(t) ? 0 : 1)),
            T.tick,
          ) +
            22) <=
        plotW
      )
        return ts;
    }
    return rawScale.ticks(2);
  })();
  const [hoverBin, setHoverBin] = useState<number | null>(null);

  const marker = (zv: number, label: string, kind: 'live' | 'ink' | 'ghost', row: number) => {
    if (!Number.isFinite(zv)) return null;
    const px = x(zv);
    const color = kind === 'live' ? 'var(--accent)' : kind === 'ink' ? 'var(--ink)' : 'var(--ink-2)';
    const w = textWidth(label, T.label, { weight: 600 });
    const flip = px + 8 + w > width - 2;
    const ly = histTop + 14 + row * 20;
    return (
      <g>
        <line
          x1={px}
          x2={px}
          y1={topAxisY + 6}
          y2={baseY + 6}
          style={{ stroke: color }}
          strokeWidth={kind === 'ghost' ? 1.5 : 2.5}
          strokeDasharray={kind === 'ghost' ? '4 3' : undefined}
        />
        <circle
          cx={px}
          cy={baseY}
          r={kind === 'ghost' ? 5 : 6.5}
          style={{
            fill: kind === 'ghost' ? 'var(--surface)' : color,
            stroke: kind === 'ghost' ? color : 'var(--surface)',
          }}
          strokeWidth={2}
        />
        <text
          x={px + (flip ? -8 : 8)}
          y={ly}
          textAnchor={flip ? 'end' : 'start'}
          style={{
            fill: 'var(--ink)',
            fontSize: T.label,
            fontWeight: 600,
            paintOrder: 'stroke',
            stroke: 'var(--surface)',
            strokeWidth: 4,
          }}
        >
          {label}
        </text>
      </g>
    );
  };

  return (
    <Figure
      wide
      title="From z-score to original units"
      caption={
        <>
          One axis, two labelings: z above, {task.targetCol} below. Because ŷ = ẑ·σ + μ is a straight-line map, every point has both readings. Gray
          bars: the {int(labels.length)} {task.kind === 'forecast' ? 'Train labels' : 'values in the table'} whose μ and σ define the scale. The
          violet line is the number head’s output at the masked cell; the black line is the hidden label.
        </>
      }
    >
      <LaneHeader lane={lane} />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)] lg:gap-8">
        {/* the measured element is always mounted, so its width is known when the data arrives */}
        <div ref={ref} className="min-w-0">
          {!pred || !stats ? (
            <div className="rounded-lg border border-dashed border-border-strong px-4 py-8 text-center text-[16px] text-ink-2">
              Running the forward pass…
            </div>
          ) : (
            <>
              <svg
                width={width}
                height={H}
                role="img"
                aria-label={`Prediction ${fmtUnits(yHat)} and label ${fmtUnits(rawLab)} on the ${task.targetCol} scale`}
                className="block"
              >
                {/* z axis (top) */}
                <text x={left} y={12} style={{ fill: 'var(--ink-2)', fontSize: T.axisTitle }}>
                  z-score (what the number head outputs)
                </text>
                <line x1={left} x2={width - right} y1={topAxisY + 6} y2={topAxisY + 6} style={{ stroke: 'var(--axis)' }} />
                {zTicks.map((t) => (
                  <g key={`z${t}`}>
                    <line x1={x(t)} x2={x(t)} y1={topAxisY + 2} y2={baseY} style={{ stroke: 'var(--grid)' }} />
                    <text x={x(t)} y={topAxisY} textAnchor="middle" className="tabular" style={{ fill: 'var(--muted)', fontSize: T.tick }}>
                      {t === 0 ? '0' : fmtS(t, Number.isInteger(t) ? 0 : 1)}
                    </text>
                  </g>
                ))}
                {/* histogram */}
                {hist.map((h, i) => {
                  const x0 = x(h.z0) + 0.5;
                  const x1 = x(h.z1) - 0.5;
                  const bh = (h.n / maxN) * (histH - 30);
                  return (
                    <rect
                      key={i}
                      x={x0}
                      y={baseY - bh}
                      width={Math.max(1, x1 - x0)}
                      height={bh}
                      rx={1.5}
                      style={{
                        fill: 'var(--muted)',
                        fillOpacity: hoverBin === i ? 0.9 : 0.55,
                      }}
                      onPointerEnter={() => setHoverBin(i)}
                      onPointerLeave={() => setHoverBin(null)}
                    >
                      <title>
                        {h.label}: {h.n} {h.n === 1 ? 'row' : 'rows'}
                      </title>
                    </rect>
                  );
                })}
                {beyond > 0 && (
                  <text
                    x={width - right}
                    y={baseY - 14}
                    textAnchor="end"
                    style={{ fill: 'var(--ink-2)', fontSize: T.annotation, paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 4 }}
                  >
                    {plotW < 460 ? `+${beyond} more →` : `+ ${beyond} larger ${beyond === 1 ? 'label' : 'labels'} off the right edge →`}
                  </text>
                )}
                {/* μ and μ + σ */}
                <line x1={x(0)} x2={x(0)} y1={histTop} y2={baseY} style={{ stroke: 'var(--ink-2)' }} strokeDasharray="2 3" />
                <line x1={x(1)} x2={x(1)} y1={histTop} y2={baseY} style={{ stroke: 'var(--ink-2)' }} strokeDasharray="2 3" />
                <text x={x(0) - 4} y={histTop - 4} textAnchor="end" style={{ fill: 'var(--ink-2)', fontSize: T.annotation }}>
                  μ
                </text>
                <text x={x(1) + 4} y={histTop - 4} style={{ fill: 'var(--ink-2)', fontSize: T.annotation }}>
                  μ + σ
                </text>
                {showGhost && marker(tryZ, `ẑ = ${fmtS(tryZ, 2)} → ${fmtUnits(tryZ * sd + mu)}`, 'ghost', 2)}
                {marker(zLab, `label ${rawLabelText(tok.raw)}`, 'ink', 1)}
                {marker(zHat, `ŷ = ${fmtUnits(yHat)}`, 'live', 0)}
                {/* raw axis (bottom) */}
                <line x1={left} x2={width - right} y1={baseY + 6} y2={baseY + 6} style={{ stroke: 'var(--axis)' }} />
                {rawTicks.map((t) => (
                  <text
                    key={`r${t}`}
                    x={rawScale(t)}
                    y={baseY + 26}
                    textAnchor="middle"
                    className="tabular"
                    style={{ fill: 'var(--muted)', fontSize: T.tick }}
                  >
                    {fmt(t, Number.isInteger(t) ? 0 : 1)}
                  </text>
                ))}
                <text x={left} y={baseY + 52} style={{ fill: 'var(--ink-2)', fontSize: T.axisTitle }}>
                  {task.targetCol} in original units (ẑ·σ + μ)
                </text>
              </svg>
              <Slider
                label="Try another number-head output"
                value={tryZ}
                min={Math.floor(lo * 4) / 4}
                max={Math.ceil(hi * 4) / 4}
                step={0.05}
                onChange={setTryZ}
                format={(v) => (
                  <>
                    ẑ = {fmtS(v, 2)} <span className="text-ink-2">→ {fmtUnits(v * sd + mu)}</span>
                  </>
                )}
                className="mt-2 w-full max-w-[420px]"
              />
            </>
          )}
        </div>
        {pred && stats && (
          <div className="min-w-0">
            <div className="rounded-lg border border-border bg-surface px-3 py-1">
              <Readout label="Number head output at the target (a z-score)" value={fmtS(zHat, 4)} strong />
              <Readout
                label={
                  <>
                    {task.kind === 'forecast' ? 'Train stats' : 'Stats'} of <span className="font-mono">{task.targetCol}</span>
                  </>
                }
                value={`μ = ${fmt(mu, 3)}, σ = ${fmt(sd, 3)}`}
                note={
                  task.kind === 'forecast'
                    ? 'Train split of the task table, sample std. Val and Test task rows are z-scored with these same two numbers.'
                    : 'All rows of the table, sample std.'
                }
              />
              <Readout
                label="Prediction in original units"
                value={fmtUnits(yHat)}
                note={
                  <span className="num">
                    {fmtS(zHat, 3)} × {fmt(sd, 3)} + {fmt(mu, 3)} = {fmt(yHat, 3)}
                  </span>
                }
                strong
              />
              <Readout
                label="Hidden label (raw, and as stored)"
                value={`${rawLabelText(tok.raw)}  ·  z = ${fmtS(zLab, 3)}`}
                note="The stored z-score is what the loss compares with; it never enters the input."
              />
              <Readout
                label="Error, both scales"
                value={`${fmtS(yHat - rawLab, 2)} units = ${fmtS(zHat - zLab, 3)} σ`}
                note="Dividing the error in units by σ gives the error in z-space, the d that the Huber loss sees."
              />
            </div>
          </div>
        )}
      </div>
    </Figure>
  );
}
