/**
 * Small SVG chart primitives for the decoding chapter, following the shared chart conventions
 * (src/components/viz/LineChart.tsx): hairline grid, muted tick labels at SVG_TYPE.tick, axis titles
 * in ink-2, 2px lines, ≥ 8px markers with a 2px surface ring, one y-axis, a hover readout.
 * Text never drops below SVG_TYPE.min (12px); ticks are thinned to fit instead.
 */
import { scaleLinear, type ScaleLinear } from 'd3';
import { useMemo, useState, type ReactNode } from 'react';
import { maxTextWidth, SVG_TYPE, textWidth } from '@/components';
import { useElementSize } from '@/components/viz/useSize';

const T = SVG_TYPE;

export interface CurveSeries {
  id: string;
  label: string;
  /** CSS colour (a token var). */
  color: string;
  fn: (x: number) => number;
  dash?: string;
  width?: number;
  /** Only draw where this returns true (e.g. one branch of a piecewise function). */
  where?: (x: number) => boolean;
}

export interface CurveMark {
  x: number;
  y: number;
  /** 'live' = filled accent marker; 'ghost' = hollow; 'ink' = filled ink. */
  kind: 'live' | 'ghost' | 'ink';
  label?: string;
  /** Draw dashed guides to both axes. */
  guides?: boolean;
  /** Put the label left of the marker. */
  labelLeft?: boolean;
}

export interface CurveChartProps {
  title: string;
  series: CurveSeries[];
  xDomain: [number, number];
  yDomain: [number, number];
  xLabel?: string;
  yLabel?: string;
  height?: number;
  marks?: CurveMark[];
  /** Vertical hairlines with a label. */
  vlines?: { x: number; label?: string }[];
  /** Horizontal hairlines with a label. */
  hlines?: { y: number; label?: string }[];
  /** Shaded x-intervals (e.g. the quadratic zone of the Huber loss). */
  bands?: { x0: number; x1: number; label?: string }[];
  formatX?: (v: number) => string;
  formatY?: (v: number) => string;
  /** Hover readout below the chart for an x position. */
  readout?: (x: number) => ReactNode;
  samples?: number;
  className?: string;
}

function niceTicks(s: ScaleLinear<number, number>, px: number, fmt: (v: number) => string, minGap = 22): number[] {
  for (let n = Math.max(2, Math.floor(px / 70)); n >= 2; n--) {
    const ts = s.ticks(n);
    if (ts.length * (maxTextWidth(ts.map(fmt), T.tick) + minGap) <= px + minGap) return ts;
  }
  return s.ticks(2);
}

export function CurveChart({
  title,
  series,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  height = 280,
  marks = [],
  vlines = [],
  hlines = [],
  bands = [],
  formatX = (v) => String(+v.toFixed(2)).replace('-', '−'),
  formatY = (v) => String(+v.toFixed(2)).replace('-', '−'),
  readout,
  samples = 241,
  className,
}: CurveChartProps) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const width = Math.max(280, size.width || 560);
  const y = scaleLinear().domain(yDomain);
  const top = 14;
  const bottom = xLabel ? 52 : 30;
  const plotH = height - top - bottom;
  y.range([plotH, 0]);
  const yTicks = y.ticks(5);
  const left = Math.ceil(maxTextWidth(yTicks.map(formatY), T.tick) + 12 + (yLabel ? T.axisTitle + 12 : 0));
  const right = 14;
  const plotW = width - left - right;
  const x = scaleLinear().domain(xDomain).range([0, plotW]);
  const xTicks = niceTicks(x, plotW, formatX);
  const [hx, setHx] = useState<number | null>(null);

  const paths = useMemo(
    () =>
      series.map((s) => {
        let d = '';
        let pen = false;
        for (let i = 0; i < samples; i++) {
          const xv = xDomain[0] + ((xDomain[1] - xDomain[0]) * i) / (samples - 1);
          const ok = !s.where || s.where(xv);
          const yv = s.fn(xv);
          if (!ok || !Number.isFinite(yv)) {
            pen = false;
            continue;
          }
          const yc = Math.max(yDomain[0] - (yDomain[1] - yDomain[0]) * 0.05, Math.min(yDomain[1] + (yDomain[1] - yDomain[0]) * 0.05, yv));
          d += `${pen ? 'L' : 'M'}${x(xv).toFixed(1)},${y(yc).toFixed(1)}`;
          pen = true;
        }
        return d;
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [series, xDomain[0], xDomain[1], yDomain[0], yDomain[1], plotW, plotH, samples],
  );

  const clipId = `clip-${title.replace(/[^a-z0-9]/gi, '')}`;

  return (
    <div className={className}>
      <div ref={ref} className="relative w-full">
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={title}
          className="block touch-none select-none"
          onPointerMove={(e) => {
            const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            const px = e.clientX - r.left - left;
            if (px < 0 || px > plotW) return setHx(null);
            setHx(x.invert(px));
          }}
          onPointerLeave={() => setHx(null)}
        >
          <defs>
            <clipPath id={clipId}>
              <rect x={0} y={-4} width={plotW} height={plotH + 8} />
            </clipPath>
          </defs>
          <g transform={`translate(${left},${top})`}>
            {bands.map((b, i) => (
              <g key={i}>
                <rect x={x(b.x0)} y={0} width={Math.max(0, x(b.x1) - x(b.x0))} height={plotH} style={{ fill: 'var(--accent-soft)' }} />
                {b.label && (
                  <text x={(x(b.x0) + x(b.x1)) / 2} y={14} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: T.annotation }}>
                    {b.label}
                  </text>
                )}
              </g>
            ))}
            {yTicks.map((t) => (
              <g key={t} transform={`translate(0,${y(t)})`}>
                <line x1={0} x2={plotW} style={{ stroke: 'var(--grid)' }} />
                <text x={-8} dy="0.32em" textAnchor="end" className="tabular" style={{ fill: 'var(--muted)', fontSize: T.tick }}>
                  {formatY(t)}
                </text>
              </g>
            ))}
            <line x1={0} x2={plotW} y1={plotH} y2={plotH} style={{ stroke: 'var(--axis)' }} />
            {xTicks.map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={0} y2={plotH} style={{ stroke: 'var(--grid)' }} opacity={0.6} />
                <text x={x(t)} y={plotH + 20} textAnchor="middle" className="tabular" style={{ fill: 'var(--muted)', fontSize: T.tick }}>
                  {formatX(t)}
                </text>
              </g>
            ))}
            {xLabel && (
              <text x={plotW / 2} y={plotH + 44} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: T.axisTitle }}>
                {xLabel}
              </text>
            )}
            {yLabel && (
              <text
                transform={`translate(${-(left - T.axisTitle + 2)},${plotH / 2}) rotate(-90)`}
                textAnchor="middle"
                style={{ fill: 'var(--ink-2)', fontSize: T.axisTitle }}
              >
                {yLabel}
              </text>
            )}
            {hlines.map((h, i) => (
              <g key={`h${i}`}>
                <line x1={0} x2={plotW} y1={y(h.y)} y2={y(h.y)} style={{ stroke: 'var(--axis)' }} strokeDasharray="4 3" />
                {h.label && (
                  <text x={plotW - 4} y={y(h.y) - 6} textAnchor="end" style={{ fill: 'var(--ink-2)', fontSize: T.annotation }}>
                    {h.label}
                  </text>
                )}
              </g>
            ))}
            {vlines.map((v, i) => {
              // label right of the line, else left of it, else clamped inside the plot (never cut off)
              const vw = v.label ? textWidth(v.label, T.annotation) : 0;
              const vx = x(v.x) + 6 + vw <= plotW ? x(v.x) + 6 : x(v.x) - 6 - vw >= 0 ? x(v.x) - 6 - vw : Math.max(0, plotW - vw);
              return (
                <g key={`v${i}`}>
                  <line x1={x(v.x)} x2={x(v.x)} y1={0} y2={plotH} style={{ stroke: 'var(--axis)' }} strokeDasharray="4 3" />
                  {v.label && (
                    <text
                      x={vx}
                      y={14}
                      textAnchor="start"
                      style={{
                        fill: 'var(--ink-2)',
                        fontSize: T.annotation,
                        paintOrder: 'stroke',
                        stroke: 'var(--surface)',
                        strokeWidth: 4,
                      }}
                    >
                      {v.label}
                    </text>
                  )}
                </g>
              );
            })}
            <g clipPath={`url(#${clipId})`}>
              {series.map((s, i) => (
                <path
                  key={s.id}
                  d={paths[i]}
                  fill="none"
                  style={{ stroke: s.color }}
                  strokeWidth={s.width ?? 2}
                  strokeDasharray={s.dash}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              ))}
            </g>
            {hx !== null && <line x1={x(hx)} x2={x(hx)} y1={0} y2={plotH} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1} opacity={0.5} />}
            {marks.map((m, i) => {
              if (!Number.isFinite(m.x) || !Number.isFinite(m.y)) return null;
              const mx = Math.max(0, Math.min(plotW, x(m.x)));
              const my = Math.max(0, Math.min(plotH, y(m.y)));
              const fill = m.kind === 'live' ? 'var(--accent)' : m.kind === 'ink' ? 'var(--ink)' : 'var(--surface)';
              const stroke = m.kind === 'ghost' ? 'var(--ink-2)' : 'var(--surface)';
              const lw = m.label ? textWidth(m.label, T.label, { weight: 600 }) : 0;
              // right of the marker, else left of it, else clamped inside the plot (never over the y-axis)
              const side = m.labelLeft !== true && mx + 12 + lw <= plotW ? 'right' : mx - 12 - lw >= 0 ? 'left' : 'above';
              const lx = side === 'right' ? mx + 12 : side === 'left' ? mx - 12 - lw : Math.max(0, Math.min(plotW - lw, mx - lw / 2));
              const ly = side === 'above' ? my - 16 : my - 10;
              return (
                <g key={i}>
                  {m.guides && (
                    <g
                      style={{
                        stroke: m.kind === 'live' ? 'var(--accent)' : 'var(--ink-2)',
                      }}
                      strokeDasharray="3 3"
                      strokeWidth={1.25}
                      opacity={0.8}
                    >
                      <line x1={mx} x2={mx} y1={my} y2={plotH} />
                      <line x1={0} x2={mx} y1={my} y2={my} />
                    </g>
                  )}
                  <circle cx={mx} cy={my} r={m.kind === 'ghost' ? 6 : 7} style={{ fill, stroke }} strokeWidth={m.kind === 'ghost' ? 2 : 2.5} />
                  {m.label && (
                    <text
                      x={lx}
                      y={Math.max(12, ly)}
                      textAnchor="start"
                      style={{
                        fill: 'var(--ink)',
                        fontSize: T.label,
                        fontWeight: 600,
                        paintOrder: 'stroke',
                        stroke: 'var(--surface)',
                        strokeWidth: 4,
                      }}
                    >
                      {m.label}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        </svg>
      </div>
      {readout && (
        <div className="mt-1 min-h-[24px] text-[15px] text-ink-2" aria-live="polite">
          {hx !== null ? readout(hx) : <span className="text-muted">Hover the chart to read values.</span>}
        </div>
      )}
    </div>
  );
}

/** A labelled readout row: "label · value" with an optional note. */
export function Readout({ label, value, note, strong }: { label: ReactNode; value: ReactNode; note?: ReactNode; strong?: boolean }) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 border-b border-grid/70 py-2 last:border-0">
      <span className="text-[15.5px] text-ink-2">{label}</span>
      <span className={strong ? 'num text-[19px] font-semibold text-ink' : 'num text-[17px] text-ink'}>{value}</span>
      {note && <span className="w-full text-[15px] leading-snug text-ink-2">{note}</span>}
    </div>
  );
}

/** "Live" or "local preview" chip for a lane, with its checkpoint. */
export function LaneChip({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface-2 px-2.5 py-0.5 text-[14px] text-ink-2">
      {children}
    </span>
  );
}
