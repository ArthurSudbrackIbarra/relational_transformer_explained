/**
 * LineChart: simple SVG multi-series line chart (dataviz rules): 2px lines,
 * ≥ 8px end markers with a 2px surface ring, hairline solid grid, one y-axis,
 * crosshair that snaps to the nearest x with ONE tooltip listing every series,
 * a legend for ≥ 2 series, selective direct end labels (≤ 4 series, only when
 * they do not collide), and a table-view twin.
 */
import { scaleLinear, scaleLog } from 'd3';
import { useMemo, useState, type ReactNode } from 'react';
import { catVar } from '@/styles/palette';
import { cx } from '../ui/cx';
import { ChartFrame, DataTable, Legend, TooltipBox } from './chartParts';
import { formatNumber } from './format';
import { maxTextWidth, SVG_TYPE, textWidth } from './text';
import { useElementSize } from './useSize';

export interface LinePoint {
  x: number;
  y: number;
}

export interface LineSeries {
  id: string;
  label: string;
  /** Categorical slot (default: series order 1, 2, 3, … never cycled past 8). */
  slot?: number;
  points: LinePoint[];
}

export interface LineChartProps {
  series: LineSeries[];
  /** Accessible title (also the table caption). */
  title: string;
  subtitle?: ReactNode;
  /** Show the title row (default true). */
  showTitle?: boolean;
  xLabel?: string;
  yLabel?: string;
  xScale?: 'linear' | 'log';
  xDomain?: [number, number];
  yDomain?: [number, number];
  formatX?: (x: number) => string;
  formatY?: (y: number) => string;
  /** Total height including axes (px). Default 280. */
  height?: number;
  /** Soft 10% area fill (single series only). */
  area?: boolean;
  /** Vertical annotations (hairline + label). */
  markers?: { x: number; label: string }[];
  /** Horizontal reference lines (e.g. a baseline). */
  refLines?: { y: number; label: string }[];
  /** Emphasise one series; others turn gray (dataviz "emphasis" form). */
  emphasis?: string;
  className?: string;
}

const T = SVG_TYPE;
const M0 = { top: 16, right: 16 };

export function LineChart({
  series,
  title,
  subtitle,
  showTitle = true,
  xLabel,
  yLabel,
  xScale = 'linear',
  xDomain,
  yDomain,
  formatX = (x) => formatNumber(x, 2),
  formatY = (y) => formatNumber(y, 3),
  height = 280,
  area = false,
  markers,
  refLines,
  emphasis,
  className,
}: LineChartProps) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const width = Math.max(260, size.width || 640);
  const slotOf = (s: LineSeries, i: number) => s.slot ?? Math.min(8, i + 1);

  const all = useMemo(() => series.flatMap((s) => s.points), [series]);
  const xs = useMemo(() => [...new Set(all.map((p) => p.x))].sort((a, b) => a - b), [all]);

  const bottom = xLabel ? 50 : 30;
  const plotH = height - M0.top - bottom;

  const y = useMemo(() => {
    let dom = yDomain;
    if (!dom) {
      const ys = all.map((p) => p.y).concat(refLines?.map((r) => r.y) ?? []);
      const lo = Math.min(...ys);
      const hi = Math.max(...ys);
      const pad = (hi - lo || 1) * 0.08;
      // never invent negative space below a non-negative series
      dom = [lo >= 0 && lo - pad < 0 ? 0 : lo - pad, hi + pad];
    }
    return scaleLinear().domain(dom).nice(5).range([plotH, 0]);
  }, [all, yDomain, plotH, refLines]);

  const [hoverX, setHoverX] = useState<number | null>(null);

  // Left margin fits the widest y tick label (measured) + the rotated axis title.
  const yTickW = maxTextWidth(y.ticks(5).map((t) => formatY(t)), T.tick);
  const M = { ...M0, bottom, left: Math.ceil(yTickW + 10 + (yLabel ? T.axisTitle + 14 : 4)) };
  const plotW = width - M.left - M.right;

  const path = (pts: LinePoint[]) =>
    pts
      .filter((p) => Number.isFinite(p.y) && (xScale !== 'log' || p.x > 0))
      .map((p, i) => `${i ? 'L' : 'M'}${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`)
      .join('');

  const yTicks = y.ticks(5);

  // Direct end labels: ≤ 4 series and no collisions.
  const endLabels = useMemo(() => {
    if (series.length < 2 || series.length > 4) return [];
    const ends = series
      .map((s, i) => {
        const p = s.points[s.points.length - 1];
        return p ? { id: s.id, label: s.label, slot: slotOf(s, i), yPx: y(p.y) } : null;
      })
      .filter((e): e is NonNullable<typeof e> => !!e)
      .sort((a, b) => a.yPx - b.yPx);
    for (let i = 1; i < ends.length; i++) if (ends[i].yPx - ends[i - 1].yPx < T.value + 4) return [];
    // only when the labels fit in at most 30% of the width (the legend always names the series)
    const w = Math.max(...ends.map((e) => textWidth(e.label, T.value)));
    if (w + 12 > plotW * 0.3) return [];
    return ends;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series, y, plotW]);
  const rightPad = endLabels.length ? Math.ceil(Math.max(...endLabels.map((e) => textWidth(e.label, T.value)))) + 12 : 0;
  const innerW = Math.max(60, plotW - rightPad);

  const x = useMemo(() => {
    const dom = xDomain ?? [Math.min(...xs), Math.max(...xs)];
    const lo = xScale === 'log' ? Math.max(dom[0], 1e-9) : dom[0];
    const s = xScale === 'log' ? scaleLog().domain([lo, dom[1]]) : scaleLinear().domain([lo, dom[1] === lo ? lo + 1 : dom[1]]);
    return s.range([0, innerW]);
  }, [xs, xDomain, xScale, innerW]);
  const xr = (v: number) => x(v);
  const xTicks = useMemo(() => {
    // keep ≥ 20px between tick labels (measured), never shrink them
    const fits = (ts: number[]) => ts.length * (maxTextWidth(ts.map((t) => formatX(t)), T.tick) + 20) <= innerW + 20;
    if (xScale !== 'log') {
      const lin = x as ReturnType<typeof scaleLinear>;
      for (let n = Math.max(2, Math.floor(innerW / 85)); n >= 2; n--) {
        const ts = lin.ticks(n);
        if (fits(ts)) return ts;
      }
      return lin.ticks(2);
    }
    // log axis: powers of ten, plus 2× and 5× steps only when there is room
    const all = (x as ReturnType<typeof scaleLog>).ticks();
    const lead = (t: number) => Math.round(t / 10 ** Math.floor(Math.log10(t) + 1e-9));
    const pow10 = all.filter((t) => lead(t) === 1);
    const with25 = all.filter((t) => [1, 2, 5].includes(lead(t)));
    if (fits(with25)) return with25;
    if (pow10.length >= 2 && fits(pow10)) return pow10;
    return pow10.length >= 2 ? pow10.filter((_, i) => i % 2 === 0) : all.slice(0, 4);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [x, xScale, innerW]);

  const nearest = (s: LineSeries, xv: number) => {
    let best: LinePoint | null = null;
    for (const p of s.points) if (!best || Math.abs(p.x - xv) < Math.abs(best.x - xv)) best = p;
    return best;
  };

  const legendItems = series.map((s, i) => ({ id: s.id, label: s.label, slot: slotOf(s, i), mark: 'line' as const, muted: !!emphasis && emphasis !== s.id }));

  const table = (
    <DataTable
      caption={title}
      columns={[xLabel ?? 'x', ...series.map((s) => s.label)]}
      rows={xs.map((xv) => [formatX(xv), ...series.map((s) => {
        const p = s.points.find((q) => q.x === xv);
        return p ? formatY(p.y) : '—';
      })])}
    />
  );


  return (
    <ChartFrame
      title={showTitle ? title : undefined}
      subtitle={subtitle}
      legend={<Legend items={legendItems} />}
      table={table}
      className={className}
    >
      <div ref={ref} className="relative w-full">
        <svg width={width} height={height} role="img" aria-label={title} className="block overflow-visible">
          <g transform={`translate(${M.left},${M.top})`}>
            {/* grid + y axis */}
            {yTicks.map((t) => (
              <g key={t} transform={`translate(0,${y(t)})`}>
                <line x1={0} x2={innerW} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
                <text x={-8} dy="0.32em" textAnchor="end" className="tabular" style={{ fill: 'var(--muted)', fontSize: T.tick }}>
                  {formatY(t)}
                </text>
              </g>
            ))}
            <line x1={0} x2={innerW} y1={plotH} y2={plotH} style={{ stroke: 'var(--axis)' }} strokeWidth={1} />
            {xTicks.map((t) => (
              <text key={t} x={xr(t)} y={plotH + 20} textAnchor="middle" className="tabular" style={{ fill: 'var(--muted)', fontSize: T.tick }}>
                {formatX(t)}
              </text>
            ))}
            {xLabel && (
              <text x={innerW / 2} y={plotH + 42} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: T.axisTitle }}>
                {xLabel}
              </text>
            )}
            {yLabel && (
              <text transform={`translate(${-(M.left - T.axisTitle)},${plotH / 2}) rotate(-90)`} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: T.axisTitle }}>
                {yLabel}
              </text>
            )}
            {refLines?.map((r) => (
              <g key={r.label}>
                <line x1={0} x2={innerW} y1={y(r.y)} y2={y(r.y)} style={{ stroke: 'var(--axis)' }} strokeWidth={1} />
                <text x={innerW - 4} y={y(r.y) - 6} textAnchor="end" style={{ fill: 'var(--ink-2)', fontSize: T.annotation }}>
                  {r.label}
                </text>
              </g>
            ))}
            {markers?.map((m) => {
              // label right of the hairline, or left of it when it would run off the chart
              const flip = xr(m.x) + 5 + textWidth(m.label, T.annotation) > innerW + M.right - 2;
              return (
                <g key={m.label} transform={`translate(${xr(m.x)},0)`}>
                  <line y1={0} y2={plotH} style={{ stroke: 'var(--axis)' }} strokeWidth={1} />
                  <text x={flip ? -5 : 5} y={2} dy="0.71em" textAnchor={flip ? 'end' : 'start'} style={{ fill: 'var(--ink-2)', fontSize: T.annotation }}>
                    {m.label}
                  </text>
                </g>
              );
            })}
            {/* series */}
            {series.map((s, i) => {
              const slot = slotOf(s, i);
              const muted = !!emphasis && emphasis !== s.id;
              const stroke = muted ? 'var(--axis)' : catVar(slot);
              const d = path(s.points);
              const last = s.points[s.points.length - 1];
              return (
                <g key={s.id}>
                  {area && series.length === 1 && s.points.length > 1 && (
                    <path d={`${d}L${xr(last.x)},${plotH}L${xr(s.points[0].x)},${plotH}Z`} style={{ fill: stroke, opacity: 0.1 }} />
                  )}
                  <path d={d} fill="none" style={{ stroke }} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
                  {last && Number.isFinite(last.y) && (
                    <circle cx={xr(last.x)} cy={y(last.y)} r={4} style={{ fill: stroke, stroke: 'var(--surface)' }} strokeWidth={2} />
                  )}
                </g>
              );
            })}
            {endLabels.map((e) => (
              <text key={e.id} x={innerW + 8} y={e.yPx} dy="0.32em" style={{ fill: 'var(--ink-2)', fontSize: T.value }}>
                {e.label}
              </text>
            ))}
            {/* crosshair */}
            {hoverX !== null && (
              <g>
                <line x1={xr(hoverX)} x2={xr(hoverX)} y1={0} y2={plotH} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1} />
                {series.map((s, i) => {
                  const p = nearest(s, hoverX);
                  if (!p || !Number.isFinite(p.y)) return null;
                  const muted = !!emphasis && emphasis !== s.id;
                  return (
                    <circle
                      key={s.id}
                      cx={xr(p.x)}
                      cy={y(p.y)}
                      r={4}
                      style={{ fill: muted ? 'var(--axis)' : catVar(slotOf(s, i)), stroke: 'var(--surface)' }}
                      strokeWidth={2}
                    />
                  );
                })}
              </g>
            )}
            <rect
              x={0}
              y={0}
              width={innerW}
              height={plotH}
              fill="transparent"
              className="cursor-crosshair"
              onPointerMove={(e) => {
                const rect = (e.currentTarget as SVGRectElement).getBoundingClientRect();
                const xv = x.invert(e.clientX - rect.left);
                let best = xs[0];
                for (const v of xs) if (Math.abs(x(v) - x(xv)) < Math.abs(x(best) - x(xv))) best = v;
                setHoverX(best ?? null);
              }}
              onPointerLeave={() => setHoverX(null)}
            />
          </g>
        </svg>
        {hoverX !== null && (
          <TooltipBox
            style={{
              left: Math.max(0, Math.min(width - 220, M.left + xr(hoverX) + 12)),
              top: M.top + 4,
            }}
          >
            <div className="mb-1 text-[14px] text-muted">
              {xLabel ?? 'x'} {formatX(hoverX)}
            </div>
            {series.map((s, i) => {
              const p = nearest(s, hoverX);
              return (
                <div key={s.id} className={cx('flex items-center gap-2 py-px', emphasis && emphasis !== s.id && 'opacity-60')}>
                  <span aria-hidden="true" className="inline-block h-[2px] w-3 rounded-full" style={{ background: catVar(slotOf(s, i)) }} />
                  <strong className="num font-semibold text-ink">{p ? formatY(p.y) : '—'}</strong>
                  <span className="truncate text-ink-2">{s.label}</span>
                </div>
              );
            })}
          </TooltipBox>
        )}
      </div>
    </ChartFrame>
  );
}
