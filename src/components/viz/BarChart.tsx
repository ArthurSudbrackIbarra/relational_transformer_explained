/**
 * BarChart: horizontal (default) or vertical bars, single or grouped series
 * (dataviz rules): bars ≤ 24px thick with a 4px rounded data-end and a square
 * baseline end, a 2px surface gap between touching bars, value at the tip,
 * negative values supported (bars grow from one baseline), per-bar hover
 * tooltip (the whole band is the hit target), legend for ≥ 2 series, a
 * table-view twin. Single series → one color (slot 1); `emphasis` grays the rest.
 */
import { scaleLinear } from 'd3';
import { useMemo, useState, type ReactNode } from 'react';
import { catVar } from '@/styles/palette';
import { ChartFrame, DataTable, hBarPath, Legend, TooltipBox, vBarPath } from './chartParts';
import { formatNumber } from './format';
import { fitText, maxTextWidth, SVG_TYPE as T } from './text';
import { useElementSize } from './useSize';

export interface BarSeries {
  id: string;
  label: string;
  slot?: number;
  /** One value per category (null = no value). */
  values: (number | null)[];
}

export interface BarChartProps {
  categories: string[];
  series: BarSeries[];
  title: string;
  subtitle?: ReactNode;
  showTitle?: boolean;
  orientation?: 'horizontal' | 'vertical';
  format?: (v: number) => string;
  /** Value-axis domain (default: data extent including 0). */
  domain?: [number, number];
  /** Value-axis title. */
  valueLabel?: string;
  /** Categories to emphasise (single series): the rest turn gray. */
  emphasis?: string[];
  /** Reference lines on the value axis (e.g. "random = 50"). */
  refLines?: { value: number; label: string }[];
  /** Minimum width for category labels (horizontal). They grow to fit the measured labels, up to 40% of the chart; longer ones are truncated with a tooltip. */
  labelWidth?: number;
  /** Plot height for vertical charts (px). Default 240. */
  height?: number;
  className?: string;
}

export function BarChart({
  categories,
  series,
  title,
  subtitle,
  showTitle = true,
  orientation = 'horizontal',
  format = (v) => formatNumber(v, 1),
  domain,
  valueLabel,
  emphasis,
  refLines,
  labelWidth = 0,
  height = 240,
  className,
}: BarChartProps) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const width = Math.max(260, size.width || 640);
  const nS = series.length;
  const slotOf = (s: BarSeries, i: number) => s.slot ?? Math.min(8, i + 1);
  const emph = emphasis ? new Set(emphasis) : null;
  const [hover, setHover] = useState<{ c: number; s: number } | null>(null);

  const values = series.flatMap((s) => s.values.filter((v): v is number => v !== null && Number.isFinite(v)));
  const dom = useMemo<[number, number]>(() => {
    if (domain) return domain;
    const lo = Math.min(0, ...values, ...(refLines?.map((r) => r.value) ?? []));
    const hi = Math.max(0, ...values, ...(refLines?.map((r) => r.value) ?? []));
    return [lo, hi === lo ? lo + 1 : hi];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [domain, JSON.stringify(values), refLines]);

  const legend = <Legend items={series.map((s, i) => ({ id: s.id, label: s.label, slot: slotOf(s, i), mark: 'rect' as const }))} />;
  const table = (
    <DataTable
      caption={title}
      columns={['', ...series.map((s) => s.label)]}
      rows={categories.map((c, ci) => [c, ...series.map((s) => (s.values[ci] === null ? '—' : format(s.values[ci] as number)))])}
    />
  );

  const fillOf = (s: BarSeries, si: number, c: string) => (emph && !emph.has(c) ? 'var(--axis)' : catVar(nS === 1 ? s.slot ?? 1 : slotOf(s, si)));

  const tooltip =
    hover !== null ? (
      <>
        <div className="mb-1 text-[14px] text-muted">{categories[hover.c]}</div>
        {series.map((s, si) => (
          <div key={s.id} className="flex items-center gap-2 py-px" style={{ opacity: nS > 1 && si !== hover.s ? 0.6 : 1 }}>
            {nS > 1 && <span aria-hidden="true" className="inline-block h-2 w-2 rounded-[2px]" style={{ background: catVar(slotOf(s, si)) }} />}
            <strong className="num font-semibold text-ink">{s.values[hover.c] === null ? '—' : format(s.values[hover.c] as number)}</strong>
            {nS > 1 && <span className="text-ink-2">{s.label}</span>}
          </div>
        ))}
      </>
    ) : null;

  if (orientation === 'vertical') {
    const y = scaleLinear().domain(dom).nice(5).range([height, 0]);
    const yTicks = y.ticks(5);
    const tickW = maxTextWidth(yTicks.map((t) => format(t)), T.tick);
    const M = { top: 22, right: 8, bottom: 34, left: Math.ceil(tickW + 10 + (valueLabel ? T.axisTitle + 14 : 4)) };
    const plotW = width - M.left - M.right;
    const plotH = height;
    const band = plotW / Math.max(1, categories.length);
    const barW = Math.min(24, (band * 0.7) / nS);
    const y0 = y(0);
    // category labels: truncated to the band (full name in the tooltip and the table view)
    const catLabels = categories.map((c) => fitText(c, Math.max(24, band - 8), T.category - 1));
    return (
      <ChartFrame title={showTitle ? title : undefined} subtitle={subtitle} legend={legend} table={table} className={className}>
        <div ref={ref} className="relative w-full">
          <svg width={width} height={plotH + M.top + M.bottom} role="img" aria-label={title} className="block">
            <g transform={`translate(${M.left},${M.top})`}>
              {yTicks.map((t) => (
                <g key={t} transform={`translate(0,${y(t)})`}>
                  <line x2={plotW} style={{ stroke: t === 0 ? 'var(--axis)' : 'var(--grid)' }} />
                  <text x={-8} dy="0.32em" textAnchor="end" className="tabular" style={{ fill: 'var(--muted)', fontSize: T.tick }}>
                    {format(t)}
                  </text>
                </g>
              ))}
              {refLines?.map((r) => (
                <g key={r.label}>
                  <line x2={plotW} y1={y(r.value)} y2={y(r.value)} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1} />
                  <text x={plotW} y={y(r.value) - 5} textAnchor="end" style={{ fill: 'var(--ink-2)', fontSize: T.annotation }}>
                    {r.label}
                  </text>
                </g>
              ))}
              {categories.map((c, ci) => {
                const gx = ci * band + (band - barW * nS - 2 * (nS - 1)) / 2;
                return (
                  <g key={c} onPointerEnter={() => setHover({ c: ci, s: 0 })} onPointerLeave={() => setHover(null)}>
                    <rect x={ci * band} y={0} width={band} height={plotH} fill="transparent" />
                    {series.map((s, si) => {
                      const v = s.values[ci];
                      if (v === null || !Number.isFinite(v)) return null;
                      const x = gx + si * (barW + 2);
                      return (
                        <g key={s.id} onPointerEnter={() => setHover({ c: ci, s: si })}>
                          <path d={vBarPath(x, barW, y0, y(v))} style={{ fill: fillOf(s, si, c), opacity: hover && hover.c === ci ? 0.85 : 1 }} />
                          {nS === 1 && (
                            <text x={x + barW / 2} y={v >= 0 ? y(v) - 6 : y(v) + 6} dy={v >= 0 ? 0 : '0.71em'} textAnchor="middle" className="tabular" style={{ fill: 'var(--ink-2)', fontSize: T.value }}>
                              {format(v)}
                            </text>
                          )}
                        </g>
                      );
                    })}
                    <text x={ci * band + band / 2} y={plotH + 22} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: T.category - 1 }}>
                      {catLabels[ci]}
                      {catLabels[ci] !== c && <title>{c}</title>}
                    </text>
                  </g>
                );
              })}
              {valueLabel && (
                <text transform={`translate(${-(M.left - T.axisTitle)},${plotH / 2}) rotate(-90)`} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: T.axisTitle }}>
                  {valueLabel}
                </text>
              )}
            </g>
          </svg>
          {hover && (
            <TooltipBox style={{ left: Math.max(0, Math.min(width - 220, M.left + hover.c * band + band / 2 + 10)), top: 4 }}>{tooltip}</TooltipBox>
          )}
        </div>
      </ChartFrame>
    );
  }

  // horizontal
  const barH = nS === 1 ? 20 : Math.min(18, Math.max(10, 40 / nS));
  const bandH = Math.max(nS * barH + (nS - 1) * 2 + 14, T.category + 16);
  // room for the value printed at the bar tip (measured, both signs)
  const valuePad = Math.ceil(maxTextWidth(values.map((v) => format(v)), T.value)) + 12;
  // category labels: as wide as the longest one needs (at least `labelWidth`), at most 40% of the chart
  const lw = Math.ceil(Math.min(width * 0.4, Math.max(labelWidth, maxTextWidth(categories, T.category) + 14)));
  const hasNeg = dom[0] < 0;
  const M = { top: refLines?.length ? 26 : 6, right: valuePad, bottom: valueLabel ? 50 : 30, left: lw + (hasNeg ? valuePad : 0) };
  const plotW = Math.max(80, width - M.left - M.right);
  const plotH = categories.length * bandH;
  const x = scaleLinear().domain(dom).nice(5).range([0, plotW]);
  const x0 = x(0);
  const xTicks = (() => {
    for (let n = Math.max(2, Math.floor(plotW / 85)); n >= 2; n--) {
      const ts = x.ticks(n);
      if (ts.length * (maxTextWidth(ts.map((t) => format(t)), T.tick) + 18) <= plotW + 18) return ts;
    }
    return x.ticks(2);
  })();

  return (
    <ChartFrame title={showTitle ? title : undefined} subtitle={subtitle} legend={legend} table={table} className={className}>
      <div ref={ref} className="relative w-full">
        <svg width={width} height={plotH + M.top + M.bottom} role="img" aria-label={title} className="block">
          <g transform={`translate(${M.left},${M.top})`}>
            {xTicks.map((t) => (
              <g key={t} transform={`translate(${x(t)},0)`}>
                <line y2={plotH} style={{ stroke: t === 0 ? 'var(--axis)' : 'var(--grid)' }} />
                <text y={plotH + 20} textAnchor="middle" className="tabular" style={{ fill: 'var(--muted)', fontSize: T.tick }}>
                  {format(t)}
                </text>
              </g>
            ))}
            {valueLabel && (
              <text x={plotW / 2} y={plotH + 42} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: T.axisTitle }}>
                {valueLabel}
              </text>
            )}
            {refLines?.map((r) => (
              <g key={r.label}>
                <line x1={x(r.value)} x2={x(r.value)} y1={-4} y2={plotH} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1} />
              </g>
            ))}
            {categories.map((c, ci) => {
              const top = ci * bandH + 7;
              const isHover = hover?.c === ci;
              return (
                <g key={c} onPointerEnter={() => setHover({ c: ci, s: 0 })} onPointerLeave={() => setHover(null)}>
                  <rect x={-M.left} y={ci * bandH} width={M.left + plotW + valuePad} height={bandH} style={{ fill: isHover ? 'var(--surface-2)' : 'transparent' }} />
                  <text x={-M.left + lw - 10} y={top + (nS * barH + (nS - 1) * 2) / 2} dy="0.32em" textAnchor="end" style={{ fill: 'var(--ink)', fontSize: T.category }}>
                    {fitText(c, lw - 14, T.category)}
                    {fitText(c, lw - 14, T.category) !== c && <title>{c}</title>}
                  </text>
                  {series.map((s, si) => {
                    const v = s.values[ci];
                    if (v === null || !Number.isFinite(v)) return null;
                    const yb = top + si * (barH + 2);
                    const xe = x(v);
                    return (
                      <g key={s.id} onPointerEnter={() => setHover({ c: ci, s: si })}>
                        <path d={hBarPath(x0, xe, yb, barH)} style={{ fill: fillOf(s, si, c) }} />
                        <text
                          x={v >= 0 ? xe + 5 : xe - 5}
                          y={yb + barH / 2}
                          dy="0.32em"
                          textAnchor={v >= 0 ? 'start' : 'end'}
                          className="tabular"
                          style={{ fill: 'var(--ink-2)', fontSize: T.value }}
                        >
                          {format(v)}
                        </text>
                      </g>
                    );
                  })}
                </g>
              );
            })}
            {refLines?.map((r) => (
              <text key={r.label + 't'} x={x(r.value)} y={-10} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: T.annotation }}>
                {r.label}
              </text>
            ))}
          </g>
        </svg>
        {hover && (
          <TooltipBox style={{ left: Math.max(0, Math.min(width - 220, M.left + Math.max(x0, x(series[0].values[hover.c] ?? 0)) + valuePad + 4)), top: M.top + hover.c * bandH }}>
            {tooltip}
          </TooltipBox>
        )}
      </div>
    </ChartFrame>
  );
}
