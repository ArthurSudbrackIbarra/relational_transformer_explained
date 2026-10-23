/**
 * DotPlot: one row per task (plus the printed mean row), one marker per method on a shared value axis.
 * Used for Tables 1, 2/16, 3 and 10. Off-scale values ("< −9", or below the axis) become a chevron at
 * the axis edge with the printed value. Hover or focus a row for every value; the right-hand column
 * prints the focus method's value (selective direct labeling). Markers glide when the metric changes.
 */
import { motion, useReducedMotion } from 'motion/react';
import { useState, type ReactNode } from 'react';
import { textWidth } from '@/components/viz/text';
import { useElementSize } from '@/components/viz/useSize';
import { catVar } from '@/styles/palette';
import { LT9, type Cell } from './data';
import { cellText, isNum, MethodGlyph, MethodMark, Tip, xScale } from './kit';
import { METHODS } from './methods';

export interface DotRow {
  id: string;
  label: string;
  /** Second line / tooltip detail (e.g. training-set size). */
  detail?: string;
  values: Record<string, Cell>;
  mean?: boolean;
}

export interface DotPlotProps {
  rows: DotRow[];
  /** Method ids to draw, in legend order. */
  methods: string[];
  domain: [number, number];
  /** Reference line (random = 50, global mean = 0). */
  refLine?: { value: number; label: string };
  axisLabel: string;
  /** Method whose value is printed in the right-hand column and drawn on top. */
  focus?: string;
  ariaLabel: string;
  /** Extra line at the bottom of the tooltip. */
  tipNote?: (row: DotRow) => ReactNode;
}

const ROW_H = 36;
const ROW_H_STACKED = 58;
const R = 6.5;

export function DotPlot({ rows, methods, domain, refLine, axisLabel, focus, ariaLabel, tipNote }: DotPlotProps) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const reduce = useReducedMotion();
  const [hover, setHover] = useState<number | null>(null);
  const W = Math.max(300, size.width || 720);

  const labelW = Math.ceil(Math.max(...rows.map((r) => textWidth(r.label, 15, { weight: r.mean ? 600 : 400 })))) + 16;
  const valueW = focus ? 64 : 12;
  const stacked = W - labelW - valueW < 300;
  const rowH = stacked ? ROW_H_STACKED : ROW_H;
  const plotL = stacked ? 14 : labelW + 10;
  const plotR = W - valueW - 12;
  const top = 30;
  const meanGap = 14;
  const nRows = rows.length;
  const hasMean = rows.some((r) => r.mean);
  const plotH = nRows * rowH + (hasMean ? meanGap : 0);
  const H = top + plotH + 48;
  const { x, ticks } = xScale(domain, plotL, plotR, W < 520 ? 56 : 70);
  const rowY = (i: number) => top + i * rowH + (hasMean && rows[i].mean ? meanGap : 0);
  const markY = stacked ? rowH - 18 : rowH / 2;
  const order = focus ? [...methods.filter((m) => m !== focus), focus] : methods;
  const transition = reduce ? { duration: 0 } : { type: 'spring' as const, stiffness: 170, damping: 24 };

  return (
    <div ref={ref} className="relative w-full" onPointerLeave={() => setHover(null)}>
      <svg width={W} height={H} role="img" aria-label={ariaLabel} className="block">
        {/* grid + ticks */}
        {ticks.map((t) => (
          <g key={t} transform={`translate(${x(t)},0)`}>
            <line y1={top - 6} y2={top + plotH} style={{ stroke: 'var(--grid)' }} />
            <text y={top + plotH + 20} textAnchor="middle" className="tabular" style={{ fill: 'var(--muted)', fontSize: 13.5 }}>
              {t.toString().replace('-', '−')}
            </text>
          </g>
        ))}
        <text x={(plotL + plotR) / 2} y={top + plotH + 42} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: 14.5 }}>
          {axisLabel}
        </text>
        {refLine && refLine.value >= domain[0] && refLine.value <= domain[1] && (
          <g>
            <line x1={x(refLine.value)} x2={x(refLine.value)} y1={top - 10} y2={top + plotH} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.25} />
            <text x={x(refLine.value)} y={top - 14} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: 13.5 }}>
              {refLine.label}
            </text>
          </g>
        )}
        {focus && (
          <text x={W - 4} y={top - 14} textAnchor="end" style={{ fill: 'var(--ink-2)', fontSize: 13.5, fontWeight: 600 }}>
            {METHODS[focus]?.label}
          </text>
        )}

        {rows.map((r, i) => {
          const y0 = rowY(i);
          const nums = methods.map((m) => r.values[m]).filter(isNum);
          const lo = nums.length ? Math.min(...nums) : null;
          const hi = nums.length ? Math.max(...nums) : null;
          const fv = focus ? r.values[focus] : undefined;
          return (
            <g
              key={r.id}
              tabIndex={0}
              role="listitem"
              aria-label={`${r.label}: ${methods.map((m) => `${METHODS[m]?.label} ${cellText(r.values[m] ?? null)}`).join(', ')}`}
              onPointerEnter={() => setHover(i)}
              onFocus={() => setHover(i)}
              onBlur={() => setHover(null)}
              className="outline-none"
            >
              {r.mean && <line x1={0} x2={W} y1={y0 - meanGap / 2} y2={y0 - meanGap / 2} style={{ stroke: 'var(--axis)' }} strokeWidth={1.5} />}
              <rect x={0} y={y0} width={W} height={rowH} rx={6} style={{ fill: hover === i ? 'var(--surface-2)' : 'transparent' }} />
              <text
                x={stacked ? plotL : labelW - 6}
                y={stacked ? y0 + 18 : y0 + rowH / 2}
                dy="0.32em"
                textAnchor={stacked ? 'start' : 'end'}
                style={{
                  fill: 'var(--ink)',
                  fontSize: 15,
                  fontWeight: r.mean ? 600 : 400,
                  stroke: hover === i ? 'var(--surface-2)' : 'var(--surface)',
                  strokeWidth: 4,
                  paintOrder: 'stroke',
                  strokeLinejoin: 'round',
                }}
              >
                {r.label}
                {r.detail && stacked ? <tspan style={{ fill: 'var(--muted)', fontFamily: 'var(--font-sans)' }}>{`  ${r.detail}`}</tspan> : null}
              </text>
              {lo !== null && hi !== null && hi > lo && (
                <line
                  x1={x(Math.max(lo, domain[0]))}
                  x2={x(Math.min(hi, domain[1]))}
                  y1={y0 + markY}
                  y2={y0 + markY}
                  style={{ stroke: 'var(--axis)' }}
                  strokeWidth={2}
                  strokeLinecap="round"
                />
              )}
              {order.map((m) => {
                const st = METHODS[m];
                const v = r.values[m];
                if (!st || v === null || v === undefined) return null;
                const off = v === LT9 || (isNum(v) && v < domain[0]);
                if (off) {
                  return (
                    <g key={m} transform={`translate(${plotL - 2},${y0 + markY})`}>
                      <path
                        d="M5,-6 L-2,0 L5,6"
                        style={{ fill: 'none', stroke: catVar(st.slot) }}
                        strokeWidth={2.4}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </g>
                  );
                }
                const xv = x(Math.min(v as number, domain[1]));
                return (
                  <motion.g key={m} initial={false} animate={{ x: xv, y: y0 + markY }} transition={transition}>
                    <MethodMark m={st} r={m === focus ? R + 1 : R} />
                  </motion.g>
                );
              })}
              {focus && (
                <text
                  x={W - 4}
                  y={y0 + markY}
                  dy="0.32em"
                  textAnchor="end"
                  className="tabular"
                  style={{ fill: 'var(--ink)', fontSize: 15, fontWeight: r.mean ? 700 : 500 }}
                >
                  {fv === undefined ? '' : cellText(fv)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {hover !== null && rows[hover] && (
        <Tip x={W * 0.55} y={rowY(hover) + rowH + 2} width={W} flip={false}>
          <div className="mb-1 text-[14.5px] text-ink">{rows[hover].label}</div>
          {rows[hover].detail && <div className="mb-1 text-[14px] text-muted">{rows[hover].detail}</div>}
          {[...methods]
            .sort((a, b) => {
              const va = rows[hover].values[a];
              const vb = rows[hover].values[b];
              const na = isNum(va) ? va : -1e9;
              const nb = isNum(vb) ? vb : -1e9;
              return nb - na;
            })
            .map((m) => (
              <div key={m} className="flex items-center gap-2 py-px">
                <MethodGlyph id={m} size={12} />
                <span className="num w-[52px] shrink-0 text-right font-semibold text-ink">{cellText(rows[hover].values[m] ?? null)}</span>
                <span className="truncate text-ink-2">{METHODS[m]?.label}</span>
              </div>
            ))}
          {tipNote && <div className="mt-1 text-[14px] text-muted">{tipNote(rows[hover])}</div>}
        </Tip>
      )}
    </div>
  );
}
