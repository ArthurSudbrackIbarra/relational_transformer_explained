/**
 * Shared SVG timeline pieces for chapter 3: one UTC time scale for every figure (so the
 * builder, the splits chart and the sales chart line up), a hairline month axis, lane labels
 * and the task-table marker (slot 5 hexagon, BUILD_PLAN §3.5 / shapes.tsx).
 */
import { scaleUtc, type ScaleTime } from 'd3';
import { useMemo } from 'react';
import { shapePath } from '@/components';

export type TimeScale = ScaleTime<number, number>;

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** Month-aligned domain around the DB's time range. */
export function monthDomain(min: number, max: number): [Date, Date] {
  const a = new Date(min);
  const b = new Date(max);
  return [new Date(Date.UTC(a.getUTCFullYear(), a.getUTCMonth(), 1)), new Date(Date.UTC(b.getUTCFullYear(), b.getUTCMonth() + 1, 1))];
}

export function useTimeScale(domain: [Date, Date], x0: number, x1: number): TimeScale {
  return useMemo(() => scaleUtc().domain(domain).range([x0, Math.max(x0 + 10, x1)]), [domain, x0, x1]);
}

/** Hairline month axis with labels every `every` months (Jan always carries the year). */
export function TimeAxis({ x, y, narrow }: { x: TimeScale; y: number; narrow: boolean }) {
  const [d0, d1] = x.domain();
  const ticks: Date[] = [];
  for (let d = new Date(d0); d <= d1; d = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1))) ticks.push(d);
  const every = narrow ? 6 : 3;
  const [r0, r1] = x.range();
  return (
    <g aria-hidden="true">
      <line x1={r0} x2={r1} y1={y} y2={y} style={{ stroke: 'var(--axis)' }} strokeWidth={1} />
      {ticks.map((d) => {
        const m = d.getUTCMonth();
        const labelled = m % every === 0;
        const px = x(d);
        return (
          <g key={d.getTime()}>
            <line x1={px} x2={px} y1={y} y2={y + (labelled ? 5 : 3)} style={{ stroke: 'var(--axis)' }} strokeWidth={1} />
            {labelled && (
              <text
                x={px}
                y={y + 17}
                textAnchor={px - r0 < 16 ? 'start' : r1 - px < 16 ? 'end' : 'middle'}
                className="fill-[var(--muted)] text-[13px]"
                style={{ fontVariantNumeric: 'tabular-nums' }}
              >
                {MONTHS[m]}
                {m === 0 ? ` ${d.getUTCFullYear()}` : ''}
              </text>
            )}
          </g>
        );
      })}
    </g>
  );
}

/** Small lane title inside a timeline (muted, left aligned). */
export function LaneLabel({ x, y, children }: { x: number; y: number; children: React.ReactNode }) {
  return (
    <text
      x={x}
      y={y}
      className="fill-[var(--ink-2)] text-[13.5px] font-medium"
      style={{ paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 4, strokeLinejoin: 'round' }}
    >
      {children}
    </text>
  );
}

/** Hexagon = the task-table shape (slot 5), centred at (cx, cy). */
export function TaskMarker({
  cx,
  cy,
  r = 5.5,
  filled = true,
  className,
  style,
}: {
  cx: number;
  cy: number;
  r?: number;
  filled?: boolean;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <path
      d={shapePath('hexagon', r)}
      transform={`translate(${cx},${cy})`}
      className={className}
      style={{
        fill: filled ? 'var(--cat-5)' : 'var(--surface)',
        stroke: filled ? 'var(--surface)' : 'var(--cat-5)',
        strokeWidth: filled ? 1.5 : 2,
        ...style,
      }}
    />
  );
}
