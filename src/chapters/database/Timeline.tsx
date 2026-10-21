/**
 * Timeline of the time-stamped rows linked to one entity (e.g. all orders and
 * reviews of a customer), with a draggable prediction time t. Rows after t are
 * "the future": using them to predict at t would leak (paper p.3 §2.1).
 */
import { scaleTime, utcMonth } from 'd3';
import { RotateCcw } from 'lucide-react';
import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { Button, Slider, TableSwatch, formatDate } from '@/components';
import { TooltipBox } from '@/components/viz/chartParts';
import { shapeOfSlot, shapePath } from '@/components/viz/shapes';
import { useElementSize } from '@/components/viz/useSize';
import type { RawDatabase } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { rowCaption, rowKey, rowLabel, sameRow, type LinkIndex, type RowRef } from './model';

export interface TimelineProps {
  db: RawDatabase;
  idx: LinkIndex;
  slots: Record<string, number>;
  anchor: RowRef;
  focus: RowRef;
  onFocus: (r: RowRef) => void;
  /** Default prediction time (the seed's timestamp, ms). */
  seedTime: number | null;
}

const LANE_H = 46;
const M = { top: 26, right: 14, bottom: 28, left: 12 };
const DAY = 86_400_000;

interface Mark {
  ref: RowRef;
  t: number;
  lane: number;
  jitter: number;
}

export const Timeline = memo(function Timeline({ db, idx, slots, anchor, focus, onFocus, seedTime }: TimelineProps) {
  const [wrap, size] = useElementSize<HTMLDivElement>();
  const width = Math.max(300, size.width || 700);
  const narrow = width < 560;

  const lanes = useMemo(
    () =>
      idx
        .childrenOf(anchor)
        .filter((g) => !!idx.table(g.childTable)?.schema.timeCol)
        .map((g) => ({ ...g, timeCol: idx.table(g.childTable)!.schema.timeCol! })),
    [idx, anchor],
  );
  const marks = useMemo(() => {
    const out: Mark[] = [];
    lanes.forEach((g, lane) => {
      const t = idx.table(g.childTable)!;
      for (const r of g.rows) {
        const v = t.rows[r]?.[g.timeCol];
        if (typeof v !== 'number') continue;
        // deterministic vertical jitter so same-day rows do not hide each other
        const h = Math.sin(r * 12.9898 + lane * 78.233) * 43758.5453;
        out.push({ ref: { table: g.childTable, row: r }, t: v, lane, jitter: (h - Math.floor(h) - 0.5) * 18 });
      }
    });
    return out;
  }, [idx, lanes]);

  const t0 = utcMonth.floor(new Date(db.minTimestamp)).getTime();
  const t1 = utcMonth.ceil(new Date(db.maxTimestamp)).getTime();
  const fallback = seedTime ?? t0 + (t1 - t0) / 2;
  const [cut, setCut] = useState(fallback);
  useEffect(() => setCut(fallback), [fallback]);

  const H = M.top + Math.max(1, lanes.length) * LANE_H + M.bottom;
  const x = scaleTime()
    .domain([new Date(t0), new Date(t1)])
    .range([M.left, width - M.right]);
  const months = utcMonth.range(new Date(t0), new Date(t1 + 1));
  const every = narrow ? 6 : 3;

  const [hover, setHover] = useState<Mark | null>(null);
  const dragging = useRef(false);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const setFromEvent = (e: React.PointerEvent) => {
    const svg = svgRef.current;
    if (!svg) return;
    const r = svg.getBoundingClientRect();
    const px = Math.max(M.left, Math.min(width - M.right, e.clientX - r.left));
    setCut(x.invert(px).getTime());
  };

  const before = lanes.map((_, l) => marks.filter((m) => m.lane === l && m.t <= cut).length);
  const after = lanes.map((_, l) => marks.filter((m) => m.lane === l && m.t > cut).length);
  const xc = x(new Date(cut));

  if (lanes.length === 0) {
    return <p className="text-[16px] text-ink-2">No time-stamped rows are linked to {rowLabel(anchor)}.</p>;
  }

  return (
    <div className="min-w-0">
      <div className="mb-3 flex flex-wrap items-end gap-x-4 gap-y-3">
        <Slider
          label="Prediction time t"
          value={Math.round((cut - t0) / DAY)}
          min={0}
          max={Math.round((t1 - t0) / DAY)}
          onChange={(d) => setCut(t0 + d * DAY)}
          format={() => formatDate(cut, 'never')}
          className="w-full max-w-[340px]"
        />
        {seedTime !== null && Math.abs(cut - seedTime) > DAY / 2 && (
          <Button size="sm" variant="ghost" icon={<RotateCcw size={14} />} onClick={() => setCut(seedTime)}>
            Back to the seed’s time ({formatDate(seedTime, 'never')})
          </Button>
        )}
      </div>
      <div ref={wrap} className="relative w-full select-none">
        <svg
          ref={svgRef}
          width={width}
          height={H}
          role="img"
          aria-label={`Timeline of the rows linked to ${rowLabel(anchor)}. Prediction time ${formatDate(cut, 'never')}: ${lanes
            .map((g, l) => `${before[l]} ${g.childTable} before, ${after[l]} after`)
            .join('; ')}.`}
          className="block touch-pan-y"
          onPointerDown={(e) => {
            dragging.current = true;
            (e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId);
            setFromEvent(e);
          }}
          onPointerMove={(e) => dragging.current && setFromEvent(e)}
          onPointerUp={() => (dragging.current = false)}
          onPointerCancel={() => (dragging.current = false)}
          style={{ cursor: 'ew-resize' }}
        >
          {/* future region */}
          <rect x={xc} y={M.top - 6} width={Math.max(0, width - M.right - xc)} height={H - M.top - M.bottom + 6} style={{ fill: 'var(--surface-2)' }} />
          {/* month grid */}
          {months.map((m, i) => (
            <g key={m.getTime()}>
              <line x1={x(m)} x2={x(m)} y1={M.top - 6} y2={H - M.bottom} style={{ stroke: 'var(--grid)' }} strokeWidth={i % every === 0 ? 1 : 0.5} />
              {i % every === 0 && x(m) + 58 < width && (
                <text x={x(m) + 3} y={H - M.bottom + 16} style={{ fill: 'var(--muted)', fontSize: 13 }}>
                  {m.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' })} {m.getUTCFullYear()}
                </text>
              )}
            </g>
          ))}
          {/* lanes */}
          {lanes.map((g, l) => (
            <g key={g.childTable + g.fkCol}>
              <line
                x1={M.left}
                x2={width - M.right}
                y1={M.top + l * LANE_H + LANE_H / 2}
                y2={M.top + l * LANE_H + LANE_H / 2}
                style={{ stroke: 'var(--axis)' }}
                strokeWidth={1}
              />
              <text
                x={M.left + 2}
                y={M.top + l * LANE_H + 7}
                className="font-mono"
                style={{ fill: 'var(--ink-2)', fontSize: 13, paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 3 }}
              >
                {g.childTable}
              </text>
            </g>
          ))}
          {marks.map((m) => {
            const slot = slots[m.ref.table] ?? 0;
            const fut = m.t > cut;
            const isFocus = sameRow(m.ref, focus);
            const cx = x(new Date(m.t));
            const cy = M.top + m.lane * LANE_H + LANE_H / 2 + m.jitter;
            return (
              <g
                key={rowKey(m.ref)}
                transform={`translate(${cx},${cy})`}
                style={{ opacity: fut ? 0.3 : 1, cursor: 'pointer' }}
                onPointerEnter={() => setHover(m)}
                onPointerLeave={() => setHover(null)}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={() => onFocus(m.ref)}
              >
                <circle r={9} fill="transparent" />
                {isFocus && <circle r={9} fill="none" style={{ stroke: 'var(--accent)' }} strokeWidth={2} />}
                <path d={shapePath(shapeOfSlot(slot), 4.6)} style={{ fill: catVar(slot), stroke: 'var(--surface)' }} strokeWidth={1.5} paintOrder="stroke" />
              </g>
            );
          })}
          {/* seed time (if moved away) */}
          {seedTime !== null && Math.abs(cut - seedTime) > DAY / 2 && (
            <line x1={x(new Date(seedTime))} x2={x(new Date(seedTime))} y1={M.top - 6} y2={H - M.bottom} style={{ stroke: 'var(--ink-2)' }} strokeDasharray="3 3" strokeWidth={1} />
          )}
          {/* cutoff */}
          <line x1={xc} x2={xc} y1={M.top - 10} y2={H - M.bottom} style={{ stroke: 'var(--accent)' }} strokeWidth={2} />
          <circle cx={xc} cy={M.top - 10} r={5} style={{ fill: 'var(--accent)' }} />
          <text
            x={xc + (xc > width * 0.72 ? -8 : 8)}
            y={M.top - 8}
            textAnchor={xc > width * 0.72 ? 'end' : 'start'}
            style={{ fill: 'var(--ink)', fontSize: 13.5, fontWeight: 600, paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 3 }}
          >
            t = {formatDate(cut, 'never')}
            {seedTime !== null && Math.abs(cut - seedTime) <= DAY / 2 ? ' (seed)' : ''}
          </text>
          {xc < width - M.right - 70 && (
            <text x={width - M.right - 4} y={M.top + 6} textAnchor="end" style={{ fill: 'var(--muted)', fontSize: 13 }}>
              after t: the future
            </text>
          )}
        </svg>
        {hover && (
          <TooltipBox
            style={{
              left: Math.min(width - 210, Math.max(0, x(new Date(hover.t)) - 100)),
              top: Math.max(0, M.top + hover.lane * LANE_H + LANE_H / 2 + hover.jitter + 14),
            }}
          >
            <div className="font-mono font-semibold text-ink">{rowLabel(hover.ref)}</div>
            <div className="num text-ink-2">{formatDate(hover.t, 'always')}</div>
            {idx
              .parentsOf(hover.ref)
              .filter((p) => p.parentRow !== null && !(p.parentTable === anchor.table && p.parentRow === anchor.row))
              .map((p) => (
                <div key={p.fkCol} className="text-ink-2">
                  → {rowLabel({ table: p.parentTable, row: p.parentRow! })} {rowCaption(db, { table: p.parentTable, row: p.parentRow! }, 22)}
                </div>
              ))}
            <div className="mt-1 text-[14px] text-ink-2">{hover.t > cut ? 'after t: not available at t' : 'before t: available at t'}</div>
          </TooltipBox>
        )}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1.5 text-[15px] text-ink-2" aria-live="polite">
        {lanes.map((g, l) => (
          <span key={g.childTable + g.fkCol} className="inline-flex items-center gap-1.5">
            <TableSwatch slot={slots[g.childTable] ?? 0} shape label={g.childTable} size={10} />
            <span>
              <span className="num font-semibold text-ink">{before[l]}</span> before t ·{' '}
              <span className="num font-semibold text-ink">{after[l]}</span> after
            </span>
          </span>
        ))}
        <span>Drag on the chart or use the slider to move t.</span>
      </div>
    </div>
  );
});
