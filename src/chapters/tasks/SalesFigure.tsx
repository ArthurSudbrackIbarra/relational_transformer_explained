/**
 * Figure: the regression task product-sales (units ordered in (t, t+30d]). Top lane: the product's
 * orders as spikes (height = quantity). Bottom lane: one bar per prediction timestamp spanning its
 * window, height = the stored label. Hover, tap or focus a bar to see which orders it sums (the
 * window is shaded on the orders lane too); click / Enter pins it, arrow keys move between windows.
 * Labels and Train stats come from the engine (task table, PreprocessedDB.colStats).
 */
import { scaleLinear } from 'd3';
import { useEffect, useMemo, useRef, useState } from 'react';
import { ChartFrame, DataTable, Select, StatusBadge, formatNumber, formatSigned } from '@/components';
import { TooltipBox, vBarPath } from '@/components/viz/chartParts';
import { useElementSize } from '@/components/viz/useSize';
import type { PreprocessedDB, RawDatabase, TaskDef } from '@/engine/types';
import {
  aggregateWindow,
  allStamps,
  deltaMsOf,
  entityName,
  forecastSpecOf,
  isoDay,
  labelEvents,
  labelMatches,
  windowEvents,
} from './taskData';
import { LaneLabel, TimeAxis, monthDomain, useTimeScale } from './timeline';

export function SalesFigure({
  db,
  pdb,
  task,
  product,
  onProductChange,
}: {
  db: RawDatabase;
  pdb: PreprocessedDB;
  task: TaskDef;
  product: number;
  onProductChange: (p: number) => void;
}) {
  const spec = forecastSpecOf(task)!;
  const delta = deltaMsOf(task);
  const products = db.tables.find((t) => t.schema.name === task.entityTable)!;
  const events = useMemo(() => labelEvents(db, spec, product), [db, spec, product]);
  const stamps = useMemo(() => allStamps(task), [task]);
  const rows = useMemo(
    () =>
      stamps.map((s) => {
        const tbl = task.tables![s.split].rows;
        const rowPos = tbl.findIndex((r) => r.timestamp === s.t && r[task.entityCol!] === product);
        const inWin = windowEvents(events, s.t, delta);
        return {
          ...s,
          rowPos,
          stored: rowPos >= 0 ? (tbl[rowPos][task.targetCol] as number) : null,
          inWin,
          computed: aggregateWindow(spec, inWin) as number,
        };
      }),
    [stamps, task, product, events, delta, spec],
  );
  const present = rows.filter((r) => r.stored !== null);
  const allMatch = present.every((r) => labelMatches(r.computed, r.stored));
  const stats = pdb.colStats[`${task.name}:Train`]?.[task.targetCol];

  const valRow = rows.find((r) => r.split === 'Val' && r.stored !== null);
  const valZ = useMemo(() => {
    if (!valRow) return null;
    const info = pdb.tableInfo[`${task.name}:Val`];
    const node = pdb.nodes[info.nodeIdxOffset + valRow.rowPos];
    return node?.cells.find((c) => c.colName === task.targetCol)?.number ?? null;
  }, [valRow, pdb, task]);

  // hover previews a window; click / tap / Enter pins it; focus (keyboard) shows it too
  const [hoverT, setHoverT] = useState<number | null>(null);
  const [focusT, setFocusT] = useState<number | null>(null);
  const [pinnedT, setPinnedT] = useState<number | null>(null);
  useEffect(() => setPinnedT(null), [product]);
  const activeT = hoverT ?? focusT ?? pinnedT;
  const hover = activeT === null ? null : (rows.find((r) => r.t === activeT && r.stored !== null) ?? null);
  const barRefs = useRef(new Map<number, SVGGElement>());
  const presentTs = present.map((r) => r.t);
  const moveFocus = (from: number, dir: 1 | -1) => {
    const i = presentTs.indexOf(from);
    const j = Math.max(0, Math.min(presentTs.length - 1, i + dir));
    const t = presentTs[j];
    setPinnedT(t);
    barRefs.current.get(t)?.focus();
  };
  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(300, size.width || 760);
  const narrow = W < 560;
  const domain = useMemo(() => monthDomain(db.minTimestamp, db.maxTimestamp), [db]);
  const x = useTimeScale(domain, 10, W - 10);
  const maxQ = Math.max(1, ...events.map((e) => e.v));
  const spikeH = 26;
  const days = task.timedeltaDays ?? Math.round(delta / 86_400_000);
  const Y = { orderLabel: 13, orderBase: 58, salesLabel: 84, salesTop: 108, salesBase: 206, axis: 206 };
  const ymax = Math.max(1, ...present.map((r) => r.stored as number));
  const y = scaleLinear().domain([0, ymax]).nice(4).range([Y.salesBase, Y.salesTop]);
  const H = Y.axis + 30;
  // beside the bar, on the side with more room, so it never covers the bar it describes
  const TIP_W = 236;
  const tipStyle = (t: number): React.CSSProperties => {
    const a = x(new Date(t));
    const b = x(new Date(t + delta));
    const right = W - b >= TIP_W + 12 || a < W / 2;
    const left = right ? Math.min(W - TIP_W - 4, b + 8) : Math.max(4, a - 8 - TIP_W);
    return { left, top: Y.salesTop - 30, width: TIP_W };
  };

  const table = (
    <DataTable
      caption={`${task.name} labels of product ${product}`}
      columns={['timestamp', 'split', 'window', 'orders', `${task.targetCol} (stored)`]}
      rows={rows.map((r) => [
        isoDay(r.t),
        r.split,
        `(${isoDay(r.t).slice(5)}, ${isoDay(r.t + delta).slice(5)}]`,
        r.inWin.length,
        r.stored === null ? 'no row' : formatNumber(r.stored),
      ])}
    />
  );

  return (
    <div className="min-w-0">
      <div className="mb-4 flex flex-wrap items-end gap-x-4 gap-y-3">
        <Select label="Product" value={product} onChange={(e) => onProductChange(Number(e.target.value))} className="w-[360px] max-w-full">
          {products.rows.map((r, i) => (
            <option key={i} value={i}>
              #{i} · {String(r.name ?? '')} ({String(r.category ?? '∅')})
            </option>
          ))}
        </Select>
        <div className="pb-1">
          <StatusBadge status={allMatch ? 'good' : 'warning'}>
            {allMatch ? 'Recomputed sums match the engine’s task table' : 'A recomputed sum differs from the task table'}
          </StatusBadge>
        </div>
      </div>
      <ChartFrame
        title={`product-sales for #${product} ${entityName(db, task.entityTable!, product)}`}
        subtitle={`Each bar spans its window (t, t + ${days} d]; its height is the number of units ordered in that window. A flat line marks a window with 0 units.`}
        table={table}
      >
        <div ref={ref} className="relative w-full">
          <svg
            width={W}
            height={H}
            role="group"
            aria-label={`Orders and product-sales labels of product ${product}. Each label bar is a button: select it to see the orders it sums; arrow keys move between windows.`}
            className="block"
          >
            <LaneLabel x={10} y={Y.orderLabel}>
              {narrow ? `Orders of #${product}, height = quantity` : `Orders of product #${product} (spike height = quantity)`}
            </LaneLabel>
            {hover && (
              <rect
                x={x(new Date(hover.t))}
                y={Y.orderBase - spikeH - 6}
                width={Math.max(2, x(new Date(hover.t + delta)) - x(new Date(hover.t)))}
                height={spikeH + 6}
                rx={3}
                style={{ fill: 'var(--accent-soft)' }}
                aria-hidden="true"
              />
            )}
            <line x1={10} x2={W - 10} y1={Y.orderBase} y2={Y.orderBase} style={{ stroke: 'var(--grid)' }} />
            {events.map((e) => {
              const inHover = hover ? e.t > hover.t && e.t <= hover.t + delta : false;
              const h = (e.v / maxQ) * spikeH;
              return (
                <line
                  key={`${e.table}-${e.row}`}
                  x1={x(new Date(e.t))}
                  x2={x(new Date(e.t))}
                  y1={Y.orderBase}
                  y2={Y.orderBase - h}
                  strokeLinecap="round"
                  strokeWidth={inHover ? 3 : 2}
                  style={{ stroke: 'var(--cat-3)', opacity: hover && !inHover ? 0.3 : 1, transition: 'opacity 150ms' }}
                >
                  <title>{`order #${e.row} · ${isoDay(e.t)} · quantity ${e.v}`}</title>
                </line>
              );
            })}

            <LaneLabel x={10} y={Y.salesLabel}>
              {narrow ? 'sales label per timestamp' : 'sales label per timestamp (task table)'}
            </LaneLabel>
            {y.ticks(4).map((tk) => (
              <g key={tk}>
                <line x1={10} x2={W - 10} y1={y(tk)} y2={y(tk)} style={{ stroke: tk === 0 ? 'var(--axis)' : 'var(--grid)' }} />
                {tk > 0 && (
                  <text x={12} y={y(tk) - 3} className="fill-[var(--muted)] text-[13px]" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    {tk}
                  </text>
                )}
              </g>
            ))}
            {rows.map((r) => {
              if (r.stored === null) return null;
              const a = x(new Date(r.t)) + 1;
              const b = x(new Date(r.t + delta)) - 1;
              const isHover = hover?.t === r.t;
              return (
                <g
                  key={r.t}
                  ref={(el) => {
                    if (el) barRefs.current.set(r.t, el);
                    else barRefs.current.delete(r.t);
                  }}
                  role="button"
                  tabIndex={0}
                  aria-pressed={pinnedT === r.t}
                  aria-label={`${r.split} window (${isoDay(r.t)}, ${isoDay(r.t + delta)}]: ${formatNumber(r.stored)} units from ${r.inWin.length} order${r.inWin.length === 1 ? '' : 's'}`}
                  className="cursor-pointer outline-none"
                  onPointerEnter={(e) => e.pointerType === 'mouse' && setHoverT(r.t)}
                  onPointerLeave={() => setHoverT(null)}
                  onClick={() => setPinnedT((p) => (p === r.t ? null : r.t))}
                  onFocus={() => setFocusT(r.t)}
                  onBlur={() => setFocusT(null)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setPinnedT((p) => (p === r.t ? null : r.t));
                    } else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
                      e.preventDefault();
                      moveFocus(r.t, e.key === 'ArrowRight' ? 1 : -1);
                    } else if (e.key === 'Escape') {
                      setPinnedT(null);
                    }
                  }}
                >
                  {focusT === r.t && (
                    <rect
                      x={a - 2}
                      y={Math.min(y(r.stored), y(0) - 4) - 3}
                      width={Math.max(2, b - a) + 4}
                      height={y(0) - Math.min(y(r.stored), y(0) - 4) + 5}
                      rx={4}
                      style={{ fill: 'none', stroke: 'var(--accent)' }}
                      strokeWidth={2}
                    />
                  )}
                  <rect x={a} y={Y.salesTop - 8} width={Math.max(2, b - a)} height={Y.salesBase - Y.salesTop + 8} fill="transparent" />
                  {r.stored > 0 ? (
                    <path
                      d={vBarPath(a, Math.max(2, b - a), y(0), y(r.stored), 3)}
                      style={{
                        fill: 'var(--cat-5)',
                        opacity: hover && !isHover ? 0.45 : 1,
                        transition: 'opacity 150ms',
                      }}
                    />
                  ) : (
                    // a stored 0 is a real row: draw it as a flat line on the baseline
                    <line
                      x1={a + 1}
                      x2={Math.max(a + 2, b - 1)}
                      y1={y(0) - 1.5}
                      y2={y(0) - 1.5}
                      strokeLinecap="round"
                      style={{ stroke: 'var(--cat-5)', opacity: hover && !isHover ? 0.45 : 1 }}
                      strokeWidth={3}
                    />
                  )}
                  {r.split !== 'Train' &&
                    (narrow ? (
                      // narrow: the split name on its own line above the number, so neighbouring labels never collide
                      <text x={(a + b) / 2} y={y(r.stored) - 19} textAnchor="middle" className="fill-[var(--ink-2)] text-[13px]">
                        <tspan x={(a + b) / 2}>{r.split === 'Val' ? 'val' : 'test'}</tspan>
                        <tspan x={(a + b) / 2} dy={14} style={{ fontVariantNumeric: 'tabular-nums' }}>
                          {formatNumber(r.stored)}
                        </tspan>
                      </text>
                    ) : (
                      <text
                        x={(a + b) / 2}
                        y={y(r.stored) - 5}
                        textAnchor="middle"
                        className="fill-[var(--ink-2)] text-[13px]"
                        style={{ fontVariantNumeric: 'tabular-nums' }}
                      >
                        {r.split === 'Val' ? 'val ' : 'test '}
                        {formatNumber(r.stored)}
                      </text>
                    ))}
                </g>
              );
            })}
            <TimeAxis x={x} y={Y.axis} narrow={narrow} />
          </svg>
          {hover && hover.stored !== null && (
            <TooltipBox style={tipStyle(hover.t)}>
              <div className="text-[13.5px] text-muted">
                {hover.split} · t = {isoDay(hover.t)}
              </div>
              <div className="mt-0.5">
                <strong className="num text-ink">{formatNumber(hover.stored)}</strong> <span className="text-ink-2">units in the window</span>
              </div>
              <div className="num text-[14px] text-ink-2">
                ({isoDay(hover.t)}, {isoDay(hover.t + delta)}]
              </div>
              <div className="text-[14px] text-ink-2">
                {hover.inWin.length} order{hover.inWin.length === 1 ? '' : 's'}
                {hover.inWin.length > 0 ? `: ${hover.inWin.map((e) => e.v).join(' + ')}` : ''}
              </div>
            </TooltipBox>
          )}
        </div>
      </ChartFrame>
      {stats && (
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <div className="rounded-xl border border-border bg-surface-2 px-3.5 py-2.5">
            <div className="text-[14.5px] text-ink-2">Train μ of sales (all products’ rows)</div>
            <div className="num mt-0.5 text-[20px] font-semibold text-ink">{formatNumber(stats.mean, 3)}</div>
          </div>
          <div className="rounded-xl border border-border bg-surface-2 px-3.5 py-2.5">
            <div className="text-[14.5px] text-ink-2">Train σ of sales (sample, all rows)</div>
            <div className="num mt-0.5 text-[20px] font-semibold text-ink">{formatNumber(stats.std, 3)}</div>
          </div>
          <div className="rounded-xl border border-border bg-surface-2 px-3.5 py-2.5">
            <div className="text-[14.5px] text-ink-2">Val label of #{product}, as stored for the model</div>
            <div className="num mt-0.5 text-[20px] font-semibold text-ink">
              {valRow ? formatNumber(valRow.stored as number) : '—'}
              {valZ !== null && <span className="ml-2 text-[15.5px] font-normal text-ink-2">z = {formatSigned(valZ)}</span>}
            </div>
            {valRow && valZ !== null && (
              <div className="num mt-0.5 text-[14.5px] text-ink-2">
                = ({formatNumber(valRow.stored as number)} − {formatNumber(stats.mean, 3)}) / {formatNumber(stats.std, 3)}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
