/**
 * Figure: the regression task product-sales (units ordered in (t, t+30d]). Top lane: the product's
 * orders as spikes (height = quantity). Bottom lane: one bar per prediction timestamp spanning its
 * window, height = the stored label. Hover a bar to see which orders it sums.
 * Labels and Train stats come from the engine (task table, PreprocessedDB.colStats).
 */
import { scaleLinear } from 'd3';
import { useMemo, useState } from 'react';
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

  const [hover, setHover] = useState<(typeof rows)[number] | null>(null);
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
          <svg width={W} height={H} role="img" aria-label={`Orders and product-sales labels of product ${product}`} className="block">
            <LaneLabel x={10} y={Y.orderLabel}>
              {narrow ? `Orders of #${product}, height = quantity` : `Orders of product #${product} (spike height = quantity)`}
            </LaneLabel>
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
                <g key={r.t} onPointerEnter={() => setHover(r)} onPointerLeave={() => setHover(null)}>
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
                  {r.split !== 'Train' && (
                    <text
                      x={(a + b) / 2}
                      y={y(r.stored) - 5}
                      textAnchor="middle"
                      className="fill-[var(--ink-2)] text-[13px]"
                      style={{ fontVariantNumeric: 'tabular-nums' }}
                    >
                      {narrow ? '' : r.split === 'Val' ? 'val ' : 'test '}
                      {formatNumber(r.stored)}
                    </text>
                  )}
                </g>
              );
            })}
            <TimeAxis x={x} y={Y.axis} narrow={narrow} />
          </svg>
          {hover && hover.stored !== null && (
            <TooltipBox style={{ left: Math.max(4, Math.min(W - 230, x(new Date(hover.t)) + 12)), top: Y.salesTop - 30 }}>
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
            <div className="text-[14.5px] text-ink-2">Train mean μ of sales</div>
            <div className="num mt-0.5 text-[20px] font-semibold text-ink">{formatNumber(stats.mean, 3)}</div>
          </div>
          <div className="rounded-xl border border-border bg-surface-2 px-3.5 py-2.5">
            <div className="text-[14.5px] text-ink-2">Train std σ (sample)</div>
            <div className="num mt-0.5 text-[20px] font-semibold text-ink">{formatNumber(stats.std, 3)}</div>
          </div>
          <div className="rounded-xl border border-border bg-surface-2 px-3.5 py-2.5">
            <div className="text-[14.5px] text-ink-2">Val label as stored</div>
            <div className="num mt-0.5 text-[20px] font-semibold text-ink">
              {valRow ? formatNumber(valRow.stored as number) : '—'}
              {valZ !== null && <span className="ml-2 text-[15.5px] font-normal text-ink-2">z = {formatSigned(valZ)}</span>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
