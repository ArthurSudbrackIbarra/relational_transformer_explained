/**
 * Parents vs children: every row of a parent table as one bar, height = number
 * of child rows (stacked by child table), sorted. A row has at most as many
 * parents as its table has FK columns, but children are unbounded (paper p.16
 * App. A); the dashed line marks the sampler's width bound w (spec §4.7).
 */
import { scaleLinear } from 'd3';
import { memo, useMemo, useState } from 'react';
import { ChartFrame, DataTable, Legend, Segmented, formatNumber } from '@/components';
import { TooltipBox, vBarPath } from '@/components/viz/chartParts';
import { useElementSize } from '@/components/viz/useSize';
import type { RawDatabase } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { rowCaption, rowLabel, type LinkIndex, type RowRef } from './model';

export interface DegreeChartProps {
  db: RawDatabase;
  idx: LinkIndex;
  slots: Record<string, number>;
  focus: RowRef;
  onFocus: (r: RowRef) => void;
  w: number;
  table: string;
  onTable: (t: string) => void;
}

const H = 230;
const M = { top: 14, right: 10, bottom: 30, left: 40 };

export const DegreeChart = memo(function DegreeChart({ db, idx, slots, focus, onFocus, w, table, onTable }: DegreeChartProps) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const width = Math.max(280, size.width || 640);
  const [hover, setHover] = useState<number | null>(null);

  const parentTables = useMemo(() => db.tables.filter((t) => idx.childTablesOf(t.schema.name).length > 0).map((t) => t.schema.name), [db, idx]);
  const groups = idx.childTablesOf(table);
  const data = useMemo(() => {
    const t = idx.table(table);
    if (!t) return [];
    return t.rows
      .map((_, r) => {
        const g = idx.childrenOf({ table, row: r });
        return { row: r, parts: g.map((x) => x.rows.length), total: g.reduce((s, x) => s + x.rows.length, 0) };
      })
      .sort((a, b) => b.total - a.total || a.row - b.row);
  }, [idx, table]);

  const maxV = Math.max(w, ...data.map((d) => d.total), 1);
  const y = scaleLinear().domain([0, maxV]).nice(4).range([H - M.bottom, M.top]);
  const plotW = width - M.left - M.right;
  const band = plotW / Math.max(1, data.length);
  const barW = Math.max(1, band - (band > 4 ? 2 : 1));
  const ticks = y.ticks(4);
  const focusIdx = focus.table === table ? data.findIndex((d) => d.row === focus.row) : -1;
  const hd = hover !== null ? data[hover] : null;

  const seriesLabel = (i: number) => {
    const g = groups[i];
    const dup = groups.filter((x) => x.childTable === g.childTable).length > 1;
    return dup ? `${g.childTable} (${g.fkCol})` : g.childTable;
  };

  return (
    <div className="min-w-0">
      {parentTables.length > 1 && (
        <div className="mb-3">
          <Segmented label="Parent table" value={table} onChange={onTable} options={parentTables.map((t) => ({ value: t, label: t }))} />
        </div>
      )}
      <ChartFrame
        title={`Children per ${table} row, sorted`}
        subtitle={`Each bar is one ${table} row; its height is the number of rows whose foreign key points to it.`}
        legend={
          <Legend items={groups.map((g, i) => ({ id: `${g.childTable}.${g.fkCol}`, label: seriesLabel(i), slot: slots[g.childTable] ?? 0, mark: 'rect' as const }))} />
        }
        table={
          <DataTable
            caption={`Children per ${table} row`}
            columns={[`${table} row`, ...groups.map((_, i) => seriesLabel(i)), 'total']}
            rows={data.map((d) => [`${rowLabel({ table, row: d.row })} ${rowCaption(db, { table, row: d.row })}`, ...d.parts, d.total])}
          />
        }
      >
        <div ref={ref} className="relative w-full">
          <svg width={width} height={H} role="img" aria-label={`Bar chart: number of child rows for each of the ${data.length} ${table} rows, sorted from most to fewest. Maximum ${data[0]?.total ?? 0}.`} className="block">
            {ticks.map((t) => (
              <g key={t}>
                <line x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
                <text x={M.left - 6} y={y(t)} textAnchor="end" dominantBaseline="central" className="num" style={{ fill: 'var(--muted)', fontSize: 13 }}>
                  {formatNumber(t)}
                </text>
              </g>
            ))}
            {data.map((d, i) => {
              const x = M.left + i * band + (band - barW) / 2;
              let cum = 0;
              const dim = hover !== null && hover !== i;
              return (
                <g key={d.row} style={{ opacity: dim ? 0.55 : 1 }}>
                  {d.parts.map((v, s) => {
                    if (v <= 0) return null;
                    const base = y(cum);
                    cum += v;
                    const top = y(cum);
                    const isTop = d.parts.slice(s + 1).every((u) => u <= 0);
                    const gap = base < y(0) ? 1.5 : 0;
                    const path = isTop && barW >= 5 ? vBarPath(x, barW, base - gap, top, 3) : `M${x},${base - gap}V${top}H${x + barW}V${base - gap}Z`;
                    return <path key={s} d={path} style={{ fill: catVar(slots[groups[s].childTable] ?? 0) }} />;
                  })}
                </g>
              );
            })}
            {/* width bound w */}
            <line x1={M.left} x2={width - M.right} y1={y(w)} y2={y(w)} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.25} strokeDasharray="5 4" />
            <text x={width - M.right - 2} y={y(w) - 5} textAnchor="end" style={{ fill: 'var(--ink-2)', fontSize: 13.5, fontWeight: 600, paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 3 }}>
              w = {w}
            </text>
            {/* baseline */}
            <line x1={M.left} x2={width - M.right} y1={y(0)} y2={y(0)} style={{ stroke: 'var(--axis)' }} strokeWidth={1} />
            {focusIdx >= 0 && (
              <g>
                <path
                  d={`M${M.left + focusIdx * band + band / 2},${y(0) + 3} l-5,8 h10 Z`}
                  style={{ fill: 'var(--accent)' }}
                />
                <text
                  x={Math.min(width - M.right - 4, Math.max(M.left + 4, M.left + focusIdx * band + band / 2))}
                  y={y(0) + 24}
                  textAnchor={focusIdx < data.length * 0.15 ? 'start' : focusIdx > data.length * 0.85 ? 'end' : 'middle'}
                  style={{ fill: 'var(--ink)', fontSize: 13 }}
                >
                  {rowLabel(focus)}
                </text>
              </g>
            )}
            {focusIdx < 0 && (
              <text x={M.left} y={y(0) + 20} style={{ fill: 'var(--muted)', fontSize: 13 }}>
                {table} rows, most children first →
              </text>
            )}
            {/* hit targets: the whole band */}
            {data.map((d, i) => (
              <rect
                key={d.row}
                x={M.left + i * band}
                y={M.top}
                width={band}
                height={H - M.top - M.bottom}
                fill="transparent"
                style={{ cursor: 'pointer' }}
                onPointerEnter={() => setHover(i)}
                onPointerLeave={() => setHover(null)}
                onClick={() => onFocus({ table, row: d.row })}
              />
            ))}
          </svg>
          {hd && (
            <TooltipBox
              style={{
                left: Math.min(width - 200, Math.max(0, M.left + (hover ?? 0) * band + band / 2 - 90)),
                top: Math.max(0, y(hd.total) - 78),
              }}
            >
              <div className="font-semibold text-ink">{rowLabel({ table, row: hd.row })}</div>
              <div className="mb-1 text-ink-2">{rowCaption(db, { table, row: hd.row }, 26)}</div>
              {groups.map((g, s) => (
                <div key={s} className="flex items-center gap-1.5 text-ink-2">
                  <span aria-hidden="true" className="inline-block h-2 w-2 rounded-[2px]" style={{ background: catVar(slots[g.childTable] ?? 0) }} />
                  <span className="num font-semibold text-ink">{hd.parts[s]}</span> {seriesLabel(s)}
                </div>
              ))}
              <div className="mt-1 text-[14px] text-ink-2">click to make it the focus row</div>
            </TooltipBox>
          )}
        </div>
      </ChartFrame>
    </div>
  );
});
