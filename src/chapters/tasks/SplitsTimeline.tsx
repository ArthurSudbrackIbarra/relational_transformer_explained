/**
 * Figure: the three splits of a forecast task table on one timeline (spec §2.2), with a preview
 * of which task rows a seed at time t could ever see (spec §2.5, fly.rs:421-424: a P→F child is
 * skipped iff child_ts > seed_ts, so equal timestamps pass).
 *
 * Bars = number of task rows per timestamp (engine task table). Nothing here samples a context:
 * the colours say only whether a row passes the time filter, not whether it is in the context.
 */
import { useId, useMemo, useState } from 'react';
import { ChartFrame, DataTable, Select, formatPct } from '@/components';
import { TooltipBox, vBarPath } from '@/components/viz/chartParts';
import { useElementSize } from '@/components/viz/useSize';
import type { RawDatabase, Split, TaskDef } from '@/engine/types';
import { isoDay, stampSummaries, type StampSummary } from './taskData';
import { TimeAxis, monthDomain, useTimeScale } from './timeline';
import { scaleLinear } from 'd3';

type Vis = 'earlier' | 'same' | 'later';

// What the time filter decides (fly.rs:421-424). Passing it does not mean "in the context": that is
// up to the search and L (Chapter 6).
const VIS_TEXT: Record<Vis, string> = {
  earlier: 'earlier than the seed: passes the time filter',
  same: 'same timestamp as the seed: passes the time filter too (fly.rs:422 skips only later rows)',
  later: 'later than the seed: skipped by the time filter',
};
const VIS_SHORT: Record<Vis, string> = { earlier: 'passes', same: 'passes (same t)', later: 'skipped' };

export function SplitsTimeline({ db, task, defaultSeedT }: { db: RawDatabase; task: TaskDef; defaultSeedT: number }) {
  const sums = useMemo(() => stampSummaries(task), [task]);
  const [seedT, setSeedT] = useState<number>(() => (sums.some((s) => s.t === defaultSeedT) ? defaultSeedT : task.valTimestamp!));
  const [hover, setHover] = useState<StampSummary | null>(null);
  const hatchId = useId().replace(/:/g, '');

  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(300, size.width || 760);
  const narrow = W < 560;
  const domain = useMemo(() => monthDomain(db.minTimestamp, db.maxTimestamp), [db]);
  const x = useTimeScale(domain, 10, W - 10);
  const M = { top: 80, plotH: 120 };
  const axisY = M.top + M.plotH;
  const H = axisY + 30;
  const maxRows = Math.max(...sums.map((s) => s.rows));
  const y = scaleLinear().domain([0, maxRows]).nice(4).range([M.top + M.plotH, M.top]);
  const pxPerDelta = x(new Date(task.valTimestamp!)) - x(new Date(task.valTimestamp! - (task.timedeltaDays ?? 30) * 86_400_000));
  const barW = Math.max(6, Math.min(18, pxPerDelta * 0.6));

  const visOf = (t: number): Vis => (t < seedT ? 'earlier' : t === seedT ? 'same' : 'later');
  // later = hollow dashed outline in --muted (>= 3:1 against the surface in both themes)
  const fillOf = (v: Vis) => (v === 'earlier' ? 'var(--cat-5)' : v === 'same' ? `url(#${hatchId})` : 'transparent');

  const groups = (['Train', 'Val', 'Test'] as Split[]).map((s) => {
    const items = sums.filter((x) => x.split === s);
    return { split: s, items, rows: items.reduce((a, b) => a + b.rows, 0), t0: items[0]?.t ?? 0, t1: items[items.length - 1]?.t ?? 0 };
  });

  const seedSplit = sums.find((s) => s.t === seedT)?.split ?? 'Val';
  const binary = task.targetType === 'binary';

  const table = (
    <DataTable
      caption={`${task.name}: rows per timestamp`}
      columns={['timestamp', 'split', 'rows', binary ? `${task.targetCol} = true` : `mean ${task.targetCol}`, 'time filter']}
      rows={sums.map((s) => [
        isoDay(s.t),
        s.split,
        s.rows,
        binary ? formatPct(s.positives / Math.max(1, s.rows), 0) : s.mean.toFixed(2),
        VIS_SHORT[visOf(s.t)],
      ])}
    />
  );

  return (
    <ChartFrame
      title={`${task.name}: one table, three splits`}
      subtitle="Bars count the task rows at each prediction timestamp. Colour shows which of them a seed at t may see under the time filter (fly.rs:421-424); whether they then land in the context is up to the search (Chapter 6)."
      table={table}
    >
      <div className="mb-3 flex flex-wrap items-end gap-x-4 gap-y-2">
        <Select label="Suppose the seed row is at" value={seedT} onChange={(e) => setSeedT(Number(e.target.value))} className="w-[220px]">
          {sums.map((s) => (
            <option key={s.t} value={s.t}>
              {s.split} · {isoDay(s.t)}
            </option>
          ))}
        </Select>
        <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 pb-1 text-[14.5px] text-ink-2" aria-label="Legend">
          <li className="inline-flex items-center gap-1.5">
            <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: 'var(--cat-5)' }} />
            earlier: passes the time filter
          </li>
          <li className="inline-flex items-center gap-1.5">
            <svg width="11" height="11" aria-hidden="true">
              <rect width="11" height="11" rx="3" fill={`url(#${hatchId})`} />
            </svg>
            same timestamp: passes too
          </li>
          <li className="inline-flex items-center gap-1.5">
            <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-[3px] border-[1.5px] border-dashed" style={{ borderColor: 'var(--muted)' }} />
            later: skipped
          </li>
        </ul>
      </div>
      <div className="mb-2 flex flex-wrap gap-2">
        {groups.map((g) => (
          <span key={g.split} className="inline-flex items-baseline gap-1.5 rounded-full border border-border bg-surface-2 px-2.5 py-0.5 text-[14.5px] text-ink-2">
            <span className="font-semibold text-ink">{g.split}</span>
            <span className="num">
              {g.items.length} timestamp{g.items.length === 1 ? '' : 's'} · {g.rows.toLocaleString('en-US')} rows
            </span>
          </span>
        ))}
      </div>
      <div ref={ref} className="relative w-full">
        <svg width={W} height={H} role="img" aria-label={`${task.name} rows per timestamp, split into Train, Val and Test`} className="block">
          <defs>
            <pattern id={hatchId} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="5" height="5" style={{ fill: 'var(--surface)' }} />
              <line x1="0" y1="0" x2="0" y2="5" style={{ stroke: 'var(--cat-5)' }} strokeWidth={3} />
            </pattern>
          </defs>

          {/* split brackets */}
          {groups.map((g) => {
            const a = x(new Date(g.t0)) - barW / 2 - 2;
            const b = x(new Date(g.t1)) + barW / 2 + 2;
            const yb = 34;
            const anchor = g.split === 'Train' ? 'middle' : g.split === 'Val' ? 'end' : 'start';
            const lx = g.split === 'Train' ? (a + b) / 2 : g.split === 'Val' ? b : a;
            return (
              <g key={g.split} aria-hidden="true">
                <path d={`M${a},${yb + 6} V${yb} H${b} V${yb + 6}`} fill="none" style={{ stroke: 'var(--ink-2)' }} strokeWidth={1} />
                <text x={lx} y={yb - 7} textAnchor={anchor} className="fill-[var(--ink)] text-[14px] font-semibold">
                  {g.split}
                </text>
              </g>
            );
          })}

          {/* gridlines */}
          {y.ticks(4).map((tk) => (
            <g key={tk}>
              <line x1={10} x2={W - 10} y1={y(tk)} y2={y(tk)} style={{ stroke: tk === 0 ? 'var(--axis)' : 'var(--grid)' }} strokeWidth={1} />
              {tk > 0 && (
                <text x={12} y={y(tk) - 3} className="fill-[var(--muted)] text-[13px]" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {tk}
                </text>
              )}
            </g>
          ))}

          {/* seed marker */}
          <line
            x1={x(new Date(seedT))}
            x2={x(new Date(seedT))}
            y1={M.top - 24}
            y2={y(sums.find((s) => s.t === seedT)?.rows ?? 0) - 17}
            style={{ stroke: 'var(--accent)' }}
            strokeWidth={1.5}
          />
          <text
            x={x(new Date(seedT)) + (seedSplit === 'Test' ? 4 : -4)}
            y={M.top - 16}
            textAnchor={seedSplit === 'Test' ? 'start' : 'end'}
            className="fill-[var(--accent-text)] text-[13.5px] font-semibold"
          >
            seed t
          </text>

          {/* bars */}
          {sums.map((s) => {
            const cx = x(new Date(s.t));
            const v = visOf(s.t);
            const labelled = s.split !== 'Train' || s.t === seedT;
            return (
              <g key={s.t} onPointerEnter={() => setHover(s)} onPointerLeave={() => setHover(null)}>
                <rect x={cx - Math.max(barW, 14) / 2} y={M.top} width={Math.max(barW, 14)} height={M.plotH} fill="transparent" />
                <path
                  d={vBarPath(cx - barW / 2, barW, y(0), y(s.rows), 3)}
                  style={{
                    fill: fillOf(v),
                    stroke: v === 'same' ? 'var(--cat-5)' : v === 'later' ? 'var(--muted)' : 'none',
                    strokeWidth: v === 'earlier' ? 0 : v === 'later' ? 1.5 : 1,
                    strokeDasharray: v === 'later' ? '3 2' : undefined,
                    opacity: hover && hover.t !== s.t ? 0.75 : 1,
                    transition: 'fill 200ms',
                  }}
                />
                {labelled && (
                  <text x={cx} y={y(s.rows) - 5} textAnchor="middle" className="fill-[var(--ink-2)] text-[13px]" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    {s.rows}
                  </text>
                )}
              </g>
            );
          })}
          <TimeAxis x={x} y={axisY} narrow={narrow} />
        </svg>
        {hover && (
          <TooltipBox style={{ left: Math.max(4, Math.min(W - 230, x(new Date(hover.t)) + 12)), top: M.top - 6 }}>
            <div className="text-[13.5px] text-muted">
              {hover.split} · {isoDay(hover.t)}
            </div>
            <div className="mt-0.5">
              <strong className="num text-ink">{hover.rows}</strong> <span className="text-ink-2">task rows</span>
            </div>
            <div className="text-ink-2">
              {binary ? (
                <>
                  <span className="num text-ink">{formatPct(hover.positives / Math.max(1, hover.rows), 0)}</span> with {task.targetCol} = true
                </>
              ) : (
                <>
                  mean {task.targetCol} <span className="num text-ink">{hover.mean.toFixed(2)}</span>
                </>
              )}
            </div>
            <div className="mt-1 max-w-[210px] text-[14px] leading-snug text-ink-2">{VIS_TEXT[visOf(hover.t)]}</div>
          </TooltipBox>
        )}
      </div>
    </ChartFrame>
  );
}
