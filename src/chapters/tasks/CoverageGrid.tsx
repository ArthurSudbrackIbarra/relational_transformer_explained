/**
 * Step (e) of the customer-churn builder: "repeat for every active customer". One row per
 * customer, one column per prediction timestamp (chronological; the last two are Val and Test).
 * A cell is drawn only where the engine's task table HAS a row (TaskDef.tables); its shade is the
 * stored label. Columns fade in left to right, so the table visibly fills up across customers.
 */
import { motion } from 'motion/react';
import { useMemo } from 'react';
import { useElementSize } from '@/components/viz/useSize';
import type { Split, TaskDef } from '@/engine/types';
import { usePrefersReducedMotion } from '@/styles/theme';
import { SPLITS, isoDay, type StampInfo } from './taskData';

export function CoverageGrid({
  task,
  stamps,
  entityCount,
  entity,
}: {
  task: TaskDef;
  stamps: readonly StampInfo[];
  entityCount: number;
  /** The customer followed in the builder (outlined). */
  entity: number;
}) {
  const reduced = usePrefersReducedMotion();
  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(240, size.width || 400);
  const gutter = 34;
  const top = 22;
  const rowH = Math.max(2.5, Math.min(3.5, 280 / Math.max(1, entityCount)));
  const gridH = rowH * entityCount;
  const colW = (W - gutter - 2) / Math.max(1, stamps.length);
  const H = top + gridH + 24;

  const cells = useMemo(() => {
    const out: { k: number; e: number; v: boolean }[] = [];
    const col = new Map(stamps.map((s, k) => [`${s.split}|${s.t}`, k]));
    for (const sp of SPLITS) {
      for (const r of task.tables![sp].rows) {
        const k = col.get(`${sp}|${r.timestamp as number}`);
        if (k === undefined) continue;
        out.push({ k, e: r[task.entityCol!] as number, v: r[task.targetCol] === true });
      }
    }
    return out;
  }, [task, stamps]);
  const byCol = useMemo(() => {
    const g: { e: number; v: boolean }[][] = stamps.map(() => []);
    for (const c of cells) g[c.k].push(c);
    return g;
  }, [cells, stamps]);

  const trainK = stamps.map((s, k) => (s.split === 'Train' ? k : -1)).filter((k) => k >= 0);
  const xOf = (k: number) => gutter + k * colW;
  const splitK = (sp: Split) => stamps.findIndex((s) => s.split === sp);

  return (
    <div ref={ref} className="min-w-0">
      <svg
        width={W}
        height={H}
        className="block"
        role="img"
        aria-label={`${task.name}: ${cells.length} rows, one per active customer and timestamp. Rows are customers, columns are the ${stamps.length} timestamps.`}
      >
        {/* split labels above the columns */}
        {trainK.length > 0 && (
          <g aria-hidden="true">
            <path
              d={`M${xOf(trainK[0]) + 1},${top - 4} V${top - 8} H${xOf(trainK[trainK.length - 1]) + colW - 1} V${top - 4}`}
              fill="none"
              style={{ stroke: 'var(--ink-2)' }}
            />
            <text x={(xOf(trainK[0]) + xOf(trainK[trainK.length - 1]) + colW) / 2} y={top - 11} textAnchor="middle" className="fill-[var(--ink-2)] text-[13px] font-medium">
              Train
            </text>
          </g>
        )}
        {(['Val', 'Test'] as Split[]).map((sp) => {
          const k = splitK(sp);
          if (k < 0) return null;
          return (
            <text
              key={sp}
              x={xOf(k) + colW / 2}
              y={sp === 'Val' ? top - 6 : top + gridH + 16}
              textAnchor="middle"
              className="fill-[var(--ink-2)] text-[13px] font-medium"
              aria-hidden="true"
            >
              {sp.toLowerCase()}
            </text>
          );
        })}
        <text x={gutter - 6} y={top + 10} textAnchor="end" className="fill-[var(--muted)] text-[13px]" style={{ fontVariantNumeric: 'tabular-nums' }} aria-hidden="true">
          #0
        </text>
        <text x={gutter - 6} y={top + gridH} textAnchor="end" className="fill-[var(--muted)] text-[13px]" style={{ fontVariantNumeric: 'tabular-nums' }} aria-hidden="true">
          #{entityCount - 1}
        </text>
        <rect x={gutter} y={top} width={W - gutter - 2} height={gridH} style={{ fill: 'none', stroke: 'var(--grid)' }} />

        {byCol.map((col, k) => (
          <motion.g
            key={stamps[k].t}
            initial={reduced ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.25, delay: reduced ? 0 : 0.15 + k * 0.08 }}
          >
            <title>{`${stamps[k].split} · ${isoDay(stamps[k].t)}: ${col.length} rows`}</title>
            {col.map((c) => (
              <rect
                key={c.e}
                x={xOf(k) + 1}
                y={top + c.e * rowH}
                width={Math.max(1, colW - 2)}
                height={rowH}
                style={{ fill: 'var(--cat-5)', opacity: c.v ? 1 : 0.3 }}
              />
            ))}
          </motion.g>
        ))}
        {/* the followed customer */}
        <rect
          x={gutter - 2}
          y={top + entity * rowH - 1.5}
          width={W - gutter + 1}
          height={rowH + 3}
          rx={1.5}
          style={{ fill: 'none', stroke: 'var(--accent)' }}
          strokeWidth={1.5}
        />
      </svg>
      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[14px] text-ink-2" aria-label="Legend">
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3 w-3 rounded-[2px]" style={{ background: 'var(--cat-5)' }} />
          churn = true
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3 w-3 rounded-[2px]" style={{ background: 'var(--cat-5)', opacity: 0.3 }} />
          false
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3 w-3 rounded-[2px] border border-[var(--grid)]" />
          no row
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3 w-4 rounded-[2px] border-[1.5px] border-[var(--accent)]" />
          customer #{entity}
        </span>
      </div>
    </div>
  );
}
