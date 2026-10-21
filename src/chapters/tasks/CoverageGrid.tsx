/**
 * Step (e) of the customer-churn builder: "repeat for every active customer". One row per
 * customer, one column per prediction timestamp (chronological; the last two are Val and Test).
 * A cell is drawn only where the engine's task table HAS a row (TaskDef.tables): solid = churn true,
 * hollow = churn false; grey = no row. Columns fade in left to right, so the table visibly fills up.
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
  const top = 40;
  // one 4 px line per entity, 1 px apart, and a 3 px break after every 10 entities
  const cellH = 4;
  const pitch = 5;
  const groupGap = 3;
  const yOf = (e: number) => top + e * pitch + Math.floor(e / 10) * groupGap;
  const gridH = yOf(Math.max(0, entityCount - 1)) + cellH - top;
  // a small gap before the Val column and before the Test column
  const splitGap = 6;
  const nGaps = stamps.reduce((a, s, k) => a + (k > 0 && s.split !== stamps[k - 1].split ? 1 : 0), 0);
  const colW = (W - gutter - 2 - nGaps * splitGap) / Math.max(1, stamps.length);
  const H = top + gridH + 6;

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

  const xOf = (k: number) => {
    let gaps = 0;
    for (let j = 1; j <= k; j++) if (stamps[j].split !== stamps[j - 1].split) gaps++;
    return gutter + k * colW + gaps * splitGap;
  };
  const trainK = stamps.map((s, k) => (s.split === 'Train' ? k : -1)).filter((k) => k >= 0);
  const splitK = (sp: Split) => stamps.findIndex((s) => s.split === sp);
  const kVal = splitK('Val');
  const kTest = splitK('Test');
  // narrow columns: centred "Val" and "Test" would collide (or "Test" would pass the right edge)
  const tight = colW < 25;

  return (
    <div ref={ref} className="min-w-0">
      <svg
        width={W}
        height={H}
        className="block"
        role="img"
        aria-label={`${task.name}: ${cells.length} rows, one per active customer and timestamp. Rows are customers, columns are the ${stamps.length} timestamps: ${trainK.length} Train, then Val and Test.`}
      >
        {/* split labels above the columns: Train with a bracket on the upper line, val and test on the lower line */}
        {trainK.length > 0 && (
          <g aria-hidden="true">
            <path
              d={`M${xOf(trainK[0]) + 1},${top - 18} V${top - 22} H${xOf(trainK[trainK.length - 1]) + colW - 1} V${top - 18}`}
              fill="none"
              style={{ stroke: 'var(--ink-2)' }}
            />
            <text x={(xOf(trainK[0]) + xOf(trainK[trainK.length - 1]) + colW) / 2} y={top - 26} textAnchor="middle" className="fill-[var(--ink-2)] text-[13px] font-medium">
              Train
            </text>
          </g>
        )}
        {kVal >= 0 && (
          <text x={xOf(kVal) + colW / 2} y={top - 5} textAnchor="middle" className="fill-[var(--ink-2)] text-[13px] font-medium" aria-hidden="true">
            Val
          </text>
        )}
        {kTest >= 0 && (
          // narrow columns: "Test" moves up to the Train line, right-aligned, so it never touches "Val" or the edge
          <text
            x={tight ? xOf(kTest) + colW : xOf(kTest) + colW / 2}
            y={tight ? top - 26 : top - 5}
            textAnchor={tight ? 'end' : 'middle'}
            className="fill-[var(--ink-2)] text-[13px] font-medium"
            aria-hidden="true"
          >
            Test
          </text>
        )}
        <text x={gutter - 6} y={top + 8} textAnchor="end" className="fill-[var(--muted)] text-[13px]" style={{ fontVariantNumeric: 'tabular-nums' }} aria-hidden="true">
          #0
        </text>
        <text x={gutter - 6} y={top + gridH} textAnchor="end" className="fill-[var(--muted)] text-[13px]" style={{ fontVariantNumeric: 'tabular-nums' }} aria-hidden="true">
          #{entityCount - 1}
        </text>

        {/* "no row" background: every (customer, timestamp) slot starts grey; a task row covers it */}
        {stamps.map((s, k) => (
          <g key={`bg-${s.t}`} aria-hidden="true">
            {Array.from({ length: Math.ceil(entityCount / 10) }, (_, g) => {
              const e0 = g * 10;
              const e1 = Math.min(entityCount, e0 + 10) - 1;
              return (
                <rect
                  key={g}
                  x={xOf(k) + 1}
                  y={yOf(e0)}
                  width={Math.max(1, colW - 2)}
                  height={yOf(e1) + cellH - yOf(e0)}
                  style={{ fill: 'var(--axis)', opacity: 0.55 }}
                />
              );
            })}
          </g>
        ))}

        {byCol.map((col, k) => (
          <motion.g
            key={stamps[k].t}
            initial={reduced ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.25, delay: reduced ? 0 : 0.15 + k * 0.08 }}
          >
            <title>{`${stamps[k].split} · ${isoDay(stamps[k].t)}: ${col.length} rows`}</title>
            {col.map((c) =>
              c.v ? (
                <rect key={c.e} x={xOf(k) + 1} y={yOf(c.e) - 0.5} width={Math.max(1, colW - 2)} height={cellH + 1} style={{ fill: 'var(--cat-5)' }} />
              ) : (
                // churn = false: hollow, like the hollow circle in the timeline above
                <rect
                  key={c.e}
                  x={xOf(k) + 1.5}
                  y={yOf(c.e)}
                  width={Math.max(1, colW - 3)}
                  height={cellH}
                  style={{ fill: 'var(--surface)', stroke: 'var(--cat-5)' }}
                  strokeWidth={1}
                />
              ),
            )}
          </motion.g>
        ))}
        {/* the followed customer */}
        <rect
          x={gutter - 2}
          y={yOf(entity) - 2}
          width={W - gutter + 1}
          height={cellH + 4}
          rx={1.5}
          style={{ fill: 'none', stroke: 'var(--accent)' }}
          strokeWidth={1.5}
        />
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[14px] text-ink-2" aria-label="Legend">
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3 w-3 rounded-[2px]" style={{ background: 'var(--cat-5)' }} />
          churn = true
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3 w-3 rounded-[2px] border-[1.5px]" style={{ borderColor: 'var(--cat-5)', background: 'var(--surface)' }} />
          churn = false
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3 w-3 rounded-[2px]" style={{ background: 'var(--axis)', opacity: 0.55 }} />
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
