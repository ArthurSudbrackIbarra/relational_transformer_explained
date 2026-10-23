/**
 * Figure: every task is "fill in the masked cell" (paper p.3 §2.2, masked token prediction).
 * Left: autocomplete hides a value that exists (reviews.rating); the rating task's leakage column
 * (review_text) is shown as dropped for the seed row, as fly.rs:457-461 does. Right: forecasting
 * asks for a value that does not exist yet (customer-churn), computed from the rows of a future
 * window. Live ShopDB rows; the forecast row and label are the engine's task-table values, and a
 * customer without a task row gets a note, never an invented row.
 */
import { AnimatePresence, motion, useInView } from 'motion/react';
import { Eye, EyeOff } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Button, SemBadge, cx, formatRaw, semOfDType, textWidth } from '@/components';
import type { PreprocessedDB, RawDatabase, TaskDef } from '@/engine/types';
import { usePrefersReducedMotion } from '@/styles/theme';
import { useElementSize } from '@/components/viz/useSize';
import {
  boolText,
  deltaMsOf,
  entityName,
  exampleTaskRow,
  firstActivity,
  forecastSpecOf,
  isoDay,
  isoSecond,
  labelEvents,
  windowEvents,
  type EntityEvent,
} from './taskData';

function HiddenCell({ masked, value, wide }: { masked: boolean; value: string; wide?: boolean }) {
  return (
    <span className={cx('relative inline-flex h-[24px] items-center', wide ? 'min-w-[52px]' : 'min-w-[30px]')}>
      <AnimatePresence mode="wait" initial={false}>
        {masked ? (
          <motion.span
            key="q"
            initial={{ opacity: 0, scale: 0.6 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.6 }}
            transition={{ duration: 0.22 }}
            className="target-outline inline-flex h-[22px] min-w-[26px] items-center justify-center rounded-[5px] bg-accent-soft px-1.5 text-[15.5px] font-bold text-accent-text"
            aria-label="masked cell"
          >
            ?
          </motion.span>
        ) : (
          <motion.span
            key="v"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.22 }}
            className="num inline-flex h-[22px] items-center rounded-[5px] px-1.5 text-[15.5px] font-semibold text-ink ring-1 ring-accent/50"
          >
            {value}
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  );
}

/**
 * A ±45-day strip around t: the past (known at t) and the label window (t, t + Δ]. While the label
 * is masked the window's orders are not drawn (the future is unknown); they appear on reveal.
 */
function FutureStrip({ t, delta, events, masked, reduced }: { t: number; delta: number; events: readonly EntityEvent[]; masked: boolean; reduced: boolean }) {
  const span = 1.5 * delta;
  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(240, size.width || 320);
  const x = (ms: number) => 8 + ((ms - (t - span)) / (2 * span)) * (W - 16);
  const shown = events.filter((e) => e.t >= t - span && e.t <= t + span);
  const inWin = (e: EntityEvent) => e.t > t && e.t <= t + delta;
  const bandW = x(t + delta) - x(t);
  const winLabel = bandW >= textWidth('future window', 13, { weight: 500 }) + 8 ? 'future window' : 'window';
  const Y = { label: 13, bandTop: 20, bandBot: 46, dot: 33, axis: 42, t: 62 };
  return (
    <div ref={ref} className="mt-3 min-w-0">
      <svg
        width={W}
        height={68}
        className="block"
        role="img"
        aria-label={`Orders around the prediction time t, and the window (t, t + ${Math.round(delta / 86_400_000)} days]${masked ? '; the orders inside the window are hidden until you reveal the label' : ''}`}
      >
        <rect x={x(t)} y={Y.bandTop} width={bandW} height={Y.bandBot - Y.bandTop} rx={3} style={{ fill: 'var(--accent-soft)' }} />
        <line x1={x(t + delta)} x2={x(t + delta)} y1={Y.bandTop} y2={Y.bandBot} style={{ stroke: 'var(--accent)' }} strokeWidth={1.2} />
        <line x1={8} x2={W - 8} y1={Y.axis} y2={Y.axis} style={{ stroke: 'var(--axis)' }} />
        <line x1={x(t)} x2={x(t)} y1={Y.bandTop - 2} y2={Y.axis + 8} style={{ stroke: 'var(--ink)' }} strokeWidth={1.5} />
        {shown
          .filter((e) => !inWin(e))
          .map((e) => (
            <circle key={e.row} cx={x(e.t)} cy={Y.dot} r={4} style={{ fill: 'var(--cat-3)', stroke: 'var(--surface)', opacity: 0.45 }} strokeWidth={1} />
          ))}
        <AnimatePresence initial={false}>
          {masked ? (
            <motion.text
              key="q"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: reduced ? 0 : 0.2 }}
              x={x(t) + bandW / 2}
              y={Y.dot + 5}
              textAnchor="middle"
              className="fill-[var(--accent-text)] text-[15px] font-bold"
            >
              ?
            </motion.text>
          ) : (
            shown.filter(inWin).map((e, i) => (
              <motion.circle
                key={e.row}
                initial={reduced ? false : { opacity: 0, scale: 0.3 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.25, delay: reduced ? 0 : 0.08 * i }}
                cx={x(e.t)}
                cy={Y.dot}
                r={4}
                style={{ fill: 'var(--cat-3)', stroke: 'var(--surface)', transformOrigin: `${x(e.t)}px ${Y.dot}px` }}
                strokeWidth={1}
              />
            ))
          )}
        </AnimatePresence>
        <text x={x(t) - 8} y={Y.label} textAnchor="end" className="fill-[var(--muted)] text-[13px]">
          known at t
        </text>
        <text x={x(t) + bandW / 2} y={Y.label} textAnchor="middle" className="fill-[var(--accent-text)] text-[13px] font-medium">
          {winLabel}
        </text>
        <text x={x(t)} y={Y.t} textAnchor="middle" className="fill-[var(--ink)] text-[13px] font-semibold">
          t
        </text>
      </svg>
    </div>
  );
}

export function MaskedCellFigure({
  db,
  pdb,
  ratingTask,
  churnTask,
  review,
  customer,
}: {
  db: RawDatabase;
  pdb: PreprocessedDB;
  ratingTask: TaskDef;
  churnTask: TaskDef;
  review: number;
  customer: number;
}) {
  const reduced = usePrefersReducedMotion();
  const ref = useRef<HTMLDivElement | null>(null);
  const inView = useInView(ref, { once: true, amount: 0.5 });
  const [masked, setMasked] = useState(reduced);
  const [touched, setTouched] = useState(false);
  useEffect(() => {
    if (!inView || touched) return;
    const id = window.setTimeout(() => setMasked(true), reduced ? 0 : 900);
    return () => window.clearTimeout(id);
  }, [inView, touched, reduced]);

  // ── autocomplete panel: a few rows of reviews around the example review ──
  const reviews = db.tables.find((t) => t.schema.name === ratingTask.table)!;
  const shownCols = ['review_time', 'rating', 'review_text'];
  const [acRef, acSize] = useElementSize<HTMLDivElement>();
  const acStacked = (acSize.width || 480) < 440;
  const rowIdx = useMemo(() => {
    const n = reviews.rows.length;
    const start = Math.max(0, Math.min(n - 4, review - 1));
    return [start, start + 1, start + 2, start + 3].filter((r) => r < n);
  }, [reviews, review]);
  // fly.rs:457-461: a leakage cell is dropped for the seed row and for rows with the seed's timestamp
  const offset = pdb.tableInfo[`${ratingTask.table}:Db`].nodeIdxOffset;
  const seedTs = pdb.nodes[offset + review]?.ts ?? null;
  const isDropped = (r: number, c: string) =>
    ratingTask.columnsToDrop.includes(c) && (r === review || (pdb.nodes[offset + r]?.ts ?? null) === seedTs);

  // ── forecast panel: the customer's own task row (Val, else Test), never an invented one ──
  const spec = forecastSpecOf(churnTask)!;
  const delta = deltaMsOf(churnTask);
  const days = churnTask.timedeltaDays ?? Math.round(delta / 86_400_000);
  const exRow = useMemo(() => exampleTaskRow(churnTask, customer), [churnTask, customer]);
  const firstOrder = useMemo(() => firstActivity(db, spec, customer), [db, spec, customer]);
  const allEvents = useMemo(() => labelEvents(db, spec, customer), [db, spec, customer]);
  const future = useMemo(() => (exRow ? windowEvents(allEvents, exRow.t, delta) : []), [allEvents, exRow, delta]);
  const customersT = db.tables.find((x) => x.schema.name === churnTask.entityTable)!;
  const cust = customersT.rows[customer];

  const dtypeOf = (c: string) => reviews.schema.columns.find((x) => x.name === c)!.dtype;
  const leak = ratingTask.columnsToDrop;

  return (
    <div ref={ref} className="min-w-0">
      <div className="grid gap-4 md:grid-cols-2">
        {/* autocomplete */}
        <div ref={acRef} className="card flex min-w-0 flex-col overflow-hidden" style={{ borderLeft: '3px solid var(--cat-7)' }}>
          <div className="border-b border-grid px-4 py-3">
            <div className="eyebrow">Autocomplete</div>
            <div className="mt-0.5 text-[17px] font-semibold text-ink">The value exists. We hide it.</div>
          </div>
          {/* the reviews around the seed: time (to the second, the sampler's resolution), rating, review_text */}
          <div
            className={cx('grid gap-x-3 px-3 pt-2 pb-3', acStacked ? 'grid-cols-[minmax(0,1fr)_auto]' : 'grid-cols-[auto_auto_minmax(0,1fr)]')}
            role="table"
            aria-label={`Reviews ${rowIdx[0]} to ${rowIdx[rowIdx.length - 1]}`}
          >
            <div role="rowgroup" className="contents">
              <div
                role="row"
                className={cx(
                  'grid grid-cols-subgrid items-center border-b border-grid px-1.5 pb-1.5 text-[14px] text-ink-2',
                  acStacked ? 'col-span-2' : 'col-span-3',
                )}
              >
                {shownCols.map((c) => (
                  <span
                    key={c}
                    role="columnheader"
                    className={cx('inline-flex min-w-0 items-center gap-1 font-medium', acStacked && c === 'review_text' && 'col-span-2')}
                  >
                    <SemBadge sem={semOfDType(dtypeOf(c))} size="sm" />
                    <span>{c}</span>
                  </span>
                ))}
              </div>
            </div>
            <div role="rowgroup" className="contents">
              {rowIdx.map((r) => {
                const rec = reviews.rows[r];
                const isSeed = r === review;
                return (
                  <div
                    key={r}
                    role="row"
                    className={cx(
                      'grid grid-cols-subgrid items-start gap-y-0.5 border-b border-grid/70 px-1.5 py-1.5 text-[15px] last:border-0',
                      acStacked ? 'col-span-2' : 'col-span-3',
                      isSeed && 'rounded-md bg-surface-2',
                    )}
                  >
                    {shownCols.map((c) => {
                      const dropped = isDropped(r, c);
                      const v = rec[c];
                      return (
                        <span key={c} role="cell" className={cx('min-w-0', acStacked && c === 'review_text' && 'col-span-2')}>
                          {c === 'review_time' ? (
                            <span className="num text-ink">
                              <span className="mr-1.5 text-[14px] whitespace-nowrap text-muted">#{r}</span>
                              {typeof v === 'number' ? (
                                <>
                                  <span className="whitespace-nowrap">{isoSecond(v).slice(0, 10)}</span>{' '}
                                  <span className="whitespace-nowrap">{isoSecond(v).slice(11)}</span>
                                </>
                              ) : (
                                formatRaw(v, dtypeOf(c))
                              )}
                            </span>
                          ) : isSeed && c === ratingTask.targetCol ? (
                            <HiddenCell masked={masked} value={formatRaw(v, dtypeOf(c))} />
                          ) : c === 'review_text' ? (
                            <span
                              className={cx(
                                'block leading-snug',
                                dropped ? 'text-muted line-through decoration-[var(--critical)]' : 'line-clamp-2 text-ink-2',
                              )}
                              title={
                                dropped
                                  ? 'Leakage column: this cell is dropped from the context (fly.rs:457-461)'
                                  : typeof v === 'string'
                                    ? v
                                    : undefined
                              }
                            >
                              {typeof v === 'string' ? `“${v}”` : formatRaw(v, 'str')}
                              {dropped && <span className="sr-only"> (dropped: leakage column)</span>}
                            </span>
                          ) : (
                            <span className="num text-ink">{formatRaw(v, dtypeOf(c))}</span>
                          )}
                        </span>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </div>
          <dl className="mx-4 mb-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 rounded-lg border border-border bg-surface-2 px-3 py-2 text-[15px]">
            <dt className="text-ink-2">seed row</dt>
            <dd className="num text-ink">reviews[{review}]</dd>
            <dt className="text-ink-2">target cell</dt>
            <dd className="text-ink">
              <span>{ratingTask.targetCol}</span>, masked
            </dd>
            {leak.length > 0 && (
              <>
                <dt className="text-ink-2">leakage column</dt>
                <dd className="text-ink">
                  <span className="line-through decoration-[var(--critical)]">{leak.join(', ')}</span>, dropped
                </dd>
              </>
            )}
          </dl>
          <p className="mt-auto border-t border-grid px-4 py-2.5 text-[15.5px] leading-snug text-ink-2">
            <span className="text-ink">reviews.rating</span> of row {review}: the answer sits in the database already. The
            model sees the rest of the row, minus the leakage column{leak.length === 1 ? '' : 's'} that would give the answer away
            (section 4), and the row’s neighborhood.
          </p>
        </div>

        {/* forecasting */}
        <div className="card flex min-w-0 flex-col overflow-hidden" style={{ borderLeft: '3px solid var(--cat-5)' }}>
          <div className="border-b border-grid px-4 py-3">
            <div className="eyebrow">Forecasting</div>
            <div className="mt-0.5 text-[17px] font-semibold text-ink">The value does not exist yet.</div>
          </div>
          <div className="px-4 py-3">
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[15px]">
              <span className="inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: 'var(--cat-1)' }} aria-hidden="true" />
              <span className="text-ink-2">customers[{customer}]</span>
              <span className="text-ink">{entityName(db, churnTask.entityTable!, customer)}</span>
              <span className="num text-muted">
                {String(cust?.country ?? '∅')} · age {formatRaw(cust?.age ?? null)} · premium {formatRaw(cust?.is_premium ?? null)}
              </span>
            </div>
            {exRow ? (
              <>
                <div className="scroll-quiet mt-3 overflow-x-auto rounded-lg border border-border">
                  <table className="w-full border-collapse text-[15px]">
                    <thead>
                      <tr className="bg-surface-2 text-left text-[14px] text-ink-2">
                        <th className="px-2.5 py-1 font-medium">timestamp</th>
                        <th className="px-2.5 py-1 font-medium">customer_id</th>
                        <th className="px-2.5 py-1 font-medium">churn</th>
                        <th className="px-2.5 py-1 text-right font-medium">split</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="num px-2.5 py-1.5 whitespace-nowrap text-ink">{isoDay(exRow.t)}</td>
                        <td className="num px-2.5 py-1.5 text-ink-2">{customer}</td>
                        <td className="px-2.5 py-1.5">
                          <HiddenCell masked={masked} value={boolText(exRow.label)} wide />
                        </td>
                        <td className="px-2.5 py-1.5 text-right">
                          <span className="rounded bg-surface-3 px-1.5 text-[13.5px] font-medium text-ink-2">{exRow.split}</span>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                {exRow.split === 'Test' && (
                  <p className="mt-2 text-[14.5px] leading-snug text-ink-2">
                    Customer {customer} has no Val row: their first order ({isoDay(firstOrder)}) came after the validation cutoff, so
                    this is their Test row.
                  </p>
                )}
                <FutureStrip t={exRow.t} delta={delta} events={allEvents} masked={masked} reduced={reduced} />
                <p className="mt-1 text-[15px] leading-snug text-ink-2">
                  The label is computed from customer {customer}’s orders in{' '}
                  <span className="num whitespace-nowrap text-ink">
                    ({isoDay(exRow.t)}, {isoDay(exRow.t + delta)}]
                  </span>
                  , the {days} days after t:{' '}
                  <span className="num font-semibold whitespace-nowrap text-ink">
                    {masked ? '? orders' : `${future.length} order${future.length === 1 ? '' : 's'}`}
                  </span>
                  .
                </p>
              </>
            ) : (
              <p className="mt-3 rounded-lg border border-dashed border-border-strong px-3 py-2.5 text-[15px] leading-snug text-ink-2">
                Customer {customer} has <strong className="text-ink">no row</strong> in customer-churn.{' '}
                {Number.isFinite(firstOrder)
                  ? `Their first order is on ${isoDay(firstOrder)}, after the last prediction timestamp (${isoDay(churnTask.testTimestamp!)}), so the task table never includes them.`
                  : 'They never place an order, so the task table never includes them.'}{' '}
                Pick another customer in the next section to see a task row here.
              </p>
            )}
          </div>
          <p className="mt-auto border-t border-grid px-4 py-2.5 text-[15.5px] leading-snug text-ink-2">
            No ShopDB table stores “will this customer churn?”. The label is built from future rows and kept in a new{' '}
            <span className="text-ink">customer-churn</span> table.
          </p>
        </div>
      </div>
      <div className="mt-3 flex justify-center">
        <Button
          size="sm"
          icon={masked ? <Eye size={15} aria-hidden="true" /> : <EyeOff size={15} aria-hidden="true" />}
          onClick={() => {
            setTouched(true);
            setMasked((m) => !m);
          }}
        >
          {masked ? 'Reveal the hidden values' : touched ? 'Hide them again' : 'Hide the values'}
        </Button>
      </div>
    </div>
  );
}
