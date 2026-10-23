/**
 * Figure: building the customer-churn task table step by step (spec §2.1-2.2).
 *
 *  (a) timestamps every Δ going back from the val cutoff, (b) the window (t, t+Δ] on one
 *  customer's order timeline, (c) churn = no order in the window, (d) append the row,
 *  (e) repeat for every active customer.
 *
 * Timestamps and stored labels come from the engine's task table (TaskDef.tables); the order
 * counts are re-derived from the raw orders table to show WHICH rows produced each label, and are
 * cross-checked against the stored label.
 */
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useMemo, useRef } from 'react';
import { Link2 } from 'lucide-react';
import { InlineMath, SemBadge, Select, StatusBadge, StepPlayer, cx, useStepPlayer } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { SemType, type RawDatabase, type Split, type TaskDef } from '@/engine/types';
import { CoverageGrid } from './CoverageGrid';
import {
  SPLITS,
  aggregateWindow,
  allStamps,
  boolText,
  deltaMsOf,
  entityName,
  firstActivity,
  forecastSpecOf,
  isoDay,
  isoMinute,
  labelEvents,
  labelMatches,
  windowEvents,
  type EntityEvent,
} from './taskData';
import { LaneLabel, TaskMarker, TimeAxis, monthDomain, useTimeScale } from './timeline';

type BStep =
  | { kind: 'cutoffs' }
  | { kind: 'train-first' }
  | { kind: 'train-all' }
  | { kind: 'inactive'; i: number }
  | { kind: 'window'; i: number }
  | { kind: 'label'; i: number }
  | { kind: 'append'; i: number }
  | { kind: 'repeat'; i: number }
  | { kind: 'all' };

interface StampCalc {
  t: number;
  split: Split;
  active: boolean;
  inWindow: EntityEvent[];
  computed: boolean | number;
  /** Engine's stored label for (entity, t), undefined when the entity has no row at t. */
  stored: boolean | number | undefined;
  rowPos: number | undefined;
}

const SPLIT_TAG: Record<Split, string> = { Train: 'Train', Val: 'Val', Test: 'Test' };

export function ChurnBuilder({
  db,
  task,
  customer,
  onCustomerChange,
}: {
  db: RawDatabase;
  task: TaskDef;
  customer: number;
  onCustomerChange: (c: number) => void;
}) {
  const spec = forecastSpecOf(task)!;
  const delta = deltaMsOf(task);
  const stamps = useMemo(() => allStamps(task), [task]);
  const val = task.valTimestamp!;
  const test = task.testTimestamp!;
  const trainStamps = stamps.filter((s) => s.split === 'Train');
  const customers = db.tables.find((t) => t.schema.name === task.entityTable)!;
  const days = task.timedeltaDays ?? Math.round(delta / 86_400_000);

  // every row of the finished table (all splits, all customers), chronologically, for step (e)
  const allRows = useMemo(() => {
    const out: { split: Split; t: number; e: number; label: boolean | number }[] = [];
    for (const sp of SPLITS)
      for (const r of task.tables![sp].rows)
        out.push({ split: sp, t: r.timestamp as number, e: r[task.entityCol!] as number, label: r[task.targetCol] as boolean | number });
    return out.sort((a, b) => a.t - b.t || a.e - b.e);
  }, [task]);

  const calc: StampCalc[] = useMemo(() => {
    const ev = labelEvents(db, spec, customer);
    const first = firstActivity(db, spec, customer);
    return stamps.map((s) => {
      const inWindow = windowEvents(ev, s.t, delta);
      const rows = task.tables![s.split].rows;
      const rowPos = rows.findIndex((r) => r.timestamp === s.t && r[task.entityCol!] === customer);
      const stored = rowPos >= 0 ? (rows[rowPos][task.targetCol] as boolean | number) : undefined;
      return {
        t: s.t,
        split: s.split,
        active: first <= s.t,
        inWindow,
        computed: aggregateWindow(spec, inWindow),
        stored,
        rowPos: rowPos >= 0 ? rowPos : undefined,
      };
    });
  }, [db, spec, customer, stamps, delta, task]);

  const events = useMemo(() => labelEvents(db, spec, customer), [db, spec, customer]);
  const allMatch = calc.every((c) => (c.active ? c.stored !== undefined && labelMatches(c.computed, c.stored) : c.stored === undefined));

  const steps: BStep[] = useMemo(() => {
    const out: BStep[] = [{ kind: 'cutoffs' }, { kind: 'train-first' }, { kind: 'train-all' }];
    let firstDone = false;
    calc.forEach((c, i) => {
      if (!c.active) out.push({ kind: 'inactive', i });
      else if (!firstDone) {
        firstDone = true;
        out.push({ kind: 'window', i }, { kind: 'label', i }, { kind: 'append', i });
      } else out.push({ kind: 'repeat', i });
    });
    out.push({ kind: 'all' });
    return out;
  }, [calc]);

  const player = useStepPlayer(steps.length, { intervalMs: 1500 });
  const step = steps[player.index] ?? steps[0];

  // ── derived view state for the current step ───────────────────────────
  const cur = 'i' in step ? step.i : step.kind === 'all' ? calc.length - 1 : -1;
  const timestampsShown =
    step.kind === 'cutoffs' ? 'cutoffs' : step.kind === 'train-first' ? 'first' : ('all' as 'cutoffs' | 'first' | 'all');
  const processedUpTo = (() => {
    // stamps whose row (or "no row") is final at this step
    if (step.kind === 'all') return calc.length - 1;
    if (!('i' in step)) return -1;
    if (step.kind === 'window' || step.kind === 'label') return step.i - 1;
    return step.i;
  })();
  const windowStamp = step.kind === 'window' || step.kind === 'label' || step.kind === 'append' || step.kind === 'repeat' ? step.i : -1;
  const labelVisible = (i: number) => i <= processedUpTo || (step.kind === 'label' && step.i === i);
  const appended = calc
    .map((c, i) => ({ c, i }))
    .filter(({ c, i }) => c.active && i <= processedUpTo);

  // ── geometry ─────────────────────────────────────────────────────────────
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(300, size.width || 760);
  const narrow = W < 560;
  // wide: lane titles sit in a left gutter, so the window band never runs under them
  const gutter = W >= 640 ? 124 : 0;
  const H = 196;
  const domain = useMemo(() => monthDomain(db.minTimestamp, db.maxTimestamp), [db]);
  const x = useTimeScale(domain, gutter + 10, W - 10);
  const Y = { stampLabel: 13, stamp: 38, orderLabel: 66, order: 86, labelLabel: 118, label: 138, axis: 160, bandTop: 24, bandBot: 100 };

  const [ruleRef, ruleSize] = useElementSize<HTMLDivElement>();
  const twoLineRule = (ruleSize.width || 480) < 470;

  const isAll = step.kind === 'all';
  const compact = W < 640;
  const tableScroll = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = tableScroll.current;
    if (!el) return;
    if (!isAll) {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
      return;
    }
    // all customers: center the followed customer's latest row
    const rows = el.querySelectorAll<HTMLElement>('[data-own-row]');
    const last = rows[rows.length - 1];
    if (last) {
      const top = last.getBoundingClientRect().top - el.getBoundingClientRect().top + el.scrollTop;
      el.scrollTo({ top: Math.max(0, top - el.clientHeight / 2) });
    }
  }, [appended.length, isAll, customer]);

  const name = entityName(db, task.entityTable!, customer);
  const entityTitle = `customer #${customer}`;

  const describe = (idx: number) => {
    const s = steps[idx];
    if (!s) return null;
    const c = 'i' in s ? calc[s.i] : undefined;
    const win = c ? `(${isoDay(c.t)}, ${isoDay(c.t + delta)}]` : '';
    switch (s.kind) {
      case 'cutoffs':
        return (
          <>
            The dataset fixes two cutoff dates: the <strong>validation timestamp {isoDay(val)}</strong> and the{' '}
            <strong>test timestamp {isoDay(test)}</strong>. Each evaluation split gets one prediction time there.
          </>
        );
      case 'train-first':
        return (
          <>
            Training timestamps step back from the validation cutoff by Δ = {task.timedeltaDays} days. One step back from the val cutoff
            gives <strong>{isoDay(trainStamps[trainStamps.length - 1]?.t ?? val)}</strong>, the latest training timestamp.
          </>
        );
      case 'train-all':
        return (
          <>
            …and keep stepping back by Δ: <strong>{trainStamps.length} training timestamps</strong>, from {isoDay(trainStamps[0]?.t ?? val)} to{' '}
            {isoDay(trainStamps[trainStamps.length - 1]?.t ?? val)}.
          </>
        );
      case 'inactive':
        return (
          <>
            At t = {isoDay(c!.t)}, {entityTitle} has not placed an order yet, so the table gets <strong>no row</strong> for them at this
            time.
          </>
        );
      case 'window':
        return (
          <>
            Take t = <strong>{isoMinute(c!.t)}</strong> UTC. The label looks only at the future window{' '}
            <span className="num">{win}</span>: <strong>{c!.inWindow.length}</strong> order{c!.inWindow.length === 1 ? '' : 's'} of{' '}
            {entityTitle} fall inside. Since t is midnight, an order later that same day is already in the future.
          </>
        );
      case 'label':
        return (
          <>
            churn is true exactly when the window holds no order. Here it holds {c!.inWindow.length}, so{' '}
            <strong>churn = {boolText(c!.computed as boolean)}</strong>.
          </>
        );
      case 'append':
        return (
          <>
            Append the row <span className="num">(timestamp = {isoDay(c!.t)}, customer_id = {customer}, churn = {boolText(c!.computed as boolean)})</span>{' '}
            to the <strong>{c!.split}</strong> split.
          </>
        );
      case 'repeat':
        return (
          <>
            t = {isoDay(c!.t)}: {c!.inWindow.length} order{c!.inWindow.length === 1 ? '' : 's'} in <span className="num">{win}</span> → churn ={' '}
            <strong>{boolText(c!.computed as boolean)}</strong>, appended to the {c!.split} split.
          </>
        );
      case 'all':
        return (
          <>
            Do the same for <strong>every customer</strong> who is active at each t (in the grid: one line per customer, one column per
            timestamp). The finished table holds{' '}
            {(['Train', 'Val', 'Test'] as Split[]).map((sp, k) => (
              <span key={sp}>
                {k > 0 ? (k === 2 ? ' and ' : ', ') : ''}
                <strong>{task.tables![sp].rows.length}</strong> {sp} rows
              </span>
            ))}
            .
          </>
        );
    }
  };

  const playerEl = (
    <StepPlayer {...player} className={compact ? 'mt-3' : 'mt-4'} unit="step" label="Build the customer-churn task table" describe={describe} />
  );

  const markerVisible = (sp: Split, t: number) => {
    if (sp !== 'Train') return true;
    if (timestampsShown === 'cutoffs') return false;
    if (timestampsShown === 'first') return t === trainStamps[trainStamps.length - 1]?.t;
    return true;
  };

  const c = cur >= 0 ? calc[cur] : undefined;

  /** Lane title: in the left gutter when there is room, else just above the lane. */
  const lane = (lines: string[], cy: number, aboveY: number) =>
    gutter > 0 ? (
      <text x={0} y={cy - (lines.length - 1) * 8 + 4} className="fill-[var(--ink-2)] text-[13.5px] font-medium">
        {lines.map((l, i) => (
          <tspan key={i} x={0} dy={i === 0 ? 0 : 16}>
            {l}
          </tspan>
        ))}
      </text>
    ) : (
      <LaneLabel x={10} y={aboveY}>
        {lines.join(' ')}
      </LaneLabel>
    );

  return (
    <div className="min-w-0">
      <div className="mb-4 flex flex-wrap items-end gap-x-4 gap-y-3">
        <Select
          label="Customer"
          value={customer}
          onChange={(e) => onCustomerChange(Number(e.target.value))}
          className="w-[250px] max-w-full"
        >
          {customers.rows.map((r, i) => (
            <option key={i} value={i}>
              #{i} · {String(r.name ?? '')} ({String(r.country ?? '∅')})
            </option>
          ))}
        </Select>
        <div className="flex flex-wrap items-center gap-2 pb-1">
          <StatusBadge status={allMatch ? 'good' : 'warning'}>
            {allMatch ? 'Recomputed labels match the engine’s task table' : 'A recomputed label differs from the task table'}
          </StatusBadge>
        </div>
      </div>

      <div ref={wrapRef} className="min-w-0">
        <svg
          width={W}
          height={H}
          viewBox={`0 0 ${W} ${H}`}
          role="group"
          aria-label={`Timeline of ${entityTitle}'s orders and the customer-churn prediction timestamps. Each timestamp marker is a button that jumps to its step.`}
          className="block"
        >
          {/* window band (t, t+Δ] */}
          <AnimatePresence>
            {windowStamp >= 0 && (
              <motion.g
                key="band"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.25 }}
              >
                <motion.rect
                  initial={false}
                  animate={{ x: x(calc[windowStamp].t), width: Math.max(2, x(calc[windowStamp].t + delta) - x(calc[windowStamp].t)) }}
                  transition={{ type: 'spring', stiffness: 260, damping: 30 }}
                  y={Y.bandTop}
                  height={Y.bandBot - Y.bandTop}
                  rx={3}
                  style={{ fill: 'var(--accent-soft)' }}
                />
                <motion.line
                  initial={false}
                  animate={{ x1: x(calc[windowStamp].t), x2: x(calc[windowStamp].t) }}
                  transition={{ type: 'spring', stiffness: 260, damping: 30 }}
                  y1={Y.bandTop}
                  y2={Y.bandBot}
                  strokeDasharray="3 3"
                  style={{ stroke: 'var(--accent)' }}
                  strokeWidth={1.5}
                />
                <motion.line
                  initial={false}
                  animate={{ x1: x(calc[windowStamp].t + delta), x2: x(calc[windowStamp].t + delta) }}
                  transition={{ type: 'spring', stiffness: 260, damping: 30 }}
                  y1={Y.bandTop}
                  y2={Y.bandBot}
                  style={{ stroke: 'var(--accent)' }}
                  strokeWidth={1.5}
                />
              </motion.g>
            )}
          </AnimatePresence>

          {/* lane: prediction timestamps */}
          {lane(['Prediction', 'timestamps t'], Y.stamp, Y.stampLabel)}
          <AnimatePresence>
          {calc.map((s, i) => {
            const show = markerVisible(s.split, s.t);
            const isCur = i === cur && step.kind !== 'all';
            const trainK = trainStamps.findIndex((ts) => ts.t === s.t);
            const delay = timestampsShown === 'all' && s.split === 'Train' ? (trainStamps.length - 1 - trainK) * 0.06 : 0;
            if (!show) return null;
            return (
                  <motion.g
                    key={s.t}
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3, delay }}
                    role="button"
                    tabIndex={0}
                    aria-label={`${s.split} timestamp ${isoDay(s.t)}: jump to this step`}
                    className="cursor-pointer"
                    onClick={() => {
                      const k = steps.findIndex((st) => 'i' in st && st.i === i && st.kind !== 'label' && st.kind !== 'append');
                      if (k >= 0) player.setIndex(k);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        const k = steps.findIndex((st) => 'i' in st && st.i === i && st.kind !== 'label' && st.kind !== 'append');
                        if (k >= 0) player.setIndex(k);
                      }
                    }}
                  >
                    <rect x={x(s.t) - 9} y={Y.stamp - 12} width={18} height={24} fill="transparent" />
                    {isCur && <circle cx={x(s.t)} cy={Y.stamp} r={10} style={{ fill: 'none', stroke: 'var(--accent)' }} strokeWidth={2} />}
                    <TaskMarker cx={x(s.t)} cy={Y.stamp} r={s.split === 'Train' ? 5 : 6} />
                    {s.split !== 'Train' && (
                      <text
                        x={x(s.t) + (s.split === 'Val' ? -4 : 4)}
                        y={Y.stamp - 11}
                        textAnchor={s.split === 'Val' ? 'end' : 'start'}
                        className="fill-[var(--ink)] text-[13px] font-semibold"
                      >
                        {s.split === 'Val' ? 'val' : 'test'}
                      </text>
                    )}
                  </motion.g>
            );
          })}
          </AnimatePresence>

          {/* lane: the customer's orders */}
          {lane(['Orders of', entityTitle], Y.order, Y.orderLabel)}
          {events.map((e) => {
            const inWin = windowStamp >= 0 && e.t > calc[windowStamp].t && e.t <= calc[windowStamp].t + delta;
            return (
              <circle
                key={`${e.table}-${e.row}`}
                cx={x(e.t)}
                cy={Y.order}
                r={inWin ? 5 : 4}
                style={{
                  fill: 'var(--cat-3)',
                  stroke: inWin ? 'var(--ink)' : 'var(--surface)',
                  opacity: windowStamp >= 0 && !inWin ? 0.35 : 1,
                  transition: 'opacity 200ms, r 200ms',
                }}
                strokeWidth={inWin ? 1.5 : 1}
              >
                <title>{`order #${e.row} · ${isoDay(e.t)}`}</title>
              </circle>
            );
          })}

          {/* lane: churn label per timestamp */}
          {lane(['churn at t'], Y.label, Y.labelLabel)}
          {calc.map((s, i) => {
            if (!labelVisible(i)) return null;
            if (!s.active)
              return (
                <line
                  key={s.t}
                  x1={x(s.t) - 4}
                  x2={x(s.t) + 4}
                  y1={Y.label}
                  y2={Y.label}
                  style={{ stroke: 'var(--muted)' }}
                  strokeWidth={2}
                  strokeLinecap="round"
                >
                  <title>{`${isoDay(s.t)}: not active yet, no row`}</title>
                </line>
              );
            const v = s.stored ?? s.computed;
            return (
              <motion.g
                key={s.t}
                initial={{ opacity: 0, scale: 0.4 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
              >
                <circle
                  cx={x(s.t)}
                  cy={Y.label}
                  r={6}
                  style={{
                    fill: v === true ? 'var(--cat-5)' : 'var(--surface)',
                    stroke: i === cur && step.kind !== 'all' ? 'var(--accent)' : 'var(--cat-5)',
                  }}
                  strokeWidth={2}
                >
                  <title>{`${isoDay(s.t)}: churn = ${boolText(v)}`}</title>
                </circle>
              </motion.g>
            );
          })}

          <TimeAxis x={x} y={Y.axis} narrow={narrow} />
        </svg>
      </div>

      <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[14.5px] text-ink-2" aria-label="Legend">
        <span className="inline-flex items-center gap-1.5">
          <svg width="14" height="14" viewBox="-7 -7 14 14" aria-hidden="true">
            <TaskMarker cx={0} cy={0} r={5} />
          </svg>
          task-table timestamp
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: 'var(--cat-3)' }} />
          order (orders table)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className="inline-block h-3 w-5 rounded-[3px]"
            style={{
              background: 'var(--accent-soft)',
              borderLeft: '1.5px dashed var(--accent)',
              borderRight: '1.5px solid var(--accent)',
            }}
          />
          window (t, t + {days} d]
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3 w-3 rounded-full" style={{ background: 'var(--cat-5)' }} />
          churn = true
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3 w-3 rounded-full border-2" style={{ borderColor: 'var(--cat-5)' }} />
          churn = false
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width="14" height="14" aria-hidden="true">
            <line x1={3} x2={11} y1={7} y2={7} style={{ stroke: 'var(--muted)' }} strokeWidth={2} strokeLinecap="round" />
          </svg>
          no row (not active yet)
        </span>
      </div>

      {/* on narrow screens the player sits right under the timeline, so "next" and the moving window stay in view together */}
      {compact && playerEl}

      <div className="mt-4 grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
        {/* the computation (steps a-d), or every customer at once (step e) */}
        <div ref={ruleRef} className="min-w-0 rounded-xl border border-border bg-surface-2 px-4 py-3.5">
          {isAll ? (
            <>
              <div className="eyebrow mb-2">Every customer, every timestamp</div>
              <CoverageGrid task={task} stamps={stamps} entityCount={customers.rows.length} entity={customer} />
            </>
          ) : (
            <>
              <div className="eyebrow mb-2">The label rule</div>
              {twoLineRule ? (
                <div className="flex flex-col gap-1 pb-1 text-[16.5px]">
                  <InlineMath tex={String.raw`\text{churn}(c, t) = \big[\, n = 0 \,\big]`} />
                  <InlineMath tex={String.raw`n = \#\{\text{orders of } c \text{ in } (t,\ t+${days}\,\text{d}]\}`} />
                </div>
              ) : (
                <div className="pb-1 text-[17.5px] whitespace-nowrap">
                  <InlineMath tex={String.raw`\text{churn}(c, t) = \big[\,\#\{\text{orders of } c \text{ in } (t,\ t+${days}\,\text{d}]\} = 0\,\big]`} />
                </div>
              )}
              <dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-[16px]">
                <dt className="text-ink-2">customer c</dt>
                <dd className="min-w-0 truncate text-ink">
                  <span className="num">#{customer}</span> {name}
                </dd>
                <dt className="text-ink-2">timestamp t</dt>
                <dd className="num text-ink">{c && cur >= 0 ? `${isoMinute(c.t)} (${SPLIT_TAG[c.split]})` : '—'}</dd>
                <dt className="text-ink-2">window</dt>
                <dd className="num text-ink">
                  {windowStamp >= 0 ? (
                    <>
                      <span className="whitespace-nowrap">({isoMinute(calc[windowStamp].t)},</span>{' '}
                      <span className="whitespace-nowrap">{isoMinute(calc[windowStamp].t + delta)}]</span>
                    </>
                  ) : (
                    '—'
                  )}
                </dd>
                <dt className="text-ink-2">orders inside</dt>
                <dd className="num text-ink">
                  {windowStamp >= 0 ? (
                    <>
                      {calc[windowStamp].inWindow.length}
                      {calc[windowStamp].inWindow.length > 0 && (
                        <span className="text-ink-2">
                          {' '}
                          ({calc[windowStamp].inWindow
                            .slice(0, 3)
                            .map((e) => `#${e.row} at ${isoMinute(e.t).slice(5)}`)
                            .join(', ')}
                          {calc[windowStamp].inWindow.length > 3 ? ', …' : ''})
                        </span>
                      )}
                    </>
                  ) : (
                    '—'
                  )}
                </dd>
                <dt className="text-ink-2">churn</dt>
                <dd className="num font-semibold text-ink">
                  {step.kind === 'window' ? '?' : windowStamp >= 0 ? boolText(calc[windowStamp].stored ?? calc[windowStamp].computed) : '—'}
                </dd>
              </dl>
            </>
          )}
        </div>

        {/* the growing task table */}
        <div className="card min-w-0 overflow-hidden" style={{ borderLeft: '3px solid var(--cat-5)' }}>
          <div className="flex flex-wrap items-center justify-between gap-x-2 border-b border-grid px-3 py-2">
            <span className="text-[15.5px] font-semibold text-ink">customer-churn</span>
            <span className="num text-[14.5px] text-muted">
              {isAll
                ? `${allRows.length.toLocaleString('en-US')} rows, all customers`
                : `${appended.length} row${appended.length === 1 ? '' : 's'} for #${customer}`}
            </span>
          </div>
          <div ref={tableScroll} className={cx('scroll-quiet overflow-auto', isAll ? 'max-h-[340px]' : 'max-h-[248px] min-h-[150px]')}>
            <table className={cx('w-full border-collapse', narrow ? 'text-[14px] [&_td]:!px-2 [&_th]:!px-2' : 'text-[15px]')}>
              <thead className="sticky top-0 z-[1] bg-surface-2">
                <tr className="border-b border-grid text-left text-[14px] text-ink-2">
                  <th scope="col" className="px-3 py-1.5 font-semibold">
                    <span className="inline-flex items-center gap-1 whitespace-nowrap">
                      {!narrow && <SemBadge sem={SemType.DateTime} size="xs" />}
                      <span>timestamp</span>
                    </span>
                  </th>
                  <th scope="col" className="px-2 py-1.5 font-semibold" title="Foreign key → customers: a link, no token">
                    <span className={cx('inline-flex items-center gap-1', !narrow && 'whitespace-nowrap')}>
                      {!narrow && <Link2 aria-label="foreign key" size={13} className="shrink-0" />}
                      <span>
                        customer_
                        <wbr />
                        id
                      </span>
                    </span>
                  </th>
                  <th scope="col" className="px-2 py-1.5 font-semibold">
                    <span className="inline-flex items-center gap-1 whitespace-nowrap">
                      {!narrow && <SemBadge sem={SemType.Boolean} size="xs" />}
                      <span>churn</span>
                    </span>
                  </th>
                  <th scope="col" className="px-3 py-1.5 text-right font-semibold">
                    split
                  </th>
                </tr>
              </thead>
              <tbody>
                {isAll ? (
                  allRows.map((r) => {
                    const own = r.e === customer;
                    return (
                      <tr
                        key={`${r.split}-${r.t}-${r.e}`}
                        data-own-row={own ? '' : undefined}
                        className={cx('border-b border-grid/70', own && 'bg-accent-soft')}
                      >
                        <td className="num px-3 py-[4px] whitespace-nowrap text-ink">{isoDay(r.t)}</td>
                        <td className={cx('num px-2 py-[4px]', own ? 'font-semibold text-ink' : 'text-ink-2')}>{r.e}</td>
                        <td className="num px-2 py-[4px] text-ink">{boolText(r.label)}</td>
                        <td className="px-3 py-[4px] text-right">
                          <span className="rounded bg-surface-3 px-1.5 text-[13.5px] font-medium text-ink-2">{r.split}</span>
                        </td>
                      </tr>
                    );
                  })
                ) : appended.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="px-3 py-6 text-center text-[15px] text-muted">
                      No rows yet.
                    </td>
                  </tr>
                ) : (
                  <AnimatePresence initial={false}>
                    {appended.map(({ c: r, i }) => (
                      <motion.tr
                        key={r.t}
                        initial={{ opacity: 0, y: -6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.25 }}
                        className={cx('border-b border-grid/70', i === cur && 'bg-accent-soft')}
                      >
                        <td className="num px-3 py-[5px] whitespace-nowrap text-ink">{isoDay(r.t)}</td>
                        <td className="num px-2 py-[5px] text-ink-2">{customer}</td>
                        <td className="num px-2 py-[5px] text-ink">{boolText(r.stored ?? r.computed)}</td>
                        <td className="px-3 py-[5px] text-right">
                          <span className="rounded bg-surface-3 px-1.5 text-[13.5px] font-medium text-ink-2">{r.split}</span>
                        </td>
                      </motion.tr>
                    ))}
                  </AnimatePresence>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {!compact && playerEl}
    </div>
  );
}
