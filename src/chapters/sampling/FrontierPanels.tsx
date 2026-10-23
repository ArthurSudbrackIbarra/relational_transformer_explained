/**
 * State panels under the BFS graph: the F→P stack (LIFO), the P→F depth buckets (exact contents and
 * order, so the swap-remove of a uniform draw is visible), counters, and the tokens-used / L meter.
 */
import { ArrowUp, Dice5 } from 'lucide-react';
import { motion } from 'motion/react';
import { ShapeGlyph, cx, shapeOfSlot } from '@/components';
import type { PreprocessedDB, RawDatabase, SampledContext } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { usePrefersReducedMotion } from '@/styles/theme';
import { plural, rowName, type ReplayState, type Step } from './trace';

interface Common {
  db: RawDatabase;
  pdb: PreprocessedDB;
  slots: Record<string, number>;
  state: ReplayState;
  before: ReplayState;
  step: Step;
  ctx: SampledContext;
  hover: number | null;
  onHover: (n: number | null) => void;
}

function PanelTitle({ children, aside }: { children: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-2">
      <div className="text-[15px] font-semibold whitespace-nowrap text-ink">{children}</div>
      {aside && <div className="num text-[14px] whitespace-nowrap text-muted">{aside}</div>}
    </div>
  );
}

/** The F→P stack, top first. */
export function StackPanel({ db, pdb, slots, state, before, step, ctx, hover, onHover }: Common) {
  const pop = step.kind === 'select' || step.kind === 'row' ? ctx.events[step.start] : undefined;
  const popped = pop && pop.kind === 'pop' && pop.from === 'stack' ? before.stack[before.stack.length - 1] : undefined;
  const items = [...state.stack].reverse();
  const MAX = 9;
  return (
    <div className="min-w-0">
      <PanelTitle aside={plural(state.stack.length, 'entry', 'entries')}>F→P stack (LIFO)</PanelTitle>
      <div className="flex min-h-[88px] flex-col gap-1 rounded-lg border border-border bg-surface-2/60 p-1.5" role="list" aria-label="F→P stack, top first">
        {popped && (
          <div className="flex items-center gap-1.5 rounded-md border border-dashed border-accent px-2 py-1 text-[14px] text-accent-text" role="listitem">
            <ArrowUp aria-hidden="true" size={12} /> popped: {rowName(db, pdb, popped.node).short}
          </div>
        )}
        {items.length === 0 && !popped && <div className="px-2 py-1 text-[14.5px] text-muted">empty: the buckets are used</div>}
        {items.slice(0, MAX).map((e, i) => {
          const g = pdb.nodes[e.node];
          const slot = slots[g.tableName] ?? 0;
          const dup = state.visitedDepth.has(e.node);
          return (
            <div
              key={`${e.node}-${i}-${items.length}`}
              role="listitem"
              onPointerEnter={() => onHover(e.node)}
              onPointerLeave={() => onHover(null)}
              className={cx(
                'flex min-w-0 items-center gap-1.5 rounded-md border bg-surface px-2 py-1 text-[14.5px]',
                i === 0 ? 'border-border-strong' : 'border-border',
                hover === e.node && 'ring-2 ring-accent',
                dup && 'opacity-55',
              )}
            >
              <ShapeGlyph shape={shapeOfSlot(slot)} size={11} fill={catVar(slot)} />
              <span className={cx('truncate text-ink', dup && 'line-through decoration-[var(--muted)]')}>{rowName(db, pdb, e.node).short}</span>
              <span className="num ml-auto shrink-0 text-[13.5px] text-muted">
                d{e.depth}
                {i === 0 ? ' · top' : ''}
              </span>
            </div>
          );
        })}
        {items.length > MAX && <div className="px-2 text-[14px] text-muted">+ {items.length - MAX} more below</div>}
      </div>
      <p className="mt-1.5 text-[14px] leading-snug text-muted">Struck through: already visited, skipped when popped.</p>
    </div>
  );
}

/** The P→F depth buckets with exact contents. */
export function BucketsPanel({ db, pdb, slots, state, before, step, ctx, hover, onHover }: Common) {
  const reduce = usePrefersReducedMotion();
  const ev = step.kind === 'select' || step.kind === 'row' ? ctx.events[step.start] : undefined;
  const draw = ev && ev.kind === 'pop' && ev.from === 'bucket' ? ev : undefined;
  // every depth from the lowest to the highest non-empty bucket (empty ones too, so the reader sees why
  // "the lowest non-empty bucket" skips them), plus the bucket just drawn from
  const nonEmpty = state.buckets.map((b, d) => (b.length > 0 || (draw && draw.depth === d) ? d : -1)).filter((d) => d >= 0);
  const dLo = nonEmpty.length ? Math.min(...nonEmpty) : 0;
  const dHi = nonEmpty.length ? Math.max(...nonEmpty) : -1;
  const depths = Array.from({ length: Math.max(0, dHi - dLo + 1) }, (_, i) => ({ d: dLo + i, b: state.buckets[dLo + i] ?? [] }));
  const lowest = state.buckets.findIndex((b) => b.length > 0);
  const total = state.buckets.reduce((s, b) => s + b.length, 0);
  return (
    <div className="min-w-0">
      <PanelTitle aside={plural(total, 'entry', 'entries')}>P→F depth buckets</PanelTitle>
      <div className="scroll-quiet flex max-h-[220px] min-h-[88px] flex-col gap-2 overflow-y-auto rounded-lg border border-border bg-surface-2/60 p-2">
        {depths.length === 0 && <div className="px-1 py-1 text-[14.5px] text-muted">empty</div>}
        {state.finished?.kind === 'full' && total > 0 && <div className="text-[14px] text-muted">L reached: these entries are never drawn.</div>}
        {depths.map(({ d, b }) => {
          const drawnHere = draw && draw.depth === d;
          if (b.length === 0 && !drawnHere)
            return (
              <div key={d} className="text-[14px] text-muted">
                <span className="font-semibold text-ink-2">depth {d}</span> · empty
              </div>
            );
          return (
            <div key={d} className="min-w-0">
              <div className="mb-1 flex flex-wrap items-center gap-x-2 text-[14px] text-ink-2">
                <span className="font-semibold text-ink">depth {d}</span>
                <span className="num">{b.length}</span>
                {d === lowest && state.stack.length === 0 && !state.finished && <span className="text-accent-text">next draw comes from here</span>}
                {drawnHere && draw && (
                  <span className="inline-flex items-center gap-1 text-accent-text">
                    <Dice5 aria-hidden="true" size={12} /> drew index {draw.draw} of {draw.bucketSize}: {rowName(db, pdb, draw.node).short}
                  </span>
                )}
              </div>
              <div className="flex flex-wrap gap-[3px]" role="list" aria-label={`Bucket at depth ${d}`}>
                {drawnHere && draw && (
                  <motion.span
                    key={`drawn-${step.start}`}
                    role="listitem"
                    aria-label={`drawn: ${rowName(db, pdb, draw.node).short}`}
                    initial={reduce ? false : { y: 6, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    className="inline-flex h-[13px] w-[13px] items-center justify-center rounded-[3px] ring-2 ring-accent ring-offset-1 ring-offset-surface"
                    style={{ background: catVar(slots[pdb.nodes[draw.node].tableName] ?? 0) }}
                    onPointerEnter={() => onHover(draw.node)}
                    onPointerLeave={() => onHover(null)}
                  />
                )}
                {b.map((n, i) => {
                  const slot = slots[pdb.nodes[n].tableName] ?? 0;
                  const dup = state.visitedDepth.has(n);
                  const swapped = drawnHere && draw && i === draw.draw && i < b.length && (before.buckets[d]?.length ?? 0) - 1 !== draw.draw;
                  return (
                    <span
                      key={`${n}-${i}`}
                      role="listitem"
                      title={`${rowName(db, pdb, n).long}${dup ? ' (already visited: will be skipped)' : ''}${swapped ? ' (moved here from the end by the swap-remove)' : ''}`}
                      onPointerEnter={() => onHover(n)}
                      onPointerLeave={() => onHover(null)}
                      className={cx(
                        'inline-block h-[11px] w-[11px] rounded-[3px] border',
                        dup && 'opacity-35',
                        hover === n && 'ring-2 ring-accent',
                        swapped && 'outline-dashed outline-[1.5px] outline-offset-1 outline-accent',
                      )}
                      style={{
                        borderColor: catVar(slot),
                        background: dup ? 'transparent' : `color-mix(in srgb, ${catVar(slot)} 30%, transparent)`,
                      }}
                    />
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
      <p className="mt-1.5 text-[14px] leading-snug text-muted">
        Exact contents in order. A draw swaps the picked entry with the last one and pops it. Faded: already visited.
      </p>
    </div>
  );
}

/** Counters + the tokens-used / L meter. */
export function CountersPanel({ state, ctx, slots, hover, onHover }: Omit<Common, 'db' | 'pdb' | 'before' | 'step'>) {
  const L = ctx.params.L;
  const filled = state.emitted;
  const padding = state.finished?.kind === 'empty' ? L - filled : 0;
  const W = 300;
  const cw = W / L;
  return (
    <div className="min-w-0">
      <PanelTitle>Progress</PanelTitle>
      <dl className="grid grid-cols-2 gap-2">
        <div className="rounded-lg border border-border bg-surface px-2.5 py-2">
          <dt className="text-[14px] text-ink-2">Visited rows</dt>
          <dd className="num text-[22px] font-semibold text-ink">{state.visitedOrder.length}</dd>
        </div>
        <div className="rounded-lg border border-border bg-surface px-2.5 py-2">
          <dt className="text-[14px] text-ink-2">Pops</dt>
          <dd className="num text-[22px] font-semibold text-ink">
            {state.pops}
            <span className="ml-1 text-[14px] font-normal text-muted">{state.skips} skipped</span>
          </dd>
        </div>
      </dl>
      <div className="mt-3">
        <div className="flex items-baseline justify-between text-[14px] text-ink-2">
          <span>Tokens used</span>
          <span className="num text-[15.5px] text-ink">
            <strong className="font-semibold">{filled}</strong> / {L}
          </span>
        </div>
        <svg viewBox={`0 0 ${W} 14`} width="100%" height={14} preserveAspectRatio="none" className="mt-1 block" role="img" aria-label={`${filled} of ${L} token positions filled${padding ? `, ${padding} padding` : ''}`}>
          <rect x={0} y={0} width={W} height={14} rx={3} style={{ fill: 'var(--surface-2)' }} />
          {ctx.tokens.slice(0, filled).map((t) => {
            const isHover = hover !== null && t.nodeIdx === hover;
            return (
              <rect
                key={t.pos}
                x={t.pos * cw}
                y={isHover ? 0 : 2}
                width={Math.max(cw - (cw > 2.5 ? 0.6 : 0), 0.5)}
                height={isHover ? 14 : 10}
                style={{ fill: t.isTarget ? 'var(--accent)' : catVar(slots[t.tableName] ?? 0) }}
                onPointerEnter={() => onHover(t.nodeIdx)}
                onPointerLeave={() => onHover(null)}
              />
            );
          })}
          {padding > 0 && (
            <rect x={filled * cw} y={2} width={padding * cw} height={10} style={{ fill: 'none', stroke: 'var(--axis)', strokeDasharray: '3 2' }} strokeWidth={1} />
          )}
        </svg>
        <p className="mt-1 text-[14px] leading-snug text-muted">
          {state.finished?.kind === 'full'
            ? `Full: the hard stop at L = ${L} fired.`
            : state.finished?.kind === 'empty'
              ? `Frontier empty: ${plural(padding, 'padding token')}.`
              : `${plural(L - filled, 'position')} left. Accent = the masked target.`}
        </p>
      </div>
    </div>
  );
}
