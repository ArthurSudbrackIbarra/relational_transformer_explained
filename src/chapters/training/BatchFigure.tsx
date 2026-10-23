/**
 * "From ten tasks to one batch": how pretraining batches are formed (spec §4.14, §9.1).
 *
 *  1. every task of the three pretraining DBs contributes its seed rows to ONE item list
 *     (fly.rs:287-296, main.py:123-136), no task balancing;
 *  2. the list is shuffled once per epoch with seed epoch + seed (fly.rs:324-328, main.py:370);
 *  3. batch b takes 32 consecutive items (fly.rs:336-355);
 *  4. each item becomes one context with exactly ONE masked cell, its own target (fly.rs:492-496);
 *  5. the loss is the mean over the batch's masked cells (model.py:250-267).
 *
 * The seed lists and every context are the engine's (the same ones exported for training). The
 * shuffle is drawn with this site's PRNG: same rule, different random numbers from Rust's StdRng
 * (and from the NumPy permutation our trainer used), so the batch shown is illustrative.
 */
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { useMemo, useState } from 'react';
import { Cite, MathBlock, SemBadge, Slider, StepPlayer, TokenStrip, useStepPlayer } from '@/components';
import { ShapeGlyph, shapeOfSlot } from '@/components/viz/shapes';
import { useElementSize } from '@/components/viz/useSize';
import { formatDate } from '@/components/viz/format';
import { getContext, makeSampleParams } from '@/engine/pipeline';
import { Rng } from '@/engine/rng';
import { SemType } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { DB_LABEL, int, type Dataset } from './data';
import type { PretrainSet, PreTask } from './pretrainSet';

const B = 32;
const SHOWN_BATCHES = 8;

const STEPS = [
  {
    title: 'Ten tasks, one list',
    text: 'Every task of the three pretraining databases contributes its seed rows. They are concatenated into a single item list.',
  },
  {
    title: 'In list order',
    text: 'Unshuffled, the list runs task by task, so the first batches would hold nothing but student-dropout rows.',
  },
  {
    title: 'Shuffle',
    text: 'Once per pass over the data, the whole list is shuffled with the seed epoch + seed. Tasks and databases now mix.',
  },
  {
    title: 'One batch',
    text: 'Batch b takes the next 32 items of the shuffled list. It mixes databases, tasks, booleans and numbers.',
  },
  {
    title: 'One context per item',
    text: 'Each item is a seed row. The sampler builds its context window, and exactly one cell is masked: the seed’s own target.',
  },
  {
    title: 'One loss per batch',
    text: 'Each masked cell gets the loss of its type. The batch loss is their mean, and one optimizer step follows.',
  },
];

/** Fisher–Yates over 0..N−1 with the site PRNG (illustrative; the code uses StdRng(epoch + seed)). */
function shuffledOrder(N: number, epoch: number): Int32Array {
  const p = new Int32Array(N);
  for (let i = 0; i < N; i++) p[i] = i;
  const rng = new Rng(epoch + 0);
  for (let i = N - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    const t = p[i];
    p[i] = p[j];
    p[j] = t;
  }
  return p;
}

function seedRowText(t: PreTask, seed: number): { what: string; when: string | null; target: string; label: string } {
  const node = t.pdb.nodes[seed];
  const cell = node.cells.find((c) => c.colName === t.task.targetCol);
  const label = cell ? String(cell.raw) : '∅';
  const when = node.ts !== null ? formatDate(node.ts * 1000) : null;
  if (t.task.kind === 'forecast') {
    const parent = node.f2pNbrIdxs[0] !== undefined ? t.pdb.nodes[node.f2pNbrIdxs[0]] : undefined;
    const ent = parent ? `${parent.tableName.replace(/s$/, '')} ${parent.rowPos}` : 'entity';
    return {
      what: `${t.short} Train row #${node.rowPos} (${ent})`,
      when,
      target: t.task.targetCol,
      label,
    };
  }
  return {
    what: `${node.tableName} row #${node.rowPos}`,
    when,
    target: t.task.targetCol,
    label,
  };
}

export function BatchFigure({ set, dataset }: { set: PretrainSet | null; dataset: Dataset | null }) {
  const reduce = useReducedMotion();
  const player = useStepPlayer(STEPS.length, { intervalMs: 2600 });
  const step = player.index;
  const maxEpoch = Math.max(1, Math.floor((dataset?.pretrain?.epochsSeen ?? 8) - 1e-9));
  const [epoch, setEpoch] = useState(0);
  const [batch, setBatch] = useState(0);
  const [slotInBatch, setSlotInBatch] = useState(0);

  const order = useMemo(() => (set ? shuffledOrder(set.N, epoch) : null), [set, epoch]);
  const L = dataset?.L ?? 128;
  const w = dataset?.w ?? 16;

  if (!set || !order) {
    return (
      <div className="wide my-8 card px-5 py-6 text-[16px] text-ink-2" aria-busy="true">
        Building the pretraining item list from SchoolDB, RacingDB and ForumDB…
      </div>
    );
  }

  const shuffled = step >= 2;
  const itemAt = (k: number) => (shuffled ? order[k] : k);
  const batchItems = Array.from({ length: B }, (_, i) => itemAt((batch * B + i) % set.N));
  const sel = batchItems[Math.min(slotInBatch, B - 1)];
  const selTask = set.tasks[set.taskOf[sel]];
  const selSeed = set.seedOf[sel];
  const exported = dataset?.pretrain?.itemsPerEpoch;

  return (
    <figure className="wide my-8 min-w-0">
      <div className="card px-4 py-4 sm:px-5 sm:py-5">
        <div className="mb-3 text-[17px] font-semibold text-ink">From ten tasks to one batch</div>
        <StepPlayer {...player} unit="stage" label="Batch construction" describe={undefined} />
        <div className="mt-3 rounded-lg bg-surface-2 px-3.5 py-2.5" aria-live="polite">
          <div className="text-[16px] font-semibold text-ink">
            {step + 1}. {STEPS[step].title}
          </div>
          <div className="text-[16px] leading-snug text-ink-2">{STEPS[step].text}</div>
        </div>

        {/* 1. the item list */}
        <div className="mt-5">
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <div className="text-[16px] font-semibold text-ink">
              The item list: {int(set.N)} seed rows from {set.tasks.length} tasks
            </div>
            <div className="text-[14.5px] text-ink-2">
              {exported !== undefined && (exported === set.N ? `same count as the exported training set` : `exported set: ${int(exported)} items`)}{' '}
              <Cite kind="code" ref="rustler/src/fly.rs:287-296" />
            </div>
          </div>
          <ListBar set={set} />
          <TaskTable set={set} />
        </div>

        {/* 2-3. batches */}
        <AnimatePresence initial={false}>
          {step >= 1 && (
            <motion.div
              key="batches"
              initial={reduce ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reduce ? undefined : { opacity: 0 }}
              className="mt-6"
            >
              <div className="mb-2 flex flex-wrap items-end justify-between gap-x-4 gap-y-2">
                <div className="min-w-0">
                  <div className="text-[16px] font-semibold text-ink">
                    {shuffled
                      ? `The first ${SHOWN_BATCHES} batches after the epoch-${epoch} shuffle`
                      : `The first ${SHOWN_BATCHES} batches without a shuffle`}
                  </div>
                  <div className="text-[14.5px] text-ink-2">
                    One row = one batch of {B} items. Color and shape = database. Click a batch, then a cell.
                  </div>
                </div>
                {shuffled && (
                  <Slider
                    className="w-[220px]"
                    label="Epoch (shuffle seed)"
                    value={epoch}
                    min={0}
                    max={maxEpoch}
                    onChange={(v) => {
                      setEpoch(v);
                      player.setPlaying(false);
                    }}
                  />
                )}
              </div>
              <div>
                <BatchGrid
                  set={set}
                  itemAt={itemAt}
                  batch={step >= 3 ? batch : -1}
                  sel={step >= 4 ? slotInBatch : -1}
                  animateKey={`${shuffled}-${epoch}`}
                  reduce={!!reduce}
                  onPick={(b, i) => {
                    setBatch(b);
                    if (i !== null) setSlotInBatch(i);
                    player.setPlaying(false);
                    if (player.index < (i !== null ? 4 : 3)) player.setIndex(i !== null ? 4 : 3);
                  }}
                />
              </div>
              <p className="mt-2 text-[14.5px] leading-snug text-muted">
                Shuffle drawn with this site’s random generator. The official sampler shuffles with Rust’s <code>StdRng</code> seeded by epoch + seed{' '}
                <Cite kind="code" ref="rustler/src/fly.rs:324-328" />, and our trainer with a NumPy permutation: the same rule with different random
                numbers.
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* 4. one batch */}
        {step >= 3 && (
          <BatchSummary
            set={set}
            items={batchItems}
            batch={batch}
            sel={step >= 4 ? slotInBatch : -1}
            onPick={(i) => {
              setSlotInBatch(i);
              player.setPlaying(false);
              if (player.index < 4) player.setIndex(4);
            }}
          />
        )}

        {/* 5. one context */}
        {step >= 4 && <ItemContext set={set} task={selTask} seed={selSeed} epoch={epoch} L={L} w={w} slot={slotInBatch} batch={batch} />}

        {/* 6. the loss */}
        {step >= 5 && <BatchLoss set={set} items={batchItems} />}
      </div>
      <figcaption className="mx-auto mt-3 max-w-[var(--prose-max)] text-[16px] leading-snug text-ink-2">
        Seed lists and contexts are computed live by this site’s engine, the same code that exported our training data. Tasks enter in proportion to
        their number of seed rows: autocomplete tasks, which use every row of a table, dominate <Cite kind="spec" ref="§4.14" />.
      </figcaption>
    </figure>
  );
}

function ListBar({ set }: { set: PretrainSet }) {
  return (
    <div className="flex h-7 w-full overflow-hidden rounded-md" role="img" aria-label="Share of each task in the item list">
      {set.tasks.map((t) => (
        <div
          key={t.id}
          title={`${t.id}: ${int(t.seeds.length)} items`}
          className="h-full border-r-2 border-surface last:border-r-0"
          style={{
            width: `${(100 * t.seeds.length) / set.N}%`,
            background: catVar(t.slot),
            opacity: t.task.kind === 'forecast' ? 1 : 0.62,
          }}
        />
      ))}
    </div>
  );
}

function TaskTable({ set }: { set: PretrainSet }) {
  const maxSeeds = Math.max(...set.tasks.map((t) => t.seeds.length));
  return (
    <div className="relative mt-3 overflow-x-auto">
      <table className="w-full border-collapse text-[15px]">
        <thead>
          <tr className="border-b border-grid text-left text-ink-2">
            <th scope="col" className="hidden py-1.5 pr-3 font-semibold sm:table-cell">
              Database
            </th>
            <th scope="col" className="py-1.5 pr-3 font-semibold">
              Task
            </th>
            <th scope="col" className="hidden py-1.5 pr-3 font-semibold md:table-cell">
              Kind
            </th>
            <th scope="col" className="hidden py-1.5 pr-3 font-semibold md:table-cell">
              Target
            </th>
            <th scope="col" className="py-1.5 pr-3 text-right font-semibold">
              Seed rows
            </th>
            <th scope="col" className="w-[22%] py-1.5 font-semibold sm:w-[28%]">
              <span className="sr-only">Share</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {set.tasks.map((t) => (
            <tr key={t.id} className="border-b border-grid/70 last:border-0">
              <td className="hidden py-1 pr-3 whitespace-nowrap text-ink sm:table-cell">
                <span className="inline-flex items-center gap-1.5">
                  <ShapeGlyph shape={shapeOfSlot(t.slot)} size={13} fill={catVar(t.slot)} />
                  {DB_LABEL[t.dbName]}
                </span>
              </td>
              <td className="py-1 pr-3">
                <span className="inline-flex items-center gap-1.5 font-mono text-[14.5px] break-words text-ink">
                  <span className="sm:hidden">
                    <ShapeGlyph shape={shapeOfSlot(t.slot)} size={13} fill={catVar(t.slot)} />
                  </span>
                  {t.short}
                </span>
                {/* on narrow screens the kind and target type fold under the task name */}
                <span className="flex items-center gap-1.5 text-[14px] text-ink-2 md:hidden">
                  <SemBadge sem={t.task.targetType === 'binary' ? SemType.Boolean : SemType.Number} size="xs" />
                  {t.task.kind === 'forecast' ? 'forecast, Train split' : 'autocomplete, all rows'}
                </span>
              </td>
              <td className="hidden py-1 pr-3 text-ink-2 md:table-cell">
                {t.task.kind === 'forecast' ? 'forecast (Train split)' : 'autocomplete (all rows)'}
              </td>
              <td className="hidden py-1 pr-3 whitespace-nowrap text-ink-2 md:table-cell">
                <span className="inline-flex items-center gap-1.5">
                  <SemBadge sem={t.task.targetType === 'binary' ? SemType.Boolean : SemType.Number} size="xs" />
                  {t.task.targetType === 'binary' ? 'boolean' : 'number'}
                </span>
              </td>
              <td className="num py-1 pr-3 text-right text-ink">{int(t.seeds.length)}</td>
              <td className="py-1">
                <div
                  className="h-2.5 rounded-r-[3px]"
                  style={{
                    width: `${(100 * t.seeds.length) / maxSeeds}%`,
                    background: catVar(t.slot),
                    opacity: t.task.kind === 'forecast' ? 1 : 0.62,
                  }}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BatchGrid({
  set,
  itemAt,
  batch,
  sel,
  animateKey,
  reduce,
  onPick,
}: {
  set: PretrainSet;
  itemAt: (k: number) => number;
  batch: number;
  sel: number;
  animateKey: string;
  reduce: boolean;
  onPick: (batch: number, slot: number | null) => void;
}) {
  // measured here (this component mounts when the batches appear), not in the parent
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const cols = size.width && size.width < 620 ? 16 : 32;
  const rowsPerBatch = B / cols;
  return (
    <div ref={wrapRef}>
      <div className="grid gap-1.5" role="list" aria-label="Batches">
        {Array.from({ length: SHOWN_BATCHES }, (_, b) => {
          const active = b === batch;
          return (
            <div
              key={b}
              role="listitem"
              className={`flex items-center gap-2 rounded-md px-1 py-0.5 ${active ? 'bg-accent-soft ring-1 ring-accent' : ''}`}
            >
              <button
                type="button"
                onClick={() => onPick(b, null)}
                className={`w-[72px] shrink-0 rounded px-1 text-left text-[14px] whitespace-nowrap ${active ? 'font-semibold text-accent-text' : 'text-ink-2 hover:text-ink'}`}
                aria-label={`Select batch ${b}`}
              >
                batch {b}
              </button>
              <div
                className="grid min-w-0 flex-1 gap-[2px]"
                style={{
                  gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
                  gridTemplateRows: `repeat(${rowsPerBatch}, auto)`,
                }}
              >
                {Array.from({ length: B }, (_, i) => {
                  const k = itemAt((b * B + i) % set.N);
                  const t = set.tasks[set.taskOf[k]];
                  const isSel = active && i === sel;
                  return (
                    <motion.button
                      key={`${animateKey}-${i}`}
                      type="button"
                      title={`${t.id} · seed node ${set.seedOf[k]}`}
                      aria-label={`Batch ${b}, item ${i}: ${t.id}`}
                      onClick={() => onPick(b, i)}
                      initial={reduce ? false : { opacity: 0, scale: 0.4 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{
                        duration: 0.25,
                        delay: reduce ? 0 : (b * B + i) * 0.0016,
                      }}
                      className={`flex aspect-square min-h-[9px] items-center justify-center rounded-[3px] ${isSel ? 'outline-2 outline-offset-1 outline-[var(--ink)]' : ''}`}
                      style={{ background: 'var(--surface-2)' }}
                    >
                      <ShapeGlyph shape={shapeOfSlot(t.slot)} size={cols === 32 ? 16 : 12} fill={catVar(t.slot)} />
                    </motion.button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[15px] text-ink-2" aria-label="Legend">
        {(['school', 'racing', 'forum'] as const).map((d) => {
          const slot = set.tasks.find((t) => t.dbName === d)?.slot ?? 1;
          return (
            <span key={d} className="inline-flex items-center gap-1.5">
              <ShapeGlyph shape={shapeOfSlot(slot)} size={14} fill={catVar(slot)} />
              {DB_LABEL[d]}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function BatchSummary({
  set,
  items,
  batch,
  sel,
  onPick,
}: {
  set: PretrainSet;
  items: number[];
  batch: number;
  sel: number;
  onPick: (slot: number) => void;
}) {
  const groups = new Map<number, number[]>();
  items.forEach((k, i) => {
    const t = set.taskOf[k];
    if (!groups.has(t)) groups.set(t, []);
    groups.get(t)!.push(i);
  });
  const nBool = items.filter((k) => set.tasks[set.taskOf[k]].task.targetType === 'binary').length;
  const dbCount = (d: string) => items.filter((k) => set.tasks[set.taskOf[k]].dbName === d).length;
  return (
    <div className="mt-6">
      <div className="text-[16px] font-semibold text-ink">Inside batch {batch}</div>
      <div className="text-[15px] text-ink-2">
        {(['school', 'racing', 'forum'] as const).map((d) => `${dbCount(d)} ${DB_LABEL[d]}`).join(' · ')} · {nBool} boolean targets, {B - nBool}{' '}
        number targets. Click a task to open one of its contexts.
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        {[...groups.entries()]
          .sort((a, b) => b[1].length - a[1].length || a[0] - b[0])
          .map(([ti, slots]) => {
            const t = set.tasks[ti];
            const active = slots.includes(sel);
            return (
              <button
                key={ti}
                type="button"
                onClick={() => onPick(active ? slots[(slots.indexOf(sel) + 1) % slots.length] : slots[0])}
                className={`inline-flex items-center gap-2 rounded-lg border px-2.5 py-1.5 text-[15px] transition-colors ${active ? 'border-accent bg-accent-soft text-ink' : 'border-border bg-surface text-ink hover:bg-surface-2'}`}
                title={active && slots.length > 1 ? 'Click again for the next item of this task' : undefined}
              >
                <ShapeGlyph shape={shapeOfSlot(t.slot)} size={13} fill={catVar(t.slot)} />
                <span className="font-mono text-[14.5px]">{t.short}</span>
                <SemBadge sem={t.task.targetType === 'binary' ? SemType.Boolean : SemType.Number} size="xs" />
                <span className="num text-ink-2">×{slots.length}</span>
              </button>
            );
          })}
      </div>
    </div>
  );
}

function ItemContext({
  set,
  task,
  seed,
  epoch,
  L,
  w,
  slot,
  batch,
}: {
  set: PretrainSet;
  task: PreTask;
  seed: number;
  epoch: number;
  L: number;
  w: number;
  slot: number;
  batch: number;
}) {
  const ctx = useMemo(() => getContext(task.pdb, makeSampleParams(task.pdb, task.task, seed, L, w, epoch)), [task, seed, L, w, epoch]);
  const info = seedRowText(task, seed);
  const isBool = task.task.targetType === 'binary';
  const real = ctx.numReal;
  return (
    <div className="mt-6">
      <div className="text-[16px] font-semibold text-ink">
        Item {slot} of batch {batch}: its context window
      </div>
      <div className="mt-1 grid gap-x-6 gap-y-1 text-[15.5px] text-ink-2 md:grid-cols-2">
        <div>
          <span className="inline-flex items-center gap-1.5 text-ink">
            <ShapeGlyph shape={shapeOfSlot(task.slot)} size={13} fill={catVar(task.slot)} />
            {DB_LABEL[task.dbName]}
          </span>
          {' · '}
          <span className="font-mono text-[14.5px] text-ink">{info.what}</span>
          {info.when ? ` at ${info.when}` : ''}
        </div>
        <div>
          Masked cell: <span className="font-mono text-[14.5px] text-ink">{info.target} = ?</span> · hidden label{' '}
          <span className="font-mono text-[14.5px] text-ink">{info.label}</span> · loss{' '}
          <strong className="font-semibold text-ink">{isBool ? 'BCE with logits' : 'Huber (δ = 1)'}</strong>
        </div>
      </div>
      <div className="mt-2.5 rounded-lg border border-border bg-surface px-2.5 py-2">
        <TokenStrip tokens={ctx.tokens} tableSlots={set.tableSlots[task.dbName]} variant="cells" label={`Context of ${task.id} seed ${seed}`} />
      </div>
      <p className="mt-2 text-[14.5px] leading-snug text-muted">
        {int(real)} cell tokens{real < L ? ` and ${int(L - real)} padding` : ''} (L = {L}, w = {w}), sampled with the RNG seed epoch + seed row + 0 ={' '}
        {int(ctx.params.rngSeed)}: the exact context exported for this row at epoch {epoch}. The dashed “?” is the only masked cell{' '}
        <Cite kind="code" ref="rustler/src/fly.rs:492-496" />.
      </p>
    </div>
  );
}

function BatchLoss({ set, items }: { set: PretrainSet; items: number[] }) {
  const nBool = items.filter((k) => set.tasks[set.taskOf[k]].task.targetType === 'binary').length;
  const nNum = B - nBool;
  // three short lines, so the formula also fits a phone screen
  const tex = String.raw`\begin{aligned}\mathcal{L}_{\text{batch}} &= \frac{1}{${B}}\Big(\textstyle\sum_{${nBool}\ \text{boolean}} \operatorname{BCE}\big(\hat z_i,\ \mathbf 1\{y_i>0\}\big)\\[2pt] &\qquad +\; \textstyle\sum_{${nNum}\ \text{number}} \operatorname{Huber}_{\delta=1}\big(\hat y_i,\ y_i\big)\Big)\end{aligned}`;
  return (
    <div className="mt-6">
      <div className="text-[16px] font-semibold text-ink">The batch loss</div>
      <div className="relative mt-2 grid grid-cols-[minmax(0,1fr)] overflow-hidden rounded-lg bg-surface-2 px-3 py-1 text-[15px] sm:text-[17px]">
        <MathBlock tex={tex} className="my-2 min-w-0" />
      </div>
      <p className="mt-2 text-[15px] leading-snug text-ink-2">
        BCE compares a boolean cell’s logit ẑ with its label; Huber compares a number cell’s prediction ŷ with its z-scored value. One masked cell
        per context, so the mean is over {B} cells: every context weighs the same, whatever its database or type{' '}
        <Cite kind="code" ref="rt/model.py:250-267" />. Then one AdamW step updates every weight, and the next batch follows.
      </p>
    </div>
  );
}
