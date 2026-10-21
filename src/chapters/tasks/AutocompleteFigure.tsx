/**
 * Figure: autocomplete tasks on ShopDB's reviews table (spec §2.4, §4.9).
 * For the chosen review and task it shows the fate of every cell of the seed row (PK/FK → no
 * token, target → masked, leakage column → dropped on the fly, null → nothing) and the tokens the
 * seed row actually contributes in a context sampled by the real sampler with the live L, w, seed.
 */
import { ChevronLeft, ChevronRight, KeyRound, Link2 } from 'lucide-react';
import { useMemo } from 'react';
import { IconButton, NumberField, ScaleNote, SemBadge, Segmented, TokenStrip, cx, formatRaw, semOfDType } from '@/components';
import { getContext, getSeeds, inContextLabels, makeSampleParams } from '@/engine/pipeline';
import type { PreprocessedDB, RawDatabase, TaskDef } from '@/engine/types';
import { useSim } from '@/state/store';
import { useShallow } from 'zustand/react/shallow';

type Fate = 'pk' | 'fk' | 'target' | 'dropped' | 'null' | 'null-leak' | 'token';

const FATE_TEXT: Record<Fate, string> = {
  pk: 'primary key: no token',
  fk: 'foreign key: a link, no token',
  target: 'masked target',
  dropped: 'leakage column: dropped',
  null: 'missing: no token',
  'null-leak': 'missing: no token (leakage column)',
  token: 'token',
};

export function AutocompleteFigure({
  db,
  pdb,
  tasks,
  taskId,
  onTaskChange,
  review,
  onReviewChange,
  tableSlots,
}: {
  db: RawDatabase;
  pdb: PreprocessedDB;
  tasks: TaskDef[];
  taskId: string;
  onTaskChange: (id: string) => void;
  review: number;
  onReviewChange: (r: number) => void;
  tableSlots: Record<string, number>;
}) {
  const task = tasks.find((t) => t.id === taskId) ?? tasks[0];
  const table = db.tables.find((t) => t.schema.name === task.table)!;
  const n = table.rows.length;
  const row = table.rows[review];
  const { L, w, rngSeed } = useSim(useShallow((s) => ({ L: s.L, w: s.w, rngSeed: s.rngSeed })));

  const seedNode = pdb.tableInfo[`${task.table}:Db`].nodeIdxOffset + review;
  const ctx = useMemo(() => getContext(pdb, makeSampleParams(pdb, task, seedNode, L, w, rngSeed)), [pdb, task, seedNode, L, w, rngSeed]);
  const seedTokens = useMemo(() => ctx.tokens.filter((t) => t.nodeIdx === seedNode), [ctx, seedNode]);
  const labels = useMemo(() => inContextLabels(pdb, ctx, task), [pdb, ctx, task]);
  const seedCount = useMemo(() => getSeeds(pdb, task, 'Val').length, [pdb, task]);
  const drops = ctx.events.filter((e) => e.kind === 'drop-leakage-cell');
  const dropsOther = drops.filter((e) => e.kind === 'drop-leakage-cell' && e.node !== seedNode).length;
  // leakage-column cells of OTHER rows (other timestamps) are not dropped (spec §4.9): count them
  const leakKept = useMemo(
    () => ctx.tokens.filter((t) => !t.isPadding && t.nodeIdx !== seedNode && t.tableName === task.table && task.columnsToDrop.includes(t.colName)).length,
    [ctx, seedNode, task],
  );
  const targetMissing = row[task.targetCol] === null || (typeof row[task.targetCol] === 'number' && Number.isNaN(row[task.targetCol] as number));

  const fates = table.schema.columns.map((c) => {
    const v = row[c.name];
    let fate: Fate;
    if (c.name === table.schema.pkey) fate = 'pk';
    else if (c.name in table.schema.fkeys) fate = 'fk';
    else if (c.name === task.targetCol) fate = 'target';
    // a null cell makes no cell at all (pre.rs:552), so there is nothing for the drop rule to remove
    else if (v === null || (typeof v === 'number' && Number.isNaN(v))) fate = task.columnsToDrop.includes(c.name) ? 'null-leak' : 'null';
    else if (task.columnsToDrop.includes(c.name)) fate = 'dropped';
    else fate = 'token';
    return { col: c, v, fate };
  });

  return (
    <div className="min-w-0">
      <div className="mb-4 flex flex-wrap items-end gap-x-4 gap-y-3">
        <div className="flex flex-col gap-1">
          <span className="text-[14.5px] font-medium text-ink-2">Task</span>
          <Segmented
            label="Autocomplete task"
            value={task.id}
            onChange={onTaskChange}
            options={tasks.map((t) => ({ value: t.id, label: <span className="font-mono">{`${t.table}.${t.targetCol}`}</span> }))}
          />
        </div>
        <div className="flex items-end gap-1">
          <NumberField label={`Review row (0–${n - 1})`} value={review} min={0} max={n - 1} onChange={(v) => onReviewChange(Math.max(0, Math.min(n - 1, Math.round(v))))} className="w-[130px]" />
          <IconButton label="Previous review" size="sm" onClick={() => onReviewChange((review - 1 + n) % n)}>
            <ChevronLeft size={16} />
          </IconButton>
          <IconButton label="Next review" size="sm" onClick={() => onReviewChange((review + 1) % n)}>
            <ChevronRight size={16} />
          </IconButton>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div className="card min-w-0 overflow-hidden" style={{ borderLeft: '3px solid var(--cat-7)' }}>
          <div className="flex items-baseline justify-between gap-2 border-b border-grid px-3.5 py-2">
            <span className="font-mono text-[15.5px] font-semibold text-ink">
              reviews[{review}] <span className="font-normal text-muted">= the seed row</span>
            </span>
          </div>
          <ul className="divide-y divide-[var(--grid)]">
            {fates.map(({ col, v, fate }) => (
              <li
                key={col.name}
                className={cx(
                  'grid grid-cols-[18px_minmax(0,1fr)] items-center gap-x-2.5 gap-y-1 px-3.5 py-2 sm:grid-cols-[18px_minmax(0,0.85fr)_minmax(0,1.35fr)]',
                  fate === 'dropped' && 'bg-surface-2',
                )}
              >
                <span className="flex justify-center text-ink-2">
                  {fate === 'pk' ? (
                    <KeyRound aria-label="primary key" size={15} />
                  ) : fate === 'fk' ? (
                    <Link2 aria-label="foreign key" size={15} />
                  ) : (
                    <SemBadge sem={semOfDType(col.dtype)} size="xs" />
                  )}
                </span>
                <span className={cx('truncate font-mono text-[15px]', fate === 'pk' || fate === 'fk' ? 'text-ink-2' : 'text-ink')}>{col.name}</span>
                <span className="col-start-2 flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 sm:col-start-auto">
                  {fate === 'target' ? (
                    <span className="target-outline inline-flex h-[22px] min-w-[26px] items-center justify-center rounded-[5px] bg-accent-soft px-1.5 font-mono text-[15.5px] font-bold text-accent-text">
                      ?
                    </span>
                  ) : (
                    <span
                      className={cx(
                        'min-w-0 truncate text-[15px]',
                        col.dtype === 'str' ? '' : 'num',
                        fate === 'dropped' ? 'text-muted line-through decoration-[var(--critical)]' : fate === 'pk' || fate === 'fk' ? 'text-ink-2' : 'text-ink',
                      )}
                      title={typeof v === 'string' ? v : undefined}
                    >
                      {col.dtype === 'str' && v !== null ? `“${formatRaw(v, col.dtype)}”` : formatRaw(v, col.dtype)}
                      {fate === 'fk' ? ` → ${table.schema.fkeys[col.name]}` : ''}
                    </span>
                  )}
                  <span
                    className={cx(
                      'min-w-0 rounded px-1.5 py-px text-[13.5px] leading-snug font-medium whitespace-normal',
                      fate === 'target' ? 'bg-accent-soft text-accent-text' : fate === 'dropped' ? 'bg-surface-3 text-ink' : 'bg-surface-2 text-ink-2',
                    )}
                  >
                    {fate === 'target' ? `masked target (${task.targetType})` : FATE_TEXT[fate]}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </div>

        <div className="min-w-0">
          <div className="eyebrow mb-2">What the seed row puts in the context</div>
          {targetMissing ? (
            <p className="rounded-lg border border-dashed border-border-strong px-3 py-2.5 text-[15px] leading-snug text-ink-2">
              reviews[{review}] has no {task.targetCol} value, so it is not a seed of this task: seeds are the rows that have a target
              cell (fly.rs:287-296). Pick another review.
            </p>
          ) : (
            <>
              <TokenStrip tokens={seedTokens} tableSlots={tableSlots} showPadding={false} label="Tokens of the seed review" />
              <p className="mt-2 text-[15px] leading-snug text-ink-2">
                Positions {seedTokens.length > 0 ? `${seedTokens[0].pos}–${seedTokens[seedTokens.length - 1].pos}` : '—'} of a context
                sampled right now with L = {L}, w = {w} <ScaleNote inline />. The seed row always comes first.
              </p>
            </>
          )}
          <dl className="mt-4 grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-2 rounded-xl border border-border bg-surface-2 px-3.5 py-3 text-[15.5px]">
            <dt className="text-ink-2">Seeds of this task (all reviews with a {task.targetCol} value)</dt>
            <dd className="num text-right font-semibold text-ink">{seedCount.toLocaleString('en-US')}</dd>
            <dt className="text-ink-2">Other reviews’ {task.targetCol} values visible in this context</dt>
            <dd className="num text-right font-semibold text-ink">{labels.other.length}</dd>
            {task.columnsToDrop.length > 0 ? (
              <>
                <dt className="text-ink-2">Leakage cells dropped (seed row · other rows with the seed’s timestamp)</dt>
                <dd className="num text-right font-semibold text-ink">
                  {drops.length - dropsOther} · {dropsOther}
                </dd>
                <dt className="text-ink-2">
                  Other reviews’ <span className="font-mono">{task.columnsToDrop.join(', ')}</span> tokens kept (other timestamps)
                </dt>
                <dd className="num text-right font-semibold text-ink">{leakKept}</dd>
              </>
            ) : (
              <>
                <dt className="text-ink-2">Leakage columns of this task</dt>
                <dd className="text-right font-semibold text-ink">none</dd>
              </>
            )}
          </dl>
        </div>
      </div>
    </div>
  );
}
