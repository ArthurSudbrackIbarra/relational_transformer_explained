/**
 * Leave-one-DB-out pretraining (spec §9.1, paper p.6 §4.1).
 *  - "This site": SchoolDB + RacingDB + ForumDB → pretraining; ShopDB held out (live engine counts).
 *  - "Paper": RelBench's 7 databases, one pretraining run per target database; task counts from
 *    rt/tasks.py (spec §9.1); rel-event is in every pretraining mix but never evaluated (p.6).
 * Switching the held-out database moves its card across (motion layout animation).
 */
import { LayoutGroup, motion, useReducedMotion } from 'motion/react';
import { ArrowDown, Lock } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';
import { Cite, SemBadge, Segmented } from '@/components';
import { ShapeGlyph, shapeOfSlot } from '@/components/viz/shapes';
import { getDatabase } from '@/engine/db/registry';
import { getPreprocessed, getSeeds } from '@/engine/pipeline';
import { SemType } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { DB_LABEL, DB_SLOT, int, type Dataset, type Results } from './data';
import { RELBENCH_DBS } from './paper';
import type { PretrainSet } from './pretrainSet';

export function LeaveOneOut({ set, dataset, results }: { set: PretrainSet | null; dataset: Dataset | null; results: Results | null }) {
  const [view, setView] = useState<'site' | 'paper'>('site');
  return (
    <figure className="wide my-8 min-w-0">
      <div className="card px-4 py-4 sm:px-5 sm:py-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div className="text-[17px] font-semibold text-ink">Who is seen in pretraining, who is held out</div>
          <Segmented
            label="Setting"
            value={view}
            onChange={setView}
            options={[
              { value: 'site', label: 'This site' },
              { value: 'paper', label: 'Paper (RelBench)' },
            ]}
          />
        </div>
        {view === 'site' ? <SiteView set={set} dataset={dataset} results={results} /> : <PaperView />}
      </div>
      <figcaption className="mx-auto mt-3 max-w-[var(--prose-max)] text-[16px] leading-snug text-ink-2">
        {view === 'site' ? (
          <>
            Counts are computed live by the engine. Forecast tasks contribute their Train split; autocomplete tasks every row of their table{' '}
            <Cite kind="code" ref="rt/data.py:77-81" />. ShopDB’s tables, column names and rows never reach the optimizer.
          </>
        ) : (
          <>
            Task counts from the official task list <Cite kind="code" ref="rt/tasks.py:1-97" />. Pick a target: its forecast tasks are evaluated, and
            every task of the other six databases is pretrained on <Cite kind="code" ref="scripts/example_pretrain.py:17-18" />.
          </>
        )}
      </figcaption>
    </figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared layout: pretraining DBs on top, the weights in the middle, the held-out DB below
// ─────────────────────────────────────────────────────────────────────────────

function Flow({
  pre,
  preTitle,
  weights,
  held,
  heldTitle,
  note,
}: {
  pre: ReactNode;
  preTitle: ReactNode;
  weights: ReactNode;
  held: ReactNode;
  heldTitle: ReactNode;
  note?: ReactNode;
}) {
  return (
    <div className="grid gap-0">
      <div className="rounded-xl border border-border bg-surface-2 p-3">
        <div className="eyebrow mb-2">{preTitle}</div>
        {pre}
      </div>
      <div className="flex flex-col items-center py-1.5">
        <ArrowDown aria-hidden="true" size={20} className="text-accent" />
        <div className="my-1 rounded-lg border border-accent bg-accent-soft px-3 py-1.5 text-center">
          <span className="text-[15px] font-semibold text-ink">RT weights</span>
          <span className="num ml-2 text-[14px] text-ink-2">{weights}</span>
        </div>
        <div className="grid w-full grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-2 text-[14px] text-ink-2">
          <span />
          <ArrowDown aria-hidden="true" size={20} className="text-accent" />
          <span>frozen: no weight update on the held-out database</span>
        </div>
      </div>
      <div className="rounded-xl border-2 border-dashed border-accent/60 p-3">
        <div className="eyebrow mb-2 flex items-center gap-1.5">
          <Lock aria-hidden="true" size={13} /> {heldTitle}
        </div>
        {held}
        {note}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

function SiteView({ set, dataset, results }: { set: PretrainSet | null; dataset: Dataset | null; results: Results | null }) {
  const shop = useMemo(() => {
    const db = getDatabase('shop');
    const pdb = getPreprocessed(db);
    return {
      db,
      rows: db.tables.reduce((s, t) => s + t.rows.length, 0),
      tasks: db.tasks.map((t) => ({
        t,
        val: getSeeds(pdb, t, 'Val').length,
        test: getSeeds(pdb, t, 'Test').length,
      })),
    };
  }, []);
  const steps = results?.pretrain?.steps ?? dataset?.pretrain?.steps;
  const batch = dataset?.batchSize ?? results?.config?.batch;
  return (
    <Flow
      preTitle={<>Pretraining: {set ? `${set.tasks.length} tasks, ${int(set.N)} seed rows per epoch` : '…'}</>}
      pre={
        <div className="grid gap-2.5 md:grid-cols-3">
          {(set?.dbs ?? []).map((d) => (
            <div key={d.name} className="min-w-0 rounded-lg border border-border bg-surface px-3 py-2.5">
              <div className="flex items-center gap-1.5 text-[16px] font-semibold text-ink">
                <ShapeGlyph shape={shapeOfSlot(DB_SLOT[d.name])} size={14} fill={catVar(DB_SLOT[d.name])} />
                {DB_LABEL[d.name]}
                <span className="ml-auto text-[14px] font-normal text-ink-2">
                  {d.tables} tables · {int(d.rows)} rows
                </span>
              </div>
              <ul className="!mt-2 !mb-0 grid gap-1 !pl-0" style={{ listStyle: 'none' }}>
                {d.tasks.map((t) => (
                  <li key={t.id} className="!m-0 flex flex-wrap items-center gap-x-1.5 !pl-0 text-[14.5px]">
                    <SemBadge sem={t.task.targetType === 'binary' ? SemType.Boolean : SemType.Number} size="xs" />
                    <span className="min-w-0 break-words text-ink" title={t.id}>
                      {t.short}
                    </span>
                    <span className="num ml-auto shrink-0 pl-2 text-ink-2">{int(t.seeds.length)}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
          {!set && <div className="text-[15px] text-ink-2">Building the three databases…</div>}
        </div>
      }
      weights={`${steps ? `${int(steps)} steps` : ''}${batch ? ` × ${batch} contexts` : ''}`}
      heldTitle="Held out: zero-shot evaluation only"
      held={
        <div className="rounded-lg border border-border bg-surface px-3 py-2.5">
          <div className="flex flex-wrap items-center gap-x-1.5 text-[16px] font-semibold text-ink">
            <ShapeGlyph shape={shapeOfSlot(DB_SLOT.shop)} size={14} fill={catVar(DB_SLOT.shop)} />
            ShopDB
            <span className="text-[14px] font-normal text-ink-2 sm:ml-auto">
              {shop.db.tables.length} tables · {int(shop.rows)} rows · never seen in pretraining
            </span>
          </div>
          <ul className="!mt-2 !mb-0 grid gap-x-6 gap-y-1 !pl-0 md:grid-cols-2" style={{ listStyle: 'none' }}>
            {shop.tasks.map(({ t, val, test }) => (
              <li key={t.id} className="!m-0 flex flex-wrap items-center gap-x-1.5 !pl-0 text-[14.5px]">
                <SemBadge sem={t.targetType === 'binary' ? SemType.Boolean : SemType.Number} size="xs" />
                <span className="min-w-0 break-words text-ink" title={t.id}>
                  {t.id.split('/')[1]}
                </span>
                <span className="num ml-auto shrink-0 pl-2 text-ink-2">{t.kind === 'forecast' ? `val ${val} · test ${test}` : 'our extension'}</span>
              </li>
            ))}
          </ul>
        </div>
      }
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────────

function PaperView() {
  const reduce = useReducedMotion();
  const evaluable = RELBENCH_DBS.filter((d) => d.evaluated).map((d) => d.name);
  const [target, setTarget] = useState('rel-amazon');
  const tgt = RELBENCH_DBS.find((d) => d.name === target)!;
  const pre = RELBENCH_DBS.filter((d) => d.name !== target);
  const nTrain = pre.reduce((s, d) => s + d.forecast + d.autocomplete, 0);
  const card = (d: (typeof RELBENCH_DBS)[number], held: boolean) => (
    <motion.div
      key={d.name}
      layoutId={`db-${d.name}`}
      layout={!reduce}
      transition={{ type: 'spring', stiffness: 380, damping: 34 }}
      className={`min-w-0 rounded-lg border px-3 py-2 ${held ? 'border-accent bg-surface' : 'border-border bg-surface'}`}
    >
      <div className="text-[15.5px] font-semibold text-ink">{d.name}</div>
      <div className="text-[14.5px] text-ink-2">
        {held ? `${d.forecast} forecast tasks evaluated` : `${d.forecast} forecast + ${d.autocomplete} autocomplete`}
      </div>
      {!d.evaluated && <div className="text-[14px] text-muted">never a target (temporal leakage)</div>}
    </motion.div>
  );
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="text-[15px] font-medium text-ink-2">Held-out target</span>
        <Segmented
          label="Held-out target database"
          value={target}
          onChange={setTarget}
          options={evaluable.map((n) => ({
            value: n,
            label: n.replace('rel-', ''),
          }))}
        />
      </div>
      <LayoutGroup>
        <Flow
          preTitle={
            <>
              Pretraining: {nTrain} tasks from {pre.length} databases
            </>
          }
          pre={<div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">{pre.map((d) => card(d, false))}</div>}
          weights="50,001 steps × 256 contexts"
          heldTitle="Held out"
          held={<div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">{card(tgt, true)}</div>}
          note={
            <p className="!mt-2 !mb-0 text-[14.5px] leading-snug text-ink-2">
              One pretraining run per target: {evaluable.length} runs, each with its own checkpoints <Cite kind="paper" ref="p.6 §4.1" />.
            </p>
          }
        />
      </LayoutGroup>
    </div>
  );
}
