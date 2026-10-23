/**
 * Figure: what the model sees of a task row (spec §2.1): exactly two tokens (the timestamp cell
 * and the label cell); the FK yields only a link (f2p_nbr_idxs), no token. Shown for the live
 * seed row (label masked) and for one of the entity's past rows in the same context (label
 * visible = an in-context label, spec §2.5). Tokens are the real sampled context's tokens.
 */
import { motion, useInView } from 'motion/react';
import { ArrowDown, ArrowRight, Link2 } from 'lucide-react';
import { useMemo, useRef, type ReactNode } from 'react';
import { Cite, SemBadge, TokenStrip, formatRaw, semOfDType } from '@/components';
import { MAX_F2P_NBRS, type GraphNode, type PreprocessedDB, type RawDatabase, type SampledContext } from '@/engine/types';
import { useElementSize } from '@/components/viz/useSize';
import { isoDay } from './taskData';

function RawRow({ db, node, masked }: { db: RawDatabase; node: GraphNode; masked: boolean }) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  // narrow card: one "name  value" line per column instead of three cramped columns
  const stacked = (size.width || 480) < 420;
  const task = db.tasks.find((t) => t.name === node.tableName && t.tables);
  const split = node.tableType === 'Db' ? null : node.tableType;
  const rec = task && split ? task.tables![split].rows[node.rowPos] : undefined;
  const schema = task?.tables?.Train.schema;
  if (!rec || !schema) return null;
  return (
    <div ref={ref} className="card min-w-0 overflow-hidden" style={{ borderLeft: '3px solid var(--cat-5)' }}>
      <div className="border-b border-grid px-3 py-1.5 text-[14.5px] break-words text-ink-2">
        {node.tableName}:{node.tableType}[{node.rowPos}] · node {node.nodeIdx}
      </div>
      <div className={stacked ? 'flex flex-col divide-y divide-[var(--grid)]' : 'grid grid-cols-3'}>
        {schema.columns.map((c) => {
          const isFk = c.name in schema.fkeys;
          const isTarget = c.name === task!.targetCol;
          return (
            <div
              key={c.name}
              className={
                stacked
                  ? 'flex min-w-0 items-center justify-between gap-3 px-3 py-1.5'
                  : 'min-w-0 border-r border-grid px-2.5 py-2 last:border-r-0'
              }
            >
              <div className="flex min-w-0 items-center gap-1 text-[14px] text-ink-2">
                {isFk ? <Link2 size={13} aria-label="foreign key" className="shrink-0" /> : <SemBadge sem={semOfDType(c.dtype)} size="xs" bare />}
                <span className="break-words">{c.name}</span>
              </div>
              <div className={stacked ? 'num shrink-0 text-[15.5px] text-ink' : 'num mt-1 text-[15.5px] text-ink'}>
                {isTarget && masked ? (
                  <span className="target-outline inline-flex h-[20px] min-w-[22px] items-center justify-center rounded-[4px] bg-accent-soft px-1 font-bold text-accent-text">
                    ?
                  </span>
                ) : c.dtype === 'datetime' ? (
                  <span className="whitespace-nowrap">{isoDay(rec[c.name] as number)}</span>
                ) : (
                  formatRaw(rec[c.name], c.dtype)
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Flow({ left, right, inView, delay = 0 }: { left: ReactNode; right: ReactNode; inView: boolean; delay?: number }) {
  return (
    <div className="grid items-center gap-3 md:grid-cols-[minmax(0,1fr)_28px_minmax(0,1.15fr)]">
      <div className="min-w-0">{left}</div>
      <div className="flex justify-center text-accent" aria-hidden="true">
        <ArrowRight size={20} className="hidden md:block" />
        <ArrowDown size={20} className="md:hidden" />
      </div>
      <motion.div
        className="min-w-0"
        initial={{ opacity: 0, x: -14 }}
        animate={inView ? { opacity: 1, x: 0 } : { opacity: 0, x: -14 }}
        transition={{ duration: 0.45, delay }}
      >
        {right}
      </motion.div>
    </div>
  );
}

export function TaskRowTokens({
  db,
  pdb,
  ctx,
  slots,
  pastNode,
}: {
  db: RawDatabase;
  pdb: PreprocessedDB;
  ctx: SampledContext;
  slots: Record<string, number>;
  /** A past row of the same entity that is in the context (a self in-context label), or null. */
  pastNode: number | null;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const inView = useInView(ref, { once: true, amount: 0.3 });
  const seed = pdb.nodes[ctx.params.seedNode];
  const seedTokens = ctx.tokens.filter((t) => t.nodeIdx === seed.nodeIdx);
  const past = pastNode !== null ? pdb.nodes[pastNode] : null;
  const pastTokens = past ? ctx.tokens.filter((t) => t.nodeIdx === past.nodeIdx) : [];
  const parent = seed.f2pNbrIdxs[0] !== undefined ? pdb.nodes[seed.f2pNbrIdxs[0]] : null;
  // batch layout: MAX_F2P_NBRS slots, -1 padded (fly.rs:25, 69, 465-467)
  const nbrs = [...seed.f2pNbrIdxs, ...Array(Math.max(0, MAX_F2P_NBRS - seed.f2pNbrIdxs.length)).fill(-1)];
  // pre.rs:477-479: parent node = parent table offset + FK value
  const parentOffset = parent ? (pdb.tableInfo[`${parent.tableName}:Db`]?.nodeIdxOffset ?? 0) : 0;
  const fkValue = useMemo(() => {
    const task = db.tasks.find((t) => t.name === seed.tableName && t.tables);
    const split = seed.tableType === 'Db' ? null : seed.tableType;
    const v = task && split && task.entityCol ? task.tables![split].rows[seed.rowPos]?.[task.entityCol] : null;
    return typeof v === 'number' ? v : null;
  }, [db, seed]);

  return (
    <div ref={ref} className="flex min-w-0 flex-col gap-6">
      <div>
        <div className="eyebrow mb-2">The seed row (label masked)</div>
        <Flow
          inView={inView}
          left={<RawRow db={db} node={seed} masked />}
          right={
            <div className="flex flex-col gap-2">
              <TokenStrip tokens={seedTokens} tableSlots={slots} showPadding={false} label="Tokens of the seed task row" />
              <div className="rounded-lg border border-dashed border-border-strong px-3 py-2 text-[15px] leading-snug text-ink-2">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <Link2 size={14} aria-hidden="true" className="shrink-0" />
                  <span>The FK makes no token, only a link:</span>
                  <span className="font-mono text-ink">f2p_nbr_idxs = [{nbrs.join(', ')}]</span>
                </div>
                <p className="mt-1.5">
                  {parent && fkValue !== null && (
                    <>
                      Each entry is a node index: the start of the {parent.tableName} block (node {parentOffset}) plus the FK value {fkValue}{' '}
                      gives node {parent.nodeIdx}, which is {parent.tableName}[{parent.rowPos}]{' '}
                      <Cite kind="code" ref="rustler/src/pre.rs:477-479" />.{' '}
                      {parentOffset === 0 && `It equals the FK value only because ${parent.tableName} is the first block. `}
                    </>
                  )}
                  In the batch every row gets {MAX_F2P_NBRS} parent slots, and unused slots hold −1{' '}
                  <Cite kind="code" ref="rustler/src/fly.rs:25" /> <Cite kind="code" ref="rustler/src/fly.rs:465-467" />{' '}
                  <Cite kind="spec" ref="§4.13" />.
                </p>
              </div>
            </div>
          }
        />
      </div>
      <div>
        <div className="eyebrow mb-2">A past row of the same entity, in the same context (label visible)</div>
        {past && pastTokens.length > 0 ? (
          <Flow
            inView={inView}
            delay={0.25}
            left={<RawRow db={db} node={past} masked={false} />}
            right={<TokenStrip tokens={pastTokens} tableSlots={slots} showPadding={false} label="Tokens of a past task row" />}
          />
        ) : (
          <p className="text-[16px] text-ink-2">
            The live context holds none of this entity’s past rows (for example, the seed is the entity’s earliest row, or the context
            filled up first).
          </p>
        )}
      </div>
    </div>
  );
}
