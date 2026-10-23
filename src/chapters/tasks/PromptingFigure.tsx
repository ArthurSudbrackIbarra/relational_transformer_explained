/**
 * Figure: task-table prompting (paper p.4 §3.1). The task table is attached to the database as an
 * ordinary table with one FK to the entity; its rows become children of the entity, exactly like
 * orders or reviews. Live schema (DB tables: key and time columns only, the figure is about links;
 * the task table: all columns, label shaded) + the live P→F adjacency of the example entity
 * (PreprocessedDB.p2fAdj), filtered exactly as the sampler's table test filters it (fly.rs:417):
 * task-table children only from a seed of that table. The task table sits in a column left of its
 * entity, so its one FK edge crosses no card. Below 860 px the schema is a compact stack.
 */
import { AnimatePresence, motion } from 'motion/react';
import { Clock, Link2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Cite, SchemaDiagram, SemBadge, Segmented, TableSwatch, cx, semOfDType, type SchemaTableSpec } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import type { PreprocessedDB, RawDatabase, TaskDef } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { entityName } from './taskData';

type Choice = 'none' | string;

/** Narrow-screen stand-in for the schema diagram: DB tables as chips, the task table as a card. */
function CompactSchema({ specs, task }: { specs: SchemaTableSpec[]; task: TaskDef | undefined }) {
  const dbSpecs = specs.filter((t) => t.kind !== 'task');
  const taskSpec = specs.find((t) => t.kind === 'task');
  return (
    <div className="flex flex-col gap-3">
      <ul className="flex flex-wrap gap-2" aria-label="Database tables">
        {dbSpecs.map((t) => (
          <li
            key={t.schema.name}
            className="inline-flex items-center gap-2 rounded-lg border border-border bg-surface px-2.5 py-1 text-[15px]"
            style={t.schema.name === task?.entityTable ? { boxShadow: '0 0 0 2px var(--accent)' } : undefined}
          >
            <TableSwatch slot={t.slot} label={t.schema.name} />
            <span className="num text-[14px] text-muted">{t.rowCount?.toLocaleString('en-US')}</span>
          </li>
        ))}
      </ul>
      {taskSpec && task ? (
        <div className="card relative overflow-hidden ring-2 ring-accent">
          <div aria-hidden="true" className="absolute inset-x-0 top-0 h-[3px]" style={{ background: catVar(taskSpec.slot) }} />
          <div className="flex items-baseline justify-between gap-2 px-3 pt-3 pb-1">
            <span className="truncate text-[16.5px] font-semibold text-ink">{taskSpec.schema.name}</span>
            <span className="shrink-0 text-[14.5px] text-muted">task table</span>
          </div>
          {taskSpec.subtitle && <div className="px-3 pb-2 text-[14.5px] text-muted">{taskSpec.subtitle}</div>}
          <ul className="border-t border-grid py-1">
            {taskSpec.schema.columns.map((c) => {
              const fk = taskSpec.schema.fkeys[c.name];
              const isTime = c.name === taskSpec.schema.timeCol;
              const isLabel = c.name === task.targetCol;
              return (
                <li key={c.name} className={cx('flex items-center gap-2 px-3 py-[6px] text-[15.5px]', isLabel && 'bg-accent-soft')}>
                  <span className="flex w-[20px] shrink-0 justify-center text-ink-2">
                    {fk ? <Link2 aria-label="foreign key" size={15} /> : <SemBadge sem={semOfDType(c.dtype)} size="xs" />}
                  </span>
                  <span className={fk ? 'min-w-0 flex-1 truncate text-ink-2' : 'min-w-0 flex-1 truncate text-ink'}>{c.name}</span>
                  {fk && <span className="shrink-0 text-[14px] text-accent-text">→ {fk}</span>}
                  {isTime && (
                    <span className="inline-flex shrink-0 items-center gap-1 rounded bg-surface-3 px-1.5 text-[13.5px] font-medium text-ink-2">
                      <Clock aria-hidden="true" size={12} /> time
                    </span>
                  )}
                  {isLabel && <span className="shrink-0 rounded bg-surface px-1.5 text-[13.5px] font-medium text-accent-text">label</span>}
                </li>
              );
            })}
          </ul>
        </div>
      ) : (
        <div className="rounded-xl border border-dashed border-border-strong px-3 py-3 text-[15px] text-muted">No task table attached.</div>
      )}
    </div>
  );
}

export function PromptingFigure({
  db,
  pdb,
  tasks,
  slots,
  customer,
  product,
}: {
  db: RawDatabase;
  pdb: PreprocessedDB;
  tasks: TaskDef[];
  slots: Record<string, number>;
  customer: number;
  product: number;
}) {
  const forecast = tasks.filter((t) => t.kind === 'forecast' && t.tables);
  const [choice, setChoice] = useState<Choice>(forecast[0]?.name ?? 'none');
  const task = forecast.find((t) => t.name === choice);
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  // the three-column diagram needs ~852 px (SchemaDiagram: 3 × 244 + 2 × 56 + 8); below that, a compact stack
  const narrow = (size.width || 900) < 860;

  // DB tables: key and time columns only (the figure is about links); the task table: every column,
  // because its label column is what makes past task rows in-context labels (spec §2.5).
  const specs: SchemaTableSpec[] = useMemo(() => {
    const out: SchemaTableSpec[] = db.tables.map((t) => ({
      schema: {
        ...t.schema,
        columns: t.schema.columns.filter((c) => c.name === t.schema.pkey || c.name in t.schema.fkeys || c.name === t.schema.timeCol),
      },
      slot: slots[t.schema.name] ?? 0,
      rowCount: t.rows.length,
      kind: 'db',
    }));
    if (task?.tables) {
      const n = (['Train', 'Val', 'Test'] as const).map((s) => `${s} ${task.tables![s].rows.length}`).join(' · ');
      out.push({ schema: task.tables.Train.schema, slot: slots[task.name] ?? 5, kind: 'task', subtitle: `${n} rows` });
    }
    return out;
  }, [db, task, slots]);

  // Layout: the task table in its own column LEFT of its entity, the DB tables shifted one column
  // right, so the task table's single FK edge runs between adjacent columns and crosses no card.
  const columnsOf = useMemo(() => {
    if (!task) return undefined;
    const byName = new Map(db.tables.map((t) => [t.schema.name, t]));
    const memo = new Map<string, number>();
    const level = (name: string, seen: Set<string>): number => {
      if (memo.has(name)) return memo.get(name)!;
      const t = byName.get(name);
      if (!t || seen.has(name)) return 0;
      seen.add(name);
      const parents = Object.values(t.schema.fkeys).filter((p) => byName.has(p) && p !== name);
      const l = parents.length === 0 ? 0 : 1 + Math.max(...parents.map((p) => level(p, seen)));
      memo.set(name, l);
      return l;
    };
    const out: Record<string, number> = { [task.name]: 0 };
    for (const t of db.tables) out[t.schema.name] = level(t.schema.name, new Set()) + 1;
    return out;
  }, [db, task]);

  // the entity whose children we list: the attached task's entity (default: the customer)
  const entityTable = task?.entityTable ?? 'customers';
  const entityRow = entityTable === 'products' ? product : customer;
  const entityNode = pdb.tableInfo[`${entityTable}:Db`].nodeIdxOffset + entityRow;
  const { children, stored, skipped } = useMemo(() => {
    const m = new Map<string, { table: string; type: string; n: number }>();
    const adj = pdb.p2fAdj[entityNode] ?? [];
    let skippedN = 0;
    for (const e of adj) {
      const node = pdb.nodes[e.node];
      // fly.rs:417: a task-table child is followed only from a seed of that same task table
      if (e.tableType !== 'Db' && (!task || node.tableName !== task.name)) {
        skippedN++;
        continue;
      }
      const k = node.tableKey;
      const cur = m.get(k) ?? { table: node.tableName, type: node.tableType, n: 0 };
      cur.n++;
      m.set(k, cur);
    }
    const list = [...m.values()].sort((a, b) => (a.type === 'Db' ? 0 : 1) - (b.type === 'Db' ? 0 : 1) || a.table.localeCompare(b.table));
    return { children: list, stored: adj.length, skipped: skippedN };
  }, [pdb, entityNode, task]);

  return (
    <div ref={wrapRef} className="min-w-0">
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <span className="text-[14.5px] font-medium text-ink-2">Attach a task table</span>
          <Segmented
            label="Attached task table"
            value={choice}
            onChange={setChoice}
            options={[
              { value: 'none', label: 'none' },
              ...forecast.map((t) => ({ value: t.name, label: <span>{t.name}</span> })),
            ]}
          />
        </div>
      </div>
      <AnimatePresence mode="wait" initial={false}>
        <motion.div key={choice} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -12 }} transition={{ duration: 0.25 }}>
          {narrow ? (
            <CompactSchema specs={specs} task={task} />
          ) : (
            <SchemaDiagram
              tables={specs}
              columnsOf={columnsOf}
              highlightTables={task ? [task.name] : []}
              highlightColumns={task ? [`${task.name}.${task.targetCol}`] : []}
              selectedTable={task?.name ?? null}
            />
          )}
        </motion.div>
      </AnimatePresence>
      <p className="mt-1 px-1 text-[15px] leading-snug text-muted">
        Database tables: only key and time columns are listed.
        {task && (
          <>
            {' '}
            The task table is listed in full; its label column <span className="text-ink-2">{task.targetCol}</span>, the cell a
            seed masks, is shaded.
          </>
        )}
      </p>

      <div className="mt-5 rounded-xl border border-border bg-surface-2 px-4 py-3">
        <div className="text-[15.5px] text-ink-2">
          Children of <span className="text-ink">{entityTable}[{entityRow}]</span> ({entityName(db, entityTable, entityRow)}) that pass
          the sampler’s table test for {task ? <>a <span className="text-ink">{task.name}</span> seed</> : 'a seed in a database table'}{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:415-419" /> (the temporal filter and the width bound come after it):
        </div>
        <ul className="mt-2 flex flex-wrap gap-2" aria-label="Child rows by table">
          {children.map((c) => (
            <motion.li
              key={`${c.table}:${c.type}`}
              layout
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="inline-flex items-center gap-2 rounded-lg border border-border bg-surface px-2.5 py-1 text-[15px]"
            >
              <TableSwatch slot={slots[c.table] ?? 0} label={c.type === 'Db' ? c.table : `${c.table}:${c.type}`} />
              <span className="num font-semibold text-ink">{c.n}</span>
            </motion.li>
          ))}
        </ul>
        <p className="mt-2 text-[15px] leading-snug text-ink-2">
          {task
            ? `The ${task.name} rows hang off the ${task.entityTable!.replace(/s$/, '')} through one foreign key, just like its other child rows. `
            : 'With no task table attached, the entity keeps only its database children; this is also what an autocomplete seed (a database row) gets. '}
          The stored P→F list of {entityTable}[{entityRow}] holds <span className="num text-ink">{stored}</span> children, because
          preprocessing indexes every task table <Cite kind="code" ref="rustler/src/pre.rs:79-92" />; the sampler skips the{' '}
          <span className="num text-ink">{skipped}</span> that belong to {task ? 'other task tables' : 'task tables'}.
        </p>
      </div>
    </div>
  );
}
