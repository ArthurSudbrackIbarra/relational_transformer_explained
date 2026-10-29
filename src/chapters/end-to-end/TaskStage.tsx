/**
 * Stage 1: the question. Pick a database, a task and a seed row (writing the global store, so every
 * chapter follows), then read the seed row as the sampler will: which cells become tokens, which
 * one is masked, which columns are links, and what the model will output.
 *
 * Forecasting tasks seed on a task-table row (timestamp, one FK to the entity, the label); an
 * autocomplete task seeds on a database row and masks one of its columns, with leakage columns
 * dropped on the fly (spec §2.1-2.4, §4.9; rt/data.py:31-108, rustler/src/fly.rs:454-511).
 */
import { Shuffle } from 'lucide-react';
import { useMemo } from 'react';
import { Cite, Figure, IconButton, SemBadge, Segmented, Select, StatusBadge, TableSwatch, cx, formatDate, formatRaw } from '@/components';
import { getDatabase, listDatabases } from '@/engine/db/registry';
import { autoCheckpoint, CHECKPOINT_INFO } from '@/engine/pipeline';
import { semOfDType } from '@/components/viz/format';
import { SEM_TYPE_NAMES, type RawValue, type Split } from '@/engine/types';
import { useSeeds } from '@/state/hooks';
import { useSim, type DbName } from '@/state/store';
import { rawRowOf, rawTableOf } from '../sampling/trace';
import { int, singular } from './format';
import { runKind, useE2E } from './live';

const NAME_COLS = ['name', 'display_name', 'title', 'circuit'];

/** "Ana Souza" for a DB row with a name column, else "customer 35". */
export function entityName(db: ReturnType<typeof getDatabase>, table: string, rowPos: number): string {
  const t = db.tables.find((x) => x.schema.name === table);
  const r = t?.rows[rowPos];
  const col = t?.schema.columns.find((c) => NAME_COLS.includes(c.name))?.name;
  if (r && col && typeof r[col] === 'string') return r[col] as string;
  return `${singular(table)} ${rowPos}`;
}

function rawText(v: RawValue, dtype?: string): string {
  if (v === null) return '∅';
  if (dtype === 'datetime' && typeof v === 'number') return formatDate(v, 'never');
  return formatRaw(v);
}

export function TaskStage() {
  const live = useE2E();
  const { db, pdb, task, seedNode, assets } = live;
  const dbs = useMemo(() => listDatabases(), []);
  const seeds = useSeeds();
  const split = useSim((s) => s.split);
  const stored = useSim((s) => s.seedNode);
  const setDb = useSim((s) => s.setDb);
  const setTask = useSim((s) => s.setTask);
  const setSplit = useSim((s) => s.setSplit);
  const setSeedNode = useSim((s) => s.setSeedNode);
  const auto = autoCheckpoint(task.id);
  const kind = runKind(live);

  const seed = pdb.nodes[seedNode];
  const table = rawTableOf(db, pdb, seedNode);
  const row = rawRowOf(db, pdb, seedNode);
  const seedLabels = useMemo(() => {
    const m = new Map<number, string>();
    for (const s of seeds) {
      const n = pdb.nodes[s];
      const p = n.f2pNbrIdxs[0];
      const ent = task.kind === 'forecast' && p !== undefined ? entityName(db, pdb.nodes[p].tableName, pdb.nodes[p].rowPos) : '';
      const date = n.ts !== null ? formatDate(n.ts * 1000, 'never') : '';
      const tbl = n.tableType !== 'Db' ? `${n.tableName}:${n.tableType}` : n.tableName;
      m.set(s, `${tbl}[${n.rowPos}]${ent ? ` · ${ent}` : ''}${date && task.kind === 'forecast' ? ` · ${date}` : ''}`);
    }
    return m;
  }, [seeds, pdb, db, task]);
  const MAX = 400;
  const shown = seeds.length > MAX ? [...seeds.slice(0, MAX), ...(seeds.slice(MAX).includes(seedNode) ? [seedNode] : [])] : seeds;

  // the seed row, column by column, as the sampler and the model see it
  const cols = useMemo(() => {
    if (!table || !row || !seed) return [];
    const drop = new Set(task.columnsToDrop);
    return table.schema.columns.map((c) => {
      const v = row[c.name] ?? null;
      const isPk = table.schema.pkey === c.name;
      const fk = table.schema.fkeys[c.name];
      const isTarget = c.name === task.targetCol;
      const role: 'pk' | 'fk' | 'target' | 'drop' | 'null' | 'cell' = isPk
        ? 'pk'
        : fk
          ? 'fk'
          : isTarget
            ? 'target'
            : drop.has(c.name)
              ? 'drop'
              : v === null || (typeof v === 'number' && Number.isNaN(v))
                ? 'null'
                : 'cell';
      return { name: c.name, dtype: c.dtype, v, fk, role };
    });
  }, [table, row, seed, task]);
  const nTokens = cols.filter((c) => c.role === 'cell' || c.role === 'target').length;
  const dbShort = (name: string) => dbs.find((d) => d.name === name)?.title.split(' ')[0] ?? name;
  const hidden = cols.find((c) => c.role === 'target')?.v ?? null;

  return (
    <Figure
      wide
      title="The question the model will answer"
      caption={
        <>
          These pickers change the simulation for the whole course, like the settings in the top bar. ShopDB is the held-out database: with the
          default checkpoints its tasks run zero-shot, while the other three databases were seen in pretraining.
        </>
      }
    >
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
        {/* pickers */}
        <div className="grid grid-cols-1 min-w-0 content-start gap-4">
          <div className="grid grid-cols-1 gap-1.5">
            <span className="text-[15px] font-medium text-ink-2">Database</span>
            <Segmented<DbName>
              label="Database"
              value={db.name as DbName}
              onChange={(name) => {
                const first = getDatabase(name).tasks[0];
                if (first) setDb(name, first.id);
              }}
              options={dbs.map((d) => ({ value: d.name, label: d.title.split(' ')[0], title: d.role === 'target' ? 'Held out from pretraining' : 'Seen in pretraining' }))}
            />
          </div>
          <div className="grid grid-cols-1 gap-1.5">
            <span className="text-[15px] font-medium text-ink-2">Task</span>
            <ul className="!m-0 grid grid-cols-1 gap-2 !p-0 sm:grid-cols-2" style={{ listStyle: 'none' }} role="radiogroup" aria-label="Task">
              {db.tasks.map((t) => {
                const on = t.id === task.id;
                return (
                  <li key={t.id} className="!m-0 !p-0">
                    <button
                      type="button"
                      role="radio"
                      aria-checked={on}
                      onClick={() => setTask(t.id)}
                      className={cx(
                        'flex h-full w-full flex-col items-start gap-1 rounded-lg border px-3 py-2.5 text-left transition-colors',
                        on ? 'border-accent bg-accent-soft' : 'border-border bg-surface hover:bg-surface-2',
                      )}
                    >
                      <span className={cx('text-[16px] font-semibold', on ? 'text-accent-text' : 'text-ink')}>{t.id.split('/')[1]}</span>
                      <span className="text-[14.5px] text-ink-2">
                        {t.kind}, {t.targetType}
                      </span>
                      <span className="text-[14.5px] leading-snug text-muted">{t.description}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-[auto_minmax(0,1fr)] sm:items-end">
            <div className="grid grid-cols-1 gap-1.5">
              <span className="text-[15px] font-medium text-ink-2">Seed split</span>
              <Segmented<Split>
                label="Seed split"
                value={split}
                onChange={setSplit}
                disabled={task.kind === 'autocomplete'}
                options={[
                  { value: 'Train', label: 'Train' },
                  { value: 'Val', label: 'Val' },
                  { value: 'Test', label: 'Test' },
                ]}
              />
            </div>
            <div className="flex min-w-0 items-end gap-1.5">
              <Select
                className="min-w-0 flex-1"
                label={`Seed row (${int(seeds.length)} in this ${task.kind === 'autocomplete' ? 'table' : 'split'})`}
                value={stored === null ? 'auto' : String(seedNode)}
                onChange={(e) => setSeedNode(e.target.value === 'auto' ? null : Number(e.target.value))}
              >
                <option value="auto">auto · {seedLabels.get(seedNode) ?? `#${seedNode}`}</option>
                {shown.map((s) => (
                  <option key={s} value={s}>
                    {seedLabels.get(s) ?? `#${s}`}
                  </option>
                ))}
              </Select>
              <IconButton
                label="Random seed row"
                size="sm"
                className="mb-0 border border-border-strong"
                onClick={() => seeds.length && setSeedNode(seeds[Math.floor(Math.random() * seeds.length)])}
              >
                <Shuffle size={14} />
              </IconButton>
            </div>
          </div>
          {task.kind === 'autocomplete' && (
            <p className="!m-0 text-[15px] leading-snug text-muted">
              An autocomplete task seeds on every row of its table that has a value in the target column, from every period, so the split does
              not apply <Cite kind="spec" ref="§2.4" />.
            </p>
          )}
        </div>

        {/* the seed row */}
        <div className="grid grid-cols-1 min-w-0 content-start gap-3">
          <div className="rounded-xl border border-border bg-surface px-4 py-3">
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
              <TableSwatch slot={live.slots[seed?.tableName ?? ''] ?? 0} />
              <span className="text-[16.5px] font-semibold text-ink">
                {seed ? (seed.tableType !== 'Db' ? `${seed.tableName}:${seed.tableType}[${seed.rowPos}]` : `${seed.tableName}[${seed.rowPos}]`) : '–'}
              </span>
              <span className="text-[15px] text-muted">
                {task.kind === 'forecast' ? `a row of the task table, ${seed?.tableType} split` : `a row of the ${dbShort(db.name)} table`} · node{' '}
                {seedNode}
              </span>
            </div>
            <div className="scroll-quiet mt-2.5 overflow-x-auto">
              <table className="!my-0 w-full min-w-[420px] border-collapse text-[15px]">
                <thead>
                  <tr className="text-left text-[14px] text-muted">
                    <th className="py-1 pr-3 font-medium">column</th>
                    <th className="py-1 pr-3 font-medium">value</th>
                    <th className="py-1 font-medium">becomes</th>
                  </tr>
                </thead>
                <tbody>
                  {cols.map((c) => (
                    <tr key={c.name} className={cx('border-t border-grid', c.role === 'target' && 'bg-accent-soft')}>
                      <td className="py-1.5 pr-3 align-top">
                        <code className="!text-[14.5px]">{c.name}</code>
                      </td>
                      <td className="num py-1.5 pr-3 align-top text-ink">
                        {c.role === 'target' ? (
                          <span className="inline-flex items-center rounded border border-dashed border-accent px-1.5 font-semibold text-accent-text">?</span>
                        ) : c.role === 'fk' && typeof c.v === 'number' ? (
                          <>
                            {c.v} <span className="text-muted">→ {entityName(db, c.fk!, c.v)}</span>
                          </>
                        ) : (
                          rawText(c.v, c.dtype)
                        )}
                      </td>
                      <td className="py-1.5 align-top text-ink-2">
                        {c.role === 'pk' && 'nothing: a primary key is not a token'}
                        {c.role === 'fk' && <>a link: the parent’s node id goes into every token’s parent list</>}
                        {c.role === 'null' && 'nothing: a missing value is not a token'}
                        {c.role === 'drop' && 'nothing: dropped as a leakage column'}
                        {(c.role === 'cell' || c.role === 'target') && (
                          <span className="inline-flex flex-wrap items-center gap-1.5">
                            <SemBadge sem={semOfDType(c.dtype)} size="xs" />
                            {c.role === 'target' ? (
                              <strong className="text-accent-text">the masked target, a {SEM_TYPE_NAMES[semOfDType(c.dtype)]} token</strong>
                            ) : (
                              `a ${SEM_TYPE_NAMES[semOfDType(c.dtype)]} token`
                            )}
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="!mt-2.5 !mb-0 text-[15px] leading-snug text-ink-2">
              {nTokens === 1 ? 'One cell' : `${nTokens} cells`} of this row become tokens, the first of the context: the sampler visits the seed
              row before any other <Cite kind="code" ref="rustler/src/fly.rs:366" /> <Cite kind="code" ref="rustler/src/fly.rs:454-500" />. The label is still stored in the target token (
              <span className="num">{rawText(hidden)}</span>), but the model never reads it: the input embedding uses a learned mask vector in its place{' '}
              <Cite kind="spec" ref="§5.3" />.
            </p>
          </div>
          <div className="grid grid-cols-1 gap-2 rounded-xl bg-surface-2 px-4 py-3 text-[15.5px] leading-snug text-ink-2">
            <div>
              <span className="font-semibold text-ink">What comes out: </span>
              {task.targetType === 'binary' ? (
                <>
                  one logit from the boolean decoder at the masked position; its sigmoid is the probability that <code>{task.targetCol}</code> is true{' '}
                  <Cite kind="code" ref="rt/main.py:283-285" /> <Cite kind="spec" ref="§8.4" />.
                </>
              ) : (
                <>
                  one z-score from the number decoder at the masked position; times the column’s standard deviation plus its mean, it is{' '}
                  <code>{task.targetCol}</code> in its own units <Cite kind="code" ref="rt/main.py:286-288" /> <Cite kind="spec" ref="§8.4" />.
                </>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
              <span className="font-semibold text-ink">Weights:</span>
              <code>{assets.ckpt}</code>
              {assets.weightsKind === 'init' && <StatusBadge status="warning">untrained</StatusBadge>}
              {kind === 'zero-shot' && <StatusBadge status="good">zero-shot</StatusBadge>}
              {kind === 'seen' && <StatusBadge status="info">database seen in pretraining</StatusBadge>}
              {kind === 'finetuned' && <StatusBadge status="info">fine-tuned on ShopDB</StatusBadge>}
              <span className="basis-full text-[15px] text-muted">{assets.ckpt === auto.name ? auto.note : CHECKPOINT_INFO[assets.ckpt]}</span>
            </div>
          </div>
        </div>
      </div>
    </Figure>
  );
}
