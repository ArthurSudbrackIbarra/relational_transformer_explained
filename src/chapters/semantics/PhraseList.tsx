/**
 * Figure "Column phrases of a database": every "<col> of <table>" string pre.rs interns
 * (pre.rs:432-437), grouped by table, with how many tokens of the LIVE context carry it.
 * PK / FK phrases are interned too but never reach a token (pre.rs:439-533).
 */
import { motion, useReducedMotion } from 'motion/react';
import { useMemo } from 'react';
import { Segmented, Switch, TableSwatch, cx } from '@/components';
import { DB_NAMES, type DbName } from '@/engine/db';
import { getPreprocessed } from '@/state/hooks';
import { Phrase, RoleGlyph } from './bits';
import { DB_SHORT, ROLE_LABEL, int, type Catalog, type PhraseInfo } from './data';
import type { SemanticsLive } from './live';

export interface PhraseListProps {
  live: SemanticsLive;
  catalog: Catalog | null;
  view: DbName;
  setView: (db: DbName) => void;
  showKeys: boolean;
  setShowKeys: (v: boolean) => void;
  focus: string;
  onFocus: (phrase: string) => void;
}

export function PhraseList({ live, catalog, view, setView, showKeys, setShowKeys, focus, onFocus }: PhraseListProps) {
  const reduce = useReducedMotion();
  const isLive = view === live.dbName;
  const phrases = useMemo<PhraseInfo[]>(() => {
    if (catalog) return catalog.byDb[view];
    return [];
  }, [catalog, view]);

  // text index of each phrase in this DB's string table (column_index.json, pre.rs:641-645)
  const pdb = useMemo(() => (catalog ? getPreprocessed(catalog.dbs[view]) : null), [catalog, view]);

  const groups = useMemo(() => {
    const m = new Map<string, PhraseInfo[]>();
    for (const p of phrases) {
      if (!showKeys && !p.onToken) continue;
      if (!m.has(p.table)) m.set(p.table, []);
      m.get(p.table)!.push(p);
    }
    return [...m.entries()];
  }, [phrases, showKeys]);

  const maxCount = Math.max(1, ...[...live.tokensPerPhrase.values()]);
  const nOnToken = phrases.filter((p) => p.onToken).length;
  const nKeys = phrases.length - nOnToken;
  const used = isLive ? [...live.tokensPerPhrase.keys()].length : 0;

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-3">
        <Segmented<DbName>
          label="Database"
          value={view}
          onChange={setView}
          options={DB_NAMES.map((n) => ({ value: n, label: n === live.dbName ? `${DB_SHORT[n]} · live` : DB_SHORT[n] }))}
        />
        <Switch checked={showKeys} onChange={setShowKeys} label="Show key columns" />
      </div>

      {!catalog ? (
        <div className="flex min-h-[220px] items-center justify-center rounded-lg bg-surface-2 text-[15px] text-ink-2">Reading the four schemas…</div>
      ) : (
        <motion.div
          key={view}
          initial={reduce ? false : { opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.25 }}
          className="grid gap-3 md:grid-cols-2"
        >
          {groups.map(([table, ps]) => {
            const kind = ps[0].tableKind;
            return (
              <section key={table} aria-label={`${table} phrases`} className="min-w-0 rounded-lg border border-border bg-surface">
                <header className="flex items-center justify-between gap-2 border-b border-grid px-3 py-2">
                  <TableSwatch slot={ps[0].slot} shape label={table} size={10} />
                  <span className="text-[14px] text-muted">{kind === 'Task' ? 'task table' : 'DB table'}</span>
                </header>
                <ul className="!m-0 !list-none !p-1" role="list">
                  {ps.map((p) => {
                    const count = isLive ? live.tokensPerPhrase.get(p.text) ?? 0 : 0;
                    const selected = p.text === focus;
                    const isTarget = isLive && p.text === live.targetPhrase;
                    const idx = pdb?.columnIndex[p.text];
                    return (
                      <li key={p.text} className="!m-0 !p-0">
                        <button
                          type="button"
                          onClick={() => onFocus(p.text)}
                          aria-pressed={selected}
                          title={`${p.text} · ${ROLE_LABEL[p.role]}${idx !== undefined ? ` · text index ${idx}` : ''}`}
                          className={cx(
                            'group flex w-full min-w-0 items-center gap-2 rounded-md px-2 py-1.5 text-left transition-colors',
                            selected ? 'bg-accent-soft ring-1 ring-accent' : 'hover:bg-surface-2',
                            !p.onToken && 'opacity-70',
                          )}
                        >
                          <span className="flex w-6 shrink-0 justify-center">
                            <RoleGlyph role={p.role} dtype={p.dtype} />
                          </span>
                          <span className={cx('min-w-0 flex-1 text-[15px] leading-snug', !p.onToken && 'line-through decoration-axis')}>
                            <Phrase text={p.text} />
                            {isTarget && (
                              <span className="target-outline ml-1.5 inline-flex h-5 items-center rounded px-1 align-[1px] text-[13px] text-accent-text">
                                ? target
                              </span>
                            )}
                            {!isTarget && p.role === 'label' && (
                              <span className="ml-1.5 inline-flex h-5 items-center rounded border border-border-strong px-1 align-[1px] text-[13px] text-ink-2">
                                task label
                              </span>
                            )}
                          </span>
                          {isLive && p.onToken ? (
                            <span className="flex w-[74px] shrink-0 items-center justify-end gap-1.5" aria-label={`${count} tokens in the live context`}>
                              <span className="num text-[14px] text-ink-2">{count}</span>
                              <span className="relative h-2 w-9 rounded-full bg-surface-3">
                                <span
                                  className="absolute inset-y-0 left-0 rounded-full bg-seq-400"
                                  style={{ width: `${(100 * count) / maxCount}%` }}
                                />
                              </span>
                            </span>
                          ) : !p.onToken ? (
                            <span className="w-[74px] shrink-0 text-right text-[13.5px] text-muted">no cell</span>
                          ) : null}
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </section>
            );
          })}
        </motion.div>
      )}

      {catalog && (
        <p className="mt-3 text-[15px] leading-snug text-ink-2">
          {DB_SHORT[view]} has <strong className="text-ink">{phrases.length}</strong> column phrases: {nOnToken} can appear on a token, {nKeys}{' '}
          belong to key columns and never do.{' '}
          {isLive ? (
            <>
              The {int(live.numReal)} tokens of the live context use {used} distinct phrases (bars: tokens per phrase).
            </>
          ) : (
            <>This database is not the live one, so there are no token counts.</>
          )}{' '}
          Click a phrase to follow it through the rest of the chapter.
        </p>
      )}
    </div>
  );
}

const KIND_FILL = ['var(--seq-700)', 'var(--seq-450)', 'var(--seq-200)'];

/**
 * Figure "the string table": pdb.text in index order (text.json, pre.rs:622-630), showing how
 * table names, column phrases and string values are interleaved in order of first appearance.
 */
export function StringTable({ catalog, view }: { catalog: Catalog | null; view: DbName }) {
  const pdb = useMemo(() => (catalog ? getPreprocessed(catalog.dbs[view]) : null), [catalog, view]);
  const info = useMemo(() => {
    if (!pdb || !catalog) return null;
    const tableNames = new Set<string>([...catalog.dbs[view].tables.map((t) => t.schema.name), ...catalog.dbs[view].tasks.filter((t) => t.kind === 'forecast').map((t) => t.name)]);
    const phrases = new Set(Object.keys(pdb.columnIndex));
    const kindOf = (s: string): 'table' | 'phrase' | 'value' => (tableNames.has(s) ? 'table' : phrases.has(s) ? 'phrase' : 'value');
    const counts = { table: 0, phrase: 0, value: 0 };
    for (const s of pdb.text) counts[kindOf(s)]++;
    return { kindOf, counts, total: pdb.text.length };
  }, [pdb, catalog, view]);

  if (!pdb || !info) return <div className="min-h-[120px] rounded-lg bg-surface-2" />;
  const shown = pdb.text.slice(0, 18);
  const KIND_LABEL = { table: 'table name', phrase: 'column phrase', value: 'string value' } as const;
  return (
    <div>
      <div className="mb-3 grid grid-cols-3 gap-2">
        {(['table', 'phrase', 'value'] as const).map((k, i) => (
          <div key={k} className="rounded-lg bg-surface-2 px-3 py-2">
            <div className="flex items-center gap-1.5 text-[14.5px] text-ink-2">
              <span aria-hidden="true" className="inline-block h-2.5 w-2.5 shrink-0 rounded-[3px]" style={{ background: KIND_FILL[i] }} />
              {KIND_LABEL[k]}s
            </div>
            <div className="num text-[20px] font-semibold text-ink">{int(info.counts[k])}</div>
          </div>
        ))}
      </div>
      <div className="relative h-3 overflow-hidden rounded-full bg-surface-3" aria-hidden="true">
        {(() => {
          let x = 0;
          return (['table', 'phrase', 'value'] as const).map((k, i) => {
            const w = (100 * info.counts[k]) / info.total;
            const el = (
              <span
                key={k}
                className="absolute inset-y-0"
                style={{ left: `${x}%`, width: `calc(${w}% - 2px)`, background: KIND_FILL[i] }}
              />
            );
            x += w;
            return el;
          });
        })()}
      </div>
      <p className="mt-2 mb-3 text-[14.5px] text-ink-2">
        {int(info.total)} strings in all, one embedding row each. The first {shown.length}, in index order:
      </p>
      <ol className="scroll-quiet !m-0 flex !list-none gap-2 overflow-x-auto !p-0 pb-2" aria-label="First entries of the string table">
        {shown.map((s, i) => {
          const k = info.kindOf(s);
          return (
            <li key={i} className="!m-0 flex w-[168px] shrink-0 flex-col gap-1 rounded-lg border border-border bg-surface px-2.5 py-2">
              <span className="font-mono text-[13.5px] text-muted">text[{i}]</span>
              <span className="text-[14.5px] leading-snug break-words text-ink">{s}</span>
              <span
                className={cx(
                  'mt-auto inline-flex w-fit items-center rounded px-1.5 text-[13px]',
                  k === 'table' ? 'bg-surface-3 text-ink' : k === 'phrase' ? 'border border-border-strong text-ink' : 'border border-dashed border-border-strong text-ink-2',
                )}
              >
                {KIND_LABEL[k]}
              </span>
            </li>
          );
        })}
        <li className="!m-0 flex w-[120px] shrink-0 items-center justify-center rounded-lg border border-dashed border-border text-[15px] text-ink-2">
          … {int(info.total - shown.length)} more
        </li>
      </ol>
    </div>
  );
}
