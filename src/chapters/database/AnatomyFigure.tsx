/**
 * "Anatomy of a table": a few live rows of one table, with a picker that
 * highlights what the paper calls a table, a row, a column, a cell, the
 * feature columns and the ID (key) columns (paper p.3 §2.1, p.4 §3.1).
 * On narrow screens the table scrolls sideways (edge fades show it); it only
 * scrolls to a highlighted column after the reader picks one.
 */
import { AnimatePresence, motion } from 'motion/react';
import { KeyRound, Link2 } from 'lucide-react';
import { memo, useEffect, useRef, useState, type ReactNode } from 'react';
import { Cite, Segmented, SemBadge, Term, cx, formatRaw, semOfDType } from '@/components';
import type { RawTable } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { usePrefersReducedMotion } from '@/styles/theme';
import { useScrollEdges } from './hooks';
import { columnKind, isMissing } from './model';

type Part = 'table' | 'row' | 'column' | 'cell' | 'features' | 'keys';

export const AnatomyFigure = memo(function AnatomyFigure({ table, rows, slot, dbLabel, nTables }: { table: RawTable; rows: number[]; slot: number; dbLabel: string; nTables: number }) {
  const [part, setPart] = useState<Part>('table');
  const [scroller, edges] = useScrollEdges<HTMLDivElement>();
  const picked = useRef(false);
  const reduceMotion = usePrefersReducedMotion();
  const s = table.schema;
  const feats = s.columns.filter((c) => columnKind(table, c.name) === 'feature');
  // the example column: the first numeric feature column (else the first feature column)
  const exCol = feats.find((c) => c.dtype === 'int' || c.dtype === 'float') ?? feats[0] ?? s.columns[0];
  const exRow = rows[Math.min(1, rows.length - 1)];
  const exVal = table.rows[exRow]?.[exCol.name];
  // other rows of this table whose cell in the example column holds the same value
  const sameValueRows = table.rows.reduce((n, rec, i) => (i !== exRow && rec[exCol.name] === exVal && !isMissing(exVal) ? n + 1 : n), 0);
  const sameValueShown = rows.find((r) => r !== exRow && table.rows[r]?.[exCol.name] === exVal);

  const isHlCell = (r: number, col: string) => {
    const k = columnKind(table, col);
    switch (part) {
      case 'row':
        return r === exRow;
      case 'column':
        return col === exCol.name;
      case 'cell':
        return r === exRow && col === exCol.name;
      case 'features':
        return k === 'feature';
      case 'keys':
        return k !== 'feature';
      default:
        return false;
    }
  };
  const isHlHeader = (col: string) =>
    (part === 'column' && col === exCol.name) ||
    (part === 'features' && columnKind(table, col) === 'feature') ||
    (part === 'keys' && columnKind(table, col) !== 'feature');

  // Narrow screens: after the reader picks a column or a cell, bring it into view; for the whole
  // table, a row or the key columns, scroll back to the start (where the keys are). Never on mount.
  useEffect(() => {
    if (!picked.current) return;
    const sc = scroller.current;
    if (!sc) return;
    const behavior = reduceMotion ? 'auto' : 'smooth';
    const th = part === 'column' || part === 'cell' ? sc.querySelector<HTMLElement>('th[data-hl="1"]') : null;
    if (!th) {
      if (sc.scrollLeft > 0) sc.scrollTo({ left: 0, behavior });
      return;
    }
    const left = th.offsetLeft;
    const right = left + th.offsetWidth;
    if (left < sc.scrollLeft || right > sc.scrollLeft + sc.clientWidth) sc.scrollTo({ left: Math.max(0, left - 24), behavior });
  }, [part, reduceMotion, scroller]);

  const keyCols = s.columns.filter((c) => columnKind(table, c.name) !== 'feature').map((c) => c.name);

  const DEF: Record<Part, ReactNode> = {
    table: (
      <>
        A <strong>table</strong> is a collection of rows that share the same columns. A relational database is a set of tables linked through
        inter-table relationships <Cite kind="paper" ref="p.3 §2.1" />. {dbLabel} has {nTables} of them; this one is <code>{s.name}</code>.
      </>
    ),
    row: (
      <>
        A <strong>row</strong> is one record, here one {s.name === 'orders' ? 'order' : `${s.name} record`}. Every row is a set of cells, one
        for each column <Cite kind="paper" ref="p.3 §2.1" />. RT uses whole rows: once a row is chosen, all of its non-missing feature cells go
        into its <Term id="context-window">context</Term> <Cite kind="paper" ref="p.16 App. A" />. Chapter 6 adds two small exceptions (a context
        that fills up, and leakage columns of autocomplete tasks) <Cite kind="code" ref="rustler/src/fly.rs:502-509" />
        <Cite kind="code" ref="rustler/src/fly.rs:457-461" />.
      </>
    ),
    column: (
      <>
        A <strong>column</strong> has a name and one data type for all its cells. RT turns the name into the phrase{' '}
        <code>
          “{exCol.name} of {s.name}”
        </code>{' '}
        so that every cell knows which column and table it comes from <Cite kind="spec" ref="§3.6" />.
      </>
    ),
    cell: (
      <>
        A <strong>cell</strong> is one value at one row and column. RT represents it as a triple <em>(v, c, t)</em>: the value, the column name
        and the table name <Cite kind="paper" ref="p.4 §3.1" />. The highlighted cell is{' '}
        <span className="num whitespace-nowrap text-ink">
          ({formatRaw(exVal, exCol.dtype)}, {exCol.name}, {s.name})
        </span>
        . In RT one such cell becomes one token. Notice that the triple does not say which row the cell is in:{' '}
        {sameValueShown !== undefined ? (
          <>
            <code>
              {s.name}[{sameValueShown}]
            </code>{' '}
            has the same triple
            {sameValueRows > 1 ? (
              <>
                , and so {sameValueRows - 1 === 1 ? 'does' : 'do'} <span className="num">{(sameValueRows - 1).toLocaleString('en-US')}</span> more {s.name}{' '}
                row{sameValueRows - 1 === 1 ? '' : 's'}
              </>
            ) : null}
          </>
        ) : sameValueRows > 0 ? (
          <>
            <span className="num">{sameValueRows.toLocaleString('en-US')}</span> other {s.name} row{sameValueRows === 1 ? ' has' : 's have'} the same
            triple
          </>
        ) : (
          <>any {s.name} row with that value would have the same triple</>
        )}
        . Which cells share a row reaches the model only through the attention masks (chapter 9){' '}
        <Cite kind="code" ref="rt/model.py:168-178" />.
      </>
    ),
    features: (
      <>
        <strong>Feature columns</strong> hold the information: numbers, text, datetimes and booleans <Cite kind="paper" ref="p.4 §3.1" />
        <Cite kind="code" ref="rustler/src/common.rs:38-44" />. Only their cells become tokens.
      </>
    ),
    keys: (
      <>
        <strong>Key columns</strong>, which the paper calls ID columns ({keyCols.map((c, i) => (
          <span key={c}>
            {i > 0 && ', '}
            <code>{c}</code>
          </span>
        ))}
        ), “define how rows are uniquely identified and connected across tables” <Cite kind="paper" ref="p.3 §2.1" />: the primary key and the
        foreign keys. Neither ever becomes a token. RT keeps the foreign keys as structure (links between rows) and skips the primary key
        entirely, because a row is identified by its position <Cite kind="code" ref="rustler/src/pre.rs:439-442" />
        <Cite kind="spec" ref="§1.2" />.
      </>
    ),
  };

  return (
    <div>
      <Segmented
        label="Highlight a part of the table"
        value={part}
        onChange={(p) => {
          picked.current = true;
          setPart(p);
        }}
        options={[
          { value: 'table', label: 'Table' },
          { value: 'row', label: 'Row' },
          { value: 'column', label: 'Column' },
          { value: 'cell', label: 'Cell' },
          { value: 'features', label: 'Feature columns' },
          { value: 'keys', label: 'Key (ID) columns' },
        ]}
      />
      <div className="mt-4 grid items-start gap-4">
        <div
          className={cx(
            'min-w-0 overflow-hidden rounded-xl border transition-[box-shadow,border-color] duration-300',
            part === 'table' ? 'border-accent shadow-[0_0_0_3px_var(--accent-ring)]' : 'border-border',
          )}
          style={{ borderLeft: `3px solid ${catVar(slot)}` }}
        >
          <div className="flex items-center gap-2 border-b border-grid px-3 py-1.5">
            <span className="text-[15.5px] font-semibold text-ink">{s.name}</span>
            <span className="num text-[14px] text-muted">
              {rows.length} of {table.rows.length.toLocaleString('en-US')} rows
            </span>
          </div>
          <div className="relative">
          <div
            aria-hidden="true"
            className={cx('pointer-events-none absolute inset-y-0 left-0 z-[1] w-7 transition-opacity duration-200', edges.left ? 'opacity-100' : 'opacity-0')}
            style={{ background: 'linear-gradient(to right, var(--surface), transparent)' }}
          />
          <div
            aria-hidden="true"
            className={cx('pointer-events-none absolute inset-y-0 right-0 z-[1] w-7 transition-opacity duration-200', edges.right ? 'opacity-100' : 'opacity-0')}
            style={{ background: 'linear-gradient(to left, var(--surface), transparent)' }}
          />
          <div ref={scroller} className="scroll-quiet overflow-x-auto">
            <table className="w-full border-collapse text-[15px]">
              <thead>
                <tr className="bg-surface-2">
                  {s.columns.map((c) => {
                    const k = columnKind(table, c.name);
                    return (
                      <th
                        key={c.name}
                        data-hl={isHlHeader(c.name) || (part === 'cell' && c.name === exCol.name) ? '1' : undefined}
                        scope="col"
                        className={cx(
                          'px-2.5 py-1.5 text-left font-semibold whitespace-nowrap transition-colors duration-300',
                          k === 'feature' ? 'text-ink' : 'text-ink-2',
                          isHlHeader(c.name) && 'bg-accent-soft',
                        )}
                      >
                        <span className="inline-flex items-center gap-1.5">
                          {k === 'pk' ? (
                            <KeyRound aria-label="primary key" size={15} />
                          ) : k === 'fk' ? (
                            <Link2 aria-label="foreign key" size={15} />
                          ) : (
                            <SemBadge sem={semOfDType(c.dtype)} size="sm" />
                          )}
                          {c.name}
                        </span>
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr
                    key={r}
                    className={cx('border-t border-grid/70 transition-colors duration-300', part === 'row' && r === exRow && 'bg-accent-soft')}
                  >
                    {s.columns.map((c) => {
                      const v = table.rows[r]?.[c.name];
                      const hl = isHlCell(r, c.name);
                      const k = columnKind(table, c.name);
                      return (
                        <td
                          key={c.name}
                          className={cx(
                            'px-2.5 py-1.5 whitespace-nowrap transition-colors duration-300',
                            c.dtype !== 'str' && 'num',
                            k !== 'feature' && 'text-ink-2',
                            isMissing(v) && 'text-muted',
                            hl && part !== 'row' && 'bg-accent-soft',
                          )}
                        >
                          <span className={cx('rounded px-0.5', part === 'cell' && hl && 'outline-2 outline-accent outline-solid font-semibold text-ink')}>
                            {formatRaw(v, c.dtype)}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          </div>
        </div>
        <div className="min-h-[104px] rounded-xl bg-surface-2 px-4 py-3.5 text-[17.5px] leading-[1.6] text-ink" aria-live="polite">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={part}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.18 }}
            >
              {DEF[part]}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
});
