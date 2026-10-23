/**
 * TableView: one relational table, virtualised rows, with key/time column
 * markers, sem-type badges on feature columns, null as "∅", a cell/row
 * highlight API, the masked target cell, and an optional raw ↔ z-scored toggle.
 *
 * Semantics shown (spec §1, §3): the PK column produces no token (row position
 * = PK, §1.2); FK columns produce no token, only links; every other non-null
 * cell is a feature cell whose sem type follows its physical dtype (§3.2).
 */
import { KeyRound, Link2 } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type { DType, RawValue, TableSchema } from '@/engine/types';
import { SemType } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { Segmented } from '../ui/controls';
import { cx } from '../ui/cx';
import { formatRaw, formatSigned, NULL_GLYPH, semOfDType } from './format';
import { SemBadge } from './SemBadge';
import { textWidth } from './text';
import { toSet } from './useSize';

export interface TableViewProps {
  schema: TableSchema;
  rows: readonly Record<string, RawValue>[];
  /** Table color slot (left border + header swatch). */
  slot?: number;
  /** Display title (default schema.name); e.g. "customer-churn (Train)". */
  title?: ReactNode;
  /** Columns to show (default all, in schema order). */
  columns?: string[];
  /**
   * z-scored value for a cell (PreprocessedDB cells), or null/undefined if the
   * cell has no number/datetime/boolean encoding. Enables the raw ↔ z toggle.
   */
  zValue?: (row: number, col: string) => number | null | undefined;
  /** Controlled display mode; uncontrolled when omitted. */
  mode?: 'raw' | 'z';
  onModeChange?: (mode: 'raw' | 'z') => void;
  highlightRows?: ReadonlySet<number> | readonly number[];
  /** Cell keys "row:col". */
  highlightCells?: ReadonlySet<string> | readonly string[];
  /** Rows to dim. */
  mutedRows?: ReadonlySet<number> | readonly number[];
  /** Cells to strike through and dim, keyed "row:col"; the value is the cell's tooltip (e.g. why it produced no token). */
  mutedCells?: ReadonlyMap<string, string>;
  /** The masked target cell (dashed accent outline + "?"). */
  targetCell?: { row: number; col: string } | null;
  /** Show the target's true value instead of "?" (e.g. in a "reveal" mode). */
  revealTarget?: boolean;
  onRowHover?: (row: number | null) => void;
  onRowClick?: (row: number) => void;
  onCellClick?: (row: number, col: string) => void;
  /** Scroll this row into view when it changes. */
  scrollToRow?: number | null;
  /** Scroll this column into view (horizontally) when it changes. */
  scrollToCol?: string | null;
  /** Max scroll height (px). Default 360. */
  maxHeight?: number;
  /** Optional row subset (row positions) to display, in this order. */
  rowFilter?: readonly number[];
  /** Text shown in the footer (e.g. "1,204 rows"). */
  footer?: ReactNode;
  className?: string;
}

const ROW_H = 32;
const OVERSCAN = 10;

type RawRow = Readonly<Record<string, RawValue>>;

/**
 * Column width from the measured header (icon + name + time tag) and a sample
 * of formatted values, so headers and values are not cut at the 14-15px type
 * sizes. Long strings still truncate (full value on hover).
 */
function measureColumn(name: string, dtype: string, isTime: boolean, isKey: boolean, sample: readonly RawRow[]): number {
  const header = (isKey ? 13 : 18) + 6 + textWidth(name, 14.5, { mono: true, weight: 600 }) + (isTime ? 6 + textWidth('time', 13.5) + 8 : 0) + 18;
  let content = 0;
  for (const r of sample) {
    const v = r[name];
    if (v === null || v === undefined) continue;
    content = Math.max(content, textWidth(formatRaw(v, dtype as DType), 15, { mono: dtype !== 'str' }));
  }
  content += 16 + 8 + 4; // cell padding + value pill padding
  const cap = dtype === 'str' ? 280 : 210;
  return Math.ceil(Math.min(cap, Math.max(72, header, content)));
}

export function TableView({
  schema,
  rows,
  slot = 0,
  title,
  columns,
  zValue,
  mode: modeProp,
  onModeChange,
  highlightRows,
  highlightCells,
  mutedRows,
  mutedCells,
  targetCell,
  revealTarget = false,
  onRowHover,
  onRowClick,
  onCellClick,
  scrollToRow,
  scrollToCol,
  maxHeight = 360,
  rowFilter,
  footer,
  className,
}: TableViewProps) {
  const [modeState, setModeState] = useState<'raw' | 'z'>('raw');
  const mode = modeProp ?? modeState;
  const setMode = (m: 'raw' | 'z') => {
    if (modeProp === undefined) setModeState(m);
    onModeChange?.(m);
  };
  const cols = useMemo(() => {
    const names = columns ?? schema.columns.map((c) => c.name);
    return names
      .map((n) => schema.columns.find((c) => c.name === n))
      .filter((c): c is NonNullable<typeof c> => !!c);
  }, [schema, columns]);

  const order = useMemo(() => (rowFilter ? [...rowFilter] : rows.map((_, i) => i)), [rowFilter, rows]);
  const hlRows = toSet(highlightRows);
  const hlCells = toSet(highlightCells);
  const muted = toSet(mutedRows);

  const scroller = useRef<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const viewH = Math.min(maxHeight, ROW_H * (order.length + 1));
  const start = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN);
  const end = Math.min(order.length, Math.ceil((scrollTop + viewH) / ROW_H) + OVERSCAN);

  // vertical target of a smooth scroll started in this commit, so a horizontal scroll (scrollToCol) does not cancel it
  const pendingTop = useRef<number | null>(null);
  useEffect(() => {
    if (scrollToRow === null || scrollToRow === undefined || !scroller.current) return;
    const idx = order.indexOf(scrollToRow);
    if (idx < 0) return;
    const el = scroller.current;
    const top = idx * ROW_H;
    if (top < el.scrollTop || top + ROW_H * 2 > el.scrollTop + el.clientHeight) {
      // snap to whole rows so no row is left half-hidden under the sticky header
      const target = Math.max(0, Math.round((top - el.clientHeight / 3) / ROW_H) * ROW_H);
      el.scrollTo({ top: target, behavior: 'smooth' });
      pendingTop.current = target;
      requestAnimationFrame(() => (pendingTop.current = null));
    }
  }, [scrollToRow, order]);

  const fkCols = schema.fkeys;
  const idxW = Math.ceil(Math.max(48, textWidth(String(Math.max(0, rows.length - 1)), 14.5, { mono: true }) + 20));
  const widths = useMemo(() => {
    const step = Math.max(1, Math.floor(rows.length / 300));
    const sample: RawRow[] = [];
    for (let i = 0; i < rows.length; i += step) if (rows[i]) sample.push(rows[i]);
    const out: Record<string, number> = {};
    for (const c of cols) {
      const isKey = c.name === schema.pkey || c.name in schema.fkeys;
      out[c.name] = measureColumn(c.name, c.dtype, c.name === schema.timeCol, isKey, sample);
    }
    return out;
  }, [cols, rows, schema]);
  const colWidth = (name: string) => widths[name] ?? 120;
  const totalW = idxW + cols.reduce((s, c) => s + colWidth(c.name), 0);

  useEffect(() => {
    if (!scrollToCol || !scroller.current) return;
    const i = cols.findIndex((c) => c.name === scrollToCol);
    if (i < 0) return;
    const el = scroller.current;
    const left = idxW + cols.slice(0, i).reduce((s, c) => s + colWidth(c.name), 0);
    const right = left + colWidth(scrollToCol);
    if (left < el.scrollLeft || right > el.scrollLeft + el.clientWidth) {
      // center the column (the row-position column scrolls with the table)
      const target = Math.max(0, left - Math.max(0, (el.clientWidth - (right - left)) / 2));
      el.scrollTo(pendingTop.current !== null ? { left: target, top: pendingTop.current, behavior: 'smooth' } : { left: target, behavior: 'smooth' });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scrollToCol, cols, widths]);

  const kindOf = (name: string): 'pk' | 'fk' | 'feature' =>
    name === schema.pkey ? 'pk' : name in fkCols ? 'fk' : 'feature';

  return (
    <div className={cx('card overflow-hidden', className)} style={{ borderLeft: slot ? `3px solid ${catVar(slot)}` : undefined }}>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-grid px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate font-mono text-[16px] font-semibold text-ink">{title ?? schema.name}</span>
          <span className="num shrink-0 text-[14.5px] text-muted">{order.length.toLocaleString('en-US')} {order.length === 1 ? 'row' : 'rows'}</span>
        </div>
        {zValue && (
          <Segmented
            label="Value display"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'raw', label: 'raw' },
              { value: 'z', label: 'z-scored', title: 'Values as stored for the model (spec §3.3)' },
            ]}
          />
        )}
      </div>
      <div
        ref={scroller}
        className="scroll-quiet overflow-auto"
        style={{ maxHeight }}
        onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
        role="table"
        aria-label={`Table ${schema.name}`}
        aria-rowcount={order.length + 1}
      >
        <div style={{ width: totalW, minWidth: '100%' }}>
          <div role="row" className="sticky top-0 z-10 flex border-b border-grid bg-surface-2" style={{ height: ROW_H + 4 }}>
            <div role="columnheader" className="flex shrink-0 items-center px-2 font-mono text-[14px] text-muted" style={{ width: idxW }} title="row position (= primary-key value, spec §1.2)">
              row
            </div>
            {cols.map((c) => {
              const kind = kindOf(c.name);
              const isTime = c.name === schema.timeCol;
              return (
                <div
                  key={c.name}
                  role="columnheader"
                  className="flex shrink-0 items-center gap-1.5 px-2 text-[14.5px]"
                  style={{ width: colWidth(c.name) }}
                  title={
                    kind === 'pk'
                      ? 'Primary key: produces no token (spec §1.2)'
                      : kind === 'fk'
                        ? `Foreign key → ${fkCols[c.name]}: produces no token, only a link (spec §1.2)`
                        : `${c.description ?? c.name}${isTime ? ' · time column (also a datetime token)' : ''}`
                  }
                >
                  {kind === 'pk' ? (
                    <KeyRound aria-label="primary key" size={13} className="shrink-0 text-ink-2" />
                  ) : kind === 'fk' ? (
                    <Link2 aria-label="foreign key" size={13} className="shrink-0 text-ink-2" />
                  ) : (
                    <SemBadge sem={semOfDType(c.dtype)} size="xs" />
                  )}
                  <span className={cx('truncate font-mono font-semibold', kind === 'feature' ? 'text-ink' : 'text-ink-2')}>{c.name}</span>
                  {isTime && <span className="shrink-0 rounded bg-surface-3 px-1 text-[13.5px] font-medium text-ink-2">time</span>}
                </div>
              );
            })}
          </div>
          <div style={{ height: order.length * ROW_H, position: 'relative' }}>
            {order.slice(start, end).map((r, i) => {
              const rec = rows[r];
              if (!rec) return null;
              const top = (start + i) * ROW_H;
              const isHl = hlRows.has(r);
              return (
                <div
                  key={r}
                  role="row"
                  aria-rowindex={start + i + 2}
                  className={cx(
                    'absolute left-0 flex w-full border-b border-grid/70 text-[15px]',
                    isHl ? 'bg-accent-soft' : 'hover:bg-surface-2',
                    muted.has(r) && 'opacity-35',
                    onRowClick && 'cursor-pointer',
                  )}
                  style={{ top, height: ROW_H }}
                  onPointerEnter={() => onRowHover?.(r)}
                  onPointerLeave={() => onRowHover?.(null)}
                  onClick={() => onRowClick?.(r)}
                >
                  <div role="cell" className="num flex shrink-0 items-center px-2 text-[14.5px] text-muted" style={{ width: idxW }}>
                    {r}
                  </div>
                  {cols.map((c) => {
                    const v = rec[c.name];
                    const kind = kindOf(c.name);
                    const isTarget = !!targetCell && targetCell.row === r && targetCell.col === c.name;
                    const isNull = v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v));
                    let text: string;
                    if (isTarget && !revealTarget) text = '?';
                    else if (mode === 'z' && kind === 'feature' && zValue && !isNull) {
                      const z = zValue(r, c.name);
                      text = z === null || z === undefined ? (semOfDType(c.dtype) === SemType.Text ? `“${formatRaw(v, c.dtype)}”` : '—') : formatSigned(z);
                    } else text = formatRaw(v, c.dtype);
                    const cellHl = hlCells.has(`${r}:${c.name}`);
                    const mutedTitle = mutedCells?.get(`${r}:${c.name}`);
                    return (
                      <div
                        key={c.name}
                        role="cell"
                        className={cx(
                          'flex shrink-0 items-center px-2',
                          c.dtype === 'str' ? '' : 'num',
                          isNull && 'text-muted',
                          kind !== 'feature' && 'text-ink-2',
                          onCellClick && 'cursor-pointer',
                        )}
                        style={{ width: colWidth(c.name) }}
                        onClick={
                          onCellClick
                            ? (e) => {
                                e.stopPropagation();
                                onCellClick(r, c.name);
                              }
                            : undefined
                        }
                      >
                        <span
                          className={cx(
                            'truncate rounded-[4px] px-1 py-px',
                            isTarget && 'target-outline font-bold text-accent-text',
                            cellHl && !isTarget && 'bg-accent-soft ring-1 ring-accent',
                            mutedTitle !== undefined && 'text-muted line-through opacity-70',
                          )}
                          title={
                            mutedTitle !== undefined
                              ? mutedTitle
                              : isNull
                                ? 'missing value: produces no token (spec §3.5)'
                                : c.dtype === 'str' && !isTarget
                                  ? text
                                  : undefined
                          }
                        >
                          {isNull && !isTarget ? NULL_GLYPH : text}
                        </span>
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>
      </div>
      {footer && <div className="border-t border-grid px-3 py-2 text-[14.5px] text-muted">{footer}</div>}
    </div>
  );
}
