/**
 * TokenTable: the batch dictionary, one row per token, one column per field
 * (spec §5.1; fly.rs:27-44, 462-500; data.py:116-154). Virtualised rows.
 *
 * Field names are the official batch keys. Display-only columns ("cell") are
 * marked as such: the model never reads them.
 */
import { useMemo, useRef, useState } from 'react';
import { SEM_TYPE_NAMES, type Token } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { cx } from '../ui/cx';
import { formatSigned, truncate } from './format';
import { textWidth } from './text';
import { tokenValueText } from './TokenStrip';
import { toSet } from './useSize';

export type TokenField =
  | 'pos'
  | 'cell'
  | 'node_idxs'
  | 'f2p_nbr_idxs'
  | 'table_name_idxs'
  | 'col_name_idxs'
  | 'sem_types'
  | 'number_values'
  | 'text_values'
  | 'datetime_values'
  | 'boolean_values'
  | 'masks'
  | 'is_targets'
  | 'is_task_nodes'
  | 'class_value_idxs'
  | 'is_padding';

export const ALL_TOKEN_FIELDS: TokenField[] = [
  'pos',
  'cell',
  'node_idxs',
  'f2p_nbr_idxs',
  'table_name_idxs',
  'col_name_idxs',
  'sem_types',
  'number_values',
  'text_values',
  'datetime_values',
  'boolean_values',
  'masks',
  'is_targets',
  'is_task_nodes',
  'class_value_idxs',
  'is_padding',
];

/**
 * Field → whether the paper-era model reads it (spec §5.1 "used by model?"). 'eval' = read only by the
 * evaluation readout (main.py:283-288), never by model.py.
 */
export const FIELD_USED_BY_MODEL: Record<TokenField, boolean | 'display' | 'eval'> = {
  pos: 'display',
  cell: 'display',
  node_idxs: true,
  f2p_nbr_idxs: true,
  table_name_idxs: true,
  col_name_idxs: true,
  sem_types: true,
  number_values: true,
  text_values: true,
  datetime_values: true,
  boolean_values: true,
  masks: true,
  is_targets: 'eval', // eval readout only (main.py:283-288)
  is_task_nodes: false,
  class_value_idxs: false,
  is_padding: true,
};

const WIDTH: Record<TokenField, number> = {
  pos: 52,
  cell: 330,
  node_idxs: 84,
  f2p_nbr_idxs: 132,
  table_name_idxs: 112,
  col_name_idxs: 104,
  sem_types: 128,
  number_values: 116,
  text_values: 160,
  datetime_values: 120,
  boolean_values: 116,
  masks: 64,
  is_targets: 84,
  is_task_nodes: 100,
  class_value_idxs: 122,
  is_padding: 86,
};

export interface TokenTableProps {
  tokens: readonly Token[];
  /** The DB string table (PreprocessedDB.text) to resolve text/column indices. Optional. */
  text?: readonly string[];
  tableSlots?: Record<string, number>;
  /** Which columns to show (default: all, in batch order). */
  fields?: TokenField[];
  highlight?: ReadonlySet<number> | readonly number[];
  selected?: number | null;
  onHover?: (pos: number | null) => void;
  onSelect?: (pos: number) => void;
  /** Max height of the scroll area (px). Default 420. */
  maxHeight?: number;
  /** Hide padding rows. Default false (padding is part of the batch). */
  hidePadding?: boolean;
  /** Per-column minimum widths (px) that override the defaults, e.g. to show f2p_nbr_idxs in full. */
  colWidths?: Partial<Record<TokenField, number>>;
  /** Characters of the table name shown after table_name_idxs (default 12; 0 = id only). */
  tableNameChars?: number;
  className?: string;
}

const ROW_H = 32;
/** Column width: the content minimum above, but never narrower than the (measured) field name. */
const colW = (f: TokenField, over?: Partial<Record<TokenField, number>>) =>
  Math.max(over?.[f] ?? WIDTH[f], Math.ceil(textWidth(f, 14, { mono: true, weight: 600 })) + 22);
const OVERSCAN = 8;

const bool = (b: boolean) => (b ? 'true' : 'false');

export function TokenTable({
  tokens,
  text,
  tableSlots = {},
  fields = ALL_TOKEN_FIELDS,
  highlight,
  selected = null,
  onHover,
  onSelect,
  maxHeight = 420,
  hidePadding = false,
  colWidths,
  tableNameChars = 12,
  className,
}: TokenTableProps) {
  const rows = useMemo(() => (hidePadding ? tokens.filter((t) => !t.isPadding) : tokens), [tokens, hidePadding]);
  const hl = toSet(highlight);
  const scroller = useRef<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const viewH = Math.min(maxHeight, ROW_H * rows.length + ROW_H);
  const start = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN);
  const end = Math.min(rows.length, Math.ceil((scrollTop + viewH) / ROW_H) + OVERSCAN);
  const totalW = fields.reduce((s, f) => s + colW(f, colWidths), 0);

  const cellText = (t: Token, f: TokenField): string => {
    switch (f) {
      case 'pos':
        return String(t.pos);
      case 'cell':
        return t.isPadding
          ? '(padding)'
          : `${t.tableName}${t.tableType && t.tableType !== 'Db' ? `:${t.tableType}` : ''}[${t.rowPos}].${t.colName} = ${t.isTarget ? '?' : tokenValueText(t)}`;
      case 'node_idxs':
        return String(t.nodeIdx);
      case 'f2p_nbr_idxs':
        return `[${t.f2pNbrIdxs.join(',')}]`;
      case 'table_name_idxs':
        return text && !t.isPadding && tableNameChars > 0 ? `${t.tableNameIdx} ${truncate(text[t.tableNameIdx] ?? '', tableNameChars)}` : String(t.tableNameIdx);
      case 'col_name_idxs':
        return String(t.colNameIdx);
      case 'sem_types':
        return `${t.sem} ${SEM_TYPE_NAMES[t.sem] ?? ''}`;
      case 'number_values':
        return formatSigned(t.numberValue);
      case 'text_values':
        return t.isPadding ? 'all zeros' : `E(text[${t.textIdx}])${text && t.sem === 1 ? ` “${truncate(text[t.textIdx] ?? '', 14)}”` : ''}`;
      case 'datetime_values':
        return formatSigned(t.datetimeValue);
      case 'boolean_values':
        return formatSigned(t.booleanValue);
      case 'masks':
        return bool(t.mask);
      case 'is_targets':
        return bool(t.isTarget);
      case 'is_task_nodes':
        return bool(t.isTaskNode);
      case 'class_value_idxs':
        return String(t.classValueIdx);
      case 'is_padding':
        return bool(t.isPadding);
    }
  };

  return (
    <div className={cx('card overflow-hidden', className)}>
      <div
        ref={scroller}
        className="scroll-quiet overflow-auto"
        style={{ maxHeight }}
        onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
        role="table"
        aria-label="Batch dictionary: one row per token"
        aria-rowcount={rows.length + 1}
      >
        <div style={{ width: totalW, minWidth: '100%' }}>
          <div role="row" className="sticky top-0 z-10 flex border-b border-grid bg-surface-2" style={{ height: ROW_H }}>
            {fields.map((f) => (
              <div
                key={f}
                role="columnheader"
                title={
                  FIELD_USED_BY_MODEL[f] === 'display'
                    ? 'display only'
                    : FIELD_USED_BY_MODEL[f] === 'eval'
                      ? 'read by the evaluation readout (main.py:283-288), not by the model'
                      : FIELD_USED_BY_MODEL[f]
                        ? 'read by the model'
                        : 'carried in the batch, never read by the paper-era model'
                }
                className={cx(
                  'flex shrink-0 items-center gap-1 px-2 font-mono text-[14px] font-semibold whitespace-nowrap',
                  FIELD_USED_BY_MODEL[f] === false ? 'text-muted' : FIELD_USED_BY_MODEL[f] === 'display' ? 'text-ink-2 italic' : 'text-ink',
                )}
                style={{ width: colW(f, colWidths) }}
              >
                {f}
              </div>
            ))}
          </div>
          <div style={{ height: rows.length * ROW_H, position: 'relative' }}>
            {rows.slice(start, end).map((t, i) => {
              const top = (start + i) * ROW_H;
              const isSel = selected === t.pos;
              const isHl = hl.has(t.pos);
              return (
                <div
                  key={t.pos}
                  role="row"
                  aria-rowindex={start + i + 2}
                  aria-selected={onSelect ? isSel : undefined}
                  tabIndex={onSelect ? 0 : undefined}
                  onPointerEnter={() => onHover?.(t.pos)}
                  onPointerLeave={() => onHover?.(null)}
                  onFocus={() => onHover?.(t.pos)}
                  onClick={() => onSelect?.(t.pos)}
                  onKeyDown={(e) => {
                    if ((e.key === 'Enter' || e.key === ' ') && onSelect) {
                      e.preventDefault();
                      onSelect(t.pos);
                    }
                  }}
                  className={cx(
                    'absolute left-0 flex w-full border-b border-grid/70 font-mono text-[14.5px] tabular-nums',
                    t.isPadding && 'text-muted',
                    isSel ? 'bg-accent-soft' : isHl ? 'bg-surface-2' : 'hover:bg-surface-2',
                    onSelect && 'cursor-pointer',
                  )}
                  style={{ top, height: ROW_H }}
                >
                  {fields.map((f) => (
                    <div
                      key={f}
                      role="cell"
                      className={cx(
                        'relative flex shrink-0 items-center truncate px-2 whitespace-nowrap',
                        t.isTarget && (f === 'masks' || f === 'is_targets') && 'font-bold text-accent-text',
                      )}
                      style={{ width: colW(f, colWidths) }}
                      title={f === 'cell' || f === 'text_values' || f === 'f2p_nbr_idxs' || f === 'table_name_idxs' ? cellText(t, f) : undefined}
                    >
                      {f === 'cell' && !t.isPadding && (
                        <span
                          aria-hidden="true"
                          className="mr-1.5 inline-block h-2.5 w-2.5 shrink-0 rounded-[3px]"
                          style={{ background: catVar(tableSlots[t.tableName] ?? 0) }}
                        />
                      )}
                      <span className={cx('truncate', f === 'cell' && t.isTarget && 'target-outline rounded-[3px] px-0.5')}>
                        {cellText(t, f)}
                      </span>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-grid px-3 py-2 text-[14.5px] text-muted">
        <span>
          <span className="font-semibold text-ink">bold</span> = read by the model
        </span>
        <span>gray = carried in the batch but never read</span>
        <span className="italic">italic = display only</span>
        <span className="num">{rows.length} rows</span>
      </div>
    </div>
  );
}
