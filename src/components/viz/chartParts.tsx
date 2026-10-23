/**
 * Shared chart chrome (dataviz skill): legend (≥ 2 series), table-view twin,
 * tooltip box, and the rounded-data-end bar path.
 */
import { Table2 } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { catVar } from '@/styles/palette';
import { cx } from '../ui/cx';

export interface LegendItem {
  id: string;
  label: string;
  slot: number;
  /** 'line' key for line charts, 'rect' for bars (legend mirrors the mark). */
  mark: 'line' | 'rect';
  muted?: boolean;
}

/** Horizontal legend; text stays in ink, the key carries the color. */
export function Legend({ items, className }: { items: LegendItem[]; className?: string }) {
  if (items.length < 2) return null;
  return (
    <ul className={cx('flex flex-wrap items-center gap-x-4 gap-y-1 text-[15px] text-ink-2', className)} aria-label="Legend">
      {items.map((it) => (
        <li key={it.id} className={cx('inline-flex items-center gap-1.5', it.muted && 'opacity-50')}>
          {it.mark === 'line' ? (
            <span aria-hidden="true" className="inline-block h-[2px] w-4 rounded-full" style={{ background: catVar(it.slot) }} />
          ) : (
            <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: catVar(it.slot) }} />
          )}
          {it.label}
        </li>
      ))}
    </ul>
  );
}

/** Figure frame with a title row, optional legend, and a chart ↔ table toggle. */
export function ChartFrame({
  title,
  subtitle,
  legend,
  table,
  children,
  className,
  defaultTable = false,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  legend?: ReactNode;
  /** The table-view twin (always provide one: tooltips enhance, never gate). */
  table?: ReactNode;
  children: ReactNode;
  className?: string;
  defaultTable?: boolean;
}) {
  const [showTable, setShowTable] = useState(defaultTable);
  return (
    <div className={cx('min-w-0', className)}>
      {(title || table) && (
        <div className="mb-2 flex items-start justify-between gap-3">
          <div className="min-w-0">
            {title && <div className="text-[17px] font-semibold text-ink">{title}</div>}
            {subtitle && <div className="mt-0.5 text-[15px] text-ink-2">{subtitle}</div>}
          </div>
          {table && (
            <button
              type="button"
              onClick={() => setShowTable((v) => !v)}
              aria-pressed={showTable}
              className={cx(
                'inline-flex h-8 shrink-0 items-center gap-1.5 rounded-md px-2.5 text-[15px] font-medium transition-colors',
                showTable ? 'bg-accent-soft text-accent-text' : 'text-ink-2 hover:bg-surface-2 hover:text-ink',
              )}
            >
              <Table2 aria-hidden="true" size={15} />
              {showTable ? 'Chart' : 'Table'}
            </button>
          )}
        </div>
      )}
      {legend && !showTable && <div className="mb-2">{legend}</div>}
      {showTable ? <div className="scroll-quiet max-h-[420px] overflow-auto">{table}</div> : children}
    </div>
  );
}

/** Plain data table used as the accessible twin of a chart. */
export function DataTable({ columns, rows, caption }: { columns: string[]; rows: (string | number)[][]; caption?: string }) {
  return (
    <table className="w-full border-collapse text-[15px]">
      {caption && <caption className="sr-only">{caption}</caption>}
      <thead>
        <tr className="border-b border-grid text-left text-ink-2">
          {columns.map((c, i) => (
            <th key={i} scope="col" className={cx('px-2 py-1.5 font-semibold', i > 0 && 'text-right')}>
              {c}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className="border-b border-grid/70">
            {r.map((v, j) => (
              <td key={j} className={cx('px-2 py-1', j > 0 ? 'num text-right' : 'text-ink')}>
                {v}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Tooltip box (values lead, labels follow; line keys, not boxes). */
export function TooltipBox({ children, style }: { children: ReactNode; style?: React.CSSProperties }) {
  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-20 max-w-[280px] min-w-[120px] rounded-[9px] border border-border bg-surface px-3 py-2 text-[15px] leading-snug shadow-[var(--shadow-pop)]"
      style={style}
    >
      {children}
    </div>
  );
}

/**
 * Bar path with a 4px rounded data-end and a square baseline end
 * (dataviz marks-and-anatomy). Horizontal: from x0 (baseline) to x1 (data end).
 */
export function hBarPath(x0: number, x1: number, y: number, h: number, r = 4): string {
  const w = Math.abs(x1 - x0);
  const rr = Math.min(r, w, h / 2);
  if (w < 0.5) return `M${x0},${y}h0.5v${h}h-0.5Z`;
  if (x1 >= x0) {
    return `M${x0},${y}H${x1 - rr}Q${x1},${y} ${x1},${y + rr}V${y + h - rr}Q${x1},${y + h} ${x1 - rr},${y + h}H${x0}Z`;
  }
  return `M${x0},${y}H${x1 + rr}Q${x1},${y} ${x1},${y + rr}V${y + h - rr}Q${x1},${y + h} ${x1 + rr},${y + h}H${x0}Z`;
}

/** Vertical bar path: baseline y0, data end y1. */
export function vBarPath(x: number, w: number, y0: number, y1: number, r = 4): string {
  const h = Math.abs(y1 - y0);
  const rr = Math.min(r, h, w / 2);
  if (h < 0.5) return `M${x},${y0}h${w}v-0.5h${-w}Z`;
  if (y1 <= y0) {
    return `M${x},${y0}V${y1 + rr}Q${x},${y1} ${x + rr},${y1}H${x + w - rr}Q${x + w},${y1} ${x + w},${y1 + rr}V${y0}Z`;
  }
  return `M${x},${y0}V${y1 - rr}Q${x},${y1} ${x + rr},${y1}H${x + w - rr}Q${x + w},${y1} ${x + w},${y1 - rr}V${y0}Z`;
}
