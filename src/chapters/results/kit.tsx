/**
 * Small building blocks shared by the results chapter's figures: method markers, number formatting,
 * linear scales with "nice" ticks, a tooltip box, source badges and a faithful paper-table renderer.
 */
import { scaleLinear } from 'd3';
import { BookOpen, FlaskConical, Table2 } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { cx, shapePath, ShapeGlyph } from '@/components';
import { catVar } from '@/styles/palette';
import { LT9, type Cell } from './data';
import { METHODS, type MethodStyle } from './methods';

// ─────────────────────────────────────────────────────────────────────────────
// numbers
// ─────────────────────────────────────────────────────────────────────────────

/** One decimal with a typographic minus, as the paper prints it. */
export function f1(v: number): string {
  return v.toFixed(1).replace('-', '−');
}

/** Signed one-decimal change: "+0.6", "−16.3", "0.0". */
export function d1(v: number): string {
  const r = Math.round(v * 10) / 10;
  if (r === 0) return '0.0';
  return (r > 0 ? '+' : '−') + Math.abs(r).toFixed(1);
}

/** A table cell as printed. */
export function cellText(c: Cell): string {
  if (c === null) return '—';
  if (c === LT9) return '< −9';
  return f1(c);
}

export const isNum = (c: Cell): c is number => typeof c === 'number' && Number.isFinite(c);

export function mean(xs: number[]): number {
  return xs.reduce((s, x) => s + x, 0) / Math.max(1, xs.length);
}

export function sd(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) * (x - m), 0) / (xs.length - 1));
}

// ─────────────────────────────────────────────────────────────────────────────
// scales
// ─────────────────────────────────────────────────────────────────────────────

/** Linear x scale over [lo, hi] mapped to [x0, x1], with ticks that fit the width. */
export function xScale(domain: [number, number], x0: number, x1: number, minGap = 64) {
  const s = scaleLinear().domain(domain).range([x0, x1]);
  const n = Math.max(2, Math.floor((x1 - x0) / minGap));
  const ticks = s.ticks(n).filter((t) => t >= domain[0] - 1e-9 && t <= domain[1] + 1e-9);
  return { x: (v: number) => s(v), ticks };
}

// ─────────────────────────────────────────────────────────────────────────────
// markers
// ─────────────────────────────────────────────────────────────────────────────

/** A method marker centered at (0,0): filled with a 2px surface ring, or hollow with a 2.2px stroke. */
export function MethodMark({ m, r = 6.5, dim = false }: { m: MethodStyle; r?: number; dim?: boolean }) {
  const color = catVar(m.slot);
  return (
    <path
      d={shapePath(m.shape, r)}
      style={{
        fill: m.hollow ? 'var(--surface)' : color,
        stroke: m.hollow ? color : 'var(--surface)',
        opacity: dim ? 0.35 : 1,
      }}
      strokeWidth={m.hollow ? 2.2 : 2}
    />
  );
}

/** Legend glyph for a method (HTML). */
export function MethodGlyph({ id, size = 14 }: { id: string; size?: number }) {
  const m = METHODS[id];
  if (!m) return null;
  return (
    <ShapeGlyph
      shape={m.shape}
      size={size}
      fill={m.hollow ? 'var(--surface)' : catVar(m.slot)}
      stroke={m.hollow ? catVar(m.slot) : undefined}
      className="shrink-0"
    />
  );
}

/** Legend row of methods: glyph + label (text in ink, never in the series color). */
export function MethodLegend({ ids, className }: { ids: string[]; className?: string }) {
  return (
    <ul
      className={cx('!m-0 flex !list-none flex-wrap items-center gap-x-4 gap-y-1.5 !p-0 pt-1 text-[15px] font-normal text-ink-2', className)}
      aria-label="Legend"
    >
      {ids.map((id) => (
        <li key={id} className="!m-0 inline-flex items-center gap-1.5 !p-0">
          <MethodGlyph id={id} />
          {METHODS[id]?.label ?? id}
        </li>
      ))}
    </ul>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// chrome
// ─────────────────────────────────────────────────────────────────────────────

/** Floating tooltip inside a `relative` container. */
export function Tip({ x, y, width, children, flip }: { x: number; y: number; width: number; children: ReactNode; flip?: boolean }) {
  const w = Math.min(300, width - 8);
  const left = Math.max(4, Math.min(width - w - 4, flip ? x - w - 14 : x + 14));
  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-20 rounded-[9px] border border-border bg-surface px-3 py-2 text-[15px] leading-snug shadow-[var(--shadow-pop)]"
      style={{ left, top: y, width: w }}
    >
      {children}
    </div>
  );
}

/** "Paper" / "This site" provenance badge on a figure title. */
export function SourceBadge({ kind, children }: { kind: 'paper' | 'site'; children?: ReactNode }) {
  const Icon = kind === 'paper' ? BookOpen : FlaskConical;
  return (
    <span
      className={cx(
        'inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-0.5 align-middle text-[14px] font-medium whitespace-nowrap',
        kind === 'paper' ? 'border-border bg-surface-2 text-ink-2' : 'border-accent/40 bg-accent-soft text-accent-text',
      )}
    >
      <Icon aria-hidden="true" size={14} />
      {children ?? (kind === 'paper' ? 'Paper numbers' : 'This site’s miniature')}
    </span>
  );
}

/** Figure title row with a provenance badge. */
export function TitleWithBadge({ title, kind, badge }: { title: ReactNode; kind: 'paper' | 'site'; badge?: ReactNode }) {
  return (
    <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
      <span>{title}</span>
      <SourceBadge kind={kind}>{badge}</SourceBadge>
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Paper table (faithful rendition, the table-view twin of every chart)
// ─────────────────────────────────────────────────────────────────────────────

export interface PaperTableColumn {
  id: string;
  label: ReactNode;
  /** Optional group header ("Zero-shot", "No", …). Adjacent equal groups merge. */
  group?: string;
  /** Method id: shows its glyph in the header. */
  method?: string;
}

export interface PaperTableRow {
  id: string;
  /** First-column cells (e.g. dataset, task). */
  head: ReactNode[];
  cells: Cell[];
  mean?: boolean;
  /** Optional per-cell background (e.g. diverging shading vs the reference column). */
  shade?: (string | undefined)[];
}

export function PaperTableView({
  headCols,
  columns,
  rows,
  caption,
  markMax = false,
  className,
}: {
  headCols: string[];
  columns: PaperTableColumn[];
  rows: PaperTableRow[];
  caption: ReactNode;
  /** Underline the row maximum among numeric cells (the paper bolds the best entry). */
  markMax?: boolean;
  className?: string;
}) {
  const groups: { label: string; span: number }[] = [];
  for (const c of columns) {
    const g = c.group ?? '';
    const last = groups[groups.length - 1];
    if (last && last.label === g) last.span++;
    else groups.push({ label: g, span: 1 });
  }
  const hasGroups = groups.some((g) => g.label);
  return (
    <div className={cx('min-w-0', className)}>
      <p className="!mt-0 !mb-2 text-[15px] leading-snug text-ink-2">{caption}</p>
      <div className="scroll-quiet overflow-x-auto">
        <table className="w-full border-collapse text-[15px]">
          <thead>
            {hasGroups && (
              <tr className="text-ink-2">
                <th colSpan={headCols.length} className="sticky left-0 z-[1] bg-surface" />
                {groups.map((g, i) => (
                  <th key={i} colSpan={g.span} scope="colgroup" className="border-b border-grid px-2 pt-1 pb-1 text-center text-[14px] font-semibold">
                    {g.label}
                  </th>
                ))}
              </tr>
            )}
            <tr className="border-b border-axis text-ink-2">
              {headCols.map((h, i) => (
                <th
                  key={h}
                  scope="col"
                  className={cx('bg-surface px-2 py-1.5 text-left text-[14px] font-semibold whitespace-nowrap', i === 0 && 'sticky left-0 z-[1]')}
                >
                  {h}
                </th>
              ))}
              {columns.map((c) => (
                <th key={c.id} scope="col" className="px-2 py-1.5 text-right align-bottom text-[14px] font-semibold whitespace-nowrap">
                  <span className="inline-flex items-center gap-1.5">
                    {c.method && <MethodGlyph id={c.method} size={12} />}
                    {c.label}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const nums = r.cells.filter(isNum);
              const max = nums.length ? Math.max(...nums) : NaN;
              return (
                <tr key={r.id} className={cx('border-b border-grid/70', r.mean && 'border-t-2 border-t-axis font-semibold')}>
                  {r.head.map((h, i) => (
                    <td
                      key={i}
                      className={cx(
                        'bg-surface px-2 py-1 text-[14.5px] whitespace-nowrap',
                        i === 0 && 'sticky left-0 z-[1]',
                        !r.mean && 'text-ink',
                      )}
                    >
                      {h}
                    </td>
                  ))}
                  {r.cells.map((c, j) => {
                    const best = markMax && isNum(c) && c === max && !r.mean;
                    const bg = r.shade?.[j];
                    return (
                      <td
                        key={j}
                        className={cx(
                          'num px-2 py-1 text-right whitespace-nowrap',
                          best && 'font-bold text-ink underline decoration-2 underline-offset-[3px]',
                          c === null && 'text-muted',
                        )}
                        style={bg ? { background: bg } : undefined}
                      >
                        {cellText(c)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Chart ↔ table frame (like the shared ChartFrame, but the table is shown in full height)
// ─────────────────────────────────────────────────────────────────────────────

export function ChartTableFrame({
  title,
  subtitle,
  legend,
  table,
  children,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  legend?: ReactNode;
  table: ReactNode;
  children: ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);
  return (
    <div className="min-w-0">
      <div className="mb-2 flex items-start justify-between gap-3">
        <div className="min-w-0">
          {title && <div className="text-[17px] font-semibold text-ink">{title}</div>}
          {subtitle && <div className="mt-0.5 text-[15px] text-ink-2">{subtitle}</div>}
        </div>
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
      </div>
      {legend && !showTable && <div className="mb-2">{legend}</div>}
      {showTable ? table : children}
    </div>
  );
}
