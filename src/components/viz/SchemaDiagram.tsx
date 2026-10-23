/**
 * SchemaDiagram: tables as cards listing their columns with PK / FK / time
 * markers and sem-type badges; FK arrows drawn F→P (child row → parent row,
 * spec §1.3), card accent = table color (BUILD_PLAN §5).
 *
 * Auto layout: tables without FKs (pure parents) in the left column, each
 * child one column right of its deepest parent, task tables last. Arrows are
 * measured from the DOM, so the cards can be ordinary responsive HTML.
 */
import { Clock, KeyRound, Link2 } from 'lucide-react';
import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { TableSchema } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { cx } from '../ui/cx';
import { semOfDType } from './format';
import { SemBadge } from './SemBadge';
import { toSet } from './useSize';

export interface SchemaTableSpec {
  schema: TableSchema;
  /** Categorical slot (BUILD_PLAN §3.5). */
  slot: number;
  rowCount?: number;
  /** 'task' renders the "task table" tag and places it in the last column. */
  kind?: 'db' | 'task';
  /** e.g. "Train · Val · Test" for task tables. */
  subtitle?: string;
}

export interface SchemaDiagramProps {
  tables: SchemaTableSpec[];
  highlightTables?: ReadonlySet<string> | readonly string[];
  /** Keys "table.column". */
  highlightColumns?: ReadonlySet<string> | readonly string[];
  selectedTable?: string | null;
  onTableClick?: (table: string) => void;
  onTableHover?: (table: string | null) => void;
  onColumnClick?: (table: string, column: string) => void;
  /** Manual placement: table → layout column index (0 = leftmost). */
  columnsOf?: Record<string, number>;
  /** Only list key and time columns (compact overview). */
  keysOnly?: boolean;
  /** Horizontal gap between layout columns, where the arrows run (px). Default 56. */
  gutter?: number;
  /** Color each arrow by its child table (at rest); hover still uses the accent. Default false. */
  tintArrows?: boolean;
  className?: string;
}

interface Arrow {
  id: string;
  from: string; // child table
  to: string; // parent table
  d: string;
  tip: { x: number; y: number; angle: number };
}

export function SchemaDiagram({
  tables,
  highlightTables,
  highlightColumns,
  selectedTable = null,
  onTableClick,
  onTableHover,
  onColumnClick,
  columnsOf,
  keysOnly = false,
  gutter = 56,
  tintArrows = false,
  className,
}: SchemaDiagramProps) {
  const hlT = toSet(highlightTables);
  const hlC = toSet(highlightColumns);
  const [hoverT, setHoverT] = useState<string | null>(null);

  // ── layered auto-layout ────────────────────────────────────────────────
  const levels = useMemo(() => {
    const byName = new Map(tables.map((t) => [t.schema.name, t]));
    const memo = new Map<string, number>();
    const level = (name: string, stack: Set<string>): number => {
      if (columnsOf && name in columnsOf) return columnsOf[name];
      if (memo.has(name)) return memo.get(name)!;
      const t = byName.get(name);
      if (!t || stack.has(name)) return 0;
      stack.add(name);
      const parents = Object.values(t.schema.fkeys).filter((p) => byName.has(p) && p !== name);
      const l = parents.length === 0 ? 0 : 1 + Math.max(...parents.map((p) => level(p, stack)));
      stack.delete(name);
      memo.set(name, l);
      return l;
    };
    const dbLevels = tables.filter((t) => t.kind !== 'task').map((t) => level(t.schema.name, new Set()));
    const maxDb = dbLevels.length ? Math.max(...dbLevels) : 0;
    const out = new Map<string, number>();
    for (const t of tables) {
      const n = t.schema.name;
      out.set(n, columnsOf && n in columnsOf ? columnsOf[n] : t.kind === 'task' ? maxDb + 1 : level(n, new Set()));
    }
    return out;
  }, [tables, columnsOf]);

  const nCols = Math.max(1, ...[...levels.values()].map((l) => l + 1));
  const grouped = useMemo(() => {
    const g: SchemaTableSpec[][] = Array.from({ length: nCols }, () => []);
    for (const t of tables) g[levels.get(t.schema.name) ?? 0].push(t);
    return g;
  }, [tables, levels, nCols]);

  // ── arrows measured from the DOM ───────────────────────────────────────
  const wrap = useRef<HTMLDivElement | null>(null);
  const anchors = useRef(new Map<string, HTMLElement>());
  const [arrows, setArrows] = useState<Arrow[]>([]);
  const [box, setBox] = useState({ w: 0, h: 0 });

  const register = useCallback(
    (key: string) => (el: HTMLElement | null) => {
      if (el) anchors.current.set(key, el);
      else anchors.current.delete(key);
    },
    [],
  );

  const measure = useCallback(() => {
    const root = wrap.current;
    if (!root) return;
    const R = root.getBoundingClientRect();
    const rel = (el: HTMLElement) => {
      const r = el.getBoundingClientRect();
      return { l: r.left - R.left + root.scrollLeft, r: r.right - R.left + root.scrollLeft, t: r.top - R.top, b: r.bottom - R.top };
    };
    const out: Arrow[] = [];
    for (const t of tables) {
      for (const [fkCol, parent] of Object.entries(t.schema.fkeys)) {
        const fromEl = anchors.current.get(`${t.schema.name}.${fkCol}`);
        const parentSpec = tables.find((x) => x.schema.name === parent);
        if (!fromEl || !parentSpec) continue;
        const pk = parentSpec.schema.pkey;
        const toEl = anchors.current.get(`${parent}.${pk ?? ''}`) ?? anchors.current.get(`${parent}.__header`);
        if (!toEl) continue;
        const a = rel(fromEl);
        const b = rel(toEl);
        const ay = (a.t + a.b) / 2;
        const by = (b.t + b.b) / 2;
        const leftward = (b.r + b.l) / 2 < (a.l + a.r) / 2;
        // child's FK row edge → parent's PK row edge
        const x1 = leftward ? a.l - 2 : a.r + 2;
        const x2 = leftward ? b.r + 6 : b.l - 6;
        const dx = Math.max(28, Math.abs(x2 - x1) * 0.45);
        const c1 = leftward ? x1 - dx : x1 + dx;
        const c2 = leftward ? x2 + dx : x2 - dx;
        out.push({
          id: `${t.schema.name}.${fkCol}`,
          from: t.schema.name,
          to: parent,
          d: `M${x1},${ay} C${c1},${ay} ${c2},${by} ${x2},${by}`,
          tip: { x: x2, y: by, angle: leftward ? 180 : 0 },
        });
      }
    }
    setArrows(out);
    setBox({ w: root.scrollWidth, h: root.scrollHeight });
  }, [tables]);

  useLayoutEffect(() => {
    measure();
    const root = wrap.current;
    if (!root || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => measure());
    ro.observe(root);
    for (const el of root.querySelectorAll('[data-schema-card]')) ro.observe(el);
    return () => ro.disconnect();
  }, [measure, keysOnly]);

  const active = hoverT ?? selectedTable;

  return (
    <div className={cx('scroll-quiet relative w-full overflow-x-auto', className)}>
      <div
        ref={wrap}
        className="relative grid items-start gap-y-5 p-1"
        style={{ columnGap: gutter, gridTemplateColumns: `repeat(${nCols}, minmax(244px, 1fr))`, minWidth: nCols * 244 + (nCols - 1) * gutter + 8 }}
      >
        <svg
          aria-hidden="true"
          className="pointer-events-none absolute top-0 left-0 overflow-visible"
          width={box.w}
          height={box.h}
        >
          {arrows.map((a) => {
            const on = !!active && (a.from === active || a.to === active);
            const slot = tables.find((t) => t.schema.name === a.from)?.slot ?? 0;
            const rest = tintArrows ? catVar(slot) : 'var(--axis)';
            return (
              <g key={a.id} style={{ opacity: active && !on ? 0.25 : 1 }}>
                <path d={a.d} fill="none" style={{ stroke: on ? 'var(--accent)' : rest }} strokeWidth={on ? 2 : 1.5} />
                <path
                  d="M0,0 L-7,-4 L-7,4 Z"
                  transform={`translate(${a.tip.x},${a.tip.y}) rotate(${a.tip.angle})`}
                  style={{ fill: on ? 'var(--accent)' : rest }}
                />
              </g>
            );
          })}
        </svg>
        {grouped.map((col, ci) => (
          <div key={ci} className="relative z-[1] flex flex-col gap-5">
            {col.map((t) => {
              const name = t.schema.name;
              const isHl = hlT.has(name) || active === name;
              const isLinked =
                !!active &&
                active !== name &&
                arrows.some((a) => (a.from === active && a.to === name) || (a.to === active && a.from === name));
              const dim = !!active && !isHl && !isLinked && hlT.size === 0;
              const cols = t.schema.columns.filter(
                (c) => !keysOnly || c.name === t.schema.pkey || c.name in t.schema.fkeys || c.name === t.schema.timeCol,
              );
              return (
                <div
                  key={name}
                  data-schema-card
                  className={cx(
                    'card relative overflow-hidden transition-[opacity,box-shadow] duration-200',
                    isHl && 'ring-2 ring-accent',
                    dim && 'opacity-45',
                    onTableClick && 'cursor-pointer',
                  )}
                  onPointerEnter={() => {
                    setHoverT(name);
                    onTableHover?.(name);
                  }}
                  onPointerLeave={() => {
                    setHoverT(null);
                    onTableHover?.(null);
                  }}
                  onClick={() => onTableClick?.(name)}
                >
                  <div aria-hidden="true" className="absolute inset-x-0 top-0 h-[3px]" style={{ background: catVar(t.slot) }} />
                  <div ref={register(`${name}.__header`)} className="flex items-baseline justify-between gap-2 px-3 pt-3 pb-2">
                    {onTableClick ? (
                      <button type="button" className="truncate text-left text-[16.5px] font-semibold text-ink" onClick={() => onTableClick(name)}>
                        {name}
                      </button>
                    ) : (
                      <span className="truncate text-[16.5px] font-semibold text-ink">{name}</span>
                    )}
                    <span className="num shrink-0 text-[14.5px] text-muted">
                      {t.kind === 'task' ? 'task table' : t.rowCount !== undefined ? `${t.rowCount.toLocaleString('en-US')} rows` : ''}
                    </span>
                  </div>
                  {t.subtitle && <div className="-mt-1 px-3 pb-2 text-[14.5px] text-muted">{t.subtitle}</div>}
                  <ul className="border-t border-grid py-1">
                    {cols.map((c) => {
                      const isPk = c.name === t.schema.pkey;
                      const fk = t.schema.fkeys[c.name];
                      const isTime = c.name === t.schema.timeCol;
                      const key = `${name}.${c.name}`;
                      const colHl = hlC.has(key);
                      const Row = onColumnClick ? 'button' : 'div';
                      return (
                        <li key={c.name}>
                          <Row
                            ref={register(key) as never}
                            type={onColumnClick ? 'button' : undefined}
                            onClick={
                              onColumnClick
                                ? (e: React.MouseEvent) => {
                                    e.stopPropagation();
                                    onColumnClick(name, c.name);
                                  }
                                : undefined
                            }
                            className={cx(
                              'flex w-full items-center gap-2 px-3 py-[6px] text-left text-[15.5px]',
                              colHl ? 'bg-accent-soft' : onColumnClick && 'hover:bg-surface-2',
                            )}
                            title={
                              isPk
                                ? 'Primary key: no token, no edge (spec §1.2)'
                                : fk
                                  ? `Foreign key → ${fk}: no token; an F→P link (spec §1.2)`
                                  : c.description
                            }
                          >
                            <span className="flex w-[20px] shrink-0 justify-center text-ink-2">
                              {isPk ? (
                                <KeyRound aria-label="primary key" size={15} />
                              ) : fk ? (
                                <Link2 aria-label="foreign key" size={15} />
                              ) : (
                                <SemBadge sem={semOfDType(c.dtype)} size="xs" />
                              )}
                            </span>
                            <span className={cx('min-w-0 flex-1 truncate', isPk || fk ? 'text-ink-2' : 'text-ink')}>{c.name}</span>
                            {fk && <span className="max-w-[48%] shrink-0 truncate text-[14px] text-muted">→ {fk}</span>}
                            {isTime && (
                              <span className="inline-flex shrink-0 items-center gap-1 rounded bg-surface-3 px-1.5 text-[13.5px] font-medium text-ink-2">
                                <Clock aria-hidden="true" size={12} /> time
                              </span>
                            )}
                          </Row>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              );
            })}
          </div>
        ))}

      </div>
      <p className="mt-2 px-1 text-[15px] text-muted">
        Arrows point F→P: from a foreign-key column to the parent table’s primary key.
      </p>
    </div>
  );
}
