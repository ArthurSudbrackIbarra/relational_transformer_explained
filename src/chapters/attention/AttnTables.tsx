/**
 * AttnTables: the live context drawn as small per-table grids (one square per cell token; rows = rows
 * in the context, columns = that table's columns), in the spirit of paper Fig. 1 (right). Each square
 * is filled by the attention weight one query gives that key, so you can read "where the query looks"
 * as a heat over the database. Keys the mask blocks are drawn hollow. Display only: the weights are
 * the engine's capture.
 */
import { useMemo, type KeyboardEvent } from 'react';
import { TableSwatch, sequentialCss } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import type { Token } from '@/engine/types';
import { useResolvedTheme } from '@/styles/theme';

interface Row {
  node: number;
  byColumn: number[];
}
interface Table {
  name: string;
  slot: number;
  columns: string[];
  rows: Row[];
}

export function groupTables(tokens: readonly Token[], slots: Record<string, number>): Table[] {
  const tables: Table[] = [];
  const byName = new Map<string, Table>();
  const rowsByNode = new Map<number, { table: Table; positions: number[] }>();
  for (const t of tokens) {
    if (t.isPadding) continue;
    let tbl = byName.get(t.tableName);
    if (!tbl) {
      tbl = { name: t.tableName, slot: slots[t.tableName] ?? 0, columns: [], rows: [] };
      byName.set(t.tableName, tbl);
      tables.push(tbl);
    }
    if (!tbl.columns.includes(t.colName)) tbl.columns.push(t.colName);
    let r = rowsByNode.get(t.nodeIdx);
    if (!r) {
      r = { table: tbl, positions: [] };
      rowsByNode.set(t.nodeIdx, r);
      tbl.rows.push({ node: t.nodeIdx, byColumn: [] });
    }
    r.positions.push(t.pos);
  }
  for (const tbl of tables) {
    for (const row of tbl.rows) {
      const ps = rowsByNode.get(row.node)!.positions;
      row.byColumn = tbl.columns.map((c) => ps.find((p) => tokens[p].colName === c) ?? -1);
    }
  }
  return tables;
}

export interface AttnTablesProps {
  tokens: readonly Token[];
  slots: Record<string, number>;
  /** weights[k] for every position (0 for blocked keys). */
  weights: ArrayLike<number>;
  /** allowed[k] = 1 if the mask lets the query read key k. */
  allowed: ArrayLike<number>;
  query: number;
  hoverKey?: number | null;
  onHover?: (pos: number | null) => void;
  onSelect?: (pos: number) => void;
  /** Fill ∝ √(w / max w) instead of w / max w. */
  sqrt?: boolean;
  label: string;
  maxRowsPerStack?: number;
}

const GAP = 2;
const STACK_GAP = 6;
const TITLE_H = 22;
const BLOCK_GAP_X = 18;
const BLOCK_GAP_Y = 14;

export function AttnTables({ tokens, slots, weights, allowed, query, hoverKey = null, onHover, onSelect, sqrt = true, label, maxRowsPerStack = 14 }: AttnTablesProps) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const mode = useResolvedTheme();
  const width = Math.max(220, size.width || 600);
  const cell = width < 480 ? 10 : 13;
  const step = cell + GAP;
  const tables = useMemo(() => groupTables(tokens, slots), [tokens, slots]);
  const maxW = useMemo(() => {
    let m = 0;
    for (let k = 0; k < tokens.length; k++) if (allowed[k]) m = Math.max(m, weights[k]);
    return m;
  }, [tokens, weights, allowed]);

  const layout = useMemo(() => {
    const blocks: { name: string; slot: number; x: number; y: number; w: number; cells: { pos: number; x: number; y: number }[]; holes: { x: number; y: number }[] }[] = [];
    let x = 0;
    let y = 0;
    let lineH = 0;
    for (const tbl of tables) {
      const cols = tbl.columns.length;
      const nRows = tbl.rows.length;
      const stacks = Math.max(1, Math.ceil(nRows / maxRowsPerStack));
      const perStack = Math.ceil(nRows / stacks);
      const stackW = cols * step - GAP;
      const gridW = stacks * stackW + (stacks - 1) * STACK_GAP;
      const gridH = perStack * step - GAP;
      const w = Math.max(gridW, tbl.name.length * 8.2 + 22);
      if (x > 0 && x + w > width) {
        x = 0;
        y += lineH + BLOCK_GAP_Y;
        lineH = 0;
      }
      const cells: { pos: number; x: number; y: number }[] = [];
      const holes: { x: number; y: number }[] = [];
      tbl.rows.forEach((row, ri) => {
        const s = Math.floor(ri / perStack);
        const r = ri % perStack;
        row.byColumn.forEach((pos, c) => {
          const cx = x + s * (stackW + STACK_GAP) + c * step;
          const cy = y + TITLE_H + r * step;
          if (pos < 0) holes.push({ x: cx, y: cy });
          else cells.push({ pos, x: cx, y: cy });
        });
      });
      blocks.push({ name: tbl.name, slot: tbl.slot, x, y, w, cells, holes });
      x += w + BLOCK_GAP_X;
      lineH = Math.max(lineH, TITLE_H + gridH);
    }
    return { blocks, height: y + lineH };
  }, [tables, width, step, maxRowsPerStack]);

  // Keyboard reading: the arrow keys walk the cells in drawing order (table by table, row by row).
  const cellOrder = useMemo(() => layout.blocks.flatMap((b) => b.cells.map((c) => c.pos)), [layout]);
  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    if (!onHover || !cellOrder.length) return;
    const fwd = e.key === 'ArrowRight' || e.key === 'ArrowDown';
    const back = e.key === 'ArrowLeft' || e.key === 'ArrowUp';
    if (!fwd && !back && e.key !== 'Home' && e.key !== 'End' && e.key !== 'Escape') return;
    e.preventDefault();
    if (e.key === 'Escape') return onHover(null);
    const i = hoverKey !== null ? cellOrder.indexOf(hoverKey) : -1;
    const j = e.key === 'Home' ? 0 : e.key === 'End' ? cellOrder.length - 1 : i < 0 ? 0 : Math.max(0, Math.min(cellOrder.length - 1, i + (fwd ? 1 : -1)));
    onHover(cellOrder[j]);
  };

  const fillOf = (pos: number) => {
    const w = weights[pos];
    if (!allowed[pos] || !(w > 0) || maxW <= 0) return null;
    const t = sqrt ? Math.sqrt(w / maxW) : w / maxW;
    return sequentialCss(0.08 + 0.92 * t, mode);
  };

  return (
    <div ref={ref} className="min-w-0">
      <svg
        width="100%"
        height={layout.height + 4}
        viewBox={`-2 -2 ${width + 4} ${layout.height + 4}`}
        role="img"
        aria-label={label}
        className="block overflow-visible rounded-[4px] select-none focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-accent"
        tabIndex={onHover ? 0 : undefined}
        onKeyDown={onKey}
        onPointerLeave={() => onHover?.(null)}
      >
        {layout.blocks.map((b) => (
          <g key={b.name}>
            <foreignObject x={b.x} y={b.y - 1} width={b.w + 4} height={TITLE_H}>
              <div className="flex h-full items-start text-[14px] leading-none">
                <TableSwatch slot={b.slot} label={b.name} size={8} />
              </div>
            </foreignObject>
            {b.holes.map((h, i) => (
              <circle key={`h${i}`} cx={h.x + cell / 2} cy={h.y + cell / 2} r={1.1} style={{ fill: 'var(--axis)' }} />
            ))}
            {b.cells.map((g) => {
              const fill = fillOf(g.pos);
              const isQ = g.pos === query;
              const isHover = g.pos === hoverKey;
              const blocked = !allowed[g.pos];
              return (
                <g
                  key={g.pos}
                  onPointerEnter={() => onHover?.(g.pos)}
                  onClick={onSelect ? () => onSelect(g.pos) : undefined}
                  className={onSelect ? 'cursor-pointer' : undefined}
                >
                  <rect
                    x={g.x}
                    y={g.y}
                    width={cell}
                    height={cell}
                    rx={2}
                    style={{
                      fill: fill ?? (blocked ? 'transparent' : 'var(--surface-2)'),
                      stroke: blocked ? 'var(--grid)' : fill ? 'none' : 'var(--axis)',
                    }}
                    strokeWidth={blocked ? 1 : 0.8}
                  />
                  {tokens[g.pos].isTarget && (
                    <text x={g.x + cell / 2} y={g.y + cell / 2 + 3.5} textAnchor="middle" className="text-[13px] font-bold" style={{ fill: 'var(--accent-text)' }}>
                      ?
                    </text>
                  )}
                  {isQ && (
                    <rect x={g.x - 2} y={g.y - 2} width={cell + 4} height={cell + 4} rx={3} fill="none" strokeDasharray="3 2" style={{ stroke: 'var(--accent)' }} strokeWidth={1.6} />
                  )}
                  {isHover && !isQ && <rect x={g.x - 1.5} y={g.y - 1.5} width={cell + 3} height={cell + 3} rx={3} fill="none" style={{ stroke: 'var(--ink)' }} strokeWidth={1.5} />}
                </g>
              );
            })}
          </g>
        ))}
      </svg>
    </div>
  );
}
