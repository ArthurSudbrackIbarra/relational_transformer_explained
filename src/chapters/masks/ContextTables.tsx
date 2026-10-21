/**
 * ContextTables: the context window drawn as small per-table grids (one square per cell token,
 * rows = rows in the context, columns = that table's columns), in the spirit of paper Fig. 1 (right).
 * Given one row of an attention mask (the query's allowed keys) it fills the allowed cells, outlines
 * the query, and can draw animated "flow" lines from every allowed key into the query.
 *
 * Purely presentational: the allowed set always comes from the engine's masks, and the reason for a
 * missing cell (null, dropped leakage column, cut at L) comes from the sampler's event log.
 */
import { motion, useReducedMotion } from 'motion/react';
import { memo, useId, useMemo, useState, type ReactNode } from 'react';
import { Cite, sequentialCss, textWidth } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import type { Token } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { useResolvedTheme } from '@/styles/theme';
import type { CtxGrouping, HoleKind } from './maskUtils';

export interface ContextTablesProps {
  tokens: readonly Token[];
  grouping: CtxGrouping;
  /** allowed[k] = 1 if the query may attend to key k (one row of a mask), or null for no highlight. */
  allowed: ArrayLike<number> | null;
  query: number | null;
  /** A key to outline (e.g. the key hovered in another view). */
  hoverKey?: number | null;
  /** Cell size in px. Default 12. */
  cell?: number;
  /** Wrap a table's rows into side-by-side stacks of at most this many rows. Default 18. */
  maxRowsPerStack?: number;
  onHover?: (pos: number | null) => void;
  onSelect?: (pos: number) => void;
  /** Tooltip content for a hovered cell. */
  describe?: (pos: number) => ReactNode;
  /** Draw animated lines from each allowed key into the query. */
  flow?: boolean;
  /** Change to replay the flow animation. */
  replay?: number;
  /** Accessible description. */
  label: string;
  /** Horizontal gap between tables. Default 16. */
  gapX?: number;
  /** Show the "+N padding" note. Default true. */
  showPadding?: boolean;
  /** Background behind the figure (used as a halo so table titles stay legible over flow lines). */
  halo?: string;
  className?: string;
}

const GAP = 2;
const STACK_GAP = 6;
const TITLE_PX = 14.5;
const TITLE_H = 22;
const SWATCH = 9;
const BLOCK_GAP_Y = 14;
const FLOW_PAD = 44;

interface CellGeom {
  pos: number;
  x: number;
  y: number;
  /** Top of the wrapped line of blocks this cell belongs to (title band starts here). */
  lineY: number;
  block: number;
  /** Right edge of this cell's table grid. */
  gridRight: number;
}

export const HOLE_TEXT: Record<HoleKind, string> = {
  null: 'no token: the value is null, so no cell was stored',
  dropped: 'no token: leakage column dropped for this row (fly.rs:457-461)',
  cut: 'no token: the sampler stopped at exactly L tokens inside this row (fly.rs:502-509)',
};

export function ContextTables({
  tokens,
  grouping,
  allowed,
  query,
  hoverKey = null,
  cell = 12,
  maxRowsPerStack = 18,
  onHover,
  onSelect,
  describe,
  flow = false,
  replay = 0,
  label,
  showPadding = true,
  gapX = 16,
  halo = 'var(--surface-2)',
  className,
}: ContextTablesProps) {
  const uid = useId().replace(/:/g, '');
  const [ref, size] = useElementSize<HTMLDivElement>();
  const mode = useResolvedTheme();
  const reduce = useReducedMotion();
  const fill = sequentialCss(mode === 'dark' ? 0.62 : 0.66, mode);
  // the viewBox is 4 px wider than the layout (2 px margin each side), so lay out in (measured width − 4):
  // the SVG then draws at scale 1 and every text keeps its nominal size
  const width = Math.max(200, (size.width || 524) - 4);
  const step = cell + GAP;
  const [tip, setTip] = useState<number | null>(null);

  const layout = useMemo(() => {
    const blocks: {
      name: string;
      slot: number;
      x: number;
      y: number;
      w: number;
      cells: CellGeom[];
      holes: { x: number; y: number; kind: HoleKind }[];
    }[] = [];
    let x = 0;
    let y = 0;
    let lineH = 0;
    const cellAt = new Map<number, CellGeom>();
    grouping.tables.forEach((tbl, bi) => {
      const cols = tbl.columns.length;
      const nRows = tbl.rows.length;
      const stacks = Math.max(1, Math.ceil(nRows / maxRowsPerStack));
      const perStack = Math.ceil(nRows / stacks);
      const stackW = cols * step - GAP;
      const gridW = stacks * stackW + (stacks - 1) * STACK_GAP;
      const gridH = perStack * step - GAP;
      // measured, never truncated: the block is as wide as its title when the title is wider
      const titleW = SWATCH + 6 + textWidth(tbl.name, TITLE_PX, { mono: true }) + 2;
      const w = Math.max(gridW, titleW);
      const h = TITLE_H + gridH;
      if (x > 0 && x + w > width) {
        x = 0;
        y += lineH + BLOCK_GAP_Y;
        lineH = 0;
      }
      const cells: CellGeom[] = [];
      const holes: { x: number; y: number; kind: HoleKind }[] = [];
      tbl.rows.forEach((row, ri) => {
        const s = Math.floor(ri / perStack);
        const r = ri % perStack;
        row.byColumn.forEach((pos, c) => {
          const cx = x + s * (stackW + STACK_GAP) + c * step;
          const cy = y + TITLE_H + r * step;
          if (pos < 0) holes.push({ x: cx, y: cy, kind: row.holes[c] ?? 'null' });
          else {
            const g = { pos, x: cx, y: cy, lineY: y, block: bi, gridRight: x + gridW };
            cells.push(g);
            cellAt.set(pos, g);
          }
        });
      });
      blocks.push({ name: tbl.name, slot: tbl.slot, x, y, w, cells, holes });
      x += w + gapX;
      lineH = Math.max(lineH, h);
    });
    return { blocks, height: y + lineH, cellAt };
  }, [grouping, width, step, maxRowsPerStack, gapX]);

  // allowed keys in emission order, for a staggered reveal
  const allowedList = useMemo(() => {
    if (!allowed) return [] as number[];
    const out: number[] = [];
    for (let k = 0; k < tokens.length; k++) if (allowed[k] && !tokens[k].isPadding) out.push(k);
    return out;
  }, [allowed, tokens]);
  const rank = useMemo(() => new Map(allowedList.map((k, i) => [k, i])), [allowedList]);

  const qGeom = query !== null ? layout.cellAt.get(query) : undefined;
  const flows = flow && qGeom ? allowedList.filter((k) => k !== query && layout.cellAt.has(k)) : [];
  const half = cell / 2;
  // room above the tables for the flow arcs
  const top = flow ? FLOW_PAD : 0;
  const H = Math.max(layout.height, cell + TITLE_H) + top;
  const stagger = (i: number) => (reduce ? 0 : Math.min(0.3, i * 0.004));
  const targetGeom = [...layout.cellAt.values()].find((g) => tokens[g.pos]?.isTarget);
  const glyphPx = Math.max(13, cell - 4);

  const scale = size.width ? size.width / (width + 4) : 1;
  const tipGeom = tip !== null && describe ? layout.cellAt.get(tip) : undefined;
  const tipLeft = tipGeom ? (tipGeom.x + 2 + cell + 8) * scale : 0;
  const tipFlip = tipGeom ? tipLeft + 260 > (size.width || width) : false;

  const setHover = (p: number | null) => {
    setTip(p);
    onHover?.(p);
  };

  return (
    <div ref={ref} className={`relative ${className ?? ''}`}>
      <svg
        width="100%"
        height={H + 4}
        viewBox={`-2 ${-2 - top} ${width + 4} ${H + 4}`}
        role="img"
        aria-label={label}
        className="block overflow-visible select-none"
        onPointerLeave={() => setHover(null)}
        onPointerOver={(e) => {
          const p = posOf(e.target);
          if (p !== tip) setHover(p);
        }}
        onClick={
          onSelect
            ? (e) => {
                const p = posOf(e.target);
                if (p !== null) onSelect(p);
              }
            : undefined
        }
      >
        <defs>
          <marker id={`${uid}-arr`} viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0.8 L7.2,4 L0,7.2 Z" style={{ fill: 'var(--accent)' }} />
          </marker>
        </defs>
        {layout.blocks.map((b) => (
          <g key={b.name}>
            {b.holes.map((h, i) => (
              <Hole key={`h${i}`} x={h.x} y={h.y} cell={cell} kind={h.kind} />
            ))}
            {b.cells.map((g) => (
              <Cell
                key={g.pos}
                pos={g.pos}
                x={g.x}
                y={g.y}
                cell={cell}
                fill={fill}
                allowed={!!allowed?.[g.pos]}
                delay={allowed?.[g.pos] ? stagger(rank.get(g.pos) ?? 0) : 0}
                isTarget={tokens[g.pos].isTarget}
                hovered={(hoverKey === g.pos || tip === g.pos) && g.pos !== query}
                clickable={!!onSelect}
              />
            ))}
          </g>
        ))}

        {/* flow lines: key → query */}
        {qGeom &&
          flows.map((k, i) => {
            const a = layout.cellAt.get(k)!;
            return (
              <motion.path
                key={`${replay}-${k}`}
                d={flowPath(a, qGeom, cell, step)}
                fill="none"
                style={{ stroke: 'var(--accent)' }}
                strokeWidth={1.4}
                strokeOpacity={0.72}
                strokeLinecap="round"
                markerEnd={`url(#${uid}-arr)`}
                initial={reduce ? false : { pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ duration: reduce ? 0 : 0.7, delay: reduce ? 0 : 0.15 + i * 0.06, ease: 'easeOut' }}
              />
            );
          })}

        {/* table titles above the flow lines, with a halo in the figure's background */}
        {layout.blocks.map((b) => (
          <g key={`t-${b.name}`} pointerEvents="none">
            <rect x={b.x} y={b.y + 3} width={SWATCH} height={SWATCH} rx={2.5} style={{ fill: catVar(b.slot) }} />
            <text
              x={b.x + SWATCH + 6}
              y={b.y + 3 + SWATCH / 2}
              dominantBaseline="central"
              className="font-mono"
              style={{
                fill: 'var(--ink)',
                fontSize: TITLE_PX,
                paintOrder: 'stroke',
                stroke: flow ? halo : 'none',
                strokeWidth: flow ? 5 : 0,
                strokeLinejoin: 'round',
              }}
            >
              {b.name}
            </text>
          </g>
        ))}

        {/* the target's "?" and the query outline go on top of everything */}
        {targetGeom && cell >= 12 && (
          <text
            x={targetGeom.x + half}
            y={targetGeom.y + half + 0.5}
            textAnchor="middle"
            dominantBaseline="central"
            pointerEvents="none"
            style={{ fill: allowed?.[targetGeom.pos] ? '#ffffff' : 'var(--accent-text)', fontSize: glyphPx, fontWeight: 700 }}
          >
            ?
          </text>
        )}
        {qGeom && (
          <rect
            x={qGeom.x - 2.5}
            y={qGeom.y - 2.5}
            width={cell + 5}
            height={cell + 5}
            rx={3.5}
            fill="none"
            style={{ stroke: 'var(--accent)' }}
            strokeWidth={2.4}
            pointerEvents="none"
          />
        )}
      </svg>
      {tipGeom && (
        <div
          role="tooltip"
          className="pointer-events-none absolute z-20 w-max max-w-[260px] rounded-lg border border-border bg-surface px-3 py-2 text-[15px] leading-snug text-ink shadow-lg"
          style={{
            top: (tipGeom.y + 2 + top) * scale - 6,
            ...(tipFlip ? { right: (size.width || width) - (tipGeom.x + 2 - 8) * scale } : { left: tipLeft }),
          }}
        >
          {describe!(tipGeom.pos)}
        </div>
      )}
      {showPadding && grouping.padding > 0 && (
        <div className="mt-1.5 flex items-center gap-1.5 text-[14px] text-muted">
          <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-[2px] border border-dashed border-axis" />
          +{grouping.padding} padding tokens (they attend to nothing and are never attended to)
        </div>
      )}
    </div>
  );
}

/**
 * Path of one flow line from key `a` into query `q`:
 *  - same column of the same table: a bracket on the right of the grid, so stacked links nest;
 *  - other cells of the same table, below or above: a bracket on the right of that table's grid;
 *  - same wrapped line of tables: an arc that peaks above the title band;
 *  - otherwise (tables on different lines): a plain bent curve.
 */
function flowPath(a: CellGeom, q: CellGeom, cell: number, step: number): string {
  const half = cell / 2;
  const x1 = a.x + half;
  const y1 = a.y + half;
  const x2 = q.x + half;
  const y2 = q.y + half;
  const dx = x2 - x1;
  const dy = y2 - y1;
  if (a.block === q.block && Math.abs(dy) >= step * 0.5 && Math.abs(dx) <= step * 1.5) {
    const right = Math.max(a.gridRight, q.gridRight);
    const off = Math.min(80, 9 + Math.abs(dy) * 0.2);
    const sx = a.x + cell;
    const ex = q.x + cell + 3.5;
    return `M${sx},${y1} C${right + off},${y1} ${right + off},${y2} ${ex},${y2}`;
  }
  const sameRowBand = a.block === q.block && Math.abs(dy) < step * 0.5;
  if (sameRowBand && Math.abs(dx) <= step * 3.5) {
    const bend = 12 + Math.abs(dx) * 0.2;
    const cx = (x1 + x2) / 2;
    const cy = Math.min(y1, y2) - bend;
    return endShort(x1, y1, cx, cy, x2, y2, half);
  }
  if (a.lineY === q.lineY) {
    // peak of the quadratic (t = ½) = ¼·y1 + ½·cy + ¼·y2; put it above the title band
    const peak = a.lineY - 6 - Math.min(FLOW_PAD - 12, Math.abs(dx) * 0.06);
    const cy = 2 * peak - (y1 + y2) / 2;
    return endShort(x1, y1, (x1 + x2) / 2, cy, x2, y2, half);
  }
  const len = Math.hypot(dx, dy) || 1;
  let nx = -dy / len;
  let ny = dx / len;
  if (Math.abs(dx) >= Math.abs(dy) ? ny > 0 : nx < 0) {
    nx = -nx;
    ny = -ny;
  }
  const bend = Math.min(44, 16 + len * 0.2);
  return endShort(x1, y1, (x1 + x2) / 2 + nx * bend, (y1 + y2) / 2 + ny * bend, x2, y2, half);
}

/** Quadratic from (x1,y1) via (cx,cy), stopping just outside the query square along the end tangent. */
function endShort(x1: number, y1: number, cx: number, cy: number, x2: number, y2: number, half: number): string {
  const tx = x2 - cx;
  const ty = y2 - cy;
  const tl = Math.hypot(tx, ty) || 1;
  // distance from the square's centre to its edge along the tangent, plus a small gap
  const edge = half / Math.max(Math.abs(tx / tl), Math.abs(ty / tl)) + 3.5;
  return `M${x1},${y1} Q${cx},${cy} ${x2 - (tx / tl) * edge},${y2 - (ty / tl) * edge}`;
}

/** Legend for ContextTables: key fill, blocked, query, target, and the kinds of missing cell present. */
export function ContextTablesLegend({ className, holes = ['null'] }: { className?: string; holes?: readonly HoleKind[] }) {
  const mode = useResolvedTheme();
  const fill = sequentialCss(mode === 'dark' ? 0.62 : 0.66, mode);
  return (
    <div className={`flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[14.5px] text-ink-2 ${className ?? ''}`}>
      <span className="inline-flex items-center gap-1.5">
        <span aria-hidden="true" className="inline-block h-3 w-3 rounded-[3px]" style={{ background: fill }} />
        key the query may attend to
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span aria-hidden="true" className="inline-block h-3 w-3 rounded-[3px] border border-border-strong bg-surface-2" />
        blocked
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span aria-hidden="true" className="inline-block h-3 w-3 rounded-[3px]" style={{ outline: '2px solid var(--accent)', outlineOffset: 1 }} />
        query
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span aria-hidden="true" className="target-outline inline-flex h-[18px] w-[18px] items-center justify-center rounded-[3px] text-[13px] font-bold text-accent-text">
          ?
        </span>
        masked target
      </span>
      {holes.includes('null') && (
        <span className="inline-flex items-center gap-1.5" title={HOLE_TEXT.null}>
          <HoleSwatch kind="null" />
          null value (no cell stored) <Cite kind="spec" ref="§3.5" />
        </span>
      )}
      {holes.includes('dropped') && (
        <span className="inline-flex items-center gap-1.5" title={HOLE_TEXT.dropped}>
          <HoleSwatch kind="dropped" />
          leakage column dropped <Cite kind="code" ref="rustler/src/fly.rs:457-461" />
        </span>
      )}
      {holes.includes('cut') && (
        <span className="inline-flex items-center gap-1.5" title={HOLE_TEXT.cut}>
          <HoleSwatch kind="cut" />
          cut by the stop at L <Cite kind="code" ref="rustler/src/fly.rs:502-509" />
        </span>
      )}
    </div>
  );
}

function HoleSwatch({ kind }: { kind: HoleKind }) {
  return (
    <svg aria-hidden="true" width={14} height={14} viewBox="0 0 14 14" className="shrink-0">
      <Hole x={1} y={1} cell={12} kind={kind} />
    </svg>
  );
}

/** A missing cell: a dot (null), a slash (dropped leakage column) or a dashed square (cut at L). */
function Hole({ x, y, cell, kind }: { x: number; y: number; cell: number; kind: HoleKind }) {
  const half = cell / 2;
  return (
    <g>
      <title>{HOLE_TEXT[kind]}</title>
      {kind === 'null' && <circle cx={x + half} cy={y + half} r={1.4} style={{ fill: 'var(--muted)' }} />}
      {kind === 'dropped' && (
        <line
          x1={x + cell * 0.2}
          y1={y + cell * 0.8}
          x2={x + cell * 0.8}
          y2={y + cell * 0.2}
          style={{ stroke: 'var(--muted)' }}
          strokeWidth={1.5}
          strokeLinecap="round"
        />
      )}
      {kind === 'cut' && (
        <rect
          x={x + 0.75}
          y={y + 0.75}
          width={cell - 1.5}
          height={cell - 1.5}
          rx={2}
          fill="none"
          style={{ stroke: 'var(--muted)' }}
          strokeWidth={1.2}
          strokeDasharray="2 2"
        />
      )}
    </g>
  );
}

/** Token position of the cell under an event target (event delegation), or null. */
function posOf(target: EventTarget | null): number | null {
  const el = (target as Element | null)?.closest?.('[data-pos]');
  const v = el?.getAttribute('data-pos');
  return v === null || v === undefined ? null : Number(v);
}

/**
 * One cell square. Memoised on primitive props, so a hover only re-renders the cells whose state
 * changes. The allowed fill fades with a CSS transition (staggered by `delay`), which the global
 * reduced-motion rule turns off. The target's "?" glyph is drawn in the overlay layer.
 */
const Cell = memo(function Cell({
  pos,
  x,
  y,
  cell,
  fill,
  allowed,
  delay,
  isTarget,
  hovered,
  clickable,
}: {
  pos: number;
  x: number;
  y: number;
  cell: number;
  fill: string;
  allowed: boolean;
  delay: number;
  isTarget: boolean;
  hovered: boolean;
  clickable: boolean;
}) {
  return (
    <g data-pos={pos} style={{ cursor: clickable ? 'pointer' : 'default' }}>
      {isTarget && <title>masked target</title>}
      <rect x={x} y={y} width={cell} height={cell} rx={2} style={{ fill: 'var(--surface-2)', stroke: 'var(--border-strong)' }} strokeWidth={0.75} />
      <rect
        x={x}
        y={y}
        width={cell}
        height={cell}
        rx={2}
        style={{ fill, opacity: allowed ? 1 : 0, transition: `opacity 220ms ease ${allowed ? delay : 0}s` }}
      />
      {isTarget && (
        <rect
          x={x - 1.5}
          y={y - 1.5}
          width={cell + 3}
          height={cell + 3}
          rx={3}
          fill="none"
          style={{ stroke: 'var(--accent)' }}
          strokeWidth={1.25}
          strokeDasharray="2.5 1.8"
        />
      )}
      {hovered && <rect x={x - 1.5} y={y - 1.5} width={cell + 3} height={cell + 3} rx={2.5} fill="none" style={{ stroke: 'var(--ink)' }} strokeWidth={1.75} />}
    </g>
  );
});
