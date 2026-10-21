/**
 * Landing-page motif: database cells turning into tokens, then relational
 * attention arcs from the masked target. A loop of three phases:
 *   tables → tokens → attention.
 * Illustrative (a hand-made toy, not a sampled context). It pauses when off
 * screen, has a pause button, and shows the final state statically under
 * prefers-reduced-motion.
 *
 * Geometry is computed in CSS pixels from the measured width (the SVG is
 * drawn 1:1, never scaled), so every label keeps its real size (≥ 13px) and
 * the tables and the token strip wrap onto more rows on narrow screens.
 */
import { Pause, Play } from 'lucide-react';
import { motion, useInView } from 'motion/react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { textWidth } from '@/components/viz/text';
import { useElementSize } from '@/components/viz/useSize';
import { cx } from '@/components/ui/cx';
import { usePrefersReducedMotion } from '@/styles/theme';
import { catVar } from '@/styles/palette';

type Phase = 'tables' | 'tokens' | 'attention';
const PHASES: Phase[] = ['tables', 'tokens', 'attention'];
const DURATION: Record<Phase, number> = { tables: 2800, tokens: 2600, attention: 3600 };

interface TableSpec {
  name: string;
  slot: number;
  cols: string[];
  rows: string[][];
  /** Parent table index (F→P link), if any. */
  parent?: number;
}

// Toy rows in the spirit of spec §6.3 (customers, orders, the churn task table).
const TABLES: TableSpec[] = [
  { name: 'customers', slot: 1, cols: ['country', 'age', 'is_premium'], rows: [['BR', '34', 'true'], ['US', '51', 'false']] },
  { name: 'orders', slot: 3, cols: ['order_time', 'quantity'], rows: [['02-05', '2'], ['02-20', '1']], parent: 0 },
  { name: 'customer-churn', slot: 5, cols: ['timestamp', 'churn'], rows: [['02-01', 'false'], ['03-01', '?']], parent: 0 },
];

// Token order: seed row first (the row with "?"), then rows in a BFS-like order.
// [tableIndex, rowIndex]
const ROW_ORDER: [number, number][] = [
  [2, 1],
  [0, 0],
  [2, 0],
  [1, 0],
  [0, 1],
  [1, 1],
];

// Type (CSS px, drawn 1:1)
const VALUE_PX = 13.5;
const HEAD_PX = 14;
const NOTE_PX = 13.5;
// Table cards
const CELL_H = 28;
const GAP = 3;
const PAD = 8;
const HEAD_H = 24;
const TABLE_GAP_X = 16;
const TABLE_GAP_Y = 34;
// Token strip
const CHIP_H = 34;
const CHIP_GAP_X = 5;
const CHIP_GAP_Y = 16;
const ARC_SPACE = 66;

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface CellGeom {
  key: string;
  table: number;
  row: number;
  col: number;
  text: string;
  slot: number;
  isTarget: boolean;
  /** position in the table card */
  tx: number;
  ty: number;
  /** position in the token strip */
  kx: number;
  ky: number;
  /** strip row of the token */
  kRow: number;
  order: number;
}

/** Lay the toy out for a container `W` px wide. */
function layoutFor(W: number) {
  const allValues = TABLES.flatMap((t) => t.rows.flat());
  const valueW = Math.max(...allValues.map((v) => textWidth(v, VALUE_PX, { mono: true })));
  const cellW = Math.ceil(valueW + 16);
  const chipW = Math.ceil(valueW + 20);

  // table cards, flowed into centred rows
  const sizes = TABLES.map((t) => ({
    w: Math.ceil(Math.max(t.cols.length * cellW + (t.cols.length - 1) * GAP + 2 * PAD, textWidth(t.name, HEAD_PX, { mono: true, weight: 600 }) + 2 * PAD)),
    h: HEAD_H + t.rows.length * CELL_H + (t.rows.length - 1) * GAP + PAD,
  }));
  const rows: number[][] = [[]];
  let lineW = 0;
  sizes.forEach((s, i) => {
    const need = (rows[rows.length - 1].length ? TABLE_GAP_X : 0) + s.w;
    if (rows[rows.length - 1].length && lineW + need > W) {
      rows.push([i]);
      lineW = s.w;
    } else {
      rows[rows.length - 1].push(i);
      lineW += need;
    }
  });
  const singleRow = rows.length === 1;
  const rowOf: number[] = new Array(TABLES.length);
  rows.forEach((r, ri) => r.forEach((i) => (rowOf[i] = ri)));
  // same-row F→P arcs rise above the cards: reserve room for them (and the label on one row)
  const sameRowArc = TABLES.some((t, i) => t.parent !== undefined && rowOf[i] === rowOf[t.parent]);
  const topSpace = singleRow ? 56 : sameRowArc ? 40 : 8;
  const cards: Box[] = new Array(TABLES.length);
  let y = topSpace;
  rows.forEach((r) => {
    const total = r.reduce((s, i) => s + sizes[i].w, 0) + TABLE_GAP_X * (r.length - 1);
    let x = Math.max(0, (W - total) / 2);
    const rowH = Math.max(...r.map((i) => sizes[i].h));
    for (const i of r) {
      cards[i] = { x, y, w: sizes[i].w, h: sizes[i].h };
      x += sizes[i].w + TABLE_GAP_X;
    }
    y += rowH + TABLE_GAP_Y;
  });
  const tablesBottom = y - TABLE_GAP_Y;

  // token strip, wrapped into rows of `perRow` chips
  const nTokens = ROW_ORDER.reduce((s, [t]) => s + TABLES[t].cols.length, 0);
  const perRow = Math.max(3, Math.min(nTokens, Math.floor((W + CHIP_GAP_X) / (chipW + CHIP_GAP_X))));
  const stripRows = Math.ceil(nTokens / perRow);
  const stripTop = tablesBottom + ARC_SPACE;
  const rowWidth = (n: number) => n * chipW + (n - 1) * CHIP_GAP_X;
  const stripX0 = Math.max(0, (W - rowWidth(Math.min(perRow, nTokens))) / 2);

  const cells: CellGeom[] = [];
  let order = 0;
  for (const [ti, ri] of ROW_ORDER) {
    const t = TABLES[ti];
    t.cols.forEach((_, ci) => {
      const text = t.rows[ri][ci];
      const kRow = Math.floor(order / perRow);
      const kCol = order % perRow;
      cells.push({
        key: `${ti}-${ri}-${ci}`,
        table: ti,
        row: ri,
        col: ci,
        text,
        slot: t.slot,
        isTarget: text === '?',
        tx: cards[ti].x + PAD + ci * (cellW + GAP),
        ty: cards[ti].y + HEAD_H + ri * (CELL_H + GAP),
        kx: stripX0 + kCol * (chipW + CHIP_GAP_X),
        ky: stripTop + kRow * (CHIP_H + CHIP_GAP_Y),
        kRow,
        order,
      });
      order++;
    });
  }
  const height = Math.ceil(stripTop + stripRows * CHIP_H + (stripRows - 1) * CHIP_GAP_Y + 6);
  return { cellW, chipW, cards, rowOf, singleRow, cells, stripTop, height };
}

const EASE = [0.25, 0.8, 0.25, 1] as const;

export function HeroMotif() {
  const reduced = usePrefersReducedMotion();
  const ref = useRef<HTMLDivElement | null>(null);
  const [sizeRef, size] = useElementSize<HTMLDivElement>();
  const inView = useInView(ref, { amount: 0.3 });
  const [paused, setPaused] = useState(false);
  const [phase, setPhase] = useState<Phase>('tables');
  const W = Math.max(280, Math.round(size.width || 520));
  const { cellW, chipW, cards, rowOf, singleRow, cells, stripTop, height } = useMemo(() => layoutFor(W), [W]);

  const animating = !reduced && !paused && inView;
  useEffect(() => {
    if (!animating) return;
    const t = window.setTimeout(() => setPhase((p) => PHASES[(PHASES.indexOf(p) + 1) % PHASES.length]), DURATION[phase]);
    return () => window.clearTimeout(t);
  }, [phase, animating]);

  const shown: Phase = reduced ? 'attention' : phase;
  const asTokens = shown !== 'tables';
  const target = cells.find((c) => c.isTarget)!;

  // Who the masked target may read (spec §6.2): feature = own row + parent row (customers[0]);
  // column = other cells of "churn of customer-churn".
  const featureKeys = cells.filter((c) => (c.table === target.table && c.row === target.row && !c.isTarget) || (c.table === 0 && c.row === 0));
  const columnKeys = cells.filter((c) => c.table === target.table && c.col === target.col && !c.isTarget);

  const caption: Record<Phase, string> = {
    tables: 'A relational database: rows, columns, keys, and a task table whose label is masked.',
    tokens: 'Every non-missing feature cell becomes one token: value, column name, table name.',
    attention: 'Relational attention: the masked cell reads its own row, its parent row, and other labels in its column.',
  };

  // Attention arc from the target chip to a key chip: above the strip within a
  // row, through the gap between rows otherwise (chips are drawn on top).
  const arc = (from: CellGeom, to: CellGeom) => {
    const x1 = from.kx + chipW / 2;
    const x2 = to.kx + chipW / 2;
    if (from.kRow === to.kRow) {
      const y = from.ky - 3;
      const lift = Math.min(ARC_SPACE - 10, 24 + Math.abs(to.order - from.order) * 7);
      return `M${x1},${y} Q${(x1 + x2) / 2},${y - lift} ${x2},${y}`;
    }
    const down = to.kRow > from.kRow;
    const y1 = down ? from.ky + CHIP_H + 2 : from.ky - 2;
    const y2 = down ? to.ky - 2 : to.ky + CHIP_H + 2;
    const bend = (CHIP_GAP_Y + CHIP_H) * 0.6 * (down ? 1 : -1);
    return `M${x1},${y1} C${x1},${y1 + bend} ${x2},${y2 - bend} ${x2},${y2}`;
  };

  // F→P arrows between table cards (child → parent).
  const fkPaths = TABLES.flatMap((t, ci) => {
    if (t.parent === undefined) return [];
    const c = cards[ci];
    const p = cards[t.parent];
    if (rowOf[ci] === rowOf[t.parent]) {
      // same row: arc over the cards, from the child's top to the parent's top
      const lift = ci === 1 ? 26 : 42;
      const x1 = c.x + Math.min(36, c.w / 3);
      const x2 = p.x + p.w - (ci === 1 ? 40 : 22);
      return [`M${x1},${c.y - 2} C${x1 - 16},${c.y - lift} ${x2 + 16},${p.y - lift} ${x2},${p.y - 3}`];
    }
    // parent in a row above: from the child's top up to the parent's bottom edge
    const x1 = c.x + c.w / 2;
    const x2 = Math.min(p.x + p.w - 16, Math.max(p.x + 16, x1));
    const y2 = p.y + p.h + 3;
    return [`M${x1},${c.y - 2} C${x1},${c.y - TABLE_GAP_Y * 0.6} ${x2},${y2 + TABLE_GAP_Y * 0.6} ${x2},${y2}`];
  });

  const legendLine = (dashed: boolean) => (
    <svg aria-hidden="true" width="22" height="8" viewBox="0 0 22 8" className="shrink-0">
      <line x1="1" x2="21" y1="4" y2="4" style={{ stroke: 'var(--accent)' }} strokeWidth="2" strokeDasharray={dashed ? '4 3' : undefined} strokeLinecap="round" />
    </svg>
  );

  return (
    <div ref={ref} className="relative">
      <div ref={sizeRef} className="w-full">
        <svg
          width={W}
          height={height}
          viewBox={`0 0 ${W} ${height}`}
          className="block max-w-full"
          role="img"
          aria-label="Animation: database cells become tokens, then the masked target cell attends to related tokens."
        >
          <defs>
            <marker id="hero-fk" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto" markerUnits="userSpaceOnUse">
              <path d="M0,0.8 L7.4,4 L0,7.2 Z" style={{ fill: 'var(--axis)' }} />
            </marker>
          </defs>

          {/* table cards */}
          {cards.map((c, i) => (
            <motion.g key={TABLES[i].name} initial={false} animate={{ opacity: asTokens ? 0.42 : 1 }} transition={{ duration: 0.6 }}>
              <rect x={c.x} y={c.y} width={c.w} height={c.h} rx={9} style={{ fill: 'var(--surface)', stroke: 'var(--border-strong)' }} strokeWidth={1} />
              <rect x={c.x} y={c.y} width={c.w} height={3} rx={1.5} style={{ fill: catVar(TABLES[i].slot) }} />
              <text x={c.x + PAD} y={c.y + 18} className="font-mono" style={{ fill: 'var(--ink)', fontSize: HEAD_PX, fontWeight: 600 }}>
                {TABLES[i].name}
              </text>
            </motion.g>
          ))}
          {/* F→P arrows between cards: orders → customers, customer-churn → customers */}
          <motion.g initial={false} animate={{ opacity: asTokens ? 0 : 1 }} transition={{ duration: 0.4 }}>
            {fkPaths.map((d, i) => (
              <path key={i} d={d} fill="none" style={{ stroke: 'var(--axis)' }} strokeWidth={1.5} markerEnd="url(#hero-fk)" />
            ))}
            {singleRow && (
              <text x={W / 2} y={14} textAnchor="middle" style={{ fill: 'var(--muted)', fontSize: NOTE_PX }}>
                foreign keys (F→P)
              </text>
            )}
          </motion.g>

          {/* ghost cells stay in the tables (faint values once the tokens have left) */}
          {cells.map((c) => (
            <g key={`g-${c.key}`}>
              <rect
                x={c.tx}
                y={c.ty}
                width={cellW}
                height={CELL_H}
                rx={5}
                style={{ fill: 'var(--surface-2)', stroke: 'var(--grid)' }}
                strokeWidth={1}
                strokeDasharray={c.isTarget ? '3 2.5' : undefined}
              />
              <text
                x={c.tx + cellW / 2}
                y={c.ty + CELL_H / 2}
                textAnchor="middle"
                dominantBaseline="central"
                className="font-mono"
                style={{ fill: 'var(--muted)', fontSize: VALUE_PX, opacity: 0.85 }}
              >
                {c.text}
              </text>
            </g>
          ))}

          {/* the context window: one slot per token (always visible) */}
          {cells.map((c) => (
            <rect
              key={`slot-${c.key}`}
              x={c.kx}
              y={c.ky}
              width={chipW}
              height={CHIP_H}
              rx={6}
              fill="none"
              style={{ stroke: 'var(--axis)' }}
              strokeWidth={1}
              strokeDasharray="2.5 2.5"
            />
          ))}
          <motion.text
            x={cells[0].kx}
            y={stripTop - 10}
            style={{ fill: 'var(--muted)', fontSize: NOTE_PX }}
            initial={false}
            animate={{ opacity: asTokens ? 0 : 1 }}
            transition={{ duration: reduced ? 0 : 0.3 }}
          >
            context window: one slot per token
          </motion.text>

          {/* attention arcs */}
          <g>
            {[...featureKeys.map((k) => ({ k, kind: 'feat' as const })), ...columnKeys.map((k) => ({ k, kind: 'col' as const }))].map(({ k, kind }, i) => (
              <motion.path
                key={`${kind}-${k.key}`}
                d={arc(target, k)}
                fill="none"
                style={{ stroke: 'var(--accent)' }}
                strokeWidth={1.8}
                strokeLinecap="round"
                strokeDasharray={kind === 'col' ? '4 3' : undefined}
                initial={false}
                animate={{ pathLength: shown === 'attention' ? 1 : 0, opacity: shown === 'attention' ? 0.9 : 0 }}
                transition={{ duration: reduced ? 0 : 0.7, delay: reduced ? 0 : shown === 'attention' ? 0.1 + i * 0.08 : 0, ease: EASE }}
              />
            ))}
          </g>

          {/* moving cells */}
          {cells.map((c) => {
            const x = asTokens ? c.kx : c.tx;
            const y = asTokens ? c.ky : c.ty;
            const w = asTokens ? chipW : cellW;
            const h = asTokens ? CHIP_H : CELL_H;
            const delay = reduced ? 0 : asTokens ? (shown === 'tokens' ? c.order * 0.045 : 0) : (cells.length - c.order) * 0.02;
            return (
              <motion.g key={c.key} initial={false} animate={{ x, y }} transition={{ duration: reduced ? 0 : 0.75, delay, ease: EASE }}>
                <motion.rect
                  initial={false}
                  animate={{ width: w, height: h }}
                  transition={{ duration: reduced ? 0 : 0.75, delay, ease: EASE }}
                  rx={6}
                  style={{
                    fill: c.isTarget ? 'var(--accent-soft)' : 'var(--surface)',
                    stroke: c.isTarget ? 'var(--accent)' : 'var(--border-strong)',
                  }}
                  strokeWidth={c.isTarget ? 1.5 : 1}
                  strokeDasharray={c.isTarget ? '3.5 2.5' : undefined}
                />
                {!c.isTarget && (
                  <motion.rect
                    initial={false}
                    animate={{ height: h }}
                    transition={{ duration: reduced ? 0 : 0.75, delay, ease: EASE }}
                    x={0}
                    y={0}
                    width={4}
                    rx={1.5}
                    style={{ fill: catVar(c.slot) }}
                  />
                )}
                <motion.text
                  initial={false}
                  animate={{ x: w / 2 + (c.isTarget ? 0 : 2), y: h / 2 }}
                  transition={{ duration: reduced ? 0 : 0.75, delay, ease: EASE }}
                  textAnchor="middle"
                  dominantBaseline="central"
                  className="font-mono"
                  style={{
                    fill: c.isTarget ? 'var(--accent-text)' : 'var(--ink)',
                    fontSize: c.isTarget ? 16 : VALUE_PX,
                    fontWeight: c.isTarget ? 800 : 500,
                  }}
                >
                  {c.text}
                </motion.text>
              </motion.g>
            );
          })}
        </svg>
      </div>

      <div className="mt-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="min-h-[5.5em] text-[16.5px] leading-snug text-ink-2 sm:min-h-[2.9em]" aria-live="off">
            {caption[shown]}
          </p>
          {/* phase notes, in HTML so they wrap instead of shrinking */}
          <div className="mt-1.5 grid text-[15px] leading-snug text-ink-2">
            <div
              className={cx('flex flex-wrap items-center gap-x-4 gap-y-1 transition-opacity duration-300 [grid-area:1/1]', shown === 'attention' ? 'opacity-100' : 'pointer-events-none opacity-0')}
              aria-hidden={shown !== 'attention'}
            >
              <span className="inline-flex items-center gap-2">{legendLine(false)} feature: own row + parent row</span>
              <span className="inline-flex items-center gap-2">{legendLine(true)} column: other labels</span>
            </div>
            <div
              className={cx('transition-opacity duration-300 [grid-area:1/1]', shown === 'tokens' ? 'opacity-100' : 'pointer-events-none opacity-0')}
              aria-hidden={shown !== 'tokens'}
            >
              one token per cell · seed row first · no positional encoding
            </div>
          </div>
        </div>
        {!reduced && (
          <button
            type="button"
            onClick={() => setPaused((p) => !p)}
            aria-label={paused ? 'Play animation' : 'Pause animation'}
            title={paused ? 'Play animation' : 'Pause animation'}
            className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border text-ink-2 hover:bg-surface-2 hover:text-ink"
          >
            {paused ? <Play size={13} fill="currentColor" /> : <Pause size={13} fill="currentColor" />}
          </button>
        )}
      </div>
    </div>
  );
}
