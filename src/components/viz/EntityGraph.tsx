/**
 * EntityGraph: rows as nodes, F→P links as arrows (child → parent, spec §1.3),
 * laid out with d3-force (deterministic: seeded random source, fixed ticks).
 *
 * Table identity = hue × shape (composite encoding; the ShopDB slots fail the
 * all-pairs colour check, see index.css) plus a legend. Highlight API:
 * `highlight`, `muted`, `rings`, `badges`, `highlightEdges`, `selected`.
 * Pass `positions` to pin a layout (e.g. keep nodes still while a BFS animates).
 */
import { forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY, type SimulationNodeDatum } from 'd3';
import { useEffect, useId, useMemo, useState } from 'react';
import { catVar } from '@/styles/palette';
import { cx } from '../ui/cx';
import { shapeOfSlot, shapePath } from './shapes';
import { textWidth } from './text';
import { TableSwatch } from './TableSwatch';
import { toSet, useElementSize } from './useSize';

export type NodeId = number | string;

export interface GraphNodeSpec {
  id: NodeId;
  /** Short label, e.g. "c12" or "orders[3]". */
  label: string;
  table: string;
  slot: number;
  /** Tooltip text (defaults to label). */
  title?: string;
}

export interface GraphEdgeSpec {
  /** Child row (holds the FK). */
  source: NodeId;
  /** Parent row (referenced by the FK). */
  target: NodeId;
}

export type RingStyle = 'accent' | 'dashed' | 'target' | 'muted';

export interface EntityGraphProps {
  nodes: readonly GraphNodeSpec[];
  edges: readonly GraphEdgeSpec[];
  highlight?: ReadonlySet<NodeId> | readonly NodeId[];
  /** Dim everything not highlighted (when highlight is non-empty). */
  dimOthers?: boolean;
  muted?: ReadonlySet<NodeId> | readonly NodeId[];
  /** Per-node ring: accent (e.g. visited), dashed (frontier), target (seed), muted. */
  rings?: ReadonlyMap<NodeId, RingStyle>;
  /** Small text badge next to a node (e.g. BFS depth). */
  badges?: ReadonlyMap<NodeId, string>;
  /** Edge keys "source->target" to draw in accent. */
  highlightEdges?: ReadonlySet<string> | readonly string[];
  selected?: NodeId | null;
  onNodeHover?: (id: NodeId | null) => void;
  onNodeClick?: (id: NodeId) => void;
  /** Pinned positions (layout units); nodes without one are simulated. */
  positions?: ReadonlyMap<NodeId, { x: number; y: number }>;
  /** Called once per layout with the final positions (to pin them later). */
  onLayout?: (positions: Map<NodeId, { x: number; y: number }>) => void;
  /** 'all' | 'highlighted' | 'none' (default: 'highlighted', or 'all' for ≤ 40 nodes). */
  labels?: 'all' | 'highlighted' | 'none';
  height?: number;
  /** Layout seed (deterministic). */
  seed?: number;
  linkDistance?: number;
  charge?: number;
  legend?: boolean;
  /** Accessible name. */
  title?: string;
  className?: string;
}

interface SimNode extends SimulationNodeDatum {
  id: NodeId;
}

const R = 7.5;
/** Node label and badge sizes (screen px; constant whatever the zoom). */
const LABEL_PX = 13.5;
const BADGE_PX = 13;
/** Gap between a node's centre and its label. */
const LABEL_GAP = R + 7;

type LabelPlacement = { dx: number; dy: number; anchor: 'start' | 'end' | 'middle' };
interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  owner: NodeId | null;
}
const overlaps = (a: Box, b: Box) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

/** Small deterministic LCG for d3's randomSource. */
function lcg(seed: number) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export const edgeKey = (s: NodeId, t: NodeId) => `${s}->${t}`;

export function EntityGraph({
  nodes,
  edges,
  highlight,
  dimOthers = true,
  muted,
  rings,
  badges,
  highlightEdges,
  selected = null,
  onNodeHover,
  onNodeClick,
  positions,
  onLayout,
  labels,
  height = 420,
  seed = 7,
  linkDistance = 34,
  charge = -70,
  legend = true,
  title = 'Entity graph: rows linked by foreign keys',
  className,
}: EntityGraphProps) {
  const uid = useId().replace(/:/g, '');
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<NodeId | null>(null);
  const hl = toSet(highlight);
  const mu = toSet(muted);
  const hlE = toSet(highlightEdges);

  // Structure key: re-layout only when nodes/edges change, not on highlight changes.
  const structKey = useMemo(
    () => nodes.map((n) => n.id).join(',') + '|' + edges.map((e) => edgeKey(e.source, e.target)).join(','),
    [nodes, edges],
  );

  const layout = useMemo(() => {
    const simNodes: SimNode[] = nodes.map((n) => {
      const p = positions?.get(n.id);
      return p ? { id: n.id, x: p.x, y: p.y, fx: p.x, fy: p.y } : { id: n.id };
    });
    const index = new Map(simNodes.map((n, i) => [n.id, i]));
    const links = edges
      .filter((e) => index.has(e.source) && index.has(e.target))
      .map((e) => ({ source: index.get(e.source)!, target: index.get(e.target)! }));
    const needSim = simNodes.some((n) => n.fx === undefined);
    if (needSim) {
      const sim = forceSimulation(simNodes)
        .randomSource(lcg(seed))
        .force('link', forceLink(links).distance(linkDistance).strength(0.7))
        .force('charge', forceManyBody().strength(charge).distanceMax(260))
        .force('collide', forceCollide(R + 3))
        .force('x', forceX(0).strength(0.06))
        .force('y', forceY(0).strength(0.09))
        .stop();
      const ticks = Math.min(400, 120 + Math.ceil(Math.log2(simNodes.length + 1) * 30));
      for (let i = 0; i < ticks; i++) sim.tick();
    }
    const pos = new Map<NodeId, { x: number; y: number }>();
    for (const n of simNodes) pos.set(n.id, { x: n.x ?? 0, y: n.y ?? 0 });
    return pos;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structKey, seed, linkDistance, charge, positions]);

  useEffect(() => {
    onLayout?.(layout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout]);

  // Fit the layout into the viewBox.
  const view = useMemo(() => {
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (const p of layout.values()) {
      x0 = Math.min(x0, p.x);
      y0 = Math.min(y0, p.y);
      x1 = Math.max(x1, p.x);
      y1 = Math.max(y1, p.y);
    }
    if (!Number.isFinite(x0)) return { x: -50, y: -50, w: 100, h: 100 };
    const pad = 28;
    return { x: x0 - pad, y: y0 - pad, w: Math.max(60, x1 - x0 + 2 * pad), h: Math.max(60, y1 - y0 + 2 * pad) };
  }, [layout]);

  const width = size.width || 640;
  const scale = Math.min(width / view.w, height / view.h, 2.2);
  const showLabels = labels ?? (nodes.length <= 40 ? 'all' : 'highlighted');
  const nodeById = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const tables = useMemo(() => {
    const seen = new Map<string, number>();
    for (const n of nodes) if (!seen.has(n.table)) seen.set(n.table, n.slot);
    return [...seen.entries()];
  }, [nodes]);

  // ── label placement (screen px): greedy, priority first, never overlapping ──
  // Labels that would collide with another label, a node or a badge are left
  // out instead of being shrunk; every node still names itself on hover/focus.
  const vx0 = view.x + view.w / 2 - width / scale / 2;
  const vy0 = view.y + view.h / 2 - height / scale / 2;
  const placements = useMemo(() => {
    const out = new Map<NodeId, LabelPlacement>();
    if (showLabels === 'none') return out;
    const hlSet = toSet(highlight);
    const boxes: Box[] = [];
    const scr = (id: NodeId) => {
      const p = layout.get(id);
      return p ? { x: (p.x - vx0) * scale, y: (p.y - vy0) * scale } : null;
    };
    for (const n of nodes) {
      const q = scr(n.id);
      if (q) boxes.push({ x0: q.x - R - 2, y0: q.y - R - 2, x1: q.x + R + 2, y1: q.y + R + 2, owner: n.id });
      const b = badges?.get(n.id);
      if (q && b) {
        const w = textWidth(b, BADGE_PX, { mono: true });
        const base = q.y - (R + 7);
        boxes.push({ x0: q.x - w / 2 - 1, y0: base - BADGE_PX * 0.8, x1: q.x + w / 2 + 1, y1: base + BADGE_PX * 0.25, owner: null });
      }
    }
    const prio = (n: GraphNodeSpec) => {
      const ring = rings?.get(n.id);
      if (n.id === selected) return 0;
      if (ring === 'target') return 1;
      if (hlSet.has(n.id)) return 2;
      if (ring && ring !== 'muted') return 3;
      return 4;
    };
    const wanted = nodes
      .filter((n) => showLabels === 'all' || prio(n) <= 2)
      .map((n, i) => ({ n, i, p: prio(n) }))
      .sort((a, b) => a.p - b.p || a.i - b.i);
    const h = LABEL_PX * 1.25;
    for (const { n } of wanted) {
      const q = scr(n.id);
      if (!q) continue;
      const w = textWidth(n.label, LABEL_PX, { mono: true }) + 2;
      const hasBadge = !!badges?.get(n.id);
      const cands: (LabelPlacement & { box: Box })[] = [
        { dx: LABEL_GAP, dy: 0, anchor: 'start', box: { x0: q.x + LABEL_GAP, y0: q.y - h / 2, x1: q.x + LABEL_GAP + w, y1: q.y + h / 2, owner: n.id } },
        { dx: -LABEL_GAP, dy: 0, anchor: 'end', box: { x0: q.x - LABEL_GAP - w, y0: q.y - h / 2, x1: q.x - LABEL_GAP, y1: q.y + h / 2, owner: n.id } },
        { dx: 0, dy: R + 4 + h / 2, anchor: 'middle', box: { x0: q.x - w / 2, y0: q.y + R + 4, x1: q.x + w / 2, y1: q.y + R + 4 + h, owner: n.id } },
      ];
      if (!hasBadge) {
        cands.push({ dx: 0, dy: -(R + 4 + h / 2), anchor: 'middle', box: { x0: q.x - w / 2, y0: q.y - R - 4 - h, x1: q.x + w / 2, y1: q.y - R - 4, owner: n.id } });
      }
      const ok = cands.find(
        (c) =>
          c.box.x0 >= 2 &&
          c.box.x1 <= width - 2 &&
          c.box.y0 >= 2 &&
          c.box.y1 <= height - 2 &&
          !boxes.some((b) => b.owner !== n.id && overlaps(b, c.box)),
      );
      if (ok) {
        boxes.push(ok.box);
        out.set(n.id, { dx: ok.dx, dy: ok.dy, anchor: ok.anchor });
      }
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, nodes, badges, rings, highlight, selected, showLabels, width, height, scale, vx0, vy0]);

  const active = hover ?? selected;
  const neighbors = useMemo(() => {
    if (active === null) return null;
    const s = new Set<NodeId>([active]);
    for (const e of edges) {
      if (e.source === active) s.add(e.target);
      if (e.target === active) s.add(e.source);
    }
    return s;
  }, [active, edges]);

  const isDim = (id: NodeId) => {
    if (mu.has(id)) return true;
    if (dimOthers && hl.size > 0 && !hl.has(id)) return true;
    if (neighbors && !neighbors.has(id) && hl.size === 0) return true;
    return false;
  };

  const focusable = nodes.length <= 150 && (!!onNodeClick || !!onNodeHover);
  const k = 1 / scale; // keep strokes/text constant in screen px

  const hovered = hover !== null ? nodeById.get(hover) : undefined;
  const hoveredPos = hover !== null ? layout.get(hover) : undefined;

  return (
    <div ref={wrapRef} className={cx('relative w-full min-w-0', className)}>
      <svg
        role="img"
        aria-label={`${title}. ${nodes.length} rows, ${edges.length} links.`}
        width="100%"
        height={height}
        viewBox={`${view.x + view.w / 2 - width / scale / 2} ${view.y + view.h / 2 - height / scale / 2} ${width / scale} ${height / scale}`}
        className="block touch-manipulation select-none"
      >
        <defs>
          {(['n', 'a'] as const).map((v) => (
            <marker
              key={v}
              id={`${uid}-arrow-${v}`}
              viewBox="0 0 8 8"
              refX="7.2"
              refY="4"
              markerWidth={7}
              markerHeight={7}
              markerUnits="userSpaceOnUse"
              orient="auto"
            >
              <path d="M0,0.6 L7.6,4 L0,7.4 Z" style={{ fill: v === 'a' ? 'var(--accent)' : 'var(--axis)' }} />
            </marker>
          ))}
        </defs>
        <g>
          {edges.map((e) => {
            const a = layout.get(e.source);
            const b = layout.get(e.target);
            if (!a || !b) return null;
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const len = Math.hypot(dx, dy) || 1;
            const shrink = (R + 3) * k;
            const x2 = b.x - (dx / len) * shrink;
            const y2 = b.y - (dy / len) * shrink;
            const key = edgeKey(e.source, e.target);
            const on = hlE.has(key) || (active !== null && (e.source === active || e.target === active));
            const dim = !on && (isDim(e.source) || isDim(e.target));
            return (
              <line
                key={key}
                x1={a.x}
                y1={a.y}
                x2={x2}
                y2={y2}
                style={{ stroke: on ? 'var(--accent)' : 'var(--axis)', opacity: dim ? 0.25 : 1 }}
                strokeWidth={(on ? 1.75 : 1.1) * k}
                markerEnd={`url(#${uid}-arrow-${on ? 'a' : 'n'})`}
              />
            );
          })}
        </g>
        <g>
          {nodes.map((n) => {
            const p = layout.get(n.id);
            if (!p) return null;
            const ring = rings?.get(n.id);
            const dim = isDim(n.id);
            const isSel = selected === n.id || hover === n.id;
            const isHl = hl.has(n.id);
            const labelOn = showLabels === 'all' || (showLabels === 'highlighted' && (isHl || isSel || ring === 'target'));
            const placement = placements.get(n.id);
            const shape = shapeOfSlot(n.slot);
            return (
              <g
                key={String(n.id)}
                transform={`translate(${p.x},${p.y})`}
                style={{ opacity: dim ? 0.28 : 1, cursor: onNodeClick ? 'pointer' : 'default' }}
                tabIndex={focusable ? 0 : undefined}
                role={focusable ? 'button' : undefined}
                aria-label={focusable ? `${n.title ?? n.label} (${n.table})` : undefined}
                onPointerEnter={() => {
                  setHover(n.id);
                  onNodeHover?.(n.id);
                }}
                onPointerLeave={() => {
                  setHover(null);
                  onNodeHover?.(null);
                }}
                onFocus={() => {
                  setHover(n.id);
                  onNodeHover?.(n.id);
                }}
                onBlur={() => {
                  setHover(null);
                  onNodeHover?.(null);
                }}
                onClick={() => onNodeClick?.(n.id)}
                onKeyDown={(e) => {
                  if ((e.key === 'Enter' || e.key === ' ') && onNodeClick) {
                    e.preventDefault();
                    onNodeClick(n.id);
                  }
                }}
              >
                {/* hit target ≥ 24px (dataviz interaction rule) */}
                <circle r={14 * k} fill="transparent" />
                {(ring || isSel) && (
                  <circle
                    r={(R + 4.5) * k}
                    fill="none"
                    style={{ stroke: ring === 'muted' ? 'var(--axis)' : 'var(--accent)' }}
                    strokeWidth={(ring === 'target' || isSel ? 2 : 1.5) * k}
                    strokeDasharray={ring === 'dashed' || ring === 'target' ? `${3 * k} ${2.5 * k}` : undefined}
                  />
                )}
                <path
                  d={shapePath(shape, R * k)}
                  style={{ fill: catVar(n.slot), stroke: 'var(--surface)' }}
                  strokeWidth={2 * k}
                  paintOrder="stroke"
                />
                {ring === 'target' && (
                  <text
                    y={0.5 * k}
                    textAnchor="middle"
                    dominantBaseline="central"
                    style={{ fill: '#ffffff', fontSize: 12.5 * k, fontWeight: 700 }}
                  >
                    ?
                  </text>
                )}
                {labelOn && placement && (
                  <text
                    x={placement.dx * k}
                    y={placement.dy * k}
                    textAnchor={placement.anchor}
                    dominantBaseline="central"
                    className="font-mono"
                    style={{
                      fill: 'var(--ink)',
                      fontSize: LABEL_PX * k,
                      stroke: 'var(--surface)',
                      strokeWidth: 3 * k,
                      paintOrder: 'stroke',
                      strokeLinejoin: 'round',
                    }}
                  >
                    {n.label}
                  </text>
                )}
                {badges?.get(n.id) && (
                  <text
                    x={0}
                    y={-(R + 7) * k}
                    textAnchor="middle"
                    className="font-mono"
                    style={{
                      fill: 'var(--ink-2)',
                      fontSize: BADGE_PX * k,
                      stroke: 'var(--surface)',
                      strokeWidth: 3 * k,
                      paintOrder: 'stroke',
                    }}
                  >
                    {badges.get(n.id)}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>

      {hovered && hoveredPos && (
        <div
          role="tooltip"
          className="pointer-events-none absolute z-10 max-w-[260px] rounded-[9px] border border-border bg-surface px-3 py-2 text-[15px] shadow-[var(--shadow-pop)]"
          style={{
            left: Math.min(
              width - 200,
              Math.max(4, (hoveredPos.x - (view.x + view.w / 2 - width / scale / 2)) * scale + 12),
            ),
            top: Math.max(4, (hoveredPos.y - (view.y + view.h / 2 - height / scale / 2)) * scale - 40),
          }}
        >
          <div className="font-mono font-semibold text-ink">{hovered.title ?? hovered.label}</div>
          <div className="mt-0.5 flex items-center gap-1.5 text-ink-2">
            <TableSwatch slot={hovered.slot} shape label={hovered.table} />
          </div>
        </div>
      )}

      {legend && tables.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[15.5px]" aria-label="Legend">
          {tables.map(([t, slot]) => (
            <TableSwatch key={t} slot={slot} shape label={t} size={11} />
          ))}
          <span className="inline-flex items-center gap-1.5 text-muted">
            <svg aria-hidden="true" width="26" height="8" viewBox="0 0 26 8">
              <line x1="1" y1="4" x2="20" y2="4" style={{ stroke: 'var(--axis)' }} strokeWidth="1.5" />
              <path d="M18,0.8 L25,4 L18,7.2 Z" style={{ fill: 'var(--axis)' }} />
            </svg>
            F→P (child → parent)
          </span>
        </div>
      )}
    </div>
  );
}
