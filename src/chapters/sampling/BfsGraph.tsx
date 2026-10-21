/**
 * Radial view of the BFS (left pane of the chapter's centrepiece, after paper Fig. 6a).
 * The seed sits in the centre; every row the sampler has touched sits on the ring of its depth label
 * (spec §4.5): a visited row on the depth it was visited with, inside the sector of the row that pushed
 * the entry it was popped from; a frontier row where it was first pushed. Filled marks are visited
 * rows (their cells are in the context), hollow marks are frontier entries still waiting.
 *
 * Text is sized in screen pixels (font size / scale), so labels stay at 13-14px at every width.
 *
 * Positions come from `buildLayout` over the WHOLE trace, so nothing moves while the replay runs.
 */
import { motion } from 'motion/react';
import { memo, useEffect, useId, useMemo, useRef, useState } from 'react';
import { ShapeGlyph, shapeOfSlot, shapePath, TableSwatch, cx, textWidth } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import type { PreprocessedDB, RawDatabase } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { usePrefersReducedMotion } from '@/styles/theme';
import { rowName, tsDate, type Layout, type ReplayState } from './trace';

export interface BfsGraphProps {
  layout: Layout;
  state: ReplayState;
  pdb: PreprocessedDB;
  db: RawDatabase;
  slots: Record<string, number>;
  seed: number;
  /** Row being processed in the current step. */
  current: number | null;
  /** Rows pushed during the current step (their links are drawn in accent). */
  fresh: ReadonlySet<number>;
  /** Events [stepStart, cursor) are the current step; `cursor` events have been replayed. */
  stepStart: number;
  cursor: number;
  /** Rows ringed as "candidates" (the bucket a uniform draw is taken from). */
  candidates?: ReadonlySet<number>;
  /** A row whose cells were cut by the hard stop at L. */
  cut?: number | null;
  hover: number | null;
  onHover: (n: number | null) => void;
  onSelect: (n: number) => void;
  /** Max height in px. */
  maxHeight?: number;
  className?: string;
}

const R_VISITED = 5.6;
const PAD = 30;

export function BfsGraph({
  layout,
  state,
  pdb,
  db,
  slots,
  seed,
  current,
  fresh,
  stepStart,
  cursor,
  candidates,
  cut,
  hover,
  onHover,
  onSelect,
  maxHeight = 560,
  className,
}: BfsGraphProps) {
  const uid = useId().replace(/:/g, '');
  const reduce = usePrefersReducedMotion();
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const [focus, setFocus] = useState<number | null>(null);
  const width = size.width || 560;
  // On narrow screens the view zooms to the rings reached so far (all positions stay fixed), so the early
  // steps are not crammed into the middle of a 300px figure.
  const narrow = width < 480;
  const maxShownDepth = useMemo(() => {
    let m = 1;
    for (const n of state.discovered) m = Math.max(m, layout.nodes.get(n)?.depth ?? 0);
    return m;
  }, [state.discovered, layout]);
  const R = narrow ? Math.max(layout.ring(1), layout.ring(Math.min(layout.maxDepth, maxShownDepth))) : layout.radius;
  const vb = { x: -R - PAD, y: -R - PAD, w: 2 * (R + PAD), h: 2 * (R + PAD) };
  const height = Math.min(width, maxHeight);
  const scale = Math.min(width / vb.w, height / vb.h);
  const offX = (width - vb.w * scale) / 2;
  const offY = (height - vb.h * scale) / 2;
  const toPx = (x: number, y: number) => ({ left: offX + (x - vb.x) * scale, top: offY + (y - vb.y) * scale });
  // keep marks legible on small screens
  const k = Math.max(1, 0.62 / scale);
  // text: user units per screen pixel, so `px * t` renders at `px` CSS pixels whatever the scale
  const t = 1 / scale;
  // marks never shrink below these on-screen radii (px): visited rows 5, frontier entries 2.5
  const rVisited = Math.max(R_VISITED, (5 * t) / k);
  const rFrontierMin = (2.5 * t) / k;

  // animate the zoom (viewBox) between steps; React renders the target, the effect tweens the attribute
  const svgRef = useRef<SVGSVGElement | null>(null);
  const shownVb = useRef(vb);
  const vbKey = `${vb.x} ${vb.y} ${vb.w} ${vb.h}`;
  useEffect(() => {
    const el = svgRef.current;
    const from = shownVb.current;
    const to = vb;
    shownVb.current = to;
    if (!el || reduce || (from.w === to.w && from.x === to.x)) return;
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      // rAF timestamps can precede t0 (frame start): clamp, or the ease would overshoot backwards
      const u = Math.max(0, Math.min(1, (now - t0) / 380));
      const e = 1 - (1 - u) ** 3;
      const lerp = (a: number, b: number) => a + (b - a) * e;
      el.setAttribute('viewBox', `${lerp(from.x, to.x)} ${lerp(from.y, to.y)} ${lerp(from.w, to.w)} ${lerp(from.h, to.h)}`);
      if (u < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vbKey, reduce]);

  // frontier marks shrink on crowded rings
  const frontierR = useMemo(() => {
    const out: number[] = [];
    for (let d = 0; d < layout.perDepth.length; d++) {
      const circ = 2 * Math.PI * Math.max(1, layout.ring(d));
      const per = circ / Math.max(1, layout.perDepth[d]);
      out[d] = Math.max(1.6, rFrontierMin, Math.min(3.6, per * 0.38));
    }
    return out;
  }, [layout, rFrontierMin]);

  const nodes = useMemo(() => [...layout.nodes.values()], [layout]);
  const shown = nodes.filter((n) => state.discovered.has(n.node));
  const tablesShown = useMemo(() => {
    const m = new Map<string, number>();
    for (const n of shown) {
      const g = pdb.nodes[n.node];
      if (!m.has(g.tableName)) m.set(g.tableName, slots[g.tableName] ?? 0);
    }
    return [...m.entries()];
  }, [shown, pdb, slots]);

  const active = hover ?? focus;
  const tip = active !== null ? layout.nodes.get(active) : undefined;
  const tipName = active !== null ? rowName(db, pdb, active) : null;

  // ring labels are 13px; label every ring only when the rings are at least 18px apart
  const ringGapPx = (layout.ring(2) - layout.ring(1)) * scale || layout.ring(1) * scale;
  const ringLabelEvery = Math.max(1, Math.ceil(18 / Math.max(1, ringGapPx)));
  const curRing = current !== null ? (state.visitedDepth.get(current) ?? layout.nodes.get(current)?.depth ?? null) : null;

  return (
    <div ref={wrapRef} className={cx('relative w-full min-w-0', className)}>
      <svg
        ref={svgRef}
        role="group"
        aria-label={`Breadth-first search around the seed row: ${state.visitedOrder.length} rows visited, ${shown.length - state.visitedOrder.length} waiting in the frontier.`}
        width={width}
        height={height}
        viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`}
        className="block touch-manipulation select-none"
      >
        <defs>
          <marker id={`${uid}-a`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth={6 * k} markerHeight={6 * k} markerUnits="userSpaceOnUse" orient="auto">
            <path d="M0,0.8 L7.4,4 L0,7.2 Z" style={{ fill: 'var(--ink-2)' }} />
          </marker>
          <marker id={`${uid}-ac`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth={7 * k} markerHeight={7 * k} markerUnits="userSpaceOnUse" orient="auto">
            <path d="M0,0.8 L7.4,4 L0,7.2 Z" style={{ fill: 'var(--accent)' }} />
          </marker>
        </defs>

        {/* depth rings */}
        <g aria-hidden="true">
          {Array.from({ length: narrow ? Math.min(layout.maxDepth, maxShownDepth) : layout.maxDepth }, (_, i) => i + 1).map((d) => (
            <g key={d}>
              <circle r={layout.ring(d)} fill="none" style={{ stroke: d === curRing ? 'var(--accent)' : 'var(--grid)', opacity: d === curRing ? 0.45 : 1 }} strokeWidth={1 * k} />
              {(d - 1) % ringLabelEvery === 0 && (
                <text
                  x={0}
                  y={-layout.ring(d) - 3 * t}
                  textAnchor="middle"
                  className="font-mono"
                  style={{ fill: d === curRing ? 'var(--accent-text)' : 'var(--muted)', fontSize: 13 * t, fontWeight: d === curRing ? 600 : 400, stroke: 'var(--surface)', strokeWidth: 3 * t, paintOrder: 'stroke' }}
                >
                  d{d}
                </text>
              )}
            </g>
          ))}
        </g>

        {/* discovery links */}
        <g aria-hidden="true">
          {shown.map((n) => {
            if (n.parent === null) return null;
            // before the push that the row will be visited through, draw the link of its first push
            const usePath = !n.first || n.pathEvent < cursor;
            const from = usePath ? n.parent : n.first!.parent;
            const via = usePath ? n.via : n.first!.via;
            const p = layout.nodes.get(from);
            if (!p) return null;
            const isFresh = fresh.has(n.node) || (!!n.first && n.pathEvent >= stepStart && n.pathEvent < cursor);
            const on = isFresh || active === n.node || current === n.node;
            // arrows point child → parent (the FK direction): for an F→P link the new row is the parent
            const child = via === 'f2p' ? p : n;
            const parent = via === 'f2p' ? n : p;
            return (
              <Link
                key={isFresh && !reduce ? `${n.node}-fresh-${usePath ? 1 : 0}` : `${n.node}-${usePath ? 1 : 0}`}
                x1={child.x}
                y1={child.y}
                x2={parent.x}
                y2={parent.y}
                f2p={via === 'f2p'}
                on={on}
                bothVisited={state.visitedDepth.has(n.node) && state.visitedDepth.has(from)}
                animate={isFresh && !reduce}
                k={k}
                rTip={rVisited}
                uid={uid}
              />
            );
          })}
        </g>

        {/* rows */}
        <g>
          {shown.map((n) => {
            const g = pdb.nodes[n.node];
            const visited = state.visitedDepth.has(n.node);
            const isSeed = n.node === seed;
            const isFresh = fresh.has(n.node);
            return (
              <Node
                key={isFresh && !reduce ? `${n.node}-fresh` : n.node}
                node={n.node}
                x={n.x}
                y={n.y}
                slot={slots[g.tableName] ?? 0}
                r={isSeed ? Math.max(8.5, 9.5 / scale / k) : visited ? rVisited : (frontierR[n.depth] ?? 2.4)}
                visited={visited}
                isSeed={isSeed}
                isCur={current === n.node}
                isCand={candidates?.has(n.node) ?? false}
                isActive={active === n.node}
                isCut={cut === n.node}
                animate={isFresh && !reduce}
                pulse={!reduce}
                k={k}
                t={t}
                label={visited ? `${rowName(db, pdb, n.node).long}, depth ${state.visitedDepth.get(n.node)}. Jump to its visit.` : undefined}
                onHover={onHover}
                onFocusNode={setFocus}
                onSelect={onSelect}
              />
            );
          })}
        </g>

        {/* label of the current row */}
        {current !== null && layout.nodes.get(current) && (() => {
          const ln = layout.nodes.get(current)!;
          const label = rowName(db, pdb, current).short;
          const wUser = textWidth(label, 14, { mono: true, weight: 600 }) * t;
          const right = ln.x + 12 * t + wUser <= vb.x + vb.w - 2 * t;
          const y = Math.max(vb.y + 16 * t, ln.y - 10 * t);
          return (
            <text
              x={right ? ln.x + 12 * t : ln.x - 12 * t}
              y={y}
              textAnchor={right ? 'start' : 'end'}
              className="font-mono"
              style={{
                fill: 'var(--ink)',
                fontSize: 14 * t,
                fontWeight: 600,
                stroke: 'var(--surface)',
                strokeWidth: 4 * t,
                paintOrder: 'stroke',
                strokeLinejoin: 'round',
                pointerEvents: 'none',
              }}
            >
              {label}
            </text>
          );
        })()}
      </svg>

      {tip && tipName && (
        <div
          role="tooltip"
          className="pointer-events-none absolute z-10 w-max max-w-[260px] rounded-[9px] border border-border bg-surface px-2.5 py-1.5 text-[14.5px] leading-snug shadow-[var(--shadow-pop)]"
          style={{
            left: Math.min(width - 200, Math.max(4, toPx(tip.x, tip.y).left + 14)),
            top: Math.max(4, toPx(tip.x, tip.y).top - 56),
          }}
        >
          <div className="font-mono font-semibold text-ink">{tipName.long}</div>
          <div className="mt-0.5 flex items-center gap-1.5 text-ink-2">
            <TableSwatch slot={slots[pdb.nodes[tip.node].tableName] ?? 0} shape label={tipName.key} />
          </div>
          <div className="num mt-1 text-[14px] text-muted">
            {state.visitedDepth.has(tip.node)
              ? `visited #${state.visitedOrder.indexOf(tip.node) + 1} at depth ${state.visitedDepth.get(tip.node)} · ${state.emittedByNode.get(tip.node)?.length ?? 0} tokens`
              : `in the frontier · depth label ${tip.depth}`}
          </div>
          <div className="mt-0.5 text-[14px] text-muted">
            {tip.via === 'seed'
              ? 'the seed row'
              : state.visitedDepth.has(tip.node) || !tip.first
                ? `${state.visitedDepth.has(tip.node) ? 'visited ' : 'reached '}${tip.via === 'f2p' ? 'as a parent (F→P)' : 'as a child (P→F)'} of ${rowName(db, pdb, tip.parent!).short}`
                : `reached ${tip.first.via === 'f2p' ? 'as a parent (F→P)' : 'as a child (P→F)'} of ${rowName(db, pdb, tip.first.parent).short}`}
            {tip.first && state.visitedDepth.has(tip.node) ? ` (first pushed by ${rowName(db, pdb, tip.first.parent).short} at depth ${tip.first.depth})` : ''}
            {pdb.nodes[tip.node].ts !== null ? ` · ${tsDate(pdb.nodes[tip.node].ts)}` : ''}
          </div>
        </div>
      )}

      <Key narrow={narrow}>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[14.5px] text-ink-2" aria-label="Legend">
        {tablesShown.map(([t, slot]) => (
          <TableSwatch key={t} slot={slot} shape label={t} size={10} />
        ))}
        <span className="inline-flex items-center gap-1.5">
          <ShapeGlyph shape="circle" size={11} fill="var(--ink-2)" /> visited
        </span>
        <span className="inline-flex items-center gap-1.5">
          <ShapeGlyph shape="circle" size={11} fill="var(--surface)" stroke="var(--ink-2)" /> in frontier
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg aria-hidden="true" width="24" height="8" viewBox="0 0 24 8">
            <line x1="1" y1="4" x2="18" y2="4" style={{ stroke: 'var(--ink-2)' }} strokeWidth="1.4" />
            <path d="M16,0.8 L23,4 L16,7.2 Z" style={{ fill: 'var(--ink-2)' }} />
          </svg>
          reached as a parent (F→P)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg aria-hidden="true" width="20" height="8" viewBox="0 0 20 8">
            <line x1="1" y1="4" x2="19" y2="4" style={{ stroke: 'var(--axis)' }} strokeWidth="1.4" />
          </svg>
          reached as a child (P→F)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg aria-hidden="true" width="14" height="14" viewBox="-7 -7 14 14">
            <circle r={6} fill="none" style={{ stroke: 'var(--axis)' }} strokeWidth={1.2} />
          </svg>
          rings d1, d2, …: depth labels (visit depth)
        </span>
      </div>
      </Key>
      {narrow && maxShownDepth < layout.maxDepth && (
        <p className="mt-1 text-[14px] text-muted">
          Zoomed to the rings reached so far (d1–d{maxShownDepth} of {layout.maxDepth}).
        </p>
      )}
    </div>
  );
}

/** The legend; on narrow screens it folds into a "Key" disclosure so the graph keeps the room. */
function Key({ narrow, children }: { narrow: boolean; children: React.ReactNode }) {
  if (!narrow) return <>{children}</>;
  return (
    <details className="mt-1">
      <summary className="cursor-pointer text-[15px] font-medium text-ink-2 select-none">Key</summary>
      {children}
    </details>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Memoised marks: only the rows whose state changes re-render on a step
// ─────────────────────────────────────────────────────────────────────────────

const Link = memo(function Link({ x1, y1, x2, y2, f2p, on, bothVisited, animate, k, rTip, uid }: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  f2p: boolean;
  on: boolean;
  bothVisited: boolean;
  animate: boolean;
  k: number;
  /** Radius of a visited mark (the arrow stops at its edge). */
  rTip: number;
  uid: string;
}) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const shrink = (f2p ? rTip + 2 : 2) * k;
  const ex = x2 - (dx / len) * shrink;
  const ey = y2 - (dy / len) * shrink;
  const d = `M${x1},${y1}L${ex},${ey}`;
  const style = { stroke: on ? 'var(--accent)' : f2p ? 'var(--ink-2)' : 'var(--axis)', opacity: on ? 1 : bothVisited ? 0.85 : 0.45 };
  const sw = (on ? 1.8 : f2p ? 1.2 : 0.8) * k;
  const marker = f2p ? `url(#${uid}-${on ? 'ac' : 'a'})` : undefined;
  if (animate)
    return (
      <motion.path
        d={d}
        fill="none"
        style={style}
        strokeWidth={sw}
        markerEnd={marker}
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 0.45, ease: 'easeOut' }}
      />
    );
  return <path d={d} fill="none" style={style} strokeWidth={sw} markerEnd={marker} />;
});

const Node = memo(function Node({
  node,
  x,
  y,
  slot,
  r,
  visited,
  isSeed,
  isCur,
  isCand,
  isActive,
  isCut,
  animate,
  pulse,
  k,
  t,
  label,
  onHover,
  onFocusNode,
  onSelect,
}: {
  node: number;
  x: number;
  y: number;
  slot: number;
  r: number;
  visited: boolean;
  isSeed: boolean;
  isCur: boolean;
  isCand: boolean;
  isActive: boolean;
  isCut: boolean;
  animate: boolean;
  pulse: boolean;
  k: number;
  t: number;
  label: string | undefined;
  onHover: (n: number | null) => void;
  onFocusNode: (n: number | null) => void;
  onSelect: (n: number) => void;
}) {
  const filled = visited || isSeed;
  const body = (
    <>
      <circle r={Math.max(11 * k, 12 * t)} fill="transparent" />
      {isCand && !isCur && <circle r={(r + 3.2) * k} fill="none" style={{ stroke: 'var(--accent)' }} strokeWidth={1 * k} strokeDasharray={`${1.5 * k} ${1.5 * k}`} />}
      {(isCur || isActive) && <circle r={(r + 4.5) * k} fill="none" style={{ stroke: 'var(--accent)' }} strokeWidth={2 * k} />}
      {isCur && pulse && (
        <motion.circle
          r={(r + 4.5) * k}
          fill="none"
          style={{ stroke: 'var(--accent)' }}
          strokeWidth={1.5 * k}
          initial={{ scale: 1, opacity: 0.7 }}
          animate={{ scale: 2.1, opacity: 0 }}
          transition={{ duration: 1.1, repeat: Infinity, ease: 'easeOut' }}
        />
      )}
      <path
        d={shapePath(shapeOfSlot(slot), r * k)}
        style={{ fill: filled ? catVar(slot) : 'var(--surface)', stroke: filled ? 'var(--surface)' : catVar(slot) }}
        strokeWidth={(filled ? 1.6 : 1.1) * k}
        paintOrder="stroke"
      />
      {isSeed && (
        <>
          <circle r={(r + 4) * k} fill="none" style={{ stroke: 'var(--accent)' }} strokeWidth={1.6 * k} strokeDasharray={`${3 * k} ${2.4 * k}`} />
          <text y={0.6 * t} textAnchor="middle" dominantBaseline="central" style={{ fill: '#fff', fontSize: 13 * t, fontWeight: 700 }}>
            ?
          </text>
        </>
      )}
      {isCut && <circle r={(r + 3) * k} fill="none" style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.2 * k} strokeDasharray={`${2 * k} ${2 * k}`} />}
    </>
  );
  const focusable = visited;
  return (
    <g
      transform={`translate(${x},${y})`}
      tabIndex={focusable ? 0 : undefined}
      role={focusable ? 'button' : undefined}
      aria-label={label}
      style={{ cursor: 'pointer', outline: 'none' }}
      onPointerEnter={() => onHover(node)}
      onPointerLeave={() => onHover(null)}
      onFocus={() => onFocusNode(node)}
      onBlur={() => onFocusNode(null)}
      onClick={() => onSelect(node)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect(node);
        }
      }}
    >
      {animate ? (
        <motion.g initial={{ scale: 0.2, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.35, ease: 'easeOut' }}>
          {body}
        </motion.g>
      ) : (
        body
      )}
    </g>
  );
});
