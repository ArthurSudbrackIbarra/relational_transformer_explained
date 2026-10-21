/**
 * "RT in one picture": the whole pipeline on the live running example, one stage at a time
 * (a re-imagining of paper Fig. 1).
 *
 *   database → task row → context (BFS) → cell tokens → column / feature / neighbor / full
 *   attention → decoder → probability
 *
 * Every mark is live engine output: the dots are the rows of pdb.tableInfo, the lit rows and their
 * links are ctx.visited and the sampler's event trace, the chips are ctx.tokens, the attention
 * keys are the target's row of each mask and the weights come from the captured forward pass
 * (head-averaged), and the output is the decoder's value at the masked cell.
 *
 * The SVG is drawn 1:1 in CSS pixels from the measured width, so labels never shrink (≥ 13px).
 * Motion explains causality (rows light up in visit order, cells fly into their token slots),
 * and is instant under prefers-reduced-motion.
 */
import { Boxes, Database, Gauge, Network, Table2, Waypoints, type LucideIcon } from 'lucide-react';
import { motion, useInView } from 'motion/react';
import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react';
import { LAYER_INFO, StepPlayer, formatNumber, formatPct, tokenValueText, useStepPlayer } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { textWidth } from '@/components/viz/text';
import { cx } from '@/components/ui/cx';
import { ATTN_ORDER, SEM_TYPE_NAMES, type AttnLayer } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { usePrefersReducedMotion } from '@/styles/theme';
import type { OverviewLive } from './live';
import { rowLabel } from './live';
import { T, blockSpecs, discoveries, dotIndex, gridPath, layoutFigure, nodeAt, tokenAt, tokenXY, type ModelBox } from './pipelineLayout';
import { ChapterChips, StepText } from './pipelineText';
import { STEPS, STEP_CHAPTER_SLUGS, type StepId } from './steps';

// ─────────────────────────────────────────────────────────────────────────────
// Stage groups
// ─────────────────────────────────────────────────────────────────────────────

interface Group {
  id: string;
  label: string;
  icon: LucideIcon;
  steps: number[];
}

const GROUPS: Group[] = [
  { id: 'db', label: 'Database', icon: Database, steps: [0] },
  { id: 'task', label: 'Task row', icon: Table2, steps: [1] },
  { id: 'context', label: 'Context', icon: Waypoints, steps: [2] },
  { id: 'tokens', label: 'Tokens', icon: Boxes, steps: [3] },
  { id: 'attention', label: 'Attention', icon: Network, steps: [4, 5, 6, 7] },
  { id: 'predict', label: 'Prediction', icon: Gauge, steps: [8] },
];

const EASE = [0.25, 0.8, 0.25, 1] as const;
const halo = {
  paintOrder: 'stroke',
  stroke: 'var(--surface)',
  strokeWidth: 4,
  strokeLinejoin: 'round',
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// Stage rail
// ─────────────────────────────────────────────────────────────────────────────

function StageRail({ index, onPick }: { index: number; onPick: (step: number) => void }) {
  return (
    <ol className="mb-4 grid grid-cols-3 gap-2 !p-0 lg:grid-cols-6" style={{ listStyle: 'none' }} aria-label="Pipeline stages">
      {GROUPS.map((g, gi) => {
        const Icon = g.icon;
        const active = g.steps.includes(index);
        const done = g.steps[g.steps.length - 1] < index;
        return (
          <li key={g.id} className="!m-0 min-w-0 !p-0">
            <button
              type="button"
              onClick={() => onPick(g.steps[0])}
              aria-pressed={active}
              className={cx(
                'flex h-full w-full min-w-0 items-center gap-2 rounded-lg border px-2.5 py-2 text-left transition-colors',
                active ? 'border-accent bg-accent-soft' : 'border-border bg-surface hover:bg-surface-2',
              )}
            >
              <span
                className={cx(
                  'hidden h-7 w-7 shrink-0 items-center justify-center rounded-full text-[14px] font-semibold sm:inline-flex',
                  active ? 'bg-accent text-accent-ink' : done ? 'bg-surface-3 text-ink' : 'bg-surface-2 text-ink-2',
                )}
              >
                <Icon aria-hidden="true" size={15} strokeWidth={2.1} />
              </span>
              <span className="min-w-0">
                <span className="num block truncate text-[13.5px] leading-none text-muted">
                  {gi + 1}
                  {g.steps.length > 1 && active && ` · ${g.steps.indexOf(index) + 1}/${g.steps.length}`}
                </span>
                <span className={cx('block truncate text-[15px] leading-tight font-medium', active ? 'text-ink' : 'text-ink-2')}>{g.label}</span>
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Figure
// ─────────────────────────────────────────────────────────────────────────────

interface Hover {
  x: number;
  y: number;
  body: ReactNode;
  /** Row under the pointer (a dot, or the row of a token): cross-highlighted in both views. */
  node: number | null;
}

export function PipelineFigure({ live }: { live: OverviewLive }) {
  const { db, pdb, task, ctx, masks, slots, fr, pred, assets } = live;
  const reduced = usePrefersReducedMotion();
  const [sizeRef, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(300, Math.round(size.width || 960));

  const specs = useMemo(() => blockSpecs(db, pdb, task, slots), [db, pdb, task, slots]);
  const lay = useMemo(() => layoutFigure(W, specs, ctx.tokens.length), [W, specs, ctx.tokens.length]);
  const dots = useMemo(() => dotIndex(lay.map), [lay.map]);
  const disc = useMemo(() => discoveries(ctx), [ctx]);
  const visitedSet = useMemo(() => new Map(disc.map((d) => [d.node, d])), [disc]);

  const player = useStepPlayer(STEPS.length, { intervalMs: 4600 });
  const s = player.index;
  const stepId = STEPS[s];
  const showTask = s >= 1;
  const showBfs = s >= 2;
  const showTokens = s >= 3;
  const layer: AttnLayer | null = s >= 4 && s <= 7 ? ATTN_ORDER[s - 4] : null;
  const predicting = s === 8;

  // autoplay once, the first time the figure is well in view
  const rootRef = useRef<HTMLDivElement | null>(null);
  const inView = useInView(rootRef, { once: true, amount: 0.2 });
  const started = useRef(false);
  useEffect(() => {
    if (inView && !reduced && !started.current) {
      started.current = true;
      player.setPlaying(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, reduced]);

  const [attnBlock, setAttnBlock] = useState(0);
  const nBlocks = fr?.blocks.length ?? 4;

  const S = ctx.tokens.length;
  const q = ctx.targetPos;
  const half = lay.map.dot / 2;
  const mapY = lay.mapY;
  const seedPos = dots.get(ctx.params.seedNode);
  const entityPos = live.entityNode !== null ? dots.get(live.entityNode) : undefined;

  // ── mask rows of the target, and head-averaged weights of the chosen block ──
  const allowed = useMemo(() => {
    const out = {} as Record<AttnLayer, Set<number>>;
    for (const l of ATTN_ORDER) {
      const set = new Set<number>();
      if (q >= 0) for (let k = 0; k < S; k++) if (masks[l][q * S + k]) set.add(k);
      out[l] = set;
    }
    return out;
  }, [masks, q, S]);

  const weights = useMemo(() => {
    if (!fr || !layer || q < 0) return null;
    const cap = fr.blocks[Math.min(attnBlock, fr.blocks.length - 1)]?.attn[layer];
    if (!cap || cap.skipped) return null;
    const H = cap.probs.length / (S * S);
    const w = new Float32Array(S);
    for (let h = 0; h < H; h++) for (let k = 0; k < S; k++) w[k] += cap.probs[h * S * S + q * S + k] / H;
    return w;
  }, [fr, layer, attnBlock, q, S]);

  const topKeys = useMemo(() => {
    if (!layer) return [] as { k: number; w: number }[];
    const keys = [...allowed[layer]].filter((k) => k !== q);
    const list = keys.map((k) => ({
      k,
      w: weights ? weights[k] : 1 / Math.max(1, keys.length),
    }));
    list.sort((a, b) => b.w - a.w);
    return list.slice(0, 14);
  }, [layer, allowed, weights, q]);
  const wMax = useMemo(() => {
    if (!layer || !weights) return 1;
    let m = 0;
    for (const k of allowed[layer]) if (k !== q) m = Math.max(m, weights[k]);
    return m || 1;
  }, [layer, weights, allowed, q]);

  // ── tooltip ──
  const [hover, setHover] = useState<Hover | null>(null);
  const onMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    if (showTokens) {
      const i = tokenAt(lay.tokens, S, x, y);
      if (i !== null) {
        const t = ctx.tokens[i];
        if (t.isPadding) {
          setHover({
            x,
            y,
            node: null,
            body: <span>Padding (position {i}): an empty slot, masked out of every attention layer.</span>,
          });
          return;
        }
        const wk = layer && weights && allowed[layer].has(i) ? weights[i] : null;
        setHover({
          x,
          y,
          node: t.nodeIdx,
          body: (
            <>
              <div className="font-mono font-semibold text-ink">
                {t.tableName}.{t.colName}
              </div>
              <div className="text-ink-2">
                {t.isTarget ? (
                  <>masked target · value hidden from the model</>
                ) : (
                  <>
                    {tokenValueText(t)} · {SEM_TYPE_NAMES[t.sem]}
                  </>
                )}
              </div>
              <div className="text-muted">
                position {i} · {rowLabel(db, pdb, t.nodeIdx).short}
              </div>
              {layer && i !== q && (
                <div className="mt-0.5 text-ink">
                  {allowed[layer].has(i)
                    ? wk !== null
                      ? `weight ${formatPct(wk, 1)} (block ${attnBlock + 1}, head mean)`
                      : 'visible to the masked cell'
                    : 'blocked for the masked cell'}
                </div>
              )}
            </>
          ),
        });
        return;
      }
    }
    const node = nodeAt(lay.map, x, y - mapY);
    if (node !== null) {
      const n = pdb.nodes[node];
      if (n.isTaskNode && !showTask) {
        setHover(null);
        return;
      }
      const d = visitedSet.get(node);
      setHover({
        x,
        y,
        node,
        body: (
          <>
            <div className="font-semibold text-ink">{rowLabel(db, pdb, node).long}</div>
            <div className="text-ink-2">
              {n.cells.length} feature {n.cells.length === 1 ? 'cell' : 'cells'} · {n.f2pNbrIdxs.length}{' '}
              {n.f2pNbrIdxs.length === 1 ? 'parent' : 'parents'}
            </div>
            {showBfs && (
              <div className="text-muted">
                {d
                  ? `in the context: visited ${d.order + 1} of ${disc.length}, depth ${d.depth}${showTokens ? `, ${n.cells.length === 1 ? 'its token is' : 'its tokens are'} outlined below` : ''}`
                  : 'not in the context'}
              </div>
            )}
          </>
        ),
      });
      return;
    }
    setHover(null);
  };

  // ── stagger timings (only when the step is the one that introduces the motion) ──
  const visitStagger = Math.min(70, 2400 / Math.max(1, disc.length));
  const tokenStagger = Math.min(16, 1700 / Math.max(1, ctx.numReal));

  const center = (p: { x: number; y: number } | undefined) => (p ? { x: p.x + half, y: p.y + half + mapY } : null);

  // ── model column helpers ──
  const m = lay.model;
  const activeEmbed = stepId === 'tokens';
  const activeBlock = layer !== null;
  const activeOut = predicting;
  const modelDim = s < 3 ? 0.5 : 1;

  const box = (b: ModelBox, active: boolean, key: string, children: ReactNode, dashed = false) => (
    <g key={key}>
      <rect
        x={b.x + 0.5}
        y={b.y + 0.5}
        width={b.w - 1}
        height={b.h - 1}
        rx={8}
        strokeWidth={active ? 2 : 1}
        style={{
          fill: active ? 'var(--accent-soft)' : 'var(--surface)',
          stroke: active ? 'var(--accent)' : 'var(--border-strong)',
          transition: reduced ? undefined : 'stroke-width .3s, fill .3s, stroke .3s',
        }}
        strokeDasharray={dashed ? '5 4' : undefined}
      />
      {children}
    </g>
  );
  /** A right-aligned note inside a model box, only if it fits next to the left label. */
  const boxNote = (b: ModelBox, left: string, leftPx: number, note: string) => {
    const room = b.w - 14 - textWidth(left, leftPx, { weight: 600 }) - 24;
    if (textWidth(note, T.boxSmall) > room) return null;
    return (
      <text x={b.x + b.w - 12} y={b.y + b.h / 2} dominantBaseline="central" textAnchor="end" style={{ fill: 'var(--ink-2)', fontSize: T.boxSmall }}>
        {note}
      </text>
    );
  };

  const layerCount = (l: AttnLayer) => allowed[l].size;
  const prob = pred?.taskKind === 'binary' ? (pred.probability ?? null) : null;

  const tokenCenter = (i: number) => {
    const p = tokenXY(lay.tokens, i);
    return { x: p.x + lay.tokens.chip / 2, y: p.y + lay.tokens.chip / 2 };
  };
  const qc = q >= 0 ? tokenCenter(q) : null;

  const ariaStep: Record<StepId, string> = {
    db: 'The database: one dot per row of every table.',
    task: 'The task row: the seed row of the task table, linked to its entity.',
    context: 'The context: rows visited by the sampler light up in visit order.',
    tokens: 'The tokens: every feature cell of the visited rows moves into the context window.',
    col: 'Column attention: the keys the masked cell may attend to.',
    feat: 'Feature attention: the keys the masked cell may attend to.',
    nbr: 'Neighbor attention: the keys the masked cell may attend to.',
    full: 'Full attention: the keys the masked cell may attend to.',
    predict: 'The prediction read at the masked cell.',
  };

  return (
    <div ref={rootRef}>
      <StageRail
        index={s}
        onPick={(i) => {
          player.setPlaying(false);
          player.setIndex(i);
        }}
      />
      <div ref={sizeRef} className="relative w-full">
        <svg
          width={W}
          height={lay.H}
          viewBox={`0 0 ${W} ${lay.H}`}
          className="block max-w-full select-none"
          role="img"
          aria-label={`RT pipeline on the live example. ${ariaStep[stepId]}`}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
        >
          <defs>
            <marker id="ov-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto" markerUnits="userSpaceOnUse">
              <path d="M0,0.8 L7.4,4 L0,7.2 Z" style={{ fill: 'var(--ink-2)' }} />
            </marker>
            <marker
              id="ov-arrow-accent"
              viewBox="0 0 8 8"
              refX="7"
              refY="4"
              markerWidth="8"
              markerHeight="8"
              orient="auto"
              markerUnits="userSpaceOnUse"
            >
              <path d="M0,0.8 L7.4,4 L0,7.2 Z" style={{ fill: 'var(--accent)' }} />
            </marker>
          </defs>

          {/* ── band titles ── */}
          <text x={0} y={lay.mapLabelY} style={{ fill: 'var(--ink-2)', fontSize: T.band, fontWeight: 600 }}>
            {db.title.split(' ')[0]} · one dot per row
          </text>
          <text x={0} y={lay.tokensLabelY} style={{ fill: 'var(--ink-2)', fontSize: T.band, fontWeight: 600 }}>
            Context window · L = {ctx.params.L} tokens
          </text>
          <text x={m.x} y={lay.modelLabelY} style={{ fill: 'var(--ink-2)', fontSize: T.band, fontWeight: 600 }}>
            Mini RT · {assets.weightsKind === 'init' ? 'untrained weights' : 'zero-shot checkpoint'}
          </text>

          {/* ── database map ── */}
          <g transform={`translate(0 ${mapY})`}>
            {lay.map.blocks.map((b) => (
              <motion.g key={b.key} initial={false} animate={{ opacity: b.isTask && !showTask ? 0 : 1 }} transition={{ duration: reduced ? 0 : 0.5 }}>
                <rect x={b.x} y={b.y + 5} width={10} height={10} rx={2.5} style={{ fill: catVar(b.slot) }} />
                <text
                  x={b.x + 16}
                  y={b.y + 14}
                  className="font-mono"
                  style={{
                    fill: 'var(--ink)',
                    fontSize: T.title,
                    fontWeight: 600,
                  }}
                >
                  {b.title}
                  <tspan
                    className="font-sans"
                    dx={6}
                    style={{
                      fill: 'var(--muted)',
                      fontSize: T.count,
                      fontWeight: 400,
                    }}
                  >
                    {b.count.toLocaleString('en-US')}
                  </tspan>
                </text>
                {b.grids.map((g) => (
                  <g key={g.key}>
                    {g.label && (
                      <text x={g.x} y={g.y - 6} style={{ fill: 'var(--ink-2)', fontSize: T.sub }}>
                        {g.label}
                        <tspan dx={4} style={{ fill: 'var(--muted)' }}>
                          {g.nodes.length}
                        </tspan>
                      </text>
                    )}
                    <motion.path
                      d={gridPath(g, lay.map.pitch, lay.map.dot)}
                      initial={false}
                      animate={{ opacity: showBfs ? 0.2 : 0.85 }}
                      transition={{ duration: reduced ? 0 : 0.5 }}
                      style={{ fill: catVar(b.slot) }}
                    />
                  </g>
                ))}
              </motion.g>
            ))}
          </g>

          {/* F→P link from the seed row to its entity (task step) */}
          {seedPos &&
            entityPos &&
            (() => {
              const a = center(seedPos)!;
              const b = center(entityPos)!;
              const mx = (a.x + b.x) / 2;
              const my = Math.min(a.y, b.y) - 26;
              return (
                <motion.path
                  d={`M${a.x},${a.y - half - 2} Q${mx},${my} ${b.x},${b.y - half - 3}`}
                  fill="none"
                  style={{ stroke: 'var(--accent)' }}
                  strokeWidth={2}
                  markerEnd="url(#ov-arrow-accent)"
                  initial={false}
                  animate={{
                    pathLength: s === 1 ? 1 : 0,
                    opacity: s === 1 ? 1 : 0,
                  }}
                  transition={{
                    duration: reduced ? 0 : 0.7,
                    delay: reduced || s !== 1 ? 0 : 0.35,
                    ease: EASE,
                  }}
                />
              );
            })()}

          {/* BFS discovery links (context step and after) */}
          <g>
            {disc.map((d, i) => {
              if (d.from === null) return null;
              const a = center(dots.get(d.from));
              const b = center(dots.get(d.node));
              if (!a || !b) return null;
              const mx = (a.x + b.x) / 2 + (b.y - a.y) * 0.12;
              const my = (a.y + b.y) / 2 - (b.x - a.x) * 0.12;
              return (
                <motion.path
                  key={`e-${d.node}`}
                  d={`M${a.x},${a.y} Q${mx},${my} ${b.x},${b.y}`}
                  fill="none"
                  style={{ stroke: 'var(--ink-2)' }}
                  strokeWidth={1.3}
                  strokeDasharray={d.via === 'child' ? '4 3' : undefined}
                  initial={false}
                  animate={{
                    pathLength: showBfs ? 1 : 0,
                    opacity: showBfs ? (s >= 4 ? 0.16 : s === 3 ? 0.28 : 0.6) : 0,
                  }}
                  transition={{
                    duration: reduced ? 0 : 0.45,
                    delay: reduced || s !== 2 ? 0 : (i * visitStagger) / 1000,
                  }}
                />
              );
            })}
          </g>

          {/* visited rows, lit in visit order */}
          <g>
            {disc.map((d, i) => {
              const p = dots.get(d.node);
              if (!p) return null;
              const slot = slots[pdb.nodes[d.node].tableName] ?? 0;
              const size = lay.map.dot + 3;
              return (
                <motion.rect
                  key={`v-${d.node}`}
                  x={p.x - 1.5}
                  y={p.y - 1.5 + mapY}
                  width={size}
                  height={size}
                  rx={1.5}
                  style={{ fill: catVar(slot), stroke: 'var(--surface)' }}
                  strokeWidth={1}
                  initial={false}
                  animate={{ opacity: showBfs ? (s >= 4 ? 0.6 : 1) : 0 }}
                  transition={{
                    duration: reduced ? 0 : 0.25,
                    delay: reduced || s !== 2 ? 0 : (i * visitStagger) / 1000,
                  }}
                />
              );
            })}
          </g>

          {/* seed ring + labels */}
          {seedPos &&
            showTask &&
            (() => {
              const c = center(seedPos)!;
              return (
                <g>
                  <rect
                    x={c.x - half - 4}
                    y={c.y - half - 4}
                    width={lay.map.dot + 8}
                    height={lay.map.dot + 8}
                    rx={3}
                    fill="none"
                    style={{ stroke: 'var(--accent)' }}
                    strokeWidth={2}
                  />
                  {s === 1 && !reduced && (
                    <motion.rect
                      x={c.x - half - 4}
                      y={c.y - half - 4}
                      width={lay.map.dot + 8}
                      height={lay.map.dot + 8}
                      rx={3}
                      fill="none"
                      style={{
                        stroke: 'var(--accent)',
                        transformBox: 'fill-box',
                        transformOrigin: 'center',
                      }}
                      strokeWidth={2}
                      initial={{ scale: 1, opacity: 0.8 }}
                      animate={{ scale: 2.4, opacity: 0 }}
                      transition={{
                        duration: 1.4,
                        repeat: Infinity,
                        ease: 'easeOut',
                      }}
                    />
                  )}
                  <text
                    x={Math.min(W - 4, c.x + half + 9)}
                    y={c.y + 5}
                    textAnchor={c.x + half + 60 > W ? 'end' : 'start'}
                    style={{
                      fill: 'var(--accent-text)',
                      fontSize: T.sub,
                      fontWeight: 600,
                      ...halo,
                    }}
                  >
                    seed
                  </text>
                </g>
              );
            })()}
          {entityPos &&
            s === 1 &&
            (() => {
              const c = center(entityPos)!;
              return (
                <g>
                  <rect
                    x={c.x - half - 4}
                    y={c.y - half - 4}
                    width={lay.map.dot + 8}
                    height={lay.map.dot + 8}
                    rx={3}
                    fill="none"
                    style={{ stroke: 'var(--ink)' }}
                    strokeWidth={1.6}
                  />
                  <text
                    x={Math.min(
                      Math.max(
                        c.x,
                        textWidth(live.entityShort ?? '', T.sub, {
                          weight: 600,
                        }) /
                          2 +
                          2,
                      ),
                      W -
                        textWidth(live.entityShort ?? '', T.sub, {
                          weight: 600,
                        }) /
                          2 -
                        2,
                    )}
                    y={c.y + half + 20}
                    textAnchor="middle"
                    style={{
                      fill: 'var(--ink)',
                      fontSize: T.sub,
                      fontWeight: 600,
                      ...halo,
                    }}
                  >
                    {live.entityShort}
                  </text>
                </g>
              );
            })()}

          {/* ── token slots ── */}
          <g>
            {ctx.tokens.map((t, i) => {
              const p = tokenXY(lay.tokens, i);
              return (
                <rect
                  key={`slot-${i}`}
                  x={p.x + 0.5}
                  y={p.y + 0.5}
                  width={lay.tokens.chip - 1}
                  height={lay.tokens.chip - 1}
                  rx={4}
                  fill="none"
                  style={{
                    stroke: 'var(--axis)',
                    opacity: t.isPadding ? 0.9 : showTokens ? 0 : 0.9,
                  }}
                  strokeWidth={1}
                  strokeDasharray="2.5 2.5"
                />
              );
            })}
          </g>

          {/* attention links from the top keys to the masked cell */}
          {layer && qc && (
            <g>
              {topKeys.map(({ k, w }, i) => {
                const a = tokenCenter(k);
                const rel = w / wMax;
                const dx = qc.x - a.x;
                const dy = qc.y - a.y;
                const bend = Math.min(40, Math.hypot(dx, dy) * 0.18);
                const len = Math.hypot(dx, dy) || 1;
                const mx = (a.x + qc.x) / 2 - (dy / len) * bend;
                const my = (a.y + qc.y) / 2 + (dx / len) * bend;
                return (
                  <motion.path
                    key={`${layer}-${k}`}
                    d={`M${a.x},${a.y} Q${mx},${my} ${qc.x},${qc.y}`}
                    fill="none"
                    style={{ stroke: 'var(--accent)' }}
                    strokeLinecap="round"
                    strokeWidth={1 + 2.2 * rel}
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 0.25 + 0.65 * rel }}
                    transition={{
                      duration: reduced ? 0 : 0.6,
                      delay: reduced ? 0 : 0.15 + i * 0.04,
                      ease: EASE,
                    }}
                  />
                );
              })}
            </g>
          )}

          {/* ── tokens (cells fly from their row's dot into their slot) ── */}
          <g>
            {ctx.tokens.map((t, i) => {
              if (t.isPadding) return null;
              const dest = tokenXY(lay.tokens, i);
              const origin = dots.get(t.nodeIdx);
              const o = origin ? { x: origin.x, y: origin.y + mapY } : dest;
              const isT = t.isTarget;
              const slot = slots[t.tableName] ?? 0;
              let opacity = 1;
              let fillOpacity = 0.3;
              if (layer) {
                const ok = allowed[layer].has(i);
                opacity = isT ? 1 : ok ? 1 : 0.16;
                if (ok && !isT) fillOpacity = weights ? 0.18 + 0.72 * (weights[i] / wMax) : 0.55;
              } else if (predicting) {
                opacity = isT ? 1 : 0.28;
              }
              const shown = showTokens;
              const c = lay.tokens.chip;
              const delay = reduced || s !== 3 ? 0 : (i * tokenStagger) / 1000;
              return (
                <motion.g
                  key={`tok-${i}`}
                  initial={false}
                  animate={{
                    x: shown ? dest.x : o.x,
                    y: shown ? dest.y : o.y,
                    opacity: shown ? opacity : 0,
                  }}
                  transition={{
                    x: { duration: reduced ? 0 : 0.85, delay, ease: EASE },
                    y: { duration: reduced ? 0 : 0.85, delay, ease: EASE },
                    opacity: {
                      duration: reduced ? 0 : shown && s === 3 ? 0.15 : 0.35,
                      delay: shown && s === 3 ? delay : 0,
                    },
                  }}
                >
                  {isT && <rect width={c} height={c} rx={5} style={{ fill: 'var(--surface)', opacity: shown ? 1 : 0 }} />}
                  <motion.rect
                    initial={false}
                    animate={{
                      width: shown ? c : lay.map.dot,
                      height: shown ? c : lay.map.dot,
                    }}
                    transition={{
                      duration: reduced ? 0 : 0.85,
                      delay,
                      ease: EASE,
                    }}
                    rx={isT ? 5 : 4}
                    style={
                      isT
                        ? {
                            fill: 'var(--accent-soft)',
                            stroke: 'var(--accent)',
                          }
                        : {
                            fill: catVar(slot),
                            fillOpacity,
                            stroke: catVar(slot),
                            strokeOpacity: 0.95,
                          }
                    }
                    strokeWidth={isT ? 2 : 1.2}
                    strokeDasharray={isT ? '3.5 2.5' : undefined}
                  />
                  {isT && shown && (
                    <text
                      x={c / 2}
                      y={c / 2 + 1}
                      textAnchor="middle"
                      dominantBaseline="central"
                      style={{
                        fill: 'var(--accent-text)',
                        fontSize: 15,
                        fontWeight: 800,
                      }}
                    >
                      ?
                    </text>
                  )}
                </motion.g>
              );
            })}
          </g>

          {/* cross-highlight: the hovered row's dot and its tokens */}
          {hover?.node != null &&
            (() => {
              const hn = hover.node;
              const dp = dots.get(hn);
              const c = lay.tokens.chip;
              return (
                <g pointerEvents="none">
                  {dp && (
                    <rect
                      x={dp.x - 3.5}
                      y={dp.y - 3.5 + mapY}
                      width={lay.map.dot + 7}
                      height={lay.map.dot + 7}
                      rx={3}
                      fill="none"
                      style={{ stroke: 'var(--ink)' }}
                      strokeWidth={1.8}
                    />
                  )}
                  {showTokens &&
                    ctx.tokens.map((t, i) => {
                      if (t.isPadding || t.nodeIdx !== hn) return null;
                      const p = tokenXY(lay.tokens, i);
                      return (
                        <rect
                          key={i}
                          x={p.x - 2.5}
                          y={p.y - 2.5}
                          width={c + 5}
                          height={c + 5}
                          rx={6}
                          fill="none"
                          style={{ stroke: 'var(--ink)' }}
                          strokeWidth={1.8}
                        />
                      );
                    })}
                </g>
              );
            })()}

          {/* empty neighbor row note */}
          {layer &&
            allowed[layer].size === 0 &&
            qc &&
            (() => {
              const x0 = qc.x + lay.tokens.chip / 2 + 10;
              const room = (lay.wide ? lay.tokens.x + lay.tokens.w : W) - x0;
              const long = 'no keys: this layer adds 0 to the masked cell';
              const text = textWidth(long, T.sub, { weight: 600 }) <= room ? long : 'no keys: adds 0 here';
              return (
                <text
                  x={x0}
                  y={qc.y}
                  dominantBaseline="central"
                  style={{
                    fill: 'var(--ink)',
                    fontSize: T.sub,
                    fontWeight: 600,
                    ...halo,
                  }}
                >
                  {text}
                </text>
              );
            })()}

          {/* ── tokens → model connector ── */}
          {(() => {
            const tg = lay.tokens;
            if (lay.wide) {
              const x1 = tg.x + tg.w + 8;
              const y1 = tg.y + tg.h / 2;
              const x2 = m.embed.x - 6;
              const y2 = m.embed.y + m.embed.h / 2;
              return (
                <motion.path
                  d={`M${x1},${y1} C${x1 + 26},${y1} ${x2 - 30},${y2} ${x2},${y2}`}
                  fill="none"
                  style={{ stroke: 'var(--ink-2)' }}
                  strokeWidth={1.5}
                  markerEnd="url(#ov-arrow)"
                  initial={false}
                  animate={{ opacity: showTokens ? 0.9 : 0.25 }}
                  transition={{ duration: reduced ? 0 : 0.4 }}
                />
              );
            }
            const x = m.x + 24;
            return (
              <motion.path
                d={`M${x},${tg.y + tg.h + 8} L${x},${m.embed.y - 6}`}
                fill="none"
                style={{ stroke: 'var(--ink-2)' }}
                strokeWidth={1.5}
                markerEnd="url(#ov-arrow)"
                initial={false}
                animate={{ opacity: showTokens ? 0.9 : 0.25 }}
                transition={{ duration: reduced ? 0 : 0.4 }}
              />
            );
          })()}

          {/* ── model column ── */}
          <motion.g initial={false} animate={{ opacity: modelDim }} transition={{ duration: reduced ? 0 : 0.4 }}>
            {m.arrows.map((d, i) => (
              <path key={i} d={d} fill="none" style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.4} markerEnd="url(#ov-arrow)" />
            ))}
            {box(
              m.embed,
              activeEmbed,
              'embed',
              <>
                <text
                  x={m.embed.x + 14}
                  y={m.embed.y + m.embed.h / 2}
                  dominantBaseline="central"
                  style={{
                    fill: 'var(--ink)',
                    fontSize: T.box,
                    fontWeight: 600,
                  }}
                >
                  Cell embedding
                </text>
                {boxNote(m.embed, 'Cell embedding', T.box, 'value + column name')}
              </>,
            )}
            {box(
              m.block,
              false,
              'block',
              <>
                <text
                  x={m.block.x + 12}
                  y={m.block.y + 20}
                  style={{
                    fill: 'var(--ink)',
                    fontSize: T.box,
                    fontWeight: 600,
                  }}
                >
                  RT block × {nBlocks}
                  <tspan
                    dx={8}
                    style={{
                      fill: 'var(--muted)',
                      fontSize: T.boxSmall,
                      fontWeight: 400,
                    }}
                  >
                    (paper: × 12)
                  </tspan>
                </text>
              </>,
              true,
            )}
            {[...ATTN_ORDER, 'mlp' as const].map((l, i) => {
              const b = m.layers[i];
              const isAttn = l !== 'mlp';
              const active = isAttn ? layer === l : false;
              const Icon = isAttn ? LAYER_INFO[l].icon : Boxes;
              const name = isAttn ? LAYER_INFO[l].name : 'MLP';
              const count = isAttn ? `${layerCount(l)} ${layerCount(l) === 1 ? 'key' : 'keys'}` : 'per token';
              if (m.compact)
                return box(
                  b,
                  active,
                  `layer-${l}`,
                  <>
                    <text
                      x={b.x + b.w / 2}
                      y={b.y + 17}
                      textAnchor="middle"
                      dominantBaseline="central"
                      style={{
                        fill: 'var(--ink)',
                        fontSize: T.boxSmall,
                        fontWeight: active ? 650 : 500,
                      }}
                    >
                      {name}
                    </text>
                    <text
                      x={b.x + b.w / 2}
                      y={b.y + 33}
                      textAnchor="middle"
                      dominantBaseline="central"
                      style={{ fill: 'var(--ink-2)', fontSize: T.boxSmall }}
                    >
                      {count}
                    </text>
                  </>,
                );
              return box(
                b,
                active,
                `layer-${l}`,
                <>
                  <Icon
                    x={b.x + 10}
                    y={b.y + b.h / 2 - 8}
                    width={16}
                    height={16}
                    strokeWidth={2}
                    style={{ color: 'var(--ink-2)' }}
                    aria-hidden="true"
                  />
                  <text
                    x={b.x + 34}
                    y={b.y + b.h / 2}
                    dominantBaseline="central"
                    style={{
                      fill: 'var(--ink)',
                      fontSize: T.boxSmall,
                      fontWeight: active ? 650 : 500,
                    }}
                  >
                    {isAttn ? `${name} attention` : 'MLP (SwiGLU)'}
                  </text>
                  <text
                    x={b.x + b.w - 10}
                    y={b.y + b.h / 2}
                    dominantBaseline="central"
                    textAnchor="end"
                    style={{ fill: 'var(--ink-2)', fontSize: T.boxSmall }}
                  >
                    {isAttn ? count : 'each token'}
                  </text>
                </>,
              );
            })}
            {box(
              m.norm,
              activeOut,
              'norm',
              <text
                x={m.norm.x + 14}
                y={m.norm.y + m.norm.h / 2}
                dominantBaseline="central"
                style={{
                  fill: 'var(--ink)',
                  fontSize: m.compact ? T.boxSmall : T.box,
                  fontWeight: 600,
                }}
              >
                Final RMSNorm
              </text>,
            )}
            {box(
              m.decoder,
              activeOut,
              'decoder',
              <>
                <text
                  x={m.decoder.x + 14}
                  y={m.decoder.y + m.decoder.h / 2}
                  dominantBaseline="central"
                  style={{
                    fill: 'var(--ink)',
                    fontSize: m.compact ? T.boxSmall : T.box,
                    fontWeight: 600,
                  }}
                >
                  {task.targetType === 'binary' ? 'Boolean decoder' : 'Number decoder'}
                </text>
                {boxNote(m.decoder, task.targetType === 'binary' ? 'Boolean decoder' : 'Number decoder', T.box, 'read at the ? cell')}
              </>,
            )}
            {/* output */}
            <g>
              <rect
                x={m.out.x + 0.5}
                y={m.out.y + 0.5}
                width={m.out.w - 1}
                height={m.out.h - 1}
                rx={10}
                style={{
                  fill: 'var(--surface-2)',
                  stroke: activeOut ? 'var(--accent)' : 'var(--border)',
                }}
                strokeWidth={activeOut ? 2 : 1}
              />
              {pred ? (
                pred.taskKind === 'binary' && prob !== null ? (
                  <>
                    <text x={m.out.x + 14} y={m.out.y + 22} style={{ fill: 'var(--ink-2)', fontSize: T.boxSmall }}>
                      logit {formatNumber(pred.output, 3)} → σ →
                      <tspan
                        dx={6}
                        style={{
                          fill: 'var(--ink)',
                          fontSize: T.value,
                          fontWeight: 700,
                        }}
                      >
                        P({task.targetCol}) = {formatPct(prob)}
                      </tspan>
                    </text>
                    <rect x={m.out.x + 14} y={m.out.y + 34} width={m.out.w - 28} height={10} rx={5} style={{ fill: 'var(--grid)' }} />
                    <motion.rect
                      x={m.out.x + 14}
                      y={m.out.y + 34}
                      height={10}
                      rx={5}
                      style={{ fill: 'var(--accent)' }}
                      initial={false}
                      animate={{
                        width: (m.out.w - 28) * (predicting ? prob : 0),
                      }}
                      transition={{
                        duration: reduced ? 0 : 0.9,
                        delay: reduced ? 0 : 0.2,
                        ease: EASE,
                      }}
                    />
                    <line
                      x1={m.out.x + 14 + (m.out.w - 28) / 2}
                      x2={m.out.x + 14 + (m.out.w - 28) / 2}
                      y1={m.out.y + 31}
                      y2={m.out.y + 47}
                      style={{ stroke: 'var(--ink-2)' }}
                      strokeWidth={1}
                    />
                    <text x={m.out.x + 14} y={m.out.y + 66} style={{ fill: 'var(--ink-2)', fontSize: T.boxSmall }}>
                      hidden true label:
                      <tspan dx={5} className="font-mono" style={{ fill: 'var(--ink)', fontWeight: 600 }}>
                        {predicting ? String(pred.label.raw) : '•••'}
                      </tspan>
                    </text>
                  </>
                ) : (
                  <>
                    <text x={m.out.x + 14} y={m.out.y + 24} style={{ fill: 'var(--ink-2)', fontSize: T.boxSmall }}>
                      z {formatNumber(pred.output, 3)} → original units
                    </text>
                    <text
                      x={m.out.x + 14}
                      y={m.out.y + 46}
                      style={{
                        fill: 'var(--ink)',
                        fontSize: T.value,
                        fontWeight: 700,
                      }}
                    >
                      {task.targetCol} ≈ {formatNumber(pred.value ?? NaN, 2)}
                    </text>
                    <text x={m.out.x + 14} y={m.out.y + 66} style={{ fill: 'var(--ink-2)', fontSize: T.boxSmall }}>
                      hidden true value:
                      <tspan dx={5} className="font-mono" style={{ fill: 'var(--ink)', fontWeight: 600 }}>
                        {predicting ? formatNumber(Number(pred.label.raw), 2) : '•••'}
                      </tspan>
                    </text>
                  </>
                )
              ) : (
                <text x={m.out.x + 14} y={m.out.y + m.out.h / 2} dominantBaseline="central" style={{ fill: 'var(--ink-2)', fontSize: T.boxSmall }}>
                  {assets.status === 'error' ? 'weights failed to load' : 'loading weights and embeddings…'}
                </text>
              )}
            </g>
          </motion.g>
        </svg>

        {hover && (
          <div
            className="pointer-events-none absolute z-10 max-w-[300px] rounded-lg border border-border bg-surface px-3 py-2 text-[14.5px] leading-snug shadow-lg"
            style={{
              left: Math.min(Math.max(8, hover.x + 14), Math.max(8, W - 300)),
              top: hover.y + 18,
            }}
          >
            {hover.body}
          </div>
        )}
      </div>

      <StepPlayer
        {...player}
        unit="step"
        label="Pipeline steps"
        className="mt-4"
        describe={(i) => (
          <StepText
            step={STEPS[i]}
            live={live}
            disc={disc}
            allowed={allowed}
            attnBlock={attnBlock}
            nBlocks={nBlocks}
            setAttnBlock={setAttnBlock}
            hasWeights={!!weights}
            topKeys={topKeys}
          />
        )}
      />
      <ChapterChips slugs={STEP_CHAPTER_SLUGS[stepId]} className="mt-3" />
    </div>
  );
}
