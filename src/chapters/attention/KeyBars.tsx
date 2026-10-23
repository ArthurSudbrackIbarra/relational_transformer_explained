/**
 * KeyBars: one query's row of an attention head, drawn over every key of the context and animated
 * through three stages of the computation (rt/model.py:43 → FlexAttention; spec §6.1):
 *
 *   'scores'  — the scaled score q·k/√d_head for every key (signed bars around 0), before masking;
 *   'mask'    — keys the layer's mask forbids are pushed to −∞ (hatched, dropped below the axis);
 *   'softmax' — the masked softmax: blocked keys get exactly 0 and collapse away; the allowed keys
 *               spread out and grow to their probabilities, which sum to 1 (or all stay 0).
 *
 * Every number is the engine's capture (`AttnCapture.scores` / `.probs` / the mask row). Keys are
 * grouped by table (task splits share one group, as they share table_name_idx).
 */
import { motion, useReducedMotion } from 'motion/react';
import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { cx, sequentialCss } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import type { Token } from '@/engine/types';
import { TOKENS, catVar } from '@/styles/palette';
import { useReadingMode, useResolvedTheme } from '@/styles/theme';
import { pctw, sgn, type KeyGroup } from './utils';

export type BarStage = 'scores' | 'mask' | 'softmax';

export interface KeyBarsProps {
  tokens: readonly Token[];
  groups: KeyGroup[];
  order: number[];
  scores: ArrayLike<number>;
  probs: ArrayLike<number>;
  allowed: ArrayLike<number>;
  stage: BarStage;
  query: number;
  inspected: number | null;
  labels: string[];
  onInspect: (pos: number) => void;
  height?: number;
  className?: string;
}

const M_LEFT = 46;
const LABEL_MAX = 22;
const M_RIGHT = 6;
const M_TOP = 16;
const GROUP_GAP = 7;

export function KeyBars({ tokens, groups, order, scores, probs, allowed, stage, query, inspected, labels, onInspect, height = 230, className }: KeyBarsProps) {
  const uid = useId().replace(/:/g, '');
  const [ref, size] = useElementSize<HTMLDivElement>();
  const mode = useResolvedTheme();
  useReadingMode(); // TOKENS follow reading mode
  const reduce = useReducedMotion();
  const [hover, setHover] = useState<number | null>(null);
  const W = Math.max(280, size.width || 720);
  const plotW = W - M_LEFT - M_RIGHT;
  const n = order.length;
  const allowedOrder = useMemo(() => order.filter((k) => allowed[k]), [order, allowed]);
  const nAllowed = allowedOrder.length;
  const collapse = stage === 'softmax' && nAllowed > 0 && nAllowed < n;
  const fewLabels = collapse && nAllowed <= 40 && Math.min(34, plotW / Math.max(1, nAllowed)) >= 24; // 45° labels need ≥ 24 px per bar
  const LANE = 20; // the "−∞" lane under the plot (stage 'mask')
  // Collapsed softmax bars are labeled "key = value", so equal values (and equal weights) are easy to spot.
  const barLabel = (k: number) => truncate(`${labels[k]} = ${tokens[k]?.isTarget ? '?' : tokenValue(tokens[k])}`, LABEL_MAX);
  // Rotated (45°) key labels under the collapsed softmax bars need room: ~7.8 px per 13 px mono character.
  const maxLabel = fewLabels ? Math.max(4, ...allowedOrder.map((k) => barLabel(k).length)) : 0;
  const bottom = fewLabels ? Math.ceil(26 + maxLabel * 7.8 * 0.72) : stage === 'mask' ? 58 : 36;
  const plotH = height;
  const H = M_TOP + plotH + bottom;

  // ── x layout: spread (every key, grouped) and collapsed (allowed keys only) ──
  const spread = useMemo(() => {
    const slot = (plotW - GROUP_GAP * Math.max(0, groups.length - 1)) / Math.max(1, n);
    const x = new Map<number, number>();
    const bands: { name: string; slot: number; x0: number; x1: number }[] = [];
    let cur = M_LEFT;
    for (const g of groups) {
      const x0 = cur;
      for (const p of g.positions) {
        x.set(p, cur);
        cur += slot;
      }
      bands.push({ name: g.name, slot: g.slot, x0, x1: cur });
      cur += GROUP_GAP;
    }
    const bw = slot >= 4 ? slot * 0.74 : Math.max(0.7, slot - 0.6);
    return { x, slot, bw, bands };
  }, [groups, n, plotW]);

  const collapsed = useMemo(() => {
    const slot = Math.min(34, plotW / Math.max(1, nAllowed));
    const x0 = M_LEFT + (plotW - slot * nAllowed) / 2;
    const x = new Map<number, number>();
    allowedOrder.forEach((p, i) => x.set(p, x0 + i * slot));
    const bw = slot >= 4 ? Math.min(24, slot * 0.72) : Math.max(0.7, slot - 0.6);
    return { x, slot, bw };
  }, [allowedOrder, nAllowed, plotW]);

  // ── y scales ──
  const maxAbsScore = useMemo(() => {
    let m = 0;
    for (const k of order) m = Math.max(m, Math.abs(scores[k]));
    return m > 0 ? m * 1.06 : 1;
  }, [order, scores]);
  const maxP = useMemo(() => {
    let m = 0;
    for (const k of allowedOrder) m = Math.max(m, probs[k]);
    return m > 0 ? m * 1.08 : 1;
  }, [allowedOrder, probs]);
  const yMid = M_TOP + plotH / 2;
  const yBase = M_TOP + plotH;

  const geom = (k: number) => {
    const isAllowed = !!allowed[k];
    if (stage === 'softmax') {
      const lay = collapse ? collapsed : spread;
      const xk = lay.x.get(k);
      const visible = isAllowed && xk !== undefined;
      const x = visible ? xk! + (lay.slot - lay.bw) / 2 : (spread.x.get(k) ?? M_LEFT) + spread.slot / 2;
      const h = visible ? (probs[k] / maxP) * plotH : 0;
      return { x, y: yBase - h, w: visible ? lay.bw : 0, h, opacity: visible ? 1 : 0, blocked: !isAllowed };
    }
    const xk = (spread.x.get(k) ?? M_LEFT) + (spread.slot - spread.bw) / 2;
    const s = scores[k];
    if (stage === 'mask' && !isAllowed) {
      return { x: xk, y: yMid, w: spread.bw, h: plotH / 2 + LANE, opacity: 1, blocked: true };
    }
    const h = (Math.abs(s) / maxAbsScore) * (plotH / 2);
    return { x: xk, y: s >= 0 ? yMid - h : yMid, w: spread.bw, h, opacity: 1, blocked: false };
  };

  // Resolved colors (not CSS variables) so motion can blend a score bar into a weight bar.
  const scoreFill = TOKENS[mode].ink2;
  const probFill = sequentialCss(mode === 'dark' ? 0.55 : 0.68, mode);
  // Animate only real changes of the data or the stage. A change of width (resize, sidebar, scrollbar) moves the bars at once,
  // like the plain-SVG labels, outline and marker, so the two never drift apart.
  const prevW = useRef(W);
  const resized = prevW.current !== W;
  useEffect(() => {
    prevW.current = W;
  }, [W]);
  const transition = reduce || resized ? { duration: 0 } : { duration: 0.65, ease: [0.22, 1, 0.36, 1] as const };

  const visibleKeys = stage === 'softmax' && collapse ? allowedOrder : order;
  const hitSlot = stage === 'softmax' && collapse ? collapsed : spread;
  const shown = hover ?? inspected;

  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft' && e.key !== 'Home' && e.key !== 'End') return;
    e.preventDefault();
    const list = visibleKeys;
    if (!list.length) return;
    const i = inspected !== null ? list.indexOf(inspected) : -1;
    let j = e.key === 'ArrowRight' ? i + 1 : e.key === 'ArrowLeft' ? i - 1 : e.key === 'Home' ? 0 : list.length - 1;
    j = Math.max(0, Math.min(list.length - 1, j < 0 ? 0 : j));
    onInspect(list[j]);
  };

  const tip = shown !== null && shown >= 0 && shown < tokens.length ? tokens[shown] : null;
  const tipG = shown !== null ? geom(shown) : null;
  const empty = stage === 'softmax' && nAllowed === 0;

  const ariaSummary =
    stage === 'scores'
      ? `Scores of the query against ${n} keys, from ${sgn(-maxAbsScore / 1.06, 2)} to ${sgn(maxAbsScore / 1.06, 2)}.`
      : stage === 'mask'
        ? `${nAllowed} of ${n} keys allowed by the mask; ${n - nAllowed} set to minus infinity.`
        : nAllowed === 0
          ? 'No allowed key: every weight is 0.'
          : `Softmax weights over ${nAllowed} allowed keys; the largest is ${pctw(maxP / 1.08)}.`;

  return (
    <div ref={ref} className={cx('relative min-w-0 select-none', className)}>
      <svg
        width="100%"
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${ariaSummary} Use the left and right arrow keys to inspect keys.`}
        tabIndex={0}
        onKeyDown={onKey}
        className="block overflow-visible rounded-[4px] focus-visible:outline-2 focus-visible:outline-accent"
        onPointerLeave={() => setHover(null)}
      >
        <defs>
          <pattern id={`${uid}-hatch`} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="5" height="5" style={{ fill: 'var(--surface-2)' }} />
            <line x1="0" y1="0" x2="0" y2="5" style={{ stroke: 'var(--axis)' }} strokeWidth="2.2" />
          </pattern>
          <linearGradient id={`${uid}-fade`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="white" stopOpacity="1" />
            <stop offset="1" stopColor="white" stopOpacity="0.15" />
          </linearGradient>
          <mask id={`${uid}-fademask`} maskUnits="userSpaceOnUse" x="0" y="0" width={W} height={H}>
            <rect x="0" y={yMid} width={W} height={plotH / 2 + LANE} fill={`url(#${uid}-fade)`} />
            <rect x="0" y="0" width={W} height={yMid} fill="white" />
          </mask>
        </defs>

        {/* axis + grid */}
        {stage !== 'softmax' ? (
          <g className="text-[13px]">
            <line x1={M_LEFT} x2={W - M_RIGHT} y1={M_TOP} y2={M_TOP} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
            <line x1={M_LEFT} x2={W - M_RIGHT} y1={yBase} y2={yBase} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
            <text x={M_LEFT - 6} y={M_TOP + 3.5} textAnchor="end" className="num" style={{ fill: 'var(--muted)' }}>
              {sgn(maxAbsScore, 2)}
            </text>
            <text x={M_LEFT - 6} y={yMid + 3.5} textAnchor="end" className="num" style={{ fill: 'var(--muted)' }}>
              0
            </text>
            {stage !== 'mask' && (
              <text x={M_LEFT - 6} y={yBase + 3.5} textAnchor="end" className="num" style={{ fill: 'var(--muted)' }}>
                {sgn(-maxAbsScore, 2)}
              </text>
            )}
            {stage === 'mask' && (
              <text x={M_LEFT - 6} y={yBase + LANE - 2} textAnchor="end" className="num text-[14.5px] font-semibold" style={{ fill: 'var(--ink-2)' }}>
                −∞
              </text>
            )}
          </g>
        ) : (
          <g className="text-[13px]">
            <line x1={M_LEFT} x2={W - M_RIGHT} y1={M_TOP} y2={M_TOP} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
            <line x1={M_LEFT} x2={W - M_RIGHT} y1={M_TOP + plotH / 2} y2={M_TOP + plotH / 2} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
            <text x={M_LEFT - 6} y={M_TOP + 3.5} textAnchor="end" className="num" style={{ fill: 'var(--muted)' }}>
              {empty ? '1' : pctw(maxP)}
            </text>
            <text x={M_LEFT - 6} y={M_TOP + plotH / 2 + 3.5} textAnchor="end" className="num" style={{ fill: 'var(--muted)' }}>
              {empty ? '' : pctw(maxP / 2)}
            </text>
            <text x={M_LEFT - 6} y={yBase + 3.5} textAnchor="end" className="num" style={{ fill: 'var(--muted)' }}>
              0
            </text>
          </g>
        )}

        {/* bars: allowed and blocked keys in two groups (membership never changes with the stage) */}
        {[false, true].map((blockedGroup) => (
          <g key={String(blockedGroup)} mask={blockedGroup && stage === 'mask' ? `url(#${uid}-fademask)` : undefined}>
            {order.map((k) => {
              if (!allowed[k] !== blockedGroup) return null;
              const g = geom(k);
              const box = { x: g.x, y: g.y, width: Math.max(0, g.w), height: Math.max(0, g.h), opacity: g.opacity };
              if (!blockedGroup) {
                // Allowed keys: the gray score bar turns into a blue weight bar at the softmax stage.
                const soft = stage === 'softmax';
                return (
                  <motion.rect
                    key={k}
                    initial={false}
                    animate={{ ...box, fill: soft ? probFill : scoreFill, fillOpacity: soft ? 1 : 0.62 }}
                    transition={transition}
                    rx={g.w >= 6 ? 1.5 : 0}
                  />
                );
              }
              return (
                <motion.rect
                  key={k}
                  initial={false}
                  animate={box}
                  transition={transition}
                  style={{ fill: g.blocked ? `url(#${uid}-hatch)` : scoreFill, fillOpacity: g.blocked ? 1 : 0.62 }}
                  rx={g.w >= 6 ? 1.5 : 0}
                />
              );
            })}
          </g>
        ))}

        {/* even split 1/n over the allowed keys (softmax stage) */}
        {stage === 'softmax' && nAllowed > 1 && (() => {
          const even = 1 / nAllowed;
          if (even > maxP) return null;
          const y = yBase - (even / maxP) * plotH;
          return (
            <g pointerEvents="none">
              <line x1={M_LEFT} x2={W - M_RIGHT} y1={y} y2={y} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.2} strokeDasharray="5 4" />
              <text x={W - M_RIGHT - 2} y={y - 5} textAnchor="end" className="num text-[13px]" style={{ fill: 'var(--ink-2)', paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 3 }}>
                even split 1/{nAllowed} = {pctw(even)}
              </text>
            </g>
          );
        })()}

        {/* zero line */}
        <line
          x1={M_LEFT}
          x2={W - M_RIGHT}
          y1={stage === 'softmax' ? yBase : yMid}
          y2={stage === 'softmax' ? yBase : yMid}
          style={{ stroke: 'var(--axis)' }}
          strokeWidth={1}
        />

        {/* inspected key */}
        {inspected !== null && (() => {
          const g = geom(inspected);
          if (g.opacity === 0) return null;
          const top = Math.min(g.y, stage === 'softmax' ? yBase - 2 : yMid - 2);
          const bot = Math.max(g.y + g.h, stage === 'softmax' ? yBase : yMid + 2);
          return (
            <rect
              x={g.x - 2}
              y={top - 2}
              width={g.w + 4}
              height={bot - top + 4}
              rx={2}
              fill="none"
              style={{ stroke: 'var(--accent)' }}
              strokeWidth={1.8}
              pointerEvents="none"
            />
          );
        })()}

        {/* query marker: the query is also one of the keys (self-attention is allowed in col, feat, full) */}
        {(() => {
          const g = geom(query);
          if (g.opacity === 0) return null;
          const cx0 = g.x + g.w / 2;
          const y = M_TOP - 9;
          return <path d={`M${cx0 - 4.5},${y} L${cx0 + 4.5},${y} L${cx0},${y + 6.5} Z`} style={{ fill: 'var(--accent)' }} pointerEvents="none" />;
        })()}

        {/* mask row (stage 'mask'): literally row q of the layer's mask */}
        {stage === 'mask' &&
          order.map((k) => {
            const x = spread.x.get(k)!;
            return (
              <rect
                key={`m${k}`}
                x={x}
                y={yBase + LANE + 4}
                width={Math.max(0.6, spread.slot - (spread.slot >= 3 ? 0.6 : 0))}
                height={7}
                style={{ fill: allowed[k] ? sequentialCss(mode === 'dark' ? 0.62 : 0.66, mode) : 'var(--surface-3)' }}
              />
            );
          })}

        {/* table bands / per-bar identity */}
        {!collapse
          ? spread.bands.map((b) => {
              const y = yBase + (stage === 'mask' ? LANE + 17 : 10);
              const w = b.x1 - b.x0;
              return (
                <g key={b.name}>
                  <rect x={b.x0} y={y} width={Math.max(1, w)} height={4} rx={1} style={{ fill: catVar(b.slot) }} />
                  <title>{b.name}</title>
                  {w > 36 && (
                    <text x={b.x0} y={y + 16} className="text-[13px]" style={{ fill: 'var(--ink-2)' }}>
                      {fitText(b.name, w)}
                    </text>
                  )}
                </g>
              );
            })
          : allowedOrder.map((k) => {
              const xk = collapsed.x.get(k)! + collapsed.slot / 2;
              const t = tokens[k];
              return (
                <g key={`l${k}`}>
                  <rect x={xk - 3} y={yBase + 6} width={6} height={6} rx={1} style={{ fill: catVar(tokens[k].isPadding ? 0 : groups.find((g) => g.name === t.tableName)?.slot ?? 0) }} />
                  {fewLabels && (
                    <text
                      x={xk + 3}
                      y={yBase + 18}
                      transform={`rotate(45 ${xk + 3} ${yBase + 18})`}
                      className="num text-[13px]"
                      style={{ fill: k === inspected ? 'var(--accent-text)' : 'var(--ink-2)' }}
                    >
                      {barLabel(k)}
                    </text>
                  )}
                </g>
              );
            })}

        {/* per-bar value labels when there is room */}
        {collapse &&
          collapsed.slot >= 22 &&
          allowedOrder.map((k) => {
            const g = geom(k);
            return (
              <motion.text
                key={`v${k}`}
                initial={false}
                animate={{ opacity: 1 }}
                x={g.x + g.w / 2}
                y={g.y - 4}
                textAnchor="middle"
                className="num text-[13px]"
                style={{ fill: 'var(--ink-2)' }}
              >
                {pctw(probs[k]).replace('%', '')}
              </motion.text>
            );
          })}

        {/* hit areas (none for an empty row at the softmax stage: there is nothing to read) */}
        {!empty && visibleKeys.map((k) => {
          const x = hitSlot.x.get(k);
          if (x === undefined) return null;
          return (
            <rect
              key={`h${k}`}
              x={x}
              y={M_TOP}
              width={Math.max(1, hitSlot.slot)}
              height={plotH + 14}
              fill="transparent"
              className="cursor-pointer"
              onPointerEnter={() => setHover(k)}
              onClick={() => onInspect(k)}
            />
          );
        })}

        {empty && (
          <text x={M_LEFT + plotW / 2} y={M_TOP + plotH / 2} textAnchor="middle" className="text-[15.5px]" style={{ fill: 'var(--ink-2)' }}>
            No allowed key: all {n} weights are exactly 0
          </text>
        )}
      </svg>

      {/* tooltip */}
      {tip && tipG && hover !== null && !empty && (
        <div
          className="pointer-events-none absolute z-10 w-[220px] rounded-[9px] border border-border bg-surface px-3 py-2 text-[14.5px] leading-snug shadow-[var(--shadow-pop)]"
          style={{ left: Math.max(0, Math.min(W - 222, tipG.x + tipG.w / 2 - 110)), top: Math.max(0, M_TOP - 6) }}
          role="tooltip"
        >
          <div className="num font-semibold text-ink">
            #{shown} {labels[shown!]}
          </div>
          <div className="num mt-0.5 text-ink-2">{tip.isTarget ? 'masked target “?”' : truncate(String(tokenValue(tip)), 28)}</div>
          <div className="num mt-1 grid grid-cols-[auto_1fr] gap-x-3 text-[14px]">
            <span className="text-muted">score</span>
            <span className="text-right text-ink">{sgn(scores[shown!], 3)}</span>
            <span className="text-muted">mask</span>
            <span className="text-right text-ink">{allowed[shown!] ? 'allowed' : 'blocked → −∞'}</span>
            {stage === 'softmax' && (
              <>
                <span className="text-muted">weight</span>
                <span className="text-right text-ink">{pctw(probs[shown!])}</span>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function tokenValue(t: Token): string {
  if (t.raw === null || t.raw === undefined) return '∅';
  if (typeof t.raw === 'number' && t.sem === 2) return new Date(t.raw).toISOString().slice(0, 10);
  return String(t.raw);
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}…`;
}

/** Fit a label into `w` px at ~6px per character. */
function fitText(s: string, w: number): string {
  const n = Math.floor((w - 2) / 6);
  return truncate(s, Math.max(3, n));
}
