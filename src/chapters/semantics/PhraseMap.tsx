/**
 * Figure "A map of every column phrase": the 2-D PCA of all column phrases of the four databases
 * (public/data/embeddings/pca2d.json, python/embed.py), colored and shaped by database. Hollow
 * markers = key columns (interned but never on a token). A pinned phrase draws lines to its
 * nearest neighbors in the full 384-d space, computed live, because two axes keep only part of the
 * variance.
 */
import { Minus, Plus, RotateCcw } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent as RPointerEvent } from 'react';
import { IconButton, cx } from '@/components';
import { shapeOfSlot, shapePath } from '@/components/viz/shapes';
import { textWidth } from '@/components/viz/text';
import { useElementSize } from '@/components/viz/useSize';
import { DB_NAMES, type DbName } from '@/engine/db';
import type { EmbeddingStore } from '@/engine/embeddings';
import { catVar } from '@/styles/palette';
import { DbTag, Phrase } from './bits';
import { DB_SHORT, DB_SLOT, ROLE_LABEL, cosine, f3, type Catalog, type PcaPoint } from './data';

export interface PhraseMapProps {
  points: PcaPoint[];
  store: EmbeddingStore;
  catalog: Catalog;
  focus: string;
  onFocus: (phrase: string) => void;
}

const LABEL_PX = 13.5;
const K_NEIGHBORS = 5;

export function PhraseMap({ points, store, catalog, focus, onFocus }: PhraseMapProps) {
  const reduce = useReducedMotion();
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const [view, setView] = useState({ k: 1, tx: 0, ty: 0 });
  const [hover, setHover] = useState<number | null>(null);
  const [hiddenDbs, setHiddenDbs] = useState<Set<DbName>>(new Set());
  const drag = useRef<{ x: number; y: number; tx: number; ty: number; moved: boolean } | null>(null);

  const W = Math.max(300, size.width || 900);
  const narrow = W < 600;
  const H = Math.round(narrow ? Math.min(560, W * 1.1) : Math.min(640, Math.max(380, W * 0.6)));
  // ~8% padding on every side, so no marker sits on the frame and the axis names stay clear
  const padX = Math.round(W * 0.08) + 6;
  const padY = Math.round(H * 0.08) + 8;

  const ext = useMemo(() => {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
  }, [points]);

  // Each axis fills the frame, but one unit of PC 1 is never drawn more than 1.6× (or less than 1/1.6×)
  // as long as one unit of PC 2; the ratio used is stated under the map.
  const base = useMemo(() => {
    const fx = (W - 2 * padX) / (ext.x1 - ext.x0 || 1);
    const fy = (H - 2 * padY) / (ext.y1 - ext.y0 || 1);
    const sx = Math.min(fx, fy * 1.6);
    const sy = Math.min(fy, fx * 1.6);
    const ox = padX + (W - 2 * padX - sx * (ext.x1 - ext.x0)) / 2;
    const oy = padY + (H - 2 * padY - sy * (ext.y1 - ext.y0)) / 2;
    return { sx, sy, ox, oy };
  }, [W, H, ext, padX, padY]);
  const stretch = base.sx / base.sy;

  const px = (p: { x: number; y: number }) => ({
    x: (base.ox + (p.x - ext.x0) * base.sx) * view.k + view.tx,
    y: (base.oy + (ext.y1 - p.y) * base.sy) * view.k + view.ty,
  });

  const focusIdx = points.findIndex((p) => p.text === focus);
  const neighbors = useMemo(() => {
    if (focusIdx < 0 || !store.has(points[focusIdx].text)) return [] as { i: number; cos: number }[];
    const q = store.get(points[focusIdx].text);
    return points
      .map((p, i) => ({ i, cos: i === focusIdx || !store.has(p.text) ? -Infinity : cosine(q, store.get(p.text)) }))
      .sort((a, b) => b.cos - a.cos)
      .slice(0, K_NEIGHBORS);
  }, [points, focusIdx, store]);

  const zoomAt = (factor: number, cx = W / 2, cy = H / 2) =>
    setView((v) => {
      const k = Math.min(8, Math.max(1, v.k * factor));
      const r = k / v.k;
      if (k === 1) return { k: 1, tx: 0, ty: 0 };
      return { k, tx: cx - (cx - v.tx) * r, ty: cy - (cy - v.ty) * r };
    });

  const onDown = (e: RPointerEvent<SVGSVGElement>) => {
    if (view.k === 1) return;
    drag.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty, moved: false };
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
  };
  const onMove = (e: RPointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
    setView((v) => ({ ...v, tx: d.tx + dx, ty: d.ty + dy }));
  };
  const onUp = () => {
    window.setTimeout(() => (drag.current = null), 0);
  };

  // text labels: the pinned phrase and the hovered one (neighbors get numbered badges instead)
  const labels = useMemo(() => {
    const ids: { i: number; strong: boolean }[] = [];
    if (focusIdx >= 0) ids.push({ i: focusIdx, strong: true });
    if (hover !== null && !ids.some((d) => d.i === hover)) ids.push({ i: hover, strong: false });
    const placed: { x: number; y: number; w: number; h: number; i: number; strong: boolean; anchor: 'start' | 'end'; px: number; py: number; right: boolean }[] = [];
    for (const { i, strong } of ids) {
      if (hiddenDbs.has(points[i].db)) continue;
      const p = px(points[i]);
      const w = textWidth(points[i].text, LABEL_PX, { weight: strong ? 650 : 400 }) + 6;
      const right = p.x + 10 + w < W - 4;
      let x = right ? p.x + 9 : p.x - 9 - w;
      x = Math.max(2, Math.min(W - w - 2, x));
      const free = (yy: number) => yy >= 2 && yy <= H - 20 && !placed.some((b) => x < b.x + b.w && b.x < x + w && yy < b.y + b.h + 1 && b.y < yy + 19);
      let y = p.y - 9;
      for (const off of [0, 20, -20, 40, -40, 60, -60, 80, -80]) {
        if (free(p.y - 9 + off)) {
          y = p.y - 9 + off;
          break;
        }
      }
      y = Math.max(2, Math.min(H - 20, y));
      placed.push({ x, y, w, h: 18, i, strong, anchor: 'start', px: p.x, py: p.y, right });
    }
    return placed;
  }, [focusIdx, hover, view, base, W, H, hiddenDbs, points]);

  const hp = hover !== null ? points[hover] : null;
  const hpPx = hp ? px(hp) : null;
  const hpInfo = hp ? catalog.byText.get(hp.text) : undefined;

  // the read-out card sits in the corner farthest from the hovered and the pinned point
  const TIP_W = Math.min(280, W - 16);
  const TIP_H = 104;
  const corner = useMemo(() => {
    if (!hpPx) return { left: 8, top: 8 };
    const pin = focusIdx >= 0 ? px(points[focusIdx]) : null;
    const cands = [
      { left: 8, top: 8 },
      { left: W - TIP_W - 8, top: 8 },
      { left: 8, top: H - TIP_H - 8 },
      { left: W - TIP_W - 8, top: H - TIP_H - 8 },
    ];
    const dist = (c: { left: number; top: number }, q: { x: number; y: number }) => {
      const dx = Math.max(c.left - q.x, 0, q.x - (c.left + TIP_W));
      const dy = Math.max(c.top - q.y, 0, q.y - (c.top + TIP_H));
      return Math.hypot(dx, dy);
    };
    let best = cands[0];
    let bd = -1;
    for (const c of cands) {
      const d = Math.min(dist(c, hpPx), pin ? dist(c, pin) + 40 : Infinity);
      if (d > bd) (bd = d), (best = c);
    }
    return best;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hpPx?.x, hpPx?.y, focusIdx, W, H, TIP_W, view, base]);

  // keyboard: arrows move to the nearest visible point in that direction, Enter pins it
  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    const visible = points.map((p, i) => ({ p, i })).filter(({ p }) => !hiddenDbs.has(p.db));
    if (!visible.length) return;
    const cur = hover ?? (focusIdx >= 0 && !hiddenDbs.has(points[focusIdx].db) ? focusIdx : null);
    const dirs: Record<string, [number, number]> = { ArrowRight: [1, 0], ArrowLeft: [-1, 0], ArrowDown: [0, 1], ArrowUp: [0, -1] };
    if (e.key in dirs) {
      e.preventDefault();
      if (cur === null) {
        setHover(visible[0].i);
        return;
      }
      const [ux, uy] = dirs[e.key];
      const a = px(points[cur]);
      let best: number | null = null;
      let bs = Infinity;
      for (const { p, i } of visible) {
        if (i === cur) continue;
        const b = px(p);
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const along = dx * ux + dy * uy;
        if (along <= 0.5) continue;
        const across = Math.abs(dx * uy - dy * ux);
        const score = along + 2.5 * across;
        if (score < bs) (bs = score), (best = i);
      }
      if (best !== null) setHover(best);
    } else if ((e.key === 'Enter' || e.key === ' ') && cur !== null) {
      e.preventDefault();
      onFocus(points[cur].text);
    } else if (e.key === 'Escape') setHover(null);
  };

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Show or hide a database">
          {DB_NAMES.map((d) => {
            const off = hiddenDbs.has(d);
            return (
              <button
                key={d}
                type="button"
                aria-pressed={!off}
                onClick={() =>
                  setHiddenDbs((s) => {
                    const n = new Set(s);
                    if (n.has(d)) n.delete(d);
                    else n.add(d);
                    return n;
                  })
                }
                className={cx(
                  'inline-flex h-8 items-center rounded-full border px-3 text-[15px] transition-colors',
                  off ? 'border-dashed border-border text-muted' : 'border-border bg-surface text-ink hover:bg-surface-2',
                )}
              >
                <DbTag db={d} className={off ? 'opacity-40' : ''} />
              </button>
            );
          })}
          <span className="inline-flex items-center gap-1.5 pl-1 text-[15px] text-ink-2">
            <svg width="14" height="14" aria-hidden="true">
              <circle cx="7" cy="7" r="5" fill="none" style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.6} />
            </svg>
            hollow = key column
          </span>
        </div>
        <div className="flex items-center gap-1">
          <IconButton label="Zoom in" onClick={() => zoomAt(1.6)}>
            <Plus size={16} />
          </IconButton>
          <IconButton label="Zoom out" onClick={() => zoomAt(1 / 1.6)} disabled={view.k <= 1}>
            <Minus size={16} />
          </IconButton>
          <IconButton label="Reset view" onClick={() => setView({ k: 1, tx: 0, ty: 0 })} disabled={view.k === 1}>
            <RotateCcw size={16} />
          </IconButton>
        </div>
      </div>

      <div ref={wrapRef} className="relative w-full overflow-hidden rounded-lg border border-border bg-surface">
        <svg
          width={W}
          height={H}
          role="img"
          aria-label={`2-D PCA map of ${points.length} column phrases from four databases. Arrow keys move between points, Enter pins the point and lists its nearest neighbors.`}
          tabIndex={0}
          onKeyDown={onKey}
          className={cx('block select-none focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent', view.k > 1 ? 'cursor-grab active:cursor-grabbing' : '')}
          style={{ touchAction: view.k > 1 ? 'none' : 'pan-y' }}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          onPointerLeave={() => setHover(null)}
          onDoubleClick={(e) => {
            const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            zoomAt(1.8, e.clientX - r.left, e.clientY - r.top);
          }}
        >
          {/* zero axes (hairlines); axis names in the padded margin, clear of the markers */}
          <line x1={0} x2={W} y1={px({ x: 0, y: 0 }).y} y2={px({ x: 0, y: 0 }).y} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
          <line x1={px({ x: 0, y: 0 }).x} x2={px({ x: 0, y: 0 }).x} y1={0} y2={H} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
          <text x={W - 10} y={H - 10} textAnchor="end" style={{ fontSize: 13.5, fill: 'var(--muted)' }}>
            PC 1 →
          </text>
          <text x={10} y={20} style={{ fontSize: 13.5, fill: 'var(--muted)' }}>
            ↑ PC 2
          </text>

          {/* neighbor lines */}
          {focusIdx >= 0 &&
            !hiddenDbs.has(points[focusIdx].db) &&
            neighbors.map((n) => {
              if (hiddenDbs.has(points[n.i].db)) return null;
              const a = px(points[focusIdx]);
              const b = px(points[n.i]);
              return (
                <motion.line
                  key={`${focus}-${n.i}`}
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  initial={reduce ? false : { opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.45 }}
                  style={{ stroke: 'var(--ink-2)' }}
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                />
              );
            })}

          {/* points */}
          {points.map((p, i) => {
            if (hiddenDbs.has(p.db)) return null;
            const info = catalog.byText.get(p.text);
            const hollow = info ? !info.onToken : false;
            const q = px(p);
            const isFocus = i === focusIdx;
            const isNb = neighbors.some((n) => n.i === i);
            const r = isFocus ? 7.5 : 5.5;
            return (
              <g
                key={p.text}
                transform={`translate(${q.x},${q.y})`}
                className="cursor-pointer"
                onPointerEnter={() => setHover(i)}
                onClick={() => {
                  if (drag.current?.moved) return;
                  onFocus(p.text);
                }}
              >
                <circle r={12} fill="transparent" />
                {(isFocus || isNb) && <circle r={r + 4} fill="none" style={{ stroke: isFocus ? 'var(--accent)' : 'var(--ink-2)' }} strokeWidth={isFocus ? 2 : 1.2} />}
                <path
                  d={shapePath(shapeOfSlot(DB_SLOT[p.db]), r)}
                  style={{
                    fill: hollow ? 'var(--surface)' : catVar(DB_SLOT[p.db]),
                    stroke: hollow ? catVar(DB_SLOT[p.db]) : 'var(--surface)',
                  }}
                  strokeWidth={hollow ? 1.8 : 1.2}
                />
              </g>
            );
          })}

          {/* neighbor badges: the numbers match the list under the map */}
          {focusIdx >= 0 &&
            !hiddenDbs.has(points[focusIdx].db) &&
            neighbors.map((n, j) => {
              if (hiddenDbs.has(points[n.i].db)) return null;
              const q = px(points[n.i]);
              const bx = Math.min(W - 10, q.x + 11);
              const by = Math.max(10, q.y - 11);
              return (
                <g key={`nb-${n.i}`} pointerEvents="none" transform={`translate(${bx},${by})`}>
                  <circle r={9} style={{ fill: 'var(--ink)', stroke: 'var(--surface)' }} strokeWidth={1.5} />
                  <text textAnchor="middle" dominantBaseline="central" className="num" style={{ fontSize: 13, fontWeight: 650, fill: 'var(--surface)' }}>
                    {j + 1}
                  </text>
                </g>
              );
            })}

          {/* labels */}
          {labels.map((l) => (
            <g key={`l-${l.i}`} pointerEvents="none">
              {Math.abs(l.y + 9 - l.py) > 4 && (
                <line x1={l.px} y1={l.py} x2={l.right ? l.x - 2 : l.x + l.w - 4} y2={l.y + 9} style={{ stroke: 'var(--axis)' }} strokeWidth={1} />
              )}
              <rect x={l.x - 3} y={l.y - 1} width={l.w} height={19} rx={4} style={{ fill: 'var(--surface)', opacity: 0.88 }} />
              <text x={l.x} y={l.y + 13} style={{ fontSize: LABEL_PX, fill: 'var(--ink)', fontWeight: l.strong ? 650 : 400 }}>
                {points[l.i].text}
              </text>
            </g>
          ))}
        </svg>
        <div className="sr-only" aria-live="polite">
          {hp ? `${hp.text}, ${DB_SHORT[hp.db]}, ${hpInfo ? ROLE_LABEL[hpInfo.role] : ''}${focusIdx >= 0 && hover === focusIdx ? ', pinned' : ''}` : ''}
        </div>

        {hp && hpPx && (
          <div
            className="pointer-events-none absolute z-10 rounded-lg border border-border bg-surface px-3 py-2 text-[14.5px] shadow-[var(--shadow-pop)]"
            style={{ left: corner.left, top: corner.top, width: TIP_W, minHeight: TIP_H - 12 }}
          >
            <div className="text-[15px] leading-snug">
              <Phrase text={hp.text} strong />
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-x-2 text-ink-2">
              <DbTag db={hp.db} />
              <span>· {hpInfo ? ROLE_LABEL[hpInfo.role] : ''}</span>
              {hpInfo && !hpInfo.onToken && <span>· never on a token</span>}
            </div>
            <div className="num mt-1 text-[14px] text-muted">
              PC1 {f3(hp.x)} · PC2 {f3(hp.y)}
            </div>
          </div>
        )}
      </div>

      {focusIdx >= 0 && (
        <div className="mt-3 rounded-lg bg-surface-2 px-4 py-3">
          <div className="mb-1.5 text-[15.5px] text-ink">
            Pinned: <Phrase text={points[focusIdx].text} strong />. Its {K_NEIGHBORS} nearest phrases in the full 384-d space (numbered on the
            map, joined by dashed lines):
          </div>
          <ol className="!m-0 grid !list-none gap-x-6 gap-y-1 !p-0 sm:grid-cols-2">
            {neighbors.map((n, j) => (
              <li key={n.i} className="!m-0 flex min-w-0 items-center gap-2 !p-0 text-[15px]">
                <span className="num inline-flex h-[20px] w-[20px] shrink-0 items-center justify-center rounded-full bg-ink text-[13px] font-semibold text-surface">
                  {j + 1}
                </span>
                <DbTag db={points[n.i].db} showName={false} />
                <span className="min-w-0 truncate" title={points[n.i].text}>
                  <Phrase text={points[n.i].text} />
                </span>
                <span className="num ml-auto text-ink">{f3(n.cos)}</span>
              </li>
            ))}
          </ol>
        </div>
      )}
      <p className="mt-2 text-[14.5px] text-muted">
        Hover a point to read it and click to pin it, or click a phrase in Step 1. With the keyboard, focus the map, move with the arrow keys
        and press Enter to pin. Zoom with the buttons or a double-click, then drag to pan.{' '}
        {Math.abs(stretch - 1) < 0.08
          ? 'Both axes use the same scale.'
          : `To fill the frame, one unit of PC 1 is drawn ${stretch.toFixed(2)}× as long as one unit of PC 2.`}
      </p>
    </div>
  );
}
