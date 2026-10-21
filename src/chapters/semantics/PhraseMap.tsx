/**
 * Figure "A map of every column phrase": the 2-D PCA of all column phrases of the four databases
 * (public/data/embeddings/pca2d.json, python/embed.py), coloured and shaped by database. Hollow
 * markers = key columns (interned but never on a token). A pinned phrase draws lines to its
 * nearest neighbours in the full 384-d space, computed live, because two axes keep only part of the
 * variance.
 */
import { Minus, Plus, RotateCcw } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import { useMemo, useRef, useState, type PointerEvent as RPointerEvent } from 'react';
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
  const H = Math.round(Math.min(680, Math.max(340, W * 0.66)));
  const pad = 28;

  const ext = useMemo(() => {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
  }, [points]);

  // equal aspect ratio: one unit on PC1 = one unit on PC2
  const base = useMemo(() => {
    const sx = (W - 2 * pad) / (ext.x1 - ext.x0 || 1);
    const sy = (H - 2 * pad) / (ext.y1 - ext.y0 || 1);
    const s = Math.min(sx, sy);
    const ox = pad + (W - 2 * pad - s * (ext.x1 - ext.x0)) / 2;
    const oy = pad + (H - 2 * pad - s * (ext.y1 - ext.y0)) / 2;
    return { s, ox, oy };
  }, [W, H, ext]);

  const px = (p: { x: number; y: number }) => ({
    x: (base.ox + (p.x - ext.x0) * base.s) * view.k + view.tx,
    y: (base.oy + (ext.y1 - p.y) * base.s) * view.k + view.ty,
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

  // labels: pinned phrase + its neighbours + the hovered phrase, nudged to avoid overlaps
  const labels = useMemo(() => {
    const ids: { i: number; strong: boolean }[] = [];
    if (focusIdx >= 0) ids.push({ i: focusIdx, strong: true });
    if (W >= 560) for (const n of neighbors) ids.push({ i: n.i, strong: false });
    if (hover !== null && !ids.some((d) => d.i === hover)) ids.push({ i: hover, strong: false });
    const placed: { x: number; y: number; w: number; h: number; i: number; strong: boolean; anchor: 'start' | 'end'; px: number; py: number; right: boolean }[] = [];
    for (const { i, strong } of ids) {
      if (hiddenDbs.has(points[i].db)) continue;
      const p = px(points[i]);
      const w = textWidth(points[i].text, LABEL_PX, { mono: true, weight: strong ? 650 : 400 }) + 6;
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
  }, [focusIdx, neighbors, hover, view, base, W, H, hiddenDbs, points]);

  const zero = px({ x: 0, y: 0 });
  const hp = hover !== null ? points[hover] : null;
  const hpPx = hp ? px(hp) : null;
  const hpInfo = hp ? catalog.byText.get(hp.text) : undefined;

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
          aria-label={`2-D PCA map of ${points.length} column phrases from four databases. Click a point to pin it and see its nearest neighbours.`}
          className={cx('block select-none', view.k > 1 ? 'cursor-grab active:cursor-grabbing' : '')}
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
          {/* zero axes (hairlines) */}
          <line x1={0} x2={W} y1={zero.y} y2={zero.y} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
          <line x1={zero.x} x2={zero.x} y1={0} y2={H} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
          <text x={W - 8} y={Math.min(H - 8, Math.max(16, zero.y - 6))} textAnchor="end" style={{ fontSize: 13.5, fill: 'var(--muted)' }}>
            PC 1
          </text>
          <text x={Math.min(W - 40, Math.max(8, zero.x + 6))} y={16} style={{ fontSize: 13.5, fill: 'var(--muted)' }}>
            PC 2
          </text>

          {/* neighbour lines */}
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

          {/* labels */}
          {labels.map((l) => (
            <g key={`l-${l.i}`} pointerEvents="none">
              {Math.abs(l.y + 9 - l.py) > 4 && (
                <line x1={l.px} y1={l.py} x2={l.right ? l.x - 2 : l.x + l.w - 4} y2={l.y + 9} style={{ stroke: 'var(--axis)' }} strokeWidth={1} />
              )}
              <rect x={l.x - 3} y={l.y - 1} width={l.w} height={19} rx={4} style={{ fill: 'var(--surface)', opacity: 0.88 }} />
              <text x={l.x} y={l.y + 13} className="font-mono" style={{ fontSize: LABEL_PX, fill: 'var(--ink)', fontWeight: l.strong ? 650 : 400 }}>
                {points[l.i].text}
              </text>
            </g>
          ))}
        </svg>

        {hp && hpPx && (
          <div
            className="pointer-events-none absolute z-10 w-[280px] max-w-[calc(100%-16px)] rounded-lg border border-border bg-surface px-3 py-2 text-[14.5px] shadow-[var(--shadow-pop)]"
            style={{
              left: Math.max(8, Math.min(W - 288, hpPx.x + 14)),
              top: hpPx.y > H / 2 ? Math.max(8, hpPx.y - 118) : Math.min(H - 110, hpPx.y + 14),
            }}
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
            Pinned: <Phrase text={points[focusIdx].text} strong />. Its {K_NEIGHBORS} nearest phrases in the full 384-d space (dashed lines):
          </div>
          <ol className="!m-0 grid !list-none gap-x-6 gap-y-1 !p-0 sm:grid-cols-2">
            {neighbors.map((n, j) => (
              <li key={n.i} className="!m-0 flex min-w-0 items-center gap-2 !p-0 text-[15px]">
                <span className="num w-4 text-muted">{j + 1}</span>
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
        Hover a point to read it, click to pin it. Zoom with the buttons or a double-click, then drag to pan. Both axes use the same scale.
      </p>
    </div>
  );
}
