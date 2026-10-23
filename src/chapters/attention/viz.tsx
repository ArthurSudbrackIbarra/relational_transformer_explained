/**
 * Small local visualisations for the attention chapter (display only):
 *  - MatrixCanvas: a dense matrix (weights or activations) on canvas, diverging blue↔red, with an
 *    emphasised band of rows or columns (e.g. the 16 rows of W_q that make head j's query).
 *  - VStrip: a vector drawn vertically (one cell per dimension), for "h → W → q" layouts.
 * Colours come from the shared diverging scale (BUILD_PLAN §5: signed vector values).
 */
import { useEffect, useRef, useState } from 'react';
import { divergingRgb, rgbCss, sequentialCss } from '@/components/viz/scales';
import { TOKENS } from '@/styles/palette';
import { useReadingMode, useResolvedTheme } from '@/styles/theme';
import { cx } from '@/components';

export interface MatrixCanvasProps {
  data: ArrayLike<number>;
  rows: number;
  cols: number;
  /** CSS px per cell (width, height). */
  cellW: number;
  cellH: number;
  /** Symmetric colour domain max |v|. */
  maxAbs: number;
  /** Emphasised half-open ranges; everything else is faded. */
  bandRows?: [number, number] | null;
  bandCols?: [number, number] | null;
  /** Accessible name. */
  label: string;
  onHover?: (cell: { r: number; c: number; v: number } | null) => void;
  className?: string;
}

export function MatrixCanvas({ data, rows, cols, cellW, cellH, maxAbs, bandRows, bandCols, label, onHover, className }: MatrixCanvasProps) {
  const mode = useResolvedTheme();
  const reading = useReadingMode();
  const ref = useRef<HTMLCanvasElement | null>(null);
  const W = Math.max(1, cols * cellW);
  const H = Math.max(1, rows * cellH);
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const dpr = typeof window !== 'undefined' ? Math.min(2, window.devicePixelRatio || 1) : 1;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(cols, rows);
    const surf = TOKENS[mode].surface;
    const sr = parseInt(surf.slice(1, 3), 16);
    const sg = parseInt(surf.slice(3, 5), 16);
    const sb = parseInt(surf.slice(5, 7), 16);
    const banded = !!bandRows || !!bandCols;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const [cr, cg, cb] = divergingRgb(Number(data[r * cols + c]), maxAbs || 1, mode);
        const inBand = (!bandRows || (r >= bandRows[0] && r < bandRows[1])) && (!bandCols || (c >= bandCols[0] && c < bandCols[1]));
        const a = banded && !inBand ? 0.28 : 1;
        const o = (r * cols + c) * 4;
        img.data[o] = Math.round(cr * a + sr * (1 - a));
        img.data[o + 1] = Math.round(cg * a + sg * (1 - a));
        img.data[o + 2] = Math.round(cb * a + sb * (1 - a));
        img.data[o + 3] = 255;
      }
    }
    const off = document.createElement('canvas');
    off.width = cols;
    off.height = rows;
    off.getContext('2d')!.putImageData(img, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, W * dpr, H * dpr);
  }, [data, rows, cols, W, H, dpr, maxAbs, mode, reading, bandRows, bandCols]);

  const band = bandRows
    ? { x: 0, y: bandRows[0] * cellH, w: W, h: (bandRows[1] - bandRows[0]) * cellH }
    : bandCols
      ? { x: bandCols[0] * cellW, y: 0, w: (bandCols[1] - bandCols[0]) * cellW, h: H }
      : null;

  return (
    <div className={cx('relative shrink-0', className)} style={{ width: W, height: H }}>
      <canvas
        ref={ref}
        role="img"
        aria-label={label}
        style={{ width: W, height: H, display: 'block' }}
        className="cursor-crosshair rounded-[2px]"
        onPointerMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const c = Math.floor((e.clientX - rect.left) / cellW);
          const r = Math.floor((e.clientY - rect.top) / cellH);
          if (r < 0 || c < 0 || r >= rows || c >= cols) {
            setHover(null);
            onHover?.(null);
            return;
          }
          setHover({ r, c });
          onHover?.({ r, c, v: Number(data[r * cols + c]) });
        }}
        onPointerLeave={() => {
          setHover(null);
          onHover?.(null);
        }}
      />
      {band && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute rounded-[2px]"
          style={{ left: band.x - 1.5, top: band.y - 1.5, width: band.w + 3, height: band.h + 3, border: '1.5px solid var(--accent)' }}
        />
      )}
      {hover && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute"
          style={{ left: hover.c * cellW - 0.5, top: hover.r * cellH - 0.5, width: cellW + 1, height: cellH + 1, border: '1px solid var(--ink)' }}
        />
      )}
    </div>
  );
}

export interface VStripProps {
  values: ArrayLike<number>;
  cellW: number;
  cellH: number;
  maxAbs: number;
  /** Outlined half-open ranges. */
  bands?: { start: number; end: number; strong?: boolean }[];
  /** Fade everything outside the strong band. */
  fadeOutside?: boolean;
  label: string;
  onHover?: (i: number | null) => void;
  highlight?: number | null;
  className?: string;
}

/** A vector drawn top-to-bottom, one cell per dimension. */
export function VStrip({ values, cellW, cellH, maxAbs, bands, fadeOutside, label, onHover, highlight = null, className }: VStripProps) {
  const mode = useResolvedTheme();
  const n = values.length;
  const strong = bands?.find((b) => b.strong);
  return (
    <svg
      width={cellW + 4}
      height={n * cellH + 4}
      viewBox={`-2 -2 ${cellW + 4} ${n * cellH + 4}`}
      role="img"
      aria-label={label}
      className={cx('block shrink-0 overflow-visible', className)}
      onPointerMove={(e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const i = Math.floor((e.clientY - rect.top - 2) / cellH);
        onHover?.(i >= 0 && i < n ? i : null);
      }}
      onPointerLeave={() => onHover?.(null)}
    >
      {Array.from({ length: n }, (_, i) => {
        const faded = fadeOutside && strong && (i < strong.start || i >= strong.end);
        return (
          <rect
            key={i}
            x={0}
            y={i * cellH}
            width={cellW}
            height={Math.max(0.5, cellH - (cellH >= 5 ? 0.6 : 0))}
            style={{ fill: rgbCss(divergingRgb(values[i], maxAbs || 1, mode)), opacity: faded ? 0.3 : 1 }}
          />
        );
      })}
      {bands?.map((b, k) => (
        <rect
          key={k}
          x={-1}
          y={b.start * cellH - 1}
          width={cellW + 2}
          height={(b.end - b.start) * cellH + 2}
          fill="none"
          rx={1.5}
          style={{ stroke: b.strong ? 'var(--accent)' : 'var(--axis)' }}
          strokeWidth={b.strong ? 1.6 : 1}
        />
      ))}
      {highlight !== null && highlight >= 0 && highlight < n && (
        <rect x={-1.5} y={highlight * cellH - 0.5} width={cellW + 3} height={cellH + 1} fill="none" style={{ stroke: 'var(--ink)' }} strokeWidth={1.2} />
      )}
    </svg>
  );
}

/** A tiny "label: value" line used in the arithmetic panels. */
export function KV({ k, v, className }: { k: React.ReactNode; v: React.ReactNode; className?: string }) {
  return (
    <div className={cx('flex items-baseline justify-between gap-3 border-b border-grid/70 py-1 last:border-0', className)}>
      <span className="text-ink-2">{k}</span>
      <span className="num text-right text-ink">{v}</span>
    </div>
  );
}

/**
 * A positive vector (e.g. an RMSNorm gain γ) on the sequential scale over its own range, so its
 * variation is visible (a signed diverging scale would paint every cell the same red).
 */
export function SeqStrip({ values, label, cellHeight = 14 }: { values: ArrayLike<number>; label: string; cellHeight?: number }) {
  const mode = useResolvedTheme();
  const [hover, setHover] = useState<number | null>(null);
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    lo = Math.min(lo, values[i]);
    hi = Math.max(hi, values[i]);
  }
  const t = (v: number) => (hi > lo ? 0.12 + 0.88 * ((v - lo) / (hi - lo)) : 0.6);
  return (
    <div className="grid gap-1">
      <div
        role="img"
        aria-label={`${label}: ${values.length} values from ${lo.toFixed(3)} to ${hi.toFixed(3)}`}
        className="grid overflow-hidden rounded-[3px]"
        style={{ gridTemplateColumns: `repeat(${values.length}, minmax(0, 1fr))`, height: cellHeight }}
        onPointerLeave={() => setHover(null)}
      >
        {Array.from({ length: values.length }, (_, i) => (
          <div
            key={i}
            onPointerEnter={() => setHover(i)}
            style={{ background: sequentialCss(t(values[i]), mode), outline: hover === i ? '1.5px solid var(--ink)' : undefined, outlineOffset: -1 }}
          />
        ))}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-x-3 text-[14px] text-muted">
        <span className="inline-flex items-center gap-1.5">
          <span className="num">{lo.toFixed(3)}</span>
          <span
            aria-hidden="true"
            className="inline-block h-2.5 w-16 rounded-full"
            style={{ background: `linear-gradient(90deg, ${sequentialCss(0.12, mode)}, ${sequentialCss(1, mode)})` }}
          />
          <span className="num">{hi.toFixed(3)}</span>
          <span>(this vector’s own range; all values are positive)</span>
        </span>
        <span className="num min-h-[1.3em]">{hover !== null ? `[${hover}] = ${values[hover].toFixed(4)}` : ''}</span>
      </div>
    </div>
  );
}
