/**
 * Matrix: one tensor of the forward pass drawn on canvas, one rectangle per entry (cellW × cellH CSS
 * px, so S × d activations and d × d weights keep their proportions at any size). Display only.
 *
 * Colors are the site's scales (BUILD_PLAN §5): diverging blue ↔ red with a gray midpoint for signed
 * values (activations, weights), sequential blue for magnitudes (attention weights), binary for
 * masks; masked entries are drawn as the empty surface. Every value is reachable without hover: the
 * overlay canvas takes keyboard focus and the arrow keys move a cursor that drives the same readout.
 *
 * `transpose` draws a row-major [cols × rows] array transposed, so a PyTorch nn.Linear weight
 * stored [out, in] can be shown as Wᵀ ([in, out]) without copying it.
 */
import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent as RPointerEvent } from 'react';
import { cx } from '@/components';
import { divergingRgb, maxAbs as maxAbsOf, sequentialRgb, type RGB } from '@/components/viz/scales';
import { catHex, TOKENS } from '@/styles/palette';
import { useReadingMode, useResolvedTheme } from '@/styles/theme';

export type MatScale = 'diverging' | 'sequential' | 'binary';

export interface Cell {
  r: number;
  c: number;
}

export interface MatrixProps {
  data: ArrayLike<number>;
  rows: number;
  cols: number;
  /** `data` is [cols × rows] row-major and is drawn transposed (Wᵀ of an [out, in] weight). */
  transpose?: boolean;
  /** Draw only the first rows / columns (e.g. the non-padding tokens). */
  drawRows?: number;
  drawCols?: number;
  cellW: number;
  cellH: number;
  scale?: MatScale;
  /** Diverging: max |v| (default: the drawn entries' max |v|). Sequential: [lo, hi] (default [0, max]). */
  domain?: number | [number, number];
  /** 1 = masked entry, row-major [rows × cols] in display coordinates. */
  blocked?: ArrayLike<number> | null;
  /** Categorical slot per row / column: a thin identity strip on the left / top. */
  rowSlots?: readonly number[];
  colSlots?: readonly number[];
  /** Outlined row (the token in focus). */
  focusRow?: number | null;
  /** A row / column / entry marked from elsewhere (linked hover). */
  markRow?: number | null;
  markCol?: number | null;
  markCell?: Cell | null;
  /** Emphasized half-open ranges; the rest is faded (e.g. one head's 16 dimensions). */
  bandRows?: readonly [number, number] | null;
  bandCols?: readonly [number, number] | null;
  /** Fade the entries outside the band (default true); false only outlines it. */
  fadeOutside?: boolean;
  onHover?: (cell: Cell | null) => void;
  onPick?: (cell: Cell) => void;
  /** Accessible name. */
  label: string;
  className?: string;
}

const STRIP = 4;
const STRIP_GAP = 2;

function hexRgb(hex: string): RGB {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

/** Value of display entry (r, c). */
export function valueAt(data: ArrayLike<number>, rows: number, cols: number, r: number, c: number, transpose = false): number {
  return Number(transpose ? data[c * rows + r] : data[r * cols + c]);
}

/** The color domain a Matrix would pick for these entries (so linked panels can share it). */
export function autoDomain(data: ArrayLike<number>, scale: MatScale): number | [number, number] {
  if (scale === 'diverging') return maxAbsOf(data) || 1;
  if (scale === 'binary') return [0, 1];
  let hi = 0;
  for (let i = 0; i < data.length; i++) if (Number.isFinite(data[i]) && data[i] > hi) hi = data[i];
  return [0, hi || 1];
}

export function Matrix(p: MatrixProps) {
  const { data, rows, cols, transpose = false, cellW, cellH, scale = 'diverging' } = p;
  const R = Math.max(0, Math.min(rows, p.drawRows ?? rows));
  const C = Math.max(0, Math.min(cols, p.drawCols ?? cols));
  const mode = useResolvedTheme();
  const reading = useReadingMode();
  const base = useRef<HTMLCanvasElement | null>(null);
  const overlay = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<Cell | null>(null);
  const [cursor, setCursor] = useState<Cell | null>(null);
  const dpr = typeof window !== 'undefined' ? Math.min(2, window.devicePixelRatio || 1) : 1;
  const sx = p.rowSlots ? STRIP + STRIP_GAP : 0;
  const sy = p.colSlots ? STRIP + STRIP_GAP : 0;
  const plotW = C * cellW;
  const plotH = R * cellH;
  const Wd = sx + plotW;
  const Ht = sy + plotH;

  const dom = useMemo(() => {
    if (p.domain !== undefined) return p.domain;
    if (scale === 'diverging') {
      let m = 0;
      for (let r = 0; r < R; r++)
        for (let c = 0; c < C; c++) {
          const v = Math.abs(valueAt(data, rows, cols, r, c, transpose));
          if (Number.isFinite(v) && v > m) m = v;
        }
      return m || 1;
    }
    if (scale === 'binary') return [0, 1] as [number, number];
    let hi = 0;
    for (let r = 0; r < R; r++)
      for (let c = 0; c < C; c++) {
        const v = valueAt(data, rows, cols, r, c, transpose);
        if (Number.isFinite(v) && v > hi) hi = v;
      }
    return [0, hi || 1] as [number, number];
  }, [p.domain, scale, data, rows, cols, R, C, transpose]);

  // effects depend on primitives, so a parent re-render with equal-valued props never redraws
  const d0 = typeof dom === 'number' ? dom : dom[0];
  const d1 = typeof dom === 'number' ? Number.NaN : dom[1];
  const br0 = p.bandRows?.[0] ?? -1;
  const br1 = p.bandRows?.[1] ?? -1;
  const bc0 = p.bandCols?.[0] ?? -1;
  const bc1 = p.bandCols?.[1] ?? -1;
  const fade = p.fadeOutside !== false;

  // base layer: the entries and the identity strips
  useEffect(() => {
    const cv = base.current;
    if (!cv || R === 0 || C === 0) return;
    cv.width = Math.max(1, Math.round(Wd * dpr));
    cv.height = Math.max(1, Math.round(Ht * dpr));
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const t = TOKENS[mode];
    const surf = hexRgb(t.surface);
    const blockedRgb = hexRgb(t.surface2);
    const img = ctx.createImageData(C, R);
    const banded = (br0 >= 0 || bc0 >= 0) && fade;
    const maxAbs = Number.isNaN(d1) ? d0 : Math.max(Math.abs(d0), Math.abs(d1));
    const [lo, hi] = Number.isNaN(d1) ? [0, d0] : [d0, d1];
    const onRgb = sequentialRgb(mode === 'dark' ? 0.62 : 0.66, mode);
    for (let r = 0; r < R; r++) {
      for (let c = 0; c < C; c++) {
        let rgb: RGB;
        if (p.blocked && p.blocked[r * cols + c]) rgb = blockedRgb;
        else {
          const v = valueAt(data, rows, cols, r, c, transpose);
          rgb =
            scale === 'diverging'
              ? divergingRgb(v, maxAbs || 1, mode)
              : scale === 'binary'
                ? v > 0
                  ? onRgb
                  : blockedRgb
                : sequentialRgb(hi > lo ? (v - lo) / (hi - lo) : 0, mode);
        }
        const inBand = (br0 < 0 || (r >= br0 && r < br1)) && (bc0 < 0 || (c >= bc0 && c < bc1));
        const a = banded && !inBand ? 0.26 : 1;
        const o = (r * C + c) * 4;
        img.data[o] = Math.round(rgb[0] * a + surf[0] * (1 - a));
        img.data[o + 1] = Math.round(rgb[1] * a + surf[1] * (1 - a));
        img.data[o + 2] = Math.round(rgb[2] * a + surf[2] * (1 - a));
        img.data[o + 3] = 255;
      }
    }
    const off = document.createElement('canvas');
    off.width = C;
    off.height = R;
    off.getContext('2d')!.putImageData(img, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, sx * dpr, sy * dpr, plotW * dpr, plotH * dpr);
    if (p.rowSlots) {
      for (let r = 0; r < R; r++) {
        ctx.fillStyle = catHex(p.rowSlots[r], mode);
        ctx.fillRect(0, (sy + r * cellH) * dpr, STRIP * dpr, Math.max(1, cellH * dpr));
      }
    }
    if (p.colSlots) {
      for (let c = 0; c < C; c++) {
        ctx.fillStyle = catHex(p.colSlots[c], mode);
        ctx.fillRect((sx + c * cellW) * dpr, 0, Math.max(1, cellW * dpr), STRIP * dpr);
      }
    }
  }, [data, rows, cols, transpose, R, C, cellW, cellH, scale, d0, d1, p.blocked, p.rowSlots, p.colSlots, br0, br1, bc0, bc1, fade, mode, reading, dpr, Wd, Ht, sx, sy, plotW, plotH]);

  // overlay: focus row, linked marks, hover / keyboard cursor
  const shown = hover ?? cursor;
  const mcR = p.markCell?.r ?? -1;
  const mcC = p.markCell?.c ?? -1;
  useEffect(() => {
    const cv = overlay.current;
    if (!cv) return;
    cv.width = Math.max(1, Math.round(Wd * dpr));
    cv.height = Math.max(1, Math.round(Ht * dpr));
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, Wd, Ht);
    const t = TOKENS[mode];
    const box = (x: number, y: number, w: number, h: number, color: string, lw: number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.strokeRect(x - lw / 2, y - lw / 2, w + lw, h + lw);
    };
    if (br0 >= 0) box(sx, sy + br0 * cellH, plotW, (br1 - br0) * cellH, t.accent, 1.5);
    if (bc0 >= 0) box(sx + bc0 * cellW, sy, (bc1 - bc0) * cellW, plotH, t.accent, 1.5);
    const tint = mode === 'dark' ? 'rgba(255,255,255,0.10)' : 'rgba(11,11,11,0.07)';
    if (p.markRow !== null && p.markRow !== undefined && p.markRow < R) {
      ctx.fillStyle = tint;
      ctx.fillRect(sx, sy + p.markRow * cellH, plotW, cellH);
      box(sx, sy + p.markRow * cellH, plotW, cellH, t.ink, 1);
    }
    if (p.markCol !== null && p.markCol !== undefined && p.markCol < C) {
      ctx.fillStyle = tint;
      ctx.fillRect(sx + p.markCol * cellW, sy, cellW, plotH);
      box(sx + p.markCol * cellW, sy, cellW, plotH, t.ink, 1);
    }
    if (p.focusRow !== null && p.focusRow !== undefined && p.focusRow >= 0 && p.focusRow < R) {
      box(sx - (p.rowSlots ? STRIP + STRIP_GAP : 0), sy + p.focusRow * cellH, plotW + (p.rowSlots ? STRIP + STRIP_GAP : 0), cellH, t.accent, 2);
    }
    const mc = mcR >= 0 && mcR < R && mcC >= 0 && mcC < C ? { r: mcR, c: mcC } : null;
    const cell = shown ?? mc;
    if (cell) {
      const w = Math.max(cellW, 6);
      const h = Math.max(cellH, 6);
      box(sx + cell.c * cellW - (w - cellW) / 2, sy + cell.r * cellH - (h - cellH) / 2, w, h, t.ink, 1.5);
    }
  }, [shown, mcR, mcC, p.markRow, p.markCol, p.focusRow, br0, br1, bc0, bc1, p.rowSlots, R, C, cellW, cellH, mode, reading, dpr, Wd, Ht, sx, sy, plotW, plotH]);

  const locate = (e: RPointerEvent<HTMLCanvasElement>): Cell | null => {
    const rect = e.currentTarget.getBoundingClientRect();
    const c = Math.floor((e.clientX - rect.left - sx) / cellW);
    const r = Math.floor((e.clientY - rect.top - sy) / cellH);
    return r < 0 || c < 0 || r >= R || c >= C ? null : { r, c };
  };
  const emit = (cell: Cell | null) => {
    setHover(cell);
    p.onHover?.(cell);
  };
  const onKey = (e: KeyboardEvent<HTMLCanvasElement>) => {
    const cur = cursor ?? { r: p.focusRow !== null && p.focusRow !== undefined && p.focusRow < R ? p.focusRow : 0, c: 0 };
    let { r, c } = cur;
    if (e.key === 'ArrowRight') c = Math.min(C - 1, c + 1);
    else if (e.key === 'ArrowLeft') c = Math.max(0, c - 1);
    else if (e.key === 'ArrowDown') r = Math.min(R - 1, r + 1);
    else if (e.key === 'ArrowUp') r = Math.max(0, r - 1);
    else if (e.key === 'Enter' || e.key === ' ') {
      p.onPick?.(cur);
      e.preventDefault();
      return;
    } else if (e.key === 'Escape') {
      setCursor(null);
      p.onHover?.(null);
      return;
    } else return;
    e.preventDefault();
    setCursor({ r, c });
    p.onHover?.({ r, c });
  };

  if (R === 0 || C === 0) return null;
  return (
    <div className={cx('relative shrink-0', p.className)} style={{ width: Wd, height: Ht }}>
      <canvas ref={base} aria-hidden="true" style={{ width: Wd, height: Ht, display: 'block' }} />
      <canvas
        ref={overlay}
        tabIndex={0}
        role="img"
        aria-label={`${p.label}. ${R} by ${C}. Use the arrow keys to read entries.`}
        className="absolute inset-0 cursor-crosshair rounded-[2px] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        style={{ width: Wd, height: Ht }}
        onPointerMove={(e) => emit(locate(e))}
        onPointerLeave={() => emit(null)}
        onClick={(e) => {
          const l = locate(e as unknown as RPointerEvent<HTMLCanvasElement>);
          if (l) p.onPick?.(l);
        }}
        onKeyDown={onKey}
        onBlur={() => {
          setCursor(null);
          if (!hover) p.onHover?.(null);
        }}
      />
    </div>
  );
}
