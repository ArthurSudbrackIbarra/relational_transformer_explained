/**
 * Heatmap: canvas S×S (or H stacked S×S panels) with hover readout, row/column
 * highlighting, optional axis labels and table-identity strips.
 *
 * Built for attention maps (AttnCapture.probs is [H*S*S], query-major:
 * data[(h*S + q)*S + k]) and masks (MaskMatrix wraps this). Handles 128×128
 * per panel and H stacks; draws through a 1-pixel-per-cell ImageData scaled
 * with smoothing off, so it stays fast.
 *
 * Colour (dataviz): sequential = one blue hue, near-zero recedes to the
 * surface; diverging = blue↔red with gray midpoint; binary = one sequential
 * step vs the surface. A scale legend is always drawn; values are reachable
 * without hover through the readout line and keyboard focus.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { catHex, TOKENS } from '@/styles/palette';
import { useResolvedTheme } from '@/styles/theme';
import { cx } from '../ui/cx';
import { formatNumber, formatSigned } from './format';
import { divergingRgb, maxAbs as maxAbsOf, rgbCss, sequentialRgb, type RGB } from './scales';
import { toSet, useElementSize } from './useSize';

export interface HeatmapCell {
  stack: number;
  row: number;
  col: number;
  value: number;
  blocked: boolean;
}

export interface HeatmapProps {
  /** Values, row-major per panel, panels concatenated: data[(s*rows + r)*cols + c]. */
  data: ArrayLike<number>;
  rows: number;
  cols: number;
  /** Number of stacked panels (e.g. attention heads). Default 1. */
  stacks?: number;
  /** Panel titles, e.g. ["head 0", …]. */
  stackLabels?: string[];
  /** Colour scale. Default 'sequential'. */
  scale?: 'sequential' | 'diverging' | 'binary';
  /** Sequential: [min, max] (default [0, max]). Diverging: max |v| comes from domain[1] (default data max |v|). */
  domain?: [number, number];
  /**
   * Cells to render as blocked (drawn as empty surface, readout says "masked").
   * Length rows*cols (shared by all panels) or stacks*rows*cols. 1 = blocked.
   */
  blocked?: ArrayLike<number>;
  /** Only show the top-left n×n (e.g. the non-padding tokens). Default: all. */
  crop?: { rows?: number; cols?: number };
  rowLabels?: string[];
  colLabels?: string[];
  /** Categorical slot per row / column: drawn as a thin identity strip on the axis. */
  rowSlots?: number[];
  colSlots?: number[];
  /** Draw text labels when cells are large enough (≥ 13 px: 12px text; ≥ 15 px: 13px text). Default true. */
  showLabels?: boolean;
  highlightRows?: ReadonlySet<number> | readonly number[];
  highlightCols?: ReadonlySet<number> | readonly number[];
  /** Outlined cell. */
  selected?: { row: number; col: number } | null;
  /** Hover / keyboard-focus callback (null on leave). */
  onHover?: (cell: HeatmapCell | null) => void;
  onCellClick?: (cell: HeatmapCell) => void;
  /** Fixed cell size in CSS px. Default: fit the container width. */
  cellSize?: number;
  maxCellSize?: number;
  minCellSize?: number;
  /** Panels per row when stacks > 1 (default: as many as fit at ≥ minCellSize). */
  panelsPerRow?: number;
  formatValue?: (v: number) => string;
  /** Axis names for the readout, e.g. "query" / "key". */
  rowName?: string;
  colName?: string;
  /** Accessible name. */
  title: string;
  /** Hide the readout line / legend. */
  showReadout?: boolean;
  showLegend?: boolean;
  /** Legend captions for binary scale. */
  binaryLabels?: [string, string];
  className?: string;
  /** Extra content right of the legend (e.g. a note). */
  legendExtra?: ReactNode;
  /** Fill of blocked cells as '#rrggbb' (default: the theme's surface-2). */
  blockedFill?: string;
}

const STRIP = 5; // identity strip thickness
const STRIP_GAP = 2;
const LABEL_W = 140;
const LABEL_H = 104;
/** Smallest cell that can carry a text label without labels overlapping (12px text). */
const MIN_LABEL_CELL = 13;
/** Label font for a cell size: 13px when there is room, never below 12px. */
const labelPx = (cell: number) => (cell >= 15 ? 13 : 12);
const PANEL_GAP = 20;

/**
 * Layout constants, for callers that size a Heatmap themselves (e.g. pick a cell size that leaves
 * room for labels). `strip` is the identity strip plus its gap.
 */
export const HEATMAP_GEOMETRY = { strip: STRIP + STRIP_GAP, labelW: LABEL_W, labelH: LABEL_H, minLabelCell: MIN_LABEL_CELL } as const;

/** Canvas heatmap with hover readout. See HeatmapProps. */
export function Heatmap(props: HeatmapProps) {
  const {
    data,
    rows,
    cols,
    stacks = 1,
    stackLabels,
    scale = 'sequential',
    domain,
    blocked,
    crop,
    rowLabels,
    colLabels,
    rowSlots,
    colSlots,
    showLabels = true,
    highlightRows,
    highlightCols,
    selected,
    onHover,
    onCellClick,
    cellSize,
    maxCellSize = 22,
    minCellSize = 2,
    panelsPerRow,
    formatValue,
    rowName = 'row',
    colName = 'col',
    title,
    showReadout = true,
    showLegend = true,
    binaryLabels = ['allowed', 'blocked'],
    className,
    legendExtra,
    blockedFill,
  } = props;

  const mode = useResolvedTheme();
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const R = Math.min(rows, crop?.rows ?? rows);
  const C = Math.min(cols, crop?.cols ?? cols);

  // Layout: panels per row and cell size.
  const hasRowStrip = !!rowSlots;
  const hasColStrip = !!colSlots;
  const width = size.width || 640;
  const layout = useMemo(() => {
    const stripW = hasRowStrip ? STRIP + STRIP_GAP : 0;
    const fit = (ppr: number, labelW: number) =>
      Math.floor((width - PANEL_GAP * (ppr - 1)) / ppr - labelW - stripW) / C;
    let ppr = panelsPerRow ?? 1;
    if (!panelsPerRow && stacks > 1) {
      ppr = Math.max(1, Math.min(stacks, Math.floor((width + PANEL_GAP) / (C * Math.max(minCellSize, 3) + stripW + PANEL_GAP))));
    }
    let cell = cellSize ?? Math.floor(Math.min(maxCellSize, Math.max(minCellSize, fit(ppr, 0))));
    const wantLabels = showLabels && (!!rowLabels || !!colLabels);
    let labels = wantLabels && cell >= MIN_LABEL_CELL;
    if (labels && !cellSize) {
      cell = Math.floor(Math.min(maxCellSize, Math.max(minCellSize, fit(ppr, rowLabels ? LABEL_W : 0))));
      labels = cell >= MIN_LABEL_CELL;
      if (!labels) cell = Math.floor(Math.min(maxCellSize, Math.max(minCellSize, fit(ppr, 0))));
    }
    return { ppr, cell: Math.max(1, cell), labels, stripW };
  }, [width, C, stacks, panelsPerRow, cellSize, maxCellSize, minCellSize, showLabels, rowLabels, colLabels, hasRowStrip]);

  const hlRows = toSet(highlightRows);
  const hlCols = toSet(highlightCols);

  const domainResolved = useMemo<[number, number]>(() => {
    if (domain) return domain;
    if (scale === 'diverging') {
      const m = maxAbsOf(data);
      return [-m, m || 1];
    }
    if (scale === 'binary') return [0, 1];
    let hi = 0;
    for (let i = 0; i < data.length; i++) if (Number.isFinite(data[i]) && data[i] > hi) hi = data[i];
    return [0, hi || 1];
  }, [data, domain, scale]);

  const isBlocked = useCallback(
    (s: number, r: number, c: number) => {
      if (!blocked) return false;
      const per = rows * cols;
      const idx = blocked.length >= per * stacks ? s * per + r * cols + c : r * cols + c;
      return !!blocked[idx];
    },
    [blocked, rows, cols, stacks],
  );

  const colorOf = useCallback(
    (v: number): RGB => {
      if (scale === 'diverging') return divergingRgb(v, Math.max(Math.abs(domainResolved[0]), Math.abs(domainResolved[1])), mode);
      if (scale === 'binary') return v > 0 ? sequentialRgb(mode === 'dark' ? 0.62 : 0.66, mode) : hexRgb(TOKENS[mode].surface2);
      const [lo, hi] = domainResolved;
      return sequentialRgb(hi > lo ? (v - lo) / (hi - lo) : 0, mode);
    },
    [scale, domainResolved, mode],
  );

  const [hover, setHover] = useState<HeatmapCell | null>(null);
  const [focusCell, setFocusCell] = useState<{ stack: number; row: number; col: number } | null>(null);

  const cellAt = useCallback(
    (s: number, r: number, c: number): HeatmapCell => ({
      stack: s,
      row: r,
      col: c,
      value: Number(data[(s * rows + r) * cols + c]),
      blocked: isBlocked(s, r, c),
    }),
    [data, rows, cols, isBlocked],
  );

  const emitHover = useCallback(
    (cell: HeatmapCell | null) => {
      setHover(cell);
      onHover?.(cell);
    },
    [onHover],
  );

  const fmt = useCallback(
    (cell: HeatmapCell) => {
      if (cell.blocked) return 'masked (−∞ before softmax)';
      if (formatValue) return formatValue(cell.value);
      if (scale === 'binary') return cell.value > 0 ? binaryLabels[0] : binaryLabels[1];
      if (scale === 'diverging') return formatSigned(cell.value, 3);
      return formatNumber(cell.value, 4);
    },
    [formatValue, scale, binaryLabels],
  );

  const panels = Array.from({ length: stacks }, (_, s) => s);
  const { cell, labels, stripW } = layout;
  const plotW = C * cell;
  const plotH = R * cell;

  return (
    <div ref={wrapRef} className={cx('w-full min-w-0', className)}>
      <div className="scroll-quiet flex flex-wrap gap-x-5 gap-y-4 overflow-x-auto" role="group" aria-label={title}>
        {panels.map((s) => (
          <HeatmapPanel
            key={s}
            stack={s}
            title={stacks > 1 ? stackLabels?.[s] ?? `panel ${s}` : undefined}
            ariaLabel={`${title}${stacks > 1 ? `, ${stackLabels?.[s] ?? `panel ${s}`}` : ''}`}
            R={R}
            C={C}
            cell={cell}
            plotW={plotW}
            plotH={plotH}
            stripW={stripW}
            labels={labels}
            rowLabels={rowLabels}
            colLabels={colLabels}
            rowSlots={rowSlots}
            colSlots={colSlots}
            mode={mode}
            draw={(ctx) => {
              const img = ctx.createImageData(C, R);
              const surface2 = hexRgb(blockedFill ?? TOKENS[mode].surface2);
              for (let r = 0; r < R; r++) {
                for (let c = 0; c < C; c++) {
                  const v = Number(data[(s * rows + r) * cols + c]);
                  const rgb = isBlocked(s, r, c) ? surface2 : colorOf(v);
                  const o = (r * C + c) * 4;
                  img.data[o] = rgb[0];
                  img.data[o + 1] = rgb[1];
                  img.data[o + 2] = rgb[2];
                  img.data[o + 3] = 255;
                }
              }
              return img;
            }}
            drawKey={`${mode}|${scale}|${domainResolved.join(',')}|${R}|${C}|${s}|${blockedFill ?? ''}`}
            data={data}
            blocked={blocked}
            hlRows={hlRows}
            hlCols={hlCols}
            selected={selected ?? null}
            hover={hover && hover.stack === s ? hover : null}
            focusCell={focusCell && focusCell.stack === s ? focusCell : null}
            onMove={(r, c) => emitHover(r === null || c === null ? null : cellAt(s, r, c))}
            onClick={(r, c) => onCellClick?.(cellAt(s, r, c))}
            onKey={(e) => {
              const cur = focusCell && focusCell.stack === s ? focusCell : { stack: s, row: 0, col: 0 };
              let { row, col } = cur;
              if (e.key === 'ArrowRight') col = Math.min(C - 1, col + 1);
              else if (e.key === 'ArrowLeft') col = Math.max(0, col - 1);
              else if (e.key === 'ArrowDown') row = Math.min(R - 1, row + 1);
              else if (e.key === 'ArrowUp') row = Math.max(0, row - 1);
              else if (e.key === 'Enter' || e.key === ' ') {
                onCellClick?.(cellAt(s, row, col));
                e.preventDefault();
                return;
              } else if (e.key === 'Escape') {
                setFocusCell(null);
                emitHover(null);
                return;
              } else return;
              e.preventDefault();
              setFocusCell({ stack: s, row, col });
              emitHover(cellAt(s, row, col));
            }}
            onBlur={() => {
              setFocusCell(null);
            }}
          />
        ))}
      </div>

      {(showReadout || showLegend) && (
        <div className="mt-3 flex flex-wrap items-center justify-between gap-x-6 gap-y-2 text-[15.5px] text-ink-2">
          {showReadout && (
            <div className="num min-h-[1.4em] min-w-0" aria-live="polite">
              {hover ? (
                <>
                  {stacks > 1 && <span className="text-muted">{stackLabels?.[hover.stack] ?? `panel ${hover.stack}`} · </span>}
                  <span className="text-muted">{rowName} </span>
                  <span className="text-ink">{rowLabels?.[hover.row] ?? hover.row}</span>
                  <span className="text-muted"> · {colName} </span>
                  <span className="text-ink">{colLabels?.[hover.col] ?? hover.col}</span>
                  <span className="text-muted"> → </span>
                  <strong className="font-semibold text-ink">{fmt(hover)}</strong>
                </>
              ) : (
                <span className="text-muted">Hover or focus a cell to read its value.</span>
              )}
            </div>
          )}
          {showLegend && (
            <div className="flex items-center gap-3">
              <ScaleLegend scale={scale} domain={domainResolved} mode={mode} binaryLabels={binaryLabels} hasBlocked={!!blocked} />
              {legendExtra}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function hexRgb(hex: string): RGB {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

// ─────────────────────────────────────────────────────────────────────────────

interface PanelProps {
  stack: number;
  title?: string;
  ariaLabel: string;
  R: number;
  C: number;
  cell: number;
  plotW: number;
  plotH: number;
  stripW: number;
  labels: boolean;
  rowLabels?: string[];
  colLabels?: string[];
  rowSlots?: number[];
  colSlots?: number[];
  mode: 'light' | 'dark';
  draw: (ctx: CanvasRenderingContext2D) => ImageData;
  drawKey: string;
  data: ArrayLike<number>;
  blocked?: ArrayLike<number>;
  hlRows: ReadonlySet<number>;
  hlCols: ReadonlySet<number>;
  selected: { row: number; col: number } | null;
  hover: HeatmapCell | null;
  focusCell: { row: number; col: number } | null;
  onMove: (r: number | null, c: number | null) => void;
  onClick: (r: number, c: number) => void;
  onKey: (e: KeyboardEvent<HTMLCanvasElement>) => void;
  onBlur: () => void;
}

function HeatmapPanel(p: PanelProps) {
  const base = useRef<HTMLCanvasElement | null>(null);
  const overlay = useRef<HTMLCanvasElement | null>(null);
  const { R, C, cell, plotW, plotH, stripW, mode } = p;
  const colStripH = p.colSlots ? STRIP + STRIP_GAP : 0;
  const dpr = typeof window !== 'undefined' ? Math.min(2, window.devicePixelRatio || 1) : 1;
  const W = stripW + plotW;
  const H = colStripH + plotH;

  // Base layer: cells (+ identity strips). Redrawn when data/theme/layout change.
  useEffect(() => {
    const cv = base.current;
    if (!cv || R === 0 || C === 0) return;
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    const img = p.draw(ctx);
    const off = document.createElement('canvas');
    off.width = C;
    off.height = R;
    off.getContext('2d')!.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, stripW * dpr, colStripH * dpr, plotW * dpr, plotH * dpr);
    // 1px surface gap between larger cells (dataviz "surface gap")
    if (cell >= 8) {
      ctx.fillStyle = TOKENS[mode].surface;
      for (let i = 1; i < C; i++) ctx.fillRect((stripW + i * cell) * dpr - dpr * 0.5, colStripH * dpr, dpr, plotH * dpr);
      for (let j = 1; j < R; j++) ctx.fillRect(stripW * dpr, (colStripH + j * cell) * dpr - dpr * 0.5, plotW * dpr, dpr);
    }
    // identity strips
    if (p.rowSlots) {
      for (let r = 0; r < R; r++) {
        ctx.fillStyle = catHex(p.rowSlots[r], mode);
        ctx.fillRect(0, (colStripH + r * cell) * dpr, STRIP * dpr, Math.max(1, cell - (cell >= 8 ? 1 : 0)) * dpr);
      }
    }
    if (p.colSlots) {
      for (let c = 0; c < C; c++) {
        ctx.fillStyle = catHex(p.colSlots[c], mode);
        ctx.fillRect((stripW + c * cell) * dpr, 0, Math.max(1, cell - (cell >= 8 ? 1 : 0)) * dpr, STRIP * dpr);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.drawKey, p.data, p.blocked, W, H, cell, dpr, p.rowSlots, p.colSlots]);

  // Overlay: highlights, selection, hover crosshair.
  useEffect(() => {
    const cv = overlay.current;
    if (!cv) return;
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const accent = TOKENS[mode].accent;
    const ink = TOKENS[mode].ink;
    const x0 = stripW;
    const y0 = colStripH;
    const band = (x: number, y: number, w: number, h: number, color: string, lw: number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.strokeRect(x + lw / 2, y + lw / 2, Math.max(0, w - lw), Math.max(0, h - lw));
    };
    for (const r of p.hlRows) if (r < R) band(x0, y0 + r * cell, plotW, cell, accent, Math.min(2, Math.max(1, cell / 4)));
    for (const c of p.hlCols) if (c < C) band(x0 + c * cell, y0, cell, plotH, accent, Math.min(2, Math.max(1, cell / 4)));
    const h = p.hover ?? (p.focusCell ? { row: p.focusCell.row, col: p.focusCell.col } : null);
    if (h) {
      ctx.fillStyle = mode === 'dark' ? 'rgba(255,255,255,0.07)' : 'rgba(11,11,11,0.05)';
      ctx.fillRect(x0, y0 + h.row * cell, plotW, cell);
      ctx.fillRect(x0 + h.col * cell, y0, cell, plotH);
      const s = Math.max(cell, 6);
      const off = (s - cell) / 2;
      band(x0 + h.col * cell - off, y0 + h.row * cell - off, s, s, ink, 1.5);
    }
    if (p.selected && p.selected.row < R && p.selected.col < C) {
      const s = Math.max(cell, 7);
      const off = (s - cell) / 2;
      band(x0 + p.selected.col * cell - off, y0 + p.selected.row * cell - off, s, s, accent, 2);
    }
  }, [p.hlRows, p.hlCols, p.hover, p.focusCell, p.selected, W, H, cell, dpr, mode, R, C, plotW, plotH, stripW, colStripH]);

  const locate = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left - stripW;
    const y = e.clientY - rect.top - colStripH;
    const c = Math.floor(x / cell);
    const r = Math.floor(y / cell);
    if (r < 0 || c < 0 || r >= R || c >= C) return null;
    return { r, c };
  };

  const showRowLabels = p.labels && !!p.rowLabels;
  const showColLabels = p.labels && !!p.colLabels;

  return (
    <figure className="m-0 min-w-0">
      {p.title && <figcaption className="mb-1.5 text-[15px] font-medium text-ink-2">{p.title}</figcaption>}
      <div className="inline-grid" style={{ gridTemplateColumns: `${showRowLabels ? LABEL_W : 0}px auto` }}>
        <div />
        {showColLabels ? (
          <div className="relative overflow-hidden" style={{ height: LABEL_H, width: W }} aria-hidden="true">
            {p.colLabels!.slice(0, C).map((l, c) => (
              <span
                key={c}
                className="num absolute overflow-hidden text-ellipsis whitespace-nowrap text-ink-2"
                style={{
                  fontSize: labelPx(cell),
                  left: stripW + c * cell,
                  width: cell,
                  height: LABEL_H - 4,
                  bottom: 3,
                  lineHeight: `${cell}px`,
                  writingMode: 'vertical-rl',
                  transform: 'rotate(180deg)',
                  textAlign: 'start',
                }}
                title={l}
              >
                {l}
              </span>
            ))}
          </div>
        ) : (
          <div />
        )}
        {showRowLabels ? (
          <div className="relative" style={{ height: H }} aria-hidden="true">
            {p.rowLabels!.slice(0, R).map((l, r) => (
              <span
                key={r}
                className="num absolute right-2 truncate text-right text-ink-2"
                style={{ top: colStripH + r * cell, width: LABEL_W - 10, height: cell, lineHeight: `${cell}px`, fontSize: labelPx(cell) }}
                title={l}
              >
                {l}
              </span>
            ))}
          </div>
        ) : (
          <div />
        )}
        <div className="relative" style={{ width: W, height: H }}>
          <canvas ref={base} aria-hidden="true" style={{ width: W, height: H, display: 'block' }} />
          <canvas
            ref={overlay}
            tabIndex={0}
            role="img"
            aria-label={`${p.ariaLabel}. ${R} by ${C}. Use arrow keys to read cells.`}
            className="absolute inset-0 cursor-crosshair rounded-[2px]"
            style={{ width: W, height: H }}
            onPointerMove={(e) => {
              const l = locate(e);
              p.onMove(l ? l.r : null, l ? l.c : null);
            }}
            onPointerLeave={() => p.onMove(null, null)}
            onClick={(e) => {
              const l = locate(e as unknown as React.PointerEvent<HTMLCanvasElement>);
              if (l) p.onClick(l.r, l.c);
            }}
            onKeyDown={p.onKey}
            onBlur={p.onBlur}
          />
        </div>
      </div>
    </figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

/** Small scale legend (sequential ramp / diverging ramp / binary swatches). */
export function ScaleLegend({
  scale,
  domain,
  mode,
  binaryLabels = ['allowed', 'blocked'],
  hasBlocked = false,
}: {
  scale: 'sequential' | 'diverging' | 'binary';
  domain: [number, number];
  mode: 'light' | 'dark';
  binaryLabels?: [string, string];
  hasBlocked?: boolean;
}) {
  if (scale === 'binary') {
    return (
      <div className="flex items-center gap-3 text-[15px] text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3.5 w-3.5 rounded-[3px]" style={{ background: rgbCss(sequentialRgb(mode === 'dark' ? 0.62 : 0.66, mode)) }} />
          {binaryLabels[0]}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3.5 w-3.5 rounded-[3px] border border-border" style={{ background: TOKENS[mode].surface2 }} />
          {binaryLabels[1]}
        </span>
      </div>
    );
  }
  const stops = Array.from({ length: 11 }, (_, i) => i / 10);
  const grad =
    scale === 'diverging'
      ? stops.map((t) => rgbCss(divergingRgb(t * 2 - 1, 1, mode))).join(', ')
      : stops.map((t) => rgbCss(sequentialRgb(t, mode))).join(', ');
  const m = Math.max(Math.abs(domain[0]), Math.abs(domain[1]));
  return (
    <div className="flex items-center gap-2 text-[14.5px] text-ink-2">
      {hasBlocked && (
        <span className="mr-1 inline-flex items-center gap-1.5">
          <span className="inline-block h-3.5 w-3.5 rounded-[3px] border border-border" style={{ background: TOKENS[mode].surface2 }} />
          masked
        </span>
      )}
      <span className="num">{scale === 'diverging' ? formatSigned(-m, 2) : formatNumber(domain[0], 2)}</span>
      <span className="inline-block h-2.5 w-28 rounded-full" style={{ background: `linear-gradient(90deg, ${grad})` }} aria-hidden="true" />
      <span className="num">{scale === 'diverging' ? formatSigned(m, 2) : formatNumber(domain[1], 2)}</span>
    </div>
  );
}
