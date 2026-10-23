/**
 * FitMask: a MaskMatrix whose cell size comes from the measured column width, with axis titles
 * ("keys →" above, "queries →" on the left), optional position ticks, outlined "marks" and a visible
 * band around the pinned query row.
 *
 * Labels use the Heatmap's own thresholds (HEATMAP_GEOMETRY) and label bands sized to the longest
 * label (measured, so no label is truncated). When `labeled` and the labels do not fit at the
 * smallest legible cell, the matrix either keeps that cell size and scrolls sideways (default) or,
 * with `fitLabels`, drops the labels and shows position ticks instead, so it never scrolls.
 */
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { HEATMAP_GEOMETRY, MaskMatrix, type MaskMatrixProps } from '@/components';
import { maxTextWidth } from '@/components/viz/text';
import { useElementSize } from '@/components/viz/useSize';

const AXIS_W = 22;
/** Room above the matrix for the "keys →" title. */
const KEYS_H = 18;
/** Tick label band: height above the matrix, width left of it. */
const TICK_H = 18;
const TICK_W = 30;
const TICK_PX = 13;
/** Label font the bands are measured for (the Heatmap draws 13 px labels on ≥ 15 px cells, else 12 px). */
const LABEL_PX = 13;
const G = HEATMAP_GEOMETRY;

export interface FitMaskProps {
  /** Tokens shown (top-left n×n). */
  n: number;
  /** Ask for row/column labels. */
  labeled: boolean;
  maxCell: number;
  /** Query row to mark with an accent band (null: none). */
  bandRow?: number | null;
  /** Draw the "keys →" / "queries →" axis titles. Default true. */
  axisTitles?: boolean;
  /** Without labels: print the position every `ticks` tokens on both axes. */
  ticks?: number;
  /** Drop the labels (and show ticks) instead of scrolling when they do not fit. */
  fitLabels?: boolean;
  /** Smallest cell that may carry labels. Default the Heatmap's (13 px → 12 px labels); 15 gives 13 px labels. */
  minLabelCell?: number;
  /** Cells to outline, as q·n + k in the n×n view (e.g. pairs that differ from another mask). */
  marks?: readonly number[];
  header?: ReactNode;
  matrix: Omit<MaskMatrixProps, 'cellSize' | 'maxCellSize' | 'highlightRows' | 'labelWidth' | 'labelHeight'>;
}

export function FitMask({
  n,
  labeled,
  maxCell,
  bandRow = null,
  axisTitles = true,
  ticks,
  fitLabels = false,
  minLabelCell = G.minLabelCell,
  marks,
  header,
  matrix,
}: FitMaskProps) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const boxRef = useRef<HTMLDivElement | null>(null);
  // where the Heatmap's canvas actually sits inside our box (its inline grid can add a line-box offset)
  const [off, setOff] = useState({ x: 0, y: 0, gap: 0 });
  const w0 = (size.width || 480) - (axisTitles ? AXIS_W : 0);
  const N = Math.max(1, n);

  // label bands sized to the longest label actually shown
  const shownLabels = matrix.labels?.slice(0, N) ?? [];
  const labelW = shownLabels.length ? Math.min(230, Math.max(60, Math.ceil(maxTextWidth(shownLabels, LABEL_PX) + 14))) : G.labelW;
  const labelH = shownLabels.length ? Math.min(190, Math.max(40, Math.ceil(maxTextWidth(shownLabels, LABEL_PX) + 10))) : G.labelH;

  let withLabels = false;
  let cell = 1;
  if (labeled && matrix.labels) {
    const c = Math.min(maxCell, Math.floor((w0 - G.strip - labelW) / N));
    if (c >= minLabelCell) {
      cell = c;
      withLabels = true;
    } else if (!fitLabels) {
      cell = minLabelCell;
      withLabels = true;
    }
  }
  const showTicks = !withLabels && !!ticks;
  if (!withLabels) cell = Math.max(1, Math.min(maxCell, Math.floor((w0 - (showTicks ? TICK_W : 0) - G.strip) / N)));
  const bandL = withLabels ? labelW : 0;
  const plotW = bandL + G.strip + N * cell;
  const leftW = (axisTitles ? AXIS_W : 0) + (showTicks ? TICK_W : 0);
  const topH = (axisTitles ? KEYS_H : 0) + (showTicks ? TICK_H : 0);
  const band = bandRow !== null && bandRow < N ? bandRow : null;
  // With labels the cells are ≥ 13 px and the Heatmap's own 2 px row band reads well (and scrolls with
  // the matrix); without labels the cells can be 2-3 px, so draw a band around the row instead.
  const overlay = band !== null && !withLabels;
  const tickList = showTicks ? Array.from({ length: Math.ceil(N / ticks!) }, (_, i) => i * ticks!) : [];
  useLayoutEffect(() => {
    const box = boxRef.current;
    const cv = box?.querySelector('canvas');
    if (!box || !cv) return;
    const a = box.getBoundingClientRect();
    const b = cv.getBoundingClientRect();
    const grid = cv.closest('.inline-grid')?.getBoundingClientRect();
    const fig = cv.closest('figure')?.getBoundingClientRect();
    const gap = grid && fig ? Math.round(grid.top - fig.top) : 0;
    const next = { x: Math.round(b.left - a.left), y: Math.round(b.top - a.top), gap };
    setOff((o) => (o.x === next.x && o.y === next.y && o.gap === next.gap ? o : next));
  });
  return (
    <div ref={ref} className="min-w-0">
      {header && <div style={{ maxWidth: plotW + leftW }}>{header}</div>}
      <div className="flex min-w-0">
        {leftW > 0 && (
          <div className="relative shrink-0" style={{ width: leftW }} aria-hidden="true">
            {axisTitles && (
              <span
                className="absolute left-0 text-[14px] leading-none whitespace-nowrap text-muted"
                style={{ top: off.y + G.strip, writingMode: 'vertical-rl' }}
              >
                queries →
              </span>
            )}
            {tickList.map((t) => (
              <span
                key={t}
                className="num absolute text-right leading-none text-muted"
                style={{ right: 4, top: off.y + G.strip + t * cell + cell / 2, transform: 'translateY(-50%)', fontSize: TICK_PX }}
              >
                {t}
              </span>
            ))}
          </div>
        )}
        <div ref={boxRef} className="relative min-w-0 flex-1" style={{ paddingTop: Math.max(0, topH - off.gap) }}>
          {axisTitles && (
            <div
              className="absolute text-[14px] leading-none whitespace-nowrap text-muted"
              style={{ top: Math.max(0, off.gap - topH), left: bandL + G.strip }}
              aria-hidden="true"
            >
              keys →
            </div>
          )}
          {tickList.map((t) => (
            <span
              key={t}
              aria-hidden="true"
              className="num absolute leading-none text-muted"
              style={{ left: off.x + G.strip + t * cell + cell / 2, top: off.y - TICK_H + 3, transform: 'translateX(-50%)', fontSize: TICK_PX }}
            >
              {t}
            </span>
          ))}
          <MaskMatrix
            {...matrix}
            cellSize={cell}
            labels={withLabels ? matrix.labels : undefined}
            labelWidth={labelW}
            labelHeight={labelH}
            highlightRows={band !== null && withLabels ? [band] : []}
          />
          {overlay && (
            <div
              aria-hidden="true"
              className="pointer-events-none absolute rounded-[2px]"
              style={{
                left: off.x + G.strip - 3,
                top: off.y + G.strip + band * cell - 3,
                width: N * cell + 6,
                height: cell + 6,
                border: '2px solid var(--accent)',
              }}
            />
          )}
          {marks && marks.length > 0 && (
            <svg
              aria-hidden="true"
              className="pointer-events-none absolute overflow-visible"
              style={{ left: off.x + G.strip, top: off.y + G.strip }}
              width={N * cell}
              height={N * cell}
            >
              {marks.map((m) => {
                const q = Math.floor(m / N);
                const k = m % N;
                return (
                  <rect
                    key={m}
                    x={k * cell + 1.25}
                    y={q * cell + 1.25}
                    width={Math.max(1, cell - 2.5)}
                    height={Math.max(1, cell - 2.5)}
                    rx={1.5}
                    fill="none"
                    style={{ stroke: 'var(--critical)' }}
                    strokeWidth={2}
                  />
                );
              })}
            </svg>
          )}
        </div>
      </div>
    </div>
  );
}
