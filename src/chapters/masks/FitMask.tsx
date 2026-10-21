/**
 * FitMask: a MaskMatrix whose cell size comes from the measured column width, with axis titles
 * ("keys →" above, "queries ↓" on the left) and a visible band around the pinned query row.
 *
 * Labels use the Heatmap's own thresholds (HEATMAP_GEOMETRY), so a matrix never shrinks its cells for
 * labels that will not be drawn. When `labelled` and the labels do not fit at the smallest legible
 * cell, the matrix keeps that cell size and scrolls sideways instead of dropping the labels.
 */
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { HEATMAP_GEOMETRY, MaskMatrix, type MaskMatrixProps } from '@/components';
import { useElementSize } from '@/components/viz/useSize';

const AXIS_W = 22;
/** Room above the matrix for the "keys →" title. */
const KEYS_H = 18;
const G = HEATMAP_GEOMETRY;

export interface FitMaskProps {
  /** Tokens shown (top-left n×n). */
  n: number;
  /** Ask for row/column labels. */
  labelled: boolean;
  maxCell: number;
  /** Query row to mark with an accent band (null: none). */
  bandRow?: number | null;
  /** Draw the "keys →" / "queries ↓" axis titles. Default true. */
  axisTitles?: boolean;
  header?: ReactNode;
  matrix: Omit<MaskMatrixProps, 'cellSize' | 'maxCellSize' | 'highlightRows'>;
}

export function FitMask({ n, labelled, maxCell, bandRow = null, axisTitles = true, header, matrix }: FitMaskProps) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const boxRef = useRef<HTMLDivElement | null>(null);
  // where the Heatmap's canvas actually sits inside our box (its inline grid can add a line-box offset)
  const [off, setOff] = useState({ x: 0, y: 0, gap: 0 });
  const w = (size.width || 480) - (axisTitles ? AXIS_W : 0);
  const N = Math.max(1, n);
  let cell = Math.max(1, Math.min(maxCell, Math.floor((w - G.strip) / N)));
  let withLabels = false;
  if (labelled && matrix.labels) {
    const c = Math.min(maxCell, Math.floor((w - G.strip - G.labelW) / N));
    cell = Math.max(c, G.minLabelCell);
    withLabels = true;
  }
  const labelW = withLabels ? G.labelW : 0;
  const plotW = labelW + G.strip + N * cell;
  const band = bandRow !== null && bandRow < N ? bandRow : null;
  // With labels the cells are ≥ 13 px and the Heatmap's own 2 px row band reads well (and scrolls with
  // the matrix); without labels the cells can be 2-3 px, so draw a band around the row instead.
  const overlay = band !== null && !withLabels;
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
      {header && <div style={{ maxWidth: plotW + (axisTitles ? AXIS_W : 0) }}>{header}</div>}
      <div className="flex min-w-0">
        {axisTitles && (
          <div className="relative shrink-0" style={{ width: AXIS_W }} aria-hidden="true">
            <span
              className="absolute left-0 text-[14px] leading-none whitespace-nowrap text-muted"
              style={{ top: off.y + G.strip, writingMode: 'vertical-rl' }}
            >
              queries →
            </span>
          </div>
        )}
        <div ref={boxRef} className="relative min-w-0 flex-1" style={{ paddingTop: axisTitles ? Math.max(0, KEYS_H - off.gap) : 0 }}>
          {axisTitles && (
            <div
              className="absolute text-[14px] leading-none whitespace-nowrap text-muted"
              style={{ top: Math.max(0, off.gap - KEYS_H), left: labelW + G.strip }}
              aria-hidden="true"
            >
              keys →
            </div>
          )}
          <MaskMatrix
            {...matrix}
            cellSize={cell}
            labels={withLabels ? matrix.labels : undefined}
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
        </div>
      </div>
    </div>
  );
}
