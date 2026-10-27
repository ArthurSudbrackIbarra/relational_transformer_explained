/**
 * VectorStrip: one d-dimensional vector as a strip of diverging-color cells
 * (blue < 0 < red, gray midpoint) with hover values. Wraps into rows of
 * `columns` cells (e.g. 64 per row for a 384-d MiniLM embedding).
 *
 * Optional `ranges` outline contiguous slices (e.g. head j = dims [j·d_head, (j+1)·d_head), spec §6.1).
 * The readout line starts with the vector's ShapeTag (1 × d) and what its columns are.
 */
import { useMemo, useState } from 'react';
import { useResolvedTheme } from '@/styles/theme';
import { cx } from '../ui/cx';
import { formatSigned } from './format';
import { ShapeTag, type Shape } from './ShapeTag';
import { divergingCss, maxAbs } from './scales';
import { useElementSize } from './useSize';

export interface VectorStripProps {
  values: ArrayLike<number>;
  /** Label shown left/above, e.g. "x₀ (target)". */
  label?: string;
  /** Cells per row (default min(d, 64)). */
  columns?: number;
  /** Cell height px (default 14). Width fits the container. */
  cellHeight?: number;
  /** Symmetric color domain max |v| (default: this vector's max |v|). Share it to compare vectors. */
  domain?: number;
  /** Outlined slices [start, end). */
  ranges?: { start: number; end: number; label?: string }[];
  onHover?: (i: number | null, v: number | null) => void;
  /** Highlight one index. */
  highlightIndex?: number | null;
  /** Show the hover readout line (default true). */
  showReadout?: boolean;
  /** Show the norm in the readout line. */
  showNorm?: boolean;
  /** Show the "hover a cell" hint in the idle readout (default true). */
  hint?: boolean;
  /** What the vector and its columns are, for the ShapeTag; null only in component demos. */
  shape: Shape | null;
  className?: string;
}

export function VectorStrip({
  values,
  label,
  columns,
  cellHeight = 14,
  domain,
  ranges,
  onHover,
  highlightIndex = null,
  showReadout = true,
  showNorm = true,
  hint = true,
  shape,
  className,
}: VectorStripProps) {
  const mode = useResolvedTheme();
  const [ref, size] = useElementSize<HTMLDivElement>();
  const d = values.length;
  const cols = Math.max(1, Math.min(d, columns ?? 64));
  const nRows = Math.ceil(d / cols);
  const m = domain ?? (maxAbs(values) || 1);
  const width = Math.max(cols, size.width || cols * 8);
  const cw = width / cols;
  const H = nRows * (cellHeight + 2) - 2;
  const [hover, setHover] = useState<number | null>(null);

  const norm = useMemo(() => {
    let s = 0;
    for (let i = 0; i < d; i++) s += values[i] * values[i];
    return Math.sqrt(s);
  }, [values, d]);

  const cells = useMemo(() => {
    const out: { x: number; y: number; fill: string; i: number }[] = [];
    for (let i = 0; i < d; i++) {
      const r = Math.floor(i / cols);
      const c = i % cols;
      out.push({ x: c * cw, y: r * (cellHeight + 2), fill: divergingCss(values[i], m, mode), i });
    }
    return out;
  }, [values, d, cols, cw, cellHeight, m, mode]);

  const set = (i: number | null) => {
    setHover(i);
    onHover?.(i, i === null ? null : values[i]);
  };

  const gap = cw >= 6 ? 1 : 0;
  const shown = hover ?? highlightIndex;
  const wrapNote = nRows > 1 ? `Drawn as ${nRows} lines of ${cols} cells to fit: read left to right, then top to bottom.` : null;
  const tag = shape ? (
    <ShapeTag rows={1} cols={d} shape={wrapNote ? { ...shape, note: shape.note ? <>{shape.note} {wrapNote}</> : wrapNote } : shape} />
  ) : null;

  return (
    <div className={cx('min-w-0', className)}>
      {label && <div className="mb-1.5 text-[15px] text-ink-2">{label}</div>}
      <div ref={ref} className="w-full">
        <svg
          width="100%"
          height={H}
          viewBox={`0 0 ${width} ${H}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={`${label ?? 'vector'} with ${d} dimensions, norm ${norm.toFixed(3)}`}
          className="block cursor-crosshair touch-none"
          onPointerMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * width;
            const y = e.clientY - rect.top;
            const c = Math.floor(x / cw);
            const r = Math.floor(y / (cellHeight + 2));
            const i = r * cols + c;
            set(i >= 0 && i < d && c < cols ? i : null);
          }}
          onPointerLeave={() => set(null)}
        >
          {cells.map((c) => (
            <rect key={c.i} x={c.x} y={c.y} width={Math.max(0.5, cw - gap)} height={cellHeight} style={{ fill: c.fill }} />
          ))}
          {ranges?.map((rg, k) => {
            // outline each row-segment of the range
            const segs: { x: number; y: number; w: number }[] = [];
            for (let r = Math.floor(rg.start / cols); r <= Math.floor((rg.end - 1) / cols); r++) {
              const a = Math.max(rg.start, r * cols);
              const b = Math.min(rg.end, (r + 1) * cols);
              segs.push({ x: (a % cols) * cw, y: r * (cellHeight + 2), w: (b - a) * cw - gap });
            }
            return segs.map((s, j) => (
              <rect
                key={`${k}-${j}`}
                x={s.x + 0.75}
                y={s.y + 0.75}
                width={Math.max(0, s.w - 1.5)}
                height={cellHeight - 1.5}
                fill="none"
                style={{ stroke: 'var(--accent)' }}
                strokeWidth={1.5}
                vectorEffect="non-scaling-stroke"
              />
            ));
          })}
          {shown !== null && shown < d && (
            <rect
              x={(shown % cols) * cw - 0.5}
              y={Math.floor(shown / cols) * (cellHeight + 2) - 0.5}
              width={cw - gap + 1}
              height={cellHeight + 1}
              fill="none"
              style={{ stroke: 'var(--ink)' }}
              strokeWidth={1.5}
              vectorEffect="non-scaling-stroke"
            />
          )}
        </svg>
      </div>
      {showReadout ? (
        <div className="num mt-1.5 flex min-h-[1.3em] flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5 text-[14.5px] text-ink-2" aria-live="polite">
          <span className="inline-flex min-w-0 flex-wrap items-baseline gap-x-2.5">
            {tag}
            {shown !== null && shown < d ? (
              <span>
                <span className="text-muted">dim </span>
                <span className="text-ink">{shown}</span>
                <span className="text-muted"> = </span>
                <strong className="font-semibold text-ink">{formatSigned(values[shown], 4)}</strong>
              </span>
            ) : tag ? (
              hint && <span className="text-muted">hover a cell</span>
            ) : (
              <span className="text-muted">{hint ? `${d}-d · hover a cell` : `${d}-d`}</span>
            )}
          </span>
          {showNorm && (
            <span className="text-muted">
              ‖v‖ = <span className="text-ink">{norm.toFixed(3)}</span> · scale ±{m.toFixed(2)}
            </span>
          )}
        </div>
      ) : (
        tag && <div className="mt-1">{tag}</div>
      )}
    </div>
  );
}
