/**
 * ArrayStrip: one boolean field of the batch (e.g. masks[0, :] or is_padding[0, :]) drawn as the
 * literal array, one cell per position. Wraps onto several lines on narrow screens.
 */
import { useMemo, useState } from 'react';
import { ShapeTag, cx, type Shape } from '@/components';
import { useElementSize } from '@/components/viz/useSize';

export interface ArrayStripProps {
  /** Field name, e.g. "masks". Rendered as `name[0, :]`. */
  name: string;
  values: readonly boolean[];
  /** Ring this position. */
  highlight?: number | null;
  /** Minimum cell width in px (default 9). */
  minCell?: number;
  cellH?: number;
  onHover?: (pos: number | null) => void;
  /** Extra text after the counts. */
  note?: string;
  /** What the row and the columns of this array are, for its ShapeTag. */
  shape: Shape | null;
  className?: string;
}

export function ArrayStrip({ name, values, highlight = null, minCell = 9, cellH = 20, onHover, note, shape, className }: ArrayStripProps) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const n = values.length;
  const width = Math.max(120, size.width || 600);
  // Balanced wrapping: as few lines as the minimum cell width allows, equally filled.
  const maxPerLine = Math.max(8, Math.min(n, Math.floor(width / minCell)));
  const lines = Math.max(1, Math.ceil(n / maxPerLine));
  const perLine = Math.ceil(n / lines);
  const cw = width / perLine;
  const H = lines * (cellH + 4) - 4;
  const nTrue = useMemo(() => values.reduce((s, v) => s + (v ? 1 : 0), 0), [values]);
  const set = (i: number | null) => {
    setHover(i);
    onHover?.(i);
  };
  const shown = hover ?? highlight;

  return (
    <div className={cx('min-w-0', className)}>
      <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 text-[15.5px]">
        <span className="font-mono text-ink">
          {name}
          <span className="text-ink-2">[0, :]</span>
        </span>
        <span className="num text-ink-2" aria-live="polite">
          {hover !== null ? (
            <>
              position {hover}: <span className="font-semibold text-ink">{values[hover] ? 'true' : 'false'}</span>
            </>
          ) : (
            <>
              <span className="font-semibold text-ink">{nTrue}</span> × true · {n - nTrue} × false{note ? ` · ${note}` : ''}
            </>
          )}
        </span>
      </div>
      <div
        ref={ref}
        className="w-full rounded-[3px] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        tabIndex={0}
        role="group"
        aria-label={`${name}: use the arrow keys to read one position at a time`}
        onKeyDown={(e) => {
          const cur = hover ?? highlight ?? 0;
          let next: number | null = null;
          if (e.key === 'ArrowRight') next = hover === null ? cur : cur + 1;
          else if (e.key === 'ArrowLeft') next = hover === null ? cur : cur - 1;
          else if (e.key === 'ArrowDown') next = cur + perLine;
          else if (e.key === 'ArrowUp') next = cur - perLine;
          else if (e.key === 'Home') next = 0;
          else if (e.key === 'End') next = n - 1;
          else if (e.key === 'Escape') {
            set(null);
            return;
          } else return;
          e.preventDefault();
          set(Math.max(0, Math.min(n - 1, next)));
        }}
        onBlur={() => set(null)}
      >
        <svg
          width="100%"
          height={H}
          viewBox={`0 0 ${width} ${H}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={`${name}: ${nTrue} of ${n} positions are true`}
          className="block touch-none"
          onPointerMove={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            const x = ((e.clientX - r.left) / r.width) * width;
            const line = Math.floor((e.clientY - r.top) / (cellH + 4));
            const i = line * perLine + Math.floor(x / cw);
            set(i >= 0 && i < n ? i : null);
          }}
          onPointerLeave={() => set(null)}
        >
          {values.map((v, i) => {
            const x = (i % perLine) * cw;
            const y = Math.floor(i / perLine) * (cellH + 4);
            return (
              <rect
                key={i}
                x={x + 0.5}
                y={y}
                width={Math.max(0.6, cw - 1)}
                height={cellH}
                rx={Math.min(2, cw / 4)}
                style={{ fill: v ? 'var(--accent)' : 'var(--surface-3)' }}
              />
            );
          })}
          {shown !== null && shown < n && (
            <rect
              x={(shown % perLine) * cw - 0.5}
              y={Math.floor(shown / perLine) * (cellH + 4) - 1.5}
              width={cw + 1}
              height={cellH + 3}
              rx={2}
              fill="none"
              style={{ stroke: 'var(--ink)' }}
              strokeWidth={1.5}
              vectorEffect="non-scaling-stroke"
            />
          )}
        </svg>
      </div>
      {shape && (
        <div className="mt-1">
          <ShapeTag
            rows={1}
            cols={n}
            shape={lines > 1 ? { ...shape, note: <>{shape.note} Drawn as {lines} lines of up to {perLine} cells to fit: read left to right, then top to bottom.</> } : shape}
          />
        </div>
      )}
    </div>
  );
}
