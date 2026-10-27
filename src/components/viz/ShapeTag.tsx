/**
 * ShapeTag: the shape of a drawn matrix or vector ("128 × 64") as small muted text under its
 * bottom-left corner. Hover, focus or tap opens a card that says what one row and one column are
 * and where those sizes come from (the wording lives with the figures: src/chapters/shapes.ts).
 *
 * The numbers are the drawn ones, passed by the figure component itself, so the tag always matches
 * what is on screen; callers only supply the meanings.
 */
import type { ReactNode } from 'react';
import { HoverCard } from '../ui/floating';
import { cx } from '../ui/cx';

export interface Shape {
  /** What one row is and where the row count comes from, phrased to follow "128 rows:". */
  rows: ReactNode;
  /** What one column is and where the column count comes from, phrased to follow "64 columns:". */
  cols: ReactNode;
  /** How the drawing relates to the stored tensor (transposed, wrapped, cropped…), when it matters. */
  note?: ReactNode;
}

export interface ShapeTagProps {
  rows: number;
  cols: number;
  shape: Shape;
  className?: string;
}

const fmt = (n: number) => n.toLocaleString('en-US');

export function ShapeTag({ rows, cols, shape, className }: ShapeTagProps) {
  const text = `${fmt(rows)} × ${fmt(cols)}`;
  return (
    <HoverCard
      placement="top"
      align="start"
      width={380}
      label={`Shape ${fmt(rows)} by ${fmt(cols)}`}
      trigger={(p) => (
        <span
          {...p}
          ref={p.ref}
          role="button"
          tabIndex={0}
          className={cx(
            'num tap-pad inline-block cursor-help whitespace-nowrap text-[13.5px] leading-tight text-muted underline decoration-[var(--axis)] decoration-dotted decoration-1 underline-offset-[3px] transition-colors hover:text-ink-2 hover:decoration-[var(--accent)]',
            p['aria-expanded'] && 'text-ink-2 decoration-[var(--accent)]',
            className,
          )}
          onKeyDown={(e) => {
            p.onKeyDown(e);
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              p.onClick(e as unknown as React.MouseEvent);
            }
          }}
        >
          {text}
        </span>
      )}
    >
      <div className="num text-[16.5px] font-semibold text-ink">{text}</div>
      <dl className="!m-0 mt-1.5 grid gap-1.5 text-[15.5px] leading-snug text-ink-2">
        <div>
          <dt className="num inline font-semibold text-ink">
            {fmt(rows)} {rows === 1 ? 'row' : 'rows'}:{' '}
          </dt>
          <dd className="!m-0 inline">{shape.rows}</dd>
        </div>
        <div>
          <dt className="num inline font-semibold text-ink">
            {fmt(cols)} {cols === 1 ? 'column' : 'columns'}:{' '}
          </dt>
          <dd className="!m-0 inline">{shape.cols}</dd>
        </div>
      </dl>
      {shape.note && <p className="!mt-2 !mb-0 text-[15px] leading-snug text-muted">{shape.note}</p>}
    </HoverCard>
  );
}
