/**
 * TableSwatch: table identity mark (BUILD_PLAN §5 "Table identity =
 * categorical color", §3.5 slots; task tables share slot 5). The label, if
 * shown, stays in ink (text never wears the series color).
 */
import type { ReactNode } from 'react';
import { catVar } from '@/styles/palette';
import { cx } from '../ui/cx';
import { ShapeGlyph, shapeOfSlot } from './shapes';

export interface TableSwatchProps {
  /** Categorical slot 1..8 (0 = neutral). */
  slot: number;
  /** Optional label (table name) in ink, right of the swatch. */
  label?: ReactNode;
  size?: number;
  /** Draw the slot's shape instead of a rounded square (matches EntityGraph). */
  shape?: boolean;
  className?: string;
  /** Monospace label (table names are identifiers). Default true. */
  mono?: boolean;
}

export function TableSwatch({ slot, label, size = 10, shape = false, className, mono = true }: TableSwatchProps) {
  return (
    <span className={cx('inline-flex min-w-0 items-center gap-1.5 align-middle', className)}>
      {shape ? (
        <ShapeGlyph shape={shapeOfSlot(slot)} size={size + 3} fill={catVar(slot)} />
      ) : (
        <span
          aria-hidden="true"
          className="inline-block shrink-0 rounded-[3px]"
          style={{ width: size, height: size, background: catVar(slot) }}
        />
      )}
      {label !== undefined && (
        <span className={cx('truncate text-ink', mono ? 'font-mono text-[length:max(14px,0.92em)]' : 'text-[length:max(14px,0.95em)]')}>{label}</span>
      )}
    </span>
  );
}
