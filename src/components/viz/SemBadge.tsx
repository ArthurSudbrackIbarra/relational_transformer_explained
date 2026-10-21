/**
 * SemBadge: the semantic type of a cell, as a GLYPH (BUILD_PLAN §5: "Sem type
 * = glyph, not colour"). number `#`, text `Aa`, datetime clock, boolean toggle.
 * SemType codes are exactly common.rs:38-44 (Number=0, Text=1, DateTime=2,
 * Boolean=3; spec §3.2).
 */
import { Clock, ToggleRight } from 'lucide-react';
import { SEM_TYPE_NAMES, SemType, type SemTypeName } from '@/engine/types';
import { cx } from '../ui/cx';

export interface SemBadgeProps {
  /** SemType code or name. */
  sem: SemType | SemTypeName;
  size?: 'xs' | 'sm' | 'md';
  /** Show the type name next to the glyph. */
  showLabel?: boolean;
  /** Bare glyph without the pill background. */
  bare?: boolean;
  className?: string;
}

function toCode(sem: SemType | SemTypeName): SemType {
  if (typeof sem === 'number') return sem;
  const i = SEM_TYPE_NAMES.indexOf(sem);
  return (i < 0 ? 0 : i) as SemType;
}

// Box sizes; the glyph text never drops below 12px (BUILD_PLAN §5).
const DIM = { xs: 18, sm: 21, md: 24 } as const;

/** Glyph for a sem type; accessible name = the type name. */
export function SemBadge({ sem, size = 'sm', showLabel = false, bare = false, className }: SemBadgeProps) {
  const code = toCode(sem);
  const name = SEM_TYPE_NAMES[code] ?? 'number';
  const px = DIM[size];
  const icon = Math.round(px * 0.7);
  const glyph =
    code === SemType.Number ? (
      <span className="font-mono font-semibold" style={{ fontSize: Math.max(12.5, Math.round(px * 0.7)) }}>
        #
      </span>
    ) : code === SemType.Text ? (
      <span className="font-semibold tracking-[-0.04em]" style={{ fontSize: Math.max(12, Math.round(px * 0.6)) }}>
        Aa
      </span>
    ) : code === SemType.DateTime ? (
      <Clock aria-hidden="true" size={icon} strokeWidth={2.1} />
    ) : (
      <ToggleRight aria-hidden="true" size={icon + 1} strokeWidth={2} />
    );
  return (
    <span
      role="img"
      aria-label={`${name} cell`}
      title={name}
      className={cx('inline-flex shrink-0 items-center gap-1 align-middle text-ink-2', className)}
    >
      <span
        aria-hidden="true"
        className={cx('inline-flex items-center justify-center leading-none', !bare && 'rounded-[5px] border border-border bg-surface-2')}
        style={{ width: px, height: px }}
      >
        {glyph}
      </span>
      {showLabel && <span className="text-[15px] leading-none">{name}</span>}
    </span>
  );
}
