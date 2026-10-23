/**
 * Stat tile (dataviz "figure contract"): label (sentence case, no colon) ·
 * value (sans semibold, proportional figures) · optional delta vs a named
 * reference · optional caption. Use when the data is one number.
 */
import { ArrowDownRight, ArrowUpRight, Minus } from 'lucide-react';
import type { ReactNode } from 'react';
import { cx } from '../ui/cx';

export interface StatProps {
  label: ReactNode;
  value: ReactNode;
  /** Unit suffix shown smaller, e.g. "%". */
  unit?: ReactNode;
  /** Signed change, e.g. +3.1. Shown with an arrow. */
  delta?: number;
  /** What the delta compares against, e.g. "vs EntityMean". */
  deltaLabel?: ReactNode;
  /** Whether an increase is good (decides the delta color). Default true. */
  upIsGood?: boolean;
  formatDelta?: (d: number) => string;
  caption?: ReactNode;
  /** 'lg' for a hero figure (one per view). */
  size?: 'md' | 'lg';
  /** Mark this tile as the emphasised one (accent top rule). */
  emphasis?: boolean;
  className?: string;
}

export function Stat({
  label,
  value,
  unit,
  delta,
  deltaLabel,
  upIsGood = true,
  formatDelta = (d) => `${d > 0 ? '+' : d < 0 ? '−' : '±'}${Math.abs(d).toFixed(1)}`,
  caption,
  size = 'md',
  emphasis = false,
  className,
}: StatProps) {
  const dir = delta === undefined ? 0 : Math.sign(delta);
  const good = dir === 0 ? null : (dir > 0) === upIsGood;
  const Icon = dir > 0 ? ArrowUpRight : dir < 0 ? ArrowDownRight : Minus;
  return (
    <div className={cx('card relative min-w-0 overflow-hidden px-4 py-3.5', className)}>
      {emphasis && <div aria-hidden="true" className="absolute inset-x-0 top-0 h-[2px] bg-accent" />}
      <div className="text-[15.5px] font-medium text-ink-2">{label}</div>
      <div className={cx('mt-1 flex items-baseline gap-1 font-semibold tracking-tight text-ink', size === 'lg' ? 'text-[48px] leading-none' : 'text-[28px] leading-tight')}>
        <span>{value}</span>
        {unit && <span className={cx('font-medium text-ink-2', size === 'lg' ? 'text-[22px]' : 'text-[17.5px]')}>{unit}</span>}
      </div>
      {delta !== undefined && (
        <div className="mt-1.5 flex items-center gap-1 text-[15px]">
          <Icon
            aria-hidden="true"
            size={14}
            strokeWidth={2.2}
            style={{ color: good === null ? 'var(--muted)' : good ? 'var(--good-text)' : 'var(--critical)' }}
          />
          <span className="num font-semibold" style={{ color: good === null ? 'var(--ink-2)' : good ? 'var(--good-text)' : 'var(--critical)' }}>
            {formatDelta(delta)}
          </span>
          {deltaLabel && <span className="text-ink-2">{deltaLabel}</span>}
        </div>
      )}
      {caption && <div className="mt-1.5 text-[15px] leading-snug text-muted">{caption}</div>}
    </div>
  );
}
