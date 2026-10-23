/**
 * Status badge for verification results and warnings ONLY (BUILD_PLAN §5:
 * status colors never encode series). Always icon + label, never color alone.
 */
import { CircleCheck, CircleX, Info, TriangleAlert } from 'lucide-react';
import type { ReactNode } from 'react';
import { cx } from './cx';

export type Status = 'good' | 'warning' | 'serious' | 'critical' | 'info';

const ICON = { good: CircleCheck, warning: TriangleAlert, serious: TriangleAlert, critical: CircleX, info: Info };
const COLOR: Record<Status, string> = {
  good: 'var(--good)',
  warning: 'var(--warning)',
  serious: 'var(--serious)',
  critical: 'var(--critical)',
  info: 'var(--muted)',
};

export interface StatusBadgeProps {
  status: Status;
  /** Visible label, e.g. "Parity: max |Δ| 3e-6". */
  children: ReactNode;
  className?: string;
}

/** Pill with a status-colored icon and an ink-colored label. */
export function StatusBadge({ status, children, className }: StatusBadgeProps) {
  const Icon = ICON[status];
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-0.5 text-[15px] font-medium text-ink',
        className,
      )}
    >
      <Icon aria-hidden="true" size={14} strokeWidth={2.2} style={{ color: COLOR[status] }} />
      <span className="sr-only">{status}: </span>
      {children}
    </span>
  );
}
