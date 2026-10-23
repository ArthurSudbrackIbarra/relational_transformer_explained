/**
 * Callout boxes for prose (BUILD_PLAN §5):
 *  - note: neutral aside
 *  - paper-vs-code: where the paper and the paper-era code differ (BUILD_PLAN
 *    §0.3: show the code, cite both). Pass `paper` and `code` to get the
 *    two-column "Paper says / Code does" layout.
 *  - warning: a pitfall (status color + icon + label, never color alone)
 *  - insight: the "why" behind a design choice
 */
import { GitCompare, Info, Lightbulb, TriangleAlert, type LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';
import { cx } from '../ui/cx';

export type CalloutVariant = 'note' | 'paper-vs-code' | 'warning' | 'insight';

const META: Record<CalloutVariant, { icon: LucideIcon; label: string; color: string }> = {
  note: { icon: Info, label: 'Note', color: 'var(--muted)' },
  'paper-vs-code': { icon: GitCompare, label: 'Paper vs code', color: 'var(--accent)' },
  warning: { icon: TriangleAlert, label: 'Watch out', color: 'var(--warning)' },
  insight: { icon: Lightbulb, label: 'Why it works this way', color: 'var(--accent)' },
};

export interface CalloutProps {
  variant?: CalloutVariant;
  /** Heading; defaults to the variant label. */
  title?: ReactNode;
  children?: ReactNode;
  /** paper-vs-code: what the paper says (with a <Cite kind="paper">). */
  paper?: ReactNode;
  /** paper-vs-code: what the code does (with a <Cite kind="code">). */
  code?: ReactNode;
  /** paper-vs-code: which one this site shows. Default "This site follows the code." */
  verdict?: ReactNode;
  className?: string;
}

export function Callout({ variant = 'note', title, children, paper, code, verdict, className }: CalloutProps) {
  const m = META[variant];
  const Icon = m.icon;
  const split = variant === 'paper-vs-code' && (paper || code);
  return (
    <aside
      className={cx(
        'my-5 rounded-xl border border-border bg-surface px-4 py-3.5 text-[18px] leading-[1.6]',
        variant === 'paper-vs-code' && 'border-l-[3px] border-l-accent',
        variant === 'warning' && 'border-l-[3px] border-l-[var(--warning)]',
        variant === 'insight' && 'bg-accent-soft border-transparent',
        className,
      )}
      aria-label={typeof title === 'string' ? title : m.label}
    >
      <div className="mb-1.5 flex items-center gap-2">
        <Icon aria-hidden="true" size={17} strokeWidth={2.2} style={{ color: m.color }} className="shrink-0" />
        <span className="text-[16.5px] font-semibold tracking-[0.005em] text-ink">
          {title ?? m.label}
          {title && variant !== 'note' && <span className="sr-only"> ({m.label})</span>}
        </span>
      </div>
      {split ? (
        <>
          <div className="mt-2 grid gap-3 sm:grid-cols-2">
            <div className="rounded-lg bg-surface-2 px-3 py-2.5">
              <div className="eyebrow mb-1">Paper says</div>
              <div className="text-[18px] leading-[1.55] text-ink">{paper}</div>
            </div>
            <div className="rounded-lg bg-surface-2 px-3 py-2.5">
              <div className="eyebrow mb-1">Code does</div>
              <div className="text-[18px] leading-[1.55] text-ink">{code}</div>
            </div>
          </div>
          {children && <div className="mt-2.5 text-ink-2 [&>p:last-child]:mb-0 [&>p]:mb-2">{children}</div>}
          <div className="mt-2 text-[16px] font-medium text-ink-2">{verdict ?? 'This site follows the code.'}</div>
        </>
      ) : (
        <div className="text-ink [&>p:last-child]:mb-0 [&>p]:mb-2">{children}</div>
      )}
    </aside>
  );
}
