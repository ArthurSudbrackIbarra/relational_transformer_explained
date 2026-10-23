/**
 * Term: a glossary word with a hover / focus definition (src/components/glossary.ts).
 *
 *   <Term id="seed-row">seed row</Term>      (children default to the glossary term)
 */
import type { ReactNode } from 'react';
import { getGlossaryEntry, type GlossaryId } from '../glossary';
import { HoverCard } from '../ui/floating';
import { cx } from '../ui/cx';
import { Cite } from './Cite';

export interface TermProps {
  id: GlossaryId;
  children?: ReactNode;
  className?: string;
}

export function Term({ id, children, className }: TermProps) {
  const entry = getGlossaryEntry(id);
  if (!entry) {
    if (import.meta.env.DEV) console.warn(`Term: unknown glossary id "${id}"`);
    return <>{children}</>;
  }
  return (
    <HoverCard
      placement="top"
      width={360}
      label={`Definition: ${entry.term}`}
      trigger={(p) => (
        <span
          {...p}
          ref={p.ref}
          role="button"
          tabIndex={0}
          className={cx(
            'tap-pad cursor-help underline decoration-[var(--axis)] decoration-dotted decoration-[1.5px] underline-offset-[3px] transition-colors hover:decoration-[var(--accent)]',
            p['aria-expanded'] && 'decoration-[var(--accent)]',
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
          {children ?? entry.term}
        </span>
      )}
    >
      <div className="text-[16.5px] font-semibold text-ink">{entry.term}</div>
      <p className="mt-1 text-[16px] leading-snug text-ink-2">{entry.def}</p>
      <div className="mt-1.5 text-[15.5px]">
        {entry.cite.map((c, i) => (
          <Cite key={i} kind={c.kind} ref={c.ref} />
        ))}
      </div>
    </HoverCard>
  );
}
