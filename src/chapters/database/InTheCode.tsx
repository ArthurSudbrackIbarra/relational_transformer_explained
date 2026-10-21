/**
 * A collapsed "In the code" box: code-level detail that a first reading can skip
 * (CodeRefs, edge cases, live tables), kept one click away so the main text stays
 * on the concepts.
 */
import { ChevronDown, FileCode2 } from 'lucide-react';
import type { ReactNode } from 'react';

export function InTheCode({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <details className="group my-6 rounded-xl border border-border bg-surface">
      <summary className="flex cursor-pointer list-none items-center gap-2.5 rounded-xl px-4 py-3 text-[17px] font-semibold text-ink hover:bg-surface-2/60 [&::-webkit-details-marker]:hidden">
        <FileCode2 aria-hidden="true" size={17} className="shrink-0 text-muted" />
        <span className="min-w-0 flex-1">
          <span className="eyebrow mr-2 text-muted">In the code</span>
          {title}
        </span>
        <ChevronDown aria-hidden="true" size={17} className="shrink-0 text-muted transition-transform group-open:rotate-180" />
      </summary>
      <div className="border-t border-border px-4 pt-3 pb-1 text-[17px] leading-[1.6] text-ink-2 [&>p]:mb-3 [&>p]:text-ink">{children}</div>
    </details>
  );
}
