/**
 * In-page navigation for the "embedding" chapter: links to a Step and the "now inspecting" chip that
 * figures far from Step 2 show after they change the inspected token (display only).
 */
import { ArrowUp } from 'lucide-react';
import { useReducedMotion } from 'motion/react';
import { useCallback, useState, type ReactNode } from 'react';
import { cx } from '@/components';
import type { EmbeddingStable } from './live';
import { rowRef, useInspected } from './live';

/** Id of the Step-2 figure (TokenEmbedder) and the event that makes its token header flash. */
export const FOLLOW_ID = 'embed-follow';
export const FLASH_EVENT = 'rt-embedding-flash';

function scrollToId(id: string, reduce: boolean) {
  document.getElementById(id)?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
}

/** A link to a section or figure of this page (the app uses hash routes, so a bare #id would navigate away). */
export function JumpLink({ to, children, className }: { to: string; children: ReactNode; className?: string }) {
  const reduce = !!useReducedMotion();
  return (
    <a
      href={`#/embedding?section=${to}`}
      onClick={(e) => {
        e.preventDefault();
        scrollToId(to, reduce);
      }}
      className={className}
    >
      {children}
    </a>
  );
}

/**
 * For a figure that can change the inspected token: `pick(pos)` selects it and remembers that this
 * figure did; `chip` is then a visible confirmation with a jump to Step 2 (until another view picks).
 * Also returns the inspected token itself (`useInspected`).
 */
export function useCrossPick(live: Pick<EmbeddingStable, 'S' | 'target' | 'ctx'>) {
  const [picked, setPicked] = useState<number | null>(null);
  const { setInspected, inspected } = useInspected(live);
  const { ctx } = live;
  const pick = useCallback(
    (pos: number) => {
      setInspected(pos);
      setPicked(pos);
    },
    [setInspected],
  );
  const tok = picked !== null && picked === inspected ? ctx.tokens[picked] : undefined;
  const chip = tok ? <InspectChip pos={tok.pos} label={tok.isPadding ? 'padding' : `${rowRef(tok)}.${tok.colName}`} /> : null;
  return { pick, chip, inspected, setInspected };
}

function InspectChip({ pos, label }: { pos: number; label: string }) {
  const reduce = !!useReducedMotion();
  return (
    <div
      className="mt-3 inline-flex max-w-full flex-wrap items-center gap-x-3 gap-y-1.5 rounded-lg border border-accent/60 bg-accent-soft px-3 py-1.5 text-[15px] text-ink"
      role="status"
    >
      <span className="min-w-0">
        Step 2 now follows <span className="num">pos {pos}</span> · <span className="break-all">{label}</span>
      </span>
      <button
        type="button"
        onClick={() => {
          scrollToId(FOLLOW_ID, reduce);
          window.setTimeout(() => window.dispatchEvent(new CustomEvent(FLASH_EVENT)), reduce ? 0 : 450);
        }}
        className={cx(
          'inline-flex items-center gap-1 rounded-md border border-accent bg-surface px-2 py-0.5 text-[15px] font-medium text-accent-text',
          'hover:bg-accent-soft focus-visible:outline-2 focus-visible:outline-accent',
        )}
      >
        <ArrowUp size={15} aria-hidden="true" />
        Go to Step 2
      </button>
    </div>
  );
}
