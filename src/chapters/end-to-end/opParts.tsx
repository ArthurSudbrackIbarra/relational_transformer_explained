/** Pieces of the math panels shared by the stage explorers (title + formula + citation, readout, symbols). */
import type { ReactNode } from 'react';
import { TeX } from '@/components';
import type { E2ELive } from './live';

/**
 * Title, formula, prose, citation: the head of every op's math panel. Formulas are set at 18px and
 * written without nested scripts, so every sub- and superscript stays above 13px (as in chapter 10).
 */
export function MathHead({ title, tex, children, cite }: { title: ReactNode; tex: string | string[]; children?: ReactNode; cite: ReactNode }) {
  return (
    <>
      <div className="text-[16.5px] font-semibold text-ink">{title}</div>
      <div className="scroll-quiet my-2 grid gap-1 overflow-x-auto py-1 text-[18px]">
        {(Array.isArray(tex) ? tex : [tex]).map((t) => (
          <TeX key={t} tex={t} />
        ))}
      </div>
      {children}
      <div className="mt-2.5 text-[14.5px] leading-snug text-muted">{cite}</div>
    </>
  );
}

export function Readout({ children }: { children: ReactNode }) {
  return <div className="mt-3 border-t border-grid pt-2.5 text-[15px] leading-snug text-ink-2">{children}</div>;
}

export function HoverHint() {
  return <span className="text-muted">Hover or focus an entry to see how it is computed.</span>;
}

export function Sym({ children }: { children: ReactNode }) {
  return (
    <span className="self-center px-0.5 text-[20px] text-muted" aria-hidden="true">
      {children}
    </span>
  );
}

/** A token's short label for readouts. */
export function tok(live: E2ELive, i: number): ReactNode {
  return <span className="text-ink">{live.tokenLabels[i]}</span>;
}
