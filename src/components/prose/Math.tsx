/**
 * KaTeX math. `Math` (alias `TeX`) renders inline, `MathBlock` renders display
 * math with an optional equation label. Output is memoised per TeX string.
 *
 * Tip: importing `{ Math }` shadows the global `Math` object in that file;
 * prefer `import { TeX } from '@/components'` if you also need Math.max etc.
 */
import katex from 'katex';
import { useMemo, type ReactNode } from 'react';
import { cx } from '../ui/cx';

const cache = new Map<string, string>();

function render(tex: string, display: boolean): string {
  const key = `${display ? 'D' : 'I'}:${tex}`;
  let html = cache.get(key);
  if (html === undefined) {
    html = katex.renderToString(tex, {
      displayMode: display,
      throwOnError: false,
      strict: 'ignore',
      output: 'htmlAndMathml', // MathML for screen readers
      trust: false,
    });
    if (cache.size > 2000) cache.clear();
    cache.set(key, html);
  }
  return html;
}

export interface MathProps {
  /** TeX source, e.g. "x_i = \\mathrm{RMSNorm}(W c_i + b)". */
  tex: string;
  className?: string;
}

/** Inline math. */
export function InlineMath({ tex, className }: MathProps) {
  const html = useMemo(() => render(tex, false), [tex]);
  return <span className={cx('katex-inline', className)} dangerouslySetInnerHTML={{ __html: html }} />;
}

export interface MathBlockProps extends MathProps {
  /** Right-aligned label, e.g. "(1)" or a source chip. */
  label?: ReactNode;
  /** Small caption under the equation. */
  caption?: ReactNode;
}

/** Display math, centered, horizontally scrollable if too wide for a phone. */
export function MathBlock({ tex, label, caption, className }: MathBlockProps) {
  const html = useMemo(() => render(tex, true), [tex]);
  return (
    <figure className={cx('my-6', className)}>
      <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:gap-4">
        <div className="scroll-quiet min-w-0 flex-1 overflow-x-auto text-center" dangerouslySetInnerHTML={{ __html: html }} />
        {label && <div className="shrink-0 self-end text-[16px] text-muted sm:self-auto">{label}</div>}
      </div>
      {caption && <figcaption className="mt-2 text-center text-[16px] leading-snug text-ink-2">{caption}</figcaption>}
    </figure>
  );
}

export { InlineMath as Math, InlineMath as TeX };
