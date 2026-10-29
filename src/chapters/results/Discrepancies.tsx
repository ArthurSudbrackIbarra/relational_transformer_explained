/**
 * The paper-vs-code differences that change how RT behaves, one card each: what the paper says, what
 * the code does, and a link to the chapter that shows it.
 */
import { ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Cite } from '@/components';
import { getChapter } from '@/chapters/registry';
import { DISCREPANCIES } from './discrepancies';

export function Discrepancies() {
  return (
    <div className="wide my-8 grid gap-2.5">
      {DISCREPANCIES.map((d) => {
        const ch = getChapter(d.chapter);
        return (
          <article key={d.topic} className="card border-l-[3px] !border-l-accent px-4 py-3">
            <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
              <h3 className="!m-0 text-[17.5px] font-semibold text-ink">{d.topic}</h3>
              {ch && (
                <Link
                  to={`/${d.chapter}`}
                  data-plain
                  className="inline-flex items-center gap-1.5 text-[15px] font-medium text-accent-text hover:underline"
                >
                  Ch. {ch.number}: {ch.short} <ArrowRight aria-hidden="true" size={14} />
                </Link>
              )}
            </div>
            <div className="mt-2 grid gap-2 md:grid-cols-2">
              <div className="rounded-lg bg-surface-2 px-3 py-2">
                <div className="eyebrow mb-0.5">Paper says</div>
                <div className="text-[16px] leading-snug text-ink">
                  {d.paper}{' '}
                  {d.paperRefs.map((r) => (
                    <Cite key={r.ref} kind={r.kind} ref={r.ref} />
                  ))}
                </div>
              </div>
              <div className="rounded-lg bg-surface-2 px-3 py-2">
                <div className="eyebrow mb-0.5">Code does</div>
                <div className="text-[16px] leading-snug text-ink">
                  {d.code}{' '}
                  {d.codeRefs.map((r) => (
                    <Cite key={r.ref} kind={r.kind} ref={r.ref} />
                  ))}
                </div>
              </div>
            </div>
          </article>
        );
      })}
    </div>
  );
}
