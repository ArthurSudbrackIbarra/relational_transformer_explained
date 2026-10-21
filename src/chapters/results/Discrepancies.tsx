/**
 * Every paper-vs-code discrepancy of spec §10 as a filterable list: by area, by free text, each row with
 * its citations and a link to the chapter that shows it.
 */
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { ArrowRight, ChevronDown, Search } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Cite, cx } from '@/components';
import { getChapter } from '@/chapters/registry';
import { AREA_LABEL, DISCREPANCIES, type Area } from './discrepancies';

const AREAS: Area[] = ['model', 'data', 'sampling', 'training', 'reporting'];

export function Discrepancies() {
  const [area, setArea] = useState<Area | 'all'>('all');
  const [q, setQ] = useState('');
  const reduce = useReducedMotion();
  // expanded by default on wide screens, collapsed to one line per row on phones
  const [open, setOpen] = useState<Set<number>>(() => {
    let wide = true;
    try {
      wide = window.matchMedia('(min-width: 768px)').matches;
    } catch {
      /* no matchMedia */
    }
    return new Set(wide ? DISCREPANCIES.map((d) => d.n) : []);
  });
  const toggle = (n: number) =>
    setOpen((o) => {
      const next = new Set(o);
      if (next.has(n)) next.delete(n);
      else next.add(n);
      return next;
    });
  const allOpen = open.size === DISCREPANCIES.length;
  const list = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return DISCREPANCIES.filter(
      (d) => (area === 'all' || d.area === area) && (!needle || `${d.n} ${d.topic} ${d.paper} ${d.code} ${d.site}`.toLowerCase().includes(needle)),
    );
  }, [area, q]);
  const count = (a: Area | 'all') => (a === 'all' ? DISCREPANCIES.length : DISCREPANCIES.filter((d) => d.area === a).length);

  return (
    <div className="wide my-8">
      <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Filter by area">
        {(['all', ...AREAS] as const).map((a) => (
          <button
            key={a}
            type="button"
            aria-pressed={area === a}
            onClick={() => setArea(a)}
            className={cx(
              'inline-flex h-9 items-center gap-1.5 rounded-full border px-3.5 text-[15px] font-medium transition-colors',
              area === a
                ? 'border-accent bg-accent-soft text-accent-text'
                : 'border-border bg-surface text-ink-2 hover:border-border-strong hover:text-ink',
            )}
          >
            {a === 'all' ? 'All' : AREA_LABEL[a]}
            <span className="num text-[14px] text-muted">{count(a)}</span>
          </button>
        ))}
        <label className="relative ml-auto flex min-w-[220px] flex-1 items-center sm:flex-none">
          <span className="sr-only">Search the discrepancies</span>
          <Search aria-hidden="true" size={16} className="pointer-events-none absolute left-3 text-muted" />
          <input
            type="search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search, e.g. “timestamp”"
            className="h-9 w-full rounded-lg border border-border-strong bg-surface pr-3 pl-9 text-[15.5px] text-ink placeholder:text-muted focus-visible:outline-2 focus-visible:outline-accent"
          />
        </label>
      </div>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
        <p className="!m-0 text-[15px] text-ink-2" aria-live="polite">
          Showing {list.length} of {DISCREPANCIES.length}. “Code” is the paper-era code at commit 2a899c8 <Cite kind="spec" ref="§10" />.
        </p>
        <button
          type="button"
          onClick={() => setOpen(allOpen ? new Set() : new Set(DISCREPANCIES.map((d) => d.n)))}
          className="text-[15px] font-medium text-accent-text hover:underline"
        >
          {allOpen ? 'Collapse all' : 'Expand all'}
        </button>
      </div>

      <div className="mt-3 grid gap-2.5">
        <AnimatePresence initial={false}>
          {list.map((d) => {
            const ch = getChapter(d.chapter);
            return (
              <motion.article
                key={d.n}
                layout={!reduce}
                initial={reduce ? false : { opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={reduce ? { opacity: 0, transition: { duration: 0 } } : { opacity: 0, y: -4, transition: { duration: 0.15 } }}
                transition={{ duration: 0.2 }}
                className="card border-l-[3px] !border-l-accent px-4 py-3"
                aria-labelledby={`disc-${d.n}`}
              >
                <h3 id={`disc-${d.n}`} className="!m-0">
                  <button
                    type="button"
                    aria-expanded={open.has(d.n)}
                    onClick={() => toggle(d.n)}
                    className="flex w-full flex-wrap items-baseline gap-x-3 gap-y-1 text-left"
                  >
                    <span className="num text-[15px] font-semibold text-muted">#{d.n}</span>
                    <span className="text-[17.5px] font-semibold text-ink">{d.topic}</span>
                    <span className="rounded-full bg-surface-2 px-2.5 py-0.5 text-[14px] font-normal text-ink-2">{AREA_LABEL[d.area]}</span>
                    <ChevronDown
                      aria-hidden="true"
                      size={17}
                      className={cx('ml-auto self-center text-muted transition-transform', open.has(d.n) && 'rotate-180')}
                    />
                  </button>
                </h3>
                {open.has(d.n) && (
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
                      <div className="eyebrow mb-0.5">{d.codeLabel ?? 'Code does'}</div>
                      <div className="text-[16px] leading-snug text-ink">
                        {d.code}{' '}
                        {d.codeRefs.map((r) => (
                          <Cite key={r.ref} kind={r.kind} ref={r.ref} />
                        ))}
                      </div>
                    </div>
                  </div>
                )}
                <div className="mt-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1.5">
                  <div className="text-[15.5px] text-ink-2">
                    <span className="font-semibold text-ink">This site:</span> {d.site}
                  </div>
                  {ch && d.chapter !== 'results' && (
                    <Link
                      to={`/${d.chapter}`}
                      data-plain
                      className="inline-flex items-center gap-1.5 text-[15px] font-medium text-accent-text hover:underline"
                    >
                      Ch. {ch.number}: {ch.short} <ArrowRight aria-hidden="true" size={14} />
                    </Link>
                  )}
                  {d.chapter === 'results' && (
                    <Link
                      to="/results?section=provenance"
                      data-plain
                      className="inline-flex items-center gap-1.5 text-[15px] font-medium text-accent-text hover:underline"
                    >
                      See “Provenance” below <ArrowRight aria-hidden="true" size={14} />
                    </Link>
                  )}
                </div>
              </motion.article>
            );
          })}
        </AnimatePresence>
        {list.length === 0 && (
          <p className="rounded-lg border border-dashed border-border-strong px-4 py-3 text-[16px] text-ink-2">Nothing matches that search.</p>
        )}
      </div>
    </div>
  );
}
