/**
 * Chapter sidebar: numbered chapter list grouped by part, current chapter
 * highlighted, visited ticks (persisted in the store), progress meter.
 * Rendered as a fixed column ≥ 1024px and inside the drawer below that.
 */
import { Check } from 'lucide-react';
import { NavLink } from 'react-router-dom';
import { CHAPTERS, PARTS, prefetchChapter } from '@/chapters/registry';
import { useSim } from '@/state/store';
import { cx } from '../ui/cx';

export function ChapterList({ onNavigate, currentSlug }: { onNavigate?: () => void; currentSlug?: string }) {
  const visited = useSim((s) => s.visited);
  const nVisited = CHAPTERS.filter((c) => visited[c.slug]).length;
  return (
    <nav aria-label="Chapters" className="flex flex-col">
      <div className="mb-4 px-3">
        <div className="flex items-baseline justify-between">
          <span className="eyebrow">Course</span>
          <span className="num text-[14.5px] text-muted">
            {nVisited} / {CHAPTERS.length} visited
          </span>
        </div>
        <div
          className="mt-2 h-[3px] w-full overflow-hidden rounded-full bg-accent-soft"
          role="progressbar"
          aria-label="Chapters visited"
          aria-valuemin={0}
          aria-valuemax={CHAPTERS.length}
          aria-valuenow={nVisited}
        >
          <div className="h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${(nVisited / CHAPTERS.length) * 100}%` }} />
        </div>
      </div>
      {PARTS.map((part) => (
        <div key={part.id} className="mb-3">
          <div className="px-3 pt-1 pb-1.5 text-[14px] font-semibold tracking-[0.06em] text-muted uppercase">{part.title}</div>
          <ul>
            {CHAPTERS.filter((c) => c.part === part.id).map((c) => {
              const current = c.slug === currentSlug;
              return (
                <li key={c.slug}>
                  <NavLink
                    to={`/${c.slug}`}
                    onClick={onNavigate}
                    onPointerEnter={() => prefetchChapter(c.slug)}
                    onFocus={() => prefetchChapter(c.slug)}
                    aria-current={current ? 'page' : undefined}
                    className={cx(
                      'group relative flex items-center gap-2.5 rounded-lg py-[7px] pr-2.5 pl-3 text-[16.5px] leading-snug transition-colors',
                      current ? 'bg-accent-soft font-medium text-ink' : 'text-ink-2 hover:bg-surface-2 hover:text-ink',
                    )}
                  >
                    {current && <span aria-hidden="true" className="absolute top-2 bottom-2 left-0 w-[3px] rounded-full bg-accent" />}
                    <span className={cx('num w-5 shrink-0 text-right text-[15px]', current ? 'text-accent-text' : 'text-muted')}>{c.number}</span>
                    <span className="min-w-0 flex-1">{c.short}</span>
                    {visited[c.slug] && !current && (
                      <Check aria-label="visited" size={14} strokeWidth={2.4} className="shrink-0 text-muted" />
                    )}
                  </NavLink>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </nav>
  );
}
