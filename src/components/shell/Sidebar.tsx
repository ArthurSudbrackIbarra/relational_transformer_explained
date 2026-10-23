/**
 * Chapter sidebar: numbered chapter list grouped by part, current chapter
 * highlighted, completion ticks the reader toggles by hand (persisted in the
 * store; opening a chapter does not tick it), progress meter, and the
 * Questions & answers page below the course. Rendered as a fixed column
 * ≥ 1024px (foldable into <ChapterRail>) and inside the drawer below that.
 */
import { Check, MessageCircleQuestion } from 'lucide-react';
import type { ChapterMeta } from '@/chapters/registry';
import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';
import { CHAPTERS, PARTS, prefetchChapter } from '@/chapters/registry';
import { useSim } from '@/state/store';
import { cx } from '../ui/cx';

export const QA_PATH = '/qa';

export function ChapterList({
  onNavigate,
  currentSlug,
  qaCurrent,
  action,
}: {
  onNavigate?: () => void;
  currentSlug?: string;
  /** The Questions & answers page is open. */
  qaCurrent?: boolean;
  /** Control shown at the right of the header (the collapse button). */
  action?: ReactNode;
}) {
  const completed = useSim((s) => s.completed);
  const nDone = CHAPTERS.filter((c) => completed[c.slug]).length;
  return (
    <nav aria-label="Chapters" className="flex flex-col">
      <div className="mb-4 px-3">
        <div className="flex items-center justify-between gap-2">
          <span className="eyebrow">Course</span>
          <span className="flex items-center gap-1">
            <span className="text-[14.5px] text-muted tabular-nums">
              {nDone} / {CHAPTERS.length} completed
            </span>
            {action}
          </span>
        </div>
        <div
          className="mt-2 h-[3px] w-full overflow-hidden rounded-full bg-accent-soft"
          role="progressbar"
          aria-label="Chapters completed"
          aria-valuemin={0}
          aria-valuemax={CHAPTERS.length}
          aria-valuenow={nDone}
        >
          <div className="h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${(nDone / CHAPTERS.length) * 100}%` }} />
        </div>
      </div>
      {PARTS.map((part) => (
        <div key={part.id} className="mb-3">
          <div className="px-3 pt-1 pb-1.5 text-[14px] font-semibold tracking-[0.06em] text-muted uppercase">{part.title}</div>
          <ul>
            {CHAPTERS.filter((c) => c.part === part.id).map((c) => {
              const current = c.slug === currentSlug;
              return (
                <li key={c.slug} className="relative">
                  <NavLink
                    to={`/${c.slug}`}
                    onClick={onNavigate}
                    onPointerEnter={() => prefetchChapter(c.slug)}
                    onFocus={() => prefetchChapter(c.slug)}
                    aria-current={current ? 'page' : undefined}
                    className={cx(
                      'group relative flex items-center gap-2.5 rounded-lg py-[7px] pr-11 pl-3 text-[16.5px] leading-snug transition-colors',
                      current ? 'bg-accent-soft font-medium text-ink' : 'text-ink-2 hover:bg-surface-2 hover:text-ink',
                    )}
                  >
                    {current && <span aria-hidden="true" className="absolute top-2 bottom-2 left-0 w-[3px] rounded-full bg-accent" />}
                    <span className={cx('w-5 shrink-0 text-right text-[15px] tabular-nums', current ? 'text-accent-text' : 'text-muted')}>{c.number}</span>
                    <span className="min-w-0 flex-1">{c.short}</span>
                  </NavLink>
                  <CompleteToggle chapter={c} className="absolute top-1/2 right-1.5 -translate-y-1/2" />
                </li>
              );
            })}
          </ul>
        </div>
      ))}
      <div className="mt-1 border-t border-grid pt-3">
        <div className="px-3 pt-1 pb-1.5 text-[14px] font-semibold tracking-[0.06em] text-muted uppercase">Beyond the course</div>
        <NavLink
          to={QA_PATH}
          onClick={onNavigate}
          aria-current={qaCurrent ? 'page' : undefined}
          className={cx(
            'relative flex items-center gap-2.5 rounded-lg py-[7px] pr-2.5 pl-3 text-[16.5px] leading-snug transition-colors',
            qaCurrent ? 'bg-accent-soft font-medium text-ink' : 'text-ink-2 hover:bg-surface-2 hover:text-ink',
          )}
        >
          {qaCurrent && <span aria-hidden="true" className="absolute top-2 bottom-2 left-0 w-[3px] rounded-full bg-accent" />}
          <span className="flex w-5 shrink-0 justify-end">
            <MessageCircleQuestion aria-hidden="true" size={16} className={qaCurrent ? 'text-accent-text' : 'text-muted'} />
          </span>
          <span className="min-w-0 flex-1">Questions &amp; answers</span>
        </NavLink>
      </div>
    </nav>
  );
}

/** The reader's own "done" tick for a chapter: an empty circle until toggled on. */
export function CompleteToggle({ chapter: c, className }: { chapter: ChapterMeta; className?: string }) {
  const done = useSim((s) => !!s.completed[c.slug]);
  const toggle = useSim((s) => s.toggleCompleted);
  const label = done ? `Mark chapter ${c.number}, ${c.short}, as not completed` : `Mark chapter ${c.number}, ${c.short}, as completed`;
  return (
    <button
      type="button"
      onClick={() => toggle(c.slug)}
      aria-pressed={done}
      aria-label={label}
      title={done ? 'Completed (click to undo)' : 'Mark as completed'}
      className={cx('group/tick flex h-8 w-8 items-center justify-center rounded-full', className)}
    >
      <span
        aria-hidden="true"
        className={cx(
          'flex h-[19px] w-[19px] items-center justify-center rounded-full border-[1.5px] transition-colors',
          done ? 'border-accent bg-accent text-accent-ink' : 'border-axis text-transparent group-hover/tick:border-accent group-hover/tick:text-accent-text',
        )}
      >
        <Check size={12} strokeWidth={3} />
      </span>
    </button>
  );
}

/**
 * The folded sidebar (≥ 1024px): a narrow rail of chapter numbers, so the
 * content gets the width but every chapter stays one click away.
 */
export function ChapterRail({ currentSlug, qaCurrent, action }: { currentSlug?: string; qaCurrent?: boolean; action?: ReactNode }) {
  const completed = useSim((s) => s.completed);
  return (
    <nav aria-label="Chapters" className="flex flex-col items-center gap-1">
      {action}
      {PARTS.map((part) => (
        <ul key={part.id} className="flex flex-col items-center gap-1 border-t border-grid pt-2 first-of-type:mt-1" aria-label={part.title}>
          {CHAPTERS.filter((c) => c.part === part.id).map((c) => {
            const current = c.slug === currentSlug;
            return (
              <li key={c.slug}>
                <NavLink
                  to={`/${c.slug}`}
                  onPointerEnter={() => prefetchChapter(c.slug)}
                  onFocus={() => prefetchChapter(c.slug)}
                  aria-current={current ? 'page' : undefined}
                  aria-label={`${c.number}. ${c.short}${completed[c.slug] ? ' (completed)' : ''}`}
                  title={`${c.number}. ${c.short}${completed[c.slug] ? ' (completed)' : ''}`}
                  className={cx(
                    'relative flex h-9 w-9 tabular-nums items-center justify-center rounded-lg text-[15px] transition-colors',
                    current ? 'bg-accent-soft font-semibold text-accent-text' : 'text-ink-2 hover:bg-surface-2 hover:text-ink',
                  )}
                >
                  {c.number}
                  {completed[c.slug] && (
                    <span aria-hidden="true" className="absolute right-0.5 bottom-0.5 flex h-3 w-3 items-center justify-center rounded-full bg-accent text-accent-ink">
                      <Check size={8} strokeWidth={3.5} />
                    </span>
                  )}
                </NavLink>
              </li>
            );
          })}
        </ul>
      ))}
      <div className="mt-1 border-t border-grid pt-2">
        <NavLink
          to={QA_PATH}
          aria-current={qaCurrent ? 'page' : undefined}
          aria-label="Questions & answers"
          title="Questions & answers"
          className={cx(
            'flex h-9 w-9 items-center justify-center rounded-lg transition-colors',
            qaCurrent ? 'bg-accent-soft text-accent-text' : 'text-ink-2 hover:bg-surface-2 hover:text-ink',
          )}
        >
          <MessageCircleQuestion aria-hidden="true" size={17} />
        </NavLink>
      </div>
    </nav>
  );
}
