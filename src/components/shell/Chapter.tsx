/**
 * Chapter building blocks (BUILD_PLAN §6: ChapterHeader → Sections → CodeRefs
 * → callouts → Quiz → ChapterNav).
 *
 *   export default function SamplingChapter() {
 *     return (
 *       <ChapterPage slug="sampling">
 *         <ChapterHeader slug="sampling" lede="…" learn={['…', '…']} />
 *         <Section id="bfs" title="A bounded-width BFS">
 *           <p>Prose sits in the ~70ch column…</p>
 *           <Figure wide title="…" caption="…">…</Figure>
 *         </Section>
 *         <ChapterNav slug="sampling" />
 *       </ChapterPage>
 *     );
 *   }
 */
import { ArrowLeft, ArrowRight, Check, Link as LinkIcon } from 'lucide-react';
import { useEffect, type ReactNode } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { getChapter, getNeighbors, PARTS, prefetchChapter } from '@/chapters/registry';
import { useSim } from '@/state/store';
import { SITE_TITLE } from '../prose/sources';
import { cx } from '../ui/cx';
import { ContentColumn } from './Layout';

// ─────────────────────────────────────────────────────────────────────────────
// ChapterPage
// ─────────────────────────────────────────────────────────────────────────────

/** Page wrapper: content column + bottom spacing; scrolls to `?section=` on load. */
export function ChapterPage({ children, className }: { slug?: string; children: ReactNode; className?: string }) {
  const { search } = useLocation();
  useEffect(() => {
    const id = new URLSearchParams(search).get('section');
    if (!id) return;
    const t = window.setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60);
    return () => window.clearTimeout(t);
  }, [search]);
  return (
    <ContentColumn className={cx('pt-10 pb-24 sm:pt-14', className)}>
      <article>{children}</article>
    </ContentColumn>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ChapterHeader
// ─────────────────────────────────────────────────────────────────────────────

export interface ChapterHeaderProps {
  /** Fills number, title and part from the registry. */
  slug?: string;
  number?: number;
  title?: ReactNode;
  /** One or two sentences under the title. */
  lede?: ReactNode;
  /** "You will learn" bullets. */
  learn?: ReactNode[];
  /** Optional extra content under the lede (e.g. a ScaleNote). */
  children?: ReactNode;
}

export function ChapterHeader({ slug, number, title, lede, learn, children }: ChapterHeaderProps) {
  const meta = getChapter(slug);
  const n = number ?? meta?.number;
  const t = title ?? meta?.title;
  const part = meta ? PARTS.find((p) => p.id === meta.part) : undefined;
  useEffect(() => {
    if (typeof t === 'string') document.title = `${n ? `${n}. ` : ''}${t} · ${SITE_TITLE}`;
  }, [t, n]);
  return (
    <header className="page-grid mb-12">
      <div className="eyebrow flex items-center gap-2">
        {n !== undefined && <span>Chapter {n}</span>}
        {part && (
          <>
            <span aria-hidden="true" className="text-axis">
              /
            </span>
            <span>{part.title}</span>
          </>
        )}
      </div>
      <h1 className="mt-3 text-[34px] leading-[1.12] font-semibold tracking-[-0.022em] text-balance text-ink sm:text-[42px]">{t}</h1>
      {lede && <p className="mt-4 text-[21px] leading-[1.55] text-pretty text-ink-2 sm:text-[22px]">{lede}</p>}
      {learn && learn.length > 0 && (
        <div className="mt-7 rounded-xl border border-border bg-surface px-5 py-4">
          <div className="eyebrow mb-2">You will learn</div>
          <ul className="!mb-0 grid gap-1.5 !pl-0 text-[18px] leading-snug" style={{ listStyle: 'none' }}>
            {learn.map((l, i) => (
              <li key={i} className="!m-0 flex gap-2.5 !pl-0">
                <span aria-hidden="true" className="mt-[0.55em] h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
                <span>{l}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {children}
    </header>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Section
// ─────────────────────────────────────────────────────────────────────────────

export interface SectionProps {
  /** Anchor id (used for ?section= deep links). */
  id: string;
  title: ReactNode;
  /** Small eyebrow above the title, e.g. "Step 2". */
  kicker?: ReactNode;
  children: ReactNode;
  className?: string;
}

/**
 * A titled section laid out on the reading grid: direct children sit in the
 * ~70ch prose column; give a child `className="wide"` (or use <Figure wide>)
 * to span the full content width.
 */
export function Section({ id, title, kicker, children, className }: SectionProps) {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  return (
    <section id={id} aria-labelledby={`${id}-title`} className={cx('page-grid mt-16 scroll-mt-[calc(var(--topbar-h)+20px)] first-of-type:mt-0', className)}>
      {kicker && <div className="eyebrow mb-2">{kicker}</div>}
      <h2 id={`${id}-title`} className="group mb-4 flex items-baseline gap-2 text-[25px] leading-[1.25] font-semibold tracking-[-0.015em] text-balance text-ink">
        <span>{title}</span>
        <a
          href={`#${pathname}?section=${id}`}
          data-plain
          onClick={(e) => {
            e.preventDefault();
            navigate(`${pathname}?section=${id}`, { replace: true });
            document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }}
          className="translate-y-[-1px] rounded p-0.5 text-axis opacity-0 transition-opacity group-hover:opacity-100 hover:text-accent-text focus-visible:opacity-100"
          aria-label="Link to this section"
        >
          <LinkIcon size={15} />
        </a>
      </h2>
      {children}
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Figure
// ─────────────────────────────────────────────────────────────────────────────

export interface FigureProps {
  /** Span the full content width (≤ 1200px) instead of the prose column. */
  wide?: boolean;
  /** Bold title above the figure. */
  title?: ReactNode;
  /** Caption below (what to look at). */
  caption?: ReactNode;
  /** One row of controls above the figure (dataviz: filters in one row above what they scope). */
  controls?: ReactNode;
  /** No card background (e.g. for a full-bleed diagram). */
  bare?: boolean;
  children: ReactNode;
  className?: string;
  id?: string;
}

export function Figure({ wide, title, caption, controls, bare, children, className, id }: FigureProps) {
  return (
    <figure id={id} className={cx('my-8 min-w-0', wide && 'wide', className)}>
      <div className={cx('min-w-0', !bare && 'card px-4 py-4 sm:px-5 sm:py-5')}>
        {title && <div className="mb-3 text-[17px] font-semibold text-ink">{title}</div>}
        {controls && <div className="mb-4 flex flex-wrap items-end gap-x-4 gap-y-3">{controls}</div>}
        {children}
      </div>
      {caption && <figcaption className="mx-auto mt-3 max-w-[var(--prose-max)] text-[16px] leading-snug text-ink-2">{caption}</figcaption>}
    </figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ChapterNav
// ─────────────────────────────────────────────────────────────────────────────

/** End-of-chapter toggle: the reader marks the chapter completed by hand (sidebar ticks follow). */
function CompleteButton({ slug }: { slug: string }) {
  const meta = getChapter(slug);
  const done = useSim((s) => !!s.completed[slug]);
  const toggle = useSim((s) => s.toggleCompleted);
  if (!meta) return null;
  return (
    <div className="mb-6 flex flex-wrap items-center justify-center gap-3">
      <button
        type="button"
        onClick={() => toggle(slug)}
        aria-pressed={done}
        className={cx(
          'inline-flex h-11 items-center gap-2.5 rounded-xl border px-4 text-[16.5px] font-medium transition-colors',
          done ? 'border-accent bg-accent-soft text-accent-text' : 'border-border-strong bg-surface text-ink hover:border-accent',
        )}
      >
        <span
          aria-hidden="true"
          className={cx(
            'flex h-[20px] w-[20px] items-center justify-center rounded-full border-[1.5px]',
            done ? 'border-accent bg-accent text-accent-ink' : 'border-axis text-transparent',
          )}
        >
          <Check size={13} strokeWidth={3} />
        </span>
        {done ? `Chapter ${meta.number} completed` : `Mark chapter ${meta.number} as completed`}
      </button>
      {done && <span className="text-[15px] text-muted">Click again to undo.</span>}
    </div>
  );
}

export function ChapterNav({ slug }: { slug: string }) {
  const { prev, next } = getNeighbors(slug);
  return (
    <nav aria-label="Chapter navigation" className="page-grid mt-20">
      <div className="border-t border-grid pt-8">
        <CompleteButton slug={slug} />
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {prev ? (
          <Link
            to={`/${prev.slug}`}
            data-plain
            onPointerEnter={() => prefetchChapter(prev.slug)}
            className="group rounded-xl border border-border bg-surface px-4 py-3.5 transition-colors hover:border-accent"
          >
            <span className="flex items-center gap-1.5 text-[15px] text-muted">
              <ArrowLeft aria-hidden="true" size={14} className="transition-transform group-hover:-translate-x-0.5" /> Previous · {prev.number}
            </span>
            <span className="mt-1 block text-[18px] font-semibold text-ink">{prev.title}</span>
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link
            to={`/${next.slug}`}
            data-plain
            onPointerEnter={() => prefetchChapter(next.slug)}
            className="group rounded-xl border border-border bg-surface px-4 py-3.5 text-right transition-colors hover:border-accent"
          >
            <span className="flex items-center justify-end gap-1.5 text-[15px] text-muted">
              Next · {next.number} <ArrowRight aria-hidden="true" size={14} className="transition-transform group-hover:translate-x-0.5" />
            </span>
            <span className="mt-1 block text-[18px] font-semibold text-ink">{next.title}</span>
          </Link>
        ) : (
          <Link to="/" data-plain className="rounded-xl border border-border bg-surface px-4 py-3.5 text-right hover:border-accent">
            <span className="text-[15px] text-muted">The end</span>
            <span className="mt-1 block text-[18px] font-semibold text-ink">Back to the course map</span>
          </Link>
        )}
      </div>
    </nav>
  );
}
