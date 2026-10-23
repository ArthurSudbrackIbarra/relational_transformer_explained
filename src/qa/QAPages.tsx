/**
 * Questions & answers: the list (/#/qa) and one answer per page
 * (/#/qa/<id>). Answers use the chapters' building blocks (Section, Figure,
 * Callout, Cite, CodeRef), so they read like a short chapter.
 */
import { ArrowLeft, ArrowRight, MessageCircleQuestion } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getChapter, prefetchChapter } from '@/chapters/registry';
import { ChapterPage } from '@/components/shell/Chapter';
import { SITE_TITLE } from '@/components/prose/sources';
import { cx } from '@/components/ui/cx';
import { NotFound } from '@/pages/NotFound';
import { getQA, QA_ENTRIES } from './registry';
import type { QAEntry } from './types';

function formatDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });
}

function ChapterChips({ slugs }: { slugs: string[] }) {
  return (
    <span className="flex flex-wrap gap-1.5">
      {slugs.map((slug) => {
        const c = getChapter(slug);
        if (!c) return null;
        return (
          <Link
            key={slug}
            to={`/${slug}`}
            data-plain
            onPointerEnter={() => prefetchChapter(slug)}
            className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-0.5 text-[14px] text-ink-2 hover:border-accent hover:text-ink"
          >
            <span className="num text-muted">{c.number}</span> {c.short}
          </Link>
        );
      })}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// List
// ─────────────────────────────────────────────────────────────────────────────

export function QAIndex() {
  const [query, setQuery] = useState('');
  useEffect(() => {
    document.title = `Questions & answers · ${SITE_TITLE}`;
  }, []);
  const q = query.trim().toLowerCase();
  const shown = q
    ? QA_ENTRIES.filter((e) => `${e.title} ${e.question} ${e.short}`.toLowerCase().includes(q))
    : QA_ENTRIES;
  return (
    <ChapterPage>
      <header className="page-grid mb-10">
        <div className="eyebrow flex items-center gap-2">
          <MessageCircleQuestion aria-hidden="true" size={15} /> Beyond the course
        </div>
        <h1 className="mt-3 text-[34px] leading-[1.12] font-semibold tracking-[-0.022em] text-balance text-ink sm:text-[42px]">
          Questions &amp; answers
        </h1>
        <p className="mt-4 text-[21px] leading-[1.55] text-pretty text-ink-2 sm:text-[22px]">
          Questions that came up while studying the Relational Transformer. Claude answered each one and checked it against the paper
          and the official code at the paper-era commit, and every claim carries a citation, as in the chapters.
        </p>
      </header>
      <div className="page-grid">
        {QA_ENTRIES.length > 4 && (
          <label className="mb-6 block">
            <span className="sr-only">Filter questions</span>
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter questions…"
              className="h-11 w-full rounded-xl border border-border-strong bg-surface px-4 text-[17px] text-ink placeholder:text-muted focus:border-accent focus:outline-none"
            />
          </label>
        )}
        <ol className="!m-0 grid gap-4 !p-0" style={{ listStyle: 'none' }}>
          {shown.map((e) => (
            <li key={e.id} className="!m-0 !p-0">
              <QACard entry={e} />
            </li>
          ))}
        </ol>
        {shown.length === 0 && <p className="text-ink-2">No question matches “{query}”.</p>}
      </div>
    </ChapterPage>
  );
}

function QACard({ entry: e }: { entry: QAEntry }) {
  return (
    <article className="card relative px-5 py-4 transition-colors hover:border-accent">
      <div className="flex items-baseline justify-between gap-3 text-[14.5px] text-muted">
        <span className="eyebrow">Question {e.number}</span>
        <span>{formatDate(e.asked)}</span>
      </div>
      <h2 className="mt-1.5 text-[21px] leading-snug font-semibold text-ink">
        <Link to={`/qa/${e.id}`} data-plain className="after:absolute after:inset-0 after:rounded-xl after:content-['']">
          {e.title}
        </Link>
      </h2>
      <p className="mt-2 !mb-0 text-[17px] leading-[1.6] text-ink-2">{e.short}</p>
      <div className="relative z-[1] mt-3 flex flex-wrap items-center gap-x-3 gap-y-2">
        <ChapterChips slugs={e.chapters} />
        <span className="ml-auto inline-flex items-center gap-1 text-[15px] font-medium text-accent-text">
          Read the answer <ArrowRight aria-hidden="true" size={15} />
        </span>
      </div>
    </article>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// One answer
// ─────────────────────────────────────────────────────────────────────────────

export function QAPage() {
  const { qid } = useParams();
  const entry = getQA(qid);
  useEffect(() => {
    if (entry) document.title = `${entry.title} · Q&A · ${SITE_TITLE}`;
  }, [entry]);
  if (!entry) return <NotFound />;
  const i = QA_ENTRIES.indexOf(entry);
  const prev = QA_ENTRIES[i - 1];
  const next = QA_ENTRIES[i + 1];
  const { Answer } = entry;
  return (
    <ChapterPage>
      <header className="page-grid mb-12">
        <div className="eyebrow flex items-center gap-2">
          <Link to="/qa" data-plain className="hover:text-ink">
            Questions &amp; answers
          </Link>
          <span aria-hidden="true" className="text-axis">
            /
          </span>
          <span>Question {entry.number}</span>
        </div>
        <h1 className="mt-3 text-[34px] leading-[1.12] font-semibold tracking-[-0.022em] text-balance text-ink sm:text-[40px]">{entry.title}</h1>
        <figure className="mt-6 rounded-xl border border-border bg-surface px-5 py-4">
          <figcaption className="eyebrow mb-2 flex flex-wrap items-baseline justify-between gap-2">
            <span>The question</span>
            <span className="text-[14px] font-normal tracking-normal normal-case">Asked {formatDate(entry.asked)}</span>
          </figcaption>
          <blockquote className="!m-0 border-l-2 border-accent !py-0 pl-4 text-[18px] leading-[1.65] text-ink italic">{entry.question}</blockquote>
        </figure>
        <div className="mt-5 rounded-xl border border-accent/40 bg-accent-soft px-5 py-4">
          <div className="eyebrow mb-1.5 !text-accent-text">Short answer</div>
          <p className="!m-0 text-[18.5px] leading-[1.6] text-ink">{entry.short}</p>
        </div>
        <div className="mt-5 flex flex-wrap items-center gap-x-3 gap-y-2 text-[15px] text-muted">
          <span>Related chapters</span>
          <ChapterChips slugs={entry.chapters} />
        </div>
      </header>

      <Answer />

      <aside className="page-grid mt-14">
        <div className="rounded-xl border border-border bg-surface px-5 py-4 text-[15.5px] leading-[1.6] text-ink-2">
          <div className="eyebrow mb-1.5">Checked against</div>
          <ul className="!m-0 list-disc !pl-5">
            {entry.checked.map((c) => (
              <li key={c} className="!my-0.5">
                {c}
              </li>
            ))}
          </ul>
        </div>
      </aside>

      <nav aria-label="Question navigation" className="page-grid mt-16">
        <div className="grid gap-3 border-t border-grid pt-8 sm:grid-cols-2">
          {prev ? <QANavLink entry={prev} dir="prev" /> : <span />}
          {next ? (
            <QANavLink entry={next} dir="next" />
          ) : (
            <Link to="/qa" data-plain className="rounded-xl border border-border bg-surface px-4 py-3.5 text-right hover:border-accent">
              <span className="text-[15px] text-muted">All questions</span>
              <span className="mt-1 block text-[18px] font-semibold text-ink">Back to the list</span>
            </Link>
          )}
        </div>
      </nav>
    </ChapterPage>
  );
}

function QANavLink({ entry, dir }: { entry: QAEntry; dir: 'prev' | 'next' }) {
  const Icon = dir === 'prev' ? ArrowLeft : ArrowRight;
  return (
    <Link
      to={`/qa/${entry.id}`}
      data-plain
      className={cx('group rounded-xl border border-border bg-surface px-4 py-3.5 transition-colors hover:border-accent', dir === 'next' && 'text-right')}
    >
      <span className={cx('flex items-center gap-1.5 text-[15px] text-muted', dir === 'next' && 'justify-end')}>
        {dir === 'prev' && <Icon aria-hidden="true" size={14} />}
        {dir === 'prev' ? 'Previous' : 'Next'} · Question {entry.number}
        {dir === 'next' && <Icon aria-hidden="true" size={14} />}
      </span>
      <span className="mt-1 block text-[18px] font-semibold text-ink">{entry.title}</span>
    </Link>
  );
}
