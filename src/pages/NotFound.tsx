/** 404 page: unknown route → suggestions from the chapter list. */
import { useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { CHAPTERS } from '@/chapters/registry';
import { SITE_TITLE } from '@/components/prose/sources';
import { ContentColumn } from '@/components/shell/Layout';

export function NotFound() {
  const { pathname } = useLocation();
  useEffect(() => {
    document.title = `Not found · ${SITE_TITLE}`;
  }, []);
  const q = pathname.replace(/^\/+/, '').toLowerCase();
  const suggestions = CHAPTERS.filter((c) => q && (c.slug.includes(q) || q.includes(c.slug) || c.title.toLowerCase().includes(q))).slice(0, 4);
  const list = suggestions.length ? suggestions : CHAPTERS.slice(0, 4);
  return (
    <ContentColumn className="py-20">
      <div className="mx-auto max-w-[560px]">
        <div className="eyebrow">404</div>
        <h1 className="mt-3 text-[34px] leading-tight font-semibold tracking-[-0.02em] text-ink">This page is a padding token</h1>
        <p className="mt-3 text-[19px] leading-[1.6] text-ink-2">
          Nothing lives at <span className="font-mono text-[0.92em] text-ink">{pathname}</span>. Like padding in a context window,
          it attends to nothing and nothing attends to it.
        </p>
        <h2 className="mt-8 text-[16.5px] font-semibold text-ink">{suggestions.length ? 'Did you mean' : 'Start here'}</h2>
        <ul className="mt-3 grid gap-2">
          {list.map((c) => (
            <li key={c.slug}>
              <Link to={`/${c.slug}`} data-plain className="flex items-baseline gap-3 rounded-lg border border-border bg-surface px-4 py-3 hover:border-accent">
                <span className="num text-[15.5px] text-muted">{String(c.number).padStart(2, '0')}</span>
                <span className="text-[17.5px] font-medium text-ink">{c.title}</span>
              </Link>
            </li>
          ))}
        </ul>
        <p className="mt-6 text-[16.5px]">
          <Link to="/" className="text-accent-text underline underline-offset-2">
            Back to the course map
          </Link>
        </p>
      </div>
    </ContentColumn>
  );
}
