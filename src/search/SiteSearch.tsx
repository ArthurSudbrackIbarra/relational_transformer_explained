/**
 * Site search in the top bar, as in MkDocs: the search field opens a floating panel that lists the
 * matching sections of every chapter and Q&A answer as you type. Opening a match goes to that
 * section, scrolls to the matched words and highlights them briefly (landing.ts).
 *
 * The index (public/search-index.json, built by scripts/search-index.mjs from the rendered pages)
 * loads on first use. Keys: / or Ctrl/⌘ K opens, ↑ ↓ move, Enter opens, Esc closes.
 */
import { BookOpen, ChevronDown, Hash, MessageCircleQuestion, Search as SearchIcon, X } from 'lucide-react';
import { Fragment, useCallback, useDeferredValue, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { IconButton, Kbd } from '@/components/ui/controls';
import { cx } from '@/components/ui/cx';
import { Portal, useFloatingStyle } from '@/components/ui/floating';
import { markText, parseQuery, prepare, search, snippet, type PreparedPage, type Query, type ResultGroup, type Segment, type SectionHit } from './match';
import type { FindState, SearchIndex } from './types';

/** Sections listed per page before "Show more". */
const SHOWN_PER_PAGE = 3;

let indexPromise: Promise<PreparedPage[]> | null = null;
function loadIndex(): Promise<PreparedPage[]> {
  if (!indexPromise) {
    indexPromise = fetch(`${import.meta.env.BASE_URL}search-index.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`search index: HTTP ${r.status}`);
        return r.json() as Promise<SearchIndex>;
      })
      .then(prepare);
    indexPromise.catch(() => (indexPromise = null)); // a later open retries
  }
  return indexPromise;
}

function useIndex(active: boolean): { pages: PreparedPage[] | null; error: boolean } {
  const [state, setState] = useState<{ pages: PreparedPage[] | null; error: boolean }>({ pages: null, error: false });
  useEffect(() => {
    if (!active || state.pages) return;
    let live = true;
    loadIndex().then(
      (pages) => live && setState({ pages, error: false }),
      () => live && setState({ pages: null, error: true }),
    );
    return () => {
      live = false;
    };
  }, [active, state.pages]);
  return state;
}

type Item = { kind: 'link'; to: string; find?: FindState } | { kind: 'more'; path: string };

interface Row {
  group: ResultGroup;
  header: number;
  hits: { hit: SectionHit; i: number }[];
  more: { n: number; i: number } | null;
}

function findState(q: Query, hit: SectionHit): FindState {
  return { terms: q.terms, phrase: q.phrase, block: hit.block, text: hit.block >= 0 ? hit.section.blocks[hit.block] : null };
}

function Marked({ segs }: { segs: Segment[] }) {
  return (
    <>
      {segs.map((s, i) =>
        s.hit ? (
          <mark key={i} className="rounded-[3px] bg-[var(--search-hit)] px-px text-inherit">
            {s.text}
          </mark>
        ) : (
          <Fragment key={i}>{s.text}</Fragment>
        ),
      )}
    </>
  );
}

export function SiteSearch() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(new Set());
  const anchorRef = useRef<HTMLButtonElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const byKeyboard = useRef(false);
  const listId = useId();
  const navigate = useNavigate();
  const style = useFloatingStyle(open, anchorRef, panelRef, 'bottom', 'end');
  const { pages, error } = useIndex(open);
  const deferred = useDeferredValue(query);
  const q = useMemo(() => parseQuery(deferred), [deferred]);
  const groups = useMemo(() => (pages && q ? search(pages, q) : []), [pages, q]);

  useEffect(() => {
    setActive(0);
    setExpanded(new Set());
  }, [q]);

  // flat list of the options (keyboard order) and the rows that render them
  const { items, rows } = useMemo(() => {
    const items: Item[] = [];
    const rows: Row[] = [];
    if (!q) return { items, rows };
    for (const group of groups) {
      const { page } = group;
      const header = items.push({ kind: 'link', to: page.path, find: group.header ? findState(q, group.header) : undefined }) - 1;
      const n = expanded.has(page.path) ? group.hits.length : SHOWN_PER_PAGE;
      const hits = group.hits.slice(0, n).map((hit) => ({
        hit,
        i: items.push({ kind: 'link', to: `${page.path}?section=${hit.section.id}`, find: findState(q, hit) }) - 1,
      }));
      const rest = group.hits.length - hits.length;
      const more = rest > 0 ? { n: rest, i: items.push({ kind: 'more', path: page.path }) - 1 } : null;
      rows.push({ group, header, hits, more });
    }
    return { items, rows };
  }, [groups, expanded, q]);

  const close = useCallback((refocus: boolean) => {
    setOpen(false);
    if (refocus) anchorRef.current?.focus();
  }, []);

  const choose = (item: Item | undefined) => {
    if (!item) return;
    if (item.kind === 'more') {
      setExpanded((s) => new Set(s).add(item.path));
      return;
    }
    setOpen(false);
    navigate(item.to, { state: item.find ? { find: item.find } : undefined });
  };

  // "/" (outside text fields) and Ctrl/⌘ K open the search from anywhere
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const typing = !!t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
      if ((e.key === 'k' || e.key === 'K') && (e.metaKey || e.ctrlKey) && !e.altKey) {
        e.preventDefault();
        setOpen(true);
      } else if (e.key === '/' && !typing && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        setOpen(true);
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  // focus the field as soon as the panel is placed (a hidden input cannot take focus), before the next key
  const placed = open && style.visibility === 'visible';
  useLayoutEffect(() => {
    if (!placed) return;
    inputRef.current?.focus();
    inputRef.current?.select();
  }, [placed]);

  // Esc closes wherever focus is
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      e.preventDefault();
      close(true);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, close]);

  useEffect(() => {
    if (!byKeyboard.current) return;
    byKeyboard.current = false;
    document.getElementById(`${listId}-o${active}`)?.scrollIntoView({ block: 'nearest' });
  }, [active, listId]);

  const move = (d: number) => {
    if (!items.length) return;
    byKeyboard.current = true;
    setActive((a) => (a + d + items.length) % items.length);
  };

  const option = (i: number) => ({
    id: `${listId}-o${i}`,
    role: 'option' as const,
    'aria-selected': i === active,
    onMouseMove: () => i !== active && setActive(i),
    // keep focus (and the keyboard) in the search field
    onMouseDown: (e: React.MouseEvent) => e.preventDefault(),
  });

  const nSections = groups.reduce((n, g) => n + g.hits.length + (g.header ? 1 : 0), 0);
  let status: string;
  if (error) status = 'The search index failed to load. Close the search and try again.';
  else if (!query.trim()) status = '';
  else if (!pages) status = 'Loading the search index…';
  else if (!q) status = 'Type at least two letters.';
  else if (!groups.length) status = `No matches for “${query.trim()}”. Try fewer or shorter words.`;
  else status = `${nSections} matching ${nSections === 1 ? 'section' : 'sections'} in ${groups.length} ${groups.length === 1 ? 'page' : 'pages'}`;

  return (
    <>
      <button
        ref={anchorRef}
        type="button"
        onClick={() => setOpen(true)}
        onPointerEnter={() => void loadIndex().catch(() => undefined)}
        onFocus={() => void loadIndex().catch(() => undefined)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label="Search the site"
        title="Search (/)"
        className={cx(
          'inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink',
          'xl:w-[200px] xl:justify-start xl:gap-2 xl:border xl:border-border xl:bg-surface xl:px-2.5',
          open && 'bg-accent-soft text-accent-text xl:border-accent',
        )}
      >
        <SearchIcon aria-hidden="true" size={17} className="shrink-0" />
        <span className="hidden text-[15px] text-muted xl:inline">Search</span>
        <span className="ml-auto hidden xl:inline-flex">
          <Kbd>/</Kbd>
        </span>
      </button>
      {open && (
        <Portal>
          <div className="fixed inset-0 z-[64] bg-black/25" onClick={() => close(false)} aria-hidden="true" />
          <div
            ref={panelRef}
            role="dialog"
            aria-modal="true"
            aria-label="Search"
            style={{ ...style, width: 'min(640px, calc(100vw - 16px))', zIndex: 65 }}
            className="flex max-h-[min(720px,calc(100dvh-72px))] flex-col overflow-hidden rounded-2xl border border-border bg-surface shadow-[var(--shadow-pop)]"
          >
            <div className="flex items-center gap-2 border-b border-border pr-2 pl-4">
              <SearchIcon aria-hidden="true" size={18} className="shrink-0 text-muted" />
              <input
                ref={inputRef}
                type="search"
                role="combobox"
                aria-label="Search the chapters and answers"
                aria-expanded={items.length > 0}
                aria-controls={listId}
                aria-autocomplete="list"
                aria-activedescendant={items[active] ? `${listId}-o${active}` : undefined}
                placeholder="Search the chapters and answers"
                autoComplete="off"
                spellCheck={false}
                enterKeyHint="go"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    move(1);
                  } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    move(-1);
                  } else if (e.key === 'Enter') {
                    e.preventDefault();
                    choose(items[active]);
                  }
                }}
                className="h-14 min-w-0 flex-1 bg-transparent text-[17.5px] text-ink placeholder:text-muted focus:outline-none [&::-webkit-search-cancel-button]:hidden"
              />
              {query && (
                <IconButton
                  label="Clear the search"
                  size="sm"
                  onClick={() => {
                    setQuery('');
                    inputRef.current?.focus();
                  }}
                >
                  <X size={16} />
                </IconButton>
              )}
              <button
                type="button"
                onClick={() => close(true)}
                aria-label="Close the search"
                className="inline-flex h-8 shrink-0 items-center rounded-lg px-2 text-[15px] font-medium text-ink-2 hover:bg-surface-2 hover:text-ink"
              >
                <span className="sm:hidden">Close</span>
                <span className="hidden sm:inline-flex">
                  <Kbd>Esc</Kbd>
                </span>
              </button>
            </div>

            <div className="scroll-quiet min-h-0 flex-1 overflow-y-auto overscroll-contain p-2">
              <div aria-live="polite" className={cx('px-3 text-[14.5px] text-muted', status ? 'pt-1.5 pb-2' : 'sr-only')}>
                {status}
              </div>
              {!query.trim() && !error && <EmptyState />}
              {rows.length > 0 && q && (
                <div role="listbox" id={listId} aria-label="Search results">
                  {rows.map(({ group, header, hits, more }) => {
                    const { page } = group;
                    const Icon = page.kind === 'qa' ? MessageCircleQuestion : BookOpen;
                    const intro = group.header && group.header.block >= 0 ? group.header.section.blocks[group.header.block] : null;
                    return (
                      <div key={page.path} role="group" aria-label={`${page.label}: ${page.title}`} className="mb-2">
                        <Link
                          to={page.path}
                          state={group.header ? { find: findState(q, group.header) } : undefined}
                          onClick={() => setOpen(false)}
                          tabIndex={-1}
                          data-plain
                          {...option(header)}
                          className={cx('block rounded-xl px-3 py-2.5', header === active && 'bg-surface-2')}
                        >
                          <span className="flex items-center gap-1.5 text-[14.5px] text-muted">
                            <Icon aria-hidden="true" size={15} /> {page.label}
                          </span>
                          <span className="mt-0.5 block text-[17px] leading-snug font-semibold text-ink">
                            <Marked segs={markText(page.title, q)} />
                          </span>
                          {intro && (
                            <span className="mt-1 line-clamp-2 block text-[15px] leading-[1.55] text-ink-2">
                              <Marked segs={snippet(intro, q)} />
                            </span>
                          )}
                        </Link>
                        {hits.map(({ hit, i }) => (
                          <Link
                            key={hit.section.id}
                            to={`${page.path}?section=${hit.section.id}`}
                            state={{ find: findState(q, hit) }}
                            onClick={() => setOpen(false)}
                            tabIndex={-1}
                            data-plain
                            {...option(i)}
                            className={cx(
                              'ml-3 block rounded-r-xl border-l-2 py-2 pr-3 pl-3.5',
                              i === active ? 'border-accent bg-surface-2' : 'border-grid',
                            )}
                          >
                            <span className="flex items-start gap-1.5 text-[16px] leading-snug font-medium text-ink">
                              <Hash aria-hidden="true" size={14} className="mt-[0.28em] shrink-0 text-muted" />
                              <span>
                                <Marked segs={markText(hit.section.title, q)} />
                              </span>
                            </span>
                            {hit.block >= 0 && (
                              <span className="mt-1 line-clamp-2 block pl-5 text-[15px] leading-[1.55] text-ink-2">
                                <Marked segs={snippet(hit.section.blocks[hit.block], q)} />
                              </span>
                            )}
                          </Link>
                        ))}
                        {more && (
                          <div
                            {...option(more.i)}
                            onClick={() => choose(items[more.i])}
                            className={cx(
                              'ml-3 flex cursor-pointer items-center gap-1.5 rounded-r-xl border-l-2 py-1.5 pr-3 pl-3.5 text-[15px] font-medium text-accent-text',
                              more.i === active ? 'border-accent bg-surface-2' : 'border-grid',
                            )}
                          >
                            <ChevronDown aria-hidden="true" size={15} /> Show {more.n} more {more.n === 1 ? 'match' : 'matches'} in this{' '}
                            {page.kind === 'qa' ? 'answer' : 'chapter'}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </Portal>
      )}
    </>
  );
}

function EmptyState() {
  return (
    <div className="px-3 pt-2 pb-3 text-[15.5px] leading-[1.6] text-ink-2">
      <p className="!m-0">
        Search every chapter and question. Matches appear as you type; open one to go straight to that passage, highlighted for a
        moment.
      </p>
      <div className="mt-3 hidden flex-wrap gap-x-5 gap-y-2 text-[15px] text-muted sm:flex">
        <span className="inline-flex items-center gap-1.5">
          <Kbd>↑</Kbd>
          <Kbd>↓</Kbd> move
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Kbd>Enter</Kbd> open
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Kbd>/</Kbd> or <Kbd>Ctrl K</Kbd> search from any page
        </span>
      </div>
    </div>
  );
}
