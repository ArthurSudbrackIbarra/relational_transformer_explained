/**
 * Browse the test suite: every *.test.ts file of the repository is read raw (Vite glob, loaded on demand)
 * and its describe / it titles are extracted, so this list is always the suite as written. The run
 * counts (tests executed, failures) come from verification.json (vitest's JSON report).
 */
import { ChevronDown, FlaskConical, Search } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { StatusBadge, cx } from '@/components';
import { int, useVerification } from './data';

const LOADERS = import.meta.glob(['/src/**/*.test.ts', '/tests/**/*.test.ts'], { query: '?raw', import: 'default' }) as Record<string, () => Promise<string>>;

interface ParsedCase {
  title: string;
  /** describe path */
  group: string;
  each: boolean;
}

interface ParsedFile {
  path: string;
  header: string;
  cases: ParsedCase[];
  lines: number;
}

/** First doc comment of the file, as plain text (" * " prefixes removed). */
function headerOf(src: string): string {
  const m = /^\s*\/\*\*([\s\S]*?)\*\//.exec(src);
  if (!m) return '';
  return m[1]
    .split('\n')
    .map((l) => l.replace(/^\s*\*\s?/, ''))
    .join('\n')
    .trim();
}

const CALL = /\b(describe|it|test)(\.each\([^)]*\))?\(\s*(['"`])((?:\\.|(?!\3)[^\\])*)\3/g;

/**
 * Titles in source order, with the enclosing describe() titles. Nesting is read from indentation (the
 * sources are formatted by prettier): a call belongs to the nearest earlier describe() indented less.
 */
function parse(path: string, src: string): ParsedFile {
  const cases: ParsedCase[] = [];
  const stack: { title: string; indent: number }[] = [];
  for (const m of src.matchAll(CALL)) {
    const at = m.index ?? 0;
    const lineStart = src.lastIndexOf('\n', at) + 1;
    const indent = /^[ \t]*/.exec(src.slice(lineStart, at))![0].length;
    while (stack.length && stack[stack.length - 1].indent >= indent) stack.pop();
    const title = m[4].replace(/\\(['"`])/g, '$1');
    if (m[1] === 'describe') stack.push({ title, indent });
    else cases.push({ title, group: stack.map((x) => x.title).join(' › '), each: !!m[2] || /%s|\$\{/.test(title) || stack.some((x) => /%s|\$\{/.test(x.title)) });
  }
  return { path: path.replace(/^\//, ''), header: headerOf(src), cases, lines: src.split('\n').length };
}

let parsedPromise: Promise<ParsedFile[]> | null = null;
function loadAll(): Promise<ParsedFile[]> {
  if (!parsedPromise) {
    parsedPromise = Promise.all(Object.entries(LOADERS).map(([p, load]) => load().then((src) => parse(p, src)))).then((files) =>
      files.sort((a, b) => a.path.localeCompare(b.path)),
    );
    parsedPromise.catch(() => (parsedPromise = null));
  }
  return parsedPromise;
}

export function TestBrowser() {
  const { data: v } = useVerification();
  const [files, setFiles] = useState<ParsedFile[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [q, setQ] = useState('');

  useEffect(() => {
    let alive = true;
    loadAll().then(
      (f) => alive && setFiles(f),
      (e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      alive = false;
    };
  }, []);

  const needle = q.trim().toLowerCase();
  const shown = useMemo(() => {
    if (!files) return [];
    return files
      .map((f) => ({
        ...f,
        hits: needle ? f.cases.filter((c) => `${c.group} ${c.title}`.toLowerCase().includes(needle)) : f.cases,
      }))
      .filter((f) => !needle || f.hits.length > 0 || f.path.toLowerCase().includes(needle));
  }, [files, needle]);

  if (error) return <StatusBadge status="warning">The test sources could not be loaded: {error}</StatusBadge>;
  const written = files?.reduce((a, f) => a + f.cases.length, 0) ?? 0;

  return (
    <div className="min-w-0">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <label className="relative flex min-w-0 flex-1 basis-[260px] items-center">
          <span className="sr-only">Search the test titles</span>
          <Search aria-hidden="true" size={16} className="pointer-events-none absolute left-3 text-muted" />
          <input
            type="search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={files ? `Search ${written} test titles (try “padding”)` : 'Loading…'}
            className="h-10 w-full rounded-lg border border-border-strong bg-surface pr-3 pl-9 text-[16px] text-ink placeholder:text-muted focus-visible:outline-2 focus-visible:outline-accent"
          />
        </label>
        {v && (
          <StatusBadge status={v.details.testsFailed === 0 ? 'good' : 'critical'}>
            {int(v.details.testsTotal - v.details.testsFailed)} of {int(v.details.testsTotal)} tests passed
          </StatusBadge>
        )}
      </div>
      {files && (
        <p className="mt-2.5 mb-0 text-[15px] leading-snug text-ink-2">
          {files.length} files, {written} test titles in the source. Vitest runs {v ? int(v.details.testsTotal) : '…'} tests in{' '}
          {v ? v.details.testFiles : '…'} suites because loops and <span className="font-mono text-[14.5px]">describe.each</span> repeat some of
          them (one copy per fixture, database or task).
        </p>
      )}
      <ul className="mt-4 grid gap-2 !p-0" style={{ listStyle: 'none' }}>
        {shown.map((f) => {
          const isOpen = open === f.path || (needle !== '' && f.hits.length > 0 && f.hits.length <= 12);
          const firstLine = f.header.split('\n')[0];
          const groups = new Map<string, ParsedCase[]>();
          for (const c of f.hits) groups.set(c.group, [...(groups.get(c.group) ?? []), c]);
          return (
            <li key={f.path} className="!m-0 rounded-lg border border-border bg-surface !pl-0">
              <button
                type="button"
                aria-expanded={isOpen}
                onClick={() => setOpen(open === f.path ? null : f.path)}
                className="flex w-full items-start gap-3 rounded-lg px-3.5 py-2.5 text-left hover:bg-surface-2/70"
              >
                <FlaskConical aria-hidden="true" size={17} className="mt-[3px] shrink-0 text-muted" />
                <span className="min-w-0 flex-1">
                  <span className="flex flex-wrap items-baseline gap-x-2.5">
                    <span className="font-mono text-[15.5px] font-semibold break-words text-ink">
                      {f.path.split('/').map((seg, i, arr) => (
                        <span key={i}>
                          {seg}
                          {i < arr.length - 1 && (
                            <>
                              /<wbr />
                            </>
                          )}
                        </span>
                      ))}
                    </span>
                    <span className="text-[14.5px] text-ink-2">
                      {needle ? `${f.hits.length} of ${f.cases.length}` : f.cases.length} titles · {int(f.lines)} lines
                    </span>
                  </span>
                  {firstLine && <span className="mt-0.5 block text-[15px] leading-snug text-ink-2">{firstLine}</span>}
                </span>
                <ChevronDown aria-hidden="true" size={16} className={cx('mt-[4px] shrink-0 text-muted transition-transform', isOpen && 'rotate-180')} />
              </button>
              {isOpen && (
                <div className="border-t border-border px-3.5 py-3">
                  {f.header.split('\n').length > 1 && (
                    <pre className="scroll-quiet mb-3 overflow-x-auto rounded-md bg-surface-2 px-3 py-2 font-mono text-[14px] leading-relaxed whitespace-pre-wrap text-ink-2">
                      {f.header}
                    </pre>
                  )}
                  {[...groups.entries()].map(([g, cs]) => (
                    <div key={g || '_'} className="mb-2.5 last:mb-0">
                      {g && <div className="mb-1 text-[15px] font-semibold text-ink">{g}</div>}
                      <ul className="grid gap-1 !p-0" style={{ listStyle: 'none' }}>
                        {cs.map((c, i) => (
                          <li key={i} className="!m-0 flex gap-2 !pl-0 text-[15px] leading-snug text-ink-2">
                            <span aria-hidden="true" className="mt-[0.5em] h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
                            <span>
                              {c.title}
                              {c.each && <span className="ml-1.5 rounded border border-border px-1 text-[13.5px] text-muted">repeated</span>}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              )}
            </li>
          );
        })}
      </ul>
      {files && shown.length === 0 && <p className="mt-3 text-[16px] text-ink-2">No test title matches “{q}”.</p>}
    </div>
  );
}
