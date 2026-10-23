/**
 * CodeRef: a collapsible, syntax-highlighted excerpt of the official
 * paper-era code (BUILD_PLAN §5: snippets are copied VERBATIM from
 * snap-stanford/relational-transformer @2a899c8). The header names the repo,
 * commit, file and lines, and links to the GitHub permalink.
 */
import { Check, ChevronDown, Copy, ExternalLink, FileCode2 } from 'lucide-react';
import { Highlight, type PrismTheme } from 'prism-react-renderer';
import { useId, useState, type ReactNode } from 'react';
import { cx } from '../ui/cx';
import { CODE, codeUrl } from './sources';

/** Prism theme driven by CSS variables, so it follows light/dark without re-rendering. */
const THEME: PrismTheme = {
  plain: { color: 'var(--code-ink)', backgroundColor: 'transparent' },
  styles: [
    { types: ['comment', 'prolog', 'doctype', 'cdata'], style: { color: 'var(--code-comment)', fontStyle: 'italic' } },
    { types: ['keyword', 'builtin', 'important', 'atrule', 'boolean'], style: { color: 'var(--code-keyword)' } },
    { types: ['string', 'char', 'attr-value', 'regex', 'inserted'], style: { color: 'var(--code-string)' } },
    { types: ['number', 'constant', 'symbol'], style: { color: 'var(--code-number)' } },
    { types: ['function', 'class-name', 'macro', 'decorator'], style: { color: 'var(--code-function)' } },
    { types: ['punctuation', 'operator'], style: { color: 'var(--code-punct)' } },
    { types: ['deleted'], style: { color: 'var(--critical)' } },
  ],
};

export interface CodeRefProps {
  /** What the excerpt shows, e.g. "The four attention masks". */
  title: ReactNode;
  /** Repo-relative path, e.g. "rt/model.py" or "rustler/src/fly.rs". */
  file: string;
  /** Line range "165-191", a single line, or [start, end]. */
  lines: string | number | [number, number];
  /** The verbatim code. Leading/trailing blank lines are trimmed. */
  code: string;
  /** Prism language: 'python' | 'rust' | 'typescript' | … Default inferred from the file extension. */
  language?: string;
  /** Start expanded. Default false. */
  defaultOpen?: boolean;
  /** Lines to emphasise (absolute line numbers). */
  highlightLines?: number[];
  /** Short note under the header (e.g. what to look for). */
  note?: ReactNode;
  /** Remove the indentation common to all lines (display only). Default true. */
  dedent?: boolean;
  /** Span the full content width when placed on the reading grid. */
  wide?: boolean;
  /** Code from another repository (e.g. RelBench): its name, pinned ref and the permalink. Default: the paper-era RT code. */
  source?: { repo: string; ref: string; url: string };
  className?: string;
}

/** Strip the leading whitespace shared by every non-blank line. */
function dedentCode(code: string): string {
  const lines = code.split('\n');
  const indents = lines.filter((l) => l.trim().length > 0).map((l) => /^[ \t]*/.exec(l)![0].length);
  const n = indents.length ? Math.min(...indents) : 0;
  return n > 0 ? lines.map((l) => l.slice(Math.min(n, /^[ \t]*/.exec(l)![0].length))).join('\n') : code;
}

function parseLines(lines: CodeRefProps['lines']): [number, number | undefined] {
  if (Array.isArray(lines)) return [lines[0], lines[1]];
  if (typeof lines === 'number') return [lines, undefined];
  const m = /^(\d+)(?:\s*[-–]\s*(\d+))?/.exec(lines);
  return m ? [Number(m[1]), m[2] ? Number(m[2]) : undefined] : [1, undefined];
}

function inferLanguage(file: string): string {
  if (file.endsWith('.py')) return 'python';
  if (file.endsWith('.rs')) return 'rust';
  if (file.endsWith('.ts') || file.endsWith('.tsx')) return 'tsx';
  if (file.endsWith('.json')) return 'json';
  return 'plain';
}

export function CodeRef({
  title,
  file,
  lines,
  code,
  language,
  defaultOpen = false,
  highlightLines,
  note,
  dedent = true,
  wide = false,
  source,
  className,
}: CodeRefProps) {
  const [open, setOpen] = useState(defaultOpen);
  const [copied, setCopied] = useState(false);
  const bodyId = useId();
  const [start, end] = parseLines(lines);
  const trimmed = code.replace(/^(?:[ \t]*\n)+/, '').replace(/(?:\n[ \t]*)+$/, '');
  const text = dedent ? dedentCode(trimmed) : trimmed;
  const lineLabel = end && end !== start ? `${start}-${end}` : `${start}`;
  const hl = new Set(highlightLines ?? []);
  const lang = language ?? inferLanguage(file);
  const repo = source ? `${source.repo} @${source.ref}` : `${CODE.repo} @${CODE.commit}`;
  const href = source ? source.url : codeUrl(file, start, end);
  const linkTitle = source ? `Open on GitHub (${repo})` : 'Open on GitHub at the paper-era commit';

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div className={cx('my-6 min-w-0 overflow-hidden rounded-xl border border-border bg-[var(--code-bg)]', wide && 'wide', className)}>
      <div className="flex items-stretch">
        <button
          type="button"
          aria-expanded={open}
          aria-controls={bodyId}
          onClick={() => setOpen((o) => !o)}
          className="flex min-w-0 flex-1 items-start gap-2.5 px-3.5 py-2.5 text-left hover:bg-surface-2/60"
        >
          <FileCode2 aria-hidden="true" size={16} className="mt-[3px] shrink-0 text-muted" />
          <span className="min-w-0 flex-1">
            <span className="block text-[17px] font-semibold text-ink">{title}</span>
            <span className="mt-0.5 flex min-w-0 flex-wrap items-baseline gap-x-2 font-mono text-[14.5px] text-ink-2" title={`${repo} · ${file}:${lineLabel}`}>
              <span className="[overflow-wrap:anywhere]">
                {file}
                <span className="whitespace-nowrap">:{lineLabel}</span>
              </span>
              <span className="min-w-0 truncate text-muted">{repo}</span>
            </span>
          </span>
          <ChevronDown aria-hidden="true" size={16} className={cx('mt-[3px] shrink-0 text-muted transition-transform', open && 'rotate-180')} />
          <span className="sr-only">{open ? 'Hide code' : 'Show code'}</span>
        </button>
        <div className="flex shrink-0 items-start gap-0.5 py-1.5 pr-1.5">
          <button
            type="button"
            onClick={copy}
            title="Copy code"
            aria-label="Copy code"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted hover:bg-surface-2 hover:text-ink"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </button>
          <a
            href={href}
            target="_blank"
            rel="noreferrer noopener"
            data-plain
            title={linkTitle}
            aria-label={linkTitle}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted hover:bg-surface-2 hover:text-ink"
          >
            <ExternalLink size={14} />
          </a>
        </div>
      </div>
      {note && <div className="border-t border-border px-3.5 py-2.5 text-[16px] leading-snug text-ink-2">{note}</div>}
      <div id={bodyId} hidden={!open} className="border-t border-border">
        <Highlight code={text} language={lang} theme={THEME}>
          {({ tokens, getLineProps, getTokenProps }) => (
            <pre className="code-block scroll-quiet overflow-x-auto py-3">
              <code>
                {tokens.map((line, i) => {
                  const n = start + i;
                  const lp = getLineProps({ line });
                  return (
                    <div
                      key={i}
                      {...lp}
                      className={cx(lp.className, 'flex pr-4')}
                      style={{ ...lp.style, background: hl.has(n) ? 'var(--code-line-hl)' : undefined }}
                    >
                      <span aria-hidden="true" className="w-12 shrink-0 pr-3 text-right text-[var(--code-comment)] select-none">
                        {n}
                      </span>
                      <span className="whitespace-pre">
                        {line.map((token, k) => (
                          <span key={k} {...getTokenProps({ token })} />
                        ))}
                      </span>
                    </div>
                  );
                })}
              </code>
            </pre>
          )}
        </Highlight>
      </div>
    </div>
  );
}
