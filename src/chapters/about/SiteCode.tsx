/**
 * SiteCode: a code excerpt from THIS SITE's repository (not the official code, which always goes through
 * the shared <CodeRef/>). The header says so explicitly. Excerpts are read raw from the repository at
 * build time (data.ts → useRawSource), so they are verbatim and their line numbers are computed.
 */
import { Check, ChevronDown, Copy, FileCode2 } from 'lucide-react';
import { Highlight, type PrismTheme } from 'prism-react-renderer';
import { useId, useState, type ReactNode } from 'react';
import { cx } from '@/components';

const THEME: PrismTheme = {
  plain: { color: 'var(--code-ink)', backgroundColor: 'transparent' },
  styles: [
    { types: ['comment', 'prolog', 'doctype', 'cdata'], style: { color: 'var(--code-comment)', fontStyle: 'italic' } },
    { types: ['keyword', 'builtin', 'important', 'atrule', 'boolean'], style: { color: 'var(--code-keyword)' } },
    { types: ['string', 'char', 'attr-value', 'regex', 'inserted'], style: { color: 'var(--code-string)' } },
    { types: ['number', 'constant', 'symbol'], style: { color: 'var(--code-number)' } },
    { types: ['function', 'class-name', 'macro', 'decorator'], style: { color: 'var(--code-function)' } },
    { types: ['punctuation', 'operator'], style: { color: 'var(--code-punct)' } },
  ],
};

export interface SiteCodeProps {
  title: ReactNode;
  /** Repo-relative path in this site's repository. */
  file: string;
  /** First line number of `code` in the file (omit for a command block without line numbers). */
  start?: number;
  code: string | null;
  language?: string;
  defaultOpen?: boolean;
  note?: ReactNode;
  highlightLines?: number[];
  /** Label of the source, default "this site’s code". */
  source?: string;
  className?: string;
  /** Show line numbers (default true when `start` is given). */
  numbers?: boolean;
  /** Always open, no toggle (for short command blocks). */
  static?: boolean;
}

export function SiteCode({
  title,
  file,
  start,
  code,
  language = 'python',
  defaultOpen = false,
  note,
  highlightLines,
  source = 'this site’s code',
  className,
  numbers,
  static: isStatic = false,
}: SiteCodeProps) {
  const [openState, setOpen] = useState(defaultOpen);
  const open = isStatic || openState;
  const [copied, setCopied] = useState(false);
  const bodyId = useId();
  const text = (code ?? '').replace(/^(?:[ \t]*\n)+/, '').replace(/(?:\n[ \t]*)+$/, '');
  const lineCount = text.split('\n').length;
  const showNumbers = numbers ?? start !== undefined;
  const label = start !== undefined ? `${file}:${start}${lineCount > 1 ? `-${start + lineCount - 1}` : ''}` : file;
  const hl = new Set(highlightLines ?? []);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      /* clipboard unavailable */
    }
  };

  const head = (
    <span className="flex min-w-0 flex-1 items-start gap-2.5">
      <FileCode2 aria-hidden="true" size={16} className="mt-[3px] shrink-0 text-muted" />
      <span className="min-w-0 flex-1">
        <span className="block text-[17px] font-semibold text-ink">{title}</span>
        <span className="mt-0.5 flex min-w-0 flex-wrap items-baseline gap-x-2 font-mono text-[14.5px] text-ink-2">
          <span className="break-all">{label}</span>
          <span className="inline-flex items-center rounded-full border border-border bg-surface px-2 font-sans text-[13.5px] font-medium text-ink-2">
            {source}
          </span>
        </span>
      </span>
    </span>
  );

  return (
    <div className={cx('my-6 min-w-0 overflow-hidden rounded-xl border border-dashed border-border-strong bg-[var(--code-bg)]', className)}>
      <div className="flex items-stretch">
        {isStatic ? (
          <div className="flex min-w-0 flex-1 px-3.5 py-2.5">{head}</div>
        ) : (
          <button
            type="button"
            aria-expanded={open}
            aria-controls={bodyId}
            onClick={() => setOpen((o) => !o)}
            className="flex min-w-0 flex-1 items-start gap-2.5 px-3.5 py-2.5 text-left hover:bg-surface-2/60"
          >
            {head}
            <ChevronDown aria-hidden="true" size={16} className={cx('mt-[3px] shrink-0 text-muted transition-transform', open && 'rotate-180')} />
            <span className="sr-only">{open ? 'Hide code' : 'Show code'}</span>
          </button>
        )}
        <div className="flex shrink-0 items-start py-1.5 pr-1.5">
          <button
            type="button"
            onClick={copy}
            disabled={!code}
            title="Copy"
            aria-label="Copy code"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted hover:bg-surface-2 hover:text-ink disabled:opacity-40"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </button>
        </div>
      </div>
      {note && <div className="border-t border-border px-3.5 py-2.5 text-[16px] leading-snug text-ink-2">{note}</div>}
      <div id={bodyId} hidden={!open} className="border-t border-border">
        {code === null ? (
          <div className="px-4 py-3 text-[15px] text-ink-2">Loading the file…</div>
        ) : (
          <Highlight code={text} language={language} theme={THEME}>
            {({ tokens, getLineProps, getTokenProps }) => (
              <pre className="code-block scroll-quiet overflow-x-auto py-3">
                <code>
                  {tokens.map((line, i) => {
                    const n = (start ?? 1) + i;
                    const lp = getLineProps({ line });
                    return (
                      <div
                        key={i}
                        {...lp}
                        className={cx(lp.className, 'flex pr-4', !showNumbers && 'pl-4')}
                        style={{ ...lp.style, background: hl.has(n) ? 'var(--code-line-hl)' : undefined }}
                      >
                        {showNumbers && (
                          <span aria-hidden="true" className="w-12 shrink-0 pr-3 text-right text-[var(--code-comment)] select-none">
                            {n}
                          </span>
                        )}
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
        )}
      </div>
    </div>
  );
}
