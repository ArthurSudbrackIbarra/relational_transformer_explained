/**
 * Cite: inline citation chip (BUILD_PLAN §0.6, §5). Hover / focus / tap shows
 * the full reference.
 *
 *   <Cite kind="spec" ref="§4.6" />
 *   <Cite kind="paper" ref="p.5 §3.2" title="Relational attention" />
 *   <Cite kind="code" ref="rustler/src/fly.rs:421-424" />
 *   <Cite kind="url" ref="https://huggingface.co/…" title="all-MiniLM-L12-v2 card" />
 *
 * Note: `ref` is a plain string prop here (React 19 passes `ref` through to
 * function components as a prop; it is never attached to a DOM node).
 */
import { ExternalLink } from 'lucide-react';
import type { ReactNode } from 'react';
import { HoverCard } from '../ui/floating';
import { cx } from '../ui/cx';
import { CODE, codeUrl, PAPER, paperUrl, parseCodeRef, SPEC_PATH } from './sources';

export type CiteKind = 'spec' | 'paper' | 'code' | 'url';

export interface CiteProps {
  kind: CiteKind;
  /** spec: "§4.6" · paper: "p.5 §3.2" · code: "rt/model.py:20-47" · url: the URL. */
  ref: string;
  /** Optional human title shown in the card (e.g. what the cited lines do). */
  title?: string;
  /** Optional extra note in the card. */
  note?: ReactNode;
  /** Optional chip text instead of the default short label (e.g. a file name for a url cite). */
  label?: string;
  className?: string;
}

/** Short chip text per kind. */
function chipLabel(kind: CiteKind, ref: string): string {
  switch (kind) {
    case 'spec':
      return ref.startsWith('§') ? ref : `§${ref}`;
    case 'paper': {
      const p = /p\.?\s*(\d+(?:\s*[-–]\s*\d+)?)/.exec(ref);
      return p ? `p.${p[1].replace(/\s/g, '')}` : ref;
    }
    case 'code': {
      const { file, start, end } = parseCodeRef(ref);
      const base = file.split('/').pop() ?? file;
      return start ? `${base}:${start}${end && end !== start ? `–${end}` : ''}` : base;
    }
    case 'url':
      try {
        return new URL(ref).hostname.replace(/^www\./, '');
      } catch {
        return ref;
      }
  }
}

const KIND_NAME: Record<CiteKind, string> = { spec: 'Spec', paper: 'Paper', code: 'Code', url: 'Web' };

export function Cite({ kind, ref, title, note, label: chip, className }: CiteProps) {
  const label = chip ?? chipLabel(kind, ref);
  let href: string | null = null;
  let full: ReactNode;
  switch (kind) {
    case 'spec':
      full = (
        <>
          Technical specification {label}
          <span className="mt-0.5 block font-mono text-[14.5px] break-all text-muted">{SPEC_PATH}</span>
        </>
      );
      break;
    case 'paper':
      href = paperUrl(ref);
      full = (
        <>
          {PAPER.authors}, <em>{PAPER.title}</em>, {PAPER.venue} (arXiv {PAPER.arxiv}), {ref}
        </>
      );
      break;
    case 'code': {
      const { file, start, end } = parseCodeRef(ref);
      href = codeUrl(file, start, end);
      full = (
        <span className="font-mono text-[15px] break-all">
          {CODE.repo} @{CODE.commit} · {ref}
        </span>
      );
      break;
    }
    case 'url':
      href = ref;
      full = <span className="font-mono text-[15px] break-all">{ref}</span>;
      break;
  }

  return (
    <HoverCard
      placement="top"
      width={360}
      label={`${KIND_NAME[kind]} reference`}
      interactive={kind !== 'spec'}
      trigger={(p) => (
        <span
          {...p}
          ref={p.ref}
          role="button"
          tabIndex={0}
          className={cx(
            'tap-pad mx-[0.15em] inline-flex -translate-y-[0.2em] cursor-help items-center rounded-[5px] border border-border bg-surface-2 px-[0.38em] align-baseline text-[length:max(13.5px,0.74em)] leading-[1.4] font-medium whitespace-nowrap text-ink-2 tabular-nums transition-colors hover:border-accent hover:text-accent-text',
            kind === 'code' && 'font-mono',
            p['aria-expanded'] && 'border-accent text-accent-text',
            className,
          )}
          onKeyDown={(e) => {
            p.onKeyDown(e);
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              p.onClick(e as unknown as React.MouseEvent);
            }
          }}
        >
          <span className="sr-only">{KIND_NAME[kind]} citation: </span>
          {label}
        </span>
      )}
    >
      <div className="eyebrow mb-1">{KIND_NAME[kind]}</div>
      {title && <div className="mb-1 text-[16.5px] font-semibold text-ink">{title}</div>}
      <div className="text-[16px] leading-snug text-ink-2">{full}</div>
      {note && <div className="mt-1.5 text-[15.5px] text-ink-2">{note}</div>}
      {href && (
        <a
          href={href}
          target="_blank"
          rel="noreferrer noopener"
          data-plain
          className="mt-2 inline-flex items-center gap-1.5 text-[15.5px] font-medium text-accent-text hover:underline"
        >
          Open source <ExternalLink aria-hidden="true" size={14} />
        </a>
      )}
    </HoverCard>
  );
}
