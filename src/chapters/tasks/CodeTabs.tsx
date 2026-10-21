/**
 * "In the official code": several verbatim excerpts behind tabs, one at a time, instead of a wall
 * of stacked code cards. Each tab renders the shared CodeRef, expanded, with its file and lines.
 * Arrow keys / Home / End move between tabs (WAI-ARIA tabs pattern).
 */
import { useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { CodeRef } from '@/components';
import type { Snippet } from './codeSnippets';

export interface CodeTab {
  id: string;
  /** Short tab label. */
  label: string;
  s: Snippet;
  title: ReactNode;
  note?: ReactNode;
  highlight?: number[];
}

export function CodeTabs({ title = 'In the official code', intro, tabs }: { title?: ReactNode; intro?: ReactNode; tabs: CodeTab[] }) {
  const [active, setActive] = useState(tabs[0]?.id);
  const base = useId();
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const idx = Math.max(
    0,
    tabs.findIndex((t) => t.id === active),
  );
  const cur = tabs[idx];

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft' && e.key !== 'Home' && e.key !== 'End') return;
    e.preventDefault();
    let j = idx;
    if (e.key === 'ArrowRight') j = (idx + 1) % tabs.length;
    if (e.key === 'ArrowLeft') j = (idx - 1 + tabs.length) % tabs.length;
    if (e.key === 'Home') j = 0;
    if (e.key === 'End') j = tabs.length - 1;
    setActive(tabs[j].id);
    refs.current[j]?.focus();
  };

  return (
    <div className="my-7 min-w-0">
      <div className="text-[17px] font-semibold text-ink">{title}</div>
      {intro && <div className="mt-0.5 text-[16px] leading-snug text-ink-2">{intro}</div>}
      <div role="tablist" aria-label={typeof title === 'string' ? title : 'Code excerpts'} className="mt-2 flex flex-wrap gap-1.5" onKeyDown={onKey}>
        {tabs.map((t, i) => {
          const on = i === idx;
          return (
            <button
              key={t.id}
              ref={(el) => {
                refs.current[i] = el;
              }}
              id={`${base}-tab-${t.id}`}
              role="tab"
              type="button"
              aria-selected={on}
              aria-controls={`${base}-panel`}
              tabIndex={on ? 0 : -1}
              onClick={() => setActive(t.id)}
              className={`rounded-lg border px-2.5 py-1 text-[15px] transition-colors ${
                on ? 'border-accent bg-accent-soft font-medium text-ink' : 'border-border bg-surface text-ink-2 hover:bg-surface-2 hover:text-ink'
              }`}
            >
              {t.label}
            </button>
          );
        })}
      </div>
      <div id={`${base}-panel`} role="tabpanel" aria-labelledby={`${base}-tab-${cur.id}`}>
        <CodeRef
          key={cur.id}
          title={cur.title}
          file={cur.s.file}
          lines={[cur.s.start, cur.s.end]}
          code={cur.s.code}
          note={cur.note}
          highlightLines={cur.highlight}
          defaultOpen
          className="!mt-2.5 !mb-0"
        />
      </div>
    </div>
  );
}
