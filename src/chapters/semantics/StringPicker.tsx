/**
 * Search box over every string that was embedded offline (the store's string list). Nothing can be
 * embedded in the browser: RT itself never runs its encoder at train or inference time (spec §3.6),
 * and this site never fabricates a vector, so the picker only offers strings that exist.
 */
import { Search } from 'lucide-react';
import { useId, useMemo, useRef, useState } from 'react';
import { cx } from '@/components';
import { Phrase } from './bits';
import { DB_SHORT, type Catalog } from './data';

export interface StringPickerProps {
  label: string;
  value: string;
  onChange: (s: string) => void;
  strings: readonly string[];
  catalog: Catalog | null;
  className?: string;
  placeholder?: string;
}

const MAX_RESULTS = 8;

export function StringPicker({ label, value, onChange, strings, catalog, className, placeholder }: StringPickerProps) {
  const id = useId();
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [] as string[];
    const starts: string[] = [];
    const contains: string[] = [];
    for (const s of strings) {
      const l = s.toLowerCase();
      if (l.startsWith(q)) starts.push(s);
      else if (l.includes(q)) contains.push(s);
      if (starts.length >= MAX_RESULTS) break;
    }
    return [...starts, ...contains].slice(0, MAX_RESULTS);
  }, [query, strings]);

  const kindOf = (s: string) => {
    const p = catalog?.byText.get(s);
    if (p) return `phrase · ${DB_SHORT[p.db]}`;
    if (catalog && Object.values(catalog.dbs).some((d) => d.tables.some((t) => t.schema.name === s) || d.tasks.some((t) => t.name === s))) return 'table name';
    return 'value';
  };

  const commit = (s: string) => {
    onChange(s);
    setQuery('');
    setOpen(false);
    inputRef.current?.blur();
  };

  const listId = `${id}-list`;
  const showList = open && results.length > 0;

  return (
    <div className={cx('relative flex min-w-0 flex-col gap-1', className)}>
      <label htmlFor={id} className="text-[15px] font-medium text-ink-2">
        {label}
      </label>
      <div className="relative">
        <Search aria-hidden="true" size={16} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-muted" />
        <input
          ref={inputRef}
          id={id}
          type="text"
          role="combobox"
          aria-expanded={showList}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={showList ? `${id}-opt-${active}` : undefined}
          autoComplete="off"
          spellCheck={false}
          value={query}
          placeholder={placeholder ?? `Search ${strings.length} embedded strings`}
          onChange={(e) => {
            setQuery(e.target.value);
            setActive(0);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => window.setTimeout(() => setOpen(false), 120)}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') {
              e.preventDefault();
              setActive((a) => Math.min(results.length - 1, a + 1));
            } else if (e.key === 'ArrowUp') {
              e.preventDefault();
              setActive((a) => Math.max(0, a - 1));
            } else if (e.key === 'Enter') {
              if (results[active]) {
                e.preventDefault();
                commit(results[active]);
              }
            } else if (e.key === 'Escape') {
              setOpen(false);
              setQuery('');
            }
          }}
          className="h-9 w-full min-w-0 rounded-lg border border-border-strong bg-surface pr-2.5 pl-8 text-[16px] text-ink placeholder:text-muted"
        />
      </div>
      <div className="min-h-[1.3em] truncate text-[14.5px] text-ink-2" aria-live="polite">
        showing <Phrase text={value} plain={!catalog?.byText.has(value)} className="text-[14.5px]" />
      </div>
      {showList && (
        <ul
          id={listId}
          role="listbox"
          aria-label={`${label} suggestions`}
          className="absolute top-[64px] right-0 left-0 z-30 !m-0 max-h-[320px] !list-none overflow-auto rounded-lg border border-border bg-surface !p-1 shadow-[var(--shadow-pop)]"
        >
          {results.map((s, i) => (
            <li
              key={s}
              id={`${id}-opt-${i}`}
              role="option"
              aria-selected={i === active}
              onPointerDown={(e) => {
                e.preventDefault();
                commit(s);
              }}
              onPointerEnter={() => setActive(i)}
              className={cx(
                '!m-0 flex cursor-pointer items-baseline justify-between gap-3 rounded-md px-2.5 py-1.5 text-[15px]',
                i === active ? 'bg-accent-soft' : '',
              )}
            >
              <span className="min-w-0 truncate">
                <Phrase text={s} plain={!catalog?.byText.has(s)} />
              </span>
              <span className="shrink-0 text-[13.5px] text-muted">{kindOf(s)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
