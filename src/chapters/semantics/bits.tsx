/** Small presentational pieces shared by the figures of chapter 5. */
import { KeyRound, Link2 } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import { Fragment, useEffect, useId, useState, type ReactNode } from 'react';
import { CodeRef, SemBadge, ShapeGlyph, shapeOfSlot, cx, type CodeRefProps } from '@/components';
import { semOfDType } from '@/components/viz/format';
import type { DbName } from '@/engine/db';
import { catVar } from '@/styles/palette';
import { DB_SHORT, DB_SLOT, type PhraseInfo, type PhraseRole } from './data';

/** "<col> of <table>" in mono: the column in ink, the connective and table in secondary ink. */
export function Phrase({ text, className, strong = false, plain = false }: { text: string; className?: string; strong?: boolean; plain?: boolean }) {
  const i = plain ? -1 : text.indexOf(' of ');
  if (i < 0) return <span className={cx('font-mono break-words text-ink', className)}>{text}</span>;
  return (
    <span className={cx('font-mono break-words', className)}>
      <span className={cx('text-ink', strong && 'font-semibold')}>{text.slice(0, i)}</span>
      <span className="text-ink-2"> of {text.slice(i + 4)}</span>
    </span>
  );
}

/** A quoted string value, as embedded (mono). */
export function Quoted({ text, className }: { text: string; className?: string }) {
  return <span className={cx('font-mono text-ink', className)}>“{text}”</span>;
}

/** Key icon / link icon / sem-type glyph of a column. */
export function RoleGlyph({ role, dtype, size = 'sm' }: { role: PhraseRole; dtype: PhraseInfo['dtype']; size?: 'xs' | 'sm' }) {
  const px = size === 'xs' ? 15 : 17;
  if (role === 'pk') return <KeyRound aria-label="primary key" size={px} className="shrink-0 text-ink-2" />;
  if (role === 'fk') return <Link2 aria-label="foreign key" size={px} className="shrink-0 text-ink-2" />;
  return <SemBadge sem={semOfDType(dtype)} size={size} />;
}

/**
 * Database identity: shape + name. Coloured (slot 1-4) only in the figure that colours by database
 * (the PCA map); `neutral` draws the shape in ink, for figures where colour already means table.
 */
export function DbTag({ db, className, showName = true, neutral = false }: { db: DbName; className?: string; showName?: boolean; neutral?: boolean }) {
  const slot = DB_SLOT[db];
  return (
    <span className={cx('inline-flex items-center gap-1.5', className)} title={showName ? undefined : DB_SHORT[db]}>
      <ShapeGlyph shape={shapeOfSlot(slot)} size={12} fill={neutral ? 'var(--ink-2)' : catVar(slot)} />
      {showName ? <span className="text-ink">{DB_SHORT[db]}</span> : <span className="sr-only">{DB_SHORT[db]}</span>}
    </span>
  );
}

/** Keeps its children on one line (e.g. the last word, a citation chip and the full stop after it). */
export function Nw({ children }: { children: ReactNode }) {
  return <span className="whitespace-nowrap">{children}</span>;
}

/** A long identifier that may wrap only after "/", "_", "-" or "." (never inside a word). */
export function BreakId({ text }: { text: string }) {
  const parts = text.split(/(?<=[/_.-])/);
  return (
    <>
      {parts.map((p, i) => (
        <Fragment key={i}>
          {p}
          {i < parts.length - 1 && <wbr />}
        </Fragment>
      ))}
    </>
  );
}

/** True while the media query matches (false where matchMedia is unavailable). */
export function useMedia(query: string): boolean {
  const get = () => typeof window !== 'undefined' && !!window.matchMedia?.(query).matches;
  const [m, setM] = useState(get);
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mql = window.matchMedia(query);
    const on = () => setM(mql.matches);
    on();
    mql.addEventListener('change', on);
    return () => mql.removeEventListener('change', on);
  }, [query]);
  return m;
}

/** Several official-code excerpts behind one set of tabs, so related excerpts read as one block. */
export function CodeTabs({ label, tabs }: { label: string; tabs: (CodeRefProps & { tab: string })[] }) {
  const [i, setI] = useState(0);
  // after the reader picks a tab, show that excerpt expanded (the first view stays collapsed)
  const [touched, setTouched] = useState(false);
  const id = useId();
  const cur = tabs[Math.min(i, tabs.length - 1)];
  const { tab: _tab, ...ref } = cur;
  void _tab;
  return (
    <div className="my-6">
      <div role="tablist" aria-label={label} className="scroll-quiet mb-[-1px] flex gap-1 overflow-x-auto">
        {tabs.map((t, k) => {
          const on = k === i;
          return (
            <button
              key={t.tab}
              id={`${id}-tab-${k}`}
              type="button"
              role="tab"
              aria-selected={on}
              aria-controls={`${id}-panel`}
              tabIndex={on ? 0 : -1}
              onClick={() => (setI(k), setTouched(true))}
              onKeyDown={(e) => {
                if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
                e.preventDefault();
                const n = (k + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length;
                setI(n);
                setTouched(true);
                document.getElementById(`${id}-tab-${n}`)?.focus();
              }}
              className={cx(
                'shrink-0 rounded-t-lg border px-3 py-1.5 font-mono text-[14.5px] whitespace-nowrap transition-colors focus-visible:outline-2 focus-visible:outline-accent',
                on ? 'border-border border-b-transparent bg-surface text-ink' : 'border-transparent text-ink-2 hover:text-ink',
              )}
            >
              {t.tab}
            </button>
          );
        })}
      </div>
      <div id={`${id}-panel`} role="tabpanel" aria-labelledby={`${id}-tab-${i}`}>
        <CodeRef key={cur.tab} {...ref} defaultOpen={touched || ref.defaultOpen} className={cx('!my-0', ref.className)} />
      </div>
    </div>
  );
}

/** WordPiece chips. [CLS] / [SEP] are drawn as quieter special tokens. */
export function Wordpieces({ tokens, className, animateKey }: { tokens: string[]; className?: string; animateKey?: string }) {
  const reduce = useReducedMotion();
  return (
    <div className={cx('flex flex-wrap items-center gap-1.5', className)} aria-label={`${tokens.length} wordpieces: ${tokens.join(' ')}`}>
      {tokens.map((t, i) => {
        const special = t === '[CLS]' || t === '[SEP]';
        return (
          <motion.span
            key={`${animateKey ?? ''}-${i}-${t}`}
            initial={reduce ? false : { opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.22, delay: reduce ? 0 : Math.min(i, 24) * 0.04 }}
            className={cx(
              'inline-flex h-7 items-center rounded-md border px-2 font-mono text-[14.5px]',
              special ? 'border-dashed border-border-strong bg-transparent text-ink-2' : 'border-border bg-surface-2 text-ink',
            )}
          >
            {t}
          </motion.span>
        );
      })}
    </div>
  );
}

/** A labelled number tile used in figure read-outs. */
export function Readout({ label, children, hint, className }: { label: ReactNode; children: ReactNode; hint?: ReactNode; className?: string }) {
  return (
    <div className={cx('min-w-0 rounded-lg bg-surface-2 px-3 py-2', className)}>
      <div className="text-[14.5px] font-medium text-ink-2">{label}</div>
      <div className="num mt-0.5 text-[18px] leading-tight font-semibold text-ink">{children}</div>
      {hint && <div className="mt-0.5 text-[14px] leading-snug text-muted">{hint}</div>}
    </div>
  );
}

/** A small "loading the frozen vectors" placeholder that keeps the figure height. */
export function Loading({ what = 'the precomputed MiniLM vectors', height = 160 }: { what?: string; height?: number }) {
  return (
    <div className="flex items-center justify-center rounded-lg bg-surface-2 text-[15px] text-ink-2" style={{ minHeight: height }}>
      Loading {what}…
    </div>
  );
}

/** Error state (embeddings never fabricated, spec §3.6). */
export function LoadError({ error }: { error: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface-2 px-4 py-3 text-[15px] text-ink-2">
      Could not load the precomputed embeddings ({error}). This figure needs public/data/embeddings; nothing is shown rather than invented.
    </div>
  );
}
