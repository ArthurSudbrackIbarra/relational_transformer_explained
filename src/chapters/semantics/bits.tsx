/** Small presentational pieces shared by the figures of chapter 5. */
import { KeyRound, Link2 } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import type { ReactNode } from 'react';
import { SemBadge, ShapeGlyph, shapeOfSlot, cx } from '@/components';
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
export function RoleGlyph({ role, dtype, size = 'xs' }: { role: PhraseRole; dtype: PhraseInfo['dtype']; size?: 'xs' | 'sm' }) {
  const px = size === 'xs' ? 15 : 17;
  if (role === 'pk') return <KeyRound aria-label="primary key" size={px} className="shrink-0 text-ink-2" />;
  if (role === 'fk') return <Link2 aria-label="foreign key" size={px} className="shrink-0 text-ink-2" />;
  return <SemBadge sem={semOfDType(dtype)} size={size} />;
}

/** Database identity for figures coloured by database: shape + colour + name. */
export function DbTag({ db, className, showName = true }: { db: DbName; className?: string; showName?: boolean }) {
  const slot = DB_SLOT[db];
  return (
    <span className={cx('inline-flex items-center gap-1.5', className)}>
      <ShapeGlyph shape={shapeOfSlot(slot)} size={12} fill={catVar(slot)} />
      {showName && <span className="text-ink">{DB_SHORT[db]}</span>}
    </span>
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
