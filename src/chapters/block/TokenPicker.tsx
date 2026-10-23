/**
 * Token picker for the block chapter: preset chips derived from the live context, plus a disclosure
 * with every token of the context (TokenStrip). Chapter-local state: it never touches the store.
 */
import { ChevronDown } from 'lucide-react';
import { useState } from 'react';
import { TokenStrip, cx } from '@/components';
import type { BlockLive, TokenPreset } from './useBlockLive';
import { rowName, valueOf } from './utils';

export function PresetChips({
  presets,
  value,
  onChange,
  label,
}: {
  presets: TokenPreset[];
  value: number;
  onChange: (pos: number) => void;
  label: string;
}) {
  return (
    <div className="flex flex-wrap gap-1.5" role="group" aria-label={label}>
      {presets.map((p) => (
        <button
          key={p.id}
          type="button"
          aria-pressed={value === p.pos}
          onClick={() => onChange(p.pos)}
          className={cx(
            'min-h-8 rounded-full border px-3 py-0.5 text-left text-[15px] font-medium transition-colors',
            value === p.pos ? 'border-accent bg-accent-soft text-accent-text' : 'border-border bg-surface text-ink-2 hover:bg-surface-2 hover:text-ink',
          )}
        >
          {p.label}
        </button>
      ))}
    </div>
  );
}

export function TokenPicker({
  live,
  value,
  onChange,
  presets,
  label = 'Token presets',
  stripLabel = 'Context tokens: click one to choose it',
}: {
  live: BlockLive;
  value: number;
  onChange: (pos: number) => void;
  presets?: TokenPreset[];
  label?: string;
  stripLabel?: string;
}) {
  const [open, setOpen] = useState(false);
  const list = presets ?? live.presets;
  const custom = !list.some((p) => p.pos === value);
  return (
    <div className="grid grid-cols-1 min-w-0 gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <PresetChips presets={list} value={value} onChange={onChange} label={label} />
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen((o) => !o)}
          className={cx(
            'inline-flex min-h-8 items-center gap-1 rounded-full border px-3 py-0.5 text-[15px] font-medium transition-colors',
            custom ? 'border-accent bg-accent-soft text-accent-text' : 'border-dashed border-border-strong text-ink-2 hover:bg-surface-2 hover:text-ink',
          )}
        >
          {custom ? `token #${value}` : 'any token'}
          <ChevronDown aria-hidden="true" size={15} className={cx('transition-transform', open && 'rotate-180')} />
        </button>
      </div>
      {open && (
        <TokenStrip
          tokens={live.ctx.tokens}
          tableSlots={live.slots}
          variant="cells"
          selected={value}
          onSelect={(p) => onChange(p)}
          label={stripLabel}
        />
      )}
    </div>
  );
}

/** One line naming a token: "#12 c35·age · age of customers[35] · 39". */
export function TokenCaption({ live, pos, prefix = 'Token' }: { live: BlockLive; pos: number; prefix?: string }) {
  const t = live.ctx.tokens[pos];
  if (!t) return null;
  return (
    <p className="!mb-0 text-[15.5px] leading-snug text-ink-2">
      <span className="eyebrow mr-1.5 !text-[13.5px]">{prefix}</span>
      <span className="num text-ink">
        #{pos} {live.labels[pos]}
      </span>{' '}
      · {t.colName} of {rowName(t)} ·{' '}
      {pos === live.target ? <strong className="font-semibold text-accent-text">masked target “?”</strong> : <span className="num text-ink">{valueOf(t)}</span>}
    </p>
  );
}
