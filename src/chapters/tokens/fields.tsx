/**
 * The per-token fields of the batch dictionary (spec §5.1; fly.rs:27-44, 462-500; data.py:116-154),
 * with what each holds and which part of the paper-era model reads it (model.py:153-267,
 * main.py:283-288). `FieldList` renders them for one live token.
 */
import { Ban, ChevronDown, Crosshair, Eye, Layers, Lock, ScanSearch, type LucideIcon } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Cite, SemBadge, TableSwatch, VectorStrip, cx, formatSigned, truncate } from '@/components';
import type { EmbeddingStore } from '@/engine/embeddings';
import { SEM_TYPE_NAMES, SemType, type PreprocessedDB, type Token } from '@/engine/types';
import { nodeLabel } from './live';
import { ZeroVec } from './ZeroVec';

/** A stored scalar slot: an exact 0 (the default of an unused slot, fly.rs:75-78) prints as "0", not "+0.000". */
export const formatSlot = (v: number, digits?: number) => (v === 0 ? '0' : formatSigned(v, digits));

// ─────────────────────────────────────────────────────────────────────────────
// Metadata
// ─────────────────────────────────────────────────────────────────────────────

export type Reader = 'masks' | 'embed' | 'loss' | 'eval' | 'unused';

export const READER: Record<Reader, { label: string; icon: LucideIcon; title: string; meaning: string; cite?: string }> = {
  masks: {
    label: 'attention masks',
    icon: Eye,
    title: 'Read when the four attention masks are built (model.py:164-191)',
    meaning: 'the four attention masks are built from it',
    cite: 'rt/model.py:164-191',
  },
  embed: {
    label: 'embedding',
    icon: Layers,
    title: 'Read by the input embedding (model.py:208-224)',
    meaning: 'the input embedding reads it',
    cite: 'rt/model.py:208-224',
  },
  loss: {
    label: 'loss',
    icon: Crosshair,
    title: 'Read by the loss at the masked cell (model.py:236-265)',
    meaning: 'the loss at the masked cell reads it',
    cite: 'rt/model.py:236-265',
  },
  eval: {
    label: 'eval readout',
    icon: ScanSearch,
    title: 'Read when evaluation collects predictions and labels (main.py:283-293)',
    meaning: 'evaluation reads it to collect predictions and labels',
    cite: 'rt/main.py:283-293',
  },
  unused: {
    label: 'never read',
    icon: Ban,
    title: 'Carried in the batch, never read by the paper-era model or eval loop',
    meaning: 'carried in the batch, read by neither the model nor the eval loop',
  },
};

export type FieldKey =
  | 'node_idxs'
  | 'f2p_nbr_idxs'
  | 'table_name_idxs'
  | 'col_name_idxs'
  | 'col_name_values'
  | 'sem_types'
  | 'number_values'
  | 'text_values'
  | 'datetime_values'
  | 'boolean_values'
  | 'masks'
  | 'is_targets'
  | 'is_padding'
  | 'is_task_nodes'
  | 'class_value_idxs';

export interface FieldInfo {
  key: FieldKey;
  /** Tensor shape and dtype after data.py (spec §5.1). */
  shape: string;
  readers: Reader[];
  /** One sentence: what the field holds. */
  what: string;
  /** Where it is read or written (code cites). */
  readAt?: string[];
}

export const FIELDS: FieldInfo[] = [
  { key: 'node_idxs', shape: '[B, S] i32', readers: ['masks'], what: 'Global index of the row the cell belongs to.', readAt: ['rt/model.py:168-178'] },
  {
    key: 'f2p_nbr_idxs',
    shape: '[B, S, 5] i32',
    readers: ['masks'],
    what: 'The row’s parent rows, in foreign-key column order, padded with −1 to five slots.',
    readAt: ['rt/model.py:171-178'],
  },
  {
    key: 'table_name_idxs',
    shape: '[B, S] i32',
    readers: ['masks'],
    what: 'Index of the table name in the string table. Train, Val and Test of a task share it.',
    readAt: ['rt/model.py:181-183'],
  },
  { key: 'col_name_idxs', shape: '[B, S] i32', readers: ['masks'], what: 'Index of the phrase “<col> of <table>”.', readAt: ['rt/model.py:181-183'] },
  {
    key: 'col_name_values',
    shape: '[B, S, 384] bf16',
    readers: ['embed'],
    what: 'Frozen MiniLM embedding of that phrase.',
    readAt: ['rt/model.py:209-214'],
  },
  {
    key: 'sem_types',
    shape: '[B, S] i32',
    readers: ['embed', 'loss'],
    what: '0 number, 1 text, 2 datetime, 3 boolean. Picks which value slot and encoder count.',
    readAt: ['rt/model.py:216-224', 'rt/model.py:241'],
  },
  {
    key: 'number_values',
    shape: '[B, S, 1] bf16',
    readers: ['embed', 'loss', 'eval'],
    what: 'Per-column z-score of a number cell; 0 for other types.',
    readAt: ['rt/model.py:216-220', 'rt/model.py:240', 'rt/main.py:288'],
  },
  {
    key: 'text_values',
    shape: '[B, S, 384] bf16',
    readers: ['embed'],
    what: 'MiniLM embedding of a text cell’s string; other types carry the embedding of text[0].',
    readAt: ['rt/model.py:216-220'],
  },
  {
    key: 'datetime_values',
    shape: '[B, S, 1] bf16',
    readers: ['embed', 'loss'],
    what: 'Global z-score of a datetime cell; 0 for other types.',
    readAt: ['rt/model.py:216-220', 'rt/model.py:240'],
  },
  {
    key: 'boolean_values',
    shape: '[B, S, 1] bf16',
    readers: ['embed', 'loss', 'eval'],
    what: 'Per-column z-score of 0/1, so true > 0 and false < 0; 0 for other types.',
    readAt: ['rt/model.py:216-220', 'rt/model.py:240', 'rt/main.py:285'],
  },
  {
    key: 'masks',
    shape: '[B, S] bool',
    readers: ['embed', 'loss'],
    what: 'True only for the seed row’s target cell: its value is hidden and it is where the loss is taken.',
    readAt: ['rt/model.py:216-224', 'rt/model.py:236-241'],
  },
  { key: 'is_targets', shape: '[B, S] bool', readers: ['eval'], what: 'Same as masks in the paper-era sampler.', readAt: ['rt/main.py:283-288'] },
  {
    key: 'is_padding',
    shape: '[B, S] bool',
    readers: ['masks', 'embed'],
    what: 'True for the unused slots after the last cell.',
    readAt: ['rt/model.py:165', 'rt/model.py:213'],
  },
  {
    key: 'is_task_nodes',
    shape: '[B, S] bool',
    readers: ['unused'],
    what: 'True for task-table rows and for cells of the target column.',
    readAt: ['rustler/src/fly.rs:498-499'],
  },
  { key: 'class_value_idxs', shape: '[B, S] i32', readers: ['unused'], what: 'Text index for text cells, −1 otherwise.', readAt: ['rustler/src/fly.rs:471'] },
];

export const FIELD_GROUPS: { title: string; blurb: string; keys: FieldKey[] }[] = [
  {
    title: 'Which cell is it?',
    blurb: 'Integer ids. Only the attention masks read them.',
    keys: ['node_idxs', 'f2p_nbr_idxs', 'table_name_idxs', 'col_name_idxs'],
  },
  {
    title: 'What does it say?',
    blurb: 'The column phrase, the type and the value. The input embedding reads them.',
    keys: ['col_name_values', 'sem_types', 'number_values', 'text_values', 'datetime_values', 'boolean_values'],
  },
  { title: 'Flags', blurb: 'Where to hide a value, where to read a prediction, which slots are empty.', keys: ['masks', 'is_targets', 'is_padding'] },
  { title: 'Carried along, never read', blurb: 'Written by the sampler, ignored by the paper-era model.', keys: ['is_task_nodes', 'class_value_idxs'] },
];

export const FIELD_BY_KEY = Object.fromEntries(FIELDS.map((f) => [f.key, f])) as Record<FieldKey, FieldInfo>;

// ─────────────────────────────────────────────────────────────────────────────
// Small pieces
// ─────────────────────────────────────────────────────────────────────────────

export function ReaderTag({ reader, className }: { reader: Reader; className?: string }) {
  const r = READER[reader];
  const Icon = r.icon;
  return (
    <span
      title={r.title}
      className={cx(
        'inline-flex items-center gap-1 rounded-full border px-2 py-[1px] text-[14px] leading-[1.4] font-medium whitespace-nowrap',
        reader === 'unused' ? 'border-dashed border-border-strong text-ink-2' : 'border-border bg-surface-2 text-ink-2',
        className,
      )}
    >
      <Icon aria-hidden="true" size={13} strokeWidth={2.1} />
      {r.label}
    </span>
  );
}

/** Five parent slots of f2p_nbr_idxs; filled slots show the parent row and whether it is in the context. */
export function ParentSlots({
  ids,
  pdb,
  slots,
  present,
  onPick,
  compact = false,
}: {
  ids: readonly number[];
  pdb: PreprocessedDB;
  slots: Record<string, number>;
  /** nodeIdx → first token position in the context (absent = not in the context). */
  present: (node: number) => number | undefined;
  onPick?: (pos: number) => void;
  compact?: boolean;
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {ids.map((id, j) => {
        if (id < 0)
          return (
            <span
              key={j}
              className="num inline-flex h-[26px] min-w-[30px] items-center justify-center rounded-[6px] border border-dashed border-axis px-1.5 text-[14px] text-ink-2"
              title={`slot ${j}: empty (−1)`}
            >
              −1
            </span>
          );
        const node = pdb.nodes[id];
        const pos = present(id);
        const label = nodeLabel(pdb, id);
        const inner = (
          <>
            <TableSwatch slot={slots[node?.tableName ?? ''] ?? 0} size={8} />
            <span className="num text-ink">{id}</span>
            {!compact && <span className="truncate text-[14px] text-ink-2">{truncate(label, 28)}</span>}
          </>
        );
        const cls =
          'inline-flex h-[26px] max-w-full min-w-0 items-center gap-1.5 rounded-[6px] border border-border bg-surface px-1.5 text-[14px]';
        const title = `slot ${j}: ${label}${pos === undefined ? ' (not in this context)' : ` (in the context from position ${pos})`}`;
        return onPick && pos !== undefined ? (
          <button key={j} type="button" className={cx(cls, 'hover:border-accent hover:bg-surface-2')} title={title} onClick={() => onPick(pos)}>
            {inner}
          </button>
        ) : (
          <span key={j} className={cx(cls, pos === undefined && 'opacity-60')} title={title}>
            {inner}
          </span>
        );
      })}
    </div>
  );
}

const BOOL = (b: boolean) => (
  <span className={cx('num rounded-[5px] px-1.5 py-[1px] text-[14.5px]', b ? 'bg-accent-soft font-semibold text-accent-text' : 'bg-surface-2 text-ink-2')}>
    {b ? 'true' : 'false'}
  </span>
);

const VALUE_SEM: Partial<Record<FieldKey, SemType>> = {
  number_values: SemType.Number,
  text_values: SemType.Text,
  datetime_values: SemType.DateTime,
  boolean_values: SemType.Boolean,
};

// ─────────────────────────────────────────────────────────────────────────────
// FieldList
// ─────────────────────────────────────────────────────────────────────────────

export interface FieldListProps {
  token: Token;
  pdb: PreprocessedDB;
  store: EmbeddingStore | null;
  slots: Record<string, number>;
  present: (node: number) => number | undefined;
  onPick?: (pos: number) => void;
}

/** Every per-token field of one live token, grouped, with what reads it. */
export function FieldList({ token: t, pdb, store, slots, present, onPick }: FieldListProps) {
  const text = pdb.text;
  // Wide screens show every group; phones start with only the first one open (the figure is long).
  const [wide] = useState(() => typeof window === 'undefined' || !window.matchMedia || window.matchMedia('(min-width: 1024px)').matches);
  const valueCell = (k: FieldKey): ReactNode => {
    const sem = VALUE_SEM[k];
    if (sem !== undefined) {
      const active = !t.isPadding && t.sem === sem;
      if (k === 'text_values') {
        const idx = t.textIdx;
        const vec = store && !t.isPadding && store.has(text[idx] ?? '') ? store.get(text[idx]) : null;
        return (
          <div className="min-w-0">
            <div className={cx('text-[15px]', active ? 'text-ink' : 'text-ink-2')}>
              {t.isPadding ? (
                <span>
                  <ZeroVec /> (384 zeros)
                </span>
              ) : (
                <>
                  <span className="num">E(text[{idx}])</span> <span className="font-mono">“{truncate(text[idx] ?? '', 34)}”</span>
                  {!active && <span className="ml-1.5 text-[14px]">· not a text cell, gated out</span>}
                </>
              )}
            </div>
            {vec && active && <VectorStrip values={vec} columns={96} cellHeight={6} showNorm={false} className="mt-1.5" />}
          </div>
        );
      }
      const v = k === 'number_values' ? t.numberValue : k === 'datetime_values' ? t.datetimeValue : t.booleanValue;
      return (
        <div className={cx('flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[15px]', active ? 'text-ink' : 'text-ink-2')}>
          <span className={cx('num', active && 'font-semibold')}>{formatSlot(v)}</span>
          {active && t.mask && (
            <span className="inline-flex items-center gap-1 text-[14px] text-accent-text">
              <Lock aria-hidden="true" size={13} /> the hidden label, stored but gated out by masks
            </span>
          )}
          {active && !t.mask && <span className="text-[14px] text-ink-2">this cell’s value slot</span>}
          {!active && !t.isPadding && <span className="text-[14px]">not this type, gated out</span>}
        </div>
      );
    }
    switch (k) {
      case 'node_idxs':
        return (
          <div className="text-[15px]">
            <span className="num font-semibold text-ink">{t.nodeIdx}</span>
            {!t.isPadding && <span className="ml-2 text-ink-2">{nodeLabel(pdb, t.nodeIdx)}</span>}
          </div>
        );
      case 'f2p_nbr_idxs':
        return <ParentSlots ids={t.f2pNbrIdxs} pdb={pdb} slots={slots} present={present} onPick={onPick} />;
      case 'table_name_idxs':
        return (
          <div className="text-[15px]">
            <span className="num font-semibold text-ink">{t.tableNameIdx}</span>
            <span className="ml-2 font-mono text-ink-2">“{text[t.tableNameIdx]}”</span>
            {t.isPadding && <span className="ml-1.5 text-[14px] text-ink-2">(default 0)</span>}
          </div>
        );
      case 'col_name_idxs':
        return (
          <div className="text-[15px]">
            <span className="num font-semibold text-ink">{t.colNameIdx}</span>
            <span className="ml-2 font-mono text-ink-2">“{text[t.colNameIdx]}”</span>
            {t.isPadding && <span className="ml-1.5 text-[14px] text-ink-2">(default 0)</span>}
          </div>
        );
      case 'col_name_values': {
        if (t.isPadding)
          return (
            <span className="text-[15px] text-ink-2">
              <ZeroVec /> (384 zeros)
            </span>
          );
        const phrase = text[t.colNameIdx] ?? '';
        const vec = store && store.has(phrase) ? store.get(phrase) : null;
        return (
          <div className="min-w-0">
            <div className="text-[15px] text-ink">
              <span className="num">E(</span>
              <span className="font-mono">“{phrase}”</span>
              <span className="num">)</span>
              <span className="ml-1.5 text-[14px] text-ink-2">384 numbers</span>
            </div>
            {vec ? (
              <VectorStrip values={vec} columns={96} cellHeight={6} showNorm={false} className="mt-1.5" />
            ) : (
              <div className="mt-1 text-[14px] text-ink-2">loading embeddings…</div>
            )}
          </div>
        );
      }
      case 'sem_types':
        return (
          <div className="flex items-center gap-2 text-[15px]">
            <span className="num font-semibold text-ink">{t.sem}</span>
            <SemBadge sem={t.sem} size="sm" />
            <span className="text-ink-2">{SEM_TYPE_NAMES[t.sem]}</span>
            {t.isPadding && <span className="text-[14px] text-ink-2">(default 0)</span>}
          </div>
        );
      case 'masks':
        return BOOL(t.mask);
      case 'is_targets':
        return BOOL(t.isTarget);
      case 'is_padding':
        return BOOL(t.isPadding);
      case 'is_task_nodes':
        return BOOL(t.isTaskNode);
      case 'class_value_idxs':
        return <span className="num text-[15px] text-ink-2">{t.classValueIdx}</span>;
      default:
        return null;
    }
  };

  return (
    <div className="min-w-0">
      <div className="mb-3 rounded-lg bg-surface-2 px-3 py-2 text-[14.5px] leading-snug text-ink-2">
        <div className="mb-1 font-medium text-ink">Read by:</div>
        <ul className="space-y-1">
          {(Object.keys(READER) as Reader[]).map((r) => (
            <li key={r} className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5">
              <ReaderTag reader={r} />
              <span>{READER[r].meaning}</span>
              {READER[r].cite && <Cite kind="code" ref={READER[r].cite!} />}
            </li>
          ))}
        </ul>
        <div className="mt-1.5">
          <span className="num">E(·)</span> = frozen MiniLM text embedding (<Link to="/semantics">chapter 5</Link>) ·{' '}
          <span className="num">i32</span> = 32-bit integer · <span className="num">bf16</span> = bfloat16, a 16-bit float ·{' '}
          <span className="num">[B, S]</span> = sequences × positions
        </div>
      </div>
      {FIELD_GROUPS.map((g, gi) => (
        <FieldGroup key={g.title} title={g.title} blurb={g.blurb} defaultOpen={gi === 0 || wide}>
          {g.keys.map((k) => {
            const f = FIELD_BY_KEY[k];
            return (
              <div
                key={k}
                className="grid grid-cols-1 gap-x-3 gap-y-1 border-b border-grid/60 py-2 last:border-0 sm:grid-cols-[150px_minmax(0,1fr)]"
              >
                <div className="min-w-0">
                  <code className="!border-0 !bg-transparent !p-0 font-mono text-[15px] font-semibold text-ink">{k}</code>
                  <div className="num text-[14px] text-ink-2">{f.shape}</div>
                </div>
                <div className="min-w-0">
                  {valueCell(k)}
                  <div className="mt-1 flex flex-wrap items-center gap-1">
                    {f.readers.map((r) => (
                      <ReaderTag key={r} reader={r} />
                    ))}
                    <span className="text-[15px] leading-snug text-ink-2">{f.what}</span>
                    {f.readAt?.map((r) => (
                      <Cite key={r} kind="code" ref={r} />
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </FieldGroup>
      ))}
    </div>
  );
}

/** One collapsible group of fields. Open by default on wide screens; on phones only the first group is. */
function FieldGroup({ title, blurb, defaultOpen, children }: { title: string; blurb: string; defaultOpen: boolean; children: ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="mb-3 last:mb-0">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full flex-wrap items-baseline gap-x-2 border-b border-grid pb-1 text-left hover:text-accent-text"
      >
        <ChevronDown aria-hidden="true" size={16} className={cx('shrink-0 self-center text-ink-2 transition-transform', !open && '-rotate-90')} />
        <span className="text-[15.5px] font-semibold text-ink">{title}</span>
        <span className="text-[14.5px] text-ink-2">{blurb}</span>
      </button>
      {open && children}
    </div>
  );
}
