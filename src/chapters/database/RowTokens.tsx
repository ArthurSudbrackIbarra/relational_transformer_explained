/**
 * "What RT keeps of a row": the cells of one row fly into three bins. Feature
 * cells that are not missing become tokens (v, c, t) (paper p.4 §3.1, p.16 App. A);
 * each non-null FK becomes a link to its parent (pre.rs:444-533); the PK and
 * missing cells produce nothing (pre.rs:439-442, 552; spec §1.2, §3.5).
 */
import { LayoutGroup, motion } from 'motion/react';
import { KeyRound, Link2, Scissors, Undo2 } from 'lucide-react';
import { memo, useMemo, useState, type ReactNode } from 'react';
import { Button, Cite, Select, SemBadge, cx, formatRaw, semOfDType } from '@/components';
import type { RawDatabase, RawTable } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { classifyRow, columnKind, isMissing, rowCaption, rowKey, rowLabel, type LinkIndex, type RowRef } from './model';

export interface RowTokensProps {
  db: RawDatabase;
  idx: LinkIndex;
  slots: Record<string, number>;
  focus: RowRef;
}

/** One example row per table: the first row with a missing cell, else row 0. */
function examplesOf(db: RawDatabase): RowRef[] {
  return db.tables.map((t) => {
    const r = t.rows.findIndex((rec) => t.schema.columns.some((c) => isMissing(rec[c.name])));
    return { table: t.schema.name, row: r >= 0 ? r : 0 };
  });
}

export const RowTokens = memo(function RowTokens({ db, idx, slots, focus }: RowTokensProps) {
  const examples = useMemo(() => examplesOf(db), [db]);
  const [choice, setChoice] = useState<string>('focus');
  const [split, setSplit] = useState(false);
  const at: RowRef = choice === 'focus' ? focus : (examples.find((e) => rowKey(e) === choice) ?? focus);
  const table = idx.table(at.table) ?? db.tables[0];
  const rec = table.rows[at.row] ?? {};
  const cls = classifyRow(table, at.row);
  const slot = slots[at.table] ?? 0;
  const k = rowKey(at);

  const cell = (col: string) => <CellChip key={col} table={table} col={col} value={rec[col]} split={split} layoutId={`${k}:${col}`} slot={slot} />;

  return (
    <div className="min-w-0">
      <div className="mb-4 flex flex-wrap items-end gap-x-4 gap-y-3">
        <Select
          label="Row"
          value={choice}
          onChange={(e) => {
            setChoice(e.target.value);
            setSplit(false);
          }}
          className="min-w-[220px]"
        >
          <option value="focus">
            Focus row: {rowLabel(focus)} {rowCaption(db, focus, 22)}
          </option>
          {examples.map((e) => {
            const t = idx.table(e.table)!;
            const miss = t.schema.columns.filter((c) => isMissing(t.rows[e.row]?.[c.name])).map((c) => c.name);
            return (
              <option key={rowKey(e)} value={rowKey(e)}>
                {rowLabel(e)} {miss.length ? `(missing ${miss.join(', ')})` : ''}
              </option>
            );
          })}
        </Select>
        <Button variant="primary" size="sm" icon={split ? <Undo2 size={14} /> : <Scissors size={14} />} onClick={() => setSplit((s) => !s)}>
          {split ? 'Put the row back together' : 'Split the row as RT does'}
        </Button>
      </div>

      <LayoutGroup id={`rowtokens-${k}`}>
        {!split ? (
          <div className="rounded-xl border border-border" style={{ borderLeft: `3px solid ${catVar(slot)}` }}>
            <div className="flex items-baseline gap-2 border-b border-grid px-3 py-1.5">
              <span className="text-[15.5px] font-semibold text-ink">{rowLabel(at)}</span>
              <span className="text-[14.5px] text-ink-2">as stored: {table.schema.columns.length} columns</span>
            </div>
            <div className="flex flex-wrap gap-2 p-3">{table.schema.columns.map((c) => cell(c.name))}</div>
          </div>
        ) : (
          <div className="grid gap-3 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1.15fr)_minmax(0,1fr)]">
            <Bin
              title={`${cls.tokens.length} token${cls.tokens.length === 1 ? '' : 's'}`}
              sub="non-missing feature cells, each a (value, column, table) triple"
              tone="token"
            >
              {cls.tokens.map(cell)}
            </Bin>
            <Bin title={`${cls.links.length} link${cls.links.length === 1 ? '' : 's'}`} sub="foreign keys: structure, no token" tone="link">
              {cls.links.length ? cls.links.map(cell) : <Empty>no foreign keys in {at.table}</Empty>}
            </Bin>
            <Bin title="nothing" sub="primary key and missing values" tone="none">
              {[...cls.pk, ...cls.missing].length ? [...cls.pk, ...cls.missing].map(cell) : <Empty>nothing dropped</Empty>}
            </Bin>
          </div>
        )}
      </LayoutGroup>

      <p className="!mt-3 !mb-0 text-[15.5px] leading-snug text-ink-2" aria-live="polite">
        {split ? (
          <>
            <span className="text-ink">{rowLabel(at)}</span> has {table.schema.columns.length} columns. RT turns it into{' '}
            <strong className="text-ink">{cls.tokens.length} tokens</strong> and {cls.links.length} link{cls.links.length === 1 ? '' : 's'}
            {cls.missing.length > 0 && (
              <>
                ; the missing {cls.missing.join(', ')} cell{cls.missing.length === 1 ? '' : 's'} simply produce
                {cls.missing.length === 1 ? 's' : ''} no token, without imputation or a mask <Cite kind="spec" ref="§3.5" />
              </>
            )}
            .
          </>
        ) : (
          'Press the button to split the row the way RT’s preprocessing does.'
        )}
      </p>
    </div>
  );
});

function Bin({ title, sub, tone, children }: { title: string; sub: string; tone: 'token' | 'link' | 'none'; children: ReactNode }) {
  return (
    <div
      className={cx(
        'min-w-0 rounded-xl border px-3 py-2.5',
        tone === 'token' ? 'border-accent bg-accent-soft' : tone === 'link' ? 'border-border-strong border-dashed' : 'border-border bg-surface-2',
      )}
    >
      <div className="text-[15.5px] font-semibold text-ink">{title}</div>
      <div className="mb-2 text-[14px] leading-snug text-ink-2">{sub}</div>
      <div className="flex flex-wrap gap-2">{children}</div>
    </div>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return <div className="text-[15px] text-ink-2">{children}</div>;
}

function CellChip({
  table,
  col,
  value,
  split,
  layoutId,
  slot,
}: {
  table: RawTable;
  col: string;
  value: RawTable['rows'][number][string];
  split: boolean;
  layoutId: string;
  slot: number;
}) {
  const c = table.schema.columns.find((x) => x.name === col)!;
  const kind = columnKind(table, col);
  const miss = isMissing(value);
  const isTime = table.schema.timeCol === col;
  const dropped = kind === 'pk' || miss;
  return (
    <motion.div
      layoutId={layoutId}
      layout="position"
      transition={{ type: 'spring', stiffness: 380, damping: 32 }}
      className={cx(
        'min-w-0 rounded-lg border px-2.5 py-1.5',
        split && kind === 'feature' && !miss ? 'border-border bg-surface shadow-[var(--shadow-card)]' : 'border-border bg-surface',
        split && dropped && 'border-dashed',
      )}
      style={split && kind === 'feature' && !miss ? { borderLeft: `3px solid ${catVar(slot)}` } : undefined}
    >
      <div className="flex items-center gap-1.5 text-[13.5px] text-ink-2">
        {kind === 'pk' ? (
          <KeyRound aria-label="primary key" size={14} />
        ) : kind === 'fk' ? (
          <Link2 aria-label="foreign key" size={14} />
        ) : (
          <SemBadge sem={semOfDType(c.dtype)} size="sm" />
        )}
        <span className={cx(split && dropped && 'line-through decoration-[1.5px]')}>{col}</span>
        {isTime && <span className="rounded bg-surface-3 px-1 text-[13px] font-medium">time</span>}
      </div>
      <div
        className={cx(
          'mt-0.5 max-w-[220px] truncate text-[15.5px]',
          c.dtype !== 'str' && 'num',
          split && dropped ? 'text-ink-2 line-through decoration-[1.5px]' : miss ? 'text-muted' : 'text-ink',
        )}
      >
        {formatRaw(value, c.dtype)}
        {split && kind === 'fk' && !miss && <span className="text-muted"> → {table.schema.fkeys[col]}[{String(value)}]</span>}
      </div>
      {split && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.25 }} className="mt-0.5 text-[14px] leading-tight text-ink-2">
          {kind === 'pk'
            ? 'primary key: no token, no edge'
            : kind === 'fk' && miss
              ? 'null foreign key: no link'
              : miss
              ? 'missing: no token'
              : kind === 'fk'
                ? 'parent link, no token'
                : `c = ${col} · t = ${table.schema.name}`}
        </motion.div>
      )}
    </motion.div>
  );
}
