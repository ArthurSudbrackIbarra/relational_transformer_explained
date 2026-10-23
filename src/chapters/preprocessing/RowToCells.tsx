/**
 * Section 2 figure: what pre.rs makes of one row, column by column
 * (pre.rs:439-611; spec §1.2, §3.2, §3.5). A StepPlayer walks the columns in
 * DataFrame order; each value flies into the lane of its fate: a cell record,
 * a link (FK), or nothing (PK, missing value). The node record fills up as it
 * goes (node_idx, timestamp, f2p_nbr_idxs, cells).
 */
import { Check, CircleSlash, KeyRound, Link2 } from 'lucide-react';
import { LayoutGroup, motion } from 'motion/react';
import { useMemo, useState, type ReactNode } from 'react';
import { NumberField, Select, SemBadge, StepPlayer, cx, formatRaw, useStepPlayer } from '@/components';
import { SemType } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { fileLabel, isoSeconds, rowFates, tsReason, type Fate, type TableFile } from './data';
import { ColGlyph, PickChip, signed, type ChapterData } from './shared';

const SEM_LABEL = ['number', 'text', 'datetime', 'boolean'] as const;

export interface RowPick {
  label: string;
  key: string;
  row: number;
}

/** The value a cell stores, in words: "number +0.164", "text #85". */
export function storedText(f: Fate & { kind: 'cell' }): string {
  const c = f.cell;
  switch (c.sem) {
    case SemType.Number:
      return `number ${signed(c.number)}`;
    case SemType.DateTime:
      return `datetime ${signed(c.datetime)}`;
    case SemType.Boolean:
      return `boolean ${signed(c.boolean)}`;
    default:
      return `text #${c.textIdx}`;
  }
}

function rawText(f: Fate): string {
  if (f.kind === 'missing' || f.kind === 'null-fk') return '∅';
  const s = formatRaw(f.value, f.col.dtype);
  return f.col.dtype === 'str' ? `“${s.length > 26 ? s.slice(0, 25) + '…' : s}”` : s;
}

function laneOf(f: Fate): 'cell' | 'link' | 'none' {
  return f.kind === 'cell' ? 'cell' : f.kind === 'link' ? 'link' : 'none';
}

/** Plain-language rule for one column (StepPlayer description). */
function ruleText(f: Fate, file: TableFile, data: ChapterData, node: { ts: number | null }, firstLink: boolean): ReactNode {
  const name = <code>{f.col.name}</code>;
  switch (f.kind) {
    case 'pk':
      return (
        <>
          {name} is the primary key. pre.rs skips it: no cell, no edge. Its value ({String(f.value)}) is the row position, which the node index already encodes.
        </>
      );
    case 'link':
      return (
        <>
          {name} is a foreign key to <code>{f.parentTable}</code>. No cell: the parent’s node index #{f.parentNode} is pushed onto <code>f2p_nbr_idxs</code>, and{' '}
          <code>{f.parentTable}</code>[{f.value}] gets a P→F edge back to this row.
          {firstLink && file.table.schema.timeCol && node.ts !== null && (
            <>
              {' '}
              The row’s timestamp is set here too, from <code>{file.table.schema.timeCol}</code>; every later non-null foreign key sets it again, to the same value.
            </>
          )}
        </>
      );
    case 'null-fk':
      return (
        <>
          {name} is a foreign key, but its value is null: pre.rs skips it, so there is no link.
        </>
      );
    case 'missing':
      return (
        <>
          {name} is missing in this row. A missing value produces no cell at all: it is neither imputed nor masked.
        </>
      );
    case 'cell': {
      const c = f.cell;
      const phrase = data.pdb.text[c.colNameIdx];
      return (
        <>
          {name} is a feature column with a value, so it becomes one cell: sem type <strong>{SEM_LABEL[c.sem]}</strong>, stored as {storedText(f)}, with the column phrase “{phrase}”
          (string index {c.colNameIdx}).
        </>
      );
    }
  }
}

function ResultChip({ f, data }: { f: Fate; data: ChapterData }) {
  if (f.kind === 'cell') {
    return (
      <>
        <div className="flex min-w-0 items-center gap-1.5">
          <SemBadge sem={f.cell.sem} size="sm" />
          <span className="truncate text-[14.5px] font-semibold text-ink">{f.col.name}</span>
        </div>
        <div className="num mt-0.5 truncate text-[14.5px] text-ink-2">{storedText(f)}</div>
      </>
    );
  }
  if (f.kind === 'link') {
    const pf = data.fileOfNode(f.parentNode);
    return (
      <>
        <div className="flex min-w-0 items-center gap-1.5">
          <Link2 aria-hidden="true" size={13} className="shrink-0 text-ink-2" />
          <span className="truncate text-[14.5px] font-semibold text-ink">{f.col.name}</span>
        </div>
        <div className="num mt-0.5 flex items-center gap-1 truncate text-[14.5px] text-ink-2">
          → #{f.parentNode}
          {pf && (
            <span className="inline-flex items-center gap-1">
              <span aria-hidden="true" className="inline-block h-2 w-2 rounded-[2px]" style={{ background: catVar(data.slotOf(pf.tableName)) }} />
              {pf.tableName}[{f.value}]
            </span>
          )}
        </div>
      </>
    );
  }
  const why = f.kind === 'pk' ? 'primary key' : f.kind === 'null-fk' ? 'null foreign key' : 'missing value';
  return (
    <>
      <div className="flex min-w-0 items-center gap-1.5">
        {f.kind === 'pk' ? (
          <KeyRound aria-hidden="true" size={13} className="shrink-0 text-muted" />
        ) : (
          <CircleSlash aria-hidden="true" size={13} className="shrink-0 text-muted" />
        )}
        <span className="truncate text-[14.5px] font-semibold text-ink-2 line-through decoration-[var(--axis)]">{f.col.name}</span>
      </div>
      <div className="mt-0.5 truncate text-[14.5px] text-muted">{why}</div>
    </>
  );
}

function RowFlow({ data, file, row }: { data: ChapterData; file: TableFile; row: number }) {
  const fates = useMemo(() => rowFates(data.pdb, file, row), [data.pdb, file, row]);
  const node = data.pdb.nodes[file.offset + row];
  const n = fates.length;
  const player = useStepPlayer(n + 1, { initial: n, intervalMs: 1000 });
  const done = player.index; // columns [0, done) processed
  const idPrefix = `rc-${file.key}-${row}-`;

  const firstLinkIdx = fates.findIndex((f) => f.kind === 'link');
  const tsSet = node && node.ts !== null && firstLinkIdx >= 0 && done > firstLinkIdx;
  const links = fates.slice(0, done).filter((f): f is Fate & { kind: 'link' } => f.kind === 'link');
  const cells = fates.slice(0, done).filter((f) => f.kind === 'cell');

  const describe = (i: number) => {
    if (i === 0)
      return (
        <>
          Row {row} of <code>{file.path}</code>, before pre.rs reads it. Its node index is {file.offset} + {row} = <strong>{file.offset + row}</strong>. Press play to process its {n} columns in
          the order the table stores them. Next: <code>{fates[0]?.col.name}</code>.
        </>
      );
    const f = fates[i - 1];
    return (
      <>
        {ruleText(f, file, data, node ?? { ts: null }, i - 1 === firstLinkIdx)}
        {i < n && (
          <span className="text-muted">
            {' '}
            Next: <code>{fates[i].col.name}</code>.
          </span>
        )}
      </>
    );
  };

  const lane = (kind: 'cell' | 'link' | 'none', title: string, hint: string) => {
    const items = fates.map((f, i) => ({ f, i })).filter(({ f, i }) => i < done && laneOf(f) === kind);
    return (
      <div className={cx('min-w-0 rounded-xl border px-3 py-2.5', kind === 'cell' ? 'border-accent/40 bg-accent-soft' : 'border-border bg-surface-2/50')}>
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-[15px] font-semibold text-ink">{title}</span>
          <span className="num text-[14.5px] text-ink-2">{items.length}</span>
        </div>
        <div className="mb-2 text-[14px] leading-snug text-muted">{hint}</div>
        <div className="flex min-h-[46px] flex-col gap-1.5">
          {items.map(({ f, i }) => (
            <motion.div
              key={i}
              layoutId={idPrefix + i}
              transition={{ type: 'spring', stiffness: 380, damping: 34 }}
              className={cx('min-w-0 rounded-lg border bg-surface px-2.5 py-1.5', i === done - 1 ? 'border-accent' : 'border-border')}
            >
              <ResultChip f={f} data={data} />
            </motion.div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <LayoutGroup id={idPrefix}>
      <StepPlayer
        {...player}
        unit="column"
        formatCounter={(i, count) => (i === 0 ? `before · ${count - 1} columns` : `column ${i} / ${count - 1}`)}
        label="Step through pre.rs on this row"
        describe={describe}
      />

      {/* the source row */}
      <div className="mt-4 rounded-xl border border-border bg-surface px-3 py-3" style={{ borderLeft: `3px solid ${catVar(data.slotOf(file.tableName))}` }}>
        <div className="mb-2 flex flex-wrap items-baseline gap-x-2 text-[15px]">
          <span className="font-semibold text-ink">
            {fileLabel(file)}[{row}]
          </span>
          <span className="text-ink-2">raw row · node #{file.offset + row}</span>
        </div>
        <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-3 lg:grid-cols-4">
          {fates.map((f, i) => {
            const processed = i < done;
            const justDone = i === done - 1;
            // phone: one line per column (name … value); from sm up: name above value
            const box = (
              <>
                <div className="flex min-w-0 items-center gap-1.5 text-[14px] text-ink-2">
                  <ColGlyph file={file} col={f.col.name} size="sm" />
                  <span className="truncate">{f.col.name}</span>
                  {processed && <Check aria-label="processed" size={14} strokeWidth={2.5} className={cx('ml-auto hidden shrink-0 sm:block', justDone ? 'text-accent-text' : 'text-muted')} />}
                  {i === done && <span className="ml-auto shrink-0 text-[13px] font-medium text-muted">next</span>}
                </div>
                <div
                  className={cx(
                    'num ml-auto flex min-w-0 items-center justify-end gap-2 text-[15.5px] sm:mt-0.5 sm:ml-0 sm:justify-start',
                    f.kind === 'missing' || f.kind === 'null-fk' ? 'text-muted' : processed ? 'text-ink-2' : 'text-ink',
                  )}
                >
                  <span className="truncate">{rawText(f)}</span>
                  {processed && <Check aria-hidden="true" size={14} strokeWidth={2.5} className="shrink-0 text-muted sm:hidden" />}
                </div>
              </>
            );
            return processed ? (
              <div
                key={i}
                className={cx(
                  'flex min-w-0 flex-wrap items-center gap-x-2 rounded-lg border border-dashed px-2.5 py-1.5 sm:block',
                  justDone ? 'border-accent bg-accent-soft' : 'border-border-strong',
                )}
              >
                {box}
              </div>
            ) : (
              <motion.div
                key={i}
                layoutId={idPrefix + i}
                transition={{ type: 'spring', stiffness: 380, damping: 34 }}
                className="flex min-w-0 flex-wrap items-center gap-x-2 rounded-lg border border-border bg-surface-2 px-2.5 py-1.5 sm:block"
              >
                {box}
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* lanes */}
      <div className="mt-3 grid gap-3 md:grid-cols-3">
        {lane('cell', 'Cells', 'One record per non-missing feature value. These can become tokens later.')}
        {lane('link', 'Links', 'Foreign keys: the parent’s node index, plus edges. No token.')}
        {lane('none', 'Nothing', 'The primary key, and missing values.')}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[14px] text-ink-2" aria-label="Glyph legend">
        {[SemType.Number, SemType.Text, SemType.DateTime, SemType.Boolean].map((t) => (
          <span key={t} className="inline-flex items-center gap-1.5">
            <SemBadge sem={t} size="sm" />
            {SEM_LABEL[t]}
          </span>
        ))}
        <span className="inline-flex items-center gap-1.5">
          <KeyRound aria-hidden="true" size={15} className="text-ink-2" />
          primary key
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Link2 aria-hidden="true" size={15} className="text-ink-2" />
          foreign key
        </span>
        <span className="text-muted">Signed numbers are z-scores (Pass 2).</span>
      </div>

      {/* the node record */}
      <div className="mt-3 rounded-xl border border-border bg-surface-2/50 px-3 py-2.5 text-[15px]">
        <div className="mb-1.5 text-[15px] font-semibold text-ink">
          Node record <span className="font-normal text-ink-2">(common.rs <code>Node</code>)</span>
        </div>
        <dl className="grid gap-x-5 gap-y-1 sm:grid-cols-2">
          <Field k="node_idx" v={String(file.offset + row)} />
          <Field k="table_name_idx" v={`${node?.tableNameIdx ?? '—'}  (“${file.tableName}”)`} hint="its index in the string table; the numbering is arbitrary" />
          <Field k="is_task_node" v={String(file.isTask)} hint="set for task rows; not read by the model" />
          <Field
            k="timestamp"
            v={tsSet && node?.ts != null ? `${node.ts} s` : 'None'}
            hint={
              tsSet && node?.ts != null
                ? `${isoSeconds(node.ts)}; read only by the sampler`
                : node?.ts != null
                  ? `set from ${file.table.schema.timeCol} when pre.rs reads a non-null foreign key`
                  : tsReason(file, row, null)
            }
          />
          <Field k="f2p_nbr_idxs" v={`[${links.map((l) => l.parentNode).join(', ')}]`} />
          <Field k="cells" v={`${cells.length} (${cells.map((c) => SEM_LABEL[(c as Fate & { kind: 'cell' }).cell.sem]).join(', ') || 'none yet'})`} />
        </dl>
      </div>
    </LayoutGroup>
  );
}

function Field({ k, v, hint }: { k: string; v: string; hint?: string }) {
  return (
    <div className="flex min-w-0 flex-wrap items-baseline gap-x-2">
      <dt className="font-mono text-ink-2">{k}</dt>
      <dd className="num m-0 text-ink">{v}</dd>
      {hint && <dd className="m-0 w-full text-[14px] leading-snug text-muted sm:w-auto">{hint}</dd>}
    </div>
  );
}

export function RowToCellsFigure({ data, picks }: { data: ChapterData; picks: RowPick[] }) {
  const { files } = data;
  const first = picks[0] ?? { key: files[0].key, row: 0 };
  const [key, setKey] = useState(first.key);
  const [row, setRow] = useState(first.row);
  const file = files.find((f) => f.key === key) ?? files[0];
  const r = Math.max(0, Math.min(row, file.numNodes - 1));

  return (
    <div className="min-w-0">
      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[14.5px] font-medium text-ink-2">Try</span>
        {picks.map((p) => (
          <PickChip
            key={p.label}
            active={p.key === key && p.row === r}
            onClick={() => {
              setKey(p.key);
              setRow(p.row);
            }}
          >
            {p.label}
          </PickChip>
        ))}
      </div>
      <div className="mb-4 grid grid-cols-[minmax(0,1fr)_110px] gap-3 sm:grid-cols-[240px_130px]">
        <Select label="Table file" value={file.key} onChange={(e) => setKey(e.target.value)}>
          {files.map((f) => (
            <option key={f.key} value={f.key}>
              {f.key}
            </option>
          ))}
        </Select>
        <NumberField label={`Row (0–${file.numNodes - 1})`} value={r} min={0} max={file.numNodes - 1} onChange={(v) => setRow(Math.max(0, Math.min(file.numNodes - 1, Math.round(v))))} />
      </div>
      <RowFlow key={`${file.key}:${r}`} data={data} file={file} row={r} />
    </div>
  );
}
