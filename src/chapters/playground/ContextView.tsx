/**
 * Centre-bottom: what the model sees for the current seed. The task rows (with their editable label
 * cells) are listed in full; the rest of the context is summarised per table and can be expanded.
 * Everything is read from the sampled context (fly.rs mirror) and the ablation of ablate.ts.
 */
import { ChevronRight, Pencil } from 'lucide-react';
import { useMemo, useState } from 'react';
import { TableSwatch, cx, formatDate, formatRaw } from '@/components';
import type { RawValue, Token } from '@/engine/types';
import { SemType } from '@/engine/types';
import { useTableSlots } from '@/state/hooks';
import { entityName, labelText, nodeTitle, rowNoun, usePlayLive } from './context';
import { usePlay } from './state';

type Role = 'seed' | 'self' | 'other';

interface LabelRow {
  node: number;
  role: Role;
  split: string;
  ts: number | null;
  title: string;
  sub: string;
  /** Original position of the label (target-column) token, or −1 if it did not fit in L. */
  labelPos: number;
  raw: RawValue;
  /** Every original position of this row's tokens. */
  positions: number[];
}

function useLabelRows(): LabelRow[] {
  const { db, pdb, task, ablCtx, abl } = usePlayLive();
  return useMemo(() => {
    const seed = pdb.nodes[ablCtx.params.seedNode];
    const byNode = new Map<number, Token[]>();
    for (const t of ablCtx.tokens) {
      if (t.isPadding) continue;
      const isRow =
        task.kind === 'forecast'
          ? t.tableType !== 'Db' && t.tableNameIdx === seed.tableNameIdx
          : t.nodeIdx === seed.nodeIdx || abl.labelRole.has(t.pos);
      if (!isRow) continue;
      if (!byNode.has(t.nodeIdx)) byNode.set(t.nodeIdx, []);
      byNode.get(t.nodeIdx)!.push(t);
    }
    const seedEntity = seed.f2pNbrIdxs[0];
    const rows: LabelRow[] = [];
    for (const [node, toks] of byNode) {
      const n = pdb.nodes[node];
      const label = toks.find((t) => t.colNameIdx === ablCtx.params.targetColNameIdx);
      const role: Role =
        node === seed.nodeIdx ? 'seed' : task.kind === 'forecast' && n.f2pNbrIdxs[0] === seedEntity ? 'self' : 'other';
      let title: string;
      let sub: string;
      if (task.kind === 'autocomplete') {
        const row = db.tables.find((t) => t.schema.name === n.tableName)?.rows[n.rowPos] ?? {};
        title = `${rowNoun(n.tableName)} ${n.rowPos}`;
        sub = typeof row.product_id === 'number' ? entityName(db, 'products', row.product_id) : '';
      } else {
        const t = nodeTitle(db, pdb, node);
        title = t.title;
        sub = t.sub.split(' · ')[0];
      }
      rows.push({
        node,
        role,
        split: n.tableType,
        ts: n.ts,
        title,
        sub,
        labelPos: label ? label.pos : -1,
        raw: label ? label.raw : null,
        positions: toks.map((t) => t.pos),
      });
    }
    const order: Record<Role, number> = { seed: 0, self: 1, other: 2 };
    rows.sort((a, b) => order[a.role] - order[b.role] || (b.ts ?? 0) - (a.ts ?? 0) || a.node - b.node);
    return rows;
  }, [db, pdb, task, ablCtx, abl]);
}

/**
 * The range an edited number label is kept in: the observed range of the target column for an
 * autocomplete column (e.g. ratings 1-5), and ≥ the observed minimum for a forecast count. Display
 * guard only; the encoding of the value is pre.rs's z-score either way.
 */
function useLabelBounds(): { min: number; max: number | null } {
  const { db, task } = usePlayLive();
  return useMemo(() => {
    const rows = task.kind === 'autocomplete' ? (db.tables.find((t) => t.schema.name === task.table)?.rows ?? []) : (task.tables?.Train.rows ?? []);
    const vals = rows.map((r) => r[task.targetCol]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    if (vals.length === 0) return { min: 0, max: null };
    const lo = Math.min(...vals);
    return task.kind === 'autocomplete' ? { min: lo, max: Math.max(...vals) } : { min: Math.min(0, lo), max: null };
  }, [db, task]);
}

function LabelCell({ row }: { row: LabelRow }) {
  const { task, abl, scope } = usePlayLive();
  const bounds = useLabelBounds();
  const edits = usePlay((s) => (s.editScope === scope ? s.edits : null));
  const setEdit = usePlay((s) => s.setEdit);
  const [draft, setDraft] = useState<string | null>(null);

  if (row.role === 'seed') {
    return (
      <span
        className="target-outline num inline-flex h-8 min-w-[52px] items-center justify-center rounded-md bg-surface px-2 text-[15px] font-semibold text-accent-text"
        title="The masked target: its value is hidden from the model (fly.rs:492-496)"
      >
        ?
      </span>
    );
  }
  if (row.labelPos < 0) return <span className="text-[14px] text-muted">cut at L</span>;
  const removed = abl.removedPos.has(row.labelPos);
  const edited = abl.editedPos.has(row.labelPos);
  const value = edits && row.node in edits ? edits[row.node] : row.raw;
  const name = `${task.targetCol} of ${row.title}${row.ts !== null ? `, ${formatDate(row.ts * 1000)}` : ''}`;

  if (removed) {
    return (
      <span className="num inline-flex h-8 min-w-[52px] items-center justify-center rounded-md border border-dashed border-border-strong px-2 text-[15px] text-muted line-through" title="Removed from the context">
        {labelText(row.raw)}
      </span>
    );
  }
  if (task.targetType === 'binary') {
    return (
      <span className="inline-flex items-center gap-1.5">
        <button
          type="button"
          onClick={() => {
            const next = !(value === true);
            setEdit(scope, row.node, next === row.raw ? undefined : next);
          }}
          aria-label={`${name}: ${value === true ? 'true' : 'false'}${edited ? ' (edited)' : ''}. Click to flip.`}
          className={cx(
            'num inline-flex h-8 min-w-[52px] items-center justify-center gap-1 rounded-md border px-2 text-[15px] font-semibold transition-colors',
            edited ? 'border-accent bg-accent-soft text-accent-text' : 'border-border-strong bg-surface text-ink hover:bg-surface-2',
          )}
        >
          {labelText(value)}
          {edited && <Pencil aria-hidden="true" size={13} />}
        </button>
        {edited && <span className="num whitespace-nowrap text-[14px] text-muted">was {labelText(row.raw)}</span>}
      </span>
    );
  }
  const shown = draft ?? String(value);
  const commit = (s: string) => {
    setDraft(null);
    const v0 = Number(s);
    if (s.trim() === '' || !Number.isFinite(v0)) return;
    const v = Math.max(bounds.min, bounds.max === null ? v0 : Math.min(bounds.max, v0));
    setEdit(scope, row.node, v === row.raw ? undefined : v);
  };
  return (
    <span className="inline-flex items-center gap-1.5">
      <input
        type="number"
        inputMode="decimal"
        step={1}
        min={bounds.min}
        max={bounds.max ?? undefined}
        value={shown}
        title={bounds.max !== null ? `Range of this column: ${bounds.min} to ${bounds.max}` : `At least ${bounds.min}`}
        aria-label={`${name}${edited ? ' (edited)' : ''}`}
        onChange={(e) => {
          setDraft(e.target.value);
          if (e.target.value.trim() !== '' && Number.isFinite(Number(e.target.value))) commit(e.target.value);
        }}
        onBlur={() => setDraft(null)}
        className={cx(
          'num h-8 w-[76px] rounded-md border px-2 text-right text-[15px] font-semibold',
          edited ? 'border-accent bg-accent-soft text-accent-text' : 'border-border-strong bg-surface text-ink',
        )}
      />
      {edited && <span className="num whitespace-nowrap text-[14px] text-muted">was {labelText(row.raw)}</span>}
    </span>
  );
}

const ROLE_TEXT: Record<Role, string> = { seed: 'seed', self: 'self', other: 'other' };

function LabelTable() {
  const { task, abl } = usePlayLive();
  const rows = useLabelRows();
  const forecast = task.kind === 'forecast';
  return (
    <div className="scroll-quiet max-h-[360px] overflow-auto rounded-lg border border-border">
      <table className="w-full border-collapse text-[15px]">
        <caption className="sr-only">
          {forecast ? 'Task rows in the context: the seed row and the in-context labels' : 'The seed row and the other rows’ target cells in the context'}
        </caption>
        <thead className="sticky top-0 z-[1] bg-surface-2">
          <tr className="text-left text-ink-2">
            <th scope="col" className="px-2.5 py-1.5 font-semibold">
              {forecast ? `${task.name} rows in the context` : `${task.table ?? 'Rows'} in the context`}
            </th>
            <th scope="col" className="num px-2.5 py-1.5 text-right font-semibold">
              {task.targetCol}
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const rowRemoved = r.role !== 'seed' && r.positions.every((p) => abl.removedPos.has(p));
            const who = forecast ? (r.role === 'self' ? '' : r.title) : `${r.title}${r.sub ? ` · ${r.sub}` : ''}`;
            return (
              <tr key={r.node} className={cx('border-t border-grid/70', r.role === 'seed' && 'bg-accent-soft/60', rowRemoved && 'opacity-55')}>
                <td className="min-w-0 px-2.5 py-1.5 align-middle">
                  <span className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                    <span
                      className={cx(
                        'inline-flex h-6 shrink-0 items-center rounded-md px-1.5 text-[13.5px] font-semibold',
                        r.role === 'seed' ? 'bg-accent text-accent-ink' : r.role === 'self' ? 'bg-surface-3 text-ink' : 'border border-border-strong text-ink-2',
                      )}
                    >
                      {ROLE_TEXT[r.role]}
                    </span>
                    {forecast && (
                      <span className={cx('num whitespace-nowrap text-ink', rowRemoved && 'line-through')}>
                        {r.ts !== null ? formatDate(r.ts * 1000) : '—'}
                        <span className="ml-1.5 text-[14px] text-muted">{r.split}</span>
                      </span>
                    )}
                    {who && <span className={cx('min-w-0 text-[14.5px] leading-snug', forecast ? 'text-ink-2' : 'text-ink', rowRemoved && 'line-through')}>{who}</span>}
                  </span>
                </td>
                <td className="px-2.5 py-1 text-right align-middle">
                  <LabelCell row={r} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** One DB table of the context, summarised; expands to its rows' cells. */
function TableGroup({ table, nodes, tokens }: { table: string; nodes: number[]; tokens: Token[] }) {
  const { pdb } = usePlayLive();
  const slots = useTableSlots();
  const [open, setOpen] = useState(false);
  const byNode = new Map<number, Token[]>();
  for (const t of tokens) {
    if (!byNode.has(t.nodeIdx)) byNode.set(t.nodeIdx, []);
    byNode.get(t.nodeIdx)!.push(t);
  }
  return (
    <div className="rounded-lg border border-border">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-2.5 py-2 text-left text-[15px] hover:bg-surface-2"
      >
        <ChevronRight aria-hidden="true" size={15} className={cx('shrink-0 text-muted transition-transform', open && 'rotate-90')} />
        <TableSwatch slot={slots[table] ?? 0} label={table} />
        <span className="num ml-auto text-[14.5px] text-ink-2">
          {nodes.length} row{nodes.length === 1 ? '' : 's'} · {tokens.length} cells
        </span>
      </button>
      {open && (
        <ul className="scroll-quiet !m-0 max-h-[240px] !list-none overflow-y-auto border-t border-grid !p-0">
          {nodes.map((n) => {
            const g = pdb.nodes[n];
            const cells = byNode.get(n) ?? [];
            const named = cells.find((c) => c.colName === 'name');
            return (
              <li key={n} className="!m-0 border-b border-grid/60 px-2.5 py-1.5 !pl-2.5 text-[14.5px] leading-snug last:border-0">
                <span className="font-medium text-ink">{named ? String(named.raw) : `${rowNoun(table)} ${g.rowPos}`}</span>
                <span className="text-muted"> · </span>
                <span className="text-ink-2">
                  {cells
                    .filter((c) => c !== named)
                    .map((c) => `${c.colName} ${c.sem === SemType.DateTime && typeof c.raw === 'number' ? formatDate(c.raw) : formatRaw(c.raw)}`)
                    .join(' · ')}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

export function ContextView() {
  const { ablCtx, abl, task, effL, pdb } = usePlayLive();
  const groups = useMemo(() => {
    const seed = pdb.nodes[ablCtx.params.seedNode];
    const m = new Map<string, { nodes: number[]; tokens: Token[] }>();
    for (const t of ablCtx.tokens) {
      if (t.isPadding) continue;
      // forecast: the task rows are listed above; autocomplete: the target-column cells are
      const listed =
        task.kind === 'forecast'
          ? t.tableType !== 'Db' && t.tableNameIdx === seed.tableNameIdx
          : t.tableNameIdx === seed.tableNameIdx && t.colNameIdx === ablCtx.params.targetColNameIdx;
      if (listed) continue;
      const g = m.get(t.tableName) ?? { nodes: [], tokens: [] };
      if (!g.nodes.includes(t.nodeIdx)) g.nodes.push(t.nodeIdx);
      g.tokens.push(t);
      m.set(t.tableName, g);
    }
    return [...m.entries()];
  }, [ablCtx, pdb, task]);
  const nReal = abl.tokens.filter((t) => !t.isPadding).length;
  const removed = abl.removedPos.size;

  return (
    <div className="min-w-0">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <div className="text-[16.5px] font-semibold text-ink">What the model sees</div>
        <div className="num text-[14.5px] text-ink-2">
          {nReal} of L = {effL} cells{removed > 0 ? ` · ${removed} dropped` : ''}
        </div>
      </div>
      <LabelTable />
      <div className="mt-3 grid gap-1.5">
        <div className="text-[14.5px] text-ink-2">The rest of the context</div>
        {groups.map(([table, g]) => (
          <TableGroup key={table} table={table} nodes={g.nodes} tokens={g.tokens} />
        ))}
      </div>
    </div>
  );
}
