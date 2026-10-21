/**
 * Right pane of the centrepiece (after paper Fig. 6b): the context as per-table mini tables that fill
 * up in visit order. Uses the shared TableView: key columns keep their key/link icons (they produce no
 * token), the masked target shows "?", the row being processed is highlighted.
 */
import { memo, useEffect, useMemo, useRef } from 'react';
import { TableView, cx } from '@/components';
import type { PreprocessedDB, RawDatabase, SampledContext, TaskDef } from '@/engine/types';
import { plural, rawTableOf, rowName, type ReplayState } from './trace';

export interface ContextTablesProps {
  db: RawDatabase;
  pdb: PreprocessedDB;
  task: TaskDef;
  slots: Record<string, number>;
  state: ReplayState;
  /** The sampled context and the number of trace events replayed (for cells that never became tokens). */
  ctx: SampledContext;
  cursor: number;
  seed: number;
  current: number | null;
  /** Token positions emitted during the current step (cells to ring). */
  freshTokens: ReadonlySet<number>;
  hover: number | null;
  onHover: (n: number | null) => void;
  /** Row cut by the hard stop at L. */
  cut: { node: number; emitted: number; total: number } | null;
  tokenCol: (pos: number) => string | undefined;
  maxHeight?: number;
  className?: string;
}

const DROP_TITLE = 'leakage column of this task: dropped for the seed row and rows with the seed’s timestamp, so no token (fly.rs:457-461)';
const CUT_TITLE = 'not emitted: the context reached L in the middle of this row (fly.rs:502-509)';

export function ContextTables({ db, pdb, task, slots, state, ctx, cursor, seed, current, freshTokens, hover, onHover, cut, tokenCol, maxHeight = 380, className }: ContextTablesProps) {
  // Cells of visited rows that never became tokens, keyed by table: leakage drops (once replayed) and the
  // cells of the row cut at L (once the hard stop fired). Encoded as strings so the memoised tables only
  // re-render when their own set changes.
  const mutedByKey = useMemo(() => {
    const m = new Map<string, string[]>();
    const add = (node: number, col: string, title: string) => {
      const g = pdb.nodes[node];
      const arr = m.get(g.tableKey) ?? [];
      arr.push(`${g.rowPos}:${col}\t${title}`);
      m.set(g.tableKey, arr);
    };
    const dropped = new Map<number, Set<string>>();
    const n = Math.min(cursor, ctx.events.length);
    for (let i = 0; i < n; i++) {
      const e = ctx.events[i];
      if (e.kind !== 'drop-leakage-cell') continue;
      add(e.node, e.colName, DROP_TITLE);
      const s = dropped.get(e.node) ?? new Set<string>();
      s.add(e.colName);
      dropped.set(e.node, s);
    }
    if (cut && state.finished?.kind === 'full') {
      const emitted = new Set((state.emittedByNode.get(cut.node) ?? []).map((p) => tokenCol(p)));
      for (const c of pdb.nodes[cut.node].cells) {
        if (emitted.has(c.colName) || dropped.get(cut.node)?.has(c.colName)) continue;
        add(cut.node, c.colName, CUT_TITLE);
      }
    }
    const out = new Map<string, string>();
    for (const [k, v] of m) out.set(k, v.join('\n'));
    return out;
  }, [ctx, cursor, cut, state, pdb, tokenCol]);

  // group visited rows by table_info key, in order of first appearance
  const groups = useMemo(() => {
    const m = new Map<string, number[]>();
    for (const n of state.visitedOrder) {
      const key = pdb.nodes[n].tableKey;
      const arr = m.get(key);
      if (arr) arr.push(n);
      else m.set(key, [n]);
    }
    return [...m.entries()];
  }, [state.visitedOrder, pdb]);

  const pane = useRef<HTMLDivElement | null>(null);
  const curKey = current !== null ? pdb.nodes[current]?.tableKey : null;
  // keep the table of the current row in view (only this pane scrolls, never the page)
  useEffect(() => {
    const el = pane.current;
    if (!el || !curKey) return;
    const child = el.querySelector<HTMLElement>(`[data-key="${CSS.escape(curKey)}"]`);
    if (!child) return;
    const top = child.offsetTop; // the pane is `relative`, so this is relative to it
    if (top < el.scrollTop || top + 80 > el.scrollTop + el.clientHeight) el.scrollTo({ top: Math.max(0, top - 8), behavior: 'auto' });
  }, [curKey, groups.length]);

  const seedNode = pdb.nodes[seed];
  const targetCol = task.targetCol;

  if (groups.length === 0)
    return (
      <div className={cx('flex min-h-[120px] items-center justify-center rounded-xl border border-dashed border-border-strong px-4 text-center text-[15.5px] text-ink-2', className)}>
        No row visited yet. The first pop takes the seed row off the stack.
      </div>
    );

  return (
    <div ref={pane} className={cx('scroll-quiet relative grid min-w-0 content-start gap-3 overflow-y-auto pr-1', className)} style={{ maxHeight }}>
      {groups.map(([key, nodes]) => {
        const inGroup = (n: number | null) => (n !== null && pdb.nodes[n]?.tableKey === key ? n : null);
        const cur = inGroup(current);
        const cells: string[] = [];
        if (cur !== null) {
          const row = pdb.nodes[cur].rowPos;
          for (const pos of state.emittedByNode.get(cur) ?? []) {
            if (!freshTokens.has(pos)) continue;
            const c = tokenCol(pos);
            if (c) cells.push(`${row}:${c}`);
          }
        }
        const tokens = nodes.reduce((s, n) => s + (state.emittedByNode.get(n)?.length ?? 0), 0);
        const cutHere = cut && pdb.nodes[cut.node].tableKey === key && state.finished?.kind === 'full' ? cut : null;
        return (
          <TableGroup
            key={key}
            tableKey={key}
            nodesKey={nodes.join(',')}
            db={db}
            pdb={pdb}
            slots={slots}
            curNode={cur}
            hovNode={inGroup(hover)}
            cellsKey={cells.join('|')}
            tokens={tokens}
            cutText={cutHere ? `${rowName(db, pdb, cutHere.node).short} cut after ${cutHere.emitted} of ${cutHere.total} cells` : ''}
            mutedKey={mutedByKey.get(key) ?? ''}
            targetRow={seedNode.tableKey === key ? seedNode.rowPos : null}
            targetCol={targetCol}
            labelFirst={pdb.nodes[nodes[0]].tableName === seedNode.tableName}
            onHover={onHover}
          />
        );
      })}
    </div>
  );
}

const TableGroup = memo(function TableGroup({
  tableKey,
  nodesKey,
  db,
  pdb,
  slots,
  curNode,
  hovNode,
  cellsKey,
  tokens,
  cutText,
  mutedKey,
  targetRow,
  targetCol,
  labelFirst,
  onHover,
}: {
  tableKey: string;
  nodesKey: string;
  db: RawDatabase;
  pdb: PreprocessedDB;
  slots: Record<string, number>;
  curNode: number | null;
  hovNode: number | null;
  cellsKey: string;
  tokens: number;
  cutText: string;
  mutedKey: string;
  targetRow: number | null;
  targetCol: string;
  /** Rows of the seed's own table: show the target / label column first, so it never scrolls out of a narrow pane. */
  labelFirst: boolean;
  onHover: (n: number | null) => void;
}) {
  const nodes = useMemo(() => nodesKey.split(',').map(Number), [nodesKey]);
  const rows = useMemo(() => nodes.map((n) => pdb.nodes[n].rowPos), [nodes, pdb]);
  const first = pdb.nodes[nodes[0]];
  const table = rawTableOf(db, pdb, nodes[0]);
  const curRow = curNode !== null ? pdb.nodes[curNode].rowPos : null;
  const hovRow = hovNode !== null ? pdb.nodes[hovNode].rowPos : null;
  const highlightRows = useMemo(() => [curRow, hovRow].filter((x): x is number => x !== null), [curRow, hovRow]);
  const cells = useMemo(() => (cellsKey ? cellsKey.split('|') : []), [cellsKey]);
  const target = useMemo(() => (targetRow !== null ? { row: targetRow, col: targetCol } : null), [targetRow, targetCol]);
  const muted = useMemo(() => {
    if (!mutedKey) return undefined;
    const m = new Map<string, string>();
    for (const line of mutedKey.split('\n')) {
      const [k, t] = line.split('\t');
      m.set(k, t);
    }
    return m;
  }, [mutedKey]);
  const nMuted = muted?.size ?? 0;
  const columns = useMemo(() => {
    if (!labelFirst || !table) return undefined;
    const names = table.schema.columns.map((c) => c.name);
    return names.includes(targetCol) ? [targetCol, ...names.filter((c) => c !== targetCol)] : undefined;
  }, [labelFirst, table, targetCol]);
  if (!table) return null;
  return (
    <div data-key={tableKey} className="min-w-0">
      <TableView
        schema={table.schema}
        rows={table.rows}
        slot={slots[first.tableName] ?? 0}
        title={first.tableType === 'Db' ? first.tableName : `${first.tableName} (${first.tableType})`}
        columns={columns}
        rowFilter={rows}
        highlightRows={highlightRows}
        highlightCells={cells}
        targetCell={target}
        mutedCells={muted}
        scrollToRow={curRow}
        maxHeight={Math.min(176, 36 + 32 * Math.max(1, rows.length))}
        onRowHover={(r) => {
          if (r === null) onHover(null);
          else {
            const n = nodes.find((x) => pdb.nodes[x].rowPos === r);
            onHover(n ?? null);
          }
        }}
        footer={
          <>
            {plural(tokens, 'token')}
            {cutText ? ` · ${cutText}` : ''}
            {nMuted > 0 ? ` · struck-through cells produced no token` : ''}
          </>
        }
      />
    </div>
  );
});
