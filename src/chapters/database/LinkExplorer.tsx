/**
 * Link explorer: the focus row with its parents (F→P, one per non-null FK) above
 * and its children (P→F, any number) below, plus a ±1-hop EntityGraph. Click any
 * linked row to walk the database (spec §1.2-1.3, paper p.3 §2.1).
 */
import { AnimatePresence, motion } from 'motion/react';
import { ArrowDown, ArrowUp, Crosshair, KeyRound, Link2, Undo2 } from 'lucide-react';
import { Fragment, memo, useMemo, useState } from 'react';
import {
  Button,
  EntityGraph,
  SemBadge,
  TableSwatch,
  cx,
  formatRaw,
  semOfDType,
  type GraphEdgeSpec,
  type GraphNodeSpec,
  type NodeId,
  type RingStyle,
} from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import type { RawDatabase } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { LinkSummary } from './TableBrowser';
import { childrenForDisplay, columnKind, isMissing, parseRowKey, rowCaption, rowKey, rowLabel, sameRow, type LinkIndex, type RowRef } from './model';

const CHIP_CAP = 36;
/** Long text values in the focus card are shortened (full value in the tooltip). */
const LONG_TEXT = 90;
const GRAPH_CAP = 40;

export interface LinkExplorerProps {
  db: RawDatabase;
  idx: LinkIndex;
  slots: Record<string, number>;
  focus: RowRef;
  /** The row the ±1-hop graph is drawn around: a deferred copy of `focus`, so the heavy layout never blocks the click. */
  graphFocus?: RowRef;
  onFocus: (r: RowRef) => void;
  history: RowRef[];
  onBack: () => void;
  example: RowRef;
  /** Optional action that makes the focus row the simulation's running example. */
  adopt?: { label: string; title: string; onClick: () => void } | null;
  /** Task tables whose rows are also children of rows of `focus.table` (they join in chapter 3). */
  taskChildTables?: string[];
}

export const LinkExplorer = memo(function LinkExplorer({
  db,
  idx,
  slots,
  focus,
  graphFocus = focus,
  onFocus,
  history,
  onBack,
  example,
  adopt,
  taskChildTables = [],
}: LinkExplorerProps) {
  const table = idx.table(focus.table) ?? db.tables[0];
  const rec = table.rows[focus.row] ?? {};
  const parents = useMemo(() => idx.parentsOf(focus), [idx, focus]);
  const groups = useMemo(() => idx.childrenOf(focus), [idx, focus]);
  const [hover, setHover] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  // ±1 hop graph: graphFocus + parents + (capped) children
  const graph = useMemo(() => {
    const nodes: GraphNodeSpec[] = [];
    const edges: GraphEdgeSpec[] = [];
    const seen = new Set<string>();
    const add = (r: RowRef) => {
      const k = rowKey(r);
      if (seen.has(k)) return;
      seen.add(k);
      nodes.push({ id: k, label: rowLabel(r), table: r.table, slot: slots[r.table] ?? 0, title: `${rowLabel(r)} ${rowCaption(db, r)}` });
    };
    add(graphFocus);
    for (const p of idx.parentsOf(graphFocus)) {
      if (p.parentRow === null) continue;
      const pr = { table: p.parentTable, row: p.parentRow };
      add(pr);
      edges.push({ source: rowKey(graphFocus), target: rowKey(pr) });
    }
    const gGroups = idx.childrenOf(graphFocus);
    const total = gGroups.reduce((s, g) => s + g.rows.length, 0);
    const kids = childrenForDisplay(gGroups, GRAPH_CAP);
    for (const cr of kids) {
      add(cr);
      edges.push({ source: rowKey(cr), target: rowKey(graphFocus) });
    }
    const shown = kids.length;
    return { nodes, edges, shown, total };
  }, [db, idx, graphFocus, slots]);
  const nChildren = useMemo(() => groups.reduce((s, g) => s + g.rows.length, 0), [groups]);

  const nParents = parents.filter((p) => p.parentRow !== null).length;
  const focusRing = useMemo(() => new Map<NodeId, RingStyle>([[rowKey(graphFocus), 'accent']]), [graphFocus]);
  const graphHighlight = useMemo(() => [rowKey(graphFocus)], [graphFocus]);
  const noHighlight = useMemo(() => [] as string[], []);
  const drawnIds = useMemo(() => new Set(graph.nodes.map((n) => String(n.id))), [graph]);
  // a hovered chip whose row is not drawn (display limit) must not gray out the whole graph
  const hoverDrawn = hover !== null && drawnIds.has(hover);
  // small neighborhoods get a small box; on a phone the box is never much taller than it is wide
  const [graphBox, graphSize] = useElementSize<HTMLDivElement>();
  const graphH = Math.round(Math.min(380, graphSize.width ? graphSize.width * 0.9 : 380, Math.max(220, 140 + 40 * Math.sqrt(graph.nodes.length))));

  return (
    <div className="grid min-w-0 gap-5 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
      <div className="min-w-0">
        {/* trail */}
        <div className="mb-3 flex h-9 min-w-0 flex-nowrap items-center gap-x-2 text-[15px] text-ink-2">
          <Button size="sm" variant="secondary" icon={<Undo2 size={14} />} disabled={history.length === 0} onClick={onBack}>
            Back
          </Button>
          {!sameRow(focus, example) && (
            <Button size="sm" variant="ghost" icon={<Crosshair size={14} />} onClick={() => onFocus(example)} title={`Back to the running example, ${rowLabel(example)}`}>
              {rowLabel(example)}
            </Button>
          )}
          {history.length > 0 && (
            <span className="min-w-0 flex-1 truncate text-[14px] text-muted" title="The rows you walked through">
              {history
                .slice(-4)
                .map((r) => rowLabel(r))
                .join(' → ')}{' '}
              → <span className="text-ink">{rowLabel(focus)}</span>
            </span>
          )}
        </div>

        {/* parents */}
        <div className="rounded-xl border border-dashed border-border-strong px-3 py-2.5">
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
            <span className="text-[15px] font-semibold text-ink">
              Parents <span className="font-normal text-ink-2">(F→P: this row’s foreign keys point to them)</span>
            </span>
            {parents.length > 0 && (
              <span className="num text-[14px] text-muted">
                {nParents} of at most {parents.length}
              </span>
            )}
          </div>
          {parents.length === 0 ? (
            <p className="!m-0 text-[15.5px] text-ink-2">
              <span>{focus.table}</span> has no foreign-key column, so its rows have no parents.
            </p>
          ) : (
            <div className="grid gap-2 sm:grid-cols-2">
              {parents.map((p) =>
                p.parentRow === null ? (
                  <div key={p.fkCol} className="rounded-lg bg-surface-2 px-3 py-2 text-[15px] text-ink-2">
                    <span>{p.fkCol}</span> = ∅: a null foreign key makes no link
                  </div>
                ) : (
                  <RowButton
                    key={p.fkCol}
                    db={db}
                    r={{ table: p.parentTable, row: p.parentRow }}
                    slot={slots[p.parentTable] ?? 0}
                    via={`${p.fkCol} = ${p.parentRow}`}
                    onClick={onFocus}
                    onHover={setHover}
                    hovered={hover === rowKey({ table: p.parentTable, row: p.parentRow })}
                  />
                ),
              )}
            </div>
          )}
        </div>

        <div className="flex items-center justify-center gap-1.5 py-1.5 text-[14px] text-muted" aria-hidden="true">
          <ArrowUp size={14} /> F→P
        </div>

        {/* focus card */}
        <AnimatePresence mode="popLayout" initial={false}>
          <motion.div
            key={rowKey(focus)}
            initial={{ opacity: 0, scale: 0.97, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: -8 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
            className="card relative overflow-hidden ring-2 ring-accent"
          >
            <div aria-hidden="true" className="absolute inset-y-0 left-0 w-[4px]" style={{ background: catVar(slots[focus.table] ?? 0) }} />
            <div className="flex flex-wrap items-baseline justify-between gap-2 px-4 pt-3 pb-2">
              <span className="flex items-baseline gap-2">
                <span className="text-[17.5px] font-semibold text-ink">{rowLabel(focus)}</span>
                <span className="text-[15.5px] text-ink-2">{rowCaption(db, focus, 40)}</span>
              </span>
              <span className="flex flex-wrap items-center gap-2">
                {adopt && (
                  <Button size="sm" variant="secondary" icon={<Crosshair size={14} />} onClick={adopt.onClick} title={adopt.title}>
                    {adopt.label}
                  </Button>
                )}
                <span className="eyebrow">focus row</span>
              </span>
            </div>
            <dl className="grid grid-cols-[minmax(0,auto)_minmax(0,1fr)] items-baseline gap-x-4 gap-y-1 border-t border-grid px-4 py-2.5 text-[15px]">
              {table.schema.columns.map((c) => {
                const k = columnKind(table, c.name);
                const v = rec[c.name];
                const text = formatRaw(v, c.dtype);
                return (
                  <Fragment key={c.name}>
                    <dt className="flex min-w-0 items-center gap-1.5 text-ink-2">
                      {k === 'pk' ? (
                        <KeyRound aria-label="primary key" size={15} className="shrink-0" />
                      ) : k === 'fk' ? (
                        <Link2 aria-label="foreign key" size={15} className="shrink-0" />
                      ) : (
                        <SemBadge sem={semOfDType(c.dtype)} size="sm" />
                      )}
                      <span className="break-all">{c.name}</span>
                    </dt>
                    <dd
                      className={cx('m-0 min-w-0 break-words', c.dtype !== 'str' && 'num', isMissing(v) ? 'text-muted' : k === 'feature' ? 'text-ink' : 'text-ink-2')}
                      title={text.length > LONG_TEXT ? text : undefined}
                    >
                      {text.length > LONG_TEXT ? `${text.slice(0, LONG_TEXT - 1)}…` : text}
                      {k === 'fk' && !isMissing(v) && (
                        <span className="text-ink-2">
                          {' '}
                          → {table.schema.fkeys[c.name]}[{String(v)}]
                        </span>
                      )}
                    </dd>
                  </Fragment>
                );
              })}
            </dl>
            <div className="border-t border-grid px-4 py-2 text-[15px] leading-snug text-ink-2">
              <LinkSummary db={db} idx={idx} at={focus} />
            </div>
          </motion.div>
        </AnimatePresence>

        <div className="flex items-center justify-center gap-1.5 py-1.5 text-[14px] text-muted" aria-hidden="true">
          <ArrowDown size={14} /> P→F
        </div>

        {/* children */}
        <div className="rounded-xl border border-dashed border-border-strong px-3 py-2.5">
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
            <span className="text-[15px] font-semibold text-ink">
              Children <span className="font-normal text-ink-2">(P→F: their foreign keys point here)</span>
            </span>
            {groups.length > 0 && <span className="num text-[14px] text-ink-2">{nChildren} rows, no upper limit</span>}
          </div>
          {groups.length === 0 ? (
            <p className="!m-0 text-[15.5px] text-ink-2">
              No table has a foreign key into <span>{focus.table}</span>, so its rows have no children.
            </p>
          ) : (
            <div className="grid gap-3">
              {groups.map((g) => {
                const key = `${g.childTable}.${g.fkCol}`;
                const all = expanded[key];
                const list = all ? g.rows : g.rows.slice(0, CHIP_CAP);
                return (
                  <div key={key}>
                    <div className="mb-1.5 flex items-center gap-2 text-[15px] text-ink-2">
                      <TableSwatch slot={slots[g.childTable] ?? 0} shape label={g.childTable} />
                      <span>
                        via <span>{g.fkCol}</span> · <span className="num text-ink">{g.rows.length}</span> rows
                      </span>
                    </div>
                    {g.rows.length === 0 ? (
                      <div className="text-[15px] text-muted">none</div>
                    ) : (
                      <div className="flex flex-wrap gap-1">
                        {list.map((r, i) => {
                          const cr = { table: g.childTable, row: r };
                          const k = rowKey(cr);
                          return (
                            <motion.button
                              key={k}
                              type="button"
                              initial={{ opacity: 0, y: 4 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ duration: 0.18, delay: Math.min(i, 24) * 0.012 }}
                              onClick={() => onFocus(cr)}
                              onPointerEnter={() => setHover(k)}
                              onPointerLeave={() => setHover(null)}
                              onFocus={() => setHover(k)}
                              onBlur={() => setHover(null)}
                              title={`${rowLabel(cr)} ${rowCaption(db, cr)}`}
                              aria-label={`Go to ${rowLabel(cr)} ${rowCaption(db, cr)}`}
                              className={cx(
                                'num h-7 min-w-8 rounded-md border px-1.5 text-[14px] transition-colors',
                                hover === k ? 'border-accent bg-accent-soft text-ink' : 'border-border bg-surface text-ink-2 hover:border-accent hover:text-ink',
                              )}
                            >
                              {r}
                            </motion.button>
                          );
                        })}
                        {g.rows.length > CHIP_CAP && (
                          <button
                            type="button"
                            onClick={() => setExpanded((e) => ({ ...e, [key]: !all }))}
                            className="h-7 rounded-md px-1.5 text-[14px] font-medium text-accent-text hover:bg-accent-soft"
                          >
                            {all ? 'show fewer' : `+${g.rows.length - CHIP_CAP} more`}
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
          {taskChildTables.length > 0 && (
            <p className="!mt-2.5 !mb-0 text-[15px] leading-snug text-ink-2">
              Chapter 3 adds more children here: the rows of the {taskChildTables.join(' and ')} task table
              {taskChildTables.length === 1 ? '' : 's'} point to {focus.table} rows too.
            </p>
          )}
        </div>
      </div>

      {/* ±1 hop graph */}
      <div ref={graphBox} className="min-w-0">
        <div className="rounded-xl border border-border bg-surface-2/40 px-2 pt-2 pb-2.5">
          <div className="px-1.5 pb-1 text-[15px] font-semibold text-ink">
            {rowLabel(graphFocus)} and its neighbors, one hop
          </div>
          <EntityGraph
            nodes={graph.nodes}
            edges={graph.edges}
            selected={hoverDrawn ? hover : null}
            rings={focusRing}
            highlight={hoverDrawn ? noHighlight : graphHighlight}
            dimOthers={false}
            onNodeClick={(id: NodeId) => onFocus(parseRowKey(String(id)))}
            onNodeHover={(id) => setHover(id === null ? null : String(id))}
            height={graphH}
            labels="all"
            linkDistance={46}
            charge={-90}
            roving
            title={`Rows linked to ${rowLabel(graphFocus)}`}
          />
          <p className="!mt-1.5 !mb-0 min-h-[1.4em] px-1.5 text-[14.5px] leading-snug text-ink-2" aria-live="polite">
            {hover !== null && !hoverDrawn ? (
              <>
                <span className="text-ink">{rowLabel(parseRowKey(hover))}</span> is not drawn here (display limit).
              </>
            ) : graph.total > graph.shown ? (
              <>Drawing {graph.shown} of {graph.total} children to keep the picture readable (a display limit, not part of RT).</>
            ) : null}
          </p>
        </div>
        <p className="!mt-2.5 !mb-0 px-1 text-[15px] leading-snug text-ink-2">Click a node or a chip to walk there; Back retraces your steps.</p>
      </div>
    </div>
  );
});

function RowButton({
  db,
  r,
  slot,
  via,
  onClick,
  onHover,
  hovered,
}: {
  db: RawDatabase;
  r: RowRef;
  slot: number;
  via: string;
  onClick: (r: RowRef) => void;
  onHover: (k: string | null) => void;
  hovered: boolean;
}) {
  const k = rowKey(r);
  return (
    <button
      type="button"
      onClick={() => onClick(r)}
      onPointerEnter={() => onHover(k)}
      onPointerLeave={() => onHover(null)}
      onFocus={() => onHover(k)}
      onBlur={() => onHover(null)}
      className={cx(
        'flex min-w-0 items-center gap-2.5 rounded-lg border bg-surface px-3 py-2 text-left transition-colors',
        hovered ? 'border-accent' : 'border-border hover:border-accent',
      )}
    >
      <span aria-hidden="true" className="w-[3px] shrink-0 self-stretch rounded-full" style={{ background: catVar(slot) }} />
      <span className="min-w-0">
        <span className="block text-[15.5px] font-semibold break-words text-ink">{rowLabel(r)}</span>
        {rowCaption(db, r) && <span className="block text-[15px] leading-snug break-words text-ink">{rowCaption(db, r, 40)}</span>}
        <span className="block text-[14px] leading-snug break-words text-ink-2">via {via}</span>
      </span>
    </button>
  );
}
