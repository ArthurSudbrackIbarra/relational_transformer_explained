/**
 * Section 1 figure: the batch dictionary of the live context.
 *
 *  - a compact strip of all L tokens (sampler emission order);
 *  - for the inspected token: where it comes from (its row among the context rows of its table)
 *    and every per-token field, with what reads it (spec §5.1);
 *  - the full TokenTable (one row per token, one column per field).
 *
 * Hover previews a token everywhere; click pins it (the store's selectedPos, shared with the masks
 * and attention chapters).
 */
import { ChevronDown, ChevronLeft, ChevronRight } from 'lucide-react';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import {
  Figure,
  IconButton,
  SemBadge,
  StatusBadge,
  TableSwatch,
  TableView,
  TokenStrip,
  TokenTable,
  type TokenField,
  cx,
  formatRaw,
  formatSigned,
} from '@/components';
import { SemType, type Token } from '@/engine/types';
import { EgoGraph, type EgoNode } from './EgoGraph';
import { FieldList } from './fields';
import { INSPECTOR_ID, nodeLabel, pickVisibleLabel, rowTags, sourceTable, type TokensLive } from './live';

/** TokenTable columns: the flags this chapter is about right after the cell, then the batch order. */
const TABLE_FIELDS: TokenField[] = [
  'pos',
  'cell',
  'masks',
  'is_targets',
  'is_padding',
  'node_idxs',
  'f2p_nbr_idxs',
  'table_name_idxs',
  'col_name_idxs',
  'sem_types',
  'number_values',
  'text_values',
  'datetime_values',
  'boolean_values',
  'is_task_nodes',
  'class_value_idxs',
];

/**
 * True when `el` fits in the viewport below the top bar, so it can stay sticky without an inner
 * scroll area (wide screens only).
 */
function useFitsViewport(ref: React.RefObject<HTMLDivElement | null>, minWidth: number): boolean {
  const [fits, setFits] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof window === 'undefined') return;
    const check = () => {
      const topbar = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--topbar-h')) || 56;
      setFits(window.innerWidth >= minWidth && el.offsetHeight + topbar + 24 <= window.innerHeight);
    };
    check();
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(check) : null;
    ro?.observe(el);
    window.addEventListener('resize', check);
    return () => {
      ro?.disconnect();
      window.removeEventListener('resize', check);
    };
  }, [ref, minWidth]);
  return fits;
}

function storedLine(t: Token): string {
  switch (t.sem) {
    case SemType.Number:
      return `number_values = ${formatSigned(t.numberValue)}`;
    case SemType.DateTime:
      return `datetime_values = ${formatSigned(t.datetimeValue)}`;
    case SemType.Boolean:
      return `boolean_values = ${formatSigned(t.booleanValue)}`;
    default:
      return `text_values = E(text[${t.textIdx}])`;
  }
}

export function TokenInspector({ live }: { live: TokensLive }) {
  const { db, pdb, ctx, S, slots, assets, inspected, setInspected, tokensByNode, labels, target, jump, hoverAllowed } = live;
  const [hover, setHover] = useState<number | null>(null);
  // Hover previews are paused after an "inspect in Step 1" jump until the pointer really moves.
  const preview = (p: number | null) => {
    if (p !== null && !hoverAllowed()) return;
    setHover(p);
  };
  useEffect(() => setHover(null), [jump]);
  const [showTable, setShowTable] = useState(false);
  const leftRef = useRef<HTMLDivElement | null>(null);
  const sticky = useFitsViewport(leftRef, 1280);
  const tableWrap = useRef<HTMLDivElement | null>(null);
  const shownPos = hover ?? inspected;
  const t = ctx.tokens[shownPos] ?? ctx.tokens[target];
  const rowPositions = useMemo(() => (t && !t.isPadding ? tokensByNode.get(t.nodeIdx) ?? [] : []), [t, tokensByNode]);
  const present = (n: number) => tokensByNode.get(n)?.[0];

  const picks = useMemo(() => {
    const out: { label: string; pos: number }[] = [];
    if (ctx.targetPos >= 0) out.push({ label: 'masked target', pos: ctx.targetPos });
    const lab = pickVisibleLabel(labels); // the same label Step 3 compares with the target
    if (lab) out.push({ label: 'a visible label', pos: lab.pos });
    const firstText = ctx.tokens.find((x) => !x.isPadding && x.sem === SemType.Text);
    if (firstText) out.push({ label: 'a text cell', pos: firstText.pos });
    const firstNum = ctx.tokens.find((x) => !x.isPadding && x.sem === SemType.Number);
    if (firstNum) out.push({ label: 'a number cell', pos: firstNum.pos });
    const twoParents = ctx.tokens.find((x) => !x.isPadding && x.f2pNbrIdxs.filter((p) => p >= 0).length >= 2);
    if (twoParents) out.push({ label: 'a row with two parents', pos: twoParents.pos });
    const pad = ctx.tokens.find((x) => x.isPadding);
    if (pad) out.push({ label: 'padding', pos: pad.pos });
    return out;
  }, [ctx, labels]);

  // Rows of the shown token's table that are in the context (visit order).
  const tableRows = useMemo(() => {
    if (!t || t.isPadding) return [];
    const out: number[] = [];
    for (const v of ctx.visited) {
      const n = pdb.nodes[v.node];
      if (n && n.tableName === t.tableName && n.tableType === t.tableType && tokensByNode.has(v.node)) out.push(n.rowPos);
    }
    return out;
  }, [t, ctx, pdb, tokensByNode]);
  const raw = t && !t.isPadding ? sourceTable(db, t.tableName, t.tableType) : undefined;
  const nodeOfRow = (row: number) => {
    if (!t || !raw) return undefined;
    const key = `${t.tableName}:${t.tableType ?? 'Db'}`;
    const info = pdb.tableInfo[key];
    return info ? info.nodeIdxOffset + row : undefined;
  };

  // Ego view of the context graph: the token's row, its parents (f2p_nbr_idxs) above, and the rows of
  // the context that list it as a parent below. Arrows go child → parent, as foreign keys do.
  const tags = useMemo(() => rowTags(pdb, ctx), [pdb, ctx]);
  const rowNode = t && !t.isPadding ? t.nodeIdx : null;
  const parentsIn = useMemo(() => (t && !t.isPadding ? [...new Set(t.f2pNbrIdxs)].filter((p) => p >= 0 && tokensByNode.has(p)) : []), [t, tokensByNode]);
  const children = useMemo(() => {
    if (rowNode === null) return [];
    const out: number[] = [];
    for (const v of ctx.visited) if (tokensByNode.has(v.node) && pdb.nodes[v.node].f2pNbrIdxs.includes(rowNode)) out.push(v.node);
    return out;
  }, [rowNode, ctx, pdb, tokensByNode]);
  const egoNode = (n: number): EgoNode => ({ id: n, tag: tags.get(n) ?? String(n), name: nodeLabel(pdb, n), slot: slots[pdb.nodes[n].tableName] ?? 0 });
  // Mini table view: the columns that become tokens first, then the key columns (structure only).
  const tableCols = useMemo(() => {
    if (!raw) return undefined;
    const keys = (c: string) => c === raw.schema.pkey || c in raw.schema.fkeys;
    const names = raw.schema.columns.map((c) => c.name);
    return [...names.filter((c) => !keys(c)), ...names.filter(keys)];
  }, [raw]);
  // Bring the inspected cell into view sideways, inside the table's own scroller (never the page).
  // A layout effect, so it runs before TableView's own (passive) smooth vertical scroll to the row:
  // a horizontal scroll issued after it would cancel that animation.
  const colName = t && !t.isPadding ? t.colName : null;
  useLayoutEffect(() => {
    const wrap = tableWrap.current;
    if (!wrap || !colName || !tableCols) return;
    const scroller = wrap.querySelector<HTMLElement>('[role="table"]');
    const headers = wrap.querySelectorAll<HTMLElement>('[role="columnheader"]');
    const i = tableCols.indexOf(colName);
    const h = i >= 0 ? headers[i + 1] : undefined; // headers[0] is the "row" column
    if (!scroller || !h) return;
    const left = h.offsetLeft;
    const right = left + h.offsetWidth;
    if (left >= scroller.scrollLeft && right <= scroller.scrollLeft + scroller.clientWidth) return;
    scroller.scrollLeft = Math.max(0, left - (scroller.clientWidth - h.offsetWidth) / 2);
  }, [colName, tableCols, t?.nodeIdx]);

  const step = (d: number) => setInspected(Math.max(0, Math.min(S - 1, inspected + d)));
  // Tables present in the context, in first-appearance order (strip legend).
  const stripTables = useMemo(() => {
    const seen: string[] = [];
    for (const x of ctx.tokens) if (!x.isPadding && !seen.includes(x.tableName)) seen.push(x.tableName);
    return seen;
  }, [ctx]);
  const hasPadding = ctx.numReal < ctx.tokens.length;

  return (
    <Figure
      wide
      id={INSPECTOR_ID}
      className="scroll-mt-[calc(var(--topbar-h)+16px)]"
      title={`The batch dictionary of the live context: ${S} positions`}
      controls={
        <div className="flex w-full flex-wrap items-center gap-x-3 gap-y-2">
          <div className="flex items-center gap-1">
            <IconButton size="sm" label="Previous token" onClick={() => step(-1)} disabled={inspected <= 0}>
              <ChevronLeft size={16} />
            </IconButton>
            <span className="num min-w-[86px] text-center text-[15px] text-ink-2">
              pos <span className="font-semibold text-ink">{inspected}</span> / {S - 1}
            </span>
            <IconButton size="sm" label="Next token" onClick={() => step(1)} disabled={inspected >= S - 1}>
              <ChevronRight size={16} />
            </IconButton>
          </div>
          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Jump to a token">
            <span className="text-[14.5px] text-ink-2">Jump to</span>
            {picks.map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => setInspected(p.pos)}
                aria-pressed={inspected === p.pos}
                className={cx(
                  'h-7 rounded-full border px-2.5 text-[15px] transition-colors pointer-coarse:h-9',
                  inspected === p.pos ? 'border-accent bg-accent-soft text-accent-text' : 'border-border bg-surface text-ink-2 hover:bg-surface-2 hover:text-ink',
                )}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      }
      caption={
        <>
          Top: one square per position, in the order the sampler emitted the cells. Middle: where the inspected token comes from, and every field
          the model receives for it. Bottom: the whole batch dictionary as one table, one row per position (open it with the toggle). The inspected
          token is shared with the embedding, masks and attention chapters.
        </>
      }
    >
      <p className="mb-2 text-[16px] leading-snug text-ink-2">
        Each square is one position, in the order the sampler wrote it. Click one, or use the arrows, to open its record below.
      </p>
      <TokenStrip
        tokens={ctx.tokens}
        tableSlots={slots}
        variant="cells"
        selected={inspected}
        highlight={rowPositions}
        onSelect={(p) => setInspected(p)}
        onHover={preview}
        showPadding={true}
        label="All positions of the live context"
        className="pointer-coarse:[&_[role=listbox]]:gap-1 pointer-coarse:[&_[role=option]]:h-6 pointer-coarse:[&_[role=option]]:w-6"
      />
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[14.5px] text-ink-2">
        {stripTables.map((tb) => (
          <TableSwatch key={tb} slot={slots[tb] ?? 0} label={tb} />
        ))}
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="target-outline inline-flex h-[18px] w-[18px] items-center justify-center rounded-[4px] bg-accent-soft text-[13px] font-bold text-accent-text">
            ?
          </span>
          masked target
        </span>
        {hasPadding && (
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden="true" className="inline-block h-[18px] w-[18px] rounded-[4px] border border-dashed border-axis" />
            padding
          </span>
        )}
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-[14px] w-[14px] rounded-[3px] ring-2 ring-accent ring-offset-1 ring-offset-surface" />
          the inspected token’s row (a row of k cells is k tokens)
        </span>
        <span>· hover to preview, click or Enter to open</span>
      </div>

      <div className="mt-6 grid gap-7 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        {/* ── where it comes from (sticky on wide screens when it fits the viewport; never an inner scroll area) */}
        <div ref={leftRef} className={cx('min-w-0 self-start', sticky && 'sticky top-[calc(var(--topbar-h)+16px)]')}>
          <div className="eyebrow mb-2">Where it comes from</div>
          {!t || t.isPadding ? (
            <div className="rounded-xl border border-dashed border-axis px-4 py-4 text-[16.5px] leading-relaxed text-ink-2">
              <div className="font-semibold text-ink">Position {t?.pos}: padding</div>
              No database cell. The sampler ran out of rows before L, so this slot keeps the defaults it was allocated with: node −1, no parents, every
              value 0, is_padding = true (Step 5).
            </div>
          ) : (
            <>
              <div className="mb-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[16px]">
                <TableSwatch slot={slots[t.tableName] ?? 0} label={t.tableName} />
                {t.tableType && t.tableType !== 'Db' && <span className="rounded bg-surface-3 px-1.5 text-[14px] text-ink-2">{t.tableType} split</span>}
                <span className="text-ink-2">
                  row <span className="num text-ink">{t.rowPos}</span> → node <span className="num text-ink">{t.nodeIdx}</span>
                </span>
                <span className="text-[14.5px] text-ink-2">· click a cell to inspect it</span>
              </div>
              {raw && (
                <div ref={tableWrap}>
                  <TableView
                    schema={raw.schema}
                    rows={raw.rows}
                    columns={tableCols}
                    slot={slots[t.tableName] ?? 0}
                    title={`${t.tableName}${t.tableType && t.tableType !== 'Db' ? ` (${t.tableType})` : ''}`}
                    rowFilter={tableRows}
                    highlightRows={[t.rowPos]}
                    highlightCells={[`${t.rowPos}:${t.colName}`]}
                    targetCell={ctx.targetPos >= 0 && ctx.tokens[ctx.targetPos].nodeIdx === t.nodeIdx ? { row: t.rowPos, col: ctx.tokens[ctx.targetPos].colName } : null}
                    scrollToRow={t.rowPos}
                    maxHeight={132}
                    footer="Context rows in visit order. Key columns (last) give no token."
                    onCellClick={(row, col) => {
                      const n = nodeOfRow(row);
                      if (n === undefined) return;
                      const pos = (tokensByNode.get(n) ?? []).find((p) => ctx.tokens[p].colName === col);
                      if (pos !== undefined) setInspected(pos);
                    }}
                  />
                </div>
              )}
              <dl className="mt-3 grid grid-cols-[112px_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-[15.5px]">
                <dt className="text-ink-2">cell</dt>
                <dd className="min-w-0">
                  <span className="font-mono text-ink">{t.colName}</span> ={' '}
                  {t.isTarget ? (
                    <span className="target-outline rounded-[3px] px-1 font-mono font-bold text-accent-text">?</span>
                  ) : (
                    <span className="font-mono text-ink">{formatRaw(t.raw, t.sem === SemType.DateTime ? 'datetime' : undefined)}</span>
                  )}
                  {t.isTarget && <span className="ml-1.5 text-[14.5px] text-ink-2">(hidden: {formatRaw(t.raw)})</span>}
                </dd>
                <dt className="text-ink-2">phrase</dt>
                <dd className="font-mono text-ink">“{pdb.text[t.colNameIdx]}”</dd>
                <dt className="text-ink-2">type</dt>
                <dd className="flex items-center gap-1.5">
                  <SemBadge sem={t.sem} size="sm" showLabel />
                </dd>
                <dt className="text-ink-2">stored as</dt>
                <dd className="num text-ink">{storedLine(t)}</dd>
                <dt className="text-ink-2">row’s tokens</dt>
                <dd className="flex flex-wrap gap-1">
                  {rowPositions.map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => setInspected(p)}
                      className={cx(
                        'num h-6 rounded-[5px] border px-1.5 text-[14px] pointer-coarse:h-8 pointer-coarse:min-w-8',
                        p === t.pos ? 'border-accent bg-accent-soft text-accent-text' : 'border-border bg-surface text-ink-2 hover:bg-surface-2',
                      )}
                      title={`${ctx.tokens[p].colName}`}
                    >
                      #{p}
                    </button>
                  ))}
                </dd>
              </dl>
              {t.isTarget && (
                <div className="mt-3">
                  <StatusBadge status="info">the single masked cell</StatusBadge>
                </div>
              )}
            </>
          )}
          {rowNode !== null && (
            <div className="mt-5">
              <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3 text-[15px]">
                <span className="font-semibold text-ink">Its row in the context graph</span>
                <span className="text-[14px] text-ink-2">arrow = child → parent · click a node to inspect it</span>
              </div>
              <EgoGraph
                parents={parentsIn.map(egoNode)}
                row={egoNode(rowNode)}
                children={children.map(egoNode)}
                seed={ctx.params.seedNode}
                onPick={(id) => {
                  const p = tokensByNode.get(id)?.[0];
                  if (p !== undefined) setInspected(p);
                }}
              />
              <ul className="mt-1.5 space-y-0.5 text-[14.5px] leading-snug text-ink-2">
                <li>
                  <span className="font-medium text-ink">Parents (lit):</span>{' '}
                  {parentsIn.length ? <span className="font-mono">{parentsIn.map((p) => nodeLabel(pdb, p)).join(', ')}</span> : 'none in the context'}
                </li>
                <li>
                  <span className="font-medium text-ink">Rows that list it as a parent:</span>{' '}
                  {children.length ? `${children.length} in the context` : 'none in the context'}
                  {rowNode === ctx.params.seedNode ? ' · dashed ring = the seed row' : ''}
                </li>
              </ul>
            </div>
          )}
        </div>

        {/* ── what the model receives */}
        <div className="min-w-0">
          <div className="eyebrow mb-2">What the model receives at position {t?.pos}</div>
          {t && <FieldList token={t} pdb={pdb} store={assets.store} slots={slots} present={present} onPick={(p) => setInspected(p)} />}
        </div>
      </div>

      <div className="mt-7 border-t border-grid pt-4">
        <button
          type="button"
          aria-expanded={showTable}
          aria-controls="tokens-full-table"
          onClick={() => setShowTable((v) => !v)}
          className="flex w-full flex-wrap items-baseline gap-x-2 text-left hover:text-accent-text"
        >
          <ChevronDown aria-hidden="true" size={17} className={cx('shrink-0 self-center text-ink-2 transition-transform', !showTable && '-rotate-90')} />
          <span className="text-[16px] font-semibold text-ink">Every position, every field</span>
          <span className="text-[15px] text-ink-2">
            {showTable ? 'hide the table' : `show the whole batch dictionary as one table (${S} rows)`}
          </span>
        </button>
        {showTable && (
          <div id="tokens-full-table" className="mt-3">
            <div className="mb-2 text-[14.5px] text-ink-2">
              The flag columns come right after the cell; scroll sideways for the ids and values.{' '}
              <span className="font-mono">col_name_values</span> and <span className="font-mono">text_values</span> are 384-d vectors (shown above for
              the inspected token).
            </div>
            <TokenTable
              tokens={ctx.tokens}
              text={pdb.text}
              tableSlots={slots}
              fields={TABLE_FIELDS}
              selected={inspected}
              highlight={hover !== null ? [hover] : rowPositions}
              onHover={preview}
              onSelect={(p) => setInspected(p)}
              maxHeight={330}
              colWidths={{ cell: 250, f2p_nbr_idxs: 176, table_name_idxs: 184 }}
              tableNameChars={16}
            />
          </div>
        )}
      </div>
    </Figure>
  );
}
