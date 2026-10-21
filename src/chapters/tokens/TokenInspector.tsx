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
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useMemo, useState } from 'react';
import {
  EntityGraph,
  Figure,
  IconButton,
  SemBadge,
  StatusBadge,
  TableSwatch,
  TableView,
  TokenStrip,
  TokenTable,
  cx,
  formatRaw,
  formatSigned,
  edgeKey,
  type GraphEdgeSpec,
  type GraphNodeSpec,
  type NodeId,
  type RingStyle,
} from '@/components';
import { SemType, type Token } from '@/engine/types';
import { FieldList } from './fields';
import { INSPECTOR_ID, nodeLabel, pickVisibleLabel, rowTags, sourceTable, tokenRowRef, type TokensLive } from './live';

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
  const { db, pdb, ctx, S, slots, assets, inspected, setInspected, tokensByNode, labels, target } = live;
  const [hover, setHover] = useState<number | null>(null);
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

  // Mini entity graph of the context rows (child → parent arrows), cross-highlighted with the strip.
  const tags = useMemo(() => rowTags(pdb, ctx), [pdb, ctx]);
  const graph = useMemo(() => {
    const nodes: GraphNodeSpec[] = [];
    const edges: GraphEdgeSpec[] = [];
    for (const [n] of tokensByNode) {
      const node = pdb.nodes[n];
      nodes.push({ id: n, label: tags.get(n) ?? String(n), table: node.tableName, slot: slots[node.tableName] ?? 0, title: nodeLabel(pdb, n) });
    }
    for (const [n] of tokensByNode) for (const p of new Set(pdb.nodes[n].f2pNbrIdxs)) if (tokensByNode.has(p)) edges.push({ source: n, target: p });
    return { nodes, edges };
  }, [pdb, tokensByNode, tags, slots]);
  const rowNode = t && !t.isPadding ? t.nodeIdx : null;
  const parentsIn = useMemo(() => (t && !t.isPadding ? [...new Set(t.f2pNbrIdxs)].filter((p) => p >= 0 && tokensByNode.has(p)) : []), [t, tokensByNode]);
  const graphHighlight = useMemo(() => (rowNode === null ? [] : [rowNode, ...parentsIn]), [rowNode, parentsIn]);
  const graphEdges = useMemo(() => (rowNode === null ? [] : parentsIn.map((p) => edgeKey(rowNode, p))), [rowNode, parentsIn]);
  const rings = useMemo(() => {
    const m = new Map<NodeId, RingStyle>();
    m.set(ctx.params.seedNode, 'target');
    return m;
  }, [ctx]);

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
                  'h-7 rounded-full border px-2.5 text-[15px] transition-colors',
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
          the model receives for it. Bottom: the whole batch dictionary, one row per position. The inspected token is shared with the embedding,
          masks and attention chapters.
        </>
      }
    >
      <TokenStrip
        tokens={ctx.tokens}
        tableSlots={slots}
        variant="cells"
        selected={inspected}
        highlight={rowPositions}
        onSelect={(p) => setInspected(p)}
        onHover={setHover}
        showPadding={true}
        label="All positions of the live context"
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
        <span>· hover to preview, click or Enter to pin</span>
      </div>

      <div className="mt-6 grid gap-7 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        {/* ── where it comes from */}
        <div className="scroll-quiet min-w-0 lg:sticky lg:top-[calc(var(--topbar-h)+16px)] lg:max-h-[calc(100vh-var(--topbar-h)-32px)] lg:self-start lg:overflow-y-auto lg:pr-1">
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
              </div>
              {raw && (
                <TableView
                  schema={raw.schema}
                  rows={raw.rows}
                  slot={slots[t.tableName] ?? 0}
                  title={`${t.tableName}${t.tableType && t.tableType !== 'Db' ? ` (${t.tableType})` : ''}`}
                  rowFilter={tableRows}
                  highlightRows={[t.rowPos]}
                  highlightCells={[`${t.rowPos}:${t.colName}`]}
                  targetCell={ctx.targetPos >= 0 && ctx.tokens[ctx.targetPos].nodeIdx === t.nodeIdx ? { row: t.rowPos, col: ctx.tokens[ctx.targetPos].colName } : null}
                  scrollToRow={t.rowPos}
                  maxHeight={150}
                  footer="Rows of this table that are in the context, in visit order. Click a cell to inspect its token."
                  onCellClick={(row, col) => {
                    const n = nodeOfRow(row);
                    if (n === undefined) return;
                    const pos = (tokensByNode.get(n) ?? []).find((p) => ctx.tokens[p].colName === col);
                    if (pos !== undefined) setInspected(pos);
                  }}
                />
              )}
              <dl className="mt-3 grid grid-cols-[96px_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-[15.5px]">
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
                        'num h-6 rounded-[5px] border px-1.5 text-[14px]',
                        p === t.pos ? 'border-accent bg-accent-soft text-accent-text' : 'border-border bg-surface text-ink-2 hover:bg-surface-2',
                      )}
                      title={`${ctx.tokens[p].colName}`}
                    >
                      #{p}
                    </button>
                  ))}
                </dd>
              </dl>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {t.isTarget && <StatusBadge status="info">the single masked cell</StatusBadge>}
                <span className="text-[14.5px] text-ink-2">{tokenRowRef(t)}</span>
              </div>
            </>
          )}
          <div className="mt-5">
            <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3 text-[15px]">
              <span className="font-semibold text-ink">Its row in the context graph</span>
              <span className="text-[14px] text-ink-2">{graph.nodes.length} rows · arrow = child → parent</span>
            </div>
            <EntityGraph
              nodes={graph.nodes}
              edges={graph.edges}
              highlight={graphHighlight}
              highlightEdges={graphEdges}
              dimOthers
              rings={rings}
              selected={rowNode}
              labels="highlighted"
              height={210}
              linkDistance={40}
              charge={-90}
              onNodeHover={(id) => setHover(id === null ? null : tokensByNode.get(Number(id))?.[0] ?? null)}
              onNodeClick={(id) => {
                const p = tokensByNode.get(Number(id))?.[0];
                if (p !== undefined) setInspected(p);
              }}
              title="Rows of the live context, linked child → parent by foreign keys"
            />
            <div className="mt-1 text-[14px] leading-snug text-ink-2">
              Lit: the token’s row and the parents in its <span className="font-mono">f2p_nbr_idxs</span>. Dashed ring: the seed row. Click a node
              to inspect its first token.
            </div>
          </div>
        </div>

        {/* ── what the model receives */}
        <div className="min-w-0">
          <div className="eyebrow mb-2">What the model receives at position {t?.pos}</div>
          {t && <FieldList token={t} pdb={pdb} store={assets.store} slots={slots} present={present} onPick={(p) => setInspected(p)} />}
        </div>
      </div>

      <div className="mt-7">
        <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
          <div className="text-[16px] font-semibold text-ink">Every position, every field</div>
          <div className="text-[14.5px] text-ink-2">
            <span className="font-mono">col_name_values</span> and <span className="font-mono">text_values</span> are 384-d vectors (shown above for the
            inspected token). Scroll the table sideways for the value and flag columns →
          </div>
        </div>
        <TokenTable
          tokens={ctx.tokens}
          text={pdb.text}
          tableSlots={slots}
          selected={inspected}
          highlight={hover !== null ? [hover] : rowPositions}
          onHover={setHover}
          onSelect={(p) => setInspected(p)}
          maxHeight={330}
          colWidths={{ f2p_nbr_idxs: 176, table_name_idxs: 184 }}
          tableNameChars={16}
        />
      </div>
    </Figure>
  );
}
