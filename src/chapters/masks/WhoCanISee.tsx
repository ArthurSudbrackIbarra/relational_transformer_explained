/**
 * Section 3 figure, "Who can I see?": pick a query token (click, or Enter on a strip chip); in each of
 * the four masks its allowed keys light up in a per-table view of the live context (like paper Fig. 1,
 * right), in a mini entity graph and in the token strip. Hovering a cell never changes the query: it
 * names the cell and says, for each mask, whether the query may read it.
 */
import { Check, X } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';
import {
  Button,
  Cite,
  EntityGraph,
  Figure,
  LayerLabel,
  Segmented,
  TableSwatch,
  TokenStrip,
  cx,
  formatRaw,
  tokenValueText,
  type GraphEdgeSpec,
  type GraphNodeSpec,
  type NodeId,
  type RingStyle,
} from '@/components';
import { SemType, type AttnLayer, type Token } from '@/engine/types';
import { useTask } from '@/state/hooks';
import { ContextTables, ContextTablesLegend } from './ContextTables';
import { int, LAYER_ORDER, listAnd, maskRow, presets, rowName, rowTag, type HoleKind } from './maskUtils';
import type { LiveMasks } from './useLiveMasks';

const LAYER_NAME: Record<AttnLayer, string> = { col: 'Column', feat: 'Feature', nbr: 'Neighbor', full: 'Full' };

function valueText(t: Token): string {
  if (t.isTarget) return '? (masked)';
  if (t.sem === SemType.Text) return `“${tokenValueText(t)}”`;
  return formatRaw(t.raw, t.sem === SemType.DateTime ? 'datetime' : undefined);
}

export function WhoCanISee({ live }: { live: LiveMasks }) {
  const { ctx, masks, S, slots, grouping, prefixes, pinned, setPinned, target } = live;
  const task = useTask();
  const [hoverK, setHoverK] = useState<number | null>(null);
  const [layer, setLayer] = useState<AttnLayer>('col');
  const q = pinned;
  const qTok = ctx.tokens[q];
  const kTok = hoverK !== null && !ctx.tokens[hoverK]?.isPadding ? ctx.tokens[hoverK] : null;

  const rows = useMemo(() => {
    const out = {} as Record<AttnLayer, Uint8Array>;
    for (const l of LAYER_ORDER) out[l] = maskRow(masks[l], S, q);
    return out;
  }, [masks, S, q]);

  const summary = useMemo(() => {
    const out = {} as Record<AttnLayer, { keys: number; rows: number; byTable: { name: string; slot: number; cells: number; rows: number }[] }>;
    for (const l of LAYER_ORDER) {
      const rowSet = new Set<number>();
      const byTable = new Map<string, { name: string; slot: number; cells: number; rows: Set<number> }>();
      let keys = 0;
      for (let k = 0; k < S; k++) {
        if (!rows[l][k]) continue;
        const t = ctx.tokens[k];
        keys++;
        rowSet.add(t.nodeIdx);
        let e = byTable.get(t.tableName);
        if (!e) byTable.set(t.tableName, (e = { name: t.tableName, slot: slots[t.tableName] ?? 0, cells: 0, rows: new Set() }));
        e.cells++;
        e.rows.add(t.nodeIdx);
      }
      out[l] = { keys, rows: rowSet.size, byTable: [...byTable.values()].map((e) => ({ ...e, rows: e.rows.size })) };
    }
    return out;
  }, [rows, S, ctx, slots]);

  const presetList = useMemo(() => presets(ctx.tokens, ctx.targetPos), [ctx]);
  const holeKinds = useMemo(() => {
    const s = new Set<HoleKind>();
    for (const t of grouping.tables) for (const r of t.rows) for (const h of r.holes) if (h) s.add(h);
    return [...s];
  }, [grouping]);

  // mini entity graph: rows of the context, F→P links between rows that are both in the context
  const graph = useMemo(() => {
    const nodes: GraphNodeSpec[] = [];
    const edges: GraphEdgeSpec[] = [];
    for (const tbl of grouping.tables) {
      for (const r of tbl.rows) {
        nodes.push({ id: r.node, label: rowTag(r.first, prefixes), table: tbl.name, slot: tbl.slot, title: `${rowName(r.first)} · ${r.positions.length} cells` });
      }
    }
    for (const r of grouping.rowByNode.values()) {
      for (const p of new Set(r.parents)) if (grouping.rowByNode.has(p)) edges.push({ source: r.node, target: p });
    }
    return { nodes, edges };
  }, [grouping, prefixes]);

  const litRows = useMemo(() => {
    const s = new Set<NodeId>();
    for (let k = 0; k < S; k++) if (rows[layer][k]) s.add(ctx.tokens[k].nodeIdx);
    return s;
  }, [rows, layer, S, ctx]);

  // With no allowed key, nothing may light up: mute every row except the query's own (no fallback to
  // the graph neighbours, which would wrongly suggest they are keys).
  const muted = useMemo(() => (litRows.size > 0 ? undefined : graph.nodes.map((n) => n.id).filter((id) => id !== qTok?.nodeIdx)), [litRows, graph, qTok]);

  const rings = useMemo(() => {
    const m = new Map<NodeId, RingStyle>();
    const seedNode = ctx.tokens[target]?.nodeIdx;
    if (seedNode !== undefined) m.set(seedNode, 'target');
    return m;
  }, [ctx, target]);

  const strip = useMemo(() => {
    const out: number[] = [];
    for (let k = 0; k < S; k++) if (rows[layer][k]) out.push(k);
    return out;
  }, [rows, layer, S]);

  const firstPosOfNode = (id: NodeId) => grouping.rowByNode.get(Number(id))?.positions[0] ?? null;
  const pin = (p: number | null) => {
    if (p !== null && !ctx.tokens[p].isPadding) setPinned(p);
  };

  const describe = (pos: number): ReactNode => {
    const t = ctx.tokens[pos];
    return (
      <>
        <div className="font-mono text-ink">
          {rowTag(t, prefixes)}·{t.colName} <span className="text-muted">=</span> <span className="num">{valueText(t)}</span>
        </div>
        <div className="text-[14px] text-muted">
          #{pos} · {t.colName} of {rowName(t)}
          {pos === q ? ' · the query itself' : ''}
        </div>
        <PairStatus masks={masks} S={S} q={q} k={pos} className="mt-1" />
        <div className="mt-1 text-[14px] text-muted">Click to make it the query.</div>
      </>
    );
  };

  return (
    <Figure
      wide
      title="Who can I see? One query, four masks"
      controls={
        <div className="flex w-full flex-wrap items-center gap-2">
          <span className="text-[15px] font-medium text-ink-2">Query presets</span>
          {presetList.map((p) => (
            <Button key={p.id} size="sm" variant={pinned === p.pos ? 'primary' : 'secondary'} aria-pressed={pinned === p.pos} onClick={() => setPinned(p.pos)}>
              {p.label}
            </Button>
          ))}
        </div>
      }
      caption={
        <>
          Click any token (a chip in the strip, a table cell or a graph node) to make it the query. Hover a cell to name it and see which masks
          let the query read it. Each panel fills the keys the query may attend to under one mask. The strip and the graph follow the mask picked
          above the graph, or click a panel’s title. All four masks are the engine’s, computed from ids only{' '}
          <Cite kind="code" ref="rt/model.py:164-191" />.
        </>
      }
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-2 rounded-lg bg-surface-2 px-3 py-2 text-[16px]" aria-live="polite">
        <div className="min-w-0">
          <span className="text-muted">Query </span>
          <span className="num font-semibold text-ink">#{q}</span>{' '}
          {qTok && !qTok.isPadding && (
            <>
              <span className="font-mono text-ink">{qTok.colName}</span> <span className="text-muted">of</span>{' '}
              <span className="font-mono text-ink">{rowName(qTok)}</span> <span className="text-muted">=</span>{' '}
              <span className={cx('num', qTok.isTarget ? 'font-semibold text-accent-text' : 'text-ink')}>{valueText(qTok)}</span>
            </>
          )}
        </div>
        {kTok && hoverK !== null ? (
          <div className="flex min-w-0 flex-wrap items-baseline gap-x-2 text-[15px]">
            <span className="text-muted">key</span>
            <span className="num text-ink">
              #{hoverK} {rowTag(kTok, prefixes)}·{kTok.colName}
            </span>
            <PairStatus masks={masks} S={S} q={q} k={hoverK} />
          </div>
        ) : (
          <div className="num text-[15px] text-ink-2">
            {LAYER_ORDER.map((l, i) => (
              <span key={l}>
                {i > 0 && ' · '}
                {LAYER_NAME[l]} {int(summary[l].keys)}
              </span>
            ))}{' '}
            keys
          </div>
        )}
      </div>

      <div className="mt-4">
        <div className="mb-1.5 text-[15px] font-medium text-ink-2">
          Context tokens in emission order. Rings: keys allowed under the <span className="text-ink">{LAYER_NAME[layer].toLowerCase()}</span> mask.
        </div>
        <TokenStrip
          tokens={ctx.tokens}
          tableSlots={slots}
          variant="cells"
          highlight={strip}
          dimOthers
          selected={q}
          onHover={(p) => setHoverK(p !== null && !ctx.tokens[p].isPadding ? p : null)}
          onSelect={pin}
          label="Context tokens: click one to make it the query"
        />
      </div>

      <div className="mt-5 grid gap-4 md:grid-cols-2">
        {LAYER_ORDER.map((l) => {
          const s = summary[l];
          const active = l === layer;
          return (
            <div key={l} className={cx('min-w-0 rounded-xl border px-3.5 pt-3 pb-3.5 transition-colors', active ? 'border-accent bg-accent-soft' : 'border-border')}>
              <div className="flex items-start justify-between gap-3">
                <button
                  type="button"
                  onClick={() => setLayer(l)}
                  aria-pressed={active}
                  className="rounded-md text-left text-[16px] hover:text-accent-text"
                  title="Show this mask in the strip and the graph"
                >
                  <LayerLabel layer={l} />
                  <span className="text-muted"> mask</span>
                </button>
                <span className="num shrink-0 text-right text-[15px] text-ink-2">
                  {s.keys === 0 ? (
                    <span className="font-semibold text-ink">no key: zero update</span>
                  ) : (
                    <>
                      {int(s.keys)} {s.keys === 1 ? 'key' : 'keys'} in {int(s.rows)} {s.rows === 1 ? 'row' : 'rows'}
                    </>
                  )}
                </span>
              </div>
              <p className="mt-1 mb-2.5 text-[15px] leading-snug text-ink-2">
                <PanelReading layer={l} q={qTok} keys={s.keys} tokens={ctx.tokens} numReal={ctx.numReal} prefixes={prefixes} byTable={s.byTable} />
              </p>
              <ContextTables
                tokens={ctx.tokens}
                grouping={grouping}
                allowed={rows[l]}
                query={q}
                hoverKey={hoverK}
                cell={12}
                maxRowsPerStack={14}
                onHover={setHoverK}
                onSelect={pin}
                describe={describe}
                label={`${LAYER_NAME[l]} mask: query ${q} may attend to ${s.keys} keys in ${s.rows} rows.`}
                showPadding={l === 'col'}
              />
            </div>
          );
        })}
      </div>
      <ContextTablesLegend className="mt-3" holes={holeKinds} />

      <div className="mt-7 grid items-start gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
            <span className="text-[16px] font-medium text-ink">Rows of the context as a graph</span>
            <Segmented
              label="Mask shown in the strip and graph"
              value={layer}
              onChange={setLayer}
              options={LAYER_ORDER.map((l) => ({ value: l, label: <LayerLabel layer={l} size={13} /> }))}
            />
          </div>
          <EntityGraph
            nodes={graph.nodes}
            edges={graph.edges}
            highlight={litRows}
            muted={muted}
            dimOthers
            rings={rings}
            selected={qTok ? qTok.nodeIdx : null}
            onNodeClick={(id) => pin(firstPosOfNode(id))}
            labels={litRows.size <= 10 ? 'highlighted' : 'none'}
            height={330}
            linkDistance={46}
            charge={-110}
            title="Rows in the context window, linked child → parent by foreign keys"
          />
        </div>
        <div className="min-w-0 text-[17px] leading-relaxed text-ink">
          <LayerReading layer={layer} keys={summary[layer].keys} rows={summary[layer].rows} isTarget={q === ctx.targetPos} forecast={task.kind === 'forecast'} />
          <p className="mt-3 text-[15.5px] text-ink-2">
            A node lights up when at least one of its cells is an allowed key; when the mask row is empty, every node but the query’s own is
            dimmed. Arrows point from a child row (it holds the foreign key) to its parent. Row tags:{' '}
            {Object.entries(prefixes).map(([t, p], i) => `${i ? ', ' : ''}${p} = ${t}`)}; train/val/test = task rows of that split.
          </p>
        </div>
      </div>
    </Figure>
  );
}

/** ✓ / ✗ for one (query, key) pair under each of the four masks. */
function PairStatus({ masks, S, q, k, className }: { masks: LiveMasks['masks']; S: number; q: number; k: number; className?: string }) {
  return (
    <span className={cx('flex flex-wrap gap-x-3 gap-y-0.5 text-[15px]', className)}>
      {LAYER_ORDER.map((l) => {
        const ok = masks[l][q * S + k] === 1;
        return (
          <span key={l} className={cx('inline-flex items-center gap-0.5', ok ? 'text-ink' : 'text-muted')}>
            {ok ? <Check aria-hidden="true" size={14} style={{ color: 'var(--good)' }} /> : <X aria-hidden="true" size={14} />}
            {LAYER_NAME[l].toLowerCase()}
            <span className="sr-only">{ok ? ' allowed' : ' blocked'}</span>
          </span>
        );
      })}
    </span>
  );
}

/** One line per panel: what this mask row means for this query, in the live context. */
function PanelReading({
  layer,
  q,
  keys,
  tokens,
  numReal,
  prefixes,
  byTable,
}: {
  layer: AttnLayer;
  q: Token | undefined;
  keys: number;
  tokens: readonly Token[];
  numReal: number;
  prefixes: Record<string, string>;
  byTable: { name: string; slot: number; cells: number; rows: number }[];
}) {
  if (!q || q.isPadding) return null;
  const tag = rowTag(q, prefixes);
  const counts = (
    <span className="inline-flex flex-wrap gap-x-2.5 align-middle">
      {byTable.map((b) => (
        <span key={b.name} className="inline-flex items-center gap-1">
          <TableSwatch slot={b.slot} size={8} />
          <span className="num">
            {b.cells} in {b.name}
          </span>
        </span>
      ))}
    </span>
  );
  switch (layer) {
    case 'col':
      return (
        <>
          The <span className="font-mono text-ink">{q.colName}</span> cells of {q.tableName}, one per row that has one.
        </>
      );
    case 'feat': {
      const parents = [...new Set(q.f2pNbrIdxs.filter((p) => p >= 0))]
        .map((p) => tokens.find((t) => !t.isPadding && t.nodeIdx === p))
        .filter((t): t is Token => !!t)
        .map((t) => rowTag(t, prefixes));
      return parents.length ? (
        <>
          Own row {tag} + {parents.length === 1 ? 'parent' : 'parents'} {listAnd(parents)}: {counts}
        </>
      ) : (
        <>Own row {tag} only: no parent of it is in the context.</>
      );
    }
    case 'nbr':
      return keys === 0 ? (
        <>No row in the context points to {tag}, so this row of the mask is empty.</>
      ) : (
        <>
          Rows that point to {tag} (its children): {counts}
        </>
      );
    case 'full':
      return <>Every non-padding token: all {int(numReal)} of them.</>;
  }
}

function LayerReading({ layer, keys, rows, isTarget, forecast }: { layer: AttnLayer; keys: number; rows: number; isTarget: boolean; forecast: boolean }) {
  switch (layer) {
    case 'col':
      return (
        <p>
          <strong>Column.</strong> {keys} cells share the query’s column (same column name, same table), spread over {rows} rows.{' '}
          {isTarget
            ? forecast
              ? 'For the masked target these are the visible labels of the task, from every split, plus the target itself.'
              : 'For the masked target these are the visible values of the target column in other rows, plus the target itself.'
            : 'The query compares itself with the other values of its column.'}
        </p>
      );
    case 'feat':
      return (
        <p>
          <strong>Feature.</strong> {keys} cells: the query’s own row and those of its parent rows (the rows its foreign keys point to) that are
          in the context, {rows} {rows === 1 ? 'row' : 'rows'} in total. Information flows from parent to child only.
        </p>
      );
    case 'nbr':
      return keys === 0 ? (
        <p>
          <strong>Neighbor.</strong> No row in the context points to the query’s row, so its neighbor mask row is empty. The attention output is
          exactly zero and the token passes through this sublayer unchanged <Cite kind="spec" ref="§6.1" />.
        </p>
      ) : (
        <p>
          <strong>Neighbor.</strong> {keys} cells from the {rows} rows whose foreign keys point to the query’s row: its children in the context.
          This is the parent reading its children.
        </p>
      );
    case 'full':
      return (
        <p>
          <strong>Full.</strong> Every non-padding token: {keys} keys. No relational restriction at all.
        </p>
      );
  }
}
