/**
 * Step 6-7 figures.
 *  - PaddingCheck: a padding token exactly as the sampler initialises it (fly.rs:65-85) and its x₀,
 *    computed by the engine's `embedTokens` (model.py:208-224): every term is gated by ¬is_padding.
 *  - ContextMatrix: the whole S × d input matrix X₀ of the live forward pass, split into its column
 *    term and value term (ForwardResult.embedding), as a heatmap with one column per token (wide
 *    screens) or one row per token, grouped under a header per column phrase (phones).
 */
import { memo, useMemo, useState } from 'react';
import { Figure, Heatmap, ScaleLegend, Segmented, StatusBadge, SVG_TYPE, TableSwatch, VectorStrip, fitText, textWidth, type HeatmapCell } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { useResolvedTheme } from '@/styles/theme';
import { embedTokens, textEmbedder } from '@/engine/model';
import { SemType, type Token } from '@/engine/types';
import type { EmbeddingStable } from './live';
import { int, phraseOf, rowRef, useInspected } from './live';
import { norm, slice } from './math';

/** Vecs::new defaults of one batch slot (fly.rs:65-85), as a Token. */
function samplerPaddingToken(pos: number): Token {
  return {
    pos,
    isPadding: true,
    nodeIdx: -1,
    f2pNbrIdxs: [-1, -1, -1, -1, -1],
    tableNameIdx: 0,
    colNameIdx: 0,
    sem: SemType.Number,
    numberValue: 0,
    textIdx: 0,
    datetimeValue: 0,
    booleanValue: 0,
    mask: false,
    isTarget: false,
    isTaskNode: false,
    classValueIdx: -1,
    tableName: '',
    tableType: null,
    colName: '',
    rowPos: -1,
    raw: null,
    depth: -1,
  };
}

export const PaddingCheck = memo(function PaddingCheck({ live }: { live: EmbeddingStable }) {
  const { weights, store, pdb, fr, ctx, gates } = live;
  const realPad = gates.padding.length ? gates.padding[0] : null;
  const res = useMemo(() => {
    if (!weights || !store) return null;
    const d = weights.config.dModel;
    if (realPad !== null && fr) return { x0: slice(fr.embedding.x0, realPad, d), from: 'live' as const };
    const e = embedTokens([samplerPaddingToken(0)], textEmbedder(pdb, store), weights);
    return { x0: e.x0, from: 'engine' as const };
  }, [weights, store, pdb, fr, realPad]);

  const fields: [string, string][] = [
    ['is_padding', 'true'],
    ['node_idxs', '−1'],
    ['f2p_nbr_idxs', '[−1, −1, −1, −1, −1]'],
    ['sem_types', '0'],
    ['col_name_values', '0⃗ (384)'],
    ['number / datetime / boolean_values', '0'],
    ['masks', 'false'],
  ];

  return (
    <Figure
      wide
      title={realPad !== null ? `A padding position of the live context (pos ${realPad})` : 'A padding token, as the sampler initialises it'}
      caption={
        realPad !== null ? (
          <>
            The live context has {int(gates.padding.length)} padding positions; this is the first, read from the forward pass.
          </>
        ) : (
          <>
            The live context fills all {int(ctx.tokens.length)} positions, so it has no padding. This token is built from the sampler’s defaults and
            embedded by the same engine function as the live context.
          </>
        )
      }
    >
      <div className="grid items-start gap-5 md:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <dl className="min-w-0 text-[15px]">
          {fields.map(([k, v]) => (
            <div key={k} className="flex flex-wrap items-baseline justify-between gap-x-3 border-b border-grid/60 py-1 last:border-0">
              <dt className="min-w-0 font-mono break-words text-ink-2">{k}</dt>
              <dd className="num text-ink">{v}</dd>
            </div>
          ))}
        </dl>
        <div className="min-w-0">
          <div className="mb-1.5 text-[15.5px] text-ink-2">x₀ of this position</div>
          {res ? <VectorStrip values={res.x0} columns={32} cellHeight={13} label={undefined} /> : <div className="text-[16px] text-ink-2">Loading…</div>}
          {res && (
            <div className="mt-2">
              {norm(res.x0) === 0 ? (
                <StatusBadge status="good">all {res.x0.length} numbers are exactly 0</StatusBadge>
              ) : (
                <StatusBadge status="critical">‖x₀‖ = {norm(res.x0)}</StatusBadge>
              )}
            </div>
          )}
        </div>
      </div>
    </Figure>
  );
});

type Panel = 'col' | 'val' | 'x0';

interface Group {
  key: string;
  label: string;
  /** Display columns [start, end) in the current order. */
  start: number;
  end: number;
  slot: number;
  hasTarget: boolean;
}

const AXIS_W = 24;
const BAND_MAX = 220;
const NARROW = 600;

export const ContextMatrix = memo(function ContextMatrix({ live }: { live: EmbeddingStable }) {
  const { fr, weights, ctx, slots, assets, target } = live;
  const { inspected, setInspected } = useInspected(live);
  const mode = useResolvedTheme();
  const [panel, setPanel] = useState<Panel>('col');
  const [order, setOrder] = useState<'context' | 'column'>('column');
  const [hover, setHover] = useState<{ pos: number; dim: number; v: number } | null>(null);
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const width = size.width || 900;
  const narrow = width < NARROW;
  const S = ctx.tokens.length;
  const d = weights?.config.dModel ?? 64;

  const perm = useMemo(() => {
    const idx = ctx.tokens.map((t) => t.pos);
    if (order === 'context') return idx;
    return [...idx].sort((a, b) => {
      const ta = ctx.tokens[a];
      const tb = ctx.tokens[b];
      if (ta.isPadding !== tb.isPadding) return ta.isPadding ? 1 : -1;
      return ta.tableName.localeCompare(tb.tableName) || ta.colName.localeCompare(tb.colName) || a - b;
    });
  }, [ctx, order]);

  /** Runs of tokens of the same column phrase in the grouped order (padding is one run). */
  const groups = useMemo<Group[]>(() => {
    if (order !== 'column') return [];
    const out: Group[] = [];
    perm.forEach((p, i) => {
      const t = ctx.tokens[p];
      const key = t.isPadding ? '∅' : `${t.tableName}.${t.colName}`;
      const last = out[out.length - 1];
      if (last && last.key === key) {
        last.end = i + 1;
        last.hasTarget ||= p === target;
      } else {
        out.push({ key, label: t.isPadding ? 'padding' : phraseOf(t), start: i, end: i + 1, slot: t.isPadding ? 0 : slots[t.tableName] ?? 0, hasTarget: p === target });
      }
    });
    return out;
  }, [perm, ctx, order, slots, target]);

  const src = fr ? (panel === 'col' ? fr.embedding.colTerm : panel === 'val' ? fr.embedding.valueTerm : fr.embedding.x0) : null;
  // one colour scale for the three panels, so they can be compared
  const m = useMemo(() => {
    if (!fr) return 1;
    let mx = 0;
    for (const a of [fr.embedding.colTerm, fr.embedding.valueTerm, fr.embedding.x0]) for (const v of a) mx = Math.max(mx, Math.abs(v));
    return mx || 1;
  }, [fr]);

  /** Wide layout: dims are rows, tokens are columns (display order `perm`). */
  const wide = useMemo(() => {
    if (!src || narrow) return null;
    const out = new Float32Array(d * S);
    for (let c = 0; c < S; c++) {
      const p = perm[c];
      for (let r = 0; r < d; r++) out[r * S + c] = src[p * d + r];
    }
    return out;
  }, [src, narrow, perm, d, S]);

  const tokLabel = (p: number) => {
    const t = ctx.tokens[p];
    return t.isPadding ? `#${p} (padding)` : `#${p} ${rowRef(t)}.${t.colName}${t.mask ? ' (masked target)' : ''}`;
  };
  const colLabels = useMemo(() => perm.map(tokLabel), [perm, ctx]); // eslint-disable-line react-hooks/exhaustive-deps
  const colSlots = useMemo(() => perm.map((p) => (ctx.tokens[p].isPadding ? 0 : slots[ctx.tokens[p].tableName] ?? 0)), [perm, ctx, slots]);
  const dimLabels = useMemo(() => Array.from({ length: d }, (_, i) => `dim ${i}`), [d]);
  const selCol = perm.indexOf(inspected);
  const tgtCol = perm.indexOf(target);
  const fmtV = (v: number) => (v < 0 ? `−${Math.abs(v).toFixed(3)}` : v.toFixed(3));
  const panelName = panel === 'col' ? 'column terms' : panel === 'val' ? 'value terms' : 'x0';

  // wide geometry
  const cell = Math.max(2, Math.min(9, Math.floor((width - AXIS_W) / Math.max(1, S))));
  const plotW = S * cell;
  const labelPx = 13;
  const longest = groups.reduce((a, g) => Math.max(a, textWidth(g.label + (g.hasTarget ? ' ?' : ''), labelPx, { weight: g.hasTarget ? 600 : 400 })), 0);
  const bandH = groups.length ? Math.min(BAND_MAX, Math.max(60, Math.ceil(longest) + 22)) : 0;

  // narrow geometry: one row per token, the d dims across
  const nCell = Math.max(3, Math.min(6, Math.floor((width - (order === 'column' ? 2 : 9)) / d)));

  return (
    <Figure
      wide
      title={
        <>
          The matrix that enters block 1: X₀, {S} tokens × {d} dimensions
        </>
      }
      controls={
        <>
          <div className="flex flex-col gap-1">
            <span className="text-[15px] font-medium text-ink-2">Show</span>
            <Segmented
              label="Which part of x₀"
              value={panel}
              onChange={setPanel}
              options={[
                { value: 'col', label: 'column term' },
                { value: 'val', label: 'value term or mᵗ' },
                { value: 'x0', label: 'x₀ (sum)' },
              ]}
            />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[15px] font-medium text-ink-2">Token order</span>
            <Segmented
              label="Token order"
              value={order}
              onChange={setOrder}
              options={[
                { value: 'column', label: 'grouped by column' },
                { value: 'context', label: 'context order' },
              ]}
            />
          </div>
        </>
      }
      caption={
        <>
          {narrow ? 'One row per token, one column per dimension' : 'One column per token, one row per dimension'}, on one shared colour scale for the
          three views. {order === 'column' ? 'Each group is labelled with its column phrase; ' : ''}the coloured strip is the token’s table, and the
          dashed “?” marks the masked target. “Grouped by column” only reorders the display; the model has no positions{' '}
          <span className="whitespace-nowrap">(chapter 7)</span>. Click a token to inspect it in Step 2.
        </>
      }
    >
      {assets.weightsKind === 'init' && <StatusBadge status="warning">Untrained weights: values from a random initialisation</StatusBadge>}
      <div ref={wrapRef} className="w-full min-w-0">
        {!src ? (
          <div className="rounded-lg border border-dashed border-border-strong px-4 py-8 text-center text-[16px] text-ink-2">Loading the forward pass…</div>
        ) : !narrow && wide ? (
          <div>
            <div className="mb-1 text-[14.5px] text-ink-2" style={{ marginLeft: AXIS_W }}>
              tokens →{order === 'column' ? ' grouped by column phrase' : ' in context order'}
            </div>
            <div className="grid" style={{ gridTemplateColumns: `${AXIS_W}px minmax(0,1fr)` }}>
              <div className="relative" aria-hidden="true">
                <span
                  className="absolute left-0 text-[14.5px] whitespace-nowrap text-ink-2"
                  style={{ top: bandH + 18 + 10, writingMode: 'vertical-rl' }}
                >
                  dimension 0 → {d - 1}
                </span>
              </div>
              <div className="min-w-0">
                <svg width={plotW} height={bandH + 18} className="block" role="img" aria-label={order === 'column' ? `Column groups: ${groups.map((g) => `${g.label} (${g.end - g.start})`).join(', ')}` : 'Masked target marker'}>
                  {groups.map((g) => {
                    const x0 = g.start * cell;
                    const x1 = g.end * cell;
                    const w = x1 - x0;
                    const cx = (x0 + x1) / 2;
                    const yb = bandH - 3;
                    const text = fitText(g.label + (g.hasTarget ? ' ?' : ''), bandH - 18, labelPx, { weight: g.hasTarget ? 600 : 400 });
                    return (
                      <g key={g.key}>
                        <title>
                          {g.label}: {g.end - g.start} {g.end - g.start === 1 ? 'token' : 'tokens'}
                          {g.hasTarget ? ', including the masked target' : ''}
                        </title>
                        <rect x={x0} y={0} width={w} height={bandH} fill="transparent" />
                        <path
                          d={`M${x0 + 1},${yb}V${yb - 6}H${x1 - 1}V${yb}`}
                          fill="none"
                          style={{ stroke: g.hasTarget ? 'var(--accent)' : 'var(--axis)' }}
                          strokeWidth={1.25}
                        />
                        {w >= 14 && (
                          <text
                            transform={`translate(${cx + 4.5},${yb - 10}) rotate(-90)`}
                            fontSize={labelPx}
                            style={{ fill: g.hasTarget ? 'var(--accent-text)' : 'var(--ink-2)' }}
                            fontWeight={g.hasTarget ? 600 : 400}
                          >
                            {text}
                          </text>
                        )}
                      </g>
                    );
                  })}
                  {tgtCol >= 0 && (
                    <g>
                      <rect
                        x={Math.max(0.75, Math.min(plotW - 14.75, tgtCol * cell + cell / 2 - 7))}
                        y={bandH + 1}
                        width={14}
                        height={15}
                        rx={3}
                        fill="none"
                        style={{ stroke: 'var(--accent)' }}
                        strokeDasharray="2.5 2"
                        strokeWidth={1.5}
                      />
                      <text x={Math.max(7.75, Math.min(plotW - 7.75, tgtCol * cell + cell / 2))} y={bandH + 13} textAnchor="middle" fontSize={SVG_TYPE.annotation} fontWeight={700} style={{ fill: 'var(--accent-text)' }}>
                        ?
                      </text>
                    </g>
                  )}
                </svg>
                <Heatmap
                  data={wide}
                  rows={d}
                  cols={S}
                  scale="diverging"
                  domain={[-m, m]}
                  colSlots={colSlots}
                  rowLabels={dimLabels}
                  colLabels={colLabels}
                  showLabels={false}
                  rowName=""
                  colName="token"
                  cellSize={cell}
                  highlightCols={selCol >= 0 ? [selCol] : []}
                  onCellClick={(c) => setInspected(perm[c.col])}
                  formatValue={fmtV}
                  title={`Input embedding matrix, ${panelName}`}
                />
              </div>
            </div>
          </div>
        ) : (
          <NarrowMatrix
            src={src}
            d={d}
            cell={nCell}
            groups={order === 'column' ? groups : [{ key: 'all', label: '', start: 0, end: S, slot: 0, hasTarget: tgtCol >= 0 }]}
            grouped={order === 'column'}
            perm={perm}
            slots={slots}
            ctx={ctx}
            inspected={inspected}
            target={target}
            dimLabels={dimLabels}
            tokLabel={tokLabel}
            fmtV={fmtV}
            m={m}
            panelName={panelName}
            onPick={setInspected}
            onHover={setHover}
          />
        )}
        {src && narrow && (
          <div className="mt-3 flex flex-wrap items-center justify-between gap-x-6 gap-y-2 text-[15px] text-ink-2">
            <div className="num min-h-[1.4em] min-w-0" aria-live="polite">
              {hover ? (
                <>
                  <span className="text-ink">{tokLabel(hover.pos)}</span> <span className="text-muted">· dim {hover.dim} →</span>{' '}
                  <strong className="font-semibold text-ink">{fmtV(hover.v)}</strong>
                </>
              ) : (
                <span className="text-muted">Tap a row to inspect that token.</span>
              )}
            </div>
            <ScaleLegend scale="diverging" domain={[-m, m]} mode={mode} />
          </div>
        )}
      </div>
    </Figure>
  );
});

/** Phone layout: one heatmap per column group (rows = its tokens, columns = the d dims), under a header. */
function NarrowMatrix(p: {
  src: Float32Array;
  d: number;
  cell: number;
  groups: Group[];
  grouped: boolean;
  perm: number[];
  slots: Record<string, number>;
  ctx: EmbeddingStable['ctx'];
  inspected: number;
  target: number;
  dimLabels: string[];
  tokLabel: (pos: number) => string;
  fmtV: (v: number) => string;
  m: number;
  panelName: string;
  onPick: (pos: number) => void;
  onHover: (h: { pos: number; dim: number; v: number } | null) => void;
}) {
  const { src, d, cell, groups, grouped, perm, ctx, slots } = p;
  return (
    <div className="grid gap-2.5">
      <div className="text-[14.5px] text-ink-2">tokens ↓ · dimension 0 … {d - 1} →</div>
      {groups.map((g) => {
        const rows = perm.slice(g.start, g.end);
        const data = new Float32Array(rows.length * d);
        rows.forEach((pos, r) => data.set(src.subarray(pos * d, (pos + 1) * d), r * d));
        const sel = rows.indexOf(p.inspected);
        const tgt = rows.indexOf(p.target);
        return (
          <div key={g.key} className="min-w-0">
            {grouped && (
              <div className="mb-1 flex min-w-0 items-center gap-2 text-[14.5px] leading-tight">
                <TableSwatch slot={g.slot} label={undefined} />
                <span className={g.hasTarget ? 'min-w-0 truncate font-semibold text-accent-text' : 'min-w-0 truncate text-ink'} title={g.label}>
                  {g.label}
                </span>
                <span className="num shrink-0 text-muted">{rows.length}</span>
                {g.hasTarget && <span className="target-outline shrink-0 rounded px-1 font-mono text-[13.5px] font-bold text-accent-text">?</span>}
              </div>
            )}
            <Heatmap
              data={data}
              rows={rows.length}
              cols={d}
              scale="diverging"
              domain={[-p.m, p.m]}
              rowSlots={grouped ? undefined : rows.map((pos) => (ctx.tokens[pos].isPadding ? 0 : slots[ctx.tokens[pos].tableName] ?? 0))}
              rowLabels={rows.map(p.tokLabel)}
              colLabels={p.dimLabels}
              showLabels={false}
              rowName="token"
              colName=""
              cellSize={cell}
              highlightRows={[sel, tgt].filter((i) => i >= 0)}
              onHover={(c: HeatmapCell | null) => p.onHover(c ? { pos: rows[c.row], dim: c.col, v: c.value } : null)}
              onCellClick={(c) => p.onPick(rows[c.row])}
              formatValue={p.fmtV}
              showReadout={false}
              showLegend={false}
              title={`Input embedding matrix, ${p.panelName}${grouped ? `, ${g.label}` : ''}`}
            />
          </div>
        );
      })}
    </div>
  );
}
