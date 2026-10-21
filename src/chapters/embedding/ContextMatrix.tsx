/**
 * Step 6-7 figures.
 *  - PaddingCheck: a padding token exactly as the sampler initialises it (fly.rs:65-85) and its x₀,
 *    computed by the engine's `embedTokens` (model.py:208-224): every term is gated by ¬is_padding.
 *  - ContextMatrix: the whole S × d input matrix X₀ of the live forward pass, split into its column
 *    term and value term (ForwardResult.embedding), as a heatmap with one column per token.
 */
import { useMemo, useState } from 'react';
import { Figure, Heatmap, Segmented, StatusBadge, VectorStrip } from '@/components';
import { embedTokens, textEmbedder } from '@/engine/model';
import { SemType, type Token } from '@/engine/types';
import type { EmbeddingLive } from './live';
import { int, rowRef } from './live';
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

export function PaddingCheck({ live }: { live: EmbeddingLive }) {
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
}

type Panel = 'col' | 'val' | 'x0';

export function ContextMatrix({ live }: { live: EmbeddingLive }) {
  const { fr, weights, ctx, slots, inspected, setInspected, assets } = live;
  const [panel, setPanel] = useState<Panel>('col');
  const [order, setOrder] = useState<'context' | 'column'>('column');
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

  const mat = useMemo(() => {
    if (!fr) return null;
    const src = panel === 'col' ? fr.embedding.colTerm : panel === 'val' ? fr.embedding.valueTerm : fr.embedding.x0;
    const out = new Float32Array(d * S);
    for (let c = 0; c < S; c++) {
      const p = perm[c];
      for (let r = 0; r < d; r++) out[r * S + c] = src[p * d + r];
    }
    // one colour scale for the three panels, so they can be compared
    let m = 0;
    for (const a of [fr.embedding.colTerm, fr.embedding.valueTerm, fr.embedding.x0]) for (const v of a) m = Math.max(m, Math.abs(v));
    return { out, m };
  }, [fr, panel, perm, d, S]);

  const colLabels = useMemo(
    () => perm.map((p) => (ctx.tokens[p].isPadding ? `#${p} (padding)` : `#${p} ${rowRef(ctx.tokens[p])}.${ctx.tokens[p].colName}${ctx.tokens[p].mask ? ' (masked)' : ''}`)),
    [perm, ctx],
  );
  const colSlots = useMemo(() => perm.map((p) => (ctx.tokens[p].isPadding ? 0 : slots[ctx.tokens[p].tableName] ?? 0)), [perm, ctx, slots]);
  const rowLabels = useMemo(() => Array.from({ length: d }, (_, i) => `dim ${i}`), [d]);
  const selCol = perm.indexOf(inspected);

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
          One column per token, one row per dimension, one shared colour scale for the three views. The coloured strip on top is the token’s table.
          “Grouped by column” only reorders the display; the model has no positions <span className="whitespace-nowrap">(chapter 7)</span>. Click a
          column to inspect that token in Step 2.
        </>
      }
    >
      {assets.weightsKind === 'init' && <StatusBadge status="warning">Untrained weights: values from a random initialisation</StatusBadge>}
      {!mat ? (
        <div className="rounded-lg border border-dashed border-border-strong px-4 py-8 text-center text-[16px] text-ink-2">Loading the forward pass…</div>
      ) : (
        <Heatmap
          data={mat.out}
          rows={d}
          cols={S}
          scale="diverging"
          domain={[-mat.m, mat.m]}
          colSlots={colSlots}
          rowLabels={rowLabels}
          colLabels={colLabels}
          showLabels={false}
          rowName=""
          colName="token"
          maxCellSize={9}
          minCellSize={2}
          highlightCols={selCol >= 0 ? [selCol] : []}
          onCellClick={(c) => setInspected(perm[c.col])}
          formatValue={(v) => (v < 0 ? `−${Math.abs(v).toFixed(3)}` : v.toFixed(3))}
          title={`Input embedding matrix, ${panel === 'col' ? 'column terms' : panel === 'val' ? 'value terms' : 'x0'}`}
        />
      )}
    </Figure>
  );
}
