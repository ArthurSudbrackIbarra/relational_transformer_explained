/**
 * Section 3: all heads, all layers. The 4 layers × 4 heads attention matrices (AttnCapture.probs,
 * [H·S·S], query-major) of the selected block as small multiples, with the layer's mask drawn as
 * "blocked" cells. Per head: the mean largest weight and how much of its weight it places differently
 * from the mean of the heads (½ Σ_k |p_h − p̄|, averaged over real queries). A color mode shows
 * p_h − p̄ directly. Click a panel to focus it (sets the store's layer + head), click a cell to make
 * its row the query (on touch screens a tap pins the readout instead). A focused large view below has
 * a readout, table identity strips and a table legend.
 */
import { useReducedMotion } from 'motion/react';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { Button, Heatmap, LayerLabel, LAYER_INFO, Segmented, ShapeTag, TableSwatch, cx, divergingCss, sequentialCss, type HeatmapCell } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { ATTN_ORDER, type AttnLayer } from '@/engine/types';
import { TOKENS } from '@/styles/palette';
import { useReadingMode, useResolvedTheme } from '@/styles/theme';
import { axis } from '../shapes';
import { SelectionBar } from './Controls';
import { int, pctw } from './utils';
import type { AttnLive } from './useAttnLive';

type ScaleMode = 'sqrt' | 'linear' | 'diff';
type OrderMode = 'context' | 'grouped';

/**
 * Display order of the rows and columns. 'grouped' sorts the real tokens by table, then by column, then by
 * context position (padding stays last). It is a pure permutation of queries and keys: RT has no positional
 * encoding, so permuting the tokens permutes the outputs the same way (spec §6.5) and nothing is recomputed.
 */
function displayOrder(tokens: AttnLive['ctx']['tokens'], groups: AttnLive['keys']['groups'], mode: OrderMode): { perm: Int32Array; inv: Int32Array } | null {
  if (mode === 'context') return null;
  const S = tokens.length;
  const gi = new Map(groups.map((g, i) => [g.name, i]));
  const colRank = new Map<number, number>();
  for (const t of tokens) if (!t.isPadding && !colRank.has(t.colNameIdx)) colRank.set(t.colNameIdx, colRank.size);
  const real = tokens.filter((t) => !t.isPadding).map((t) => t.pos);
  real.sort((a, b) => {
    const ta = tokens[a];
    const tb = tokens[b];
    return (gi.get(ta.tableName) ?? 0) - (gi.get(tb.tableName) ?? 0) || (colRank.get(ta.colNameIdx) ?? 0) - (colRank.get(tb.colNameIdx) ?? 0) || a - b;
  });
  const perm = new Int32Array(S);
  let i = 0;
  for (const p of real) perm[i++] = p;
  for (const t of tokens) if (t.isPadding) perm[i++] = t.pos;
  const inv = new Int32Array(S);
  for (let j = 0; j < S; j++) inv[perm[j]] = j;
  return { perm, inv };
}

/** Square S×S matrix with rows and columns permuted: out[i, j] = m[perm[i], perm[j]]. */
function permute2<T extends Float32Array | Uint8Array>(m: T, perm: Int32Array, S: number, make: (n: number) => T): T {
  const out = make(S * S);
  for (let i = 0; i < S; i++) {
    const src = perm[i] * S;
    const dst = i * S;
    for (let j = 0; j < S; j++) out[dst + j] = m[src + perm[j]];
  }
  return out;
}

/** Blocked cells in the difference mode: a quiet neutral, distinct from the near-white (light) or near-surface (dark) zero. */
const DIFF_BLOCKED = { light: '#d6d5cd', dark: '#121211' } as const;

interface CellRef {
  layer: AttnLayer;
  head: number;
  q: number;
  k: number;
}

const isCoarse = () => typeof window !== 'undefined' && !!window.matchMedia?.('(pointer: coarse)').matches;

/** Signed percentage points, e.g. "+3.1 pp". */
const pp = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(Math.abs(v) >= 0.1 ? 1 : 2)} pp`;

export function HeadGrid({ live }: { live: AttnLive }) {
  const { fr, masks, block, layer, head, query, numReal, S, H, setLayer, setHead, setQuery, assets, keys } = live;
  const [scale, setScale] = useState<ScaleMode>('sqrt');
  const [order, setOrder] = useState<OrderMode>('context');
  const [hover, setHover] = useState<CellRef | null>(null);
  const [pinned, setPinned] = useState<CellRef | null>(null);
  const [ref, size] = useElementSize<HTMLDivElement>();
  const focusRef = useRef<HTMLDivElement | null>(null);
  const reduce = useReducedMotion();
  const mode = useResolvedTheme();
  const blockedFill = DIFF_BLOCKED[mode];
  const ord = useMemo(() => displayOrder(live.ctx.tokens, keys.groups, order), [live.ctx.tokens, keys.groups, order]);
  const toPos = useCallback((i: number) => (ord ? ord.perm[i] : i), [ord]);
  const toRow = (pos: number) => (ord ? ord.inv[pos] : pos);
  const W = Math.max(280, size.width || 900);
  const narrow = W < 640;
  const LABEL_W = narrow ? 0 : 168;
  const GAP = narrow ? 10 : 14;
  const perRow = narrow ? 2 : H;
  const panelW = Math.floor((W - LABEL_W - GAP * perRow) / perRow);
  const cell = Math.max(0.5, panelW / Math.max(1, numReal));

  const blocked = useMemo(() => {
    const out = {} as Record<AttnLayer, Uint8Array>;
    for (const l of ATTN_ORDER) {
      const m = masks[l];
      const b = new Uint8Array(S * S);
      for (let i = 0; i < S * S; i++) b[i] = m[i] ? 0 : 1;
      out[l] = ord ? permute2(b, ord.perm, S, (n) => new Uint8Array(n)) : b;
    }
    return out;
  }, [masks, S, ord]);

  /** Mean over heads, per layer (for the difference mode and the per-head Δ). */
  const means = useMemo(() => {
    if (!fr) return null;
    const blk = fr.blocks[Math.min(block, fr.blocks.length - 1)];
    const out = {} as Record<AttnLayer, Float32Array | null>;
    for (const l of ATTN_ORDER) {
      const cap = blk.attn[l];
      if (cap.skipped || cap.probs.length < H * S * S) {
        out[l] = null;
        continue;
      }
      const m = new Float32Array(S * S);
      for (let h = 0; h < H; h++) {
        const o = h * S * S;
        for (let i = 0; i < S * S; i++) m[i] += cap.probs[o + i] / H;
      }
      out[l] = m;
    }
    return out;
  }, [fr, block, H, S]);

  const panels = useMemo(() => {
    if (!fr || !means) return null;
    const blk = fr.blocks[Math.min(block, fr.blocks.length - 1)];
    const out = {} as Record<AttnLayer, (Float32Array | null)[]>;
    const diffMax = {} as Record<AttnLayer, number>;
    for (const l of ATTN_ORDER) {
      const cap = blk.attn[l];
      const mean = means[l];
      let dm = 0;
      out[l] = Array.from({ length: H }, (_, h) => {
        if (!mean || cap.skipped || cap.probs.length < H * S * S) return null;
        const p = cap.probs.subarray(h * S * S, (h + 1) * S * S);
        let v: Float32Array;
        if (scale === 'sqrt') v = Float32Array.from(p, Math.sqrt);
        else if (scale === 'linear') v = p;
        else {
          v = new Float32Array(S * S);
          for (let i = 0; i < S * S; i++) {
            v[i] = p[i] - mean[i];
            const a = Math.abs(v[i]);
            if (a > dm) dm = a;
          }
        }
        return ord ? permute2(v, ord.perm, S, (n) => new Float32Array(n)) : v;
      });
      diffMax[l] = dm || 1e-6;
    }
    return { data: out, diffMax };
  }, [fr, means, block, H, S, scale, ord]);

  const stats = useLayerStats(live);
  const headStats = useHeadStats(live, means);

  const scrollToFocus = useCallback(() => {
    requestAnimationFrame(() => focusRef.current?.scrollIntoView({ block: 'nearest', behavior: reduce ? 'auto' : 'smooth' }));
  }, [reduce]);

  // Stable handlers for the memoised panels (latest store setters via a ref).
  const act = useRef({ setLayer, setHead, setQuery, scrollToFocus });
  act.current = { setLayer, setHead, setQuery, scrollToFocus };
  const onFocusPanel = useCallback((l: AttnLayer, h: number) => {
    act.current.setLayer(l);
    act.current.setHead(h);
    setPinned(null);
    act.current.scrollToFocus();
  }, []);
  const onHoverCell = useCallback((l: AttnLayer, h: number, c: HeatmapCell | null) => setHover(c ? { layer: l, head: h, q: toPos(c.row), k: toPos(c.col) } : null), [toPos]);
  const onPickCell = useCallback(
    (l: AttnLayer, h: number, c: HeatmapCell) => {
      if (isCoarse()) {
        setPinned({ layer: l, head: h, q: toPos(c.row), k: toPos(c.col) });
        return;
      }
      act.current.setLayer(l);
      act.current.setHead(h);
      act.current.setQuery(toPos(c.row));
      act.current.scrollToFocus();
    },
    [toPos],
  );

  const shown = hover ?? pinned;
  const readout = (() => {
    const h = shown;
    if (!h || !fr) return null;
    const cap = fr.blocks[Math.min(block, fr.blocks.length - 1)].attn[h.layer];
    const allowed = !!masks[h.layer][h.q * S + h.k];
    const p = cap.probs[(h.head * S + h.q) * S + h.k];
    const mean = means?.[h.layer]?.[h.q * S + h.k] ?? 0;
    return (
      <>
        <span className="text-muted">
          {LAYER_INFO[h.layer].name} · head {h.head + 1} · query{' '}
        </span>
        <span className="num text-ink">
          #{h.q} {live.labels[h.q]}
        </span>
        <span className="text-muted"> → key </span>
        <span className="num text-ink">
          #{h.k} {live.labels[h.k]}
        </span>
        <span className="text-muted"> : </span>
        <strong className="num font-semibold text-ink">{allowed ? pctw(p) : 'blocked by the mask'}</strong>
        {allowed && <span className="num text-muted"> (mean of heads {pctw(mean)})</span>}
      </>
    );
  })();

  const focusData = panels?.data[layer]?.[head] ?? null;
  const rowsShown = narrow ? [layer] : ATTN_ORDER;

  return (
    <div className="wide my-8">
      <div className="card px-4 py-4 sm:px-5 sm:py-5">
        <div className="mb-3 text-[17px] font-semibold text-ink">
          Attention weights of block {block + 1}: every layer, every head{' '}
          <span className="font-normal">
            (each{' '}
            <ShapeTag
              rows={numReal}
              cols={numReal}
              shape={{
                rows: axis.queries({ shown: numReal, L: live.ctx.params.L, real: numReal, order }),
                cols: axis.keys({ shown: numReal, L: live.ctx.params.L, real: numReal, order }),
                note: `${ATTN_ORDER.length} layers × ${H} heads = ${ATTN_ORDER.length * H} matrices of this shape in every block; each head of each layer has its own.`,
              }}
            />
            )
          </span>
        </div>
        <div className="mb-2 flex flex-wrap items-end gap-x-4 gap-y-3">
          <SelectionBar live={live} showLayer={false} showHead={false} showQuery={false} />
          <Segmented
            label="Color"
            value={scale}
            onChange={setScale}
            options={[
              { value: 'sqrt', label: '√weight', title: 'Color ∝ √weight: makes small weights visible' },
              { value: 'linear', label: 'weight', title: 'Color ∝ weight' },
              { value: 'diff', label: 'head − mean of heads', title: 'Diverging: where this head puts more (red) or less (blue) weight than the mean of the heads' },
            ]}
          />
          <Segmented
            label="Order"
            value={order}
            onChange={setOrder}
            options={[
              { value: 'context' as OrderMode, label: 'context order', title: 'Rows and columns in the order of the context window' },
              { value: 'grouped' as OrderMode, label: 'by table & column', title: 'Rows and columns grouped by table, then by column (a pure permutation)' },
            ]}
          />
          {narrow && (
            <Segmented
              label="Layer shown"
              value={layer}
              onChange={setLayer}
              options={ATTN_ORDER.map((l) => ({ value: l, label: <LayerLabel layer={l} size={13} />, title: `${LAYER_INFO[l].name} attention` }))}
            />
          )}
        </div>
        <p className="!mb-4 text-[14.5px] text-muted">
          Block, layer, head and query are shared across this chapter: a choice here also changes the step-through above.
        </p>
        <div ref={ref} className="min-w-0">
          {!panels ? (
            <p className="text-[16.5px] text-ink-2">{assets.status === 'error' ? `The forward pass is unavailable: ${assets.error}` : 'Running the forward pass…'}</p>
          ) : (
            <div className="grid gap-y-5">
              {rowsShown.map((l) => (
                <div key={l} className={cx('grid items-start gap-y-2', !narrow && 'grid-cols-[168px_minmax(0,1fr)]')}>
                  <div className="pt-0.5 pr-2">
                    <div className="text-[16px]">
                      <LayerLabel layer={l} />
                    </div>
                    {stats && (
                      <div className={cx('mt-1 text-[14px] leading-snug text-muted', narrow && 'flex flex-wrap gap-x-3')}>
                        <span className="block">
                          <span className="num text-ink-2">{stats[l].keys.toFixed(stats[l].keys < 10 ? 1 : 0)}</span> keys per query
                        </span>
                        <span className="block">
                          top weight <span className="num text-ink-2">{pctw(stats[l].top)}</span>
                        </span>
                        {stats[l].empty > 0 && (
                          <span className="block">
                            <span className="num text-ink-2">{stats[l].empty}</span> empty rows
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                  <div className="grid gap-x-[var(--g)] gap-y-3" style={{ gridTemplateColumns: `repeat(${perRow}, minmax(0, 1fr))`, ['--g' as string]: `${GAP}px` }}>
                    {Array.from({ length: H }, (_, h) => (
                      <SmallPanel
                        key={h}
                        layer={l}
                        head={h}
                        block={block}
                        data={panels.data[l][h]}
                        blocked={blocked[l]}
                        diverging={scale === 'diff'}
                        diffMax={panels.diffMax[l]}
                        blockedFill={blockedFill}
                        S={S}
                        numReal={numReal}
                        cell={cell}
                        panelW={panelW}
                        query={toRow(query)}
                        on={l === layer && h === head}
                        stat={headStats?.[l][h] ?? null}
                        onFocusPanel={onFocusPanel}
                        onHoverCell={onHoverCell}
                        onPickCell={onPickCell}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="mt-4 grid gap-2 border-t border-grid pt-2 text-[14.5px] text-ink-2">
          <div className="min-h-[1.4em]" aria-live="polite">
            {readout ?? (
              <span className="text-muted">
                Hover a cell to read one weight; click it to make its row the query (on a touch screen, tap to read it). Rows are queries, columns keys.
              </span>
            )}
          </div>
          {pinned && !hover && (
            <div className="flex flex-wrap gap-2">
              <Button
                size="sm"
                variant="secondary"
                onClick={() => {
                  setLayer(pinned.layer);
                  setHead(pinned.head);
                  setQuery(pinned.q);
                  setPinned(null);
                  scrollToFocus();
                }}
              >
                Make #{pinned.q} the query
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setPinned(null)}>
                Clear
              </Button>
            </div>
          )}
          <GridLegend scale={scale} />
        </div>
      </div>

      {panels && focusData && (
        <div ref={focusRef} className="card mt-5 scroll-mt-24 px-4 py-4 sm:px-5 sm:py-5">
          <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <div className="text-[17px] font-semibold text-ink">
              Focused: block {block + 1} · <LayerLabel layer={layer} /> · head {head + 1}
            </div>
            <div className="text-[14.5px] text-muted">
              the query’s row #{query} is outlined; gray cells are blocked by the {LAYER_INFO[layer].name.toLowerCase()} mask
            </div>
          </div>
          <FocusMap live={live} data={focusData} blocked={blocked[layer]} scale={scale} diffMax={panels.diffMax[layer]} ord={ord} />
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[14px] text-ink-2">
            <span className="text-muted">Table of each token (side strips):</span>
            {keys.groups.map((g) => (
              <TableSwatch key={g.name} slot={g.slot} label={g.name} size={9} />
            ))}
          </div>
        </div>
      )}
      <p className="mx-auto mt-3 max-w-[var(--prose-max)] text-[16px] leading-snug text-ink-2">
        Each matrix is one head of one layer. Row q holds the weights that query q spreads over the keys, and it sums to 1 over the allowed keys,
        or is all zero when no key is allowed. Under each head, <em>top</em> is its largest weight averaged over the real queries, and <em>Δ</em>{' '}
        is the share of its weight that it places differently from the mean of the {H} heads (½ Σ<sub>k</sub> |p<sub>head</sub> − p<sub>mean</sub>|,
        averaged over queries). With √ scaling, an even row over {int(numReal)} keys ({pctw(1 / Math.max(1, numReal))} each) still shows as a pale
        wash. The difference mode shows where one head departs from the others.
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

interface SmallPanelProps {
  layer: AttnLayer;
  head: number;
  block: number;
  data: Float32Array | null;
  blocked: Uint8Array;
  diverging: boolean;
  diffMax: number;
  blockedFill: string;
  S: number;
  numReal: number;
  cell: number;
  panelW: number;
  query: number;
  on: boolean;
  stat: { top: number; delta: number } | null;
  onFocusPanel: (l: AttnLayer, h: number) => void;
  onHoverCell: (l: AttnLayer, h: number, c: HeatmapCell | null) => void;
  onPickCell: (l: AttnLayer, h: number, c: HeatmapCell) => void;
}

/** One head's matrix. Memoised: a query change only moves the CSS row outline, it never repaints the canvas. */
const SmallPanel = memo(function SmallPanel(p: SmallPanelProps) {
  const { layer: l, head: h } = p;
  return (
    <div className="min-w-0">
      <button
        type="button"
        onClick={() => p.onFocusPanel(l, h)}
        aria-pressed={p.on}
        className={cx(
          'mb-1 flex w-full flex-wrap items-baseline justify-between gap-x-2 rounded-md px-1 text-left text-[14px] transition-colors',
          p.on ? 'font-semibold text-accent-text' : 'text-ink-2 hover:text-ink',
        )}
      >
        <span>
          head {h + 1}
          {p.on ? ' · focused' : ''}
        </span>
        {p.stat && (
          <span className="num font-normal text-muted" title="top: mean largest weight · Δ: share of weight placed differently from the mean of heads">
            <span className="whitespace-nowrap">top {pctw(p.stat.top)}</span> · <span className="whitespace-nowrap">Δ {pctw(p.stat.delta)}</span>
          </span>
        )}
      </button>
      <div className={cx('relative inline-block rounded-[3px] p-[2px]', p.on ? 'ring-2 ring-accent' : 'ring-1 ring-border')}>
        {p.data ? (
          <>
            <Heatmap
              data={p.data}
              rows={p.S}
              cols={p.S}
              crop={{ rows: p.numReal, cols: p.numReal }}
              blocked={p.blocked}
              scale={p.diverging ? 'diverging' : 'sequential'}
              domain={p.diverging ? [-p.diffMax, p.diffMax] : [0, 1]}
              blockedFill={p.diverging ? p.blockedFill : undefined}
              cellSize={p.cell}
              showReadout={false}
              showLegend={false}
              title={`${LAYER_INFO[l].name} attention, head ${h + 1}, block ${p.block + 1}`}
              shape={null}
              onHover={(c) => p.onHoverCell(l, h, c)}
              onCellClick={(c) => p.onPickCell(l, h, c)}
            />
            {p.query < p.numReal && (
              <div
                aria-hidden="true"
                className="pointer-events-none absolute"
                style={{ left: 1, top: 2 + p.query * p.cell - 1, width: p.numReal * p.cell + 2, height: Math.max(2, p.cell) + 2, border: '1px solid var(--accent)' }}
              />
            )}
          </>
        ) : (
          <div className="flex items-center justify-center text-[14px] text-muted" style={{ width: p.panelW, height: p.panelW }}>
            switched off
          </div>
        )}
      </div>
    </div>
  );
});

function GridLegend({ scale }: { scale: ScaleMode }) {
  const mode = useResolvedTheme();
  useReadingMode(); // TOKENS follow reading mode
  const blockedBg = scale === 'diff' ? DIFF_BLOCKED[mode] : TOKENS[mode].surface2;
  const stops = Array.from({ length: 11 }, (_, i) => i / 10);
  const grad =
    scale === 'diff' ? stops.map((t) => divergingCss(t * 2 - 1, 1, mode)).join(', ') : stops.map((t) => sequentialCss(t, mode)).join(', ');
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[14px] text-ink-2">
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block h-3.5 w-3.5 rounded-[3px] border border-border" style={{ background: blockedBg }} /> blocked by the mask
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="num">{scale === 'diff' ? 'less' : '0'}</span>
        <span className="inline-block h-2.5 w-24 rounded-full" style={{ background: `linear-gradient(90deg, ${grad})` }} aria-hidden="true" />
        <span className="num">{scale === 'diff' ? 'more weight than the mean of heads' : '1'}</span>
      </span>
      <span className="text-muted">
        {scale === 'sqrt' ? 'color ∝ √weight' : scale === 'linear' ? 'color ∝ weight' : 'color ∝ p_head − p_mean, scaled per layer'}
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block h-2.5 w-5 border border-accent" /> the query’s row
      </span>
    </div>
  );
}

function FocusMap({
  live,
  data,
  blocked,
  scale,
  diffMax,
  ord,
}: {
  live: AttnLive;
  data: Float32Array;
  blocked: Uint8Array;
  scale: ScaleMode;
  diffMax: number;
  ord: { perm: Int32Array; inv: Int32Array } | null;
}) {
  const { S, numReal, query, setQuery, fr, block, layer, head, masks } = live;
  const labels = useMemo(() => (ord ? Array.from(ord.perm, (p) => live.labels[p]) : live.labels), [ord, live.labels]);
  const tokenSlots = useMemo(() => (ord ? Array.from(ord.perm, (p) => live.tokenSlots[p]) : live.tokenSlots), [ord, live.tokenSlots]);
  const toPos = (i: number) => (ord ? ord.perm[i] : i);
  const [ref, size] = useElementSize<HTMLDivElement>();
  const mode = useResolvedTheme();
  const [hover, setHover] = useState<HeatmapCell | null>(null);
  const [pinned, setPinned] = useState<HeatmapCell | null>(null);
  const W = Math.max(260, size.width || 800);
  const cell = Math.max(1, Math.min(8, Math.floor((W - 20) / Math.max(1, numReal))));
  const c0 = hover ?? pinned;
  // Readout in context positions (the displayed row / column are mapped back through the order).
  const c = c0 ? { ...c0, row: toPos(c0.row), col: toPos(c0.col) } : null;
  const cap = fr ? fr.blocks[Math.min(block, fr.blocks.length - 1)].attn[layer] : null;
  const p = c && cap ? cap.probs[(head * S + c.row) * S + c.col] : 0;
  const allowed = c ? !!masks[layer][c.row * S + c.col] : false;
  return (
    <div ref={ref} className="min-w-0">
      <Heatmap
        data={data}
        rows={S}
        cols={S}
        crop={{ rows: numReal, cols: numReal }}
        blocked={blocked}
        scale={scale === 'diff' ? 'diverging' : 'sequential'}
        domain={scale === 'diff' ? [-diffMax, diffMax] : [0, 1]}
        blockedFill={scale === 'diff' ? DIFF_BLOCKED[mode] : undefined}
        cellSize={cell}
        rowSlots={tokenSlots}
        colSlots={tokenSlots}
        rowLabels={labels}
        colLabels={labels}
        highlightRows={[ord ? ord.inv[query] : query]}
        rowName="query"
        colName="key"
        title="Focused attention matrix"
        shape={{
          rows: axis.queries({ shown: numReal, L: live.ctx.params.L, real: numReal, order: ord ? 'grouped' : 'context' }),
          cols: axis.keys({ shown: numReal, L: live.ctx.params.L, real: numReal, order: ord ? 'grouped' : 'context' }),
        }}
        showReadout={false}
        showLegend={false}
        onHover={setHover}
        onCellClick={(cl) => {
          if (isCoarse()) setPinned(cl);
          else setQuery(toPos(cl.row));
        }}
      />
      <div className="mt-2 flex min-h-[1.4em] flex-wrap items-center gap-x-3 gap-y-1 text-[15px] text-ink-2" aria-live="polite">
        {c ? (
          <>
            <span className="num">
              <span className="text-muted">query </span>
              <span className="text-ink">
                #{c.row} {live.labels[c.row]}
              </span>
              <span className="text-muted"> · key </span>
              <span className="text-ink">
                #{c.col} {live.labels[c.col]}
              </span>
              <span className="text-muted"> → </span>
              <strong className="font-semibold text-ink">{allowed ? pctw(p) : 'blocked by the mask'}</strong>
              {allowed && scale === 'diff' && <span className="text-muted"> ({pp(c.value)} vs the mean of heads)</span>}
            </span>
            {pinned && !hover && toPos(pinned.row) !== query && (
              <Button size="sm" variant="secondary" onClick={() => (setQuery(toPos(pinned.row)), setPinned(null))}>
                Make #{toPos(pinned.row)} the query
              </Button>
            )}
          </>
        ) : (
          <span className="text-muted">Hover or focus a cell to read its weight; click it to make its row the query.</span>
        )}
      </div>
    </div>
  );
}

/** Per-layer summary for one block over real queries with ≥1 allowed key: mean allowed keys, mean largest weight (mean over heads). */
export function useLayerStats(live: AttnLive) {
  const { fr, block, masks, numReal, S, H } = live;
  return useMemo(() => {
    if (!fr) return null;
    const blk = fr.blocks[Math.min(block, fr.blocks.length - 1)];
    const out = {} as Record<AttnLayer, { keys: number; top: number; empty: number }>;
    for (const l of ATTN_ORDER) {
      const cap = blk.attn[l];
      let rows = 0;
      let keys = 0;
      let top = 0;
      let empty = 0;
      for (let q = 0; q < numReal; q++) {
        let n = 0;
        for (let k = 0; k < S; k++) n += masks[l][q * S + k];
        if (n === 0) {
          empty++;
          continue;
        }
        rows++;
        keys += n;
        if (!cap.skipped && cap.probs.length >= H * S * S) {
          let t = 0;
          for (let h = 0; h < H; h++) {
            let mx = 0;
            const o = (h * S + q) * S;
            for (let k = 0; k < S; k++) if (cap.probs[o + k] > mx) mx = cap.probs[o + k];
            t += mx;
          }
          top += t / H;
        }
      }
      out[l] = { keys: rows ? keys / rows : 0, top: rows ? top / rows : 0, empty };
    }
    return out;
  }, [fr, block, masks, numReal, S, H]);
}

/**
 * Per head of each layer, over real queries with ≥1 allowed key: the mean largest weight, and
 * Δ = mean over queries of ½ Σ_k |p_head − p_mean| (the share of the row's weight placed differently
 * from the mean of the heads; 0 = identical to the mean). Display statistics of the capture only.
 */
export function useHeadStats(live: AttnLive, means: Record<AttnLayer, Float32Array | null> | null) {
  const { fr, block, masks, numReal, S, H } = live;
  return useMemo(() => {
    if (!fr || !means) return null;
    const blk = fr.blocks[Math.min(block, fr.blocks.length - 1)];
    const out = {} as Record<AttnLayer, { top: number; delta: number }[]>;
    for (const l of ATTN_ORDER) {
      const cap = blk.attn[l];
      const mean = means[l];
      out[l] = Array.from({ length: H }, () => ({ top: 0, delta: 0 }));
      if (!mean || cap.skipped || cap.probs.length < H * S * S) continue;
      let rows = 0;
      for (let q = 0; q < numReal; q++) {
        let n = 0;
        for (let k = 0; k < S; k++) n += masks[l][q * S + k];
        if (n === 0) continue;
        rows++;
        for (let h = 0; h < H; h++) {
          const o = (h * S + q) * S;
          let mx = 0;
          let tv = 0;
          for (let k = 0; k < S; k++) {
            const v = cap.probs[o + k];
            if (v > mx) mx = v;
            tv += Math.abs(v - mean[q * S + k]);
          }
          out[l][h].top += mx;
          out[l][h].delta += tv / 2;
        }
      }
      if (rows) for (const s of out[l]) (s.top /= rows), (s.delta /= rows);
    }
    return out;
  }, [fr, block, masks, numReal, S, H, means]);
}
