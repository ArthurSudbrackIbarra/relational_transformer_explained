/**
 * Section 4: where does the masked target look? For the target query of the live context:
 *  - per layer, the keys with the largest weights (one head, or the mean over heads), linked to
 *  - a per-table view of the context filled by those weights (paper Fig. 1 style), and
 *  - for the column layer, how the weight splits between the seed entity's own past labels and
 *    other entities' labels, in every block (in-context labels: useInContextLabels, spec §2.5).
 * Everything is read from the capture; the reading of it is labelled as interpretation.
 */
import { Lightbulb, Search } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Button, Cite, LayerLabel, LAYER_INFO, Segmented, TableSwatch, cx, sequentialCss } from '@/components';
import { ATTN_ORDER, type AttnLayer } from '@/engine/types';
import { useSim } from '@/state/store';
import { useResolvedTheme } from '@/styles/theme';
import { AttnTables } from './AttnTables';
import { SelectionBar } from './Controls';
import { useOtherLabelSeeds } from './otherLabelSeeds';
import { int, pctw, rowName, singular, topK, valueOf } from './utils';
import type { AttnLive } from './useAttnLive';

type HeadSel = 'mean' | number;

/** Weights of one query row: one head, or the mean over heads (rows of AttnCapture.probs). */
export function rowWeights(live: AttnLive, block: number, layer: AttnLayer, q: number, sel: HeadSel): Float32Array | null {
  const { fr, H, S } = live;
  if (!fr) return null;
  const cap = fr.blocks[Math.min(block, fr.blocks.length - 1)].attn[layer];
  if (cap.skipped || cap.probs.length < H * S * S) return null;
  const out = new Float32Array(S);
  if (sel === 'mean') {
    for (let h = 0; h < H; h++) {
      const o = (h * S + q) * S;
      for (let k = 0; k < S; k++) out[k] += cap.probs[o + k] / H;
    }
  } else {
    out.set(cap.probs.subarray((sel * S + q) * S, (sel * S + q + 1) * S));
  }
  return out;
}

const TOP = 5;
/** Two weights computed from identical inputs are bit-identical; allow float noise only. */
const tied = (a: number, b: number) => Math.abs(a - b) <= 1e-7 * Math.max(1, Math.abs(a));

export function TargetLook({ live }: { live: AttnLive }) {
  const { fr, masks, block, target, H, S, ctx, slots, labels, assets } = live;
  const [headSel, setHeadSel] = useState<HeadSel>('mean');
  const [active, setActive] = useState<AttnLayer>('col');
  const [hoverKey, setHoverKey] = useState<number | null>(null);

  const weights = useMemo(() => {
    const out = {} as Record<AttnLayer, Float32Array | null>;
    for (const l of ATTN_ORDER) out[l] = rowWeights(live, block, l, target, headSel);
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fr, block, target, headSel, S, H]);
  const allowed = useMemo(() => {
    const out = {} as Record<AttnLayer, Uint8Array>;
    for (const l of ATTN_ORDER) out[l] = masks[l].subarray(target * S, (target + 1) * S);
    return out;
  }, [masks, target, S]);
  // One bar scale for all four cards, so a 1% weight never looks as decisive as a 30% one.
  const scaleMax = useMemo(() => {
    let m = 0;
    for (const l of ATTN_ORDER) {
      const w = weights[l];
      if (!w) continue;
      for (let k = 0; k < S; k++) if (allowed[l][k] && w[k] > m) m = w[k];
    }
    return m > 0 ? m : 1;
  }, [weights, allowed, S]);

  const hovered = hoverKey !== null ? ctx.tokens[hoverKey] : null;
  const activeW = weights[active];

  return (
    <div className="wide my-8">
      <div className="card px-4 py-4 sm:px-5 sm:py-5">
        <div className="mb-3 flex flex-wrap items-center gap-2 text-[17px] font-semibold text-ink">
          <span>
            Where the masked target #{target} looks, in block {block + 1}
          </span>
          <InterpretationBadge />
        </div>
        <div className="mb-2 flex flex-wrap items-end gap-x-4 gap-y-3">
          <SelectionBar live={live} showLayer={false} showHead={false} showQuery={false} />
          <Segmented
            label="Heads"
            value={headSel}
            onChange={setHeadSel}
            options={[{ value: 'mean' as HeadSel, label: 'mean of heads' }, ...Array.from({ length: H }, (_, h) => ({ value: h as HeadSel, label: `head ${h + 1}` }))]}
          />
        </div>
        <p className="!mb-4 text-[14.5px] text-muted">The block is shared with the figures above; the head choice here is local to this card.</p>
        {!fr ? (
          <p className="text-[16.5px] text-ink-2">{assets.status === 'error' ? `The forward pass is unavailable: ${assets.error}` : 'Running the forward pass…'}</p>
        ) : (
          <>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {ATTN_ORDER.map((l) => (
                <LayerCard
                  key={l}
                  live={live}
                  layer={l}
                  w={weights[l]}
                  allowed={allowed[l]}
                  scaleMax={scaleMax}
                  active={active === l}
                  hoverKey={hoverKey}
                  onActivate={() => setActive(l)}
                  onHoverKey={(k) => {
                    setHoverKey(k);
                    if (k !== null) setActive(l);
                  }}
                />
              ))}
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[14px] text-ink-2">
              <span className="inline-flex items-center gap-1.5">
                <span className="inline-block h-[5px] w-8 rounded-full bg-seq-500" /> weight (one scale for all four cards)
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="inline-block h-3 w-[2px] rounded-full bg-ink" /> even split: 1 / allowed keys
              </span>
            </div>

            <div className="mt-6 grid gap-2">
              <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <div className="text-[15.5px] font-semibold text-ink">
                  The context, filled by the target’s <LayerLabel layer={active} size={13} /> weights
                </div>
                <Segmented
                  label="Layer shown in the context view"
                  value={active}
                  onChange={setActive}
                  options={ATTN_ORDER.map((l) => ({ value: l, label: LAYER_INFO[l].name }))}
                />
              </div>
              {activeW ? (
                <AttnTables
                  tokens={ctx.tokens}
                  slots={slots}
                  weights={activeW}
                  allowed={allowed[active]}
                  query={target}
                  hoverKey={hoverKey}
                  onHover={setHoverKey}
                  label={`The context as small tables; each cell is filled by the weight the masked target gives it in the ${LAYER_INFO[active].name} layer. Focus it and use the arrow keys to read cells.`}
                />
              ) : (
                <p className="text-[15.5px] text-ink-2">This layer is switched off by the playground’s ablation.</p>
              )}
              <div className="min-h-[1.4em] text-[14.5px] text-ink-2" aria-live="polite">
                {hovered ? (
                  <>
                    <span className="num text-ink">
                      #{hovered.pos} {labels[hovered.pos]}
                    </span>{' '}
                    <span className="num text-muted">({rowName(hovered)})</span> = <span className="num">{hovered.isTarget ? '?' : valueOf(hovered)}</span>
                    <span className="text-muted"> · weight </span>
                    <strong className="num font-semibold text-ink">{allowed[active][hovered.pos] ? pctw(activeW?.[hovered.pos] ?? 0) : 'blocked by the mask'}</strong>
                  </>
                ) : (
                  <span className="text-muted">
                    One square per cell token, grouped by table and row. Filled squares are keys the target may read (darker = more weight, colour ∝ √weight);
                    hollow squares are blocked. The dashed square is the target itself. Hover a square, or focus the view and use the arrow keys.
                  </span>
                )}
              </div>
              <Legend />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/** Marks a figure whose reading goes beyond what the paper or code states (BUILD_PLAN §0). */
export function InterpretationBadge() {
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full border border-border bg-surface-2 px-2.5 py-0.5 text-[14px] font-medium text-ink-2"
      title="The weights are real outputs of the mini model; what they mean is our reading, not a claim of the paper."
    >
      <Lightbulb aria-hidden="true" size={14} /> interpretation
    </span>
  );
}

function Legend() {
  const mode = useResolvedTheme();
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[14px] text-ink-2">
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block h-2.5 w-14 rounded-full" style={{ background: `linear-gradient(90deg, ${sequentialCss(0.08, mode)}, ${sequentialCss(1, mode)})` }} />
        weight (0 → the largest in this view)
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block h-2.5 w-2.5 rounded-[2px] border border-grid" /> blocked by the mask
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="inline-block h-2.5 w-2.5 rounded-[2px] border-[1.5px] border-dashed border-accent" /> the target (query)
      </span>
    </div>
  );
}

function LayerCard({
  live,
  layer,
  w,
  allowed,
  scaleMax,
  active,
  hoverKey,
  onActivate,
  onHoverKey,
}: {
  live: AttnLive;
  layer: AttnLayer;
  w: Float32Array | null;
  allowed: Uint8Array;
  scaleMax: number;
  active: boolean;
  hoverKey: number | null;
  onActivate: () => void;
  onHoverKey: (k: number | null) => void;
}) {
  const { ctx, labels, slots, S } = live;
  const allowedList = useMemo(() => {
    const out: number[] = [];
    for (let k = 0; k < S; k++) if (allowed[k]) out.push(k);
    return out;
  }, [allowed, S]);
  const ranked = useMemo(() => (w ? topK(w, allowedList, allowedList.length) : []), [w, allowedList]);
  const top = ranked.slice(0, TOP);
  // Keys outside the top 5 that tie exactly with the 5th (identical inputs give identical weights).
  const moreTied = w && top.length === TOP ? ranked.slice(TOP).filter((k) => tied(w[k], w[top[TOP - 1]])).length : 0;
  const even = allowedList.length ? 1 / allowedList.length : 0;
  const bar = (v: number) => `${Math.max(1.5, Math.min(100, (v / scaleMax) * 100))}%`;
  return (
    <div
      className={cx('min-w-0 rounded-xl border px-3 py-2.5 transition-colors', active ? 'border-accent bg-accent-soft' : 'border-border bg-surface')}
      onPointerLeave={() => onHoverKey(null)}
    >
      <button type="button" onClick={onActivate} className="flex w-full flex-wrap items-baseline justify-between gap-x-2 text-left" aria-pressed={active}>
        <span className="text-[16px]">
          <LayerLabel layer={layer} />
        </span>
        <span className="num text-[14px] text-muted">
          {int(allowedList.length)} allowed{allowedList.length ? ` · even ${pctw(even)}` : ''}
        </span>
      </button>
      {!w ? (
        <p className="!mb-0 mt-2 text-[14.5px] text-ink-2">Switched off by the playground’s ablation.</p>
      ) : allowedList.length === 0 ? (
        <p className="!mb-0 mt-2 text-[15px] leading-snug text-ink-2">
          No allowed key. No row in the context points to the seed row, so this row of the mask is empty and the update is exactly 0.
        </p>
      ) : (
        <>
          <ol className="!m-0 mt-2 grid gap-1 !p-0" style={{ listStyle: 'none' }}>
            {top.map((k, i) => {
              const t = ctx.tokens[k];
              return (
                <li key={k} className="!m-0 !p-0">
                  <button
                    type="button"
                    className={cx(
                      'grid w-full grid-cols-[14px_minmax(0,1fr)_auto] items-center gap-x-1.5 rounded-md px-1 py-0.5 text-left focus-visible:outline-2 focus-visible:outline-accent',
                      hoverKey === k && 'bg-surface-3',
                    )}
                    onPointerEnter={() => onHoverKey(k)}
                    onFocus={() => onHoverKey(k)}
                    onClick={() => onHoverKey(k)}
                    aria-label={`${labels[k]} = ${t.isTarget ? '?' : valueOf(t)}: weight ${pctw(w[k])}. Show it in the context view.`}
                  >
                    <span className="num text-[13px] text-muted">{i + 1}</span>
                    <span className="min-w-0">
                      <span className="flex min-w-0 items-center gap-1">
                        <TableSwatch slot={t.isPadding ? 0 : slots[t.tableName] ?? 0} size={7} />
                        <span className="num truncate text-[14px] text-ink">
                          {labels[k]}
                          <span className="text-muted"> = {t.isTarget ? '?' : valueOf(t)}</span>
                        </span>
                      </span>
                      <span className="relative mt-0.5 block h-[5px] w-full rounded-full bg-surface-3">
                        <span className="block h-full rounded-full bg-seq-500" style={{ width: bar(w[k]) }} />
                        <span aria-hidden="true" className="absolute -top-[3px] h-[11px] w-[2px] rounded-full bg-ink" style={{ left: `calc(${bar(even)} - 1px)` }} />
                      </span>
                    </span>
                    <span className="num pl-1 text-right text-[14px] text-ink">{pctw(w[k])}</span>
                  </button>
                </li>
              );
            })}
          </ol>
          {moreTied > 0 && (
            <p className="!mb-0 mt-1 pl-5 text-[14px] leading-snug text-muted">
              + {moreTied} more {moreTied === 1 ? 'key' : 'keys'} tied at {pctw(w[top[TOP - 1]])}
            </p>
          )}
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Column layer: own labels vs other entities' labels, per block
// ─────────────────────────────────────────────────────────────────────────────

const SEG_STYLE = {
  self: { background: 'var(--seq-500)', color: '#ffffff' },
  other: { background: 'var(--ink-2)', color: 'var(--page)' },
  me: { background: 'repeating-linear-gradient(135deg, var(--muted) 0 2px, var(--surface) 2px 6px)', color: 'var(--ink)' },
} as const;

export function LabelShare({ live }: { live: AttnLive }) {
  const { fr, inCtx, target, nBlocks, H, task, masks, S, pdb } = live;
  const setSeedNode = useSim((s) => s.setSeedNode);
  const storedSeed = useSim((s) => s.seedNode);
  const scan = useOtherLabelSeeds(live);
  const [headSel, setHeadSel] = useState<HeadSel>('mean');
  const selfPos = useMemo(() => new Set(inCtx.self.map((l) => l.pos)), [inCtx]);
  const otherPos = useMemo(() => new Set(inCtx.other.map((l) => l.pos)), [inCtx]);
  const nAllowed = useMemo(() => {
    let n = 0;
    for (let k = 0; k < S; k++) n += masks.col[target * S + k];
    return n;
  }, [masks, target, S]);

  const rows = useMemo(() => {
    if (!fr) return null;
    return Array.from({ length: nBlocks }, (_, b) => {
      const w = rowWeights(live, b, 'col', target, headSel);
      if (!w) return null;
      let self = 0;
      let other = 0;
      let me = 0;
      for (let k = 0; k < S; k++) {
        if (!w[k]) continue;
        if (k === target) me += w[k];
        else if (selfPos.has(k)) self += w[k];
        else if (otherPos.has(k)) other += w[k];
      }
      return { self, other, me };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fr, nBlocks, target, headSel, selfPos, otherPos, S, H]);

  const entity = task.entityTable ? singular(task.entityTable) : 'entity';
  const forecast = task.kind === 'forecast';
  const uniformSelf = nAllowed ? selfPos.size / nAllowed : 0;
  const segs = (
    [
      { id: 'self', label: `own past labels (same ${entity}, ${selfPos.size} ${selfPos.size === 1 ? 'cell' : 'cells'})`, n: selfPos.size },
      { id: 'other', label: `other ${entity}s’ labels (${otherPos.size} ${otherPos.size === 1 ? 'cell' : 'cells'})`, n: otherPos.size },
      { id: 'me', label: 'the target itself', n: 1 },
    ] as const
  ).filter((s) => s.n > 0);
  const contrast = selfPos.size > 0 && otherPos.size > 0;
  const hits = [...scan.hits].filter((h) => h.seed !== live.ctx.params.seedNode).sort((a, b) => Math.min(b.self, b.other) - Math.min(a.self, a.other));
  const best = hits.find((h) => h.self > 0 && h.other > 0) ?? hits[0];
  const seedName = (n: number) => {
    const node = pdb.nodes[n];
    return node ? `${node.tableName}:${node.tableType}[${node.rowPos}]` : `node ${n}`;
  };

  return (
    <div className="wide my-8">
      <div className="card px-4 py-4 sm:px-5 sm:py-5">
        <div className="mb-1 flex flex-wrap items-center gap-2 text-[17px] font-semibold text-ink">
          <span>Column layer: how the target splits its weight among the visible labels</span>
          <InterpretationBadge />
        </div>
        <p className="!mb-3 text-[15px] text-ink-2">
          The target may read {int(nAllowed)} cells in the column layer: every visible label of the task plus itself.
          {selfPos.size > 0 && <> The tick on each bar marks the share its own labels would get if the head spread its weight evenly ({pctw(uniformSelf)}).</>}
        </p>
        <div className="mb-3 flex flex-wrap items-end gap-x-4 gap-y-2">
          <Segmented
            label="Heads"
            value={headSel}
            onChange={setHeadSel}
            options={[{ value: 'mean' as HeadSel, label: 'mean of heads' }, ...Array.from({ length: H }, (_, h) => ({ value: h as HeadSel, label: `head ${h + 1}` }))]}
          />
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[14px] text-ink-2">
            {segs.map((s) => (
              <span key={s.id} className="inline-flex items-center gap-1.5">
                <span className="inline-block h-3 w-3 rounded-[2px] border border-border" style={{ background: SEG_STYLE[s.id].background }} />
                {s.label}
              </span>
            ))}
            {selfPos.size > 0 && (
              <span className="inline-flex items-center gap-1.5">
                <span className="inline-block h-3.5 w-[2px] rounded-full bg-ink" /> even split of the own labels
              </span>
            )}
          </div>
        </div>
        {!rows ? (
          <p className="text-[16.5px] text-ink-2">Running the forward pass…</p>
        ) : selfPos.size + otherPos.size === 0 ? (
          <p className="text-[15.5px] text-ink-2">This context holds no visible label cell, so the target can only read itself in the column layer.</p>
        ) : (
          <div className="grid gap-2">
            {rows.map((r, b) => (
              <div key={b} className="grid grid-cols-[62px_minmax(0,1fr)] items-center gap-3">
                <span className="text-[14.5px] font-medium text-ink-2">block {b + 1}</span>
                {r ? (
                  <div className="relative h-[24px] w-full">
                    <div className="flex h-full w-full overflow-hidden rounded-[5px] bg-surface-2">
                      {segs.map((s) => {
                        const v = r[s.id];
                        return v > 0 ? (
                          <div
                            key={s.id}
                            className="flex h-full items-center justify-center overflow-hidden border-r-2 border-surface text-[13.5px] font-medium whitespace-nowrap transition-[width] duration-500 last:border-r-0"
                            style={{ width: `${v * 100}%`, ...SEG_STYLE[s.id] }}
                            title={`${s.label}: ${pctw(v)}`}
                          >
                            {v > 0.09 ? <span className={s.id === 'me' ? 'rounded-[3px] bg-surface px-1' : undefined}>{pctw(v)}</span> : ''}
                          </div>
                        ) : null;
                      })}
                    </div>
                    {selfPos.size > 0 && (
                      <div
                        aria-hidden="true"
                        className="absolute -top-[3px] -bottom-[3px] w-[2px] rounded-full bg-ink"
                        style={{ left: `calc(${uniformSelf * 100}% - 1px)` }}
                        title="even split"
                      />
                    )}
                  </div>
                ) : (
                  <span className="text-[14.5px] text-muted">switched off</span>
                )}
              </div>
            ))}
          </div>
        )}
        <div className="mt-4 grid gap-2 border-t border-grid pt-3 text-[15px] leading-snug text-ink-2">
          {forecast && (
          <p className="!mb-0">
            <strong className="font-semibold text-ink">Block 1 cannot tell own from other.</strong> When block 1’s column layer runs, each token is
            still its input embedding: a column-name term plus a value term, with no row, entity, time or position information{' '}
            <Cite kind="spec" ref="§5.2" /> <Cite kind="spec" ref="§6.5" />. Two label cells with the same value are the same vector, so they get the
            same key and exactly the same weight, whoever they belong to <Cite kind="spec" ref="§6.4" />. Only after the feature layer has written
            the {entity} into the target and into each task row can later blocks weigh their own {entity}’s labels differently.
          </p>
          )}
          {forecast && !contrast && selfPos.size + otherPos.size > 0 && (
            <div className="grid gap-2">
              <p className="!mb-0">
                {otherPos.size === 0
                  ? `In this seed’s window every visible label belongs to the seed’s own ${entity}, so this chart cannot contrast own and other labels here.`
                  : `In this seed’s window no label belongs to the seed’s own ${entity}, so this chart cannot contrast own and other labels here.`}{' '}
                The sampler decides which task rows make it in (chapter 6).
              </p>
              <div className="flex flex-wrap items-center gap-2">
                {scan.status !== 'done' ? (
                  <Button size="sm" variant="secondary" icon={<Search size={15} aria-hidden="true" />} onClick={scan.start} disabled={scan.status === 'scanning'}>
                    {scan.status === 'scanning'
                      ? `Sampling every ${task.id.split('/')[1]} seed… ${scan.done} / ${scan.total}`
                      : `Find a seed whose window holds both kinds of label`}
                  </Button>
                ) : best ? (
                  <Button size="sm" variant="primary" className="h-auto min-h-8 py-1 text-left whitespace-normal" onClick={() => setSeedNode(best.seed)}>
                    Switch to seed {seedName(best.seed)}: {best.self} own, {best.other} other {best.other === 1 ? 'label' : 'labels'}
                  </Button>
                ) : (
                  <span className="text-[14.5px] text-muted">No seed of this split has another {entity}’s label in its window.</span>
                )}
                {scan.status === 'done' && (
                  <span className="text-[14.5px] text-muted">
                    {int(scan.hits.length)} of {int(scan.total)} seeds have another {entity}’s label in their window. Switching changes the seed for the whole course.
                  </span>
                )}
              </div>
            </div>
          )}
          {storedSeed !== null && contrast && (
            <div>
              <Button size="sm" variant="ghost" onClick={() => setSeedNode(null)}>
                Back to the default seed
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/** Live numbers for the prose under the label-share chart (mean over heads, every block). */
export function useLabelShareFacts(live: AttnLive) {
  const { fr, inCtx, target, nBlocks, S, masks } = live;
  return useMemo(() => {
    if (!fr) return null;
    const selfPos = new Set(inCtx.self.map((l) => l.pos));
    const otherPos = new Set(inCtx.other.map((l) => l.pos));
    const labelPos = [...selfPos, ...otherPos];
    let nAllowed = 0;
    for (let k = 0; k < S; k++) nAllowed += masks.col[target * S + k];
    const perBlock = Array.from({ length: nBlocks }, (_, b) => {
      const w = rowWeights(live, b, 'col', target, 'mean');
      if (!w) return null;
      let self = 0;
      let other = 0;
      for (const k of selfPos) self += w[k];
      for (const k of otherPos) other += w[k];
      return { self, other };
    });
    // The full layer can read the same label cells (mask = padding only), mixed with every other token.
    const fullOnLabels = Array.from({ length: nBlocks }, (_, b) => {
      const w = rowWeights(live, b, 'full', target, 'mean');
      if (!w) return null;
      let s = 0;
      for (const k of labelPos) s += w[k];
      return s;
    });
    return {
      nSelf: selfPos.size,
      nOther: otherPos.size,
      nAllowed,
      uniformSelf: nAllowed ? selfPos.size / nAllowed : 0,
      uniformOther: nAllowed ? otherPos.size / nAllowed : 0,
      perBlock,
      fullOnLabels,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fr, inCtx, target, nBlocks, S, masks]);
}
