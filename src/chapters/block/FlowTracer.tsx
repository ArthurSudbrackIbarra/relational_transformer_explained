/**
 * Figure 6: how information can reach the target (spec §6.4, a TEACHING INTERPRETATION, not a paper
 * claim). In sublayer l, query q can receive from key k iff mask_l[q,k] = 1 (rt/model.py:186-191);
 * FFNs act per token. Following the masks through the stack (same masks in every block, model.py:226-227)
 * gives every route the computation allows. The attention weights on each hop are the live trained
 * model's (mean over heads), to show how much a route is actually used in this context.
 */
import { ArrowRight } from 'lucide-react';
import { useMemo, useState } from 'react';
import { BarChart, LAYER_INFO, LayerLabel, Switch, TableSwatch, cx } from '@/components';
import { ATTN_ORDER, type AttnLayer } from '@/engine/types';
import type { BlockLive } from './useBlockLive';
import { TokenCaption, TokenPicker } from './TokenPicker';
import { arrivalAtDest, flowOrder, fx, int, meanWeight, traceFlow, type FlowStep } from './utils';

const stepName = (s: FlowStep) => `block ${s.block + 1}, ${LAYER_INFO[s.layer].name.toLowerCase()}`;

export function FlowTracer({ live }: { live: BlockLive }) {
  const { ctx, masks, fr, target, nBlocks, H, S } = live;
  const presets = useMemo(() => live.presets.filter((p) => p.pos !== target), [live.presets, target]);
  const [srcRaw, setSrc] = useState<number | null>(null);
  const src = srcRaw !== null && srcRaw < S && !ctx.tokens[srcRaw].isPadding && srcRaw !== target ? srcRaw : presets[0]?.pos ?? 0;
  const [withFull, setWithFull] = useState(true);

  const orderFull = useMemo(() => flowOrder(nBlocks, true), [nBlocks]);
  const orderRel = useMemo(() => flowOrder(nBlocks, false), [nBlocks]);
  const order = withFull ? orderFull : orderRel;
  const trace = useMemo(() => traceFlow(masks, ctx.tokens, src, target, order), [masks, ctx, src, target, order]);
  const tFull = useMemo(() => traceFlow(masks, ctx.tokens, src, target, orderFull), [masks, ctx, src, target, orderFull]);
  const tRel = useMemo(() => traceFlow(masks, ctx.tokens, src, target, orderRel), [masks, ctx, src, target, orderRel]);
  const arrival = trace.time[target];

  const direct: Record<AttnLayer, boolean> = {
    col: !!masks.col[target * S + src],
    feat: !!masks.feat[target * S + src],
    nbr: !!masks.nbr[target * S + src],
    full: !!masks.full[target * S + src],
  };

  const stepIndex = (b: number, l: AttnLayer) => order.findIndex((s) => s.block === b && s.layer === l) + 1; // 0 = not in order

  return (
    <div className="grid grid-cols-1 gap-5">
      <div className="grid grid-cols-1 gap-3">
        <div className="grid min-w-0 gap-1">
          <span className="text-[15px] font-medium text-ink-2">Source token (where the information starts)</span>
          <TokenPicker live={live} value={src} onChange={(p) => setSrc(p)} presets={presets} label="Source presets" stripLabel="Context tokens: click one to make it the source" />
        </div>
        <Switch checked={withFull} onChange={setWithFull} label="Use full attention too" />
      </div>
      <div className="grid grid-cols-1 gap-1">
        <TokenCaption live={live} pos={src} prefix="Source" />
        <TokenCaption live={live} pos={target} prefix="Destination" />
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
        {/* the grid of sublayers */}
        <div className="min-w-0">
          <div className="mb-2 text-[16px] font-semibold text-ink">Attention sublayers, in execution order</div>
          <table className="w-full table-fixed border-separate border-spacing-1">
            <caption className="sr-only">For each block and attention layer: whether the target reads the source directly, and whether the source’s information can have reached the target</caption>
            <thead>
              <tr>
                <th scope="col" className="w-[40px] sm:w-[64px]" />
                {ATTN_ORDER.map((l) => {
                  const Icon = LAYER_INFO[l].icon;
                  return (
                    <th key={l} scope="col" className="pb-1 text-center text-[14px] font-medium text-ink-2">
                      <span className="inline-flex flex-col items-center gap-0.5 sm:flex-row sm:gap-1">
                        <Icon aria-hidden="true" size={15} />
                        {LAYER_INFO[l].name}
                      </span>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: nBlocks }, (_, b) => (
                <tr key={b}>
                  <th scope="row" className="pr-1 text-left text-[15px] font-semibold whitespace-nowrap text-ink">
                    <span className="sm:hidden">B{b + 1}</span>
                    <span className="hidden sm:inline">Block {b + 1}</span>
                  </th>
                  {ATTN_ORDER.map((l) => {
                    const t = stepIndex(b, l);
                    const off = t === 0;
                    const reached = !off && arrival > 0 && t >= arrival;
                    const arrives = !off && t === arrival;
                    const w = direct[l] && fr ? meanWeight(fr, b, l, target, src, H) : NaN;
                    const carriers = !off ? trace.reached[t - 1] : null;
                    return (
                      <td key={l} className="p-0 align-top">
                        <div
                          className={cx(
                            'flex min-h-[66px] flex-col gap-0.5 rounded-lg border px-1 py-1.5 text-[13.5px] leading-tight whitespace-nowrap sm:px-2',
                            off && 'border-dashed border-border bg-transparent text-muted',
                            !off && arrives && 'border-accent bg-accent-soft ring-1 ring-accent',
                            !off && !arrives && reached && 'border-accent/40 bg-accent-soft',
                            !off && !reached && 'border-border bg-surface',
                          )}
                          title={off ? 'not used (full attention switched off)' : `${stepName(order[t - 1])}: ${carriers} tokens carry the source’s information after this sublayer`}
                        >
                          {off ? (
                            <span>off</span>
                          ) : (
                            <>
                              {direct[l] ? (
                                <span className="font-semibold text-ink">
                                  direct
                                  {Number.isFinite(w) && (
                                    <span className="num block font-normal text-ink-2">
                                      <span className="hidden sm:inline">w </span>
                                      {fx(w, 3)}
                                    </span>
                                  )}
                                </span>
                              ) : (
                                <span className="text-muted">none</span>
                              )}
                              {arrives && <span className="font-semibold text-accent-text">arrives</span>}
                              <span className="num mt-auto text-muted">
                                {int(carriers ?? 0)}
                                <span className="hidden sm:inline"> tok</span>
                              </span>
                            </>
                          )}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="!mb-0 mt-2 text-[15px] leading-snug text-ink-2">
            <strong className="font-semibold text-ink">direct</strong>: the target may read the source in that layer (the masks are the same in every
            block); <span className="num">w</span> = the trained model’s attention weight, mean over the {H} heads; <strong className="font-semibold text-ink">none</strong>: no edge.
            Shaded: the source’s information can be in the target’s stream after that sublayer. Bottom number (<span className="num">tok</span>): how
            many tokens can carry it by then.
          </p>
        </div>

        {/* the route */}
        <div className="min-w-0">
          <div className="mb-2 text-[16px] font-semibold text-ink">One earliest route</div>
          {trace.path && trace.path.length > 0 ? (
            <ol className="!m-0 grid gap-1.5 !p-0" style={{ listStyle: 'none' }}>
              <li className="!m-0 !p-0">
                <RouteToken live={live} pos={src} role="source" />
              </li>
              {trace.path.map((hop, i) => {
                const w = fr ? meanWeight(fr, hop.step.block, hop.step.layer, hop.to, hop.from, H) : NaN;
                return (
                  <li key={i} className="!m-0 grid gap-1.5 !p-0">
                    <div className="flex items-center gap-2 pl-3 text-[15px] text-ink-2">
                      <ArrowRight aria-hidden="true" size={16} className="rotate-90 text-accent" />
                      <span>
                        block {hop.step.block + 1}, <LayerLabel layer={hop.step.layer} /> attention
                        <span className="num text-muted"> · w {fx(w, 3)}</span>
                      </span>
                    </div>
                    <RouteToken live={live} pos={hop.to} role={hop.to === target ? 'destination' : 'via'} />
                  </li>
                );
              })}
            </ol>
          ) : (
            <p className="text-[16px] text-ink-2">
              Within {nBlocks} blocks, no route from this token reaches the target through the {withFull ? 'four' : 'three relational'} layers.
            </p>
          )}
          <div className="mt-3 grid gap-1 rounded-lg border border-border bg-surface-2/60 px-3 py-2.5 text-[15.5px] leading-snug text-ink-2">
            <div>
              With full attention: arrives after <strong className="font-semibold text-ink">{tFull.time[target] > 0 ? stepName(orderFull[tFull.time[target] - 1]) : 'never'}</strong>.
            </div>
            <div>
              Relational layers only: {tRel.time[target] > 0 ? (
                <>
                  arrives after <strong className="font-semibold text-ink">{stepName(orderRel[tRel.time[target] - 1])}</strong>, {tRel.path?.length ?? 0} hop
                  {(tRel.path?.length ?? 0) === 1 ? '' : 's'}.
                </>
              ) : (
                <strong className="font-semibold text-ink">not within {nBlocks} blocks</strong>
              )}
            </div>
          </div>
        </div>
      </div>
      <ArrivalSummary live={live} />
    </div>
  );
}

function RouteToken({ live, pos, role }: { live: BlockLive; pos: number; role: 'source' | 'via' | 'destination' }) {
  const t = live.ctx.tokens[pos];
  const slot = live.slots[t.tableName] ?? 0;
  return (
    <div
        className={cx(
          'flex flex-wrap items-center gap-x-2 gap-y-0.5 rounded-lg border px-3 py-1.5',
          role === 'destination' ? 'target-outline border-transparent bg-accent-soft' : role === 'source' ? 'border-accent/50 bg-surface' : 'border-border bg-surface',
        )}
      >
        <TableSwatch slot={slot} size={11} />
        <span className="num text-[15px] text-ink">
          #{pos} {live.labels[pos]}
        </span>
        <span className="text-[14.5px] text-ink-2">{role === 'destination' ? 'the masked target “?”' : role}</span>
    </div>
  );
}

/** How many tokens' information can be at the target after each block, relational layers only vs all four. */
function ArrivalSummary({ live }: { live: BlockLive }) {
  const { masks, ctx, target, nBlocks } = live;
  const data = useMemo(() => {
    const rel = arrivalAtDest(masks, ctx.tokens, target, flowOrder(nBlocks, false));
    const all = arrivalAtDest(masks, ctx.tokens, target, flowOrder(nBlocks, true));
    const per = (arr: Int16Array, perBlock: number) => {
      const cum = new Array(nBlocks).fill(0);
      let total = 0;
      for (let i = 0; i < ctx.tokens.length; i++) {
        if (ctx.tokens[i].isPadding || i === target) continue;
        total++;
        if (arr[i] > 0) {
          const b = Math.floor((arr[i] - 1) / perBlock);
          for (let k = b; k < nBlocks; k++) cum[k]++;
        }
      }
      return { cum, total };
    };
    return { rel: per(rel, 3), all: per(all, 4) };
  }, [masks, ctx, target, nBlocks]);
  const cats = Array.from({ length: nBlocks }, (_, b) => `after block ${b + 1}`);
  return (
    <div className="rounded-xl border border-border bg-surface-2/50 p-4">
      <BarChart
        title={`Tokens whose information can be in the target’s stream (of ${data.rel.total} other tokens)`}
        subtitle="Counted from the masks alone, for every token of the live context at once."
        categories={cats}
        series={[
          { id: 'rel', label: 'relational layers only (column, feature, neighbor)', slot: 1, values: data.rel.cum },
          { id: 'all', label: 'all four layers', slot: 2, values: data.all.cum },
        ]}
        format={(v) => int(v)}
        domain={[0, data.rel.total]}
        labelWidth={110}
      />
    </div>
  );
}
