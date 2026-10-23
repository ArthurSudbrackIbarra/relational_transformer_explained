/**
 * Section 5 figure: padding. Vecs::new allocates every position as padding (fly.rs:65-85); the
 * sampler overwrites only the slots it emits (is_padding = false, fly.rs:500). When the BFS runs out
 * of rows before L, the rest stays padding (fly.rs:387-389; spec §4.11).
 *
 * If the live context is full, the figure scans the task's seeds (same L, w and RNG seed) for the
 * context with the fewest cells, using the engine's sampler. It then checks, on the engine's forward
 * pass, that padding attends to nothing, is attended by nothing, and stays exactly 0 (spec §6.2, §7.2).
 */
import { useEffect, useMemo, useState } from 'react';
import { Cite, Figure, LayerLabel, StatusBadge, TableSwatch, TokenStrip, cx, formatNumber, formatSigned } from '@/components';
import { buildMasks } from '@/engine/masks';
import { forwardWithMasks, l2norm, rmsnorm, scalarValueTerm, tensor, textEmbedder } from '@/engine/model';
import { getForward, getSeeds } from '@/engine/pipeline';
import { ATTN_ORDER, SemType, type ForwardResult, type SampledContext, type Split } from '@/engine/types';
import { useSim } from '@/state/store';
import { ArrayStrip } from './ArrayStrip';
import { afterPaint, contextFor, findMostPadded, type PaddedPick } from './compute';
import { cutRow, nodeLabel, nodeRef, plural, taskShort, tsDate, type TokensLive } from './live';
import { formatSlot } from './fields';
import { useIdle, useInView } from './useInView';
import { ZeroVec } from './ZeroVec';

interface ScanState {
  done: number;
  total: number;
  finished: boolean;
  pick: PaddedPick | null;
}

export function PaddingDemo({ live }: { live: TokensLive }) {
  const { ctx, pdb, task, assets, fr, slots } = live;
  const rngSeed = useSim((s) => s.rngSeed);
  const { L, w } = ctx.params;
  const liveHasPad = ctx.numReal < ctx.tokens.length;
  const [viewRef, inView] = useInView<HTMLDivElement>('900px');
  const idle = useIdle(2500);
  const groups = useMemo(() => {
    const splits: Split[] = task.kind === 'autocomplete' ? ['Train'] : ['Train', 'Val', 'Test'];
    return splits.map((split) => ({ split, seeds: getSeeds(pdb, task, split) }));
  }, [pdb, task]);
  const nSeeds = useMemo(() => groups.reduce((s, g) => s + g.seeds.length, 0), [groups]);
  const [scan, setScan] = useState<ScanState>({ done: 0, total: 0, finished: false, pick: null });

  useEffect(() => {
    if (liveHasPad || !(inView || idle)) return;
    const signal = { cancelled: false };
    setScan({ done: 0, total: nSeeds, finished: false, pick: null });
    findMostPadded(pdb, task, groups, L, w, rngSeed, (done, total) => !signal.cancelled && setScan((s) => ({ ...s, done, total })), signal).then(
      (pick) => !signal.cancelled && setScan((s) => ({ ...s, done: s.total, finished: true, pick })),
    );
    return () => {
      signal.cancelled = true;
    };
  }, [liveHasPad, inView, idle, groups, nSeeds, pdb, task, L, w, rngSeed]);

  const demoCtx: SampledContext | null = useMemo(
    () => (liveHasPad ? ctx : scan.pick ? contextFor(pdb, task, scan.pick.seedNode, L, w, rngSeed) : null),
    [liveHasPad, ctx, scan.pick, pdb, task, L, w, rngSeed],
  );

  const [demoFr, setDemoFr] = useState<ForwardResult | null>(null);
  useEffect(() => {
    if (liveHasPad) {
      setDemoFr(fr);
      return;
    }
    setDemoFr(null);
    if (!demoCtx || !assets.store || !assets.weights) return;
    let alive = true;
    afterPaint(() => getForward(demoCtx, pdb, assets.store!, assets.weights!)).then((r) => alive && setDemoFr(r));
    return () => {
      alive = false;
    };
  }, [liveHasPad, fr, demoCtx, pdb, assets.store, assets.weights]);

  // Non-interference: drop the padding slots and run the model again on the real tokens only.
  const [unpadded, setUnpadded] = useState<{ ctxId: SampledContext; prediction: number; maxDiff: number; S: number } | null>(null);
  useEffect(() => {
    if (!demoCtx || !demoFr || !assets.store || !assets.weights) return;
    if (demoCtx.numReal >= demoCtx.tokens.length) return;
    let alive = true;
    afterPaint(() => {
      const real = demoCtx.tokens.slice(0, demoCtx.numReal);
      const r = forwardWithMasks(real, buildMasks(real), textEmbedder(pdb, assets.store!), assets.weights!, { capture: false });
      const d = demoFr.d;
      let maxDiff = 0;
      for (let i = 0; i < real.length * d; i++) maxDiff = Math.max(maxDiff, Math.abs(r.xFinal[i] - demoFr.xFinal[i]));
      return { ctxId: demoCtx, prediction: r.prediction, maxDiff, S: real.length };
    }).then((u) => alive && setUnpadded(u));
    return () => {
      alive = false;
    };
  }, [demoCtx, demoFr, pdb, assets.store, assets.weights]);
  const unp = unpadded && unpadded.ctxId === demoCtx ? unpadded : null;

  const cut = useMemo(() => cutRow(pdb, ctx), [pdb, ctx]);

  const checks = useMemo(() => {
    if (!demoCtx) return null;
    const S = demoCtx.tokens.length;
    const pad = demoCtx.tokens.filter((t) => t.isPadding).map((t) => t.pos);
    const isPad = demoCtx.tokens.map((t) => t.isPadding);
    const masks = demoFr?.masks;
    const maskRows = masks
      ? ATTN_ORDER.map((l) => {
          const m = masks[l];
          let fromPad = 0;
          let toPad = 0;
          for (let q = 0; q < S; q++)
            for (let k = 0; k < S; k++) {
              if (!m[q * S + k]) continue;
              if (isPad[q]) fromPad++;
              else if (isPad[k]) toPad++;
            }
          return { layer: l, fromPad, toPad };
        })
      : null;
    // What a padding slot would enter the model as without the ¬is_padding gates: its slots are 0,
    // but the encoders have biases (model.py:115-123), so W·0 + b = b and RMSNorm(b) ≠ 0.
    let wouldBe: { colNorm: number; valNorm: number } | null = null;
    let dec: { head: string; at: number; bias: number } | null = null;
    if (assets.weights) {
      const w = assets.weights;
      const bCol = tensor(w, 'enc_dict.col_name.bias');
      const colB = rmsnorm(bCol, 1, w.config.dModel, tensor(w, 'norm_dict.col_name.weight'), w.config.eps);
      const valB = scalarValueTerm(w, 'number', 0).term; // sem_types = 0 → the number encoder
      wouldBe = { colNorm: l2norm(colB), valNorm: l2norm(valB) };
    }
    if (demoFr && pad.length && assets.weights) {
      const t = demoCtx.tokens[demoCtx.targetPos >= 0 ? demoCtx.targetPos : 0];
      const head = t.sem === SemType.Boolean ? 'boolean' : t.sem === SemType.DateTime ? 'datetime' : 'number';
      dec = { head, at: demoFr.yhat[head as 'boolean'][pad[0]], bias: tensor(assets.weights, `dec_dict.${head}.bias`)[0] };
    }
    return { S, pad, isPad, maskRows, wouldBe, dec };
  }, [demoCtx, demoFr, assets.weights]);

  const seedNode = demoCtx ? pdb.nodes[demoCtx.params.seedNode] : null;
  const entity = seedNode && task.kind === 'forecast' && seedNode.f2pNbrIdxs.length ? seedNode.f2pNbrIdxs[0] : null;
  const firstPad = checks?.pad[0];
  const padTok = firstPad !== undefined && demoCtx ? demoCtx.tokens[firstPad] : null;

  return (
    <div ref={viewRef} className="wide">
      <Figure
        wide
        title="Padding: the slots the sampler never wrote"
        caption={
          <>
            Top: the <span className="font-mono">is_padding</span> array and the tokens of a context that ran out of rows. Left: one padding slot, every
            field at the value <span className="font-mono">Vecs::new</span> gave it. Right: what the model does with it, computed from the real weights
            and masks of this context.
          </>
        }
      >
        <div className="mb-4 text-[16.5px] leading-relaxed text-ink-2">
          {liveHasPad ? (
            <>
              The live context ran out of rows: the sampler wrote {plural(ctx.numReal, 'cell')} and left{' '}
              <span className="font-semibold text-ink">{plural(ctx.tokens.length - ctx.numReal, 'padding slot')}</span>.
            </>
          ) : (
            <>
              The live context has no padding: all {L} positions hold cells
              {cut ? (
                <>
                  , and the last row, <span className="text-ink">{nodeRef(pdb.nodes[cut.node])}</span>, was cut after {cut.emitted} of its{' '}
                  {cut.total} cells
                </>
              ) : null}
              .{' '}
              {scan.pick && seedNode ? (
                <>
                  So here is the {taskShort(task)} seed whose context is smallest at the same L = {L}, w = {w}, found by running the sampler on all{' '}
                  {scan.total.toLocaleString('en-US')} seeds of the task:{' '}
                  <span className="text-ink">{nodeRef(seedNode)}</span>
                  {entity !== null ? (
                    <>
                      {' '}
                      → <span className="text-ink">{nodeLabel(pdb, entity)}</span>
                    </>
                  ) : null}
                  {seedNode.ts !== null ? <>, at {tsDate(seedNode.ts)}</> : null}. Its BFS found only {plural(demoCtx!.visited.length, 'row')} and{' '}
                  {plural(demoCtx!.numReal, 'cell')}, so{' '}
                  <span className="font-semibold text-ink">{plural(L - demoCtx!.numReal, 'slot')}</span> stay padding.
                </>
              ) : scan.finished ? (
                <>No seed of this task leaves padding at L = {L}, w = {w}. Switch to ShopDB customer-churn, or raise L, to see some.</>
              ) : (
                <>Looking for a seed of this task whose context ends in padding…</>
              )}
            </>
          )}
        </div>

        {!liveHasPad && !scan.finished && (
          <div className="mb-4" aria-live="polite">
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
              <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${scan.total ? (100 * scan.done) / scan.total : 0}%` }} />
            </div>
            <div className="num mt-1 text-[14.5px] text-ink-2">
              sampled {scan.done.toLocaleString('en-US')} / {(scan.total || nSeeds).toLocaleString('en-US')} contexts (one per seed of {taskShort(task)}, same L, w
              and RNG seed)
            </div>
          </div>
        )}

        {demoCtx && checks && (
          <>
            <ArrayStrip name="is_padding" values={checks.isPad} />
            <div className="mt-4">
              <TokenStrip tokens={demoCtx.tokens} tableSlots={slots} variant="cells" showPadding="collapsed" label="Tokens of the padded context" />
            </div>

            <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
              <div className="min-w-0">
                <div className="eyebrow mb-2">A padding slot (position {firstPad})</div>
                {padTok && (
                  <dl className="grid grid-cols-[minmax(0,140px)_minmax(0,1fr)] gap-x-3 gap-y-1 text-[15px]">
                    {(
                      [
                        ['node_idxs', String(padTok.nodeIdx)],
                        ['f2p_nbr_idxs', `[${padTok.f2pNbrIdxs.join(', ')}]`],
                        ['table_name_idxs', String(padTok.tableNameIdx)],
                        ['col_name_idxs', String(padTok.colNameIdx)],
                        ['class_value_idxs', String(padTok.classValueIdx)],
                        ['col_name_values', <><ZeroVec /> (384 zeros)</>],
                        ['sem_types', String(padTok.sem)],
                        ['number_values', formatSlot(padTok.numberValue)],
                        ['text_values', <><ZeroVec /> (384 zeros)</>],
                        ['datetime_values', formatSlot(padTok.datetimeValue)],
                        ['boolean_values', formatSlot(padTok.booleanValue)],
                        ['masks', String(padTok.mask)],
                        ['is_targets', String(padTok.isTarget)],
                        ['is_task_nodes', String(padTok.isTaskNode)],
                        ['is_padding', String(padTok.isPadding)],
                      ] as const
                    ).map(([k, v]) => (
                      <div key={k} className="contents">
                        <dt className="font-mono text-ink-2">{k}</dt>
                        <dd className={cx('num min-w-0', k !== 'f2p_nbr_idxs' && 'whitespace-nowrap', k === 'is_padding' ? 'font-semibold text-accent-text' : 'text-ink')}>{v}</dd>
                      </div>
                    ))}
                  </dl>
                )}
              </div>

              <div className="min-w-0">
                <div className="eyebrow mb-2">What the model does with it</div>
                {!checks.maskRows ? (
                  <StatusBadge status="info">{assets.status === 'error' ? 'model assets failed to load' : 'running the forward pass…'}</StatusBadge>
                ) : (
                  <ol className="space-y-5">
                    <li>
                      <div className="mb-1 text-[16px] font-semibold text-ink">1 · It enters as the zero vector</div>
                      <p className="text-[15.5px] leading-relaxed text-ink-2">
                        Its slots hold zeros, but that alone would not give a zero input: the encoders have biases{' '}
                        <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:115-123" />,</span> so W·0 + b = b. What zeroes the slot is the factor ¬is_padding on every term{' '}
                        <Cite kind="code" ref="rt/model.py:213" /> <Cite kind="code" ref="rt/model.py:219" /> <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:223" />.</span>
                      </p>
                      {checks.wouldBe && (
                        <table className="mt-2 w-full border-collapse text-[15px]">
                          <thead>
                            <tr className="border-b border-grid text-left text-[14.5px] text-ink-2">
                              <th scope="col" className="py-1 pr-2 font-medium">
                                term of x₀
                              </th>
                              <th scope="col" className="py-1 pr-2 font-medium whitespace-nowrap">
                                without the gate
                              </th>
                              <th scope="col" className="py-1 font-medium">
                                gated
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr className="border-b border-grid/60">
                              <td className="py-1 pr-2 text-ink">column name, RMSNorm(b)</td>
                              <td className="num py-1 pr-2 whitespace-nowrap text-ink">‖·‖ = {formatNumber(checks.wouldBe.colNorm, 2)}</td>
                              <td className="num py-1 whitespace-nowrap text-ink">
                                × 0 = <ZeroVec />
                              </td>
                            </tr>
                            <tr className="border-b border-grid/60">
                              <td className="py-1 pr-2 text-ink">number value (sem_types = 0), RMSNorm(b)</td>
                              <td className="num py-1 pr-2 whitespace-nowrap text-ink">‖·‖ = {formatNumber(checks.wouldBe.valNorm, 2)}</td>
                              <td className="num py-1 whitespace-nowrap text-ink">
                                × 0 = <ZeroVec />
                              </td>
                            </tr>
                            <tr>
                              <td className="py-1 pr-2 text-ink">the four mask vectors</td>
                              <td className="font-mono py-1 pr-2 text-ink-2">masks = false</td>
                              <td className="num py-1 whitespace-nowrap text-ink">
                                × 0 = <ZeroVec />
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      )}
                    </li>

                    <li>
                      <div className="mb-1 text-[16px] font-semibold text-ink">2 · Attention cannot reach it</div>
                      <table className="w-full border-collapse text-[15px]">
                        <thead>
                          <tr className="border-b border-grid text-left text-[14.5px] text-ink-2">
                            <th scope="col" className="py-1 pr-2 font-medium">
                              mask
                            </th>
                            <th scope="col" className="py-1 pr-2 font-medium">
                              padding query → any key
                            </th>
                            <th scope="col" className="py-1 font-medium">
                              real query → padding key
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {checks.maskRows.map((r) => (
                            <tr key={r.layer} className="border-b border-grid/60 last:border-0">
                              <th scope="row" className="py-1 pr-2 text-left font-normal text-ink">
                                <LayerLabel layer={r.layer} size={14} />
                              </th>
                              <td className="num py-1 pr-2">{r.fromPad}</td>
                              <td className="num py-1">{r.toPad}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      <div className="mt-1.5">
                        {checks.maskRows.every((r) => r.fromPad === 0 && r.toPad === 0) ? (
                          <StatusBadge status="good">counted in the live masks: 0 pairs</StatusBadge>
                        ) : (
                          <StatusBadge status="critical">unexpected pairs involving padding</StatusBadge>
                        )}
                      </div>
                    </li>

                    <li>
                      <div className="mb-1 text-[16px] font-semibold text-ink">3 · It stays 0 through every block</div>
                      <ul className="space-y-1 text-[15.5px] leading-snug text-ink-2">
                        <li>
                          RMSNorm(<ZeroVec />) = <ZeroVec /> <span className="whitespace-nowrap"><Cite kind="spec" ref="§7.2" />,</span> and Q, K, V = W·<ZeroVec /> = <ZeroVec />, because the
                          projections have no bias{' '}
                          <span className="whitespace-nowrap">
                            <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:25-27" />.</span>
                          </span>
                        </li>
                        <li>
                          A query with no allowed key outputs exactly <ZeroVec /> <span className="whitespace-nowrap"><Cite kind="spec" ref="§6.1" />.</span> The output projection{' '}
                          <code>wo</code> has no bias either <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:28" />,</span> so the residual update is <ZeroVec />{' '}
                          <span className="whitespace-nowrap">
                            <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:46" />.</span>
                          </span>
                        </li>
                        <li>
                          FFN(<ZeroVec />) = W₂(SiLU(W₁<ZeroVec />) ⊙ W₃<ZeroVec />) = <ZeroVec />, again without biases{' '}
                          <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:54-59" />.</span> The final RMSNorm keeps <ZeroVec />{' '}
                          <span className="whitespace-nowrap">
                            <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:229" />.</span>
                          </span>
                        </li>
                      </ul>
                      <p className="mt-1.5 text-[14.5px] leading-snug text-ink-2">
                        This is a derivation, not a measurement: the site’s engine relies on it and skips padding rows inside the blocks, which gives the
                        same numbers.
                      </p>
                    </li>

                    <li>
                      <div className="mb-1 text-[16px] font-semibold text-ink">4 · The real tokens never notice</div>
                      {unp ? (
                        <>
                          <p className="text-[15.5px] leading-relaxed text-ink-2">
                            Running the model again on the {plural(unp.S, 'real token')} alone (S = {unp.S}, masks rebuilt) gives the prediction{' '}
                            <span className="num text-ink">{formatSigned(unp.prediction, 6)}</span>, against{' '}
                            <span className="num text-ink">{formatSigned(demoFr!.prediction, 6)}</span> with all {checks.S} slots. Largest difference
                            over the final states of the real tokens: <span className="num text-ink">{unp.maxDiff === 0 ? '0' : unp.maxDiff.toExponential(1)}</span>.
                          </p>
                          <div className="mt-1.5">
                            {unp.prediction === demoFr!.prediction ? (
                              <StatusBadge status="good">identical, with or without padding</StatusBadge>
                            ) : (
                              <StatusBadge status={Math.abs(unp.prediction - demoFr!.prediction) < 1e-5 ? 'good' : 'warning'}>
                                |Δ| = {Math.abs(unp.prediction - demoFr!.prediction).toExponential(1)}
                              </StatusBadge>
                            )}
                          </div>
                        </>
                      ) : (
                        <StatusBadge status="info">running the model without the padding…</StatusBadge>
                      )}
                    </li>

                    {checks.dec && (
                      <li className="text-[15px] leading-snug text-ink-2">
                        The decoders still run at every position <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:239" />.</span> At padding position {firstPad} the{' '}
                        {checks.dec.head} decoder outputs <span className="num text-ink">{formatSigned(checks.dec.at, 4)}</span>, and its bias is b ={' '}
                        <span className="num text-ink">{formatSigned(checks.dec.bias, 4)}</span>: the same number, because W·<ZeroVec /> = <ZeroVec />. Nothing reads it:
                        the loss and the readout look only at the masked position.
                      </li>
                    )}
                  </ol>
                )}
              </div>
            </div>
            {!liveHasPad && seedNode && (
              <div className="mt-5 flex items-start gap-2 rounded-lg bg-surface-2 px-3 py-2 text-[15px] leading-snug text-ink-2">
                <TableSwatch slot={slots[seedNode.tableName] ?? 0} className="mt-[5px] shrink-0" />
                <span>
                  <span className="text-ink">{nodeRef(seedNode)}</span> is{' '}
                  {seedNode.tableType && seedNode.tableType !== 'Db'
                    ? `a ${seedNode.tableType} row of ${taskShort(task)}`
                    : `a ${seedNode.tableName} row, one of the seeds of ${taskShort(task)}`}
                  . This context is
                  computed for this figure only; the rest of the course keeps your seed.
                </span>
              </div>
            )}
          </>
        )}
      </Figure>
    </div>
  );
}
