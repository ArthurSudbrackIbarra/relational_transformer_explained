/**
 * Stage 7: from the last block's output to the answer.
 *
 *   x_final = RMSNorm_out(x)                                   rt/model.py:229
 *   ŷ_t     = x_final W_tᵀ + b_t for t ∈ {number, datetime, boolean} at EVERY position   rt/model.py:238-239
 *   read ŷ at the masked position: boolean → logit (binary task), number → z-score (regression)
 *                                                              rt/main.py:283-288
 *   display: σ(logit), or ẑ·σ + μ with the stats the target was z-scored with (spec §8.4); loss as in training (rt/model.py:250-265)
 *
 * The text decoder Linear(d, 384) is in the checkpoint but never read (text is never masked), so the
 * engine skips it (model/forward.ts); ŷ here is the S × 3 matrix of the three scalar heads.
 */
import { useMemo } from 'react';
import { Cite, Figure, StatusBadge, StepPlayer, TeX, cx, useStepPlayer } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { bceWithLogits, huber } from '@/engine/model/loss';
import { SemType } from '@/engine/types';
import { fmtNum, fmtSigned, int, pct } from './format';
import { W, useE2E, type E2ELive } from './live';
import { DotLine, MatMul, fitScale, type LinkedHover } from './MatMul';
import { linearEntry, rowOf } from './math';
import { NormView } from './NormView';
import { HoverHint, MathHead, Readout, tok } from './opParts';
import { FocusLine, LoadingNote, OpLayout, useViewHover, useVisualWidth } from './shared';

type OutOp = 'norm' | 'decode' | 'answer';
const OPS: { id: OutOp; short: string }[] = [
  { id: 'norm', short: 'Final RMSNorm' },
  { id: 'decode', short: 'Decoders at every position' },
  { id: 'answer', short: 'Read the masked cell' },
];
const HEADS = ['number', 'datetime', 'boolean'] as const;

export function OutputStage() {
  const live = useE2E();
  const player = useStepPlayer(OPS.length, { intervalMs: 3200 });
  const op = OPS[player.index].id;
  const [hover, setHover] = useViewHover<LinkedHover>([player.index, live.ctx]);
  const [ref, size] = useElementSize<HTMLDivElement>();
  const width = useVisualWidth(size.width);
  const ready = !!live.fr && !!live.weights;
  return (
    <Figure
      wide
      title={
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span>From the last block to the answer · {OPS[player.index].short}</span>
          {live.assets.weightsKind === 'init' && <StatusBadge status="warning">untrained weights</StatusBadge>}
        </span>
      }
      caption={<>The same matrices as before, one row per token. The masked target’s row is where the answer is read.</>}
    >
      <StepPlayer {...player} unit="step" label="Play through the output stage" describe={(i) => <>{OPS[i].short}.</>} />
      <ol className="!mt-3 !mb-0 flex flex-wrap gap-1.5 !p-0" style={{ listStyle: 'none' }} aria-label="Steps of the output stage">
        {OPS.map((o, i) => (
          <li key={o.id} className="!m-0 !p-0">
            <button
              type="button"
              aria-pressed={i === player.index}
              onClick={() => {
                player.setPlaying(false);
                player.setIndex(i);
              }}
              className={cx(
                'inline-flex h-10 items-center gap-1.5 rounded-lg border px-2.5 text-[15px] transition-colors',
                i === player.index ? 'border-accent bg-accent-soft font-semibold text-accent-text' : 'border-border bg-surface text-ink-2 hover:bg-surface-2',
              )}
            >
              <span className="num text-muted">{i + 1}</span>
              {o.short}
            </button>
          </li>
        ))}
      </ol>
      {op !== 'answer' && <FocusLine className="mt-3" />}
      <div ref={ref} className="mt-4 min-w-0">
        {!ready ? (
          <LoadingNote />
        ) : op === 'norm' ? (
          <NormView
            live={live}
            x={live.fr!.blocks[live.fr!.blocks.length - 1].ffn.xOut}
            y={live.fr!.xFinal}
            gamma={W(live.weights!, 'norm_out.weight')}
            xTex="x"
            gammaTex="\gamma^{\text{out}}"
            yTex="z"
            width={width}
            hover={hover}
            setHover={setHover}
            title="One last RMSNorm, norm_out"
            cite={
              <>
                <code>self.norm_out = nn.RMSNorm(d_model)</code> <Cite kind="code" ref="rt/model.py:150" />, applied after the last block{' '}
                <Cite kind="code" ref="rt/model.py:229" />.
              </>
            }
          >
            <p className="!mb-0">
              x is the residual stream after block {live.nBlocks}. Unlike the norms inside the blocks, this one changes the stream itself: the
              decoders read its output z.
            </p>
          </NormView>
        ) : op === 'decode' ? (
          <DecodeOp live={live} width={width} hover={hover} setHover={setHover} />
        ) : (
          <AnswerOp live={live} />
        )}
      </div>
    </Figure>
  );
}

/** ŷ as an [S × 3] matrix (number, datetime, boolean) and the stacked decoder weights [3 × d]. */
function useDecoders(live: E2ELive) {
  const { fr, weights, S, d } = live;
  return useMemo(() => {
    const Wd = new Float32Array(3 * d);
    const bd = new Float32Array(3);
    const yhat = new Float32Array(S * 3);
    HEADS.forEach((k, j) => {
      Wd.set(W(weights!, `dec_dict.${k}.weight`), j * d);
      bd[j] = W(weights!, `dec_dict.${k}.bias`)[0];
      for (let s = 0; s < S; s++) yhat[s * 3 + j] = fr!.yhat[k][s];
    });
    return { Wd, bd, yhat };
  }, [fr, weights, S, d]);
}

function DecodeOp({ live, width, hover, setHover }: { live: E2ELive; width: number; hover: LinkedHover | null; setHover: (h: LinkedHover | null) => void }) {
  const { S, d, numReal, focus, target, fr } = live;
  const tokens = useMemo(() => ({ drawRows: numReal, rowSlots: live.tokenSlots, tokenRows: true }), [numReal, live.tokenSlots]);
  const { Wd, bd, yhat } = useDecoders(live);
  const s = fitScale(width, d * 3 + 3 * 16 + 16 + 60);
  const hv = hover?.which === 'C' ? linearEntry(rowOf(fr!.xFinal, hover.r, d), Wd, d, hover.c, bd) : null;
  const tt = live.ctx.tokens[target];
  const col = tt?.sem === SemType.Boolean ? 2 : tt?.sem === SemType.DateTime ? 1 : 0;
  return (
    <OpLayout
      visual={
        <MatMul
          A={{ data: fr!.xFinal, rows: S, cols: d, ...tokens, name: <TeX tex="z" />, text: 'z', label: 'final normalized vectors' }}
          B={{ data: Wd, rows: d, cols: 3, transpose: true, name: <TeX tex="[\,w_{\text{num}}\;w_{\text{dt}}\;w_{\text{bool}}\,]" />, text: 'W_decᵀ', label: 'the three scalar decoders', shape: `${d} × 3` }}
          C={{ data: yhat, rows: S, cols: 3, ...tokens, name: <TeX tex="\hat y" />, text: 'ŷ', label: 'decoder outputs at every position', shape: `${int(numReal)} × 3` }}
          bias={{ data: bd, name: <>+ <TeX tex="b" /></>, text: 'b', label: 'the three decoder biases' }}
          cellS={2.5 * s}
          cellK={3 * s}
          cellN={16}
          focusRow={target}
          onFocusRow={live.setFocus}
          hover={hover}
          setHover={setHover}
        />
      }
      math={
        <MathHead
          title="Four decoders, at every position"
          tex={`\\hat y^{t}_i = w_t \\cdot z_i + b_t`}
          cite={
            <>
              <code>dec_dict</code>: one <code>nn.Linear(d_model, 1)</code> per scalar type, plus a text head <code>nn.Linear(d_model, d_text)</code>{' '}
              <Cite kind="code" ref="rt/model.py:124-131" />, each applied to the whole sequence <Cite kind="code" ref="rt/model.py:238-239" />.
            </>
          }
        >
          <p className="!mb-0">
            Every position gets a number, a datetime and a boolean output. The official code also runs the text head ({d} → 384), but its output is
            never used because text is never masked, so this page leaves it out <Cite kind="spec" ref="§8.1" />. Columns left to right: number, datetime, boolean. The outlined row is the masked target
            {tt ? <> ({tok(live, target)}, a {['number', 'text', 'datetime', 'boolean'][tt.sem]} cell)</> : null}: its{' '}
            {['number', 'datetime', 'boolean'][col]} output is the prediction. Every other value is computed and discarded.
          </p>
          <Readout>
            {hv && hover ? (
              <>
                ŷ<sub>{HEADS[hover.c]}</sub>[{tok(live, hover.r)}] = Σ<sub>k</sub> z[·, k]·w[k] + b = <DotLine terms={hv.terms} total={hv.value} extra={<> + {fmtNum(hv.bias, 3, true)}</>} />
              </>
            ) : (
              <HoverHint />
            )}
          </Readout>
          {focus !== target && <p className="!mt-2 !mb-0 text-[14.5px] text-muted">The outline stays on the target here, whichever token you follow.</p>}
        </MathHead>
      }
    />
  );
}

function AnswerOp({ live }: { live: E2ELive }) {
  const { pred, task, ctx, target, labels, fr } = live;
  if (!pred || !fr || target < 0) return <LoadingNote />;
  const t = ctx.tokens[target];
  const binary = pred.taskKind === 'binary';
  const z = pred.output;
  const label = pred.label;
  const nSelf = labels.self.length;
  const nOther = labels.other.length;
  let loss = 0;
  if (binary) loss = bceWithLogits(z, label.binary ?? 0);
  else loss = huber(z, label.z ?? 0);
  const rawLabel = label.raw === null ? '∅' : String(label.raw);
  return (
    <OpLayout
      visual={
        <div className="grid grid-cols-1 gap-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <Big title={binary ? 'logit (boolean head)' : 'z-score (number head)'} value={fmtSigned(z)} note={`ŷ at position ${target} (${live.tokenLabels[target]})`} />
            {binary ? (
              <Big title="probability" value={pct(pred.probability!)} note={`σ(${fmtNum(z)}) that ${task.targetCol} is true`} accent />
            ) : (
              <Big title={`${task.targetCol}, in its units`} value={fmtNum(pred.value!, 2)} note={`${fmtNum(z)} × ${fmtNum(pred.stats!.std, 3)} + ${fmtNum(pred.stats!.mean, 3, true)}`} accent />
            )}
            <Big title="hidden label" value={binary ? `${rawLabel} (y = ${label.binary})` : rawLabel} note={binary ? `stored z ${fmtSigned(t.booleanValue)}; y = 1{z > 0}` : `stored z ${fmtSigned(label.z ?? 0)}`} />
          </div>
          {binary ? <SigmoidBar p={pred.probability!} y={label.binary ?? 0} /> : null}
          <div className="rounded-xl border border-border bg-surface px-4 py-3 text-[15.5px] leading-snug text-ink-2">
            <span className="font-semibold text-ink">The loss this context would give in training: </span>
            {binary ? (
              <>
                BCE-with-logits(ẑ = {fmtNum(z)}, y = {label.binary}) = <span className="num font-semibold text-ink">{fmtNum(loss, 4)}</span>
              </>
            ) : (
              <>
                Huber(ẑ − z = {fmtNum(z - (label.z ?? 0))}) = <span className="num font-semibold text-ink">{fmtNum(loss, 4)}</span>
              </>
            )}{' '}
            <Cite kind="code" ref="rt/model.py:250-265" />. With one masked cell the mean over masked cells is this single value. The evaluation loop
            computes it too <Cite kind="code" ref="rt/main.py:281" /> and only logs its average <Cite kind="code" ref="rt/main.py:316-326" />: at
            inference no weight changes.
          </div>
        </div>
      }
      math={
        <MathHead
          title="Read the answer at the masked cell"
          tex={binary ? `p = \\sigma(\\hat y) = 1 \\,/\\, (1 + e^{-\\hat y})` : `\\hat v = \\hat y \\cdot \\sigma + \\mu`}
          cite={
            <>
              Read at <code>is_targets</code> <Cite kind="code" ref="rt/main.py:283-288" />; σ and the de-normalization are display steps, as in the
              authors’ later scripts <Cite kind="spec" ref="§8.4" />.
            </>
          }
        >
          <p className="!mb-0">
            {binary ? (
              <>
                The code keeps the logit: the loss and the AUROC both take it directly <Cite kind="spec" ref="§8.5" />. The sigmoid only turns it into a
                probability for people.
              </>
            ) : (
              <>
                The number head predicts a z-score in the target column’s own scale. The mean and standard deviation that z-scored{' '}
                {task.targetCol} turn it back into units: the Train split’s for a task table, the whole table’s for an autocomplete column{' '}
                <Cite kind="spec" ref="§8.4" /> <Cite kind="spec" ref="§3.4" />.
              </>
            )}{' '}
            The model got here from the {task.kind === 'forecast' ? `${nSelf} self and ${nOther} other` : nOther} in-context label
            {(task.kind === 'forecast' ? nSelf + nOther : nOther) === 1 ? '' : 's'} and the other cells of this context, through the {live.nBlocks * 5}{' '}
            sublayers above.
          </p>
          <Readout>
            {binary ? (
              <>
                Here p = <span className="num text-ink">{pct(pred.probability!)}</span> for “{task.targetCol} = true”; the hidden label is{' '}
                <span className="num text-ink">{rawLabel}</span>.
              </>
            ) : (
              <>
                Here {task.targetCol} ≈ <span className="num text-ink">{fmtNum(pred.value!, 2)}</span>; the hidden label is{' '}
                <span className="num text-ink">{rawLabel}</span>.
              </>
            )}
          </Readout>
        </MathHead>
      }
    />
  );
}

function Big({ title, value, note, accent }: { title: string; value: string; note: string; accent?: boolean }) {
  return (
    <div className={cx('rounded-xl border px-4 py-3', accent ? 'border-accent bg-accent-soft' : 'border-border bg-surface')}>
      <div className="text-[14.5px] text-muted">{title}</div>
      <div className={cx('num mt-0.5 text-[26px] leading-tight font-semibold', accent ? 'text-accent-text' : 'text-ink')}>{value}</div>
      <div className="mt-1 text-[14px] leading-snug text-ink-2">{note}</div>
    </div>
  );
}

/** A 0..1 bar with the predicted probability and the true class. */
function SigmoidBar({ p, y }: { p: number; y: 0 | 1 }) {
  return (
    <div className="rounded-xl border border-border bg-surface px-4 py-3">
      <div className="relative h-3 rounded-full bg-surface-2">
        <div className="absolute inset-y-0 left-0 rounded-full bg-accent" style={{ width: `${p * 100}%` }} />
        <div className="absolute top-[-4px] h-5 w-0.5 bg-ink" style={{ left: '50%' }} aria-hidden="true" />
      </div>
      <div className="mt-1.5 flex justify-between text-[14px] text-muted">
        <span>0: false</span>
        <span>0.5</span>
        <span>1: true</span>
      </div>
      <p className="!mt-1.5 !mb-0 text-[15px] text-ink-2">
        Predicted {pct(p)}; the true class is <span className="font-semibold text-ink">{y === 1 ? 'true' : 'false'}</span>, so at a 0.5 threshold this
        prediction is {(p > 0.5 ? 1 : 0) === y ? 'right' : 'wrong'}. The paper scores rankings (AUROC), not thresholds.
      </p>
    </div>
  );
}
