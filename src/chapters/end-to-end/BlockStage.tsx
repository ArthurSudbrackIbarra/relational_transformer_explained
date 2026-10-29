/**
 * Stage 6: every operation of every block, as whole matrices of the live forward pass.
 *
 * Inside a block (rt/model.py:82-86) each of the four attention sublayers runs
 *   x ← x + MaskedAttention_l(RMSNorm_l(x), mask_l)                                 (model.py:84)
 * with MaskedAttention = W_q, W_k, W_v → per-head masked softmax → concat → W_o (model.py:20-47),
 * then the SwiGLU sublayer x ← x + W₂(SiLU(W₁ h) ⊙ W₃ h), h = RMSNorm_ffn(x) (model.py:50-59, 85).
 * The player walks all of them, one operation per step. Every matrix is the engine's capture
 * (ForwardResult.blocks[b]); the arithmetic shown for an entry is recomputed from the operands
 * on screen by ./math.ts, which math.test.ts checks against the capture.
 */
import { useMemo, useState, type ReactNode } from 'react';
import { Button, Cite, Figure, LAYER_INFO, LayerLabel, Segmented, StatusBadge, StepPlayer, TeX, cx, useStepPlayer } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { ATTN_ORDER, type AttnCapture, type AttnLayer, type FFNCapture } from '@/engine/types';
import { fmtNum, int, pct, plural, sub } from './format';
import { W, useE2E, type E2ELive } from './live';
import { axis } from '../shapes';
import { qk, tokD, tokFF, tokHead, weightT } from './dims';
import { DotLine, MatMul, Panel, fitScale, panelCell, type LinkedHover } from './MatMul';
import type { Cell } from './Matrix';
import {
  headPanel,
  headSlice,
  headSliceT,
  linearEntry,
  maskReasons,
  maskedSoftmaxRow,
  rowOf,
  scoreEntry,
  silu,
  weightedSumEntry,
  type MaskReason,
} from './math';
import { NormView } from './NormView';
import { HoverHint, MathHead, Readout, Sym, tok } from './opParts';
import { FocusLine, LoadingNote, OpLayout, useViewHover, useVisualWidth } from './shared';

// ─────────────────────────────────────────────────────────────────────────────
// Steps
// ─────────────────────────────────────────────────────────────────────────────

type Sub = AttnLayer | 'ffn';
const SUBS: Sub[] = [...ATTN_ORDER, 'ffn'];
type AttnOp = 'norm' | 'qkv' | 'scores' | 'mask' | 'softmax' | 'mix' | 'out';
type FfnOp = 'norm' | 'up' | 'gate' | 'down';
type Op = AttnOp | FfnOp;

const ATTN_OPS: { id: AttnOp; short: string }[] = [
  { id: 'norm', short: 'RMSNorm' },
  { id: 'qkv', short: 'Q, K, V' },
  { id: 'scores', short: 'Scores' },
  { id: 'mask', short: 'Mask' },
  { id: 'softmax', short: 'Softmax' },
  { id: 'mix', short: 'Weighted sum' },
  { id: 'out', short: 'W_o and residual' },
];
const FFN_OPS: { id: FfnOp; short: string }[] = [
  { id: 'norm', short: 'RMSNorm' },
  { id: 'up', short: 'W₁ and W₃' },
  { id: 'gate', short: 'SiLU gate' },
  { id: 'down', short: 'W₂ and residual' },
];
const opsOf = (s: Sub): { id: Op; short: string }[] => (s === 'ffn' ? FFN_OPS : ATTN_OPS);

interface StepRef {
  block: number;
  sub: Sub;
  op: Op;
  /** Index of the op inside its sublayer. */
  k: number;
}

function buildSteps(nBlocks: number): StepRef[] {
  const out: StepRef[] = [];
  for (let block = 0; block < nBlocks; block++) for (const sub of SUBS) opsOf(sub).forEach((o, k) => out.push({ block, sub, op: o.id, k }));
  return out;
}

const SUB_TITLE: Record<Sub, string> = {
  col: 'Column attention',
  feat: 'Feature attention',
  nbr: 'Neighbor attention',
  full: 'Full attention',
  ffn: 'SwiGLU feed-forward',
};
const SUB_SHORT: Record<Sub, string> = { col: 'col', feat: 'feat', nbr: 'nbr', full: 'full', ffn: 'FFN' };
/** Mid-sentence names (SwiGLU keeps its capitals). */
const SUB_PHRASE: Record<Sub, string> = {
  col: 'column attention',
  feat: 'feature attention',
  nbr: 'neighbor attention',
  full: 'full attention',
  ffn: 'SwiGLU feed-forward',
};

const REASON_TEXT: Record<MaskReason, string> = {
  'same-column': 'same column of the same table',
  'same-row': 'same row',
  'parent-row': 'the key’s row is a parent of the query’s row',
  'child-row': 'the key’s row is a child of the query’s row',
  any: 'full attention allows every non-padding pair',
};

// ─────────────────────────────────────────────────────────────────────────────
// The figure
// ─────────────────────────────────────────────────────────────────────────────

export function BlockStage() {
  const live = useE2E();
  const { fr, nBlocks, H } = live;
  const steps = useMemo(() => buildSteps(nBlocks), [nBlocks]);
  const player = useStepPlayer(steps.length, { intervalMs: 2600 });
  const { setIndex, setPlaying } = player;
  const step = steps[Math.min(player.index, steps.length - 1)];
  const [head, setHead] = useState(0);
  const [which, setWhich] = useState<'q' | 'k' | 'v'>('q');
  const [upWhich, setUpWhich] = useState<'w1' | 'w3'>('w1');
  // a new step, head, projection or context drops the hovered entry
  const [hover, setHover] = useViewHover<LinkedHover>([player.index, head, which, upWhich, live.ctx]);
  const goSub = (block: number, sub: Sub) => {
    setPlaying(false);
    setIndex(steps.findIndex((s) => s.block === block && s.sub === sub));
  };
  const goOp = (k: number) => {
    setPlaying(false);
    setIndex(steps.findIndex((s) => s.block === step.block && s.sub === step.sub && s.k === k));
  };
  const ops = opsOf(step.sub);
  const headOps: Op[] = ['scores', 'mask', 'softmax', 'mix', 'out'];

  return (
    <Figure
      wide
      title={
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span>
            Block {step.block + 1} ·{' '}
            {step.sub === 'ffn' ? (
              SUB_TITLE.ffn
            ) : (
              <>
                <LayerLabel layer={step.sub} /> attention
              </>
            )}{' '}
            · {ops[step.k].short}
          </span>
          {live.assets.weightsKind === 'init' && <StatusBadge status="warning">untrained weights</StatusBadge>}
        </span>
      }
      caption={
        <>
          Every matrix comes from the live forward pass of the mini model on this context, one row per token; the followed token’s row is
          outlined in accent. Hover or focus any entry to see the numbers it is computed from, and click a token row to follow that token. Signed
          values are red (positive) and blue (negative) around a gray zero, each matrix on its own scale; attention weights use the blue scale from 0.
        </>
      }
    >
      <SublayerRail live={live} block={step.block} sub={step.sub} onPick={goSub} />
      <StepPlayer
        {...player}
        unit="step"
        className="mt-3"
        label="Play through every operation of every block"
        formatCounter={(i, n) => `step ${i + 1} / ${n}`}
        describe={(i) => {
          const s = steps[i];
          return (
            <>
              Block {s.block + 1}, {SUB_PHRASE[s.sub]}: {opsOf(s.sub)[s.k].short}.
            </>
          );
        }}
      />
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
        <OpRail ops={ops} k={step.k} onPick={goOp} />
        {step.sub !== 'ffn' && step.op === 'qkv' && (
          <Segmented
            label="Projection"
            size="sm"
            value={which}
            onChange={setWhich}
            options={[
              { value: 'q', label: 'Q' },
              { value: 'k', label: 'K' },
              { value: 'v', label: 'V' },
            ]}
          />
        )}
        {step.sub !== 'ffn' && headOps.includes(step.op) && (
          <Segmented
            label="Head"
            size="sm"
            value={head}
            onChange={setHead}
            options={Array.from({ length: H }, (_, h) => ({ value: h, label: `head ${h + 1}` }))}
          />
        )}
        {step.sub === 'ffn' && step.op === 'up' && (
          <Segmented
            label="Projection"
            size="sm"
            value={upWhich}
            onChange={setUpWhich}
            options={[
              { value: 'w1', label: 'W₁ → a' },
              { value: 'w3', label: 'W₃ → b' },
            ]}
          />
        )}
      </div>
      <FocusLine className="mt-3" />
      {!fr || !live.weights ? (
        <LoadingNote />
      ) : step.sub !== 'ffn' && fr.blocks[step.block].attn[step.sub].skipped ? (
        <SkippedNote live={live} />
      ) : (
        <StepBody live={live} step={step} head={Math.min(head, H - 1)} which={which} upWhich={upWhich} hover={hover} setHover={setHover} />
      )}
    </Figure>
  );
}

function SkippedNote({ live }: { live: E2ELive }) {
  return (
    <div className="mt-5 grid grid-cols-1 gap-2 text-[16.5px] text-ink-2">
      <p className="!mb-0">
        This sublayer is switched off by the playground’s inference-time ablation, so it computes nothing and the residual stream passes through
        unchanged.
      </p>
      <div>
        <Button size="sm" onClick={() => live.setDisabledLayers([])}>
          Switch every attention layer back on
        </Button>
      </div>
    </div>
  );
}

/** The blocks × 5 sublayers, each with a bar of ‖Δ‖ for the followed token (the size of the update it adds). */
function SublayerRail({ live, block, sub, onPick }: { live: E2ELive; block: number; sub: Sub; onPick: (b: number, s: Sub) => void }) {
  const { fr, focus, d, nBlocks } = live;
  const norms = useMemo(() => {
    if (!fr) return null;
    return Array.from({ length: nBlocks }, (_, b) =>
      SUBS.map((s) => {
        const o = s === 'ffn' ? fr.blocks[b].ffn.out : fr.blocks[b].attn[s].out;
        let n = 0;
        for (let j = 0; j < d; j++) n += o[focus * d + j] ** 2;
        return Math.sqrt(n);
      }),
    );
  }, [fr, focus, d, nBlocks]);
  const max = norms ? Math.max(...norms.flat(), 1e-9) : 1;
  return (
    <div>
      <div className="mb-1.5 text-[15px] text-ink-2">
        The {nBlocks * SUBS.length} sublayers in the order they run. Each bar shows how much that sublayer adds to the followed token’s vector
        (‖Δ‖).
      </div>
      <ol className="!m-0 grid grid-cols-1 gap-2 !p-0 sm:grid-cols-2 xl:grid-cols-4" style={{ listStyle: 'none' }} aria-label="Blocks and sublayers">
        {Array.from({ length: nBlocks }, (_, b) => (
          <li key={b} className="!m-0 min-w-0 rounded-lg border border-border bg-surface px-2 pt-1.5 pb-1 !pl-2">
            <div className="px-0.5 text-[14.5px] font-semibold text-ink">Block {b + 1}</div>
            <div className="mt-1 flex gap-1">
              {SUBS.map((s, si) => {
                const on = b === block && s === sub;
                const n = norms?.[b][si] ?? 0;
                return (
                  <button
                    key={s}
                    type="button"
                    onClick={() => onPick(b, s)}
                    aria-pressed={on}
                    aria-label={`Block ${b + 1}, ${SUB_TITLE[s]}${norms ? `, update norm ${fmtNum(n, 2)}` : ''}`}
                    title={`${SUB_TITLE[s]}${norms ? ` · ‖Δ‖ = ${fmtNum(n, 2)}` : ''}`}
                    className={cx(
                      'flex min-h-[44px] min-w-0 flex-1 flex-col items-center justify-between gap-1 rounded-md border px-0.5 pt-1.5 pb-1 transition-colors',
                      on ? 'border-accent bg-accent-soft text-accent-text' : 'border-transparent text-ink-2 hover:bg-surface-2',
                    )}
                  >
                    <span className="text-[14px] leading-none">{SUB_SHORT[s]}</span>
                    <span aria-hidden="true" className="flex h-[14px] w-full items-end justify-center">
                      <span className={cx('w-[60%] rounded-t-[2px]', on ? 'bg-accent' : 'bg-axis')} style={{ height: `${Math.max(8, (n / max) * 100)}%` }} />
                    </span>
                  </button>
                );
              })}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

function OpRail({ ops, k, onPick }: { ops: { short: string }[]; k: number; onPick: (k: number) => void }) {
  return (
    <ol className="!m-0 flex flex-wrap gap-1.5 !p-0" style={{ listStyle: 'none' }} aria-label="Operations of this sublayer">
      {ops.map((o, i) => (
        <li key={o.short} className="!m-0 !p-0">
          <button
            type="button"
            onClick={() => onPick(i)}
            aria-pressed={i === k}
            className={cx(
              'inline-flex h-10 items-center gap-1.5 rounded-lg border px-2.5 text-[15px] transition-colors',
              i === k ? 'border-accent bg-accent-soft font-semibold text-accent-text' : 'border-border bg-surface text-ink-2 hover:bg-surface-2',
            )}
          >
            <span className="num text-muted">{i + 1}</span>
            {o.short}
          </button>
        </li>
      ))}
    </ol>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// One step
// ─────────────────────────────────────────────────────────────────────────────

interface BodyProps {
  live: E2ELive;
  step: StepRef;
  head: number;
  which: 'q' | 'k' | 'v';
  upWhich: 'w1' | 'w3';
  hover: LinkedHover | null;
  setHover: (h: LinkedHover | null) => void;
}

interface AttnProps extends BodyProps {
  cap: AttnCapture;
  layer: AttnLayer;
  width: number;
}

interface FfnProps extends BodyProps {
  cap: FFNCapture;
  width: number;
}

function StepBody(p: BodyProps) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const width = useVisualWidth(size.width);
  const fr = p.live.fr!;
  const { step } = p;
  let content: ReactNode;
  if (step.sub === 'ffn') {
    const q: FfnProps = { ...p, cap: fr.blocks[step.block].ffn, width };
    content =
      step.op === 'norm' ? (
        <BlockNorm live={p.live} block={step.block} sub="ffn" xIn={q.cap.xIn} h={q.cap.h} width={width} hover={p.hover} setHover={p.setHover} />
      ) : step.op === 'up' ? (
        <UpOp {...q} />
      ) : step.op === 'gate' ? (
        <GateOp {...q} />
      ) : (
        <DownOp {...q} />
      );
  } else {
    const a: AttnProps = { ...p, cap: fr.blocks[step.block].attn[step.sub], layer: step.sub, width };
    switch (step.op as AttnOp) {
      case 'norm':
        content = (
          <BlockNorm live={p.live} block={step.block} sub={step.sub} xIn={a.cap.xIn} h={a.cap.h} width={width} hover={p.hover} setHover={p.setHover} />
        );
        break;
      case 'qkv':
        content = <QkvOp {...a} />;
        break;
      case 'scores':
        content = <ScoresOp {...a} />;
        break;
      case 'mask':
        content = <MaskOp {...a} />;
        break;
      case 'softmax':
        content = <SoftmaxOp {...a} />;
        break;
      case 'mix':
        content = <MixOp {...a} />;
        break;
      case 'out':
        content = <OutOp {...a} />;
        break;
    }
  }
  return (
    <div ref={ref} className="mt-4 min-w-0">
      {content}
    </div>
  );
}

function l2(v: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

/** max |v| over the first `rows` rows of a row-major [· × n] tensor. */
function maxAbsRows(buf: Float32Array, rows: number, n: number): number {
  let m = 0;
  for (let i = 0; i < rows * n; i++) m = Math.max(m, Math.abs(buf[i]));
  return m || 1;
}

/** max over the first `rows` rows. */
function maxRows(buf: Float32Array, rows: number, n: number): number {
  let m = 0;
  for (let i = 0; i < rows * n; i++) m = Math.max(m, buf[i]);
  return m || 1;
}

function useTokenRows(live: E2ELive) {
  return useMemo(() => ({ drawRows: live.numReal, rowSlots: live.tokenSlots, tokenRows: true }), [live.numReal, live.tokenSlots]);
}

/** C-hover → the entry; A/B hovers do not count (for element-wise panels every panel reports 'C'). */
const cellOf = (h: LinkedHover | null): Cell | null => (h && h.which === 'C' ? { r: h.r, c: h.c } : null);

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm (all five norms of a block)
// ─────────────────────────────────────────────────────────────────────────────

function BlockNorm({
  live,
  block,
  sub,
  xIn,
  h,
  width,
  hover,
  setHover,
}: {
  live: E2ELive;
  block: number;
  sub: Sub;
  xIn: Float32Array;
  h: Float32Array;
  width: number;
  hover: LinkedHover | null;
  setHover: (h: LinkedHover | null) => void;
}) {
  return (
    <NormView
      live={live}
      x={xIn}
      y={h}
      gamma={W(live.weights!, `blocks.${block}.norms.${sub}.weight`)}
      xTex="x"
      gammaTex={`\\gamma^{\\text{${sub}}}`}
      yTex="h"
      width={width}
      hover={hover}
      setHover={setHover}
      title={<>Normalize every row (RMSNorm)</>}
      cite={
        <>
          <code>nn.RMSNorm(d)</code>, one of five per block <Cite kind="code" ref="rt/model.py:71-73" />, applied before the sublayer{' '}
          <Cite kind="code" ref={sub === 'ffn' ? 'rt/model.py:85' : 'rt/model.py:84'} />.
        </>
      }
    >
      <p className="!mb-0">
        This is pre-norm: the normalized copy h feeds the {sub === 'ffn' ? 'feed-forward network' : 'attention'}, while the residual add later uses
        the un-normalized x. Every sublayer has its own gain γ.
      </p>
    </NormView>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Attention ops
// ─────────────────────────────────────────────────────────────────────────────

function QkvOp({ live, step, cap, layer, head, which, hover, setHover, width }: AttnProps) {
  const { d, dh, S, numReal, focus, weights, H } = live;
  const tokens = useTokenRows(live);
  const Wm = W(weights!, `blocks.${step.block}.attns.${layer}.w${which}.weight`);
  const out = which === 'q' ? cap.q : which === 'k' ? cap.k : cap.v;
  const L = which.toUpperCase();
  const band: [number, number] = [head * dh, (head + 1) * dh];
  const s = fitScale(width, d * 3 + d * 3 + 16 + 60);
  const hv = hover?.which === 'C' ? linearEntry(rowOf(cap.h, hover.r, d), Wm, d, hover.c) : null;
  const what = which === 'q' ? 'queries' : which === 'k' ? 'keys' : 'values';
  return (
    <OpLayout
      visual={
        <MatMul
          A={{ data: cap.h, rows: S, cols: d, ...tokens, name: <TeX tex="h" />, text: 'h', label: 'h, the normalized input', dims: tokD(live) }}
          B={{ data: Wm, rows: d, cols: d, transpose: true, bandCols: band, fadeOutside: false, name: <TeX tex={`W_${which}^{\\top}`} />, text: `W${which}ᵀ`, label: `W_${which} transposed`, dims: weightT(`the d = ${d} dimensions of h`, `the d = ${d} dimensions of ${L}: the ${H} heads’ ${dh} each, side by side`, `${d} × ${d}`) }}
          C={{ data: out, rows: S, cols: d, ...tokens, bandCols: band, fadeOutside: false, name: <TeX tex={L} />, text: L, label: `${L}, one row per token`, dims: { rows: axis.tokens(live.span), cols: `${axis.model(d)}; the ${H} heads’ ${dh} columns each, side by side (head ${head + 1}’s outlined)` } }}
          cellS={2.5 * s}
          cellK={3 * s}
          cellN={3 * s}
          focusRow={focus}
          onFocusRow={live.setFocus}
          hover={hover}
          setHover={setHover}
          corner={<>no bias</>}
        />
      }
      math={
        <MathHead
          title={`Project to ${what}`}
          tex={[`${L} = h\\,W_${which}^{\\top}`, `W_${which} \\in \\mathbb{R}^{${d}\\times ${d}}`]}
          cite={
            <>
              <code>wq</code>, <code>wk</code>, <code>wv</code> are <code>nn.Linear(d_model, d_model, bias=False)</code>{' '}
              <Cite kind="code" ref="rt/model.py:25-33" />. PyTorch stores W as [out, in] and computes h Wᵀ.
            </>
          }
        >
          <p className="!mb-0">
            One matrix product for all {int(numReal)} tokens at once. The {d} columns of {L} are {H} heads of {dh}: head {head + 1} owns columns {band[0]}–
            {band[1] - 1}, outlined <Cite kind="code" ref="rt/model.py:35-37" />. Switch between Q, K and V above.
          </p>
          <Readout>
            {hv && hover ? (
              <>
                {L}[{tok(live, hover.r)}, {hover.c}] = Σ<sub>k</sub> h[·, k]·W<sub>{which}</sub>[{hover.c}, k] = <DotLine terms={hv.terms} total={hv.value} />
              </>
            ) : hover?.which === 'A' ? (
              <>
                h[{tok(live, hover.r)}, {hover.c}] = <strong className="num text-ink">{fmtNum(cap.h[hover.r * d + hover.c])}</strong>. It meets row {hover.c} of W
                <sub>{which}</sub>
                <sup>⊤</sup> in every entry of this token’s row of {L}.
              </>
            ) : hover?.which === 'B' ? (
              <>
                W<sub>{which}</sub>[{hover.c}, {hover.r}] = <strong className="num text-ink">{fmtNum(Wm[hover.c * d + hover.r])}</strong>: the weight from input
                dimension {hover.r} to output dimension {hover.c}, the same for every token.
              </>
            ) : (
              <HoverHint />
            )}
          </Readout>
        </MathHead>
      }
    />
  );
}

function ScoresOp({ live, cap, head, hover, setHover, width }: AttnProps) {
  const { d, dh, S, numReal, focus } = live;
  const tokens = useTokenRows(live);
  const qh = useMemo(() => headSlice(cap.q, S, d, head, dh), [cap, S, d, head, dh]);
  const khT = useMemo(() => headSliceT(cap.k, S, d, head, dh), [cap, S, d, head, dh]);
  const sc = headPanel(cap.scores, S, head);
  const s = fitScale(width, dh * 5 + numReal * 2.5 + 16 + 60);
  const hv = hover?.which === 'C' ? scoreEntry(cap.q, cap.k, d, dh, head, hover.r, hover.c) : null;
  const h1 = head + 1;
  return (
    <OpLayout
      visual={
        <MatMul
          A={{ data: qh, rows: S, cols: dh, ...tokens, name: <TeX tex={`Q_{${h1}}`} />, text: `Q${sub(h1)}`, label: `Q of head ${h1}`, dims: tokHead(live, head, 'Q', 'query') }}
          B={{ data: khT, rows: dh, cols: S, drawCols: numReal, colSlots: live.tokenSlots, tokenCols: true, name: <TeX tex={`K_{${h1}}^{\\top}`} />, text: `K${sub(h1)}ᵀ`, label: `K of head ${h1}, transposed`, dims: { rows: `${axis.head(dh, d, live.H)}; here head ${h1}, dimensions ${head * dh}–${(head + 1) * dh - 1} of K`, cols: axis.tokens(live.span, 'key'), note: 'Kₕ transposed: in Kₕ every key token is a row; here it is a column, so Qₕ Kₕᵀ pairs every query with every key.' } }}
          C={{ data: sc, rows: S, cols: S, drawCols: numReal, ...tokens, colSlots: live.tokenSlots, name: <TeX tex={`S_{${h1}}`} />, label: `raw scores of head ${h1}`, dims: qk(live) }}
          cellS={2.5 * s}
          cellK={5 * s}
          cellN={2.5 * s}
          focusRow={focus}
          onFocusRow={live.setFocus}
          hover={hover}
          setHover={setHover}
          corner={<>÷ √{dh}</>}
        />
      }
      math={
        <MathHead
          title={`Scores of head ${h1}`}
          tex={`S_{qk} = Q_{q}\\cdot K_{k} \\,/\\, \\sqrt{${dh}}`}
          cite={
            <>
              <code>flex_attention</code> divides by its default √E, where E = d<sub>head</sub> ({dh} here, 32 in the paper){' '}
              <Cite kind="code" ref="rt/model.py:43" /> <Cite kind="spec" ref="§6.1" />.
            </>
          }
        >
          <p className="!mb-0">
            Head {h1}’s {dh}-column slice of Q against the same slice of K: one score for every query row and key column, {int(numReal)} ×{' '}
            {int(numReal)}, before any mask. The strips along the edges color each token by its table.
          </p>
          <Readout>
            {hv && hover ? (
              <>
                query {tok(live, hover.r)} · key {tok(live, hover.c)}: <DotLine terms={hv.terms} total={hv.dot} /> ÷ {fmtNum(Math.sqrt(dh), 0)} ={' '}
                <strong className="num text-ink">{fmtNum(hv.value)}</strong>
              </>
            ) : (
              <HoverHint />
            )}
          </Readout>
        </MathHead>
      }
    />
  );
}

function useBlocked(mask: Uint8Array): Uint8Array {
  return useMemo(() => {
    const out = new Uint8Array(mask.length);
    for (let i = 0; i < mask.length; i++) out[i] = mask[i] ? 0 : 1;
    return out;
  }, [mask]);
}

function MaskOp({ live, cap, layer, head, hover, setHover, width }: AttnProps) {
  const { S, numReal, focus, fr } = live;
  const tokens = useTokenRows(live);
  const mask = fr!.masks[layer];
  const blocked = useBlocked(mask);
  const sc = headPanel(cap.scores, S, head);
  const dom = useMemo(() => maxAbsRows(sc, numReal, S), [sc, numReal, S]);
  const c = panelCell(width, numReal, 3, 2.3);
  const cell = cellOf(hover);
  const onH = (x: Cell | null) => setHover(x ? { which: 'C', r: x.r, c: x.c } : null);
  const reasons = cell ? maskReasons(live.ctx.tokens, layer, cell.r, cell.c) : [];
  let allowedN = 0;
  for (let k = 0; k < S; k++) allowedN += mask[focus * S + k];
  const lname = LAYER_INFO[layer].name.toLowerCase();
  const common = { ...tokens, colSlots: live.tokenSlots, rows: S, cols: S, drawCols: numReal, dims: qk(live) };
  return (
    <OpLayout
      visual={
        <div className="flex flex-wrap items-start gap-x-4 gap-y-4">
          <Panel o={{ ...common, data: sc, domain: dom, name: <TeX tex={`S_{${head + 1}}`} />, label: 'raw scores' }} cellW={c} cellH={c} focusRow={focus} onFocusRow={live.setFocus} hover={cell} onHover={onH} />
          <Panel o={{ ...common, data: mask, scale: 'binary', name: <TeX tex={`M_{\\text{${layer}}}`} />, label: `${lname} mask` }} cellW={c} cellH={c} focusRow={focus} onFocusRow={live.setFocus} hover={cell} onHover={onH} />
          <Panel o={{ ...common, data: sc, domain: dom, blocked, name: <TeX tex={`\\tilde S_{${head + 1}}`} />, label: 'masked scores' }} cellW={c} cellH={c} focusRow={focus} onFocusRow={live.setFocus} hover={cell} onHover={onH} />
        </div>
      }
      math={
        <MathHead
          title={`Apply the ${lname} mask`}
          tex={`\\tilde S_{qk} = \\begin{cases} S_{qk} & M_{qk} = 1 \\\\ -\\infty & M_{qk} = 0 \\end{cases}`}
          cite={
            <>
              One boolean mask per layer, built once from the token ids and shared by every head and every block{' '}
              <Cite kind="code" ref="rt/model.py:89-101" /> <Cite kind="code" ref="rt/model.py:204-206" />; −∞ masking <Cite kind="paper" ref="p.5 §3.2" />.
            </>
          }
        >
          <p className="!mb-0">
            The followed token may attend to {plural(allowedN, 'key')} in this layer. Every blocked score becomes −∞, so the softmax gives it weight exactly
            0.
            {layer === 'nbr' &&
              ' A row with no allowed key at all (a cell whose row has no children in the context) gets no weights, and its output is exactly 0.'}
          </p>
          <Readout>
            {cell ? (
              <>
                query {tok(live, cell.r)}, key {tok(live, cell.c)}:{' '}
                {reasons.length ? (
                  <>
                    allowed ({reasons.map((r) => REASON_TEXT[r]).join('; ')}); the score stays{' '}
                    <strong className="num text-ink">{fmtNum(sc[cell.r * S + cell.c])}</strong>
                  </>
                ) : (
                  <>
                    blocked; {fmtNum(sc[cell.r * S + cell.c])} becomes <strong className="text-ink">−∞</strong>
                  </>
                )}
              </>
            ) : (
              <HoverHint />
            )}
          </Readout>
        </MathHead>
      }
    />
  );
}

function SoftmaxOp({ live, cap, layer, head, hover, setHover, width }: AttnProps) {
  const { S, numReal, focus, fr } = live;
  const tokens = useTokenRows(live);
  const mask = fr!.masks[layer];
  const blocked = useBlocked(mask);
  const sc = headPanel(cap.scores, S, head);
  const pr = headPanel(cap.probs, S, head);
  const pmax = useMemo(() => maxRows(pr, numReal, S), [pr, numReal, S]);
  const c = panelCell(width, numReal, 2, 2.4);
  const cell = cellOf(hover);
  const onH = (x: Cell | null) => setHover(x ? { which: 'C', r: x.r, c: x.c } : null);
  const row = pr.subarray(focus * S, (focus + 1) * S);
  const sm = maskedSoftmaxRow(sc.subarray(focus * S, (focus + 1) * S), mask.subarray(focus * S, (focus + 1) * S));
  const top = Array.from({ length: numReal }, (_, k) => k)
    .filter((k) => row[k] > 0)
    .sort((a, b) => row[b] - row[a] || a - b)
    .slice(0, 5);
  const hs = cell ? maskedSoftmaxRow(sc.subarray(cell.r * S, (cell.r + 1) * S), mask.subarray(cell.r * S, (cell.r + 1) * S)) : null;
  const common = { ...tokens, colSlots: live.tokenSlots, rows: S, cols: S, drawCols: numReal, dims: qk(live) };
  return (
    <OpLayout
      visual={
        <div className="flex flex-wrap items-start gap-x-4 gap-y-4">
          <Panel o={{ ...common, data: sc, blocked, name: <TeX tex={`\\tilde S_{${head + 1}}`} />, label: 'masked scores' }} cellW={c} cellH={c} focusRow={focus} onFocusRow={live.setFocus} hover={cell} onHover={onH} />
          <Panel o={{ ...common, data: pr, scale: 'sequential', domain: [0, pmax], blocked, name: <TeX tex={`P_{${head + 1}}`} />, label: 'attention weights' }} cellW={c} cellH={c} focusRow={focus} onFocusRow={live.setFocus} hover={cell} onHover={onH} />
        </div>
      }
      math={
        <MathHead
          title="Softmax over each row"
          tex={[`P_{qk} = \\exp(\\tilde S_{qk} - m_q) \\,/\\, Z_q`, `Z_q = \\textstyle\\sum_{j} \\exp(\\tilde S_{qj} - m_q)`, `m_q = \\max_{j} \\tilde S_{qj}`]}
          cite={
            <>
              The masked softmax inside <code>flex_attention</code> <Cite kind="code" ref="rt/model.py:43" />.
            </>
          }
        >
          <p className="!mb-0">
            Each row becomes weights that sum to 1 over its allowed keys{layer === 'nbr' ? '; a row with no allowed key stays all zero' : ''}.
            Subtracting the row maximum m first leaves the result unchanged and keeps the exponentials small. The followed token’s largest weights in
            head {head + 1}:
          </p>
          {top.length ? (
            <ol className="!mt-2 !mb-0 grid grid-cols-1 gap-1 !pl-0" style={{ listStyle: 'none' }}>
              {top.map((k) => (
                <li key={k} className="!m-0 flex items-center gap-2 !pl-0 text-[15px]">
                  <span className="min-w-0 flex-1 truncate">{live.tokenLabels[k]}</span>
                  <span className="h-2 shrink-0 rounded-full bg-accent" style={{ width: `${Math.max(3, row[k] * 80)}px` }} aria-hidden="true" />
                  <span className="num w-[56px] shrink-0 text-right text-ink">{pct(row[k])}</span>
                </li>
              ))}
            </ol>
          ) : (
            <p className="!mt-2 !mb-0 text-muted">The followed token has no allowed key in this layer: its weights are all 0.</p>
          )}
          <Readout>
            {cell && hs ? (
              pr[cell.r * S + cell.c] > 0 ? (
                <>
                  query {tok(live, cell.r)}, key {tok(live, cell.c)}: e<sup>{fmtNum(sc[cell.r * S + cell.c] - hs.max)}</sup> ÷ {fmtNum(hs.Z)} ={' '}
                  <strong className="num text-ink">{fmtNum(pr[cell.r * S + cell.c], 4)}</strong>
                </>
              ) : (
                <>
                  query {tok(live, cell.r)}, key {tok(live, cell.c)}: blocked, weight <strong className="num text-ink">0</strong>
                </>
              )
            ) : sm.Z > 0 ? (
              <>
                Followed row: m = {fmtNum(sm.max)}, Σ e<sup>S̃−m</sup> = {fmtNum(sm.Z)}. <HoverHint />
              </>
            ) : (
              <HoverHint />
            )}
          </Readout>
        </MathHead>
      }
    />
  );
}

function MixOp({ live, cap, layer, head, hover, setHover, width }: AttnProps) {
  const { d, dh, S, numReal, focus, H, fr } = live;
  const tokens = useTokenRows(live);
  const blocked = useBlocked(fr!.masks[layer]);
  const pr = headPanel(cap.probs, S, head);
  const pmax = useMemo(() => maxRows(pr, numReal, S), [pr, numReal, S]);
  const vh = useMemo(() => headSlice(cap.v, S, d, head, dh), [cap, S, d, head, dh]);
  const oh = useMemo(() => headSlice(cap.headOut, S, d, head, dh), [cap, S, d, head, dh]);
  const s = fitScale(width, numReal * 2.4 + dh * 5 + 16 + 60);
  const hv = hover?.which === 'C' ? weightedSumEntry(pr.subarray(hover.r * S, (hover.r + 1) * S), cap.v, d, dh, head, hover.c) : null;
  const h1 = head + 1;
  return (
    <OpLayout
      visual={
        <MatMul
          A={{ data: pr, rows: S, cols: S, drawCols: numReal, ...tokens, colSlots: live.tokenSlots, tokenCols: true, scale: 'sequential', domain: [0, pmax], blocked, name: <TeX tex={`P_{${h1}}`} />, text: `P${sub(h1)}`, label: 'attention weights', dims: qk(live) }}
          B={{ data: vh, rows: S, cols: dh, drawRows: numReal, rowSlots: live.tokenSlots, labelRows: true, name: <TeX tex={`V_{${h1}}`} />, text: `V${sub(h1)}`, label: `V of head ${h1}`, dims: tokHead(live, head, 'V', 'key') }}
          C={{ data: oh, rows: S, cols: dh, ...tokens, name: <TeX tex={`O_{${h1}}`} />, label: `output of head ${h1}`, dims: { rows: axis.tokens(live.span, 'query'), cols: `${axis.head(dh, d, H)}; head ${h1}’s output fills dimensions ${head * dh}–${(head + 1) * dh - 1} of the concatenated heads` } }}
          cellS={2.4 * s}
          cellK={2.4 * s}
          cellN={5 * s}
          focusRow={focus}
          onFocusRow={live.setFocus}
          hover={hover}
          setHover={setHover}
        />
      }
      math={
        <MathHead
          title="Weighted sum of the values"
          tex={`O_{q} = \\sum_{k} P_{qk}\\, V_{k}`}
          cite={
            <>
              Inside <code>flex_attention</code> <Cite kind="code" ref="rt/model.py:43" />; then the heads are laid side by side,{' '}
              <code>"b h s d -&gt; b s (h d)"</code> <Cite kind="code" ref="rt/model.py:45" />.
            </>
          }
        >
          <p className="!mb-0">
            Each token’s output for head {h1} averages the value rows, weighted by its row of P; a blocked key has weight 0 and adds nothing. The {H}{' '}
            heads’ outputs, {dh} columns each, are then concatenated into one {d}-wide row per token.
          </p>
          <Readout>
            {hv && hover ? (
              <>
                O[{tok(live, hover.r)}, {hover.c}] = Σ P·V over {plural(hv.terms.length, 'allowed key')}: <DotLine terms={hv.terms} total={hv.value} />
              </>
            ) : (
              <HoverHint />
            )}
          </Readout>
        </MathHead>
      }
    />
  );
}

function ResidualRow({ live, xIn, out, xOut, width, last }: { live: E2ELive; xIn: Float32Array; out: Float32Array; xOut: Float32Array; width: number; last: string }) {
  const { d, S, numReal, focus } = live;
  const tokens = useTokenRows(live);
  const dom = useMemo(() => Math.max(maxAbsRows(xIn, numReal, d), maxAbsRows(xOut, numReal, d)), [xIn, xOut, numReal, d]);
  const c = 2.5 * fitScale(width, (d * 2.5 + 30) * 3);
  return (
    <div>
      <div className="mb-2 text-[15.5px] font-semibold text-ink">The residual add, on one shared color scale</div>
      <div className="flex flex-wrap items-start gap-x-3 gap-y-3">
        <Panel o={{ data: xIn, rows: S, cols: d, ...tokens, domain: dom, name: <TeX tex="x" />, text: 'x', label: 'x entering the sublayer', dims: tokD(live) }} cellW={c} cellH={c} focusRow={focus} onFocusRow={live.setFocus} />
        <Sym>+</Sym>
        <Panel o={{ data: out, rows: S, cols: d, ...tokens, domain: dom, name: <TeX tex="\Delta" />, text: 'Δ', label: 'the update', dims: tokD(live) }} cellW={c} cellH={c} focusRow={focus} onFocusRow={live.setFocus} />
        <Sym>=</Sym>
        <Panel o={{ data: xOut, rows: S, cols: d, ...tokens, domain: dom, name: <TeX tex="x'" />, text: 'x′', label: last, dims: tokD(live) }} cellW={c} cellH={c} focusRow={focus} onFocusRow={live.setFocus} />
      </div>
    </div>
  );
}

function OutOp({ live, step, cap, layer, head, hover, setHover, width }: AttnProps) {
  const { d, dh, S, focus, weights, H } = live;
  const tokens = useTokenRows(live);
  const Wo = W(weights!, `blocks.${step.block}.attns.${layer}.wo.weight`);
  const band: [number, number] = [head * dh, (head + 1) * dh];
  const s = fitScale(width, d * 3 + d * 3 + 16 + 60);
  const hv = hover?.which === 'C' ? linearEntry(rowOf(cap.headOut, hover.r, d), Wo, d, hover.c) : null;
  return (
    <OpLayout
      visual={
        <div className="grid grid-cols-1 gap-6">
          <MatMul
            A={{ data: cap.headOut, rows: S, cols: d, ...tokens, bandCols: band, fadeOutside: false, name: <TeX tex={`[O_1 \\cdots O_{${H}}]`} />, text: 'concat', label: 'the heads concatenated', dims: { rows: axis.tokens(live.span), cols: `one per dimension of the ${H} heads’ outputs side by side: head j’s ${dh} fill columns ${dh}·(j − 1) to ${dh}·j − 1, so ${H} × ${dh} = ${d}` } }}
            B={{ data: Wo, rows: d, cols: d, transpose: true, name: <TeX tex="W_o^{\top}" />, text: 'Woᵀ', label: 'W_o transposed', dims: weightT(`the ${d} dimensions of the concatenated head outputs`, `the model width d = ${d}`, `${d} × ${d}`) }}
            C={{ data: cap.out, rows: S, cols: d, ...tokens, name: <TeX tex="\Delta" />, label: 'the update', dims: tokD(live) }}
            cellS={2.5 * s}
            cellK={3 * s}
            cellN={3 * s}
            focusRow={focus}
            onFocusRow={live.setFocus}
            hover={hover}
            setHover={setHover}
            corner={<>no bias</>}
          />
          <ResidualRow live={live} xIn={cap.xIn} out={cap.out} xOut={cap.xOut} width={width} last="x leaving the sublayer" />
        </div>
      }
      math={
        <MathHead
          title="Output projection and the residual add"
          tex={`x' = x + \\underbrace{[O_1\\,\\cdots\\,O_{${H}}]\\,W_o^{\\top}}_{\\Delta}`}
          cite={
            <>
              <code>wo</code> has no bias <Cite kind="code" ref="rt/model.py:28" />, applied after the concatenation <Cite kind="code" ref="rt/model.py:45-46" />;
              pre-norm residual <Cite kind="code" ref="rt/model.py:84" />.
            </>
          }
        >
          <p className="!mb-0">
            W<sub>o</sub> mixes the heads back into one {d}-d update Δ per token, which is added to the token’s vector. The norm only shaped the
            attention’s input; the residual stream x itself is never normalized inside a block.
          </p>
          <FocusNorms live={live} xIn={cap.xIn} out={cap.out} xOut={cap.xOut} />
          <Readout>
            {hv && hover ? (
              <>
                Δ[{tok(live, hover.r)}, {hover.c}] = Σ<sub>k</sub> concat[·, k]·W<sub>o</sub>[{hover.c}, k] = <DotLine terms={hv.terms} total={hv.value} />
              </>
            ) : (
              <HoverHint />
            )}
          </Readout>
        </MathHead>
      }
    />
  );
}

function FocusNorms({ live, xIn, out, xOut }: { live: E2ELive; xIn: Float32Array; out: Float32Array; xOut: Float32Array }) {
  const { d, focus } = live;
  return (
    <p className="!mt-2 !mb-0">
      Followed token: ‖x‖ = <span className="num">{fmtNum(l2(rowOf(xIn, focus, d)), 2)}</span>, ‖Δ‖ ={' '}
      <span className="num">{fmtNum(l2(rowOf(out, focus, d)), 2)}</span>, ‖x′‖ = <span className="num">{fmtNum(l2(rowOf(xOut, focus, d)), 2)}</span>.
    </p>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU ops
// ─────────────────────────────────────────────────────────────────────────────

function UpOp({ live, step, cap, upWhich, hover, setHover, width }: FfnProps) {
  const { d, dFF, S, focus, weights } = live;
  const tokens = useTokenRows(live);
  const Wm = W(weights!, `blocks.${step.block}.ffn.${upWhich}.weight`);
  const out = upWhich === 'w1' ? cap.a : cap.b;
  const s = fitScale(width, d * 3 + dFF * 1.5 + 16 + 60);
  const hv = hover?.which === 'C' ? linearEntry(rowOf(cap.h, hover.r, d), Wm, d, hover.c) : null;
  const sym = upWhich === 'w1' ? 'a' : 'b';
  const wn = upWhich === 'w1' ? 'W_1' : 'W_3';
  return (
    <OpLayout
      visual={
        <MatMul
          A={{ data: cap.h, rows: S, cols: d, ...tokens, name: <TeX tex="h" />, text: 'h', label: 'normalized input', dims: tokD(live) }}
          B={{ data: Wm, rows: d, cols: dFF, transpose: true, name: <TeX tex={`${wn}^{\\top}`} />, text: upWhich === 'w1' ? 'W₁ᵀ' : 'W₃ᵀ', label: `${wn} transposed`, dims: weightT(`the d = ${d} dimensions of h`, `the d_ff = ${dFF} hidden units of the SwiGLU (${dFF / d} × d)`, `${dFF} × ${d}`) }}
          C={{ data: out, rows: S, cols: dFF, ...tokens, name: <TeX tex={sym} />, label: `${sym}, one row per token`, dims: tokFF(live) }}
          cellS={2.5 * s}
          cellK={3 * s}
          cellN={1.5 * s}
          focusRow={focus}
          onFocusRow={live.setFocus}
          hover={hover}
          setHover={setHover}
          corner={<>no bias</>}
        />
      }
      math={
        <MathHead
          title="Two projections up to the hidden width"
          tex={`a = h\\,W_1^{\\top},\\quad b = h\\,W_3^{\\top}`}
          cite={
            <>
              <code>w1</code>, <code>w3</code> are <code>nn.Linear(d_model, d_ff, bias=False)</code> <Cite kind="code" ref="rt/model.py:54-56" />.
            </>
          }
        >
          <p className="!mb-0">
            The SwiGLU widens every token from {d} to {dFF} dimensions twice, with two different {dFF} × {d} matrices W₁ and W₃; switch between them
            above. Each token is transformed on its own: in this sublayer no token sees another.
          </p>
          <Readout>
            {hv && hover ? (
              <>
                {sym}[{tok(live, hover.r)}, {hover.c}] = Σ<sub>k</sub> h[·, k]·{upWhich === 'w1' ? 'W₁' : 'W₃'}[{hover.c}, k] ={' '}
                <DotLine terms={hv.terms} total={hv.value} />
              </>
            ) : (
              <HoverHint />
            )}
          </Readout>
        </MathHead>
      }
    />
  );
}

function GateOp({ live, cap, hover, setHover, width }: FfnProps) {
  const { dFF, S, focus } = live;
  const tokens = useTokenRows(live);
  const c = Math.max(0.7, 1.5 * fitScale(width, (dFF * 1.5 + 30) * 3, 0.3));
  const cell = cellOf(hover);
  const onH = (x: Cell | null) => setHover(x ? { which: 'C', r: x.r, c: x.c } : null);
  const av = cell ? cap.a[cell.r * dFF + cell.c] : 0;
  const bv = cell ? cap.b[cell.r * dFF + cell.c] : 0;
  return (
    <OpLayout
      visual={
        <div className="flex flex-wrap items-start gap-x-3 gap-y-4">
          <Panel o={{ data: cap.a, rows: S, cols: dFF, ...tokens, name: <TeX tex="a" />, label: 'a', dims: tokFF(live) }} cellW={c} cellH={2.2} focusRow={focus} onFocusRow={live.setFocus} hover={cell} onHover={onH} />
          <Panel o={{ data: cap.b, rows: S, cols: dFF, ...tokens, name: <TeX tex="b" />, label: 'b', dims: tokFF(live) }} cellW={c} cellH={2.2} focusRow={focus} onFocusRow={live.setFocus} hover={cell} onHover={onH} />
          <Panel o={{ data: cap.gated, rows: S, cols: dFF, ...tokens, name: <TeX tex="g" />, label: 'g, the gated hidden layer', dims: tokFF(live) }} cellW={c} cellH={2.2} focusRow={focus} onFocusRow={live.setFocus} hover={cell} onHover={onH} />
        </div>
      }
      math={
        <MathHead
          title="The SiLU gate"
          tex={[`g = \\mathrm{SiLU}(a) \\odot b`, `\\mathrm{SiLU}(z) = z \\,/\\, (1 + e^{-z})`]}
          cite={
            <>
              <code>F.silu(self.w1(x)) * self.w3(x)</code> <Cite kind="code" ref="rt/model.py:59" />; “gated MLPs with SiLU activation, as found in the
              LLaMA architecture” <Cite kind="paper" ref="p.6 §4.1" />.
            </>
          }
        >
          <p className="!mb-0">
            Entry by entry, SiLU passes large positive values of a almost unchanged and pushes negative ones toward 0. The result then multiplies
            the matching entry of b, so SiLU(a) controls how much of each entry of b gets through.
          </p>
          <Readout>
            {cell ? (
              <>
                g[{tok(live, cell.r)}, {cell.c}] = SiLU({fmtNum(av)}) × {fmtNum(bv, 3, true)} = {fmtNum(silu(av), 3, true)} × {fmtNum(bv, 3, true)} ={' '}
                <strong className="num text-ink">{fmtNum(cap.gated[cell.r * dFF + cell.c])}</strong>
              </>
            ) : (
              <HoverHint />
            )}
          </Readout>
        </MathHead>
      }
    />
  );
}

function DownOp({ live, step, cap, hover, setHover, width }: FfnProps) {
  const { d, dFF, S, focus, weights, nBlocks } = live;
  const tokens = useTokenRows(live);
  const W2 = W(weights!, `blocks.${step.block}.ffn.w2.weight`);
  const s = fitScale(width, dFF * 1.5 + d * 3 + 16 + 60);
  const hv = hover?.which === 'C' ? linearEntry(rowOf(cap.gated, hover.r, dFF), W2, dFF, hover.c) : null;
  return (
    <OpLayout
      visual={
        <div className="grid grid-cols-1 gap-6">
          <MatMul
            A={{ data: cap.gated, rows: S, cols: dFF, ...tokens, name: <TeX tex="g" />, text: 'g', label: 'gated hidden layer', dims: tokFF(live) }}
            B={{ data: W2, rows: dFF, cols: d, transpose: true, name: <TeX tex="W_2^{\top}" />, text: 'W₂ᵀ', label: 'W2 transposed', dims: weightT(`the d_ff = ${dFF} hidden units of g`, `the model width d = ${d}`, `${d} × ${dFF}`) }}
            C={{ data: cap.out, rows: S, cols: d, ...tokens, name: <TeX tex="\Delta" />, label: 'the update', dims: tokD(live) }}
            cellS={2.5 * s}
            cellK={1.5 * s}
            cellN={3 * s}
            focusRow={focus}
            onFocusRow={live.setFocus}
            hover={hover}
            setHover={setHover}
            corner={<>no bias</>}
          />
          <ResidualRow live={live} xIn={cap.xIn} out={cap.out} xOut={cap.xOut} width={width} last={step.block + 1 < nBlocks ? 'x leaving the block' : 'x leaving the last block'} />
        </div>
      }
      math={
        <MathHead
          title="Back down to d, and the residual add"
          tex={[`x' = x + \\underbrace{g\\,W_2^{\\top}}_{\\Delta}`, `W_2 \\in \\mathbb{R}^{${d}\\times ${dFF}}`]}
          cite={
            <>
              <code>w2</code> is <code>nn.Linear(d_ff, d_model, bias=False)</code> <Cite kind="code" ref="rt/model.py:55" />; residual{' '}
              <Cite kind="code" ref="rt/model.py:85" />.
            </>
          }
        >
          <p className="!mb-0">
            W₂ maps the {dFF} gated values back to a {d}-d update. After this add the block is done:{' '}
            {step.block + 1 < nBlocks
              ? `x enters block ${step.block + 2}, which runs the same five sublayers with its own weights and the same four masks.`
              : 'x goes to the final RMSNorm and the decoders (stage 7).'}
          </p>
          <FocusNorms live={live} xIn={cap.xIn} out={cap.out} xOut={cap.xOut} />
          <Readout>
            {hv && hover ? (
              <>
                Δ[{tok(live, hover.r)}, {hover.c}] = Σ<sub>k</sub> g[·, k]·W₂[{hover.c}, k] = <DotLine terms={hv.terms} total={hv.value} />
              </>
            ) : (
              <HoverHint />
            )}
          </Readout>
        </MathHead>
      }
    />
  );
}
