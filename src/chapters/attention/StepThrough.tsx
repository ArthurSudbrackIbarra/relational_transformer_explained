/**
 * Section 2: one masked multi-head attention sublayer, step by step, for one query token, with the
 * live tensors of the trained mini model (ForwardResult.blocks[b].attn[l], a mirror of
 * rt/model.py:20-47 inside the pre-norm residual of model.py:84):
 *
 *   1 x            the residual stream entering the sublayer
 *   2 RMSNorm      h = RMSNorm_l(x)
 *   3 Q, K, V      q = W_q h, k = W_k h, v = W_v h (64×64, no bias); head j = dims [16j, 16j+16)
 *   4 scores       q_j·k_j / √16 for every key
 *   5 mask         keys the layer forbids → −∞
 *   6 softmax      weights over the allowed keys (sum 1, or all 0)
 *   7 output       Σ p·v per head, concat heads, W_o, residual add
 *
 * The only arithmetic done here is re-derivation for display (e.g. Σ q_i k_i shown next to the
 * captured score); every value comes from the engine's capture or the loaded weights.
 */
import { ChevronDown } from 'lucide-react';
import { motion } from 'motion/react';
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  Button,
  Cite,
  Figure,
  LAYER_INFO,
  LayerLabel,
  ShapeTag,
  StatusBadge,
  StepPlayer,
  TableSwatch,
  TeX,
  VectorStrip,
  cx,
  sequentialCss,
  useStepPlayer,
} from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { ATTN_ORDER } from '@/engine/types';
import { useResolvedTheme } from '@/styles/theme';
import { KeyBars, type BarStage } from './KeyBars';
import { SelectionBar } from './Controls';
import { provenanceSentence, useProvenance } from './provenance';
import { headRow, useQueryView, type AttnLive, type QueryView } from './useAttnLive';
import { int, maxAbsOf, norm2, pctw, rowName, sgn, texNum, topK, valueOf } from './utils';
import { axis, vec } from '../shapes';
import { KV, MatrixCanvas, SeqStrip, VStrip } from './viz';

const STAGES = [
  { id: 'x', short: 'x in' },
  { id: 'norm', short: 'RMSNorm' },
  { id: 'qkv', short: 'Q, K, V' },
  { id: 'scores', short: 'scores' },
  { id: 'mask', short: 'mask' },
  { id: 'softmax', short: 'softmax' },
  { id: 'out', short: 'output' },
] as const;

export function StepThrough({ live }: { live: AttnLive }) {
  const { fr, assets, block, layer, head, query, ctx, setDisabledLayers } = live;
  const qv = useQueryView(live);
  const player = useStepPlayer(STAGES.length, { intervalMs: 2800 });
  const { setPlaying, setIndex } = player;
  const [picked, setPicked] = useState<number | null>(null);
  const narrow = useNarrow();
  const selKey = `${block}|${layer}|${head}|${query}|${ctx.params.seedNode}|${ctx.params.L}|${ctx.params.w}|${ctx.params.rngSeed}`;
  // A new selection resets the inspected key and stops the player, so it never runs past what the reader is studying.
  useEffect(() => {
    setPicked(null);
    setPlaying(false);
  }, [selKey, setPlaying]);
  const pickStage = (i: number) => {
    setPlaying(false);
    setIndex(i);
  };
  const inspect = (k: number) => {
    setPlaying(false);
    setPicked(k);
  };

  const auto = useMemo(() => {
    if (!qv || qv === 'skipped') return query;
    return qv.allowedList.length ? topK(qv.probs, qv.allowedList, 1)[0] : query;
  }, [qv, query]);
  const inspected = picked ?? auto;
  const stage = STAGES[player.index].id;
  const prov = useProvenance(live);

  return (
    <Figure
      wide
      title={
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span>
            Block {block + 1} · <LayerLabel layer={layer} /> attention · head {head + 1}
          </span>
          {assets.weightsKind === 'trained' && (
            <span title={prov ? provenanceSentence(prov) : undefined}>
              <StatusBadge status="good">
                {narrow ? 'trained' : `trained mini model (${assets.ckpt}`}
                {prov?.kind === 'zero-shot' ? (narrow ? ', zero-shot' : `, zero-shot on ${prov.dbTitle}`) : ''}
                {narrow ? '' : ')'}
              </StatusBadge>
            </span>
          )}
          {assets.weightsKind === 'init' && <StatusBadge status="warning">untrained weights: values from a random initialization</StatusBadge>}
        </span>
      }
      controls={narrow ? <CompactSelection live={live} /> : <SelectionBar live={live} />}
      caption={
        <>
          Every vector and number is read from the forward pass of the live context, computed in your browser. Use the player or click a
          stage; click any bar (or use ← →) to inspect a key. The accent triangle marks the query’s own position: in the column, feature and full
          layers a token may attend to itself <Cite kind="spec" ref="§6.2" />.
        </>
      }
    >
      <StageRail index={player.index} onPick={pickStage} compact={narrow} />
      <StepPlayer {...player} unit="stage" className="mt-3" describe={(i) => describe(STAGES[i].id, live, qv)} label="Play through the stages of the attention sublayer" />
      {!fr || !qv ? (
        <p className="mt-5 text-[16.5px] text-ink-2">
          {assets.status === 'error' ? `The forward pass is unavailable: ${assets.error}` : 'Loading the trained weights and running the forward pass…'}
        </p>
      ) : qv === 'skipped' ? (
        <div className="mt-5 grid gap-2 text-[16.5px] text-ink-2">
          <p className="!mb-0">
            This sublayer is switched off by the playground’s inference-time ablation, so it computes nothing and the residual stream passes
            through unchanged.
          </p>
          <div>
            <Button size="sm" onClick={() => setDisabledLayers([])}>
              Switch every attention layer back on
            </Button>
          </div>
        </div>
      ) : (
        <div className="mt-5 grid gap-6 lg:grid-cols-[minmax(0,1fr)_300px]">
          <StageVisual stage={stage} live={live} qv={qv} inspected={inspected} onInspect={inspect} />
          <aside className="min-w-0 rounded-xl bg-surface-2 px-4 py-3.5 text-[15.5px] leading-snug" aria-live="polite">
            <StageMath stage={stage} live={live} qv={qv} inspected={inspected} />
          </aside>
        </div>
      )}
    </Figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

/** Phone-width layout switch (below Tailwind's sm breakpoint). */
function useNarrow(): boolean {
  const q = '(max-width: 639.98px)';
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && !!window.matchMedia?.(q).matches);
  useEffect(() => {
    const m = window.matchMedia?.(q);
    if (!m) return;
    const on = () => setNarrow(m.matches);
    on();
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, []);
  return narrow;
}

/** On a phone: one summary line that opens the block / layer / head / query controls. */
function CompactSelection({ live }: { live: AttnLive }) {
  const { query, target, labels, ctx } = live;
  const t = ctx.tokens[query];
  return (
    <details className="group w-full min-w-0 rounded-lg border border-border bg-surface">
      <summary className="flex min-h-10 cursor-pointer list-none items-center justify-between gap-2 rounded-lg px-3 py-1.5 text-[15px] focus-visible:outline-2 focus-visible:outline-accent [&::-webkit-details-marker]:hidden">
        <span className="min-w-0 truncate text-ink-2">
          query <span className="num text-ink">#{query}</span> {query === target ? '(masked target)' : `${labels[query]} = ${t ? valueOf(t) : ''}`}
        </span>
        <span className="inline-flex shrink-0 items-center gap-1 font-medium text-accent-text">
          change <ChevronDown aria-hidden="true" size={15} className="transition-transform group-open:rotate-180" />
        </span>
      </summary>
      <div className="flex flex-wrap items-end gap-x-4 gap-y-3 border-t border-grid px-3 py-3">
        <SelectionBar live={live} />
      </div>
    </details>
  );
}

function StageRail({ index, onPick, compact }: { index: number; onPick: (i: number) => void; compact?: boolean }) {
  const ref = useRef<HTMLOListElement | null>(null);
  // In the one-row (phone) rail, keep the current stage in view without scrolling the page.
  useEffect(() => {
    const ol = ref.current;
    if (!compact || !ol) return;
    const li = ol.children[index] as HTMLElement | undefined;
    if (!li) return;
    const left = li.offsetLeft - (ol.clientWidth - li.offsetWidth) / 2;
    ol.scrollTo({ left: Math.max(0, left), behavior: 'smooth' });
  }, [index, compact]);
  return (
    <ol
      ref={ref}
      className={cx('!m-0 flex gap-1.5 !p-0', compact ? 'scroll-quiet flex-nowrap overflow-x-auto pb-1' : 'flex-wrap')}
      style={{ listStyle: 'none' }}
      aria-label="Stages of the sublayer"
    >
      {STAGES.map((s, i) => {
        const on = i === index;
        const done = i < index;
        return (
          <li key={s.id} className="!m-0 !p-0 shrink-0">
            <button
              type="button"
              onClick={() => onPick(i)}
              aria-current={on ? 'step' : undefined}
              className={cx(
                'flex h-8 items-center gap-1.5 rounded-full border px-2.5 text-[15px] font-medium whitespace-nowrap transition-colors',
                on ? 'border-accent bg-accent text-accent-ink' : done ? 'border-accent/40 bg-accent-soft text-accent-text' : 'border-border bg-surface text-ink-2 hover:bg-surface-2',
              )}
            >
              <span className={cx('num inline-flex h-4.5 w-4.5 items-center justify-center rounded-full text-[13px]', on ? 'bg-accent-ink/20' : 'bg-surface-3')}>{i + 1}</span>
              {s.short}
            </button>
          </li>
        );
      })}
    </ol>
  );
}

function describe(stage: (typeof STAGES)[number]['id'], live: AttnLive, qv: QueryView | null | 'skipped'): ReactNode {
  const { block, layer, head, query, labels, dh, d, H } = live;
  const name = LAYER_INFO[layer].name.toLowerCase();
  const n = qv && qv !== 'skipped' ? qv.allowedList.length : 0;
  switch (stage) {
    case 'x':
      return (
        <>
          <strong className="font-semibold">x.</strong> Token #{query} ({labels[query]}) arrives at the {name} sublayer of block {block + 1} as a{' '}
          {d}-dimensional vector: the residual stream. Every token of the context enters at the same time.
        </>
      );
    case 'norm':
      return (
        <>
          <strong className="font-semibold">RMSNorm.</strong> Divide x by its root-mean-square, then multiply by this sublayer’s learned gain γ.
          Only the normalized copy h goes into attention; x itself waits for the residual add.
        </>
      );
    case 'qkv':
      return (
        <>
          <strong className="font-semibold">Q, K, V.</strong> Three {d}×{d} matrices without bias turn h into a query, a key and a value for every
          token. Head {head + 1} of {H} reads dimensions {head * dh} to {(head + 1) * dh - 1} of each.
        </>
      );
    case 'scores':
      return (
        <>
          <strong className="font-semibold">Scores.</strong> The query’s head slice is dotted with every key’s head slice and divided by √{dh}.
          Nothing is masked yet. That happens in the next step.
        </>
      );
    case 'mask':
      return (
        <>
          <strong className="font-semibold">Mask.</strong>{' '}
          {n === 0 ? (
            <>The {name} mask blocks every key of this query: all {live.S} scores become −∞.</>
          ) : (
            <>
              The {name} mask allows {n} of the {live.S} keys. Every other score becomes −∞.
            </>
          )}
        </>
      );
    case 'softmax':
      return n > 0 ? (
        <>
          <strong className="font-semibold">Softmax.</strong> Over the {n} allowed keys the scores become weights that sum to 1. The blocked keys get
          exactly 0, because e<sup>−∞</sup> = 0.
        </>
      ) : (
        <>
          <strong className="font-semibold">Softmax.</strong> No key is allowed, so there is nothing to normalize: the row stays all zeros and this
          head outputs exactly 0.
        </>
      );
    case 'out':
      return (
        <>
          <strong className="font-semibold">Output.</strong> Each head averages its value vectors with its weights. The {H} head outputs are
          concatenated, mixed by W<sub>o</sub>, and added to x.
        </>
      );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Visuals
// ─────────────────────────────────────────────────────────────────────────────

function StageVisual({
  stage,
  live,
  qv,
  inspected,
  onInspect,
}: {
  stage: (typeof STAGES)[number]['id'];
  live: AttnLive;
  qv: QueryView;
  inspected: number;
  onInspect: (k: number) => void;
}) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(260, size.width || 640);
  const kind = stage === 'scores' || stage === 'mask' || stage === 'softmax' ? 'bars' : stage;
  return (
    <div ref={ref} className="min-w-0">
      <motion.div key={kind} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
        {kind === 'x' && <XVisual live={live} qv={qv} W={W} />}
        {kind === 'norm' && <NormVisual live={live} qv={qv} />}
        {kind === 'qkv' && <QKVVisual live={live} qv={qv} W={W} inspected={inspected} />}
        {kind === 'bars' && <BarsVisual live={live} qv={qv} stage={stage as BarStage} inspected={inspected} onInspect={onInspect} />}
        {kind === 'out' && <OutVisual live={live} qv={qv} />}
      </motion.div>
    </div>
  );
}

function Eyebrow({ children }: { children: ReactNode }) {
  return <div className="mb-1.5 text-[14.5px] font-semibold text-ink-2">{children}</div>;
}

function XVisual({ live, qv, W }: { live: AttnLive; qv: QueryView; W: number }) {
  const { numReal, d, query, ctx } = live;
  const XT = useMemo(() => {
    const out = new Float32Array(d * numReal);
    const X = qv.cap.xIn;
    for (let s = 0; s < numReal; s++) for (let j = 0; j < d; j++) out[j * numReal + s] = X[s * d + j];
    return out;
  }, [qv.cap, d, numReal]);
  // Color domain: the 95th percentile of |x| (a few large dimensions would otherwise wash out the rest);
  // larger values are drawn at full color. Display only.
  const m = useMemo(() => {
    const a = Float32Array.from(XT, Math.abs).sort();
    return a.length ? Math.max(1e-6, a[Math.min(a.length - 1, Math.floor(a.length * 0.95))]) : 1;
  }, [XT]);
  const cellW = Math.max(1, Math.min(7, Math.floor(W / numReal)));
  const [hover, setHover] = useState<{ r: number; c: number; v: number } | null>(null);
  return (
    <div className="grid gap-5">
      <div>
        <Eyebrow>
          The residual stream entering the sublayer: one column of {d} numbers per token ({int(numReal)} tokens)
        </Eyebrow>
        <div className="scroll-quiet overflow-x-auto pt-1 pb-1">
          <MatrixCanvas
            data={XT}
            rows={d}
            cols={numReal}
            cellW={cellW}
            cellH={W < 500 ? 2 : 3}
            maxAbs={m}
            bandCols={[query, query + 1]}
            label={`Residual stream: ${numReal} tokens by ${d} dimensions; the query's column is outlined.`}
            onHover={setHover}
          />
        </div>
        <div className="num mt-1 flex min-h-[1.3em] flex-wrap items-baseline gap-x-2.5 text-[14px] text-muted">
          <ShapeTag
            rows={d}
            cols={numReal}
            shape={{
              rows: axis.model(d),
              cols: axis.tokens({ shown: numReal, L: ctx.params.L, real: numReal }),
              note: 'The code keeps this as x, one row per token (L × d); it is drawn transposed here, so each token is a column.',
            }}
          />
          <span>
            {hover ? (
              <>
                token #{hover.c} {live.labels[hover.c]} · dim {hover.r} = <span className="text-ink">{sgn(hover.v, 4)}</span>
              </>
            ) : (
              <>columns = tokens in context order · rows = the {d} dimensions · outlined: the query · color saturates at ±{m.toFixed(2)}</>
            )}
          </span>
        </div>
      </div>
      <div>
        <Eyebrow>The query’s column: x for token #{query}</Eyebrow>
        <VectorStrip values={qv.x} columns={d} cellHeight={18} label={`x (token #${query})`} shape={vec(`the query token #${query}’s x: one column of the matrix above`, axis.model(d))} />
      </div>
    </div>
  );
}

function NormVisual({ live, qv }: { live: AttnLive; qv: QueryView }) {
  const { d, cfg } = live;
  const scaled = useMemo(() => {
    let ms = 0;
    for (let i = 0; i < d; i++) ms += qv.x[i] * qv.x[i];
    const r = Math.sqrt(ms / d + cfg.eps);
    return Float32Array.from(qv.x, (v) => v / r);
  }, [qv.x, d, cfg.eps]);
  // One color scale for x, x / rms(x) and h, so the rescaling shows as a change of color strength.
  const dom = useMemo(() => maxAbsOf(qv.x, scaled, qv.h), [qv.x, scaled, qv.h]);
  const layerName = LAYER_INFO[live.layer].name.toLowerCase();
  const rows: { label: string; values: Float32Array; note: ReactNode; what: string; gamma?: boolean }[] = [
    { label: 'x', values: qv.x, note: 'the residual stream', what: 'the query token’s x, the residual stream entering this sublayer' },
    {
      label: 'x / rms(x)',
      values: scaled,
      note: `root-mean-square ≈ 1, so ‖v‖ ≈ √${d} = ${Math.sqrt(d).toFixed(Number.isInteger(Math.sqrt(d)) ? 0 : 2)}`,
      what: 'x divided by its root mean square',
    },
    {
      label: `γ of block ${live.block + 1}’s ${layerName} RMSNorm`,
      values: qv.gamma,
      note: (
        <>
          <span className="font-mono">{`blocks.${live.block}.norms.${live.layer}.weight`}</span> (the code counts blocks from 0)
        </>
      ),
      what: `γ, the learned gain of block ${live.block + 1}’s ${layerName} RMSNorm, shared by every token`,
      gamma: true,
    },
    { label: 'h = (x / rms(x)) ⊙ γ', values: qv.h, note: 'what the attention reads', what: 'h, the normalized vector the attention reads' },
  ];
  return (
    <div className="grid gap-4">
      {rows.map((r, i) => (
        <div key={r.label}>
          <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3 text-[14.5px]">
            <span className="text-ink">{r.label}</span>
            <span className="text-muted">{r.note}</span>
          </div>
          {r.gamma ? (
            <SeqStrip values={r.values} label="γ, the RMSNorm gain" shape={vec(r.what, `${axis.model(d)}; every dimension has its own gain`)} />
          ) : (
            <VectorStrip values={r.values} columns={d} cellHeight={i === 3 ? 18 : 14} domain={dom} showNorm shape={vec(r.what, axis.model(d))} />
          )}
        </div>
      ))}
      <p className="!mb-0 text-[14.5px] text-muted">
        x, x / rms(x) and h share one color scale (±{dom.toFixed(2)}), so a stronger color means a larger number; the readout under each strip gives
        its length ‖v‖.
      </p>
    </div>
  );
}

function QKVVisual({ live, qv, W, inspected }: { live: AttnLive; qv: QueryView; W: number; inspected: number }) {
  const { d, dh, head, H, query, labels } = live;
  const cell = Math.max(3, Math.min(5, Math.floor((W - 120) / d)));
  const mW = useMemo(() => maxAbsOf(qv.Wq), [qv.Wq]);
  const mh = maxAbsOf(qv.h);
  const mq = maxAbsOf(qv.q);
  const kh = headRow(qv.cap.k, inspected, head, d, dh);
  const vh = headRow(qv.cap.v, inspected, head, d, dh);
  const mHead = maxAbsOf(qv.qh, kh, vh);
  const [hover, setHover] = useState<string | null>(null);
  const bands = Array.from({ length: H }, (_, j) => ({ start: j * dh, end: (j + 1) * dh, strong: j === head }));
  return (
    <div className="grid gap-5">
      <div>
        <Eyebrow>
          q = W<sub>q</sub> h, for the query (k = W<sub>k</sub> h and v = W<sub>v</sub> h work the same way, for every token)
        </Eyebrow>
        <div className="scroll-quiet flex items-center gap-2.5 overflow-x-auto pt-1 pb-1">
          <div className="flex flex-col items-center gap-1">
            <MatrixCanvas
              data={qv.Wq}
              rows={d}
              cols={d}
              cellW={cell}
              cellH={cell}
              maxAbs={mW}
              bandRows={[head * dh, (head + 1) * dh]}
              label={`W_q of this sublayer, ${d} by ${d}; the ${dh} rows that produce head ${head + 1}'s query are outlined.`}
              onHover={(c) => setHover(c ? `W_q[${c.r}, ${c.c}] = ${sgn(c.v, 4)}` : null)}
            />
            <span className="inline-flex items-baseline gap-2 text-[14px] text-ink-2">
              <span>
                W<sub>q</sub>
              </span>
              <ShapeTag
                rows={d}
                cols={d}
                shape={{
                  rows: `one per dimension of q, the output (d = ${d}); each head owns ${dh} consecutive rows, and head ${head + 1}’s rows ${head * dh}–${(head + 1) * dh - 1} are outlined`,
                  cols: `one per dimension of h, the input (d = ${d})`,
                  note: 'Stored the way PyTorch stores every nn.Linear weight, [out, in], so q = W_q h. It is learned in training and the same for every token; each attention sublayer of each block has its own.',
                }}
              />
            </span>
          </div>
          <span className="text-[20px] text-muted">·</span>
          <div className="flex flex-col items-center gap-1">
            <VStrip values={qv.h} cellW={14} cellH={cell} maxAbs={mh} label="h, the normalized input" onHover={(i) => setHover(i === null ? null : `h[${i}] = ${sgn(qv.h[i], 4)}`)} />
            <span className="inline-flex items-baseline gap-2 text-[14px] text-ink-2">
              <span>h</span>
              <ShapeTag rows={d} cols={1} shape={{ rows: axis.model(d), cols: 'a single column: h of the query token, drawn upright to line up with the columns of W_q' }} />
            </span>
          </div>
          <span className="text-[20px] text-muted">=</span>
          <div className="flex flex-col items-center gap-1">
            <VStrip
              values={qv.q}
              cellW={14}
              cellH={cell}
              maxAbs={mq}
              bands={bands}
              fadeOutside
              label={`q, the query; head ${head + 1} uses dimensions ${head * dh} to ${(head + 1) * dh - 1}`}
              onHover={(i) => setHover(i === null ? null : `q[${i}] = ${sgn(qv.q[i], 4)} · head ${Math.floor(i / dh) + 1}`)}
            />
            <span className="inline-flex items-baseline gap-2 text-[14px] text-ink-2">
              <span>q</span>
              <ShapeTag
                rows={d}
                cols={1}
                shape={{ rows: `one per dimension of q (d = ${d}): the ${H} heads’ ${dh} dimensions, one block after another`, cols: 'a single column: q of the query token' }}
              />
            </span>
          </div>
          <div className="ml-1 hidden flex-col justify-center gap-0 self-stretch text-[13.5px] text-muted sm:flex" style={{ height: d * cell }}>
            {bands.map((b) => (
              <span key={b.start} className={cx('flex items-center', b.strong && 'font-semibold text-accent-text')} style={{ height: dh * cell }}>
                head {b.start / dh + 1}
              </span>
            ))}
          </div>
        </div>
        <div className="num mt-1 min-h-[1.3em] text-[14px] text-muted">{hover ?? `faded rows of W_q feed the other ${H - 1} heads`}</div>
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        <div>
          <Eyebrow>
            q<sup>({head + 1})</sup> of the query #{query}
          </Eyebrow>
          <VectorStrip
            values={qv.qh}
            columns={dh}
            cellHeight={18}
            domain={mHead}
            shape={vec(`head ${head + 1}’s query for token #${query}: dimensions ${head * dh}–${(head + 1) * dh - 1} of q`, axis.head(dh, d, H))}
          />
        </div>
        <div>
          <Eyebrow>
            k<sup>({head + 1})</sup> of key #{inspected}
          </Eyebrow>
          <VectorStrip
            values={kh}
            columns={dh}
            cellHeight={18}
            domain={mHead}
            shape={vec(`head ${head + 1}’s key for token #${inspected}: dimensions ${head * dh}–${(head + 1) * dh - 1} of k`, axis.head(dh, d, H))}
          />
        </div>
        <div>
          <Eyebrow>
            v<sup>({head + 1})</sup> of key #{inspected}
          </Eyebrow>
          <VectorStrip
            values={vh}
            columns={dh}
            cellHeight={18}
            domain={mHead}
            shape={vec(`head ${head + 1}’s value for token #${inspected}: dimensions ${head * dh}–${(head + 1) * dh - 1} of v`, axis.head(dh, d, H))}
          />
        </div>
      </div>
      <p className="!mb-0 text-[14.5px] text-muted">
        {inspected === query ? (
          <>
            Key #{inspected} is the query itself ({labels[inspected]}): the same h goes through three different matrices, so its q, k and v
            differ. Pick another key by clicking a bar in stages 4–6.
          </>
        ) : (
          <>
            Key #{inspected} is {labels[inspected]}. Pick another key by clicking a bar in stages 4–6.
          </>
        )}
      </p>
    </div>
  );
}

function BarsVisual({ live, qv, stage, inspected, onInspect }: { live: AttnLive; qv: QueryView; stage: BarStage; inspected: number; onInspect: (k: number) => void }) {
  const { ctx, keys, query, labels, block, layer, d, setBlock } = live;
  const mode = useResolvedTheme();
  const nAllowed = qv.allowedList.length;
  const collapse = stage === 'softmax' && nAllowed > 0 && nAllowed < keys.order.length;
  // Allowed keys whose score equals another allowed key's score exactly, confirmed by identical k vectors.
  const tied = useMemo(() => {
    const byScore = new Map<number, number[]>();
    for (const k of qv.allowedList) {
      const v = qv.scores[k];
      const g = byScore.get(v);
      if (g) g.push(k);
      else byScore.set(v, [k]);
    }
    let n = 0;
    let groups = 0;
    const K = qv.cap.k;
    for (const g of byScore.values()) {
      if (g.length < 2) continue;
      const a = g[0];
      const same = g.filter((k) => {
        for (let j = 0; j < d; j++) if (K[k * d + j] !== K[a * d + j]) return false;
        return true;
      });
      if (same.length >= 2) {
        n += same.length;
        groups++;
      }
    }
    return { n, groups };
  }, [qv, d]);
  const tables = collapse ? keys.groups.filter((g) => g.positions.some((p) => qv.allowed[p])) : [];
  return (
    <div>
      <KeyBars
        tokens={ctx.tokens}
        groups={keys.groups}
        order={keys.order}
        scores={qv.scores}
        probs={qv.probs}
        allowed={qv.allowed}
        stage={stage}
        query={query}
        inspected={inspected}
        labels={labels}
        onInspect={onInspect}
      />
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[14px] text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <svg width="10" height="8" aria-hidden="true">
            <path d="M0,0 L10,0 L5,7 Z" style={{ fill: 'var(--accent)' }} />
          </svg>
          the query itself
        </span>
        {stage !== 'softmax' && (
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-[2px]" style={{ background: 'var(--ink-2)', opacity: 0.62 }} /> score q·k/√d<sub>head</sub>
          </span>
        )}
        {stage === 'mask' && (
          <>
            <span className="inline-flex items-center gap-1.5">
              <span
                className="inline-block h-2.5 w-2.5 rounded-[2px] border border-border"
                style={{ backgroundImage: 'repeating-linear-gradient(135deg, var(--axis) 0 2px, var(--surface-2) 2px 4px)' }}
              />
              blocked by the mask → −∞
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-flex h-2.5 overflow-hidden rounded-[2px]" aria-hidden="true">
                <span className="w-2" style={{ background: sequentialCss(mode === 'dark' ? 0.62 : 0.66, mode) }} />
                <span className="w-2 bg-surface-3" />
              </span>
              mask row M[q, ·] under the bars (blue = allowed)
            </span>
          </>
        )}
        {stage === 'softmax' && (
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-[2px] bg-seq-500" /> attention weight
          </span>
        )}
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-3 rounded-[2px] border-[1.5px] border-accent" /> inspected key #{inspected}
        </span>
      </div>
      {tables.length > 0 && (
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[14px] text-ink-2">
          <span className="text-muted">Square under each bar: the key’s table</span>
          {tables.map((g) => (
            <TableSwatch key={g.name} slot={g.slot} label={g.name} size={9} />
          ))}
        </div>
      )}
      {tied.n > 0 && stage !== 'scores' && (
        <div className="mt-2 rounded-lg border border-border bg-surface-2 px-3 py-2 text-[15px] leading-snug text-ink-2">
          {block === 0 && layer === 'col' ? (
            <>
              <strong className="font-semibold text-ink">Equal bars.</strong> {tied.n} of the {nAllowed} allowed keys share their score with
              another key. When block 1’s column layer runs, every token is still its input embedding: a column-name term plus a value term, with no
              row, entity or position information <Cite kind="spec" ref="§5.2" /> <Cite kind="spec" ref="§6.5" />. Two cells of one column with the
              same value are the same vector, so they get the same key, the same score and the same weight.{' '}
              <button type="button" className="font-medium text-accent-text underline underline-offset-2" onClick={() => setBlock(1)}>
                Switch to block 2
              </button>{' '}
              to see whether the ties break once the feature layer has mixed each cell’s row into it.
            </>
          ) : (
            <>
              <strong className="font-semibold text-ink">Equal bars.</strong> {tied.n} of the {nAllowed} allowed keys share their score with
              another key, because their key vectors k are identical: their inputs to this sublayer were the same vector.
            </>
          )}
        </div>
      )}
      {keys.padding.length > 0 && (
        <p className="!mb-0 mt-1 text-[14px] text-muted">
          Not drawn: {keys.padding.length} padding keys, which every mask blocks <Cite kind="spec" ref="§6.2" />.
        </p>
      )}
    </div>
  );
}

function OutVisual({ live, qv }: { live: AttnLive; qv: QueryView }) {
  const { head, H, dh, d, labels } = live;
  const top = useMemo(() => topK(qv.probs, qv.allowedList, 3).filter((k) => qv.probs[k] > 0), [qv]);
  const contribs = useMemo(
    () =>
      top.map((k) => {
        const v = headRow(qv.cap.v, k, head, d, dh);
        return { k, p: qv.probs[k], vec: Float32Array.from(v, (x) => x * qv.probs[k]) };
      }),
    [top, qv, head, d, dh],
  );
  const mC = maxAbsOf(qv.headSlice, ...contribs.map((c) => c.vec));
  const mX = maxAbsOf(qv.x, qv.xOut, qv.out);
  const rest = qv.allowedList.length - contribs.length;
  const empty = qv.allowedList.length === 0;
  const ranges = Array.from({ length: H }, (_, j) => ({ start: j * dh, end: (j + 1) * dh, label: `head ${j + 1}` })).filter((r) => r.start === head * dh);
  return (
    <div className="grid gap-5">
      <div>
        <Eyebrow>
          Head {head + 1}: o<sup>({head + 1})</sup> = Σ<sub>k</sub> p<sub>k</sub> v<sub>k</sub><sup>({head + 1})</sup>
        </Eyebrow>
        {empty ? (
          <p className="!mb-0 text-[15.5px] text-ink-2">No key is allowed, so the sum is empty: o<sup>({head + 1})</sup> = 0 in every dimension.</p>
        ) : (
          <div className="grid gap-2.5">
            {contribs.map((c, i) => (
              <div key={c.k} className="grid items-center gap-x-3 gap-y-1 sm:grid-cols-[260px_minmax(0,1fr)]">
                <span className="num truncate text-[14px] text-ink-2">
                  {i > 0 ? '+ ' : '  '}
                  {pctw(c.p)} × v(#{c.k} {labels[c.k]})
                </span>
                <VectorStrip
                  values={c.vec}
                  columns={dh}
                  cellHeight={13}
                  domain={mC}
                  showReadout={false}
                  shape={vec(`token #${c.k}’s head-${head + 1} value, times its attention weight ${pctw(c.p)}`, axis.head(dh, d, H))}
                />
              </div>
            ))}
            {rest > 0 && (
              <span className="num text-[14px] text-muted sm:pl-[272px]">
                + {rest} more {rest === 1 ? 'key' : 'keys'} ({pctw(1 - contribs.reduce((a, c) => a + c.p, 0))} of the weight)
              </span>
            )}
            <div className="grid items-center gap-x-3 gap-y-1 sm:grid-cols-[260px_minmax(0,1fr)]">
              <span className="num text-[14px] font-semibold text-ink">= o({head + 1})</span>
              <VectorStrip
                values={qv.headSlice}
                columns={dh}
                cellHeight={16}
                domain={mC}
                shape={vec(`o(${head + 1}), head ${head + 1}’s output for the query: the weighted sum of the values`, axis.head(dh, d, H))}
              />
            </div>
          </div>
        )}
      </div>
      <div className="grid gap-3">
        <div>
          <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3 text-[14.5px]">
            <span className="text-ink">[o(1); …; o({H})]</span>
            <span className="text-muted">the {H} head outputs, concatenated (head {head + 1} outlined)</span>
          </div>
          <VectorStrip
            values={qv.headOut}
            columns={d}
            cellHeight={14}
            ranges={ranges}
            shape={vec(
              `the ${H} heads’ outputs for the query, side by side`,
              `one per dimension: head j’s ${dh} outputs fill dimensions ${dh}·(j − 1) to ${dh}·j − 1, so d = ${H} × ${dh} = ${d}`,
            )}
          />
        </div>
        <div>
          <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3 text-[14.5px]">
            <span className="text-ink">Δx = W_o · [o(1); …; o({H})]</span>
            <span className="text-muted">the update, on the same scale as x</span>
          </div>
          <VectorStrip values={qv.out} columns={d} cellHeight={14} domain={mX} shape={vec('Δx = W_o · [o(1); …], the update this sublayer adds to the query token', axis.model(d))} />
        </div>
        <div>
          <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3 text-[14.5px]">
            <span className="text-ink">x</span>
            <span className="text-muted">the residual stream that entered</span>
          </div>
          <VectorStrip values={qv.x} columns={d} cellHeight={14} domain={mX} shape={vec('the query token’s x as it entered this sublayer', axis.model(d))} />
        </div>
        <div>
          <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3 text-[14.5px]">
            <span className="text-ink">x′ = x + Δx</span>
            <span className="text-muted">what the next sublayer receives</span>
          </div>
          <VectorStrip values={qv.xOut} columns={d} cellHeight={18} domain={mX} shape={vec('x′ = x + Δx, the query token’s vector after this sublayer', axis.model(d))} />
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic panel
// ─────────────────────────────────────────────────────────────────────────────

function StageMath({ stage, live, qv, inspected }: { stage: (typeof STAGES)[number]['id']; live: AttnLive; qv: QueryView; inspected: number }) {
  const { d, dh, H, head, block, layer, query, labels, cfg, ctx, pdb } = live;
  const li = ATTN_ORDER.indexOf(layer);
  const before = block * (ATTN_ORDER.length + 1) + li;
  const title = (t: ReactNode) => <div className="eyebrow mb-2 !text-[14px]">{t}</div>;

  if (stage === 'x') {
    return (
      <>
        {title('Where x comes from')}
        <p className="!mb-2 text-ink">
          {before === 0 ? (
            <>
              This is the first sublayer of the model, so x is the input embedding from chapter 8 <Cite kind="code" ref="rt/model.py:208-224" />.
            </>
          ) : (
            <>
              x is the input embedding plus the updates of the {before} {before === 1 ? 'sublayer' : 'sublayers'} before this one: each sublayer adds its output to
              the stream and passes it on{' '}
              <Cite kind="code" ref="rt/model.py:82-86" />.
            </>
          )}
        </p>
        <KV k="query" v={`#${query} ${labels[query]}`} />
        <KV k="row" v={rowName(ctx.tokens[query])} />
        <KV k="dimensions" v={d} />
        <KV k="‖x‖" v={norm2(qv.x).toFixed(3)} />
        <p className="!mb-0 mt-2 text-ink-2">The same x feeds all {H} heads. The heads split the projections, and each one still sees the whole input.</p>
      </>
    );
  }

  if (stage === 'norm') {
    let ms = 0;
    for (let i = 0; i < d; i++) ms += qv.x[i] * qv.x[i];
    ms /= d;
    const r = Math.sqrt(ms + cfg.eps);
    let msScaled = 0;
    for (let i = 0; i < d; i++) msScaled += (qv.x[i] / r) ** 2;
    msScaled /= d;
    let gmin = Infinity;
    let gmax = -Infinity;
    for (const g of qv.gamma) {
      gmin = Math.min(gmin, g);
      gmax = Math.max(gmax, g);
    }
    return (
      <>
        {title('RMSNorm, with this token’s numbers')}
        <Formula tex={String.raw`\mathrm{rms}(x)=\sqrt{\mathrm{mean}(x^2)+\varepsilon}`} />
        <Formula tex={String.raw`h=\big(x\,/\,\mathrm{rms}(x)\big)\odot\gamma`} />
        <KV k={`mean(x²) over ${d} dims`} v={ms.toFixed(4)} />
        <KV k={<T tex="\varepsilon = 2^{-7}" />} v={cfg.eps.toFixed(7)} />
        <KV k="rms(x)" v={r.toFixed(4)} />
        <KV k="mean square of x / rms" v={msScaled.toFixed(4)} />
        <KV k="γ range" v={`${sgn(gmin, 3)} … ${sgn(gmax, 3)}`} />
        <KV k="‖h‖" v={norm2(qv.h).toFixed(3)} />
        <p className="!mb-0 mt-2 text-ink-2">
          Each attention layer of each block has its own RMSNorm, five per block counting the feed-forward one{' '}
          <Cite kind="code" ref="rt/model.py:71-73" />. ε is a small constant that keeps the division away from zero <Cite kind="spec" ref="§7.2" />.
        </p>
      </>
    );
  }

  if (stage === 'qkv') {
    const kh = headRow(qv.cap.k, inspected, head, d, dh);
    const vh = headRow(qv.cap.v, inspected, head, d, dh);
    return (
      <>
        {title('Projections and the head split')}
        <Formula tex={String.raw`q=W_q h,\;\; k=W_k h,\;\; v=W_v h`} />
        <p className="!mb-2 text-ink-2">
          Each W is {d}×{d} with no bias <Cite kind="code" ref="rt/model.py:25-28" />. The code then reshapes{' '}
          <code>"b s (h d) -&gt; b h s d"</code>, so head j owns dimensions{' '}
          <TeX tex={String.raw`[${dh}j,\ ${dh}j+${dh})`} /> <Cite kind="code" ref="rt/model.py:35-37" />.
        </p>
        <KV k={`head ${head + 1} dims`} v={`${head * dh} … ${(head + 1) * dh - 1}`} />
        <KV k={<T tex={String.raw`\lVert q^{(${head + 1})}\rVert`} />} v={norm2(qv.qh).toFixed(3)} />
        <KV k={<T tex={String.raw`\lVert k^{(${head + 1})}_{\#${inspected}}\rVert`} />} v={norm2(kh).toFixed(3)} />
        <KV k={<T tex={String.raw`\lVert v^{(${head + 1})}_{\#${inspected}}\rVert`} />} v={norm2(vh).toFixed(3)} />
        <KV k="head width" v={`${d} / ${H} = ${dh}`} />
        <p className="!mb-0 mt-2 text-ink-2">
          The formula above the figure uses the row form <T tex={String.raw`h\,W_q^{\top}`} />. Written for one token as a column vector, it
          becomes <T tex="q=W_q h" />.
        </p>
      </>
    );
  }

  if (stage === 'scores') {
    const kh = headRow(qv.cap.k, inspected, head, d, dh);
    let dot = 0;
    for (let i = 0; i < dh; i++) dot += qv.qh[i] * kh[i];
    const s = dot / Math.sqrt(dh);
    const best = topK(qv.scores, live.keys.order, 1)[0];
    // Spread of the scores over the real keys, before and after the divisor (display only).
    const ks = live.keys.order;
    let mean = 0;
    for (const k of ks) mean += qv.scores[k];
    mean /= Math.max(1, ks.length);
    let vr = 0;
    for (const k of ks) vr += (qv.scores[k] - mean) ** 2;
    const sdScaled = Math.sqrt(vr / Math.max(1, ks.length));
    return (
      <>
        {title(`Score of key #${inspected}`)}
        <Formula tex={String.raw`s_k=q^{(${head + 1})}\cdot k^{(${head + 1})}_k\,/\,\sqrt{${dh}}`} />
        <KV k={<T tex={String.raw`q^{(${head + 1})}\cdot k^{(${head + 1})}_{\#${inspected}}`} />} v={sgn(dot, 4)} />
        <KV k={<T tex={String.raw`\div\sqrt{${dh}}`} />} v={`= ${sgn(s, 4)}`} />
        <KV k="captured score" v={sgn(qv.scores[inspected], 4)} />
        <KV k="highest score" v={`#${best} · ${sgn(qv.scores[best], 3)}`} />
        <KV k={`std of q·k over the ${int(ks.length)} keys`} v={(sdScaled * Math.sqrt(dh)).toFixed(3)} />
        <KV k={`std after ÷√${dh}`} v={sdScaled.toFixed(3)} />
        <p className="!mb-0 mt-2 text-ink-2">
          FlexAttention’s default scale is 1 over the square root of the head width, here 1/√{dh} = {(1 / Math.sqrt(dh)).toFixed(4)}{' '}
          <Cite kind="spec" ref="§6.1" />. In the paper model the head width is 32, so the factor is 1/√32 ≈ 0.1768.
        </p>
      </>
    );
  }

  if (stage === 'mask') {
    const t = ctx.tokens[query];
    const n = qv.allowedList.length;
    const parents = t.f2pNbrIdxs.filter((p) => p >= 0);
    return (
      <>
        {title(`Why these ${n} keys`)}
        <MaskRule layer={layer} />
        <div className="mt-2">
          {layer === 'col' && (
            <>
              <KV k="query’s column" v={<span className="text-[14px]">{pdb.text[t.colNameIdx] ?? t.colName}</span>} />
              <KV k="col / table id" v={`${t.colNameIdx} / ${t.tableNameIdx}`} />
            </>
          )}
          {layer === 'feat' && (
            <>
              <KV k="query’s node" v={int(t.nodeIdx)} />
              <KV k="its parents (f2p)" v={parents.length ? parents.map(int).join(', ') : 'none'} />
            </>
          )}
          {layer === 'nbr' && <KV k="query’s node" v={int(t.nodeIdx)} />}
          <KV k="allowed keys" v={int(n)} />
          <KV k="blocked → −∞" v={int(live.S - n)} />
        </div>
        <p className="!mb-0 mt-2 text-ink-2">
          {layer === 'nbr' && n === 0
            ? 'No token in the context lists this row among its parents, so the row of the neighbor mask is empty.'
            : `All ${H} heads use this same mask row.`}{' '}
          {layer === 'nbr' && n === 0 ? <Cite kind="spec" ref="§6.2" /> : <Cite kind="code" ref="rt/model.py:89-101" />}
        </p>
      </>
    );
  }

  if (stage === 'softmax') {
    const n = qv.allowedList.length;
    if (n === 0) {
      return (
        <>
          {title('A row with no allowed key')}
          <p className="!mb-2 text-ink">
            Every score is −∞, so there is nothing to normalize. FlexAttention leaves this row of weights at 0, and the head outputs exactly 0{' '}
            <Cite kind="spec" ref="§6.1" />.
          </p>
          <KV k="allowed keys" v="0" />
          <KV k="Σ p" v="0" />
          <KV k="head output" v={`0 in all ${dh} dims`} />
        </>
      );
    }
    let m = -Infinity;
    for (const k of qv.allowedList) m = Math.max(m, qv.scores[k]);
    let Z = 0;
    let sumP = 0;
    for (const k of qv.allowedList) {
      Z += Math.exp(qv.scores[k] - m);
      sumP += qv.probs[k];
    }
    const top = topK(qv.probs, qv.allowedList, 5);
    const isAllowed = !!qv.allowed[inspected];
    return (
      <>
        {title('Softmax over the allowed keys')}
        <Formula tex={String.raw`p_k=\exp(s_k-m)\,/\,Z`} />
        <Formula tex={String.raw`Z=\textstyle\sum_{j\,\in\,\text{allowed}}\exp(s_j-m)`} />
        <KV k={<T tex="m = \max s_k" />} v={sgn(m, 4)} />
        <KV k={<T tex="Z" />} v={Z.toFixed(4)} />
        <KV
          k={<T tex={String.raw`p_{\#${inspected}}`} />}
          v={isAllowed ? (
            <>
              <T tex={String.raw`\exp(${texNum(qv.scores[inspected] - m, 3)})/${texNum(Z, 3)}`} /> = {pctw(qv.probs[inspected])}
            </>
          ) : (
            '0 (blocked)'
          )}
        />
        <KV k="Σ p over the row" v={sumP.toFixed(6)} />
        <div className="mt-3 mb-1 text-[14px] font-semibold text-ink-2">Largest weights</div>
        <ol className="!m-0 grid gap-0.5 !p-0" style={{ listStyle: 'none' }}>
          {top.map((k) => (
            <li key={k} className="num !m-0 flex items-baseline justify-between gap-2 !p-0 text-[14px]">
              <span className={cx('truncate', k === inspected ? 'text-accent-text' : 'text-ink-2')}>
                #{k} {labels[k]}
              </span>
              <span className="text-ink">{pctw(qv.probs[k])}</span>
            </li>
          ))}
        </ol>
        <p className="!mb-0 mt-2 text-ink-2">Subtracting the maximum m changes nothing mathematically; it only keeps exp from overflowing.</p>
      </>
    );
  }

  // out
  const nx = norm2(qv.x);
  const nout = norm2(qv.out);
  return (
    <>
      {title('Back into the residual stream')}
      <Formula tex={String.raw`x' = x + W_o\,[o^{(1)};\dots;o^{(${H})}]`} />
      <KV k={<T tex={String.raw`\lVert o^{(${head + 1})}\rVert`} />} v={norm2(qv.headSlice).toFixed(3)} />
      <KV k="‖concat‖" v={norm2(qv.headOut).toFixed(3)} />
      <KV k="‖Δx‖ = ‖W_o · concat‖" v={nout.toFixed(3)} />
      <KV k="‖x‖ → ‖x′‖" v={`${nx.toFixed(3)} → ${norm2(qv.xOut).toFixed(3)}`} />
      <KV k="‖Δx‖ / ‖x‖" v={nx > 0 ? (nout / nx).toFixed(3) : '–'} />
      <p className="!mb-0 mt-2 text-ink-2">
        W<sub>o</sub> is {d}×{d} without bias <Cite kind="code" ref="rt/model.py:28" /> and mixes all {H} heads <Cite kind="code" ref="rt/model.py:45-46" />. So if every head outputs 0, the update is
        exactly 0 and x passes through unchanged <Cite kind="spec" ref="§6.1" />.
      </p>
    </>
  );
}

/** A display formula in the arithmetic panel: 18px, never shrunk, scrolls if it does not fit. */
function Formula({ tex }: { tex: string }) {
  return (
    <div className="scroll-quiet my-1 overflow-x-auto py-0.5 text-[18px]">
      <TeX tex={tex} />
    </div>
  );
}

/** Inline TeX in the arithmetic panel at 17.5px, so its sub- and superscripts stay above 13px. */
function T({ tex }: { tex: string }) {
  return (
    <span className="text-[17.5px]">
      <TeX tex={tex} />
    </span>
  );
}

function MaskRule({ layer }: { layer: (typeof ATTN_ORDER)[number] }) {
  const tex: Record<string, string> = {
    col: String.raw`\mathbf{M}[q,k]=\mathbf{1}\{\mathrm{Col}(k)=\mathrm{Col}(q)\}`,
    feat: String.raw`\mathbf{M}[q,k]=\mathbf{1}\{\mathrm{Row}(k)=\mathrm{Row}(q)\lor\mathrm{Row}(k)\in\mathrm{OutLinks}(q)\}`,
    nbr: String.raw`\mathbf{M}[q,k]=\mathbf{1}\{\mathrm{Row}(q)\in\mathrm{OutLinks}(k)\}`,
    full: String.raw`\mathbf{M}[q,k]=1`,
  };
  const code: Record<string, { text: string; ref: string }> = {
    col: { text: 'same column id and same table id, both tokens non-padding', ref: 'rt/model.py:181-183' },
    feat: { text: 'same row, or the key’s row is one of the query’s parents; both tokens non-padding', ref: 'rt/model.py:187' },
    nbr: { text: 'the query’s row is one of the key’s parents; both tokens non-padding', ref: 'rt/model.py:188' },
    full: { text: 'every pair of non-padding tokens', ref: 'rt/model.py:190' },
  };
  return (
    <div className="grid gap-1.5">
      <div className="text-ink-2">
        <LayerLabel layer={layer} size={13} /> mask in the paper <Cite kind="paper" ref="p.5 §3.2" />:
      </div>
      <div className="scroll-quiet overflow-x-auto text-[16px]">
        <TeX tex={tex[layer]} />
      </div>
      <div className="text-ink-2">
        In the code: {code[layer].text} <Cite kind="code" ref={code[layer].ref} />.
      </div>
    </div>
  );
}
