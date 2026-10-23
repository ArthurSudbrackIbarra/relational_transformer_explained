/**
 * "Follow one token through": the input embedding of any token of the live context, computed in
 * seven stages with the weights of the checkpoint the page runs (model.py:208-224, spec §5.2).
 * The figure's units are "stages" (①-⑦) so they never clash with the chapter's "Step N" sections.
 *
 * Every vector shown is the forward pass's own capture (`ForwardResult.embedding`: colPreNorm,
 * colTerm, valuePreNorm, valueTerm, x0) or the precomputed MiniLM embedding it was computed from;
 * the per-dimension arithmetic on hover re-reads the same weights.
 */
import { ArrowDown, ArrowRight, Plus, Equal } from 'lucide-react';
import { motion, useInView, useReducedMotion } from 'motion/react';
import { memo, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  Button,
  Figure,
  SemBadge,
  StatusBadge,
  StepPlayer,
  TableSwatch,
  TeX,
  TokenStrip,
  VectorStrip,
  cx,
  formatRaw,
  maxAbs,
  useStepPlayer,
} from '@/components';
import { SEM_KEYS, tensor } from '@/engine/model';
import { SemType, type Token } from '@/engine/types';
import { type EmbeddingStable, fmt, fmtSigned, phraseOf, rowRef, SEM_LABEL, useInspected, VALUE_FIELD } from './live';
import { dot, norm, rmsInfo, slice, tokenScalar } from './math';
import { useElementSize } from '@/components/viz/useSize';
import { FLASH_EVENT, FOLLOW_ID, JumpLink } from './nav';

const STEPS = 7;

type Lane = 'col' | 'val' | 'sum';
const STAGE_LANE: Lane[] = ['col', 'col', 'col', 'val', 'val', 'val', 'sum'];

export const TokenEmbedder = memo(function TokenEmbedder({ live }: { live: EmbeddingStable }) {
  const { ctx, slots, target, firstOfType, fr, weights, store, pdb, ready, assets } = live;
  const { inspected, setInspected } = useInspected(live);
  const tok = ctx.tokens[inspected];
  const reduce = useReducedMotion();
  const rootRef = useRef<HTMLDivElement | null>(null);
  // The figure is taller than a phone screen, so an area threshold would never be reached there: start
  // once its top has scrolled into the upper part of the viewport.
  const inView = useInView(rootRef, { once: true, amount: 'some', margin: '0px 0px -45% 0px' });
  const player = useStepPlayer(STEPS, { intervalMs: 1700 });
  // Stages the reader steps to (buttons, scrubber, keys) are scrolled into view if hidden; autoplay never scrolls.
  const stageRefs = useRef<(HTMLDivElement | null)[]>([]);
  const userNav = useRef(false);
  const [playerRef, playerSize] = useElementSize<HTMLDivElement>();
  useEffect(() => {
    if (!userNav.current) return;
    userNav.current = false;
    stageRefs.current[player.index]?.scrollIntoView({ block: 'nearest', behavior: reduce ? 'auto' : 'smooth' });
  }, [player.index, reduce]);
  const userStep = useCallback(
    (delta: number) => {
      userNav.current = true;
      player.step(delta);
    },
    [player],
  );
  const userSetIndex = useCallback(
    (i: number) => {
      userNav.current = true;
      player.setIndex(i);
    },
    [player],
  );
  const stageRef = (i: number) => (el: HTMLDivElement | null) => {
    stageRefs.current[i] = el;
  };
  const [hoverDim, setHoverDim] = useState<number | null>(null);
  // "Go to Step 2" from another figure: flash the token header once it is in view.
  const [flash, setFlash] = useState(false);
  useEffect(() => {
    let t = 0;
    const on = () => {
      setFlash(true);
      window.clearTimeout(t);
      t = window.setTimeout(() => setFlash(false), 900);
    };
    window.addEventListener(FLASH_EVENT, on);
    return () => {
      window.removeEventListener(FLASH_EVENT, on);
      window.clearTimeout(t);
    };
  }, []);

  // Autoplay once when the figure first scrolls into view (reduced motion: show every step at once).
  const started = useRef(false);
  useEffect(() => {
    if (!ready || started.current) return;
    if (reduce) {
      started.current = true;
      player.setIndex(STEPS - 1);
      return;
    }
    if (inView) {
      started.current = true;
      player.setPlaying(true);
    }
  }, [inView, ready, reduce, player]);

  const data = useMemo(() => {
    if (!fr || !weights || !store || !tok) return null;
    const d = weights.config.dModel;
    const eps = weights.config.eps;
    const pos = tok.pos;
    const key = SEM_KEYS[tok.sem];
    const phrase = tok.isPadding ? '' : pdb.text[tok.colNameIdx];
    const c = tok.isPadding ? new Float32Array(weights.config.dText) : store.get(phrase);
    const colPre = slice(fr.embedding.colPreNorm, pos, d);
    const colTerm = slice(fr.embedding.colTerm, pos, d);
    const valPre = slice(fr.embedding.valuePreNorm, pos, d);
    const valTerm = slice(fr.embedding.valueTerm, pos, d);
    const x0 = slice(fr.embedding.x0, pos, d);
    const textInput = !tok.isPadding && tok.sem === SemType.Text ? store.get(pdb.text[tok.textIdx]) : null;
    const textString = !tok.isPadding && tok.sem === SemType.Text ? pdb.text[tok.textIdx] : null;
    return {
      d,
      eps,
      key,
      phrase,
      c,
      colPre,
      colTerm,
      valPre,
      valTerm,
      x0,
      textInput,
      textString,
      colInfo: rmsInfo(colPre, eps),
      valInfo: rmsInfo(valPre, eps),
      Wc: tensor(weights, 'enc_dict.col_name.weight'),
      bc: tensor(weights, 'enc_dict.col_name.bias'),
      gc: tensor(weights, 'norm_dict.col_name.weight'),
      Wt: tensor(weights, `enc_dict.${key}.weight`),
      bt: tensor(weights, `enc_dict.${key}.bias`),
      gt: tensor(weights, `norm_dict.${key}.weight`),
      postDomain: Math.max(maxAbs(colTerm), maxAbs(valTerm), maxAbs(x0)) || 1,
    };
  }, [fr, weights, store, tok, pdb]);

  const step = player.index;
  const revealed = (i: number) => i <= step;
  const masked = !!tok?.mask;
  const padding = !!tok?.isPadding;

  // per-dimension arithmetic for the hovered dim (64-d strips only)
  const dimLine = (stage: number): ReactNode => {
    if (!data || hoverDim === null || hoverDim >= data.d) return null;
    const j = hoverDim;
    switch (stage) {
      case 1: {
        const wc = dot(data.Wc.subarray(j * data.c.length, (j + 1) * data.c.length), data.c);
        return (
          <>
            dim {j}: W[{j}]·c = {fmtSigned(wc)}, b[{j}] = {fmtSigned(data.bc[j])} → {fmtSigned(data.colPre[j])}
          </>
        );
      }
      case 2:
        return (
          <>
            dim {j}: {fmtSigned(data.colPre[j])} × {fmt(data.colInfo.scale, 2)} × γ[{j}] ({fmt(data.gc[j])}) = {fmtSigned(data.colTerm[j])}
          </>
        );
      case 3:
        if (!masked) return null;
        return (
          <>
            dim {j}: m_{data.key}[{j}] = {fmtSigned(data.valTerm[j])} (used as is)
          </>
        );
      case 4:
        if (masked) return null;
        if (data.textInput) {
          const wt = dot(data.Wt.subarray(j * data.textInput.length, (j + 1) * data.textInput.length), data.textInput);
          return (
            <>
              dim {j}: W[{j}]·e = {fmtSigned(wt)}, b[{j}] = {fmtSigned(data.bt[j])} → {fmtSigned(data.valPre[j])}
            </>
          );
        }
        return (
          <>
            dim {j}: w[{j}]·z + b[{j}] = ({fmtSigned(data.Wt[j])})({fmtSigned(tokenScalar(tok))}) + ({fmtSigned(data.bt[j])}) ={' '}
            {fmtSigned(data.valPre[j])}
          </>
        );
      case 5:
        if (masked) return null;
        return (
          <>
            dim {j}: {fmtSigned(data.valPre[j])} × {fmt(data.valInfo.scale, 2)} × γ[{j}] ({fmt(data.gt[j])}) = {fmtSigned(data.valTerm[j])}
          </>
        );
      case 6:
        return (
          <>
            dim {j}: {fmtSigned(data.colTerm[j])} + ({fmtSigned(data.valTerm[j])}) = {fmtSigned(data.x0[j])}
          </>
        );
      default:
        return null;
    }
  };

  const describe = (i: number): ReactNode => {
    if (!data || !tok) return 'Loading the embeddings and the weights…';
    if (padding) return 'This is a padding position: every term is multiplied by ¬is_padding = 0, so x₀ is the zero vector.';
    const key = data.key;
    const m = <TeX tex={`m_{\\mathrm{${key}}}`} />;
    switch (i) {
      case 0:
        return (
          <>
            The column phrase <q className="font-mono">{data.phrase}</q> is looked up in the frozen MiniLM table: <em>c</em> is a{' '}
            {data.c.length}-d unit vector (‖c‖ = {fmt(norm(data.c))}). Every cell of this column gets the same <em>c</em>.
          </>
        );
      case 1:
        return (
          <>
            A linear layer maps <em>c</em> to {data.d} numbers, <TeX tex="W_{\mathrm{col}}\,c + b_{\mathrm{col}}" />. The result has length{' '}
            {fmt(norm(data.colPre))} and mean square {fmt(data.colInfo.ms, 4)}
            {data.colInfo.epsShare > 0.2 ? `, small next to ε = ${fmt(data.eps, 4)}` : ''}.
          </>
        );
      case 2:
        return (
          <>
            RMSNorm divides by √(mean square + ε) = √({fmt(data.colInfo.ms, 4)} + {fmt(data.eps, 4)}), a factor of ×{fmt(data.colInfo.scale, 2)}, then
            multiplies by <TeX tex="\gamma_{\mathrm{col}}" />. The column term has length {fmt(norm(data.colTerm))}. Here ε makes up{' '}
            {(data.colInfo.epsShare * 100).toFixed(0)}% of the sum under the square root
            {data.colInfo.epsShare > 0.2 ? ', so this term comes out short' : ''}. <JumpLink to="rmsnorm">Step 3</JumpLink> explains ε.
          </>
        );
      case 3:
        if (masked)
          return (
            <>
              This is the masked target. Its stored label ({VALUE_FIELD[tok.sem]} = {fmtSigned(tokenScalar(tok))}) is still in the batch, but
              the gate ignores it. The mask vector {m} of its type takes the place of the value term.
            </>
          );
        if (data.textString !== null)
          return (
            <>
              The value is the string <q>{data.textString}</q>, embedded by the same frozen MiniLM, on its own, without the column name.
            </>
          );
        return (
          <>
            The value is one number: {VALUE_FIELD[tok.sem]} = {fmtSigned(tokenScalar(tok))}, the z-score computed in preprocessing (raw value{' '}
            {formatRaw(tok.raw, tok.sem === SemType.DateTime ? 'datetime' : undefined)}).
          </>
        );
      case 4:
        if (masked)
          return (
            <>
              No linear layer for the masked cell. (The code still evaluates the {key} branch on the stored label and multiplies it by the gate 0.)
            </>
          );
        return data.textInput ? (
          <>
            <code>enc_dict["text"]</code> is Linear(384, {data.d}): <TeX tex="W_{\mathrm{text}}\,e + b_{\mathrm{text}}" />, length{' '}
            {fmt(norm(data.valPre))}.
          </>
        ) : (
          <>
            <code>enc_dict["{key}"]</code> is Linear(1, {data.d}): the scalar times one weight column, plus a bias, <TeX tex="z\,w + b" />. Length{' '}
            {fmt(norm(data.valPre))}.
          </>
        );
      case 5:
        if (masked)
          return (
            <>
              No RMSNorm either: {m} is added raw, with its length {fmt(norm(data.valTerm))}.
            </>
          );
        return (
          <>
            <TeX tex={`\\mathrm{RMSNorm}_{\\mathrm{${key}}}`} /> rescales by ×{fmt(data.valInfo.scale, 2)} and applies{' '}
            <TeX tex={`\\gamma_{\\mathrm{${key}}}`} />: the value term has length {fmt(norm(data.valTerm))}. Here ε makes up{' '}
            {(data.valInfo.epsShare * 100).toFixed(data.valInfo.epsShare < 0.01 ? 1 : 0)}% of the sum under the square root.
          </>
        );
      default:
        return (
          <>
            x₀ = column term + {masked ? m : 'value term'}: {data.d} numbers of length {fmt(norm(data.x0))}. This is the vector the token brings
            into block 1.
          </>
        );
    }
  };

  const presets: { label: ReactNode; pos: number | undefined; title: string }[] = [
    { label: <>Masked target</>, pos: target, title: 'The seed row’s target cell' },
    { label: <>Number</>, pos: firstOfType[SemType.Number], title: 'The first number cell of the context' },
    { label: <>Text</>, pos: firstOfType[SemType.Text], title: 'The first text cell of the context' },
    { label: <>Datetime</>, pos: firstOfType[SemType.DateTime], title: 'The first datetime cell of the context' },
    { label: <>Boolean</>, pos: firstOfType[SemType.Boolean], title: 'The first visible boolean cell of the context' },
  ];

  return (
    <Figure
      wide
      title="Compute x₀ for one token of the live context"
      caption={
        <>
          Pick any token: click a square, or a preset; in the strip the arrow keys move and Enter picks. The seven stages follow model.py:208-224,
          with the weights of the checkpoint this page runs. The player stays at the top of the screen while you scroll through the stages. Hover a
          cell of any 64-d strip to see the arithmetic of that dimension in every stage. Post-norm vectors (③, ⑥, ⑦) share one color scale; the
          others have their own (shown as “scale ±…”).
        </>
      }
    >
      <div
        ref={rootRef}
        id={FOLLOW_ID}
        className="scroll-mt-[calc(var(--topbar-h)+24px)]"
        style={{ ['--emb-player-h' as string]: `${Math.round(playerSize.height || 0)}px` }}
      >
        <div className="mb-3 flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-[14.5px] font-medium text-ink-2">Presets</span>
          {presets.map((p, i) => (
            <Button
              key={i}
              size="sm"
              variant={p.pos === inspected ? 'primary' : 'secondary'}
              disabled={p.pos === undefined}
              title={p.title}
              onClick={() => p.pos !== undefined && setInspected(p.pos)}
              aria-pressed={p.pos === inspected}
            >
              {p.label}
            </Button>
          ))}
        </div>
        <TokenStrip
          tokens={ctx.tokens}
          tableSlots={slots}
          variant="cells"
          selected={inspected}
          onSelect={(pos) => setInspected(pos)}
          label="Context window tokens: pick one to embed"
        />

        {tok && <TokenHeader tok={tok} slot={tok.isPadding ? 0 : slots[tok.tableName] ?? 0} flash={flash} />}

        {!ready || !data ? (
          <div className="mt-4 rounded-lg border border-dashed border-border-strong px-4 py-6 text-center text-[16.5px] text-ink-2">
            {assets.status === 'error' ? `Could not load the model assets: ${assets.error ?? 'unknown error'}` : 'Loading the embeddings and the weights…'}
          </div>
        ) : (
          <>
            <div className="mt-2 flex flex-wrap gap-2">
              {assets.weightsKind === 'trained' && <StatusBadge status="good">trained mini model ({assets.ckpt})</StatusBadge>}
              {assets.weightsKind === 'init' && <StatusBadge status="warning">untrained weights: values from a random initialization</StatusBadge>}
            </div>

            {/* The player and its narration stay in view (sticky) while the reader scrolls through the stages. */}
            <div ref={playerRef} className="sticky top-[calc(var(--topbar-h)+8px)] z-20 mt-3">
              <StepPlayer
                {...player}
                step={userStep}
                setIndex={userSetIndex}
                className="shadow-[var(--shadow-card)]"
                unit="stage"
                label="Embedding stage-through controls"
                describe={(i) => (
                  <span>
                    <span className="mr-1.5 font-semibold text-accent-text">{'①②③④⑤⑥⑦'[i]}</span>
                    {describe(i)}
                  </span>
                )}
              />
            </div>

            <div className="mt-4 grid gap-4 lg:grid-cols-2">
              {/* ── column lane ── */}
              <LaneBox
                title="Column term"
                subtitle="Which column of which table is this?"
                gate={<TeX tex={`[\\lnot\\,\\mathrm{pad}] = ${padding ? 0 : 1}`} />}
              >
                <Stage innerRef={stageRef(0)} n={1} active={step === 0} revealed={revealed(0)} title="Phrase → frozen MiniLM" tex={String.raw`c = E(\text{“}\mathit{col}\text{ of }\mathit{table}\text{”}) \in \mathbb{R}^{384}`}>
                  <div className="mb-1.5 font-mono text-[15px] text-ink">“{padding ? '—' : data.phrase}”</div>
                  <VectorStrip values={data.c} columns={64} cellHeight={7} label={`c (${data.c.length}-d, col_name_values)`} hint={false} />
                </Stage>
                <Arrow on={revealed(1)} />
                <Stage innerRef={stageRef(1)} n={2} active={step === 1} revealed={revealed(1)} title={`Linear 384 → ${data.d}`} tex={String.raw`W_{\mathrm{col}}\,c + b_{\mathrm{col}}`} foot={dimLine(1)}>
                  <VectorStrip values={data.colPre} columns={32} cellHeight={13} onHover={(i) => setHoverDim(i)} highlightIndex={hoverDim} hint={false} />
                </Stage>
                <Arrow on={revealed(2)} />
                <Stage
                  innerRef={stageRef(2)}
                  n={3}
                  active={step === 2}
                  revealed={revealed(2)}
                  title="RMSNorm_col"
                  tex={String.raw`x/\sqrt{\operatorname{mean}(x^2)+\varepsilon}\,\odot\,\gamma_{\mathrm{col}}`}
                  foot={dimLine(2)}
                  side={<NormReadout info={data.colInfo} eps={data.eps} />}
                >
                  <VectorStrip
                    values={data.colTerm}
                    columns={32}
                    cellHeight={13}
                    domain={data.postDomain}
                    onHover={(i) => setHoverDim(i)}
                    highlightIndex={hoverDim}
                    hint={false}
                  />
                </Stage>
              </LaneBox>

              {/* ── value lane ── */}
              <LaneBox
                title={masked ? 'Mask vector' : 'Value term'}
                subtitle={masked ? 'The value is hidden: one stand-in vector per type' : 'What value does the cell hold?'}
                gate={<GateSwitch tok={tok} />}
              >
                <Stage
                  innerRef={stageRef(3)}
                  n={4}
                  active={step === 3}
                  revealed={revealed(3)}
                  title={masked ? `Stored label ignored → m_${data.key}` : data.textInput ? 'String → frozen MiniLM' : `Scalar input (${SEM_LABEL[tok.sem]})`}
                  tex={
                    masked
                      ? String.raw`m_{\mathrm{${data.key}}} \in \mathbb{R}^{${data.d}}`
                      : data.textInput
                        ? String.raw`e = E(\text{value}) \in \mathbb{R}^{384}`
                        : String.raw`v = z \in \mathbb{R}^{1}`
                  }
                  foot={masked ? dimLine(3) : undefined}
                >
                  {masked ? (
                    <div>
                      <div className="mb-1.5 flex flex-wrap items-baseline gap-x-2 text-[15px] text-ink-2">
                        <span className="num text-ink-2 line-through decoration-[1.5px]">
                          {VALUE_FIELD[tok.sem]} = {fmtSigned(tokenScalar(tok))}
                        </span>
                        <span>
                          ({formatRaw(tok.raw, tok.sem === SemType.DateTime ? 'datetime' : undefined)}) gated out (× 0)
                        </span>
                      </div>
                      <VectorStrip
                        values={data.valTerm}
                        columns={32}
                        cellHeight={13}
                        domain={data.postDomain}
                        label={`m_${data.key} = mask_embs.${data.key}`}
                        onHover={(i) => setHoverDim(i)}
                        highlightIndex={hoverDim}
                        hint={false}
                      />
                    </div>
                  ) : data.textInput ? (
                    <>
                      <div className="mb-1.5 truncate font-mono text-[15px] text-ink">“{data.textString}”</div>
                      <VectorStrip values={data.textInput} columns={64} cellHeight={7} label="e (384-d, text_values)" hint={false} />
                    </>
                  ) : (
                    <ScalarInput tok={tok} />
                  )}
                </Stage>
                <Arrow on={revealed(4)} />
                <Stage
                  innerRef={stageRef(4)}
                  n={5}
                  active={step === 4}
                  revealed={revealed(4)}
                  skipped={masked}
                  title={masked ? 'No linear layer' : data.textInput ? `Linear 384 → ${data.d}` : `Linear 1 → ${data.d}`}
                  tex={masked ? undefined : data.textInput ? String.raw`W_{\mathrm{text}}\,e + b_{\mathrm{text}}` : String.raw`z\,w_{\mathrm{${data.key}}} + b_{\mathrm{${data.key}}}`}
                  foot={masked ? undefined : dimLine(4)}
                >
                  {masked ? (
                    <SkipNote>
                      The code computes the {data.key} branch for every position and multiplies it by the gate{' '}
                      <span className="whitespace-nowrap text-[17.5px]">
                        <TeX tex={String.raw`[t_i{=}\mathrm{${data.key}}]\land\lnot\,\mathrm{mask}\land\lnot\,\mathrm{pad} = 0.`} />
                      </span>
                    </SkipNote>
                  ) : (
                    <VectorStrip values={data.valPre} columns={32} cellHeight={13} onHover={(i) => setHoverDim(i)} highlightIndex={hoverDim} hint={false} />
                  )}
                </Stage>
                <Arrow on={revealed(5)} />
                <Stage
                  innerRef={stageRef(5)}
                  n={6}
                  active={step === 5}
                  revealed={revealed(5)}
                  skipped={masked}
                  title={masked ? 'No RMSNorm' : `RMSNorm_${data.key}`}
                  tex={masked ? undefined : String.raw`x/\sqrt{\operatorname{mean}(x^2)+\varepsilon}\,\odot\,\gamma_{\mathrm{${data.key}}}`}
                  foot={masked ? undefined : dimLine(5)}
                  side={masked ? undefined : <NormReadout info={data.valInfo} eps={data.eps} />}
                >
                  {masked ? (
                    <SkipNote>
                      <span className="font-mono">m_{data.key}</span> is added to the sum as it is, times its gate, which is 1 here (model.py:221-224).
                    </SkipNote>
                  ) : (
                    <VectorStrip
                      values={data.valTerm}
                      columns={32}
                      cellHeight={13}
                      domain={data.postDomain}
                      onHover={(i) => setHoverDim(i)}
                      highlightIndex={hoverDim}
                      hint={false}
                    />
                  )}
                </Stage>
              </LaneBox>
            </div>

            {/* ── the sum ── */}
            <div className="mt-2 flex justify-center">
              <Arrow on={revealed(6)} />
            </div>
            <Stage
              innerRef={stageRef(6)}
              n={7}
              active={step === 6}
              revealed={revealed(6)}
              title="Sum: the token’s input vector"
              tex={masked ? String.raw`x_0 = \text{③} + m_{\mathrm{${data.key}}}` : String.raw`x_0 = \text{③} + \text{⑥}`}
              foot={dimLine(6)}
              className="mt-0"
            >
              <div className="grid items-center gap-x-3 gap-y-1.5 sm:grid-cols-[120px_minmax(0,1fr)]">
                <SumLabel icon={null}>column term</SumLabel>
                <VectorStrip values={data.colTerm} columns={64} cellHeight={12} domain={data.postDomain} showReadout={false} highlightIndex={hoverDim} onHover={(i) => setHoverDim(i)} />
                <SumLabel icon={<Plus size={13} aria-hidden="true" />}>{masked ? `m_${data.key}` : 'value term'}</SumLabel>
                <VectorStrip values={data.valTerm} columns={64} cellHeight={12} domain={data.postDomain} showReadout={false} highlightIndex={hoverDim} onHover={(i) => setHoverDim(i)} />
                <SumLabel icon={<Equal size={13} aria-hidden="true" />} strong>
                  x₀
                </SumLabel>
                <VectorStrip values={data.x0} columns={64} cellHeight={16} domain={data.postDomain} highlightIndex={hoverDim} onHover={(i) => setHoverDim(i)} hint={false} />
              </div>
            </Stage>

          </>
        )}
      </div>
    </Figure>
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// Pieces
// ─────────────────────────────────────────────────────────────────────────────

function TokenHeader({ tok, slot, flash }: { tok: Token; slot: number; flash?: boolean }) {
  return (
    <div
      className={cx(
        'mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg bg-surface-2 px-3 py-2 text-[16px] ring-accent transition-shadow duration-300',
        flash ? 'ring-[3px]' : 'ring-0',
      )}
    >
      <span className="num text-muted">pos {tok.pos}</span>
      {tok.isPadding ? (
        <span className="text-ink-2">padding (is_padding = true)</span>
      ) : (
        <>
          <TableSwatch slot={slot} label={rowRef(tok)} />
          <span className="font-mono text-ink">{phraseOf(tok)}</span>
          <SemBadge sem={tok.sem} size="sm" showLabel />
          <span className="text-ink-2">
            value{' '}
            {tok.isTarget ? (
              <span className="target-outline rounded px-1 font-mono font-bold text-accent-text">?</span>
            ) : (
              <span className="font-mono text-ink">{formatRaw(tok.raw, tok.sem === SemType.DateTime ? 'datetime' : undefined)}</span>
            )}
          </span>
          {tok.isTarget && <span className="text-[15px] text-ink-2">masked target (masks = true)</span>}
        </>
      )}
    </div>
  );
}

function LaneBox({ title, subtitle, gate, children }: { title: string; subtitle: string; gate: ReactNode; children: ReactNode }) {
  return (
    <section className="min-w-0 rounded-xl border border-border bg-surface-2/50 p-3" aria-label={title}>
      <div className="mb-2.5 flex flex-wrap items-start justify-between gap-x-3 gap-y-1">
        <div>
          <div className="text-[16.5px] font-semibold text-ink">{title}</div>
          <div className="text-[15px] text-ink-2">{subtitle}</div>
        </div>
        <div className="text-[15px] text-ink-2">{gate}</div>
      </div>
      <div className="flex flex-col">{children}</div>
    </section>
  );
}

/** The value-branch selector of model.py:216-224 for this token: which of the 8 gates is 1. */
function GateSwitch({ tok }: { tok: Token }) {
  return (
    <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Value branch selected by sem_type and mask">
      <span className="mr-0.5 text-[15px] text-ink-2">branch</span>
      {[SemType.Number, SemType.Text, SemType.DateTime, SemType.Boolean].map((s) => {
        const on = !tok.isPadding && tok.sem === s;
        return (
          <span
            key={s}
            title={`${SEM_LABEL[s]} branch${on ? (tok.mask ? ': mask vector' : ': encoder') : ': gate 0'}`}
            aria-label={`${SEM_LABEL[s]} branch${on ? (tok.mask ? ': mask vector' : ': encoder') : ': gate 0'}`}
            className={cx(
              'inline-flex h-6 items-center gap-1 rounded-md border px-1.5 text-[14px]',
              on ? (tok.mask ? 'target-outline border-transparent bg-accent-soft text-accent-text' : 'border-accent bg-accent-soft text-ink') : 'border-border text-muted',
            )}
          >
            <SemBadge sem={s} size="sm" bare />
            {on && (tok.mask ? <span className="font-mono">m</span> : <span className="font-mono">enc</span>)}
          </span>
        );
      })}
    </div>
  );
}

function Stage({
  innerRef,
  n,
  title,
  tex,
  active,
  revealed,
  skipped,
  foot,
  side,
  className,
  children,
}: {
  innerRef?: (el: HTMLDivElement | null) => void;
  n: number;
  title: ReactNode;
  tex?: string;
  active: boolean;
  revealed: boolean;
  skipped?: boolean;
  foot?: ReactNode;
  side?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <motion.div
      ref={innerRef}
      initial={false}
      animate={{ y: revealed ? 0 : 4 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      className={cx(
        'min-w-0 scroll-mt-[calc(var(--topbar-h)+var(--emb-player-h,0px)+20px)] scroll-mb-4 rounded-lg border bg-surface px-3 py-2.5 transition-shadow',
        skipped ? 'border-dashed border-border-strong' : 'border-border',
        active && 'ring-2 ring-accent',
        className,
      )}
      aria-current={active ? 'step' : undefined}
      data-revealed={revealed}
    >
      <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
        <div className="flex items-baseline gap-2">
          <span
            aria-hidden="true"
            className={cx(
              'inline-flex h-5 w-5 shrink-0 translate-y-[3px] items-center justify-center rounded-full text-[13.5px] font-bold',
              active ? 'bg-accent text-accent-ink' : 'bg-surface-3 text-ink-2',
            )}
          >
            {n}
          </span>
          <span className={cx('text-[16px] font-semibold', revealed ? 'text-ink' : 'text-ink-2')}>{title}</span>
        </div>
        {tex && (
          <span className="min-w-0 max-w-full overflow-x-auto text-[17.5px] text-ink-2">
            <TeX tex={tex} />
          </span>
        )}
      </div>
      {/* Not reached yet: only the data fades (strips gray and light); every text stays at full contrast. */}
      <div
        className={cx(
          side && 'grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto]',
          '[&_svg]:transition-[opacity,filter] [&_svg]:duration-300',
          !revealed && '[&_svg]:opacity-35 [&_svg]:grayscale',
        )}
      >
        <div className="min-w-0">{children}</div>
        {side}
      </div>
      {foot !== undefined && (
        <div className="num mt-1 min-h-[1.4em] text-[14.5px] leading-snug text-ink-2" aria-live="polite">
          {foot}
        </div>
      )}
    </motion.div>
  );
}

function Arrow({ on }: { on: boolean }) {
  return (
    <div className="flex h-6 items-center justify-center" aria-hidden="true">
      <motion.span initial={false} animate={{ opacity: on ? 1 : 0.3, y: on ? 0 : -2 }} transition={{ duration: 0.3 }}>
        <ArrowDown size={16} className={on ? 'text-accent' : 'text-axis'} />
      </motion.span>
    </div>
  );
}

function SkipNote({ children }: { children: ReactNode }) {
  return <p className="!m-0 text-[15px] leading-snug text-ink-2">{children}</p>;
}

function SumLabel({ icon, strong, children }: { icon: ReactNode; strong?: boolean; children: ReactNode }) {
  return (
    <div className={cx('flex items-center gap-1.5 text-[15px] sm:justify-end', strong ? 'font-semibold text-ink' : 'text-ink-2')}>
      {icon && <span className="text-muted">{icon}</span>}
      <span className="font-mono">{children}</span>
    </div>
  );
}

function NormReadout({ info, eps }: { info: ReturnType<typeof rmsInfo>; eps: number }) {
  return (
    <dl className="num grid grid-cols-[auto_auto] content-start gap-x-2 gap-y-0.5 text-[14.5px] text-ink-2 sm:min-w-[150px]">
      <dt className="text-muted">mean(x²)</dt>
      <dd className="text-right text-ink">{fmt(info.ms, 4)}</dd>
      <dt className="text-muted">+ ε</dt>
      <dd className="text-right text-ink">{fmt(eps, 4)}</dd>
      <dt className="text-muted">scale</dt>
      <dd className="text-right text-ink">×{fmt(info.scale, 2)}</dd>
      <dt className="text-muted" title="ε as a share of mean(x²) + ε, the sum under the square root">
        ε of the sum
      </dt>
      <dd className="text-right text-ink">{(info.epsShare * 100).toFixed(info.epsShare < 0.01 ? 1 : 0)}%</dd>
    </dl>
  );
}

function ScalarInput({ tok }: { tok: Token }) {
  const z = tokenScalar(tok);
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <div className="flex items-center gap-2">
        <span className="font-mono text-[15px] text-ink-2">{formatRaw(tok.raw, tok.sem === SemType.DateTime ? 'datetime' : undefined)}</span>
        <ArrowRight size={14} className="text-muted" aria-hidden="true" />
        <span className="rounded-md border border-border bg-surface-2 px-2 py-1 font-mono text-[17.5px] font-semibold text-ink">{fmtSigned(z, 4)}</span>
      </div>
      <span className="text-[14.5px] leading-snug text-ink-2">
        {VALUE_FIELD[tok.sem]}: the z-score from preprocessing{tok.sem === SemType.DateTime ? ' (one global datetime z-score per DB)' : ''}
        {tok.sem === SemType.Boolean ? ' of 0/1 (true > 0, false < 0)' : ''}
      </span>
    </div>
  );
}
