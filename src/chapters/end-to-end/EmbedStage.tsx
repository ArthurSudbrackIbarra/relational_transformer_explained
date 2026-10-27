/**
 * Stage 4: the input embedding x₀, as whole matrices (rt/model.py:208-224; spec §5.2):
 *
 *   x_i = [¬pad] RMSNorm_col(W_col c_i + b_col)
 *       + Σ_t [sem_i = t ∧ ¬mask_i ∧ ¬pad] RMSNorm_t(W_t v_i + b_t) + [sem_i = t ∧ mask_i ∧ ¬pad] m_t
 *
 * c_i is the frozen MiniLM embedding of "<col> of <table>" (col_name_values) and v_i the token's
 * value (a z-scored scalar, or the MiniLM embedding of a text value). Every panel is the engine's
 * capture (ForwardResult.embedding) or a batch field rebuilt exactly as fly.rs writes it
 * (live.ts `textFields`); math.test.ts checks every recomputed entry against the capture.
 */
import { useMemo, type ReactNode } from 'react';
import { Cite, Figure, StatusBadge, StepPlayer, TeX, cx, useStepPlayer } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { SEM_TYPE_NAMES, SemType } from '@/engine/types';
import { fmtNum, int, plural } from './format';
import { W, textFields, useE2E, type E2ELive } from './live';
import { DotLine, MatMul, Panel, fitScale, type LinkedHover } from './MatMul';
import type { Cell } from './Matrix';
import { linearEntry, rowOf, scalarInput } from './math';
import { NormView } from './NormView';
import { HoverHint, MathHead, Readout, Sym, tok } from './opParts';
import { FocusLine, LoadingNote, OpLayout, useViewHover, useVisualWidth } from './shared';

type EmbOp = 'lookup' | 'colProj' | 'colNorm' | 'value' | 'sum';
const OPS: { id: EmbOp; short: string }[] = [
  { id: 'lookup', short: 'Text lookups' },
  { id: 'colProj', short: 'Column-name projection' },
  { id: 'colNorm', short: 'Its RMSNorm' },
  { id: 'value', short: 'Value term' },
  { id: 'sum', short: 'Sum: x₀' },
];
const KEYS = ['number', 'text', 'datetime', 'boolean'] as const;

export function EmbedStage() {
  const live = useE2E();
  const player = useStepPlayer(OPS.length, { intervalMs: 3200 });
  const op = OPS[player.index].id;
  const [hover, setHover] = useViewHover<LinkedHover>([player.index, live.ctx]);
  const [ref, size] = useElementSize<HTMLDivElement>();
  const width = useVisualWidth(size.width);
  const ready = !!live.fr && !!live.weights && !!live.store;
  return (
    <Figure
      wide
      title={
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span>From batch fields to x₀ · {OPS[player.index].short}</span>
          {live.assets.weightsKind === 'init' && <StatusBadge status="warning">untrained weights</StatusBadge>}
        </span>
      }
      caption={
        <>
          One row per token, the followed token outlined. Hover any entry for its arithmetic; click a row to follow that token. Text embeddings
          are the frozen MiniLM vectors; everything after them uses the trained mini model’s weights.
        </>
      }
    >
      <StepPlayer {...player} unit="step" label="Play through the input embedding" describe={(i) => <>{OPS[i].short}.</>} />
      <ol className="!mt-3 !mb-0 flex flex-wrap gap-1.5 !p-0" style={{ listStyle: 'none' }} aria-label="Steps of the input embedding">
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
      <FocusLine className="mt-3" />
      <div ref={ref} className="mt-4 min-w-0">
        {!ready ? (
          <LoadingNote />
        ) : op === 'lookup' ? (
          <LookupOp live={live} width={width} hover={hover} setHover={setHover} />
        ) : op === 'colProj' ? (
          <ColProjOp live={live} width={width} hover={hover} setHover={setHover} />
        ) : op === 'colNorm' ? (
          <NormView
            live={live}
            x={live.fr!.embedding.colPreNorm}
            y={live.fr!.embedding.colTerm}
            gamma={W(live.weights!, 'norm_dict.col_name.weight')}
            xTex="u"
            gammaTex="\gamma^{\text{col}}"
            yTex="c'"
            width={width}
            hover={hover}
            setHover={setHover}
            title="Normalize the column-name term"
            cite={
              <>
                <code>norm_dict["col_name"]</code>, one of five input RMSNorms <Cite kind="code" ref="rt/model.py:132-140" />, applied at{' '}
                <Cite kind="code" ref="rt/model.py:209-214" />.
              </>
            }
          >
            <p className="!mb-0">
              u is the projection from the previous step. In the paper’s formula the column-name term is a plain linear map; the code adds a bias and
              this RMSNorm <Cite kind="spec" ref="§10 #2" />.
            </p>
          </NormView>
        ) : op === 'value' ? (
          <ValueOp live={live} width={width} hover={hover} setHover={setHover} />
        ) : (
          <SumOp live={live} width={width} hover={hover} setHover={setHover} />
        )}
      </div>
    </Figure>
  );
}

interface OpProps {
  live: E2ELive;
  width: number;
  hover: LinkedHover | null;
  setHover: (h: LinkedHover | null) => void;
}

function useTokens(live: E2ELive) {
  return useMemo(() => ({ drawRows: live.numReal, rowSlots: live.tokenSlots, tokenRows: true }), [live.numReal, live.tokenSlots]);
}

function useFields(live: E2ELive) {
  const { ctx, pdb, dText, store } = live;
  return useMemo(() => textFields({ ctx, pdb, dText }, store!), [ctx, pdb, dText, store]);
}

function LookupOp({ live, width, hover, setHover }: OpProps) {
  const { ctx, pdb, S, dText, numReal, focus } = live;
  const tokens = useTokens(live);
  const f = useFields(live);
  const nPhrases = useMemo(() => new Set(ctx.tokens.slice(0, numReal).map((t) => t.colNameIdx)).size, [ctx, numReal]);
  const nText = ctx.tokens.slice(0, numReal).filter((t) => t.sem === SemType.Text).length;
  const cw = Math.max(0.6, Math.min(1.4, (width - 24) / (dText * 2 + 30)));
  const cell = hover && hover.which === 'C' ? { r: hover.r, c: hover.c } : null;
  const panelHover = (which: 'A' | 'C') => (k: Cell | null) => setHover(k ? { which, r: k.r, c: k.c } : null);
  const hA = hover && hover.which === 'A' ? { r: hover.r, c: hover.c } : null;
  const row = cell?.r ?? hA?.r ?? null;
  const t = row !== null ? ctx.tokens[row] : null;
  return (
    <OpLayout
      visual={
        <div className="flex flex-wrap items-start gap-x-6 gap-y-4">
          <Panel
            o={{ data: f.colName, rows: S, cols: dText, ...tokens, name: <code>col_name_values</code>, label: 'column-phrase embeddings' }}
            cellW={cw}
            cellH={2.5}
            focusRow={focus}
            onFocusRow={live.setFocus}
            hover={hA}
            onHover={panelHover('A')}
            markRow={cell?.r ?? null}
          />
          <Panel
            o={{ data: f.text, rows: S, cols: dText, ...tokens, name: <code>text_values</code>, label: 'text-value embeddings' }}
            cellW={cw}
            cellH={2.5}
            focusRow={focus}
            onFocusRow={live.setFocus}
            hover={cell}
            onHover={panelHover('C')}
            markRow={hA?.r ?? null}
          />
        </div>
      }
      math={
        <MathHead
          title="Look up the frozen text embeddings"
          tex={[`c_i = E(\\text{“}\\langle\\text{col}\\rangle\\text{ of }\\langle\\text{table}\\rangle\\text{”})`, `c_i \\in \\mathbb{R}^{${dText}}`]}
          cite={
            <>
              <code>get_text_emb</code> copies precomputed rows into the batch <Cite kind="code" ref="rustler/src/fly.rs:469-486" />{' '}
              <Cite kind="code" ref="rustler/src/fly.rs:528-532" />; E is all-MiniLM-L12-v2, frozen, unit-norm <Cite kind="spec" ref="§3.6" />.
            </>
          }
        >
          <p className="!mb-0">
            No text encoder runs here: every string of the database was embedded once, before training, and the sampler copies rows of that table.{' '}
            <code>col_name_values</code> holds the phrase of each token’s column, so the {int(numReal)} rows take only {plural(nPhrases, 'distinct value')}.{' '}
            <code>text_values</code> holds the value of the {plural(nText, 'text cell')}; every other token stores the embedding of string 0 (“{pdb.text[0]}”),
            because the preprocessing writes text index 0 for non-text cells <Cite kind="code" ref="rustler/src/pre.rs:553-605" />. The input gate
            multiplies those rows by 0.
          </p>
          <Readout>
            {t ? (
              <>
                {tok(live, t.pos)}: column phrase <q>{pdb.text[t.colNameIdx]}</q> (text index {t.colNameIdx}); text value{' '}
                {t.sem === SemType.Text ? (
                  <>
                    <q>{pdb.text[t.textIdx]}</q> (index {t.textIdx})
                  </>
                ) : (
                  <>none, so the row is E(“{pdb.text[0]}”), gated out</>
                )}
                .
              </>
            ) : (
              <span className="text-muted">Hover a row to see which strings it holds.</span>
            )}
          </Readout>
        </MathHead>
      }
    />
  );
}

function ColProjOp({ live, width, hover, setHover }: OpProps) {
  const { S, d, dText, focus, weights, fr } = live;
  const tokens = useTokens(live);
  const f = useFields(live);
  const Wc = W(weights!, 'enc_dict.col_name.weight');
  const bc = W(weights!, 'enc_dict.col_name.bias');
  const s = fitScale(width, dText * 0.9 + d * 3 + 16 + 60, 0.5);
  const hv = hover?.which === 'C' ? linearEntry(rowOf(f.colName, hover.r, dText), Wc, dText, hover.c, bc) : null;
  return (
    <OpLayout
      visual={
        <MatMul
          A={{ data: f.colName, rows: S, cols: dText, ...tokens, name: <TeX tex="C" />, text: 'C', label: 'col_name_values' }}
          B={{ data: Wc, rows: dText, cols: d, transpose: true, name: <TeX tex="W_{\text{col}}^{\top}" />, text: 'W_colᵀ', label: 'W_col transposed' }}
          C={{ data: fr!.embedding.colPreNorm, rows: S, cols: d, ...tokens, name: <TeX tex="u = C\,W_{\text{col}}^{\top} + b" />, text: 'u', label: 'column-name projection' }}
          bias={{ data: bc, name: <>+ <TeX tex="b_{\text{col}}" /></>, text: 'b_col', label: 'bias of the column-name encoder' }}
          cellS={2.5 * s}
          cellK={0.9 * s}
          cellN={3 * s}
          focusRow={focus}
          onFocusRow={live.setFocus}
          hover={hover}
          setHover={setHover}
        />
      }
      math={
        <MathHead
          title="Project the column phrase to d"
          tex={[`u_i = W_{\\text{col}}\\,c_i + b_{\\text{col}}`, `W_{\\text{col}} \\in \\mathbb{R}^{${d}\\times ${dText}}`]}
          cite={
            <>
              <code>enc_dict["col_name"] = nn.Linear(d_text, d_model, bias=True)</code> <Cite kind="code" ref="rt/model.py:115-123" />, applied at{' '}
              <Cite kind="code" ref="rt/model.py:209-214" />.
            </>
          }
        >
          <p className="!mb-0">
            One linear map takes every token’s {dText}-d phrase embedding to the model width {d}. It is shared by all columns of all tables of all
            databases: the column’s identity is only in the phrase it reads.
          </p>
          <Readout>
            {hv && hover ? (
              <>
                u[{tok(live, hover.r)}, {hover.c}] = Σ<sub>k</sub> c[·, k]·W<sub>col</sub>[{hover.c}, k] + b ={' '}
                <DotLine terms={hv.terms} total={hv.value} extra={<> + {fmtNum(hv.bias, 3, true)}</>} />
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

function ValueOp({ live, width, hover, setHover }: OpProps) {
  const { ctx, S, d, dText, numReal, focus, weights, fr, pdb } = live;
  const tokens = useTokens(live);
  const f = useFields(live);
  const w = weights!;
  // The engine never computes a masked or padding token's value projection (the gate is 0, spec §5.2):
  // show those rows of u_v as gated out rather than as zeros.
  const gatedPre = useMemo(() => {
    const out = new Uint8Array(S * d);
    for (const t of ctx.tokens) if (t.isPadding || t.mask) out.fill(1, t.pos * d, (t.pos + 1) * d);
    return out;
  }, [ctx, S, d]);
  const c = 3 * fitScale(width, d * 3 * 2 + 40 + 60);
  const cell = hover && hover.which === 'C' ? { r: hover.r, c: hover.c } : null;
  const onH = (k: Cell | null) => setHover(k ? { which: 'C', r: k.r, c: k.c } : null);
  const t = cell ? ctx.tokens[cell.r] : null;
  const key = t ? KEYS[t.sem] : 'number';
  const masked = ctx.tokens.filter((x) => x.mask);
  // readout arithmetic for the hovered entry
  let pre: ReturnType<typeof linearEntry> | null = null;
  if (t && !t.mask && !t.isPadding) {
    const Wt = W(w, `enc_dict.${key}.weight`);
    const bt = W(w, `enc_dict.${key}.bias`);
    pre = t.sem === SemType.Text ? linearEntry(rowOf(f.text, t.pos, dText), Wt, dText, cell!.c, bt) : linearEntry(Float32Array.of(scalarInput(t)!), Wt, 1, cell!.c, bt);
  }
  return (
    <OpLayout
      visual={
        <div className="grid grid-cols-1 gap-5">
          <div className="flex flex-wrap items-start gap-x-3 gap-y-3">
            <Panel
              o={{ data: fr!.embedding.valuePreNorm, rows: S, cols: d, ...tokens, blocked: gatedPre, name: <TeX tex="u^{\text{val}}_i = W_t v_i + b_t" />, label: 'value projection, by type' }}
              cellW={c}
              cellH={2.6}
              focusRow={focus}
              onFocusRow={live.setFocus}
              hover={cell}
              onHover={onH}
            />
            <Sym>→</Sym>
            <Panel
              o={{ data: fr!.embedding.valueTerm, rows: S, cols: d, ...tokens, name: <TeX tex="\mathrm{RMSNorm}_t(u^{\text{val}}_i)\ \text{or}\ m_t" />, label: 'value term' }}
              cellW={c}
              cellH={2.6}
              focusRow={focus}
              onFocusRow={live.setFocus}
              hover={cell}
              onHover={onH}
            />
          </div>
          <ValueWeights live={live} width={width} />
        </div>
      }
      math={
        <MathHead
          title="The value term: one encoder per type, or the mask vector"
          tex={[`\\text{val}_i = \\mathrm{RMSNorm}_t(W_t\\,v_i + b_t)`, `\\text{val}_i = m_t \\;\\;\\text{(the masked cell)}`]}
          cite={
            <>
              <code>enc_dict</code>, <code>norm_dict</code>, <code>mask_embs</code> <Cite kind="code" ref="rt/model.py:115-146" />; the gated sum{' '}
              <Cite kind="code" ref="rt/model.py:216-224" />; m<sub>t</sub> is added without a norm <Cite kind="spec" ref="§5.2" />.
            </>
          }
        >
          <p className="!mb-0">
            Each token uses the encoder of its own type t: a Linear(1, d) for a z-scored number, datetime or boolean, a Linear({dText}, d) for a text
            embedding, each followed by its own RMSNorm. The code runs every type’s encoder on every position and multiplies the wrong types by 0.
            The masked cell gets its type’s learned vector m<sub>t</sub> instead
            {masked.length === 1 ? (
              <>
                {' '}
                ({tok(live, masked[0].pos)} gets m<sub>{SEM_TYPE_NAMES[masked[0].sem]}</sub>)
              </>
            ) : null}
            , so its label never enters the input.
          </p>
          <Readout>
            {t && cell ? (
              t.mask ? (
                <>
                  {tok(live, t.pos)} is the masked cell: its row is m<sub>{key}</sub>, entry {cell.c} ={' '}
                  <strong className="num text-ink">{fmtNum(fr!.embedding.valueTerm[t.pos * d + cell.c])}</strong>. Its projection is gated out (gray).
                </>
              ) : pre ? (
                t.sem === SemType.Text ? (
                  <>
                    {tok(live, t.pos)} (text <q>{pdb.text[t.textIdx]}</q>): u = <DotLine terms={pre.terms} total={pre.value} extra={<> + {fmtNum(pre.bias, 3, true)}</>} />; after
                    RMSNorm<sub>text</sub>: <strong className="num text-ink">{fmtNum(fr!.embedding.valueTerm[t.pos * d + cell.c])}</strong>
                  </>
                ) : (
                  <>
                    {tok(live, t.pos)} ({key}, v = {fmtNum(scalarInput(t)!)}): u = {fmtNum(scalarInput(t)!, 3, true)}·{fmtNum(pre.terms[0].b, 3, true)} +{' '}
                    {fmtNum(pre.bias, 3, true)} = {fmtNum(pre.value)}; after RMSNorm<sub>{key}</sub>:{' '}
                    <strong className="num text-ink">{fmtNum(fr!.embedding.valueTerm[t.pos * d + cell.c])}</strong>
                  </>
                )
              ) : (
                <>padding: every term is 0</>
              )
            ) : (
              <HoverHint />
            )}
          </Readout>
          <p className="!mt-2 !mb-0 text-[14.5px] text-muted">
            m<sub>text</sub> exists but is never used: text cells are never masked <Cite kind="code" ref="rt/model.py:256-257" />.{' '}
            {int(numReal)} rows: {(['number', 'text', 'datetime', 'boolean'] as const).map((k, i) => `${ctx.tokens.slice(0, numReal).filter((x) => x.sem === i).length} ${k}`).join(', ')}.
          </p>
        </MathHead>
      }
    />
  );
}

/**
 * Every weight of the value term, by type: the encoder W_t and bias b_t (enc_dict), the RMSNorm gain
 * γ_t (norm_dict) and the mask vector m_t (mask_embs), rt/model.py:115-146. A scalar encoder
 * Linear(1, d) has a d × 1 weight, drawn here as a row; the text encoder's is d × 384.
 */
function ValueWeights({ live, width }: { live: E2ELive; width: number }) {
  const w = live.weights!;
  const { d, dText } = live;
  const strip = Math.max(1.6, Math.min(3, (width - 120) / d));
  const textW = Math.max(0.5, Math.min(1, (width - 120) / dText));
  const vec = (key: string, text: string, label: string, name: ReactNode, scale?: 'sequential') => {
    const data = W(w, key);
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of data) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    return (
      <Panel
        o={{ data, rows: 1, cols: d, scale, domain: scale ? [Math.min(0, lo), hi] : undefined, name, text, label, shape: `1 × ${d}` }}
        cellW={strip}
        cellH={12}
      />
    );
  };
  return (
    <div>
      <div className="mb-2 text-[15.5px] font-semibold text-ink">Every weight of the value term, one set per type</div>
      <div className="grid grid-cols-1 gap-3">
        {KEYS.map((k) => (
          <div key={k} className="rounded-lg border border-border px-3 py-2.5">
            <div className="mb-1.5 text-[15px] font-semibold text-ink">{k}</div>
            <div className="flex flex-wrap items-start gap-x-5 gap-y-2">
              {k === 'text' ? (
                <Panel
                  o={{ data: W(w, `enc_dict.text.weight`), rows: d, cols: dText, name: <TeX tex="W_{\text{text}}" />, text: 'W_text', label: 'text encoder weight' }}
                  cellW={textW}
                  cellH={1.2}
                />
              ) : (
                vec(`enc_dict.${k}.weight`, `W_${k}`, `${k} encoder weight, one value per output dimension`, <TeX tex={`W_{\\text{${k}}}`} />)
              )}
              {vec(`enc_dict.${k}.bias`, `b_${k}`, `${k} encoder bias`, <TeX tex={`b_{\\text{${k}}}`} />)}
              {vec(`norm_dict.${k}.weight`, `γ_${k}`, `${k} RMSNorm gain`, <TeX tex={`\\gamma^{\\text{${k}}}`} />, 'sequential')}
              {vec(`mask_embs.${k}`, `m_${k}`, `${k} mask vector`, <TeX tex={`m_{\\text{${k}}}`} />)}
            </div>
          </div>
        ))}
      </div>
      <p className="!mt-2 !mb-0 text-[14.5px] leading-snug text-muted">
        A scalar encoder is <code>nn.Linear(1, d)</code>, so its weight is a single column of {d} values, drawn here as a row. γ is on its own
        sequential scale; the other rows use the diverging scale.
      </p>
    </div>
  );
}

function SumOp({ live, width, hover, setHover }: OpProps) {
  const { S, d, numReal, focus, fr } = live;
  const tokens = useTokens(live);
  const e = fr!.embedding;
  const dom = useMemo(() => {
    let m = 0;
    for (let i = 0; i < numReal * d; i++) m = Math.max(m, Math.abs(e.colTerm[i]), Math.abs(e.valueTerm[i]), Math.abs(e.x0[i]));
    return m || 1;
  }, [e, numReal, d]);
  const c = 2.6 * fitScale(width, (d * 2.6 + 30) * 3);
  const cell = hover && hover.which === 'C' ? { r: hover.r, c: hover.c } : null;
  const onH = (k: Cell | null) => setHover(k ? { which: 'C', r: k.r, c: k.c } : null);
  const panel = (data: Float32Array, name: ReactNode, label: string) => (
    <Panel o={{ data, rows: S, cols: d, ...tokens, domain: dom, name, label }} cellW={c} cellH={c} focusRow={focus} onFocusRow={live.setFocus} hover={cell} onHover={onH} />
  );
  let n = 0;
  for (let j = 0; j < d; j++) n += e.x0[focus * d + j] ** 2;
  return (
    <OpLayout
      visual={
        <div className="flex flex-wrap items-start gap-x-3 gap-y-3">
          {panel(e.colTerm, <TeX tex="c'" />, 'column-name term')}
          <Sym>+</Sym>
          {panel(e.valueTerm, <TeX tex="\text{val}" />, 'value term')}
          <Sym>=</Sym>
          {panel(e.x0, <TeX tex="x_0" />, 'input embedding')}
        </div>
      }
      math={
        <MathHead
          title="Add the two terms: x₀"
          tex={[`x_{0,i} = c'_i + \\text{val}_i`, `x_0 \\in \\mathbb{R}^{${int(S)}\\times ${d}}`]}
          cite={
            <>
              <code>x = 0</code>, then each term is added with its gate <Cite kind="code" ref="rt/model.py:208-224" />; padding rows are exactly 0{' '}
              <Cite kind="spec" ref="§5.2" />.
            </>
          }
        >
          <p className="!mb-0">
            Each token’s starting vector is its column-name term plus its value term. Nothing in it says where the token sits in the context: there are
            no positional encodings <Cite kind="paper" ref="p.5 §3.1" />. Followed token: ‖x₀‖ = {fmtNum(Math.sqrt(n), 2)}. This matrix enters block 1.
          </p>
          <Readout>
            {cell ? (
              <>
                x₀[{tok(live, cell.r)}, {cell.c}] = {fmtNum(e.colTerm[cell.r * d + cell.c], 3, true)} + {fmtNum(e.valueTerm[cell.r * d + cell.c], 3, true)} ={' '}
                <strong className="num text-ink">{fmtNum(e.x0[cell.r * d + cell.c])}</strong>
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
