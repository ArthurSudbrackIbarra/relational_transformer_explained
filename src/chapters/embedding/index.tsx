/**
 * Chapter 8 "embedding": The token embedding (BUILD_PLAN §6, docs/CHAPTER_BRIEFS.md §8).
 *
 * Computes x₀ for every token of the live context with the checkpoint the page runs. All vectors are
 * the forward pass's own capture (ForwardResult.embedding, a mirror of PE/rt/model.py:208-224) or
 * engine kernels on the live weights. Spec: research/RT_TECHNICAL_SPEC.md §5.1-5.3, §7.2, §7.4, §10 #2.
 */
import {
  Callout,
  ChapterHeader,
  ChapterNav,
  ChapterPage,
  Cite,
  CodeRef,
  MathBlock,
  Quiz,
  ScaleNote,
  Section,
  TeX,
  Term,
} from '@/components';
import { MINI_CONFIG, SEM_KEYS } from '@/engine/model';
import { SemType } from '@/engine/types';
import type { ReactNode } from 'react';
import { ContextMatrix, PaddingCheck } from './ContextMatrix';
import { EpsFinding } from './EpsFinding';
import { EpsKnee } from './EpsKnee';
import { EmbeddingParamStats, GateGrid, SymbolTable } from './FormulaParts';
import { GainStrips } from './GainStrips';
import { fmt, int, phraseOf, rowRef, useEmbeddingStable, type EmbeddingStable } from './live';
import { MaskVectors } from './MaskVectors';
import { meanSquare, slice } from './math';
import { RmsNormPlane } from './RmsNormPlane';
import { FLY_PADDING_65_85, MODEL_EMBED_208_224, MODEL_ENCODERS_115_146 } from './snippets';
import { JumpLink } from './nav';
import { TokenEmbedder } from './TokenEmbedder';
import { ValueCurve } from './ValueCurve';

const CODE_FORMULA = String.raw`\begin{aligned}
x_i \;=\;& [\lnot\mathrm{pad}_i]\cdot \underbrace{\mathrm{RMSNorm}_{\mathrm{col}}\!\left(W_{\mathrm{col}}\,c_i + b_{\mathrm{col}}\right)}_{\small\text{column term}}\\[2pt]
&+\sum_{t}\,[t_i{=}t \land \lnot\mathrm{mask}_i \land \lnot\mathrm{pad}_i]\cdot \underbrace{\mathrm{RMSNorm}_t\!\left(W_t\,v_i^{(t)} + b_t\right)}_{\small\text{value term}}\\[2pt]
&+\sum_{t}\,[t_i{=}t \land \mathrm{mask}_i \land \lnot\mathrm{pad}_i]\cdot \underbrace{m_t}_{\small\text{mask vector}}
\end{aligned}`;

/** The same formula broken into more lines for phone widths (no horizontal scrolling at 17.5px, scripts ≥ 12.5px). */
const CODE_FORMULA_NARROW = String.raw`\begin{aligned}
x_i \;=\;& [\lnot\mathrm{pad}_i]\\
&\cdot \underbrace{\mathrm{RMSNorm}_{\mathrm{col}}\!\left(W_{\mathrm{col}}\,c_i + b_{\mathrm{col}}\right)}_{\text{column term}}\\[2pt]
&+\sum_{t}\,[t_i{=}t \land \lnot\mathrm{mask}_i \land \lnot\mathrm{pad}_i]\\
&\cdot \underbrace{\mathrm{RMSNorm}_t\!\left(W_t\,v_i^{(t)} + b_t\right)}_{\text{value term}}\\[2pt]
&+\sum_{t}\,[t_i{=}t \land \mathrm{mask}_i \land \lnot\mathrm{pad}_i]\\
&\cdot \underbrace{m_t}_{\text{mask vector}}
\end{aligned}`;

const PAPER_FORMULA = String.raw`\mathbf{x} \;=\; \mathbf{W}_d\,\mathbf{r} \;+\; \mathbf{W}\,\mathcal{E}^{\text{schema}}(c, t)`;

const RMSNORM_FORMULA = String.raw`\mathrm{RMSNorm}(x)_j \;=\; \frac{x_j}{\sqrt{\tfrac{1}{d}\sum_{k=1}^{d} x_k^2 \;+\; \varepsilon}}\;\gamma_j,\qquad \varepsilon = 2^{-7} = 0.0078125`;
/** Phone width: mean(x²) on its own line, so no script sits inside another script (every glyph ≥ 12.5px at 17.5px). */
const RMSNORM_FORMULA_NARROW = String.raw`\begin{gathered}\mathrm{RMSNorm}(x)_j \;=\; \frac{x_j}{\sqrt{\operatorname{mean}(x^2) + \varepsilon}}\,\gamma_j\\[8pt] \operatorname{mean}(x^2) = \frac{1}{d}\sum_{k=1}^{d} x_k^2\\[8pt] \varepsilon = 2^{-7} = 0.0078125\end{gathered}`;

export default function EmbeddingChapter() {
  const live = useEmbeddingStable();
  const { ctx, target, weights, assets } = live;
  const tgt = ctx.tokens[target];
  const d = weights?.config.dModel ?? 64;

  return (
    <ChapterPage slug="embedding">
      <ChapterHeader
        slug="embedding"
        lede={
          <>
            Before any attention happens, every cell token becomes one vector of d numbers, x₀. It is the sum of two parts: a term that says which
            column of which table the cell belongs to, and a term that says what value it holds. The masked target gets a trainable mask vector
            in place of its value. This chapter computes x₀ for the live context, number by number.
          </>
        }
        learn={[
          'The input embedding exactly as the code computes it, and how it relates to the paper’s shorter formula',
          'How a column phrase and a value become two 64-d vectors, with the real numbers of the checkpoint this page runs',
          'What RMSNorm does, and why its ε shortens small inputs such as the column projection',
          'Why every number column shares one encoder, and what the masked target’s vector still knows',
        ]}
      >
        <p className="mt-6 text-[17px] leading-relaxed text-ink-2">
          <span className="eyebrow mr-2">Live context</span>
          {tgt ? (
            <>
              target <span className="text-ink">{rowRef(tgt)}</span> “{phraseOf(tgt)}” ·{' '}
            </>
          ) : null}
          {int(ctx.numReal)} cell tokens{ctx.tokens.length > ctx.numReal ? ` + ${int(ctx.tokens.length - ctx.numReal)} padding` : ''} · d = {d}{' '}
          <ScaleNote inline />
        </p>
      </ChapterHeader>

      {/* ───────────────────────────────────────────────────────────── 1 */}
      <Section id="formula" title="Two terms per token" kicker="Step 1">
        <p>
          The previous chapter turned every cell into a <Term id="cell-token">cell token</Term>: a column phrase, a{' '}
          <Term id="sem-type">semantic type</Term>, a stored value and a few integer ids <Cite kind="spec" ref="§5.1" />. The ids only steer the
          attention masks of the next chapter. Everything the transformer knows about a cell’s content enters through one vector per token, the
          input embedding <TeX tex="x_i \in \mathbb{R}^d" />, with d = {d} here and 256 in the paper <Cite kind="paper" ref="p.6 §4.1" />.
        </p>
        <p>The code builds it as follows <Cite kind="code" ref="rt/model.py:208-224" /> <Cite kind="spec" ref="§5.2" />:</p>
        <MathBlock tex={CODE_FORMULA} className="relative hidden overflow-hidden text-[18px] sm:block" label={<Cite kind="code" ref="rt/model.py:208-224" />} />
        <MathBlock tex={CODE_FORMULA_NARROW} className="relative overflow-hidden text-[17.5px] sm:hidden" label={<Cite kind="code" ref="rt/model.py:208-224" />} />
        <p>Read it one piece at a time.</p>
        <ul>
          <li>
            The <strong>column term</strong> says where the cell lives. <TeX tex="c_i" /> is the frozen MiniLM embedding of its{' '}
            <Term id="column-phrase">column phrase</Term> “&lt;column&gt; of &lt;table&gt;”, for example “{tgt ? phraseOf(tgt) : 'churn of customer-churn'}”{' '}
            <Cite kind="spec" ref="§3.6" />. One linear layer <TeX tex="W_{\mathrm{col}}" /> is shared by every column of every table.
          </li>
          <li>
            The <strong>value term</strong> says what the cell holds. <TeX tex="v_i^{(t)}" /> is the value slot of type t: a single z-score for
            numbers, datetimes and booleans, and the 384-d MiniLM embedding of the string for text <Cite kind="spec" ref="§5.1" />. Every cell has
            all four slots filled, and the gate keeps only the slot of the cell’s own type <Cite kind="code" ref="rustler/src/fly.rs:472-490" />. Each
            type has its own linear layer <TeX tex="W_t" /> and its own RMSNorm.
          </li>
          <li>
            The <Term id="mask-vector">mask vector</Term> <TeX tex="m_t" /> takes the value term’s place for the masked target. It is a trainable
            parameter, one per type, and is added as is, with no RMSNorm <Cite kind="code" ref="rt/model.py:221-224" />.
          </li>
          <li>
            The square brackets are 0/1 gates. For a real token exactly one of the eight value-side gates is 1. For a{' '}
            <Term id="padding">padding</Term> token every gate is 0, the column gate included, so its x₀ is the zero vector{' '}
            <Cite kind="spec" ref="§5.2" />.
          </li>
        </ul>
        <p>
          The figure below shows which gate fires at each of the {int(ctx.tokens.length)} positions of the live context. Most real tokens take a
          value encoder; only one, the target, takes a mask vector <Cite kind="spec" ref="§4.10" />.
        </p>
        <GateGrid live={live} />
        <SymbolTable config={weights?.config ?? MINI_CONFIG} />
        <p>
          The paper writes the same idea in one line, shown below. In it, <TeX tex="\mathbf{r}" /> is the normalized value (for text, its frozen
          MiniLM embedding <TeX tex="\mathbf{r} = \mathcal{E}^{\text{text}}(v)" />) and <TeX tex="\mathcal{E}^{\text{schema}}(c,t)" /> is the text
          embedding of the phrase built from column c and table t <Cite kind="paper" ref="p.5 §3.1" />. So in the paper’s formula t is the table,
          while in the code formula above t is the type.
        </p>
        <MathBlock
          tex={PAPER_FORMULA}
          className="relative overflow-hidden"
          label={<Cite kind="paper" ref="p.5 §3.1" />}
          caption={
            <>
              “<Big tex="\mathbf{W}_d" /> is datatype-specific and <Big tex="\mathbf{W}" /> is shared. For masked cells, the value embedding{' '}
              <Big tex="\mathbf{W}_d\mathbf{r}" /> is replaced with a learned mask vector <Big tex="\mathbf{m}_d" />.”
            </>
          }
        />
        <Callout
          variant="paper-vs-code"
          title="Biases and an RMSNorm after every projection"
          paper={
            <>
              two linear maps, summed, with no bias and no norm written: <TeX tex="\mathbf{W}_d\mathbf{r} + \mathbf{W}\,\mathcal{E}^{\text{schema}}(c,t)" />, with{' '}
              <TeX tex="\mathbf{m}_d" /> replacing <TeX tex="\mathbf{W}_d\mathbf{r}" /> for masked cells <Cite kind="paper" ref="p.5 §3.1" />.
            </>
          }
          code={
            <span className="[&_code]:text-[14.5px]">
              every encoder in <code>enc_dict</code> is an <code>nn.Linear</code> with <code>bias=True</code>{' '}
              <Cite kind="code" ref="rt/model.py:117-121" />, and each projection goes through its own <code>nn.RMSNorm</code>, five of them in{' '}
              <code>norm_dict</code> <Cite kind="code" ref="rt/model.py:132-140" />. The mask vector is added raw{' '}
              <Cite kind="code" ref="rt/model.py:221-224" />.
            </span>
          }
          verdict="Same structure. The code adds a bias and an RMSNorm to each projected term and leaves the mask vector raw. This site computes the code’s version."
        />
        <CodeRef
          title="The encoders, the five input norms and the four mask vectors"
          file="rt/model.py"
          lines="115-146"
          code={MODEL_ENCODERS_115_146}
          highlightLines={[117, 118, 119, 120, 121, 134, 135, 136, 137, 138, 143]}
          note="dec_dict (lines 124-131) belongs to the output side; chapter 12 uses it."
        />
        <CodeRef
          title="The input embedding: column term, then one value term or mask vector per type"
          file="rt/model.py"
          lines="208-224"
          code={MODEL_EMBED_208_224}
          highlightLines={[213, 219, 223]}
          note="Lines 213, 219 and 223 are the gates: boolean masks broadcast over the d dimensions with [..., None]."
        />
        <ScaleNote paper="d = 256" ours={`d = ${d}`}>
          The 384-d MiniLM inputs come from the same frozen encoder in both models; only d changes.
        </ScaleNote>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="step-through" title="Follow one token through" kicker="Step 2">
        <p>
          The figure below computes x₀ for one token in seven stages, ① to ⑦. The left lane is the column term, the right lane the value term or
          the mask vector, and stage ⑦ adds them. Every strip holds real numbers from the forward pass of the checkpoint this page runs; hover a 64-d
          strip to see the arithmetic of a single dimension.
        </p>
        <p>
          Stages ③ and ⑥ are RMSNorms. An RMSNorm divides a vector by its root mean square, plus a small constant ε inside the square root, and
          multiplies it by a learned gain γ <Cite kind="spec" ref="§7.2" />. <JumpLink to="rmsnorm">Step 3</JumpLink> covers it in detail; for now,
          the panel beside each norm lists the numbers it uses.
        </p>
        <p>
          The figure opens on the token you last inspected (the masked target by default) and plays once when it scrolls into view. Replay it, or
          switch between the presets and compare: two cells of the same column get the same column term, and a text value goes through MiniLM on
          its own, without its column name <Cite kind="spec" ref="§3.6" />. For the masked target, stages ⑤ and ⑥ are skipped and its mask vector
          goes straight into the sum.
        </p>
        <TokenEmbedder live={live} />
        <p>
          The code has no if-statement on the type. It runs all four value encoders at every position and multiplies each result by its gate, so
          only the value term of the cell’s own type reaches the sum, and for the masked target none does{' '}
          <Cite kind="code" ref="rt/model.py:216-224" />.
        </p>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 3 */}
      <Section id="rmsnorm" title="RMSNorm, and why its ε matters here" kicker="Step 3">
        <p>
          Each projection in the embedding is followed by an <Term id="rmsnorm">RMSNorm</Term>. It divides a vector by its root mean square and then
          multiplies dimension j by a learned gain <TeX tex="\gamma_j" /> <Cite kind="spec" ref="§7.2" />{' '}
          <Cite kind="url" ref="https://docs.pytorch.org/docs/2.8/generated/torch.nn.RMSNorm.html" title="torch.nn.RMSNorm" />:
        </p>
        <MathBlock tex={RMSNORM_FORMULA} className="relative hidden overflow-hidden text-[18px] sm:block" label={<Cite kind="spec" ref="§7.2" />} />
        <MathBlock tex={RMSNORM_FORMULA_NARROW} className="relative overflow-hidden text-[17.5px] sm:hidden" label={<Cite kind="spec" ref="§7.2" />} />
        <p>
          RMSNorm does not subtract the mean and has no bias. With γ = 1, which is how every gain starts <Cite kind="spec" ref="§7.4" />, the output
          keeps the input’s direction and has a root mean square close to 1, unless the input is tiny.
        </p>
        <p>
          The small constant ε keeps the division safe when the input is close to zero. In RT, ε = 2⁻⁷ ≈ 0.0078 <Cite kind="spec" ref="§7.2" />,
          which is large enough to matter when the input itself is short. Drag x into the shaded ε zone of the figure below and the output falls
          short of the circle.
        </p>
        <RmsNormPlane presetRms={medianColumnRms(live)} />
        <p>
          In d dimensions the circle becomes a sphere. With γ = 1 the output length is{' '}
          <TeX tex="\|y\| = \sqrt{d}\cdot r/\sqrt{r^2 + \varepsilon}" />, where r is the input’s root mean square. When <TeX tex="r^2 \gg \varepsilon" />{' '}
          that is <TeX tex="\sqrt{d}" />, {fmt(Math.sqrt(d), 0)} here and 16 in the paper. When <TeX tex="r^2" /> is comparable to ε, the output
          stays shorter, because RMSNorm can scale a vector up by at most <TeX tex="1/\sqrt{\varepsilon} \approx 11.3" />. And{' '}
          <TeX tex="\mathrm{RMSNorm}(\vec 0) = \vec 0" /> exactly.
        </p>
        <EpsKnee live={live} />
        <EpsFinding live={live} />
        <p>
          The gains γ are the only parameters of an RMSNorm. The embedding has five of them, one per projection, each with {d} entries{' '}
          <Cite kind="code" ref="rt/model.py:132-140" />.
        </p>
        <GainStrips weights={weights} kind={assets.weightsKind} />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="one-encoder-per-type" title="One encoder per type, shared by every column" kicker="Step 4">
        <p>
          A number, a datetime or a boolean enters as one scalar z, the <Term id="z-score">z-score</Term> computed in preprocessing{' '}
          <Cite kind="spec" ref="§3.3" />. Its encoder is <code>nn.Linear(1, d)</code>: one weight column <TeX tex="w_t \in \mathbb{R}^d" /> and a bias{' '}
          <TeX tex="b_t" /> <Cite kind="code" ref="rt/model.py:117" /> <Cite kind="code" ref="rt/model.py:119" />{' '}
          <Cite kind="code" ref="rt/model.py:121" />. So the value term is
        </p>
        <MathBlock className="relative overflow-hidden" tex={String.raw`\mathrm{RMSNorm}_t\!\left(z\,w_t + b_t\right)`} label={<Cite kind="code" ref="rt/model.py:218" />} />
        <p>
          As z changes, <TeX tex="z\,w_t + b_t" /> moves along a straight line in <TeX tex="\mathbb{R}^d" />. RMSNorm pushes every point of that
          line onto the sphere of radius √d (nearly, because of ε), and γ stretches the result. All values of one type therefore lie on a single
          one-dimensional curve <Cite kind="spec" ref="§5.2" />.
        </p>
        <p>
          Encoders belong to types, and every column of a type shares one. {numberColumnsText(live)} all go through{' '}
          <code>enc_dict["number"]</code> <Cite kind="code" ref="rt/model.py:216-220" />. Two cells with the same z get exactly the same value
          term, whichever column they come from. What tells them apart in x₀ is the column term <Cite kind="spec" ref="§5.2" />.
        </p>
        <ValueCurve live={live} />
        <p>
          Drawing the curve in two dimensions loses nothing. <TeX tex="z\,w_t + b_t" /> always lies in the plane spanned by <TeX tex="w_t" /> and{' '}
          <TeX tex="b_t" />; RMSNorm only slides each point along its own ray, and γ is a linear map. So the whole curve lies in one plane, and its
          first two principal directions capture all of it. As z runs from −∞ to +∞, the direction of <TeX tex="z\,w_t + b_t" /> turns from{' '}
          <TeX tex="-w_t" /> to <TeX tex="+w_t" />. Before γ the curve is half a circle of radius about √d, and γ stretches it into half an ellipse.
        </p>
        <p>
          The value term sees only the z-score, and a z-score counts standard deviations of its own column <Cite kind="spec" ref="§3.3" />. So
          z = 0.5 means “half a standard deviation above the mean {sameZExample(live)[0]}” in one column and “half a standard deviation above the
          mean {sameZExample(live)[1]}” in another, and the column term has to carry the difference. For very large |z| the bias becomes negligible
          next to <TeX tex="z\,w_t" />, so the curve flattens out toward the two points marked × in the figure, as its cosine chart shows. Extreme
          values of one type all look alike.
        </p>
        <p>
          Text is different. Its encoder is <code>nn.Linear(384, d)</code> applied to the string’s MiniLM vector, so text values do not share one
          curve; each distinct string is its own point <Cite kind="code" ref="rt/model.py:118" />. Like the column term, it starts from a
          unit-length MiniLM vector <Cite kind="spec" ref="§3.6" />, and the figure in Step 3 shows where this context’s text values land relative
          to √ε.
        </p>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 5 */}
      <Section id="mask-vector" title="The mask vector" kicker="Step 5">
        <p>
          The seed row’s target cell is the only masked token in a context <Cite kind="spec" ref="§4.10" />. Its value term is replaced by{' '}
          <TeX tex="m_t" />, a trainable vector of the target’s type, created as <code>nn.Parameter(torch.randn(d_model))</code>{' '}
          <Cite kind="code" ref="rt/model.py:141-146" />. It skips both the linear layer and the RMSNorm <Cite kind="code" ref="rt/model.py:221-224" />.
        </p>
        <p>
          The masked token still knows a lot. Its column term names the column and the table, its mask vector names the type, and its ids tie it to
          its row and to the entity it is about <Cite kind="spec" ref="§5.3" />. What it lacks is its value, and the attention layers have to infer
          that from the rest of the context.
        </p>
        <MaskVectors live={live} />
        <p>
          The label itself is still in the batch. The sampler writes the true value into the target’s slot like any other cell, and only{' '}
          <code>is_targets</code> and <code>masks</code> mark it <Cite kind="code" ref="rustler/src/fly.rs:482-496" />. The gate <TeX tex="\lnot\mathrm{mask}_i" /> keeps it out of
          x₀; the loss reads it later as the training target <Cite kind="spec" ref="§5.3" />.
        </p>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 6 */}
      <Section id="padding" title="Padding is the zero vector" kicker="Step 6">
        <p>
          When the sampler runs out of rows before it reaches L tokens, the remaining positions are padding. Every field keeps the default it was
          created with, and <code>is_padding</code> stays true <Cite kind="code" ref="rustler/src/fly.rs:65-85" /> <Cite kind="spec" ref="§4.11" />. Every term of
          x₀ is multiplied by <TeX tex="\lnot\mathrm{pad}_i" />, so a padding token’s input is exactly zero{' '}
          <Cite kind="code" ref="rt/model.py:213" /> <Cite kind="code" ref="rt/model.py:219" /> <Cite kind="code" ref="rt/model.py:223" />.
        </p>
        <PaddingCheck live={live} />
        <p>
          Every attention mask of the next chapter also excludes padding, so a padding token never affects a real one <Cite kind="spec" ref="§6.2" />.
        </p>
        <CodeRef
          title="Every batch slot starts as padding"
          file="rustler/src/fly.rs"
          lines="65-85"
          code={FLY_PADDING_65_85}
          highlightLines={[68, 73, 74, 75, 82]}
          note="A cell overwrites these defaults (fly.rs:462-500); a slot no cell reaches keeps them."
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 7 */}
      <Section id="matrix" title="The whole context at once" kicker="Step 7">
        <p>
          Stack the x₀ of all {int(ctx.tokens.length)} positions and you get the matrix that enters the first block. Grouped by column, the column-term
          view shows its structure plainly: every token of a column carries the same column term, so the matrix is made of solid blocks. The value-term
          view is where the tokens of a column differ; there the masked target, marked “?”, is the one token whose value side is a mask vector.
        </p>
        <ScaleNote paper="X₀: 1024 × 256" ours={`X₀: ${int(ctx.tokens.length)} × ${d}`}>
          The paper’s context holds L = 1024 tokens of d = 256 numbers each; the operations that build every column are the same.
        </ScaleNote>
        <ContextMatrix live={live} />
        <p>
          Few weights build this matrix. The input embedding has this many parameters in each model{' '}
          <Cite kind="code" ref="rt/model.py:115-146" />:
        </p>
        <EmbeddingParamStats />
        <p>
          This is all the transformer gets. From here on, information moves between tokens only through attention, along the four attention masks of
          the next chapter.
        </p>
      </Section>

      <div className="page-grid">
        <Quiz
          questions={[
            {
              prompt: 'Which part of x₀ tells the model which column a cell belongs to?',
              choices: [
                {
                  text: 'The value term: each column has its own encoder',
                  why: 'Encoders are per type, not per column. Every number column shares enc_dict["number"].',
                },
                {
                  text: (
                    <>
                      The column term, <TeX tex="\mathrm{RMSNorm}_{\mathrm{col}}(W_{\mathrm{col}}\,c + b_{\mathrm{col}})" /> of the phrase “&lt;column&gt; of
                      &lt;table&gt;”
                    </>
                  ),
                  correct: true,
                  why: 'It is the only part of x₀ that identifies the column. The value term depends only on the type and the stored value (a z-score, or the string’s MiniLM vector for text), and the masked target’s is m_t. The column id also exists, but it only enters the attention masks.',
                },
                {
                  text: 'The token’s position in the sequence',
                  why: 'RT has no positional encodings; position carries no information (chapter 7).',
                },
              ],
              explanation: 'Column identity reaches x₀ only through the MiniLM embedding of the column phrase, projected by the shared W_col.',
            },
            {
              prompt: 'What replaces the value term for the masked target cell?',
              choices: [
                { text: 'A zero vector', why: 'Only padding gets zero. The target gets m_t, which is not zero.' },
                {
                  text: 'Its true label, which the model must copy',
                  why: 'The label is still stored in the batch, but the gate ¬mask multiplies its value term by 0.',
                },
                {
                  text: (
                    <>
                      The mask vector <TeX tex="m_t" /> of its type, added without a norm
                    </>
                  ),
                  correct: true,
                  why: 'model.py:221-224 adds mask_embs[t] gated by (type = t) ∧ mask ∧ ¬padding.',
                },
              ],
              explanation: `There are four mask vectors (${SEM_KEYS.join(', ')}). In practice only number and boolean cells are masked.`,
            },
            {
              prompt: 'A cell of “age of customers” and a cell of “rating of reviews” both have z = 0.5. What is the same in their x₀?',
              choices: [
                { text: 'The column term', why: 'The column phrases differ, so the column terms differ.' },
                { text: 'Nothing: different tables use different weights', why: 'No weight in the embedding depends on the table.' },
                { text: 'The value term', correct: true, why: 'Both go through the same number encoder with the same input z = 0.5.' },
              ],
              explanation: 'x₀ = column term + value term; here the value terms coincide and the column terms tell the two cells apart.',
            },
            {
              prompt: (
                <>
                  A vector with <TeX tex="\operatorname{mean}(x^2) = \varepsilon/3" /> goes through RMSNorm with γ = 1. What is the root mean square of
                  the output?
                </>
              ),
              choices: [
                {
                  text: '1: RMSNorm always outputs a root mean square of 1',
                  why: 'Only when mean(x²) is much larger than ε. Here ε is three quarters of the sum under the square root.',
                },
                {
                  text: '1/2',
                  correct: true,
                  why: 'The output rms is √(ε/3) / √(ε/3 + ε) = √(1/4) = 1/2: ε makes up 75% of the sum under the root.',
                },
                {
                  text: '1/√ε ≈ 11.3',
                  why: 'That is the largest factor RMSNorm can scale a vector up by, not the rms of its output.',
                },
              ],
              explanation: q4Explanation(live),
            },
          ]}
        />
      </div>

      <ChapterNav slug="embedding" />
    </ChapterPage>
  );
}

/** Median rms of W_col·c + b_col over the distinct column phrases of the live context (forward-pass capture). */
function medianColumnRms(live: EmbeddingStable): number | undefined {
  const { fr, weights, ctx } = live;
  if (!fr || !weights) return undefined;
  const d = weights.config.dModel;
  const seen = new Set<number>();
  const r: number[] = [];
  for (const t of ctx.tokens) {
    if (t.isPadding || seen.has(t.colNameIdx)) continue;
    seen.add(t.colNameIdx);
    r.push(Math.sqrt(meanSquare(slice(fr.embedding.colPreNorm, t.pos, d))));
  }
  if (!r.length) return undefined;
  r.sort((a, b) => a - b);
  const m = r.length >> 1;
  return r.length % 2 ? r[m] : (r[m - 1] + r[m]) / 2;
}

/** "Age of customers, price of products, … and rating of reviews": the number columns of the live DB (context tables). */
function numberColumnsText(live: EmbeddingStable): string {
  const names = live.contextColumns.filter((c) => c.sem === SemType.Number).map((c) => `${c.col} of ${c.table}`);
  if (names.length === 0) return 'Every number column';
  const list = names.length === 1 ? names[0] : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
  return list.charAt(0).toUpperCase() + list.slice(1);
}

/** Two number columns of the live context for the "same z" example: "price" / "age" style names (live). */
function sameZExample(live: EmbeddingStable): [string, string] {
  const nums = live.contextColumns.filter((c) => c.sem === SemType.Number);
  if (nums.length >= 2) return [nums[0].col, nums[1].col];
  return ['of the first column', 'of the second column'];
}

/** Quiz 4 explanation, worded from the live checkpoint (ε share for the median column projection). */
function q4Explanation(live: EmbeddingStable): ReactNode {
  const r = medianColumnRms(live);
  const eps = live.weights?.config.eps ?? 2 ** -7;
  const base = 'When mean(x²) is not much larger than ε, RMSNorm cannot bring the term up to full length, so it comes out shorter than √d.';
  if (r === undefined) return base;
  const share = eps / (r * r + eps);
  return (
    <>
      {base} In the checkpoint this page runs, ε is {Math.round(share * 100)}% of the sum for a typical column projection
      {share > 0.5 ? ', so the column terms are in this regime' : share > 0.2 ? ', so the column terms are close to this regime' : ', so the column terms are mostly out of it'}{' '}
      (Step 3). At initialization they are in it in both models, ours and the paper’s.
    </>
  );
}

/** Inline TeX at 17.5px, so its sub- and superscripts stay ≥ 12.5px inside smaller caption text. */
function Big({ tex }: { tex: string }) {
  return (
    <span className="text-[17.5px]">
      <TeX tex={tex} />
    </span>
  );
}
