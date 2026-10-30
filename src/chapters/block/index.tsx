/**
 * Chapter 11 "block": The RT block and the residual stream (BUILD_PLAN §6, docs/CHAPTER_BRIEFS.md §11).
 *
 * Every number comes from the live forward pass of the global context (src/engine/model/forward.ts, a
 * mirror of rt/model.py:50-86 and :226-229) through the shared hooks, so the seed, context and tokens match
 * every other chapter. Spec: research/RT_TECHNICAL_SPEC.md §7 (all), §6.4, §10 #1, §11.4-11.5.
 */
import { useReducedMotion } from 'motion/react';
import { useMemo } from 'react';
import {
  Button,
  Callout,
  ChapterHeader,
  ChapterNav,
  ChapterPage,
  Cite,
  CodeRef,
  Figure,
  LayerLabel,
  MathBlock,
  Quiz,
  ScaleNote,
  Section,
  StatusBadge,
  TeX,
  Term,
} from '@/components';
import { MINI_CONFIG, PAPER_CONFIG, paramCount } from '@/engine/model/config';
import { SemType } from '@/engine/types';
import { BlockDiagram } from './BlockDiagram';
import { AppECards, Table15, Table5 } from './Expressivity';
import { FlowTracer } from './FlowTracer';
import { StreamNormChart } from './PreNorm';
import { ResidualStream } from './ResidualStream';
import { MODEL_BLOCK_62_86, MODEL_FFN_50_59, MODEL_LOOP_226_229, MODEL_STACK_147_150, PRETRAIN_MODEL_28_35 } from './snippets';
import { ParamTable, StackFigure } from './Stack';
import { SwiGLUFigure } from './SwiGLU';
import { useBlockLive } from './useBlockLive';
import { allowedKeys, article, flowOrder, fx, int, rowName, singular, streamStages, SUB_NAME, traceFlow } from './utils';

const RMSNORM_TEX = String.raw`\begin{gathered}\mathrm{RMSNorm}(x)_i \;=\; \frac{x_i}{\sqrt{\tfrac{1}{d}\sum_{j=1}^{d} x_j^2 \;+\; \varepsilon}}\;\gamma_i\\[4pt] \varepsilon = 2^{-7}\end{gathered}`;
const CODE_TEX = String.raw`\textbf{code:}\quad x \leftarrow x + \mathrm{MHA}_\ell\big(\mathrm{RMSNorm}_\ell(x);\,\mathbf{M}_\ell\big)`;
const BLOCK_TEX = String.raw`\begin{aligned}
x &\leftarrow x + \mathrm{MHA}_{\text{col}}\big(\mathrm{RMSNorm}_{\text{col}}(x);\ \mathbf{M}^{\text{column}}\big)\\
x &\leftarrow x + \mathrm{MHA}_{\text{feat}}\big(\mathrm{RMSNorm}_{\text{feat}}(x);\ \mathbf{M}^{\text{feature}}\big)\\
x &\leftarrow x + \mathrm{MHA}_{\text{nbr}}\big(\mathrm{RMSNorm}_{\text{nbr}}(x);\ \mathbf{M}^{\text{neighbor}}\big)\\
x &\leftarrow x + \mathrm{MHA}_{\text{full}}\big(\mathrm{RMSNorm}_{\text{full}}(x);\ \mathbf{M}^{\text{full}}\big)\\
x &\leftarrow x + \mathrm{SwiGLU}\big(\mathrm{RMSNorm}_{\text{ffn}}(x)\big)
\end{aligned}`;
const CODE_TEX_NARROW = String.raw`\begin{aligned}\textbf{code:}\quad x &\leftarrow x\,+\\ &\quad \mathrm{MHA}_\ell\big(\mathrm{RMSNorm}_\ell(x);\,\mathbf{M}_\ell\big)\end{aligned}`;
const BLOCK_TEX_NARROW = String.raw`\begin{aligned}
x &\leftarrow x + \mathrm{MHA}_{\text{col}}\big(\mathrm{RMSNorm}_{\text{col}}(x);\\ &\qquad \mathbf{M}^{\text{column}}\big)\\[4pt]
x &\leftarrow x + \mathrm{MHA}_{\text{feat}}\big(\mathrm{RMSNorm}_{\text{feat}}(x);\\ &\qquad \mathbf{M}^{\text{feature}}\big)\\[4pt]
x &\leftarrow x + \mathrm{MHA}_{\text{nbr}}\big(\mathrm{RMSNorm}_{\text{nbr}}(x);\\ &\qquad \mathbf{M}^{\text{neighbor}}\big)\\[4pt]
x &\leftarrow x + \mathrm{MHA}_{\text{full}}\big(\mathrm{RMSNorm}_{\text{full}}(x);\\ &\qquad \mathbf{M}^{\text{full}}\big)\\[4pt]
x &\leftarrow x + \mathrm{SwiGLU}\big(\mathrm{RMSNorm}_{\text{ffn}}(x)\big)
\end{aligned}`;
const RUNNING_TASK = 'shop/customer-churn';
const SWIGLU_TEX = String.raw`\begin{gathered}\mathrm{FFN}(h) \;=\; W_2\Big(\mathrm{SiLU}(W_1 h)\;\odot\;W_3 h\Big)\\[6pt] \mathrm{SiLU}(z) = z\,\sigma(z) = \frac{z}{1+e^{-z}}\end{gathered}`;

export default function BlockChapter() {
  const live = useBlockLive();
  const { ctx, fr, task, target, nBlocks, d, dFF, H, assets, inCtx, masks, S, follow, setFollow } = live;
  const reduceMotion = useReducedMotion();
  const forecast = task.kind === 'forecast';

  const seed = ctx.tokens[target];
  const entityNode = seed?.f2pNbrIdxs.find((p) => p >= 0);
  const entityTok = entityNode !== undefined ? ctx.tokens.find((t) => !t.isPadding && t.nodeIdx === entityNode) : undefined;
  const entityText = entityNode !== undefined ? ctx.tokens.find((t) => !t.isPadding && t.nodeIdx === entityNode && t.sem === SemType.Text) : undefined;
  // forecast: the task's entity table; autocomplete: the first row the target's row points to
  const entity = task.entityTable ? singular(task.entityTable) : entityTok ? singular(entityTok.tableName) : 'parent row';
  const childTables = useMemo(
    () => live.db.tables.filter((t) => task.entityTable && Object.values(t.schema.fkeys).includes(task.entityTable)).map((t) => t.schema.name),
    [live.db, task],
  );

  // live facts about the target, for the prose
  const tStages = useMemo(() => (fr ? streamStages(fr, target) : null), [fr, target]);
  const facts = useMemo(() => {
    if (!tStages) return null;
    const subs = tStages.filter((s) => s.kind !== 'embed' && s.kind !== 'out');
    const top = subs.reduce((a, b) => (b.updNorm > a.updNorm ? b : a), subs[0]);
    const lastBlock = tStages[tStages.length - 2];
    const firstBlockOut = tStages[5];
    const ran = subs.filter((s) => !s.skipped && Number.isFinite(s.updNorm));
    const nonZero = ran.filter((s) => s.updNorm > 0);
    return {
      x0: tStages[0].xNorm,
      xLast: lastBlock.xNorm,
      xOut: tStages[tStages.length - 1].xNorm,
      top,
      cos1: firstBlockOut.cosLast,
      hMin: Math.min(...subs.filter((s) => Number.isFinite(s.hNorm)).map((s) => s.hNorm)),
      hMax: Math.max(...subs.filter((s) => Number.isFinite(s.hNorm)).map((s) => s.hNorm)),
      dMin: nonZero.length ? Math.min(...nonZero.map((s) => s.updNorm)) : NaN,
      dMax: ran.length ? Math.max(...ran.map((s) => s.updNorm)) : NaN,
      nZero: ran.length - nonZero.length,
    };
  }, [tStages]);
  const targetNbrKeys = allowedKeys(masks, 'nbr', target, S);
  const entityNbrKeys = entityTok ? allowedKeys(masks, 'nbr', entityTok.pos, S) : 0;

  // worked flow examples on the live context, through the relational layers that ran (no full attention)
  const flowFacts = useMemo(() => {
    const order = flowOrder(nBlocks, false, live.disabledLayers);
    const ex = (id: string, pos: number | undefined, what: string) => {
      if (pos === undefined) return null;
      const rel = traceFlow(masks, ctx.tokens, pos, target, order);
      return { id, what, pos, path: rel.path };
    };
    const child = live.presets.find((p) => p.id === 'child');
    const far = live.presets.find((p) => p.id === 'far');
    return [
      ex('self', inCtx.self[0]?.pos, `a past label of the same ${entity}`),
      child ? ex('child', child.pos, child.label) : null,
      far ? ex('far', far.pos, `a cell of ${far.label}`) : null,
    ].filter((x): x is NonNullable<typeof x> => x !== null);
  }, [masks, ctx, target, nBlocks, inCtx, live.presets, live.disabledLayers, entity]);
  const farRoute = flowFacts.find((f) => f.id === 'far')?.path;
  const nRoutesWord = ['One', 'Two', 'Three', 'Four'][flowFacts.length - 1] ?? String(flowFacts.length);

  const followEntity = (pos: number) => {
    setFollow(pos);
    window.setTimeout(
      () => document.getElementById('block-diagram')?.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'nearest' }),
      30,
    );
  };

  const miniParams = paramCount(MINI_CONFIG);
  const paperParams = paramCount(PAPER_CONFIG);
  const perBlockMini = 4 * 4 * d * d + 3 * d * dFF + 5 * d;

  return (
    <ChapterPage slug="block">
      <ChapterHeader
        slug="block"
        lede={
          <>
            Attention is one sublayer. A block chains five sublayers: four attention layers (three relational, one full) and a gated feed-forward
            network. Each adds its result to a running vector per token. This chapter builds the block, stacks it, and follows the masked target
            through the whole stack with the live numbers of the trained mini model.
          </>
        }
        learn={[
          'The five sublayers of an RT block, in the order the code runs them',
          'Pre-norm residuals: what each sublayer reads and what it adds to the stream',
          'SwiGLU, the gated feed-forward network, on the hidden units of a live token',
          `The stack: ${nBlocks} blocks here, 12 in the paper, then a final RMSNorm, and where the parameters live`,
          'How information from any cell can reach the masked target, block by block',
        ]}
      >
        <p className="mt-6 text-[17px] leading-relaxed text-ink-2">
          <span className="eyebrow mr-2">Live context</span>
          seed <span className="text-ink">{seed ? rowName(seed) : '–'}</span>
          {entityTok && (
            <>
              {' '}
              → <span className="text-ink">{rowName(entityTok)}</span>
              {entityText ? ` (${String(entityText.raw)})` : ''}
            </>
          )}
          {' · '}
          {int(ctx.numReal)} cell tokens · checkpoint <span className="font-mono text-ink">{assets.ckpt}</span>
          {assets.weightsKind === 'init' && (
            <>
              {' '}
              <StatusBadge status="warning">untrained weights</StatusBadge>
            </>
          )}{' '}
          <ScaleNote inline />
        </p>
      </ChapterHeader>

      {(task.id !== RUNNING_TASK || live.disabledLayers.length > 0) && (
        <div className="page-grid -mt-6 mb-8">
          <div className="grid gap-4">
            {task.id !== RUNNING_TASK && (
              <Callout variant="note" title="You are viewing another task">
                <p>
                  The figures follow your simulation settings: {live.db.name} ·{' '}
                  {task.id.split('/')[1]} ({task.kind}). The prose is written for the running example, ShopDB’s{' '}
                  customer-churn, so a few sentences may not apply.
                </p>
                <Button size="sm" variant="secondary" onClick={() => live.setTask(RUNNING_TASK)}>
                  Switch the course to the running example
                </Button>
              </Callout>
            )}
            {live.disabledLayers.length > 0 && (
              <Callout variant="warning" title="An inference-time ablation is active">
                <p>
                  The playground switched off {live.disabledLayers.map((l) => SUB_NAME[l]).join(', ')} attention in every block, so the numbers below
                  come from that modified forward pass, and the route tracer of step 6 leaves those layers out.{' '}
                  <Button size="sm" variant="secondary" onClick={() => live.setDisabledLayers([])}>
                    Restore the full model
                  </Button>
                </p>
              </Callout>
            )}
          </div>
        </div>
      )}

      {/* ───────────────────────────────────────────────────────────── 1 */}
      <Section id="block" title="One block, five sublayers" kicker="Step 1">
        <p>
          After the embedding of chapter 8, every token of the context is a vector of d = {d} numbers (256 in the paper). Inside the blocks the
          model only ever adds to that vector. Each <strong>sublayer</strong> reads the token’s current vector, computes an update, and adds it
          back: <TeX tex="x \leftarrow x + f(x)" /> <Cite kind="code" ref="rt/model.py:82-86" />. The running vector is called the{' '}
          <strong>residual stream</strong>, and the identity path that carries it past every sublayer is the residual connection.
        </p>
        <p>
          An RT block has five sublayers. The first four are attention layers, always in the order column, feature, neighbor, full, and the fifth
          is a feed-forward network <Cite kind="code" ref="rt/model.py:83-85" /> <Cite kind="paper" ref="p.26 App. J" />. Each has its own{' '}
          <Term id="rmsnorm">RMSNorm</Term> and its own weights, so a block holds five RMSNorms, four separate attention modules and one
          feed-forward module <Cite kind="code" ref="rt/model.py:71-80" />.
        </p>
        <div className="sm:hidden">
          <MathBlock tex={BLOCK_TEX_NARROW} className="text-[18px]" label={<Cite kind="code" ref="rt/model.py:82-86" />} caption="One RT block, as the code runs it. Each line reads the x left by the line above." />
        </div>
        <div className="hidden sm:block">
          <MathBlock tex={BLOCK_TEX} className="text-[18px]" label={<Cite kind="code" ref="rt/model.py:82-86" />} caption="One RT block, as the code runs it. Each line reads the x left by the line above." />
        </div>
        <p>
          Walk through a block below. The pulse follows one token, the masked target by default, and each box shows ‖Δ‖, the length of the
          update that sublayer adds in the live forward pass. Click a box to inspect it.
        </p>
        <Figure wide id="block-diagram" title="An RT block, with the live numbers of one token">
          <BlockDiagram live={live} follow={follow} setFollow={setFollow} />
        </Figure>
        {targetNbrKeys === 0 && (
          <p>
            For the target, the neighbor box stays at zero in every block. The neighbor mask lets a row read only the rows that point to it,
            and{' '}
            {forecast ? (
              <>
                no row has a foreign key to a task row: task tables have no primary key, so their rows are never parents{' '}
                <Cite kind="spec" ref="§1.3" />
              </>
            ) : (
              <>no row in the context points to the target’s row</>
            )}
            . The target has no key to read <Cite kind="spec" ref="§6.2" />, so its attention output is 0, and since <TeX tex="W_o" /> has no
            bias, the update is the zero vector <Cite kind="spec" ref="§6.1" />.
            {entityTok && entityNbrKeys > 0 && (
              <>
                {' '}
                <button type="button" className="font-medium text-accent-text underline decoration-dotted underline-offset-4" onClick={() => followEntity(entityTok.pos)}>
                  Follow the {entity}’s {entityTok.colName}
                </button>{' '}
                instead: its neighbor row has {int(entityNbrKeys)} keys, so its neighbor update is not zero.
              </>
            )}
          </p>
        )}
        <CodeRef
          title="RelationalBlock: five norms, four attentions, one FFN, and the pre-norm forward"
          file="rt/model.py"
          lines="62-86"
          code={MODEL_BLOCK_62_86}
          highlightLines={[83, 84, 85]}
          note="Line 83 sets the order the layers run in. The order in which __init__ lists them does not matter. Line 84 is the whole attention sublayer: normalize a copy, attend under this layer's mask, add the result. Line 85 does the same with the FFN."
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="pre-norm" title="Pre-norm: where the norm sits" kicker="Step 2">
        <p>
          Every sublayer starts with an RMSNorm. It divides a vector by its root mean square and multiplies each dimension by a learned weight{' '}
          <TeX tex="\gamma_i" /> <Cite kind="spec" ref="§7.2" />. Unlike LayerNorm, it subtracts no mean and adds no bias. The ε under the root is
          a small constant that keeps the denominator above zero.
        </p>
        <MathBlock tex={RMSNORM_TEX} className="text-[18px] sm:text-[19px]" label={<Cite kind="spec" ref="§7.2" />} />
        <p>
          The norm sits on the sublayer’s <em>input</em>. The sublayer reads <TeX tex="\mathrm{RMSNorm}_\ell(x)" />, and its raw output is added
          to the stream <Cite kind="code" ref="rt/model.py:84-85" />. This placement is called <Term id="pre-norm">pre-norm</Term>. The paper says
          RT follows LLaMA for “normalization, activations, etc.” <Cite kind="paper" ref="p.5 §3.2" />, and LLaMA normalizes the input of each
          sublayer for training stability{' '}
          <Cite kind="url" ref="https://arxiv.org/abs/2302.13971" title="Touvron et al. 2023, LLaMA, §2.2 Pre-normalization" />.
        </p>
        <div className="sm:hidden">
          <MathBlock tex={CODE_TEX_NARROW} className="text-[18px]" label={<Cite kind="code" ref="rt/model.py:84" />} />
        </div>
        <div className="hidden sm:block">
          <MathBlock tex={CODE_TEX} className="text-[18px]" label={<Cite kind="code" ref="rt/model.py:84" />} />
        </div>
        <p>
          The paper’s Algorithm 2 puts the norm on the sublayer’s output instead,{' '}
          <TeX tex="\mathbf{X} \leftarrow \mathbf{X} + \mathrm{Norm}(\mathrm{MHA}(\mathbf{X}; \mathbf{M}))" />{' '}
          <Cite kind="paper" ref="p.26 Alg. 2" />. The code is pre-norm, and this site follows the code <Cite kind="spec" ref="§7.1" />.
        </p>
        <p>
          With the norm on the input, a sublayer always reads a vector of controlled length. However long x grows, the length of{' '}
          <TeX tex="\mathrm{RMSNorm}_\ell(x)" /> depends only on the direction of x and the weights γ. The output is not normalized. It goes
          straight back into the stream, so an update can be as long as the sublayer makes it. Inside a block the stream itself is never rescaled,
          only added to.
        </p>
        <p>
          The chart shows this for the target in the trained mini model.
          {facts && (
            <>
              {' '}
              Its stream {facts.xLast > facts.x0 ? 'grows' : 'goes'} from ‖x‖ = {fx(facts.x0)} at the embedding to {fx(facts.xLast)} after block{' '}
              {nBlocks} (blue), while every sublayer reads a copy whose length stays between {fx(facts.hMin)} and {fx(facts.hMax)} (orange). The
              updates have lengths from {fx(facts.dMin)} to {fx(facts.dMax)}
              {facts.nZero > 0 ? ', and exactly 0 where a sublayer has no key to read' : ''}. The final RMSNorm, <code>norm_out</code>, is the only
              norm applied to the stream itself. It brings the length to {fx(facts.xOut)} before the decoders read it{' '}
              <Cite kind="code" ref="rt/model.py:229" />.
            </>
          )}
        </p>
        {fr ? (
          <Figure wide>
            <StreamNormChart fr={fr} pos={target} tokenName="the masked target" />
          </Figure>
        ) : (
          <p className="text-muted">Loading the trained weights…</p>
        )}
      </Section>

      {/* ───────────────────────────────────────────────────────────── 3 */}
      <Section id="swiglu" title="SwiGLU: the feed-forward sublayer" kicker="Step 3">
        <p>
          The fifth sublayer is a gated MLP. The paper describes “gated MLPs with SiLU activation, as found in the LLaMA architecture”, with a
          hidden dimension of 1024 <Cite kind="paper" ref="p.6 §4.1" />. In the code it is three linear maps without bias{' '}
          <Cite kind="code" ref="rt/model.py:50-59" />:
        </p>
        <MathBlock tex={SWIGLU_TEX} className="text-[18px] sm:text-[19px]" label={<Cite kind="code" ref="rt/model.py:59" />} />
        <p>
          <TeX tex="W_1" /> and <TeX tex="W_3" /> map the normalized vector h from d = {d} to d<sub>ff</sub> = {dFF} hidden units, and{' '}
          <TeX tex="W_2" /> maps back to {d}. The paper model uses 256 → 1024 → 256. In both, the hidden layer is four times as wide as the
          stream, d<sub>ff</sub> = 4d <Cite kind="spec" ref="§7.1" />. The name <Term id="swiglu">SwiGLU</Term> joins Swish, another name for SiLU, with GLU, a gated linear
          unit <Cite kind="url" ref="https://arxiv.org/abs/2002.05202" title="Shazeer 2020, GLU Variants Improve Transformer" />. SiLU(W₁h) is the
          activation, and W₃h is a learned linear gate that multiplies it unit by unit.
        </p>
        <p>
          The FFN works on each token separately. It mixes the {d} dimensions of one token’s vector and never looks at another token, so inside a
          block, information moves between tokens only in the four attention sublayers.
        </p>
        <Figure wide title="SiLU, the gate, and the hidden units of a live token">
          <SwiGLUFigure live={live} follow={follow} setFollow={setFollow} />
        </Figure>
        <CodeRef title="FFN: SwiGLU with three bias-free maps" file="rt/model.py" lines="50-59" code={MODEL_FFN_50_59} highlightLines={[59]} />
        <ScaleNote
          paper={`${PAPER_CONFIG.dModel} → ${PAPER_CONFIG.dFF} → ${PAPER_CONFIG.dModel} (${int(3 * PAPER_CONFIG.dModel * PAPER_CONFIG.dFF)} weights per FFN)`}
          ours={`${d} → ${dFF} → ${d} (${int(3 * d * dFF)} weights per FFN)`}
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="stack" title={`The stack: ${nBlocks} blocks here, 12 in the paper`} kicker="Step 4">
        <p>
          The model is a stack of blocks with the same shape and separate weights, followed by one more RMSNorm, <code>norm_out</code>{' '}
          <Cite kind="code" ref="rt/model.py:147-150" />. The forward pass runs the blocks in sequence, hands each one the same four masks (built
          once from the token ids), and then applies <code>norm_out</code> <Cite kind="code" ref="rt/model.py:226-229" />. The paper model has 12
          blocks of width 256 with 8 heads <Cite kind="paper" ref="p.6 §4.1" /> <Cite kind="code" ref="scripts/example_pretrain.py:32" />, and the
          mini model has {nBlocks} blocks of width {d} with {H} heads.
        </p>
        <Figure wide title="The forward pass, end to end">
          <StackFigure live={live} follow={follow} />
        </Figure>
        <p>
          Counting every weight in <code>model.py</code> at the paper’s sizes gives {int(paperParams)} parameters, the “about 22M” of the paper{' '}
          <Cite kind="paper" ref="p.6 §4.1" />. The frozen MiniLM text encoder is not included, because its 384-d embeddings are computed once,
          before training <Cite kind="spec" ref="§7.3" />. The mini model has {int(miniParams)}, of which {int(nBlocks * perBlockMini)} sit in the
          blocks. In both models the blocks hold most of the weights, and the attention matrices are the largest single group.
        </p>
        <ParamTable weights={assets.weights} />
        <ScaleNote paper={`12 blocks, d = 256, d_ff = 1024: ${int(paperParams)} parameters`} ours={`${nBlocks} blocks, d = ${d}, d_ff = ${dFF}: ${int(miniParams)} parameters`} />
        <CodeRef title="The block stack and the final norm" file="rt/model.py" lines="147-150" code={MODEL_STACK_147_150} />
        <CodeRef
          title="Same masks for every block, then norm_out"
          file="rt/model.py"
          lines="226-229"
          code={MODEL_LOOP_226_229}
          note="block_masks was built once, before the loop (model.py:204-206)."
        />
        <CodeRef title="The paper model’s sizes in the pretraining script" file="scripts/example_pretrain.py" lines="28-35" code={PRETRAIN_MODEL_28_35} highlightLines={[32, 33, 34, 35]} />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 5 */}
      <Section id="residual-stream" title="The residual stream of the masked target" kicker="Step 5">
        <p>
          In the mini model, {nBlocks * 5} sublayers ({nBlocks} blocks × 5) sit between the embedding and the decoders, and each one adds its
          update to the target’s stream. The grid gives the length of every update. The matrix shows the vector itself after each stage, one row
          per stage, on a shared color scale. Pick another token to compare.
        </p>
        <Figure wide title="One token through the whole stack">
          <ResidualStream live={live} follow={follow} setFollow={setFollow} />
        </Figure>
        {facts && (
          <p>
            For the target, the largest single update comes from block {facts.top.block + 1}’s {SUB_NAME[facts.top.kind as keyof typeof SUB_NAME].toLowerCase()}{' '}
            {facts.top.kind === 'ffn' ? 'sublayer' : 'attention'} (‖Δ‖ = {fx(facts.top.updNorm)}).
            {facts.top.kind === 'col' && (
              <> That is the layer in which the masked cell reads the visible {forecast ? 'labels' : 'values'} of its column directly (chapter 9).</>
            )}{' '}
            After block 1, the target’s vector has cosine {fx(facts.cos1)} with the output of block {nBlocks}
            {facts.cos1 > 0.8 ? ', so the later blocks mostly refine its direction.' : ', so the later blocks still turn it a good deal.'} These
            numbers describe the mini model on this one context.
          </p>
        )}
        <Callout variant="insight" title="Why a residual stream?">
          <p>
            Every sublayer adds to the vector and none replaces it, so a token’s vector after any sublayer is exactly its embedding x⁽⁰⁾ (the
            column-name and value terms of chapter 8) plus the sum of all updates so far. A later sublayer never overwrites what an earlier one
            wrote, and a sublayer with nothing to read leaves the vector as it was: the empty neighbor rows add exactly zero. RT keeps this
            structure from “the standard transformer design” and adds the relational attention layers to each block{' '}
            <Cite kind="paper" ref="p.5 §3.2" />.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 6 */}
      <Section id="information-flow" title="How information can reach the target" kicker="Step 6">
        <p>
          In an attention sublayer, query q can receive from key k exactly when the mask allows the pair, and the feed-forward sublayer moves
          nothing between tokens. The masks are therefore a map of where information <em>can</em> flow <Cite kind="spec" ref="§6.4" />. Whether
          the trained model uses a route is a separate question, which the attention weights shown on each hop of the tracer below only partly
          answer. In block order, the four attention layers move information like this:
        </p>
        {forecast ? (
          <ol>
            <li>
              <LayerLabel layer="col" />: the target reads every visible label of its column directly, its own {entity}’s past labels and other{' '}
              {entity}s’ alike. It cannot yet tell which are which.
            </li>
            <li>
              <LayerLabel layer="feat" />: the target and every past task row read their parent {entity}’s cells, so each of them now carries
              information about which {entity} it concerns.
            </li>
            <li>
              <LayerLabel layer="nbr" />: {entity} cells read their child rows, the labeled task rows
              {childTables.length > 0 ? ` and the ${childTables.join(' and ')} rows` : ''} alike. This is where per-{entity} aggregates can form,
              “analogously to message-passing in GNNs” <Cite kind="paper" ref="p.5 §3.2" />.
            </li>
            <li>
              <LayerLabel layer="full" />: unrestricted mixing among all non-padding tokens.
            </li>
          </ol>
        ) : (
          <ol>
            <li>
              <LayerLabel layer="col" />: the target reads every visible cell of its column directly, that is, the same column of the other{' '}
              {seed?.tableName} rows in the context.
            </li>
            <li>
              <LayerLabel layer="feat" />: the target reads the other cells of its own row and the cells of the rows its row points to.
            </li>
            <li>
              <LayerLabel layer="nbr" />: the cells of every row read the rows that point to it. This is where per-row aggregates can form,
              “analogously to message-passing in GNNs” <Cite kind="paper" ref="p.5 §3.2" />.
            </li>
            <li>
              <LayerLabel layer="full" />: unrestricted mixing among all non-padding tokens.
            </li>
          </ol>
        )}
        <p>
          Stacking blocks composes these hops: what {article(entity)} {entity} gathered in block 1’s neighbor layer reaches the target in block 2’s
          feature layer. Pick a source token below and trace it.
        </p>
        <Figure wide title="Route tracer: from any token to the masked target">
          <FlowTracer live={live} />
        </Figure>
        {flowFacts.length > 0 && (
          <>
            <p>
              {nRoutesWord} route{flowFacts.length > 1 ? 's' : ''} on the live context, through the relational layers only (full attention switched
              off):
            </p>
            <ul>
              {flowFacts.map((f) => (
                <li key={f.id}>
                  {f.what}:{' '}
                  {f.path ? (
                    <span className="num text-[16px]">
                      #{f.pos} {live.labels[f.pos]}
                      {f.path.map((h, i) => (
                        <span key={i}>
                          {' '}
                          <span className="font-sans text-ink-2">
                            → B{h.step.block + 1} {SUB_NAME[h.step.layer].toLowerCase()} →
                          </span>{' '}
                          {h.to === target ? 'target' : `#${h.to} ${live.labels[h.to]}`}
                        </span>
                      ))}
                    </span>
                  ) : (
                    <>no route within {nBlocks} blocks</>
                  )}
                </li>
              ))}
            </ul>
            <p>
              {farRoute && farRoute.length > 1 && farRoute[0].step.layer === 'col' && (
                <>
                  The last route starts in a column layer: the {entity}’s own cells read the same column of every other {entity} in the context,
                  and the target then reads its {entity} in a feature layer. Column attention links rows that no foreign key connects.{' '}
                </>
              )}
              With full attention every token reaches the target in block 1, since the full mask connects every pair of non-padding tokens. In
              the paper’s words, full attention “enables long-range interactions in a fixed number of layers, independent of database or graph
              diameter” <Cite kind="paper" ref="p.20 App. E" />.
            </p>
          </>
        )}
        <ScaleNote paper="12 blocks: up to 36 relational hops" ours={`${nBlocks} blocks: up to ${nBlocks * 3} relational hops`}>
          Each block offers one column, one feature and one neighbor hop, so the deeper paper model allows longer relational routes than the mini
          model.
        </ScaleNote>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 7 */}
      <Section id="expressivity" title="Expressivity notes" kicker="Step 7">
        <p>
          If full attention can reach everything in one layer, why keep the three relational layers? Appendix E of the paper discusses what each
          layer adds to the expressive power of RT <Cite kind="paper" ref="p.19-20 App. E" />:
        </p>
        <AppECards />
        <p>
          The paper leaves a full theory of RT’s expressive power to future work <Cite kind="paper" ref="p.20 App. E" />. For evidence it turns to
          the layer ablation of Table 5, which retrains the model with one layer type removed.
        </p>
        <Table5 />
        <p>
          The paper also tried other orders of the three relational layers and found that “the specific order of relational attention layers is
          not critical” <Cite kind="paper" ref="p.23 App. H.1" />. Across orders, mean zero-shot AUROC stays between 69.3 and 70.6, and mean R²
          between 22.1 and 23.6 <Cite kind="paper" ref="p.24 Table 15" />.
        </p>
        <Table15 />
        <p>
          The playground (chapter 15) lets you switch layers off at inference time on the live context. That is a different experiment, because
          the paper’s ablations retrain the model without the layer <Cite kind="spec" ref="§11.4" />.
        </p>
      </Section>

      <div className="page-grid">
        <Quiz
          title="Check your understanding"
          questions={[
            {
              prompt: 'In what order does an RT block run its four attention layers?',
              choices: [
                { text: 'column → feature → neighbor → full', correct: true, why: 'The loop on model.py:83 runs ["col", "feat", "nbr", "full"], the same order as Algorithm 2 and Figure 1 of the paper.' },
                {
                  text: 'All four at once, with their outputs summed',
                  why: 'They run one after another, and each reads the stream as the previous one left it. Summing the three relational layers in parallel is one of the variants the paper tested in Table 15.',
                },
                { text: 'full → column → feature → neighbor', why: 'Full attention comes last among the four, just before the SwiGLU.' },
              ],
              explanation: 'Each sublayer reads what the previous ones added, so the order decides which hops can chain inside one block. The paper reports that other orders of the three relational layers score about the same (Table 15).',
            },
            {
              prompt: 'Is RMSNorm applied before or after each sublayer?',
              choices: [
                {
                  text: 'Before: x + MHA(RMSNorm(x)), so the stream itself is never normalized inside a block.',
                  correct: true,
                  why: 'model.py:84-85 normalize a copy for the sublayer and add its raw output. This is LLaMA-style pre-norm.',
                },
                {
                  text: 'After: x + Norm(MHA(x)), as Algorithm 2 writes it.',
                  why: 'Algorithm 2 (p.26) prints it that way, but the code normalizes the sublayer’s input, and this site follows the code.',
                },
                { text: 'Both before and after.', why: 'Each sublayer has exactly one RMSNorm, on its input. The other RMSNorms are outside the blocks: one per input projection in the embedding, and norm_out after the last block.' },
              ],
              explanation: (
                <>
                  Pre-norm: <TeX tex="x \leftarrow x + f(\mathrm{RMSNorm}(x))" />. You saw the consequence in step 2: ‖x‖ grows through the stack until{' '}
                  <code>norm_out</code>.
                </>
              ),
            },
            {
              prompt: 'The masked target is a cell of a task row. What does the neighbor sublayer add to its residual stream?',
              choices: [
                {
                  text: 'Exactly zero, in every block.',
                  correct: true,
                  why: 'No row has a foreign key to a task row, so the target’s neighbor mask row is empty; the attention output is 0 and W_o has no bias.',
                },
                { text: 'The average of its child rows.', why: 'A task row has no children in the context: nothing points to it.' },
                { text: 'The same as the full layer.', why: 'The two masks differ: full allows every non-padding key, neighbor only rows that point to the query’s row.' },
              ],
              explanation: 'Information still reaches the target from the entity’s children. Full attention lets the target read them directly in every block, and along the relational path the entity reads them in the neighbor layer and the target reads the entity in the next feature layer.',
            },
          ]}
        />
      </div>

      <ChapterNav slug="block" />
    </ChapterPage>
  );
}
