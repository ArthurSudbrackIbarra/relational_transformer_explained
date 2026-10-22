/**
 * Chapter 11 "block": The RT block and the residual stream (BUILD_PLAN §6, docs/CHAPTER_BRIEFS.md §11).
 *
 * Every number comes from the live forward pass of the global context (src/engine/model/forward.ts, a
 * mirror of rt/model.py:50-86 and :226-229) through the shared hooks, so the seed, context and tokens match
 * every other chapter. Spec: research/RT_TECHNICAL_SPEC.md §7 (all), §6.4, §10 #1, #23, §11.4-11.5, §13 #1.
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
import { StreamNormChart, WiringCompare } from './PreNorm';
import { ResidualStream } from './ResidualStream';
import { MODEL_BLOCK_62_86, MODEL_FFN_50_59, MODEL_LOOP_226_229, MODEL_STACK_147_150, PRETRAIN_MODEL_28_35 } from './snippets';
import { ParamTable, StackFigure } from './Stack';
import { SwiGLUFigure } from './SwiGLU';
import { useBlockLive } from './useBlockLive';
import { allowedKeys, article, flowOrder, fx, int, rowName, singular, streamStages, SUB_NAME, traceFlow } from './utils';

const RMSNORM_TEX = String.raw`\begin{gathered}\mathrm{RMSNorm}(x)_i \;=\; \frac{x_i}{\sqrt{\tfrac{1}{d}\sum_{j=1}^{d} x_j^2 \;+\; \varepsilon}}\;\gamma_i\\[4pt] \varepsilon = 2^{-7}\end{gathered}`;
const ALG2_TEX = String.raw`\textbf{Alg. 2:}\quad \mathbf{X} \leftarrow \mathbf{X} + \mathrm{Norm}\big(\mathrm{MHA}(\mathbf{X};\,\mathbf{M})\big)`;
const CODE_TEX = String.raw`\textbf{code:}\quad x \leftarrow x + \mathrm{MHA}_\ell\big(\mathrm{RMSNorm}_\ell(x);\,\mathbf{M}_\ell\big)`;
const BLOCK_TEX = String.raw`\begin{aligned}
x &\leftarrow x + \mathrm{MHA}_{\text{col}}\big(\mathrm{RMSNorm}_{\text{col}}(x);\ \mathbf{M}^{\text{column}}\big)\\
x &\leftarrow x + \mathrm{MHA}_{\text{feat}}\big(\mathrm{RMSNorm}_{\text{feat}}(x);\ \mathbf{M}^{\text{feature}}\big)\\
x &\leftarrow x + \mathrm{MHA}_{\text{nbr}}\big(\mathrm{RMSNorm}_{\text{nbr}}(x);\ \mathbf{M}^{\text{neighbor}}\big)\\
x &\leftarrow x + \mathrm{MHA}_{\text{full}}\big(\mathrm{RMSNorm}_{\text{full}}(x);\ \mathbf{M}^{\text{full}}\big)\\
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
            Attention is one sublayer. A block chains five of them: four relational attentions and a gated feed-forward network, each adding its
            result to a running vector per token. This chapter assembles the block, stacks it, and follows the masked target through the whole stack
            with the live numbers of the trained mini model.
          </>
        }
        learn={[
          'The five sublayers of an RT block, in the order the code runs them',
          'Pre-norm residuals: how the code differs from the paper’s Algorithm 2, and what each sublayer reads and adds as a result',
          'SwiGLU, the gated feed-forward network, with the hidden units of a live token',
          `The stack: ${nBlocks} blocks here, 12 in the paper, then a final RMSNorm, and where the parameters live`,
          'How information from any cell can reach the masked target, block by block (an interpretation of the masks)',
        ]}
      >
        <p className="mt-6 text-[17px] leading-relaxed text-ink-2">
          <span className="eyebrow mr-2">Live context</span>
          seed <span className="font-mono text-ink">{seed ? rowName(seed) : '–'}</span>
          {entityTok && (
            <>
              {' '}
              → <span className="font-mono text-ink">{rowName(entityTok)}</span>
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
                  The figures follow your simulation settings: <span className="font-mono">{live.db.name}</span> ·{' '}
                  <span className="font-mono">{task.id.split('/')[1]}</span> ({task.kind}). The prose is written for the running example, ShopDB’s{' '}
                  <span className="font-mono">customer-churn</span>, so a few sentences may not apply.
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
          model never replaces that vector; it only adds to it. Each <strong>sublayer</strong> reads the token’s current vector, computes an update, and
          adds the update back: <TeX tex="x \leftarrow x + f(x)" /> <Cite kind="code" ref="rt/model.py:82-86" />. The running vector is called the{' '}
          <strong>residual stream</strong>, and the identity path that carries it past every sublayer is the residual connection.
        </p>
        <p>
          An RT block has five sublayers. Four are relational attentions, always in the order column, feature, neighbor, full; the fifth is a
          feed-forward network <Cite kind="code" ref="rt/model.py:83-85" /> <Cite kind="paper" ref="p.26 App. J" />. Each has its own{' '}
          <Term id="rmsnorm">RMSNorm</Term> and its own weights: a block holds five RMSNorms, four separate attention modules and one feed-forward
          module <Cite kind="code" ref="rt/model.py:71-80" />.
        </p>
        <MathBlock tex={BLOCK_TEX} className="text-[18px]" label={<Cite kind="code" ref="rt/model.py:82-86" />} caption="One RT block, as the code runs it. Each line reads the x left by the line above." />
        <p>
          Walk through a block below. The pulse travels with the followed token, the masked target by default; each box shows the length of the
          update it adds, ‖Δ‖, from the live forward pass. Click a box to inspect that sublayer.
        </p>
        <Figure wide id="block-diagram" title="An RT block, with the live numbers of one token">
          <BlockDiagram live={live} follow={follow} setFollow={setFollow} />
        </Figure>
        {targetNbrKeys === 0 && (
          <p>
            One box stays at zero for the target in every block: neighbor. The neighbor mask lets a row read only the rows that point to it, and{' '}
            {forecast ? (
              <>
                no row has a foreign key to a task row: task tables have no primary key, so their rows are never parents{' '}
                <Cite kind="spec" ref="§1.3" />
              </>
            ) : (
              <>no row in the context points to the target’s row</>
            )}
            , so the target has no key to read <Cite kind="spec" ref="§6.2" />. A fully masked query returns exactly 0, and <TeX tex="W_o" /> has
            no bias, so the update is the zero vector <Cite kind="spec" ref="§6.1" />.
            {entityTok && entityNbrKeys > 0 && (
              <>
                {' '}
                <button type="button" className="font-medium text-accent-text underline decoration-dotted underline-offset-4" onClick={() => followEntity(entityTok.pos)}>
                  Follow the {entity}’s {entityTok.colName}
                </button>{' '}
                instead: its neighbor row has {int(entityNbrKeys)} keys, and its neighbor update is not zero.
              </>
            )}
          </p>
        )}
        <Callout variant="note" title="Two orders in the same file">
          <p>
            <code>__init__</code> registers the modules as feat, nbr, col, full, but <code>forward</code> runs them as col, feat, nbr, full{' '}
            <Cite kind="code" ref="rt/model.py:71-84" />. Only the execution order matters. It matches Algorithm 2 and Figure 1 of the paper, and has
            been the code’s order since commit 7f891d1; the ICLR submission ran feat, nbr, col, full <Cite kind="spec" ref="§6.6" />{' '}
            <Cite kind="spec" ref="§10 #23" />. The paper later tested other orders and found that “the specific order of relational attention layers
            is not critical” <Cite kind="paper" ref="p.23 App. H.1" />; the numbers are in step 7.
          </p>
        </Callout>
        <CodeRef
          title="RelationalBlock: five norms, four attentions, one FFN, and the pre-norm forward"
          file="rt/model.py"
          lines="62-86"
          code={MODEL_BLOCK_62_86}
          highlightLines={[83, 84, 85]}
          note="Line 84 is the whole attention sublayer: normalise a copy, attend under this layer's mask, add the result. Line 85 does the same with the FFN."
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="pre-norm" title="Pre-norm: where the norm sits" kicker="Step 2">
        <p>
          Every sublayer starts with an RMSNorm. It divides a vector by its root mean square and multiplies each dimension by a learned weight{' '}
          <TeX tex="\gamma_i" /> <Cite kind="spec" ref="§7.2" />. Unlike LayerNorm it subtracts no mean and adds no bias.
        </p>
        <MathBlock tex={RMSNORM_TEX} className="text-[18px] sm:text-[19px]" label={<Cite kind="spec" ref="§7.2" />} />
        <p>
          The small constant ε = 2⁻⁷ is not a typo. The paper trained in bfloat16 <Cite kind="paper" ref="p.6 §4.1" />, and the official code pins
          PyTorch 2.6.0, where <code>nn.RMSNorm</code> without an explicit ε uses the machine epsilon of the input type: 2⁻⁷ for bfloat16{' '}
          <Cite kind="spec" ref="§7.2" />. This site runs in float32 and keeps ε = 2⁻⁷ to match.
        </p>
        <h3>Algorithm 2 and the code disagree on placement</h3>
        <p>
          The paper’s Algorithm 2 normalises the <em>output</em> of each sublayer before adding it. The code normalises the <em>input</em>: the
          sublayer reads <TeX tex="\mathrm{RMSNorm}_\ell(x)" />, and its raw output is added to the stream <Cite kind="paper" ref="p.26 Alg. 2" />{' '}
          <Cite kind="code" ref="rt/model.py:84-85" />. This placement is called <Term id="pre-norm">pre-norm</Term>.
        </p>
        <MathBlock tex={ALG2_TEX} className="text-[18px]" label={<Cite kind="paper" ref="p.26 Alg. 2" />} />
        <MathBlock tex={CODE_TEX} className="text-[18px]" label={<Cite kind="code" ref="rt/model.py:84" />} />
        <WiringCompare />
        <Callout
          variant="paper-vs-code"
          title="Post-sublayer norm in Algorithm 2, pre-norm in the code"
          paper={
            <>
              <TeX tex="\mathbf{X} \leftarrow \mathbf{X} + \mathrm{Norm}(\mathrm{MHA}(\mathbf{X}; \mathbf{M}))" />, and the same for the MLP{' '}
              <Cite kind="paper" ref="p.26 Alg. 2" />
            </>
          }
          code={
            <>
              <code className="!whitespace-normal !break-words">x = x + self.attns[l](self.norms[l](x), …)</code> and{' '}
              <code className="!whitespace-normal !break-words">x = x + self.ffn(self.norms["ffn"](x))</code>{' '}
              <Cite kind="code" ref="rt/model.py:84-85" />
            </>
          }
          verdict="This site follows the code (spec §7.1, §10 #1)."
        >
          <p>
            The code agrees with the paper’s own statement that “other architectural details (normalization, activations, etc.) follow the design
            choices of LLaMA” <Cite kind="paper" ref="p.5 §3.2" />. LLaMA normalises the input of each sublayer instead of the output, citing
            training stability <Cite kind="url" ref="https://arxiv.org/abs/2302.13971" title="Touvron et al. 2023, LLaMA, §2.2 Pre-normalization" />.
          </p>
        </Callout>
        <p>
          Neither wiring rescales the stream itself inside a block: in both, x only ever receives additions. The original Transformer did
          rescale it, with <TeX tex="x \leftarrow \mathrm{LayerNorm}(x + f(x))" />{' '}
          <Cite kind="url" ref="https://arxiv.org/abs/1706.03762" title="Vaswani et al. 2017, Attention Is All You Need, §3.1" />. What the placement
          changes is what each sublayer reads and how large its update can be:
        </p>
        <ul>
          <li>
            <strong>What a sublayer reads.</strong> In Algorithm 2 it reads the raw stream x. In the code it reads{' '}
            <TeX tex="\mathrm{RMSNorm}_\ell(x)" />, whose length depends only on the direction of x and the weights γ, whatever the length of x.
          </li>
          <li>
            <strong>How large an update can be.</strong> In Algorithm 2 every update passes through Norm, so its length is set the same way, whatever
            the sublayer computed. In the code the raw output is added, so the length of the update is free.
          </li>
        </ul>
        <p>
          The chart shows both effects in the trained mini model, for the target.
          {facts && (
            <>
              {' '}
              Its stream {facts.xLast > facts.x0 ? 'grows' : 'goes'} from ‖x‖ = {fx(facts.x0)} at the embedding to {fx(facts.xLast)} after block{' '}
              {nBlocks} (blue); a sublayer that read the raw stream, as in Algorithm 2, would see vectors of these lengths. In the code every sublayer
              reads a copy whose length stays between {fx(facts.hMin)} and {fx(facts.hMax)} (orange). The updates it adds have lengths from{' '}
              {fx(facts.dMin)} to {fx(facts.dMax)}
              {facts.nZero > 0 ? ', and exactly 0 where a sublayer has no key to read' : ''}. The final RMSNorm, <code>norm_out</code>, is the only
              norm applied to the stream itself: it brings the length to {fx(facts.xOut)} before the decoders read it{' '}
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
          The fifth sublayer is a gated MLP. The paper says it uses “gated MLPs with SiLU activation, as found in the LLaMA architecture”, with a
          hidden dimension of 1024 <Cite kind="paper" ref="p.6 §4.1" />. In the code it is three bias-free linear maps <Cite kind="code" ref="rt/model.py:50-59" />:
        </p>
        <MathBlock tex={SWIGLU_TEX} className="text-[18px] sm:text-[19px]" label={<Cite kind="code" ref="rt/model.py:59" />} />
        <p>
          <TeX tex="W_1" /> and <TeX tex="W_3" /> map the normalised vector h from d = {d} to d<sub>ff</sub> = {dFF} hidden units; <TeX tex="W_2" />{' '}
          maps back to {d}. The paper model uses 256 → 1024 → 256. In both, d<sub>ff</sub> = 4d: RT does not use LLaMA’s ⅔ sizing of the hidden
          layer <Cite kind="spec" ref="§7.1" />. The name <Term id="swiglu">SwiGLU</Term> joins Swish, another name for SiLU, with GLU, a gated linear
          unit <Cite kind="url" ref="https://arxiv.org/abs/2002.05202" title="Shazeer 2020, GLU Variants Improve Transformer" />: SiLU(W₁h) is the
          activation, and W₃h is a learned linear gate that multiplies it unit by unit.
        </p>
        <p>
          The FFN works on each token by itself. It mixes the {d} dimensions of one token’s vector but never looks at another token. In an RT block,
          information moves between tokens only in the four attention sublayers.
        </p>
        <Figure wide title="SiLU, the gate, and the hidden units of a live token">
          <SwiGLUFigure live={live} follow={follow} setFollow={setFollow} />
        </Figure>
        <Callout variant="note" title="The only gate in the block">
          <p>
            Besides SwiGLU’s W₃ branch, the paper-era block has no other gate. It also has no dropout, no attention bias, no positional encoding and
            no normalisation of queries and keys <Cite kind="spec" ref="§7.1" />.
          </p>
        </Callout>
        <CodeRef title="FFN: SwiGLU with three bias-free maps" file="rt/model.py" lines="50-59" code={MODEL_FFN_50_59} highlightLines={[59]} />
        <ScaleNote
          paper={`${PAPER_CONFIG.dModel} → ${PAPER_CONFIG.dFF} → ${PAPER_CONFIG.dModel} (${int(3 * PAPER_CONFIG.dModel * PAPER_CONFIG.dFF)} weights per FFN)`}
          ours={`${d} → ${dFF} → ${d} (${int(3 * d * dFF)} weights per FFN)`}
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="stack" title={`The stack: ${nBlocks} blocks here, 12 in the paper`} kicker="Step 4">
        <p>
          The model is a list of identical blocks with separate weights, followed by one more RMSNorm, <code>norm_out</code>{' '}
          <Cite kind="code" ref="rt/model.py:147-150" />. The forward pass runs the blocks in sequence, passing each the same four block masks, which
          were built once from the token ids, and then normalises <Cite kind="code" ref="rt/model.py:226-229" />. The paper model has 12 blocks of
          width 256 with 8 heads <Cite kind="paper" ref="p.6 §4.1" /> <Cite kind="code" ref="scripts/example_pretrain.py:32" />; the mini model has{' '}
          {nBlocks} blocks of width {d} with {H} heads.
        </p>
        <Figure wide title="The forward pass, end to end">
          <StackFigure live={live} follow={follow} />
        </Figure>
        <p>
          Counting every weight of <code>model.py</code> at the paper’s sizes gives {int(paperParams)} parameters, which the paper rounds to “about
          22M” <Cite kind="paper" ref="p.6 §4.1" /> and “22.3M” <Cite kind="paper" ref="p.26 App. I.4" />. The frozen MiniLM text encoder is not part
          of it: its 384-d embeddings are computed once, before training <Cite kind="spec" ref="§7.3" />. The mini model has {int(miniParams)}, of
          which {int(nBlocks * perBlockMini)} sit in the blocks. In both, the blocks hold most of the weights, and the attention matrices are the
          largest single group.
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
      <Section id="residual-stream" title="The residual stream of the target token" kicker="Step 5">
        <p>
          {nBlocks * 5} sublayers sit between the embedding and the decoders of the mini model ({nBlocks} blocks × 5), and each one adds its update to the
          target’s stream. The grid measures every update; the matrix shows the vector itself after each stage, one row per stage, on a shared
          colour scale. Pick another token to compare.
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
            After block 1 the target’s vector has cosine {fx(facts.cos1)} with the output of block {nBlocks}
            {facts.cos1 > 0.8 ? ': here the later blocks refine its direction more than they replace it.' : ': here the later blocks still turn it substantially.'}{' '}
            These are the mini model’s numbers on one context, not a general law.
          </p>
        )}
        <Callout variant="insight" title="Why a residual stream?">
          <p>
            Because every sublayer adds instead of replacing, a token’s vector after any sublayer is exactly its embedding x⁽⁰⁾ (the column-name and
            value terms of chapter 8) plus the sum of all updates so far. Nothing a sublayer computed is overwritten, and a sublayer with nothing to
            read changes nothing: the empty neighbor rows add exactly zero. RT takes this structure from “the standard transformer design”, adding
            the relational attention layers to each block <Cite kind="paper" ref="p.5 §3.2" />.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 6 */}
      <Section id="information-flow" title="How information can reach the target" kicker="Step 6">
        <Callout variant="note" title="An interpretation, not a claim of the paper">
          <p>
            This step reads the masks as a map of where information <em>can</em> flow. It follows from the mask definitions; the paper does not
            present it this way <Cite kind="spec" ref="§6.4" />. Whether the trained model uses a route is a separate question, which the live
            attention weights on each hop only begin to answer.
          </p>
        </Callout>
        <p>
          In an attention sublayer, query q can receive from key k exactly when the mask allows the pair. The feed-forward sublayer works per token
          and moves nothing between tokens. Read in block order, the masks give a story for the target <Cite kind="spec" ref="§6.4" />:
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
              <LayerLabel layer="nbr" />: {entity} cells read their child rows, the labelled task rows
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
              <LayerLabel layer="col" />: the target reads every visible cell of its column directly: the same column of the other{' '}
              <span className="font-mono">{seed?.tableName}</span> rows in the context.
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
              With full attention every token reaches the target in block 1, since the full mask connects every pair of non-padding tokens. That is
              the point of the layer: it “enables long-range interactions in a fixed number of layers, independent of database or graph diameter”{' '}
              <Cite kind="paper" ref="p.20 App. E" />.
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
          If full attention can reach everything in one layer, why keep the three relational layers? The paper’s App. E discusses what each layer
          contributes to the expressive power of RT <Cite kind="paper" ref="p.19-20 App. E" />:
        </p>
        <AppECards />
        <p>
          The paper closes the discussion with “we leave the full theoretical characterization of RT’s expressive power to future work”{' '}
          <Cite kind="paper" ref="p.20 App. E" />. Its empirical evidence is the layer ablation of Table 5: models trained with one layer type
          removed.
        </p>
        <Table5 />
        <Callout variant="note" title="Which code produced these ablations">
          <p>
            Tables 4, 5 and 13 are unchanged since arXiv v1 and predate the commit that set the released attention order, so they were run with the
            submission-era order feat, nbr, col, full; their “none” columns do not equal the headline Table 1 <Cite kind="spec" ref="§13 #1" />{' '}
            <Cite kind="spec" ref="§10 #25" />. Table 15 is new in v3.
          </p>
        </Callout>
        <p>
          Table 15 asks the order question directly. Mean zero-shot AUROC moves between 69.3 and 70.6 across orders, and mean R² between 22.1 and
          23.6 <Cite kind="paper" ref="p.24 Table 15" />.
        </p>
        <Table15 />
        <p>
          The playground (chapter 14) lets you switch layers off at inference time on the live context. That is a different experiment: the paper’s
          ablations retrain the model without the layer <Cite kind="spec" ref="§11.4" />.
        </p>
      </Section>

      <div className="page-grid">
        <Quiz
          title="Check your understanding"
          questions={[
            {
              prompt: 'In what order does the released code apply the four attention layers inside a block?',
              choices: [
                { text: 'column → feature → neighbor → full', correct: true, why: 'model.py:83 loops over ["col", "feat", "nbr", "full"]; Algorithm 2 prints the same order.' },
                {
                  text: 'feature → neighbor → column → full',
                  why: 'That was the ICLR-submission order (and the registration order in __init__ starts with feat, nbr). The forward loop runs col first since commit 7f891d1.',
                },
                { text: 'full → column → feature → neighbor', why: 'Full attention comes last among the four, just before the SwiGLU.' },
              ],
              explanation: 'Only the execution order in forward() matters. The paper also reports that other orders score about the same (Table 15).',
            },
            {
              prompt: 'In the code, is RMSNorm applied before or after each sublayer?',
              choices: [
                {
                  text: 'Before: x + MHA(RMSNorm(x)), so the stream itself is never normalised inside a block.',
                  correct: true,
                  why: 'model.py:84-85 normalise a copy for the sublayer and add the raw output. This is LLaMA-style pre-norm.',
                },
                {
                  text: 'After: x + Norm(MHA(x)), as in Algorithm 2.',
                  why: 'That is how Algorithm 2 (p.26) prints it, but the code is pre-norm. The site follows the code.',
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
