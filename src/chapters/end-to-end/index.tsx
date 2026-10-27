/**
 * Chapter 13 "end-to-end": One prediction, start to end (docs/CHAPTER_BRIEFS.md §13).
 *
 * The whole forward pass on one seed, on one page: the task and seed row, the sampler's trace, the
 * batch fields, the input embedding, the four masks, every operation of every block, the decoders and
 * the read-out, each as the live matrices of the engine (src/state/hooks.ts → src/engine/pipeline.ts
 * → model/forward.ts, a mirror of rt/model.py:153-267) with the trained weights that produce them.
 * Displayed arithmetic is recomputed by ./math.ts and checked against the capture in math.test.ts.
 * Spec: research/RT_TECHNICAL_SPEC.md §0, §2-§8, §10 #1, #2, #5, #7, #8.
 */
import { Link } from 'react-router-dom';
import {
  Button,
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
  StatusBadge,
  Term,
} from '@/components';
import { PAPER_CONFIG, PAPER_L, PAPER_W, paramCount } from '@/engine/model/config';
import { MODEL_MASKED_ATTENTION_20_47 } from '../attention/snippets';
import { MODEL_BLOCK_62_86 } from '../block/snippets';
import { MAIN_READOUT_283_293, MODEL_BLOCKS_NORM_OUT_226_229 } from '../decoding/snippets';
import { MODEL_EMBED_208_224 } from '../embedding/snippets';
import { MODEL_MASKS_164_191 } from '../masks/snippets';
import { FLY_EXPAND } from '../sampling/code';
import { FLY_FIELDS_462_500 } from '../tokens/snippets';
import { BlockStage } from './BlockStage';
import { EmbedStage } from './EmbedStage';
import { int } from './format';
import { E2EProvider, runKind, useE2E } from './live';
import { MaskStage } from './MaskStage';
import { OutputStage } from './OutputStage';
import { PassMap } from './PassMap';
import { SampleStage } from './SampleStage';
import { TaskStage } from './TaskStage';
import { TokenStage } from './TokenStage';

const EMBED_TEX = String.raw`x_{0,i} \;=\; \underbrace{\mathrm{RMSNorm}_{\text{col}}\!\big(W_{\text{col}}\,c_i + b_{\text{col}}\big)}_{\text{column name}} \;+\; \underbrace{\begin{cases}\mathrm{RMSNorm}_{t}\!\big(W_{t}\,v_i + b_{t}\big) & \text{not masked}\\ m_{t} & \text{masked}\end{cases}}_{\text{value, of type } t}`;
const BLOCK_TEX = String.raw`\begin{aligned} x &\leftarrow x + \mathrm{MHA}_{\ell}\big(\mathrm{RMSNorm}_{\ell}(x);\,M_{\ell}\big)\quad \ell = \text{col},\,\text{feat},\,\text{nbr},\,\text{full}\\ x &\leftarrow x + W_2\big(\mathrm{SiLU}(W_1 h) \odot W_3 h\big),\quad h = \mathrm{RMSNorm}_{\text{ffn}}(x)\end{aligned}`;

export default function EndToEndChapter() {
  return (
    <E2EProvider>
      <Body />
    </E2EProvider>
  );
}

function Body() {
  const live = useE2E();
  const { db, task, ctx, cfg, assets, numReal, nBlocks, d, H, dh, dFF, disabledLayers, setDisabledLayers, labels } = live;
  const kind = runKind(live);
  const forecast = task.kind === 'forecast';
  const binary = task.targetType === 'binary';
  const miniParams = paramCount(cfg);

  return (
    <ChapterPage slug="end-to-end">
      <ChapterHeader
        slug="end-to-end"
        lede={
          <>
            Each earlier chapter took one step of the model apart. This one runs a single prediction from the first choice to the last number, on one
            page: pick the question, watch the sampler gather the context, then follow every matrix of the forward pass, with the weights that make it,
            to the probability or the value at the end.
          </>
        }
        learn={[
          'How one seed row becomes a context of L cells, a batch of token fields and an L × d matrix',
          'Every operation of every block as a matrix: products with the trained weights, norms, masks, softmax, residual adds',
          'Where any single number comes from: hover an entry to see its arithmetic',
          'How the masked cell’s last vector becomes the answer, and the loss it would give in training',
        ]}
      >
        <p className="mt-6 text-[17px] leading-relaxed text-ink-2">
          <span className="eyebrow mr-2">Live run</span>
          task <span className="text-ink">{task.id.split('/')[1]}</span> · L = {ctx.params.L}, w = {ctx.params.w} · checkpoint{' '}
          <code>{assets.ckpt}</code>{' '}
          {kind === 'zero-shot' && <StatusBadge status="good">zero-shot</StatusBadge>}
          {assets.weightsKind === 'init' && <StatusBadge status="warning">untrained weights</StatusBadge>} <ScaleNote inline />
        </p>
      </ChapterHeader>

      {/* ───────────────────────────────────────────────────────────── 0 */}
      <Section id="map" title="The whole pass on one page">
        <p>
          A forward pass of RT answers one question: the value of one <Term id="target-cell">masked cell</Term>. Exactly one cell of the context is
          masked, the seed row’s target cell <Cite kind="code" ref="rustler/src/fly.rs:492-496" /> <Cite kind="spec" ref="§4.10" />. Everything else on
          this page serves that single number. The map below lists the seven stages with the shapes that flow between them; each section after it
          opens one stage on the live seed.
        </p>
        <PassMap />
        <p>
          The figures share one <em>followed token</em>. It starts as the masked target, and you can follow any other token by clicking its row in any
          matrix or picking it from the list. Its row is outlined in every matrix and its numbers are written out next to the figure. Matrices show
          one small rectangle per entry: red for positive values, blue for negative, gray near zero, each on its own scale unless a figure says
          otherwise. Hover or focus any entry, or move through a matrix with the arrow keys, to read its value and how it was computed.
        </p>
        {disabledLayers.length > 0 && (
          <Callout variant="warning" title="An inference-time ablation is active">
            <p>
              The playground switched off {disabledLayers.join(', ')} attention in every block, so those sublayers pass the residual stream through
              unchanged on this page too.
            </p>
            <Button size="sm" onClick={() => setDisabledLayers([])}>
              Switch every attention layer back on
            </Button>
          </Callout>
        )}
      </Section>

      {/* ───────────────────────────────────────────────────────────── 1 */}
      <Section id="question" title="Pick a task and a seed row" kicker="Stage 1">
        <p>
          A forecasting task lives in a <Term id="task-table">task table</Term> that “stores the forecasting labels, together with foreign keys linking
          to the relevant entities … and a timestamp specifying the prediction horizon” <Cite kind="paper" ref="p.4 §2.2" />. RT attaches it to the
          database like any other table, and its rows serve as <Term id="seed-row">seed rows</Term> <Cite kind="paper" ref="p.4 §3.1" />. An{' '}
          <Term id="autocomplete-task">autocomplete task</Term> seeds on a row of an ordinary table instead and masks one of its columns, after dropping
          the columns that would give the answer away <Cite kind="paper" ref="p.19 App. D" /> <Cite kind="spec" ref="§2.4" />.
        </p>
        <p>
          The candidate seeds are the rows of the chosen table whose target cell exists: the code walks the table and keeps a row only if it has a cell
          in the target column <Cite kind="code" ref="rustler/src/fly.rs:287-296" />. For a forecasting task the table is the chosen split of the task
          table; for autocomplete it is the database table, whatever the split <Cite kind="code" ref="rt/data.py:77-81" />. Pick a task and a seed
          below, or keep the course’s running example.
        </p>
        <TaskStage />
        <p>
          The type of the target decides the rest of the pipeline’s end: a boolean target is read from the boolean decoder as a logit, a number target
          from the number decoder as a z-score <Cite kind="code" ref="rt/main.py:283-288" />. <Link to="/tasks">Chapter 3</Link> builds the task tables
          and their splits.
        </p>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="sample" title="Sample the context" kicker="Stage 2">
        <p>
          A database does not fit in one context, so the sampler gathers cells around the seed, row by row, until it has L = {ctx.params.L} of them
          (1024 in the paper) or runs out of rows <Cite kind="paper" ref="p.15 Alg. 1" /> <Cite kind="code" ref="rustler/src/fly.rs:357-511" />. Four
          rules drive it <Cite kind="spec" ref="§4.4" />:
        </p>
        <ul>
          <li>
            Parent rows, reached through foreign keys, come first, from a last-in, first-out stack: “F→P links are immediately followed”{' '}
            <Cite kind="paper" ref="p.16 App. A" /> <Cite kind="code" ref="rustler/src/fly.rs:378-379" />.
          </li>
          <li>
            Otherwise the next row is drawn uniformly among the child rows with the lowest depth label <Cite kind="code" ref="rustler/src/fly.rs:381-396" />
            .
          </li>
          <li>
            A child row with a timestamp later than the seed’s is skipped, and at most w = {ctx.params.w} database children of each row are kept, chosen
            uniformly; task-table children, which carry the labels, are all kept <Cite kind="code" ref="rustler/src/fly.rs:415-452" />.
          </li>
          <li>
            Each visited row adds its cells in order, and the sampler stops at exactly L, even in the middle of a row{' '}
            <Cite kind="code" ref="rustler/src/fly.rs:454-509" />.
          </li>
        </ul>
        <SampleStage />
        <p>
          {forecast ? (
            <>
              For this seed the context holds {labels.self.length} earlier label{labels.self.length === 1 ? '' : 's'} of the same entity and{' '}
              {labels.other.length} of other entities: unmasked cells of the target column, the{' '}
              <Term id="in-context-label">in-context labels</Term> <Cite kind="spec" ref="§2.5" />.
            </>
          ) : (
            <>
              For this seed the context holds {labels.other.length} other visible cell{labels.other.length === 1 ? '' : 's'} of the target column, the{' '}
              <Term id="in-context-label">in-context labels</Term> of an autocomplete task <Cite kind="spec" ref="§2.4" />.
            </>
          )}{' '}
          The paper’s Algorithm 1 differs from the released code in a few places, such as picking parents at random instead of from a stack; the site
          follows the code (<Link to="/sampling">chapter 6</Link>) <Cite kind="spec" ref="§10 #5" />.
        </p>
        <CodeRef
          title="Expanding a visited row: parents, the time filter, task rows, the width bound"
          file={FLY_EXPAND.file}
          lines={[FLY_EXPAND.start, FLY_EXPAND.end]}
          code={FLY_EXPAND.code}
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 3 */}
      <Section id="tokens" title="Write the tokens" kicker="Stage 3">
        <p>
          Each emitted cell becomes one <Term id="cell-token">token</Term>, written into flat arrays position by position: the row’s node id and its up
          to five parent ids, the table and column-phrase ids, the semantic type, the value slots and the flags{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:462-500" /> <Cite kind="spec" ref="§5.1" />. Positions left over after the sampler stops keep the
          padding defaults <Cite kind="code" ref="rustler/src/fly.rs:65-85" />. The masked target keeps its true label in its value slot; only its{' '}
          <code>masks</code> and <code>is_targets</code> flags mark it, and the model’s input gate ignores the value{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:492-496" /> <Cite kind="spec" ref="§5.3" />.
        </p>
        <TokenStage />
        <CodeRef title="Writing one token’s fields" file="rustler/src/fly.rs" lines="462-500" code={FLY_FIELDS_462_500} highlightLines={[492, 494, 496]} />
        <p>
          From here on, two kinds of field go two ways. The ids are compared between pairs of tokens to build the masks (stage 5); they are never
          embedded. The values become the input vectors (stage 4). <Link to="/tokens">Chapter 7</Link> goes through the fields one by one.
        </p>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="embed" title="Embed every token" kicker="Stage 4">
        <p>
          Every token starts as the sum of two {d}-d vectors: one for its column, one for its value <Cite kind="code" ref="rt/model.py:208-224" />{' '}
          <Cite kind="spec" ref="§5.2" />:
        </p>
        <MathBlock tex={EMBED_TEX} label={<Cite kind="spec" ref="§5.2" />} />
        <p>
          c<sub>i</sub> is the frozen MiniLM embedding of the phrase “&lt;col&gt; of &lt;table&gt;” and v<sub>i</sub> the token’s value: a z-scored
          number, datetime or boolean <Cite kind="spec" ref="§3.3" />, or the MiniLM embedding of a text value <Cite kind="spec" ref="§3.6" />. Each
          type t has its own linear encoder and its own RMSNorm. The masked cell gets its type’s learned mask vector m<sub>t</sub> in place of a value,
          as the paper puts it: “For masked cells, the value embedding … is replaced with a learned mask vector” <Cite kind="paper" ref="p.5 §3.1" />.
          Padding gets the zero vector. Nothing encodes a token’s position in the context <Cite kind="paper" ref="p.5 §3.1" />.
        </p>
        <EmbedStage />
        <Callout
          variant="paper-vs-code"
          paper={<>x = W_d r + W E_schema(c, t): two linear maps, no bias, no norm (p.5 §3.1).</>}
          code={<>A bias and an RMSNorm after every encoder; the mask vector is added raw (model.py:115-146, 208-224).</>}
          verdict={<>The site follows the code <Cite kind="spec" ref="§10 #2" />.</>}
        />
        <CodeRef title="The input embedding" file="rt/model.py" lines="208-224" code={MODEL_EMBED_208_224} />
        <p>
          <Link to="/embedding">Chapter 8</Link> takes the embedding apart term by term, and <Link to="/semantics">chapter 5</Link> shows what the frozen
          text encoder makes of the column phrases.
        </p>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 5 */}
      <Section id="masks" title="Build the four masks" kicker="Stage 5">
        <p>
          Which rows link to which enters the model only here <Cite kind="spec" ref="§0" />. From the ids alone, the forward pass builds four boolean{' '}
          {int(numReal)} × {int(numReal)} matrices, one per attention layer <Cite kind="code" ref="rt/model.py:164-191" />: <em>column</em> (same column of the same table), <em>feature</em> (the same row
          and its parent rows), <em>neighbor</em> (the child rows) and <em>full</em> (everything) <Cite kind="paper" ref="p.5 §3.2" />. Each is ANDed
          with “neither token is padding”, built once, and shared by every head of every block <Cite kind="code" ref="rt/model.py:89-101" />{' '}
          <Cite kind="code" ref="rt/model.py:204-206" />.
        </p>
        <MaskStage />
        <CodeRef title="The four masks" file="rt/model.py" lines="164-191" code={MODEL_MASKS_164_191} />
        <p>
          <Link to="/masks">Chapter 9</Link> explores who can see whom in each layer and why.
        </p>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 6 */}
      <Section id="blocks" title={`Run the ${nBlocks} blocks`} kicker="Stage 6">
        <p>
          The {int(numReal)} × {d} matrix x₀ now passes through {nBlocks} blocks with separate weights (12 in the paper){' '}
          <Cite kind="paper" ref="p.6 §4.1" />. Inside each block, four attention sublayers run in the order column, feature, neighbor, full, then a
          SwiGLU feed-forward sublayer, each wrapped in a pre-norm residual <Cite kind="code" ref="rt/model.py:82-86" /> <Cite kind="spec" ref="§7.1" />:
        </p>
        <MathBlock tex={BLOCK_TEX} label={<Cite kind="code" ref="rt/model.py:82-86" />} />
        <p>
          An attention sublayer projects the normalized rows to queries, keys and values with three {d} × {d} matrices, splits them into {H} heads of{' '}
          {dh} dimensions, scores every query against every key, masks the forbidden pairs to −∞, takes a softmax per row, averages the values with those
          weights, and projects the concatenated heads back with W<sub>o</sub> <Cite kind="code" ref="rt/model.py:20-47" /> <Cite kind="spec" ref="§6.1" />.
          The SwiGLU widens each token to {dFF} twice, gates one copy with the other and projects back <Cite kind="code" ref="rt/model.py:50-59" />. None
          of these layers has a bias. The player below walks every operation of every block, {nBlocks * 32} steps in all; the sublayer grid jumps
          straight to one.
        </p>
        <BlockStage />
        <Callout
          variant="paper-vs-code"
          paper={<>Algorithm 2 writes X ← X + Norm(MHA(X; M)), a norm after the sublayer (p.26).</>}
          code={<>Pre-norm, x + MHA(RMSNorm(x)), as in LLaMA (model.py:84-85).</>}
          verdict={
            <>
              The site follows the code <Cite kind="spec" ref="§10 #1" />.
            </>
          }
        />
        <CodeRef title="One block: four masked attentions, then the SwiGLU" file="rt/model.py" lines="62-86" code={MODEL_BLOCK_62_86} highlightLines={[83, 84, 85]} />
        <CodeRef title="Masked multi-head attention" file="rt/model.py" lines="20-47" code={MODEL_MASKED_ATTENTION_20_47} />
        <p>
          <Link to="/attention">Chapter 10</Link> follows one query through one sublayer with bar charts, and <Link to="/block">chapter 11</Link> follows
          one token’s vector through the whole stack.
        </p>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 7 */}
      <Section id="output" title="Decode and read the answer" kicker="Stage 7">
        <p>
          After the last block one more RMSNorm, <code>norm_out</code>, normalizes every row <Cite kind="code" ref="rt/model.py:229" />. Then four linear
          decoders, one per data type, run at every position <Cite kind="code" ref="rt/model.py:238-239" />, and the one that matches the masked cell is
          read there: “the decoder to select for final prediction depends on the task type, or equivalently on the datatype of the masked cell”{' '}
          <Cite kind="paper" ref="p.6 §3.3" />.
        </p>
        <OutputStage />
        <p>
          {binary ? (
            <>
              For this task the answer is a logit; the probability is its sigmoid <Cite kind="spec" ref="§8.4" />. The paper’s metric, AUROC, ranks the
              logits of many seeds directly <Cite kind="code" ref="rt/main.py:328-337" />.
            </>
          ) : (
            <>
              For this task the answer is a z-score; multiplied by the column’s standard deviation and shifted by its mean, the statistics it was
              z-scored with, it is a value in the column’s units <Cite kind="spec" ref="§8.4" />. The paper’s metric, R², is computed on the z-scores{' '}
              <Cite kind="code" ref="rt/main.py:328-337" />.
            </>
          )}{' '}
          That is the whole pass, and no weight changed during it.{' '}
          {kind === 'zero-shot' ? (
            <>
              The model never trained on {db.title.split(' ')[0]}: everything it used about this database came through the context window{' '}
              <Cite kind="paper" ref="p.4 §2.3" />.
            </>
          ) : kind === 'seen' ? (
            <>{db.title.split(' ')[0]} was one of the pretraining databases, so this prediction is not zero-shot (<Link to="/training">chapter 14</Link>).</>
          ) : kind === 'finetuned' ? (
            <>These weights were fine-tuned on ShopDB, so this prediction is not zero-shot (<Link to="/training">chapter 14</Link>).</>
          ) : (
            <>These weights are untrained, so the numbers show the mechanism only.</>
          )}{' '}
          <Link to="/decoding">Chapter 12</Link> covers the decoders, the losses and the metrics in depth.
        </p>
        <CodeRef title="The blocks, then norm_out" file="rt/model.py" lines="226-229" code={MODEL_BLOCKS_NORM_OUT_226_229} />
        <CodeRef title="Reading the prediction and the label at is_targets" file="rt/main.py" lines="283-293" code={MAIN_READOUT_283_293} highlightLines={[284, 287]} />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 8 */}
      <Section id="scale" title="The same pass at the paper’s size">
        <p>
          Every operation on this page is the paper-era code’s; only the sizes differ. At the paper’s scale the context has {PAPER_L} cells gathered
          with w = {PAPER_W}, the matrices are {PAPER_L} × {PAPER_CONFIG.dModel}, the heads are {PAPER_CONFIG.nHeads} of{' '}
          {PAPER_CONFIG.dModel / PAPER_CONFIG.nHeads}, the SwiGLU is {PAPER_CONFIG.dFF} wide and there are {PAPER_CONFIG.nBlocks} blocks:{' '}
          {int(paramCount(PAPER_CONFIG))} weights against this model’s {int(miniParams)} <Cite kind="spec" ref="§7.3" />. The map at the top of the page
          switches its shapes between the two.
        </p>
        <ScaleNote
          paper={`L = ${PAPER_L}, d = ${PAPER_CONFIG.dModel}, ${PAPER_CONFIG.nHeads} heads, ${PAPER_CONFIG.nBlocks} blocks`}
          ours={`L = ${ctx.params.L}, d = ${d}, ${H} heads, ${nBlocks} blocks`}
        >
          The paper runs in bfloat16 and this site in float32; the RMSNorm ε is 2⁻⁷ in both <Cite kind="spec" ref="§7.2" />.
        </ScaleNote>
        <p>
          The numbers here come from the site’s TypeScript forward pass. It matches a PyTorch port, and that port matches the official{' '}
          <code>model.py</code> class loaded with the same weights; <Link to="/about">chapter 17</Link> shows those checks.
        </p>
      </Section>

      {/* ───────────────────────────────────────────────────────────── quiz */}
      <Section id="quiz" title="Check your understanding">
        <Quiz
          questions={[
            {
              prompt: 'How many cells does one forward pass predict?',
              choices: [
                { text: 'Every cell of the context', why: 'The decoders run at every position, but only one output is read.' },
                { text: 'One: the seed row’s target cell', correct: true },
                { text: 'One per row of the context', why: 'Only the seed row has a masked cell.' },
              ],
              explanation: (
                <>
                  Exactly one token is masked, the seed’s target cell <Cite kind="code" ref="rustler/src/fly.rs:492-496" />; the prediction is read at that
                  position only <Cite kind="code" ref="rt/main.py:283-288" />.
                </>
              ),
            },
            {
              prompt: 'Which fields of the batch decide who may attend to whom?',
              choices: [
                { text: 'The value slots and the column-phrase embeddings', why: 'Those build the input vectors, not the masks.' },
                { text: 'The node, parent, column and table ids, and the padding flag', correct: true },
                { text: 'The token positions', why: 'Positions are used nowhere in the model.' },
              ],
              explanation: (
                <>
                  The four masks compare ids between pairs of tokens <Cite kind="code" ref="rt/model.py:164-191" />.
                </>
              ),
            },
            {
              prompt: 'Inside one block, what runs right after column attention?',
              choices: [
                { text: 'Feature attention', correct: true },
                { text: 'The SwiGLU feed-forward', why: 'The SwiGLU comes last, after all four attentions.' },
                { text: 'A RMSNorm of the residual stream itself', why: 'The norms only feed each sublayer; x is not normalized between sublayers.' },
              ],
              explanation: (
                <>
                  col → feat → nbr → full, then the SwiGLU, each pre-norm <Cite kind="code" ref="rt/model.py:82-86" />.
                </>
              ),
            },
            {
              prompt: binary ? 'What does the boolean decoder output at the masked cell?' : 'What does the number decoder output at the masked cell?',
              choices: binary
                ? [
                    { text: 'A probability between 0 and 1', why: 'The sigmoid is a display step; the code keeps the logit.' },
                    { text: 'A logit', correct: true },
                    { text: 'The class label 0 or 1', why: 'The label is read from the token, not predicted as a class.' },
                  ]
                : [
                    { text: 'The value in the column’s units', why: 'That needs the Train mean and standard deviation, a display step.' },
                    { text: 'A z-score in the column’s own scale', correct: true },
                    { text: 'A rank among the seeds', why: 'R² compares z-scores; nothing is ranked.' },
                  ],
              explanation: (
                <>
                  The decoders are linear layers <Cite kind="code" ref="rt/model.py:124-131" />; σ and the de-normalization are applied for display{' '}
                  <Cite kind="spec" ref="§8.4" />.
                </>
              ),
            },
          ]}
        />
      </Section>

      <ChapterNav slug="end-to-end" />
    </ChapterPage>
  );
}
