/**
 * Chapter 10 "attention": Inside one attention sublayer (BUILD_PLAN §6, docs/CHAPTER_BRIEFS.md §10).
 *
 * Follows one query token through one masked multi-head attention sublayer of the trained mini model,
 * with every tensor read from the live forward pass (src/engine/model/forward.ts, a mirror of
 * rt/model.py:20-47 inside the pre-norm residual of :82-86), then shows the attention maps of every
 * layer and head and where the masked target looks.
 * Spec: research/RT_TECHNICAL_SPEC.md §6.1, §6.2, §6.4, §7.1, §7.2, §10 #1.
 */
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
import { paramCount } from '@/engine/model/config';
import { useSim } from '@/state/store';
import { HeadGrid, useLayerStats } from './HeadGrid';
import { ImplementationChecks } from './Implementation';
import { QueryPicker, SublayerMap } from './Controls';
import { MODEL_BLOCK_FWD_82_86, MODEL_FLEX_14_17, MODEL_MAKE_BLOCK_MASK_89_101, MODEL_MASKED_ATTENTION_20_47, MODEL_NORMS_ATTNS_71_79 } from './snippets';
import { StepThrough } from './StepThrough';
import { LabelShare, TargetLook, useLabelShareFacts } from './TargetLook';
import { useAttnLive, type AttnLive } from './useAttnLive';
import { int, pctw, rowName, singular } from './utils';

const MHA_TEX = String.raw`\begin{aligned}
h &= \mathrm{RMSNorm}_\ell(x)\\[2pt]
Q &= h\,W_q^{\top},\qquad K = h\,W_k^{\top},\qquad V = h\,W_v^{\top}\\[2pt]
o^{(j)}_q &= \sum_{k} \operatorname{softmax}_k\!\left(\frac{\operatorname{Mask}\big(Q^{(j)}_q\cdot K^{(j)}_k;\ \mathbf{M}_\ell\big)}{\sqrt{d_{\text{head}}}}\right) V^{(j)}_k\\[2pt]
x' &= x + \big[\,o^{(1)};\ \dots;\ o^{(H)}\,\big]\,W_o^{\top}
\end{aligned}`;

export default function AttentionChapter() {
  const live = useAttnLive();
  const { ctx, task, assets, target, query, H, dh, d, nBlocks, masks, S } = live;
  const seed = ctx.tokens[target];
  const entityNode = seed?.f2pNbrIdxs.find((p) => p >= 0);
  const entityTok = entityNode !== undefined ? ctx.tokens.find((t) => !t.isPadding && t.nodeIdx === entityNode) : undefined;
  const entityName = entityNode !== undefined ? ctx.tokens.find((t) => !t.isPadding && t.nodeIdx === entityNode && t.sem === 1) : undefined;
  const entity = task.entityTable ? singular(task.entityTable) : 'entity';
  let nbrKeys = 0;
  for (let k = 0; k < S; k++) nbrKeys += masks.nbr[query * S + k];

  return (
    <ChapterPage slug="attention">
      <ChapterHeader
        slug="attention"
        lede={
          <>
            All four kinds of attention layer in RT run the same computation, each with its own weights; what sets them apart is the mask that
            decides which token may read which. This chapter follows one token through one attention sublayer with the real numbers of the trained
            mini model, from the normalised input to the residual update, then shows the attention patterns of every layer and head.
          </>
        }
        learn={[
          'The seven steps of a masked multi-head attention sublayer, with live numbers',
          `How ${H} heads split a ${d}-dimensional query into slices of ${dh}, and where the factor 1/√${dh} comes from`,
          'Why a blocked key gets exactly zero weight, and a query with no allowed key gets exactly zero update',
          'Where the masked target looks, layer by layer and head by head, in the trained mini model',
        ]}
      >
        <p className="mt-6 text-[17px] leading-relaxed text-ink-2">
          <span className="eyebrow mr-2">Live context</span>
          seed <span className="font-mono text-ink">{seed ? rowName(seed) : '–'}</span>
          {entityTok && (
            <>
              {' '}
              → <span className="font-mono text-ink">{rowName(entityTok)}</span>
              {entityName ? ` (${String(entityName.raw)})` : ''}
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

      <RunningExampleNote live={live} />

      {/* ───────────────────────────────────────────────────────────── 1 */}
      <Section id="choose" title="Pick a sublayer and a query" kicker="Step 1">
        <p>
          Chapter 9 built four masks from the token ids. Each RT block uses them in four attention sublayers, always in the order column,
          feature, neighbor, full, and then applies a feed-forward sublayer <Cite kind="code" ref="rt/model.py:82-86" />. Every attention
          sublayer has its own <Term id="rmsnorm">RMSNorm</Term> and its own four projection matrices <Cite kind="code" ref="rt/model.py:71-79" />.
          The {nBlocks} blocks of the mini model therefore hold {nBlocks * 4} different attention sublayers; the 12 blocks of the paper model hold 48{' '}
          <Cite kind="code" ref="scripts/example_pretrain.py:32" />.
        </p>
        <p>
          Choose one in the map. Each cell shows the length of the update, ‖Δx‖, that the sublayer adds to the query’s residual stream in the
          live forward pass. Then choose a query token: the <Term id="target-cell">masked target</Term> by default, or any other cell of the
          context. The choice is shared with the rest of this chapter and with chapters 7 to 9, so a token you inspected there is the query here.
        </p>
        <Figure wide title="The attention sublayers of the mini model, and the query token">
          <div className="grid gap-7 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] xl:gap-8">
            <div className="min-w-0">
              <div className="mb-2 text-[15px] font-semibold text-ink-2">Sublayer: click a cell to select it</div>
              <SublayerMap live={live} />
            </div>
            <div className="min-w-0">
              <div className="mb-2 text-[15px] font-semibold text-ink-2">Query token: a preset, or click any token</div>
              <QueryPicker live={live} />
            </div>
          </div>
        </Figure>
        {nbrKeys === 0 ? (
          <p>
            For this query the Neighbor row of the map reads 0 in every block. No token in the context has the query’s row among its parents, so its row
            of the neighbor mask is empty, and an empty row gives an update of exactly zero <Cite kind="spec" ref="§6.1" />. The other three layers
            always contain at least the query itself <Cite kind="spec" ref="§6.2" />.
          </p>
        ) : (
          <p>
            This query’s row has {int(nbrKeys)} {nbrKeys === 1 ? 'key' : 'keys'} in the neighbor mask: tokens of child rows that point to it. So
            here even the neighbor sublayers add something. Try the masked target to see an empty neighbor row <Cite kind="spec" ref="§6.2" />.
          </p>
        )}
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="steps" title="Seven steps through one sublayer" kicker="Step 2">
        <p>
          Here is the sublayer as the code writes it. The block computes <TeX tex="x + \mathrm{MHA}(\mathrm{RMSNorm}(x))" />, a{' '}
          <Term id="pre-norm">pre-norm</Term> residual <Cite kind="code" ref="rt/model.py:84" />. Inside MHA, three linear maps without bias give
          queries, keys and values; these are split into heads; FlexAttention computes a masked softmax per head; and a fourth map{' '}
          <TeX tex="W_o" /> mixes the concatenated head outputs <Cite kind="code" ref="rt/model.py:20-47" />. For head <em>j</em> and query token{' '}
          <em>q</em>:
        </p>
        <MathBlock
          tex={MHA_TEX}
          className="text-[14.5px] sm:text-[18.5px]"
          label={<Cite kind="spec" ref="§6.1" />}
          caption={
            <span className="text-[15.5px]">
              ℓ is the layer (column, feature, neighbor or full), <TeX tex="\mathbf{M}_\ell" /> its mask, and{' '}
              <TeX tex={String.raw`\mathrm{Mask}(a;\mathbf{M})=a`} /> where <TeX tex={String.raw`\mathbf{M}[q,k]=1`} />, <TeX tex="-\infty" />{' '}
              otherwise.
            </span>
          }
        />
        <p>
          The paper writes attention for a single head, with <TeX tex={String.raw`\mathbf{Q},\mathbf{K}\in\mathbb{R}^{n\times d_k}`} /> and a
          divisor <TeX tex={String.raw`\sqrt{d_k}`} /> <Cite kind="paper" ref="p.5 §3.2" />. With heads, <TeX tex="d_k" /> is the width of one
          head: the code calls FlexAttention without a scale, and its default is <TeX tex={String.raw`1/\sqrt{E}`} />, where <em>E</em> is the
          last dimension of each head’s query <Cite kind="spec" ref="§6.1" />. That is 256 / 8 = 32 in the paper model and {d} / {H} = {dh} here.
        </p>
        <p>
          Step through it below. Each step shows the live tensors on the left and the arithmetic, with this token’s numbers, on the right.
        </p>
        <StepThrough live={live} />
        <p>Three things to watch as you step:</p>
        <ul>
          <li>
            In step 3, head {live.head + 1}’s query uses all {d} numbers of <TeX tex="h" />, but only its own {dh} rows of <TeX tex="W_q" />. The
            faded rows produce the other heads’ queries. Splitting the output this way is the reshape{' '}
            <code>"b s (h d) -&gt; b h s d"</code> <Cite kind="code" ref="rt/model.py:35-37" />.
          </li>
          <li>
            In step 5, switch the layer while the query is the masked target. The column mask keeps the cells of the label column, the feature mask
            keeps the target’s own row and its {entity}, the neighbor mask keeps nothing, and the full mask keeps every real token{' '}
            <Cite kind="spec" ref="§6.2" />.
          </li>
          <li>
            In step 6, the blocked keys do not get a small weight; they get exactly 0, because <TeX tex="e^{-\infty}=0" />. The allowed weights
            alone sum to 1 <Cite kind="paper" ref="p.5 §3.2" />.
          </li>
        </ul>
        <Callout
          variant="paper-vs-code"
          title="Where the norm sits"
          paper={
            <>
              Algorithm 2 writes each sublayer as
              <span className="scroll-quiet my-1 block overflow-x-auto">
                <TeX tex={String.raw`X \leftarrow X + \mathrm{Norm}(\mathrm{MHA}(X;\mathbf{M}))`} />
              </span>
              a norm after the sublayer <Cite kind="paper" ref="p.26 Alg. 2" />.
            </>
          }
          code={
            <>
              <code className="my-1 block text-[15px] break-words whitespace-pre-wrap">x = x + self.attns[l](self.norms[l](x), …)</code>
              The norm is applied to the sublayer’s input, and the residual adds the raw output <Cite kind="code" ref="rt/model.py:84" />.
            </>
          }
          verdict="This site follows the code (pre-norm), consistent with the paper's statement that RT follows LLaMA's design choices (spec §7.1, §10 #1). Chapter 11 compares the two forms."
        />
        <Callout variant="note" title="What this sublayer does not have">
          <p>
            No bias in any of the four projections <Cite kind="code" ref="rt/model.py:25-28" />, no dropout, no positional encoding and no
            normalisation of queries or keys <Cite kind="spec" ref="§7.1" />. Attention is bidirectional: there is no causal mask, only the
            relational one <Cite kind="spec" ref="§6.1" />.
          </p>
        </Callout>
        <ScaleNote paper="d = 256 · 8 heads × 32 dims · scale 1/√32 ≈ 0.177" ours={`d = ${d} · ${H} heads × ${dh} dims · scale 1/√${dh} = ${(1 / Math.sqrt(dh)).toFixed(3).replace(/0+$/, '')}`}>
          The operations are the same; the paper model has more and wider heads. Values here are float32, the paper’s are bfloat16.
        </ScaleNote>
        <CodeRef
          title="MaskedAttention: projections, head split, FlexAttention, output projection"
          file="rt/model.py"
          lines="20-47"
          code={MODEL_MASKED_ATTENTION_20_47}
          highlightLines={[25, 26, 27, 28, 35, 36, 37, 43, 45, 46]}
          defaultOpen
          note="The SDPA branch (lines 39-41) never runs: every layer always receives a block mask (spec §6.1)."
        />
        <CodeRef
          title="RelationalBlock.forward: each sublayer is a pre-norm residual"
          file="rt/model.py"
          lines="82-86"
          code={MODEL_BLOCK_FWD_82_86}
          highlightLines={[84]}
        />
        <CodeRef title="One RMSNorm and one MaskedAttention per layer" file="rt/model.py" lines="71-79" code={MODEL_NORMS_ATTNS_71_79} />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 3 */}
      <Section id="all-heads" title="All heads, all layers" kicker="Step 3">
        <p>
          The step-through followed one row of one head. A sublayer computes such a row for every token at once: one S×S matrix of weights per
          head, where row <em>q</em> is the softmax over the keys that <em>q</em> may read. Below are all {4 * H} matrices of one block. Every head
          of a layer uses the same mask, because <code>_make_block_mask</code> passes <code>H=None</code> to <code>create_block_mask</code>{' '}
          <Cite kind="code" ref="rt/model.py:89-101" />, so the grey pattern repeats along each row of panels. The heads differ only in how they
          spread weight inside it.
        </p>
        <HeadGrid live={live} />
        <LayerStatsProse live={live} />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="target" title="Where does the target look?" kicker="Step 4">
        <p>
          Now fix the query to the masked target and compare the four layers. Each card lists the keys with the largest weights; hover one to find
          it in the context below, which is drawn as small tables in the style of the paper’s Figure 1 <Cite kind="paper" ref="p.2 Fig. 1" />.
        </p>
        <p>
          The masks decide what the target can read, whatever the weights. In the column layer: the cells of its own column, which are the task’s
          labels from every split, and itself. In the feature layer: its own row and the row of its {entity}. In the neighbor layer: nothing,
          because no row points to a task row. In the full layer: every real token <Cite kind="spec" ref="§6.2" /> <Cite kind="spec" ref="§6.4" />.
        </p>
        <TargetLook live={live} />
        <p>
          The column layer is the only place where the target reads other task rows’ labels directly. Some of those labels belong to the same{' '}
          {entity} as the seed row (its own past labels), others to other {entity}s <Cite kind="spec" ref="§2.5" />. The paper finds that “zero-shot
          transfer is driven primarily by the model’s ability to leverage past labels of the target entity”: removing them drops the mean zero-shot
          AUROC from 70.1 to 53.8 <Cite kind="paper" ref="p.9 Table 4" />, and removing the column layer hurts zero-shot R² the most among the
          attention layers (22.8 → 11.9) <Cite kind="paper" ref="p.10 Table 5" />. The chart below shows how the mini model’s target divides its
          column-layer weight.
        </p>
        <LabelShare live={live} />
        <LabelShareProse live={live} />
        <Callout variant="warning" title="Reading attention weights with care">
          <p>
            These are the weights of a {int(paramCount(live.cfg))}-parameter replica trained on four small synthetic databases, so they describe this model, not
            the paper’s. And a weight says where a head looks, not what it takes away: the value vectors, <TeX tex="W_o" /> and every later
            sublayer decide what the information becomes. Treat the patterns as clues, not explanations.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 5 */}
      <Section id="implementation" title="Implementation note: FlexAttention and a dense loop" kicker="Step 5">
        <p>
          The paper implements the relational layers “with sparse attention masks, and compiled to efficient FlashAttention-based kernels using
          FlexAttention” <Cite kind="paper" ref="p.5 §3.2" />. The code compiles <code>flex_attention</code> once and, before the first block,
          turns each dense boolean mask into a block mask, which records which tiles of (query, key) pairs contain any allowed pair, so the kernel
          can skip the empty tiles <Cite kind="code" ref="rt/model.py:14-17" /> <Cite kind="code" ref="rt/model.py:89-101" />{' '}
          <Cite kind="url" ref="https://docs.pytorch.org/docs/2.8/nn.attention.flex_attention.html" title="PyTorch FlexAttention documentation" />.
        </p>
        <p>
          This site runs in your browser without a GPU, so its engine loops over each query’s allowed keys in plain TypeScript. The arithmetic is
          the same masked softmax: scores scaled by <TeX tex={String.raw`1/\sqrt{d_{\text{head}}}`} />, blocked keys excluded, weights
          normalised over the allowed keys <Cite kind="spec" ref="§6.1" />. One case needs care. A query with no allowed key would divide 0 by 0.
          In the pinned PyTorch 2.6 kernel such a row outputs exactly 0, neither NaN nor a uniform average, and with a bias-free{' '}
          <TeX tex="W_o" /> the update is exactly 0 too <Cite kind="spec" ref="§6.1" />. Our loop skips such rows, which gives the same zeros. The
          live forward pass confirms it:
        </p>
        <ImplementationChecks live={live} />
        <p>
          The remaining difference is precision. The paper runs the model in bfloat16 <Cite kind="code" ref="rt/main.py:187" />; this site stores
          values in float32 and accumulates in float64, so its numbers are slightly more precise, not differently computed{' '}
          <Cite kind="spec" ref="§10 #20" />.
        </p>
        <CodeRef title="FlexAttention, compiled once" file="rt/model.py" lines="14-17" code={MODEL_FLEX_14_17} highlightLines={[17]} />
        <CodeRef title="_make_block_mask: one mask, every head" file="rt/model.py" lines="89-101" code={MODEL_MAKE_BLOCK_MASK_89_101} highlightLines={[96]} />
      </Section>

      <div className="page-grid">
        <Quiz
          title="Check your understanding"
          questions={[
            {
              prompt: 'By what factor are the attention scores multiplied before the softmax in the paper’s model? And in this site’s mini model?',
              choices: [
                { text: '1/√256 in the paper, 1/√64 here', why: 'That would be the full model width. FlexAttention scales by the width of one head.' },
                {
                  text: '1/√32 in the paper, 1/√16 here',
                  correct: true,
                  why: 'd_head = 256/8 = 32 in the paper and 64/4 = 16 here; FlexAttention’s default scale is 1/√E with E the head width.',
                },
                { text: 'No scaling: FlexAttention uses raw dot products', why: 'The code passes no scale, so FlexAttention applies its default 1/√E.' },
              ],
              explanation: 'The scale is 1 over the square root of the head width: 1/√32 ≈ 0.177 in the paper, 1/√16 = 0.25 here (spec §6.1).',
            },
            {
              prompt: 'The masked target’s row of the neighbor mask is empty. What does the neighbor sublayer add to the target?',
              choices: [
                { text: 'Exactly zero: the target passes through unchanged.', correct: true, why: 'A fully masked row outputs 0, and W_o has no bias (spec §6.1).' },
                { text: 'The mean of all value vectors.', why: 'That would be a uniform softmax over every key. The kernel returns 0 for a row without allowed keys.' },
                { text: 'Its own value vector, since a token always attends to itself.', why: 'Self-attention is allowed in column, feature and full, but not in neighbor: no row is its own parent.' },
              ],
              explanation: 'Empty rows only occur in the neighbor mask, and they always give a zero update. The live check in Step 5 confirms it.',
            },
            {
              prompt: 'Do the heads of one attention layer use different masks?',
              choices: [
                { text: 'Yes: each head learns its own mask.', why: 'Masks are built from ids, not learned, and _make_block_mask passes H=None.' },
                {
                  text: 'No: all heads share the layer’s mask; they differ in their projection weights, and so in how they spread weight over the allowed keys.',
                  correct: true,
                },
                { text: 'Only in the full layer, which has no mask.', why: 'The full layer’s mask is the padding mask, and it is shared like the others.' },
              ],
              explanation: 'create_block_mask(..., H=None) broadcasts one mask over every head (rt/model.py:89-101).',
            },
          ]}
        />
      </div>

      <ChapterNav slug="attention" />
    </ChapterPage>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

function RunningExampleNote({ live }: { live: AttnLive }) {
  const setTask = useSim((s) => s.setTask);
  const { db, task } = live;
  if (task.id === 'shop/customer-churn') return null;
  return (
    <div className="page-grid">
      <Callout variant="note" title="You are viewing another task">
        <p>
          The figures follow your simulation settings: <span className="font-mono">{db.name}</span> ·{' '}
          <span className="font-mono">{task.id.split('/')[1]}</span> ({task.kind}). The prose is written for the running example, ShopDB’s{' '}
          <span className="font-mono">customer-churn</span>, so a few sentences may not apply.
        </p>
        <Button size="sm" variant="secondary" onClick={() => setTask('shop/customer-churn')}>
          Switch the course to the running example
        </Button>
      </Callout>
    </div>
  );
}

function LayerStatsProse({ live }: { live: AttnLive }) {
  const stats = useLayerStats(live);
  if (!stats) return null;
  const { block } = live;
  const keys = (n: number) => n.toFixed(n < 10 ? 1 : 0);
  return (
    <p>
      Some numbers for block {block + 1}, averaged over the real query tokens and the heads. A <LayerLabel layer="col" /> query may read{' '}
      {keys(stats.col.keys)} keys on average, and its largest weight averages {pctw(stats.col.top)}. A <LayerLabel layer="feat" /> query may read{' '}
      {keys(stats.feat.keys)} keys, with a largest weight of {pctw(stats.feat.top)}. In the <LayerLabel layer="nbr" /> layer,{' '}
      {int(stats.nbr.empty)} of the {int(live.numReal)} queries have an empty row and receive no update at all <Cite kind="spec" ref="§6.2" />; the
      others read {keys(stats.nbr.keys)} keys on average. A <LayerLabel layer="full" /> query may read all {keys(stats.full.keys)}, and its largest
      weight averages {pctw(stats.full.top)}, against {pctw(1 / Math.max(1, stats.full.keys))} for a perfectly even spread.
    </p>
  );
}

function LabelShareProse({ live }: { live: AttnLive }) {
  const f = useLabelShareFacts(live);
  const { task } = live;
  const entity = task.entityTable ? singular(task.entityTable) : 'entity';
  if (!f || f.perBlock[0] === null || f.nSelf === 0) return null;
  const first = f.perBlock[0] ?? 0;
  const last = f.perBlock[f.perBlock.length - 1] ?? 0;
  const above = first > f.uniform;
  return (
    <p>
      In block 1, averaged over heads, the target gives {pctw(first)} of its column-layer weight to its {f.nSelf} own past labels, which make up{' '}
      {pctw(f.uniform)} of the cells it can read there. That is {above ? 'more' : 'less'} than an even split would give
      {f.perBlock.length > 1 ? `; in block ${f.perBlock.length} the share is ${pctw(last)}` : ''}.{' '}
      {f.nOther === 0
        ? `In this context every visible label belongs to the seed’s own ${entity}: no task row of another ${entity} made it into the window (chapter 6 shows how the sampler picks rows). `
        : ''}
      Whether the model leans on the {entity}’s own history shows up here first, but only as weights: what the target then does with those values
      is decided by everything after.
    </p>
  );
}
