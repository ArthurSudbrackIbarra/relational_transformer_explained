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
import { MODEL_BLOCK_FWD_82_86, MODEL_MAKE_BLOCK_MASK_89_101, MODEL_MASKED_ATTENTION_20_47, MODEL_NORMS_ATTNS_71_79 } from './snippets';
import { StepThrough } from './StepThrough';
import { LabelShare, TargetLook, useLabelShareFacts } from './TargetLook';
import { useAttnLive, type AttnLive } from './useAttnLive';
import { provenanceClause, useProvenance } from './provenance';
import { int, pctw, rowName, singular, trimNum } from './utils';

const MHA_TEX = String.raw`\begin{aligned}
h &= \mathrm{RMSNorm}_\ell(x)\\[2pt]
Q &= h\,W_q^{\top},\qquad K = h\,W_k^{\top},\qquad V = h\,W_v^{\top}\\[2pt]
o^{(j)}_q &= \sum_{k} \operatorname{softmax}_k\!\left(\frac{\operatorname{Mask}\big(Q^{(j)}_q\cdot K^{(j)}_k;\ \mathbf{M}_\ell\big)}{\sqrt{d_{\text{head}}}}\right) V^{(j)}_k\\[2pt]
x' &= x + \big[\,o^{(1)};\ \dots;\ o^{(H)}\,\big]\,W_o^{\top}
\end{aligned}`;

const MHA_TEX_NARROW = String.raw`\begin{aligned}
h &= \mathrm{RMSNorm}_\ell(x)\\[2pt]
Q &= h\,W_q^{\top}\\ K &= h\,W_k^{\top}\\ V &= h\,W_v^{\top}\\[2pt]
o^{(j)}_q &= \sum_{k}\\ &\quad \operatorname{softmax}_k\!\bigg(\\ &\qquad \frac{\operatorname{Mask}\big(Q^{(j)}_q\cdot K^{(j)}_k;\ \mathbf{M}_\ell\big)}{\sqrt{d_{\text{head}}}}\bigg) V^{(j)}_k\\[2pt]
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
  const prov = useProvenance(live);
  let nbrKeys = 0;
  for (let k = 0; k < S; k++) nbrKeys += masks.nbr[query * S + k];

  return (
    <ChapterPage slug="attention" className="[&_sub]:[font-size:max(0.8em,13px)] [&_sup]:[font-size:max(0.8em,13px)]">
      <ChapterHeader
        slug="attention"
        lede={
          <>
            The four kinds of attention layer in RT run the same computation, each with its own weights. What sets them apart is the mask, which
            decides which token may read which. This chapter follows one token through one attention sublayer of the trained mini model, with its
            real numbers, from the normalized input to the residual update. Then it shows the attention patterns of every layer and head.
          </>
        }
        learn={[
          'The seven steps of a masked multi-head attention sublayer, with live numbers',
          `How ${H} heads split a ${d}-dimensional query into slices of ${dh}, and where the factor 1/√${dh} comes from`,
          'Why a blocked key gets exactly zero weight, and why a query with no allowed key passes through unchanged',
          'Where the masked target looks, layer by layer and head by head, in the trained mini model',
        ]}
      >
        <p className="mt-6 text-[17px] leading-relaxed text-ink-2">
          <span className="eyebrow mr-2">Live context</span>
          seed <span className="text-ink">{seed ? rowName(seed) : '–'}</span>
          {entityTok && (
            <>
              {' '}
              → <span className="text-ink">{rowName(entityTok)}</span>
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
          So the {nBlocks} blocks of the mini model hold {nBlocks * 4} different attention sublayers, and the 12 blocks of the paper model hold 48{' '}
          <Cite kind="code" ref="scripts/example_pretrain.py:32" />.
        </p>
        <p>
          Pick one in the map. Each cell shows ‖Δx‖, the length of the update that the sublayer adds to the query’s residual stream in the live
          forward pass. Then pick a query token: the <Term id="target-cell">masked target</Term> by default, or any other cell of the context.
          The rest of this chapter and chapters 7 to 9 share this choice, so a token you inspected there is the query here.
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
            of the neighbor mask is empty, and an empty row gives an update of exactly zero <Cite kind="spec" ref="§6.1" />. In the other three layers
            a token can always read at least itself <Cite kind="spec" ref="§6.2" />.
          </p>
        ) : (
          <p>
            This query’s row has {int(nbrKeys)} {nbrKeys === 1 ? 'key' : 'keys'} in the neighbor mask: tokens of child rows that point to it, so
            here even the neighbor sublayers add something. Pick the masked target to see an empty neighbor row <Cite kind="spec" ref="§6.2" />.
          </p>
        )}
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="steps" title="Seven steps through one sublayer" kicker="Step 2">
        <p>
          Each attention sublayer sits inside a <Term id="pre-norm">pre-norm</Term> residual: the block computes{' '}
          <TeX tex="x + \mathrm{MHA}(\mathrm{RMSNorm}(x))" /> <Cite kind="code" ref="rt/model.py:84" />. The norm applies to the sublayer’s
          input, and the raw output is added back to <TeX tex="x" />. Chapter 11 looks at this placement more closely. Inside MHA, three linear maps without
          bias give the queries, keys and values, which are then split into heads. FlexAttention computes a masked softmax for each head, and a
          fourth map <TeX tex="W_o" /> mixes the concatenated head outputs <Cite kind="code" ref="rt/model.py:20-47" />. For head <em>j</em> and
          query token <em>q</em>:
        </p>
        <div className="sm:hidden">
          <MathBlock
            tex={MHA_TEX_NARROW}
            className="text-[18px] sm:text-[18.5px]"
            label={<Cite kind="spec" ref="§6.1" />}
            caption={
              <span className="text-[18px]">
                ℓ is the layer (column, feature, neighbor or full), <TeX tex="\mathbf{M}_\ell" /> its mask, and{' '}
                <TeX tex={String.raw`\mathrm{Mask}(a;\mathbf{M})=a`} /> where <TeX tex={String.raw`\mathbf{M}[q,k]=1`} />, <TeX tex="-\infty" />{' '}
                otherwise.
              </span>
            }
          />
        </div>
        <div className="hidden sm:block">
          <MathBlock
            tex={MHA_TEX}
            className="text-[18px] sm:text-[18.5px]"
            label={<Cite kind="spec" ref="§6.1" />}
            caption={
              <span className="text-[18px]">
                ℓ is the layer (column, feature, neighbor or full), <TeX tex="\mathbf{M}_\ell" /> its mask, and{' '}
                <TeX tex={String.raw`\mathrm{Mask}(a;\mathbf{M})=a`} /> where <TeX tex={String.raw`\mathbf{M}[q,k]=1`} />, <TeX tex="-\infty" />{' '}
                otherwise.
              </span>
            }
          />
        </div>
        <p>
          The paper writes attention for a single head, with queries and keys of <TeX tex="d_k" /> numbers each and a divisor{' '}
          <TeX tex={String.raw`\sqrt{d_k}`} /> <Cite kind="paper" ref="p.5 §3.2" />. With several heads, <TeX tex="d_k" /> is the width of one
          head, the <TeX tex={String.raw`d_\text{head}`} /> of other chapters: 256 / 8 = 32 in the paper model and {d} / {H} = {dh} here. The code
          passes no scale to FlexAttention, which then divides by the square root of the head width by default <Cite kind="spec" ref="§6.1" />.
        </p>
        <p>
          Two pieces of general Transformer background help with the steps. They come from the original Transformer paper rather than from RT{' '}
          <Cite kind="url" ref="https://arxiv.org/abs/1706.03762" title="Vaswani et al. (2017), Attention Is All You Need, §3.2.1" label="Vaswani 2017" />.
          First, the three roles. A token’s query describes what it is looking for, each key describes what a token offers, and the score{' '}
          <TeX tex={String.raw`q\cdot k`} /> measures how well the two match. The values are what the tokens hand over, averaged with the softmax
          weights. Second, the divisor. If the numbers in <TeX tex="q" /> and <TeX tex="k" /> were independent with mean 0 and variance 1,
          their dot product would have variance <TeX tex="d_k" />. Wider heads would then give larger scores and push the softmax towards a single
          key, where its gradients are tiny. Dividing by <TeX tex={String.raw`\sqrt{d_k}`} /> brings the variance back to about 1.
        </p>
        <p>
          The figure below walks through the sublayer one step at a time. Each step shows the live tensors on the left and the arithmetic, with
          this token’s numbers, on the right.
        </p>
        <StepThrough live={live} />
        <p>Three things to watch as you step:</p>
        <ul>
          <li>
            In step 3, head {live.head + 1}’s query uses all {d} numbers of <TeX tex="h" />, but only its own {dh} rows of <TeX tex="W_q" />. The
            faded rows produce the other heads’ queries. In the code this split is the reshape <code>"b s (h d) -&gt; b h s d"</code>{' '}
            <Cite kind="code" ref="rt/model.py:35-37" />.
          </li>
          <li>
            In step 5, switch the layer while the query is the masked target. The column mask keeps the cells of the label column, the feature mask
            keeps the target’s own row and its {entity}, the neighbor mask keeps nothing, and the full mask keeps every real token{' '}
            <Cite kind="spec" ref="§6.2" />.
          </li>
          <li>
            In step 6, every blocked key gets a weight of exactly 0, because <TeX tex="e^{-\infty}=0" />, so the allowed weights alone sum to 1{' '}
            <Cite kind="paper" ref="p.5 §3.2" />.
          </li>
        </ul>
        <Callout variant="note" title="What this sublayer does not have">
          <p>
            The four projections have no bias <Cite kind="code" ref="rt/model.py:25-28" />, and there is no positional encoding{' '}
            <Cite kind="spec" ref="§7.1" />. Attention is bidirectional: there is no causal mask, only the relational one{' '}
            <Cite kind="spec" ref="§6.1" />.
          </p>
        </Callout>
        <ScaleNote paper="d = 256 · 8 heads × 32 dims · scale 1/√32 ≈ 0.177" ours={`d = ${d} · ${H} heads × ${dh} dims · scale 1/√${dh} = ${trimNum(1 / Math.sqrt(dh), 3)}`}>
          The operations are the same. The paper model has more heads, and wider ones.
        </ScaleNote>
        <CodeRef
          title="MaskedAttention: projections, head split, FlexAttention, output projection"
          file="rt/model.py"
          lines="20-47"
          code={MODEL_MASKED_ATTENTION_20_47}
          highlightLines={[25, 26, 27, 28, 35, 36, 37, 43, 45, 46]}
          defaultOpen
        />
        <CodeRef
          title={
            <>
              RelationalBlock.<wbr />
              forward: each sublayer is a pre-norm residual
            </>
          }
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
          head, where row <em>q</em> is the softmax over the keys that <em>q</em> may read. Below are all {4 * H} matrices of one block. All heads
          of a layer share its mask <Cite kind="code" ref="rt/model.py:89-101" />, so the gray pattern repeats along each row of panels, and the
          heads differ only in how they spread weight inside it.
        </p>
        <HeadGrid live={live} />
        <LayerStatsProse live={live} />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="target" title="Where does the target look?" kicker="Step 4">
        <p>
          In this figure the query is always the masked target, and the four layers sit side by side. Each card lists the keys with the largest
          weights. Hover one to find it in the context underneath, drawn as small tables in the style of the paper’s Figure 1{' '}
          <Cite kind="paper" ref="p.2 Fig. 1" />.
        </p>
        <p>
          The masks decide what the target can read, whatever the weights. In the column layer it reads the cells of its own column (the task’s
          labels from every split) and itself. In the feature layer it reads its own row and the row of its {entity}. In the neighbor layer it reads
          nothing, because no row points to a task row, and in the full layer it reads every real token <Cite kind="spec" ref="§6.2" />{' '}
          <Cite kind="spec" ref="§6.4" />.
        </p>
        <TargetLook live={live} />
        <LabelsIntroProse live={live} />
        <LabelShare live={live} />
        <LabelShareProse live={live} />
        <Callout variant="warning" title="Reading attention weights with care">
          <p>
            These are the weights of a {int(paramCount(live.cfg))}-parameter replica{' '}
            {prov ? provenanceClause(prov) : 'trained on small synthetic databases'}. They show how this small model attends, and the paper’s
            model may attend differently. A weight also tells you only where a head looks. The value vectors, <TeX tex="W_o" /> and every
            later sublayer decide what the target makes of it, so read these patterns as clues rather than explanations.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 5 */}
      <Section id="implementation" title="Implementation note: FlexAttention and a dense loop" kicker="Step 5">
        <p>
          The paper implements the relational layers “with sparse attention masks, and compiled to efficient FlashAttention-based kernels using
          FlexAttention” <Cite kind="paper" ref="p.5 §3.2" />. In every forward pass the code turns the four masks into FlexAttention block masks
          once, before the first block, and every block reuses them <Cite kind="code" ref="rt/model.py:197-206" />{' '}
          <Cite kind="code" ref="rt/model.py:226-227" />. A block mask lets the GPU skip the parts of the score matrix that a mask blocks
          entirely <Cite kind="code" ref="rt/model.py:89-101" />{' '}
          <Cite kind="url" ref="https://docs.pytorch.org/docs/2.8/nn.attention.flex_attention.html" title="PyTorch FlexAttention documentation" />.
        </p>
        <p>
          This site runs in your browser without a GPU, so its engine loops over each query’s allowed keys in plain TypeScript. The arithmetic is
          the same masked softmax: scores scaled by <TeX tex={String.raw`1/\sqrt{d_{\text{head}}}`} />, blocked keys excluded, weights
          normalized over the allowed keys <Cite kind="spec" ref="§6.1" />. A query with no allowed key, such as the masked target in the neighbor
          layer, has nothing to normalize. FlexAttention outputs 0 for such a row, and since <TeX tex="W_o" /> has no bias, the update is 0 as
          well <Cite kind="spec" ref="§6.1" />. The checks below test these properties on the engine’s live forward pass:
        </p>
        <ImplementationChecks live={live} />
        <CodeRef title="_make_block_mask: one mask, every head" file="rt/model.py" lines="89-101" code={MODEL_MAKE_BLOCK_MASK_89_101} highlightLines={[96]} />
      </Section>

      <div className="page-grid">
        <Quiz
          title="Check your understanding"
          questions={[
            {
              prompt: 'By what factor are the attention scores multiplied before the softmax in the paper’s model? And in this site’s mini model?',
              choices: [
                { text: `1/√256 in the paper, 1/√${d} here`, why: 'That would be the full model width. The scale uses the width of one head.' },
                {
                  text: `1/√32 in the paper, 1/√${dh} here`,
                  correct: true,
                  why: `A head is 256/8 = 32 numbers wide in the paper and ${d}/${H} = ${dh} here, and the scale is 1 over the square root of that width.`,
                },
                {
                  text: 'No scaling: the scores go into the softmax as raw dot products',
                  why: 'The code passes no scale to FlexAttention, so FlexAttention applies its default, 1 over the square root of the head width.',
                },
              ],
              explanation: `The scale is 1 over the square root of the head width: 1/√32 ≈ 0.177 in the paper, 1/√${dh} ${Number.isInteger(Math.sqrt(dh)) ? '=' : '≈'} ${trimNum(1 / Math.sqrt(dh), 3)} here. Without it, wider heads would give larger scores and a sharper softmax (spec §6.1).`,
            },
            {
              prompt: 'The masked target’s row of the neighbor mask is empty. What does the neighbor sublayer add to the target?',
              choices: [
                { text: 'Exactly zero: the target passes through unchanged.', correct: true, why: 'A row with no allowed key outputs 0, and W_o has no bias (spec §6.1).' },
                { text: 'The mean of all value vectors.', why: 'That would be an even softmax over every key. A row without allowed keys outputs 0 instead.' },
                { text: 'Its own value vector, since a token always attends to itself.', why: 'Self-attention is allowed in column, feature and full, but not in neighbor: no row is its own parent.' },
              ],
              explanation:
                'For real tokens, only the neighbor mask can have an empty row, and an empty row always gives a zero update. The live checks in Step 5 confirm this on the engine’s output.',
            },
            {
              prompt: 'Do the heads of one attention layer use different masks?',
              choices: [
                { text: 'Yes: each head learns its own mask.', why: 'Masks are built from the token ids, not learned, and each layer has a single mask for all its heads.' },
                {
                  text: 'No: all heads share the layer’s mask; they differ in their projection weights, and so in how they spread weight over the allowed keys.',
                  correct: true,
                },
                { text: 'Only in the full layer, which has no mask.', why: 'The full layer’s mask is the padding mask, and it is shared like the others.' },
              ],
              explanation: 'The code builds one block mask per layer and applies it to every head of that layer (rt/model.py:89-101).',
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
          The figures follow your simulation settings: <span>{db.name}</span> ·{' '}
          <span>{task.id.split('/')[1]}</span> ({task.kind}). The prose is written for the running example, ShopDB’s{' '}
          <span>customer-churn</span>, so a few sentences may not apply.
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
      {int(stats.nbr.empty)} of the {int(live.numReal)} queries have an empty row and get no update at all <Cite kind="spec" ref="§6.2" />, while
      the others read {keys(stats.nbr.keys)} keys on average. A <LayerLabel layer="full" /> query may read all {keys(stats.full.keys)}. Its
      largest weight averages {pctw(stats.full.top)}, where a perfectly even spread would give {pctw(1 / Math.max(1, stats.full.keys))}.
    </p>
  );
}

/** The paragraph between the per-layer cards and the label-share chart (live share of the full layer). */
function LabelsIntroProse({ live }: { live: AttnLive }) {
  const f = useLabelShareFacts(live);
  const { task, numReal } = live;
  const forecast = task.kind === 'forecast';
  const entity = task.entityTable ? singular(task.entityTable) : 'entity';
  const entities = task.entityTable ?? 'entities';
  const full0 = f?.fullOnLabels[0];
  return (
    <>
      <p>
        Only the column layer confines the target to its own column, so there the visible labels compete with nothing but each other and the
        target itself <Cite kind="spec" ref="§6.2" />. The full layer can read the same label cells, but mixed among all{' '}
        {int(numReal)} tokens
        {full0 !== null && full0 !== undefined && f && f.nSelf + f.nOther > 0 ? (
          <>
            : in block 1 it gives them {pctw(full0)} of its weight (mean of heads), against {pctw((f.nSelf + f.nOther) / Math.max(1, numReal))} for
            an even spread
          </>
        ) : null}
        .{' '}
        {forecast && (
          <>
            Some of those labels belong to the same {entity} as the seed row (its own past labels: chapter 6’s self labels), others to other {entities}{' '}
            <Cite kind="spec" ref="§2.5" />.
          </>
        )}
      </p>
      <p>
        The paper finds that “zero-shot transfer is driven primarily by the model’s ability to leverage past labels of the target entity”:
        removing them drops the mean zero-shot AUROC from 70.1 to 53.8 <Cite kind="paper" ref="p.9 Table 4" />. Among the attention layers,
        removing the column layer hurts zero-shot R² the most (22.8 → 11.9) <Cite kind="paper" ref="p.10 Table 5" />, although for
        classification, the task type of the running example, the effect is small (70.1 → 69.8 AUROC){' '}
        <Cite kind="paper" ref="p.23 Table 13" />. Each of these ablations trains a separate model, with extra blocks to keep the parameter
        count the same <Cite kind="paper" ref="p.9 §5.2" />. The chart below shows how the mini model’s target divides its column-layer weight.
      </p>
    </>
  );
}

function LabelShareProse({ live }: { live: AttnLive }) {
  const f = useLabelShareFacts(live);
  const { task } = live;
  const entity = task.entityTable ? singular(task.entityTable) : 'entity';
  const entities = task.entityTable ?? 'entities';
  if (!f || task.kind !== 'forecast' || f.nSelf === 0) return null;
  const first = f.perBlock[0];
  const last = f.perBlock[f.perBlock.length - 1];
  if (!first) return null;
  const nB = f.perBlock.length;
  const cmp = (v: number, u: number) => (Math.abs(v - u) < 0.0005 ? 'about what' : v > u ? 'more than' : 'less than');
  return (
    <p>
      In block 1, averaged over heads, the target gives {pctw(first.self)} of its column-layer weight to its {f.nSelf} self labels. They are{' '}
      {f.nSelf} of the {f.nAllowed} cells it can read there ({pctw(f.uniformSelf)}), so that is {cmp(first.self, f.uniformSelf)} an even split
      would give.
      {f.nOther > 0 && (
        <>
          {' '}
          The {f.nOther} labels of other {entities} get {pctw(first.other)}, against an even share of {pctw(f.uniformOther)}.
        </>
      )}
      {last && nB > 1 && (
        <>
          {' '}
          In block {nB} the own labels get {pctw(last.self)}
          {f.nOther > 0 ? ` and the other ${entities}’ labels ${pctw(last.other)}` : ''}.
        </>
      )}{' '}
      If the model leans on the {entity}’s own history, these weights are the first place where that can show.
    </p>
  );
}
