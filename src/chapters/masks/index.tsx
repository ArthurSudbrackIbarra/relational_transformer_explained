/**
 * Chapter 9 "masks": Relational attention masks (BUILD_PLAN §6, docs/CHAPTER_BRIEFS.md §9).
 *
 * Every mask shown is the engine's (src/engine/masks.ts, a mirror of rt/model.py:164-191) for the
 * live context of the global store, so the seed, context and tokens match every other chapter.
 * Spec: research/RT_TECHNICAL_SPEC.md §6.1-6.3, §6.6, §10 #12, #24.
 */
import { useMemo } from 'react';
import {
  Callout,
  ChapterHeader,
  ChapterNav,
  ChapterPage,
  Cite,
  CodeRef,
  LayerLabel,
  MathBlock,
  Quiz,
  ScaleNote,
  Section,
  TeX,
  Term,
} from '@/components';
import { SemType } from '@/engine/types';
import { ColumnTableCheck, PaddingDemo, TransposeFlip, ZeroUpdateCheck } from './Details';
import { FourMasks } from './FourMasks';
import { LayerCards } from './LayerCards';
import { MaskCounts } from './MaskCounts';
import { article, childlessTables, int, LAYER_ORDER, listAnd, maskStats, pct, rowName, rowTag, singular } from './maskUtils';
import {
  MODEL_ATTN_FWD_30_47,
  MODEL_BLOCK_FWD_82_86,
  MODEL_BLOCK_MASKS_193_206,
  MODEL_BLOCKS_LOOP_226_227,
  MODEL_MAKE_BLOCK_MASK_89_101,
  MODEL_MASKS_164_191,
} from './snippets';
import { SoftmaxDemo } from './SoftmaxDemo';
import { useLiveMasks } from './useLiveMasks';
import { WhoCanISee } from './WhoCanISee';
import { WorkedExample } from './WorkedExample';

const SDPA_TEX = String.raw`\begin{gathered}\operatorname{SDPA}(\mathbf{Q},\mathbf{K},\mathbf{V};\mathbf{M}) = \operatorname{Softmax}\!\left(\frac{\operatorname{Mask}\!\left(\mathbf{Q}\mathbf{K}^{\top};\,\mathbf{M}\right)}{\sqrt{d_k}}\right)\mathbf{V}\\[6pt] \operatorname{Mask}(\mathbf{A};\mathbf{M})_{ij} = \begin{cases}\mathbf{A}_{ij} & \text{if } \mathbf{M}_{ij} = 1\\ -\infty & \text{if } \mathbf{M}_{ij} = 0\end{cases}\end{gathered}`;

export default function MasksChapter() {
  const live = useLiveMasks();
  const { ctx, masks, S, target, grouping, db, prefixes } = live;
  const stats = useMemo(() => maskStats(masks, ctx.tokens), [masks, ctx]);
  const byLayer = Object.fromEntries(stats.map((s) => [s.layer, s])) as Record<(typeof LAYER_ORDER)[number], (typeof stats)[number]>;

  const seed = ctx.tokens[target];
  const entityNode = seed?.f2pNbrIdxs.find((p) => p >= 0);
  const entityTok = entityNode !== undefined ? ctx.tokens.find((t) => !t.isPadding && t.nodeIdx === entityNode) : undefined;
  const entityText = entityTok ? ctx.tokens.find((t) => !t.isPadding && t.nodeIdx === entityNode && t.sem === SemType.Text) : undefined;
  const nRows = grouping.rowByNode.size;
  const dbShort = db.title.split(' ')[0];
  const hasPad = ctx.numReal < S;

  // live examples for the prose (never hard-coded): a child row of the entity in another table, its parents
  const inCtx = (n: number) => ctx.tokens.find((t) => !t.isPadding && t.nodeIdx === n);
  const childTok =
    entityNode === undefined || !seed
      ? undefined
      : ctx.tokens.find((t) => !t.isPadding && t.tableType === 'Db' && t.tableNameIdx !== seed.tableNameIdx && t.f2pNbrIdxs.includes(entityNode));
  const parentsOf = (t: NonNullable<typeof seed>) =>
    [...new Set(t.f2pNbrIdxs.filter((p) => p >= 0))].map(inCtx).filter((x): x is NonNullable<typeof x> => !!x).map(rowName);
  const entityPositions = entityNode !== undefined ? ctx.tokens.filter((t) => !t.isPadding && t.nodeIdx === entityNode).map((t) => t.pos) : [];
  const entityChildren = entityNode !== undefined ? new Set(ctx.tokens.filter((t) => !t.isPadding && t.f2pNbrIdxs.includes(entityNode)).map((t) => t.nodeIdx)).size : 0;
  const entRange = entityPositions.length ? `${entityPositions[0]}–${entityPositions[entityPositions.length - 1]}` : '';
  const childless = childlessTables(db);
  const forecast = !!seed && seed.tableType !== null && seed.tableType !== 'Db';
  const targetNbrKeys = seed ? masks.nbr.subarray(target * S, (target + 1) * S).reduce((a, b) => a + b, 0) : 0;
  const parentTables = db.tables.map((t) => t.schema.name).filter((n) => !childless.includes(n));

  return (
    <ChapterPage slug="masks">
      <ChapterHeader
        slug="masks"
        lede={
          <>
            Each RT block runs four attention layers with the same arithmetic, each with its own weights and its own normalization. What makes
            them relational is a boolean mask that decides which token may look at which. This chapter builds the four masks for the live context
            and shows what each one lets through.
          </>
        }
        learn={[
          'How a 0/1 mask becomes −∞ scores inside scaled dot-product attention',
          'The column, feature, neighbor and full masks, in the paper’s notation and in the code',
          `Which keys any token can see, on the live ${dbShort} context`,
          'The details that keep it correct: the padding AND, empty neighbor rows, shared heads',
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
          {int(ctx.numReal)} cell tokens from {nRows} rows · L = {ctx.params.L}, w = {ctx.params.w} <ScaleNote inline />
        </p>
      </ChapterHeader>

      {/* ───────────────────────────────────────────────────────────── 1 */}
      <Section id="sdpa" title="Attention with a mask" kicker="Step 1">
        <p>
          Inside every attention layer, RT computes standard scaled dot-product attention. Each token’s query vector is compared with every key
          vector, the scores pass through a softmax, and the resulting weights average the value vectors <Cite kind="paper" ref="p.5 §3.2" />. Each
          of the four layers has its own query, key, value and output weights and its own RMSNorm{' '}
          <Cite kind="code" ref="rt/model.py:70-78" />. What makes them <Term id="relational-attention">relational attention</Term> layers is
          the <strong>attention mask</strong>{' '}
          <TeX tex="\mathbf{M}\in\{0,1\}^{n\times n}" />: <TeX tex="\mathbf{M}[q,k]=1" /> means that token <em>q</em> may attend to token{' '}
          <em>k</em>, and 0 means it cannot <Cite kind="paper" ref="p.5 §3.2" />.
        </p>
        <MathBlock
          tex={SDPA_TEX}
          className="text-[15px] sm:text-[19px]"
          label={<Cite kind="paper" ref="p.5 §3.2" />}
          caption={<span className="text-[15.5px]">Scaled dot-product attention with masking, as printed in the paper (v3).</span>}
        />
        <p>
          The mask acts <em>before</em> the softmax. A blocked pair gets the score −∞, and <TeX tex="e^{-\infty}=0" />, so the softmax shares all
          of the weight among the allowed keys as if the blocked ones did not exist. Rows are queries and columns are keys; every matrix in this
          chapter is drawn that way.
        </p>
        <Callout variant="note" title="An erratum in earlier versions of the paper">
          <p>
            arXiv v1 and v2 wrote <TeX tex="\operatorname{Softmax}\big((\mathbf{Q}\mathbf{K}^{\top}\odot\mathbf{M})/\sqrt{d_K}\big)" />, an
            element-wise product with the mask. Version 3 corrected it to the −∞ form above, which is what the code computes through FlexAttention{' '}
            <Cite kind="spec" ref="§6.6" /> <Cite kind="code" ref="rt/model.py:43" />. The figure shows what the product would do if read
            literally: a score multiplied by 0 becomes 0, and <TeX tex="e^{0}=1" /> is still a weight, so blocked keys would keep part of the
            attention.
          </p>
        </Callout>
        <p>
          The figure uses the real scores of the trained mini model at the masked target. It lets you pick one of the three relational layers,
          which Step 2 defines; for now, watch only what happens to the blocked keys. Tokens are labelled row·column
          {seed && entityTok ? (
            <>
              : <span className="font-mono">{rowTag(seed, prefixes)}</span> is the seed row {rowName(seed)} and{' '}
              <span className="font-mono">{rowTag(entityTok, prefixes)}</span> is {rowName(entityTok)}
            </>
          ) : null}
          .
        </p>
        <SoftmaxDemo live={live} />
        <p>
          In the code the mask never appears in the arithmetic itself. Each layer hands a precompiled block mask to FlexAttention. The plain
          scaled-dot-product branch is never taken, because all four layers always receive a mask <Cite kind="spec" ref="§6.1" />.
        </p>
        <CodeRef
          title="MaskedAttention.forward: the mask goes straight into flex_attention"
          file="rt/model.py"
          lines="30-47"
          code={MODEL_ATTN_FWD_30_47}
          highlightLines={[42, 43]}
          note="flex_attention applies the default scale 1/√E with E = d_head, and treats a 0 in the block mask as −∞ (spec §6.1)."
        />
        <ScaleNote paper="8 heads, d_head = 32" ours="4 heads, d_head = 16">
          The masks themselves do not depend on the model size: they are the same 0/1 matrices for any d or number of heads.
        </ScaleNote>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="four-masks" title="The four masks, side by side" kicker="Step 2">
        <p>
          The paper defines the masks with three helpers. For the cell of token <em>i</em>, <TeX tex="\mathrm{Col}(i)" /> is its column,{' '}
          <TeX tex="\mathrm{Row}(i)" /> its row, and <TeX tex="\mathrm{OutLinks}(i)" /> the set of rows, possibly in other tables, that the{' '}
          <Term id="foreign-key">foreign keys</Term> of <TeX tex="\mathrm{Row}(i)" /> point to <Cite kind="paper" ref="p.5 §3.2" />.
          {seed && (
            <>
              {' '}
              In the live context, the OutLinks of the seed row {rowName(seed)} {parentsOf(seed).length > 1 ? 'are' : 'is'}{' '}
              {listAnd(parentsOf(seed)) || 'empty'}
              {childTok ? (
                <>
                  ; the OutLinks of {rowName(childTok)} {parentsOf(childTok).length > 1 ? 'are' : 'is'} {listAnd(parentsOf(childTok))}
                </>
              ) : null}
              .
            </>
          )}
        </p>
        <MaskDefinitions />
        <p>
          The code builds all four masks from five arrays of the token batch: <code>node_idxs</code> (which row),{' '}
          <code>f2p_nbr_idxs</code> (that row’s parents, up to five, padded with −1), <code>col_name_idxs</code> and <code>table_name_idxs</code>{' '}
          (which column of which table) and <code>is_padding</code> <Cite kind="code" ref="rt/model.py:154-158" />. No value, no embedding and no
          position enters a mask. RT has no positional encodings; relational structure enters only through these masks, which do not care about
          token order <Cite kind="paper" ref="p.5 §3.1" /> <Cite kind="spec" ref="§6.5" />.
        </p>
        <FourMasks live={live} />
        <p>
          Each mask has a shape you can read. The <Term id="column-attention">column</Term> mask is made of diagonal stripes. Rows of one table
          emit their cells in the same column order, so where such rows follow each other, the cells of one column recur at a fixed stride and
          their pairs line up parallel to the diagonal. Task rows have only two cells, so a run of task rows makes a small checkerboard. The{' '}
          <Term id="feature-attention">feature</Term> mask has a small block on the diagonal for each row, because a row’s cells are emitted
          together and read each other.
          {entityTok && entRange ? (
            <>
              {' '}
              It also has a tall bar at keys {entRange}: the cells of {rowName(entityTok)}, which each of its {entityChildren} child rows in the
              context reads as a parent. The <Term id="neighbor-attention">neighbor</Term> mask holds the same bar turned on its side, a band in
              query rows {entRange} where {rowName(entityTok)} reads its children. Step 6 shows that this is an exact transpose.
            </>
          ) : (
            <>
              {' '}
              The <Term id="neighbor-attention">neighbor</Term> mask is mostly empty: only rows that have children in the context have anything
              in it.
            </>
          )}{' '}
          The <Term id="full-attention">full</Term> mask is solid{hasPad ? ', apart from the padding rows and columns' : ''}.
        </p>
        <CodeRef
          title="The four masks, built from ids"
          file="rt/model.py"
          lines="164-191"
          code={MODEL_MASKS_164_191}
          highlightLines={[165, 187, 188, 189, 190]}
          note="Each comparison broadcasts to a (B, S, S) boolean tensor: row = query, column = key. The engine's buildMasks mirrors these lines one by one."
        />
        <Callout
          variant="paper-vs-code"
          title="“Global” and “full” are the same layer"
          paper={
            <>
              §3.2 lists “four attention types: column, feature, neighbor, and global”, and App. E and J also say “global”{' '}
              <Cite kind="paper" ref="p.5 §3.2" /> <Cite kind="paper" ref="p.19 App. E" />. The bullet title and Algorithm 2 say “Full attention”.
            </>
          }
          code={
            <>
              the dictionary key is <code>"full"</code>, and its mask is just <code>pad</code> <Cite kind="code" ref="rt/model.py:186-191" />
            </>
          }
          verdict="There are four layers, not five. This site calls the last one full (spec §6.2)."
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 3 */}
      <Section id="who-can-i-see" title="Who can I see?" kicker="Step 3">
        <p>
          One row of a mask answers one question: which tokens can this query read? Pick a query below with a preset or by clicking any token.
          Each panel draws the live context the way paper Figure 1 does, as small tables with one square per cell, and fills the keys the query
          may attend to under one mask <Cite kind="paper" ref="p.2 Fig. 1" />. Hover a cell to see what it is and which masks let the query read
          it.
        </p>
        <p>
          Start with the <Term id="target-cell">masked target</Term>, then try the presets. Some things to look for. The target sees{' '}
          {forecast ? 'the task’s visible labels' : `the other ${seed?.colName ?? ''} values`} in the column panel and{' '}
          {targetNbrKeys === 0 ? 'nothing in the neighbor panel' : 'its children in the neighbor panel'}.
          {entityTok && (
            <>
              {' '}
              {article(singular(entityTok.tableName)).replace(/^./, (c) => c.toUpperCase())} {singular(entityTok.tableName)} cell sees{' '}
              {parentsOf(entityTok).length ? 'its own row and its parents' : 'only its own row'} in the feature panel, yet in the neighbor panel it
              lights up every row in the context that points to it.
            </>
          )}
          {childTok && (
            <>
              {' '}
              {article(singular(childTok.tableName)).replace(/^./, (c) => c.toUpperCase())} {singular(childTok.tableName)} cell sees its own row
              and its {parentsOf(childTok).length === 1 ? 'parent' : 'parents'} in the feature panel.
            </>
          )}
        </p>
        <WhoCanISee live={live} />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="what-flows" title="What each layer is for" kicker="Step 4">
        <p>
          The paper motivates each layer in one sentence, and its Appendix E adds what each contributes to expressive power. The cards pair those
          sentences with what the mask does in the live context. Each diagram is a small, real excerpt of the context, chosen for that layer: the
          live mask restricted to those tokens. Lines run from every key the query may read into the query.
        </p>
        <LayerCards live={live} />
        <Callout variant="insight" title="Why four masked layers instead of one full layer?">
          <p>
            App. E argues that two of the layers are not strictly needed for expressivity: global attention can emulate column attention, and
            feature and global attention together can simulate neighbor attention <Cite kind="paper" ref="p.19-20 App. E" />. Feature attention
            cannot be replaced that way: it is “the only mechanism that explicitly groups cells into rows” <Cite kind="paper" ref="p.19 App. E" />.
            The paper still hard-wires all four patterns, describing the relational layers as ones that “effectively encode relational structure
            and inductive bias” <Cite kind="paper" ref="p.5 §3.2" />. Its ablation (Table 5) trains models with one layer type removed, adding
            blocks to keep the parameter count constant <Cite kind="spec" ref="§11.4" />; the results chapter shows those numbers.
          </p>
        </Callout>
        <Callout variant="note" title="How information can reach the target (a teaching interpretation, not a paper claim)">
          <p>
            Reading the masks in block order: in the column layer the target reads every visible label directly, but cannot yet tell which are
            its own entity’s. In the feature layer, the target and every past task row read their customer, so all of them now carry information
            about which entity they concern. In the neighbor layer the customer reads its children, labelled task rows and orders included. The
            full layer mixes everything, and stacking blocks composes these hops <Cite kind="spec" ref="§6.4" />. Chapter 11 follows this path
            through the residual stream.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 5 */}
      <Section id="counts" title="How much each mask lets through" kicker="Step 5">
        <p>
          The masks differ widely in density. In the live context of {S} positions, full attention allows {int(byLayer.full.pairs)} ordered
          pairs
          {hasPad
            ? `: every pair of the ${int(ctx.numReal)} non-padding tokens (padding is excluded), ${pct(byLayer.full.density)} of the ${int(S * S)}`
            : `, all ${int(S * S)} of them`}
          . The relational masks allow a small share of the {int(S * S)}: column {pct(byLayer.col.density)}, feature{' '}
          {pct(byLayer.feat.density)}, neighbor {pct(byLayer.nbr.density)}. The paper implements the layers “with sparse attention masks, and compiled to efficient FlashAttention-based
          kernels using FlexAttention” <Cite kind="paper" ref="p.5 §3.2" />.
        </p>
        <MaskCounts live={live} />
        <ScaleNote paper={`L = 1024: ${int(1024 * 1024)} pairs per mask`} ours={`L = ${S}: ${int(S * S)} pairs per mask`} />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 6 */}
      <Section id="details" title="Details that matter" kicker="Step 6">
        <h3>The AND with “not padding”</h3>
        <p>
          When the sampler runs out of rows before L, the remaining positions are <Term id="padding">padding</Term>: node id −1, parent list
          [−1, −1, −1, −1, −1], every other field at its default <Cite kind="spec" ref="§4.11" />. Left alone, those −1s would match each other. A
          real row with fewer than five parents has −1 slots in its parent list, so the test “key row ∈ query’s parents” would count every padding
          token as a parent. The code therefore ANDs every mask with <TeX tex="\mathrm{pad}[q,k] = \lnot\,\mathrm{is\_padding}[q] \land \lnot\,\mathrm{is\_padding}[k]" />{' '}
          <Cite kind="code" ref="rt/model.py:165" /> <Cite kind="spec" ref="§6.2" />. The full mask is nothing but <code>pad</code>.
        </p>
        <PaddingDemo />
        <p>
          With the AND in place, a padding query has an all-zero row in every mask and a padding key is never read. Padding inputs are the zero
          vector as well <Cite kind="spec" ref="§5.2" />.
        </p>

        <h3>An empty row gives exactly zero</h3>
        <p>
          Column, feature and full rows always contain at least the query itself, so they are never empty. Only a neighbor row can be: when no row
          in the context points to the query’s row. In {dbShort} that holds for{' '}
          {listAnd([
            ...(ctx.tokens.some((t) => !t.isPadding && t.tableType !== 'Db') ? ['every task row'] : []),
            ...(childless.length ? [`every row of ${listAnd(childless)} (no table has a foreign key to ${childless.length > 1 ? 'them' : 'it'})`] : []),
            ...(parentTables.length
              ? [`any ${listAnd(parentTables.map((n) => singular(n)), 'or')} row whose child rows did not make it into the context`]
              : []),
          ])}
          . A softmax over
          nothing but −∞ would be 0/0. In the pinned PyTorch 2.6 FlexAttention kernel, a fully masked row outputs exactly 0, not NaN and not a
          uniform average <Cite kind="spec" ref="§6.1" />. Because <TeX tex="W_o" /> has no bias, the residual update is 0 and the token passes
          through the sublayer unchanged. Our forward pass reproduces this, and you can check it live:
        </p>
        <ZeroUpdateCheck live={live} />

        <h3>Neighbor is feature’s parent part, transposed</h3>
        <p>
          The feature mask has two parts: “same row” and “the key’s row is one of my parents”. The neighbor mask asks the reverse question: “the
          query’s row is one of the key’s parents” <Cite kind="paper" ref="p.5 §3.2" />. Swapping the roles of query and key is a transpose, so
          neighbor is exactly the parent part of feature, flipped across the diagonal <Cite kind="spec" ref="§6.2" />. Information that flows from
          parent to child in the feature layer flows from child to parent in the neighbor layer.
        </p>
        <TransposeFlip live={live} />

        <h3>The column test also checks the table</h3>
        <p>
          The column mask compares two ids, <code>col_name_idxs</code> and <code>table_name_idxs</code>, where the paper writes a single
          condition. The ids also decide which cells count as one column: the preprocessor interns the table name and the phrase “&lt;col&gt; of
          &lt;table&gt;” once per table, so Train, Val and Test task rows share both ids and the labels of every split meet in a single column
          group <Cite kind="code" ref="rustler/src/pre.rs:424-437" /> <Cite kind="spec" ref="§6.2" />.
        </p>
        <Callout
          variant="paper-vs-code"
          title="The column mask also compares tables"
          paper={
            <>
              <TeX tex="\mathbf{M}^{\text{column}}[q,k]=\mathbf{1}\{\mathrm{Col}(k)=\mathrm{Col}(q)\}" /> <Cite kind="paper" ref="p.5 §3.2" />
            </>
          }
          code={
            <>
              same <code>col_name_idx</code> <strong>and</strong> same <code>table_name_idx</code> <Cite kind="code" ref="rt/model.py:181-183" />
            </>
          }
          verdict="Equivalent. The column id indexes the phrase “<col> of <table>”, which already names the table, so the second test never changes a pair (spec §6.2, §10 #12)."
        >
          <div className="mt-1">
            <ColumnTableCheck live={live} />
          </div>
        </Callout>

        <h3>One mask for all heads</h3>
        <p>
          <code>_make_block_mask</code> calls <code>create_block_mask</code> with <code>H=None</code>, so a layer’s mask is broadcast over every
          head <Cite kind="spec" ref="§6.1" />. Heads differ in which allowed keys they learn to weigh, never in which keys are allowed.
        </p>
        <CodeRef title="_make_block_mask: one mask, every head" file="rt/model.py" lines="89-101" code={MODEL_MAKE_BLOCK_MASK_89_101} highlightLines={[96]} />

        <h3>Built once, used by every block</h3>
        <p>
          Because the masks depend only on ids, the forward pass builds them once, before the first block, converts each to a FlexAttention block
          mask, and passes the same four block masks to all 12 blocks of the paper model (4 blocks here){' '}
          <Cite kind="code" ref="rt/model.py:193-206" /> <Cite kind="code" ref="rt/model.py:226-227" />. Inside a block the layers run in the order{' '}
          <LayerLabel layer="col" /> → <LayerLabel layer="feat" /> → <LayerLabel layer="nbr" /> → <LayerLabel layer="full" />, each picking its own
          mask <Cite kind="code" ref="rt/model.py:82-86" />.
        </p>
        <ScaleNote paper="12 blocks" ours="4 blocks">
          The masks are the same whatever the number of blocks: they are built once from ids and every block reuses them.
        </ScaleNote>
        <CodeRef title="Dense masks become block masks, once per forward pass" file="rt/model.py" lines="193-206" code={MODEL_BLOCK_MASKS_193_206} />
        <CodeRef title="The same block masks go to every block" file="rt/model.py" lines="226-227" code={MODEL_BLOCKS_LOOP_226_227} />
        <CodeRef
          title="RelationalBlock.forward: each layer reads its own mask"
          file="rt/model.py"
          lines="82-86"
          code={MODEL_BLOCK_FWD_82_86}
          highlightLines={[83, 84]}
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 7 */}
      <Section id="worked-example" title="Worked example: the spec’s toy database" kicker="Step 7">
        <p>
          The technical specification prints the masks of a toy ShopDB, three customers, two products, five orders and a churn task table,
          computed by an independent re-implementation <Cite kind="spec" ref="§6.3" />. Our engine reproduces them character for character. Its context, seeded at customer c0’s validation row with L = 24 and w = 1, holds 19 cells and 5
          padding tokens. Step through the query rows; the line under the player explains why each key is allowed.
        </p>
        <WorkedExample />
        <h3>Reading the toy masks</h3>
        <ul>
          <li>
            <strong>Column.</strong> The masked target (row 1) sees the label cells 6, 12 and 18. Two are c0’s own past labels (false, false), one
            belongs to another customer (c2: true). Every timestamp cell (0, 5, 11, 17) sees all task timestamps; <code>order_time</code> sees
            only <code>order_time</code>.
          </li>
          <li>
            <strong>Feature.</strong> The target sees its own row (0, 1) and its parent c0 (2-4). Order o1 (7, 8) sees itself, c0 (2-4) and p1 (9,
            10). Customer cells see only their own row.
          </li>
          <li>
            <strong>Neighbor.</strong> c0’s cells (2-4) see their children in the context: #16 (0, 1), #14 (5, 6), o1 (7, 8) and #12 (11, 12). p1
            (9, 10) sees o1 and o4. Rows 0, 1, 5-8, 11-14, 17 and 18 are empty, so the neighbor sublayer adds 0 to them.
          </li>
          <li>
            <strong>Full.</strong> Rows 0-18 see keys 0-18; the five padding rows see nothing <Cite kind="spec" ref="§6.3" />.
          </li>
        </ul>
      </Section>

      <div className="page-grid">
        <Quiz
          title="Check your understanding"
          questions={[
            {
              prompt: 'In the feature layer, can a customer cell attend to the cells of that customer’s orders?',
              choices: [
                {
                  text: 'Yes: feature attention connects a row with every row it is linked to.',
                  why: 'Feature attention only goes from a row to its parents, the rows its own foreign keys point to. A customer row has no foreign key.',
                },
                {
                  text: 'No. The customer reads its orders in the neighbor layer.',
                  correct: true,
                  why: 'Orders point to the customer, so the customer is in their OutLinks: that is the neighbor mask 1{Row(q) ∈ OutLinks(k)}.',
                },
                {
                  text: 'Only for orders placed before the seed’s timestamp.',
                  why: 'Time never enters a mask. It only decides which rows the sampler puts into the context.',
                },
              ],
              explanation: (
                <>
                  Feature is <TeX tex="\mathbf{1}\{\mathrm{Row}(k)=\mathrm{Row}(q) \lor \mathrm{Row}(k)\in\mathrm{OutLinks}(q)\}" />: a child reads
                  its parents. For a customer cell, OutLinks is empty, so it sees only its own row. The reverse direction, a parent reading its
                  children, is the neighbor mask.
                </>
              ),
            },
            {
              prompt: 'Full attention sees everything. Among the three relational masks, which one lets the masked target read the visible task labels directly?',
              choices: [
                { text: 'Column', correct: true, why: 'All task rows share the label column (all splits share one table id), so the target reads every visible label.' },
                { text: 'Feature', why: 'Feature reaches the target’s own row and its parent customer. Other task rows are siblings, not parents.' },
                { text: 'Neighbor', why: 'No row points to a task row, so the target’s neighbor row is empty.' },
              ],
              explanation: 'The column layer is where the masked label meets the in-context labels: same column, same table, every split.',
            },
            {
              prompt: 'A churn task row has no children in the context. What does the neighbor sublayer add to its cells?',
              choices: [
                { text: 'Exactly zero: they pass through unchanged.', correct: true, why: 'A fully masked row outputs 0 in FlexAttention, and W_o has no bias (spec §6.1).' },
                { text: 'The average of all value vectors.', why: 'That would be a uniform softmax over every key. The kernel returns 0 for a row with no allowed key.' },
                { text: 'NaN, which the loss then ignores.', why: 'The pinned FlexAttention kernel guards this case: no NaN appears.' },
              ],
              explanation: 'Only neighbor rows can be empty, and an empty row means a zero update. You saw it checked live in step 6.',
            },
          ]}
        />
      </div>

      <ChapterNav slug="masks" />
    </ChapterPage>
  );
}

/** Spec §6.2 table: paper notation, code, and what each mask lets flow. */
function MaskDefinitions() {
  const rows: { layer: (typeof LAYER_ORDER)[number]; paper: string; code: string; words: string }[] = [
    {
      layer: 'col',
      paper: String.raw`\mathbf{1}\{\mathrm{Col}(k)=\mathrm{Col}(q)\}`,
      code: 'col[q]==col[k] & table[q]==table[k] & pad',
      words: 'a cell reads the other cells of its column in the same table, task labels of every split included',
    },
    {
      layer: 'feat',
      paper: String.raw`\mathbf{1}\{\mathrm{Row}(k)=\mathrm{Row}(q)\lor\mathrm{Row}(k)\in\mathrm{OutLinks}(q)\}`,
      code: '(node[q]==node[k] | node[k] in f2p[q]) & pad',
      words: 'a cell reads its own row and its parent rows: information flows parent → child, like a join',
    },
    {
      layer: 'nbr',
      paper: String.raw`\mathbf{1}\{\mathrm{Row}(q)\in\mathrm{OutLinks}(k)\}`,
      code: 'node[q] in f2p[k] & pad',
      words: 'a parent’s cells read their child rows: information flows child → parent, like GNN message passing',
    },
    { layer: 'full', paper: '1', code: 'pad', words: 'every non-padding token reads every other one' },
  ];
  return (
    <div className="wide my-7">
      <div className="card overflow-hidden">
        <div className="hidden grid-cols-[150px_minmax(0,1.25fr)_minmax(0,1.1fr)_minmax(0,1.2fr)] gap-4 border-b border-grid bg-surface-2 px-4 py-2 text-[14.5px] font-semibold text-ink-2 md:grid">
          <span>Layer</span>
          <span>
            Paper mask <Cite kind="paper" ref="p.5 §3.2" />
          </span>
          <span>
            Code <Cite kind="code" ref="rt/model.py:165-191" />
          </span>
          <span>
            What it lets flow <Cite kind="spec" ref="§6.2" />
          </span>
        </div>
        {rows.map((r) => (
          <div
            key={r.layer}
            className="grid gap-x-4 gap-y-1.5 border-b border-grid/70 px-4 py-3 last:border-0 md:grid-cols-[150px_minmax(0,1.25fr)_minmax(0,1.1fr)_minmax(0,1.2fr)] md:items-center"
          >
            <span className="text-[17px]">
              <LayerLabel layer={r.layer} />
            </span>
            <span className="min-w-0 overflow-x-auto text-[16.5px]">
              <TeX tex={`\\mathbf{M}[q,k] = ${r.paper}`} />
            </span>
            <code className="!whitespace-normal !break-words text-[15px]">{r.code}</code>
            <span className="text-[16.5px] leading-snug text-ink-2">{r.words}</span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[15px] text-ink-2">
        <code>node</code>, <code>f2p</code>, <code>col</code> and <code>table</code> abbreviate the batch fields <code>node_idxs</code>,{' '}
        <code>f2p_nbr_idxs</code>, <code>col_name_idxs</code> and <code>table_name_idxs</code>; <code>pad[q,k]</code> = neither token is padding.
      </p>
    </div>
  );
}
