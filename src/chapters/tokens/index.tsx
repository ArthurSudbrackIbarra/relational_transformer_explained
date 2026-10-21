/**
 * Chapter 7 "tokens": Cells become tokens (BUILD_PLAN §6, docs/CHAPTER_BRIEFS.md §7).
 *
 * Every token shown is the engine's (src/engine/sampler.ts, a mirror of rustler/src/fly.rs:357-511)
 * for the live context of the global store, so seed, context and tokens match every other chapter.
 * The two experiments (hidden label, shuffled order) rerun the engine's forward pass
 * (src/engine/model/forward.ts, a mirror of rt/model.py:153-267) on modified token arrays.
 * Spec: research/RT_TECHNICAL_SPEC.md §5.1, §5.3, §4.10-4.13, §6.5, §10 #8, #14, #17.
 */
import { useMemo } from 'react';
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
  TeX,
  Term,
} from '@/components';
import { SemType } from '@/engine/types';
import { useSim } from '@/state/store';
import { cutRow, nodeLabel, nodeRef, plural, taskShort, tsDate, useTokensLive, type TokensLive } from './live';
import { HiddenLabelDemo, OneMask, TargetVsLabel } from './MaskedLabel';
import { PaddingDemo } from './PaddingDemo';
import { RowTokens } from './RowTokens';
import { ShuffleDemo } from './ShuffleDemo';
import {
  DATA_RESHAPE_116_154,
  FLY_FIELDS_462_500,
  FLY_PAD_65_85,
  FLY_STOP_502_509,
  FLY_VECS_27_44,
  MAIN_READOUT_283_288,
  MODEL_EMBED_208_224,
  MODEL_LOSS_238_255,
} from './snippets';
import { TokenInspector } from './TokenInspector';

const PERM_TEX = String.raw`\mathbf{H}\big(P\mathbf{X};\;P\,\mathbf{M}\,P^{\top}\big) \;=\; P\,\mathbf{H}(\mathbf{X};\,\mathbf{M})`;

function RunningExampleNote({ live }: { live: TokensLive }) {
  const setTask = useSim((s) => s.setTask);
  if (live.task.id === 'shop/customer-churn') return null;
  return (
    <Callout variant="note" title="You are viewing another task">
      <p>
        The figures follow your simulation settings: <span className="font-mono">{live.db.name}</span> ·{' '}
        <span className="font-mono">{taskShort(live.task)}</span>. The prose is written for the running example, ShopDB’s{' '}
        <span className="font-mono">customer-churn</span>, so a few sentences may not match what you see.
      </p>
      <Button size="sm" variant="secondary" onClick={() => setTask('shop/customer-churn')}>
        Switch the course to the running example
      </Button>
    </Callout>
  );
}

export default function TokensChapter() {
  const live = useTokensLive();
  const { db, pdb, ctx, S, seed, entityNode, task, assets, tokensByNode, labels } = live;
  const target = ctx.targetPos >= 0 ? ctx.tokens[ctx.targetPos] : null;
  const nRows = tokensByNode.size;
  const cut = useMemo(() => cutRow(pdb, ctx), [pdb, ctx]);
  const missingParents = useMemo(() => {
    let rows = 0;
    for (const [n, positions] of tokensByNode) {
      const t = ctx.tokens[positions[0]];
      if (t.f2pNbrIdxs.some((p) => p >= 0 && !tokensByNode.has(p))) rows++;
      void n;
    }
    return rows;
  }, [ctx, tokensByNode]);
  const exampleRow = useMemo(() => {
    for (const v of ctx.visited) {
      const n = pdb.nodes[v.node];
      if (n && n.f2pNbrIdxs.length >= 2 && (tokensByNode.get(v.node)?.length ?? 0) > 1) return n;
    }
    return null;
  }, [ctx, pdb, tokensByNode]);
  const valueField = target ? (target.sem === SemType.Boolean ? 'boolean_values' : target.sem === SemType.Number ? 'number_values' : 'datetime_values') : 'boolean_values';

  return (
    <ChapterPage slug="tokens">
      <ChapterHeader
        slug="tokens"
        lede={
          <>
            The sampler hands the model L positions: up to L cells, and padding if it runs out of rows first. Each cell travels as one position of a{' '}
            <strong>batch dictionary</strong>: a few integer ids, a type, a value and some flags. This chapter opens that record field by field, then tests two of its promises live: exactly one cell is masked,
            and the order of the cells does not matter.
          </>
        }
        learn={[
          'The per-token fields of the batch dictionary, and which part of the model reads each one',
          'Why a row of k cells becomes k tokens that repeat the row’s id and parent list',
          'How the masked target keeps its true label in the arrays without leaking it into the prediction',
          'Why shuffling the tokens leaves the prediction unchanged, and what padding is',
        ]}
      >
        <div className="mt-6 flex flex-wrap items-center gap-x-2 gap-y-2 text-[17px] leading-relaxed text-ink-2">
          <span className="eyebrow mr-1">Live context</span>
          <span>
            seed <span className="font-mono text-ink">{seed ? nodeRef(seed) : '–'}</span>
            {entityNode !== null && (
              <>
                {' '}
                → <span className="font-mono text-ink">{nodeLabel(pdb, entityNode)}</span>
              </>
            )}
            {seed?.ts != null && <> · {tsDate(seed.ts)}</>}
            {' · '}
            {plural(ctx.numReal, 'cell token')} from {plural(nRows, 'row')} · L = {ctx.params.L}, w = {ctx.params.w}
          </span>
          <ScaleNote inline />
          {assets.status === 'ready' && assets.weightsKind === 'trained' && <StatusBadge status="good">trained weights: {assets.ckpt}</StatusBadge>}
          {assets.status === 'ready' && assets.weightsKind === 'init' && <StatusBadge status="warning">untrained weights</StatusBadge>}
          {assets.status === 'loading' && <StatusBadge status="info">loading model…</StatusBadge>}
        </div>
      </ChapterHeader>

      <div className="page-grid">
        <RunningExampleNote live={live} />
      </div>

      {/* ───────────────────────────────────────────────────────────── 1 */}
      <Section id="batch-dict" title="The batch dictionary" kicker="Step 1">
        <p>
          The sampler of <Link to="/sampling">chapter 6</Link> does not hand the model a table or a graph. It writes the context into parallel arrays, one slot per position.
          In <span className="font-mono">fly.rs</span> a struct holds one vector per field <Cite kind="code" ref="rustler/src/fly.rs:27-44" />, every
          slot starts as padding <Cite kind="code" ref="rustler/src/fly.rs:65-85" />, and each emitted cell fills position <code>seq_i</code> of
          every field <Cite kind="code" ref="rustler/src/fly.rs:462-500" />. On the Python side, <span className="font-mono">data.py</span> turns
          the buffers into tensors of shape <code>[B, S]</code> (one number per token), <code>[B, S, 5]</code> (the parent lists) or{' '}
          <code>[B, S, 384]</code> (embeddings) <Cite kind="code" ref="rt/data.py:116-154" />. That dictionary of tensors is the{' '}
          <strong>batch dictionary</strong> <Cite kind="spec" ref="§5.1" />.
        </p>
        <p>
          B counts sequences: 32 per GPU in the paper’s runs <Cite kind="spec" ref="§4.14" />. S is the context length L. One sequence is one
          context window, so on this page B = 1 and S = {S}. Each position holds one <Term id="cell-token">cell token</Term>, or padding.
        </p>
        <p>
          The fields fall into four groups, and the figure below follows them. The <strong>ids</strong> say which cell a token is: which row (
          <code>node_idxs</code>), which parent rows (<code>f2p_nbr_idxs</code>), which table and which column (<code>table_name_idxs</code>,{' '}
          <code>col_name_idxs</code>). The four attention masks are built from these ids and from <code>is_padding</code>, and from nothing else{' '}
          <Cite kind="code" ref="rt/model.py:154-191" />. The <strong>content</strong> fields say what the cell holds: the embedding of its column
          phrase, its <Term id="sem-type">semantic type</Term> and one value slot per type. The input embedding reads them{' '}
          <Cite kind="code" ref="rt/model.py:208-224" />. Three <strong>flags</strong> mark the masked target and the padding. Two fields ride along
          unused: <code>is_task_nodes</code> and <code>class_value_idxs</code> are written by the sampler but never read by the paper-era model{' '}
          <Cite kind="spec" ref="§5.1" />.
        </p>
        <p>
          Watch out for one name clash. The batch field <code>masks</code> marks the one hidden cell (Step 3). The four{' '}
          <em>attention masks</em> of <Link to="/masks">chapter 9</Link> are something else: S × S tables of allowed pairs, built from the ids{' '}
          <Cite kind="code" ref="rt/model.py:164-191" />. The figure tags a field “attention masks” when those tables read it.
        </p>
        <TokenInspector live={live} />
        <p>
          Every token carries all four value slots, but only the one that matches its <code>sem_types</code> counts. The code computes every type’s
          value term and multiplies the others by 0 <Cite kind="code" ref="rt/model.py:216-220" />. The unused number, datetime and boolean slots hold
          0, and <code>text_values</code> of a non-text cell holds the embedding of <code>text[0]</code>, whatever string has index 0 in the
          database’s string table (here “{pdb.text[0]}”) <Cite kind="code" ref="rustler/src/pre.rs:553-604" /> <Cite kind="spec" ref="§5.1" />.
        </p>
        <Callout
          variant="paper-vs-code"
          title="Is boolean its own type?"
          paper={
            <>
              The problem setup defines feature columns as those that “contain numeric, text, and datetime information”{' '}
              <Cite kind="paper" ref="p.3 §2.1" />. The cell encoding then says “The value v can be numeric, boolean, datetime, or text”, but gives
              numeric and boolean values one shared normalization rule, “Numeric/boolean” <Cite kind="paper" ref="p.4 §3.1" />. The decoders section
              ties binary classification to “the boolean datatype” <Cite kind="paper" ref="p.6 §3.3" />.
            </>
          }
          code={
            <>
              Boolean is a separate sem type: Number = 0, Text = 1, DateTime = 2, Boolean = 3{' '}
              <Cite kind="code" ref="rustler/src/common.rs:38-44" />. Each of the four has its own encoder, RMSNorm, mask vector and decoder{' '}
              <Cite kind="code" ref="rt/model.py:115-146" />.
            </>
          }
          verdict="This site shows the code’s four types (spec §10 #17)."
        />
        <ScaleNote paper="values and embeddings stored as bfloat16" ours="float32">
          The sampler rounds every scalar value to bfloat16 as it writes it <Cite kind="code" ref="rustler/src/fly.rs:482-490" />, and the MiniLM
          embeddings are stored as bfloat16 to begin with <Cite kind="code" ref="rt/embed.py:71" /> <Cite kind="code" ref="rustler/src/fly.rs:528-532" />.
          This site keeps float32. Ids and flags are identical.
        </ScaleNote>
        <CodeRef
          title="The batch fields: one vector per field"
          file="rustler/src/fly.rs"
          lines="27-44"
          code={FLY_VECS_27_44}
          note="true_batch_size is per batch, not per token: it counts the real sequences of a wrapped last batch and is used only in evaluation (spec §4.14)."
        />
        <CodeRef
          title="Writing one cell into every field"
          file="rustler/src/fly.rs"
          lines="462-500"
          code={FLY_FIELDS_462_500}
          highlightLines={[462, 466, 469, 470, 482, 490, 496, 500]}
          note="seq_i is the next free position. The row-level fields (node, parents, table) are copied from the node for every one of its cells."
        />
        <CodeRef
          title="From Rust buffers to the batch dictionary"
          file="rt/data.py"
          lines="116-154"
          code={DATA_RESHAPE_116_154}
          highlightLines={[133, 143, 149, 150]}
          note="Every field is reshaped to [B, S], [B, S, 5], [B, S, 1] or [B, S, 384]. The value fields arrive as raw bf16 bytes."
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="cells" title="Rows are not tokens, cells are" kicker="Step 2">
        <p>
          RT samples rows but tokenizes cells. The paper calls this cell-level tokenization: “each database cell is modeled as an individual token”{' '}
          <Cite kind="paper" ref="p.4 §3.1" />. Rows remain the sampling unit: once a row is selected, its non-missing feature cells enter the
          context <Cite kind="paper" ref="p.16 App. A" />.
          {exampleRow ? (
            <>
              {' '}
              In the live context, <span className="font-mono">{nodeRef(exampleRow)}</span> becomes {plural(tokensByNode.get(exampleRow.nodeIdx)?.length ?? 0, 'token')}.
            </>
          ) : null}
        </p>
        <p>
          What ties k separate tokens back into one row? Ids. Each token repeats its row’s <code>node_idxs</code>, <code>f2p_nbr_idxs</code> and{' '}
          <code>table_name_idxs</code> <Cite kind="code" ref="rustler/src/fly.rs:462-469" />. The feature mask uses the shared node id to let the
          cells of a row read each other, and the parent ids to let them read their parent rows <Cite kind="code" ref="rt/model.py:168-173" />. The
          neighbor mask uses the same parent ids in the other direction <Cite kind="code" ref="rt/model.py:176-178" />. A token that did not carry
          its row’s id and parents could not take part in those layers.
        </p>
        <RowTokens live={live} />
        <p>
          The key columns never become tokens. The <Term id="primary-key">primary key</Term> is the row’s position, and it survives only as the node
          id. Each <Term id="foreign-key">foreign key</Term> becomes a parent id in <code>f2p_nbr_idxs</code> and a link back from the parent{' '}
          <Cite kind="spec" ref="§1.2" />. A missing value produces no token at all <Cite kind="spec" ref="§3.5" />.
        </p>
        <h3>The parent list</h3>
        <p>
          <code>f2p_nbr_idxs</code> always has five slots. The sampler asserts that no row has more than <code>MAX_F2P_NBRS = 5</code> parents and
          fills the unused slots with −1 <Cite kind="code" ref="rustler/src/fly.rs:464-467" /> <Cite kind="code" ref="rustler/src/fly.rs:69" />. The
          list holds all of the row’s parents, in foreign-key column order, whether or not they made it into the context. A parent can only change
          attention if some of its cells are present <Cite kind="spec" ref="§4.13" />.
        </p>
        <p>
          In practice the parents are almost always there. When the sampler visits a row it pushes the row’s parents on its F→P stack{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:406-408" />, which has priority over everything else{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:378-379" />. So they, and their own parents in turn, are visited
          before any child row. The stack is last in, first out, so the parent of the last foreign key comes first <Cite kind="spec" ref="§4.4" />.
          A listed parent ends up without tokens only if the hard
          stop at L comes first, or if the parent row has no non-null feature cell to emit <Cite kind="spec" ref="§4.16" />. In the live context,{' '}
          {missingParents === 0
            ? 'every parent listed by any token is in the context.'
            : `${plural(missingParents, 'row')} ${missingParents === 1 ? 'lists' : 'list'} a parent that did not make it in.`}
        </p>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 3 */}
      <Section id="masked" title="Exactly one masked cell" kicker="Step 3">
        <p>
          Exactly one token of a context has <code>masks = true</code>: the <Term id="target-cell">target cell</Term> of the{' '}
          <Term id="seed-row">seed row</Term>. The sampler sets <code>is_target</code> when the row is the seed and the column is the target column,
          and copies it into <code>masks</code> <Cite kind="code" ref="rustler/src/fly.rs:492-496" />. There is no random masking anywhere in the
          paper-era sampler or model; training and evaluation build sequences the same way <Cite kind="spec" ref="§4.10" />. Every other cell is
          visible, including other rows’ values of the target column: the <Term id="in-context-label">in-context labels</Term> of{' '}
          <Link to="/sampling">chapter 6</Link>
          {task.kind === 'forecast' ? (
            <>
              {' '}
              (here {labels.self.length ? plural(labels.self.length, 'label') : 'no labels'} of the seed’s own entity and{' '}
              {labels.other.length ? plural(labels.other.length, 'label') : 'none'} of other entities)
            </>
          ) : null}
          .
        </p>
        <OneMask live={live} />
        <Callout
          variant="paper-vs-code"
          title="Masked token prediction with more than one masked cell"
          paper={
            <>
              Table 14 pretrains with multi-cell masking: “The target cell is always masked. Other cells are masked with probability P(mask)”{' '}
              <Cite kind="paper" ref="p.24 Table 14" />. It is an ablation; the main results use P(mask) = 0 <Cite kind="spec" ref="§11.6" />.
            </>
          }
          code={
            <>
              Only the seed’s target cell is masked <Cite kind="code" ref="rustler/src/fly.rs:492-496" />. The P(mask) sampler is not in the released
              code <Cite kind="spec" ref="§10" />.
            </>
          }
          verdict="This site masks only the target cell, as the released code does (spec §10 #14)."
        />
        <p>
          The masked token is not removed and not blanked. It keeps its row and parent ids, its column phrase and its type, so the model knows which
          column of which table it must fill, and in which row <Cite kind="spec" ref="§5.3" />. Only the value term of its input embedding changes:
          instead of encoding the value, the model adds a learned <Term id="mask-vector">mask vector</Term> for the cell’s type{' '}
          <Cite kind="code" ref="rt/model.py:221-224" />. <Link to="/embedding">Chapter 8</Link> computes it with real numbers.
        </p>
        <TargetVsLabel live={live} />

        <h3>The label is still in the arrays</h3>
        <p>
          Here is a detail that surprises most readers. The sampler writes the value of every emitted cell, the target included{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:482-490" />. The true label of the seed row
          {target ? (
            <>
              {' '}
              ({target.colName} = <span className="font-mono">{String(target.raw)}</span>)
            </>
          ) : null}{' '}
          therefore sits in the target token’s <code>{valueField}</code> slot, in plain sight. It cannot leak: the input embedding multiplies that
          value’s term by the gate <span className="font-mono text-[0.92em] text-ink">[sem = t ∧ ¬mask ∧ ¬pad]</span>, which is 0 for the target{' '}
          <Cite kind="code" ref="rt/model.py:219" />. The square brackets are a 0/1 switch: 1 if the condition inside holds, 0 otherwise. The stored
          label has two readers, and neither is the input: the loss uses it as its target <Cite kind="code" ref="rt/model.py:240" />, and the
          evaluation uses it as the label for the metrics <Cite kind="code" ref="rt/main.py:283-288" />.
        </p>
        <p>Try it. Replace the stored label and run the model again.</p>
        <HiddenLabelDemo live={live} />
        <p>
          The prediction does not move by a single bit, while the loss can. For a boolean target the loss reads only the sign of the stored value,
          1{'{'}z &gt; 0{'}'}, so any positive number gives the same loss as “true” <Cite kind="code" ref="rt/model.py:253-255" />. That is the
          contract of the masked cell: its value is invisible to the model, yet available to grade it <Cite kind="spec" ref="§5.3" />.
        </p>
        <Callout variant="warning" title="Finite values only">
          <p>
            The gate is a multiplication, not a branch. The code computes the value term of every type at every position and then multiplies by 0{' '}
            <Cite kind="code" ref="rt/model.py:216-220" />. For a finite value 0 · v = 0, but a NaN or an infinity would survive as NaN · 0 = NaN and
            spread through attention to every output <Cite kind="spec" ref="§5.3" />. The engine on this site refuses such inputs rather than return a
            number the official code would not produce.
          </p>
        </Callout>
        <CodeRef
          title="The input embedding: the value term is gated by ~masks"
          file="rt/model.py"
          lines="208-224"
          code={MODEL_EMBED_208_224}
          highlightLines={[219, 223]}
          note="Line 219 keeps a value term only where the token is not masked and not padding. Line 223 adds the mask vector exactly where it is masked."
        />
        <CodeRef
          title="Where the stored label is read: the loss"
          file="rt/model.py"
          lines="238-255"
          code={MODEL_LOSS_238_255}
          highlightLines={[240, 241]}
          note="y is the same value array the input gate ignored; sem_type_mask keeps only the masked position."
        />
        <CodeRef
          title="Evaluation reads prediction and label at is_targets"
          file="rt/main.py"
          lines="283-288"
          code={MAIN_READOUT_283_288}
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="order" title="No positional encodings" kicker="Step 4">
        <p>
          A language model must tell “dog bites man” from “man bites dog”, so it adds positional encodings to its tokens. RT leaves them out on
          purpose: “To preserve the natural symmetries of relational data, RT does not incorporate positional encodings”{' '}
          <Cite kind="paper" ref="p.5 §3.1" />. A table has no first row and no first column, and the paper wants a model that is “invariant to
          permutations of rows, columns, and tables” <Cite kind="paper" ref="p.19 App. E" />.
        </p>
        <p>
          Nothing in the code depends on where a token sits. Its input vector is computed from its own fields only: column phrase, type, value, and
          the mask and padding flags, never its index <Cite kind="code" ref="rt/model.py:208-224" />. The four attention masks compare ids and padding
          flags only <Cite kind="code" ref="rt/model.py:165-191" />. Shuffle the tokens and each one gets the same input vector and the same set of
          allowed keys as before, at a new index. Write the shuffle as a permutation matrix P acting on the rows of the token matrix X; the masks M
          become <TeX tex="P\mathbf{M}P^{\top}" />, and the final hidden states H are permuted the same way <Cite kind="spec" ref="§6.5" />:
        </p>
        <MathBlock
          tex={PERM_TEX}
          className="text-[15.5px] sm:text-[19px]"
          caption={<span className="text-[15.5px]">Shuffle the inputs, and the outputs come out shuffled the same way, each one unchanged.</span>}
        />
        <p>
          The hidden states are <Term id="permutation-equivariance">equivariant</Term>: each one moves with its token. The prediction is read at the
          target, wherever it lands, so it is <Term id="permutation-invariance">invariant</Term>. That invariance is what the paper means when it
          says RT has no positional encodings. In exact arithmetic the equality is exact. In floating point, a different order of the keys can change
          the order of the sums and so the last bits; the official model runs FlexAttention <Cite kind="code" ref="rt/model.py:17" />{' '}
          <Cite kind="code" ref="rt/model.py:43" /> in bfloat16 <Cite kind="code" ref="rt/main.py:187" />, where equality holds up to rounding. Do not take it on faith: shuffle the live context and
          run the model again.
        </p>
        <ShuffleDemo live={live} />
        <Callout variant="insight" title="Then where does the order matter?">
          <p>
            Only inside the sampler. The BFS decides which rows are visited before the context is full, and where the cut at L falls{' '}
            <Cite kind="spec" ref="§4.12" />. Once the L tokens exist, their order carries no information. The paper credits this symmetry as “a crucial
            driver of sample-efficient generalization from limited schemas in pretraining” <Cite kind="paper" ref="p.5 §3.1" />, and contrasts it with
            LLM-based approaches, which “are often highly sensitive to prompt order and formatting” <Cite kind="paper" ref="p.19 App. E" />.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 5 */}
      <Section id="padding" title="Padding" kicker="Step 5">
        <p>
          Every sequence of a batch has the same length, so the sampler allocates batch_size × seq_len slots up front and gives each one a padding
          default: node −1, parents −1, table and column 0, every value 0, <code>is_padding = true</code>{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:65-85" />. Emitting a cell overwrites its slot and sets <code>is_padding = false</code>{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:500" />. A context ends in one of two ways <Cite kind="spec" ref="§4.11" />. Either the sampler
          reaches exactly L cells and stops, even in the middle of a row <Cite kind="code" ref="rustler/src/fly.rs:502-509" />, or the frontier runs
          out of rows first <Cite kind="code" ref="rustler/src/fly.rs:387-389" /> and the remaining slots stay <Term id="padding">padding</Term>.
        </p>
        <Callout
          variant="paper-vs-code"
          title="A row can be cut at L"
          paper={
            <>
              “Once a row is selected, all of its non-missing feature cells are added to the context” <Cite kind="paper" ref="p.4 §3.1" />, and the
              traversal “stops when the total number of cells reaches the context length” <Cite kind="paper" ref="p.16 App. A" />.
            </>
          }
          code={
            <>
              The loop stops after exactly L cells, even in the middle of a row <Cite kind="code" ref="rustler/src/fly.rs:502-509" />.
            </>
          }
          verdict="This site follows the code (spec §10 #8)."
        >
          <p>
            {cut ? (
              <>
                It happens in the live context: the last row, <span className="font-mono">{nodeRef(pdb.nodes[cut.node])}</span>, kept {cut.emitted} of
                its {cut.total} cells.
              </>
            ) : ctx.numReal < S ? (
              <>The live context ends in padding, so no row was cut.</>
            ) : (
              <>In the live context the last row happened to fit exactly.</>
            )}
          </p>
        </Callout>
        <ScaleNote paper="L = 1024 tokens per context" ours={`L = ${ctx.params.L}`} />
        <PaddingDemo live={live} />
        <p>
          Padding is inert by construction. Every mask is ANDed with “neither token is padding” <Cite kind="code" ref="rt/model.py:165" />, and
          every term of the input embedding is multiplied by ¬is_padding <Cite kind="code" ref="rt/model.py:208-224" />. The last check in the figure
          is the consequence that matters: the same cells give the same prediction with or without the padding slots around them{' '}
          <Cite kind="spec" ref="§7.2" />.
        </p>
        <CodeRef
          title="Vecs::new: every slot starts as padding"
          file="rustler/src/fly.rs"
          lines="65-85"
          code={FLY_PAD_65_85}
          highlightLines={[68, 69, 82]}
        />
        <CodeRef title="The hard stop at L, even mid-row" file="rustler/src/fly.rs" lines="502-509" code={FLY_STOP_502_509} />
      </Section>

      <div className="page-grid">
        <Quiz
          title="Check your understanding"
          questions={[
            {
              prompt: 'How many tokens of one context have masks = true?',
              choices: [
                {
                  text: 'About 15% of them, chosen at random as in BERT.',
                  why: 'There is no random masking in the paper-era code. Table 14’s multi-cell masking is an ablation that was not released.',
                },
                {
                  text: 'One per in-context label.',
                  why: 'The in-context labels are visible on purpose: they are what the model reads to make its prediction.',
                },
                {
                  text: 'Exactly one: the target cell of the seed row.',
                  correct: true,
                  why: 'is_target = (row is the seed) ∧ (column is the target column), and masks = is_target (fly.rs:492-496).',
                },
              ],
              explanation: 'One masked cell per sequence, in training and in evaluation alike (spec §4.10).',
            },
            {
              prompt: 'You shuffle the order of the tokens and run the model again. What happens to the prediction?',
              choices: [
                {
                  text: 'It changes a little, because the positional encodings shift.',
                  why: 'RT has no positional encodings (paper p.5 §3.1).',
                },
                {
                  text: 'Nothing: the same value comes out, read at the target’s new position.',
                  correct: true,
                  why: 'Each input depends only on its own token’s fields and the masks only on ids and padding flags, so every output is permuted along with its token.',
                },
                {
                  text: 'It changes, because the masks are built from positions.',
                  why: 'The masks compare node, parent, table and column ids and the padding flags, never positions (model.py:165-191).',
                },
              ],
              explanation: 'Order matters only inside the sampler, which decides which rows fit before L (spec §4.12, §6.5).',
            },
            {
              prompt: 'The target token still stores the true label in its value slot. Why can the model not use it?',
              choices: [
                {
                  text: 'The input embedding multiplies the value term by a gate that is 0 when masks is true, and adds the mask vector instead.',
                  correct: true,
                  why: 'model.py:219 keeps a value term only where ~masks; model.py:223 adds m_t where masks.',
                },
                {
                  text: 'The sampler overwrites it with 0 before the model runs.',
                  why: 'It does not: the value is written for every emitted cell, the target included (fly.rs:482-490).',
                },
                {
                  text: 'Attention cannot reach the target position.',
                  why: 'The target attends and is attended like any token; it is the target’s own input that hides the value.',
                },
              ],
              explanation: 'The stored label is read only by the loss and the evaluation readout. You saw the prediction stay identical when you changed it.',
            },
          ]}
        />
      </div>

      <ChapterNav slug="tokens" />
    </ChapterPage>
  );
}
