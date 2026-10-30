/**
 * Chapter 7 "tokens": Cells become tokens (BUILD_PLAN §6, docs/CHAPTER_BRIEFS.md §7).
 *
 * Every token shown is the engine's (src/engine/sampler.ts, a mirror of rustler/src/fly.rs:357-511)
 * for the live context of the global store, so seed, context and tokens match every other chapter.
 * The two experiments (hidden label, shuffled order) rerun the engine's forward pass
 * (src/engine/model/forward.ts, a mirror of rt/model.py:153-267) on modified token arrays.
 * Spec: research/RT_TECHNICAL_SPEC.md §5.1, §5.3, §4.10-4.13, §6.5.
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
import { JumpBack } from './JumpBack';
import { HiddenLabelDemo, OneMask, TargetVsLabel } from './MaskedLabel';
import { PaddingDemo } from './PaddingDemo';
import { RowTokens } from './RowTokens';
import { ShuffleDemo } from './ShuffleDemo';
import {
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
        The figures follow your simulation settings: <span>{live.db.name}</span> ·{' '}
        <span>{taskShort(live.task)}</span>. The prose is written for the running example, ShopDB’s{' '}
        <span>customer-churn</span>, so a few sentences may not match what you see.
      </p>
      <Button size="sm" variant="secondary" onClick={() => setTask('shop/customer-churn')}>
        Switch the course to the running example
      </Button>
    </Callout>
  );
}

export default function TokensChapter() {
  const live = useTokensLive();
  const { pdb, ctx, S, seed, entityNode, task, assets, tokensByNode, labels } = live;
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
            The sampler hands the model L positions: up to L cells, plus padding if it runs out of rows first. Each cell fills one position of a{' '}
            <strong>batch dictionary</strong>, where it is stored as a few integer ids, a type, a value and some flags. This chapter goes through that
            record field by field, then checks two properties on the live context: exactly one cell is masked, and the order of the cells does not
            matter.
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
            seed <span className="text-ink">{seed ? nodeRef(seed) : '–'}</span>
            {entityNode !== null && (
              <>
                {' '}
                → <span className="text-ink">{nodeLabel(pdb, entityNode)}</span>
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
          The sampler of <Link to="/sampling">chapter 6</Link> writes the context into parallel arrays: one array per field, one slot per position{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/fly.rs:27-44" />.</span> Every slot starts as padding{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/fly.rs:65-85" />,</span> and each emitted cell fills its position in every array{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/fly.rs:462-500" />.</span> The model receives these arrays as tensors{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rt/data.py:116-154" />:</span> <code>[B, S]</code> for ids, types and flags, <code>[B, S, 1]</code> for the
          scalar values, <code>[B, S, 5]</code> for the parent lists and <code>[B, S, 384]</code> for the text embeddings. This dictionary of tensors
          is the <strong>batch dictionary</strong> <span className="whitespace-nowrap"><Cite kind="spec" ref="§5.1" />.</span>
        </p>
        <p>
          B is the number of sequences in a batch (32 per GPU in the paper’s runs) and S is the context length L{' '}
          <span className="whitespace-nowrap"><Cite kind="spec" ref="§4.14" />.</span> One sequence is one context window, so on this page B = 1 and S = {S}. Each position holds one{' '}
          <Term id="cell-token">cell token</Term> or padding. Open any position below to see its record.
        </p>
        <TokenInspector live={live} />
        <p>
          The figure sorts the fields into four groups. The ids say which cell a token is: its row (<code>node_idxs</code>), its parent rows (
          <code>f2p_nbr_idxs</code>), its table and its column (<code>table_name_idxs</code>, <code>col_name_idxs</code>). The four attention masks
          are built from these ids and <code>is_padding</code>, and from nothing else{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:154-191" />.</span> The content fields say what the cell holds: the embedding of its column phrase, its{' '}
          <Term id="sem-type">semantic type</Term> and one value slot per type, all read by the input embedding{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:208-224" />.</span> Three flags mark the masked target and the padding. The last group holds{' '}
          <code>is_task_nodes</code> and <code>class_value_idxs</code>, which the sampler writes and the paper-era model never reads{' '}
          <span className="whitespace-nowrap"><Cite kind="spec" ref="§5.1" />.</span>
        </p>
        <p>
          One name can mislead. The batch field <code>masks</code> marks the single hidden cell (Step 3). The four <em>attention masks</em> of{' '}
          <Link to="/masks">chapter 9</Link> are a different thing: S × S tables of allowed pairs, built from the ids{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:164-191" />.</span> In the figure, the tag “attention masks” marks the fields those tables read.
        </p>
        <p>
          Every token carries four value slots, one per semantic type. The types are number, text, datetime and boolean, numbered 0 to 3 in{' '}
          <code>sem_types</code> <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/common.rs:38-44" />.</span> Each type has its own encoder, normalization, mask vector and decoder{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:115-146" />.</span> Only the slot that matches the token’s type counts: the code computes every type’s
          value term and multiplies the others by 0 <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:216-220" />.</span> The unused slots hold filler: 0 for the
          scalar types and, for <code>text_values</code>, the embedding of the first string in the database’s string table (here “{pdb.text[0]}”){' '}
          <Cite kind="code" ref="rustler/src/pre.rs:553-604" /> <span className="whitespace-nowrap"><Cite kind="spec" ref="§5.1" />.</span>
        </p>
        <CodeRef
          title="The batch fields: one vector per field"
          file="rustler/src/fly.rs"
          lines="27-44"
          code={FLY_VECS_27_44}
          note="One vector per field, with one slot per position of the batch. The last member, true_batch_size, holds a single number for the whole batch."
        />
        <CodeRef
          title="Writing one cell into every field"
          file="rustler/src/fly.rs"
          lines="462-500"
          code={FLY_FIELDS_462_500}
          highlightLines={[462, 466, 469, 470, 482, 490, 496, 500]}
          note="seq_i is the next free position. The row-level fields (node, parents, table) are copied from the node for every one of its cells."
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="cells" title="Rows are not tokens, cells are" kicker="Step 2">
        <p>
          RT samples rows but tokenizes cells: “each database cell is modeled as an individual token”{' '}
          <span className="whitespace-nowrap"><Cite kind="paper" ref="p.4 §3.1" />.</span> Once the sampler selects a row, each of its non-missing feature cells enters the context as its
          own token <span className="whitespace-nowrap"><Cite kind="paper" ref="p.16 App. A" />.</span> Only the last row of a full context can be cut short (Step 5).
          {exampleRow ? (
            <>
              {' '}
              In the live context, <span>{nodeRef(exampleRow)}</span> becomes {plural(tokensByNode.get(exampleRow.nodeIdx)?.length ?? 0, 'token')}.
            </>
          ) : null}
        </p>
        <p>
          Ids tie a row’s tokens back together. Each token repeats its row’s <code>node_idxs</code>, <code>f2p_nbr_idxs</code> and{' '}
          <code>table_name_idxs</code> <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/fly.rs:462-469" />.</span> The feature mask uses the shared node id to let the
          cells of a row read each other, and the parent ids to let them read their parent rows <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:168-173" />.</span> The
          neighbor mask uses the same parent ids in the other direction <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:176-178" />.</span> Without its row’s id and
          parents, a token could not take part in those layers.
        </p>
        <RowTokens live={live} />
        <p>
          The key columns never become tokens. The <Term id="primary-key">primary key</Term> column is skipped{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/pre.rs:439-442" />,</span> and the row’s node id takes its place. Node ids number every row of the
          database in one sequence: each table gets an offset, the number of rows in the tables loaded before it{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/pre.rs:302-305" />,</span> and a row’s node id is that offset plus its position{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/pre.rs:537" />.</span> Keys in RelBench-style data are 0 … n−1 in row order{' '}
          <span className="whitespace-nowrap"><Cite kind="spec" ref="§1.2" />,</span> so a <Term id="foreign-key">foreign key</Term> value plus the parent table’s offset is the parent’s
          node id, and that id goes into <code>f2p_nbr_idxs</code> <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/pre.rs:477-479" />.</span> A missing value
          produces no token at all <span className="whitespace-nowrap"><Cite kind="spec" ref="§3.5" />.</span>
        </p>
        <h3>The parent list</h3>
        <p>
          <code>f2p_nbr_idxs</code> always has five slots, because no row may have more than five parents{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/fly.rs:464" />.</span> The row’s parents fill the first slots in foreign-key column order{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/fly.rs:465-467" />,</span> and the other slots keep −1{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/fly.rs:69" />.</span> The list names all of the row’s parents, whether or not they made it into the
          context. A parent can only change attention if some of its cells are present <span className="whitespace-nowrap"><Cite kind="spec" ref="§4.13" />.</span>
        </p>
        <p>
          In practice the parents are almost always there. The sampler visits a row’s parents, and their parents in turn, before any child row{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:406-408" /> <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/fly.rs:378-379" />.</span> A listed parent ends up
          without tokens only if the context fills up at L first, or if the parent row has no non-missing feature cell to emit{' '}
          <span className="whitespace-nowrap"><Cite kind="spec" ref="§4.16" />.</span> In the live context,{' '}
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
          and copies it into <code>masks</code> <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/fly.rs:492-496" />.</span> Nothing is masked at random, and training
          and evaluation build their sequences the same way <span className="whitespace-nowrap"><Cite kind="spec" ref="§4.10" />.</span> Every other cell is
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
        <p>
          The masked token stays in the sequence. It keeps its row and parent ids, its column phrase and its type, so the model knows which column of
          which table it must fill, and in which row <span className="whitespace-nowrap"><Cite kind="spec" ref="§5.3" />.</span> Only the value term of its input embedding changes: in
          place of the encoded value, the model adds a learned <Term id="mask-vector">mask vector</Term> for the cell’s type{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:221-224" />.</span> <Link to="/embedding">Chapter 8</Link> computes it with real numbers.
        </p>
        <TargetVsLabel live={live} />

        <h3>The label is still in the arrays</h3>
        <p>
          The sampler writes the value of every emitted cell, the target included{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/fly.rs:482-490" />.</span> So the true label of the seed row
          {target ? (
            <>
              {' '}
              ({target.colName} = <span>{String(target.raw)}</span>)
            </>
          ) : null}{' '}
          sits in the target token’s <code>{valueField}</code> slot, in plain sight. The model still cannot read it. The input embedding computes
          this value’s term and then multiplies it by the gate <span className="text-[0.92em] text-ink">[sem = t ∧ ¬mask ∧ ¬pad]</span>, which is 0
          for the target <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:218-219" />.</span> The square brackets are a 0/1 switch: 1 if the condition inside holds, 0
          otherwise. The stored label is used in only two places: the loss takes it as its target{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:240" />,</span> and evaluation reads it as the label for the metrics{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rt/main.py:283-288" />.</span>
        </p>
        <p>
          Try it: replace the stored label and run the model again. The rerun computes every value term from the arrays, the hidden label included,
          and multiplies each one by its gate, as the code does.
        </p>
        <HiddenLabelDemo live={live} />
        <p>
          The prediction stays the same to the last bit, while the loss can change. For a boolean target the loss reads only the sign of the stored
          value, 1{'{'}z &gt; 0{'}'}, so any positive number counts as “true” <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:253-255" />.</span> The masked cell’s
          value is hidden from the model but still there to grade it <span className="whitespace-nowrap"><Cite kind="spec" ref="§5.3" />.</span>
        </p>
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
          <span className="whitespace-nowrap"><Cite kind="paper" ref="p.5 §3.1" />.</span> A table has no first row and no first column, and the paper wants a model that is “invariant to
          permutations of rows, columns, and tables” <span className="whitespace-nowrap"><Cite kind="paper" ref="p.19 App. E" />.</span>
        </p>
        <p>
          Nothing in the code depends on where a token sits. Its input vector is computed from its own fields only (column phrase, type, value, and
          the mask and padding flags), never from its index <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:208-224" />.</span> The four attention masks compare only ids
          and padding flags <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:165-191" />.</span> Shuffle the tokens, and each one gets the same input vector and the same
          set of allowed keys as before, at a new index. Write the shuffle as a permutation matrix P acting on the rows of the token matrix X: the
          masks M become <TeX tex="P\mathbf{M}P^{\top}" />, and the final hidden states H are permuted the same way{' '}
          <span className="whitespace-nowrap"><Cite kind="spec" ref="§6.5" />:</span>
        </p>
        <MathBlock
          tex={PERM_TEX}
          className="text-[18px] sm:text-[19px]"
          caption={<span className="text-[15.5px]">Shuffle the inputs, and the outputs come out shuffled the same way, each one unchanged.</span>}
        />
        <p>
          The hidden states are <Term id="permutation-equivariance">equivariant</Term>: each one moves with its token. The prediction is read at the
          target, wherever it lands, so it is <Term id="permutation-invariance">invariant</Term>. The model cannot tell in which order the sampler
          wrote the cells. Shuffle the live context below and run the model again to see it.
        </p>
        <ShuffleDemo live={live} />
        <Callout variant="insight" title="Then where does the order matter?">
          <p>
            Only inside the sampler, which decides which rows are visited before the context is full and where the cut at L falls{' '}
            <span className="whitespace-nowrap"><Cite kind="spec" ref="§4.12" />.</span> Once the L tokens exist, their order carries no information. The paper credits this symmetry as “a
            crucial driver of sample-efficient generalization from limited schemas in pretraining” <span className="whitespace-nowrap"><Cite kind="paper" ref="p.5 §3.1" />,</span> and contrasts
            it with LLM-based approaches, which “are often highly sensitive to prompt order and formatting”{' '}
            <span className="whitespace-nowrap"><Cite kind="paper" ref="p.19 App. E" />.</span>
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 5 */}
      <Section id="padding" title="Padding" kicker="Step 5">
        <p>
          Every sequence in a batch has the same length, so the sampler allocates all the slots up front and gives each one padding defaults: node
          −1, parents −1, table and column 0, every value 0, <code>is_padding = true</code>{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/fly.rs:65-85" />.</span> Emitting a cell overwrites its slot and sets <code>is_padding = false</code>{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/fly.rs:500" />.</span> A context ends in one of two ways <span className="whitespace-nowrap"><Cite kind="spec" ref="§4.11" />.</span> Either the sampler
          reaches exactly L cells and stops, even in the middle of a row <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/fly.rs:502-509" />,</span> or the frontier runs
          out of rows first <Cite kind="code" ref="rustler/src/fly.rs:387-389" /> and the remaining slots stay <Term id="padding">padding</Term>.
        </p>
        <p>
          So the last row of a full context can lose some of its cells.{' '}
          {cut ? (
            <>
              It happens in the live context: the last row, <span>{nodeRef(pdb.nodes[cut.node])}</span>, kept {cut.emitted} of its {cut.total} cells.
            </>
          ) : ctx.numReal < S ? (
            <>The live context ends in padding, so no row was cut.</>
          ) : (
            <>In the live context the last row happened to fit exactly.</>
          )}
        </p>
        <ScaleNote paper="L = 1024 tokens per context" ours={`L = ${ctx.params.L}`} />
        <PaddingDemo live={live} />
        <p>
          Padding never touches the real tokens. Every attention mask requires that neither token be padding{' '}
          <Cite kind="code" ref="rt/model.py:165" /> <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:186-191" />,</span> and every term of the input embedding is
          multiplied by ¬is_padding <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:208-224" />,</span> so a padding slot enters as 0 and stays 0 through every block{' '}
          <span className="whitespace-nowrap"><Cite kind="spec" ref="§7.2" />.</span> The figure’s last check shows the result: the same cells give the same prediction with or without the
          padding slots around them <span className="whitespace-nowrap"><Cite kind="spec" ref="§6.2" />.</span>
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
                  why: 'Nothing is masked at random: the sampler hides only the seed row’s target cell, in training and in evaluation.',
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
              explanation: 'One masked cell per sequence. Every other cell, the in-context labels included, stays visible (spec §4.10).',
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
                  text: 'A gate in the input embedding multiplies its value term by 0.',
                  correct: true,
                  why: 'model.py:219 keeps a value term only where ~masks; model.py:223 adds the mask vector where masks.',
                },
                {
                  text: 'The sampler overwrites the stored value with 0 before the model runs.',
                  why: 'It does not: the value is written for every emitted cell, the target included (fly.rs:482-490).',
                },
                {
                  text: 'The attention masks stop every query from reading the target.',
                  why: 'The target attends and is attended like any token; it is the target’s own input that hides the value.',
                },
              ],
              explanation:
                'The stored label affects only the loss and the evaluation readout. The input embedding computes its term and multiplies it by 0, which is why the prediction stayed identical when you changed it.',
            },
          ]}
        />
      </div>

      <ChapterNav slug="tokens" />
      <JumpBack live={live} />
    </ChapterPage>
  );
}
