/**
 * Chapter 14 "playground": Zero-shot playground and ablations (BUILD_PLAN §6, docs/CHAPTER_BRIEFS.md §14).
 *
 * The seed, context and checkpoint are the course's live state (src/state); every ablation is an
 * inference-time change to the INPUT of the engine's forward pass (ablate.ts), computed in the
 * browser. Spec: research/RT_TECHNICAL_SPEC.md §9.4, §11.2-11.5, §13 #3.
 */
import {
  Callout,
  ChapterHeader,
  ChapterNav,
  ChapterPage,
  Cite,
  CodeRef,
  Quiz,
  ScaleNote,
  Section,
  StatusBadge,
  Term,
  formatDate,
} from '@/components';
import { MINI_CONFIG, paramCount } from '@/engine/model';
import { AttentionShare } from './AttentionShare';
import { BatchMode } from './BatchMode';
import { nodeTitle, PlayProvider, usePlayLive } from './context';
import { CkptCompare, LabelEffects, LayerEffects, LSweepFigure, ShuffleSpread } from './Experiments';
import { CkptMetrics, LabelCounts, LabelEncoding, NamesMap } from './Extras';
import { DRIVERS, FIG4, HEADLINE, LAYER_ABLATION, TABLE10, TABLE4 } from './paper';
import { Playground } from './Playground';
import { FLY_COLNAME_469_477, FLY_TARGET_492_496, MAIN_METRICS_328_337, MODEL_BLOCK_FWD_82_86, MODEL_COLNAME_TERM_208_214 } from './snippets';
import { targetPhrase } from './usePlayground';

export default function PlaygroundChapter() {
  return (
    <PlayProvider>
      <Body />
    </PlayProvider>
  );
}

function f1(v: number): string {
  return (v < 0 ? '−' : '') + Math.abs(v).toFixed(1);
}

function Body() {
  const live = usePlayLive();
  const { db, pdb, task, seedNode, globalL, w, baseCkpt, baseAssets, abl, effL } = live;
  const seed = pdb.nodes[seedNode];
  const who = nodeTitle(db, pdb, seedNode);
  const forecast = task.kind === 'forecast';
  const nSelf = abl.labels.self.length;
  const nOther = abl.labels.other.length;
  const phrase = targetPhrase(task);
  const cfg = baseAssets.weights?.config ?? MINI_CONFIG;
  const nBlocks = cfg.nBlocks;
  const nParams = paramCount(cfg);
  const isShop = db.name === 'shop';

  return (
    <ChapterPage slug="playground">
      <ChapterHeader
        slug="playground"
        lede={
          <>
            So far every chapter ran one fixed forward pass. Here you take the controls: pick any ShopDB seed, change what the model sees, and watch
            its zero-shot prediction move. Each switch mirrors one of the paper’s ablations, run live on the miniature model.
          </>
        }
        learn={[
          'How to run the zero-shot model on any ShopDB seed and read its prediction',
          'What happens when in-context labels are edited or removed (paper Table 4)',
          'How much the column names and the context length matter',
          'Why switching off a trained attention layer is not the paper’s layer ablation',
        ]}
      >
        <p className="mt-6 text-[17px] leading-relaxed text-ink-2">
          <span className="eyebrow mr-2">Live seed</span>
          <span className="text-ink">{who.title}</span>
          {who.sub ? ` · ${who.sub}` : ''}
          {seed && seed.ts !== null && !who.sub.includes(formatDate(seed.ts * 1000)) ? ` · ${formatDate(seed.ts * 1000)}` : ''} · task{' '}
          <span className="text-ink">{task.name}</span> · checkpoint <span className="font-mono text-ink">{baseCkpt}</span> · L = {globalL}, w ={' '}
          {w}{' '}
          {baseAssets.weightsKind === 'init' && (
            <>
              <StatusBadge status="warning">untrained weights</StatusBadge>{' '}
            </>
          )}
          <ScaleNote inline />
        </p>
      </ChapterHeader>

      {/* ───────────────────────────────────────────────────────────── 1 */}
      <Section id="playground" title="The playground" kicker="Try it">
        <p>
          The model on this page has never seen a ShopDB row. It was pretrained on the other three databases only, and its checkpoint was chosen
          on this task’s validation split, as the paper does <Cite kind="spec" ref="§9.1" />. Everything it knows about ShopDB arrives through the
          context window. That is what <Term id="zero-shot">zero-shot</Term> means here: “information about the new RDB is available only at
          inference time, solely through the context window” <Cite kind="paper" ref="p.4 §2.3" />.
        </p>
        <p>
          The first panel picks the <Term id="seed-row">seed row</Term>. That choice is shared with the whole course, so the other chapters follow
          it. The prediction panel shows what the model predicts at the <Term id="target-cell">masked cell</Term> and the rows it sees; the
          ablation panel holds the switches. The <strong>baseline</strong> is the course’s live context and checkpoint with every layer on.
          “With your changes” applies all the switches at once, and the paper’s number for each switch you turn on appears under the prediction.
        </p>
        <Playground />
        {!isShop && (
          <Callout variant="warning" title="Not a zero-shot setting">
            <p>
              {db.title} was one of the pretraining databases, so predictions on it are not zero-shot. Pick a ShopDB task in the left panel to see
              transfer to an unseen database.
            </p>
          </Callout>
        )}
        <p>
          Every prediction on this page comes from a forward pass that runs in your browser on the same engine as every other chapter. Only the
          frozen text embeddings, the trained weights and our training log (Experiment 5) are precomputed.
        </p>
        <ScaleNote paper="d = 256, 12 blocks, L = 1024, w = 256, trained on RelBench" ours={`d = ${cfg.dModel}, ${nBlocks} blocks, L = ${globalL}, w = ${w}, trained on 3 synthetic DBs`}>
          The ablations change the model’s input or skip a sublayer; the operations of the forward pass are the paper-era code’s. Predictions of a{' '}
          {nParams.toLocaleString('en-US')}-parameter model on a toy database show how the mechanism behaves, not how large the paper’s effects are.
        </ScaleNote>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="labels" title="Past labels carry most of the signal" kicker="Experiment 1">
        <p>
          The context of a forecasting seed holds more than the entity’s orders and reviews. It also holds earlier rows of the task table, with
          their labels visible: <Term id="in-context-label">in-context labels</Term>. Some belong to the same customer (
          <Term id="self-label">self labels</Term>), others to other customers the sampler reached, including{' '}
          <Term id="same-horizon">same-horizon</Term> rows <Cite kind="spec" ref="§2.5" />. Only the seed’s own label is masked{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:492-496" />.
        </p>
        <p>
          {forecast ? (
            <>
              For the live seed at L = {effL}, the context holds <strong>{nSelf}</strong> self label{nSelf === 1 ? '' : 's'} and{' '}
              <strong>{nOther}</strong> other-entity label{nOther === 1 ? '' : 's'}.
            </>
          ) : (
            <>
              The live task is an autocomplete task, so there are no task rows: its in-context labels are the target column of the other rows in the
              context ({nOther} at L = {effL}). Pick a forecasting task to try the self/other split.
            </>
          )}
        </p>
        <LabelCounts />
        <p>
          In the paper this is the strongest effect of all. Removing the self labels drops the mean zero-shot AUROC from {f1(TABLE4.none.zsClf)} to{' '}
          {f1(TABLE4.selfLabels.zsClf)}, close to the random baseline of 50, and the mean R² from {f1(TABLE4.none.zsReg)} to{' '}
          {f1(TABLE4.selfLabels.zsReg)}, worse than always predicting the global mean (R² = 0). Removing the other entities’ labels barely moves
          them ({f1(TABLE4.otherLabels.zsClf)} and {f1(TABLE4.otherLabels.zsReg)}) <Cite kind="paper" ref="p.9 Table 4" />. In the authors’ words, “zero-shot transfer is driven primarily by the model’s ability to
          leverage past labels of the target entity” <Cite kind="paper" ref="p.9 §5.1" />.
        </p>
        <Callout variant="note" title="How the removal works here">
          <p>
            The paper applies its zero-shot context ablations “directly to the sampled context used as input” <Cite kind="paper" ref="p.22 App. G" />
            , but the code for them is not released <Cite kind="spec" ref="§13 #3" />. We drop the label tokens from the token list before the forward
            pass and do not refill the context. Dropping a token is the same as turning it into padding, because padding is removed from every mask
            on both sides <Cite kind="code" ref="rt/model.py:165" /> <Cite kind="code" ref="rt/model.py:186-191" />.
          </p>
          <p>
            The paper words it two ways. §5.1 says “removing past task rows which refer to the target entity”, while Table 11 and Table 12 speak of
            labels, “unmasked cells from the target column” <Cite kind="paper" ref="p.9 §5.1" /> <Cite kind="paper" ref="p.23 Table 12" />. The
            switch “What is removed” lets you try both: only the label cells, or every cell of those task rows.
          </p>
        </Callout>
        <h3>Edit a label</h3>
        <p>
          Click a label in the playground to flip it, or type a new number for a regression task. The new value is stored exactly the way the
          preprocessing stored every label of this task: z-scored with the target column’s Train statistics <Cite kind="code" ref="rustler/src/pre.rs:553-569" />{' '}
          <Cite kind="spec" ref="§3.3" />. Val and Test task rows reuse the Train statistics <Cite kind="code" ref="rustler/src/pre.rs:383-396" />, so
          one formula covers every visible label:
        </p>
        <LabelEncoding />
        <p>
          The masked cell itself cannot be edited, and changing it would not matter: its value is still in the token array but gated out of the
          input <Cite kind="code" ref="rt/model.py:219" /> (chapter 7). The sampler marks exactly one cell of the context as masked:
        </p>
        <CodeRef
          title="The one masked cell: the seed row’s target column"
          file="rustler/src/fly.rs"
          lines="492-496"
          code={FLY_TARGET_492_496}
          note="Every other cell of the context, labels included, is an ordinary visible token."
        />
        <p>
          The figure below takes one label at a time, changes it, and reruns the model, starting each time from the full live context. It shows
          which past labels this model leans on for this seed. Try a few seeds: does the newest label move the prediction more than the old ones?
        </p>
        <LabelEffects />
        <h3>How a label reaches the masked cell</h3>
        <p>
          In <Term id="column-attention">column attention</Term> the masked cell can attend only to cells of its own column in its own table{' '}
          <Cite kind="code" ref="rt/model.py:181-189" />. For a forecasting seed those are exactly the labels and itself. This is the most direct
          path from a past label to the prediction (an interpretation, not a claim of the paper; spec §6.4). The chart below reads the attention
          weights of the masked cell from the live forward passes. In the playground, switch on “Self labels” under “Remove in-context labels” and watch the column bars.
        </p>
        <AttentionShare />
        <Callout variant="insight" title="More than an average">
          <p>
            If the model only averaged the self labels, it would equal the <Term id="entity-mean">EntityMean</Term> baseline. In the paper RT beats
            EntityMean on average, so it learns “more complex functions than averaging self labels” <Cite kind="paper" ref="p.9 §5.1" />{' '}
            <Cite kind="paper" ref="p.8 Table 1" />. Fig. 5 fine-tunes with the self labels removed and still finds positive transfer from
            pretraining, although “self labels do constitute the dominant share of transfer” <Cite kind="paper" ref="p.9 §5.1" /> <Cite kind="paper" ref="p.9 Fig. 5" />.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 3 */}
      <Section id="names" title="Names carry meaning" kicker="Experiment 2">
        <p>
          Every token’s input contains the embedding of its <Term id="column-phrase">column phrase</Term>, such as “churn of customer-churn” or
          “price of products”, from the frozen text encoder <Cite kind="spec" ref="§3.6" /> <Cite kind="code" ref="rt/model.py:208-214" />. On an
          unseen database these names are the model’s only clue to what a column means. Shuffling them asks how much it relies on that clue.
        </p>
        <p>
          In the paper, shuffling column and table names lowers the mean zero-shot AUROC from {f1(TABLE4.none.zsClf)} to {f1(TABLE4.colNames.zsClf)}{' '}
          and the mean R² from {f1(TABLE4.none.zsReg)} to {f1(TABLE4.colNames.zsReg)}: “wrong names lead to consistent degradation”{' '}
          <Cite kind="paper" ref="p.9 §5.1" /> <Cite kind="paper" ref="p.9 Table 4" />. After fine-tuning the effect disappears (
          {f1(TABLE4.colNames.ftClf)} vs {f1(TABLE4.none.ftClf)} AUROC).
        </p>
        <p>
          Our version permutes the 384-dimensional phrase embeddings among the columns present in the context, in one random cycle, so every column
          receives another column’s name. The phrase includes the table name, so table names move with it. The masks are untouched, because the
          sampler stores a column’s id (read by the masks) and its phrase embedding (read by the input layer) in two separate arrays:
        </p>
        <CodeRef
          title="Column id and column-phrase embedding are stored separately"
          file="rustler/src/fly.rs"
          lines="469-477"
          code={FLY_COLNAME_469_477}
          highlightLines={[470, 472]}
          note="col_name_idxs feeds the column mask (model.py:181-183); col_name_values feeds the input embedding (model.py:211). The shuffle permutes only the second."
        />
        <CodeRef
          title="The input layer reads the phrase embedding, not the column id"
          file="rt/model.py"
          lines="208-214"
          code={MODEL_COLNAME_TERM_208_214}
          highlightLines={[211]}
          note="Every non-padding token, the masked cell included, gets RMSNorm(W·c + b) of its column phrase c. Shuffling swaps c between columns."
          defaultOpen={false}
        />
        <NamesMap />
        <ShuffleSpread />
        <Callout variant="note" title="An emulation, not the paper’s procedure">
          <p>
            The paper does not say how the names were shuffled (whether within a database, across databases, or per context), and its code is not
            released <Cite kind="spec" ref="§13 #3" />. A cyclic permutation inside the context is our choice; it guarantees that every name is wrong.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="length" title="A shorter or a longer context" kicker="Experiment 3">
        <p>
          Attention cost grows with the context length L. Figure 4 of the paper evaluates a model pretrained at L = {FIG4.pretrainedAt} with
          shorter contexts at inference: L = {FIG4.contexts.join(', ')} <Cite kind="paper" ref="p.8 §4.5" />. Its summary: “{FIG4.quote}”{' '}
          <Cite kind="paper" ref="p.8 §4.5" />.
        </p>
        <p>
          The sampler’s random generator is seeded with the epoch, the seed row and a fixed offset, never with L{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:371-375" />, and L is only read by the stop test <Cite kind="code" ref="rustler/src/fly.rs:502-509" />
          . So with the same seed row a shorter context is exactly the first L cells of a longer one. We checked this for all 1,733 ShopDB seeds
          of every task and split, at every L of the sweep against L = 256. Shrinking L removes the rows the search reached last. Use the L slider
          in the playground, or look at the whole sweep for the live seed:
        </p>
        <LSweepFigure />
        <ScaleNote paper={`pretrained at L = ${FIG4.pretrainedAt}; Fig. 4 goes down to 256`} ours={`trained at L = ${globalL}; the sweep goes from 16 to 256`}>
          Lengths above the training length were never seen in training. The paper only evaluates shorter contexts.
        </ScaleNote>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 5 */}
      <Section id="layers" title="Switching off an attention layer" kicker="Experiment 4">
        <Callout variant="warning" title="This is not the paper’s layer ablation">
          <p>
            In Table 5 and Table 13 the authors train a <strong>new model</strong> without the layer and keep the parameter count constant by
            adding blocks <Cite kind="paper" ref="p.9 §5.2" /> <Cite kind="paper" ref="p.10 Table 5" />. Here we take the trained model and skip one
            sublayer at inference. The rest of the network was trained to expect that layer’s contribution, so the effects are larger and not
            comparable with the paper’s numbers.
          </p>
        </Callout>
        <p>
          Inside a block the four attention layers each add an update to the residual stream, in the order column, feature, neighbor, full.
          Switching a layer off removes its <code>x + …</code> term in every block; the residual stream passes through unchanged{' '}
          <Cite kind="code" ref="rt/model.py:82-86" />.
        </p>
        <CodeRef
          title="RelationalBlock.forward: one residual update per attention layer"
          file="rt/model.py"
          lines="82-86"
          code={MODEL_BLOCK_FWD_82_86}
          highlightLines={[84]}
          note="An inference-time switch skips line 84 for that layer, in every block. The paper instead trains without the layer."
        />
        <LayerEffects />
        <p>
          For orientation, the paper’s retrained models. Removing column attention costs the most zero-shot R² ({f1(LAYER_ABLATION.none.zsReg)} →{' '}
          {f1(LAYER_ABLATION.col.zsReg)}), while removing full attention costs the least ({f1(LAYER_ABLATION.full.zsReg)}){' '}
          <Cite kind="paper" ref="p.10 Table 5" />. On classification the differences are small ({f1(LAYER_ABLATION.col.zsClf)} to{' '}
          {f1(LAYER_ABLATION.full.zsClf)} AUROC vs {f1(LAYER_ABLATION.none.zsClf)}) <Cite kind="paper" ref="p.23 Table 13" />. “Column attention
          has the highest impact on zero-shot transfer. However, on fine-tuning the impact is less pronounced, and the impact of feature and
          neighbor attention is more significant” <Cite kind="paper" ref="p.9 §5.2" />.
        </p>
        <Callout variant="note" title="The switch is shared with the course">
          <p>
            The attention-layer switch is the one ablation that other chapters see: it stays on for this browser session, the top bar flags it, and
            the attention and block chapters show it. The playground’s other switches stay on this page.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 6 */}
      <Section id="checkpoints" title="Untrained, zero-shot, fine-tuned" kicker="Experiment 5">
        <p>
          The checkpoint switch swaps the weights and keeps the context. “Untrained” is step 0 of our pretraining run (random PyTorch-default
          weights). “Zero-shot” is the pretraining step with the best validation score on this task <Cite kind="spec" ref="§9.1" />. “Fine-tuned”
          continues from it on ShopDB’s own Train labels, so it is no longer zero-shot <Cite kind="spec" ref="§9.2" />.
        </p>
        <CkptCompare />
        <CkptMetrics />
        <p>
          The paper reports no zero-shot score for an untrained RT. After full fine-tuning on each task’s training set, pretrained RT reaches a
          mean of {f1(TABLE10.pretrained.clf)} AUROC and {f1(TABLE10.pretrained.reg)} R², and untrained RT {f1(TABLE10.untrained.clf)} and{' '}
          {f1(TABLE10.untrained.reg)}, against {f1(HEADLINE.zsClf)} and {f1(HEADLINE.zsReg)} zero-shot <Cite kind="paper" ref="p.21 Table 10" />{' '}
          <Cite kind="paper" ref="p.8 Tables 1-2" />.
        </p>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 7 */}
      <Section id="batch" title="Does it hold across seeds?" kicker="Batch mode">
        <p>
          One seed is an anecdote. The paper’s tables report a metric over many seeds: <Term id="auroc">AUROC</Term> for binary tasks and{' '}
          <Term id="r2">R²</Term> for regression, both computed from one forward pass per seed <Cite kind="spec" ref="§8.5" />. Batch mode runs
          each switch on its own over every seed of the live split, which gives this model’s own version of the paper’s ablation tables, and then
          whatever combination you switched on in the playground. Both are compared with the full model on the same seeds.
        </p>
        <CodeRef
          title="The evaluation metrics"
          file="rt/main.py"
          lines="328-337"
          code={MAIN_METRICS_328_337}
          note="AUROC is computed on raw logits with labels int(z > 0); R² on z-scores. The batch below uses the engine's mirrors of both."
        />
        <BatchMode />
        <Callout variant="note" title="Small splits, large variance">
          <p>
            ShopDB’s splits are small ({live.seeds.length} seeds in the live one), so a few seeds can move AUROC by several points. Our checkpoints
            are also selected on the Val split, which flatters Val (spec §9.1). The paper’s ablation tables predate the released sampler (spec §13
            #1), so even the paper’s own “none” row differs from its headline numbers.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 8 */}
      <Section id="drivers" title="What the paper concluded" kicker="Summary">
        <p>
          The authors rank the components that enable zero-shot transfer, “in order of importance” <Cite kind="paper" ref="p.10 §7" />:
        </p>
        <ol>
          {DRIVERS.map((d) => (
            <li key={d}>{d.charAt(0).toUpperCase() + d.slice(1)}.</li>
          ))}
        </ol>
        <p>
          Experiments 1 and 2 of this page touch points 1, 4 and 5 directly. Points 2 and 3 come from retrained models, which a playground cannot
          reproduce; Experiment 4 only shows how much the trained model depends on each layer. For {phrase.long}, which of them moved your
          prediction most?
        </p>
      </Section>

      <div className="page-grid">
        <Quiz
          questions={[
            {
              prompt: 'Which part of the context drives zero-shot transfer most, according to the paper?',
              choices: [
                { text: 'The column and table names', why: 'Shuffling them costs less: 70.1 → 69.5 AUROC and 22.8 → 20.5 R² (Table 4, p.9).' },
                {
                  text: 'Past labels of the target entity (self labels)',
                  correct: true,
                  why: 'Removing them drops mean zero-shot AUROC from 70.1 to 53.8 and R² from 22.8 to −5.5 (Table 4, p.9).',
                },
                { text: 'Labels of other entities', why: 'Removing them changes almost nothing: 70.6 AUROC, 22.9 R².' },
                { text: 'The neighbor attention layer', why: 'Layer ablations are retrained models; neighbor removal moves R² from 22.8 to 22.0.' },
              ],
            },
            {
              prompt: `The batch table shows how this model’s AUROC changes when column attention is switched off. Can you compare that change with Table 13 (${f1(LAYER_ABLATION.none.zsClf)} → ${f1(LAYER_ABLATION.col.zsClf)})?`,
              choices: [
                { text: 'Yes, both remove the same layer', why: 'They remove it in different ways.' },
                {
                  text: 'No: the paper retrains a model without the layer (with extra blocks); here a trained layer is skipped at inference',
                  correct: true,
                  why: 'The paper trains a new model without the layer and adds blocks to keep the parameter count (p.9 §5.2). A network trained with the layer expects its update in every block, so skipping it at inference measures something else.',
                },
                { text: 'Only if L = 1024', why: 'The context length is not the issue; the training is.' },
              ],
            },
            {
              prompt: 'What does “Shuffle column names” change in this playground?',
              choices: [
                {
                  text: 'Only the column-phrase embeddings fed to the input layer; the masks still use the true column ids',
                  correct: true,
                  why: 'fly.rs stores col_name_idxs (masks) and col_name_values (input) separately; only the values are permuted.',
                },
                { text: 'The column mask, so cells attend to other columns', why: 'The masks are built from col_name_idxs, which stay unchanged.' },
                { text: 'The cell values of each column', why: 'Values are untouched; only the names move.' },
              ],
            },
          ]}
        />
      </div>

      <ChapterNav slug="playground" />
    </ChapterPage>
  );
}
