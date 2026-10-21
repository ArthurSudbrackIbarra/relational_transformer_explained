/**
 * Chapter 13 "training": Pretraining, zero-shot, fine-tuning (BUILD_PLAN §6, docs/CHAPTER_BRIEFS.md §13).
 *
 * Everything about OUR run is read at runtime from public/data/training/*.json (python/train.py) and
 * from the live engine (pretraining item list, contexts, forward passes); paper numbers are the
 * published ones (paper.ts). Spec: research/RT_TECHNICAL_SPEC.md §9 (all), §4.14, §8.5-8.6, §10 #10,
 * #13, #14, #18, #21, §11.1, §11.6, §11.7.
 */
import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import { Callout, ChapterHeader, ChapterNav, ChapterPage, Cite, CodeRef, Quiz, ScaleNote, Section, StatusBadge, Term } from '@/components';
import { useAssets, useSeedNode, useTask, usePreprocessed } from '@/state/hooks';
import { BatchFigure } from './BatchFigure';
import { big, fx, int, mainRun, mean, pct, useTrainingData, type CheckpointInfo } from './data';
import { FineTuneFigure } from './FineTuneFigure';
import { HyperTable } from './HyperTable';
import { LeaveOneOut } from './LeaveOneOut';
import { LiveCheckpoints } from './LiveCheckpoints';
import { usePretrainSet } from './pretrainSet';
import { InitSpread, ReproducePanel, ResultsFigure } from './ResultsFigure';
import { RunFigure } from './RunFigure';
import { ScheduleFigure } from './ScheduleFigure';
import { SelectionFigure } from './SelectionFigure';
import {
  FINETUNE_SCRIPT_1_37,
  FLY_BATCH_336_355,
  FLY_ITEMS_287_296,
  FLY_MASK_492_496,
  FLY_SHUFFLE_324_328,
  MAIN_BEST_441_448,
  MAIN_DATASET_123_136,
  MAIN_EVAL_SELECT_373_392,
  MAIN_OPT_186_204,
  MAIN_SHUFFLE_369_371,
  MAIN_STEP_394_424,
  MODEL_LOSS_250_267,
  PRETRAIN_SCRIPT_1_36,
} from './snippets';
import { buildStory, cmp } from './story';
import { CodeTabs } from './CodeTabs';

const P = (v: number | undefined, d = 1) => pct(v, d);

export default function TrainingChapter() {
  const data = useTrainingData();
  const { curves, results, schedule, dataset } = data;
  const set = usePretrainSet();
  const story = useMemo(() => buildStory(results, curves, dataset, set), [results, curves, dataset, set]);
  const run = mainRun(curves);
  const { churn, sales } = story;

  // live running example
  const task = useTask();
  const seed = useSeedNode();
  const pdb = usePreprocessed();
  const assets = useAssets();
  const seedNode = pdb.nodes[seed];
  const entity = seedNode?.f2pNbrIdxs[0] !== undefined ? pdb.nodes[seedNode.f2pNbrIdxs[0]] : undefined;
  const entityName = entity?.cells.find((c) => c.colName === 'name')?.raw;
  const liveCkpt = results?.checkpoints?.[assets.ckpt] as CheckpointInfo | undefined;
  const steps = run?.maxSteps ?? dataset?.pretrain?.steps;

  return (
    <ChapterPage slug="training">
      <ChapterHeader
        slug="training"
        lede={
          <>
            Every chapter so far ran the model with trained weights. This one shows where they come from: one hidden cell per context, a database held
            out, a one-cycle learning-rate schedule, a checkpoint chosen per task, then fine-tuning. All of it from this site’s real training run,
            next to the paper’s recipe.
          </>
        }
        learn={[
          'What RT is trained to do: recover one masked cell per context, with one loss for pretraining and fine-tuning',
          'What leave-one-database-out means, and why choosing a checkpoint on the target’s validation labels still counts as zero-shot',
          'How the learning rate and Adam’s β1 move during training, and where the code differs from the paper’s description',
          'What our miniature run achieved on ShopDB, before and after fine-tuning, and what it cannot tell you',
        ]}
      >
        <p className="mt-6 text-[17px] leading-relaxed text-ink-2">
          <span className="eyebrow mr-2">Live context</span>
          task <span className="font-mono text-ink">{task.id.split('/')[1]}</span>, seed{' '}
          <span className="text-ink">
            {entity ? `${entity.tableName.replace(/s$/, '')} ${entity.rowPos}` : `node ${seed}`}
            {entityName ? ` (${String(entityName)})` : ''}
          </span>{' '}
          · checkpoint <span className="font-mono text-ink">{assets.ckpt}</span>
          {liveCkpt?.step !== undefined && <> = step {int(liveCkpt.step)} of the run below</>}{' '}
          {assets.weightsKind === 'init' && <StatusBadge status="warning">untrained weights</StatusBadge>} <ScaleNote inline />
        </p>
      </ChapterHeader>

      {data.status === 'error' && (
        <div className="page-grid">
          <Callout variant="warning" title="The training files did not load">
            <p>public/data/training/*.json could not be fetched ({data.error}). The figures about our run are empty; the explanations still hold.</p>
          </Callout>
        </div>
      )}

      {/* ───────────────────────────────────────────────────────────── 1 */}
      <Section id="objective" title="One masked cell per context" kicker="Step 1">
        <p>
          Pretraining gives RT the same exercise it will face at test time. Take a <Term id="seed-row">seed row</Term> of some task, sample its{' '}
          <Term id="context-window">context window</Term> (chapter 6), hide one cell, and ask the model to recover it. The hidden cell is always the
          seed’s own target: the label of a forecasting task row, or the target column of a row for an autocomplete task. There is exactly one{' '}
          <Term id="target-cell">masked cell</Term> per context, and every other cell, past labels included, stays visible{' '}
          <Cite kind="spec" ref="§4.10" /> <Cite kind="code" ref="rustler/src/fly.rs:492-496" />.
        </p>
        <CodeRef
          title="The only masked cell is the seed row’s target"
          file="rustler/src/fly.rs"
          lines="492-496"
          code={FLY_MASK_492_496}
          highlightLines={[496]}
          note="masks marks the cells the loss is computed on; it is set to is_target, so one cell per sequence."
        />
        <p>
          The model predicts that cell with the decoder of its type and is scored with the loss of that type:{' '}
          <Term id="bce">binary cross-entropy</Term> against 1{'{'}y &gt; 0{'}'} for a boolean, the <Term id="huber">Huber loss</Term> with δ = 1 for
          a number, both on z-scored values. “The overall loss is the mean over all masked cells in the batch”, and “this formulation is used in both
          pretraining and fine-tuning” <Cite kind="paper" ref="p.6 §3.3" />. Fine-tuning changes the data, never the objective. Chapter 12 works
          through both losses with live numbers.
        </p>
        <p>
          Which seeds? Pretraining builds one dataset over every training task of every pretraining database, all on their training split{' '}
          <Cite kind="code" ref="rt/main.py:123-136" />. A forecasting task contributes the rows of its Train split; an autocomplete task contributes
          every row of its table that has the target cell <Cite kind="code" ref="rt/data.py:77-81" /> <Cite kind="spec" ref="§9.1" />. These items go
          into a single list with no balancing, so a task’s share of the training signal is its share of the rows <Cite kind="spec" ref="§4.14" />.
          Step through how one batch of our run comes together:
        </p>
        <BatchFigure set={set} dataset={dataset} />
        <p>
          {story.autocompleteShare !== undefined ? (
            <>
              In our run the four autocomplete tasks supply {fx(100 * story.autocompleteShare, 0)}% of all items, because they use every row of their
              table (submissions, votes, results), while a forecasting task has one row per entity and timestamp.{' '}
            </>
          ) : null}
          A batch therefore mixes databases, tasks, booleans and numbers, and every optimizer step averages its gradient over all of them. In the
          running example the masked cell is {entity ? `${entity.tableName.replace(/s$/, '')} ${entity.rowPos}’s` : 'the seed’s'} label in{' '}
          <span className="font-mono">{task.id.split('/')[1]}</span>. ShopDB never appears in pretraining: for it, the same exercise is the test.
        </p>
        <CodeTabs
          title="The code behind the figure"
          intro="From the task list to the batch loss, in the order the figure steps through. Pick an excerpt."
          tabs={[
            {
              id: 'dataset',
              label: '1 · one dataset',
              code: {
                title: 'One dataset over every training task',
                file: 'rt/main.py',
                lines: '123-136',
                code: MAIN_DATASET_123_136,
                highlightLines: [125],
                note: 'Every task tuple gets the split "train"; for autocomplete tasks data.py then resolves it to the whole DB table.',
              },
            },
            {
              id: 'items',
              label: '2 · items',
              code: {
                title: 'One item per row that has the target cell',
                file: 'rustler/src/fly.rs',
                lines: '287-296',
                code: FLY_ITEMS_287_296,
                highlightLines: [290, 291],
                note: 'Run once per task; the items of all tasks are appended to the same list, in task order.',
              },
            },
            {
              id: 'shuffle',
              label: '3 · shuffle',
              code: {
                title: 'One shuffle of the whole list per epoch',
                file: 'rustler/src/fly.rs',
                lines: '324-328',
                code: FLY_SHUFFLE_324_328,
                highlightLines: [326, 327],
              },
            },
            {
              id: 'pass',
              label: '4 · every pass',
              code: {
                title: 'The training loop re-shuffles at every pass',
                file: 'rt/main.py',
                lines: '369-371',
                code: MAIN_SHUFFLE_369_371,
                highlightLines: [370],
                note: 'int(steps / len(loader)) is the epoch number, so pass e shuffles with seed e + 0.',
              },
            },
            {
              id: 'batch',
              label: '5 · a batch',
              code: {
                title: 'A batch is 32 consecutive items of the shuffled list',
                file: 'rustler/src/fly.rs',
                lines: '336-355',
                code: FLY_BATCH_336_355,
                highlightLines: [347, 348, 350],
                note: 'With 8 GPUs (world_size), rank k takes the k-th block of 32; the last batch wraps around to the start of the list.',
              },
            },
            {
              id: 'loss',
              label: '6 · the loss',
              code: {
                title: 'The loss: per type, then the mean over masked cells',
                file: 'rt/model.py',
                lines: '250-267',
                code: MODEL_LOSS_250_267,
                highlightLines: [251, 253, 254, 255, 265],
              },
            },
          ]}
        />
        <Callout variant="note" title="Masking more cells is an ablation, not the default">
          <p>
            The paper’s Table 14 also pretrains with extra cells masked at random, with probability P(mask) = 0.2 or 0.4, while the target cell is
            always masked. Zero-shot quality goes down: mean AUROC 70.1 → 69.8 → 69.3 and mean R² 22.8 → 16.9 → 15.3{' '}
            <Cite kind="paper" ref="p.24 Table 14" />. The released code has no such sampler: it masks the target cell only{' '}
            <Cite kind="spec" ref="§10 #14" /> <Cite kind="spec" ref="§11.6" />, and so does this site. (Table 14’s P(mask) = 0 column, 70.1,
            comes from the paper’s first version; the headline 69.7 of Table 1 was re-run later <Cite kind="spec" ref="§10 #25" />.)
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="leave-one-out" title="Leave one database out" kicker="Step 2">
        <p>
          <Term id="zero-shot">Zero-shot</Term> means “predicting new targets on a new RDB with a new schema, without weight updates. Information
          about the new RDB is available only at inference time, solely through the context window” <Cite kind="paper" ref="p.4 §2.3" />. To test this
          honestly, the target database must stay out of pretraining. So the paper pretrains “separately for each target dataset on all tasks from all
          other datasets” <Cite kind="paper" ref="p.6 §4.1" />: <Term id="leave-one-db-out">leave-one-database-out</Term>. RelBench has seven
          databases. rel-event is in every pretraining mix <Cite kind="spec" ref="§9.1" /> but never evaluated, “as it has been found to have temporal
          leakage issues” <Cite kind="paper" ref="p.6 §4.1" />.
        </p>
        <LeaveOneOut set={set} dataset={dataset} results={results} />
        <p>
          The pretraining script expresses the whole protocol as two list filters: train on every task whose database is not rel-amazon, evaluate
          rel-amazon’s forecasting tasks, on val and test <Cite kind="code" ref="scripts/example_pretrain.py:17-18" />.
        </p>
        <CodeRef
          title="The leave-one-DB-out pretraining script (target: rel-amazon)"
          file="scripts/example_pretrain.py"
          lines="1-36"
          code={PRETRAIN_SCRIPT_1_36}
          highlightLines={[8, 17, 18]}
          note="That leaves 30 training tasks for rel-amazon; across targets, 29 to 33 (spec §9.1)."
        />
        <p>
          We do the same with four synthetic databases: SchoolDB, RacingDB and ForumDB for pretraining, ShopDB held out. ShopDB’s tables, column names
          and rows never reach the optimizer. The one thing it shares with the pretraining data is the frozen text encoder, which embeds every column
          phrase and string of every database the same way (chapter 5).
        </p>
        <p>
          Why should anything transfer? The paper expects it “as databases share structural commonalities, e.g., dimension tables, fact tables, hubs
          and tripartite structures, as well as semantic similarities” <Cite kind="paper" ref="p.6 §4.1" />. It later finds the main driver is the
          target entity’s own past labels <Cite kind="paper" ref="p.9 §5.1" />.
          {story.lag1Min !== undefined && (
            <>
              {' '}
              Our generators give every pretraining forecast task that property: an entity’s label in one window correlates with its label in the
              next, with lag-1 correlations from {fx(story.lag1Min, 2)} to {fx(story.lag1Max, 2)} (results.json). A model that learns to read an
              entity’s history in SchoolDB can use the same skill on a ShopDB customer.
            </>
          )}
        </p>
        <Callout
          variant="paper-vs-code"
          title="No cutoff on database rows"
          paper={
            <>
              “We pretrain and fine-tune only on the training splits of the tasks, using database rows only up to the training cutoff timestamps”{' '}
              <Cite kind="paper" ref="p.6 §4.1" />
            </>
          }
          code={
            <>
              no cutoff anywhere: forecasting seeds are Train rows, but autocomplete seeds are every row of the table, from every period{' '}
              <Cite kind="code" ref="rt/data.py:77-81" />
            </>
          }
          verdict="This site follows the code: our autocomplete tasks use every row (spec §10 #10)."
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 3 */}
      <Section id="schedule" title="The learning-rate schedule" kicker="Step 3">
        <p>
          The optimizer is AdamW with weight decay 0.1 and a peak learning rate of 10⁻³ <Cite kind="paper" ref="p.6 §4.1" />. The schedule is
          PyTorch’s OneCycleLR: the learning rate η climbs linearly for the first 20% of the steps, then falls linearly for the rest{' '}
          <Cite kind="code" ref="rt/main.py:197-204" />. The code passes only the peak, the number of steps, the 20% and the word “linear”, so
          PyTorch’s defaults decide the rest. The climb starts at the peak divided by 25, 4·10⁻⁵; the fall ends at that value divided by 10⁴, 4·10⁻⁹;
          and with cycle_momentum on, Adam’s β1 moves the opposite way, from 0.95 down to 0.85 at the peak and back <Cite kind="spec" ref="§9.2" />.
        </p>
        <ScheduleFigure schedule={schedule} />
        <p>
          β1 is Adam’s momentum: how much of the previous update direction each step keeps. The paper does not explain its choice of schedule. The
          one-cycle policy comes from Smith and Topin, who warm up to a large learning rate and then anneal it for most of training{' '}
          <Cite kind="url" ref="https://arxiv.org/abs/1708.07120" title="Smith & Topin, Super-Convergence (2017)" />, and Smith recommends lowering
          momentum while the learning rate is high{' '}
          <Cite kind="url" ref="https://arxiv.org/abs/1803.09820" title="Smith, A disciplined approach to neural network hyper-parameters (2018)" />.
          The coded betas = (0.9, 0.999) therefore never apply as written: the scheduler overwrites β1 at every step.
        </p>
        <CodeRef
          title="bf16, AdamW with one parameter group, OneCycleLR"
          file="rt/main.py"
          lines="186-204"
          code={MAIN_OPT_186_204}
          highlightLines={[187, 189, 191, 192, 198, 199, 200, 201, 202, 203]}
          note="net.parameters() is a single group, so weight decay also applies to RMSNorm gains, biases and the mask vectors (spec §9.2)."
        />
        <Callout
          variant="paper-vs-code"
          title="Warmup from zero, decay to zero?"
          paper={
            <>
              “a peak learning rate of 10⁻³, with linear warmup from zero for the first 20% of training, and linear decay to zero for the remainder”{' '}
              <Cite kind="paper" ref="p.6 §4.1" />
            </>
          }
          code={
            <>
              η goes 4·10⁻⁵ → 10⁻³ → 4·10⁻⁹, β1 is cycled 0.95 → 0.85 → 0.95, and gradients are clipped to norm 1.0, which the paper does not mention{' '}
              <Cite kind="code" ref="rt/main.py:197-204" /> <Cite kind="code" ref="rt/main.py:413-418" />
            </>
          }
          verdict="Close for η (4·10⁻⁵ is 4% of the peak); this site follows the code, β1 cycle and clipping included (spec §10 #13)."
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="run" title="Our training run" kicker="Step 4">
        <p>
          This is the run behind every trained number on the site. It follows the paper’s recipe at a smaller scale: the same objective, optimizer,
          schedule and clipping, with {steps !== undefined ? int(steps) : '…'} steps of {dataset?.batchSize ?? 32} contexts instead of 50,001 steps of
          256. Each step is the code’s train step: forward pass and loss, clear the old gradients, backward pass, measure the global gradient norm,
          clip it to 1.0, one AdamW update, one scheduler step <Cite kind="code" ref="rt/main.py:394-424" />.
        </p>
        <CodeRef
          title="One optimizer step"
          file="rt/main.py"
          lines="394-424"
          code={MAIN_STEP_394_424}
          highlightLines={[409, 410, 411, 413, 416, 420, 422]}
        />
        <RunFigure curves={curves} results={results} />
        {run && (
          <p>
            The loss falls from about {fx(mean(run.loss.slice(0, 100)), 2)} in the first 100 steps to about {fx(mean(run.loss.slice(-200)), 2)} over
            the last 200, and it stays noisy from step to step because each batch draws a different mix of tasks. It never approaches zero: a context
            holds only so much information about its hidden cell, and our generators draw every label from a random process.
          </p>
        )}
        <ScaleNote
          paper={`50,001 × 256 contexts × 1024 tokens ≈ ${big(dataset?.paper?.tokensSeen)} tokens`}
          ours={`${int(steps)} × ${dataset?.batchSize ?? 32} contexts × ${dataset?.L ?? 128} tokens ≈ ${big(dataset?.pretrain?.tokensSeen)} tokens`}
        >
          {story.tokenRatio !== undefined
            ? `About ${int(story.tokenRatio)} times fewer tokens, on CPU in float32 instead of 8 A100s in bfloat16. Every operation is the same.`
            : undefined}
        </ScaleNote>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 5 */}
      <Section id="selection" title="Choosing the checkpoint, task by task" kicker="Step 5">
        <p>
          While it pretrains, the code evaluates the held-out database at step 0, every eval_freq steps and at every power of two. For each target
          task it keeps the checkpoint with the best validation metric and, at that moment, records the test metric of the same step{' '}
          <Cite kind="code" ref="rt/main.py:373-392" />. The numbers the run prints at the end, and the paper reports, are those recorded test metrics{' '}
          <Cite kind="code" ref="rt/main.py:441-448" />. Selection is per task because the authors “found significant overfitting during pretraining
          due to limited pretraining tasks” <Cite kind="paper" ref="p.6 §4.1" />.
        </p>
        <CodeRef
          title="Evaluate, then keep the best val checkpoint per task"
          file="rt/main.py"
          lines="373-392"
          code={MAIN_EVAL_SELECT_373_392}
          highlightLines={[373, 374, 383, 384, 385, 386]}
          note="Both metrics are higher-is-better (AUROC on logits, R² in z-space), so one comparison serves both task types."
        />
        <SelectionFigure curves={curves} initialTask={task.id} />
        <p>
          {churn.zsStep !== undefined && sales.zsStep !== undefined ? (
            <>
              In our run, customer-churn keeps step {int(churn.zsStep)} and product-sales keeps step {int(sales.zsStep)}. One pretraining run, two
              checkpoints: that is why the site ships a <span className="font-mono">pretrain-churn</span> and a{' '}
              <span className="font-mono">pretrain-sales</span> file, and why the global “auto” checkpoint depends on the task.{' '}
            </>
          ) : null}
          {churn.valJump !== undefined && (
            <>
              Notice also how much the validation curve moves between neighbouring evaluations: up to {fx(churn.valJump * 100, 1)} AUROC points for
              customer-churn, on {churn.nVal ?? '…'} validation seeds. With so few seeds, which step wins is partly chance. At the chosen step the
              test AUROC is {P(churn.zsTest)}; at the last step it would have been {P(churn.lastTest)}.
            </>
          )}
        </p>
        <Callout variant="insight" title="Is this still zero-shot?">
          <p>
            Yes, in the paper’s sense, which is about weight updates and where information enters. No gradient ever comes from ShopDB, and at
            inference ShopDB reaches the model only through the context window. The validation labels of the target task do choose among the
            checkpoints of one run, and the paper says so openly <Cite kind="paper" ref="p.6 §4.1" /> <Cite kind="spec" ref="§9.1" />. It is the same
            rule the paper applies to every method it fine-tunes: “for all methods, the best checkpoint is selected based on validation set
            performance” <Cite kind="paper" ref="p.20 App. F.1" />. What you should not do is pick by the test metric; the replay shows how much that
            would flatter the result.
          </p>
        </Callout>
        <Callout variant="note" title="Step 0 is a candidate too">
          <p>
            The loop evaluates at step 0, before any update, and the best value starts at −∞. In continued pretraining (the paper’s “Yes” columns,
            which train on the target database’s other tasks) the step-0 model is the leave-one-out checkpoint itself, so if further training never
            improves validation, both columns report the same number. Table 1 shows exactly that for rel-f1: driver-dnf 81.2 and 81.2, driver-top3
            89.3 and 89.3 <Cite kind="paper" ref="p.8 Table 1" /> <Cite kind="spec" ref="§9.3" />.
          </p>
        </Callout>
        <CodeRef title="What the run reports at the end" file="rt/main.py" lines="441-448" code={MAIN_BEST_441_448} />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 6 */}
      <Section id="results" title="Results on ShopDB" kicker="Step 6">
        <p>
          Now the numbers. For each ShopDB forecasting task the figure places five models on one axis: an untrained network, the{' '}
          <Term id="entity-mean">EntityMean</Term> baseline, RT zero-shot, and RT fine-tuned on ShopDB’s Train split from a random init and from the
          pretrained checkpoint. Churn is scored by <Term id="auroc">AUROC</Term> and sales by <Term id="r2">R²</Term>, both in percent as in the
          paper’s tables.
        </p>
        <ResultsFigure results={results} />
        {results && (
          <>
            <p>What the run shows:</p>
            <ul>
              <li>
                <strong>Customer churn.</strong> Zero-shot RT reaches a test AUROC of {P(churn.zsTest)}, {cmp(churn.zsTest, churn.emTest)} EntityMean
                ({P(churn.emTest)}) and {cmp(churn.zsTest, churn.initTest)} the untrained network ({P(churn.initTest)}). On validation the order is
                the same: {P(churn.zsVal)} vs {P(churn.emVal)} vs {P(churn.initVal)}.
              </li>
              <li>
                <strong>Product sales.</strong> Zero-shot RT reaches a test R² of {P(sales.zsTest)}, {cmp(sales.zsTest, sales.emTest)} EntityMean (
                {P(sales.emTest)}). Pretraining alone lifts it from {P(sales.initTest)} for the untrained network, without a single gradient from
                ShopDB. EntityMean is a strong baseline for counts that persist; the paper sees the same on two of its eight regression tasks, where
                EntityMean beats RT (item-ltv 54.2 vs 32.5, post-votes 43.7 vs 33.9) <Cite kind="paper" ref="p.8 Table 2" />.
              </li>
              <li>
                <strong>Fine-tuning.</strong> Selected on validation, fine-tuning from the pretrained checkpoint gives a test {P(churn.ftPreTest)} on
                churn and {P(sales.ftPreTest)} on sales; from random init, {P(churn.ftUntTest)} and {P(sales.ftUntTest)}. Step 7 shows the curves
                behind these numbers.
              </li>
            </ul>
            <p>And what it does not show:</p>
            <ul>
              <li>
                <strong>Small test sets.</strong> {churn.nTest ?? '…'} test seeds for churn and {sales.nTest ?? '…'} for sales. Differences of a few
                points are within noise. The next figure makes this concrete: untrained networks, which have learned nothing, already spread
                {story.initSpread ? ` from ${P(story.initSpread.lo)} to ${P(story.initSpread.hi)}` : ''} test AUROC on churn, depending only on their
                random seed.
              </li>
              <li>
                <strong>Simple synthetic data.</strong> Three small pretraining databases and ten tasks, against the paper’s six real databases and
                about thirty tasks. Our numbers show that the mechanism runs end to end; they do not show how RT compares with anything.
              </li>
            </ul>
          </>
        )}
        <InitSpread results={results} />
        <p>
          Why is an untrained network not at 50? Its output is a fixed but random function of the context, and the context carries information about
          the label, past labels included. On {churn.nTest ?? 'a few dozen'} test seeds, some random functions happen to rank churners above loyal
          customers and others below. PyTorch draws every initial weight and bias from an interval symmetric around zero{' '}
          <Cite kind="spec" ref="§7.4" />, so a network and its twin with the boolean decoder’s weights and bias negated are equally likely. The twin
          outputs the opposite logit for every seed and ranks every pair the other way round (AUROC 100 − x), so the expected AUROC over
          initialisations is 50. Any single network is one draw from a wide distribution.
        </p>
        <p>
          The metrics average over every seed. To see what training changes for one prediction, here is the running example’s context passed through
          the three checkpoints of its task:
        </p>
        <LiveCheckpoints results={results} />
        <p>
          Every number in this section can be recomputed in your browser: the panel below rebuilds each seed’s context with the engine, runs the
          exported weights, and computes the metric as the code does, AUROC on raw logits and R² in z-space{' '}
          <Cite kind="code" ref="rt/main.py:328-337" />.
        </p>
        <ReproducePanel results={results} />
        <Callout variant="note" title="How the paper’s evaluation differs">
          <p>
            The official loop evaluates at most 40 batches per GPU, a fixed subset of at most 10,240 seeds per split, the same at every evaluation{' '}
            <Cite kind="code" ref="rt/main.py:299-300" /> <Cite kind="spec" ref="§8.6" />. Under 8-GPU training, the last batch of a small split can
            also count some seeds twice <Cite kind="spec" ref="§10 #18" />. Our splits are tiny and evaluated in one process, so every seed counts
            exactly once.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 7 */}
      <Section id="finetune" title="Fine-tuning: pretrained vs random init" kicker="Step 7">
        <p>
          Fine-tuning trains on the target task itself: its Train split only, with the same objective, a constant learning rate of 10⁻⁴ and no weight
          decay <Cite kind="code" ref="scripts/example_finetune.py:18-28" />. Checkpoints store only the model weights, so a new AdamW starts with
          empty moments and no warmup <Cite kind="code" ref="rt/main.py:177-195" /> <Cite kind="spec" ref="§9.2" />. The paper compares starting from
          the pretrained checkpoint with starting from random weights, which it calls untrained RT, and plots both against fine-tuning steps on a log
          axis <Cite kind="paper" ref="p.7 §4.2" />.
        </p>
        <FineTuneFigure curves={curves} />
        {sales.ft && churn.ft && (
          <ul>
            <li>
              <strong>A head start.</strong> The pretrained run starts where zero-shot ends (sales test R² {P(sales.ft.step0)}), the random init at{' '}
              {P(sales.ft.untStep0)}. The paper’s Figure 3 shows the same kind of gap: its pretrained curves start at the labelled zero-shot points,
              70.8 AUROC and 29.0 R², and the untrained ones start below the bottom of its axes <Cite kind="paper" ref="p.7 Fig. 3" />.
            </li>
            {sales.ft.earlyMin !== undefined && sales.ft.step0 !== undefined && sales.ft.earlyMin < sales.ft.step0 && (
              <li>
                <strong>A dip first.</strong> In its first updates the pretrained sales run drops from {P(sales.ft.step0)} to {P(sales.ft.earlyMin)}{' '}
                (step {int(sales.ft.earlyMinStep)}) before it climbs. The paper sees “a small dip in performance in the very early steps, which we
                conjecture might be due to the model transitioning from a zero-shot, in-context-labels- and semantics-driven regime to a data-driven
                prediction regime” <Cite kind="paper" ref="p.7 §4.2" />.
              </li>
            )}
            {churn.ft.peak !== undefined && churn.ft.final !== undefined && churn.ft.final < churn.ft.peak && (
              <li>
                <strong>Overfitting.</strong> ShopDB’s churn Train split has {int(story.trainRowsChurn)} rows, and{' '}
                {int(curves?.runs['finetune-churn-pretrained']?.maxSteps)} steps of 32 contexts revisit each about {fx(story.ftEpochs, 0)} times.
                Switch to “training loss”: it keeps falling, while the test AUROC peaks at {P(churn.ft.peak)} (step {int(churn.ft.peakStep)}) and ends
                at {P(churn.ft.final)}. Validation selection keeps step {int(churn.ftPreStep)}, with test {P(churn.ftPreTest)},{' '}
                {cmp(churn.ftPreTest, churn.zsTest)} zero-shot ({P(churn.zsTest)}). The paper’s Figure 3 averages only tasks “which do not show
                overfitting” <Cite kind="paper" ref="p.7 Fig. 3" />; its per-task curves for the small rel-f1 tasks bend down the same way{' '}
                <Cite kind="paper" ref="p.20 Fig. 7" />.
              </li>
            )}
          </ul>
        )}
        <CodeRef
          title="The fine-tuning script"
          file="scripts/example_finetune.py"
          lines="1-37"
          code={FINETUNE_SCRIPT_1_37}
          highlightLines={[10, 12, 14, 18, 24, 25, 26, 28]}
        />
        <Callout
          variant="paper-vs-code"
          title="The released fine-tuning script cannot draw Figure 3 as is"
          paper={
            <>
              “we fine-tune for 33k steps”, best checkpoint on validation, curves on a log axis from step 0 <Cite kind="paper" ref="p.6 §4.1" />{' '}
              <Cite kind="paper" ref="p.7 Fig. 3" />
            </>
          }
          code={
            <>
              32,769 steps; <code>eval_pow2=False</code>, so no evaluations at steps 1, 2, 4, …; <code>save_ckpt_dir=None</code>, so nothing is saved;
              and <code>load_ckpt_path</code> names a file no script writes <Cite kind="code" ref="scripts/example_finetune.py:10-14" />
            </>
          }
          verdict="This site follows the paper’s text: fine-tuning starts from the task’s pretrained checkpoint and evaluates at powers of two (spec §10 #21, §9.2)."
        >
          <p>
            Which checkpoint the paper actually fine-tuned from is not certain: an earlier version of the script started from the
            continued-pretraining checkpoint. Figure 3’s caption, “Target datasets and tasks are unseen during pretraining”, points to the
            leave-one-out checkpoint <Cite kind="spec" ref="§9.2" />.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 8 */}
      <Section id="hyperparameters" title="Hyperparameters, paper vs this site" kicker="Step 8">
        <p>
          The table lists every training setting of the paper-era scripts next to ours. Only sizes, data, precision and hardware differ; the
          optimizer, schedule, clipping, objective and selection rule are the same code paths <Cite kind="spec" ref="§9.2" />.
        </p>
        <HyperTable curves={curves} results={results} dataset={dataset} />
        <ScaleNote />
      </Section>

      <div className="page-grid">
        <Quiz
          title="Check your understanding"
          questions={[
            {
              prompt: 'Which database is never seen during pretraining in this site’s run?',
              choices: [
                {
                  text: 'ShopDB',
                  correct: true,
                  why: 'Leave-one-DB-out: ShopDB is the held-out target; its tasks are only evaluated, zero-shot.',
                },
                {
                  text: 'SchoolDB',
                  why: 'SchoolDB is one of the three pretraining databases.',
                },
                {
                  text: 'None: all four are mixed, and ShopDB’s test split is held out',
                  why: 'That would be a row-level split. Leave-one-DB-out keeps the whole database out.',
                },
              ],
              explanation: 'SchoolDB, RacingDB and ForumDB provide all ten pretraining tasks; ShopDB is only ever evaluated.',
            },
            {
              prompt: 'The checkpoint is chosen with ShopDB’s validation labels. Why is the result still called zero-shot?',
              choices: [
                {
                  text: 'Because no weight is ever updated on ShopDB: the labels only choose among checkpoints of one run',
                  correct: true,
                  why: 'The paper defines zero-shot as no weight updates, with the new database seen only through the context window (p.4 §2.3).',
                },
                {
                  text: 'Because validation labels are never used',
                  why: 'They are: main.py:377-392 keeps the best val step per target task.',
                },
                {
                  text: 'Because the test labels are used instead',
                  why: 'Test labels are only recorded, never used to choose.',
                },
              ],
              explanation:
                'Selection by target-task validation labels is stated in the paper (p.6). It is a mild use of labels, but no gradient comes from the target database.',
            },
            {
              prompt: 'What learning rate does the paper-era code use for the very first pretraining step?',
              choices: [
                {
                  text: '0, as the paper says (“warmup from zero”)',
                  why: 'OneCycleLR starts at max_lr / div_factor, and the default div_factor is 25.',
                },
                {
                  text: '4·10⁻⁵',
                  correct: true,
                  why: '10⁻³ / 25 = 4·10⁻⁵, rising linearly to 10⁻³ at 20% of the steps.',
                },
                {
                  text: '10⁻³',
                  why: 'That is the peak, reached only after 20% of the steps.',
                },
              ],
              explanation: 'The decay likewise ends at 4·10⁻⁹, not 0, and β1 cycles between 0.95 and 0.85 (spec §10 #13).',
            },
          ]}
        />
      </div>
      <div className="page-grid">
        <p className="mt-8 text-[17px] text-ink-2">
          Next, the <Link to="/playground">playground</Link> lets you poke the zero-shot model directly: edit labels, shuffle names, shrink the
          context.
        </p>
      </div>

      <ChapterNav slug="training" />
    </ChapterPage>
  );
}
