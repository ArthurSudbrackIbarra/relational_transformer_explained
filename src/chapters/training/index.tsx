/**
 * Chapter 14 "training": Pretraining, zero-shot, fine-tuning (BUILD_PLAN §6, docs/CHAPTER_BRIEFS.md §14).
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
            Every chapter so far ran the model with trained weights. This one shows how they were made: pretraining that hides one cell per context,
            a database held out for testing, a one-cycle learning-rate schedule, a checkpoint chosen per task, and fine-tuning at the end. The figures
            come from this site’s own training run, shown next to the paper’s recipe.
          </>
        }
        learn={[
          'What RT is trained to do: recover one masked cell per context, with one loss for pretraining and fine-tuning',
          'What leave-one-database-out means, and why choosing a checkpoint on the target’s validation labels still counts as zero-shot',
          'How the learning rate and Adam’s β1 change over a pretraining run',
          'What our miniature run achieved on ShopDB, before and after fine-tuning, and what it cannot tell you',
        ]}
      >
        <p className="mt-6 text-[17px] leading-relaxed text-ink-2">
          <span className="eyebrow mr-2">Live context</span>
          task <span className="text-ink">{task.id.split('/')[1]}</span>, seed{' '}
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
          pretraining and fine-tuning” <Cite kind="paper" ref="p.6 §3.3" />. Fine-tuning changes the data; the objective stays the same. Chapter 12
          works through both losses with live numbers.
        </p>
        <p>
          Which seed rows does pretraining use? It builds one dataset over every training task of every pretraining database, all on their training
          split <Cite kind="code" ref="rt/main.py:123-136" />. A forecasting task contributes the rows of its Train split; an autocomplete task
          contributes every row of its table that has the target cell <Cite kind="code" ref="rt/data.py:77-81" /> <Cite kind="spec" ref="§9.1" />.
          These items go into a single list with no balancing, so a task’s share of the training signal is its share of the rows{' '}
          <Cite kind="spec" ref="§4.14" />. Step through the figure to watch one batch of our run come together.
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
          <span>{task.id.split('/')[1]}</span>. ShopDB never appears in pretraining: for it, the same exercise is the test.
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
                note: 'int(steps / len(loader)) is the epoch number, so every pass shuffles the list with a new seed: the epoch plus the run seed, 0.',
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
                highlightLines: [347, 348],
                note: 'With 8 GPUs, GPU k takes the k-th block of 32 items, so one optimizer step sees 256 contexts.',
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
        <Callout variant="note" title="Why not mask more cells?">
          <p>
            The paper also tried masking extra cells at random during pretraining, on top of the target cell. Zero-shot quality went down as the
            masking probability rose from 0 to 0.2 to 0.4: mean AUROC 70.1 → 69.8 → 69.3 and mean R² 22.8 → 16.9 → 15.3{' '}
            <Cite kind="paper" ref="p.24 Table 14" />. The released code masks the target cell only <Cite kind="spec" ref="§10 #14" />{' '}
            <Cite kind="spec" ref="§11.6" />, and so does this site.
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
          In the pretraining script the whole protocol is two list filters: train on every task whose database is not rel-amazon, and evaluate
          rel-amazon’s forecasting tasks on val and test <Cite kind="code" ref="scripts/example_pretrain.py:17-18" />.
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
              next, with lag-1 correlations from {fx(story.lag1Min, 2)} to {fx(story.lag1Max, 2)}. A model that learns to read an
              entity’s history in SchoolDB can use the same skill on a ShopDB customer.
            </>
          )}
        </p>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 3 */}
      <Section id="schedule" title="The learning-rate schedule" kicker="Step 3">
        <p>
          The optimizer is AdamW with weight decay 0.1 and a peak learning rate of 10⁻³ <Cite kind="paper" ref="p.6 §4.1" />. The schedule is
          PyTorch’s OneCycleLR <Cite kind="code" ref="rt/main.py:197-204" />. The learning rate η climbs linearly from 4·10⁻⁵ (the peak divided by
          25) to the peak over the first 20% of the steps, then falls linearly to almost zero over the rest. The same scheduler moves Adam’s β1 the
          opposite way, from 0.95 down to 0.85 at the peak and back up <Cite kind="spec" ref="§9.2" />.
        </p>
        <ScheduleFigure schedule={schedule} />
        <p>
          β1 is Adam’s momentum: how much of the previous update direction each step keeps. The one-cycle policy comes from Smith and Topin, who
          warm up to a large learning rate and then anneal it over most of training{' '}
          <Cite kind="url" ref="https://arxiv.org/abs/1708.07120" title="Smith & Topin, Super-Convergence (2017)" />, and Smith recommends lower
          momentum while the learning rate is high{' '}
          <Cite kind="url" ref="https://arxiv.org/abs/1803.09820" title="Smith, A disciplined approach to neural network hyper-parameters (2018)" />.
          Because the scheduler sets β1 at every step, the β1 = 0.9 in the optimizer call below is never used.
        </p>
        <CodeRef
          title="AdamW and the OneCycleLR schedule"
          file="rt/main.py"
          lines="186-204"
          code={MAIN_OPT_186_204}
          highlightLines={[190, 191, 192, 198, 200, 201, 202, 203]}
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="run" title="Our training run" kicker="Step 4">
        <p>
          Every trained number on the site comes from this run. It follows the paper’s recipe at a smaller scale: the same objective, optimizer,
          schedule and gradient clipping, with {steps !== undefined ? int(steps) : '…'} steps of {dataset?.batchSize ?? 32} contexts instead of
          50,001 steps of 256. Each step runs the code’s train step. A forward pass computes the loss, the old gradients are cleared and a backward
          pass computes new ones. If the global gradient norm is above 1.0, the whole gradient is scaled down to norm 1. One AdamW update and one
          scheduler step finish it <Cite kind="code" ref="rt/main.py:394-424" />.
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
              In our run, customer-churn keeps step {int(churn.zsStep)} and product-sales keeps step {int(sales.zsStep)}. One pretraining run
              gives one checkpoint per task. That is why the site ships a separate zero-shot checkpoint for each task, such as{' '}
              <span className="font-mono">pretrain-churn</span> and <span className="font-mono">pretrain-sales</span>, and why the “auto” checkpoint
              setting depends on the task.{' '}
            </>
          ) : null}
          {churn.valJump !== undefined && (
            <>
              The validation curve also jumps between neighboring evaluations, by up to {fx(churn.valJump * 100, 1)} AUROC points for customer-churn
              on its {churn.nVal ?? '…'} validation seeds. With so few seeds, which step wins is partly chance. At the chosen step the test AUROC is{' '}
              {P(churn.zsTest)}; at the last step it would have been {P(churn.lastTest)}.
            </>
          )}
        </p>
        <Callout variant="insight" title="Is this still zero-shot?">
          <p>
            Yes, in the paper’s sense, which is about weight updates and where information enters. No gradient ever comes from ShopDB, and at
            inference ShopDB reaches the model only through the context window. The validation labels of the target task do choose among the
            checkpoints of one run, and the paper says so openly <Cite kind="paper" ref="p.6 §4.1" /> <Cite kind="spec" ref="§9.1" />. It applies
            the same rule to every method it fine-tunes: “for all methods, the best checkpoint is selected based on validation set performance”{' '}
            <Cite kind="paper" ref="p.20 App. F.1" />. Choosing by the test metric would not be fair: the “best test over all evals” card at the
            end of the replay shows how much it would flatter the result.
          </p>
        </Callout>
        <CodeRef title="What the run reports at the end" file="rt/main.py" lines="441-448" code={MAIN_BEST_441_448} />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 6 */}
      <Section id="results" title="Results on ShopDB" kicker="Step 6">
        <p>
          For each ShopDB forecasting task, the figure places five models on one axis: an untrained network, the{' '}
          <Term id="entity-mean">EntityMean</Term> baseline, RT zero-shot, and RT fine-tuned on ShopDB’s Train split from a random init and from the
          pretrained checkpoint. Churn is scored by <Term id="auroc">AUROC</Term> and sales by <Term id="r2">R²</Term>, both in percent as in the
          paper’s tables.
        </p>
        <ResultsFigure results={results} />
        {results && (
          <>
            <p>
              On customer churn, zero-shot RT reaches a test AUROC of {P(churn.zsTest)}, {cmp(churn.zsTest, churn.emTest)} EntityMean
              ({P(churn.emTest)}) and {cmp(churn.zsTest, churn.initTest)} the untrained network ({P(churn.initTest)}). Validation gives the same
              order: {P(churn.zsVal)} vs {P(churn.emVal)} vs {P(churn.initVal)}.
            </p>
            <p>
              On product sales, zero-shot RT reaches a test R² of {P(sales.zsTest)}, {cmp(sales.zsTest, sales.emTest)} EntityMean
              ({P(sales.emTest)}). Pretraining alone lifts it from {P(sales.initTest)} for the untrained network, without a single gradient from
              ShopDB. EntityMean is a strong baseline when counts persist over time. In the paper it beats RT on two of the eight regression tasks
              (item-ltv 54.2 vs 32.5, post-votes 43.7 vs 33.9) <Cite kind="paper" ref="p.8 Table 2" />.
            </p>
            <p>
              Fine-tuning, again selected on validation, reaches a test AUROC of {P(churn.ftPreTest)} on churn and a test R² of {P(sales.ftPreTest)}{' '}
              on sales when it starts from the pretrained checkpoint. From random init it reaches {P(churn.ftUntTest)} and {P(sales.ftUntTest)}. Step
              7 shows the curves behind these numbers.
            </p>
            <p>
              Two limits apply to all of this. The test sets are small, {churn.nTest ?? '…'} seeds for churn and {sales.nTest ?? '…'} for sales, so
              differences of a few points are within noise. The next figure makes this concrete: untrained networks, which have learned nothing,
              already spread{story.initSpread ? ` from ${P(story.initSpread.lo)} to ${P(story.initSpread.hi)}` : ''} in test AUROC on churn,
              depending only on their random seed. The data are also simple and synthetic: three small pretraining databases and ten tasks, against
              the paper’s six real databases and about thirty tasks. Our numbers show that the mechanism runs end to end. They say nothing about how
              RT compares with other models.
            </p>
          </>
        )}
        <InitSpread results={results} />
        <p>
          Why isn’t an untrained network at 50? Its output is a fixed but random function of the context, and the context carries information about
          the label, past labels included. On only {churn.nTest ?? 'a few dozen'} test seeds, some random functions happen to rank churners above
          loyal customers and others below. The average over initializations is still 50. PyTorch draws every initial weight and bias symmetrically
          around zero <Cite kind="spec" ref="§7.4" />, so a network and its twin with the boolean decoder’s weights and bias negated are equally
          likely. The twin outputs the opposite logit for every seed and ranks every pair the other way round (AUROC 100 − x). Any single network
          is one draw from a wide distribution.
        </p>
        <p>
          The metrics average over every seed. The next figure follows a single prediction instead: the running example’s context, passed through
          the three checkpoints of its task.
        </p>
        <LiveCheckpoints results={results} />
        <p>
          You can recompute every number in this section in your browser. The panel below rebuilds each seed’s context with the engine, runs the
          exported weights and computes the metric the way the code does: AUROC on raw logits, R² in z-space{' '}
          <Cite kind="code" ref="rt/main.py:328-337" />.
        </p>
        <ReproducePanel results={results} />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 7 */}
      <Section id="finetune" title="Fine-tuning: pretrained vs random init" kicker="Step 7">
        <p>
          Fine-tuning trains on the target task itself, on its Train split only, with the same objective, a constant learning rate of 10⁻⁴ and no
          weight decay <Cite kind="code" ref="scripts/example_finetune.py:18-28" />. Only the weights carry over from the checkpoint: AdamW starts
          fresh, with no warmup <Cite kind="code" ref="rt/main.py:177-195" /> <Cite kind="spec" ref="§9.2" />. The paper compares starting from the
          pretrained checkpoint with starting from random weights (which it calls untrained RT) and plots both against fine-tuning steps on a log
          axis <Cite kind="paper" ref="p.7 §4.2" />.
        </p>
        <FineTuneFigure curves={curves} />
        {sales.ft && churn.ft && (
          <>
            <p>
              The pretrained run has a head start. It begins where zero-shot ends (sales test R² {P(sales.ft.step0)}), while the random init begins
              at {P(sales.ft.untStep0)}. The paper’s Figure 3 shows the same kind of gap: its untrained curves start below the bottom of the axes{' '}
              <Cite kind="paper" ref="p.7 Fig. 3" />.
            </p>
            {sales.ft.earlyMin !== undefined && sales.ft.step0 !== undefined && sales.ft.earlyMin < sales.ft.step0 && (
              <p>
                In its first updates the pretrained sales run dips, from {P(sales.ft.step0)} to {P(sales.ft.earlyMin)} at step{' '}
                {int(sales.ft.earlyMinStep)}, before it climbs. The paper sees “a small dip in performance in the very early steps, which we conjecture
                might be due to the model transitioning from a zero-shot, in-context-labels- and semantics-driven regime to a data-driven prediction
                regime” <Cite kind="paper" ref="p.7 §4.2" />.
              </p>
            )}
            {churn.ft.peak !== undefined && churn.ft.final !== undefined && churn.ft.final < churn.ft.peak && (
              <p>
                Train long enough and fine-tuning overfits. ShopDB’s churn Train split has {int(story.trainRowsChurn)} rows, and{' '}
                {int(curves?.runs['finetune-churn-pretrained']?.maxSteps)} steps of 32 contexts revisit each one about {fx(story.ftEpochs, 0)} times.
                Switch the figure to “training loss”: the loss keeps falling, while the test AUROC peaks at {P(churn.ft.peak)} (step{' '}
                {int(churn.ft.peakStep)}) and ends at {P(churn.ft.final)}. Validation selection keeps step {int(churn.ftPreStep)}, with a test AUROC
                of {P(churn.ftPreTest)}, {cmp(churn.ftPreTest, churn.zsTest)} zero-shot ({P(churn.zsTest)}). The paper’s per-task curves for the small
                rel-f1 tasks bend down the same way <Cite kind="paper" ref="p.20 Fig. 7" />.
              </p>
            )}
          </>
        )}
        <CodeRef
          title="The fine-tuning script"
          file="scripts/example_finetune.py"
          lines="1-37"
          code={FINETUNE_SCRIPT_1_37}
          highlightLines={[12, 18, 19, 24, 25, 26, 28]}
          note="It loads a pretrained checkpoint, then trains and evaluates on a single target task, here rel-amazon’s user-churn."
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 8 */}
      <Section id="hyperparameters" title="Hyperparameters, paper vs this site" kicker="Step 8">
        <p>
          The table puts every training setting of the paper’s scripts next to ours. Only sizes, data, precision and hardware differ. The
          optimizer, schedule, clipping, objective and selection rule run the same code <Cite kind="spec" ref="§9.2" />.
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
              prompt: 'What does fine-tuning change compared with pretraining?',
              choices: [
                {
                  text: 'The rows it trains on: only the target task’s Train split. The masked-cell loss stays the same.',
                  correct: true,
                  why: 'The paper uses one formulation for both stages (p.6 §3.3). Fine-tuning swaps the seeds and uses a constant, smaller learning rate.',
                },
                {
                  text: 'The objective: a new prediction head replaces the decoders and gets its own loss',
                  why: 'There is no new head. The same decoders and the same per-type losses (BCE, Huber) are used.',
                },
                {
                  text: 'Nothing: fine-tuning just continues pretraining on the same task list',
                  why: 'Fine-tuning trains on the target task, which pretraining never saw.',
                },
              ],
              explanation:
                'Fine-tuning keeps the decoders and the masked-cell loss. It changes the data (the target task’s Train split) and the optimizer settings (a constant 10⁻⁴, no weight decay).',
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
