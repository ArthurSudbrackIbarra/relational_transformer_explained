/**
 * Chapter 12 "decoding": Decoders, loss and prediction (BUILD_PLAN §6, docs/CHAPTER_BRIEFS.md §12).
 *
 * From the last residual vector to a probability or a number, the training loss, and the metrics.
 * Every number is the live engine: the store's context and forward pass (src/state/hooks.ts →
 * src/engine/pipeline.ts → model/forward.ts, a mirror of rt/model.py:153-267), plus a local preview
 * of the other target type (live.ts) and in-browser evaluations over whole splits (evaluate.ts).
 * Spec: research/RT_TECHNICAL_SPEC.md §8 (all), §3.3, §9.2, §10 #15, #19, §12.
 */
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
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
import { AllPositions } from './AllPositions';
import { BinaryFigure } from './BinaryFigure';
import { DecoderFan } from './DecoderFan';
import { fmt, fmtS, seedEntity, taskLabel, useDecodingLive, type Lane } from './live';
import { BatchFigure, BceFigure, HuberFigure } from './LossFigures';
import { R2Figure, RocFigure } from './MetricsFigures';
import { RegressionFigure } from './RegressionFigure';
import { DecoderSizes, SharedHeadsFigure, TextHeadCheck } from './SharedHeads';
import {
  MAIN_ADAMW_188_195,
  MAIN_METRICS_328_337,
  MAIN_READOUT_283_293,
  MAIN_TASK_TYPE_228_241,
  MODEL_BLOCKS_NORM_OUT_226_229,
  MODEL_DEC_DICT_124_131,
  MODEL_DECODE_231_248,
  MODEL_LOSS_250_267,
} from './snippets';

const SIGMOID_TEX = String.raw`p \;=\; \sigma(\hat z) \;=\; \frac{1}{1 + e^{-\hat z}}`;
const DENORM_TEX = String.raw`\hat y \;=\; \hat z \cdot \sigma_{\text{train}} \;+\; \mu_{\text{train}}`;
const BCE_TEX = String.raw`\begin{gathered}\ell_{\text{BCE}}(\hat z, y) \;=\; \max(\hat z, 0) \;-\; \hat z\,y \;+\; \log\!\big(1 + e^{-|\hat z|}\big)\\[4pt] y = \mathbf{1}\{z_{\text{label}} > 0\}\end{gathered}`;
const BCE_TEX_NARROW = String.raw`\begin{aligned}\ell_{\text{BCE}}(\hat z, y) &= \max(\hat z, 0) - \hat z\,y\\ &\quad + \log\!\big(1 + e^{-|\hat z|}\big)\\[4pt] y &= \mathbf{1}\{z_{\text{label}} > 0\}\end{aligned}`;
const HUBER_TEX = String.raw`\begin{gathered}\ell_{\text{Huber}}(d) \;=\; \begin{cases} \tfrac12 d^2 & |d| < 1 \\[2pt] |d| - \tfrac12 & |d| \ge 1 \end{cases}\\[6pt] d = \hat z - z_{\text{label}}\end{gathered}`;
const MEAN_TEX = String.raw`\mathcal{L} \;=\; \frac{\sum_i \text{masks}_i \,\ell_i}{\sum_i \text{masks}_i}`;
const R2_TEX = String.raw`R^2 \;=\; 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}`;

export default function DecodingChapter() {
  const live = useDecodingLive();
  const g = live.global;
  const target = g.ctx.targetPos;
  const [pos, setPos] = useState(target);
  // a new context (task, seed, L, w, RNG seed) starts again at its masked cell
  useEffect(() => setPos(target), [target, g.ctx]);

  return (
    <ChapterPage slug="decoding">
      <ChapterHeader
        slug="decoding"
        lede={
          <>
            After the last block, every position holds a vector. This chapter turns the masked cell’s vector into an answer: four small linear heads,
            one per data type; a probability or a number in the column’s own units; the loss that training minimizes; and the two metrics the paper
            reports.
          </>
        }
        learn={[
          'Why all four decoders run at every position, and which single output is read',
          'How a logit becomes a probability, and a z-score becomes a number in the column’s units',
          'The BCE and Huber losses with live numbers, and how a batch averages them',
          'Why AUROC is computed on logits and R² in z-space, measured live on a whole split',
        ]}
      >
        <HeaderLine lane={g} />
      </ChapterHeader>

      {/* ───────────────────────────────────────────────────────────── 1 */}
      <Section id="decoders" title="A final norm, then four decoders at every position" kicker="Step 1">
        <p>
          After the last block the forward pass applies one more <Term id="rmsnorm">RMSNorm</Term>, <code>norm_out</code>, to every position{' '}
          <Cite kind="code" ref="rt/model.py:229" />. Chapter 11 followed the vector up to this point. Next come the decoders, which the paper calls
          “cell decoders (also called prediction heads), one for each datatype” <Cite kind="paper" ref="p.6 §3.3" />.
        </p>
        <CodeRef title="The blocks, then norm_out" file="rt/model.py" lines="226-229" code={MODEL_BLOCKS_NORM_OUT_226_229} highlightLines={[229]} />
        <p>
          In the code the decoders are four <code>nn.Linear</code> layers in a dictionary, <code>dec_dict</code>. The number, datetime and boolean
          heads map the d-dimensional vector to a single value. The text head maps it to 384 values, the size of a MiniLM sentence embedding{' '}
          <Cite kind="code" ref="rt/model.py:124-131" />. All four have a bias.
        </p>
        <CodeRef title="dec_dict: one linear decoder per data type" file="rt/model.py" lines="124-131" code={MODEL_DEC_DICT_124_131} defaultOpen />
        <p>
          The forward pass applies every head to the whole sequence, so each position, padding included, gets four outputs{' '}
          <Cite kind="code" ref="rt/model.py:238-239" />. Only one of these values is ever used: the output of the head that matches the masked cell’s
          type, at the masked position <Cite kind="spec" ref="§8.1" />. In the paper’s words, “the decoder to select for final prediction depends on
          the task type, or equivalently on the datatype of the masked cell” <Cite kind="paper" ref="p.6 §3.3" />. Step through it for the live
          context:
        </p>
        <DecoderFan lane={g} pos={pos} setPos={setPos} />
        <p>
          The figure below zooms out to the whole context. Each column is one position and each row one decoder. Switch the view to see how the loss
          picks its single value: for each type <TeX tex="i" />, the code builds{' '}
          <code className="!whitespace-normal !break-words">sem_type_mask = (sem_types == i) &amp; masks</code>, so a position counts only if its
          token has that type <em>and</em> is the masked cell <Cite kind="code" ref="rt/model.py:241" />. Every other output is multiplied by zero.
        </p>
        <AllPositions lane={g} pos={pos} setPos={setPos} />
        <CodeRef
          title="Every decoder runs on every position"
          file="rt/model.py"
          lines="231-248"
          code={MODEL_DECODE_231_248}
          highlightLines={[239, 241, 246]}
          note="yhat = self.dec_dict[t](x) is a (B, S, D_t) tensor: all positions at once. A type with no masked cell in the batch still gets a zero-weight “touch”."
        />
        <Callout variant="insight" title="Why compute outputs nobody reads?">
          <p>
            Applying a head to the whole <TeX tex="(B, S, d)" /> tensor is one matrix product; picking out the masked cells afterwards with a 0/1 mask
            is the plain way to write this in batched PyTorch. The zero-weight “touch” <code>yhat.sum() * 0.0</code> has a reason stated in the code
            itself: “still touch the param to avoid unused param error” <Cite kind="code" ref="rt/model.py:243-247" />. Distributed training expects
            every parameter to take part in each backward pass, and a batch whose targets are all boolean would otherwise leave the number and
            datetime heads out <Cite kind="spec" ref="§8.1" />.
          </p>
        </Callout>

        <h3>The text head is never trained</h3>
        <p>
          Text cells are never masked. The paper only masks “cells in boolean or numeric columns” <Cite kind="paper" ref="p.6 §3.3" />, and the code
          refuses outright: <code className="!whitespace-normal !break-words">raise ValueError("masking text not supported")</code>{' '}
          <Cite kind="code" ref="rt/model.py:256-257" />. So no text output ever enters a loss, and the text head always falls in the “touch” branch:
          its gradient is exactly zero <Cite kind="spec" ref="§8.1" />. One thing still moves it. The optimizer is a single AdamW parameter group with
          weight decay on every parameter <Cite kind="code" ref="rt/main.py:188-195" /> <Cite kind="spec" ref="§9.2" />, and you can see the trace of
          that in our checkpoint:
        </p>
        <TextHeadCheck lane={g} />
        <CodeRef
          title="One AdamW parameter group: weight decay reaches every parameter"
          file="rt/main.py"
          lines="188-195"
          code={MAIN_ADAMW_188_195}
        />

        <h3>Reading the prediction</h3>
        <p>
          At evaluation time the code reads the boolean head for classification tasks and the number head for regression tasks, always at the
          positions flagged <code>is_targets</code> <Cite kind="code" ref="rt/main.py:283-293" />. Which tasks count as regression is decided by a
          hard-coded list of RelBench task names <Cite kind="code" ref="rt/main.py:228-241" /> <Cite kind="spec" ref="§8.3" />.
        </p>
        <CodeRef
          title="Read the prediction and the label at is_targets"
          file="rt/main.py"
          lines="283-293"
          code={MAIN_READOUT_283_293}
          highlightLines={[284, 287]}
        />
        <Callout variant="note" title="How this site picks the head">
          <p>
            The list names RelBench tasks. Only one of our tasks, ForumDB’s post-votes, happens to share a name with one of them. The engine therefore
            reads the head that matches the masked cell’s type. On this site every binary task has a boolean target and every regression task a number
            target, so this picks the same head as a list naming all our regression tasks would <Cite kind="spec" ref="§8.3" />.
          </p>
          <CodeRef title="main.py: the hard-coded list of regression tasks" file="rt/main.py" lines="228-241" code={MAIN_TASK_TYPE_228_241} />
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="binary" title="Binary targets: from logit to probability" kicker="Step 2">
        <p>
          For a binary task the masked cell is a boolean, so the boolean head is read. Its output <TeX tex="\hat z" /> is an unbounded real number, a{' '}
          <em>logit</em>. The paper-era code never turns it into a probability: the loss takes the logit directly{' '}
          <Cite kind="code" ref="rt/model.py:253-255" />, and so does the AUROC <Cite kind="code" ref="rt/main.py:337" />. A probability is still the
          natural number to show a person, and the authors’ later leaderboard script computes it with the sigmoid <Cite kind="spec" ref="§8.4" />:
        </p>
        <MathBlock tex={SIGMOID_TEX} label={<Cite kind="spec" ref="§8.4" />} />
        <BinarySentence lane={live.binary} />
        <BinaryFigure lane={live.binary} />
        <p>
          The label is still inside the masked token: its <code>boolean_values</code> slot holds the true value, gated out of the input so the model
          cannot read it (<Link to="/tokens">chapter 7</Link>) <Cite kind="spec" ref="§5.3" />. That stored value is a z-score, like every boolean (
          <Link to="/preprocessing">chapter 4</Link>). With a share μ of true values strictly between 0 and 1, true becomes{' '}
          <TeX tex="(1-\mu)/\sigma > 0" /> and false becomes <TeX tex="-\mu/\sigma < 0" />, so the sign alone recovers the class{' '}
          <Cite kind="spec" ref="§3.3" />. That is why the paper writes the loss as <TeX tex="\mathrm{BCE}(\mathbf{1}\{r>0\}, r')" />{' '}
          <Cite kind="paper" ref="p.6 §3.3" /> and why the evaluation converts labels with <code>int(x &gt; 0)</code>{' '}
          <Cite kind="code" ref="rt/main.py:336" />.
        </p>
        <Callout variant="warning" title="A constant boolean column would break this">
          <p>
            If every value of a boolean column were the same, its standard deviation would be 0 and every z-score 0/0 = NaN. The code has no guard for
            this, and the NaN would spread through full attention into the loss of the whole batch <Cite kind="spec" ref="§3.3" />. Our four databases
            are built so that no boolean column is constant and every task’s Train split holds both classes.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 3 */}
      <Section id="regression" title="Regression targets: from z‑score to the column’s units" kicker="Step 3">
        <p>
          For a regression task the masked cell is a number, so the number head is read. Its output is a z-score: the prediction lives in the same
          normalized scale as the stored label <Cite kind="spec" ref="§8.3" />. Each numeric column was z-scored with its own mean and sample standard
          deviation, and the Val and Test rows of a task table reuse the Train split’s statistics{' '}
          <Cite kind="code" ref="rustler/src/pre.rs:383-396" /> <Cite kind="spec" ref="§3.3" />. Undoing the z-score gives the prediction in the
          column’s original units:
        </p>
        <MathBlock tex={DENORM_TEX} label={<Cite kind="spec" ref="§8.4" />} />
        <p>
          Like the sigmoid, this step is for display only. The paper-era code never leaves z-space; the authors’ later leaderboard script performs
          exactly this inversion with the Train statistics <Cite kind="spec" ref="§8.4" />.
        </p>
        <RegressionSentence lane={live.regression} />
        <RegressionFigure lane={live.regression} />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="loss" title="The loss, with numbers" kicker="Step 4">
        <p>
          Training compares the read output with the hidden label. The paper: “we apply HuberLoss(r, r′) for regression and binary cross-entropy loss
          BCE(1{'{'}r &gt; 0{'}'}, r′) for binary classification. The overall loss is the mean over all masked cells in the batch”{' '}
          <Cite kind="paper" ref="p.6 §3.3" />. Here r is the cell’s normalized value, so both losses work on z-scores <Cite kind="spec" ref="§8.2" />
          . The same objective is used in pretraining and in fine-tuning <Cite kind="paper" ref="p.6 §3.3" />.
        </p>

        <h3>Binary cross-entropy with logits</h3>
        <p>
          <code>F.binary_cross_entropy_with_logits</code> takes the logit and the 0/1 target and computes <TeX tex="-\log\sigma(\hat z)" /> for a
          positive and <TeX tex="-\log(1-\sigma(\hat z))" /> for a negative, rearranged so that no exponential can overflow{' '}
          <Cite kind="code" ref="rt/model.py:253-255" /> <Cite kind="spec" ref="§8.2" />:
        </p>
        <div className="sm:hidden">
          <MathBlock tex={BCE_TEX_NARROW} label={<Cite kind="spec" ref="§8.2" />} />
        </div>
        <div className="hidden sm:block">
          <MathBlock tex={BCE_TEX} label={<Cite kind="spec" ref="§8.2" />} />
        </div>
        <BceFigure lane={live.binary} />

        <h3>Huber loss</h3>
        <p>
          Number targets use <code>F.huber_loss</code> with its default δ = 1, on the residual between the predicted and the stored z-score{' '}
          <Cite kind="code" ref="rt/model.py:250-251" /> <Cite kind="spec" ref="§8.2" />. The paper does not say why it chose Huber; the curve shows
          what it does. Small errors are penalized like a squared error, large ones only linearly, so the gradient of a single cell never exceeds 1 in
          size: an unusual label, such as a product with a sudden burst of sales, pulls on the weights no harder than any other badly predicted cell.
        </p>
        <MathBlock tex={HUBER_TEX} label={<Cite kind="spec" ref="§8.2" />} />
        <HuberFigure lane={live.regression} />

        <h3>The mean over the masked cells of a batch</h3>
        <p>
          The code adds up the loss of every masked cell, type by type, and divides by the number of masked cells, <code>loss_out / masks.sum()</code>{' '}
          <Cite kind="code" ref="rt/model.py:260-265" />:
        </p>
        <MathBlock tex={MEAN_TEX} label={<Cite kind="code" ref="rt/model.py:265" />} />
        <p>
          Each context has exactly one masked cell, so this is the plain mean over the contexts of the batch, 32 per GPU in the paper; DDP then
          averages the gradients over 8 GPUs, which gives the mean over 256 <Cite kind="spec" ref="§8.2" />. There is no weight per task, per type or
          per class, and no label smoothing <Cite kind="spec" ref="§8.2" />. Pretraining shuffles the items of all its tasks into one list, so a batch
          can mix binary and regression cells <Cite kind="spec" ref="§4.14" />:
        </p>
        <BatchFigure binary={live.binary} regression={live.regression} model={g} />
        <CodeRef
          title="The loss: Huber, BCE against 1{y > 0}, mean over masked cells"
          file="rt/model.py"
          lines="250-267"
          code={MODEL_LOSS_250_267}
          highlightLines={[251, 253, 254, 257, 265]}
        />
        <Callout
          variant="paper-vs-code"
          title="Datetime and text targets"
          paper={
            <>
              “we only mask cells in boolean or numeric columns … HuberLoss(r, r′) for regression and … BCE(1{'{'}r &gt; 0{'}'}, r′) for binary
              classification” <Cite kind="paper" ref="p.6 §3.3" />
            </>
          }
          code={
            <>
              Huber for <code>number</code> <em>and</em> <code>datetime</code> targets; a <code>ValueError</code> for <code>text</code>{' '}
              <Cite kind="code" ref="rt/model.py:250-257" />
            </>
          }
          verdict="Identical for every task in the paper and on this site, whose targets are all boolean or number (spec §10 #15, #19). The site follows the code."
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 5 */}
      <Section id="metrics" title="Metrics: AUROC on logits, R² in z‑space" kicker="Step 5">
        <p>
          The paper scores binary tasks with <Term id="auroc">AUROC</Term> and regression tasks with <Term id="r2">R²</Term>{' '}
          <Cite kind="paper" ref="p.8 Tables 1-2" />. The evaluation loop collects the read output and the stored label of every evaluated seed, then
          calls scikit-learn once per task and split <Cite kind="code" ref="rt/main.py:328-337" />. Each seed gets one context and one forward pass,
          with no averaging over several contexts <Cite kind="spec" ref="§8.5" />: “a single forward pass” <Cite kind="paper" ref="p.1" />.
        </p>
        <CodeRef
          title="The metrics: r2_score on z-scores, roc_auc_score on raw logits"
          file="rt/main.py"
          lines="328-337"
          code={MAIN_METRICS_328_337}
          highlightLines={[333, 336, 337]}
          defaultOpen
        />

        <h3>AUROC</h3>
        <p>
          AUROC is the area under the ROC curve. It also has a direct reading: pick one seed whose label is 1 and one whose label is 0 at random;
          AUROC is the probability that the first gets the higher score, with ties counting half. Only the <em>order</em> of the scores matters. The
          sigmoid never changes the order of two logits, so AUROC on logits equals AUROC on probabilities, and the code can skip the sigmoid{' '}
          <Cite kind="spec" ref="§8.5" />. The figure evaluates every seed of the split in your browser.
        </p>
        <RocFigure lane={live.binary} />

        <h3>R²</h3>
        <p>
          R² compares the squared errors of the predictions with those of the simplest possible predictor, the mean label <TeX tex="\bar y" /> of the
          evaluated seeds:
        </p>
        <MathBlock tex={R2_TEX} label={<Cite kind="spec" ref="§8.5" />} />
        <p>
          Predicting <TeX tex="\bar y" /> for every seed scores exactly 0, which is why the paper’s Table 2 notes “Global mean baseline is 0.0”{' '}
          <Cite kind="paper" ref="p.8 Table 2" />. A negative R² is worse than that baseline. The code computes R² on z-scores. Converting labels and
          predictions to original units multiplies both sums by σ², and the ratio does not change <Cite kind="spec" ref="§8.5" />.
        </p>
        <R2Figure lane={live.regression} />
        <Callout variant="note" title="This evaluation and the paper’s">
          <p>
            The paper evaluates a fixed subset of at most 10,240 seeds per split and task, with labels and predictions in bfloat16, whose coarse
            rounding produces tied scores. Under 8-GPU evaluation the last batch of a small split can even count some seeds twice{' '}
            <Cite kind="spec" ref="§8.6" /> <Cite kind="spec" ref="§12" />. Here every seed is evaluated once, in float32, in one process. And the
            numbers are our mini model’s on synthetic data: the paper’s own results are in chapter 16.
          </p>
        </Callout>
        <ScaleNote
          paper="12 blocks, d = 256, L = 1024, RelBench test splits"
          ours={`4 blocks, d = 64, L = ${g.L}, ShopDB splits of a few dozen seeds`}
        />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 6 */}
      <Section id="shared" title="One decoder per type, shared by everything" kicker="Step 6">
        <p>
          There is one number head and one boolean head for the whole model, not one per column, table or database{' '}
          <Cite kind="code" ref="rt/model.py:115-131" /> <Cite kind="spec" ref="§8.1" />. The same handful of weights predicts the sales of a product,
          the rating of a review and, in the pretraining databases, the grade of a submission or the points of a driver. Its output is read in the
          target column’s own z-space, as a number of standard deviations above that column’s mean <Cite kind="spec" ref="§8.1" />. The head never
          learns which column it is predicting; that information reaches it only through <TeX tex="\hat x" />, which carries the column-name embedding
          of the masked token (<Link to="/embedding">chapter 8</Link>).
        </p>
        <SharedHeadsFigure />
        <p>
          This sharing is what lets a pretrained model answer for a column it has never seen. Our pretrained heads learned from the targets of
          SchoolDB, RacingDB and ForumDB, and are read on ShopDB without any weight update <Cite kind="spec" ref="§9.4" />. Per-column z-scoring is
          what lets one output scale serve them all: over the rows that define its statistics, every regression target has mean 0 and standard
          deviation 1.
        </p>
        <DecoderSizes />
        <Callout variant="note" title="The vector before the decoders">
          <p>
            The normalized vector <TeX tex="\hat x" /> at the masked position is useful on its own. The paper’s Table 3 trains only a small MLP on top
            of it (“RT as feature extractor”) <Cite kind="paper" ref="p.8 §4.4" />. That probe is not part of the released paper-era code{' '}
            <Cite kind="spec" ref="§9.5" />.
          </p>
        </Callout>
        <p>
          This was the last piece of the forward pass. <Link to="/end-to-end">The next chapter</Link> puts all of them together on one page, from the
          task row to this output, with every matrix and weight of a single prediction.
        </p>
      </Section>

      <div className="page-grid">
        <Quiz
          title="Check your understanding"
          questions={[
            {
              prompt: 'Is the sigmoid needed to compute AUROC?',
              choices: [
                {
                  text: 'Yes: AUROC needs probabilities between 0 and 1.',
                  why: 'AUROC only compares scores with each other; any real numbers work.',
                },
                {
                  text: 'No. The sigmoid keeps the order of the scores, and AUROC depends only on that order.',
                  correct: true,
                  why: 'That is why main.py passes the raw logits to roc_auc_score.',
                },
                {
                  text: 'Only when the two classes are imbalanced.',
                  why: 'Class balance does not change the fact that AUROC is a ranking metric.',
                },
              ],
              explanation: 'σ is strictly increasing, so every pair of seeds keeps its order and the ROC curve is identical (spec §8.5).',
            },
            {
              prompt: 'Which loss trains a number target such as product-sales?',
              choices: [
                {
                  text: 'Mean squared error in original units.',
                  why: 'The code never leaves z-space during training.',
                },
                {
                  text: 'Huber loss with δ = 1, on the z-scores.',
                  correct: true,
                  why: 'F.huber_loss(yhat, y) with the default δ = 1.0, where y is the stored z-score.',
                },
                {
                  text: 'BCE on 1{z > 0}.',
                  why: 'That is the loss of boolean targets.',
                },
              ],
              explanation: 'model.py:250-251 applies F.huber_loss to number (and datetime) targets; the residual is in units of the column’s σ.',
            },
            {
              prompt:
                'A pretraining batch holds 24 contexts with a boolean target and 8 with a number target. How much does each masked cell weigh in the loss?',
              choices: [
                {
                  text: '1/32 each, whatever its task or type.',
                  correct: true,
                  why: 'The loss is the sum over masked cells divided by masks.sum() = 32.',
                },
                {
                  text: 'The number cells weigh more, so that each type contributes half.',
                  why: 'The code has no per-type or per-task weighting.',
                },
                {
                  text: 'It depends on each column’s σ.',
                  why: 'σ only sets the scale of a number cell’s residual, not its weight in the mean.',
                },
              ],
              explanation: 'loss_out / masks.sum(): a plain mean over the masked cells of the batch (model.py:265, spec §8.2).',
            },
          ]}
        />
      </div>

      <ChapterNav slug="decoding" />
    </ChapterPage>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Live sentences
// ─────────────────────────────────────────────────────────────────────────────

function HeaderLine({ lane }: { lane: Lane }) {
  const ent = seedEntity(lane);
  const pred = lane.pred;
  return (
    <p className="mt-6 text-[17px] leading-relaxed text-ink-2">
      <span className="eyebrow mr-2">Live context</span>
      task <span className="text-ink">{taskLabel(lane.task)}</span>, seed <span className="text-ink">{ent.label}</span>
      {ent.name ? ` (${ent.name})` : ''}
      {pred && (
        <>
          {' · '}
          {pred.taskKind === 'binary' ? (
            <>
              logit <span className="num text-ink">{fmtS(pred.output, 3)}</span>, p ={' '}
              <span className="num text-ink">{fmt(pred.probability ?? NaN, 3)}</span>
            </>
          ) : (
            <>
              ẑ = <span className="num text-ink">{fmtS(pred.output, 3)}</span>, ŷ = <span className="num text-ink">{fmt(pred.value ?? NaN, 2)}</span>
            </>
          )}
        </>
      )}
      {' · '}checkpoint <span className="font-mono text-ink">{lane.ckpt}</span>
      {lane.assets.weightsKind === 'init' && (
        <>
          {' '}
          <StatusBadge status="warning">untrained weights</StatusBadge>
        </>
      )}{' '}
      <ScaleNote inline />
    </p>
  );
}

function BinarySentence({ lane }: { lane: Lane }) {
  const ent = seedEntity(lane);
  const p = lane.pred;
  if (!p) return <p>The forward pass for {taskLabel(lane.task)} is still running…</p>;
  const tok = lane.ctx.tokens[lane.ctx.targetPos];
  return (
    <p>
      {lane.isGlobal ? 'In the live context' : `In a local preview of ${taskLabel(lane.task)} (the global task is a regression task)`}, the seed is{' '}
      {ent.label}
      {ent.name ? ` (${ent.name})` : ''}. The task asks: “{lane.task.description}” The boolean head outputs{' '}
      <TeX tex={`\\hat z = ${fmtS(p.output, 3).replace('−', '-')}`} />, so <TeX tex={`\\sigma(\\hat z) = ${fmt(p.probability ?? NaN, 3)}`} />: the
      model gives a {fmt(100 * (p.probability ?? NaN), 1)}% chance that <code>{lane.task.targetCol}</code> is true. The hidden answer is{' '}
      <code>{String(tok?.raw)}</code>.
    </p>
  );
}

function RegressionSentence({ lane }: { lane: Lane }) {
  const ent = seedEntity(lane);
  const p = lane.pred;
  if (!p || !lane.stats) return <p>The forward pass for {taskLabel(lane.task)} is still running…</p>;
  const tok = lane.ctx.tokens[lane.ctx.targetPos];
  return (
    <p>
      {lane.isGlobal ? 'In the live context' : `The global task is binary, so here is a local preview of ${taskLabel(lane.task)}`}: the seed is{' '}
      {ent.label}
      {ent.name ? ` (${ent.name})` : ''}, and the task asks “{lane.task.description}” The number head outputs{' '}
      <TeX tex={`\\hat z = ${fmtS(p.output, 3).replace('−', '-')}`} />. With the Train statistics μ = {fmt(lane.stats.mean, 3)} and σ ={' '}
      {fmt(lane.stats.std, 3)}, that is <TeX tex={`\\hat y = ${fmt(p.value ?? NaN, 2).replace('−', '-')}`} /> {lane.task.targetCol}; the hidden answer
      is {String(tok?.raw)}.
    </p>
  );
}
