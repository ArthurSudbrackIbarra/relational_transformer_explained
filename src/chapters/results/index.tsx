/**
 * Chapter 16 "results": What the paper found (BUILD_PLAN §6, docs/CHAPTER_BRIEFS.md §16).
 *
 * Every published number comes from ./data.ts, a transcription of the paper's tables checked against
 * the PDF text. Numbers derived from them (ratios, counts, differences) are computed at render time.
 * The few panels about this site's miniature read public/data/training/results.json or run the live
 * sampler, and carry a "This site" badge and a ScaleNote.
 * Spec: research/RT_TECHNICAL_SPEC.md §9.4-9.5, §10, §11.
 */
import { Link } from 'react-router-dom';
import { Callout, ChapterHeader, ChapterNav, ChapterPage, Cite, CodeRef, Figure, Quiz, ScaleNote, Section, Term } from '@/components';
import { AblationExplorer } from './Ablation';
import { ATTENTION_SPEC, CONTEXT_SPEC, MASKING_SPEC, ORDER_SPEC } from './ablationSpecs';
import { LiveLabelStats, MiniLabelAblation, SelfLabelScatter } from './ContextFigures';
import { TABLE1, TABLE10, TABLE14, TABLE15, TABLE2, TABLE4 } from './data';
import { Discrepancies } from './Discrepancies';
import { Drivers } from './Drivers';
import { Fig3InWords, SupervisedExplorer } from './Efficiency';
import { CostQuality, FeatureExtraction } from './FeatureCost';
import { f1, SourceBadge } from './kit';
import { LimitationCards, TwoForeignKeys } from './Limitations';
import { MiniZeroShot } from './MiniPanel';
import { GlossaryList, References } from './Reference';
import { MODEL_F2P_MASKS_170_178 } from './snippets';
import { rtVsEntityMean, SupervisedRatio, ZeroShotMeans, ZeroShotTasks } from './ZeroShot';

const m1 = (col: string) => TABLE1.mean[TABLE1.cols.indexOf(col as never)] as number;
const m2 = (col: string) => TABLE2.mean[TABLE2.cols.indexOf(col as never)] as number;
const T4 = Object.fromEntries(TABLE4.rows.map((r) => [r.id, r])) as Record<string, (typeof TABLE4.rows)[number]>;

export default function ResultsChapter() {
  const clfVsEm = rtVsEntityMean('clf');
  const regVsEm = rtVsEntityMean('reg');
  const ordAuroc = TABLE15.clf!.mean.zs;
  const ordR2 = TABLE15.reg!.mean.zs;
  const t10 = (m: 'clf' | 'reg', c: string) => TABLE10[m].mean[TABLE10[m].cols.indexOf(c as never)] as number;

  return (
    <ChapterPage slug="results">
      <ChapterHeader
        slug="results"
        lede={
          <>
            The earlier chapters showed how RT turns one masked cell into a prediction. This chapter collects what the paper measured: how well RT
            predicts on databases it has never seen, how quickly it learns once fine-tuned, which parts of the context and the architecture the
            transfer depends on, and what RT cannot do yet.
          </>
        }
        learn={[
          'How RT’s zero-shot results compare with LLMs, Griffin and the EntityMean baseline, and which comparison gives the headline “93%”',
          'How fast RT learns when fine-tuned, how well it works as a frozen feature extractor, and what a shorter context costs',
          'Which context and attention ablations move the numbers, and the paper’s ranked drivers of transfer',
          'The limitations the authors name, and the few places where the released code differs from the paper',
        ]}
      >
        <div className="mt-6 rounded-xl border border-border bg-surface px-4 py-3.5 text-[17px] leading-relaxed text-ink-2">
          <span className="eyebrow mr-2">How to read the figures</span>
          A <SourceBadge kind="paper" /> figure shows numbers typed from the paper’s tables as printed (in %, one decimal). Anything derived from
          them, such as a ratio or a difference, is computed on this page and labeled as such. A <SourceBadge kind="site" /> figure shows this site’s
          small model or the live sampler instead <ScaleNote inline />. The two never share a chart.
        </div>
      </ChapterHeader>

      {/* ───────────────────────────────────────────────────────────── 1 */}
      <Section id="zero-shot" title="Zero-shot classification" kicker="Table 1">
        <p>
          <Term id="zero-shot">Zero-shot</Term> has a precise meaning in the paper: “predicting new targets on a new RDB with a new schema, without
          weight updates. Information about the new RDB is available only at inference time, solely through the context window”{' '}
          <Cite kind="paper" ref="p.4 §2.3" />. In every column of Table 1 the target task is unseen during pretraining{' '}
          <Cite kind="paper" ref="p.8 Table 1" />. The columns differ in what else the model has seen:
        </p>
        <ul>
          <li>
            <strong>No</strong>: the target database was held out entirely. RT is pretrained on the six other RelBench databases (
            <Term id="leave-one-db-out">leave-one-database-out</Term>) <Cite kind="paper" ref="p.6 §4.1" />.
          </li>
          <li>
            <strong>Yes</strong>: starting from that model, RT continues pretraining on the target database’s <em>other</em> tasks. The target task
            itself stays unseen <Cite kind="paper" ref="p.7 §4.3" /> <Cite kind="spec" ref="§9.3" />.
          </li>
          <li>
            <strong>Maybe</strong>: the Gemma3 LLMs, whose pretraining data “is unknown, but likely includes similar public datasets”{' '}
            <Cite kind="paper" ref="p.7 §4.3" />. They read the same context subgraph serialized as JSON, plus dataset and task descriptions{' '}
            <Cite kind="paper" ref="p.24 App. I.2" />.
          </li>
        </ul>
        <p>
          The non-neural <Term id="entity-mean">EntityMean</Term> baseline predicts the mean of the target entity’s past labels{' '}
          <Cite kind="paper" ref="p.23 App. I.1" />. Those past labels carry a lot of signal, which makes it a hard baseline to beat (the context
          ablations below come back to this). Random guessing scores 50.
        </p>
        <ZeroShotMeans metric="clf" />
        <p>
          Among the methods that never saw the target database, RT (No) has the highest mean, {f1(m1('rtNo'))}, ahead of EntityMean (
          {f1(m1('entityMean'))}) and Griffin ({f1(m1('griffinNo'))}); the largest LLM reaches {f1(m1('gemma27b'))}. In the paper’s words, RT
          “attains the best average AUROC and is the only method to consistently beat the entity mean baseline” <Cite kind="paper" ref="p.7 §4.3" />.
          Task by task the picture is more mixed: EntityMean is higher on {clfVsEm.losses.length} of {clfVsEm.total} tasks (EntityMean vs RT:{' '}
          {clfVsEm.losses.join('; ')}).
        </p>
        <ZeroShotTasks metric="clf" />
        <SupervisedRatio />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 2 */}
      <Section id="zero-shot-regression" title="Zero-shot regression" kicker="Tables 2 and 16">
        <p>
          Regression is scored with <Term id="r2">R²</Term>: 0 means “as good as predicting the global mean”, and negative values are worse than that{' '}
          <Cite kind="paper" ref="p.8 Table 2" />. RT (No) averages {f1(m2('rtNo'))} and RT (Yes) {f1(m2('rtYes'))}. The paper sums it up: “RT is the
          only model to consistently achieve positive R² and surpass the EntityMean baseline on average” <Cite kind="paper" ref="p.8 §4.3" />.
        </p>
        <ZeroShotMeans metric="reg" />
        <p>
          EntityMean still wins {regVsEm.losses.length} of {regVsEm.total} tasks outright (EntityMean vs RT: {regVsEm.losses.join('; ')}). The LLMs
          fail almost everywhere: with the same context, the Gemma3 models score below −9 on most tasks. The exception is f1 driver-pos, where
          Gemma3-27B ties RT (No) at 52.4 <Cite kind="paper" ref="p.26 Table 16" />. The paper attributes the failure to “unconstrained number
          generation and a context not optimized for LLM regression” <Cite kind="paper" ref="p.25 App. I.3" />.
        </p>
        <ZeroShotTasks metric="reg" />
        <p>
          This site’s miniature runs the same kind of zero-shot evaluation on ShopDB. Its numbers come from a much smaller model on a synthetic
          database, so they get their own panel and never share the paper’s axes.
        </p>
        <MiniZeroShot />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 3 */}
      <Section id="efficiency" title="Learning efficiency" kicker="Figure 3 and Table 10">
        <p>
          Zero-shot is the first point of a longer curve. “The key promise of pretraining is efficient adaptation to new datasets and tasks”{' '}
          <Cite kind="paper" ref="p.7 §4.2" />. On each unseen task the paper fine-tunes for 33k steps at a constant learning rate of 10⁻⁴ with no
          weight decay <Cite kind="paper" ref="p.6 §4.1" />, and compares pretrained RT with untrained RT, an untrained RDL-GNN and Griffin{' '}
          <Cite kind="paper" ref="p.7 §4.2" />.
        </p>
        <Fig3InWords />
        <p>
          Table 10 shows where that curve ends: every model fine-tuned on the full training set until it converges. Pretrained RT has the best mean
          on both metrics, AUROC {f1(t10('clf', 'rtYes'))} and R² {f1(t10('reg', 'rtYes'))} <Cite kind="paper" ref="p.21 Table 10" />. On
          classification, RelGNN and RelLLM ({f1(t10('clf', 'relGnn'))}) come within 0.1 points and win several tasks; the paper notes that they
          “utilize custom setups for each task, whereas RT uses a single unified hyperparameter setup” <Cite kind="paper" ref="p.20 App. F.1" />. On
          regression the gap is wider: the next-best means are untrained RT ({f1(t10('reg', 'rtNo'))}) and untrained Griffin (
          {f1(t10('reg', 'griffinNo'))}).
        </p>
        <SupervisedExplorer />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 4 */}
      <Section id="context-ablations" title="What the context contributes" kicker="Tables 4, 11 and 12">
        <p>
          To find out where zero-shot transfer comes from, the authors keep the pretrained checkpoints and change the context: they shuffle the column
          and table names, remove the target entity’s past task rows (<Term id="self-label">self labels</Term>), or remove other entities’ task rows
          (other labels) <Cite kind="paper" ref="p.9 §5.1" />. They repeat each ablation with fine-tuning on the modified contexts{' '}
          <Cite kind="paper" ref="p.22 App. G" />. The matrix holds Table 4’s means; click any cell to see it task by task.
        </p>
        <Figure
          wide
          title={
            <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
              Context ablations <SourceBadge kind="paper">Tables 4 and 11 · p.9, p.22</SourceBadge>
            </span>
          }
        >
          <AblationExplorer spec={CONTEXT_SPEC} initial={{ variant: 'selfLabels', setting: 'zs', metric: 'clf' }} />
        </Figure>
        <p>
          Self labels dominate. Without them, zero-shot AUROC falls from {f1(T4.none.zsClf)} to {f1(T4.selfLabels.zsClf)} and R² from{' '}
          {f1(T4.none.zsReg)} to {f1(T4.selfLabels.zsReg)}, worse than predicting the mean. “Zero-shot transfer is driven primarily by the model’s
          ability to leverage past labels of the target entity” <Cite kind="paper" ref="p.9 §5.1" />. RT does more than average those labels,
          though: it beats EntityMean, which does exactly that <Cite kind="paper" ref="p.9 §5.1" />. After fine-tuning, the ablations barely move the
          means, except that removing self labels still costs regression {f1(T4.none.ftReg - T4.selfLabels.ftReg)} points of R²{' '}
          <Cite kind="paper" ref="p.9 §5.1" />.
        </p>
        <p>
          If self labels drive transfer, tasks whose contexts hold none should not suffer when they are removed. The paper’s own label counts (Table
          12) make this easy to check.
        </p>
        <SelfLabelScatter />
        <p>
          The paper’s Figure 5 asks a follow-up question: without self labels, does pretraining still help fine-tuning? It compares fine-tuning
          curves of pretrained RT with and without self labels, and of untrained RT without them. The gap between the two runs without self labels
          is the transfer that remains <Cite kind="paper" ref="p.9 Fig. 5" />. The authors conclude that RT does learn other transferable patterns,
          “but self labels do constitute the dominant share of transfer from pretraining even during initial steps of fine-tuning”{' '}
          <Cite kind="paper" ref="p.9 §5.1" />.
        </p>
        <h3>The same statistic on this site</h3>
        <p>
          Self labels exist because <Term id="task-table-prompting">task-table prompting</Term> puts the entity’s past task rows within reach of the
          sampler, and column attention lets the masked cell read their labels. The first panel below counts them for the live task, as Table 12
          does. The second shows what this site’s miniature does when its contexts hold no labels at all.
        </p>
        <LiveLabelStats />
        <MiniLabelAblation />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 5 */}
      <Section id="architecture-ablations" title="What the architecture contributes" kicker="Tables 5, 13, 14 and 15">
        <p>
          The architecture ablations retrain RT with one of the four attention layer types removed from every block, adding blocks to keep the
          parameter count constant <Cite kind="paper" ref="p.9 §5.2" />.
        </p>
        <Figure
          wide
          title={
            <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
              Attention-layer ablations <SourceBadge kind="paper">Tables 5 and 13 · p.10, p.23</SourceBadge>
            </span>
          }
        >
          <AblationExplorer spec={ATTENTION_SPEC} initial={{ variant: 'col', setting: 'zs', metric: 'reg' }} />
        </Figure>
        <p>
          <Term id="column-attention">Column attention</Term> matters most for zero-shot transfer: without it, zero-shot R² drops from 22.8 to 11.9{' '}
          <Cite kind="paper" ref="p.10 Table 5" />. After fine-tuning, no ablation costs more than two points of mean R²{' '}
          <Cite kind="paper" ref="p.10 Table 5" />, and on classification the differences are small in both settings{' '}
          <Cite kind="paper" ref="p.23 Table 13" />. Full attention matters least: “removing full attention has the least
          impact in both zero-shot and fine-tuned settings, even significantly improving the results on some tasks”{' '}
          <Cite kind="paper" ref="p.9 §5.2" />.
        </p>
        <Callout variant="warning" title="The paper retrains; the playground does not">
          <p>
            Chapter 15’s playground can switch an attention layer off in the trained mini model at inference time. That shows how a trained model
            behaves when part of it goes missing. Table 5 is a different experiment: each variant there is pretrained without the layer from the
            start <Cite kind="paper" ref="p.9 §5.2" />. <Link to="/playground">Open the playground</Link>.
          </p>
        </Callout>

        <h3>Does the order of the layers matter?</h3>
        <p>
          RT’s block runs column → feature → neighbor → full attention, or “CFN” in Table 15’s letters (full attention always comes last, so the
          letters leave it out) <Cite kind="code" ref="rt/model.py:83" />. Table 15 trains every order, plus a variant that runs the three relational
          layers in parallel and sums their outputs <Cite kind="paper" ref="p.24 Table 15" />. Zero-shot means span only{' '}
          {f1(Math.min(...ordAuroc))}–{f1(Math.max(...ordAuroc))} AUROC and {f1(Math.min(...ordR2))}–{f1(Math.max(...ordR2))} R²: “the specific
          order of relational attention layers is not critical to performance” <Cite kind="paper" ref="p.23 App. H.1" />.
        </p>
        <Figure
          wide
          title={
            <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
              Attention order, zero-shot <SourceBadge kind="paper">Table 15 · p.24</SourceBadge>
            </span>
          }
        >
          <AblationExplorer spec={ORDER_SPEC} initial={{ variant: 'FNC', setting: 'zs', metric: 'clf' }} />
        </Figure>

        <h3>Masking more than one cell</h3>
        <p>
          RT as released masks exactly one cell per context, the seed’s target <Cite kind="code" ref="rustler/src/fly.rs:492-496" />, and this site
          does the same. Table 14 pretrains with extra cells masked too, each with probability P(mask) and each adding to the loss{' '}
          <Cite kind="paper" ref="p.24 Table 14" />. More masking lowers zero-shot R² ({f1(TABLE14.reg!.mean.zs[0])} →{' '}
          {f1(TABLE14.reg!.mean.zs[1])} → {f1(TABLE14.reg!.mean.zs[2])}) but raises fine-tuned R² ({f1(TABLE14.reg!.mean.ft![0])} →{' '}
          {f1(TABLE14.reg!.mean.ft![1])} at 0.2, {f1(TABLE14.reg!.mean.ft![2])} at 0.4). The “no pretraining” column fine-tunes from random
          initialization.
        </p>
        <Figure
          wide
          title={
            <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
              Multi-cell masking <SourceBadge kind="paper">Table 14 · p.24</SourceBadge>
            </span>
          }
        >
          <AblationExplorer spec={MASKING_SPEC} initial={{ variant: 'p02', setting: 'zs', metric: 'reg' }} />
        </Figure>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 6 */}
      <Section id="features-cost" title="RT as a feature extractor, and the cost of context" kicker="Table 3 and Figure 4">
        <p>
          Instead of fine-tuning all of RT, you can freeze it and train a small head on its output. The paper takes “the masked token representation
          before the decoder layer from a pretrained RT” and trains only a 2-layer MLP on it <Cite kind="paper" ref="p.8 §4.4" />. Table 3 covers
          five classification and four regression tasks, so its means are not comparable with Table 10’s averages over all tasks{' '}
          <Cite kind="paper" ref="p.8 Table 3" />.
        </p>
        <FeatureExtraction />
        <p>
          The MLP on pretrained RT beats full fine-tuning on R² and comes within 1.6 AUROC points of it. Even untrained RT embeddings are “sometimes
          competitive, and much better than untrained GNN embeddings” <Cite kind="paper" ref="p.8 §4.4" />.
        </p>

        <h3>Paying for context</h3>
        <p>
          Attention cost grows with the context length L. Figure 4 evaluates RT zero-shot with shorter contexts than it was pretrained with, and
          varies Griffin’s neighbor sampling (fanout:hops) the same way <Cite kind="paper" ref="p.8 §4.5" />.
        </p>
        <CostQuality />
        <p>
          The playground lets you shrink L for the live seed and watch the prediction change (<Link to="/playground">chapter 15</Link>).
        </p>
        <ScaleNote paper="pretrained and evaluated at L = 1024" ours="pretrained and evaluated at L = 128" />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 7 */}
      <Section id="drivers" title="The ranked drivers of zero-shot transfer" kicker="Paper §7">
        <p>
          The paper ends its discussion with the components that enable zero-shot transfer, listed “in order of importance”{' '}
          <Cite kind="paper" ref="p.10 §7" />. Each card pairs the authors’ words with the ablation behind them and the chapter where you met that
          component.
        </p>
        <Drivers />
        <Callout variant="insight" title="Why could transfer work at all?">
          <p>
            Before running the experiments, the authors expected transfer across domains because databases “share structural commonalities, e.g.,
            dimension tables, fact tables, hubs and tripartite structures [4], as well as semantic similarities, e.g., similar user behaviors”{' '}
            <Cite kind="paper" ref="p.6 §4.1" />. The ablations point to a more direct source. In a forecasting task, the entity’s own past labels
            sit in the context as visible cells of the label column, and column attention lets the masked cell read each of them{' '}
            <Cite kind="spec" ref="§2.5" /> <Cite kind="spec" ref="§6.4" />.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 8 */}
      <Section id="limitations" title="Limitations" kicker="Paper §7">
        <p>The authors name three limitations and one direction for future work.</p>
        <LimitationCards />
        <TwoForeignKeys />
        <CodeRef
          title="The feature and neighbor masks only test membership in the parent list"
          file="rt/model.py"
          lines="170-178"
          code={MODEL_F2P_MASKS_170_178}
          highlightLines={[171, 176]}
          note=".any(-1) reduces over the five parent slots: which slot matched is never used."
        />
        <h3>Scope of the evaluation</h3>
        <p>
          All results cover RelBench’s forecasting tasks. Autocomplete tasks are used in pretraining but are not part of the standard benchmark, and
          rel-event is skipped “as it has been found to have temporal leakage issues” <Cite kind="paper" ref="p.6 §4.1" />. Each run pretrains on six
          databases <Cite kind="paper" ref="p.10 §7" />.
        </p>
      </Section>

      {/* ───────────────────────────────────────────────────────────── 9 */}
      <Section id="discrepancies" title="Where the code differs from the paper" kicker="Paper vs code">
        <p>
          Where the paper and its released code differ, this course follows the code <Cite kind="spec" ref="§10" />. Most of the differences are
          small. The ones below change what RT computes or what its context holds, and each card links to the chapter that shows it.
        </p>
        <Discrepancies />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 10 */}
      <Section id="glossary" title="Glossary" kicker="Every term of the course">
        <p>Every term defined in this course, with its sources. Hover a dotted term anywhere on the site to see the same definition.</p>
        <GlossaryList />
      </Section>

      {/* ───────────────────────────────────────────────────────────── 11 */}
      <Section id="references" title="References" kicker="Sources">
        <References />
      </Section>

      <div className="page-grid">
        <Quiz
          title="Check your understanding"
          questions={[
            {
              prompt: 'The abstract says zero-shot RT averages 93% of fully supervised AUROC. Which comparison gives 93%?',
              choices: [
                {
                  text: 'RT on a database it never saw vs supervised fine-tuned RT: 69.7 / 77.2',
                  why: 'That ratio is 90.3%, the number the paper gives for unseen databases (p.2).',
                },
                {
                  text: 'RT with continued pretraining on the target database (target task unseen) vs supervised fine-tuned RT: 71.9 / 77.2',
                  correct: true,
                  why: '71.9 / 77.2 = 0.931. The paper states 93.1% for this setting (p.2).',
                },
                {
                  text: 'RT (No) vs Gemma3-27B: 69.7 / 64.6',
                  why: 'Gemma3-27B itself reaches 83.7% of supervised AUROC. The 93% divides an RT score by supervised RT.',
                },
              ],
              explanation: 'The headline number is the continued-pretraining setting. On a database RT has never seen, it is 90.3%.',
            },
            {
              prompt: 'Which change to the context hurt zero-shot performance the most?',
              choices: [
                { text: 'Shuffling column and table names', why: 'It hurt consistently but mildly: 70.1 → 69.5 AUROC, 22.8 → 20.5 R² (Table 4).' },
                {
                  text: 'Removing the target entity’s past labels (self labels)',
                  correct: true,
                  why: '70.1 → 53.8 AUROC and 22.8 → −5.5 R²: by far the largest drop (Table 4).',
                },
                { text: 'Removing other entities’ labels', why: 'The means did not drop: 70.6 AUROC and 22.9 R² (Table 4).' },
              ],
              explanation:
                'Self labels are the first of the paper’s ranked drivers. Where contexts contain none (rel-trial study-outcome and study-adverse), removing them changes nothing.',
            },
            {
              prompt: 'Without column attention, zero-shot R² fell from 22.8 to 11.9 (Table 5). How was that model obtained?',
              choices: [
                {
                  text: 'The trained RT with its column-attention sublayers switched off at inference',
                  why: 'That is what this site’s playground does. It is a different experiment from Table 5.',
                },
                {
                  text: 'Retrained without column attention, with extra blocks to keep the parameter count constant',
                  correct: true,
                  why: 'That is the paper’s setup (p.9 §5.2).',
                },
                {
                  text: 'Fine-tuned on the target task without column attention',
                  why: 'Table 5 reports zero-shot and fine-tuned numbers separately; the zero-shot ones involve no fine-tuning.',
                },
              ],
              explanation: 'Architecture ablations need retraining, because a model trained with a layer has learned to depend on it.',
            },
          ]}
        />
      </div>

      <ChapterNav slug="results" />
    </ChapterPage>
  );
}
