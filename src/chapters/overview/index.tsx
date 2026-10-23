/**
 * Chapter 1 · "overview": The big picture (BUILD_PLAN §6.1, docs/CHAPTER_BRIEFS.md §1).
 *
 * Motivation and a map: what RT predicts, what goes in, what comes out, and its three ideas, all on
 * the live running example (ShopDB, customer-churn, the store's seed). Every number comes from the
 * shared simulation hooks (src/state/hooks.ts) or from public/data; the paper's published numbers
 * (Tables 1, 2, 10) are quoted exactly. Spec: research/RT_TECHNICAL_SPEC.md §0, §9.4, §10 #26, §11.1.
 */
import { ArrowRightLeft, Crosshair } from 'lucide-react';
import { useMemo } from 'react';
import {
  Button,
  Callout,
  ChapterHeader,
  ChapterNav,
  ChapterPage,
  Cite,
  CodeRef,
  Figure,
  Quiz,
  ScaleNote,
  Section,
  StatusBadge,
  TableSwatch,
  TeX,
  Term,
  type QuizQuestion,
} from '@/components';
import { MINI_CONFIG, PAPER_CONFIG, PAPER_L, PAPER_W } from '@/engine/model';
import { useSim } from '@/state/store';
import { Approaches } from './Approaches';
import { CourseMap } from './CourseMap';
import { CellCards, LabelTimeline, MaskRows, useEntityTaskRows } from './Ideas';
import { int, plural, useOverviewLive, type OverviewLive } from './live';
import { PipelineFigure } from './PipelineFigure';
import { MiniResults, PaperHeadline } from './Results';
import { SameAndDifferent } from './Scale';
import { MODEL_BLOCK_82_86, MODEL_FORWARD_208_229, MODEL_INIT_104_151 } from './snippets';
import { AskTheModel, LeaveOneOut } from './ZeroShot';

const SLUG = 'overview';

/** 8 → "eight times", 2.5 → "2.5 times". */
function ratioText(r: number): string {
  const words = [
    '',
    'once',
    'twice',
    'three times',
    'four times',
    'five times',
    'six times',
    'seven times',
    'eight times',
    'nine times',
    'ten times',
  ];
  if (Math.abs(r - Math.round(r)) < 1e-9 && Math.round(r) < words.length) return words[Math.round(r)];
  return `${r.toFixed(1)} times`;
}

/** Facts about the running example used in the prose (all live). */
function useFacts(live: OverviewLive) {
  const { db, pdb, ctx, task, labels, masks } = live;
  return useMemo(() => {
    const seedTs = live.seed?.ts ?? null;
    let ordersBefore = 0;
    let ordersAll = 0;
    if (live.entity && seedTs !== null) {
      for (const e of pdb.p2fAdj[live.entity.nodeIdx] ?? []) {
        if (e.tableType !== 'Db') continue;
        const child = pdb.nodes[e.node];
        if (child.tableName !== 'orders') continue;
        ordersAll++;
        if (e.ts !== null && e.ts <= seedTs) ordersBefore++;
      }
    }
    const entityCells = live.entity?.cells.length ?? 0;
    const S = ctx.tokens.length;
    const q = ctx.targetPos;
    const count = (l: 'col' | 'feat' | 'nbr' | 'full') => {
      let n = 0;
      if (q >= 0) for (let k = 0; k < S; k++) if (masks[l][q * S + k]) n++;
      return n;
    };
    return {
      ordersBefore,
      ordersAll,
      entityCells,
      rows: ctx.visited.length,
      self: labels.self.length,
      other: labels.other.length,
      col: count('col'),
      feat: count('feat'),
      nbr: count('nbr'),
      full: count('full'),
      dbShort: db.title.split(' ')[0],
      targetPhrase: live.target ? pdb.text[live.target.colNameIdx] : `${task.targetCol} of ${task.name}`,
    };
  }, [live, db, pdb, ctx, task, labels, masks]);
}

function RunningExample({ live }: { live: OverviewLive }) {
  const setTask = useSim((s) => s.setTask);
  const { task, slots } = live;
  return (
    <div className="mt-6 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-xl border border-border bg-surface px-4 py-3 text-[17px] leading-snug">
      <Crosshair aria-hidden="true" size={16} className="shrink-0 text-accent-text" />
      <span className="min-w-0 flex-1 text-ink-2">
        Running example:{' '}
        {live.entityName ? (
          <>
            <span className="inline-flex items-center gap-1.5 align-middle">
              <TableSwatch slot={slots[live.entity?.tableName ?? ''] ?? 0} />
              <span className="font-semibold text-ink">{live.entityName}</span>
            </span>
            {live.country ? ` from ${live.country}` : ''}, task <code>{task.name}</code>, seed row dated {live.seedDate} ({live.seed?.tableType}{' '}
            split).
          </>
        ) : (
          <>
            {live.seedLabel?.long}, task <code>{task.id.split('/')[1]}</code>.
          </>
        )}{' '}
        Every figure in the course follows this seed; change it in the top bar. <ScaleNote inline />
      </span>
      {!live.isRunningExample && (
        <Button size="sm" variant="secondary" icon={<ArrowRightLeft size={14} />} onClick={() => setTask('shop/customer-churn')}>
          Back to ShopDB customer-churn
        </Button>
      )}
    </div>
  );
}

export default function OverviewChapter() {
  const live = useOverviewLive();
  const f = useFacts(live);
  const taskRows = useEntityTaskRows(live);
  const { task, ctx, assets } = live;
  const who = live.firstName ?? live.entityShort ?? 'the entity';
  const ex = live.isRunningExample;
  const nBlocks = assets.weights?.config.nBlocks ?? MINI_CONFIG.nBlocks;

  const quiz: QuizQuestion[] = [
    {
      prompt: 'What is a token in RT?',
      choices: [
        {
          text: 'A row of a table',
          why: 'Rows are what the sampler visits, but each row is then split into its cells.',
        },
        {
          text: 'A non-missing feature cell, together with its column and table name',
          correct: true,
          why: 'Each cell is one token, built from its value, its column name and its table name (p.2).',
        },
        {
          text: 'A whole table, summarized into one vector',
          why: 'No table-level summary exists anywhere in RT.',
        },
        {
          text: 'A word piece of the database written out as text',
          why: 'That is how an LLM would read it. RT only sends names and text values through its frozen text encoder.',
        },
      ],
      explanation: (
        <>
          In the live context,{' '}
          {live.entityShort
            ? `${live.entityShort}’s row gives ${plural(f.entityCells, 'token')}, one per non-missing feature cell; its primary key gives none.`
            : 'each row gives one token per non-missing feature cell; keys give none.'}{' '}
          <Cite kind="paper" ref="p.2 §1" />
          <Cite kind="spec" ref="§1.2" />
        </>
      ),
    },
    {
      prompt: 'RT is given a database it has never seen. How does it learn about that database?',
      choices: [
        {
          text: 'Its weights are first fine-tuned on the new database’s training rows',
          why: 'That is fine-tuning (chapter 13). Zero-shot means no weight updates.',
        },
        {
          text: 'Only through the context window: the cells, names and links around the seed row',
          correct: true,
        },
        {
          text: 'From a natural-language description of the database and the task',
          why: 'The LLM baselines get such descriptions; RT gets none (Table 1 caption, p.8).',
        },
        {
          text: 'From an encoder trained for that database’s tables',
          why: 'That would be schema-specific, the limitation RT is built to avoid.',
        },
      ],
      explanation: (
        <>
          “Information about the new RDB is available only at inference time, solely through the context window” <Cite kind="paper" ref="p.4 §2.3" />.
        </>
      ),
    },
    {
      prompt: (
        <>
          Full attention lets the masked <code>{task.targetCol}</code> cell see every token. Which other layer lets it read{' '}
          {ex ? `${who}’s` : 'the entity’s'} earlier labels directly?
        </>
      ),
      choices: [
        {
          text: 'Column attention',
          correct: true,
          why: `The labels sit in the same column of the same table as the masked cell: ${plural(f.col, 'key')} in the live context.`,
        },
        { text: 'Feature attention', why: `It covers the seed row and its parent row (${plural(f.feat, 'key')} here), not other task rows.` },
        { text: 'Neighbor attention', why: 'It reads rows that point to the masked cell’s row, and nothing points to a task row: no keys at all.' },
        { text: 'None: every label in the context is hidden', why: 'Only the seed row’s own label is masked; the other rows’ labels stay visible.' },
      ],
      explanation: (
        <>
          Column attention allows a pair only when both cells come from the same column of the same table{' '}
          <Cite kind="code" ref="rt/model.py:181-183" />, so for the masked cell it keeps exactly the visible {task.targetCol} cells. Full attention
          reaches them too, mixed in with everything else.
        </>
      ),
    },
  ];

  return (
    <ChapterPage slug={SLUG}>
      <ChapterHeader
        slug={SLUG}
        lede={
          <>
            The Relational Transformer fills in one missing cell of a relational database, for a database it has never seen, from a single forward
            pass over a small window of cells. This chapter runs the whole pipeline once on a live example, before the rest of the course takes it
            apart.
          </>
        }
        learn={[
          'Why relational data is hard for the models we already have',
          'What RT takes in (cells of a database plus a task table) and what it puts out (one masked cell)',
          'Its three ideas: cell tokens, task-table prompting and relational attention',
          'What “zero-shot” means for RT, and the paper’s headline results',
          'What this site’s miniature changes (sizes, data, precision) and what it keeps (every operation)',
        ]}
      >
        <RunningExample live={live} />
      </ChapterHeader>

      {/* ─────────────────────────────── 1 ─────────────────────────────── */}
      <Section id="why" title="Why relational data is different">
        <p>
          Pretrained transformers changed language and vision: one general architecture, trained on a lot of data, transfers to new tasks with little
          or no extra training. Relational data has no such model yet, although <Term id="relational-database">relational databases</Term> are the
          dominant store of structured enterprise information <Cite kind="paper" ref="p.1 §1" />. The difficulty is the shape of the data. A database
          is several tables with columns of mixed types, and rows are linked across tables by primary and foreign keys. The signal for a prediction is
          scattered across rows, columns, linked tables and time <Cite kind="paper" ref="p.1 §1" />.
        </p>
        <p>
          Take the question this course follows.{' '}
          {ex && live.entityName ? (
            <>
              Will {live.entityName}, a customer of the synthetic shop ShopDB, place no order in the 30 days after {live.seedDate}? The customer’s own
              row in <code>customers</code> holds a name, a country, an age, a premium flag and a signup date. The evidence is in other tables: the{' '}
              {plural(f.ordersBefore, 'order')} placed before that date, the products those orders point to, the reviews written, and whether the
              customer went quiet in earlier months.
            </>
          ) : (
            <>
              The simulation is currently set to <code>{task.id}</code>. Whatever the task, the evidence for one row sits in other rows of other
              tables, reached through keys.
            </>
          )}{' '}
          A model has to gather that evidence across keys and respect time: to predict at a time t, it may only use rows created before t{' '}
          <Cite kind="paper" ref="p.3 §2.1" />.
        </p>
        <p>There are four established ways to hand such a database to a model. Each keeps something and loses something.</p>
        <Figure
          wide
          title="Four ways to hand ShopDB to a model"
          caption="Every view is built from the live ShopDB around the running example. Switch tabs to compare; none of these is how RT reads the data."
        >
          <Approaches live={live} />
        </Figure>
        <p>
          The paper’s verdict is that none of these provides a viable backbone for a foundation model on relational databases{' '}
          <Cite kind="paper" ref="p.2 §1" />. RT is designed around three principles instead: capture the relational structure while keeping the
          natural symmetries of relational data, support flexible self-supervised pretraining, and generalize zero-shot across heterogeneous schemas{' '}
          <Cite kind="paper" ref="p.4 §3" />.
        </p>
      </Section>

      {/* ─────────────────────────────── 2 ─────────────────────────────── */}
      <Section id="pipeline" title="RT in one picture">
        <p>
          Here is the whole pipeline on the live example, in six stages. Press play or pick a stage; each one names the chapters that open it up.
          Everything you see is computed in your browser by the same simulation every chapter uses, so the seed, the context and the forward pass here
          are the ones you will meet again later.
        </p>
        <Figure
          wide
          title="One prediction, end to end"
          caption={
            <>
              Hover the dots and the tokens to see rows and cells. Attention steps show the keys the masked cell may attend to, shaded by the
              attention weights of the chosen block. {assets.weightsKind === 'init' && <StatusBadge status="warning">untrained weights</StatusBadge>}
            </>
          }
        >
          <PipelineFigure live={live} />
        </Figure>
        <ScaleNote
          paper={`L = ${PAPER_L}, w = ${PAPER_W}, ${PAPER_CONFIG.nBlocks} blocks`}
          ours={`L = ${ctx.params.L}, w = ${ctx.params.w}, ${nBlocks} blocks`}
        >
          A paper context holds {ratioText(PAPER_L / ctx.params.L)} as many cells, and the paper’s model is{' '}
          {ratioText(PAPER_CONFIG.nBlocks / nBlocks)} as deep and {ratioText(PAPER_CONFIG.dModel / MINI_CONFIG.dModel)} as wide. Every step above is
          the same.
        </ScaleNote>
        <p>
          In one sentence: RT reads a <Term id="context-window">context window</Term> of cells sampled around a <Term id="seed-row">seed row</Term>,
          and predicts the one masked cell of that row <Cite kind="spec" ref="§0" />. The prediction is either a logit, for a yes/no target like{' '}
          <code>churn</code>, or a z-score, for a number <Cite kind="spec" ref="§8.3" />. The model itself is short: once the masks are built, the
          path from cells to the final vectors is the 22 lines below.
        </p>
        <CodeRef
          title="The forward pass: embedding, blocks, final norm"
          file="rt/model.py"
          lines="208-229"
          code={MODEL_FORWARD_208_229}
          highlightLines={[226, 227]}
          note={
            <>
              Lines 208-214 add the column-name term, 216-224 add each token’s value term, or the learned mask vector for the masked cell, 226-227 run
              the blocks, and 229 is the final RMSNorm. The decoders then run at every position <Cite kind="code" ref="rt/model.py:238-239" />.
            </>
          }
        />
        <Callout variant="note" title="The answer is in the sequence, and still cannot leak">
          <p>
            The masked cell keeps its column name and its type, so the model knows which cell of which row it must fill in. Its true value even stays
            stored in the token, because the sampler writes every cell’s value; the embedding simply ignores it for the masked cell and uses the mask
            vector instead <Cite kind="spec" ref="§5.3" />
            <Cite kind="code" ref="rt/model.py:216-224" />. Chapter 7 changes the hidden label and shows that the prediction does not move.
          </p>
        </Callout>
      </Section>

      {/* ─────────────────────────────── 3 ─────────────────────────────── */}
      <Section id="ideas" title="Three ideas">
        <p>
          The paper names three innovations <Cite kind="paper" ref="p.2 §1" />. Each one below comes with the live example.
        </p>

        <h3>1. Cells are tokens</h3>
        <p>
          RT “represents each database cell as a token, with embeddings constructed from its value, column name and table name”{' '}
          <Cite kind="paper" ref="p.2 §1" />. A <Term id="cell-token">cell token</Term> is a triple (v, c, t): value, column, table{' '}
          <Cite kind="paper" ref="p.4 §3.1" />. The value goes through a small encoder for its type: number, text, datetime or boolean. The column and
          table enter together as the text embedding of the phrase “<span className="font-mono">&lt;column&gt; of &lt;table&gt;</span>”, made by a
          frozen sentence encoder <Cite kind="paper" ref="p.5 §3.1" />
          . Key columns produce no tokens, only links, and missing values produce nothing <Cite kind="spec" ref="§1.2" />
          <Cite kind="spec" ref="§3.5" />.
        </p>
        <Figure
          wide
          title={
            live.entityNode !== null ? 'What becomes of every cell of the seed row and its parent row' : 'What becomes of every cell of the seed row'
          }
          caption="Keys turn into links, missing values disappear, and every other cell becomes one token labeled with its phrase."
        >
          <CellCards live={live} />
        </Figure>
        <p>
          Why cells and not rows? A cell is the unit every database shares, whatever its schema. And once cells are tokens, any prediction task
          becomes “fill in this masked cell”, the same objective RT is pretrained on <Cite kind="paper" ref="p.2 §1" />
          <Cite kind="paper" ref="p.3 §2.2" />.
        </p>
        <Callout
          variant="paper-vs-code"
          title="The embedding formula"
          paper={
            <>
              <TeX tex="\mathbf{x} = \mathbf{W}_d\,\mathbf{r} + \mathbf{W}\,\mathcal{E}^{\text{schema}}(c,t)" />
              , with the value term replaced by a learned mask vector for masked cells <Cite kind="paper" ref="p.5 §3.1" />.
            </>
          }
          code={
            <>
              Each projection has a bias and is followed by its own RMSNorm; the mask vector is added as is{' '}
              <Cite kind="code" ref="rt/model.py:208-224" />
              <Cite kind="spec" ref="§10" />.
            </>
          }
        >
          <p>Chapter 8 takes this formula apart with live numbers.</p>
        </Callout>

        <h3>2. The task is a table</h3>
        <p>
          A forecasting task such as churn has no column in the database: its labels must be computed from future rows{' '}
          <Cite kind="paper" ref="p.3 §2.2" />. RT stores them in a <Term id="task-table">task table</Term>, one row per entity and timestamp, holding
          a foreign key to the entity, the timestamp and the label <Cite kind="paper" ref="p.4 §2.2" />. With{' '}
          <Term id="task-table-prompting">task-table prompting</Term>, that table is attached to the database. Only one task table is active at a
          time, its rows serve as seed rows, and to the model it is an ordinary table <Cite kind="paper" ref="p.4 §3.1" />.
        </p>
        <p>
          The payoff is in what the sampler finds next to the seed. The same entity’s earlier task rows are its children, so they enter the context
          like any other row, with their labels visible <Cite kind="spec" ref="§2.5" />: <Term id="in-context-label">in-context labels</Term>. The
          paper stresses that this is not few-shot learning, since no explicit subgraph-label pairs are required <Cite kind="paper" ref="p.2 §1" />.
          {ex && live.entityShort && f.self > 0 && (
            <>
              {' '}
              In the live context, {plural(f.self, `earlier ${task.targetCol} label`)} of {who} {f.self === 1 ? 'is' : 'are'} visible.
            </>
          )}
        </p>
        <Figure
          wide
          title={live.entityShort ? `Every ${task.name} row of ${live.entityShort}, on a time line` : 'Task rows on a time line'}
          caption={
            <>
              Rows dated after the seed never reach its context <Cite kind="spec" ref="§2.5" />. Hover a marker for its split and label.
            </>
          }
        >
          <LabelTimeline live={live} rows={taskRows} />
        </Figure>
        <Callout variant="insight" title="Past labels of the same entity drive zero-shot transfer">
          <p>
            In the paper’s context ablation, removing the target entity’s own past labels (its self labels) drops the mean zero-shot AUROC from 70.1 to 53.8 and the
            mean R² from 22.8 to −5.5. Removing other entities’ labels changes little (70.6 and 22.9) <Cite kind="paper" ref="p.9 Table 4" />. RT
            still does more than average them: it beats the EntityMean baseline, which predicts the mean of those past labels{' '}
            <Cite kind="paper" ref="p.9 §5.1" />.
          </p>
        </Callout>

        <h3>3. Structure lives in the attention masks</h3>
        <p>
          The database’s structure is not written into the tokens. It enters through <Term id="relational-attention">relational attention</Term>: each
          block runs four attention layers with the same arithmetic and different masks. Column attention connects cells of the same column, feature
          attention connects a cell to its own row and the rows it points to, neighbor attention connects it to the rows that point to its row, and
          full attention connects everything <Cite kind="paper" ref="p.5 §3.2" />. There are no positional encodings, so the order of tables, rows and
          columns does not matter <Cite kind="paper" ref="p.5 §3.1" />.
        </p>
        <Figure
          wide
          title="Who the masked cell may attend to, in each layer"
          caption={
            <>
              One strip per mask, one mark per token of the live context, in sequence order; colored marks are the allowed keys. Hover a strip to
              name a token.
            </>
          }
        >
          <MaskRows live={live} />
        </Figure>
        <p>
          On this context the masked cell sees {plural(f.col, 'token')} through column attention, {plural(f.feat, 'token')} through feature attention,{' '}
          {f.nbr === 0 ? 'none' : plural(f.nbr, 'token')} through neighbor attention and all {int(f.full)} through full attention. Information from
          elsewhere still arrives, over several steps:{' '}
          {ex
            ? 'the customer’s cells read the orders that point to them in a neighbor layer, and the masked cell reads the customer’s cells in the next block’s feature layer.'
            : 'a parent row’s cells read the rows that point to them in a neighbor layer, and the masked cell reads its parent row’s cells in the next block’s feature layer.'}{' '}
          That reading follows from the masks; it is not a claim of the paper <Cite kind="spec" ref="§6.4" />. Chapters 9 to 11 follow those paths.
        </p>
        <CodeRef
          title="One block: four masked attentions, then the MLP"
          file="rt/model.py"
          lines="82-86"
          code={MODEL_BLOCK_82_86}
          note={
            <>
              Each sublayer is a residual update of the normalized stream (pre-norm), in the order column, feature, neighbor, full{' '}
              <Cite kind="spec" ref="§7.1" />. The paper’s Algorithm 2 writes the norm after the attention instead; the code is what ran{' '}
              <Cite kind="paper" ref="p.26 Alg. 2" />.
            </>
          }
        />
      </Section>

      {/* ─────────────────────────────── 4 ─────────────────────────────── */}
      <Section id="zero-shot" title="What zero-shot means here">
        <p>
          The paper defines <Term id="zero-shot">zero-shot relational learning</Term> as “predicting new targets on a new RDB with a new schema,
          without weight updates. Information about the new RDB is available only at inference time, solely through the context window”{' '}
          <Cite kind="paper" ref="p.4 §2.3" />. To test it, the authors pretrain separately for each target database on all tasks of all the other
          databases, a <Term id="leave-one-db-out">leave-one-database-out</Term> protocol <Cite kind="paper" ref="p.6 §4.1" />. This site runs one
          such split: it pretrains on SchoolDB, RacingDB and ForumDB and holds out ShopDB.
        </p>
        <Figure wide title="Leave one database out, on this site">
          <LeaveOneOut weightsKind={assets.weightsKind} />
        </Figure>
        <p>
          At test time nothing is retrained. For each seed row of the new database the sampler builds a context, one forward pass runs with frozen
          weights, and the output at the masked cell is the prediction <Cite kind="spec" ref="§9.4" />. Whatever the model learns about ShopDB, it
          learns from that window: the cell values, the column and table names as text, which cells share a column, a row or a link, and the past
          labels of the task that happen to be in it <Cite kind="paper" ref="p.4 §2.3" />.
        </p>
        <Figure
          wide
          title="Ask the frozen model about another seed row"
          caption="Each pick samples that row’s context and runs one forward pass with the same weights. It does not change the rest of the course unless you press the button."
        >
          <AskTheModel live={live} />
        </Figure>
        <Callout variant="note" title="One caveat: the checkpoint is chosen on the target task’s validation labels">
          <p>
            No gradient step ever touches the target database. The checkpoint, though, is chosen per target task by its validation score, which reads
            that task’s validation labels. The paper does this because it found significant overfitting during pretraining{' '}
            <Cite kind="paper" ref="p.6 §4.1" />
            <Cite kind="spec" ref="§9.1" />. This site follows the same protocol.
          </p>
        </Callout>
      </Section>

      {/* ─────────────────────────────── 5 ─────────────────────────────── */}
      <Section id="results" title="Headline results">
        <p>
          Pretrained on RelBench databases, the paper’s RT has about 22 million parameters and is evaluated zero-shot on ten binary and eight
          regression forecasting tasks of databases it never saw <Cite kind="paper" ref="p.6 §4.1" />
          <Cite kind="paper" ref="p.8 Tables 1-2" />. Measured by <Term id="auroc">AUROC</Term>, where 50 is random, it beats every other method in
          the same setting, including the far larger Gemma3 language models and the <Term id="entity-mean">EntityMean</Term> baseline, on average over
          the tasks.
        </p>
        <PaperHeadline />
        <Callout variant="note" title="Which “93%”?">
          <p>
            The abstract says RT averages “93% of fully supervised AUROC”. That figure is the continued-pretraining setting, where the target database
            (not the task) was seen: 71.9 / 77.2. On a database never seen in pretraining the share is 69.7 / 77.2 = 90.3%. Both are in the paper{' '}
            <Cite kind="paper" ref="p.1" />
            <Cite kind="paper" ref="p.2 §1" />
            <Cite kind="spec" ref="§10" />. The ablation tables report 70.1 for the unseen-database setting because they come from an earlier run of
            the experiments; the headline tables are the later one <Cite kind="spec" ref="§10" />.
          </p>
        </Callout>
        <p>And here is what the miniature on this site achieves on ShopDB, the database it never saw:</p>
        <MiniResults />
      </Section>

      {/* ─────────────────────────────── 6 ─────────────────────────────── */}
      <Section id="scale" title="The paper’s model and this site’s miniature">
        <p>
          The model in your browser is a real RT, not a sketch of one. Every operation is the paper-era code’s; only the sizes, the data and the
          floating-point precision change.
        </p>
        <ScaleNote />
        <SameAndDifferent />
        <CodeRef
          title="Every learned module of the model"
          file="rt/model.py"
          lines="104-151"
          code={MODEL_INIT_104_151}
          note={
            <>
              Five encoders with their RMSNorms, four decoders, four mask vectors, the blocks and the final norm. With d = 256, 8 heads, d_ff = 1024
              and 12 blocks this adds up to 22,336,131 parameters <Cite kind="spec" ref="§7.3" />. The frozen text encoder is not part of the model:
              its vectors are computed once, beforehand.
            </>
          }
        />
      </Section>

      {/* ─────────────────────────────── 7 ─────────────────────────────── */}
      <Section id="map" title="The course map">
        <p>
          The next eleven chapters each open one stage of the picture above, in pipeline order, on the same live context. The last four step back: how
          the model is trained, what happens when you change its inputs, what the paper found, and how this simulation was checked against the
          official code.
        </p>
        <CourseMap current={SLUG} />
        <Quiz questions={quiz} />
      </Section>

      <ChapterNav slug={SLUG} />
    </ChapterPage>
  );
}
