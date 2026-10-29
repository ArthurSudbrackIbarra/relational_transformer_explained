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
import { Link } from 'react-router-dom';
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
          why: 'Each cell is one token, built from its value, its column name and its table name.',
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
          why: 'That is fine-tuning (chapter 14). Zero-shot means no weight updates.',
        },
        {
          text: 'Only through the context window: the cells, names and links around the seed row',
          correct: true,
        },
        {
          text: 'From a natural-language description of the database and the task',
          why: 'The LLM baselines in the paper get such descriptions. RT gets none.',
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
            The Relational Transformer (RT) fills in one missing cell of a relational database it has never seen, in a single forward pass over a
            small window of cells. This chapter runs the whole pipeline once on a live example. The rest of the course then takes it apart.
          </>
        }
        learn={[
          'Why relational data is hard for existing models',
          'What RT takes in (cells of a database plus a task table) and what it puts out (one masked cell)',
          'Its three ideas: cell tokens, task-table prompting and relational attention',
          'What “zero-shot” means for RT, and the paper’s headline results',
          'How this site’s miniature differs from the paper’s model (size, data, training) and what it keeps (every operation)',
        ]}
      >
        <RunningExample live={live} />
      </ChapterHeader>

      {/* ─────────────────────────────── 1 ─────────────────────────────── */}
      <Section id="why" title="Why relational data is different">
        <p>
          In language and vision, a single pretrained transformer can move to a new task with little or no extra training. Relational data has no
          such model yet, even though most structured enterprise data lives in <Term id="relational-database">relational databases</Term>{' '}
          <Cite kind="paper" ref="p.1 §1" />. The trouble is the shape of the data. A database is several tables with columns of mixed types, and
          primary and foreign keys link rows across tables. The signal for one prediction is spread over rows, columns, linked tables and time{' '}
          <Cite kind="paper" ref="p.1 §1" />.
        </p>
        <p>
          Take the question this course follows.{' '}
          {ex && live.entityName ? (
            <>
              Will {live.entityName}, a customer of the synthetic shop ShopDB, place no order in the 30 days after {live.seedDate}? The customer’s own
              row in <code>customers</code> holds a name, a country, an age, a premium flag and a signup date. Most of the evidence sits in other
              tables: the {plural(f.ordersBefore, 'order')} placed before that date, the products those orders point to, the reviews the customer
              wrote, and whether they went quiet in earlier months.
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
        <p>There are four established ways to hand such a database to a model, and each of them loses something on the way.</p>
        <Figure
          wide
          title="Four ways to hand ShopDB to a model"
          caption="Each view is built from the live ShopDB around the running example. Switch tabs to compare them. RT reads the data differently from all four."
        >
          <Approaches live={live} />
        </Figure>
        <p>
          The paper concludes that none of them can be the backbone of a foundation model for relational databases{' '}
          <Cite kind="paper" ref="p.2 §1" />. RT is designed around three principles instead. It should capture the relational structure while
          keeping the natural symmetries of relational data, support flexible self-supervised pretraining, and generalize zero-shot to databases with
          other schemas <Cite kind="paper" ref="p.4 §3" />.
        </p>
      </Section>

      {/* ─────────────────────────────── 2 ─────────────────────────────── */}
      <Section id="pipeline" title="RT in one picture">
        <p>
          The figure below runs the whole pipeline on the live example in six stages. Press play or pick a stage; each stage links to the chapters
          that cover it in detail. Your browser computes all of it with the same simulation every chapter uses, so the seed, the context and the
          forward pass shown here come back in later chapters.
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
          path from cells to the final vectors is the 22 lines below. <Link to="/end-to-end">Chapter 13</Link> runs this same pass again, one operation at
          a time, with every matrix and weight on screen.
        </p>
        <CodeRef
          title="The forward pass: embedding, blocks, final norm"
          file="rt/model.py"
          lines="208-229"
          code={MODEL_FORWARD_208_229}
          highlightLines={[226, 227]}
          note={
            <>
              Lines 208-214 add the column-name term, 216-224 add each token’s value term (or the learned mask vector, for the masked cell), 226-227
              run the blocks, and 229 is the final RMSNorm. A decoder for the target’s type then turns the masked cell’s final vector into the
              prediction <Cite kind="code" ref="rt/model.py:238-239" />.
            </>
          }
        />
        <Callout variant="note" title="The masked cell keeps its name and loses its value">
          <p>
            The masked cell’s column name and type stay in the input, so the model knows which cell of which row to fill in. In place of the value,
            the embedding uses a learned mask vector, so the answer cannot affect the prediction <Cite kind="spec" ref="§5.3" />
            <Cite kind="code" ref="rt/model.py:216-224" />. Chapter 7 overwrites the true label and shows that the prediction stays the same.
          </p>
        </Callout>
      </Section>

      {/* ─────────────────────────────── 3 ─────────────────────────────── */}
      <Section id="ideas" title="Three ideas">
        <p>
          The paper lists three innovations <Cite kind="paper" ref="p.2 §1" />. This section shows each one on the live example.
        </p>

        <h3>1. Cells are tokens</h3>
        <p>
          RT “represents each database cell as a token, with embeddings constructed from its value, column name and table name”{' '}
          <Cite kind="paper" ref="p.2 §1" />. A <Term id="cell-token">cell token</Term> is a triple (v, c, t): value, column, table{' '}
          <Cite kind="paper" ref="p.4 §3.1" />. The value goes through a small encoder for its type: number, text, datetime or boolean. The column and
          table enter together as the text embedding of the phrase “<span>&lt;column&gt; of &lt;table&gt;</span>”, made by a
          frozen sentence encoder <Cite kind="paper" ref="p.5 §3.1" />
          . Key columns become links instead of tokens, and missing values are skipped <Cite kind="spec" ref="§1.2" />
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
        <p>
          In the paper’s notation a token’s embedding is{' '}
          <TeX tex="\mathbf{x} = \mathbf{W}_d\,\mathbf{r} + \mathbf{W}\,\mathcal{E}^{\text{schema}}(c,t)" />: the value{' '}
          <TeX tex="\mathbf{r}" />, projected by a matrix for its data type <TeX tex="d" />, plus a projection of the phrase’s text embedding. For a
          masked cell, a learned mask vector takes the place of the value term <Cite kind="paper" ref="p.5 §3.1" />. In the forward pass shown
          earlier, each projected term also passes through its own RMSNorm before the terms are added{' '}
          <Cite kind="code" ref="rt/model.py:208-224" />. Chapter 8 works through the formula with live numbers.
        </p>

        <h3>2. The task is a table</h3>
        <p>
          A forecasting task such as churn has no column in the database: its labels must be computed from future rows{' '}
          <Cite kind="paper" ref="p.3 §2.2" />. RT stores them in a <Term id="task-table">task table</Term>, one row per entity and timestamp, holding
          a foreign key to the entity, the timestamp and the label <Cite kind="paper" ref="p.4 §2.2" />. With{' '}
          <Term id="task-table-prompting">task-table prompting</Term>, that table is attached to the database. Only one task table is active at a
          time, and its rows are the seed rows. To the model it is just another table <Cite kind="paper" ref="p.4 §3.1" />.
        </p>
        <p>
          Attaching the task table pays off when the sampler explores around the seed. The entity’s earlier task rows are children of the entity’s
          row, so they enter the context like any other rows, labels included <Cite kind="spec" ref="§2.5" />. These are{' '}
          <Term id="in-context-label">in-context labels</Term>. Unlike few-shot prompting, nobody has to pick labeled examples for the model: the
          labels arrive because they sit in rows linked to the entity <Cite kind="paper" ref="p.2 §1" />.
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
        <Callout variant="insight" title="The entity’s own past labels matter most">
          <p>
            When the paper removes the target entity’s own past labels from the context, the mean zero-shot AUROC falls to 53.8, barely above random,
            and the mean R² falls to −5.5, worse than always predicting the mean. Removing other entities’ labels changes almost nothing{' '}
            <Cite kind="paper" ref="p.9 Table 4" />. RT still does more than average the past labels: it beats the EntityMean baseline, which
            predicts exactly that average <Cite kind="paper" ref="p.9 §5.1" />.
          </p>
        </Callout>

        <h3>3. Structure lives in the attention masks</h3>
        <p>
          The tokens themselves say nothing about how rows are linked. That structure enters through{' '}
          <Term id="relational-attention">relational attention</Term>: each block runs four attention layers with the same arithmetic and different
          masks. Column attention connects cells of the same column, feature attention connects a cell to its own row and the rows it points to,
          neighbor attention connects it to the rows that point to its row, and full attention connects everything{' '}
          <Cite kind="paper" ref="p.5 §3.2" />. There are no positional encodings, so the order of tables, rows and columns does not matter{' '}
          <Cite kind="paper" ref="p.5 §3.1" />.
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
          rows farther away also arrives along the links, over more than one layer. For example,{' '}
          {ex
            ? 'the customer’s cells read the orders that point to them in a neighbor layer, and the masked cell reads the customer’s cells in the next block’s feature layer'
            : 'a parent row’s cells read the rows that point to them in a neighbor layer, and the masked cell reads its parent row’s cells in the next block’s feature layer'}{' '}
          <Cite kind="spec" ref="§6.4" />. Chapters 9 to 11 follow those paths.
        </p>
        <CodeRef
          title="One block: four masked attentions, then the MLP"
          file="rt/model.py"
          lines="82-86"
          code={MODEL_BLOCK_82_86}
          note={
            <>
              Each sublayer normalizes the stream, computes its update and adds it back (a pre-norm residual), in the order column, feature, neighbor,
              full, then the MLP <Cite kind="spec" ref="§7.1" />.
            </>
          }
        />
      </Section>

      {/* ─────────────────────────────── 4 ─────────────────────────────── */}
      <Section id="zero-shot" title="What zero-shot means here">
        <p>
          The paper defines <Term id="zero-shot">zero-shot relational learning</Term> as “predicting new targets on a new RDB with a new schema,
          without weight updates. Information about the new RDB is available only at inference time, solely through the context window”{' '}
          <Cite kind="paper" ref="p.4 §2.3" />. To test this, the authors pretrain one model per target database, on all the tasks of all the other
          databases: a <Term id="leave-one-db-out">leave-one-database-out</Term> protocol <Cite kind="paper" ref="p.6 §4.1" />. This site runs one
          such split. It pretrains on SchoolDB, RacingDB and ForumDB and holds out ShopDB.
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
      </Section>

      {/* ─────────────────────────────── 5 ─────────────────────────────── */}
      <Section id="results" title="Headline results">
        <p>
          The paper’s RT has about 22 million parameters and is pretrained on RelBench databases. It is tested zero-shot on ten binary and eight
          regression forecasting tasks from databases it never saw <Cite kind="paper" ref="p.6 §4.1" />
          <Cite kind="paper" ref="p.8 Tables 1-2" />. On mean <Term id="auroc">AUROC</Term>, where 50 is random, it beats every other method in that
          setting, including the far larger Gemma3 language models and the <Term id="entity-mean">EntityMean</Term> baseline. Its mean zero-shot AUROC
          is 90% of what an RT fine-tuned on each task reaches, and 93% when the target database (but not the task) was part of pretraining{' '}
          <Cite kind="paper" ref="p.2 §1" />
          <Cite kind="paper" ref="p.21 Table 10" />.
        </p>
        <PaperHeadline />
        <p>This site’s miniature gets the same test on ShopDB, the database it never saw:</p>
        <MiniResults />
      </Section>

      {/* ─────────────────────────────── 6 ─────────────────────────────── */}
      <Section id="scale" title="The paper’s model and this site’s miniature">
        <p>
          The model in your browser is a complete RT: it runs every operation of the paper-era code. Only the scale differs. The model is
          smaller, it trains on four small synthetic databases instead of RelBench, and its training run is much shorter.
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
              and 12 blocks this adds up to 22,336,131 parameters <Cite kind="spec" ref="§7.3" />. The frozen text encoder is not among them: its
              vectors are computed once, ahead of time.
            </>
          }
        />
      </Section>

      {/* ─────────────────────────────── 7 ─────────────────────────────── */}
      <Section id="map" title="The course map">
        <p>
          Chapters 2 to 12 each open one stage of the picture above, in pipeline order, on the same live context. Chapter 13 puts the stages back
          together for one prediction. The last four step back: how the model is trained, what happens when you change its inputs, what the paper
          found, and how this simulation was checked against the official code.
        </p>
        <CourseMap current={SLUG} />
        <Quiz questions={quiz} />
      </Section>

      <ChapterNav slug={SLUG} />
    </ChapterPage>
  );
}
