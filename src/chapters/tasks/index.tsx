/**
 * Chapter 3 "tasks": predictive tasks and task tables (docs/CHAPTER_BRIEFS.md §3).
 *
 * Ground truth: research/RT_TECHNICAL_SPEC.md §2 (all), §4.9, §4.10; paper p.2-4, p.6, p.19;
 * PE/rt/tasks.py, PE/rt/data.py:77-92, PE/rustler/src/fly.rs:287-296, 415-424, 454-461, 492-496,
 * PE/rustler/src/pre.rs:152-174, 208-215, 424-437. All numbers come from the live engine.
 */
import { useEffect, useMemo, useState } from 'react';
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
  Segmented,
  TableView,
  Term,
} from '@/components';
import { tableColorSlot } from '@/engine/db';
import { getContext, getSeeds, inContextLabels, makeSampleParams, pickDefaultSeed } from '@/engine/pipeline';
import type { Split } from '@/engine/types';
import { useContextWindow, useDatabase, useInContextLabels, usePreprocessed, useTableSlots } from '@/state/hooks';
import { useSim } from '@/state/store';
import { useShallow } from 'zustand/react/shallow';
import { AutocompleteFigure } from './AutocompleteFigure';
import { ChurnBuilder } from './ChurnBuilder';
import { CodeTabs } from './CodeTabs';
import {
  DATA_TABLE_KEY,
  FLY_LEAKAGE_DROP,
  FLY_MASK,
  FLY_SEEDS,
  FLY_TASK_TABLE_SKIP,
  PRE_CAST_BOOL,
  PRE_NAME_INTERN,
  PRE_TABLE_TYPE,
  TASKS_AUTOCOMPLETE,
  TASKS_FORECAST,
  type Snippet,
} from './codeSnippets';
import { MaskedCellFigure } from './MaskedCellFigure';
import { PromptingFigure } from './PromptingFigure';
import { SalesFigure } from './SalesFigure';
import { SplitsTimeline } from './SplitsTimeline';
import { TaskRowTokens } from './TaskRowTokens';
import { SPLITS, entityName, isoDay } from './taskData';
import { useChapterExample } from './useExample';
import { useElementSize } from '@/components/viz/useSize';

const RELBENCH_TASK_BASE = 'https://github.com/snap-stanford/relbench/blob/v1.1.0/relbench/base/task_base.py';

/** Inline code inside a 16px figure caption: keeps the 14px caption floor (the global rule gives 0.86em). */
function CapCode({ children }: { children: React.ReactNode }) {
  return <code style={{ fontSize: 'max(14px, 0.9em)' }}>{children}</code>;
}

function Code({ s, title, note, highlight, defaultOpen }: { s: Snippet; title: string; note?: React.ReactNode; highlight?: number[]; defaultOpen?: boolean }) {
  return <CodeRef title={title} file={s.file} lines={[s.start, s.end]} code={s.code} note={note} highlightLines={highlight} defaultOpen={defaultOpen} />;
}

export default function TasksChapter() {
  const ex = useChapterExample();
  const { shop, shopPdb, churn, sales, rating, verified } = ex;
  const setTask = useSim((s) => s.setTask);
  const { L, w, rngSeed } = useSim(useShallow((s) => ({ L: s.L, w: s.w, rngSeed: s.rngSeed })));

  // local, chapter-only choices (never written to the global store)
  const [customer, setCustomer] = useState(ex.customer);
  const [product, setProduct] = useState(ex.product);
  const [review, setReview] = useState(ex.review);
  const [acTask, setAcTask] = useState(ex.liveTask.id === verified.id ? verified.id : rating.id);
  const [tableSplit, setTableSplit] = useState<Split>('Train');
  useEffect(() => setCustomer(ex.customer), [ex.customer]);
  useEffect(() => setProduct(ex.product), [ex.product]);
  useEffect(() => setReview(ex.review), [ex.review]);

  const shopSlots = useMemo(() => {
    const out: Record<string, number> = {};
    for (const t of shop.tables) out[t.schema.name] = tableColorSlot('shop', t.schema.name);
    for (const t of shop.tasks) if (t.kind === 'forecast') out[t.name] = tableColorSlot('shop', t.name);
    return out;
  }, [shop]);

  // ── live context for the last section: the live seed if the live task is a forecast task ──
  const liveDb = useDatabase();
  const livePdb = usePreprocessed();
  const liveCtx = useContextWindow();
  const liveSlots = useTableSlots();
  const liveLabels = useInContextLabels();
  const liveForecast = ex.liveTask.kind === 'forecast';
  const fallback = useMemo(() => {
    if (liveForecast) return null;
    const seeds = getSeeds(shopPdb, churn, 'Val');
    const seed = pickDefaultSeed(shopPdb, shop, churn, 'Val', seeds);
    const ctx = getContext(shopPdb, makeSampleParams(shopPdb, churn, seed, L, w, rngSeed));
    return { ctx, labels: inContextLabels(shopPdb, ctx, churn) };
  }, [liveForecast, shopPdb, shop, churn, L, w, rngSeed]);
  const rowDb = liveForecast ? liveDb : shop;
  const rowPdb = liveForecast ? livePdb : shopPdb;
  const rowCtx = liveForecast ? liveCtx : fallback!.ctx;
  const rowSlots = liveForecast ? liveSlots : shopSlots;
  const rowLabels = liveForecast ? liveLabels : fallback!.labels;
  const rowTask = liveForecast ? ex.liveTask : churn;
  const pastNode = useMemo(() => {
    const self = [...rowLabels.self].sort((a, b) => (b.ts ?? 0) - (a.ts ?? 0));
    return self[0]?.nodeIdx ?? null;
  }, [rowLabels]);
  const rowSeed = rowPdb.nodes[rowCtx.params.seedNode];

  // why other entities' labels are (often) absent: where the search met other entity rows (spec §2.5, §4.4)
  const reach = useMemo(() => {
    const entNode = rowSeed?.f2pNbrIdxs[0];
    const entTable = entNode !== undefined ? rowPdb.nodes[entNode].tableName : (rowTask.entityTable ?? 'entities');
    const entityWord = entTable.replace(/s$/, '');
    const others = rowCtx.visited.filter((v) => v.node !== entNode && rowPdb.nodes[v.node]?.tableName === entTable);
    const maxDepth = rowCtx.visited.reduce((m, v) => Math.max(m, v.depth), 0);
    const full = rowCtx.numReal >= rowCtx.params.L;
    const d = others.length ? Math.min(...others.map((o) => o.depth)) : null;
    const nOther = rowLabels.other.length;
    let text: React.ReactNode;
    if (nOther > 0) {
      text = (
        <>
          Those come from the task rows of other {entityWord}s, which the search reaches only through longer paths{' '}
          <Cite kind="spec" ref="§2.5" />.
        </>
      );
    } else if (full && d !== null && maxDepth <= d) {
      text = (
        <>
          The search did reach {others.length} other {entityWord}
          {others.length === 1 ? '' : 's'}, but only far from the seed ({d} hops), and the {rowCtx.params.L} slots filled up with closer
          rows before their task rows were taken <Cite kind="spec" ref="§4.4" /> <Cite kind="spec" ref="§2.5" />.{' '}
          <Link to="/sampling">Chapter 6</Link> shows the order in which the search takes rows.
        </>
      );
    } else if (full && d === null) {
      text = (
        <>
          Their task rows hang off other {entityWord} rows, and all {rowCtx.params.L} slots were full before the search reached any other{' '}
          {entityWord} <Cite kind="spec" ref="§2.5" />. <Link to="/sampling">Chapter 6</Link> shows the order in which the search takes
          rows.
        </>
      );
    } else {
      text = (
        <>
          Other {entityWord}s’ task rows can only be reached through other {entityWord} rows, further from the seed{' '}
          <Cite kind="spec" ref="§2.5" /> (<Link to="/sampling">Chapter 6</Link>).
        </>
      );
    }
    // who the seed is: "customer #35 Anna Müller’s Val row"
    const entRow = entNode !== undefined ? rowPdb.nodes[entNode].rowPos : null;
    const seedWho =
      entRow !== null ? `${entityWord} #${entRow} ${entityName(rowDb, entTable, entRow)}’s ${rowSeed?.tableType ?? ''} row` : null;
    return { entityWord, text, seedWho };
  }, [rowSeed, rowPdb, rowDb, rowCtx, rowTask, rowLabels]);

  // live seed timestamp for the splits figure (only meaningful for the live customer-churn seed)
  const liveSeedT = ex.liveTask.id === churn.id && ex.liveSeed >= 0 ? (shopPdb.nodes[ex.liveSeed]?.ts ?? 0) * 1000 : churn.valTimestamp!;

  // split identity table (live)
  const churnTargetIdx = shopPdb.columnIndex[`churn of ${churn.name}`];
  const churnTsIdx = shopPdb.columnIndex[`timestamp of ${churn.name}`];
  // rows sorted by node offset, i.e. in the order preprocessing assigns the blocks (pre.rs:79-92, 291-306)
  const splitIds = useMemo(
    () =>
      SPLITS.map((s) => {
        const info = shopPdb.tableInfo[`${churn.name}:${s}`];
        return { s, info, n: shopPdb.nodes[info.nodeIdxOffset] };
      }).sort((a, b) => a.info.nodeIdxOffset - b.info.nodeIdxOffset),
    [shopPdb, churn],
  );
  const [idWrapRef, idWrapSize] = useElementSize<HTMLDivElement>();
  const idNarrow = (idWrapSize.width || 800) < 560;
  const acSeedCounts = useMemo(
    () => Object.fromEntries(SPLITS.map((sp) => [sp, getSeeds(shopPdb, rating, sp).length])) as Record<Split, number>,
    [shopPdb, rating],
  );

  // finished task table view
  const tableInfo = shopPdb.tableInfo[`${churn.name}:${tableSplit}`];
  const tableRows = churn.tables![tableSplit].rows;
  const [tableWrapRef, tableWrapSize] = useElementSize<HTMLDivElement>();
  const tableNarrow = (tableWrapSize.width || 800) < 520;
  const customerRows = useMemo(
    () => tableRows.map((r, i) => (r[churn.entityCol!] === customer ? i : -1)).filter((i) => i >= 0),
    [tableRows, churn, customer],
  );

  const customersCount = shop.tables.find((t) => t.schema.name === 'customers')!.rows.length;
  const ordersCount = shop.tables.find((t) => t.schema.name === 'orders')!.rows.length;
  const churnRows = SPLITS.reduce((a, s) => a + churn.tables![s].rows.length, 0);
  const earliestTrain = useMemo(() => Math.min(...churn.tables!.Train.rows.map((r) => r.timestamp as number)), [churn]);
  const custName = entityName(shop, 'customers', customer);
  // a live example of leakage: the first one-star review with text
  const leakExample = useMemo(() => {
    const t = shop.tables.find((x) => x.schema.name === 'reviews');
    const i = t ? t.rows.findIndex((r) => r.rating === 1 && typeof r.review_text === 'string') : -1;
    return t && i >= 0 ? { row: i, text: t.rows[i].review_text as string, rating: t.rows[i].rating as number } : null;
  }, [shop]);

  // same-timestamp groups among the reviews (sampler resolution: 1 s; None == None also matches, spec §1.4, §4.9)
  const { reviewTimeDupes, reviewCount } = useMemo(() => {
    const info = shopPdb.tableInfo[`${rating.table}:Db`];
    const m = new Map<number | null, number>();
    for (let i = 0; i < info.numNodes; i++) {
      const ts = shopPdb.nodes[info.nodeIdxOffset + i]?.ts ?? null;
      m.set(ts, (m.get(ts) ?? 0) + 1);
    }
    return { reviewTimeDupes: [...m.values()].filter((k) => k > 1).length, reviewCount: info.numNodes };
  }, [shopPdb, rating]);

  return (
    <ChapterPage slug="tasks">
      <ChapterHeader
        slug="tasks"
        lede={
          <>
            RT answers every question about a database in the same way: hide one cell, then predict it. This chapter shows where that
            cell comes from, how a forecasting question becomes an ordinary table, and why the model needs nothing else to know what to
            predict.
          </>
        }
        learn={[
          'Why autocomplete and forecasting are both masked-cell prediction',
          'How the customer-churn task table is built: timestamps, windows (t, t + Δ], labels',
          'How the Train, Val and Test splits relate, and which labels a seed can reach',
          'Regression targets, autocomplete tasks and leakage columns',
          'Task-table prompting, and what a task row looks like to the model: two tokens',
        ]}
      >
        {!ex.liveIsShop && (
          <Callout variant="note" title="This chapter uses ShopDB" className="mt-6">
            <p>
              Your simulation is set to <strong>{liveDb.title}</strong>. The figures below teach ShopDB’s four tasks;{' '}
              {liveForecast
                ? 'the last section follows your live setting.'
                : 'the last section uses ShopDB’s default customer-churn seed, because your live task is not a forecasting task.'}
            </p>
            <Button size="sm" variant="secondary" onClick={() => setTask(churn.id)}>
              Switch the simulation to ShopDB · customer-churn
            </Button>
          </Callout>
        )}
      </ChapterHeader>

      {/* ───────────────────────────── 1 ───────────────────────────── */}
      <Section id="masked-cell" title="Every task is “fill in the masked cell”">
        <p>
          RT has a single job: predict the value of a <Term id="target-cell">masked cell</Term>. The paper calls this{' '}
          <strong>masked token prediction</strong>: “predict the value of a masked cell in the database, conditioned on the rest of the
          observed database” <Cite kind="paper" ref="p.3 §2.2" />. Two broad families of tasks on relational data fit this mould{' '}
          <Cite kind="paper" ref="p.3 §2.2" title="Autocomplete and forecasting tasks" />:
        </p>
        <ul>
          <li>
            <Term id="autocomplete-task">Autocomplete</Term>: the missing value belongs to a column that already exists, such as the star
            rating of a review.
          </li>
          <li>
            <Term id="forecasting-task">Forecasting</Term>: the value is about the future, such as whether a customer will order again.
            Such targets “do not exist in the original database and must be constructed from future rows”{' '}
            <Cite kind="paper" ref="p.3 §2.2" />.
          </li>
        </ul>
        <Figure
          wide
          title="Two ways a cell can be missing"
          caption={
            <>
              Live ShopDB rows. Left: the rating of review {review} is hidden, and the leakage column that would reveal it is dropped.
              Right: customer {customer}’s churn label is stored in no ShopDB table; it is computed from the orders after t.
            </>
          }
        >
          <MaskedCellFigure db={shop} pdb={shopPdb} ratingTask={rating} churnTask={churn} review={review} customer={customer} />
        </Figure>
        <p>
          Either way, each training or evaluation example masks exactly one cell: the target cell of its{' '}
          <Term id="seed-row">seed row</Term>. Every other cell that enters the context stays visible{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:492-496" /> <Cite kind="spec" ref="§4.10" />; autocomplete tasks first drop their
          leakage columns, as section 4 shows <Cite kind="spec" ref="§4.9" />. The type of that cell decides the loss: a boolean
          target is trained as binary classification, a numeric one as regression <Cite kind="paper" ref="p.6 §3.3" />{' '}
          <Cite kind="code" ref="rt/model.py:250-255" />.
        </p>
        <Callout
          variant="paper-vs-code"
          title="Which cells can be masked"
          paper={
            <>
              “In this work, we only mask cells in boolean or numeric columns as RelBench tasks are either binary classification or
              regression” <Cite kind="paper" ref="p.6 §3.3" />.
            </>
          }
          code={
            <>
              The loss follows the target cell’s type: number and datetime get a Huber loss, boolean gets binary cross-entropy, and a
              text target raises a <code>ValueError</code>, “masking text not supported” <Cite kind="code" ref="rt/model.py:250-257" />. A
              datetime target would train; no task uses one.
            </>
          }
          verdict="This site follows the code. ShopDB’s tasks, like the paper’s, use boolean and numeric targets only."
        >
          <p>
            See <Cite kind="spec" ref="§2.6" /> and <Cite kind="spec" ref="§10 #15" />.
          </p>
        </Callout>
      </Section>

      {/* ───────────────────────────── 2 ───────────────────────────── */}
      <Section id="build-churn" kicker="Forecasting" title="Building customer-churn, step by step">
        <p>
          A forecasting label needs a home, since no table stores it. RT takes the answer from RelBench: a{' '}
          <Term id="task-table">task table</Term>, which “stores the forecasting labels, together with foreign keys linking to the
          relevant entities (e.g., user IDs or item IDs) and a timestamp specifying the prediction horizon”{' '}
          <Cite kind="paper" ref="p.4 §2.2" />.
        </p>
        <p>
          ShopDB’s version is <code>customer-churn</code>: will the customer place <em>no</em> order in the next {churn.timedeltaDays} days?
          Each row has three
          columns: a <code>timestamp</code> t, a foreign key <code>customer_id</code>, and the label <code>churn</code>. Like RelBench task
          tables, it has no primary key <Cite kind="spec" ref="§2.1" />. The label summarises the future window (t, t + Δ], here with
          Δ = {churn.timedeltaDays} days <Cite kind="spec" ref="§2.2" />.
        </p>
        <Figure
          wide
          title="Building the task table for one customer"
          caption={
            <>
              Step through the construction, or click a timestamp. Pick any customer: this choice stays in this chapter. Timestamps and
              stored labels are read from the engine’s task table; the order counts are recomputed from the orders table to show which
              rows produce each label.
            </>
          }
        >
          <ChurnBuilder db={shop} task={churn} customer={customer} onCustomerChange={setCustomer} />
        </Figure>
        <p>
          Where do the timestamps come from? The dataset fixes a validation cutoff and a test cutoff, and training timestamps step back
          from the validation cutoff in steps of Δ, as in RelBench’s task code{' '}
          <Cite kind="url" ref={RELBENCH_TASK_BASE} label="RelBench task_base.py" title="RelBench BaseTask._get_table (v1.1.0)" />{' '}
          <Cite kind="spec" ref="§2.2" />.
        </p>
        <Callout variant="note" title="What is ours, not RT’s">
          <p>
            Two details belong to our synthetic ShopDB task, not to RT. RelBench steps back all the way to the database’s first timestamp;
            our generator stops one step sooner, so the earliest training timestamp ({isoDay(earliestTrain)}) lies at least Δ after the
            database’s first row ({isoDay(shop.minTimestamp)}), and the first window already has at least Δ of history behind it. And a
            customer gets a row at time t once they have placed at least one order at or before t, our stand-in for RelBench’s per-task
            rule for which entities are included. RT itself only ever reads the finished table.
          </p>
        </Callout>
        <ScaleNote
          paper={
            <>
              RelBench databases; rel-amazon alone has 20,862,040 reviews with a rating <Cite kind="paper" ref="p.19 Table 9" />
            </>
          }
          ours={`ShopDB: ${customersCount} customers, ${ordersCount.toLocaleString('en-US')} orders, ${churnRows.toLocaleString('en-US')} customer-churn rows`}
        >
          The task-table recipe is the same; only the amount of data differs.
        </ScaleNote>
        <p>The finished table, as the engine stores it. Rows of customer {customer} are highlighted.</p>
        <div ref={tableWrapRef} className="wide my-6">
          <div className="mb-2 flex flex-wrap items-center gap-3">
            <Segmented label="Split" value={tableSplit} onChange={setTableSplit} options={SPLITS.map((s) => ({ value: s, label: s }))} />
            <span className="text-[15px] text-ink-2">
              {customerRows.length > 0
                ? `${customerRows.length} row${customerRows.length === 1 ? '' : 's'} of #${customer} ${custName} in this split`
                : `#${customer} has no row in this split`}
            </span>
          </div>
          <TableView
            schema={churn.tables![tableSplit].schema}
            columns={tableNarrow ? ['timestamp', 'churn', 'customer_id'] : undefined}
            rows={tableRows}
            slot={shopSlots[churn.name]}
            title={`${churn.name} (${tableSplit})`}
            highlightRows={customerRows}
            scrollToRow={customerRows[0] ?? null}
            maxHeight={300}
            zValue={(r, c) => {
              const node = shopPdb.nodes[tableInfo.nodeIdxOffset + r];
              const cell = node?.cells.find((x) => x.colName === c);
              if (!cell) return null;
              return cell.sem === 3 ? cell.boolean : cell.sem === 2 ? cell.datetime : cell.sem === 0 ? cell.number : null;
            }}
            footer={
              <>
                customer_id is a foreign key: no token, only a link to the customer. The z-scored view shows the values as stored for the
                model (Chapter 4).{tableNarrow && ' On narrow screens the label column is listed before customer_id.'}
              </>
            }
          />
        </div>
      </Section>

      {/* ───────────────────────────── 3 ───────────────────────────── */}
      <Section id="splits" title="Three splits, one table">
        <p>
          RelBench gives every forecasting task three splits <Cite kind="spec" ref="§2.2" />: <strong>Train</strong> rows at the training
          timestamps, <strong>Val</strong> rows at the validation cutoff ({isoDay(churn.valTimestamp!)}) and <strong>Test</strong> rows at
          the test cutoff ({isoDay(churn.testTimestamp!)}). Each split is a separate file <Cite kind="code" ref="rustler/src/pre.rs:79-92" />,
          and RT gives each its own block of node indices <Cite kind="code" ref="rustler/src/pre.rs:291-306" />{' '}
          <Cite kind="spec" ref="§1.5" />. The Test rows keep their labels on disk, as
          in RelBench’s cached files: the sampler masks only the seed’s own target cell, and evaluation reads the true label from the
          batch <Cite kind="spec" ref="§2.2" /> <Cite kind="code" ref="rt/main.py:283-288" />.
        </p>
        <Figure
          wide
          caption={
            <>
              Choose where a seed row sits. The sampler skips a task row only if its timestamp is <em>later</em> than the seed’s, so rows at
              the same timestamp pass <Cite kind="code" ref="rustler/src/fly.rs:421-424" />. The figure shows only this time filter: whether
              a row that passes it actually lands in the context depends on the search and on L (<Link to="/sampling">Chapter 6</Link>).
            </>
          }
        >
          <SplitsTimeline db={shop} task={churn} defaultSeedT={liveSeedT} />
        </Figure>
        <p>
          Move the seed to Test: the Val rows are now earlier than the seed, so they pass the filter too, and a Test prediction can use
          Val labels as context. The time filter skips only rows that lie in the seed’s future{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:421-424" />.
        </p>
        <p>
          To RT, the three splits are one logical table. Its name is the task’s directory name and the split only sets the table type{' '}
          <Cite kind="code" ref="rustler/src/pre.rs:152-174" />. Names are interned once, so all three splits share one table name index
          and one index per column <Cite kind="code" ref="rustler/src/pre.rs:424-437" /> <Cite kind="spec" ref="§2.2" />. Live from the
          preprocessed ShopDB:
        </p>
        <div ref={idWrapRef} className="scroll-quiet my-5 overflow-x-auto">
          <table className="card w-full border-collapse text-[15.5px]">
            <thead>
              <tr className="border-b border-grid bg-surface-2 text-left text-[14.5px] text-ink-2">
                <th scope="col" className="px-3 py-2 font-semibold">
                  split
                </th>
                <th scope="col" className="px-3 py-2 text-right font-semibold" title="table_name_idx: index of the interned table name">
                  {idNarrow ? 'table idx' : 'table_name_idx'}
                </th>
                <th scope="col" className="px-3 py-2 text-right font-semibold" title={`col_name_idx of “timestamp of ${churn.name}”`}>
                  {idNarrow ? 'ts col' : 'timestamp col'}
                </th>
                <th scope="col" className="px-3 py-2 text-right font-semibold" title={`col_name_idx of “${churn.targetCol} of ${churn.name}”`}>
                  {churn.targetCol} col
                </th>
                <th scope="col" className="px-3 py-2 text-right font-semibold">
                  rows
                </th>
                {!idNarrow && (
                  <>
                    <th scope="col" className="px-3 py-2 text-right font-semibold" title="node_idx_offset: the split’s first node index">
                      offset
                    </th>
                    <th scope="col" className="px-3 py-2 font-semibold">
                      table_info key
                    </th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {splitIds.map(({ s, info, n }) => (
                <tr key={s} className="border-b border-grid/70 last:border-0">
                  <th scope="row" className="px-3 py-1.5 text-left font-medium text-ink">
                    {s}
                  </th>
                  <td className="num bg-accent-soft px-3 py-1.5 text-right font-semibold text-ink">{n.tableNameIdx}</td>
                  <td className="num bg-accent-soft px-3 py-1.5 text-right text-ink">
                    {n.cells.find((c) => c.colName === 'timestamp')?.colNameIdx ?? churnTsIdx}
                  </td>
                  <td className="num bg-accent-soft px-3 py-1.5 text-right text-ink">
                    {n.cells.find((c) => c.colName === churn.targetCol)?.colNameIdx ?? churnTargetIdx}
                  </td>
                  <td className="num px-3 py-1.5 text-right text-ink">{info.numNodes}</td>
                  {!idNarrow && (
                    <>
                      <td className="num px-3 py-1.5 text-right text-ink">{info.nodeIdxOffset}</td>
                      <td className="px-3 py-1.5 font-mono whitespace-nowrap text-ink-2">
                        {churn.name}:{s}
                      </td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 px-1 text-[15px] leading-snug text-ink-2">
            Shaded: the same in all three splits.{' '}
            {idNarrow
              ? `Each split also has its own block of node indices, starting at ${splitIds.map(({ s, info }) => `${s} ${info.nodeIdxOffset}`).join(', ')}. `
              : 'Each split has its own block of node indices (offset … offset + rows − 1). '}
            The blocks follow the file order, test &lt; train &lt; val <Cite kind="code" ref="rustler/src/pre.rs:79-92" />{' '}
            <Cite kind="spec" ref="§1.5" />.
          </p>
        </div>
        <p>
          This is why the past matters. Past rows of the same task, from any split, look to the model like ordinary rows of the same
          table. The sampler reaches them through the customer, and their label cells stay unmasked. They are the{' '}
          <Term id="in-context-label">in-context labels</Term> <Cite kind="spec" ref="§2.5" />: “while task rows provide ‘in-context
          labels’, our setting is not few-shot as explicit subgraph-label pairs are not required” <Cite kind="paper" ref="p.2" />.
        </p>
        <p>
          {liveForecast ? 'In your live context' : 'In the default customer-churn context'} ({rowTask.name}, L = {rowCtx.params.L}, w ={' '}
          {rowCtx.params.w} <ScaleNote inline />) the seed is {reach.seedWho ?? `node ${rowSeed?.nodeIdx}`}. The seed’s own{' '}
          {reach.entityWord} contributes <strong>{rowLabels.self.length}</strong> visible past label
          {rowLabels.self.length === 1 ? '' : 's'}, and other {reach.entityWord}s contribute <strong>{rowLabels.other.length}</strong>.{' '}
          {reach.text}
        </p>
        <p>
          Compare the paper’s counts on RelBench, as mean ± sd per 1024-cell context (target entity / other entities): rel-f1 driver-dnf
          19 ± 14 / 0 ± 0, and rel-amazon user-churn 4 ± 4 / 1 ± 5 <Cite kind="paper" ref="p.23 Table 12" />. Those counts predate a
          sampler change that keeps same-timestamp task rows <Cite kind="spec" ref="§11.3" />, which the next note is about.
        </p>
        <Callout
          variant="paper-vs-code"
          title="Labels at the seed’s own timestamp"
          paper={
            <>
              Not discussed. The leakage rule reads “rows with timestamps later than the seed row are excluded”{' '}
              <Cite kind="paper" ref="p.4 §3.1" />.
            </>
          }
          code={
            <>
              Equal timestamps pass the filter, so a Val seed can reach other customers’ Val rows with their labels visible{' '}
              <Cite kind="code" ref="rustler/src/fly.rs:422" />.
            </>
          }
        >
          <p>
            The ICLR-submission code excluded these same-horizon task rows, and the later RT-J code excludes them again; the paper-era code
            keeps them <Cite kind="spec" ref="§2.5" /> <Cite kind="spec" ref="§10 #11" />. They are the hatched bar in the figure.
          </p>
        </Callout>
        <p>
          One more difference concerns time. The paper says it uses “database rows only up to the training cutoff timestamps”{' '}
          <Cite kind="paper" ref="p.6 §4.1" />, but the paper-era code drops no database row by date; the sampler’s filter on child rows
          is the only time guard <Cite kind="code" ref="rustler/src/fly.rs:421-424" /> <Cite kind="spec" ref="§10 #10" />. ShopDB keeps
          every row, as the code does (<Link to="/training">Chapter 13</Link> has the details).
        </p>
        <CodeTabs
          tabs={[
            {
              id: 'type',
              label: 'table name and split',
              s: PRE_TABLE_TYPE,
              title: 'Task tables: the name comes from the directory, the split from the file',
              note: 'pq_path is tasks/<task>/<split>.parquet: every split gets the same table_name and a different TableType.',
            },
            {
              id: 'intern',
              label: 'name interning',
              s: PRE_NAME_INTERN,
              title: 'Names are interned once per string',
              note: 'Same string, same index: the three splits share table_name_idx and col_name_idx.',
            },
          ]}
        />
      </Section>

      {/* ───────────────────────────── 4 ───────────────────────────── */}
      <Section id="regression-autocomplete" title="Regression targets and autocomplete">
        <h3>A regression task: product-sales</h3>
        <p>
          Forecasting labels need not be booleans. <code>product-sales</code> asks how many units of a product will be ordered in{' '}
          <span className="whitespace-nowrap">(t, t + {sales.timedeltaDays} d]</span>. The recipe is the same as for churn, with a different aggregation: add up the quantities of the window’s orders instead
          of checking that there are none. The label is a number, so this is regression: a Number target trained with a Huber loss on its
          z-score <Cite kind="code" ref="rt/model.py:250-251" /> <Cite kind="spec" ref="§2.6" />. The z-score uses the Train split’s
          statistics, also for Val and Test rows <Cite kind="spec" ref="§3.3" />.
        </p>
        <Figure wide caption="Hover, tap or focus a bar to see the orders it sums; a click pins it, and arrow keys move between windows. Pick any product: this choice stays in this chapter.">
          <SalesFigure db={shop} pdb={shopPdb} task={sales} product={product} onProductChange={setProduct} />
        </Figure>
        <p>
          In the official code, the loss is picked by the type of the target cell, not by a list of task names{' '}
          <Cite kind="code" ref="rt/model.py:238-258" />. Only evaluation uses a list of task names: it decides which decoder’s output is read
          and whether the metric is AUROC or R² <Cite kind="code" ref="rt/main.py:228-241" /> <Cite kind="code" ref="rt/main.py:282-287" />. RelBench stores some binary labels as 0/1 integers, so preprocessing casts them
          to booleans, task by task <Cite kind="code" ref="rustler/src/pre.rs:208-289" /> <Cite kind="spec" ref="§2.6" />.
        </p>
        <CodeTabs
          tabs={[
            {
              id: 'fc',
              label: 'forecasting tasks',
              s: TASKS_FORECAST,
              title: 'The paper’s forecasting tasks',
              note: 'Each tuple is (database, table, target column, leakage columns). For forecasting, the table is the task table and the leakage list is empty.',
            },
            {
              id: 'bool',
              label: '0/1 labels → Boolean',
              s: PRE_CAST_BOOL,
              title: 'Binary labels are cast to Boolean during preprocessing',
              note: 'rel-amazon user-churn and item-churn: churn becomes a Boolean column, so it gets the BCE loss.',
            },
          ]}
        />

        <h3>Autocomplete: the target is an existing column</h3>
        <p>
          Autocomplete needs no new table: “sampling starts from the masked database row” <Cite kind="paper" ref="p.19 App. D" />. The
          seed is a database row and the target is one of its cells. The code looks up the seed table as <code>&lt;table&gt;:Db</code>{' '}
          whenever such a database table exists, and falls back to a task split only otherwise{' '}
          <Cite kind="code" ref="rt/data.py:77-81" />. The seeds are all rows whose target cell is present{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:287-296" />.
        </p>
        <p>
          Autocomplete hides a trap: other columns of the same row can give the answer away.{' '}
          {leakExample && (
            <>
              ShopDB’s review {leakExample.row} reads “{leakExample.text}” and its rating is {leakExample.rating}.{' '}
            </>
          )}
          So each autocomplete task lists <Term id="leakage-columns">leakage columns</Term>, which the sampler drops on the fly{' '}
          <Cite kind="paper" ref="p.19 App. D" />. For the paper’s rel-amazon rating task they are <code>review_text</code> and{' '}
          <code>summary</code> <Cite kind="code" ref="rt/tasks.py:67" />; ShopDB’s rating task drops <code>review_text</code>.
        </p>
        <Callout
          variant="paper-vs-code"
          title="Which leakage cells are dropped"
          paper={
            <>
              “For each task, we also identify potential sources of information leakage and discard these columns on the fly when
              building the input sequence” <Cite kind="paper" ref="p.19 App. D" />. This reads as if the whole column disappears.
            </>
          }
          code={
            <>
              A leakage cell is skipped only in the seed row and in rows whose timestamp equals the seed’s. Other rows of the table keep it,
              so other reviews still show their <code>review_text</code> <Cite kind="code" ref="rustler/src/fly.rs:457-461" />{' '}
              <Cite kind="spec" ref="§4.9" />.
            </>
          }
          verdict="This site follows the code."
        >
          <p>
            “Equal” means the same second, not the same day <Cite kind="spec" ref="§1.4" />.{' '}
            {reviewTimeDupes === 0
              ? `In ShopDB no two of the ${reviewCount} reviews share a timestamp, so only the seed’s own review_text is ever dropped. In a dataset with coarser timestamps, such as dates only, every review on the seed’s day would lose its text too.`
              : `In ShopDB, ${reviewTimeDupes} group${reviewTimeDupes === 1 ? '' : 's'} of reviews share a timestamp; a seed in such a group also drops the others’ review_text.`}
          </p>
        </Callout>
        <Figure
          wide
          title="An autocomplete seed: which cells become tokens"
          caption={
            <>
              Pick a review and a task. The right side samples a real context with your L, w and RNG seed, then shows the seed row’s tokens.
              With <CapCode>reviews.rating</CapCode> the seed’s own <CapCode>review_text</CapCode> never enters; other reviews keep theirs
              unless they share the seed’s timestamp. With <CapCode>reviews.verified</CapCode> nothing is dropped, like the paper’s
              rel-amazon <CapCode>review.verified</CapCode> task <Cite kind="code" ref="rt/tasks.py:33" />.
            </>
          }
        >
          <AutocompleteFigure
            db={shop}
            pdb={shopPdb}
            tasks={[rating, verified]}
            taskId={acTask}
            onTaskChange={setAcTask}
            review={review}
            onReviewChange={setReview}
            tableSlots={shopSlots}
          />
        </Figure>
        <p>
          No task rows appear in an autocomplete context, because the sampler follows task-table children only from a seed of that same
          task table <Cite kind="code" ref="rustler/src/fly.rs:415-419" /> <Cite kind="spec" ref="§2.4" />. The in-context labels are the
          visible target values of other rows of the same table, such as other reviews of the same product. <ScaleNote inline />
        </p>
        <Callout
          variant="paper-vs-code"
          title="Autocomplete ignores the split"
          paper={
            <>
              “We pretrain and fine-tune only on the training splits of the tasks” <Cite kind="paper" ref="p.6 §4.1" />, and autocomplete
              inputs are built “in the same way as for forecasting tasks, preserving the relational structure and temporal{' '}
              <span className="whitespace-nowrap">
                order” <Cite kind="paper" ref="p.19 App. D" />.
              </span>
            </>
          }
          code={
            <>
              For an autocomplete task, data.py always picks <code>&lt;table&gt;:Db</code>, so the requested split is ignored: the seeds are
              every row of the table, from every period, and no cutoff applies <Cite kind="code" ref="rt/data.py:77-81" />.
            </>
          }
        >
          <p>
            So here too the split setting has no effect on autocomplete tasks. The engine’s seed lists for {rating.table}.{rating.targetCol}:{' '}
            {SPLITS.map((sp, k) => (
              <span key={sp} className="whitespace-nowrap">
                {k > 0 ? ' · ' : ''}
                {sp} <strong className="num">{acSeedCounts[sp]}</strong>
              </span>
            ))}{' '}
            <Cite kind="spec" ref="§2.4" /> <Cite kind="spec" ref="§10 #10" />.
          </p>
        </Callout>
        <CodeTabs
          tabs={[
            {
              id: 'ac',
              label: 'autocomplete tasks',
              s: TASKS_AUTOCOMPLETE,
              title: 'Autocomplete tasks with leakage columns',
              highlight: [67],
              note: 'The rating of a rel-amazon review would leak through its text and summary, so both are dropped.',
            },
            {
              id: 'key',
              label: 'seed table',
              s: DATA_TABLE_KEY,
              title: 'Seed table: “<table>:Db” if it exists, else “<table>:<split>”',
              note: 'Also resolves the target column and the leakage columns to “<col> of <table>” indices.',
            },
            { id: 'seeds', label: 'seeds', s: FLY_SEEDS, title: 'Seeds: every row that has a target cell' },
            {
              id: 'drop',
              label: 'leakage drop',
              s: FLY_LEAKAGE_DROP,
              title: 'Dropping leakage cells on the fly',
              note: 'A cell is skipped if its column is a leakage column and the row is the seed or shares the seed’s timestamp.',
            },
          ]}
        />
      </Section>

      {/* ───────────────────────────── 5 ───────────────────────────── */}
      <Section id="prompting" title="Task-table prompting">
        <p>
          This is the first of RT’s two innovations in input representation; the second, cell-level tokenization, is{' '}
          <Link to="/tokens">Chapter 7</Link> <Cite kind="paper" ref="p.4 §3.1" />. “For each task, we attach a dedicated task table to the
          database. Only one task table
          is active at a time, ensuring that sampling is task-specific. Rows from the task table serve as seed rows for context
          construction, and from the model’s perspective task tables are treated as ordinary relational tables”{' '}
          <Cite kind="paper" ref="p.4 §3.1" />. The paper calls it <Term id="task-table-prompting">task-table prompting</Term>.
        </p>
        <Figure
          wide
          title="Attaching a task table to ShopDB"
          caption="Attach either forecasting task. The task table joins the schema like any other table, with one foreign key to its entity, and its rows become children of that entity."
        >
          <PromptingFigure db={shop} pdb={shopPdb} tasks={shop.tasks} slots={shopSlots} customer={customer} product={product} />
        </Figure>
        <p>
          Why a table and not a special input? Because then a task lives in the same input space as pretraining: the model reads it
          through the same cells, column names and links as the rest of the database. In the paper’s words, task-table prompting
          “augments the database input with task-specific context, enabling zero-shot predictions across diverse schemas”{' '}
          <Cite kind="paper" ref="p.4 §3.1" />, that is, predictions on a new database without weight updates{' '}
          <Cite kind="paper" ref="p.4 §2.3" />. And since the task’s past rows carry their labels, the table also shows the
          model what answers look like for this entity and its neighbours <Cite kind="spec" ref="§2.5" />.
        </p>
        <p>
          In the code, “only one task table is active” works like this. Preprocessing indexes every task table of the database{' '}
          <Cite kind="code" ref="rustler/src/pre.rs:79-92" />. The sampler then follows a link into a task table only when the seed belongs
          to that same table <Cite kind="code" ref="rustler/src/fly.rs:415-419" />. A customer-churn seed never pulls in product-sales rows,
          and an autocomplete seed, being a database row, pulls in no task rows at all.
        </p>
        <Code
          s={FLY_TASK_TABLE_SKIP}
          title="Only the seed’s own task table is followed"
          note="table_name_idx is shared by Train, Val and Test, so past rows of the same task pass the first test from any split; the second test is the temporal filter."
          highlight={[416, 417]}
        />
      </Section>

      {/* ───────────────────────────── 6 ───────────────────────────── */}
      <Section id="task-row-tokens" title="What the model sees of a task row">
        <p>
          Strip away the table and a task row is small. After preprocessing it has exactly two tokens: the timestamp cell (a datetime) and
          the label cell (a boolean or a number). The foreign key becomes a link, not a token: the entity’s node index goes into the row’s{' '}
          <code>f2p_nbr_idxs</code> list <Cite kind="spec" ref="§2.1" /> <Cite kind="code" ref="rustler/src/pre.rs:444-613" />.
        </p>
        <Figure
          wide
          title="A task row becomes two tokens"
          caption={
            <>
              From {liveForecast ? 'your live context' : 'the default customer-churn context (your live task is not a forecasting task)'}:{' '}
              {rowTask.name}, seed node {rowSeed?.nodeIdx}, L = {rowCtx.params.L}, w = {rowCtx.params.w} <ScaleNote inline />. Hover a token
              for the value stored for the model. Both rows have the same two tokens; only the seed’s label is replaced by the mask.
            </>
          }
        >
          <TaskRowTokens db={rowDb} pdb={rowPdb} ctx={rowCtx} slots={rowSlots} pastNode={pastNode} />
        </Figure>
        <p>
          Nothing tells the model which row holds the question. Every task row carries an <code>is_task_node</code> flag, but the model
          never reads it <Cite kind="spec" ref="§2.1" />. The seed is special only because its label cell is the one masked token{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:492-496" />. The next chapter turns all these rows into numbers: which cells become
          tokens, and how each value is normalized (<Link to="/preprocessing">Chapter 4</Link>).
        </p>
        <Code s={FLY_MASK} title="The one masked cell: the seed row’s target column" />

        <Quiz
          questions={[
            {
              prompt: (
                <>
                  Is <code>product-sales</code> an autocomplete task or a forecasting task?
                </>
              ),
              choices: [
                {
                  text: 'Autocomplete: sales is a column of the products table.',
                  why: 'ShopDB’s products table has no sales column. The value is built from orders that happen after t.',
                },
                {
                  text: 'Neither: RT cannot predict counts.',
                  why: 'It can. A numeric target is regression, trained with a Huber loss on its z-score.',
                },
                {
                  text: `Forecasting: the label is computed from the orders in (t, t + ${sales.timedeltaDays} d] and stored in a task table.`,
                  correct: true,
                  why: 'Right. It is also a regression task, because the label is a number.',
                },
              ],
              explanation: (
                <>
                  Forecasting targets do not exist in the database and are “constructed from future rows” (paper p.3). That is why
                  product-sales has its own task table, with Train, Val and Test splits.
                </>
              ),
            },
            {
              prompt: (
                <>
                  Why does the <code>reviews.rating</code> task drop <code>review_text</code>?
                </>
              ),
              choices: [
                { text: 'The text gives the rating away, so keeping it would leak the answer.', correct: true },
                { text: 'Text cells cannot become tokens.', why: 'They can: reviews.verified keeps review_text as a text token.' },
                { text: 'To keep the context window short.', why: 'Context length is set by L, not by leakage columns.' },
              ],
              explanation: (
                <>
                  Leakage columns are listed per task and dropped on the fly, for the seed row and for rows with the seed’s timestamp
                  (fly.rs:457-461). Other reviews keep their text. The paper’s rel-amazon rating task drops review_text and summary.
                </>
              ),
            },
            {
              prompt: 'How many tokens does one customer-churn row contribute to a context?',
              choices: [
                {
                  text: 'Three: timestamp, customer_id and churn.',
                  why: 'customer_id is a foreign key. It becomes a link (f2p_nbr_idxs), not a token.',
                },
                { text: 'Two: the timestamp and the churn label.', correct: true },
                {
                  text: 'One: only the label.',
                  why: 'The timestamp is the table’s time column, and a time column is an ordinary datetime token.',
                },
              ],
              explanation: 'Task rows are ordinary rows: every non-key, non-missing cell is a token. For a task row that means two cells.',
            },
            {
              prompt: (
                <>
                  A customer-churn seed sits at the Val timestamp ({isoDay(churn.valTimestamp!)}). Which rows of the task table pass the
                  sampler’s temporal filter?
                </>
              ),
              choices: [
                {
                  text: 'Only Train rows: Val and Test rows are held out.',
                  why: 'Equal timestamps pass the filter, so other customers’ Val rows pass too (fly.rs:422).',
                },
                {
                  text: 'Train, Val and Test rows: the splits are separate tables, so the filter does not compare them.',
                  why: 'The splits share one table name, so the table test lets all three through, and the time test then skips the Test rows, which are later than the seed (fly.rs:417, 421-424).',
                },
                {
                  text: 'Train rows and the other customers’ Val rows, which share the seed’s timestamp.',
                  correct: true,
                  why: 'Right. A row is skipped only if its timestamp is later than the seed’s. Whether it then lands in the context depends on the search and on L.',
                },
              ],
              explanation: (
                <>
                  The sampler never looks at the split: it compares table names (all three splits share one) and timestamps. That is the
                  hatched bar in the splits figure, and the paper-vs-code note on same-timestamp labels.
                </>
              ),
            },
          ]}
        />
      </Section>

      <ChapterNav slug="tasks" />
    </ChapterPage>
  );
}
