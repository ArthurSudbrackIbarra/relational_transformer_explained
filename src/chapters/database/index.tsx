/**
 * Chapter 2 · "The relational database" (BUILD_PLAN §6.2, CHAPTER_BRIEFS §2).
 *
 * Fluency with tables, rows, cells, PK/FK, F→P vs P→F, time columns and feature
 * vs key columns, exactly as RT's preprocessing uses them (spec §1). Every figure
 * reads the live database through src/state/hooks.ts. The chapter follows one
 * "focus row" (local state): by default the entity of the simulation's seed row;
 * clicking rows in any figure moves it without touching the global seed.
 */
import { ArrowRightLeft, Crosshair } from 'lucide-react';
import { useCallback, useDeferredValue, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
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
  Stat,
  TableSwatch,
  Term,
  formatDate,
  formatRaw,
  type QuizQuestion,
  type SchemaTableSpec,
} from '@/components';
import { getDatabase, listDatabases } from '@/engine/db/registry';
import { MAX_F2P_NBRS } from '@/engine/types';
import { getPreprocessed, useDatabase, usePreprocessed, useSeedNode, useSeeds, useTableSlots, useTask } from '@/state/hooks';
import { useSim } from '@/state/store';
import { usePrefersReducedMotion } from '@/styles/theme';
import { AnatomyFigure } from './AnatomyFigure';
import { CellsVsNodes } from './CellsVsNodes';
import { DegreeChart } from './DegreeChart';
import { LinkExplorer } from './LinkExplorer';
import {
  buildLinkIndex,
  defaultFocus,
  rowCaption,
  rowKey,
  rowLabel,
  sameRow,
  seedForEntity,
  seedTimeMs,
  timelineAnchor,
  timestampCoverage,
  nullCounts,
  type RowRef,
} from './model';
import { NeighbourhoodGraph } from './NeighbourhoodGraph';
import { RowTokens } from './RowTokens';
import { SchemaOverview } from './SchemaOverview';
import { CELLS, EDGES, KEYS, META, TEMPORAL, TS } from './snippets';
import { TableBrowser } from './TableBrowser';
import { Timeline } from './Timeline';
import { TwoForeignKeys } from './TwoForeignKeys';

const SLUG = 'database';

export default function DatabaseChapter() {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const task = useTask();
  const seedNode = useSeedNode();
  const seeds = useSeeds();
  const slots = useTableSlots();
  const w = useSim((s) => s.w);
  const setSeedNode = useSim((s) => s.setSeedNode);
  const setTask = useSim((s) => s.setTask);

  const reduceMotion = usePrefersReducedMotion();
  const idx = useMemo(() => buildLinkIndex(db), [db]);
  const example = useMemo(() => defaultFocus(pdb, db, seedNode), [pdb, db, seedNode]);
  const exampleKey = rowKey(example);

  // ── chapter-local focus row (+ walk history), reset when the live seed changes ──
  const [focus, setFocusRaw] = useState<RowRef>(example);
  const [history, setHistory] = useState<RowRef[]>([]);
  const [tab, setTab] = useState<string>(example.table);
  const parentTables = useMemo(() => db.tables.filter((t) => idx.childTablesOf(t.schema.name).length > 0).map((t) => t.schema.name), [db, idx]);
  const [degTable, setDegTable] = useState<string>(parentTables.includes(example.table) ? example.table : (parentTables[0] ?? example.table));

  useEffect(() => {
    setFocusRaw(example);
    setHistory([]);
    setTab(example.table);
    if (parentTables.includes(example.table)) setDegTable(example.table);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exampleKey, db]);

  // setFocus is stable (reads the current focus from a ref) so the memoized figures skip re-rendering
  // when only unrelated props change.
  const focusRef = useRef(focus);
  useLayoutEffect(() => {
    focusRef.current = focus;
  }, [focus]);
  const setFocus = useCallback(
    (r: RowRef) => {
      const cur = focusRef.current;
      if (sameRow(cur, r)) return;
      setHistory((h) => [...h, cur].slice(-12));
      setFocusRaw(r);
      setTab(r.table);
      if (parentTables.includes(r.table)) setDegTable(r.table);
    },
    [parentTables],
  );
  // The heavy figures (graph layouts, charts) follow a deferred copy of the focus row, so the control
  // that was clicked updates at once and the figures catch up without blocking it.
  const lazyFocus = useDeferredValue(focus);
  const openTable = useCallback(
    (t: string) => {
      setTab(t);
      document.getElementById('browse')?.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
    },
    [reduceMotion],
  );
  const back = useCallback(() => {
    if (!history.length) return;
    const prev = history[history.length - 1];
    setHistory(history.slice(0, -1));
    setFocusRaw(prev);
    setTab(prev.table);
    if (parentTables.includes(prev.table)) setDegTable(prev.table);
  }, [history, parentTables]);

  // ── live facts used in the prose ──
  const tables = db.tables;
  const count = (name: string) => idx.table(name)?.rows.length ?? 0;
  const isShop = db.name === 'shop';
  const dbShort = isShop ? 'ShopDB' : db.title.split(' ')[0];
  const seed = seedNode >= 0 ? pdb.nodes[seedNode] : undefined;
  const seedTime = seedTimeMs(pdb, seedNode);
  const exampleCaption = rowCaption(db, example, 40);
  const exampleCountry = idx.table(example.table)?.rows[example.row]?.country;

  const schemaTables: SchemaTableSpec[] = useMemo(
    () => db.tables.map((t) => ({ schema: t.schema, slot: slots[t.schema.name] ?? 0, rowCount: t.rows.length, kind: 'db' as const })),
    [db, slots],
  );

  // a child table with FKs and a time column (ShopDB: orders) for the anatomy figure and the FK note
  const eventTable = tables.find((t) => Object.keys(t.schema.fkeys).length > 0 && t.schema.timeCol) ?? tables.find((t) => Object.keys(t.schema.fkeys).length > 0) ?? tables[0];
  const eventRows = useMemo(() => {
    const g = idx.childrenOf(example).find((x) => x.childTable === eventTable.schema.name);
    const rows = g && g.rows.length >= 3 ? g.rows.slice(0, 4) : [0, 1, 2, 3].filter((r) => r < eventTable.rows.length);
    return rows;
  }, [idx, example, eventTable]);
  const noteRow: RowRef = { table: eventTable.schema.name, row: eventRows[0] ?? 0 };
  const noteParents = useMemo(() => idx.parentsOf(noteRow).filter((p) => p.parentRow !== null), [idx, noteRow.table, noteRow.row]); // eslint-disable-line react-hooks/exhaustive-deps
  const noteParentRefs = useMemo(() => noteParents.map((p) => ({ table: p.parentTable, row: p.parentRow! })), [noteParents]);

  const nulls = useMemo(
    () =>
      tables.flatMap((t) =>
        Object.entries(nullCounts(t))
          .filter(([, n]) => n > 0)
          .map(([col, n]) => ({ table: t.schema.name, col, n, of: t.rows.length })),
      ),
    [tables],
  );

  const degree = useMemo(() => {
    let best = { ref: { table: tables[0].schema.name, row: 0 } as RowRef, n: -1 };
    for (const t of tables)
      for (let r = 0; r < t.rows.length; r++) {
        const n = idx.childCount({ table: t.schema.name, row: r });
        if (n > best.n) best = { ref: { table: t.schema.name, row: r }, n };
      }
    const maxFk = Math.max(...tables.map((t) => Object.keys(t.schema.fkeys).length));
    const fkTables = tables.filter((t) => Object.keys(t.schema.fkeys).length === maxFk).map((t) => t.schema.name);
    return { best, maxFk, fkTables };
  }, [tables, idx]);
  const bestGroups = idx.childrenOf(degree.best.ref);

  const sameParentFks = tables.flatMap((t) => {
    const by = new Map<string, string[]>();
    for (const [fk, parent] of Object.entries(t.schema.fkeys)) by.set(parent, [...(by.get(parent) ?? []), fk]);
    return [...by.entries()].filter(([, fks]) => fks.length > 1).map(([parent, fks]) => `${t.schema.name}.${fks.join(' and ')} both point to ${parent}`);
  });
  const boolCols = tables.flatMap((t) => t.schema.columns.filter((c) => c.dtype === 'bool').map((c) => `${t.schema.name}.${c.name}`));
  const timeTables = tables.filter((t) => t.schema.timeCol);
  const noTimeTables = tables.filter((t) => !t.schema.timeCol);
  const coverage = useMemo(() => timestampCoverage(pdb, db), [pdb, db]);
  const anchor = useMemo(() => timelineAnchor(idx, lazyFocus), [idx, lazyFocus]);
  // a table with a time column but no foreign key (spec §1.4): in this database if it has one, else in another
  const localTimeOnly = useMemo(() => {
    const c = coverage.find((x) => x.timeCol && x.nFk === 0);
    return c ? { title: dbShort, table: c.table, timeCol: c.timeCol!, rows: c.rows, withTs: c.withTs, local: true } : null;
  }, [coverage, dbShort]);
  const otherTimeOnly = useTimeColumnWithoutFk(db.name, localTimeOnly === null);
  const timeNoFk = localTimeOnly ?? otherTimeOnly;
  const noTsTables = coverage.filter((c) => c.withTs === 0).map((c) => c.table);
  const noFkTables = tables.filter((t) => Object.keys(t.schema.fkeys).length === 0).map((t) => t.schema.name);
  const fkTables = tables.filter((t) => Object.keys(t.schema.fkeys).length > 0);
  const timeOnlyTables = tables.filter((t) => t.schema.timeCol && Object.keys(t.schema.fkeys).length === 0).map((t) => t.schema.name);
  // an autocomplete task with leakage columns (ShopDB: reviews.rating drops review_text), if this DB has one
  const dropTask = db.tasks.find((t) => t.kind === 'autocomplete' && t.columnsToDrop.length > 0);

  const adoptNode = seedForEntity(pdb, task, seeds, focus);
  const adopt = useMemo(
    () =>
      adoptNode !== null && !sameRow(focus, example)
        ? { label: `Make ${rowLabel(focus)} the running example (sets the simulation seed)`, onClick: () => setSeedNode(adoptNode) }
        : null,
    [adoptNode, focus, example, setSeedNode],
  );

  const quiz = useQuiz(noteRow, noteParentRefs, eventTable.schema.name, idx);

  return (
    <ChapterPage slug={SLUG}>
      <ChapterHeader
        slug={SLUG}
        lede="RT reads a database the way it is stored: tables of rows, rows of cells, and keys that link rows across tables. This chapter builds that vocabulary on ShopDB, the small online shop that the rest of the course runs on."
        learn={[
          'What tables, rows, columns and cells are, and which columns count as feature columns',
          'How primary and foreign keys link rows, in both directions (F→P and P→F)',
          'Why a row has few parents but possibly very many children',
          'Why key columns never become tokens, and what that costs',
          'What a time column is, and why prediction must respect time',
          'How the same database looks as a graph of rows, and how RT’s view differs',
        ]}
      >
        <div className="mt-5 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-xl border border-border bg-surface px-4 py-3 text-[17px] leading-snug">
          <Crosshair aria-hidden="true" size={16} className="shrink-0 text-accent-text" />
          <span className="min-w-0 flex-1 text-ink-2">
            Running example:{' '}
            <span className="inline-flex items-center gap-1.5 align-middle">
              <TableSwatch slot={slots[example.table] ?? 0} />
              <span className="font-mono font-semibold text-ink">{rowLabel(example)}</span>
            </span>
            {exampleCaption && <span className="text-ink"> {exampleCaption}</span>}
            {typeof exampleCountry === 'string' && <span> from {exampleCountry}</span>}
            {seed?.isTaskNode ? (
              <>
                , the {example.table.replace(/s$/, '')} behind the simulation’s seed row ({task.id.split('/')[1]}, {seed.tableType} split; chapter 3).
              </>
            ) : (
              <>, the simulation’s seed row (autocomplete task {task.id.split('/')[1]}; chapter 3).</>
            )}{' '}
            Change the seed in the top bar and this chapter follows.
          </span>
        </div>
      </ChapterHeader>

      {!isShop && (
        <Callout variant="warning" title={`The simulation is set to ${db.title}`}>
          <p>
            This chapter’s text describes ShopDB; the figures follow your simulation settings and therefore show {db.title}.{' '}
            <Button size="sm" variant="secondary" icon={<ArrowRightLeft size={14} />} onClick={() => setTask('shop/customer-churn')}>
              Switch the simulation to ShopDB
            </Button>
          </p>
        </Callout>
      )}

      {/* ─────────────────────────────── 1 ─────────────────────────────── */}
      <Section id="tables" title="Tables, rows, columns, cells">
        <p>
          A <Term id="relational-database">relational database</Term> is a collection of tables linked through relationships between them. Each
          table is made of rows, and every row is a set of cells, one for each column <Cite kind="paper" ref="p.3 §2.1" />. That small vocabulary is
          all RT needs, and it takes it literally: one cell becomes one <Term id="cell-token">token</Term> <Cite kind="paper" ref="p.4 §3.1" />.
        </p>
        <p>
          {isShop ? 'Our running database is ShopDB, a synthetic online shop' : `The simulation is currently set to ${db.title}`} with {tables.length}{' '}
          tables:{' '}
          {tables.map((t, i) => (
            <span key={t.schema.name}>
              {i > 0 && (i === tables.length - 1 ? ' and ' : ', ')}
              <span className="num">{t.rows.length.toLocaleString('en-US')}</span> {t.schema.name}
            </span>
          ))}
          , recorded from {formatDate(db.minTimestamp, 'never')} to {formatDate(db.maxTimestamp, 'never')}.{' '}
          {isShop
            ? 'It plays the part of a database the model has never seen: the miniature RT on this site was pretrained on three other synthetic databases, and ShopDB was held out (chapter 13).'
            : 'It is one of the three pretraining databases; ShopDB is the one held out (chapter 13).'}
        </p>
        <ScaleNote paper="RelBench, 7 real databases" ours="4 synthetic databases, ShopDB held out">
          Besides the sizes, we replace the data (and compute in float32 instead of bfloat16). Every operation applied to the data is the paper-era
          code’s.
        </ScaleNote>
        <Figure
          wide
          title={`Anatomy of a table: a few ${eventTable.schema.name} rows`}
          caption="Pick a word to see what it refers to. The rows are real rows of the live database."
        >
          <AnatomyFigure table={eventTable} rows={eventRows} slot={slots[eventTable.schema.name] ?? 0} dbLabel={isShop ? 'ShopDB' : db.title} nTables={tables.length} />
        </Figure>
      </Section>

      {/* ─────────────────────────────── 2 ─────────────────────────────── */}
      <Section id="schema" title={`The ${dbShort} schema`}>
        <p>
          A database’s <em>schema</em> lists its tables with their columns and data types, together with the relational structure{' '}
          <Cite kind="paper" ref="p.3 §2.1" />. In the diagram below each card is a table. A key icon marks the primary key and a link icon a
          foreign key, with an arrow to the table it references; a <span className="font-medium">time</span> tag marks the time column. Every other
          column is a feature column, drawn with its type glyph: <span className="font-mono">#</span> for numbers, <span className="font-medium">Aa</span>{' '}
          for text, a clock for datetimes and a toggle for booleans.
        </p>
        <Figure wide title={`${dbShort}: tables, keys and links`} caption="Hover a table to trace its links; click or tap it to open it in the table browser below.">
          <SchemaOverview tables={schemaTables} onOpen={openTable} />
        </Figure>
        <p>
          The shape is worth a second look.{' '}
          {noFkTables.length > 0 && (
            <>
              {joinAnd(noFkTables)} {noFkTables.length === 1 ? 'has' : 'have'} no foreign keys: nothing points out of{' '}
              {noFkTables.length === 1 ? 'it' : 'them'}.{' '}
            </>
          )}
          {fkTables.length > 0 && (
            <>
              {joinAnd(fkTables.map((t) => `${t.schema.name} (→ ${Object.values(t.schema.fkeys).join(', ')})`))}{' '}
              {fkTables.length === 1 ? 'holds foreign keys' : 'hold foreign keys'}
              {fkTables.every((t) => t.schema.timeCol)
                ? fkTables.length === 1
                  ? ', and it also has a time column.'
                  : ', and each also has a time column.'
                : fkTables.some((t) => t.schema.timeCol)
                  ? `; ${joinAnd(fkTables.filter((t) => t.schema.timeCol).map((t) => t.schema.name))} also ${
                      fkTables.filter((t) => t.schema.timeCol).length === 1 ? 'has' : 'have'
                    } a time column.`
                  : '.'}{' '}
            </>
          )}
          {timeOnlyTables.length > 0 && (
            <>
              {joinAnd(timeOnlyTables)} {timeOnlyTables.length === 1 ? 'has' : 'have'} a time column but no foreign key, a case the Time section
              comes back to.{' '}
            </>
          )}
          {isShop && 'Events point to the entities they involve.'}
        </p>
        <p>
          Schemas differ from one database to the next. That is why the paper argues that pretraining needs a schema-agnostic architecture, one that
          takes in the multi-table structure through attention masks <Cite kind="paper" ref="p.3 §2.1" />. RT never receives this diagram as such.
          It sees cells, each labelled with its column and table name, plus the links between rows (chapters 4, 5 and 9).
        </p>
        <CodeRef
          title="How a table declares its keys and time column"
          file={META.file}
          lines={META.lines}
          code={META.code}
          note={
            <>
              Each table file carries three metadata entries: <code>pkey_col</code>, <code>fkey_col_to_pkey_table</code> and <code>time_col</code>.
              They are how preprocessing learns which column is the primary key, which columns are foreign keys and to which table, and which column
              is the time column <Cite kind="spec" ref="§3.1" />. The <code>rel-avito</code> branch renames two foreign-key targets of one RelBench
              dataset and does not concern ShopDB.
            </>
          }
        />
      </Section>

      {/* ─────────────────────────────── 3 ─────────────────────────────── */}
      <Section id="browse" title="Browse the data">
        <p>
          Here are the tables as they are stored. Values are raw and <span className="font-mono">∅</span> marks a missing value. Click a row to make
          it the <strong>focus row</strong> of this chapter: the link explorer, the timeline and the graphs below all follow it. This only changes
          this page; the simulation’s seed stays where it is.
        </p>
        <Figure wide title="Table browser">
          <TableBrowser db={db} idx={idx} slots={slots} tab={tab} onTab={setTab} focus={focus} onFocus={setFocus} example={example} />
        </Figure>
        <p>
          The number on the left of each row is its position, and it equals the primary key: {rowLabel(example)} has{' '}
          <code>
            {idx.table(example.table)?.schema.pkey ?? 'pk'} = {example.row}
          </code>
          . RT relies on that convention, as the next section shows.
        </p>
        {nulls.length > 0 && (
          <p>
            {dbShort} has a few gaps on purpose:{' '}
            {nulls.map((n, i) => (
              <span key={n.table + n.col}>
                {i > 0 && (i === nulls.length - 1 ? ' and ' : ', ')}
                <code>{n.col}</code> is missing for {n.n} of {n.of} {n.table}
              </span>
            ))}
            . RT neither imputes nor masks a missing value. The cell simply produces no token, so that row has one token fewer{' '}
            <Cite kind="spec" ref="§3.5" />
            <Cite kind="code" ref="rustler/src/pre.rs:551-552" title="A null value matches the empty arm: no cell is recorded" />.
          </p>
        )}
      </Section>

      {/* ─────────────────────────────── 4 ─────────────────────────────── */}
      <Section id="links" title="Keys link rows: F→P and P→F">
        <p>
          Every table has a <Term id="primary-key">primary key</Term>, and some include <Term id="foreign-key">foreign keys</Term> that reference
          primary keys in other tables. This induces a graph structure. Connections from foreign keys to primary keys are{' '}
          <Term id="f2p-link">F→P links</Term>, and the reverse incoming connections are <Term id="p2f-link">P→F links</Term>{' '}
          <Cite kind="paper" ref="p.3 §2.1" />. (The task tables of chapter 3 are the one exception to “every table”: they have no primary key{' '}
          <Cite kind="spec" ref="§2.1" />.)
        </p>
        <p>
          {isShop ? (
            <>
              In ShopDB, an order’s <code>customer_id</code> and <code>product_id</code> are foreign keys. Each order therefore has two F→P links, up
              to its customer and its product. Seen from the other end, each customer and each product has P→F links down to its orders and reviews.
            </>
          ) : (
            <>
              In {dbShort}, the foreign keys of <code>{eventTable.schema.name}</code> point to {Object.values(eventTable.schema.fkeys).join(', ')}:
              each {eventTable.schema.name} row has one F→P link per non-null foreign key, and each of those parent rows has P→F links back down.
            </>
          )}{' '}
          The row a foreign key points to is the <em>parent</em>; the rows pointing to a row are its <em>children</em> <Cite kind="spec" ref="§1.3" />.
        </p>
        {noteParents.length > 0 && (
          <Callout variant="note" title="A foreign-key value is a row position">
            <p>
              The value of a foreign key is the position of the parent row in its table. Preprocessing finds the parent as the parent table’s
              offset plus that value, without searching for a row whose primary key equals it <Cite kind="code" ref="rustler/src/pre.rs:452-459" />
              <Cite kind="code" ref="rustler/src/pre.rs:477" />. So {rowLabel(noteRow)} has{' '}
              {noteParents.map((p, i) => (
                <span key={p.fkCol}>
                  {i > 0 && ' and '}
                  <code>
                    {p.fkCol} = {p.parentRow}
                  </code>
                </span>
              ))}
              , which means{' '}
              {noteParents.map((p, i) => (
                <span key={p.fkCol}>
                  {i > 0 && ' and '}
                  row {p.parentRow} of {p.parentTable} ({rowCaption(db, { table: p.parentTable, row: p.parentRow! })})
                </span>
              ))}
              . RelBench numbers primary keys 0…n−1 in row order, and a database must follow the same convention for RT to link the right rows{' '}
              <Cite kind="spec" ref="§1.2" />.
            </p>
          </Callout>
        )}
        <Figure
          wide
          title="Link explorer"
          caption="Walk the database: click a parent or a child to move there, and Back to retrace your steps. The graph draws the same rows."
        >
          <LinkExplorer
            db={db}
            idx={idx}
            slots={slots}
            focus={focus}
            graphFocus={lazyFocus}
            onFocus={setFocus}
            history={history}
            onBack={back}
            example={example}
            adopt={adopt}
          />
        </Figure>
        <p>
          Walk around for a moment and an asymmetry shows up. A row has at most one parent per foreign-key column, so no {dbShort} row has more
          than {degree.maxFk}. Children are another story: <span className="font-mono">{rowLabel(degree.best.ref)}</span> (
          {rowCaption(db, degree.best.ref)}) is referenced by{' '}
          {bestGroups.map((g, i) => (
            <span key={g.childTable + g.fkCol}>
              {i > 0 && ' and '}
              <span className="num">{g.rows.length}</span> {g.childTable}
            </span>
          ))}
          . The number of F→P links is limited by the number of foreign-key columns, while P→F links have unbounded degree{' '}
          <Cite kind="paper" ref="p.16 App. A" />.
        </p>
        <div className="wide my-6 grid gap-3 sm:grid-cols-3">
          <Stat label="Parents per row, at most" value={degree.maxFk} caption={`one per foreign-key column (${degree.fkTables.join(', ')})`} />
          <Stat label="Children per row, at most" value={degree.best.n} caption={`${rowLabel(degree.best.ref)}, no limit in principle`} emphasis />
          <Stat
            label="Non-null foreign keys RT can hold per row"
            value={MAX_F2P_NBRS}
            caption={
              <>
                the sampler’s MAX_F2P_NBRS <Cite kind="code" ref="rustler/src/fly.rs:25" />
                <Cite kind="spec" ref="§1.2" />
              </>
            }
          />
        </div>
        <Figure
          wide
          caption={
            <>
              Hover a bar to see the row; click it to make it the focus row. The dashed line marks the width bound w. The bars count every child
              ever written. When the sampler visits a row, it first drops the children that are later than the seed, then keeps a random w of the
              database children that remain (chapter 6) <Cite kind="paper" ref="p.15 Alg. 1" />
              <Cite kind="spec" ref="§4.7" />.
            </>
          }
        >
          <DegreeChart db={db} idx={idx} slots={slots} focus={lazyFocus} onFocus={setFocus} w={w} table={degTable} onTable={setDegTable} />
        </Figure>
        <p>
          The paper turns this asymmetry into its sampling rule. Parent rows typically contain important features (an order links to its customer
          and its product) and there are few of them, so RT follows every F→P link immediately, without subsampling. Informative signals from P→F
          links often arise through aggregation, with diminishing returns from including many children, so RT follows at most <em>w</em> children
          from any row <Cite kind="paper" ref="p.16 App. A" />. In the code the bound applies to database children only{' '}
          <Cite kind="spec" ref="§4.7" />. Chapter 6 runs the whole algorithm step by step.
        </p>
        <ScaleNote paper="w = 256" ours={`w = ${w}`}>
          The width bound is scaled down with the context length (paper L = 1024, here 128). The rule itself is unchanged.
        </ScaleNote>
        <CodeRef
          title="Keys in preprocessing: the primary key is skipped, each foreign-key value becomes a parent pointer"
          file={KEYS.file}
          lines={KEYS.lines}
          code={KEYS.code}
          highlightLines={[439, 440, 441, 442, 462, 463, 464, 477, 479]}
          note={
            <>
              Lines 439-442: the primary-key column is skipped, so it makes no token and no edge. For a foreign key, a null value is skipped
              (462-464), the parent is <code>ptable_offset + val</code> (477), and its index is pushed onto the row’s parent list{' '}
              <code>f2p_nbr_idxs</code> (479). The <code>continue</code> at the end of the branch (line 532, below this excerpt) means no cell is ever made from a key{' '}
              <Cite kind="spec" ref="§1.2" />.
            </>
          }
        />
        <CodeRef
          title="One foreign-key value, two edges: F→P on the child, P→F on the parent"
          file={EDGES.file}
          lines={EDGES.lines}
          code={EDGES.code}
          note={
            <>
              The F→P edge is stored on the child; the P→F edge is appended to the parent’s adjacency list and carries the <em>child’s</em> table
              and timestamp, which the sampler reads later <Cite kind="spec" ref="§1.2" />
              <Cite kind="spec" ref="§1.4" />.
            </>
          }
        />
      </Section>

      {/* ─────────────────────────────── 5 ─────────────────────────────── */}
      <Section id="features" title="Feature columns and key columns">
        <p>
          Now the consequence that matters most for the model. RT does not tokenize rows; it tokenizes cells, and only some of them. Once a row is
          selected, “all its non-missing feature cells (i.e., cells not from primary- or foreign-key columns)” enter the context{' '}
          <Cite kind="paper" ref="p.16 App. A" />. Two details of the code trim this, and chapter 6 shows both. The very last row can be cut short
          when the context fills up <Cite kind="spec" ref="§4.11" />.{' '}
          {dropTask ? (
            <>
              And an autocomplete task can name leakage columns, as {dbShort}’s <code>{`${dropTask.table}.${dropTask.targetCol}`}</code> task does with{' '}
              {dropTask.columnsToDrop.map((c, i) => (
                <span key={c}>
                  {i > 0 && (i === dropTask.columnsToDrop.length - 1 ? ' and ' : ', ')}
                  <code>{c}</code>
                </span>
              ))}
              : those cells are left out of the seed row and of every row of the seed’s table that shares its timestamp
            </>
          ) : (
            <>
              And an autocomplete task can name leakage columns, whose cells are left out of the seed row and of every row of the seed’s table that
              shares its timestamp
            </>
          )}{' '}
          <Cite kind="code" ref="rustler/src/fly.rs:457-461" />
          <Cite kind="spec" ref="§4.9" />.
        </p>
        <p>
          A <Term id="feature-column">feature column</Term> holds information about the row. A key column only says which row is which and which
          rows are linked.
        </p>
        <Figure wide title="What RT keeps of a row" caption="Split the row and watch where each cell goes. Try a row with a missing value from the menu.">
          <RowTokens db={db} idx={idx} slots={slots} focus={lazyFocus} />
        </Figure>
        <p>
          In the code, the primary-key column is skipped outright <Cite kind="code" ref="rustler/src/pre.rs:439-442" />. A foreign-key column
          produces no token either; each non-null value becomes an entry in the row’s parent list plus a pair of edges{' '}
          <Cite kind="code" ref="rustler/src/pre.rs:444-533" />. Every remaining column is a feature column. Each of its cells that is neither
          null nor a NaN float becomes a cell record <Cite kind="code" ref="rustler/src/pre.rs:552" />
          <Cite kind="code" ref="rustler/src/pre.rs:566-568" />, whose type follows its storage type{' '}
          <Cite kind="code" ref="rustler/src/pre.rs:543-611" />. Chapter 4 follows those records into numbers.
        </p>
        <Callout variant="insight" title="Our reading: keys are structure, not content">
          <p>
            The paper gives no reason for leaving key cells out; this is our interpretation. A key value is only a row position.{' '}
            {noteParents[0] ? (
              <>
                <code>
                  {noteParents[0].fkCol} = {noteParents[0].parentRow}
                </code>{' '}
                means “row {noteParents[0].parentRow} of {noteParents[0].parentTable}” and says nothing about that row itself.
              </>
            ) : (
              <>It says which row, and nothing about that row itself.</>
            )}{' '}
            What carries meaning is which row points to which, and that is what RT keeps: the parent list of every row and the edges in both
            directions <Cite kind="spec" ref="§1.2" />. The edges are what the sampler walks (chapter 6), and the parent lists later decide which
            cells may attend to which (chapter 9) <Cite kind="spec" ref="§6.2" />. Key column <em>names</em> are another matter: the paper counts
            losing them as a limitation, as the next paragraph explains.
          </p>
        </Callout>
        <p>
          The paper is candid about the price of this choice <Cite kind="paper" ref="p.10 §7" />. RT does not incorporate the names of primary-key
          and foreign-key columns, which often carry useful semantics. And it does not tell apart two foreign keys that point to the same table. The
          paper’s example is a product table with both <code>buyer_id</code> and <code>seller_id</code> referencing a user table: the model cannot
          distinguish which user bought and which sold.
        </p>
        <Figure title="Two foreign keys into the same table" caption={`After preprocessing the foreign-key column names are gone: only the parent list (pre.rs:479) and the edges (pre.rs:515-529) remain. ${
            sameParentFks.length
              ? `${dbShort} does have this case: ${sameParentFks.join('; ')}.`
              : `${dbShort} never has this problem: no table has two foreign keys into the same table.`
          }`}>
          <TwoForeignKeys />
        </Figure>
        <Callout variant="note" title="How many kinds of feature column?">
          <p>
            Section 2.1 of the paper describes feature columns as containing “numeric, text, and datetime information”{' '}
            <Cite kind="paper" ref="p.3 §2.1" />. A page later, §3.1 adds a fourth value type: “The value v can be numeric, boolean, datetime, or
            text”, and booleans are normalized like numbers <Cite kind="paper" ref="p.4 §3.1" />. The code has the same four semantic types{' '}
            <Cite kind="code" ref="rustler/src/common.rs:38-44" />, and it turns a boolean into 0 or 1 and z-scores it exactly like a number{' '}
            <Cite kind="code" ref="rustler/src/pre.rs:553-555" />.
            {boolCols.length > 0 && (
              <>
                {' '}
                {dbShort}’s boolean columns are{' '}
                {boolCols.map((c, i) => (
                  <span key={c}>
                    {i > 0 && (i === boolCols.length - 1 ? ' and ' : ', ')}
                    <code>{c}</code>
                  </span>
                ))}
                .
              </>
            )}
          </p>
        </Callout>
        <CodeRef
          title="Feature columns: one cell record per value that is not missing"
          file={CELLS.file}
          lines={CELLS.lines}
          code={CELLS.code}
          highlightLines={[552]}
          note={
            <>
              <code>Int32</code>, <code>Int64</code>, <code>UInt32</code> and <code>Float32</code> values are cast to <code>Float64</code> and
              become numbers like <code>Float64</code> itself; a storage type that no arm handles (an <code>Int16</code>, say) reaches the panic arm
              (lines 606-611). A null matches the
              empty arm on line 552 and leaves no trace, and a NaN float is skipped at lines 566-568. The boolean arm shown here is one of four
              (number, datetime and text follow in lines 565-605) <Cite kind="spec" ref="§3.2" />
              <Cite kind="spec" ref="§3.5" />.
            </>
          }
        />
      </Section>

      {/* ─────────────────────────────── 6 ─────────────────────────────── */}
      <Section id="time" title="Time">
        <p>
          Many relational databases are temporal: timestamp columns record when rows are created <Cite kind="paper" ref="p.3 §2.1" />. A table may
          declare one <Term id="time-column">time column</Term>. In {dbShort},{' '}
          {timeTables.map((t, i) => (
            <span key={t.schema.name}>
              {i > 0 && ' and '}
              {t.schema.name} have <code>{t.schema.timeCol}</code>
            </span>
          ))}
          . {noTimeTables.map((t) => t.schema.name).join(' and ')} have none.
          {isShop && (
            <>
              {' '}
              Note that customers does have a datetime column, <code>signup_date</code>, but it is an ordinary feature, not the table’s time column.
            </>
          )}
        </p>
        <p>
          The time column is still a feature column: <code>{timeTables[0]?.schema.timeCol ?? 'order_time'}</code> becomes a datetime token like any
          other <Cite kind="spec" ref="§1.4" />. It has a second job, though: it gives the row a timestamp that the sampler uses.
        </p>
        <p>
          Why time matters: “if we want to predict whether a user will buy an item at time <em>t</em>, the model must only use information
          available before <em>t</em>, otherwise it risks temporal leakage” <Cite kind="paper" ref="p.3 §2.1" />.{' '}
          {anchor ? (
            <>
              Below are all time-stamped rows linked to <span className="font-mono">{rowLabel(anchor)}</span> ({rowCaption(db, anchor)}). The accent
              line is the prediction time <em>t</em>
              {seedTime !== null ? <>, set to the simulation seed’s timestamp, {formatDate(seedTime, 'never')}</> : null}. Everything to its right is
              the future.
            </>
          ) : (
            <>The focus row has no time-stamped neighbours; pick a customer or a product to see a timeline.</>
          )}
        </p>
        {anchor && (
          <Figure wide title={`Activity of ${rowLabel(anchor)} over time`} caption="Each mark is one row; hover it for details, click it to make it the focus row.">
            <Timeline db={db} idx={idx} slots={slots} anchor={anchor} focus={lazyFocus} onFocus={setFocus} seedTime={seedTime} />
          </Figure>
        )}
        <p>
          Which rows actually carry a timestamp inside RT? The code takes it from the time column, in whole seconds, but it sets it only inside the
          loop over foreign keys <Cite kind="code" ref="rustler/src/pre.rs:481-493" />. A row therefore gets a timestamp only if its table has a
          time column, its time value is not null, and it has at least one non-null foreign key <Cite kind="spec" ref="§1.4" />.
        </p>
        <Figure title="Rows with a sampler timestamp, per table" caption="Computed from the live preprocessed graph.">
          <div className="scroll-quiet overflow-x-auto">
            <table className="w-full border-collapse text-[15.5px]">
              <thead>
                <tr className="border-b border-grid text-left align-bottom text-ink-2">
                  <th scope="col" className="py-1.5 pr-3 font-semibold">
                    table<span className="sm:hidden"> and time column</span>
                  </th>
                  <th scope="col" className="hidden py-1.5 pr-3 font-semibold sm:table-cell">
                    time column
                  </th>
                  <th scope="col" className="py-1.5 pr-3 text-right font-semibold">
                    <abbr title="foreign keys" className="no-underline sm:hidden">
                      FKs
                    </abbr>
                    <span className="hidden sm:inline">foreign keys</span>
                  </th>
                  <th scope="col" className="py-1.5 text-right font-semibold">
                    <span className="sm:hidden">with timestamp</span>
                    <span className="hidden sm:inline">rows with a timestamp</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {coverage.map((c) => (
                  <tr key={c.table} className="border-b border-grid/70">
                    <td className="py-1.5 pr-3">
                      <TableSwatch slot={slots[c.table] ?? 0} label={c.table} />
                      <div className="text-[14px] whitespace-nowrap text-ink-2 sm:hidden">
                        {c.timeCol ? <span className="font-mono">{c.timeCol}</span> : 'no time column'}
                      </div>
                    </td>
                    <td className="hidden py-1.5 pr-3 font-mono text-[15px] text-ink-2 sm:table-cell">{c.timeCol ?? '—'}</td>
                    <td className="num py-1.5 pr-3 text-right">{c.nFk}</td>
                    <td className="num py-1.5 text-right whitespace-nowrap">
                      {c.withTs.toLocaleString('en-US')} / {c.rows.toLocaleString('en-US')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Figure>
        {timeNoFk && (
          <p>
            A time column alone is therefore not enough.{' '}
            {timeNoFk.local ? `${dbShort} itself shows the edge case: its` : `${timeNoFk.title}, one of the other databases on this site, shows the edge case: its`}{' '}
            <code>{timeNoFk.table}</code> table declares <code>{timeNoFk.timeCol}</code> as its time column but has no foreign key, so{' '}
            {timeNoFk.withTs === 0 ? 'none' : timeNoFk.withTs} of its {timeNoFk.rows} rows get{timeNoFk.withTs <= 1 ? 's' : ''} a timestamp in the
            preprocessed graph <Cite kind="spec" ref="§1.4" />.
          </p>
        )}
        <Callout
          variant="paper-vs-code"
          title="Which rows does the time rule remove?"
          paper={
            <>
              Section 2.1 states the principle: modeling is conditioned only on rows created prior to the target row{' '}
              <Cite kind="paper" ref="p.3 §2.1" />. The sampler narrows it. Parent rows (F→P) are always included{' '}
              <Cite kind="paper" ref="p.4 §3.1" />, and a child q reached through a P→F link is kept only if <em>T</em>(q) ≤ <em>T</em>(s), its
              timestamp is not later than the seed’s <Cite kind="paper" ref="p.15 Alg. 1" />.
            </>
          }
          code={
            <>
              The same rule <Cite kind="code" ref="rustler/src/fly.rs:406-408" />
              <Cite kind="code" ref="rustler/src/fly.rs:421-424" />, applied only when both the child and the seed have a timestamp. A row has a
              timestamp only if it has a time value and a non-null foreign key <Cite kind="code" ref="rustler/src/pre.rs:481-493" />. Any other
              child always passes, and a seed without a timestamp switches the filter off for the whole context <Cite kind="spec" ref="§4.6" />.
            </>
          }
          verdict="Algorithm 1 and the code agree: only children are checked, and equal timestamps are kept. They differ from the broader §2.1 principle, and the code adds the missing-timestamp case."
        >
          <p>
            {noFkTables.length > 0 ? (
              <>
                In {dbShort}, {joinAnd(noFkTables)} hold{noFkTables.length === 1 ? 's' : ''} no foreign key, so {noFkTables.length === 1 ? 'its rows are' : 'their rows are'}{' '}
                never anyone’s child. They enter a context only as the seed or as a parent, and neither is ever time-filtered.{' '}
              </>
            ) : null}
            Chapter 6 shows the rule at work inside the sampler.
          </p>
        </Callout>
        <CodeRef
          title="A row’s timestamp is set inside the foreign-key loop"
          file={TS.file}
          lines={TS.lines}
          code={TS.code}
          note={
            <>
              Nanoseconds divided by 10⁹ and cast to <code>i32</code>: whole seconds. Because this runs only for non-null foreign-key values, a table
              without foreign keys ({noFkTables.join(', ')}) never gets a timestamp, time column or not <Cite kind="spec" ref="§1.4" />.
            </>
          }
        />
        <CodeRef
          title="Preview of chapter 6: parents unfiltered, children checked against the seed’s time"
          file={TEMPORAL.file}
          lines={TEMPORAL.lines}
          code={TEMPORAL.code}
          highlightLines={[406, 407, 408, 421, 422, 423, 424]}
          note={
            <>
              Lines 406-408 push every parent with no time test. Lines 421-424 drop a child only when both timestamps exist and the child’s is later
              than the seed’s. Line 417 concerns task tables (chapter 3) <Cite kind="spec" ref="§4.6" />.
            </>
          }
        />
      </Section>

      {/* ─────────────────────────────── 7 ─────────────────────────────── */}
      <Section id="graph" title="The database as a graph">
        <p>
          Put all the links together and the database becomes a graph. Relational deep learning (RDL) makes this explicit: it represents a database
          as a <em>relational entity graph</em>, “a temporal, heterogeneous graph where each table is a node-type, each row an individual node, and
          every primary-foreign key relationship an edge” <Cite kind="paper" ref="p.16 App. B" />. Graph neural networks are then applied directly
          to that graph.
        </p>
        <Figure
          wide
          title={`The neighbourhood of ${rowLabel(lazyFocus)} as an entity graph`}
          caption="Each node is one row, coloured and shaped by its table; each arrow is one foreign-key link, child to parent. Click a node to recentre."
        >
          <NeighbourhoodGraph db={db} idx={idx} slots={slots} focus={lazyFocus} onFocus={setFocus} />
        </Figure>
        <p>
          The paper’s criticism of these architectures is that they are schema-specific, which prevents pretraining and fine-tuning on diverse
          database structures <Cite kind="paper" ref="p.16 App. B" />. RT keeps the links but changes the unit. When the paper contrasts RT with
          another relational foundation model, one that first aggregates each table and then runs a graph neural network between tables, it
          describes its own design this way: “our model uses a cell-level representation of the entire database and employs attention masks to
          directly represent the foreign-key structure” <Cite kind="paper" ref="p.16 App. B" />.
        </p>
        <Figure
          wide
          title="Same rows, two representations"
          caption="Switch to the cell view: each row breaks into its feature cells, the key cells vanish as tokens, and the links remain as structure. This is not yet a context window; chapter 6 decides which rows enter one."
        >
          <CellsVsNodes db={db} idx={idx} slots={slots} focus={lazyFocus} />
        </Figure>
        <p>
          That is the raw material RT works with: a pool of cells, each knowing its value, its column and its table, plus the links between rows. The
          next chapter adds the last ingredient, the prediction task, which RT attaches to the database as one more table.
        </p>
      </Section>

      <div id="quiz" className="page-grid mt-16 scroll-mt-[calc(var(--topbar-h)+20px)]">
        <Quiz questions={quiz} />
      </div>

      <ChapterNav slug={SLUG} />
    </ChapterPage>
  );
}

/** "a", "a and b", "a, b and c". */
function joinAnd(xs: string[]): string {
  return xs.length <= 1 ? (xs[0] ?? '') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Quiz (live: question 2 uses a real row of the event table)
// ─────────────────────────────────────────────────────────────────────────────

function useQuiz(row: RowRef, parents: RowRef[], eventTable: string, idx: ReturnType<typeof buildLinkIndex>): QuizQuestion[] {
  return useMemo(() => {
    const t = idx.table(eventTable);
    const s = t?.schema;
    const cols = s?.columns.map((c) => c.name) ?? [];
    const pk = s?.pkey ?? 'id';
    const fks = Object.keys(s?.fkeys ?? {});
    const feats = cols.filter((c) => c !== pk && !fks.includes(c));
    const nonTime = feats.filter((c) => c !== s?.timeCol);
    const rec = t?.rows[row.row] ?? {};
    const parentText = parents.map((p) => rowLabel(p)).join(' and ');
    const firstParent = parents[0];
    const siblings = firstParent ? idx.childrenOf(firstParent).find((g) => g.childTable === eventTable)?.rows.length ?? 0 : 0;
    const code = (c: string) => <code key={c}>{c}</code>;
    const list = (cs: string[]) =>
      cs.map((c, i) => (
        <span key={c}>
          {i > 0 && (i === cs.length - 1 ? ' and ' : ', ')}
          {code(c)}
        </span>
      ));
    return [
      {
        prompt: (
          <>
            A row of the {eventTable} table has the columns {list(cols)}. Which of them become tokens?
          </>
        ),
        choices: [
          { text: <>All {cols.length} of them.</>, why: <>The primary key and the foreign keys never become tokens; they only define structure (spec §1.2).</> },
          {
            text: <>{list(nonTime)} only.</>,
            why: (
              <>
                The time column <code>{s?.timeCol}</code> is an ordinary datetime feature cell, so it is a token too; it also sets the row’s timestamp
                (spec §1.4).
              </>
            ),
          },
          { text: <>{list(feats)}.</>, correct: true, why: <>Exactly the non-missing feature cells (paper p.16 App. A). The time column is one of them.</> },
          { text: <>{list(fks)}, because they carry the links.</>, why: <>Foreign keys become links (parent lists and edges), never tokens (pre.rs:444-533).</> },
        ],
        explanation: <>RT tokenizes non-missing feature cells only. Keys are kept as structure, and missing cells disappear.</>,
      },
      {
        prompt: (
          <>
            {rowLabel(row)} has{' '}
            {fks.map((f, i) => (
              <span key={f}>
                {i > 0 && ' and '}
                <code>
                  {f} = {formatRaw(rec[f])}
                </code>
              </span>
            ))}
            . What are its F→P links?
          </>
        ),
        choices: [
          {
            text: firstParent ? <>The other {siblings - 1} {eventTable} rows of {rowLabel(firstParent)}.</> : <>Its sibling rows.</>,
            why: <>Those rows are other children of the same parent: they are reached through P→F links of the parent, not from this row.</>,
          },
          { text: <>{parentText}.</>, correct: true, why: <>One F→P link per non-null foreign-key value, pointing at the row with that position.</> },
          { text: <>None: F→P links start at parent rows.</>, why: <>It is the other way round: F→P links go from the row holding the foreign key to its parent.</> },
          firstParent
            ? { text: <>Only {rowLabel(firstParent)}.</>, why: <>Every non-null foreign key gives a link, so a row with {fks.length} foreign keys has {fks.length} parents.</> }
            : { text: <>Only one parent.</>, why: <>Every non-null foreign key gives a link.</> },
        ],
      },
      {
        prompt: <>Inside RT, a customers row of ShopDB has no row timestamp. Why?</>,
        choices: [
          { text: <>Because its signup_date is missing.</>, why: <>signup_date is not the table’s time column, and it is never missing in ShopDB.</> },
          {
            text: <>Because parent rows are never time-filtered.</>,
            why: <>True about the sampler (fly.rs:406-408), but that is a rule about using rows, not the reason the timestamp is empty.</>,
          },
          { text: <>Only task-table rows carry timestamps.</>, why: <>orders and reviews rows have timestamps too; see the table in the Time section.</> },
          {
            text: <>customers declares no time column and has no foreign key, and the timestamp is set only inside the foreign-key loop.</>,
            correct: true,
            why: <>pre.rs:481-493 sets the timestamp only for rows with a non-null foreign key, from the table’s time column (spec §1.4).</>,
          },
        ],
      },
    ];
  }, [row, parents, eventTable, idx]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case from another database: a table with a time column but no foreign key
// (spec §1.4). Computed with the real preprocessing, deferred so it never delays
// the first paint (~60 ms once, then cached by the engine).
// ─────────────────────────────────────────────────────────────────────────────

function useTimeColumnWithoutFk(current: string, enabled: boolean) {
  const [v, setV] = useState<{ title: string; table: string; timeCol: string; rows: number; withTs: number; local: false } | null>(null);
  useEffect(() => {
    let alive = true;
    setV(null);
    if (!enabled) return;
    const id = window.setTimeout(() => {
      try {
        for (const info of listDatabases()) {
          if (info.name === current) continue;
          const odb = getDatabase(info.name);
          const t = odb.tables.find((x) => x.schema.timeCol && Object.keys(x.schema.fkeys).length === 0);
          if (!t) continue;
          const pdb = getPreprocessed(odb);
          const ti = pdb.tableInfo[`${t.schema.name}:Db`];
          let withTs = 0;
          if (ti) for (let i = 0; i < ti.numNodes; i++) if (pdb.nodes[ti.nodeIdxOffset + i]?.ts !== null) withTs++;
          if (alive) setV({ title: info.title.split(' ')[0], table: t.schema.name, timeCol: t.schema.timeCol!, rows: t.rows.length, withTs, local: false });
          return;
        }
      } catch {
        /* optional aside: ignore */
      }
    }, 400);
    return () => {
      alive = false;
      window.clearTimeout(id);
    };
  }, [current, enabled]);
  return v;
}
