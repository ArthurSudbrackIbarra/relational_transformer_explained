/**
 * Chapter 4 "From rows to cells" (BUILD_PLAN §6, docs/CHAPTER_BRIEFS.md §4).
 *
 * How pre.rs (paper-era rustler/src/pre.rs @2a899c8) turns a database into
 * nodes and typed cells: global node indexing, which values become
 * cells, the four sem types, the three z-scoring schemes, and a cell inspector.
 * Every number on the page comes from the engine's pre.rs mirror for the live
 * database (src/engine/preprocess.ts through src/state/hooks.ts).
 */
import { useMemo } from 'react';
import {
  Callout,
  ChapterHeader,
  ChapterNav,
  ChapterPage,
  Cite,
  CodeRef,
  Figure,
  MathBlock,
  Quiz,
  ScaleNote,
  Section,
  Term,
} from '@/components';
import { SemType } from '@/engine/types';
import { BooleanNormFigure } from './BooleanNorm';
import { CellFatesChart } from './CellFates';
import { CellInspectorFigure, type CellPick } from './CellInspector';
import { booleanStats, colRole, isMissing, statColumns, type TableFile } from './data';
import { DatetimeNormFigure } from './DatetimeNorm';
import { IndexSpaceFigure } from './IndexSpace';
import { NumberNormFigure } from './NumberNorm';
import { PrecisionTable } from './Precision';
import { RowToCellsFigure, type RowPick } from './RowToCells';
import { SemTypeFigure } from './SemTypes';
import { CODE_FLOOR, Sub, signed, useChapterData, usePhone, type ChapterData } from './shared';
import {
  COMMON_SEMTYPE_38_44,
  FLY_BF16_482_490,
  MODEL_BCE_252_255,
  MODEL_EMB_216_224,
  PRE_CELLS_535_552,
  PRE_DT_342_376,
  PRE_GLOB_79_95,
  PRE_KEYS_439_479,
  PRE_OFFSETS_291_307,
  PRE_STATS_316_341,
  PRE_TRAINSTATS_383_396,
  PRE_VALUES_553_605,
} from './snippets';
import { LeakCounts, TrainStatsFigure } from './TrainStats';

const SLUG = 'preprocessing';
const nf = (n: number) => n.toLocaleString('en-US');
const singular = (s: string) => (s.endsWith('ies') ? s.slice(0, -3) + 'y' : s.endsWith('s') ? s.slice(0, -1) : s);

// ─────────────────────────────────────────────────────────────────────────────
// Live example picks (all derived from the current DB, task and seed)
// ─────────────────────────────────────────────────────────────────────────────

interface Picks {
  rows: RowPick[];
  cells: CellPick[];
  childFile?: TableFile;
  childRow?: number;
}

function findPicks(data: ChapterData): Picks {
  const { files, pdb, seedNode, entityNode, fileOfNode, task } = data;
  const rows: RowPick[] = [];
  const cells: CellPick[] = [];
  const at = (n: number) => {
    const f = fileOfNode(n);
    return f ? { f, row: n - f.offset } : null;
  };
  const seed = seedNode >= 0 ? at(seedNode) : null;
  const ent = entityNode !== null ? at(entityNode) : null;
  // a DB child of the entity (or of the seed row for autocomplete): e.g. one of the customer's orders
  const parentForChild = entityNode ?? seedNode;
  const childEdge = parentForChild >= 0 ? pdb.p2fAdj[parentForChild]?.find((e) => e.tableType === 'Db') : undefined;
  const child = childEdge ? at(childEdge.node) : null;
  // first DB row with a missing feature value
  let missing: { f: TableFile; row: number; col: string } | null = null;
  for (const f of files) {
    if (f.tableType !== 'Db' || missing) continue;
    for (let r = 0; r < f.table.rows.length && !missing; r++) {
      for (const c of f.table.schema.columns) {
        if (colRole(f.table, c.name) === 'feature' && isMissing(f.table.rows[r][c.name])) {
          missing = { f, row: r, col: c.name };
          break;
        }
      }
    }
  }

  // name the rows with live data: "the seed’s customer (Anna Müller)", "one of Anna Müller’s orders (orders[28])"
  const entName = ent ? pdb.nodes[entityNode!]?.cells.find((c) => c.sem === SemType.Text)?.raw : undefined;
  const entLabel = ent ? (typeof entName === 'string' ? entName : `${ent.f.tableName}[${ent.row}]`) : null;
  if (ent) rows.push({ label: `the seed’s ${singular(ent.f.tableName)} (${entLabel})`, key: ent.f.key, row: ent.row });
  if (child) rows.push({ label: `one of ${entLabel ? `${entLabel}’s` : 'the seed row’s'} ${child.f.tableName} (${child.f.tableName}[${child.row}])`, key: child.f.key, row: child.row });
  if (seed) rows.push({ label: 'the seed row', key: seed.f.key, row: seed.row });
  if (missing && !(ent && missing.f.key === ent.f.key && missing.row === ent.row)) rows.push({ label: 'a row with a missing value', key: missing.f.key, row: missing.row });

  // cells by type, preferring the entity row, then the child row, then the seed row
  // (datetimes: the child first, whose time column also sets the row timestamp)
  const holders = [ent, child, seed].filter((h): h is { f: TableFile; row: number } => !!h);
  const dtHolders = [child, seed, ent].filter((h): h is { f: TableFile; row: number } => !!h);
  const typeLabel: Record<number, string> = { [SemType.Number]: 'a number', [SemType.Text]: 'a text', [SemType.DateTime]: 'a datetime', [SemType.Boolean]: 'a boolean' };
  for (const sem of [SemType.Number, SemType.Text, SemType.Boolean, SemType.DateTime]) {
    for (const h of sem === SemType.DateTime ? dtHolders : holders) {
      const node = pdb.nodes[h.f.offset + h.row];
      const cell = node?.cells.find((c) => c.sem === sem && !(seed && h === seed && c.colName === task.targetCol));
      if (cell) {
        cells.push({ label: typeLabel[sem], key: h.f.key, row: h.row, col: cell.colName });
        break;
      }
    }
  }
  if (seed && seed.f.table.schema.columns.some((c) => c.name === task.targetCol)) cells.push({ label: 'the seed’s label', key: seed.f.key, row: seed.row, col: task.targetCol });
  if (missing) cells.push({ label: 'a missing value', key: missing.f.key, row: missing.row, col: missing.col });
  if (child) {
    const fk = Object.entries(child.f.table.schema.fkeys).find(([, p]) => ent && p === ent.f.tableName)?.[0] ?? Object.keys(child.f.table.schema.fkeys)[0];
    if (fk) cells.push({ label: 'a foreign key', key: child.f.key, row: child.row, col: fk });
  }
  const pkHolder = holders.find((h) => h.f.table.schema.pkey);
  if (pkHolder) cells.push({ label: 'a primary key', key: pkHolder.f.key, row: pkHolder.row, col: pkHolder.f.table.schema.pkey! });

  return { rows, cells, childFile: child?.f, childRow: child?.row };
}

// ─────────────────────────────────────────────────────────────────────────────
// The three passes of pre.rs (orientation figure)
// ─────────────────────────────────────────────────────────────────────────────

const PASSES = [
  {
    n: 1,
    title: 'Read the tables',
    lines: 'pre.rs:79-307',
    what: 'Every parquet file, DB tables then task splits. Each file gets its node_idx_offset, which fixes every row’s global index.',
    to: 'index',
    order: 'read 1st here',
  },
  {
    n: 2,
    title: 'Compute column statistics',
    lines: 'pre.rs:310-396',
    what: 'Mean and standard deviation per number and boolean column; one global pair for all datetimes.',
    to: 'normalization',
    order: 'read 3rd here',
  },
  {
    n: 3,
    title: 'Build the node vector',
    lines: 'pre.rs:399-615',
    what: 'Per table, column by column over all rows: links and edges from foreign keys, one typed cell per non-missing feature value (z-scored with the pass-2 stats), and the string table.',
    to: 'cells',
    order: 'read 2nd here',
  },
];

function PassesMap() {
  return (
    <ol className="wide !my-6 grid gap-3 !p-0 md:grid-cols-3" style={{ listStyle: 'none' }}>
      {PASSES.map((p, i) => (
        <li key={p.n} className="!m-0 !p-0">
          <a
            href={`#/${SLUG}?section=${p.to}`}
            data-plain
            onClick={(e) => {
              e.preventDefault();
              document.getElementById(p.to)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }}
            className="group relative flex h-full flex-col rounded-xl border border-border bg-surface px-4 py-3.5 transition-colors hover:border-accent"
          >
            <span className="flex items-center gap-2">
              <span className="num flex h-6 w-6 items-center justify-center rounded-full bg-accent text-[14.5px] font-semibold text-accent-ink">{p.n}</span>
              <span className="text-[17px] font-semibold text-ink">{p.title}</span>
            </span>
            <span className="mt-2 text-[16px] leading-snug text-ink-2">{p.what}</span>
            <span className="mt-auto flex flex-wrap items-baseline justify-between gap-x-3 pt-2">
              <span className="font-mono text-[14px] text-muted">{p.lines}</span>
              <span className="text-[14px] font-medium text-accent-text">{p.order}</span>
            </span>
            {i < PASSES.length - 1 && (
              <span aria-hidden="true" className="absolute top-1/2 -right-[11px] z-10 hidden h-5 w-5 -translate-y-1/2 items-center justify-center rounded-full border border-border bg-page text-[14.5px] text-muted md:flex">
                →
              </span>
            )}
          </a>
        </li>
      ))}
    </ol>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Chapter
// ─────────────────────────────────────────────────────────────────────────────

export default function PreprocessingChapter() {
  const data = useChapterData();
  const phone = usePhone();
  const { db, files, pdb, task, seed, entityNode } = data;
  const picks = useMemo(() => findPicks(data), [data]);
  const numCols = useMemo(() => statColumns(files, pdb, 'number'), [files, pdb]);
  const boolCols = useMemo(() => statColumns(files, pdb, 'boolean'), [files, pdb]);
  const dbShort = db.title.split(' (')[0];
  const totalRows = files.reduce((s, f) => s + f.numNodes, 0);
  const entity = entityNode !== null ? pdb.nodes[entityNode] : undefined;
  const entityFile = entityNode !== null ? data.fileOfNode(entityNode) : undefined;
  const entityName = entity?.cells.find((c) => c.sem === SemType.Text)?.raw;
  // a binary forecast task whose label pre.rs stores as a Boolean column (the live task when it is one)
  const binTask = task.kind === 'forecast' && task.targetType === 'binary' ? task : db.tasks.find((t) => t.kind === 'forecast' && t.targetType === 'binary');
  const binTrain = binTask ? files.find((f) => f.key === `${binTask.name}:Train`) : undefined;
  const binIsBool = !!binTrain && binTrain.table.schema.columns.some((c) => c.name === binTask!.targetCol && c.dtype === 'bool');
  const firstDb = files.filter((f) => f.tableType === 'Db');
  // a pair of DB tables whose glob (byte-wise path) order differs from the schema order
  const globPair = (() => {
    const schemaOrder = db.tables.map((t) => t.schema.name);
    for (let i = 0; i < firstDb.length; i++)
      for (let j = i + 1; j < firstDb.length; j++)
        if (schemaOrder.indexOf(firstDb[i].tableName) > schemaOrder.indexOf(firstDb[j].tableName)) return [firstDb[i].tableName, firstDb[j].tableName] as const;
    return null;
  })();
  const taskFiles = files.filter((f) => f.tableType !== 'Db');

  // defaults for the normalization figures: the entity's table first, with the entity's own values as probes
  const numDefault = numCols.find((c) => entityFile && c.file.key === entityFile.key)?.id ?? numCols[0]?.id ?? '';
  const numProbe: Record<string, number> = {};
  if (entity && entityFile) {
    for (const c of numCols) {
      if (c.file.key !== entityFile.key) continue;
      const v = entityFile.table.rows[entityNode! - entityFile.offset]?.[c.col.name];
      if (typeof v === 'number' && !Number.isNaN(v)) numProbe[c.id] = v;
    }
  }
  const boolDefault = boolCols.find((c) => entityFile && c.file.key === entityFile.key)?.id ?? boolCols[0]?.id ?? '';
  const k = `${db.name}:${data.seedNode}`;
  // the default boolean column at 25% true, for the quiz explanation (sample std, finite n)
  const quarter = (() => {
    const c = boolCols.find((x) => x.id === boolDefault);
    if (!c || c.values.length < 4) return null;
    const n = c.values.length;
    const kq = Math.round(n / 4);
    // the pure n − 1 effect at exactly p = 0.25, to separate it from the rounding of k
    const zExact = 0.75 / Math.sqrt((0.1875 * n) / (n - 1));
    return { n, k: kq, exact: kq * 4 === n, pct: (100 * kq) / n, zTrue: booleanStats(n, kq).zTrue, zExact };
  })();
  // a task in this DB that drops leakage columns (the live task first), for the "cells are not tokens yet" paragraph
  const leakTask = task.columnsToDrop.length > 0 ? task : db.tasks.find((t) => t.columnsToDrop.length > 0);

  return (
    <ChapterPage slug={SLUG} className={CODE_FLOOR}>
      <ChapterHeader
        slug={SLUG}
        lede={
          <>
            Before the Relational Transformer sees anything, a preprocessing program runs once per database. It turns the database into one long list of rows, and every row into a
            few typed cells, with numbers, booleans and datetimes already normalized. This chapter follows that program on {dbShort}, value by value.
          </>
        }
        learn={[
          'How every row of every table, task splits included, gets one global node index.',
          'Which values become cells: keys turn into links and missing values vanish.',
          'How the physical dtype alone decides a cell’s semantic type.',
          'Exactly how numbers, booleans and datetimes are z-scored, and which rows feed the statistics.',
          'What one cell looks like after preprocessing, slot by slot.',
        ]}
      />

      <Section id="overview" title="One program, run before everything else">
        <p>
          RT never reads raw tables while it trains or predicts. The official code first runs a Rust program, <code>pre.rs</code>, once per database. It writes what the sampler
          needs: one record per row, the links between rows, every <Term id="feature-column">feature</Term> value already converted into numbers, and one string table that a
          separate script embeds with the frozen text encoder <Cite kind="spec" ref="§3" /> <Cite kind="code" ref="rustler/src/pre.rs:619-698" title="files written by pre.rs" />{' '}
          <Cite kind="code" ref="rt/embed.py:14-31" />. It works in three passes.
        </p>
        <PassesMap />
        <p>
          We take the passes in the order that is easiest to learn: the index space first, then which values become cells, then the statistics that normalize them.
        </p>
        <p>
          Everything below is computed live in your browser, on {db.title}, by a TypeScript mirror of <code>pre.rs</code>: the same rows, links, cells and statistics. Its few
          bookkeeping differences are listed <a href={`#/${SLUG}?section=mirror`}>at the end of the chapter</a>. The figures follow the simulation settings in the top bar, so the
          seed row you see here is the same one every other chapter uses
          {seed ? (
            <>
              : row {seed.rowPos} of <code>{seed.tableKey}</code>
              {entity && entityFile ? (
                <>
                  , which points at {entityFile.tableName}[{entity.rowPos}]{typeof entityName === 'string' ? ` (${entityName})` : ''}
                </>
              ) : null}
            </>
          ) : null}
          .
        </p>
      </Section>

      <Section id="index" kicker="Pass 1" title="One index space for every row">
        <p>
          <code>pre.rs</code> gives every row of every table a single integer, its <Term id="global-node-index">global node index</Term>. “Every table” includes the{' '}
          <Term id="task-table">task tables</Term>: the Train, Val and Test splits of each forecasting task are files of their own, and their rows are numbered in the same space as
          the database rows <Cite kind="spec" ref="§1.5" />.
        </p>
        <p>
          The order is simply the order in which the files are read: first the database tables <code>db/*.parquet</code>, sorted by path, then the task files{' '}
          <code>tasks/&lt;task&gt;/&lt;split&gt;.parquet</code>, sorted by path <Cite kind="code" ref="rustler/src/pre.rs:79-92" title="glob order" />. Sorting by path compares the
          names byte by byte; for lowercase names like these, that is alphabetical order. That is why{' '}
          {globPair ? (
            <>
              <code>{globPair[0]}</code> comes before <code>{globPair[1]}</code> here, although the schema lists them the other way round, and why
            </>
          ) : null}{' '}
          a task’s test split comes before its train split: <code>test</code> &lt; <code>train</code> &lt; <code>val</code>.
        </p>
        <p>
          A file’s offset is the number of rows read before it, so a row’s index is <code>node_idx_offset</code> + its row position{' '}
          <Cite kind="code" ref="rustler/src/pre.rs:291-306" title="node_idx_offset = num_rows_sum" /> <Cite kind="code" ref="rustler/src/pre.rs:466" />. The offsets are saved in{' '}
          <code>table_info.json</code> under keys like <code>{firstDb[0]?.key ?? 'customers:Db'}</code> or <code>{taskFiles[taskFiles.length - 1]?.key ?? 'customer-churn:Val'}</code>{' '}
          <Cite kind="code" ref="rustler/src/pre.rs:648-664" />.
        </p>
        <Figure
          wide
          title={`${dbShort}: ${nf(totalRows)} rows in ${files.length} files, one index space`}
          caption={
            <>
              Press play to replay pass 1: <code>pre.rs</code> reads the files in glob order and each file’s offset is the running row count. Segments are coloured by table; all
              task tables share magenta. The accent pin is the live seed row; the other pin follows the row picked below.
            </>
          }
        >
          <IndexSpaceFigure key={k} data={data} />
        </Figure>
        <p>
          Why one shared space? The attention masks and the sampler compare rows only through these integers: “is this token in the same row as that one?”, “is that row one of my
          parents?” <Cite kind="spec" ref="§5.1" /> <Cite kind="spec" ref="§6.2" />. One integer per row answers those questions the same way for every table. Note also what is{' '}
          <em>not</em> in the bar: autocomplete tasks add no table, because their seeds are ordinary database rows <Cite kind="spec" ref="§2.4" />.
        </p>
        <CodeRef title="Pass 1: files in glob order (DB tables, then task splits)" file="rustler/src/pre.rs" lines="79-95" code={PRE_GLOB_79_95} note="Lines 93-95 skip link-prediction tasks, whose directory names carry two extra dashes; RT cannot do link prediction (paper p.10 §7)." />
        <CodeRef title="Each file’s node_idx_offset is the running row count" file="rustler/src/pre.rs" lines="291-307" code={PRE_OFFSETS_291_307} highlightLines={[302, 305]} />
      </Section>

      <Section id="cells" kicker="Pass 3" title="Keys become links, missing values vanish">
        <p>
          Inside a table, <code>pre.rs</code> walks the columns in their stored order and applies three rules to every value <Cite kind="spec" ref="§1.2" />{' '}
          <Cite kind="spec" ref="§3.5" />:
        </p>
        <ul>
          <li>
            The <Term id="primary-key">primary key</Term> produces nothing: no cell and no edge <Cite kind="code" ref="rustler/src/pre.rs:439-442" />. Its value is the row
            position, which the node index already encodes.
          </li>
          <li>
            A <Term id="foreign-key">foreign key</Term> produces no cell either. Its value becomes a link: the parent’s node index joins the row’s <code>f2p_nbr_idxs</code>, and
            the parent gets an edge back <Cite kind="code" ref="rustler/src/pre.rs:444-533" />. This is also where the row’s <Term id="node-timestamp">timestamp</Term> is set,
            from its <Term id="time-column">time column</Term> <Cite kind="code" ref="rustler/src/pre.rs:481-493" />.
          </li>
          <li>
            Every other value is a feature value. If it is present it becomes one <strong>cell</strong> record. If it is missing (null, or a float NaN) it produces nothing at all:
            no cell, no placeholder, no imputation <Cite kind="code" ref="rustler/src/pre.rs:552" /> <Cite kind="code" ref="rustler/src/pre.rs:566-568" />.
          </li>
        </ul>
        <p>
          The paper says the same in one line: once a row is selected, “all its non-missing feature cells (i.e., cells not from primary- or foreign-key columns) are included in the
          context” <Cite kind="paper" ref="p.16 App. A" />. Key columns shape the graph; only feature columns carry values.
        </p>
        <Figure
          wide
          title="One row through pass 3, column by column"
          caption={
            <>
              Press play (or step) to watch each value land in its lane. The node record at the bottom fills up as <code>pre.rs</code> goes. <code>pre.rs</code> itself sweeps
              each column down all rows before it moves to the next column; here we follow one row’s slice of that sweep, which ends in the same node record. The signed numbers
              in the Cells lane are z-scores from pass 2 (see <a href={`#/${SLUG}?section=normalization`}>Normalization</a> below): roughly, how many standard deviations a value
              sits above or below its column’s average. Picking another row here does not change the global seed.
            </>
          }
        >
          <RowToCellsFigure key={k} data={data} picks={picks.rows} />
        </Figure>
        <p>
          These cell records are not tokens yet. A cell becomes a <Term id="cell-token">token</Term> only when the sampler picks its row for a context window (Chapter 6).
          Even then one rule can skip it, and it belongs to the task, not to preprocessing: a task may name <strong>leakage columns</strong>, and the sampler leaves those cells
          out for the seed row and for every row with the seed’s timestamp <Cite kind="code" ref="rustler/src/fly.rs:457-460" /> <Cite kind="spec" ref="§4.9" />
          {leakTask ? (
            <>
              {' '}
              (in {dbShort}, <code>{leakTask.kind === 'autocomplete' ? `${leakTask.table}.${leakTask.targetCol}` : leakTask.name}</code> drops{' '}
              {leakTask.columnsToDrop.map((c, i) => (
                <span key={c}>
                  {i > 0 && (i === leakTask.columnsToDrop.length - 1 ? ' and ' : ', ')}
                  <code>{c}</code>
                </span>
              ))}
              )
            </>
          ) : null}
          . <code>pre.rs</code> keeps those cells like any other. Across the whole database the rules add up as follows.
        </p>
        <Figure wide>
          <CellFatesChart data={data} />
        </Figure>
        <Callout variant="note" title="Key names are thrown away">
          <p>
            A foreign key survives only as links: the parent’s node index in <code>f2p_nbr_idxs</code>, plus two edges for the sampler, one on the child (F→P) and one on the
            parent (P→F) <Cite kind="code" ref="rustler/src/pre.rs:515-529" />. Its column name is interned in the string table like every column’s, but it never reaches a token{' '}
            <Cite kind="code" ref="rustler/src/pre.rs:432-437" />. The paper lists this as a limitation: RT “does not incorporate the names of primary-key and foreign-key
            columns” and cannot tell two foreign keys to the same table apart <Cite kind="paper" ref="p.10 §7" />. <code>pre.rs</code> keeps every
            parent, but the sampler’s batch has room for five per token (<code>MAX_F2P_NBRS = 5</code>) and asserts it, so a row should have at most five non-null foreign keys{' '}
            <Cite kind="code" ref="rustler/src/fly.rs:25" /> <Cite kind="code" ref="rustler/src/fly.rs:464" /> <Cite kind="spec" ref="§1.2" />.
          </p>
        </Callout>
        <CodeRef title="Primary key → nothing; foreign key → f2p_nbr_idxs" file="rustler/src/pre.rs" lines="439-479" code={PRE_KEYS_439_479} highlightLines={[439, 440, 441, 442, 462, 463, 464, 477, 479]} />
        <CodeRef title="Every other column: cast, and null → no cell" file="rustler/src/pre.rs" lines="535-552" code={PRE_CELLS_535_552} highlightLines={[552]} />
      </Section>

      <Section id="types" kicker="Pass 3, continued" title="Four semantic types, read off the dtype">
        <p>
          Every cell carries a <Term id="sem-type">semantic type</Term>. There are four, with fixed codes from the <code>SemType</code> enum: Number = 0, Text = 1, DateTime = 2,
          Boolean = 3 <Cite kind="code" ref="rustler/src/common.rs:38-44" />. The model later loops over the types in exactly this order to pick an encoder, a mask vector and a
          decoder for each cell <Cite kind="code" ref="rt/model.py:216" /> <Cite kind="code" ref="rt/model.py:238" />.
        </p>
        <p>
          The type comes from the column’s physical dtype and from nothing else <Cite kind="spec" ref="§3.2" /> <Cite kind="code" ref="rustler/src/pre.rs:543-611" />:
        </p>
        <Figure wide title={`dtype → semantic type, with every feature column of ${dbShort}`} caption="Task-table columns appear once: the three splits share one table name and one column phrase.">
          <SemTypeFigure data={data} />
        </Figure>
        <p>
          There is no semantic inference. A string column is Text even when it is really a category, so a country code is embedded as a short string. An integer column that is
          not declared as a key is a Number, so a 1-to-5 rating is z-scored like a price, and an integer id that is not declared as a key would be too{' '}
          <Cite kind="spec" ref="§3.2" />.
        </p>
        <Callout
          variant="paper-vs-code"
          title="Is boolean a type of its own?"
          paper={
            <>
              §2.1 defines feature columns as those that “contain numeric, text, and datetime information” <Cite kind="paper" ref="p.3 §2.1" />. §3.1 then says a value “can be
              numeric, boolean, datetime, or text”, normalizes numeric and boolean values in one bullet <Cite kind="paper" ref="p.4 §3.1" />, and §3.3 has one decoder per datatype,
              where “Binary classification corresponds to the boolean datatype” <Cite kind="paper" ref="p.6 §3.3" />.
            </>
          }
          code={
            <>
              Boolean is a sem type of its own, code 3 <Cite kind="code" ref="rustler/src/common.rs:38-44" />, with its own entries in the encoder, normalization, mask-vector and
              decoder dictionaries <Cite kind="code" ref="rt/model.py:115-146" />. It uses the same z-score formula as numbers, but without the number branch’s σ guards (see the
              warning below) <Cite kind="code" ref="rustler/src/pre.rs:320-341" />.
            </>
          }
        >
          <p>
            So the code settles what the paper leaves loose: four types, not three <Cite kind="spec" ref="§10" note="Row 17 of the discrepancy table." />. It matters beyond the
            input: the sem type of the masked cell picks the loss, binary cross-entropy for a boolean and Huber for a number (Chapter 12){' '}
            <Cite kind="code" ref="rt/model.py:250-255" /> <Cite kind="spec" ref="§2.6" />.
            {binTask && binIsBool ? (
              <>
                {' '}
                In {dbShort}, the label <code>{binTask.targetCol}</code> of <code>{binTask.name}</code> is stored as a boolean column from the start.
              </>
            ) : null}
          </p>
          <details className="group mt-1 text-[16.5px]">
            <summary className="cursor-pointer font-medium text-ink-2 hover:text-ink">On the RelBench datasets: which label columns become booleans</summary>
            <p className="!mt-2">
              <code>pre.rs</code> hard-codes casts to Boolean for the RelBench label columns it knows, such as <code>did_not_finish</code>{' '}
              <Cite kind="code" ref="rustler/src/pre.rs:208-289" />; an integer label column that is not on that list stays a Number and is trained with Huber. One quirk:
              rel-event’s <code>user-attendance</code> is listed as a regression task, yet its label is cast to Boolean, so it is trained with BCE{' '}
              <Cite kind="code" ref="rustler/src/pre.rs:272-278" /> <Cite kind="spec" ref="§2.6" />.
            </p>
          </details>
        </Callout>
        <CodeRef title="SemType: the four codes" file="rustler/src/common.rs" lines="38-44" code={COMMON_SEMTYPE_38_44} />
      </Section>

      <Section id="normalization" kicker="Pass 2" title="Normalization: three kinds of z-score">
        <p>
          Numbers, booleans and datetimes are stored as <Term id="z-score">z-scores</Term>: subtract a mean, divide by a standard deviation. Text is not normalized at all; it is
          stored as an index into the database’s string table and embedded later by the frozen <Term id="text-encoder">text encoder</Term> <Cite kind="spec" ref="§3.3" />{' '}
          <Cite kind="spec" ref="§3.6" />. There is no clipping anywhere. The paper writes the shared formula as
        </p>
        <MathBlock tex={String.raw`r = \frac{v - \mu_c}{\sigma_c}`} label={<Cite kind="paper" ref="p.4 §3.1" />} />
        <p>The three kinds differ only in which values feed μ and σ, and in how σ is computed.</p>

        <h3 id="numbers" className="mt-6 mb-2 text-[21px] font-semibold tracking-[-0.01em] text-ink">
          Numbers: per column, sample standard deviation
        </h3>
        <p>
          For each number column, <code>pre.rs</code> drops nulls and NaNs, then takes the mean μ and the <strong>sample</strong> standard deviation σ, which divides by n − 1
          (ddof = 1). Two guards keep the division safe: σ becomes 1 when it is undefined (n ≤ 1) or exactly 0 <Cite kind="code" ref="rustler/src/pre.rs:330-341" />.
        </p>
        <Figure
          wide
          caption={
            <>
              The z-score is a new ruler under the same data: 0 sits on the mean and one unit is one σ (the shaded band is μ ± σ). Bars are coloured by table. Default probe: the
              seed’s own {entityFile ? singular(entityFile.tableName) : 'row'}, where it applies.
            </>
          }
        >
          <NumberNormFigure key={k} data={data} cols={numCols} defaultId={numDefault} defaultProbe={numProbe} />
        </Figure>
        <p>
          The paper does not spell out why. Here is our reading, not a claim of the paper: one shared encoder per type, a <code>Linear(1, d)</code> with d = 256 in the paper and
          64 here <ScaleNote inline />, reads every number column of every database <Cite kind="code" ref="rt/model.py:115-123" />. Standardizing first means it sees “how unusual
          is this value within its column” on a common scale, whether the column holds prices or ages.
        </p>

        <h3 id="booleans" className="mt-8 mb-2 text-[21px] font-semibold tracking-[-0.01em] text-ink">
          Booleans: true lands above zero (when both values occur)
        </h3>
        <p>
          Booleans use the same formula after casting true → 1 and false → 0 <Cite kind="code" ref="rustler/src/pre.rs:320-329" />. With a share p of trues, μ = p, so true becomes
          (1 − μ)/σ, which is positive, and false becomes −μ/σ, which is negative. The values are not ±1: they depend on how common <code>true</code> is, and a rare{' '}
          <code>true</code> sits far out on the positive side.
        </p>
        <p>
          The sign is what the rest of the code relies on. The loss for a boolean target is binary cross-entropy against 1{'{'}z &gt; 0{'}'}, and evaluation turns labels into{' '}
          <code>int(z &gt; 0)</code> <Cite kind="code" ref="rt/model.py:252-255" /> <Cite kind="code" ref="rt/main.py:336" /> <Cite kind="paper" ref="p.6 §3.3" />. Both read
          only the masked target cell <Cite kind="code" ref="rt/model.py:241" /> <Cite kind="code" ref="rt/main.py:284-285" />, so “true ⇔ z &gt; 0” must hold for every boolean column that can be a prediction target, and it
          does whenever both values occur <Cite kind="spec" ref="§3.3" />. For the other boolean columns the sign has no special role: they are simply inputs.
        </p>
        <Figure
          wide
          title="Where true and false land on the z axis"
          caption={
            <>
              Filled dot: true; hollow dot: false. Move the slider, or use the chips, to change how many values are true (n stays fixed); the dashed dots keep the real column for
              comparison. “All true” and “all false” show what a constant column does.
            </>
          }
        >
          <BooleanNormFigure key={k} data={data} cols={boolCols} defaultId={boolDefault} />
        </Figure>
        <Callout variant="warning" title="A constant boolean column poisons the batch">
          <p>
            Unlike numbers, booleans have no σ = 0 guard: an undefined σ even defaults to 0 <Cite kind="code" ref="rustler/src/pre.rs:323" />. If every value is the same, each one
            becomes (b − μ)/0 = 0/0 = NaN. Masking does not remove it (NaN · 0 = NaN). As soon as such a cell enters a context window, attention spreads the NaN to every
            non-padding token of that sequence, and the batch loss becomes NaN. The code does not check for it <Cite kind="spec" ref="§3.3" />. That is why none of this site’s databases has a constant boolean column, and why every task’s Train split
            contains both labels.
          </p>
        </Callout>
        <CodeRef title="Per-column stats: booleans (no zero guard) and numbers (σ → 1 guards)" file="rustler/src/pre.rs" lines="316-341" code={PRE_STATS_316_341} highlightLines={[323, 338, 339]} />
        <CodeRef title="The boolean target is 1{z > 0}" file="rt/model.py" lines="252-255" code={MODEL_BCE_252_255} />

        <h3 id="datetimes" className="mt-8 mb-2 text-[21px] font-semibold tracking-[-0.01em] text-ink">
          Datetimes: one clock for the whole database
        </h3>
        <p>
          Datetimes get no per-column statistics. Instead <code>pre.rs</code> pools every datetime value of every table, task splits included, into one count, one sum and one sum
          of squares. From them it derives a single mean μ<Sub>T</Sub> and a <strong>population</strong> standard deviation σ<Sub>T</Sub>, which divides by n (ddof = 0) and has
          no zero guard <Cite kind="code" ref="rustler/src/pre.rs:342-376" /> <Cite kind="spec" ref="§3.3" />.
        </p>
        <MathBlock
          tex={
            phone
              ? String.raw`\begin{aligned} z &= \frac{t - \mu_T}{\sigma_T} \\[4pt] \mu_T &= \frac{1}{n}\sum t \\[4pt] \sigma_T &= \sqrt{\frac{1}{n}\sum t^2 - \mu_T^2} \end{aligned}`
              : String.raw`z = \frac{t - \mu_T}{\sigma_T}, \qquad \mu_T = \frac{1}{n}\sum t, \qquad \sigma_T = \sqrt{\frac{1}{n}\sum t^2 - \mu_T^2}`
          }
          label={<Cite kind="code" ref="rustler/src/pre.rs:375-376" />}
        />
        <Figure wide caption="Each row is one datetime column of one table file. The accent triangle marked “seed” on the z ruler is the seed row’s timestamp.">
          <DatetimeNormFigure key={k} data={data} />
        </Figure>
        <p>
          The paper says to “normalize globally” but does not say why <Cite kind="paper" ref="p.5 §3.1" />. One consequence follows from the code (our reading, not a claim of
          the paper): the same instant gets the same value in every column. A task row’s timestamp and an order’s time are on one scale, so the model can compare them directly.
        </p>
        <Callout variant="note" title="Seconds, nanoseconds, milliseconds">
          <p>
            The paper converts timestamps to seconds <Cite kind="paper" ref="p.5 §3.1" />; the code uses nanoseconds <Cite kind="code" ref="rustler/src/pre.rs:586" />; this site
            stores milliseconds. All three give the same z-score: changing the unit scales t, μ<Sub>T</Sub> and σ<Sub>T</Sub> by the same factor <Cite kind="spec" ref="§3.3" />.
            Note the two kinds of standard deviation in this section: sample (n − 1) for numbers and booleans, population (n) for datetimes.
          </p>
        </Callout>
        <CodeRef title="One global datetime mean and population std" file="rustler/src/pre.rs" lines="342-376" code={PRE_DT_342_376} highlightLines={[346, 347, 348, 375, 376]} />

        <h3 id="labels" className="mt-8 mb-2 text-[21px] font-semibold tracking-[-0.01em] text-ink">
          Task labels: Train statistics for every split
        </h3>
        <p>
          Task tables get one extra step. <code>pre.rs</code> first computes stats for every file separately, then overwrites the Val and Test splits’ stats with the same task’s
          Train stats <Cite kind="code" ref="rustler/src/pre.rs:383-396" />. A Val label is therefore normalized with the Train mean and σ, never with its own.
        </p>
        <Figure wide caption="The split’s own label distribution differs from Train’s, but the μ and σ used are Train’s, so the same label is stored as the same value in every split.">
          <TrainStatsFigure data={data} />
        </Figure>
        <p>
          For task labels the code does what the paper asks: statistics “computed on the training split” <Cite kind="paper" ref="p.4 §3.1" />. The labels being evaluated do not
          shape their own encoding, and because the three splits share one table name and one column phrase (Chapter 3), a Train label seen in the context and the Val label being
          predicted are on the same scale <Cite kind="spec" ref="§2.2" />.
        </p>
        <Callout
          variant="paper-vs-code"
          title="Which rows feed the statistics"
          paper={
            <>
              μ<Sub>c</Sub> and σ<Sub>c</Sub> are “computed on the training split”, and μ<Sub>T</Sub>, σ<Sub>T</Sub> are the mean and standard deviation “of timestamps in the
              training split” <Cite kind="paper" ref="p.4 §3.1" /> <Cite kind="paper" ref="p.5 §3.1" />.
            </>
          }
          code={
            <>
              Task labels: Train stats, reused for Val and Test. Database tables: <strong>all rows</strong> of the table file, with no cutoff anywhere. Datetimes: every table and
              every split <Cite kind="code" ref="rustler/src/pre.rs:316-396" /> <Cite kind="spec" ref="§3.4" />.
            </>
          }
        >
          <p>Concretely, in {dbShort}:</p>
          <LeakCounts data={data} />
          <p>
            The authors later described this normalization as leaky and fixed it in a later model line; its effect on the paper’s RT was not measured{' '}
            <Cite kind="spec" ref="§3.4" /> <Cite kind="spec" ref="§10" note="Row 3 of the discrepancy table." />.
          </p>
        </Callout>
        <CodeRef title="Val and Test task splits copy the Train stats" file="rustler/src/pre.rs" lines="383-396" code={PRE_TRAINSTATS_383_396} highlightLines={[385, 386, 391, 392]} />
      </Section>

      <Section id="inspector" title="Pick a cell, see its encoding">
        <p>
          Here is the whole chapter in one panel. Click any cell of any table, task splits included. The panel walks it through <code>pre.rs</code>: its role, its semantic type,
          the statistics that normalize it, the record slots it fills, and the embedding term that will read it in Chapter 8.
        </p>
        <Figure wide title="The cell inspector" caption={<>Picking cells here is local to this figure; the global seed does not change.</>}>
          <CellInspectorFigure key={k} data={data} picks={picks.cells} />
        </Figure>
        <p>
          Two details stand out in the slot view. Every cell record has all four value slots, and only the one that matches its type is filled; a non-text cell stores text index
          0 <Cite kind="code" ref="rustler/src/pre.rs:553-605" />. The model multiplies each type’s term by a 0/1 gate on the sem type, so the empty slots never contribute{' '}
          <Cite kind="code" ref="rt/model.py:216-224" />. And the seed’s own label is stored like any other cell: preprocessing knows nothing about masking. The sampler marks it
          as the target later, and the model swaps its value term for a learned <Term id="mask-vector">mask vector</Term> <Cite kind="spec" ref="§5.3" />.
        </p>
        <CodeRef title="What each type stores in the cell record" file="rustler/src/pre.rs" lines="553-605" code={PRE_VALUES_553_605} />
        <CodeRef title="Preview of Chapter 8: one value term per cell, picked by its sem type" file="rt/model.py" lines="216-224" code={MODEL_EMB_216_224} />
      </Section>

      <Section id="precision" title="A note on precision">
        <p>
          <code>pre.rs</code> stores number, datetime and boolean values as 32-bit floats <Cite kind="code" ref="rustler/src/common.rs:57-60" />. When the sampler builds a batch, it rounds number,
          datetime and boolean values to bfloat16, which keeps 8 significant bits, about two to three decimal digits <Cite kind="code" ref="rustler/src/fly.rs:482-490" />{' '}
          <Cite kind="spec" ref="§3.7" />. The paper runs pretraining and fine-tuning at BFloat16 precision <Cite kind="paper" ref="p.6 §4.1" />; the code also evaluates in
          bf16, with the labels for the loss and the metrics in bf16 too <Cite kind="spec" ref="§3.7" /> <Cite kind="spec" ref="§10" note="Row 20 of the discrepancy table." />.
          This site keeps float32 everywhere.
        </p>
        <ScaleNote paper="bf16 values in every batch, bf16 model" ours="float32 everywhere">
          The table below applies the same rounding rule (round to nearest, ties to even) only to show its size <Cite kind="spec" ref="§3.7" />. The simulation itself never
          applies it.
        </ScaleNote>
        <Figure wide title={entityNode !== null ? 'The seed row and its entity, as this context window carries them' : 'The seed row, as this context window carries it'}>
          <PrecisionTable data={data} />
        </Figure>
        <CodeRef title="Values become bf16 when the batch is built" file="rustler/src/fly.rs" lines="482-490" code={FLY_BF16_482_490} />
      </Section>

      <Section id="mirror" title="Where this site’s mirror differs from pre.rs">
        <p>
          Every cell value, link, timestamp and statistic in this chapter comes from a TypeScript mirror of <code>pre.rs</code>. It changes four pieces of bookkeeping, none of
          which changes what a token carries:
        </p>
        <ul>
          <li>
            <strong>Time unit.</strong> Datetimes are stored in milliseconds instead of nanoseconds; the z-scores are the same (see the note in the datetime section).
          </li>
          <li>
            <strong>String-table numbering.</strong> <code>pre.rs</code> walks its tables in hash-map order <Cite kind="code" ref="rustler/src/pre.rs:415" />; the mirror walks
            them in file order. So the text indices shown here (column phrases, table names, text values) are arbitrary labels that differ from the official ones. The model only
            compares them or looks up their embeddings, so the numbering carries no meaning <Cite kind="spec" ref="§3.6" />.
          </li>
          <li>
            <strong>Order of a parent’s child links.</strong> For the same reason, the P→F edges inside each parent’s list come in a different order{' '}
            <Cite kind="code" ref="rustler/src/pre.rs:523-529" />. That changes only which random draw of the sampler picks which child (Chapter 6), not which children can be
            picked.
          </li>
          <li>
            <strong>Empty rows.</strong> A row with no non-null foreign key and no non-key column keeps <code>pre.rs</code>’s default header (node index 0){' '}
            <Cite kind="code" ref="rustler/src/pre.rs:408-410" />; the mirror fills in its real index so the figures can label it. Such a row has no cells and no links, so it never
            yields a token.
          </li>
        </ul>
      </Section>

      <div className="page-grid">
      <Quiz
        questions={[
          {
            prompt: (
              <>
                An <code>orders</code> row has the columns <code>order_id</code> (primary key), <code>customer_id</code> and <code>product_id</code> (foreign keys),{' '}
                <code>order_time</code>, <code>quantity</code> and <code>channel</code>, all non-null. How many cells does pre.rs create for it?
              </>
            ),
            choices: [
              { text: 'Six: one per column.', why: 'Key columns never produce cells.' },
              { text: 'Five: everything except the primary key.', why: 'Foreign keys produce links, not cells.' },
              { text: 'Three: order_time, quantity and channel.', correct: true },
              { text: 'Two: quantity and channel; the time column only sets the timestamp.', why: 'The time column is an ordinary datetime cell as well.' },
            ],
            explanation: (
              <>
                The primary key produces nothing, the two foreign keys become entries of <code>f2p_nbr_idxs</code>, and the three feature values become a DateTime, a Number and
                a Text cell. The time column also sets the row’s timestamp, but it stays a cell <Cite kind="spec" ref="§1.4" />.
              </>
            ),
          },
          {
            prompt: (
              <>
                A boolean column is <code>true</code> in 25% of its rows (n is large). What value does pre.rs store for <code>true</code>?
              </>
            ),
            choices: [
              { text: '+1', why: 'That is only the case for a 50/50 column.' },
              { text: '+0.25', why: 'That is μ, the share of trues.' },
              { text: 'About +1.73', correct: true },
              { text: 'About +0.75', why: 'That is 1 − μ before dividing by σ.' },
            ],
            explanation: (
              <>
                μ = 0.25 and σ ≈ √(0.25 · 0.75) ≈ 0.433, so true → (1 − 0.25)/0.433 ≈ +1.73 and false → −0.25/0.433 ≈ −0.58 <Cite kind="spec" ref="§3.3" />.
                {quarter && (
                  <>
                    {' '}
                    {quarter.exact ? (
                      <>
                        Try “25% true” in the boolean figure above: with that column’s n = {quarter.n} it shows {signed(quarter.zTrue, 3)}, slightly less, because the sample
                        standard deviation divides by n − 1.
                      </>
                    ) : (
                      <>
                        Try the “{quarter.pct.toFixed(1)}% true” chip in the boolean figure above: with that column’s n = {quarter.n} it sets k = {quarter.k} trues, so it shows{' '}
                        {signed(quarter.zTrue, 3)}. Two things move it away from +1.732: the share is not exactly 25%, and the sample standard deviation divides by n − 1 (that
                        alone would give {signed(quarter.zExact, 3)}).
                      </>
                    )}
                  </>
                )}
              </>
            ),
          },
          {
            prompt: 'Do datetime columns get their own per-column statistics?',
            choices: [
              { text: 'Yes: each column has its own mean and sample standard deviation, like numbers.', why: 'That is the rule for number columns, not datetimes.' },
              {
                text: 'No: one mean and one population standard deviation are shared by every datetime column of every table, task splits included.',
                correct: true,
              },
              { text: 'Only task-table timestamps share statistics; database columns get their own.', why: 'All datetime columns go into the one pool.' },
            ],
            explanation: (
              <>
                pre.rs accumulates one count, sum and sum of squares over every datetime value in the database and derives a single μ<Sub>T</Sub> and σ<Sub>T</Sub> (population
                std) <Cite kind="code" ref="rustler/src/pre.rs:342-376" />.
              </>
            ),
          },
        ]}
      />
      </div>

      <ChapterNav slug={SLUG} />
    </ChapterPage>
  );
}
