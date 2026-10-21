/**
 * Preprocessing: a line-by-line mirror of the paper-era `rustler/src/pre.rs`
 * (snap-stanford/relational-transformer @ 2a899c8), run in memory on a RawDatabase.
 *
 * Spec sections (research/RT_TECHNICAL_SPEC.md): §1.2 (PK/FK), §1.4 (node timestamps),
 * §1.5 (global node indexing), §3.2 (sem types), §3.3 (normalization), §3.5 (missing values),
 * §3.6 (text dictionary).
 *
 * The three phases of pre.rs are kept as three phases here:
 *   1. "reading tables"       pre.rs:79-307   → table list in glob order + node offsets
 *   2. "computing col stats"  pre.rs:310-396  → per (table, split) per column stats, global datetime stats,
 *                                               Val/Test task tables overwritten with Train stats
 *   3. "making node vector"   pre.rs:399-615  → text dictionary, FK edges, cells
 *
 * Deliberate, documented deviations (none changes what the model or the sampler sees):
 *  - pre.rs iterates its `table_map` (a HashMap with a fixed-key SipHash) in hash order in
 *    phase 3 (pre.rs:415), so the official text-dictionary indices and the order of the edges
 *    inside each `p2f_adj[parent]` list are deterministic but arbitrary. We iterate tables in
 *    node (glob) order instead. Text indices are opaque labels (only equality is ever used:
 *    model.py:165-183, fly.rs:417), and the order of a parent's child edges only changes which
 *    RNG draw selects which child, not the selection distribution (§4.2: bit-exact RNG parity
 *    is not required).
 *  - pre.rs leaves `Node::default()` (node_idx 0, table_name_idx 0) for a row that has neither a
 *    non-null FK nor any non-key column (pre.rs:408-410, 466-470, 537-541). Such a row has no
 *    cells and no parents, so it never produces a token; we fill its header fields anyway so the
 *    UI can label it.
 *  - We store datetimes as epoch MILLISECONDS instead of nanoseconds. The z-score is identical
 *    (affine transform, spec §3.3) up to float64 rounding.
 *  - Inputs that would make pre.rs panic (non-integer / out-of-range FK, missing parent table,
 *    time column that is not a datetime, Val/Test split without Train, …) throw an Error with an
 *    explanation instead. Inputs pre.rs would silently skip (two-dash task directories,
 *    pre.rs:70, 93-95) also throw, so a demo DB can never lose a table without noticing.
 */
import {
  SemType,
  type Cell,
  type ColumnSchema,
  type ColumnStats,
  type DType,
  type Edge,
  type GraphNode,
  type PreprocessedDB,
  type RawDatabase,
  type RawTable,
  type RawValue,
  type Split,
  type TableType,
} from './types';

// ─────────────────────────────────────────────────────────────────────────────
// Small public helpers
// ─────────────────────────────────────────────────────────────────────────────

/** The column-name phrase `"<col> of <table>"` (pre.rs:432). Task splits share it (§2.2). */
export function columnPhrase(colName: string, tableName: string): string {
  return `${colName} of ${tableName}`;
}

/** The table_info.json key `"<table>:<Db|Train|Val|Test>"` (pre.rs:652, `{:?}` of TableType). */
export function tableKey(tableName: string, tableType: TableType): string {
  return `${tableName}:${tableType}`;
}

/**
 * Node timestamp in whole SECONDS with Rust `i32` semantics (§1.4):
 * `(ns / 1_000_000_000) as i32` (pre.rs:489, 510). Integer division truncates toward zero
 * (= Math.trunc), and `as i32` wraps modulo 2^32 (= ToInt32, `| 0`). Null/NaN → null
 * (`.get(r)` returns None for a null time cell, pre.rs:488).
 */
export function toNodeTs(ms: RawValue | undefined): number | null {
  if (ms === null || ms === undefined || typeof ms !== 'number' || Number.isNaN(ms)) return null;
  return Math.trunc(ms / 1000) | 0;
}

/** Global node index of row `rowPos` of table `key` ("<table>:<type>"), spec §1.5. */
export function nodeIdxOf(pdb: PreprocessedDB, key: string, rowPos: number): number {
  const info = pdb.tableInfo[key];
  if (!info) throw new Error(`nodeIdxOf: unknown table key "${key}"`);
  if (!(rowPos >= 0 && rowPos < info.numNodes)) throw new Error(`nodeIdxOf: row ${rowPos} out of range for "${key}"`);
  return info.nodeIdxOffset + rowPos;
}

/** Display label in the style of spec §6.3: `customers[0]#0`, `customer-churn:Val[0]#16`. */
export function nodeLabel(pdb: PreprocessedDB, nodeIdx: number): string {
  const n = pdb.nodes[nodeIdx];
  if (!n) return `#${nodeIdx}`;
  const t = n.tableType === 'Db' ? n.tableName : `${n.tableName}:${n.tableType}`;
  return `${t}[${n.rowPos}]#${nodeIdx}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internals
// ─────────────────────────────────────────────────────────────────────────────

/** pre.rs `ColStat { mean, std }` plus the number of values it was computed from. */
interface ColStat {
  mean: number;
  std: number;
  count: number;
}

/** One parquet file of pre.rs (`Table`, pre.rs:42-51). */
interface TableEntry {
  tableName: string;
  tableType: TableType;
  table: RawTable;
  nodeIdxOffset: number;
  numRows: number;
  /** Aligned with `table.schema.columns` (pre.rs pushes one ColStat per DataFrame column). */
  colStats: ColStat[];
}

const SPLIT_FILE: Record<Split, string> = { Test: 'test', Train: 'train', Val: 'val' };

const utf8 = new TextEncoder();
/** Byte-wise comparison (Rust OsStr / Path component order on Unix). */
function cmpBytes(a: string, b: string): number {
  const x = utf8.encode(a);
  const y = utf8.encode(b);
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i++) if (x[i] !== y[i]) return x[i] - y[i];
  return x.length - y.length;
}

function countDashes(s: string): number {
  let c = 0;
  for (const ch of s) if (ch === '-') c++;
  return c;
}

function get(row: Record<string, RawValue>, col: string): RawValue {
  const v = row[col];
  return v === undefined ? null : v;
}

/** Validate one raw value against its declared dtype (pre.rs would panic on a mismatch, pre.rs:606-611). */
function checkValue(v: RawValue, dtype: DType, where: string): void {
  if (v === null) return;
  switch (dtype) {
    case 'bool':
      if (typeof v !== 'boolean') throw new Error(`${where}: expected boolean, got ${JSON.stringify(v)}`);
      return;
    case 'str':
      if (typeof v !== 'string') throw new Error(`${where}: expected string, got ${JSON.stringify(v)}`);
      return;
    case 'int':
    case 'float':
    case 'datetime':
      if (typeof v !== 'number') throw new Error(`${where}: expected number (${dtype}), got ${JSON.stringify(v)}`);
      // An integer column cannot hold fractions (NaN is tolerated and treated as missing, like a float NaN).
      if (dtype === 'int' && !Number.isNaN(v) && !Number.isInteger(v)) throw new Error(`${where}: expected integer, got ${v}`);
      return;
  }
}

/** Arithmetic mean, or null for an empty array (polars `mean()` → None). */
function mean(xs: number[]): number | null {
  if (xs.length === 0) return null;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

/** Sample standard deviation (ddof = 1), or null when undefined (n ≤ 1; polars `std(1)` → None). */
function sampleStd(xs: number[]): number | null {
  if (xs.length <= 1) return null;
  const m = mean(xs)!;
  let ss = 0;
  for (const x of xs) ss += (x - m) * (x - m);
  return Math.sqrt(ss / (xs.length - 1));
}

// ─────────────────────────────────────────────────────────────────────────────
// preprocess
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Preprocess a raw database exactly like pre.rs does for one `db_name`, including ALL task tables
 * (every split) of the DB's forecast tasks. Autocomplete tasks add no table (§2.4).
 */
export function preprocess(db: RawDatabase): PreprocessedDB {
  // ── Phase 1: tables in glob order + node offsets (pre.rs:79-307, spec §1.5) ────────────────
  // glob("db/*.parquet").sorted() then glob("tasks/*/*.parquet").sorted(). PathBuf ordering is
  // component-wise and byte-wise, so DB files compare as "<name>.parquet", task files first by
  // task directory name, then by file name "test.parquet" < "train.parquet" < "val.parquet".
  const dbTables = [...db.tables].sort((a, b) => cmpBytes(`${a.schema.name}.parquet`, `${b.schema.name}.parquet`));
  const seenDb = new Set<string>();
  for (const t of dbTables) {
    if (seenDb.has(t.schema.name)) throw new Error(`preprocess: duplicate DB table "${t.schema.name}"`);
    seenDb.add(t.schema.name);
  }

  const taskTables: { name: string; split: Split; table: RawTable }[] = [];
  const seenTask = new Set<string>();
  for (const task of db.tasks) {
    if (task.kind !== 'forecast' || !task.tables) continue; // autocomplete: no task table (§2.4)
    if (seenTask.has(task.name)) throw new Error(`preprocess: two forecast tasks share the task table name "${task.name}"`);
    if (seenDb.has(task.name)) throw new Error(`preprocess: task table "${task.name}" has the same name as a DB table`);
    seenTask.add(task.name);
    for (const split of ['Test', 'Train', 'Val'] as Split[]) {
      const t = task.tables[split];
      if (t) taskTables.push({ name: task.name, split, table: t });
    }
  }
  taskTables.sort((a, b) => cmpBytes(a.name, b.name) || cmpBytes(SPLIT_FILE[a.split], SPLIT_FILE[b.split]));

  // pre.rs:93-95 silently skips any file whose path has exactly two more dashes than the dataset
  // path (link-prediction tasks such as "user-item-purchase"). We refuse such names instead.
  for (const name of [...dbTables.map((t) => t.schema.name), ...taskTables.map((t) => t.name)]) {
    if (countDashes(name) === 2) {
      throw new Error(`preprocess: table/task name "${name}" has exactly two dashes; pre.rs:93-95 would silently skip it`);
    }
  }

  const entries: TableEntry[] = [];
  const byKey = new Map<string, TableEntry>();
  let numRowsSum = 0;
  const addEntry = (tableName: string, tableType: TableType, table: RawTable) => {
    const e: TableEntry = { tableName, tableType, table, nodeIdxOffset: numRowsSum, numRows: table.rows.length, colStats: [] };
    entries.push(e);
    byKey.set(tableKey(tableName, tableType), e);
    numRowsSum += table.rows.length; // node_idx_offset: num_rows_sum (pre.rs:302, 305)
  };
  for (const t of dbTables) addEntry(t.schema.name, 'Db', t);
  for (const t of taskTables) addEntry(t.name, t.split, t.table);

  for (const e of entries) validateTable(e);

  // ── Phase 2: column statistics (pre.rs:310-396, spec §3.3) ───────────────────────────────
  let dtCnt = 0;
  let dtSum = 0;
  let dtSumSq = 0;
  for (const e of entries) {
    // pre.rs:317 iterates EVERY DataFrame column, PK/FK included (their stats are never used).
    for (const col of e.table.schema.columns) {
      const raw = e.table.rows.map((r) => get(r, col.name)).filter((v) => v !== null);
      switch (col.dtype) {
        case 'bool': {
          // pre.rs:320-329: cast to f64, drop nulls; mean.unwrap_or(0), std(1).unwrap_or(0). NO zero guard.
          const xs = raw.map((v) => (v ? 1 : 0));
          e.colStats.push({ mean: mean(xs) ?? 0, std: sampleStd(xs) ?? 0, count: xs.length });
          break;
        }
        case 'int':
        case 'float': {
          // pre.rs:330-341: cast to f64, drop nulls, drop NaN; mean.unwrap_or(0), std(1).unwrap_or(1); σ==0 → 1.
          const xs = (raw as number[]).filter((v) => !Number.isNaN(v));
          let std = sampleStd(xs) ?? 1;
          if (std === 0) std = 1;
          e.colStats.push({ mean: mean(xs) ?? 0, std, count: xs.length });
          break;
        }
        case 'datetime': {
          // pre.rs:342-366: accumulate into ONE global (count, Σt, Σt²) over every datetime column of
          // every table file, task splits included; the per-column ColStat is a dummy (0, 0).
          for (const v of raw as number[]) {
            if (Number.isNaN(v)) continue;
            dtCnt += 1;
            dtSum += v;
            dtSumSq += v * v;
          }
          e.colStats.push({ mean: 0, std: 0, count: 0 });
          break;
        }
        default:
          e.colStats.push({ mean: 0, std: 0, count: 0 }); // pre.rs:367-370 (strings)
      }
    }
  }
  // pre.rs:375-376: population std (ddof = 0), no zero guard.
  const dtMean = dtSum / dtCnt;
  const dtStd = Math.sqrt(dtSumSq / dtCnt - dtMean * dtMean);

  // pre.rs:383-396: Val/Test task tables overwrite their stats with the same task's Train stats.
  // pre.rs copies the ColStat vector by column POSITION; we require identical column lists so that
  // position and name agree.
  for (const e of entries) {
    if (e.tableType !== 'Val' && e.tableType !== 'Test') continue;
    const train = byKey.get(tableKey(e.tableName, 'Train'));
    if (!train) {
      throw new Error(`preprocess: task "${e.tableName}" has a ${e.tableType} split but no Train split (pre.rs:392 unwrap panics)`);
    }
    const a = e.table.schema.columns;
    const b = train.table.schema.columns;
    if (a.length !== b.length || a.some((c, i) => c.name !== b[i].name || c.dtype !== b[i].dtype)) {
      throw new Error(`preprocess: task "${e.tableName}" ${e.tableType} columns differ from Train columns`);
    }
    e.colStats = train.colStats.map((s) => ({ ...s }));
  }

  // ── Phase 3: text dictionary, edges, cells (pre.rs:399-615) ──────────────────────────────
  const text: string[] = [];
  const textToIdx = new Map<string, number>();
  const columnIndex: Record<string, number> = {};
  // `*text_to_idx.entry(s).or_insert_with(|| len)` (pre.rs:424-427, 433-437, 495-498, 596-597).
  // One string table per DB, identical strings share one index (§3.6).
  const intern = (s: string): number => {
    let i = textToIdx.get(s);
    if (i === undefined) {
      i = text.length;
      textToIdx.set(s, i);
      text.push(s);
    }
    return i;
  };

  const nodes: GraphNode[] = new Array(numRowsSum);
  const p2fAdj: Edge[][] = Array.from({ length: numRowsSum }, () => []);

  for (const e of entries) {
    const { schema, rows } = e.table;
    const tableNameIdx = intern(e.tableName); // pre.rs:424-427
    for (let r = 0; r < rows.length; r++) {
      const nodeIdx = e.nodeIdxOffset + r;
      nodes[nodeIdx] = {
        nodeIdx,
        tableKey: tableKey(e.tableName, e.tableType),
        tableName: e.tableName,
        tableNameIdx,
        tableType: e.tableType,
        rowPos: r,
        isTaskNode: e.tableType !== 'Db', // pre.rs:468, 539
        ts: null, // set only inside the FK loop below (pre.rs:481-493, spec §1.4)
        f2pNbrIdxs: [],
        f2pEdges: [],
        cells: [],
      };
    }

    schema.columns.forEach((col, j) => {
      const colStat = e.colStats[j];
      const phrase = columnPhrase(col.name, e.tableName); // pre.rs:432
      // pre.rs:433-437: the phrase enters column_index.json only when THIS lookup inserts it
      // (an identical string interned earlier, e.g. a cell value, would keep it out).
      const isNew = !textToIdx.has(phrase);
      const colNameIdx = intern(phrase);
      if (isNew) columnIndex[phrase] = colNameIdx;

      // PK column → nothing at all (pre.rs:439-442).
      if (col.name === schema.pkey) return;

      // FK column → f2p_nbr_idxs, F→P edge, P→F edge; no cell (pre.rs:444-533).
      if (Object.prototype.hasOwnProperty.call(schema.fkeys, col.name)) {
        const ptableName = schema.fkeys[col.name];
        const parent = byKey.get(tableKey(ptableName, 'Db')); // parents are always "<table>:Db" (pre.rs:452-459)
        if (!parent) throw new Error(`preprocess: FK ${e.tableName}.${col.name} → "${ptableName}", which is not a DB table`);
        for (let r = 0; r < rows.length; r++) {
          const val = get(rows[r], col.name);
          if (val === null) continue; // null FKs are skipped (pre.rs:462-464)
          if (typeof val !== 'number' || !Number.isInteger(val)) {
            throw new Error(`preprocess: FK ${e.tableName}.${col.name} row ${r} is not an integer (pre.rs:472-476 panics)`);
          }
          if (val < 0 || val >= parent.numRows) {
            throw new Error(`preprocess: FK ${e.tableName}.${col.name} row ${r} = ${val} is outside 0..${parent.numRows - 1} of "${ptableName}"`);
          }
          const nodeIdx = e.nodeIdxOffset + r;
          const node = nodes[nodeIdx];
          const pnodeIdx = parent.nodeIdxOffset + val; // pre.rs:477 (FK value = parent row position)
          node.f2pNbrIdxs.push(pnodeIdx); // pre.rs:479

          // pre.rs:481-493: the row's timestamp, only here (≥1 non-null FK ∧ time column ∧ non-null value).
          const ts = schema.timeCol ? toNodeTs(get(rows[r], schema.timeCol)) : null;
          node.ts = ts;

          const ptableNameIdx = intern(ptableName); // pre.rs:495-498 (may intern the parent name here)
          // pre.rs:500-513: the F→P edge carries the PARENT's raw time value (never read by fly.rs).
          const pSchema = parent.table.schema;
          const pts = pSchema.timeCol ? toNodeTs(get(parent.table.rows[val], pSchema.timeCol)) : null;
          node.f2pEdges.push({ node: pnodeIdx, tableNameIdx: ptableNameIdx, tableType: 'Db', ts: pts }); // pre.rs:515-521
          // pre.rs:523-529: the P→F edge carries the CHILD's idx, table, table type and timestamp.
          p2fAdj[pnodeIdx].push({ node: nodeIdx, tableNameIdx, tableType: e.tableType, ts });
        }
        return;
      }

      // Every other column → one cell per non-null, non-NaN value (pre.rs:535-613, spec §3.2).
      for (let r = 0; r < rows.length; r++) {
        const val = get(rows[r], col.name);
        if (val === null) continue; // AnyValue::Null → no cell (pre.rs:552)
        const node = nodes[e.nodeIdxOffset + r];
        const cell: Cell = {
          colName: col.name,
          colNameIdx,
          sem: SemType.Number,
          number: 0,
          textIdx: 0, // non-text cells store text index 0 (pre.rs:559, 578, 589)
          datetime: 0,
          boolean: 0,
          raw: val,
        };
        switch (col.dtype) {
          case 'bool': {
            // pre.rs:553-564: ((b?1:0) − μ)/σ, stored as f32; no NaN/inf check.
            cell.sem = SemType.Boolean;
            cell.boolean = Math.fround(((val ? 1 : 0) - colStat.mean) / colStat.std);
            break;
          }
          case 'int':
          case 'float': {
            const v = val as number;
            if (Number.isNaN(v)) continue; // NaN → no cell (pre.rs:566-568)
            const z = (v - colStat.mean) / colStat.std; // pre.rs:569
            if (!Number.isFinite(z)) {
              throw new Error(`preprocess: infinite z-score in ${e.tableName}.${col.name} row ${r} (pre.rs:570-575 panics)`);
            }
            cell.sem = SemType.Number;
            cell.number = Math.fround(z);
            break;
          }
          case 'datetime': {
            const v = val as number;
            if (Number.isNaN(v)) continue; // NaN datetime treated as missing (cannot occur in polars)
            cell.sem = SemType.DateTime;
            cell.datetime = Math.fround((v - dtMean) / dtStd); // pre.rs:586 (global stats)
            break;
          }
          case 'str': {
            cell.sem = SemType.Text;
            cell.textIdx = intern(val as string); // pre.rs:596-597 (raw value only, no column context)
            break;
          }
        }
        node.cells.push(cell);
      }
    });
  }

  // ── Outputs: table_info.json, used column stats, datetime stats ───────────────────────────
  const tableInfo: PreprocessedDB['tableInfo'] = {};
  const colStats: PreprocessedDB['colStats'] = {};
  for (const e of entries) {
    const key = tableKey(e.tableName, e.tableType);
    tableInfo[key] = { nodeIdxOffset: e.nodeIdxOffset, numNodes: e.numRows }; // pre.rs:648-664
    const stats: Record<string, ColumnStats> = {};
    e.table.schema.columns.forEach((col, j) => {
      if (!isFeatureColumn(e.table.schema, col)) return;
      if (col.dtype === 'int' || col.dtype === 'float' || col.dtype === 'bool') {
        const s = e.colStats[j];
        stats[col.name] = { mean: s.mean, std: s.std, count: s.count, kind: col.dtype === 'bool' ? 'boolean' : 'number' };
      }
    });
    colStats[key] = stats;
  }

  return {
    dbName: db.name,
    nodes,
    p2fAdj,
    text,
    textToIdx,
    columnIndex,
    tableInfo,
    colStats,
    datetimeStats: { mean: dtMean, std: dtStd, count: dtCnt },
  };
}

/** True for a column that can produce cells (not the PK, not an FK; spec §1.2). */
export function isFeatureColumn(schema: RawTable['schema'], col: ColumnSchema): boolean {
  return col.name !== schema.pkey && !Object.prototype.hasOwnProperty.call(schema.fkeys, col.name);
}

/** Structural checks; each one corresponds to a pre.rs panic or a documented RelBench convention. */
function validateTable(e: TableEntry): void {
  const { schema, rows } = e.table;
  const where = tableKey(e.tableName, e.tableType);
  const names = new Set<string>();
  for (const c of schema.columns) {
    if (names.has(c.name)) throw new Error(`preprocess: ${where} has duplicate column "${c.name}"`);
    names.add(c.name);
  }
  if (schema.pkey !== null && !names.has(schema.pkey)) throw new Error(`preprocess: ${where} pkey "${schema.pkey}" is not a column`);
  for (const fk of Object.keys(schema.fkeys)) {
    const c = schema.columns.find((x) => x.name === fk);
    if (!c) throw new Error(`preprocess: ${where} FK "${fk}" is not a column`);
    // pre.rs:472-476 accepts only Int64 FK values.
    if (c.dtype !== 'int' && fk !== schema.pkey) throw new Error(`preprocess: ${where} FK "${fk}" must have dtype int (pre.rs:472-476)`);
  }
  if (schema.timeCol !== null) {
    const tc = schema.columns.find((c) => c.name === schema.timeCol);
    // pre.rs:486 `.datetime().unwrap()` panics on a non-datetime time column.
    if (!tc || tc.dtype !== 'datetime') throw new Error(`preprocess: ${where} timeCol "${schema.timeCol}" must be a datetime column`);
  }
  for (let r = 0; r < rows.length; r++) {
    for (const c of schema.columns) checkValue(get(rows[r], c.name), c.dtype, `${where} row ${r} column ${c.name}`);
    // RelBench convention relied upon by pre.rs:452-477 (FK value = parent row position): PK = row position (§1.2).
    if (schema.pkey !== null) {
      const pk = get(rows[r], schema.pkey);
      if (pk !== r) throw new Error(`preprocess: ${where} row ${r} has pkey ${JSON.stringify(pk)}; PK must equal the row position (§1.2)`);
    }
  }
}
