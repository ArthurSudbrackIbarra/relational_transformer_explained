/**
 * Shared contracts for the Relational Transformer (RT) simulation engine.
 *
 * Ground truth for every semantic below is the paper-era official code
 * (snap-stanford/relational-transformer @ 2a899c8) as documented in
 * research/RT_TECHNICAL_SPEC.md. Section numbers (§) refer to that spec.
 *
 * Nothing in this file may change without updating every module that uses it.
 */

// ─────────────────────────────────────────────────────────────────────────────
// 1. Raw relational database (what a RelBench parquet file would hold)  §1, §3.1
// ─────────────────────────────────────────────────────────────────────────────

/** Physical dtype of a column. Determines the RT sem_type (§3.2). */
export type DType = 'int' | 'float' | 'bool' | 'str' | 'datetime';

/**
 * A raw cell value. `null` = missing (produces NO token, §3.5).
 * Datetimes are stored as epoch MILLISECONDS (number). Ints/floats as number
 * (NaN for float = missing as well). Bools as boolean. Strings as string.
 */
export type RawValue = number | boolean | string | null;

export interface ColumnSchema {
  name: string;
  dtype: DType;
  /** Human description shown in the UI (not used by the model). */
  description?: string;
}

export interface TableSchema {
  name: string;
  /** Primary-key column (values MUST equal row position 0..n-1, §1.2), or null. */
  pkey: string | null;
  /** FK column -> parent table name. At most 5 FKs per table (§1.2). */
  fkeys: Record<string, string>;
  /** Time column (a datetime column), or null (§1.4). */
  timeCol: string | null;
  /** ALL columns in "DataFrame order", including pk/fk/time columns. */
  columns: ColumnSchema[];
  description?: string;
}

export interface RawTable {
  schema: TableSchema;
  /** Row-major records keyed by column name. Row i has pkey value i. */
  rows: Record<string, RawValue>[];
}

export type Split = 'Train' | 'Val' | 'Test';
/** Table type in the preprocessed graph: DB table or one split of a task table (§1.5). */
export type TableType = 'Db' | Split;

/** A RelBench-style predictive task attached to a database (§2). */
export interface TaskDef {
  /** e.g. "customer-churn" (forecast) — also the task TABLE name. For autocomplete: the DB table name. */
  name: string;
  /** Unique id across the app, e.g. "shop/customer-churn". */
  id: string;
  dbName: string;
  kind: 'forecast' | 'autocomplete';
  /** binary => Boolean target (BCE); regression => Number target (Huber) (§2.6). */
  targetType: 'binary' | 'regression';
  /** Target column name (in the task table for forecast; in `table` for autocomplete). */
  targetCol: string;
  /** Leakage columns removed on the fly for autocomplete (§4.9). [] for forecast. */
  columnsToDrop: string[];
  /** Plain-language description shown in the UI. */
  description: string;
  /** Forecast only: entity table + FK column in the task table, its time column, window. */
  entityTable?: string;
  entityCol?: string;
  timeCol?: string;
  timedeltaDays?: number;
  valTimestamp?: number; // epoch ms
  testTimestamp?: number; // epoch ms
  /** Forecast only: the three task-table splits (schema has pkey=null, one FK, timeCol). */
  tables?: Record<Split, RawTable>;
  /** Autocomplete only: the DB table whose `targetCol` is predicted. */
  table?: string;
}

export interface RawDatabase {
  name: string; // short id, e.g. "shop"
  title: string; // display, e.g. "ShopDB (e-commerce)"
  description: string;
  tables: RawTable[];
  tasks: TaskDef[];
  /** Global time range of the DB (epoch ms), for UI timelines. */
  minTimestamp: number;
  maxTimestamp: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Preprocessed graph (mirror of pre.rs output)  §1.5, §3
// ─────────────────────────────────────────────────────────────────────────────

/** SemType codes EXACTLY as common.rs:38-44. */
export enum SemType {
  Number = 0,
  Text = 1,
  DateTime = 2,
  Boolean = 3,
}
export const SEM_TYPE_NAMES = ['number', 'text', 'datetime', 'boolean'] as const;
export type SemTypeName = (typeof SEM_TYPE_NAMES)[number];

/** An edge record (common.rs Edge). For p2fAdj, node/ts/tableType are the CHILD's. */
export interface Edge {
  node: number;
  tableNameIdx: number;
  tableType: TableType;
  /** Integer SECONDS (i32 semantics) or null (§1.4). */
  ts: number | null;
}

/** One non-null, non-NaN feature cell (pre.rs:543-611). */
export interface Cell {
  colName: string; // e.g. "age"
  colNameIdx: number; // text index of "<col> of <table>"
  sem: SemType;
  /** z-scored number (sem Number) else 0. */
  number: number;
  /** text index of the string value (sem Text) else 0 (pre.rs stores 0). */
  textIdx: number;
  /** globally z-scored datetime (sem DateTime) else 0. */
  datetime: number;
  /** z-scored 0/1 (sem Boolean) else 0. */
  boolean: number;
  /** Original raw value for display (NOT used by the model). */
  raw: RawValue;
}

export interface GraphNode {
  nodeIdx: number;
  /** "<table>:<Db|Train|Val|Test>" (table_info.json key). */
  tableKey: string;
  tableName: string;
  tableNameIdx: number;
  tableType: TableType;
  /** Row position within its table file. */
  rowPos: number;
  isTaskNode: boolean;
  /** Integer seconds; set ONLY if ≥1 non-null FK, table has timeCol, value non-null (§1.4). */
  ts: number | null;
  /** Parent node idxs in FK-column order (null FKs skipped). ≤5. */
  f2pNbrIdxs: number[];
  f2pEdges: Edge[];
  /** Cells in stored column order (PK, FK, null, NaN removed). */
  cells: Cell[];
}

export interface ColumnStats {
  mean: number;
  std: number; // the σ actually used (after guards)
  count: number;
  kind: 'number' | 'boolean';
}

export interface PreprocessedDB {
  dbName: string;
  nodes: GraphNode[];
  /** p2fAdj[parent] = edges to children (child's idx, table, type, ts). */
  p2fAdj: Edge[][];
  /** One string table per DB: table names, "<col> of <table>" phrases, string values (§3.6). */
  text: string[];
  textToIdx: Map<string, number>;
  /** "<col> of <table>" -> text index (column_index.json). */
  columnIndex: Record<string, number>;
  /** "<table>:<type>" -> offset & count (table_info.json). */
  tableInfo: Record<string, { nodeIdxOffset: number; numNodes: number }>;
  /** Per "<table>:<type>" per column stats actually used for z-scoring (§3.3). */
  colStats: Record<string, Record<string, ColumnStats>>;
  /** Global datetime stats in MILLISECONDS (population std) (§3.3). */
  datetimeStats: { mean: number; std: number; count: number };
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Sampled context + tokens (mirror of fly.rs)  §4, §5
// ─────────────────────────────────────────────────────────────────────────────

export const MAX_F2P_NBRS = 5;

/** One position of the model input (one row of the batch dict), §5.1. */
export interface Token {
  pos: number;
  isPadding: boolean;
  /** −1 for padding. */
  nodeIdx: number;
  /** Always length 5, −1-padded. */
  f2pNbrIdxs: number[];
  tableNameIdx: number; // 0 for padding
  colNameIdx: number; // 0 for padding
  sem: SemType; // 0 for padding
  numberValue: number;
  textIdx: number; // index into PreprocessedDB.text (0 for non-text, like pre.rs)
  datetimeValue: number;
  booleanValue: number;
  /** = isTarget (exactly one per context) (§4.10). */
  mask: boolean;
  isTarget: boolean;
  isTaskNode: boolean;
  classValueIdx: number; // text idx for text cells else −1 (unused by model)
  // ── display-only metadata (never read by the model) ──
  tableName: string;
  tableType: TableType | null;
  colName: string;
  rowPos: number;
  raw: RawValue;
  /** BFS depth label of the row this token came from (−1 for padding). */
  depth: number;
}

/** Events recorded by the sampler, in order, to drive the BFS animation (§4.3). */
export type SamplerEvent =
  | { kind: 'pop'; from: 'stack' | 'bucket'; node: number; depth: number; bucketSize?: number; draw?: number }
  | { kind: 'skip-visited'; node: number; depth: number }
  | { kind: 'visit'; node: number; depth: number }
  | { kind: 'push-parent'; from: number; node: number; depth: number }
  | { kind: 'child-other-task-table'; from: number; node: number }
  | { kind: 'child-future'; from: number; node: number; childTs: number; seedTs: number }
  | { kind: 'push-task-child'; from: number; node: number; depth: number }
  | { kind: 'db-children-subsampled'; from: number; candidates: number[]; kept: number[]; w: number }
  | { kind: 'push-db-child'; from: number; node: number; depth: number }
  | { kind: 'drop-leakage-cell'; node: number; colName: string }
  | { kind: 'emit'; node: number; pos: number; colName: string; isTarget: boolean }
  | { kind: 'truncated'; node: number; pos: number }
  | { kind: 'frontier-empty'; tokens: number }
  | { kind: 'pad'; from: number; to: number };

export interface SampleParams {
  seedNode: number;
  /** Context length L (paper 1024). */
  L: number;
  /** BFS width bound w = max_bfs_width (paper scripts 256). */
  w: number;
  /** col_name_idx of "<target> of <table>". */
  targetColNameIdx: number;
  /** col_name_idxs of leakage columns. */
  dropColNameIdxs: number[];
  /** PRNG seed (fly.rs uses epoch + seed_node_idx + seed). */
  rngSeed: number;
}

export interface SampledContext {
  params: SampleParams;
  /** Exactly L tokens (padding included). */
  tokens: Token[];
  /** Number of non-padding tokens. */
  numReal: number;
  /** Rows in visit order with the depth label they were expanded with. */
  visited: { node: number; depth: number; order: number }[];
  events: SamplerEvent[];
  /** Position of the single masked target token. */
  targetPos: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Model  §5.2, §6, §7, §8
// ─────────────────────────────────────────────────────────────────────────────

export type AttnLayer = 'col' | 'feat' | 'nbr' | 'full';
/** Order inside a block (model.py:83, since commit 7f891d1). */
export const ATTN_ORDER: AttnLayer[] = ['col', 'feat', 'nbr', 'full'];

export interface RTConfig {
  dModel: number; // paper 256
  nHeads: number; // paper 8
  dFF: number; // paper 1024 (= 4·d)
  nBlocks: number; // paper 12
  dText: number; // 384 (all-MiniLM-L12-v2)
  /** RMSNorm epsilon. Paper env (torch 2.6, bf16) = 2^-7 (§7.2). */
  eps: number;
}

/** Row-major matrix: data[r * cols + c]; PyTorch nn.Linear weight is [out, in]. */
export interface Mat {
  rows: number;
  cols: number;
  data: Float32Array;
}

/**
 * Weights keyed EXACTLY by PyTorch state_dict names of rt/model.py, e.g.
 *  enc_dict.number.weight [d,1], enc_dict.number.bias [d], enc_dict.col_name.weight [d,384]
 *  norm_dict.col_name.weight [d], mask_embs.number [d]
 *  blocks.0.norms.col.weight [d], blocks.0.attns.col.wq.weight [d,d] (wk, wv, wo)
 *  blocks.0.ffn.w1.weight [dff,d], blocks.0.ffn.w2.weight [d,dff], blocks.0.ffn.w3.weight [dff,d]
 *  norm_out.weight [d], dec_dict.boolean.weight [1,d], dec_dict.boolean.bias [1], dec_dict.text.weight [384,d]
 */
export interface RTWeights {
  config: RTConfig;
  tensors: Record<string, { shape: number[]; data: Float32Array }>;
  /** Free-form provenance (checkpoint name, training step, etc.). */
  meta: Record<string, unknown>;
}

/** Dense masks for one context: data[q * S + k] = 1 if q may attend to k (§6.2). */
export type MaskSet = Record<AttnLayer, Uint8Array>;

export interface ForwardOptions {
  /** Skip these attention sublayers in every block (inference-time ablation, clearly labelled in UI). */
  disabledLayers?: AttnLayer[];
  /** Record full per-layer internals (Q/K/V/scores/probs). Default true. */
  capture?: boolean;
}

/** Per-token decomposition of the input embedding (§5.2). */
export interface EmbeddingParts {
  /** RMSNorm_colname(W·c + b), [S*d]. */
  colTerm: Float32Array;
  /** Pre-norm projections W_t·v + b (value) [S*d]; zeros when masked/padding. */
  valuePreNorm: Float32Array;
  /** RMSNorm_t(W_t·v + b) or m_t when masked, [S*d]. */
  valueTerm: Float32Array;
  /** colPreNorm = W_col·c + b, [S*d]. */
  colPreNorm: Float32Array;
  /** x0 = colTerm + valueTerm (0 for padding), [S*d]. */
  x0: Float32Array;
}

export interface AttnCapture {
  layer: AttnLayer;
  /** residual stream entering the sublayer [S*d] */
  xIn: Float32Array;
  /** RMSNorm_l(xIn) [S*d] */
  h: Float32Array;
  q: Float32Array; // [S*d] (heads are contiguous slices of d_head)
  k: Float32Array;
  v: Float32Array;
  /** raw scaled scores q·k/√d_head BEFORE masking, [H*S*S] */
  scores: Float32Array;
  /** softmax probabilities after masking (0 where masked; all-0 rows for fully masked), [H*S*S] */
  probs: Float32Array;
  /** concat of head outputs before W_o [S*d] */
  headOut: Float32Array;
  /** W_o · headOut [S*d] — the residual update */
  out: Float32Array;
  /** xIn + out [S*d] */
  xOut: Float32Array;
  skipped: boolean;
}

export interface FFNCapture {
  xIn: Float32Array;
  h: Float32Array; // RMSNorm_ffn(xIn)
  a: Float32Array; // W1 h  [S*dff]
  b: Float32Array; // W3 h  [S*dff]
  gated: Float32Array; // silu(a) * b [S*dff]
  out: Float32Array; // W2 gated [S*d]
  xOut: Float32Array;
}

export interface BlockCapture {
  attn: Record<AttnLayer, AttnCapture>;
  ffn: FFNCapture;
}

export interface ForwardResult {
  S: number;
  d: number;
  masks: MaskSet;
  embedding: EmbeddingParts;
  blocks: BlockCapture[];
  /** RMSNorm_out(x_last) [S*d] */
  xFinal: Float32Array;
  /** Decoder outputs at EVERY position (§8.1). text omitted (never used). */
  yhat: { number: Float32Array; datetime: Float32Array; boolean: Float32Array };
  /** Loss over masked cells (§8.2) and its parts. */
  loss: number;
  targetPos: number;
  /** Prediction at the target: logit (binary) or z-score (regression). */
  prediction: number;
}
