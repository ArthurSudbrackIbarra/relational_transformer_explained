/**
 * Small, pure helpers for the masks chapter. Nothing here adds model semantics: the masks themselves
 * always come from the engine (`buildMasks`, through `useMasks`). The two "recompute" helpers below
 * re-evaluate single expressions of rt/model.py:164-191 so the chapter can show (a) the parent part of
 * the feature mask on its own and (b) what the code would compute WITHOUT the `& pad` (a clearly
 * labelled counterfactual).
 */
import { ATTN_ORDER, type AttnLayer, type MaskSet, type PreprocessedDB, type RawDatabase, type SampledContext, type Token } from '@/engine/types';

export const LAYER_ORDER = ATTN_ORDER; // col → feat → nbr → full (model.py:83)

// ─────────────────────────────────────────────────────────────────────────────
// Names and labels (display only)
// ─────────────────────────────────────────────────────────────────────────────

/** Shortest unique prefix per table name, e.g. ShopDB → c, o, p, r. */
export function tablePrefixes(names: readonly string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const n of names) {
    let len = 1;
    while (len < n.length && names.some((m) => m !== n && m.startsWith(n.slice(0, len)))) len++;
    out[n] = n.slice(0, len);
  }
  return out;
}

/** Full row name: "customers[35]" or "customer-churn:Val[32]". */
export function rowName(t: Token): string {
  if (t.isPadding) return 'padding';
  if (t.tableType && t.tableType !== 'Db') return `${t.tableName}:${t.tableType}[${t.rowPos}]`;
  return `${t.tableName}[${t.rowPos}]`;
}

/** Short row tag: "c35", "o70", "val32", "train258". */
export function rowTag(t: Token, prefixes: Record<string, string>): string {
  if (t.isPadding) return 'pad';
  if (t.tableType && t.tableType !== 'Db') return `${t.tableType.toLowerCase()}${t.rowPos}`;
  return `${prefixes[t.tableName] ?? t.tableName[0]}${t.rowPos}`;
}

/** Short token label for matrix axes: "c35·country". */
export function tokenLabel(t: Token, prefixes: Record<string, string>): string {
  if (t.isPadding) return `pad #${t.pos}`;
  return `${rowTag(t, prefixes)}·${t.colName}`;
}

/** "customer" from "customers" (for preset button labels only). */
export function singular(table: string): string {
  if (table.endsWith('ies')) return `${table.slice(0, -3)}y`;
  if (table.endsWith('s')) return table.slice(0, -1);
  return table;
}

export function article(word: string): string {
  return /^[aeiou]/i.test(word) ? 'an' : 'a';
}

// ─────────────────────────────────────────────────────────────────────────────
// Grouping the context into tables and rows (the "per-table" view)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Why a row has no token for a column (spec §4.9, §4.11):
 *  - 'null': the value is missing, so the preprocessor stored no cell (pre.rs:429, 535-613);
 *  - 'dropped': a leakage column removed for the seed row or a same-timestamp row (fly.rs:457-461);
 *  - 'cut': the hard stop at exactly L tokens fired inside this row (fly.rs:502-509).
 */
export type HoleKind = 'null' | 'dropped' | 'cut';

export interface CtxRow {
  node: number;
  /** Token positions of this row's cells, in emission order. */
  positions: number[];
  /** Position of the cell of each column of the table (−1 = this row has no token for that column). */
  byColumn: number[];
  /** For each −1 in `byColumn`: why there is no token (null where byColumn has a token). */
  holes: (HoleKind | null)[];
  /** Parent node ids (−1 slots removed). */
  parents: number[];
  first: Token;
}

export interface CtxTable {
  /** Table name (task splits share one table, like table_name_idx does). */
  name: string;
  slot: number;
  isTask: boolean;
  columns: string[];
  rows: CtxRow[];
}

export interface CtxGrouping {
  tables: CtxTable[];
  rowByNode: Map<number, CtxRow>;
  /** Number of padding tokens (shown separately). */
  padding: number;
}

/**
 * Group the non-padding tokens by table (first-appearance order) and row. Works on the whole context or
 * on a subset. `holeKind` explains the missing cells (default: every hole is a null value).
 */
export function groupContext(
  tokens: readonly Token[],
  slots: Record<string, number>,
  holeKind: (node: number, colName: string) => HoleKind = () => 'null',
): CtxGrouping {
  const tables: CtxTable[] = [];
  const tableByName = new Map<string, CtxTable>();
  const rowByNode = new Map<number, CtxRow>();
  let padding = 0;
  for (const t of tokens) {
    if (t.isPadding) {
      padding++;
      continue;
    }
    let tbl = tableByName.get(t.tableName);
    if (!tbl) {
      tbl = { name: t.tableName, slot: slots[t.tableName] ?? 0, isTask: !!t.tableType && t.tableType !== 'Db', columns: [], rows: [] };
      tableByName.set(t.tableName, tbl);
      tables.push(tbl);
    }
    if (!tbl.columns.includes(t.colName)) tbl.columns.push(t.colName);
    let row = rowByNode.get(t.nodeIdx);
    if (!row) {
      row = { node: t.nodeIdx, positions: [], byColumn: [], holes: [], parents: t.f2pNbrIdxs.filter((p) => p >= 0), first: t };
      rowByNode.set(t.nodeIdx, row);
      tbl.rows.push(row);
    }
    row.positions.push(t.pos);
  }
  // `tokens` may be a subset of the context (an excerpt), so look tokens up by their position, not by index
  const byPos = new Map(tokens.map((t) => [t.pos, t]));
  for (const tbl of tables) {
    for (const row of tbl.rows) {
      row.byColumn = tbl.columns.map((c) => row.positions.find((p) => byPos.get(p)?.colName === c) ?? -1);
      row.holes = row.byColumn.map((p, i) => (p >= 0 ? null : holeKind(row.node, tbl.columns[i])));
    }
  }
  return { tables, rowByNode, padding };
}

/**
 * Classify the missing cells of the live context from the sampler's own event log: a
 * `drop-leakage-cell` event marks a dropped leakage column (fly.rs:457-461); in the row where the
 * `truncated` event fired, a stored cell that was never emitted was cut by the L stop (fly.rs:502-509).
 * Anything else is a null value, which the preprocessor never stores as a cell.
 */
export function holeClassifier(ctx: SampledContext, pdb: PreprocessedDB): (node: number, colName: string) => HoleKind {
  const dropped = new Set<string>();
  let cutNode = -1;
  for (const e of ctx.events) {
    if (e.kind === 'drop-leakage-cell') dropped.add(`${e.node}|${e.colName}`);
    else if (e.kind === 'truncated') cutNode = e.node;
  }
  const cut = new Set<string>();
  if (cutNode >= 0) {
    const emitted = new Set(ctx.tokens.filter((t) => !t.isPadding && t.nodeIdx === cutNode).map((t) => t.colName));
    for (const c of pdb.nodes[cutNode]?.cells ?? []) {
      if (!emitted.has(c.colName) && !dropped.has(`${cutNode}|${c.colName}`)) cut.add(c.colName);
    }
  }
  return (node, colName) => {
    if (dropped.has(`${node}|${colName}`)) return 'dropped';
    if (node === cutNode && cut.has(colName)) return 'cut';
    return 'null';
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Reading masks
// ─────────────────────────────────────────────────────────────────────────────

/** Row q of a mask as a 0/1 array of length S. */
export function maskRow(mask: Uint8Array, S: number, q: number): Uint8Array {
  return mask.subarray(q * S, (q + 1) * S);
}

export function countRow(mask: Uint8Array, S: number, q: number): number {
  let n = 0;
  const o = q * S;
  for (let k = 0; k < S; k++) n += mask[o + k];
  return n;
}

export interface MaskStats {
  layer: AttnLayer;
  pairs: number;
  /** pairs / S². */
  density: number;
  /** Keys per non-padding query. */
  min: number;
  median: number;
  max: number;
  mean: number;
  /** Non-padding queries with no allowed key. */
  empty: number;
}

export function maskStats(masks: MaskSet, tokens: readonly Token[]): MaskStats[] {
  const S = tokens.length;
  return LAYER_ORDER.map((layer) => {
    const per: number[] = [];
    let pairs = 0;
    for (let q = 0; q < S; q++) {
      const n = countRow(masks[layer], S, q);
      pairs += n;
      if (!tokens[q].isPadding) per.push(n);
    }
    per.sort((a, b) => a - b);
    const m = per.length;
    return {
      layer,
      pairs,
      density: S ? pairs / (S * S) : 0,
      min: m ? per[0] : 0,
      median: m ? (m % 2 ? per[(m - 1) / 2] : (per[m / 2 - 1] + per[m / 2]) / 2) : 0,
      max: m ? per[m - 1] : 0,
      mean: m ? pairs / m : 0,
      empty: per.filter((n) => n === 0).length,
    };
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Single expressions of model.py:164-191, re-evaluated for teaching
// ─────────────────────────────────────────────────────────────────────────────

/** `kv_in_f2p & pad` (model.py:171-173, 165): key's row is one of the query's parents. The "parent part" of feat. */
export function parentPart(tokens: readonly Token[]): Uint8Array {
  const S = tokens.length;
  const out = new Uint8Array(S * S);
  for (let q = 0; q < S; q++) {
    if (tokens[q].isPadding) continue;
    for (let k = 0; k < S; k++) {
      if (tokens[k].isPadding) continue;
      if (tokens[q].f2pNbrIdxs.includes(tokens[k].nodeIdx)) out[q * S + k] = 1;
    }
  }
  return out;
}

/** Transpose of an S×S 0/1 matrix. */
export function transpose(m: Uint8Array, S: number): Uint8Array {
  const out = new Uint8Array(S * S);
  for (let q = 0; q < S; q++) for (let k = 0; k < S; k++) out[k * S + q] = m[q * S + k];
  return out;
}

/** Number of entries where two equal-size 0/1 matrices differ. */
export function diffCount(a: Uint8Array, b: Uint8Array): number {
  let n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++;
  return n;
}

/**
 * COUNTERFACTUAL: the feature and neighbor expressions of model.py:187-188 evaluated WITHOUT `& pad`,
 * on the raw ids (padding: node −1, parents −1×5, fly.rs:65-85). Used only to show why the AND exists.
 */
export function masksWithoutPad(tokens: readonly Token[]): { feat: Uint8Array; nbr: Uint8Array } {
  const S = tokens.length;
  const feat = new Uint8Array(S * S);
  const nbr = new Uint8Array(S * S);
  for (let q = 0; q < S; q++) {
    const tq = tokens[q];
    for (let k = 0; k < S; k++) {
      const tk = tokens[k];
      const sameNode = tq.nodeIdx === tk.nodeIdx;
      const kvInF2p = tq.f2pNbrIdxs.includes(tk.nodeIdx);
      const qInF2p = tk.f2pNbrIdxs.includes(tq.nodeIdx);
      if (sameNode || kvInF2p) feat[q * S + k] = 1;
      if (qInF2p) nbr[q * S + k] = 1;
    }
  }
  return { feat, nbr };
}

/** Pairs with the same col_name_idx but a different table_name_idx (the extra table test, model.py:181-183). */
export function sameColumnOtherTable(tokens: readonly Token[]): number {
  let n = 0;
  for (const a of tokens) {
    if (a.isPadding) continue;
    for (const b of tokens) {
      if (b.isPadding) continue;
      if (a.colNameIdx === b.colNameIdx && a.tableNameIdx !== b.tableNameIdx) n++;
    }
  }
  return n;
}

// ─────────────────────────────────────────────────────────────────────────────
// Presets for "Who can I see?"
// ─────────────────────────────────────────────────────────────────────────────

export interface Preset {
  id: string;
  label: string;
  pos: number;
}

/**
 * The brief's presets, derived from the live context (never hard-coded):
 * the masked target; a cell of the seed's entity (its first parent row); a cell of a DB row that points
 * to that entity (e.g. an order); a cell of another parent of that row (e.g. a product).
 */
export function presets(tokens: readonly Token[], targetPos: number): Preset[] {
  const out: Preset[] = [];
  if (targetPos < 0) return out;
  out.push({ id: 'target', label: 'the masked target', pos: targetPos });
  const seed = tokens[targetPos];
  const entity = seed.f2pNbrIdxs.find((p) => p >= 0);
  const firstOf = (pred: (t: Token) => boolean) => tokens.find((t) => !t.isPadding && pred(t));
  if (entity === undefined) return out;
  const ent = firstOf((t) => t.nodeIdx === entity);
  if (ent) out.push({ id: 'entity', label: `${article(singular(ent.tableName))} ${singular(ent.tableName)} cell`, pos: ent.pos });
  const child = firstOf((t) => t.tableType === 'Db' && t.tableName !== seed.tableName && t.f2pNbrIdxs.includes(entity) && t.nodeIdx !== seed.nodeIdx);
  if (child) {
    out.push({ id: 'child', label: `${article(singular(child.tableName))} ${singular(child.tableName)} cell`, pos: child.pos });
    const other = child.f2pNbrIdxs.find((p) => p >= 0 && p !== entity && tokens.some((t) => !t.isPadding && t.nodeIdx === p));
    if (other !== undefined) {
      const o = firstOf((t) => t.nodeIdx === other)!;
      out.push({ id: 'other-parent', label: `${article(singular(o.tableName))} ${singular(o.tableName)} cell`, pos: o.pos });
    }
  }
  return out;
}

/**
 * A small, real excerpt of the live context for the "what each layer is for" diagrams: the seed row,
 * its parents, up to two other rows of the seed's table that point to the same entity, up to two DB
 * rows that point to the entity, the other parents of those, and one row of the seed's table that
 * points to a different entity. Returned as token positions in context order. Because a mask entry
 * depends only on its two tokens, the live masks restricted to these positions are exactly the masks
 * this excerpt would get on its own.
 */
export function excerptPositions(tokens: readonly Token[], targetPos: number, maxTokens = 30): number[] {
  if (targetPos < 0) return [];
  const seed = tokens[targetPos];
  const entity = seed.f2pNbrIdxs.find((p) => p >= 0);
  const inCtx = new Set(tokens.filter((t) => !t.isPadding).map((t) => t.nodeIdx));
  const nodes: number[] = [seed.nodeIdx];
  const add = (n: number | undefined) => {
    if (n !== undefined && n >= 0 && inCtx.has(n) && !nodes.includes(n)) nodes.push(n);
  };
  for (const p of seed.f2pNbrIdxs) add(p);
  const rowsOf = (pred: (t: Token) => boolean, limit: number) => {
    const seen: number[] = [];
    for (const t of tokens) {
      if (t.isPadding || seen.includes(t.nodeIdx) || nodes.includes(t.nodeIdx)) continue;
      if (pred(t)) seen.push(t.nodeIdx);
      if (seen.length >= limit) break;
    }
    return seen;
  };
  if (entity !== undefined) {
    for (const n of rowsOf((t) => t.tableNameIdx === seed.tableNameIdx && t.f2pNbrIdxs.includes(entity), 2)) add(n);
    const kids = rowsOf((t) => t.tableType === 'Db' && t.tableNameIdx !== seed.tableNameIdx && t.f2pNbrIdxs.includes(entity), 2);
    for (const n of kids) add(n);
    for (const n of kids) {
      const kid = tokens.find((t) => t.nodeIdx === n)!;
      for (const p of kid.f2pNbrIdxs) if (p !== entity) add(p);
    }
    for (const n of rowsOf((t) => t.tableNameIdx === seed.tableNameIdx && !t.f2pNbrIdxs.includes(entity), 1)) add(n);
  }
  const keep = new Set(nodes);
  const out: number[] = [];
  for (const t of tokens) if (!t.isPadding && keep.has(t.nodeIdx)) out.push(t.pos);
  // keep whole rows while under the cap (rows are emitted contiguously, so trimming at a row boundary is easy)
  if (out.length <= maxTokens) return out;
  const order = nodes; // priority order
  const chosen: number[] = [];
  let total = 0;
  for (const n of order) {
    const cnt = out.filter((p) => tokens[p].nodeIdx === n).length;
    if (total + cnt > maxTokens) continue;
    chosen.push(n);
    total += cnt;
  }
  const keep2 = new Set(chosen);
  return out.filter((p) => keep2.has(tokens[p].nodeIdx));
}

/**
 * Excerpt for the column card: the seed row plus up to `maxRows − 1` other rows of the seed's table
 * that have a cell in the target's column (for a forecast task: the task rows, whose labels the target
 * reads), in context order. Whole rows only.
 */
export function columnExcerpt(tokens: readonly Token[], targetPos: number, maxRows = 12): number[] {
  if (targetPos < 0) return [];
  const seed = tokens[targetPos];
  const rows: number[] = [seed.nodeIdx];
  for (const t of tokens) {
    if (rows.length >= maxRows) break;
    if (t.isPadding || rows.includes(t.nodeIdx)) continue;
    if (t.tableNameIdx === seed.tableNameIdx && t.colNameIdx === seed.colNameIdx) rows.push(t.nodeIdx);
  }
  const keep = new Set(rows);
  return tokens.filter((t) => !t.isPadding && keep.has(t.nodeIdx)).map((t) => t.pos);
}

/**
 * Excerpt for the neighbor card: one row of the entity plus up to `perTable` of its child rows from
 * EVERY child table in the context (task rows and database rows alike), so each kind of child appears.
 */
export function neighborExcerpt(tokens: readonly Token[], entityNode: number, perTable = 3): number[] {
  const rows: number[] = [entityNode];
  const perTbl = new Map<number, number>();
  for (const t of tokens) {
    if (t.isPadding || rows.includes(t.nodeIdx) || !t.f2pNbrIdxs.includes(entityNode)) continue;
    const n = perTbl.get(t.tableNameIdx) ?? 0;
    if (n >= perTable) continue;
    perTbl.set(t.tableNameIdx, n + 1);
    rows.push(t.nodeIdx);
  }
  const keep = new Set(rows);
  return tokens.filter((t) => !t.isPadding && keep.has(t.nodeIdx)).map((t) => t.pos);
}

/** Database tables that no foreign key points to (their rows can never have children). */
export function childlessTables(db: RawDatabase): string[] {
  const referenced = new Set(db.tables.flatMap((t) => Object.values(t.schema.fkeys)));
  return db.tables.map((t) => t.schema.name).filter((n) => !referenced.has(n));
}

/** "a, b and c" */
export function listAnd(items: readonly string[], conj = 'and'): string {
  if (items.length <= 1) return items.join('');
  return `${items.slice(0, -1).join(', ')} ${conj} ${items[items.length - 1]}`;
}

/** Restrict an S×S mask to the given positions (n×n, in the given order). */
export function restrict(mask: Uint8Array, S: number, positions: readonly number[]): Uint8Array {
  const n = positions.length;
  const out = new Uint8Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i * n + j] = mask[positions[i] * S + positions[j]];
  return out;
}

/** Format an integer with thousands separators. */
export const int = (n: number) => n.toLocaleString('en-US');

/** Percentage with one decimal. */
export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
