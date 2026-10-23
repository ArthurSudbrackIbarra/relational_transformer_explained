/**
 * Pure helpers for the "embedding" chapter. Every number they produce comes from the live engine's
 * weights (PyTorch state_dict names of PE/rt/model.py) and the precomputed MiniLM embeddings; the
 * kernels are the engine's own (`linearVec`, `rmsnormRow`, `scalarValueTerm`, `embedTokens`), so
 * nothing here re-implements model behavior. The extra quantities (mean square, scale factor,
 * per-dimension decompositions, a 2-D PCA of a curve) are read-outs for display only.
 */
import { scalarValueTerm, tensor, linearVec, rmsnormRow, SEM_KEYS, type SemKey } from '@/engine/model';
import type { EmbeddingStore } from '@/engine/embeddings';
import { SemType, type PreprocessedDB, type RawDatabase, type RTWeights, type Token, type ColumnStats } from '@/engine/types';
import { semOfDType } from '@/components';

// ─────────────────────────────────────────────────────────────────────────────
// Small vector helpers (display read-outs)
// ─────────────────────────────────────────────────────────────────────────────

export function norm(v: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const na = norm(a);
  const nb = norm(b);
  return na > 0 && nb > 0 ? dot(a, b) / (na * nb) : 0;
}

export function sub(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

export function maxAbsDelta(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

export function slice(a: Float32Array, row: number, d: number): Float32Array {
  return a.subarray(row * d, (row + 1) * d);
}

/** mean(x²) of a vector: the quantity under RMSNorm's square root (spec §7.2). */
export function meanSquare(v: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return v.length ? s / v.length : 0;
}

/** The RMSNorm read-outs of an input vector: mean square, the scale 1/√(ms+ε), and the share of ε. */
export function rmsInfo(v: ArrayLike<number>, eps: number) {
  const ms = meanSquare(v);
  const scale = 1 / Math.sqrt(ms + eps);
  return { ms, rms: Math.sqrt(ms), scale, epsShare: eps / (ms + eps) };
}

/** ‖RMSNorm(x)‖ with γ = 1, as a function of ‖x‖ (d dims, epsilon eps). */
export function normalizedLength(r: number, d: number, eps: number): number {
  return r / Math.sqrt((r * r) / d + eps);
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding pieces (engine kernels, real weights)
// ─────────────────────────────────────────────────────────────────────────────

export interface Projection {
  /** The encoder input (384-d MiniLM vector, or a length-1 scalar). */
  input: Float32Array;
  /** W·input + b (d). */
  pre: Float32Array;
  /** RMSNorm(W·input + b) with the type's γ (d). */
  term: Float32Array;
}

/** Column term of a column phrase: RMSNorm_col(W_col · E(phrase) + b_col)  (model.py:209-214). */
export function columnProjection(w: RTWeights, store: EmbeddingStore, phrase: string): Projection {
  const d = w.config.dModel;
  const input = store.get(phrase);
  const pre = linearVec(input, tensor(w, 'enc_dict.col_name.weight'), d, tensor(w, 'enc_dict.col_name.bias'));
  const term = new Float32Array(d);
  rmsnormRow(pre, 0, d, tensor(w, 'norm_dict.col_name.weight'), w.config.eps, term, 0);
  return { input, pre, term };
}

/** Value term of a text string: RMSNorm_text(W_text · E(string) + b_text)  (model.py:216-220). */
export function textProjection(w: RTWeights, store: EmbeddingStore, text: string): Projection {
  const d = w.config.dModel;
  const input = store.get(text);
  const pre = linearVec(input, tensor(w, 'enc_dict.text.weight'), d, tensor(w, 'enc_dict.text.bias'));
  const term = new Float32Array(d);
  rmsnormRow(pre, 0, d, tensor(w, 'norm_dict.text.weight'), w.config.eps, term, 0);
  return { input, pre, term };
}

export type ScalarKey = Exclude<SemKey, 'text'>;

/** Value term of a scalar z of a non-text type (engine `scalarValueTerm`, model.py:218). */
export function scalarProjection(w: RTWeights, key: ScalarKey, z: number): Projection {
  const { preNorm, term } = scalarValueTerm(w, key, z);
  return { input: Float32Array.of(z), pre: preNorm, term };
}

/** The stored scalar a token feeds to its type's encoder (fly.rs:482-490). */
export function tokenScalar(t: Token): number {
  return t.sem === SemType.Number ? t.numberValue : t.sem === SemType.DateTime ? t.datetimeValue : t.booleanValue;
}

export function semKey(sem: SemType): SemKey {
  return SEM_KEYS[sem];
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-D PCA of a set of d-dim points (display only)
// ─────────────────────────────────────────────────────────────────────────────

/** Top eigenvector of a symmetric d×d matrix by power iteration, deflating `prev`. */
function topEigen(C: Float64Array, d: number, prev: Float64Array | null): { v: Float64Array; lambda: number } {
  let v = new Float64Array(d);
  for (let i = 0; i < d; i++) v[i] = Math.sin(i * 1.7 + 0.3) + 0.5; // deterministic start
  const orth = (x: Float64Array) => {
    if (!prev) return;
    let p = 0;
    for (let i = 0; i < d; i++) p += x[i] * prev[i];
    for (let i = 0; i < d; i++) x[i] -= p * prev[i];
  };
  let lambda = 0;
  for (let it = 0; it < 300; it++) {
    orth(v);
    const nv = new Float64Array(d);
    for (let i = 0; i < d; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += C[i * d + j] * v[j];
      nv[i] = s;
    }
    orth(nv);
    let n = 0;
    for (let i = 0; i < d; i++) n += nv[i] * nv[i];
    n = Math.sqrt(n);
    if (n === 0) break;
    for (let i = 0; i < d; i++) nv[i] /= n;
    lambda = n;
    let diff = 0;
    for (let i = 0; i < d; i++) diff += Math.abs(nv[i] - v[i]);
    v = nv;
    if (diff < 1e-12) break;
  }
  return { v, lambda };
}

export interface Pca2 {
  e1: Float64Array;
  e2: Float64Array;
  /** Share of the (centered) variance captured by the first two components. */
  explained: number;
}

/** First two principal axes of `points` (each length d), via the covariance matrix. */
export function pca2(points: ArrayLike<number>[], d: number): Pca2 {
  const n = points.length;
  const mean = new Float64Array(d);
  for (const p of points) for (let i = 0; i < d; i++) mean[i] += p[i] / n;
  const C = new Float64Array(d * d);
  let total = 0;
  for (const p of points) {
    for (let i = 0; i < d; i++) {
      const a = p[i] - mean[i];
      total += a * a;
      for (let j = i; j < d; j++) C[i * d + j] += a * (p[j] - mean[j]);
    }
  }
  for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) C[i * d + j] = C[j * d + i];
  const a = topEigen(C, d, null);
  const b = topEigen(C, d, a.v);
  return { e1: a.v, e2: b.v, explained: total > 0 ? Math.min(1, (a.lambda + b.lambda) / total) : 1 };
}

export function project(p: ArrayLike<number>, e: Float64Array): number {
  let s = 0;
  for (let i = 0; i < e.length; i++) s += p[i] * e[i];
  return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature columns of the live DB, by sem type (pre.rs:543-611: type = physical dtype)
// ─────────────────────────────────────────────────────────────────────────────

export interface FeatureColumn {
  /** "<table>.<col>" */
  key: string;
  table: string;
  col: string;
  phrase: string;
  sem: SemType;
  /** "<table>:Db" or "<task>:Train" (the stats Val/Test reuse, pre.rs:383-396). */
  statsKey: string;
  isTaskTable: boolean;
  /** Number/boolean: the column's z-score stats; datetime: the DB's global stats (ms). */
  stats: { mean: number; std: number } | null;
  /** z of every stored cell of this column (all rows of the table file(s)). */
  zs: number[];
}

/** Every feature column (non-PK, non-FK) of the DB tables and forecast task tables. */
export function featureColumns(db: RawDatabase, pdb: PreprocessedDB): FeatureColumn[] {
  const out: FeatureColumn[] = [];
  const zsOf = (tableKeys: string[], col: string, sem: SemType) => {
    const zs: number[] = [];
    for (const key of tableKeys) {
      const info = pdb.tableInfo[key];
      if (!info) continue;
      for (let n = info.nodeIdxOffset; n < info.nodeIdxOffset + info.numNodes; n++) {
        const c = pdb.nodes[n]?.cells.find((x) => x.colName === col);
        if (!c) continue;
        zs.push(sem === SemType.Number ? c.number : sem === SemType.DateTime ? c.datetime : c.boolean);
      }
    }
    return zs;
  };
  const push = (table: string, col: string, dtype: Parameters<typeof semOfDType>[0], isTask: boolean) => {
    const sem = semOfDType(dtype);
    const statsKey = isTask ? `${table}:Train` : `${table}:Db`;
    let stats: FeatureColumn['stats'] = null;
    if (sem === SemType.DateTime) stats = { mean: pdb.datetimeStats.mean, std: pdb.datetimeStats.std };
    else if (sem !== SemType.Text) {
      const s: ColumnStats | undefined = pdb.colStats[statsKey]?.[col];
      stats = s ? { mean: s.mean, std: s.std } : null;
    }
    const keys = isTask ? [`${table}:Train`, `${table}:Val`, `${table}:Test`] : [`${table}:Db`];
    out.push({
      key: `${table}.${col}`,
      table,
      col,
      phrase: `${col} of ${table}`,
      sem,
      statsKey,
      isTaskTable: isTask,
      stats,
      zs: sem === SemType.Text ? [] : zsOf(keys, col, sem),
    });
  };
  for (const t of db.tables) {
    const s = t.schema;
    for (const c of s.columns) {
      if (c.name === s.pkey || c.name in s.fkeys) continue;
      push(s.name, c.name, c.dtype, false);
    }
  }
  for (const task of db.tasks) {
    if (task.kind !== 'forecast' || !task.tables) continue;
    const s = task.tables.Train.schema;
    for (const c of s.columns) {
      if (c.name === s.pkey || c.name in s.fkeys) continue;
      push(s.name, c.name, c.dtype, true);
    }
  }
  return out;
}

/** Raw value ↔ z for display (inverse of the z-score; datetime in epoch ms). */
export function zToRaw(col: FeatureColumn, z: number): number | null {
  return col.stats ? col.stats.mean + z * col.stats.std : null;
}

/**
 * A cell token of a scalar feature column holding z in its type's slot, built the way the sampler
 * writes one (fly.rs:472-490: the phrase index in col_name_values, the value in its type's slot, 0 in
 * the other scalar slots), so the engine's `embedTokens` can embed it through the real column and value
 * paths. null if the phrase is not in the DB's string table. Display only.
 */
export function syntheticCell(pdb: PreprocessedDB, col: FeatureColumn, z: number): Token | null {
  const colNameIdx = pdb.columnIndex[col.phrase] ?? pdb.textToIdx.get(col.phrase);
  if (colNameIdx === undefined || col.sem === SemType.Text) return null;
  return {
    pos: 0,
    isPadding: false,
    nodeIdx: -1,
    f2pNbrIdxs: [-1, -1, -1, -1, -1],
    tableNameIdx: pdb.textToIdx.get(col.table) ?? 0,
    colNameIdx,
    sem: col.sem,
    numberValue: col.sem === SemType.Number ? z : 0,
    textIdx: 0,
    datetimeValue: col.sem === SemType.DateTime ? z : 0,
    booleanValue: col.sem === SemType.Boolean ? z : 0,
    mask: false,
    isTarget: false,
    isTaskNode: false,
    classValueIdx: -1,
    tableName: col.table,
    tableType: null,
    colName: col.col,
    rowPos: -1,
    raw: null,
    depth: -1,
  };
}
