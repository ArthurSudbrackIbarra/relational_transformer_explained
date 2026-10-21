/**
 * Chapter 5 ("semantics") data layer. Everything here is either
 *  - derived live from the engine (database schemas, pre.rs string tables, the frozen MiniLM
 *    vectors in public/data/embeddings), or
 *  - copied verbatim from the paper PDF (Tables 4, 6 and 11), with the page cited next to it.
 * Nothing is hard-coded that the engine computes.
 */
import { useEffect, useState } from 'react';
import { DB_NAMES, getDatabase, tableColorSlot, type DbName } from '@/engine/db';
import { loadEmbeddingStore, type EmbeddingStore } from '@/engine/embeddings';
import { loadPublicJson } from '@/engine/model/io';
import type { DType, RawDatabase, RawTable } from '@/engine/types';
import encoderJson from './encoderData.json';

// ─────────────────────────────────────────────────────────────────────────────
// Encoder facts (generated offline by tools/encoder_data.py from the real tokenizer + model)
// ─────────────────────────────────────────────────────────────────────────────

export interface EncoderData {
  model: string;
  maxSeqLength: number;
  parameters: number;
  layers: number;
  hidden: number;
  heads: number;
  vocab: number;
  lowercase: boolean;
  modules: string[];
  meanPoolCheck: { maxAbsDiffAllWordpieces: number; minMaxAbsDiffWithoutClsSep: number };
  maxWordpieces: number;
  wordpieces: Record<string, string[]>;
}

export const ENCODER = encoderJson as unknown as EncoderData;

/** The WordPiece tokens of a string (with [CLS] and [SEP]), or null if it was never embedded. */
export function wordpiecesOf(text: string): string[] | null {
  return ENCODER.wordpieces[text] ?? null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Databases and their column phrases
// ─────────────────────────────────────────────────────────────────────────────

export const DB_SHORT: Record<DbName, string> = { shop: 'ShopDB', school: 'SchoolDB', racing: 'RacingDB', forum: 'ForumDB' };

/** Colour slot of a database in figures that colour by database (fixed order, never cycled). */
export const DB_SLOT: Record<DbName, number> = { shop: 1, school: 2, racing: 3, forum: 4 };

export type PhraseRole = 'pk' | 'fk' | 'time' | 'feature' | 'label';

export interface PhraseInfo {
  /** Exactly "<col> of <table>" (pre.rs:432). */
  text: string;
  db: DbName;
  table: string;
  col: string;
  dtype: DType;
  tableKind: 'Db' | 'Task';
  role: PhraseRole;
  /** PK and FK columns get a phrase index but never produce a cell (pre.rs:439-533). */
  onToken: boolean;
  /** Table colour slot (task tables = 5). */
  slot: number;
}

export const ROLE_LABEL: Record<PhraseRole, string> = {
  pk: 'primary key',
  fk: 'foreign key',
  time: 'time column',
  feature: 'feature',
  label: 'task label',
};

/** The task tables of a DB (forecast tasks only; all three splits share one table name). */
function taskTables(db: RawDatabase): { name: string; table: RawTable; target: string }[] {
  const out: { name: string; table: RawTable; target: string }[] = [];
  for (const t of db.tasks) if (t.kind === 'forecast' && t.tables) out.push({ name: t.name, table: t.tables.Train, target: t.targetCol });
  return out;
}

/** Every column phrase of a DB in schema order: DB tables, then task tables. */
export function phrasesOf(db: RawDatabase): PhraseInfo[] {
  const name = db.name as DbName;
  const out: PhraseInfo[] = [];
  const seen = new Set<string>();
  const visit = (table: RawTable, kind: 'Db' | 'Task', target?: string) => {
    const s = table.schema;
    for (const c of s.columns) {
      const text = `${c.name} of ${s.name}`;
      if (seen.has(text)) continue;
      seen.add(text);
      const role: PhraseRole =
        c.name === s.pkey ? 'pk' : c.name in s.fkeys ? 'fk' : c.name === target ? 'label' : c.name === s.timeCol ? 'time' : 'feature';
      out.push({
        text,
        db: name,
        table: s.name,
        col: c.name,
        dtype: c.dtype,
        tableKind: kind,
        role,
        onToken: role !== 'pk' && role !== 'fk',
        slot: tableColorSlot(name, s.name),
      });
    }
  };
  for (const t of db.tables) visit(t, 'Db');
  for (const t of taskTables(db)) visit(t.table, 'Task', t.target);
  return out;
}

export interface Catalog {
  dbs: Record<DbName, RawDatabase>;
  phrases: PhraseInfo[];
  byText: Map<string, PhraseInfo>;
  byDb: Record<DbName, PhraseInfo[]>;
}

let catalogCache: Catalog | null = null;

/** All four databases (generated deterministically by the engine) and their phrases. */
export function buildCatalog(): Catalog {
  if (catalogCache) return catalogCache;
  const dbs = {} as Record<DbName, RawDatabase>;
  const byDb = {} as Record<DbName, PhraseInfo[]>;
  const phrases: PhraseInfo[] = [];
  for (const n of DB_NAMES) {
    dbs[n] = getDatabase(n);
    byDb[n] = phrasesOf(dbs[n]);
    phrases.push(...byDb[n]);
  }
  catalogCache = { dbs, phrases, byText: new Map(phrases.map((p) => [p.text, p])), byDb };
  return catalogCache;
}

/**
 * The catalog, built after the first paint (generating the three other databases costs a few
 * hundred ms once; the engine caches them for every later chapter).
 */
export function useCatalog(): Catalog | null {
  const [cat, setCat] = useState<Catalog | null>(catalogCache);
  useEffect(() => {
    if (cat) return;
    const id = window.setTimeout(() => setCat(buildCatalog()), 30);
    return () => window.clearTimeout(id);
  }, [cat]);
  return cat;
}

// ─────────────────────────────────────────────────────────────────────────────
// The embedding store (frozen MiniLM vectors) and the 2-D PCA file
// ─────────────────────────────────────────────────────────────────────────────

export type LoadState<T> = { status: 'loading' } | { status: 'ready'; value: T } | { status: 'error'; error: string };

/** The precomputed all-MiniLM-L12-v2 table (shared with the forward pass; cached by the engine). */
export function useEmbeddingStore(): LoadState<EmbeddingStore> {
  const [st, setSt] = useState<LoadState<EmbeddingStore>>({ status: 'loading' });
  useEffect(() => {
    let alive = true;
    loadEmbeddingStore().then(
      (value) => alive && setSt({ status: 'ready', value }),
      (e: unknown) => alive && setSt({ status: 'error', error: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      alive = false;
    };
  }, []);
  return st;
}

export interface PcaPoint {
  text: string;
  db: DbName;
  table: string;
  col: string;
  x: number;
  y: number;
}

let pcaPromise: Promise<PcaPoint[]> | null = null;

/** public/data/embeddings/pca2d.json: 2-D PCA of every column phrase of the four DBs (python/embed.py). */
export function usePca(): LoadState<PcaPoint[]> {
  const [st, setSt] = useState<LoadState<PcaPoint[]>>({ status: 'loading' });
  useEffect(() => {
    let alive = true;
    if (!pcaPromise) {
      pcaPromise = loadPublicJson<PcaPoint[]>('data/embeddings/pca2d.json');
      pcaPromise.catch(() => (pcaPromise = null));
    }
    pcaPromise.then(
      (value) => alive && setSt({ status: 'ready', value }),
      (e: unknown) => alive && setSt({ status: 'error', error: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      alive = false;
    };
  }, []);
  return st;
}

// ─────────────────────────────────────────────────────────────────────────────
// Vector maths (all vectors are L2-unit-norm, so cosine = dot product; we still divide by the
// norms so that the number shown is the cosine even if a vector were not unit length)
// ─────────────────────────────────────────────────────────────────────────────

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a: ArrayLike<number>): number {
  return Math.sqrt(dot(a, a));
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = norm(a) * norm(b);
  return n > 0 ? dot(a, b) / n : 0;
}

export interface Neighbor {
  text: string;
  cos: number;
}

/** The k strings of `pool` closest to `text` by cosine (the query itself excluded). */
export function nearest(store: EmbeddingStore, text: string, pool: readonly string[], k: number): Neighbor[] {
  if (!store.has(text)) return [];
  const q = store.get(text);
  const out: Neighbor[] = [];
  for (const s of pool) {
    if (s === text || !store.has(s)) continue;
    out.push({ text: s, cos: cosine(q, store.get(s)) });
  }
  out.sort((a, b) => b.cos - a.cos);
  return out.slice(0, k);
}

/** Percentile rank (0..100) of v within xs (share of values strictly below v). */
export function percentileOf(v: number, xs: readonly number[]): number {
  if (!xs.length) return 0;
  let below = 0;
  for (const x of xs) if (x < v) below++;
  return (100 * below) / xs.length;
}

// ─────────────────────────────────────────────────────────────────────────────
// Column-name overlap (paper App. C.2 statistic, recomputed on our toy DBs)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Every column name of a database's own tables (keys included, task tables excluded). The paper does
 * not say exactly which columns its Table 6 counted; this is our stated choice.
 */
export function dbColumnNames(db: RawDatabase): Set<string> {
  return new Set(db.tables.flatMap((t) => t.schema.columns.map((c) => c.name)));
}

export function jaccard(a: ReadonlySet<string>, b: ReadonlySet<string>): { j: number; shared: string[] } {
  const shared = [...a].filter((x) => b.has(x)).sort();
  const union = new Set([...a, ...b]).size;
  return { j: union ? shared.length / union : 0, shared };
}

// ─────────────────────────────────────────────────────────────────────────────
// Paper numbers (copied from the PDF; do not edit without re-reading the page)
// ─────────────────────────────────────────────────────────────────────────────

/** Table 4 (p.9): mean AUROC (%) / R² (%), "none" vs "col names" (shuffled column and table names). */
export const TABLE4 = {
  none: { zsClf: 70.1, zsReg: 22.8, ftClf: 77.2, ftReg: 33.2 },
  colNames: { zsClf: 69.5, zsReg: 20.5, ftClf: 77.5, ftReg: 33.2 },
} as const;

export interface AblationRow {
  db: string;
  task: string;
  /** zero-shot: none, col names */
  zs: [number, number];
  /** fine-tuned: none, col names */
  ft: [number, number];
}

/** Table 11 (p.22), columns "none" and "col names" only. AUROC (%) of the 10 binary tasks. */
export const TABLE11_CLF: AblationRow[] = [
  { db: 'rel-amazon', task: 'item-churn', zs: [70.2, 71.0], ft: [83.4, 83.3] },
  { db: 'rel-amazon', task: 'user-churn', zs: [63.9, 63.9], ft: [70.8, 70.4] },
  { db: 'rel-avito', task: 'user-clicks', zs: [59.5, 58.5], ft: [59.0, 60.8] },
  { db: 'rel-avito', task: 'user-visits', zs: [61.8, 61.2], ft: [65.2, 65.1] },
  { db: 'rel-f1', task: 'driver-dnf', zs: [82.0, 81.7], ft: [84.2, 84.3] },
  { db: 'rel-f1', task: 'driver-top3', zs: [89.1, 86.5], ft: [91.9, 91.8] },
  { db: 'rel-hm', task: 'user-churn', zs: [62.8, 60.0], ft: [70.5, 70.6] },
  { db: 'rel-stack', task: 'user-badge', zs: [80.0, 79.2], ft: [88.7, 88.9] },
  { db: 'rel-stack', task: 'user-engage', zs: [77.1, 80.1], ft: [90.2, 90.1] },
  { db: 'rel-trial', task: 'study-out', zs: [54.5, 53.2], ft: [68.2, 69.0] },
];

/** Table 11 (p.22), R² (%) of the 8 regression tasks. */
export const TABLE11_REG: AblationRow[] = [
  { db: 'rel-amazon', task: 'item-ltv', zs: [33.2, 33.0], ft: [36.8, 34.7] },
  { db: 'rel-amazon', task: 'user-ltv', zs: [36.4, 33.6], ft: [47.4, 47.2] },
  { db: 'rel-avito', task: 'ad-ctr', zs: [4.5, 5.7], ft: [4.5, 4.9] },
  { db: 'rel-f1', task: 'driver-pos', zs: [54.7, 50.3], ft: [51.6, 52.3] },
  { db: 'rel-hm', task: 'item-sales', zs: [14.0, 9.2], ft: [39.0, 39.5] },
  { db: 'rel-stack', task: 'post-votes', zs: [32.4, 27.5], ft: [36.5, 35.9] },
  { db: 'rel-trial', task: 'site-succ', zs: [5.2, 3.2], ft: [6.4, 8.3] },
  { db: 'rel-trial', task: 'study-adv', zs: [2.1, 1.3], ft: [43.4, 42.4] },
];

/** Table 11 printed means (p.22). */
export const TABLE11_MEANS = {
  clf: { zs: [70.1, 69.5], ft: [77.2, 77.5] },
  reg: { zs: [22.8, 20.5], ft: [33.2, 33.2] },
} as const;

/** Table 6 (p.17): Jaccard similarity of exact column names across RelBench databases. */
export const TABLE6 = {
  dbs: ['rel-amazon', 'rel-avito', 'rel-f1', 'rel-hm', 'rel-stack', 'rel-trial'],
  m: [
    [1.0, 0.0, 0.0, 0.043, 0.0, 0.018],
    [0.0, 1.0, 0.0, 0.0, 0.018, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.022],
    [0.043, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.018, 0.0, 0.0, 1.0, 0.0],
    [0.018, 0.0, 0.022, 0.0, 0.0, 1.0],
  ],
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// Formatting
// ─────────────────────────────────────────────────────────────────────────────

export const f3 = (v: number) => (Number.isFinite(v) ? (v < 0 ? `−${Math.abs(v).toFixed(3)}` : v.toFixed(3)) : '—');
export const f2 = (v: number) => (Number.isFinite(v) ? (v < 0 ? `−${Math.abs(v).toFixed(2)}` : v.toFixed(2)) : '—');
export const f1 = (v: number) => (Number.isFinite(v) ? (v < 0 ? `−${Math.abs(v).toFixed(1)}` : v.toFixed(1)) : '—');
export const int = (n: number) => n.toLocaleString('en-US');
export const signed1 = (v: number) => (v > 0 ? `+${v.toFixed(1)}` : v < 0 ? `−${Math.abs(v).toFixed(1)}` : '0.0');

// ─────────────────────────────────────────────────────────────────────────────
// PCA map statistics (computed from pca2d.json and the stored vectors)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Share of the phrases' total variance kept by the two PCA axes:
 * Σᵢ(xᵢ² + yᵢ²) / Σᵢ‖eᵢ − ē‖². Exact because pca2d.json holds the projections of the centred
 * vectors onto the top-2 principal directions of these same phrases (python/embed.py `pca_2d`).
 */
export function pcaVarianceKept(points: readonly PcaPoint[], store: EmbeddingStore): number | null {
  const vecs = points.filter((p) => store.has(p.text)).map((p) => store.get(p.text));
  if (vecs.length !== points.length || !vecs.length) return null;
  const d = vecs[0].length;
  const mean = new Float64Array(d);
  for (const v of vecs) for (let j = 0; j < d; j++) mean[j] += v[j] / vecs.length;
  let total = 0;
  for (const v of vecs) for (let j = 0; j < d; j++) total += (v[j] - mean[j]) ** 2;
  let kept = 0;
  for (const p of points) kept += p.x * p.x + p.y * p.y;
  return total > 0 ? kept / total : null;
}

export function meanOf(xs: readonly number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;
}
