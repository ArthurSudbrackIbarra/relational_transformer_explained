/**
 * Numbers the chapter's prose quotes, computed live from the stored vectors (never assumed).
 */
import { DB_NAMES, type DbName } from '@/engine/db';
import type { EmbeddingStore } from '@/engine/embeddings';
import { cosine, meanOf, type Catalog, type PcaPoint, type PhraseInfo } from './data';

export interface LabelRank {
  phrase: PhraseInfo;
  cos: number;
  /** 1-based rank among all on-token phrases of the other databases */
  rank: number;
}

export interface TargetComparison {
  target: string;
  pool: number;
  nearest: LabelRank[];
  /** the other databases' task-label phrases, by cosine */
  labels: LabelRank[];
  meanAll: number;
}

/** How close the live target phrase is to every on-token phrase of the other databases. */
export function compareTarget(store: EmbeddingStore, catalog: Catalog, home: DbName, target: string): TargetComparison | null {
  if (!store.has(target)) return null;
  const q = store.get(target);
  const pool = DB_NAMES.filter((d) => d !== home).flatMap((d) => catalog.byDb[d].filter((p) => p.onToken));
  const scored = pool.map((p) => ({ phrase: p, cos: cosine(q, store.get(p.text)) })).sort((a, b) => b.cos - a.cos);
  const ranked = scored.map((s, i) => ({ ...s, rank: i + 1 }));
  return {
    target,
    pool: pool.length,
    nearest: ranked.slice(0, 3),
    labels: ranked.filter((r) => r.phrase.role === 'label'),
    meanAll: meanOf(scored.map((s) => s.cos)),
  };
}

/** The most similar cross-database pair of key-column phrases (home DB vs the others). */
export function topKeyPair(store: EmbeddingStore, catalog: Catalog, home: DbName): { a: string; b: string; cos: number } | null {
  const mine = catalog.byDb[home].filter((p) => !p.onToken);
  const theirs = DB_NAMES.filter((d) => d !== home).flatMap((d) => catalog.byDb[d].filter((p) => !p.onToken));
  let best: { a: string; b: string; cos: number } | null = null;
  for (const a of mine)
    for (const b of theirs) {
      const c = cosine(store.get(a.text), store.get(b.text));
      if (!best || c > best.cos) best = { a: a.text, b: b.text, cos: c };
    }
  return best;
}

/** The most similar cross-database pair of on-token phrases (home DB vs the others). */
export function topTokenPair(store: EmbeddingStore, catalog: Catalog, home: DbName): { a: string; b: string; cos: number } | null {
  const mine = catalog.byDb[home].filter((p) => p.onToken);
  const theirs = DB_NAMES.filter((d) => d !== home).flatMap((d) => catalog.byDb[d].filter((p) => p.onToken));
  let best: { a: string; b: string; cos: number } | null = null;
  for (const a of mine)
    for (const b of theirs) {
      const c = cosine(store.get(a.text), store.get(b.text));
      if (!best || c > best.cos) best = { a: a.text, b: b.text, cos: c };
    }
  return best;
}

export interface MapStats {
  keyPc1: number;
  otherPc1: number;
  pc2ByDb: { db: DbName; mean: number }[];
}

/** Simple, checkable descriptions of the PCA map. */
export function mapStats(points: readonly PcaPoint[], catalog: Catalog): MapStats {
  const isKey = (p: PcaPoint) => catalog.byText.get(p.text)?.onToken === false;
  return {
    keyPc1: meanOf(points.filter(isKey).map((p) => p.x)),
    otherPc1: meanOf(points.filter((p) => !isKey(p)).map((p) => p.x)),
    pc2ByDb: DB_NAMES.map((db) => ({ db, mean: meanOf(points.filter((p) => p.db === db).map((p) => p.y)) })).sort((a, b) => b.mean - a.mean),
  };
}

/** Mean cosine of on-token phrase pairs: same table, same database (other table), different databases. */
export function groupMeans(store: EmbeddingStore, catalog: Catalog): { sameTable: number; sameDb: number; crossDb: number } {
  const ps = catalog.phrases.filter((p) => p.onToken && store.has(p.text));
  const vs = ps.map((p) => store.get(p.text));
  const acc = { sameTable: [0, 0], sameDb: [0, 0], crossDb: [0, 0] };
  for (let i = 0; i < ps.length; i++)
    for (let k = i + 1; k < ps.length; k++) {
      const c = cosine(vs[i], vs[k]);
      const g = ps[i].db !== ps[k].db ? acc.crossDb : ps[i].table === ps[k].table ? acc.sameTable : acc.sameDb;
      g[0] += c;
      g[1] += 1;
    }
  const m = (g: number[]) => (g[1] ? g[0] / g[1] : NaN);
  return { sameTable: m(acc.sameTable), sameDb: m(acc.sameDb), crossDb: m(acc.crossDb) };
}

/**
 * 384-d backing for what the PCA map suggests: mean cosine between two key-column phrases, between a
 * key-column phrase and any other phrase, and between two non-key phrases (all four databases).
 */
export function keyMeans(store: EmbeddingStore, catalog: Catalog): { keyKey: number; keyOther: number; otherOther: number } {
  const ps = catalog.phrases.filter((p) => store.has(p.text));
  const vs = ps.map((p) => store.get(p.text));
  const acc = { keyKey: [0, 0], keyOther: [0, 0], otherOther: [0, 0] };
  for (let i = 0; i < ps.length; i++)
    for (let k = i + 1; k < ps.length; k++) {
      const a = !ps[i].onToken;
      const b = !ps[k].onToken;
      const g = a && b ? acc.keyKey : a || b ? acc.keyOther : acc.otherOther;
      g[0] += cosine(vs[i], vs[k]);
      g[1] += 1;
    }
  const m = (g: number[]) => (g[1] ? g[0] / g[1] : NaN);
  return { keyKey: m(acc.keyKey), keyOther: m(acc.keyOther), otherOther: m(acc.otherOther) };
}

/** Cosine of two stored strings, or null if either was never embedded. */
export function pairCos(store: EmbeddingStore, a: string, b: string): number | null {
  return store.has(a) && store.has(b) ? cosine(store.get(a), store.get(b)) : null;
}
