/**
 * Site search, the pure part: folding text for matching, parsing the query, ranking sections
 * and cutting snippets with the matched words marked. The DOM side is extract.ts (what a page
 * says) and landing.ts (highlighting a match after navigation).
 *
 * A term matches at the start of a word, ignoring case and accents, so "attent" finds
 * "attention" and "self-attention". Every term must occur in a section (title or text); a
 * multi-word query also scores its exact phrase.
 */
import type { SearchIndex, SearchPage, SearchSection } from './types';

/** Lowercase, accents stripped, curly apostrophes straightened. */
export function fold(s: string): string {
  return s.normalize('NFD').replace(/\p{M}/gu, '').toLowerCase().replace(/[‘’]/g, "'");
}

function foldChar(ch: string): string {
  return ch === '‘' || ch === '’' ? "'" : ch.normalize('NFD').replace(/\p{M}/gu, '').toLowerCase();
}

/** `fold` plus, for every folded char, the index of the raw char it came from (and raw.length at the end). */
export function foldWithMap(raw: string): { text: string; map: number[] } {
  let text = '';
  const map: number[] = [];
  for (let i = 0; i < raw.length; i++) {
    const f = raw.charCodeAt(i) < 128 ? raw[i].toLowerCase() : foldChar(raw[i]);
    for (let k = 0; k < f.length; k++) {
      text += f[k];
      map.push(i);
    }
  }
  map.push(raw.length);
  return { text, map };
}

export interface Query {
  terms: string[];
  /** The whole query when it has several words (whitespace collapsed), else null. */
  phrase: string | null;
}

/** Terms of two or more characters; null when nothing searchable is left. */
export function parseQuery(input: string): Query | null {
  const f = fold(input).replace(/\s+/g, ' ').trim();
  const words = f.split(/[^\p{L}\p{N}_]+/u).filter(Boolean);
  const terms = [...new Set(words.filter((w) => w.length > 1))];
  if (!terms.length) return null;
  return { terms, phrase: words.length > 1 ? f : null };
}

const reCache = new Map<string, RegExp>();
/** Matches `term` at a word start. */
export function termRe(term: string): RegExp {
  let re = reCache.get(term);
  if (!re) {
    re = new RegExp(`(?<![\\p{L}\\p{N}_])${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, 'gu');
    reCache.set(term, re);
  }
  re.lastIndex = 0;
  return re;
}

function count(re: RegExp, s: string, cap = 20): number {
  re.lastIndex = 0;
  let n = 0;
  while (n < cap && re.exec(s)) n++;
  return n;
}

/** [start, end) ranges in `text` (raw, unfolded) where the query's terms and phrase match. */
export function matchRanges(text: string, q: Query): [number, number][] {
  const { text: f, map } = foldWithMap(text);
  const out: [number, number][] = [];
  for (const t of q.terms) {
    const re = termRe(t);
    for (let m = re.exec(f); m; m = re.exec(f)) out.push([map[m.index], map[m.index + t.length]]);
  }
  if (q.phrase) for (let i = f.indexOf(q.phrase); i >= 0; i = f.indexOf(q.phrase, i + 1)) out.push([map[i], map[i + q.phrase.length]]);
  out.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const r of out) {
    const last = merged[merged.length - 1];
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
    else merged.push([r[0], r[1]]);
  }
  return merged;
}

export interface Segment {
  text: string;
  hit: boolean;
}

/** `text` split into plain and matched segments. */
export function markText(text: string, q: Query): Segment[] {
  const segs: Segment[] = [];
  let at = 0;
  for (const [s, e] of matchRanges(text, q)) {
    if (s > at) segs.push({ text: text.slice(at, s), hit: false });
    segs.push({ text: text.slice(s, e), hit: true });
    at = e;
  }
  if (at < text.length) segs.push({ text: text.slice(at), hit: false });
  return segs;
}

/** A window of `text` around its first match (cut at word breaks, with ellipses), marked. */
export function snippet(text: string, q: Query, width = 180): Segment[] {
  const ranges = matchRanges(text, q);
  const first = ranges[0]?.[0] ?? 0;
  let start = Math.max(0, first - 55);
  if (start > 0) {
    const sp = text.indexOf(' ', start);
    start = sp >= 0 && sp < first ? sp + 1 : start;
  }
  let end = Math.min(text.length, start + width);
  if (end < text.length) {
    const sp = text.lastIndexOf(' ', end);
    end = sp > first ? sp : end;
  }
  const segs = markText(text.slice(start, end), q);
  if (start > 0) segs.unshift({ text: '…', hit: false });
  if (end < text.length) segs.push({ text: '…', hit: false });
  return segs;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ranking
// ─────────────────────────────────────────────────────────────────────────────

export interface PreparedSection extends SearchSection {
  fTitle: string;
  fBlocks: string[];
}

export interface PreparedPage extends Omit<SearchPage, 'sections'> {
  fTitle: string;
  sections: PreparedSection[];
}

export function prepare(index: SearchIndex): PreparedPage[] {
  return index.pages.map((p) => ({
    ...p,
    fTitle: fold(p.title),
    sections: p.sections.map((s) => ({ ...s, fTitle: fold(s.title), fBlocks: s.blocks.map(fold) })),
  }));
}

export interface SectionHit {
  section: PreparedSection;
  /** Position of the section on its page. */
  order: number;
  score: number;
  /** Best-matching block, -1 when only the title matched. */
  block: number;
}

export interface ResultGroup {
  page: PreparedPage;
  score: number;
  /** The page header (title, lede or question) when it matched. */
  header: SectionHit | null;
  hits: SectionHit[];
}

function scoreSection(page: PreparedPage, s: PreparedSection, order: number, q: Query): SectionHit | null {
  const n = s.fBlocks.length;
  const termsIn = new Array<number>(n).fill(0);
  const hitsIn = new Array<number>(n).fill(0);
  let score = 0;
  for (const t of q.terms) {
    const re = termRe(t);
    const inTitle = count(re, s.fTitle);
    let total = 0;
    for (let b = 0; b < n; b++) {
      const c = count(re, s.fBlocks[b]);
      if (c) {
        termsIn[b]++;
        hitsIn[b] += c;
        total += c;
      }
    }
    if (!inTitle && !total) return null;
    score += (inTitle ? 12 : 0) + Math.min(total, 6) + (count(re, page.fTitle, 1) ? 2 : 0);
  }
  let phraseBlock = -1;
  if (q.phrase) {
    if (s.fTitle.includes(q.phrase)) score += 25;
    phraseBlock = s.fBlocks.findIndex((b) => b.includes(q.phrase!));
    if (phraseBlock >= 0) score += 10;
  }
  let block = phraseBlock;
  if (block < 0) {
    let best = 0;
    for (let b = 0; b < n; b++) {
      const v = termsIn[b] * 10 + Math.min(hitsIn[b], 5);
      if (v > best) {
        best = v;
        block = b;
      }
    }
  }
  // all the words in one paragraph beat words scattered over the section
  if (q.terms.length > 1 && block >= 0 && termsIn[block] === q.terms.length) score += 8;
  return { section: s, order, score, block };
}

/** Matching sections grouped by page, best page first; within a page, best section first. */
export function search(pages: PreparedPage[], q: Query): ResultGroup[] {
  const groups: ResultGroup[] = [];
  pages.forEach((page) => {
    let header: SectionHit | null = null;
    const hits: SectionHit[] = [];
    page.sections.forEach((s, i) => {
      const hit = scoreSection(page, s, i, q);
      if (!hit) return;
      if (s.id === '') header = hit;
      else hits.push(hit);
    });
    if (!header && !hits.length) return;
    hits.sort((a, b) => b.score - a.score || a.order - b.order);
    const best = Math.max((header as SectionHit | null)?.score ?? 0, hits[0]?.score ?? 0);
    groups.push({ page, score: best, header, hits });
  });
  return groups.sort((a, b) => b.score - a.score);
}
