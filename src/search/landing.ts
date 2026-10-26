/**
 * After a search result is opened: find the matched words in the target section, scroll them into
 * view and highlight them briefly. The words are marked with the CSS Custom Highlight API and the
 * block gets a fading glow (Web Animations), so React's DOM is never changed. The search passes
 * { find } in the router state; a plain ?section= link keeps ChapterPage's own scroll.
 */
import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { blockText, extractSections, pageRoot, type Block } from './extract';
import { fold, matchRanges, termRe, type Query } from './match';
import type { FindState } from './types';

const HIT = 'rt-search';
const HIT_FADE = 'rt-search-fade';
/** How long to wait for a lazy page to mount, and for a mounted section's text to settle. */
const WAIT_PAGE_MS = 6000;
const WAIT_TEXT_MS = 1500;

interface Target {
  el: Element;
  ranges: Range[];
}

function toRanges(b: Block, q: Query): Range[] {
  const at = (pos: number, end: boolean) => {
    // the text node holding raw offset `pos` (an end offset may sit at a node's end)
    for (let i = b.parts.length - 1; i >= 0; i--) {
      const p = b.parts[i];
      if (end ? pos > p.start : pos >= p.start) return { node: p.node, offset: Math.min(pos - p.start, p.node.data.length) };
    }
    return null;
  };
  const out: Range[] = [];
  for (const [s, e] of matchRanges(b.text, q)) {
    const a = at(s, false);
    const z = at(e, true);
    if (!a || !z) continue;
    const r = document.createRange();
    r.setStart(a.node, a.offset);
    r.setEnd(z.node, z.offset);
    out.push(r);
  }
  return out;
}

/** How well a block matches: distinct terms first, then the phrase, then the number of hits. */
function blockScore(b: Block, q: Query): number {
  const f = fold(b.text);
  let s = 0;
  for (const t of q.terms) {
    const re = termRe(t);
    let n = 0;
    while (n < 5 && re.exec(f)) n++;
    if (n) s += 10 + n;
  }
  return q.phrase && f.replace(/\s+/g, ' ').includes(q.phrase) ? s + 20 : s;
}

/**
 * The block to show: the one with the indexed text (the index was read at one width, and layouts
 * differ by width), else the best-matching block, nearest the indexed position on ties.
 */
function locate(sectionId: string, find: FindState): Target | 'missing' | null {
  const root = pageRoot();
  if (!root || (sectionId && !root.querySelector(`section[id="${CSS.escape(sectionId)}"]`))) return 'missing';
  const sec = extractSections(root).find((s) => s.id === sectionId);
  if (!sec) return 'missing';
  const q: Query = { terms: find.terms, phrase: find.phrase };
  const same = find.text ? sec.blocks.find((b) => blockText(b) === find.text) : undefined;
  let pick: Block | null = same && toRanges(same, q).length ? same : null;
  if (!pick) {
    let best = 0;
    for (let i = 0; i < sec.blocks.length; i++) {
      const v = blockScore(sec.blocks[i], q) - Math.abs(i - Math.max(find.block, 0)) * 1e-3;
      if (v > best) {
        best = v;
        pick = sec.blocks[i];
      }
    }
  }
  if (!pick && sec.heading && blockScore(sec.heading, q) > 0) pick = sec.heading;
  return pick ? { el: pick.el, ranges: toRanges(pick, q) } : null;
}

let cancelFlash: (() => void) | null = null;

function flash({ el, ranges }: Target): void {
  cancelFlash?.();
  const timers: number[] = [];
  const reg = typeof CSS !== 'undefined' && 'highlights' in CSS ? CSS.highlights : null;
  if (reg && ranges.length) {
    reg.set(HIT, new Highlight(...ranges));
    timers.push(
      window.setTimeout(() => {
        reg.delete(HIT);
        reg.set(HIT_FADE, new Highlight(...ranges));
      }, 2300),
      window.setTimeout(() => reg.delete(HIT_FADE), 2800),
    );
  }
  const glow = getComputedStyle(document.documentElement).getPropertyValue('--search-glow').trim() || 'rgba(250, 196, 0, 0.14)';
  const anim = el.animate?.(
    [
      { backgroundColor: glow, boxShadow: `0 0 0 6px ${glow}`, borderRadius: '6px' },
      { backgroundColor: glow, boxShadow: `0 0 0 6px ${glow}`, borderRadius: '6px', offset: 0.55 },
      { backgroundColor: 'transparent', boxShadow: '0 0 0 6px transparent', borderRadius: '6px' },
    ],
    { duration: 2800, easing: 'ease-out' },
  );
  cancelFlash = () => {
    timers.forEach((t) => window.clearTimeout(t));
    reg?.delete(HIT);
    reg?.delete(HIT_FADE);
    anim?.cancel();
    cancelFlash = null;
  };
}

function scrollToTarget({ el, ranges }: Target, behavior: ScrollBehavior): void {
  const rect = (ranges[0] ?? el).getBoundingClientRect();
  window.scrollTo({ top: window.scrollY + rect.top - window.innerHeight * 0.32, behavior });
}

function inView({ el, ranges }: Target): boolean {
  const r = (ranges[0] ?? el).getBoundingClientRect();
  return r.top >= 60 && r.bottom <= window.innerHeight - 20;
}

/** Runs on every navigation; acts only when the router state carries a search match. */
export function useSearchLanding(): void {
  const location = useLocation();
  useEffect(() => {
    const find = (location.state as { find?: FindState } | null)?.find;
    if (!find) return;
    const sectionId = new URLSearchParams(location.search).get('section') ?? '';
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const started = performance.now();
    const timers: number[] = [];
    let done = false;
    const attempt = () => {
      if (done) return;
      const t = locate(sectionId, find);
      if (!t || t === 'missing') {
        // lazy pages mount after a moment and live text settles a little later; the words can also
        // be absent at this width (a table shown as cards), and then the section itself is the target
        if (performance.now() - started < (t === 'missing' ? WAIT_PAGE_MS : WAIT_TEXT_MS)) timers.push(window.setTimeout(attempt, 150));
        else if (sectionId) document.getElementById(sectionId)?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
        return;
      }
      done = true;
      scrollToTarget(t, reduce ? 'auto' : 'smooth');
      flash(t);
      // figures above can still change height while the page settles
      timers.push(window.setTimeout(() => !inView(t) && scrollToTarget(t, 'auto'), 1100));
    };
    timers.push(window.setTimeout(attempt, 60));
    return () => {
      done = true;
      timers.forEach((t) => window.clearTimeout(t));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- once per navigation
  }, [location.key]);
}
