/**
 * What the site search can find on a rendered page: the text of each section (the page header,
 * then every chapter <Section>), split into blocks (paragraphs, list items, captions…). The index
 * builder (scripts/search-index.mjs, through window.__rtSearch) and the highlight after a result
 * is opened (landing.ts) both read pages through this file, so block numbers agree.
 *
 * Left out: figure bodies (interactive charts, tables and controls; a figure's caption and its
 * [data-fig-title] stay, and data-search="keep" keeps a whole figure), code excerpts, citation
 * chips, buttons, form controls and their labels, live readouts, eyebrows, display math,
 * screen-reader-only and hidden text, navigation, anything marked data-search="skip", and blocks
 * without a word (numbers, symbols).
 */

const SKIP = [
  'svg',
  'canvas',
  'pre',
  'button',
  'select',
  'input',
  'textarea',
  'nav',
  'aside',
  'annotation',
  'label',
  '[hidden]',
  '[aria-live]',
  '[role="status"]',
  '[role="img"]',
  '[role="tablist"]',
  '[role="radiogroup"]',
  '[role="toolbar"]',
  '[aria-hidden="true"]',
  '.sr-only',
  '.eyebrow',
  '.katex-display',
  '[data-cite]',
  '[data-search="skip"]',
].join(',');

const MATHML = 'http://www.w3.org/1998/Math/MathML';
/** Inside a figure, only these are text (the rest is the figure's interactive body). */
const FIG_TEXT = 'figcaption, [data-fig-title]';
const FIGURE = 'figure:not([data-search="keep"])';

/** Where an element sits relative to figures: plain text, a figure's caption/title, or its body. */
function figureRole(el: Element): 'text' | 'body' | 'path' {
  if (!el.closest(FIGURE) || el.closest(FIG_TEXT)) return 'text';
  // the figure or an ancestor of its caption/title: walk through it, take nothing else
  return el.querySelector(FIG_TEXT) ? 'path' : 'body';
}

/** One text node of a block and where its text starts in `Block.text`. */
export interface TextPart {
  node: Text;
  start: number;
}

export interface Block {
  el: Element;
  /** The rendered text, uncollapsed (offsets match `parts`). */
  text: string;
  parts: TextPart[];
}

export interface PageSection {
  /** The Section id, or '' for the page header. */
  id: string;
  heading: Block | null;
  blocks: Block[];
}

/** Text that flows inline stays in its block; anything laid out as a box starts a new block. */
function isInline(el: Element, style: CSSStyleDeclaration): boolean {
  if (el.namespaceURI === MATHML || el.closest('.katex')) return true;
  const d = style.display;
  return d.startsWith('inline') || d === 'contents';
}

/** The blocks of `container` in page order (empty and punctuation-only blocks dropped). */
export function walkBlocks(container: Element): Block[] {
  const styles = new Map<Element, CSSStyleDeclaration>();
  const styleOf = (el: Element) => {
    let s = styles.get(el);
    if (!s) styles.set(el, (s = getComputedStyle(el)));
    return s;
  };
  const blockOf = (el: Element): Element => {
    for (let e: Element | null = el; e && e !== container; e = e.parentElement) if (!isInline(e, styleOf(e))) return e;
    return container;
  };
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
    acceptNode(n) {
      if (n.nodeType === Node.TEXT_NODE) return n.parentElement && figureRole(n.parentElement) === 'text' ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      const el = n as Element;
      if (el.matches(SKIP)) return NodeFilter.FILTER_REJECT;
      if (figureRole(el) === 'body') return NodeFilter.FILTER_REJECT;
      const st = styleOf(el);
      if (st.display === 'none' || st.visibility === 'hidden') return NodeFilter.FILTER_REJECT;
      return el.tagName === 'BR' ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
    },
  });
  const byEl = new Map<Element, Block>();
  const order: Block[] = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const parent = n.nodeType === Node.TEXT_NODE ? n.parentElement : (n as Element).parentElement;
    if (!parent) continue;
    const el = blockOf(parent);
    let b = byEl.get(el);
    if (!b) {
      if (n.nodeType !== Node.TEXT_NODE) continue;
      b = { el, text: '', parts: [] };
      byEl.set(el, b);
      order.push(b);
    }
    if (n.nodeType === Node.TEXT_NODE) {
      b.parts.push({ node: n as Text, start: b.text.length });
      b.text += (n as Text).data;
    } else b.text += ' '; // <br>
  }
  return order.filter((b) => /\p{L}{2}/u.test(b.text));
}

/** Sections of a page: the header ('' id) and each `section[id]`, with their blocks. */
export function extractSections(root: Element): PageSection[] {
  const sections: PageSection[] = [];
  const byEl = new Map<Element, PageSection>();
  const header: PageSection = { id: '', heading: null, blocks: [] };
  sections.push(header);
  let current = header;
  for (const b of walkBlocks(root)) {
    const secEl = b.el.closest('section[id]');
    if (secEl && root.contains(secEl)) {
      let s = byEl.get(secEl);
      if (!s) {
        s = { id: secEl.id, heading: null, blocks: [] };
        byEl.set(secEl, s);
        sections.push(s);
      }
      current = s;
    }
    const headingEl = current.id ? b.el.closest(`#${CSS.escape(current.id)}-title`) : b.el.closest('h1');
    if (headingEl && !current.heading) current.heading = b;
    else current.blocks.push(b);
  }
  return sections.filter((s) => s.heading || s.blocks.length);
}

/** The rendered page's article (chapters and Q&A pages), if it has mounted. */
export function pageRoot(): Element | null {
  return document.querySelector('main article');
}

/** Collapsed text of a block, as stored in the index. */
export function blockText(b: Block): string {
  return b.text.replace(/\s+/g, ' ').trim();
}
