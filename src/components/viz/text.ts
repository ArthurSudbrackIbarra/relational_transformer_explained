/**
 * Type scale for SVG / canvas text, plus a cached text-width measurer so
 * charts can size their margins and decide what fits instead of shrinking
 * text (BUILD_PLAN §5: never below 12px; make room, wrap, abbreviate with a
 * tooltip, or show fewer items).
 *
 * The font stacks mirror `--font-sans` / `--font-mono` in src/styles/index.css
 * (SVG text inherits them from <html>).
 */

/** Pixel sizes for text drawn inside SVG or canvas figures. */
export const SVG_TYPE = {
  /** Axis tick labels. */
  tick: 13.5,
  /** Axis titles ("AUROC (%)", "step"). */
  axisTitle: 14.5,
  /** Values printed on marks (bar tips, end labels). */
  value: 13.5,
  /** Category labels next to bars. */
  category: 15,
  /** Annotations: reference-line and marker labels. */
  annotation: 13.5,
  /** Node / cell labels inside diagrams. */
  label: 13.5,
  /** Hard floor: nothing is ever drawn smaller. */
  min: 12,
} as const;

const SANS = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
const MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace';

let ctx: CanvasRenderingContext2D | null | undefined;
const cache = new Map<string, number>();

/** Rendered width (CSS px) of `text` at `px` in the site's sans (or mono) font. */
export function textWidth(text: string, px: number, opts: { mono?: boolean; weight?: number } = {}): number {
  const weight = opts.weight ?? 400;
  const key = `${opts.mono ? 1 : 0}|${weight}|${px}|${text}`;
  const hit = cache.get(key);
  if (hit !== undefined) return hit;
  if (ctx === undefined) {
    try {
      ctx = typeof document !== 'undefined' ? document.createElement('canvas').getContext('2d') : null;
    } catch {
      ctx = null;
    }
  }
  let w: number;
  if (ctx) {
    ctx.font = `${weight} ${px}px ${opts.mono ? MONO : SANS}`;
    w = ctx.measureText(text).width;
  } else {
    // no canvas (tests / SSR): a generous per-character estimate
    w = text.length * px * (opts.mono ? 0.62 : 0.6);
  }
  if (cache.size > 5000) cache.clear();
  cache.set(key, w);
  return w;
}

/** Widest of several strings. */
export function maxTextWidth(texts: readonly string[], px: number, opts?: { mono?: boolean; weight?: number }): number {
  let m = 0;
  for (const t of texts) m = Math.max(m, textWidth(t, px, opts));
  return m;
}

/** Truncate `text` with an ellipsis so it fits in `maxW` px (returns it unchanged when it fits). */
export function fitText(text: string, maxW: number, px: number, opts?: { mono?: boolean; weight?: number }): string {
  if (textWidth(text, px, opts) <= maxW) return text;
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (textWidth(text.slice(0, mid) + '…', px, opts) <= maxW) lo = mid;
    else hi = mid - 1;
  }
  return lo > 0 ? text.slice(0, lo) + '…' : '…';
}
