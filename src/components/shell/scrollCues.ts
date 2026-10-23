/**
 * Scroll cues: a horizontally scrollable box (wide table, code excerpt, display equation) fades out
 * at the edge where more content hides, so phone readers see that it scrolls (touch browsers hide
 * scrollbars). The box gets `data-cue="left|right|both"`; index.css turns that into a mask.
 */
import { useEffect } from 'react';

const SELECTOR = '.overflow-x-auto, .overflow-auto, .katex-display';

function update(el: HTMLElement): void {
  const max = el.scrollWidth - el.clientWidth;
  if (max <= 1) {
    if (el.hasAttribute('data-cue')) el.removeAttribute('data-cue');
    return;
  }
  const left = el.scrollLeft > 1;
  const right = el.scrollLeft < max - 1;
  const cue = left && right ? 'both' : left ? 'left' : 'right';
  if (el.getAttribute('data-cue') !== cue) el.setAttribute('data-cue', cue);
}

/** Keeps the cues of every scroll box in `main` current: on navigation, resize, content growth and scroll. */
export function useScrollCues(routeKey: string): void {
  useEffect(() => {
    const main = document.getElementById('main');
    if (!main) return;
    let frame = 0;
    const scan = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => main.querySelectorAll<HTMLElement>(SELECTOR).forEach(update));
    };
    const onScroll = (e: Event) => {
      const el = e.target;
      if (el instanceof HTMLElement && el.matches(SELECTOR)) update(el);
    };
    const resize = new ResizeObserver(scan);
    resize.observe(main);
    document.addEventListener('scroll', onScroll, true);
    // lazy chapters and figures settle after the first paint
    const timers = [100, 600, 1500, 3000].map((ms) => window.setTimeout(scan, ms));
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      document.removeEventListener('scroll', onScroll, true);
      timers.forEach((t) => window.clearTimeout(t));
    };
  }, [routeKey]);
}
