/**
 * Small React hooks shared by the chapter-2 figures (display only, no RT logic).
 */
import { useEffect, useRef, useState, startTransition } from 'react';

/** True once the element is within `margin` of the viewport (and while it stays there). */
export function useNearViewport<T extends Element>(margin: string): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T | null>(null);
  const [near, setNear] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') {
      setNear(true);
      return;
    }
    const io = new IntersectionObserver((entries) => {
      const v = entries.some((e) => e.isIntersecting);
      // a re-render that the figure needs when it scrolls into view never blocks the scroll itself
      startTransition(() => setNear(v));
    }, { rootMargin: margin });
    io.observe(el);
    return () => io.disconnect();
  }, [margin]);
  return [ref, near];
}

/** For a horizontal scroller: whether there is hidden content to its left / right (for edge fades). */
export function useScrollEdges<T extends HTMLElement>(): [React.RefObject<T | null>, { left: boolean; right: boolean }] {
  const ref = useRef<T | null>(null);
  const [edges, setEdges] = useState({ left: false, right: false });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => {
      const left = el.scrollLeft > 2;
      const right = el.scrollLeft + el.clientWidth < el.scrollWidth - 2;
      setEdges((e) => (e.left === left && e.right === right ? e : { left, right }));
    };
    update();
    el.addEventListener('scroll', update, { passive: true });
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(update) : null;
    ro?.observe(el);
    if (el.firstElementChild) ro?.observe(el.firstElementChild);
    return () => {
      el.removeEventListener('scroll', update);
      ro?.disconnect();
    };
  }, []);
  return [ref, edges];
}
