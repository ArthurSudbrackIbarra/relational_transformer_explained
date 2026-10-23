/** Observe an element's content-box width/height (ResizeObserver). */
import { useLayoutEffect, useRef, useState, type RefObject } from 'react';

export function useElementSize<T extends HTMLElement>(): [RefObject<T | null>, { width: number; height: number }] {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      setSize((s) => (Math.abs(s.width - r.width) < 0.5 && Math.abs(s.height - r.height) < 0.5 ? s : { width: r.width, height: r.height }));
    };
    measure();
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', measure);
      return () => window.removeEventListener('resize', measure);
    }
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

/** Normalize `Set | array | undefined` highlight props into a Set. */
export function toSet<T>(v: ReadonlySet<T> | readonly T[] | null | undefined): ReadonlySet<T> {
  if (!v) return EMPTY as ReadonlySet<T>;
  return v instanceof Set ? v : new Set(v as readonly T[]);
}
const EMPTY: ReadonlySet<never> = new Set<never>();
