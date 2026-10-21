/** True once the element has come near the viewport (IntersectionObserver; stays true afterwards). */
import { useEffect, useRef, useState } from 'react';

export function useInView<T extends HTMLElement>(rootMargin = '300px'): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T | null>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    if (seen) return;
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === 'undefined') {
      setSeen(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSeen(true);
          io.disconnect();
        }
      },
      { rootMargin },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [seen, rootMargin]);
  return [ref, seen];
}

/** True once the browser has been idle after mount (requestIdleCallback, or a timeout fallback). */
export function useIdle(timeoutMs = 2000): boolean {
  const [idle, setIdle] = useState(false);
  useEffect(() => {
    if (idle) return;
    const w = window as Window & {
      requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number;
      cancelIdleCallback?: (id: number) => void;
    };
    if (w.requestIdleCallback) {
      const id = w.requestIdleCallback(() => setIdle(true), { timeout: timeoutMs });
      return () => w.cancelIdleCallback?.(id);
    }
    const id = window.setTimeout(() => setIdle(true), timeoutMs);
    return () => window.clearTimeout(id);
  }, [idle, timeoutMs]);
  return idle;
}
