/**
 * Small async job runner for the playground's sweeps and batch evaluations. A forward pass costs
 * ~60-70 ms at L = 128, so jobs yield to the browser between forward passes (one per macrotask),
 * run one at a time (a serial queue), can be cancelled, and their results are cached at module
 * level by a key that names every input they depend on.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

const cache = new Map<string, unknown>();
const CACHE_MAX = 60;

export function cacheGet<T>(key: string): T | undefined {
  return cache.get(key) as T | undefined;
}
function cacheSet(key: string, v: unknown): void {
  cache.set(key, v);
  while (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value as string);
}

let queue: Promise<unknown> = Promise.resolve();
function enqueue<T>(job: () => Promise<T>): Promise<T> {
  const p = queue.then(job, job);
  queue = p.catch(() => undefined);
  return p;
}

/** Yield to the browser (one macrotask). */
export const yieldToBrowser = () => new Promise<void>((r) => window.setTimeout(r, 0));

export type JobFn<T> = (report: (done: number, total: number) => void, cancelled: () => boolean) => Promise<T | null>;

export type JobStatus = 'idle' | 'waiting' | 'running' | 'done' | 'error';

export interface JobState<T> {
  status: JobStatus;
  done: number;
  total: number;
  result: T | null;
  error?: string;
  start: () => void;
  cancel: () => void;
}

/**
 * Run `fn` for `key` (null = inputs not ready). With `auto`, a new key starts automatically;
 * otherwise `start()` runs it. A cached key resolves instantly.
 */
export function useJob<T>(key: string | null, fn: JobFn<T>, auto: boolean): JobState<T> {
  const [st, setSt] = useState<{ key: string | null; status: JobStatus; done: number; total: number; result: T | null; error?: string }>(() => ({
    key,
    status: key && cache.has(key) ? 'done' : 'idle',
    done: 0,
    total: 0,
    result: key ? (cacheGet<T>(key) ?? null) : null,
  }));
  const token = useRef(0);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const keyRef = useRef(key);
  keyRef.current = key;

  const start = useCallback(() => {
    const k = keyRef.current;
    if (!k) return;
    const my = ++token.current;
    const hit = cacheGet<T>(k);
    if (hit !== undefined) {
      setSt({ key: k, status: 'done', done: 1, total: 1, result: hit });
      return;
    }
    setSt({ key: k, status: 'waiting', done: 0, total: 0, result: null });
    const job = fnRef.current;
    enqueue(async () => {
      if (token.current !== my) return null;
      return job(
        (done, total) => {
          if (token.current === my) setSt((s) => ({ ...s, key: k, status: 'running', done, total }));
        },
        () => token.current !== my,
      );
    }).then(
      (res) => {
        if (res === null || res === undefined) return;
        cacheSet(k, res);
        if (token.current === my) setSt({ key: k, status: 'done', done: 1, total: 1, result: res });
      },
      (e: unknown) => {
        if (token.current === my) setSt((s) => ({ ...s, key: k, status: 'error', error: e instanceof Error ? e.message : String(e) }));
      },
    );
  }, []);

  const cancel = useCallback(() => {
    token.current++;
    setSt((s) => ({ ...s, status: 'idle', done: 0, total: 0 }));
  }, []);

  useEffect(() => {
    if (!key) return;
    const hit = cacheGet<T>(key);
    if (hit !== undefined) {
      token.current++;
      setSt({ key, status: 'done', done: 1, total: 1, result: hit });
    } else if (auto) {
      start();
    } else {
      token.current++;
      setSt({ key, status: 'idle', done: 0, total: 0, result: null });
    }
  }, [key, auto, start]);

  useEffect(
    () => () => {
      token.current++;
    },
    [],
  );

  const cur = st.key === key ? st : { key, status: 'idle' as JobStatus, done: 0, total: 0, result: null };
  return { ...cur, start, cancel };
}

/** True once the element has scrolled near the viewport (stays true). */
export function useNearView<T extends Element>(): [React.RefObject<T | null>, boolean] {
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
      { rootMargin: '300px 0px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [seen]);
  return [ref, seen];
}
