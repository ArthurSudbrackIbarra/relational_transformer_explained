/**
 * Theme plumbing: apply the persisted choice to <html data-theme>, and expose
 * the *resolved* mode (light | dark) to components that draw on canvas.
 *
 * Choice semantics (BUILD_PLAN §5, §7):
 *  - `null`  → follow `prefers-color-scheme` (no data-theme attribute; CSS media query applies)
 *  - 'light' / 'dark' → stamped on <html data-theme>, wins over the OS both ways.
 */
import { useSyncExternalStore } from 'react';
import type { ThemeMode } from './palette';

export type ThemeChoice = ThemeMode | null;

const DARK_QUERY = '(prefers-color-scheme: dark)';

function systemPrefersDark(): boolean {
  try {
    return typeof window !== 'undefined' && !!window.matchMedia?.(DARK_QUERY).matches;
  } catch {
    return false;
  }
}

/** Stamp (or clear) the theme on <html>. Safe to call before React renders. */
export function applyTheme(choice: ThemeChoice): void {
  if (typeof document === 'undefined') return;
  const root = document.documentElement;
  if (choice === 'light' || choice === 'dark') root.setAttribute('data-theme', choice);
  else root.removeAttribute('data-theme');
}

/** Stamp (or clear) reading mode on <html> (index.css: Libron + paper tones). */
export function applyReading(on: boolean): void {
  if (typeof document === 'undefined') return;
  if (on) document.documentElement.setAttribute('data-reading', 'on');
  else document.documentElement.removeAttribute('data-reading');
}

/** The mode actually on screen right now. */
export function resolvedTheme(): ThemeMode {
  if (typeof document === 'undefined') return 'light';
  const attr = document.documentElement.getAttribute('data-theme');
  if (attr === 'light' || attr === 'dark') return attr;
  return systemPrefersDark() ? 'dark' : 'light';
}

function subscribe(onChange: () => void): () => void {
  if (typeof window === 'undefined') return () => {};
  const obs = new MutationObserver(onChange);
  obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  let mql: MediaQueryList | null = null;
  try {
    mql = window.matchMedia(DARK_QUERY);
    mql.addEventListener('change', onChange);
  } catch {
    mql = null;
  }
  return () => {
    obs.disconnect();
    mql?.removeEventListener('change', onChange);
  };
}

/** React hook: the resolved theme, re-rendering on toggle and on OS changes. */
export function useResolvedTheme(): ThemeMode {
  return useSyncExternalStore(subscribe, resolvedTheme, () => 'light');
}

function subscribeReading(onChange: () => void): () => void {
  if (typeof window === 'undefined') return () => {};
  const obs = new MutationObserver(onChange);
  obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-reading'] });
  return () => obs.disconnect();
}

/** React hook: true while reading mode is on. Canvas code reads TOKENS, so it re-draws on this. */
export function useReadingMode(): boolean {
  return useSyncExternalStore(
    subscribeReading,
    () => document.documentElement.getAttribute('data-reading') === 'on',
    () => false,
  );
}

/** React hook: true when the viewer asked for reduced motion. */
export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(
    (cb) => {
      try {
        const m = window.matchMedia('(prefers-reduced-motion: reduce)');
        m.addEventListener('change', cb);
        return () => m.removeEventListener('change', cb);
      } catch {
        return () => {};
      }
    },
    () => {
      try {
        return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      } catch {
        return false;
      }
    },
    () => false,
  );
}
