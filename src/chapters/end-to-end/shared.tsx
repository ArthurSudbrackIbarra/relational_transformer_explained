/**
 * Small pieces shared by the stage figures of the "start to end" page: the focus-token line (which
 * token's row is outlined and explained), the loading note, and a media-query hook.
 */
import { Crosshair, Undo2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { Button, Select, TableSwatch, cx, tokenValueText } from '@/components';
import { SEM_TYPE_NAMES } from '@/engine/types';
import { rowName } from '../attention/utils';
import { useE2E } from './live';

export function useMedia(query: string): boolean {
  const [m, setM] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const f = () => setM(mq.matches);
    f();
    mq.addEventListener('change', f);
    return () => mq.removeEventListener('change', f);
  }, [query]);
  return m;
}

export function LoadingNote() {
  const { assets } = useE2E();
  return (
    <p className="mt-5 text-[16.5px] text-ink-2">
      {assets.status === 'error' ? `The forward pass is unavailable: ${assets.error}` : 'Loading the trained weights and running the forward pass…'}
    </p>
  );
}

/** "Following: customers[35].age = 34" with a token picker and a way back to the masked target. */
export function FocusLine({ className }: { className?: string }) {
  const live = useE2E();
  const { ctx, focus, target, setFocus, tokenLabels, numReal, slots } = live;
  const t = ctx.tokens[focus];
  const options = useMemo(
    () =>
      ctx.tokens.slice(0, numReal).map((x) => ({
        pos: x.pos,
        text: `${x.pos} · ${rowName(x)}.${x.colName} = ${x.isTarget ? '? (masked target)' : tokenValueText(x)}`,
      })),
    [ctx, numReal],
  );
  if (!t) return null;
  return (
    <div className={cx('flex flex-wrap items-end gap-x-3 gap-y-2 rounded-xl border border-border bg-surface px-3 py-2.5', className)}>
      <div className="flex min-w-0 flex-1 items-center gap-2 text-[16px] text-ink-2">
        <Crosshair aria-hidden="true" size={16} className="shrink-0 text-accent-text" />
        <span className="min-w-0">
          Following{' '}
          <span className="inline-flex items-center gap-1.5 align-middle">
            <TableSwatch slot={slots[t.tableName] ?? 0} />
            <span className="font-semibold text-ink">{tokenLabels[focus]}</span>
          </span>{' '}
          ({rowName(t)}.{t.colName}, {SEM_TYPE_NAMES[t.sem]}
          {t.isTarget ? ', the masked target' : `, value ${tokenValueText(t)}`}, position {focus})
        </span>
      </div>
      <Select label="Token to follow" hideLabel size="sm" value={String(focus)} onChange={(e) => setFocus(Number(e.target.value))} className="w-full max-w-[340px]">
        {options.map((o) => (
          <option key={o.pos} value={o.pos}>
            {o.text}
          </option>
        ))}
      </Select>
      {focus !== target && target >= 0 && (
        <Button size="sm" variant="ghost" icon={<Undo2 size={14} />} onClick={() => setFocus(null)}>
          Back to the target
        </Button>
      )}
    </div>
  );
}

/**
 * Hover state that belongs to one view of a figure (`view`: the player step, the head, the context…).
 * A hover made in another view is dropped in the same render, not in an effect after it: a player
 * that advances while the mouse rests on an entry (autoplay, keyboard) must never read the new
 * step's matrices at the old step's indices, which may not exist there.
 */
export function useViewHover<T>(view: readonly unknown[]): [T | null, (h: T | null) => void] {
  const [state, setState] = useState<{ view: readonly unknown[]; h: T } | null>(null);
  const setHover = useCallback((h: T | null) => setState(h === null ? null : { view, h }), view);
  const current = state !== null && state.view.length === view.length && state.view.every((v, i) => Object.is(v, view[i]));
  return [current ? state.h : null, setHover];
}

/** Visual on the left, the arithmetic on the right (stacked below lg). */
export function OpLayout({ visual, math }: { visual: ReactNode; math: ReactNode }) {
  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_330px]">
      <div className="min-w-0">{visual}</div>
      <aside
        className="min-w-0 self-start rounded-xl bg-surface-2 px-4 py-3.5 text-[15.5px] leading-snug [&_code]:!break-words [&_code]:!whitespace-normal [&_sub]:text-[88%] [&_sup]:text-[88%]"
        aria-live="polite"
      >
        {math}
      </aside>
    </div>
  );
}

/** Width available to the visual column of an OpLayout inside a container `width` px wide. */
export function useVisualWidth(width: number): number {
  const lg = useMedia('(min-width: 1024px)');
  return Math.max(240, (width || 640) - (lg ? 350 : 0));
}
