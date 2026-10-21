/**
 * Small shared pieces of the playground figures: job progress, a diverging Δ-bar list and the
 * sweep inputs derived from the live state.
 */
import { useRef, type ReactNode } from 'react';
import { Button, cx } from '@/components';
import type { TaskDef } from '@/engine/types';
import { usePlayLive } from './context';
import type { JobState } from './jobs';
import type { SweepBase } from './sweeps';
import { fmtDelta, fmtDisplay, oid, readout } from './usePlayground';

/** Sweep inputs: the course's live seed, context settings and checkpoint (full model). */
export function useSweepBase(): { base: SweepBase | null; key: string | null } {
  const { pdb, task, seedNode, globalL, w, rngSeed, baseAssets } = usePlayLive();
  if (baseAssets.status !== 'ready' || !baseAssets.store || !baseAssets.weights) return { base: null, key: null };
  const base: SweepBase = { pdb, task, seedNode, L: globalL, w, rngSeed, store: baseAssets.store, weights: baseAssets.weights };
  return { base, key: [oid(pdb), task.id, seedNode, globalL, w, rngSeed, oid(baseAssets.store), oid(baseAssets.weights)].join('|') };
}

export function JobProgress<T>({ job, label, onStart, startLabel }: { job: JobState<T>; label: string; onStart?: () => void; startLabel?: string }) {
  if (job.status === 'done') return null;
  if (job.status === 'error') return <p className="text-[15px] text-[var(--critical)]">Could not run: {job.error}</p>;
  if (job.status === 'idle')
    return onStart ? (
      <Button variant="primary" size="sm" onClick={onStart}>
        {startLabel ?? 'Run'}
      </Button>
    ) : (
      <p className="text-[15px] text-muted">Waiting…</p>
    );
  const frac = job.total ? job.done / job.total : 0;
  return (
    <div className="flex items-center gap-3" role="status" aria-live="polite">
      <div className="h-2 w-40 overflow-hidden rounded-full bg-surface-3">
        <div className="h-full rounded-full bg-accent transition-[width] duration-150" style={{ width: `${Math.round(frac * 100)}%` }} />
      </div>
      <span className="num text-[15px] text-ink-2">
        {label} {job.total ? `${job.done} / ${job.total}` : '…'}
      </span>
      <button type="button" className="text-[15px] font-medium text-accent-text hover:underline" onClick={job.cancel}>
        Cancel
      </button>
    </div>
  );
}

export interface DeltaRow {
  key: string;
  label: ReactNode;
  /** Decoder output (logit or z) of this variant. */
  output: number;
  emphasis?: boolean;
}

/** Rows of "variant · Δ bar · value", Δ = display(variant) − display(base), centred on zero. */
export function DeltaRows({ rows, baseOutput, task, caption }: { rows: DeltaRow[]; baseOutput: number; task: TaskDef; caption: string }) {
  const { stats } = usePlayLive();
  const base = readout(baseOutput, task, stats).display;
  const ds = rows.map((r) => readout(r.output, task, stats).display - base);
  const m = Math.max(1e-6, ...ds.map((d) => Math.abs(d)));
  return (
    <table className="w-full border-collapse text-[15px]">
      <caption className="sr-only">{caption}</caption>
      <thead>
        <tr className="border-b border-grid text-left text-ink-2">
          <th scope="col" className="py-1.5 pr-3 font-semibold">
            Variant
          </th>
          <th scope="col" className="w-[38%] py-1.5 pr-3 text-center font-semibold">
            Change vs full context
          </th>
          <th scope="col" className="py-1.5 text-right font-semibold">
            Prediction
          </th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => {
          const d = ds[i];
          const w = (Math.abs(d) / m) * 50;
          return (
            <tr key={r.key} className={cx('border-b border-grid/60', r.emphasis && 'bg-surface-2')}>
              <td className="py-1.5 pr-3 align-middle text-ink">{r.label}</td>
              <td className="py-1.5 pr-3 align-middle">
                <div className="flex items-center gap-2">
                  <div className="relative h-3.5 flex-1 rounded-sm bg-surface-2">
                    <span aria-hidden="true" className="absolute -inset-y-0.5 left-1/2 w-px bg-axis" />
                    <span
                      className="absolute inset-y-0"
                      style={{
                        left: d < 0 ? `${50 - w}%` : '50%',
                        width: `${w}%`,
                        background: d < 0 ? 'var(--div-neg)' : 'var(--div-pos)',
                        borderRadius: d < 0 ? '3px 0 0 3px' : '0 3px 3px 0',
                      }}
                    />
                  </div>
                  <span className="num w-[78px] shrink-0 text-right text-[14.5px] text-ink-2">{fmtDelta(d, task)}</span>
                </div>
              </td>
              <td className="num py-1.5 text-right align-middle text-ink">{fmtDisplay(readout(r.output, task, stats).display, task)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export interface RadioItem<T extends string> {
  value: T;
  label: ReactNode;
  sub?: ReactNode;
  title?: string;
}

/**
 * A vertical list of radio cards (role="radiogroup") with roving tab index and arrow-key
 * navigation: Tab enters the group at the checked item, arrows move and select.
 */
export function RadioList<T extends string>({ label, items, value, onChange }: { label: string; items: RadioItem<T>[]; value: T; onChange: (v: T) => void }) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const cur = Math.max(0, items.findIndex((it) => it.value === value));
  const move = (d: number) => {
    const i = (cur + d + items.length) % items.length;
    onChange(items[i].value);
    refs.current[i]?.focus();
  };
  return (
    <div
      className="flex flex-col gap-1"
      role="radiogroup"
      aria-label={label}
      onKeyDown={(e) => {
        if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
          e.preventDefault();
          move(1);
        } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
          e.preventDefault();
          move(-1);
        }
      }}
    >
      {items.map((it, i) => {
        const on = it.value === value;
        return (
          <button
            key={it.value}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={on}
            tabIndex={i === cur ? 0 : -1}
            title={it.title}
            onClick={() => onChange(it.value)}
            className={cx(
              'flex min-w-0 items-start gap-2.5 rounded-lg border px-2.5 py-1.5 text-left transition-colors',
              on ? 'border-accent bg-accent-soft' : 'border-transparent hover:bg-surface-2',
            )}
          >
            <span aria-hidden="true" className={cx('mt-[3px] grid h-4 w-4 shrink-0 place-items-center rounded-full border-2', on ? 'border-accent' : 'border-border-strong')}>
              {on && <span className="h-1.5 w-1.5 rounded-full bg-accent" />}
            </span>
            <span className="min-w-0">
              <span className={cx('block text-[15.5px] leading-tight', on ? 'font-semibold text-ink' : 'text-ink')}>{it.label}</span>
              {it.sub && <span className="block text-[14px] leading-snug text-muted">{it.sub}</span>}
            </span>
          </button>
        );
      })}
    </div>
  );
}
