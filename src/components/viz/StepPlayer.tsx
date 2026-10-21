/**
 * StepPlayer: play / pause, step back / forward, speed, reset and a scrubber
 * for any algorithm that runs in steps (BUILD_PLAN §5 "Step-through controls
 * everywhere an algorithm runs").
 *
 * Use with the `useStepPlayer` hook, which owns the timer:
 *
 *   const player = useStepPlayer(events.length);
 *   …render state for player.index…
 *   <StepPlayer {...player} describe={(i) => eventText(events[i])} />
 *
 * Keyboard (when focus is inside the player): Space = play/pause, ← / → = step,
 * Home / End = first / last.
 */
import { Pause, Play, RotateCcw, StepBack, StepForward } from 'lucide-react';
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { IconButton } from '../ui/controls';
import { cx } from '../ui/cx';

export const SPEEDS = [0.5, 1, 2, 4] as const;

export interface StepPlayerState {
  /** Current step 0..count-1. */
  index: number;
  count: number;
  playing: boolean;
  speed: number;
  setIndex: (i: number) => void;
  setPlaying: (p: boolean) => void;
  setSpeed: (s: number) => void;
  step: (delta: number) => void;
  reset: () => void;
}

export interface UseStepPlayerOptions {
  /** Base interval between steps at 1× (ms). Default 700. */
  intervalMs?: number;
  /** Start index. Default 0. */
  initial?: number;
  /** Start playing immediately. Default false. */
  autoplay?: boolean;
  /** Loop back to 0 at the end. Default false (stops at the last step). */
  loop?: boolean;
}

/** State + timer for a StepPlayer. Clamps the index when `count` changes. */
export function useStepPlayer(count: number, opts: UseStepPlayerOptions = {}): StepPlayerState {
  const { intervalMs = 700, initial = 0, autoplay = false, loop = false } = opts;
  const [index, setIndexRaw] = useState(initial);
  const [playing, setPlaying] = useState(autoplay);
  const [speed, setSpeed] = useState(1);
  const last = Math.max(0, count - 1);

  const setIndex = useCallback((i: number) => setIndexRaw(Math.max(0, Math.min(last, Math.round(i)))), [last]);

  useEffect(() => {
    if (index > last) setIndexRaw(last);
  }, [index, last]);

  useEffect(() => {
    if (!playing || count <= 1) return;
    const id = window.setInterval(() => {
      setIndexRaw((i) => {
        if (i >= last) {
          if (loop) return 0;
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, intervalMs / speed);
    return () => window.clearInterval(id);
  }, [playing, speed, intervalMs, last, loop, count]);

  const step = useCallback(
    (d: number) => {
      setPlaying(false);
      setIndexRaw((i) => Math.max(0, Math.min(last, i + d)));
    },
    [last],
  );
  const reset = useCallback(() => {
    setPlaying(false);
    setIndexRaw(0);
  }, []);
  const setPlayingSafe = useCallback(
    (p: boolean) => {
      // pressing play at the end restarts from the beginning
      if (p) setIndexRaw((i) => (i >= last ? 0 : i));
      setPlaying(p);
    },
    [last],
  );

  return { index: Math.min(index, last), count, playing, speed, setIndex, setPlaying: setPlayingSafe, setSpeed, step, reset };
}

export interface StepPlayerProps extends StepPlayerState {
  /** Text for the current step (shown under the controls, announced politely). */
  describe?: (index: number) => ReactNode;
  /** Label of the unit, e.g. "event" → "event 3 / 41". Default "step". */
  unit?: string;
  className?: string;
  /** Accessible name of the player. */
  label?: string;
}

export function StepPlayer({
  index,
  count,
  playing,
  speed,
  setIndex,
  setPlaying,
  setSpeed,
  step,
  reset,
  describe,
  unit = 'step',
  className,
  label = 'Step-through controls',
}: StepPlayerProps) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const atEnd = index >= count - 1;
  return (
    <div
      ref={rootRef}
      role="group"
      aria-label={label}
      className={cx('rounded-xl border border-border bg-surface px-3 py-2.5', className)}
      onKeyDown={(e) => {
        const tag = (e.target as HTMLElement).tagName;
        if (tag === 'SELECT') return;
        if (e.key === ' ' && tag !== 'BUTTON') {
          e.preventDefault();
          setPlaying(!playing);
        } else if (e.key === 'ArrowRight' && tag !== 'INPUT') {
          e.preventDefault();
          step(1);
        } else if (e.key === 'ArrowLeft' && tag !== 'INPUT') {
          e.preventDefault();
          step(-1);
        } else if (e.key === 'Home' && tag !== 'INPUT') {
          e.preventDefault();
          setIndex(0);
        } else if (e.key === 'End' && tag !== 'INPUT') {
          e.preventDefault();
          setIndex(count - 1);
        }
      }}
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-2">
        <div className="flex items-center gap-0.5">
          <IconButton label="Reset to the first step" size="sm" onClick={reset} disabled={count === 0}>
            <RotateCcw size={15} />
          </IconButton>
          <IconButton label="Step back" size="sm" onClick={() => step(-1)} disabled={index <= 0}>
            <StepBack size={16} />
          </IconButton>
          <button
            type="button"
            onClick={() => setPlaying(!playing)}
            disabled={count <= 1}
            aria-label={playing ? 'Pause' : atEnd ? 'Replay from the start' : 'Play'}
            className="mx-0.5 inline-flex h-8 w-8 items-center justify-center rounded-full bg-accent text-accent-ink transition hover:brightness-110 disabled:opacity-40"
          >
            {playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" className="translate-x-[1px]" />}
          </button>
          <IconButton label="Step forward" size="sm" onClick={() => step(1)} disabled={atEnd}>
            <StepForward size={16} />
          </IconButton>
        </div>
        <input
          type="range"
          min={0}
          max={Math.max(0, count - 1)}
          value={index}
          onChange={(e) => {
            setPlaying(false);
            setIndex(Number(e.target.value));
          }}
          aria-label={`${unit} position`}
          aria-valuetext={`${unit} ${index + 1} of ${count}`}
          className="h-5 min-w-[120px] flex-1 cursor-pointer"
        />
        <span
          className="num shrink-0 text-right text-[15px] whitespace-nowrap text-ink-2"
          style={{ minWidth: `${unit.length + 2 * String(Math.max(1, count)).length + 4}ch` }}
        >
          {unit} {count === 0 ? 0 : index + 1} / {count}
        </span>
        <label className="flex items-center gap-1 text-[15px] text-ink-2">
          <span className="sr-only">Speed</span>
          <select
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="h-8 rounded-md border border-border bg-surface px-1.5 text-[15px] text-ink"
          >
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </label>
      </div>
      {describe && (
        <div className="mt-2 min-h-[1.5em] border-t border-grid pt-2 text-[17px] leading-snug text-ink" aria-live="polite">
          {count > 0 ? describe(index) : null}
        </div>
      )}
    </div>
  );
}
