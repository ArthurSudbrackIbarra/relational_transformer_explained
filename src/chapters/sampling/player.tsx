/**
 * Step-through state for the BFS replay, shared by the main figure, the Algorithm 1 panel (line
 * highlighting) and the "four rules" cards ("show me" buttons jump the replay to a decision).
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from 'react';
import { useStepPlayer, type StepPlayerState } from '@/components';
import { useExplore } from './explore';
import {
  buildLayout,
  buildSteps,
  parseIterations,
  replay,
  stepAtCursor,
  type Granularity,
  type Iteration,
  type Layout,
  type ReplayState,
  type Step,
} from './trace';

export interface PlayerState {
  granularity: Granularity;
  setGranularity: (g: Granularity) => void;
  iters: Iteration[];
  steps: Step[];
  step: Step;
  player: StepPlayerState;
  /** Number of events applied for the current step (= step.end). */
  cursor: number;
  /** Frontier state after the current step, and before it (for the draw animation). */
  state: ReplayState;
  before: ReplayState;
  layout: Layout;
  /**
   * Jump to the first decision-level step satisfying `pred` and scroll the figure into view. `origin` (the
   * element that asked, e.g. a rule card) is offered back as a "Back to …" link in the figure.
   */
  jumpTo: (pred: (s: Step, it: Iteration | undefined) => boolean, origin?: JumpOrigin) => boolean;
  /** Where the last "Show me" came from (cleared once the reader goes back). */
  origin: JumpOrigin | null;
  clearOrigin: () => void;
  /** Move to the next step of the given kind (after the current one); false when there is none. */
  skip: (what: SkipKind) => boolean;
  /** Whether `skip(what)` would find a step. */
  canSkip: (what: SkipKind) => boolean;
  /** Jump to the step at which `node` was visited (or first reached). */
  jumpToNode: (node: number) => void;
  figureRef: RefObject<HTMLDivElement | null>;
}

export interface JumpOrigin {
  label: string;
  el: HTMLElement;
}

export type SkipKind = 'row' | 'draw' | 'width';

const Ctx = createContext<PlayerState | null>(null);

/** The parts that change only with the context or the granularity (not on every step). */
export type TraceState = Pick<PlayerState, 'granularity' | 'iters' | 'steps' | 'layout' | 'jumpTo' | 'jumpToNode' | 'figureRef'>;
const TraceCtx = createContext<TraceState | null>(null);

export function usePlayer(): PlayerState {
  const v = useContext(Ctx);
  if (!v) throw new Error('usePlayer must be used inside <PlayerProvider>');
  return v;
}

/** Use this instead of usePlayer when a component does not depend on the current step. */
export function useTrace(): TraceState {
  const v = useContext(TraceCtx);
  if (!v) throw new Error('useTrace must be used inside <PlayerProvider>');
  return v;
}

export function PlayerProvider({ children }: { children: ReactNode }) {
  const { ctx } = useExplore();
  const [granularity, setGranularityRaw] = useState<Granularity>('decisions');
  const iters = useMemo(() => parseIterations(ctx.events), [ctx]);
  const steps = useMemo(() => buildSteps(ctx.events, iters, granularity), [ctx, iters, granularity]);
  const layout = useMemo(() => buildLayout(ctx), [ctx]);
  // Open on the finished context (the reader sees the result first, then replays it).
  // one decision per 0.9 s at 1×; a whole row is a longer read, but rows mode is for a quick pass
  const player = useStepPlayer(steps.length, { intervalMs: granularity === 'rows' ? 650 : 900, initial: steps.length - 1 });
  const [origin, setOrigin] = useState<JumpOrigin | null>(null);
  const clearOrigin = useCallback(() => setOrigin(null), []);
  const figureRef = useRef<HTMLDivElement | null>(null);

  // A new context (seed, L, w, RNG seed …) shows its finished state; a granularity switch keeps the
  // current position in the trace.
  const pending = useRef<{ ctx: typeof ctx; cursor: number | null } | null>(null);
  const prevCtx = useRef(ctx);
  const prevSteps = useRef(steps);
  const { setIndex, setPlaying } = player;
  useEffect(() => {
    if (prevCtx.current !== ctx) {
      prevCtx.current = ctx;
      prevSteps.current = steps;
      setPlaying(false);
      setIndex(steps.length - 1);
      return;
    }
    if (prevSteps.current !== steps) {
      const cursor = pending.current?.cursor ?? null;
      prevSteps.current = steps;
      pending.current = null;
      if (cursor !== null) setIndex(stepAtCursor(steps, cursor));
    }
  }, [ctx, steps, setIndex, setPlaying]);

  const index = Math.min(player.index, steps.length - 1);
  const step = steps[index] ?? steps[0];
  const cursor = step.end;

  const setGranularity = useCallback(
    (g: Granularity) => {
      pending.current = { ctx, cursor };
      setGranularityRaw(g);
    },
    [ctx, cursor],
  );

  const state = useMemo(() => replay(ctx, cursor), [ctx, cursor]);
  const before = useMemo(() => (step.start === cursor ? state : replay(ctx, step.start)), [ctx, step.start, cursor, state]);

  const scrollToFigure = useCallback(() => {
    const el = figureRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    // the ref sits on the player (sticky on wide screens): bring it just under the top bar
    if (r.top < 0 || r.top > window.innerHeight * 0.35) {
      window.scrollTo({ top: window.scrollY + r.top - 72, behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
    }
  }, []);

  const jumpTo = useCallback(
    (pred: (s: Step, it: Iteration | undefined) => boolean, from?: JumpOrigin) => {
      const decisionSteps = granularity === 'decisions' ? steps : buildSteps(ctx.events, iters, 'decisions');
      const i = decisionSteps.findIndex((s) => pred(s, s.iter >= 0 ? iters[s.iter] : undefined));
      if (i < 0) return false;
      setPlaying(false);
      if (granularity !== 'decisions') {
        pending.current = { ctx, cursor: decisionSteps[i].end };
        setGranularityRaw('decisions');
      } else setIndex(i);
      setOrigin(from ?? null);
      scrollToFigure();
      return true;
    },
    [granularity, steps, ctx, iters, setIndex, setPlaying, scrollToFigure],
  );

  // "next row / next bucket draw / next width cut": the trace marks these step kinds
  const matches = useCallback(
    (s: Step, what: SkipKind) => {
      if (s.iter < 0) return false;
      const it = iters[s.iter];
      const startsIter = s.kind === 'select' || s.kind === 'row';
      if (what === 'row') return startsIter;
      if (what === 'draw') return startsIter && it.from === 'bucket';
      // width: a step where the bound actually cut (more than w DB children passed)
      if (s.kind !== 'width' && s.kind !== 'row') return false;
      for (let k = it.filterEnd; k < it.widthEnd; k++) if (ctx.events[k].kind === 'db-children-subsampled') return true;
      return false;
    },
    [iters, ctx],
  );
  const nextIndex = useCallback(
    (what: SkipKind) => {
      const cur = Math.min(player.index, steps.length - 1);
      for (let i = cur + 1; i < steps.length; i++) if (matches(steps[i], what)) return i;
      return -1;
    },
    [player.index, steps, matches],
  );
  const skip = useCallback(
    (what: SkipKind) => {
      const i = nextIndex(what);
      if (i < 0) return false;
      setPlaying(false);
      setIndex(i);
      return true;
    },
    [nextIndex, setIndex, setPlaying],
  );
  const canSkip = useCallback((what: SkipKind) => nextIndex(what) >= 0, [nextIndex]);

  const jumpToNode = useCallback(
    (node: number) => {
      const it = iters.find((x) => x.node === node && !x.skipped);
      if (it) {
        const i = steps.findIndex((s) => s.iter === it.iter && (s.kind === 'visit' || s.kind === 'row'));
        if (i >= 0) {
          setPlaying(false);
          setIndex(i);
          return;
        }
      }
      // not visited: jump to the moment it entered the frontier
      const ln = layout.nodes.get(node);
      if (ln && ln.firstEvent >= 0) {
        const j = steps.findIndex((s) => s.start <= ln.firstEvent && ln.firstEvent < s.end);
        if (j >= 0) {
          setPlaying(false);
          setIndex(j);
        }
      }
    },
    [iters, steps, layout, setIndex, setPlaying],
  );

  const value = useMemo<PlayerState>(
    () => ({
      granularity,
      setGranularity,
      iters,
      steps,
      step,
      player: { ...player, index },
      cursor,
      state,
      before,
      layout,
      jumpTo,
      jumpToNode,
      figureRef,
      origin,
      clearOrigin,
      skip,
      canSkip,
    }),
    [granularity, setGranularity, iters, steps, step, player, index, cursor, state, before, layout, jumpTo, jumpToNode, origin, clearOrigin, skip, canSkip],
  );
  const trace = useMemo<TraceState>(
    () => ({ granularity, iters, steps, layout, jumpTo, jumpToNode, figureRef }),
    [granularity, iters, steps, layout, jumpTo, jumpToNode],
  );
  return (
    <TraceCtx.Provider value={trace}>
      <Ctx.Provider value={value}>{children}</Ctx.Provider>
    </TraceCtx.Provider>
  );
}
