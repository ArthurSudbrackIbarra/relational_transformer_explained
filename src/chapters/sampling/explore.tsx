/**
 * Chapter-local sampler settings ("explore" state).
 *
 * The chapter starts from the course-wide simulation (store + hooks), so it shows the same seed and
 * context as every other chapter. The reader can then try another seed, split, L, w or RNG seed here
 * without silently changing the rest of the course: local choices are overrides on top of the global
 * settings. "Use in every chapter" writes them to the store; "Back to course settings" drops them.
 *
 * The context itself is always computed by the engine pipeline (`getContext`, cached per params).
 * L and w feed the sampler through `useDeferredValue`, so a slider follows the pointer at once and the
 * context, graph and tables catch up in a background render.
 */
import { createContext, useCallback, useContext, useDeferredValue, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { getContext, getSeeds, makeSampleParams, pickDefaultSeed, useDatabase, usePreprocessed, useSeedNode, useTableSlots, useTask } from '@/state/hooks';
import { useSim } from '@/state/store';
import type { PreprocessedDB, RawDatabase, SampledContext, SampleParams, Split, TaskDef } from '@/engine/types';

export interface Overrides {
  split?: Split;
  seedNode?: number;
  L?: number;
  w?: number;
  rngSeed?: number;
}

export interface ExploreState {
  db: RawDatabase;
  pdb: PreprocessedDB;
  task: TaskDef;
  slots: Record<string, number>;
  split: Split;
  seeds: number[];
  seedNode: number;
  L: number;
  w: number;
  rngSeed: number;
  params: SampleParams;
  ctx: SampledContext;
  /** True when any effective setting differs from the course-wide simulation. */
  isLocal: boolean;
  /** Which settings differ from the course-wide simulation. */
  localKeys: (keyof Overrides)[];
  global: { split: Split; seedNode: number; L: number; w: number; rngSeed: number };
  set: (o: Overrides) => void;
  reset: () => void;
  applyGlobally: () => void;
}

const Ctx = createContext<ExploreState | null>(null);

export function useExplore(): ExploreState {
  const v = useContext(Ctx);
  if (!v) throw new Error('useExplore must be used inside <ExploreProvider>');
  return v;
}

export function ExploreProvider({ children }: { children: ReactNode }) {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const task = useTask();
  const slots = useTableSlots();
  const globalSeed = useSeedNode();
  const g = useSim(useShallow((s) => ({ split: s.split, L: s.L, w: s.w, rngSeed: s.rngSeed })));
  const actions = useSim(
    useShallow((s) => ({ setSplit: s.setSplit, setSeedNode: s.setSeedNode, setL: s.setL, setW: s.setW, setRngSeed: s.setRngSeed })),
  );
  const [o, setO] = useState<Overrides>({});

  // A different database or task invalidates local seeds: start again from the course settings.
  const taskRef = useRef(task.id);
  useEffect(() => {
    if (taskRef.current !== task.id) {
      taskRef.current = task.id;
      setO({});
    }
  }, [task.id]);

  // Autocomplete ignores the split (spec §2.4, data.py:77-81).
  const split: Split = task.kind === 'autocomplete' ? g.split : (o.split ?? g.split);
  const seeds = useMemo(() => getSeeds(pdb, task, split), [pdb, task, split]);
  const seedNode = useMemo(() => {
    if (o.seedNode !== undefined && seeds.includes(o.seedNode)) return o.seedNode;
    if (split === g.split && seeds.includes(globalSeed)) return globalSeed;
    return pickDefaultSeed(pdb, db, task, split, seeds);
  }, [o.seedNode, seeds, split, g.split, globalSeed, pdb, db, task]);
  const L = o.L ?? g.L;
  const w = o.w ?? g.w;
  const rngSeed = o.rngSeed ?? g.rngSeed;
  const dL = useDeferredValue(L);
  const dW = useDeferredValue(w);

  const params = useMemo(() => makeSampleParams(pdb, task, seedNode, dL, dW, rngSeed), [pdb, task, seedNode, dL, dW, rngSeed]);
  const ctx = useMemo(() => getContext(pdb, params), [pdb, params]);

  const localKeys = useMemo(() => {
    const k: (keyof Overrides)[] = [];
    if (split !== g.split) k.push('split');
    if (seedNode !== globalSeed) k.push('seedNode');
    if (L !== g.L) k.push('L');
    if (w !== g.w) k.push('w');
    if (rngSeed !== g.rngSeed) k.push('rngSeed');
    return k;
  }, [split, g, seedNode, globalSeed, L, w, rngSeed]);

  const set = useCallback((next: Overrides) => setO((cur) => ({ ...cur, ...next })), []);
  const reset = useCallback(() => setO({}), []);
  const applyGlobally = useCallback(() => {
    // setSplit resets the stored seed, so set the seed afterwards.
    if (split !== g.split) actions.setSplit(split);
    actions.setSeedNode(seedNode);
    actions.setL(L);
    actions.setW(w);
    actions.setRngSeed(rngSeed);
    setO({});
  }, [actions, split, g.split, seedNode, L, w, rngSeed]);

  const value = useMemo<ExploreState>(
    () => ({
      db,
      pdb,
      task,
      slots,
      split,
      seeds,
      seedNode,
      L,
      w,
      rngSeed,
      params,
      ctx,
      isLocal: localKeys.length > 0,
      localKeys,
      global: { split: g.split, seedNode: globalSeed, L: g.L, w: g.w, rngSeed: g.rngSeed },
      set,
      reset,
      applyGlobally,
    }),
    [db, pdb, task, slots, split, seeds, seedNode, L, w, rngSeed, params, ctx, localKeys, g, globalSeed, set, reset, applyGlobally],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
