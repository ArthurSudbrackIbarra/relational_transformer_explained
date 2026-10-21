/** The narration of the current replay step (memoised), shared by the figure and the Algorithm 1 panel. */
import { useMemo } from 'react';
import { useExplore } from './explore';
import { narrate, type Narration } from './narrate';
import { usePlayer } from './player';

export function useNarration(): Narration {
  const { ctx, pdb, db, task } = useExplore();
  const { step, iters, before, state } = usePlayer();
  return useMemo(() => narrate({ step, iters, ctx, pdb, db, task, before, state }), [step, iters, ctx, pdb, db, task, before, state]);
}
