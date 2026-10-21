/**
 * The playground's own knobs. Everything that defines WHICH context is shown (database, task, split,
 * seed row, L, w, RNG seed, checkpoint) lives in the global store, so this chapter shows the same
 * context as every other chapter. The ablations below are local to this page: they never change what
 * the other chapters show. The one exception is the attention-layer switch, which uses the global,
 * session-only `disabledLayers` (the top bar and the attention/block chapters flag it while it is on).
 */
import { create } from 'zustand';
import type { CheckpointName } from '@/engine/pipeline';
import type { RawValue } from '@/engine/types';
import type { RemoveWhat } from './ablate';

export interface HistoryEntry {
  id: number;
  /** "<taskId>|<seed>" the prediction belongs to. */
  scope: string;
  /** Signature of the full ablation state (dedupes consecutive identical runs). */
  sig: string;
  /** Short chips describing what was changed. */
  chips: string[];
  /** Probability (binary) or value in original units (regression). */
  value: number;
}

export interface PlayState {
  removeSelf: boolean;
  removeOther: boolean;
  removeWhat: RemoveWhat;
  shuffle: boolean;
  shuffleSeed: number;
  /** null = use the global context length. */
  L: number | null;
  /** null = use the global (top bar) checkpoint. */
  ckpt: CheckpointName | null;
  /** "<taskId>|<seed>" the edits were made for; edits are ignored for any other seed. */
  editScope: string;
  /** nodeIdx → new raw label. */
  edits: Record<number, RawValue>;
  history: HistoryEntry[];

  set: (p: Partial<Pick<PlayState, 'removeSelf' | 'removeOther' | 'removeWhat' | 'shuffle' | 'L' | 'ckpt'>>) => void;
  reshuffle: () => void;
  setEdit: (scope: string, node: number, value: RawValue | undefined) => void;
  clearEdits: () => void;
  pushHistory: (e: Omit<HistoryEntry, 'id'>) => void;
  clearHistory: () => void;
  resetLocal: () => void;
}

const LOCAL_DEFAULTS = {
  removeSelf: false,
  removeOther: false,
  removeWhat: 'cells' as RemoveWhat,
  shuffle: false,
  shuffleSeed: 1,
  L: null,
  ckpt: null,
  editScope: '',
  edits: {},
};

const HISTORY_MAX = 8;
let nextId = 1;

export const usePlay = create<PlayState>()((set) => ({
  ...LOCAL_DEFAULTS,
  history: [],
  set: (p) => set(p),
  reshuffle: () => set((s) => ({ shuffle: true, shuffleSeed: s.shuffleSeed + 1 })),
  setEdit: (scope, node, value) =>
    set((s) => {
      const base = s.editScope === scope ? { ...s.edits } : {};
      if (value === undefined) delete base[node];
      else base[node] = value;
      return { editScope: scope, edits: base };
    }),
  clearEdits: () => set({ edits: {}, editScope: '' }),
  pushHistory: (e) =>
    set((s) => {
      const last = s.history[s.history.length - 1];
      if (last && last.sig === e.sig && last.scope === e.scope) return s;
      // a new seed or task starts a fresh history (the baseline it is compared with changes)
      const kept = last && last.scope !== e.scope ? [] : s.history;
      return { history: [...kept, { ...e, id: nextId++ }].slice(-HISTORY_MAX) };
    }),
  clearHistory: () => set({ history: [] }),
  resetLocal: () => set({ ...LOCAL_DEFAULTS }),
}));
