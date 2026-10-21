/**
 * Global simulation state (BUILD_PLAN §7).
 *
 * One zustand store, `useSim`, holds every knob that defines "the live
 * simulation": which database, task, split and seed row; the sampler's L, w and
 * RNG seed; which checkpoint; and what the reader is currently inspecting
 * (token, block, layer, head). Every chapter reads it, so the same context and
 * tokens flow through the whole course.
 *
 * This module is PURE state: it never imports the engine. Derived data
 * (database, preprocessed graph, context window, masks, forward pass) is
 * computed by the memoized hooks in ./hooks.ts.
 *
 * Persistence: the durable fields are saved to localStorage through a storage
 * adapter whose every access is wrapped in try/catch (private windows, blocked
 * storage, previews), so the app works identically without it.
 */
import { create } from 'zustand';
import { createJSONStorage, persist, type StateStorage } from 'zustand/middleware';
import type { AttnLayer, Split } from '@/engine/types';
import type { ThemeChoice } from '@/styles/theme';

// ─────────────────────────────────────────────────────────────────────────────
// Defaults
// ─────────────────────────────────────────────────────────────────────────────

/** Database ids (BUILD_PLAN §3). `shop` is the held-out target DB. */
export type DbName = 'shop' | 'school' | 'racing' | 'forum';
export const DB_NAMES: readonly DbName[] = ['shop', 'school', 'racing', 'forum'];

/**
 * Checkpoint choice. 'auto' = the zero-shot checkpoint for the current task
 * (pretrain-<task>, selected on that task's validation split, spec §9.1).
 * Files live in public/data/weights/<name>.{bin,json} (BUILD_PLAN §4.5).
 */
export type CkptChoice = 'auto' | 'init' | 'pretrain-churn' | 'pretrain-sales' | 'finetune-churn' | 'finetune-sales';
export const CKPT_CHOICES: readonly CkptChoice[] = [
  'auto',
  'init',
  'pretrain-churn',
  'pretrain-sales',
  'finetune-churn',
  'finetune-sales',
];

/** Mini-model sampler defaults (BUILD_PLAN §2: L = 128, w = 16, eval rngSeed = 0). Paper: L = 1024, w = 256. */
export const SIM_DEFAULTS = {
  dbName: 'shop' as DbName,
  taskId: 'shop/customer-churn',
  split: 'Val' as Split,
  /** null = "auto": hooks pick a Val seed with several past labels (see hooks.ts `pickDefaultSeed`). */
  seedNode: null as number | null,
  L: 128,
  w: 16,
  rngSeed: 0,
  ckpt: 'auto' as CkptChoice,
} as const;

/** Sane UI bounds for the sampler knobs. */
export const SIM_LIMITS = {
  L: { min: 8, max: 512, step: 8 },
  w: { min: 1, max: 64, step: 1 },
  rngSeed: { min: 0, max: 9999 },
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// State shape
// ─────────────────────────────────────────────────────────────────────────────

export interface SimParams {
  dbName: DbName;
  /** "<db>/<task>", e.g. "shop/customer-churn" (TaskDef.id). */
  taskId: string;
  /** Split of the seed row. Ignored for autocomplete tasks (spec §2.4). */
  split: Split;
  /** Global node index of the seed row, or null for the automatic default. */
  seedNode: number | null;
  /** Context length (number of cell tokens). */
  L: number;
  /** BFS width bound on DB children per visited row. */
  w: number;
  /** The `seed` term of fly.rs's `epoch + seed_node_idx + seed` (spec §4.2). Eval uses 0. */
  rngSeed: number;
  ckpt: CkptChoice;
}

export interface InspectState {
  /** Token position being inspected (null = the masked target). */
  selectedPos: number | null;
  /** Block index 0..nBlocks-1. */
  block: number;
  layer: AttnLayer;
  head: number;
  /** Inference-time ablation: skip these attention sublayers in every block. Not persisted. */
  disabledLayers: AttnLayer[];
}

export interface UiState {
  /** null = follow prefers-color-scheme. */
  theme: ThemeChoice;
  /** Chapter slugs the reader has opened (sidebar ticks). */
  visited: Record<string, true>;
}

export interface SimActions {
  /** Switch database; the caller passes the new DB's task (usually its first task). Resets the seed. */
  setDb: (dbName: DbName, taskId: string) => void;
  /** Switch task (also switches DB to the task's DB). Resets the seed. */
  setTask: (taskId: string) => void;
  setSplit: (split: Split) => void;
  setSeedNode: (seedNode: number | null) => void;
  setL: (L: number) => void;
  setW: (w: number) => void;
  setRngSeed: (rngSeed: number) => void;
  setCkpt: (ckpt: CkptChoice) => void;
  setSelectedPos: (pos: number | null) => void;
  setBlock: (block: number) => void;
  setLayer: (layer: AttnLayer) => void;
  setHead: (head: number) => void;
  toggleLayer: (layer: AttnLayer) => void;
  setDisabledLayers: (layers: AttnLayer[]) => void;
  setTheme: (theme: ThemeChoice) => void;
  markVisited: (slug: string) => void;
  clearVisited: () => void;
  /** Restore every simulation knob to SIM_DEFAULTS (keeps theme and visited). */
  resetSimulation: () => void;
}

export type SimState = SimParams & InspectState & UiState & SimActions;

// ─────────────────────────────────────────────────────────────────────────────
// Storage adapter: every access in try/catch
// ─────────────────────────────────────────────────────────────────────────────

const memoryFallback = new Map<string, string>();

/** localStorage when available, else an in-memory map. Never throws. */
export const safeStorage: StateStorage = {
  getItem: (name) => {
    try {
      const v = window.localStorage.getItem(name);
      if (v !== null) return v;
    } catch {
      /* blocked storage */
    }
    return memoryFallback.get(name) ?? null;
  },
  setItem: (name, value) => {
    memoryFallback.set(name, value);
    try {
      window.localStorage.setItem(name, value);
    } catch {
      /* quota / blocked: in-memory copy keeps the session working */
    }
  },
  removeItem: (name) => {
    memoryFallback.delete(name);
    try {
      window.localStorage.removeItem(name);
    } catch {
      /* ignore */
    }
  },
};

const clampInt = (v: number, lo: number, hi: number) =>
  Number.isFinite(v) ? Math.min(hi, Math.max(lo, Math.round(v))) : lo;

const INSPECT_RESET: Pick<InspectState, 'selectedPos'> = { selectedPos: null };

// ─────────────────────────────────────────────────────────────────────────────
// Store
// ─────────────────────────────────────────────────────────────────────────────

export const useSim = create<SimState>()(
  persist(
    (set) => ({
      ...SIM_DEFAULTS,
      selectedPos: null,
      block: 0,
      layer: 'col',
      head: 0,
      disabledLayers: [],
      theme: null,
      visited: {},

      // Changing anything that alters the context invalidates the selected token.
      setDb: (dbName, taskId) => set({ dbName, taskId, seedNode: null, ...INSPECT_RESET }),
      setTask: (taskId) => {
        const db = taskId.split('/')[0] as DbName;
        set({ taskId, dbName: DB_NAMES.includes(db) ? db : SIM_DEFAULTS.dbName, seedNode: null, ...INSPECT_RESET });
      },
      setSplit: (split) => set({ split, seedNode: null, ...INSPECT_RESET }),
      setSeedNode: (seedNode) => set({ seedNode, ...INSPECT_RESET }),
      setL: (L) => set({ L: clampInt(L, SIM_LIMITS.L.min, SIM_LIMITS.L.max), ...INSPECT_RESET }),
      setW: (w) => set({ w: clampInt(w, SIM_LIMITS.w.min, SIM_LIMITS.w.max), ...INSPECT_RESET }),
      setRngSeed: (rngSeed) =>
        set({ rngSeed: clampInt(rngSeed, SIM_LIMITS.rngSeed.min, SIM_LIMITS.rngSeed.max), ...INSPECT_RESET }),
      setCkpt: (ckpt) => set({ ckpt }),
      setSelectedPos: (selectedPos) => set({ selectedPos }),
      setBlock: (block) => set({ block: Math.max(0, Math.round(block)) }),
      setLayer: (layer) => set({ layer }),
      setHead: (head) => set({ head: Math.max(0, Math.round(head)) }),
      toggleLayer: (layer) =>
        set((s) => ({
          disabledLayers: s.disabledLayers.includes(layer)
            ? s.disabledLayers.filter((l) => l !== layer)
            : [...s.disabledLayers, layer],
        })),
      setDisabledLayers: (disabledLayers) => set({ disabledLayers: [...disabledLayers] }),
      setTheme: (theme) => set({ theme }),
      markVisited: (slug) =>
        set((s) => (s.visited[slug] ? s : { visited: { ...s.visited, [slug]: true as const } })),
      clearVisited: () => set({ visited: {} }),
      resetSimulation: () =>
        set({ ...SIM_DEFAULTS, selectedPos: null, block: 0, layer: 'col', head: 0, disabledLayers: [] }),
    }),
    {
      name: 'rt-sbs:sim',
      version: 1,
      storage: createJSONStorage(() => safeStorage),
      // Durable knobs only. Ablations (disabledLayers) and the selected token are
      // session-scoped so a reader never lands in a silently ablated model.
      partialize: (s) => ({
        dbName: s.dbName,
        taskId: s.taskId,
        split: s.split,
        seedNode: s.seedNode,
        L: s.L,
        w: s.w,
        rngSeed: s.rngSeed,
        ckpt: s.ckpt,
        block: s.block,
        layer: s.layer,
        head: s.head,
        theme: s.theme,
        visited: s.visited,
      }),
      // Defensive merge: ignore malformed persisted values.
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<SimState>;
        const out: SimState = { ...current };
        if (p.dbName && DB_NAMES.includes(p.dbName)) out.dbName = p.dbName;
        if (typeof p.taskId === 'string' && p.taskId.includes('/')) out.taskId = p.taskId;
        if (p.split === 'Train' || p.split === 'Val' || p.split === 'Test') out.split = p.split;
        if (p.seedNode === null || (typeof p.seedNode === 'number' && Number.isInteger(p.seedNode) && p.seedNode >= 0))
          out.seedNode = p.seedNode;
        if (typeof p.L === 'number') out.L = clampInt(p.L, SIM_LIMITS.L.min, SIM_LIMITS.L.max);
        if (typeof p.w === 'number') out.w = clampInt(p.w, SIM_LIMITS.w.min, SIM_LIMITS.w.max);
        if (typeof p.rngSeed === 'number')
          out.rngSeed = clampInt(p.rngSeed, SIM_LIMITS.rngSeed.min, SIM_LIMITS.rngSeed.max);
        if (p.ckpt && CKPT_CHOICES.includes(p.ckpt)) out.ckpt = p.ckpt;
        if (typeof p.block === 'number') out.block = Math.max(0, Math.round(p.block));
        if (p.layer === 'col' || p.layer === 'feat' || p.layer === 'nbr' || p.layer === 'full') out.layer = p.layer;
        if (typeof p.head === 'number') out.head = Math.max(0, Math.round(p.head));
        if (p.theme === 'light' || p.theme === 'dark' || p.theme === null) out.theme = p.theme;
        if (p.visited && typeof p.visited === 'object') out.visited = { ...(p.visited as Record<string, true>) };
        // keep db consistent with the task id
        const taskDb = out.taskId.split('/')[0] as DbName;
        if (DB_NAMES.includes(taskDb)) out.dbName = taskDb;
        return out;
      },
    },
  ),
);

/** Snapshot of the knobs that define the sampled context (handy for memo keys). */
export function selectSimParams(s: SimState): SimParams {
  return {
    dbName: s.dbName,
    taskId: s.taskId,
    split: s.split,
    seedNode: s.seedNode,
    L: s.L,
    w: s.w,
    rngSeed: s.rngSeed,
    ckpt: s.ckpt,
  };
}

/** True when any simulation knob differs from SIM_DEFAULTS. */
export function isNonDefault(s: SimParams): boolean {
  return (Object.keys(SIM_DEFAULTS) as (keyof typeof SIM_DEFAULTS)[]).some((k) => s[k] !== SIM_DEFAULTS[k]);
}
