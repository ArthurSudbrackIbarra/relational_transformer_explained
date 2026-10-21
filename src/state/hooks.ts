/**
 * Memoised hooks that turn the store's knobs into live simulation data
 * (BUILD_PLAN §7). All computation goes through the engine glue
 * `src/engine/pipeline.ts` (db → preprocess → seeds → context → masks → forward):
 *
 *   useDatabase()        → RawDatabase
 *   usePreprocessed()    → PreprocessedDB
 *   useTask()            → TaskDef
 *   useSeeds()           → number[]          (seed node idxs of the current task split)
 *   useSeedNode()        → number            (store seed, or the automatic default)
 *   useSampleParams()    → SampleParams
 *   useContextWindow()   → SampledContext
 *   useMasks()           → MaskSet
 *   useInContextLabels() → InContextLabels   (self / other-entity labels in the context)
 *   useAssets()          → { status, store, weights, weightsKind, ckpt, note, error }
 *   useForward()         → ForwardResult | null   (null while weights / embeddings load)
 *   usePrediction()      → PredictionReadout | null
 *   useTableSlots()      → tableName → categorical slot (task tables = 5)
 *   useTargetStats()     → Train stats of the target column
 *
 * Every derived object is cached by the pipeline, keyed by exactly the
 * parameters it depends on, so all chapters share one context and one forward
 * pass for the same settings (BUILD_PLAN §6 "the same seed, context and tokens
 * flow through all chapters").
 */
import { useEffect, useMemo, useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { getDatabase, listDatabases, tableColorSlot, type DbInfo } from '@/engine/db/registry';
import type { EmbeddingStore } from '@/engine/embeddings';
import {
  findTask,
  getContext,
  getForward,
  getMasks,
  getPreprocessed,
  getSeeds,
  inContextLabels,
  loadPipelineAssets,
  makeSampleParams,
  pickDefaultSeed,
  readPrediction,
  resolveCheckpoint,
  type CheckpointName,
  type InContextLabels,
  type PredictionReadout,
  type WeightsKind,
} from '@/engine/pipeline';
import type { ColumnStats, ForwardResult, MaskSet, PreprocessedDB, RawDatabase, RTWeights, SampledContext, SampleParams, TaskDef } from '@/engine/types';
import { selectSimParams, useSim, type DbName } from './store';

// The pure helpers live in the engine pipeline; re-exported here for existing imports.
export {
  findTask,
  getContext,
  getForward,
  getMasks,
  getPreprocessed,
  getSeeds,
  makeSampleParams,
  pickDefaultSeed,
  resolveCheckpoint,
  type WeightsKind,
};

// ─────────────────────────────────────────────────────────────────────────────
// Simulation hooks
// ─────────────────────────────────────────────────────────────────────────────

/** The store knobs that define the simulation (shallow-compared). */
export function useSimParams() {
  return useSim(useShallow(selectSimParams));
}

/** All databases (static metadata). */
export function useDatabaseList(): DbInfo[] {
  return useMemo(() => listDatabases(), []);
}

/** The current RawDatabase (deterministic, cached by the engine). */
export function useDatabase(): RawDatabase {
  const dbName = useSim((s) => s.dbName);
  return useMemo(() => getDatabase(dbName as DbName), [dbName]);
}

/** The current PreprocessedDB (pre.rs mirror). */
export function usePreprocessed(): PreprocessedDB {
  const db = useDatabase();
  return useMemo(() => getPreprocessed(db), [db]);
}

/** The current TaskDef. */
export function useTask(): TaskDef {
  const db = useDatabase();
  const taskId = useSim((s) => s.taskId);
  return useMemo(() => findTask(db, taskId), [db, taskId]);
}

/** Seed node idxs of the current task split (node order). */
export function useSeeds(): number[] {
  const pdb = usePreprocessed();
  const task = useTask();
  const split = useSim((s) => s.split);
  return useMemo(() => getSeeds(pdb, task, split), [pdb, task, split]);
}

/** The seed node in use: the store's choice if it is a valid seed, else the default. */
export function useSeedNode(): number {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const task = useTask();
  const split = useSim((s) => s.split);
  const stored = useSim((s) => s.seedNode);
  const seeds = useSeeds();
  return useMemo(() => {
    if (stored !== null && seeds.includes(stored)) return stored;
    return pickDefaultSeed(pdb, db, task, split, seeds);
  }, [stored, seeds, pdb, db, task, split]);
}

/** SampleParams of the live context. */
export function useSampleParams(): SampleParams {
  const pdb = usePreprocessed();
  const task = useTask();
  const seedNode = useSeedNode();
  const { L, w, rngSeed } = useSim(useShallow((s) => ({ L: s.L, w: s.w, rngSeed: s.rngSeed })));
  return useMemo(() => makeSampleParams(pdb, task, seedNode, L, w, rngSeed), [pdb, task, seedNode, L, w, rngSeed]);
}

/** The live context window (fly.rs mirror, with event trace). */
export function useContextWindow(): SampledContext {
  const pdb = usePreprocessed();
  const params = useSampleParams();
  return useMemo(() => getContext(pdb, params), [pdb, params]);
}

/** The four relational attention masks of the live context. */
export function useMasks(): MaskSet {
  const ctx = useContextWindow();
  return useMemo(() => getMasks(ctx), [ctx]);
}

/** In-context labels of the live context: the seed entity's own labels vs other entities' (spec §2.5). */
export function useInContextLabels(): InContextLabels {
  const pdb = usePreprocessed();
  const ctx = useContextWindow();
  const task = useTask();
  return useMemo(() => inContextLabels(pdb, ctx, task), [pdb, ctx, task]);
}

/** tableName → categorical slot for the current DB (BUILD_PLAN §3.5). */
export function useTableSlots(): Record<string, number> {
  const db = useDatabase();
  return useMemo(() => {
    const out: Record<string, number> = {};
    for (const t of db.tables) out[t.schema.name] = tableColorSlot(db.name, t.schema.name);
    for (const t of db.tasks) if (t.kind === 'forecast') out[t.name] = tableColorSlot(db.name, t.name);
    return out;
  }, [db]);
}

/** Train-split stats of the target column (to de-normalize regression predictions, spec §8.4). */
export function useTargetStats(): ColumnStats | null {
  const pdb = usePreprocessed();
  const task = useTask();
  return useMemo(() => {
    const table = task.kind === 'autocomplete' ? task.table ?? task.name : task.name;
    const key = task.kind === 'autocomplete' ? `${table}:Db` : `${table}:Train`;
    return pdb.colStats[key]?.[task.targetCol] ?? null;
  }, [pdb, task]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Assets (embeddings + weights) and the forward pass
// ─────────────────────────────────────────────────────────────────────────────

export type AssetStatus = 'loading' | 'ready' | 'error';

export interface AssetsState {
  status: AssetStatus;
  store: EmbeddingStore | null;
  weights: RTWeights | null;
  /**
   * 'trained' when the exported checkpoint loaded; 'init' for untrained weights (the exported init
   * checkpoint, or the initWeights(MINI_CONFIG, 0) fallback used until trained weights exist in
   * public/data/weights). null while loading or on error.
   */
  weightsKind: WeightsKind | null;
  /** True when the requested checkpoint was missing and the untrained fallback is in use. */
  fallback: boolean;
  /** Resolved checkpoint name (never 'auto'). */
  ckpt: CheckpointName;
  /** Provenance of the weights, for the UI. */
  note?: string;
  error?: string;
}

const loadingState = (ckpt: CheckpointName): AssetsState => ({
  status: 'loading',
  store: null,
  weights: null,
  weightsKind: null,
  fallback: false,
  ckpt,
});

const assetsResolved = new Map<CheckpointName, AssetsState>();
const assetsPending = new Map<CheckpointName, Promise<AssetsState>>();

/** Load (once per checkpoint) the embedding store + weights; resolves to an AssetsState, never rejects. */
export function loadAssets(ckpt: CheckpointName): Promise<AssetsState> {
  let p = assetsPending.get(ckpt);
  if (!p) {
    p = loadPipelineAssets(ckpt)
      .then(
        ({ store, checkpoint }): AssetsState => ({
          status: 'ready',
          store,
          weights: checkpoint.weights,
          weightsKind: checkpoint.kind,
          fallback: checkpoint.fallback,
          ckpt,
          note: checkpoint.note,
        }),
        (e: unknown): AssetsState => ({ ...loadingState(ckpt), status: 'error', error: e instanceof Error ? e.message : String(e) }),
      )
      .then((st) => {
        if (st.status === 'ready') assetsResolved.set(ckpt, st);
        else assetsPending.delete(ckpt); // allow a retry later
        return st;
      });
    assetsPending.set(ckpt, p);
  }
  return p;
}

/** Embedding store + weights for the resolved checkpoint. */
export function useAssets(): AssetsState {
  const taskId = useSim((s) => s.taskId);
  const ckptChoice = useSim((s) => s.ckpt);
  const ckpt = resolveCheckpoint(taskId, ckptChoice);
  const [state, setState] = useState<AssetsState>(() => assetsResolved.get(ckpt) ?? loadingState(ckpt));
  useEffect(() => {
    let alive = true;
    const done = assetsResolved.get(ckpt);
    if (done) {
      setState(done);
      return;
    }
    setState(loadingState(ckpt));
    loadAssets(ckpt).then((st) => {
      if (alive) setState(st);
    });
    return () => {
      alive = false;
    };
  }, [ckpt]);
  return state.ckpt === ckpt ? state : assetsResolved.get(ckpt) ?? loadingState(ckpt);
}

/** The forward pass of the live context (null while assets load or if they failed). */
export function useForward(): ForwardResult | null {
  const ctx = useContextWindow();
  const pdb = usePreprocessed();
  const assets = useAssets();
  const disabled = useSim(useShallow((s) => s.disabledLayers));
  return useMemo(() => {
    if (assets.status !== 'ready' || !assets.store || !assets.weights) return null;
    return getForward(ctx, pdb, assets.store, assets.weights, disabled);
  }, [ctx, pdb, assets, disabled]);
}

/** The prediction at the masked cell of the live forward pass (logit + σ, or z + original units). */
export function usePrediction(): PredictionReadout | null {
  const fr = useForward();
  const ctx = useContextWindow();
  const pdb = usePreprocessed();
  const task = useTask();
  return useMemo(() => (fr ? readPrediction(fr, ctx, pdb, task) : null), [fr, ctx, pdb, task]);
}
