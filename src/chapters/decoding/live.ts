/**
 * Live data for the decoding chapter.
 *
 * The chapter needs one binary target (logit → probability, BCE, AUROC) and one regression target
 * (z-score → original units, Huber, R²). The GLOBAL task of the store is always one of them, with
 * exactly the seed, context and forward pass every other chapter shows. The other one is a LOCAL
 * preview: the first task of the same database with the other target type (forecast tasks first),
 * its default seed in the global split, the global L / w / RNG seed, and its own zero-shot
 * checkpoint. The local preview never writes to the store; the reader can promote it with
 * "make it the global task".
 *
 * Nothing here adds model semantics: contexts come from the sampler mirror (fly.rs), forward passes
 * from model/forward.ts (model.py:153-267), read-outs from pipeline.readPrediction (main.py:283-293).
 */
import { useEffect, useMemo, useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { getDatabase } from '@/engine/db/registry';
import {
  autoCheckpoint,
  getContext,
  getForward,
  getPreprocessed,
  getSeeds,
  makeSampleParams,
  pickDefaultSeed,
  readPrediction,
  resolveCheckpoint,
  type CheckpointName,
  type PredictionReadout,
} from '@/engine/pipeline';
import { targetStats } from '@/engine/sampler';
import type {
  AttnLayer,
  ColumnStats,
  ForwardResult,
  PreprocessedDB,
  RawDatabase,
  RawValue,
  SampledContext,
  Split,
  TaskDef,
  Token,
} from '@/engine/types';
import {
  loadAssets,
  useAssets,
  useContextWindow,
  useDatabase,
  useForward,
  usePreprocessed,
  useSeedNode,
  useTask,
  type AssetsState,
} from '@/state/hooks';
import { useSim, type DbName } from '@/state/store';

export type LaneKind = 'binary' | 'regression';

export interface Lane {
  kind: LaneKind;
  /** True when this lane IS the store's live simulation (same seed/context as every chapter). */
  isGlobal: boolean;
  db: RawDatabase;
  pdb: PreprocessedDB;
  task: TaskDef;
  split: Split;
  seedNode: number;
  ctx: SampledContext;
  ckpt: CheckpointName;
  assets: AssetsState;
  /** null while the weights / embeddings load (or the local forward is still running). */
  fr: ForwardResult | null;
  pred: PredictionReadout | null;
  /** Stats of the target column used to z-score it (Train split for forecast, all rows for autocomplete). */
  stats: ColumnStats | null;
  /** Inference-time ablation of the playground (session-only). Shown as a warning when non-empty. */
  disabled: AttnLayer[];
  L: number;
  w: number;
  rngSeed: number;
}

const loadingAssets = (ckpt: CheckpointName): AssetsState => ({
  status: 'loading',
  store: null,
  weights: null,
  weightsKind: null,
  fallback: false,
  ckpt,
});

/** useAssets for an arbitrary checkpoint (the shared loader caches per checkpoint). */
export function useAssetsFor(ckpt: CheckpointName): AssetsState {
  const [state, setState] = useState<AssetsState | null>(null);
  useEffect(() => {
    let alive = true;
    loadAssets(ckpt).then((st) => {
      if (alive) setState(st);
    });
    return () => {
      alive = false;
    };
  }, [ckpt]);
  return state && state.ckpt === ckpt ? state : loadingAssets(ckpt);
}

function safeStats(pdb: PreprocessedDB, task: TaskDef): ColumnStats | null {
  try {
    return targetStats(pdb, task);
  } catch {
    return null;
  }
}

/** The global lane: exactly the store's simulation. */
function useGlobalLane(): Lane {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const task = useTask();
  const seedNode = useSeedNode();
  const ctx = useContextWindow();
  const assets = useAssets();
  const fr = useForward();
  const knobs = useSim(
    useShallow((s) => ({
      split: s.split,
      L: s.L,
      w: s.w,
      rngSeed: s.rngSeed,
      disabled: s.disabledLayers,
    })),
  );
  const stats = useMemo(() => safeStats(pdb, task), [pdb, task]);
  const pred = useMemo(() => (fr ? readPrediction(fr, ctx, pdb, task) : null), [fr, ctx, pdb, task]);
  return {
    kind: task.targetType,
    isGlobal: true,
    db,
    pdb,
    task,
    split: knobs.split,
    seedNode,
    ctx,
    ckpt: assets.ckpt,
    assets,
    fr,
    pred,
    stats,
    disabled: knobs.disabled,
    L: knobs.L,
    w: knobs.w,
    rngSeed: knobs.rngSeed,
  };
}

/** The task a local lane previews: same DB, other target type, forecast tasks first. */
export function otherTask(db: RawDatabase, kind: LaneKind): { db: RawDatabase; task: TaskDef } {
  const pick = (d: RawDatabase) => {
    const c = d.tasks.filter((t) => t.targetType === kind);
    return c.find((t) => t.kind === 'forecast') ?? c[0];
  };
  const t = pick(db);
  if (t) return { db, task: t };
  const shop = getDatabase('shop');
  return { db: shop, task: pick(shop)! };
}

function useLocalLane(globalDb: RawDatabase, kind: LaneKind): Lane {
  const knobs = useSim(
    useShallow((s) => ({
      split: s.split,
      L: s.L,
      w: s.w,
      rngSeed: s.rngSeed,
      ckptChoice: s.ckpt,
      disabled: s.disabledLayers,
    })),
  );
  const { db, task } = useMemo(() => otherTask(globalDb, kind), [globalDb, kind]);
  const pdb = useMemo(() => getPreprocessed(db), [db]);
  const seedNode = useMemo(() => {
    const s = pickDefaultSeed(pdb, db, task, knobs.split, getSeeds(pdb, task, knobs.split));
    return s >= 0 ? s : pickDefaultSeed(pdb, db, task, 'Val', getSeeds(pdb, task, 'Val'));
  }, [pdb, db, task, knobs.split]);
  const ctx = useMemo(
    () => getContext(pdb, makeSampleParams(pdb, task, seedNode, knobs.L, knobs.w, knobs.rngSeed)),
    [pdb, task, seedNode, knobs.L, knobs.w, knobs.rngSeed],
  );
  // An explicit "init" choice applies everywhere (untrained model); otherwise the local task's own
  // zero-shot checkpoint (a checkpoint selected for another task would be the wrong comparison).
  const ckpt: CheckpointName = knobs.ckptChoice === 'init' ? 'init' : resolveCheckpoint(task.id, 'auto');
  const assets = useAssetsFor(ckpt);
  const stats = useMemo(() => safeStats(pdb, task), [pdb, task]);

  // The forward pass (~60-110 ms) runs after paint, once per (context, weights, ablation).
  const [run, setRun] = useState<{
    ctx: SampledContext;
    weights: unknown;
    disabled: string;
    fr: ForwardResult;
  } | null>(null);
  const disabledKey = [...knobs.disabled].sort().join(',');
  useEffect(() => {
    if (assets.status !== 'ready' || !assets.store || !assets.weights) return;
    const store = assets.store;
    const weights = assets.weights;
    const id = window.setTimeout(() => {
      try {
        const fr = getForward(ctx, pdb, store, weights, knobs.disabled);
        setRun({ ctx, weights, disabled: disabledKey, fr });
      } catch {
        /* leave the lane in its loading state */
      }
    }, 40);
    return () => window.clearTimeout(id);
  }, [assets, ctx, pdb, knobs.disabled, disabledKey]);
  const fr = run && run.ctx === ctx && run.weights === assets.weights && run.disabled === disabledKey ? run.fr : null;
  const pred = useMemo(() => (fr ? readPrediction(fr, ctx, pdb, task) : null), [fr, ctx, pdb, task]);

  return {
    kind,
    isGlobal: false,
    db,
    pdb,
    task,
    split: knobs.split,
    seedNode,
    ctx,
    ckpt,
    assets,
    fr,
    pred,
    stats,
    disabled: knobs.disabled,
    L: knobs.L,
    w: knobs.w,
    rngSeed: knobs.rngSeed,
  };
}

export interface DecodingLive {
  global: Lane;
  binary: Lane;
  regression: Lane;
}

export function useDecodingLive(): DecodingLive {
  const g = useGlobalLane();
  const otherKind: LaneKind = g.kind === 'binary' ? 'regression' : 'binary';
  const o = useLocalLane(g.db, otherKind);
  return {
    global: g,
    binary: g.kind === 'binary' ? g : o,
    regression: g.kind === 'regression' ? g : o,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Display helpers
// ─────────────────────────────────────────────────────────────────────────────

/** "customer" from "customers". */
export function singular(table: string): string {
  if (table.endsWith('ies')) return `${table.slice(0, -3)}y`;
  if (table.endsWith('s')) return table.slice(0, -1);
  return table;
}

const NAME_COLS = ['name', 'display_name', 'title', 'circuit'];

/** The entity a seed row is about, for prose: "customer 35 (Ana Souza)" or "reviews row 120". */
export function seedEntity(lane: Pick<Lane, 'db' | 'pdb' | 'task' | 'seedNode'>): { label: string; name: string | null; node: number | null } {
  const { db, pdb, task, seedNode } = lane;
  const seed = pdb.nodes[seedNode];
  if (!seed) return { label: '–', name: null, node: null };
  if (task.kind === 'forecast' && task.entityTable && seed.f2pNbrIdxs.length > 0) {
    const ent = pdb.nodes[seed.f2pNbrIdxs[0]];
    const table = db.tables.find((t) => t.schema.name === task.entityTable);
    const row = table?.rows[ent.rowPos];
    const nameCol = table?.schema.columns.find((c) => NAME_COLS.includes(c.name))?.name;
    const name = row && nameCol && typeof row[nameCol] === 'string' ? (row[nameCol] as string) : null;
    return {
      label: `${singular(task.entityTable)} ${ent.rowPos}`,
      name,
      node: ent.nodeIdx,
    };
  }
  return {
    label: `${seed.tableName} row ${seed.rowPos}`,
    name: null,
    node: null,
  };
}

/** Short label of a task: "customer-churn" or "reviews.rating". */
export function taskLabel(task: TaskDef): string {
  return task.kind === 'autocomplete' ? `${task.table ?? task.name}.${task.targetCol}` : task.name;
}

export function rawLabelText(raw: RawValue): string {
  if (raw === null) return '∅';
  if (typeof raw === 'boolean') return raw ? 'true' : 'false';
  if (typeof raw === 'number') return Number.isInteger(raw) ? String(raw) : raw.toFixed(3).replace(/\.?0+$/, '');
  return String(raw);
}

/** Row name of a token: "customers[35]" or "customer-churn:Val[32]". */
export function rowName(t: Token): string {
  if (t.isPadding) return 'padding';
  if (t.tableType && t.tableType !== 'Db') return `${t.tableName}:${t.tableType}[${t.rowPos}]`;
  return `${t.tableName}[${t.rowPos}]`;
}

/** Fixed-digit number with a real minus sign. */
export function fmt(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'NaN' : v > 0 ? '+∞' : '−∞';
  return v.toFixed(digits).replace('-', '−');
}

/** Signed fixed-digit number: "+0.948", "−1.052". */
export function fmtS(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return fmt(v);
  return (v < 0 ? '−' : '+') + Math.abs(v).toFixed(digits);
}

export function int(n: number): string {
  return n.toLocaleString('en-US');
}

export { autoCheckpoint, getDatabase };
export type { DbName };
