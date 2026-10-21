/**
 * Convenience glue for the whole simulation (BUILD_PLAN §1 `pipeline.ts`, §2, §7):
 *
 *   getDatabase(name)                        db/registry.ts   (deterministic, cached)
 *     → getPreprocessed(db)                  preprocess.ts    (pre.rs mirror, cached per DB)
 *     → getSeeds / pickDefaultSeed           sampler.ts       (data.py:77-92, fly.rs:287-296)
 *     → makeSampleParams → getContext        sampler.ts       (fly.rs:357-511, one context per seed)
 *     → getMasks                             masks.ts         (model.py:164-191)
 *     → getForward(ctx, pdb, store, weights) model/forward.ts (model.py:153-267)
 *
 * plus the asset side (frozen MiniLM embeddings, spec §3.6; checkpoints with an untrained
 * fallback) and two read-outs every chapter needs: the prediction at the masked cell (spec §8.3-8.4)
 * and the in-context labels of the context (spec §2.5, §11.3).
 *
 * Pure TypeScript, no React: runs in the browser and in Node (tsx / vitest). Every derived object
 * is cached and keyed by exactly the inputs it depends on, so all chapters (through
 * src/state/hooks.ts) and offline scripts share ONE context and ONE forward pass per setting.
 * This module adds no model or sampler semantics of its own; it only wires the modules above.
 */
import { getDatabase, pickShowcaseEntity, type DbName } from './db/registry';
import { loadEmbeddingStore, type EmbeddingStore } from './embeddings';
import { buildMasks } from './masks';
import { binaryLabel, denormalize, sigmoid } from './metrics';
import { MINI_CONFIG } from './model/config';
import { forwardWithMasks, textEmbedder } from './model/forward';
import { initWeights, loadWeights } from './model/weights';
import { preprocess } from './preprocess';
import { contextSeed } from './rng';
import { dropColNameIdxs, listSeeds, sampleContext, targetColNameIdx, targetStats } from './sampler';
import type {
  AttnLayer,
  ColumnStats,
  ForwardResult,
  MaskSet,
  PreprocessedDB,
  RawDatabase,
  RawValue,
  RTWeights,
  SampledContext,
  SampleParams,
  Split,
  TableType,
  TaskDef,
} from './types';

// ─────────────────────────────────────────────────────────────────────────────
// Small cache helpers
// ─────────────────────────────────────────────────────────────────────────────

function lruGet<K, V>(m: Map<K, V>, k: K): V | undefined {
  const v = m.get(k);
  if (v !== undefined) {
    m.delete(k); // refresh recency
    m.set(k, v);
  }
  return v;
}

function lruSet<K, V>(m: Map<K, V>, k: K, v: V, max: number): void {
  m.set(k, v);
  while (m.size > max) m.delete(m.keys().next().value as K);
}

/** Stable small integer id per object (for composite cache keys over objects). */
function objectIdFactory() {
  const ids = new WeakMap<object, number>();
  let next = 1;
  return (o: object): number => {
    let id = ids.get(o);
    if (id === undefined) ids.set(o, (id = next++));
    return id;
  };
}

const ctxId = objectIdFactory();
const weightsId = objectIdFactory();
const pdbId = objectIdFactory();
const storeId = objectIdFactory();

// ─────────────────────────────────────────────────────────────────────────────
// 1. Database → preprocessed graph → task → seeds
// ─────────────────────────────────────────────────────────────────────────────

const pdbCache = new WeakMap<RawDatabase, PreprocessedDB>();

/** Preprocessed graph of a DB (pre.rs mirror, spec §1.5, §3), computed once per RawDatabase object. */
export function getPreprocessed(db: RawDatabase): PreprocessedDB {
  let p = pdbCache.get(db);
  if (!p) {
    p = preprocess(db);
    pdbCache.set(db, p);
  }
  return p;
}

/** The task `taskId` of `db`, or the DB's first task if `db` has no such task (e.g. a stale persisted id). */
export function findTask(db: RawDatabase, taskId: string): TaskDef {
  const t = db.tasks.find((x) => x.id === taskId) ?? db.tasks[0];
  if (!t) throw new Error(`pipeline: database "${db.name}" has no tasks`);
  return t;
}

const seedsCache = new WeakMap<PreprocessedDB, Map<string, number[]>>();

/** Cache key of a task split; autocomplete ignores the split (data.py:77-81, spec §2.4). */
function splitKey(task: TaskDef, split: Split): string {
  return `${task.id}|${task.kind === 'autocomplete' ? '*' : split}`;
}

/** Seed node idxs of a task split, in node order (sampler.listSeeds, cached). */
export function getSeeds(pdb: PreprocessedDB, task: TaskDef, split: Split): number[] {
  let m = seedsCache.get(pdb);
  if (!m) seedsCache.set(pdb, (m = new Map()));
  const key = splitKey(task, split);
  let s = m.get(key);
  if (!s) {
    s = listSeeds(pdb, task, split);
    m.set(key, s);
  }
  return s;
}

const defaultSeedCache = new WeakMap<PreprocessedDB, Map<string, number>>();

/**
 * Default seed of a task split ("a Val seed with several past labels", BUILD_PLAN §7). Deterministic.
 *  - binary forecast: the db layer's showcase entity (long, mixed label history; ShopDB → customer 35),
 *    its first row in `split`. For shop/customer-churn Val this is node 2406 (customer-churn:Val[32]),
 *    whose L=128, w=16 context holds all 12 of the customer's Train labels (pipeline.test.ts);
 *  - other forecast tasks (or if the showcase entity has no row in `split`): the seed whose entity has
 *    the most earlier rows of the same task table (= most potential self labels, spec §2.5),
 *    ties → lowest node idx;
 *  - autocomplete: the median seed in node order.
 * Returns −1 when the split has no seeds.
 */
export function pickDefaultSeed(pdb: PreprocessedDB, db: RawDatabase, task: TaskDef, split: Split, seeds?: number[]): number {
  const list = seeds ?? getSeeds(pdb, task, split);
  if (list.length === 0) return -1;
  let m = defaultSeedCache.get(pdb);
  if (!m) defaultSeedCache.set(pdb, (m = new Map()));
  const key = splitKey(task, split);
  const hit = m.get(key);
  if (hit !== undefined) return hit;

  let chosen = list[Math.floor(list.length / 2)];
  if (task.kind === 'forecast') {
    let done = false;
    if (task.targetType === 'binary') {
      try {
        const sc = pickShowcaseEntity(db, task.id);
        const rows = sc.rows[split];
        const info = pdb.tableInfo[`${task.name}:${split}`];
        if (rows && rows.length && info) {
          const node = info.nodeIdxOffset + rows[0];
          if (list.includes(node)) {
            chosen = node;
            done = true;
          }
        }
      } catch {
        /* no showcase entity for this task: fall through to the generic rule */
      }
    }
    if (!done) {
      let best = -1;
      for (const s of list) {
        const seed = pdb.nodes[s];
        const entity = seed.f2pNbrIdxs[0];
        if (entity === undefined || seed.ts === null) continue;
        let past = 0;
        for (const e of pdb.p2fAdj[entity] ?? []) {
          if (e.tableType !== 'Db' && e.tableNameIdx === seed.tableNameIdx && e.node !== s && e.ts !== null && e.ts < seed.ts) past++;
        }
        if (past > best) {
          best = past;
          chosen = s;
        }
      }
    }
  }
  m.set(key, chosen);
  return chosen;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Context window and masks
// ─────────────────────────────────────────────────────────────────────────────

/**
 * SampleParams for the eval-style context of `seedNode`. fly.rs:371-375 seeds the PRNG with
 * `epoch + seed_node_idx + seed`; evaluation uses epoch 0 and seed 0 (main.py:158-160, spec §4.2,
 * §8.6), so `rngSeed` here is the user-facing `seed` term and the sum is what the sampler receives.
 */
export function makeSampleParams(pdb: PreprocessedDB, task: TaskDef, seedNode: number, L: number, w: number, rngSeed: number): SampleParams {
  return {
    seedNode,
    L,
    w,
    targetColNameIdx: targetColNameIdx(pdb, task),
    dropColNameIdxs: dropColNameIdxs(pdb, task),
    rngSeed: contextSeed(0, seedNode, rngSeed),
  };
}

/** LRU of sampled contexts (with their event traces). */
const CTX_MAX = 48;
const ctxCache = new Map<string, SampledContext>();

/** Cached sampleContext: equal params (by value) on the same PreprocessedDB give the same object. */
export function getContext(pdb: PreprocessedDB, params: SampleParams): SampledContext {
  const key = [
    pdbId(pdb),
    params.seedNode,
    params.L,
    params.w,
    params.rngSeed,
    params.targetColNameIdx,
    params.dropColNameIdxs.join(','),
  ].join('|');
  let ctx = lruGet(ctxCache, key);
  if (!ctx) {
    ctx = sampleContext(pdb, params);
    lruSet(ctxCache, key, ctx, CTX_MAX);
  }
  return ctx;
}

const masksCache = new WeakMap<SampledContext, MaskSet>();

/** The four relational attention masks of a context (cached per context). */
export function getMasks(ctx: SampledContext): MaskSet {
  let m = masksCache.get(ctx);
  if (!m) {
    m = buildMasks(ctx.tokens);
    masksCache.set(ctx, m);
  }
  return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Assets: embeddings + checkpoints (with an untrained fallback)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Checkpoints exported by python/train.py (python/export_weights.py format) to public/data/weights
 * (BUILD_PLAN §4.5). `init` and every `pretrain-*` checkpoint come from ONE leave-one-DB-out pretraining
 * run on SchoolDB + RacingDB + ForumDB (ShopDB held out, spec §9.1): `init` is its step-0 weights, the
 * others different steps of it. `finetune-*` continue from pretrain-churn / pretrain-sales (spec §9.2).
 */
export const CHECKPOINT_NAMES = [
  'init',
  'pretrain-churn',
  'pretrain-sales',
  'pretrain-rating',
  'pretrain-verified',
  'pretrain-final',
  'finetune-churn',
  'finetune-sales',
] as const;
export type CheckpointName = (typeof CHECKPOINT_NAMES)[number];

/** One-line provenance of each checkpoint, for the UI (selection rules: python/train.py, results.json). */
export const CHECKPOINT_INFO: Record<CheckpointName, string> = {
  init: 'Untrained: PyTorch-default initialisation after torch.manual_seed(0), the starting point of pretraining.',
  'pretrain-churn': 'Pretraining step with the best ShopDB customer-churn validation AUROC (zero-shot, spec §9.1).',
  'pretrain-sales': 'Pretraining step with the best ShopDB product-sales validation R² (zero-shot, spec §9.1).',
  'pretrain-rating':
    'Pretraining step with the best ShopDB reviews.rating R² on the even review_ids (zero-shot; the odd ids are the held-out half).',
  'pretrain-verified':
    'Pretraining step with the best ShopDB reviews.verified AUROC on the even review_ids (zero-shot; the odd ids are the held-out half).',
  'pretrain-final': 'Last pretraining step. SchoolDB, RacingDB and ForumDB were seen in pretraining, so on their tasks this is not zero-shot.',
  'finetune-churn': 'pretrain-churn fine-tuned on the ShopDB customer-churn Train split (best validation step, spec §9.2).',
  'finetune-sales': 'pretrain-sales fine-tuned on the ShopDB product-sales Train split (best validation step, spec §9.2).',
};

/** The zero-shot checkpoint of each ShopDB task: best validation step for that task (spec §9.1). */
const SHOP_AUTO: Record<string, CheckpointName> = {
  'shop/customer-churn': 'pretrain-churn',
  'shop/product-sales': 'pretrain-sales',
  'shop/reviews.rating': 'pretrain-rating',
  'shop/reviews.verified': 'pretrain-verified',
};

export interface AutoCheckpoint {
  name: CheckpointName;
  /** True for ShopDB tasks: the model never saw ShopDB (leave-one-DB-out). False for the pretraining DBs. */
  zeroShot: boolean;
  /** Human-readable explanation, for the UI. */
  note: string;
}

/**
 * What 'auto' means for a task:
 *  - a ShopDB task (held out, never seen in pretraining) → its own best zero-shot checkpoint:
 *    customer-churn → pretrain-churn, product-sales → pretrain-sales, reviews.rating → pretrain-rating,
 *    reviews.verified → pretrain-verified (best val per target task, spec §9.1);
 *  - a SchoolDB / RacingDB / ForumDB task → pretrain-final. Those DBs were SEEN in pretraining (their tasks
 *    are training tasks; the forecast Val/Test rows were not trained on, but the DB was), so this is NOT
 *    zero-shot.
 */
export function autoCheckpoint(taskId: string): AutoCheckpoint {
  const shop = SHOP_AUTO[taskId];
  if (shop) {
    const auto = taskId.startsWith('shop/reviews.');
    return {
      name: shop,
      zeroShot: true,
      note:
        'Zero-shot: pretrained only on SchoolDB, RacingDB and ForumDB (ShopDB held out), checkpoint chosen on this ' +
        (auto
          ? 'task\'s validation half (even review_ids; the paper selects only forecast tasks, spec §9.1).'
          : 'task\'s ShopDB validation split (spec §9.1).'),
    };
  }
  const db = taskId.split('/')[0];
  const label = ({ school: 'SchoolDB', racing: 'RacingDB', forum: 'ForumDB' } as Record<string, string>)[db] ?? db;
  return {
    name: 'pretrain-final',
    zeroShot: false,
    note:
      `Not zero-shot: ${label} was seen in pretraining (leave-one-DB-out holds out ShopDB only), so the final ` +
      'pretraining checkpoint is used.',
  };
}

/**
 * Resolve a checkpoint choice. 'auto' = `autoCheckpoint(taskId).name`: the task's own zero-shot
 * checkpoint for ShopDB tasks, the final pretraining checkpoint for the pretraining DBs.
 */
export function resolveCheckpoint(taskId: string, choice: string): CheckpointName {
  if (choice !== 'auto') {
    if (!(CHECKPOINT_NAMES as readonly string[]).includes(choice)) throw new Error(`pipeline: unknown checkpoint "${choice}"`);
    return choice as CheckpointName;
  }
  return autoCheckpoint(taskId).name;
}

/** 'trained' = an exported trained checkpoint; 'init' = untrained (PyTorch-default) weights. */
export type WeightsKind = 'init' | 'trained';

export interface LoadedCheckpoint {
  /** The checkpoint that was requested (resolved, never 'auto'). */
  name: CheckpointName;
  weights: RTWeights;
  kind: WeightsKind;
  /** True when the requested file was unavailable and the untrained fallback is used instead. */
  fallback: boolean;
  /** Human-readable provenance, for the UI. */
  note: string;
  /** Why the file could not be loaded (fallback only). */
  error?: string;
}

let fallbackInit: RTWeights | null = null;

/**
 * The untrained weights used until trained checkpoints exist: `initWeights(MINI_CONFIG, 0)`, i.e.
 * PyTorch-default-style initialisation (nn.Linear kaiming-uniform, RMSNorm γ = 1, mask_embs ~ N(0,1);
 * spec §7.4) drawn from the site's seeded PRNG. One shared object, so every fallback shares the
 * forward-pass cache.
 */
export function fallbackInitWeights(): RTWeights {
  if (!fallbackInit) {
    const w = initWeights(MINI_CONFIG, 0);
    w.meta = { ...w.meta, name: 'init', source: 'initWeights(MINI_CONFIG, 0)' };
    fallbackInit = w;
  }
  return fallbackInit;
}

const ckptCache = new Map<CheckpointName, Promise<LoadedCheckpoint>>();

/**
 * Load `public/data/weights/<name>.{json,bin}`; if it is missing or invalid, fall back to the
 * untrained `fallbackInitWeights()` and say so (`kind: 'init'`, `fallback: true`). Never rejects.
 * A successful load is cached; a fallback is retried on the next call (the file may appear later).
 */
export function loadCheckpoint(name: CheckpointName): Promise<LoadedCheckpoint> {
  const hit = ckptCache.get(name);
  if (hit) return hit;
  const p = loadWeights(name).then(
    (weights): LoadedCheckpoint => ({
      name,
      weights,
      kind: name === 'init' ? 'init' : 'trained',
      fallback: false,
      note: name === 'init' ? 'Exported untrained initialisation (init checkpoint).' : `Trained checkpoint "${name}": ${CHECKPOINT_INFO[name]}`,
    }),
    (e: unknown): LoadedCheckpoint => {
      ckptCache.delete(name);
      return {
        name,
        weights: fallbackInitWeights(),
        kind: 'init',
        fallback: true,
        note:
          name === 'init'
            ? 'Untrained weights: initWeights(MINI_CONFIG, 0), PyTorch-default initialisation.'
            : `Checkpoint "${name}" is not available yet, so the model runs with untrained weights ` +
              '(initWeights(MINI_CONFIG, 0)). Predictions are not meaningful until trained weights are exported.',
        error: e instanceof Error ? e.message : String(e),
      };
    },
  );
  ckptCache.set(name, p);
  return p;
}

export interface PipelineAssets {
  store: EmbeddingStore;
  checkpoint: LoadedCheckpoint;
}

/**
 * Embedding store + checkpoint. Rejects only if the embeddings cannot be loaded: the simulation
 * never fabricates text embeddings (spec §3.6, BUILD_PLAN §0).
 */
export async function loadPipelineAssets(ckpt: CheckpointName): Promise<PipelineAssets> {
  const [store, checkpoint] = await Promise.all([loadEmbeddingStore(), loadCheckpoint(ckpt)]);
  return { store, checkpoint };
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Forward pass
// ─────────────────────────────────────────────────────────────────────────────

/**
 * LRU of forward results. A captured forward keeps the scores and probabilities (H·S·S each) of all
 * 16 attention sublayers: ~10 MB at L = 128 but ~130 MB at L = 512. The cache is therefore bounded
 * both in entries and in total S² (budget = 6 contexts of L = 128); the newest entry is always kept.
 */
const FWD_MAX = 6;
const FWD_BUDGET_S2 = 6 * 128 * 128;
const fwdCache = new Map<string, ForwardResult>();

function fwdCacheSet(key: string, r: ForwardResult): void {
  fwdCache.set(key, r);
  let total = 0;
  for (const v of fwdCache.values()) total += v.S * v.S;
  while (fwdCache.size > 1 && (fwdCache.size > FWD_MAX || total > FWD_BUDGET_S2)) {
    const oldest = fwdCache.keys().next().value as string;
    const v = fwdCache.get(oldest)!;
    total -= v.S * v.S;
    fwdCache.delete(oldest);
  }
}

/**
 * Forward pass of a context (model.py:153-267), cached by (context, DB, store, weights, ablation).
 * Reuses the context's cached masks, so `result.masks === getMasks(ctx)`.
 * `disabledLayers` is the inference-time ablation of the playground (skip those sublayers in every
 * block); the paper's ablations retrain instead (spec §11.4).
 */
export function getForward(
  ctx: SampledContext,
  pdb: Pick<PreprocessedDB, 'text'>,
  store: EmbeddingStore,
  weights: RTWeights,
  disabledLayers: readonly AttnLayer[] = [],
): ForwardResult {
  const ablation = [...new Set(disabledLayers)].sort();
  const key = [ctxId(ctx), pdbId(pdb), storeId(store), weightsId(weights), ablation.join(',')].join('|');
  let r = lruGet(fwdCache, key);
  if (!r) {
    if (store.dim !== weights.config.dText) {
      throw new Error(`pipeline: embedding dim ${store.dim} != model dText ${weights.config.dText}`);
    }
    r = forwardWithMasks(ctx.tokens, getMasks(ctx), textEmbedder(pdb, store), weights, {
      disabledLayers: ablation,
      capture: true,
    });
    fwdCacheSet(key, r);
  }
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Read-outs: prediction at the target, in-context labels
// ─────────────────────────────────────────────────────────────────────────────

export interface PredictionReadout {
  taskKind: TaskDef['targetType'];
  targetPos: number;
  /** Decoder output at the target: a logit (binary) or a z-score (regression), spec §8.3. */
  output: number;
  /** Binary: σ(logit) (spec §8.4; the metric itself uses the logit, §8.5). */
  probability?: number;
  /** Regression: ẑ·σ_train + μ_train in original units (spec §8.4). */
  value?: number;
  /** The hidden ground truth stored in the target token (gated out of the input, spec §5.3). */
  label: {
    raw: RawValue;
    /** Binary: int(y > 0) on the stored boolean value, as main.py:283-293 does. */
    binary?: 0 | 1;
    /** Regression: the stored z-score. */
    z?: number;
  };
  /** Train-split stats of the target column used to de-normalize (regression). */
  stats?: ColumnStats;
}

/** Read the prediction at the masked cell of a forward result (spec §8.3-8.4). null if the context has no target. */
export function readPrediction(fr: ForwardResult, ctx: SampledContext, pdb: PreprocessedDB, task: TaskDef): PredictionReadout | null {
  const pos = fr.targetPos;
  if (pos < 0) return null;
  const tok = ctx.tokens[pos];
  if (task.targetType === 'binary') {
    return {
      taskKind: 'binary',
      targetPos: pos,
      output: fr.prediction,
      probability: sigmoid(fr.prediction),
      label: { raw: tok.raw, binary: binaryLabel(tok.booleanValue) },
    };
  }
  const stats = targetStats(pdb, task);
  return {
    taskKind: 'regression',
    targetPos: pos,
    output: fr.prediction,
    value: denormalize(fr.prediction, stats),
    label: { raw: tok.raw, z: tok.numberValue },
    stats,
  };
}

export interface InContextLabel {
  pos: number;
  nodeIdx: number;
  tableType: TableType | null;
  rowPos: number;
  raw: RawValue;
  /** Row timestamp in seconds (null for rows without one). */
  ts: number | null;
}

export interface InContextLabels {
  /** The seed's entity node (forecast: the task row's FK parent), or null (autocomplete). */
  entityNode: number | null;
  /** Visible target-column cells of the seed's own entity (forecast only). */
  self: InContextLabel[];
  /** Visible target-column cells of every other row. */
  other: InContextLabel[];
}

/**
 * The in-context labels of a context: every UNMASKED cell of the target column ("unmasked cells from
 * the target column", paper Table 12 caption; spec §2.5, §11.3). For forecasting, a label is "self"
 * when its task row points to the same entity as the seed row (the task table's single FK), else
 * "other". Autocomplete has no entity notion: every visible target-column cell counts as "other"
 * (spec §2.4: they are other rows of the same DB table).
 */
export function inContextLabels(pdb: PreprocessedDB, ctx: SampledContext, task: TaskDef): InContextLabels {
  const seed = pdb.nodes[ctx.params.seedNode];
  const entityNode = task.kind === 'forecast' && seed && seed.f2pNbrIdxs.length > 0 ? seed.f2pNbrIdxs[0] : null;
  const self: InContextLabel[] = [];
  const other: InContextLabel[] = [];
  for (const t of ctx.tokens) {
    if (t.isPadding || t.isTarget || t.colNameIdx !== ctx.params.targetColNameIdx) continue;
    const node = pdb.nodes[t.nodeIdx];
    const label: InContextLabel = { pos: t.pos, nodeIdx: t.nodeIdx, tableType: t.tableType, rowPos: t.rowPos, raw: t.raw, ts: node?.ts ?? null };
    if (entityNode !== null && node?.f2pNbrIdxs[0] === entityNode) self.push(label);
    else other.push(label);
  }
  return { entityNode, self, other };
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. One-call pipeline (scripts, tests, the dev gallery)
// ─────────────────────────────────────────────────────────────────────────────

export interface SimSettings {
  dbName: DbName;
  /** "<db>/<task>"; falls back to the DB's first task. */
  taskId: string;
  split?: Split;
  /** null/undefined = pickDefaultSeed. */
  seedNode?: number | null;
  L?: number;
  w?: number;
  /** The `seed` term of fly.rs's `epoch + seed_node_idx + seed` (eval: 0). */
  rngSeed?: number;
}

export interface Simulation {
  db: RawDatabase;
  pdb: PreprocessedDB;
  task: TaskDef;
  split: Split;
  seeds: number[];
  seedNode: number;
  params: SampleParams;
  ctx: SampledContext;
  masks: MaskSet;
}

/** Mini-model defaults (BUILD_PLAN §2): L = 128, w = 16, eval seed 0, Val split. */
export const PIPELINE_DEFAULTS = { split: 'Val' as Split, L: 128, w: 16, rngSeed: 0 } as const;

/**
 * Everything up to the masks, synchronously: db → pdb → task → seeds → seed → context → masks.
 * A `seedNode` that is not a seed of the task split is replaced by the default seed.
 */
export function simulate(s: SimSettings): Simulation {
  const db = getDatabase(s.dbName);
  const pdb = getPreprocessed(db);
  const task = findTask(db, s.taskId);
  const split = s.split ?? PIPELINE_DEFAULTS.split;
  const seeds = getSeeds(pdb, task, split);
  if (seeds.length === 0) throw new Error(`pipeline: task ${task.id} has no seeds in split ${split}`);
  const seedNode =
    s.seedNode !== null && s.seedNode !== undefined && seeds.includes(s.seedNode) ? s.seedNode : pickDefaultSeed(pdb, db, task, split, seeds);
  const params = makeSampleParams(pdb, task, seedNode, s.L ?? PIPELINE_DEFAULTS.L, s.w ?? PIPELINE_DEFAULTS.w, s.rngSeed ?? PIPELINE_DEFAULTS.rngSeed);
  const ctx = getContext(pdb, params);
  return { db, pdb, task, split, seeds, seedNode, params, ctx, masks: getMasks(ctx) };
}

export interface PipelineRun extends Simulation {
  assets: PipelineAssets;
  forward: ForwardResult;
  prediction: PredictionReadout | null;
  labels: InContextLabels;
}

/** The whole pipeline including the forward pass. `ckpt` defaults to 'auto' (see resolveCheckpoint). */
export async function runPipeline(s: SimSettings & { ckpt?: string; disabledLayers?: readonly AttnLayer[] }): Promise<PipelineRun> {
  const sim = simulate(s);
  const assets = await loadPipelineAssets(resolveCheckpoint(sim.task.id, s.ckpt ?? 'auto'));
  const forward = getForward(sim.ctx, sim.pdb, assets.store, assets.checkpoint.weights, s.disabledLayers ?? []);
  return {
    ...sim,
    assets,
    forward,
    prediction: readPrediction(forward, sim.ctx, sim.pdb, sim.task),
    labels: inContextLabels(sim.pdb, sim.ctx, sim.task),
  };
}
