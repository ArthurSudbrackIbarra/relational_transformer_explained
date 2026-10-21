/**
 * Export sampled context windows for PyTorch training / evaluation (BUILD_PLAN §4 step 3).
 *
 * Every context is produced by the site's own engine — getDatabase → preprocess (pre.rs mirror) →
 * listSeeds → sampleContext (fly.rs mirror) — so the model is trained on exactly the token sequences
 * the website later shows. The per-context PRNG seed is `epoch + seed_node_idx + seed` with seed = 0,
 * like fly.rs:371-375 (spec §4.2), and evaluation contexts use epoch 0 (main.py:158-160, spec §8.6).
 *
 * Sets (one binary file each, described in index.json):
 *   pretrain                          leave-one-DB-out pretraining items of school + racing + forum
 *                                     (spec §9.1): forecast tasks → Train split; autocomplete → ALL rows
 *                                     of the DB table (data.py:77-81, spec §2.4). One item list over all
 *                                     tasks, no balancing (fly.rs:257-297, spec §4.14). Stored epoch-major
 *                                     in item order; the trainer shuffles each epoch (fly.rs:324-328).
 *   eval/<task>/<Val|Test>            shop forecast tasks, epoch 0 (zero-shot evaluation, spec §9.4)
 *   eval/<task>/All                   shop autocomplete tasks, epoch 0 (split ignored, spec §2.4)
 *   diag/<task>/Val                   forecast Val splits of the PRETRAINING DBs, epoch 0 (in-distribution
 *                                     diagnostics only; never used for checkpoint selection)
 *   finetune/<task>                   shop forecast task, Train split, epochs 0..E_ft−1 (spec §9.2)
 *
 * Batch layout per token (fly.rs:27-44, 462-500; spec §5.1). To keep files small, ids are int16
 * (asserted to fit) and the three scalar value slots are stored as ONE float32 `values` array: the
 * sampler writes cell.number / cell.datetime / cell.boolean, and pre.rs stores 0 in every slot that is
 * not the cell's own sem type, so
 *     number_values   = values · [sem = 0]
 *     datetime_values = values · [sem = 2]
 *     boolean_values  = values · [sem = 3]
 * reconstructs the official arrays exactly (checked for every token below). `text_idxs` is the per-DB
 * text index (0 for non-text cells, as pre.rs stores; −1 for padding, whose text_values are 0⃗).
 * The masked target keeps its true value in its slot (fly.rs:482-490, spec §5.3).
 *
 *   node_idxs        int16  [n, L]      −1 for padding
 *   f2p_nbr_idxs     int16  [n, L, 5]   −1-padded
 *   table_name_idxs  int16  [n, L]
 *   col_name_idxs    int16  [n, L]
 *   sem_types        uint8  [n, L]
 *   values           float32[n, L]
 *   text_idxs        int16  [n, L]
 *   flags            uint8  [n, L]      bit0 masks, bit1 is_targets, bit2 is_padding, bit3 is_task_nodes
 * Per context:
 *   task             uint16 [n]   index into index.json `tasks`
 *   seed_node        int32  [n]
 *   epoch            uint16 [n]
 *   target_pos       int16  [n]
 *   label            float32[n]   stored z-scored label at the target (boolean: >0 ⇔ positive)
 *   entity_mean      float32[n]   EntityMean baseline (paper App. I): mean of the entity's PAST labels of
 *                                 the same task table (any split, ts < seed ts); binary: mean of 1{z>0},
 *                                 regression: mean z. Fallback (no past label): Train-split mean
 *                                 (positive rate / mean z). NaN for autocomplete.
 *   entity_n_past    int16  [n]   number of past labels behind entity_mean (0 → fallback)
 *   n_real           int16  [n]   non-padding tokens
 *   n_self_labels    int16  [n]   visible target-column cells of the seed's own entity (spec §2.5)
 *   n_other_labels   int16  [n]   visible target-column cells of other rows
 *
 * Usage:
 *   npx tsx scripts/export-contexts.ts [--L 128] [--w 16] [--pretrain-epochs 8] [--finetune-epochs 64]
 *                                      [--out python/data/contexts]
 */
import { mkdirSync, writeFileSync, openSync, writeSync, closeSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { DB_NAMES, getDatabase, type DbName } from '../src/engine/db/index.ts';
import { preprocess } from '../src/engine/preprocess.ts';
import { listSeeds, sampleContext, targetColNameIdx, dropColNameIdxs, targetStats } from '../src/engine/sampler.ts';
import { contextSeed } from '../src/engine/rng.ts';
import { inContextLabels } from '../src/engine/pipeline.ts';
import { SemType, type PreprocessedDB, type Split, type TaskDef, type SampleParams } from '../src/engine/types.ts';

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────
function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}
const L = Number(arg('L', '128'));
const W = Number(arg('w', '16'));
const PRETRAIN_EPOCHS = Number(arg('pretrain-epochs', '8'));
const FINETUNE_EPOCHS = Number(arg('finetune-epochs', '64'));
const SEED = 0; // the `seed` term of fly.rs `epoch + seed_node_idx + seed` (example_*.py: seed=0)
const OUT = resolve(arg('out', 'python/data/contexts'));

const TARGET_DB: DbName = 'shop';
const PRETRAIN_DBS: DbName[] = DB_NAMES.filter((d) => d !== TARGET_DB); // leave-one-DB-out (spec §9.1)
const F2P = 5;
const I16_MAX = 32767;

// ─────────────────────────────────────────────────────────────────────────────
// Databases and tasks
// ─────────────────────────────────────────────────────────────────────────────
const pdbs = new Map<DbName, PreprocessedDB>();
for (const name of DB_NAMES) pdbs.set(name, preprocess(getDatabase(name)));

interface TaskEntry {
  idx: number;
  task: TaskDef;
  pdb: PreprocessedDB;
  targetColNameIdx: number;
  dropColNameIdxs: number[];
  /** Train-split (forecast) / all-rows (autocomplete) stats used to de-normalize (spec §8.4). */
  stats: { mean: number; std: number };
  /** Mean of the EntityMean fallback quantity over the Train split (positive rate or mean z). */
  trainMean: number;
}

/** Task order like PE/rt/tasks.py all_tasks: forecast clf, forecast reg, autocomplete clf, autocomplete reg. */
function group(t: TaskDef): number {
  return (t.kind === 'forecast' ? 0 : 2) + (t.targetType === 'binary' ? 0 : 1);
}
const allTasks: TaskDef[] = DB_NAMES.flatMap((d) => getDatabase(d).tasks).sort(
  (a, b) => group(a) - group(b) || DB_NAMES.indexOf(a.dbName as DbName) - DB_NAMES.indexOf(b.dbName as DbName),
);

const tasks: TaskEntry[] = allTasks.map((task, idx) => {
  const pdb = pdbs.get(task.dbName as DbName)!;
  const tcol = targetColNameIdx(pdb, task);
  const s = targetStats(pdb, task);
  // Train-split label mean in the space EntityMean uses (binary: positive rate; regression: z).
  let sum = 0;
  let n = 0;
  for (const seed of listSeeds(pdb, task, 'Train')) {
    const v = labelOf(pdb, seed, tcol, task);
    sum += v;
    n++;
  }
  return {
    idx,
    task,
    pdb,
    targetColNameIdx: tcol,
    dropColNameIdxs: dropColNameIdxs(pdb, task),
    stats: { mean: s.mean, std: s.std },
    trainMean: n ? sum / n : 0,
  };
});
const taskById = new Map(tasks.map((t) => [t.task.id, t]));

/** Label of a seed row in EntityMean space: binary → 1{z>0} (main.py:332), regression → z. */
function labelOf(pdb: PreprocessedDB, node: number, tcol: number, task: TaskDef): number {
  const cell = pdb.nodes[node].cells.find((c) => c.colNameIdx === tcol);
  if (!cell) throw new Error(`node ${node} has no target cell`);
  return task.targetType === 'binary' ? (cell.boolean > 0 ? 1 : 0) : cell.number;
}

/**
 * EntityMean baseline (paper p.7 §4.1 and App. I p.23): "predicts the mean of past target-entity
 * labels. When no past labels are available for the target entity, it falls back to the global mean
 * of the target column on the training set." Past = rows of the same task table (any split; all
 * splits share table_name_idx, spec §2.2) pointing to the same entity with ts strictly earlier.
 */
function entityMean(te: TaskEntry, seedNode: number): { value: number; nPast: number } {
  if (te.task.kind !== 'forecast') return { value: NaN, nPast: 0 };
  const pdb = te.pdb;
  const seed = pdb.nodes[seedNode];
  const entity = seed.f2pNbrIdxs[0];
  let sum = 0;
  let n = 0;
  if (entity !== undefined && seed.ts !== null) {
    for (const e of pdb.p2fAdj[entity]) {
      if (e.tableType === 'Db' || e.tableNameIdx !== seed.tableNameIdx || e.node === seedNode) continue;
      if (e.ts === null || e.ts >= seed.ts) continue;
      const child = pdb.nodes[e.node];
      if (!child.cells.some((c) => c.colNameIdx === te.targetColNameIdx)) continue; // null label: not a label
      sum += labelOf(pdb, e.node, te.targetColNameIdx, te.task);
      n++;
    }
  }
  return n > 0 ? { value: sum / n, nPast: n } : { value: te.trainMean, nPast: 0 };
}

// ─────────────────────────────────────────────────────────────────────────────
// Writer: one context set → typed arrays → <name>.bin
// ─────────────────────────────────────────────────────────────────────────────
type DType = 'int16' | 'uint8' | 'float32' | 'int32' | 'uint16';
interface ArrayDesc {
  dtype: DType;
  shape: number[];
  offset: number; // bytes
}
interface SetDesc {
  file: string;
  n: number;
  L: number;
  kind: 'pretrain' | 'eval' | 'diag' | 'finetune';
  tasks: string[];
  split: Split | 'All';
  epochs: number;
  /** Items (seeds) per epoch; contexts are stored epoch-major: index = epoch·itemsPerEpoch + item. */
  itemsPerEpoch: number;
  arrays: Record<string, ArrayDesc>;
  stats: { meanReal: number; fracFull: number; meanSelfLabels: number; meanOtherLabels: number; tokens: number; realTokens: number };
}

interface Item {
  te: TaskEntry;
  seed: number;
}

function buildSet(name: string, kind: SetDesc['kind'], items: Item[], epochs: number, split: Split | 'All'): SetDesc {
  const n = items.length * epochs;
  const a = {
    node_idxs: new Int16Array(n * L),
    f2p_nbr_idxs: new Int16Array(n * L * F2P),
    table_name_idxs: new Int16Array(n * L),
    col_name_idxs: new Int16Array(n * L),
    sem_types: new Uint8Array(n * L),
    values: new Float32Array(n * L),
    text_idxs: new Int16Array(n * L),
    flags: new Uint8Array(n * L),
    task: new Uint16Array(n),
    seed_node: new Int32Array(n),
    epoch: new Uint16Array(n),
    target_pos: new Int16Array(n),
    label: new Float32Array(n),
    entity_mean: new Float32Array(n),
    entity_n_past: new Int16Array(n),
    n_real: new Int16Array(n),
    n_self_labels: new Int16Array(n),
    n_other_labels: new Int16Array(n),
  };
  const i16 = (v: number, what: string) => {
    if (!Number.isInteger(v) || v < -1 || v > I16_MAX) throw new Error(`${name}: ${what}=${v} does not fit int16`);
    return v;
  };
  let sumReal = 0;
  let nFull = 0;
  let sumSelf = 0;
  let sumOther = 0;
  // Entity means do not depend on the epoch: cache per seed.
  const emCache = new Map<string, { value: number; nPast: number }>();

  let c = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (const { te, seed } of items) {
      const params: SampleParams = {
        seedNode: seed,
        L,
        w: W,
        targetColNameIdx: te.targetColNameIdx,
        dropColNameIdxs: te.dropColNameIdxs,
        rngSeed: contextSeed(epoch, seed, SEED), // fly.rs:371-375
      };
      const ctx = sampleContext(te.pdb, params, { recordEvents: false });
      if (ctx.targetPos < 0) throw new Error(`${name}: context of seed ${seed} has no target cell`);
      if (ctx.tokens.length !== L) throw new Error(`${name}: context length ${ctx.tokens.length} != ${L}`);
      let nMasked = 0;
      for (let p = 0; p < L; p++) {
        const t = ctx.tokens[p];
        const o = c * L + p;
        a.node_idxs[o] = i16(t.nodeIdx, 'node_idx');
        for (let j = 0; j < F2P; j++) a.f2p_nbr_idxs[o * F2P + j] = i16(t.f2pNbrIdxs[j], 'f2p_nbr_idx');
        a.table_name_idxs[o] = i16(t.tableNameIdx, 'table_name_idx');
        a.col_name_idxs[o] = i16(t.colNameIdx, 'col_name_idx');
        a.sem_types[o] = t.sem;
        // One value slot per token (see header): the other two slots must be exactly 0.
        const own = t.sem === SemType.Number ? t.numberValue : t.sem === SemType.DateTime ? t.datetimeValue : t.sem === SemType.Boolean ? t.booleanValue : 0;
        const slots = [t.numberValue, t.datetimeValue, t.booleanValue];
        const nonzeroOther = slots.filter((v) => v !== 0).length - (own !== 0 ? 1 : 0);
        if (nonzeroOther !== 0) throw new Error(`${name}: token ${p} of seed ${seed} has a non-zero value in another type's slot`);
        if (!Number.isFinite(own)) throw new Error(`${name}: non-finite value at token ${p} of seed ${seed}`);
        a.values[o] = own;
        a.text_idxs[o] = t.isPadding ? -1 : i16(t.textIdx, 'text_idx');
        a.flags[o] = (t.mask ? 1 : 0) | (t.isTarget ? 2 : 0) | (t.isPadding ? 4 : 0) | (t.isTaskNode ? 8 : 0);
        if (t.mask) nMasked++;
      }
      if (nMasked !== 1) throw new Error(`${name}: ${nMasked} masked cells (expected exactly 1, fly.rs:492-496)`);
      const tgt = ctx.tokens[ctx.targetPos];
      a.task[c] = te.idx;
      a.seed_node[c] = seed;
      a.epoch[c] = epoch;
      a.target_pos[c] = ctx.targetPos;
      a.label[c] = te.task.targetType === 'binary' ? tgt.booleanValue : tgt.numberValue;
      const key = `${te.idx}:${seed}`;
      let em = emCache.get(key);
      if (!em) emCache.set(key, (em = entityMean(te, seed)));
      a.entity_mean[c] = em.value;
      a.entity_n_past[c] = em.nPast;
      a.n_real[c] = ctx.numReal;
      const lab = inContextLabels(te.pdb, ctx, te.task);
      a.n_self_labels[c] = lab.self.length;
      a.n_other_labels[c] = lab.other.length;
      sumReal += ctx.numReal;
      if (ctx.numReal === L) nFull++;
      sumSelf += lab.self.length;
      sumOther += lab.other.length;
      c++;
    }
  }

  // Write all arrays back to back, each 8-byte aligned.
  const file = `${name.replace(/[/:]/g, '__')}.bin`;
  const fd = openSync(join(OUT, file), 'w');
  const arrays: Record<string, ArrayDesc> = {};
  let off = 0;
  const shapes: Record<string, number[]> = {
    node_idxs: [n, L],
    f2p_nbr_idxs: [n, L, F2P],
    table_name_idxs: [n, L],
    col_name_idxs: [n, L],
    sem_types: [n, L],
    values: [n, L],
    text_idxs: [n, L],
    flags: [n, L],
  };
  for (const [k, arr] of Object.entries(a)) {
    const dtype: DType =
      arr instanceof Int16Array ? 'int16' : arr instanceof Uint8Array ? 'uint8' : arr instanceof Float32Array ? 'float32' : arr instanceof Int32Array ? 'int32' : 'uint16';
    const buf = Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength);
    writeSync(fd, buf, 0, buf.length, off);
    arrays[k] = { dtype, shape: shapes[k] ?? [n], offset: off };
    off += buf.length;
    const pad = (8 - (off % 8)) % 8;
    if (pad) {
      writeSync(fd, Buffer.alloc(pad), 0, pad, off);
      off += pad;
    }
  }
  closeSync(fd);
  const taskIds = [...new Set(items.map((it) => it.te.task.id))];
  console.log(
    `[export] ${name.padEnd(36)} n=${String(n).padStart(6)}  epochs=${epochs}  tasks=${taskIds.length}  ` +
      `mean real tokens=${(sumReal / n).toFixed(1)}  self labels=${(sumSelf / n).toFixed(2)}  other=${(sumOther / n).toFixed(2)}  ` +
      `${(off / 1e6).toFixed(1)} MB`,
  );
  return {
    file,
    n,
    L,
    kind,
    tasks: taskIds,
    split,
    epochs,
    itemsPerEpoch: items.length,
    arrays,
    stats: {
      meanReal: sumReal / n,
      fracFull: nFull / n,
      meanSelfLabels: sumSelf / n,
      meanOtherLabels: sumOther / n,
      tokens: n * L,
      realTokens: sumReal,
    },
  };
}

function itemsOf(te: TaskEntry, split: Split): Item[] {
  return listSeeds(te.pdb, te.task, split).map((seed) => ({ te, seed }));
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
mkdirSync(OUT, { recursive: true });
const t0 = performance.now();
const sets: Record<string, SetDesc> = {};

// Pretraining: all tasks of the other DBs (example_pretrain.py:17), forecast → Train, autocomplete →
// all rows (listSeeds ignores the split for autocomplete, data.py:77-81).
const pretrainItems = tasks.filter((t) => PRETRAIN_DBS.includes(t.task.dbName as DbName)).flatMap((te) => itemsOf(te, 'Train'));
sets.pretrain = buildSet('pretrain', 'pretrain', pretrainItems, PRETRAIN_EPOCHS, 'Train');

// Zero-shot evaluation on the held-out DB: forecast val/test (example_pretrain.py:18), epoch 0.
for (const te of tasks.filter((t) => t.task.dbName === TARGET_DB)) {
  if (te.task.kind === 'forecast') {
    for (const split of ['Val', 'Test'] as Split[]) {
      const name = `eval/${te.task.id}/${split}`;
      sets[name] = buildSet(name, 'eval', itemsOf(te, split), 1, split);
    }
  } else {
    const name = `eval/${te.task.id}/All`;
    sets[name] = buildSet(name, 'eval', itemsOf(te, 'Train'), 1, 'All');
  }
}

// In-distribution diagnostics: forecast Val splits of the pretraining DBs (not used for selection).
for (const te of tasks.filter((t) => PRETRAIN_DBS.includes(t.task.dbName as DbName) && t.task.kind === 'forecast')) {
  const name = `diag/${te.task.id}/Val`;
  sets[name] = buildSet(name, 'diag', itemsOf(te, 'Val'), 1, 'Val');
}

// Fine-tuning: the target task's Train split only (example_finetune.py:18), several epochs.
for (const te of tasks.filter((t) => t.task.dbName === TARGET_DB && t.task.kind === 'forecast')) {
  const name = `finetune/${te.task.id}`;
  sets[name] = buildSet(name, 'finetune', itemsOf(te, 'Train'), FINETUNE_EPOCHS, 'Train');
}

const index = {
  format: 'rt-contexts-v1',
  generator: 'scripts/export-contexts.ts',
  createdAt: new Date().toISOString(),
  L,
  w: W,
  seed: SEED,
  rngSeedRule: 'contextSeed(epoch, seed_node_idx, seed) = epoch + seed_node_idx + seed (fly.rs:371-375)',
  targetDb: TARGET_DB,
  pretrainDbs: PRETRAIN_DBS,
  layout: {
    values: 'number_values = values·[sem=0], datetime_values = values·[sem=2], boolean_values = values·[sem=3]',
    text_idxs: 'per-DB text index; 0 for non-text cells (pre.rs), −1 for padding (text_values = 0⃗)',
    flags: 'bit0 masks, bit1 is_targets, bit2 is_padding, bit3 is_task_nodes',
  },
  embeddings: 'public/data/embeddings/minilm-l12-v2.{json,f32}: map dbs[db].text[i] (string) → embedding row',
  dbs: Object.fromEntries(
    DB_NAMES.map((d) => {
      const pdb = pdbs.get(d)!;
      return [d, { text: pdb.text, numNodes: pdb.nodes.length }];
    }),
  ),
  tasks: tasks.map((te) => ({
    idx: te.idx,
    id: te.task.id,
    db: te.task.dbName,
    name: te.task.name,
    kind: te.task.kind,
    targetType: te.task.targetType,
    targetCol: te.task.targetCol,
    columnsToDrop: te.task.columnsToDrop,
    targetColNameIdx: te.targetColNameIdx,
    targetStats: te.stats,
    trainMean: te.trainMean,
    seeds: {
      Train: listSeeds(te.pdb, te.task, 'Train').length,
      Val: listSeeds(te.pdb, te.task, 'Val').length,
      Test: listSeeds(te.pdb, te.task, 'Test').length,
    },
  })),
  sets,
};
writeFileSync(join(OUT, 'index.json'), JSON.stringify(index, null, 1));
const total = Object.values(sets).reduce((s, x) => s + x.n, 0);
console.log(`[export] ${total} contexts in ${Object.keys(sets).length} sets → ${OUT} (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
