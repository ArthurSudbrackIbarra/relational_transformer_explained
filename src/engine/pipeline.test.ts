/**
 * The engine glue (pipeline.ts): db → preprocess → seeds → context → masks → forward, the
 * checkpoint fallback, and the two read-outs (prediction, in-context labels).
 */
import { describe, expect, it } from 'vitest';
import { getDatabase, pickShowcaseEntity } from './db/registry';
import { loadEmbeddingStore } from './embeddings';
import { buildMasks } from './masks';
import { denormalize, sigmoid } from './metrics';
import { forward } from './model/forward';
import { loadWeights } from './model/weights';
import {
  CHECKPOINT_INFO,
  CHECKPOINT_NAMES,
  autoCheckpoint,
  fallbackInitWeights,
  getContext,
  getForward,
  getMasks,
  getPreprocessed,
  getSeeds,
  inContextLabels,
  loadCheckpoint,
  makeSampleParams,
  pickDefaultSeed,
  readPrediction,
  resolveCheckpoint,
  runPipeline,
  simulate,
} from './pipeline';
import { DB_NAMES } from './db/tasks';
import { targetStats } from './sampler';

/** Whether public/data/weights/<name> exists and parses (trained weights may be exported later). */
const hasCheckpoint = (name: string) => loadWeights(name).then(
  () => true,
  () => false,
);

describe('pipeline: database → context', () => {
  it('preprocesses each database once', () => {
    const db = getDatabase('shop');
    expect(getPreprocessed(db)).toBe(getPreprocessed(db));
  });

  it('default seed of shop/customer-churn Val is the showcase customer, with many self labels in its context', () => {
    const sim = simulate({ dbName: 'shop', taskId: 'shop/customer-churn', split: 'Val' });
    const showcase = pickShowcaseEntity(sim.db, 'shop/customer-churn');
    const valOffset = sim.pdb.tableInfo['customer-churn:Val'].nodeIdxOffset;
    expect(sim.seedNode).toBe(valOffset + showcase.rows.Val[0]);

    // the seed row's single FK is the showcase customer
    const seed = sim.pdb.nodes[sim.seedNode];
    const customers = sim.pdb.tableInfo['customers:Db'];
    expect(seed.f2pNbrIdxs).toEqual([customers.nodeIdxOffset + showcase.entityRow]);

    // L = 128, w = 16 (the mini defaults): the masked target is the seed's churn cell …
    expect(sim.ctx.tokens.length).toBe(128);
    expect(sim.ctx.tokens[sim.ctx.targetPos].nodeIdx).toBe(sim.seedNode);
    // … and the context shows several of the customer's own earlier labels, of both classes.
    const labels = inContextLabels(sim.pdb, sim.ctx, sim.task);
    expect(labels.entityNode).toBe(seed.f2pNbrIdxs[0]);
    expect(labels.self.length).toBeGreaterThanOrEqual(8);
    expect(labels.self.length).toBeLessThanOrEqual(showcase.rows.Train.length);
    const values = new Set(labels.self.map((l) => l.raw));
    expect(values.has(true) && values.has(false)).toBe(true);
    // a Val seed sees Train labels (ts < seed ts) of its own entity; its own Val row is the target
    for (const l of labels.self) expect(l.tableType).toBe('Train');
  });

  it('classifies every visible target-column cell exactly once', () => {
    for (const taskId of ['shop/customer-churn', 'shop/product-sales']) {
      const sim = simulate({ dbName: 'shop', taskId, split: 'Test' });
      for (const seedNode of sim.seeds.slice(0, 20)) {
        const s = simulate({ dbName: 'shop', taskId, split: 'Test', seedNode });
        const lab = inContextLabels(s.pdb, s.ctx, s.task);
        const visible = s.ctx.tokens.filter((t) => !t.isPadding && !t.isTarget && t.colNameIdx === s.params.targetColNameIdx);
        expect(lab.self.length + lab.other.length).toBe(visible.length);
        for (const l of lab.self) expect(s.pdb.nodes[l.nodeIdx].f2pNbrIdxs[0]).toBe(lab.entityNode);
        for (const l of lab.other) expect(s.pdb.nodes[l.nodeIdx].f2pNbrIdxs[0]).not.toBe(lab.entityNode);
      }
    }
  });

  it('autocomplete contexts have no task rows and no "self" labels', () => {
    const sim = simulate({ dbName: 'shop', taskId: 'shop/reviews.rating', split: 'Train' });
    expect(sim.task.kind).toBe('autocomplete');
    // every row is a DB row (fly.rs:417 rejects all task-table children). Note that the token flag
    // is_task_node is still true on target-column cells (fly.rs:498-499), so it is not checked here.
    expect(sim.ctx.tokens.every((t) => t.isPadding || t.tableType === 'Db')).toBe(true);
    const lab = inContextLabels(sim.pdb, sim.ctx, sim.task);
    expect(lab.entityNode).toBeNull();
    expect(lab.self).toEqual([]);
  });

  it('replaces an invalid seed by the default and caches contexts and masks by value', () => {
    const a = simulate({ dbName: 'shop', taskId: 'shop/customer-churn', split: 'Val', seedNode: 0 }); // 0 is a customers row
    const b = simulate({ dbName: 'shop', taskId: 'shop/customer-churn', split: 'Val' });
    expect(a.seedNode).toBe(b.seedNode);
    expect(a.ctx).toBe(b.ctx);
    expect(a.masks).toBe(getMasks(b.ctx));
    const p = makeSampleParams(a.pdb, a.task, a.seedNode, 128, 16, 0);
    expect(getContext(a.pdb, { ...p, dropColNameIdxs: [...p.dropColNameIdxs] })).toBe(a.ctx);
    // a different RNG seed term gives a different (cached) context
    expect(getContext(a.pdb, makeSampleParams(a.pdb, a.task, a.seedNode, 128, 16, 1))).not.toBe(a.ctx);
  });

  it('every task of every database has seeds and a default seed in each split', () => {
    for (const name of DB_NAMES) {
      const db = getDatabase(name);
      const pdb = getPreprocessed(db);
      for (const task of db.tasks) {
        for (const split of ['Train', 'Val', 'Test'] as const) {
          const seeds = getSeeds(pdb, task, split);
          expect(seeds.length, `${task.id} ${split}`).toBeGreaterThan(0);
          expect(seeds).toContain(pickDefaultSeed(pdb, db, task, split, seeds));
        }
      }
    }
  });
});

describe('pipeline: checkpoints and forward', () => {
  it('resolves checkpoints', () => {
    expect(resolveCheckpoint('shop/customer-churn', 'auto')).toBe('pretrain-churn');
    expect(resolveCheckpoint('shop/product-sales', 'auto')).toBe('pretrain-sales');
    expect(resolveCheckpoint('shop/reviews.rating', 'auto')).toBe('pretrain-rating');
    expect(resolveCheckpoint('shop/reviews.verified', 'auto')).toBe('pretrain-verified');
    expect(resolveCheckpoint('shop/product-sales', 'init')).toBe('init');
    expect(resolveCheckpoint('school/student-dropout', 'finetune-churn')).toBe('finetune-churn');
    expect(() => resolveCheckpoint('shop/product-sales', 'nope')).toThrow();
  });

  it("'auto' is zero-shot on every ShopDB task and the final checkpoint (NOT zero-shot) on the pretraining DBs", () => {
    for (const name of DB_NAMES) {
      for (const task of getDatabase(name).tasks) {
        const a = autoCheckpoint(task.id);
        expect(resolveCheckpoint(task.id, 'auto')).toBe(a.name);
        expect(CHECKPOINT_NAMES).toContain(a.name);
        expect(a.note.length).toBeGreaterThan(20);
        if (name === 'shop') {
          expect(a.zeroShot, task.id).toBe(true);
          expect(a.name).toMatch(/^pretrain-(churn|sales|rating|verified)$/);
        } else {
          expect(a.zeroShot, task.id).toBe(false);
          expect(a.name).toBe('pretrain-final');
          expect(a.note).toMatch(/not zero-shot/i);
        }
      }
    }
    // every ShopDB task has its own checkpoint
    expect(new Set(getDatabase('shop').tasks.map((t) => autoCheckpoint(t.id).name)).size).toBe(getDatabase('shop').tasks.length);
    for (const n of CHECKPOINT_NAMES) expect(CHECKPOINT_INFO[n]).toBeTruthy();
  });

  it('ships every checkpoint of CHECKPOINT_NAMES (python/train.py export), so no task falls back to init', async () => {
    for (const name of CHECKPOINT_NAMES) expect(await hasCheckpoint(name), name).toBe(true);
  });

  it('falls back to untrained init weights when a checkpoint file is missing', async () => {
    for (const name of CHECKPOINT_NAMES) {
      const c = await loadCheckpoint(name);
      expect(c.name).toBe(name);
      if (await hasCheckpoint(name)) {
        expect(c.fallback).toBe(false);
        expect(c.kind).toBe(name === 'init' ? 'init' : 'trained');
      } else {
        expect(c.fallback).toBe(true);
        expect(c.kind).toBe('init');
        expect(c.weights).toBe(fallbackInitWeights()); // one shared object → shared forward cache
        expect(c.error).toBeTruthy();
      }
    }
  });

  it('getForward equals model forward() on the live context, reuses the cached masks, and caches ablations', async () => {
    const sim = simulate({ dbName: 'shop', taskId: 'shop/customer-churn', split: 'Val' });
    const store = await loadEmbeddingStore();
    const w = fallbackInitWeights();
    const r = getForward(sim.ctx, sim.pdb, store, w);
    expect(r.masks).toBe(sim.masks);
    expect(getForward(sim.ctx, sim.pdb, store, w, [])).toBe(r);
    const direct = forward(sim.ctx.tokens, sim.pdb, store, w);
    expect(r.prediction).toBe(direct.prediction);
    expect(r.loss).toBe(direct.loss);
    for (const l of ['col', 'feat', 'nbr', 'full'] as const) expect(r.masks[l]).toEqual(buildMasks(sim.ctx.tokens)[l]);
    const ab = getForward(sim.ctx, sim.pdb, store, w, ['nbr', 'col']);
    expect(ab).not.toBe(r);
    expect(getForward(sim.ctx, sim.pdb, store, w, ['col', 'nbr'])).toBe(ab); // order-insensitive key
    expect(ab.blocks[0].attn.col.skipped).toBe(true);
  });

  it('reads binary and regression predictions (spec §8.3-8.4)', async () => {
    const store = await loadEmbeddingStore();
    const w = fallbackInitWeights();

    const bin = simulate({ dbName: 'shop', taskId: 'shop/customer-churn', split: 'Val' });
    const fb = getForward(bin.ctx, bin.pdb, store, w);
    const pb = readPrediction(fb, bin.ctx, bin.pdb, bin.task)!;
    expect(pb.taskKind).toBe('binary');
    expect(pb.output).toBe(fb.prediction);
    expect(pb.probability).toBeCloseTo(sigmoid(fb.prediction), 12);
    expect(pb.label.binary).toBe(bin.ctx.tokens[pb.targetPos].raw === true ? 1 : 0);

    const reg = simulate({ dbName: 'shop', taskId: 'shop/product-sales', split: 'Val' });
    const fr = getForward(reg.ctx, reg.pdb, store, w);
    const pr = readPrediction(fr, reg.ctx, reg.pdb, reg.task)!;
    expect(pr.taskKind).toBe('regression');
    const st = targetStats(reg.pdb, reg.task);
    expect(pr.value).toBeCloseTo(denormalize(fr.prediction, st), 9);
    expect(denormalize(pr.label.z!, st)).toBeCloseTo(reg.ctx.tokens[pr.targetPos].raw as number, 4);
  });

  it('runs end to end on every database', async () => {
    for (const name of DB_NAMES) {
      const db = getDatabase(name);
      const run = await runPipeline({ dbName: name, taskId: db.tasks[0].id });
      expect(run.forward.S).toBe(128);
      expect(Number.isFinite(run.forward.prediction), `${name}`).toBe(true);
      expect(run.prediction).not.toBeNull();
      expect(run.assets.store.dim).toBe(run.assets.checkpoint.weights.config.dText);
    }
  });
});
