/**
 * Integration of the hooks' pure helpers with the engine (BUILD_PLAN §2, §7):
 * db → preprocess → seeds → default seed → context → masks → forward.
 */
import { describe, expect, it } from 'vitest';
import { getDatabase } from '@/engine/db/registry';
import { loadEmbeddingStore } from '@/engine/embeddings';
import { MINI_CONFIG } from '@/engine/model/config';
import { initWeights, loadWeights } from '@/engine/model/weights';
import { contextSeed } from '@/engine/rng';
import {
  findTask,
  getContext,
  getForward,
  getMasks,
  getPreprocessed,
  getSeeds,
  loadAssets,
  makeSampleParams,
  pickDefaultSeed,
  resolveCheckpoint,
} from './hooks';

describe('hooks helpers', () => {
  const db = getDatabase('shop');
  const pdb = getPreprocessed(db);
  const task = findTask(db, 'shop/customer-churn');

  it('caches the preprocessed DB per RawDatabase', () => {
    expect(getPreprocessed(db)).toBe(pdb);
  });

  it('finds tasks and falls back to the first task', () => {
    expect(task.id).toBe('shop/customer-churn');
    expect(findTask(db, 'shop/nope').id).toBe(db.tasks[0].id);
  });

  it('default Val seed is a Val row of customer-churn with several past labels', () => {
    const seeds = getSeeds(pdb, task, 'Val');
    expect(seeds.length).toBeGreaterThan(0);
    const seed = pickDefaultSeed(pdb, db, task, 'Val', seeds);
    expect(seeds).toContain(seed);
    const node = pdb.nodes[seed];
    expect(node.tableKey).toBe('customer-churn:Val');
    const entity = node.f2pNbrIdxs[0];
    const past = pdb.p2fAdj[entity].filter(
      (e) => e.tableType !== 'Db' && e.tableNameIdx === node.tableNameIdx && e.ts !== null && node.ts !== null && e.ts < node.ts,
    );
    expect(past.length).toBeGreaterThanOrEqual(3);
    expect(pickDefaultSeed(pdb, db, task, 'Val', seeds)).toBe(seed); // deterministic
  });

  it('autocomplete seeds ignore the split', () => {
    const ac = db.tasks.find((t) => t.kind === 'autocomplete')!;
    expect(getSeeds(pdb, ac, 'Train')).toBe(getSeeds(pdb, ac, 'Test'));
  });

  it('builds eval-style SampleParams (epoch 0 + seed node + seed, fly.rs:371-375)', () => {
    const seed = getSeeds(pdb, task, 'Val')[0];
    const p = makeSampleParams(pdb, task, seed, 128, 16, 0);
    expect(p.rngSeed).toBe(contextSeed(0, seed, 0));
    expect(p.dropColNameIdxs).toEqual([]);
    const ctx = getContext(pdb, p);
    expect(ctx.tokens.length).toBe(128);
    expect(getContext(pdb, { ...p })).toBe(ctx); // cached by value
    expect(ctx.tokens.filter((t) => t.isTarget).length).toBe(1);
    const m = getMasks(ctx);
    expect(getMasks(ctx)).toBe(m);
    expect(m.col.length).toBe(128 * 128);
  });

  it('resolves the auto checkpoint per task', () => {
    expect(resolveCheckpoint('shop/customer-churn', 'auto')).toBe('pretrain-churn');
    expect(resolveCheckpoint('shop/product-sales', 'auto')).toBe('pretrain-sales');
    expect(resolveCheckpoint('shop/product-sales', 'finetune-churn')).toBe('finetune-churn');
  });

  it('runs a cached forward pass on the live context (random init weights)', async () => {
    const store = await loadEmbeddingStore();
    const weights = initWeights(MINI_CONFIG, 0);
    const seeds = getSeeds(pdb, task, 'Val');
    const ctx = getContext(pdb, makeSampleParams(pdb, task, pickDefaultSeed(pdb, db, task, 'Val', seeds), 128, 16, 0));
    const r = getForward(ctx, pdb, store, weights);
    expect(r.S).toBe(128);
    expect(Number.isFinite(r.prediction)).toBe(true);
    expect(getForward(ctx, pdb, store, weights, [])).toBe(r);
    const ablated = getForward(ctx, pdb, store, weights, ['full']);
    expect(ablated).not.toBe(r);
    expect(getForward(ctx, pdb, store, weights, ['full'])).toBe(ablated);
  });

  it('loads assets with a weightsKind; missing trained checkpoints fall back to untrained init weights', async () => {
    const st = await loadAssets('pretrain-churn');
    expect(st.status).toBe('ready');
    expect(st.store?.dim).toBe(MINI_CONFIG.dText);
    const exported = await loadWeights('pretrain-churn').then(
      () => true,
      () => false,
    );
    expect(st.weightsKind).toBe(exported ? 'trained' : 'init');
    expect(st.fallback).toBe(!exported);
    expect(st.weights?.config.dModel).toBe(MINI_CONFIG.dModel);
    expect(await loadAssets('pretrain-churn')).toBe(st); // cached
  });
});
