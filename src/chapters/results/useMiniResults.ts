/**
 * This site's own training results (public/data/training/results.json, written by python/train.py).
 * Loaded once at runtime, so the page always shows the numbers of the committed run; every field is
 * optional here because the file is produced by a separate pipeline.
 */
import { useEffect, useState } from 'react';
import { loadPublicJson } from '@/engine/model/io';

export interface MiniSplitMetric {
  step?: number;
  val?: number;
  test?: number;
}

export interface MiniZeroShot {
  metric?: 'auroc' | 'r2';
  pretrainBest?: MiniSplitMetric;
  untrainedInit?: MiniSplitMetric;
  entityMean?: { val?: number; test?: number } | null;
}

export interface MiniResults {
  config?: { d_model?: number; num_blocks?: number; num_heads?: number; L?: number; w?: number };
  pretrain?: { steps?: number };
  zeroShot?: Record<string, MiniZeroShot>;
  diagnostics?: {
    inContextLabelAblation?: Record<string, Record<string, { with?: number; withoutLabels?: number }>>;
  };
  labelPersistence?: { tasks?: Record<string, { splits?: Record<string, { rows?: number }> }> };
}

let cache: MiniResults | null = null;
let pending: Promise<MiniResults> | null = null;

export function useMiniResults(): { data: MiniResults | null; error: string | null } {
  const [data, setData] = useState<MiniResults | null>(cache);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (cache) return;
    let alive = true;
    pending ??= loadPublicJson<MiniResults>('data/training/results.json').then((d) => (cache = d));
    pending.then(
      (d) => alive && setData(d),
      (e: unknown) => {
        pending = null;
        if (alive) setError(e instanceof Error ? e.message : String(e));
      },
    );
    return () => {
      alive = false;
    };
  }, []);
  return { data, error };
}
