/**
 * Find seed rows whose context window holds labels of OTHER entities (spec §2.5), so the label-share
 * chart can contrast own vs other labels. Scans every seed of the current task split with the real
 * sampler (fly.rs mirror, no event trace), in small idle-time chunks with progress, and caches the
 * result per (db, task, split, L, w, rngSeed). Nothing is hard-coded: the seeds come from the live scan.
 */
import { useCallback, useEffect, useState } from 'react';
import { getSeeds, inContextLabels, makeSampleParams } from '@/engine/pipeline';
import { sampleContext } from '@/engine/sampler';
import { useSim } from '@/state/store';
import type { AttnLive } from './useAttnLive';

export interface SeedHit {
  seed: number;
  self: number;
  other: number;
}
export interface ScanState {
  status: 'idle' | 'scanning' | 'done';
  done: number;
  total: number;
  hits: SeedHit[];
}

const cache = new Map<string, ScanState>();
const CHUNK = 6;

export function useOtherLabelSeeds(live: AttnLive) {
  const { pdb, task } = live;
  const split = useSim((s) => s.split);
  const L = live.ctx.params.L;
  const w = live.ctx.params.w;
  const rngSeed = useSim((s) => s.rngSeed);
  const key = `${task.id}|${split}|${L}|${w}|${rngSeed}`;
  const [state, setState] = useState<ScanState>(() => cache.get(key) ?? { status: 'idle', done: 0, total: 0, hits: [] });
  const [run, setRun] = useState(0);

  useEffect(() => {
    setState(cache.get(key) ?? { status: 'idle', done: 0, total: 0, hits: [] });
  }, [key]);

  useEffect(() => {
    if (!run) return;
    const cached = cache.get(key);
    if (cached?.status === 'done') {
      setState(cached);
      return;
    }
    if (task.kind !== 'forecast') {
      const st: ScanState = { status: 'done', done: 0, total: 0, hits: [] };
      cache.set(key, st);
      setState(st);
      return;
    }
    const seeds = getSeeds(pdb, task, split);
    const hits: SeedHit[] = [];
    let i = 0;
    let alive = true;
    let timer = 0;
    setState({ status: 'scanning', done: 0, total: seeds.length, hits: [] });
    const step = () => {
      if (!alive) return;
      const end = Math.min(seeds.length, i + CHUNK);
      for (; i < end; i++) {
        const params = makeSampleParams(pdb, task, seeds[i], L, w, rngSeed);
        const ctx = sampleContext(pdb, params, { recordEvents: false });
        const labs = inContextLabels(pdb, ctx, task);
        if (labs.other.length > 0) hits.push({ seed: seeds[i], self: labs.self.length, other: labs.other.length });
      }
      if (i < seeds.length) {
        setState({ status: 'scanning', done: i, total: seeds.length, hits: [...hits] });
        timer = window.setTimeout(step, 0);
      } else {
        const st: ScanState = { status: 'done', done: seeds.length, total: seeds.length, hits };
        cache.set(key, st);
        setState(st);
      }
    };
    timer = window.setTimeout(step, 0);
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, key]);

  const start = useCallback(() => setRun((r) => r + 1), []);
  return { ...state, start };
}
