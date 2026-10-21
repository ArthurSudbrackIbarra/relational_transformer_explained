/**
 * Data loaders of the "about" chapter. Nothing here is typed in by hand: every number comes from a file
 * that the offline pipeline writes (public/data/**), from the repository's own sources (read raw), or
 * from the live engine.
 *
 *   verification.json   python/write_verification.py (parity, sampler cross-check, test counts, sha256)
 *   training/dataset.json, training/results.json   python/train.py, scripts/export-contexts.ts
 */
import { useEffect, useState } from 'react';
import { loadPublicJson } from '@/engine/model/io';

// ─────────────────────────────────────────────────────────────────────────────
// JSON shapes (only the fields this chapter reads)
// ─────────────────────────────────────────────────────────────────────────────

export interface VerificationCheckpoint {
  name: string;
  bytes: number;
  sha256: string;
  step: number | null;
  selection: string | null;
  officialVsPortMaxAbsDiff: number | null;
}

export interface TsFixtureReport {
  maxAbsDiff: number;
  contexts: number;
  checkpoint?: string;
  worstStage?: string;
}

export interface VerificationJson {
  official_vs_port_max_abs_diff: number;
  ts_vs_torch_max_abs_diff: number;
  n_contexts: number;
  checkpoints: string[];
  sampler_crosscheck: { contexts: number; mismatches: number; note: string };
  tests: number;
  date: string;
  details: {
    note: string;
    officialVsPort: {
      randomWeights: number;
      perCheckpoint: Record<string, number>;
      trainedFixtures: Record<string, number>;
      attention?: string;
    };
    tsVsTorch: Record<string, TsFixtureReport>;
    samplerCrosscheck: { contexts: number; listed?: number; mismatches: number; how: string };
    testsTotal: number;
    testsFailed: number;
    /** vitest's numTotalTestSuites (describe blocks), despite the name. */
    testFiles: number;
    checkpoints: VerificationCheckpoint[];
    torch?: string;
  };
}

export interface DatasetJson {
  L: number;
  w: number;
  batchSize: number;
  pretrain: {
    dbs: string[];
    heldOut: string;
    numTasks: number;
    itemsPerEpoch: number;
    exportedEpochs: number;
    exportedContexts: number;
    steps: number;
    contextsSeen: number;
    tokensSeen: number;
    epochsSeen: number;
  };
  paper: { steps: number; batch: number; L: number; contextsSeen: number; tokensSeen: number };
  sets: Record<string, { contexts: number; tokens: number; split: string }>;
}

export interface ResultsJson {
  config?: {
    num_blocks: number;
    d_model: number;
    num_heads: number;
    d_ff: number;
    eps: number;
    batch: number;
    dtype: string;
    device: string;
    threads?: number;
    attention?: { kernel: string; maxAbsDiffForward: number; maxAbsDiffGrad: number };
  };
  pretrain?: { steps: number; evalFreq: number; wallSeconds: number; stepSeconds: number };
}

// ─────────────────────────────────────────────────────────────────────────────
// A tiny cached JSON hook (one fetch per file per page load; failures are retried)
// ─────────────────────────────────────────────────────────────────────────────

const jsonCache = new Map<string, Promise<unknown>>();

function loadJsonOnce<T>(rel: string): Promise<T> {
  let p = jsonCache.get(rel);
  if (!p) {
    p = loadPublicJson<T>(rel);
    p.catch(() => jsonCache.delete(rel));
    jsonCache.set(rel, p);
  }
  return p as Promise<T>;
}

export interface Loaded<T> {
  data: T | null;
  error: string | null;
}

export function usePublicJson<T>(rel: string): Loaded<T> {
  const [state, setState] = useState<Loaded<T>>({ data: null, error: null });
  useEffect(() => {
    let alive = true;
    loadJsonOnce<T>(rel).then(
      (data) => alive && setState({ data, error: null }),
      (e: unknown) => alive && setState({ data: null, error: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      alive = false;
    };
  }, [rel]);
  return state;
}

export const useVerification = () => usePublicJson<VerificationJson>('data/verification.json');
export const useDataset = () => usePublicJson<DatasetJson>('data/training/dataset.json');
export const useResults = () => usePublicJson<ResultsJson>('data/training/results.json');

// ─────────────────────────────────────────────────────────────────────────────
// Raw repository sources, loaded lazily (Vite `?raw`; nothing is copied by hand)
// ─────────────────────────────────────────────────────────────────────────────

const rawCache = new Map<string, Promise<string>>();

const RAW_LOADERS: Record<string, () => Promise<string>> = {
  spec: () => import('../../../research/RT_TECHNICAL_SPEC.md?raw').then((m) => m.default),
  pythonReadme: () => import('../../../python/README.md?raw').then((m) => m.default),
  rtModelPy: () => import('../../../python/rt_model.py?raw').then((m) => m.default),
  opsTs: () => import('../../engine/model/ops.ts?raw').then((m) => m.default),
  dbTest: () => import('../../engine/db/db.test.ts?raw').then((m) => m.default),
};
export type RawSource = keyof typeof RAW_LOADERS;

export function useRawSource(name: RawSource): Loaded<string> {
  const [state, setState] = useState<Loaded<string>>({ data: null, error: null });
  useEffect(() => {
    let alive = true;
    let p = rawCache.get(name);
    if (!p) {
      p = RAW_LOADERS[name]();
      p.catch(() => rawCache.delete(name));
      rawCache.set(name, p);
    }
    p.then(
      (data) => alive && setState({ data, error: null }),
      (e: unknown) => alive && setState({ data: null, error: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      alive = false;
    };
  }, [name]);
  return state;
}

/** Lines [start, end] (1-based, inclusive) of a text. */
export function sliceLines(text: string, start: number, end: number): string {
  return text.split('\n').slice(start - 1, end).join('\n');
}

/**
 * Find a block in a source text: the first line matching `startRe` through the first later line matching
 * `endRe` (inclusive). Returns 1-based line numbers so the UI can label the excerpt exactly.
 */
export function findBlock(text: string, startRe: RegExp, endRe: RegExp): { start: number; end: number; code: string } | null {
  const lines = text.split('\n');
  const s = lines.findIndex((l) => startRe.test(l));
  if (s < 0) return null;
  for (let i = s + 1; i < lines.length; i++) {
    if (endRe.test(lines[i])) return { start: s + 1, end: i + 1, code: lines.slice(s, i + 1).join('\n') };
  }
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Number formatting
// ─────────────────────────────────────────────────────────────────────────────

const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };

/** 1.76e-6 → "1.8 × 10⁻⁶" (0 → "0"). */
export function sci(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '–';
  if (v === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(v)));
  let m = v / 10 ** e;
  let ee = e;
  if (Math.abs(Number(m.toFixed(digits))) >= 10) {
    m /= 10;
    ee += 1;
  }
  const exp = String(ee)
    .split('')
    .map((c) => SUP[c] ?? c)
    .join('');
  return `${m.toFixed(digits)} × 10${exp}`;
}

/** Thousands separators. */
export const int = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? '–' : Math.round(v).toLocaleString('en-US'));

/** Bytes → "2.2 MB". */
export function bytes(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} kB`;
  return `${n} B`;
}

/**
 * Spacing between adjacent float32 numbers at magnitude |x| (one "unit in the last place"):
 * 2^(⌊log2|x|⌋ − 23) for normal numbers.
 */
export function ulp32(x: number): number {
  const a = Math.abs(x);
  if (!Number.isFinite(a) || a === 0) return 2 ** -149;
  return 2 ** (Math.floor(Math.log2(a)) - 23);
}

/** float32 machine epsilon 2⁻²³. */
export const F32_EPS = 2 ** -23;
