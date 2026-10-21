/**
 * Shared helpers for the dummy-database generators and the task builder.
 *
 * Two concerns live here:
 *
 * 1. **Cross-engine determinism.** The plan (BUILD_PLAN §3) requires the generators to give the
 *    same output on every load, in the browser and in Node. ECMAScript only *approximates*
 *    `Math.exp/log/cos/pow` ("implementation-approximated"), so two engines may differ in the
 *    last bit, which could flip a threshold draw and change a cell. The basic IEEE-754 operations
 *    (+ − × ÷) and `Math.sqrt`/`Math.round`/`Math.floor` are exact everywhere. We therefore draw
 *    only uniforms from `Rng` (rng.ts; its uniform/int/choice paths use exact arithmetic) and
 *    build every other distribution here on top of `detExp`/`detLog`, which use only exact ops.
 *    (`Rng.normal/poisson/gamma` call Math.log/cos/exp, so the generators do not use them.)
 *
 * 2. **Time and table plumbing**: UTC epoch-ms helpers, the global data span, and a `makeTable`
 *    builder that enforces the RelBench/RT data invariants (PK = row position, spec §1.2;
 *    ≤5 FKs, spec §1.2/§4.16; rows keyed in column order).
 */
import type { ColumnSchema, DType, RawTable, RawValue, TableSchema } from '../types';
import { Rng } from '../rng';

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic elementary functions (exact IEEE ops only)
// ─────────────────────────────────────────────────────────────────────────────

const LN2 = 0.6931471805599453;
// Cody–Waite split of ln 2: LN2_HI has trailing zero bits so k·LN2_HI is exact for |k| < 2^11.
const LN2_HI = 0.693145751953125;
const LN2_LO = 1.4286068203094173e-6;
const SQRT2 = 1.4142135623730951;

/** Exact power of two 2^k by repeated multiplication (exact while in the normal range). */
function pow2(k: number): number {
  let s = 1;
  const base = k >= 0 ? 2 : 0.5;
  const n = k >= 0 ? k : -k;
  for (let i = 0; i < n; i++) s *= base;
  return s;
}

/** e^x using only + − × ÷ (≈1 ulp accurate; bit-identical on every IEEE-754 engine). */
export function detExp(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x > 709) return Infinity;
  if (x < -745) return 0;
  const k = Math.round(x / LN2);
  const r = x - k * LN2_HI - k * LN2_LO; // |r| ≤ ~0.347
  // Taylor series of e^r, Horner form, 22 terms (error < 1e-20 for |r| ≤ 0.35)
  let sum = 1;
  for (let n = 22; n >= 1; n--) sum = 1 + (r / n) * sum;
  // split the scaling so huge |k| cannot overflow/underflow an intermediate
  if (k > 1000) return sum * pow2(1000) * pow2(k - 1000);
  if (k < -1000) return sum * pow2(-1000) * pow2(k + 1000);
  return sum * pow2(k);
}

/** Natural log using only + − × ÷ (bit-identical on every IEEE-754 engine). */
export function detLog(x: number): number {
  if (Number.isNaN(x) || x < 0) return NaN;
  if (x === 0) return -Infinity;
  if (x === Infinity) return Infinity;
  // x = m · 2^e with m in [1/√2, √2) — multiplications by 2 / 0.5 are exact
  let m = x;
  let e = 0;
  while (m >= 2) {
    m *= 0.5;
    e++;
  }
  while (m < 1) {
    m *= 2;
    e--;
  }
  if (m > SQRT2) {
    m *= 0.5;
    e++;
  }
  // log m = 2·atanh(s), s = (m−1)/(m+1), |s| ≤ 0.1716 → series to s^41 is far below 1 ulp
  const s = (m - 1) / (m + 1);
  const s2 = s * s;
  let series = 0;
  for (let n = 41; n >= 1; n -= 2) series = 1 / n + s2 * series;
  return e * LN2_HI + e * LN2_LO + 2 * s * series;
}

/** Logistic sigmoid via detExp. */
export function sigmoid(x: number): number {
  return x >= 0 ? 1 / (1 + detExp(-x)) : detExp(x) / (1 + detExp(x));
}

export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/** Round to `digits` decimals (exact-op implementation; avoids Math.pow). */
export function round(x: number, digits = 0): number {
  let f = 1;
  for (let i = 0; i < digits; i++) f *= 10;
  return Math.round(x * f) / f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic distributions on top of Rng uniforms
// ─────────────────────────────────────────────────────────────────────────────

/** Standard normal via the Marsaglia polar method (uses detLog + sqrt only). */
export function normal(rng: Rng, mean = 0, std = 1): number {
  for (;;) {
    const u = 2 * rng.next() - 1;
    const v = 2 * rng.next() - 1;
    const s = u * u + v * v;
    if (s > 0 && s < 1) return mean + std * u * Math.sqrt((-2 * detLog(s)) / s);
  }
}

/** Poisson(λ) by Knuth's product method (λ is small in all our generators). */
export function poisson(rng: Rng, lambda: number): number {
  if (!(lambda > 0)) return 0;
  if (lambda > 40) return Math.max(0, Math.round(normal(rng, lambda, Math.sqrt(lambda))));
  const L = detExp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k++;
    p *= rng.next();
  } while (p > L);
  return k - 1;
}

/** Gamma(shape, scale) by Marsaglia–Tsang (deterministic variant). */
export function gamma(rng: Rng, shape: number, scale = 1): number {
  if (shape < 1) {
    // boost: Gamma(a) = Gamma(a+1) · U^(1/a)  →  exp(log(U)/a)
    const u = rng.next() || 1e-12;
    return gamma(rng, shape + 1, scale) * detExp(detLog(u) / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      x = normal(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng.next();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v * scale;
    if (u > 0 && detLog(u) < 0.5 * x * x + d * (1 - v + detLog(v))) return d * v * scale;
  }
}

/** Log-normal: exp(N(mu, sigma)). */
export function logNormal(rng: Rng, mu: number, sigma: number): number {
  return detExp(normal(rng, mu, sigma));
}

/** Index drawn from a categorical distribution (weights need not be normalized). */
export function categorical(rng: Rng, weights: readonly number[]): number {
  let total = 0;
  for (const w of weights) total += w;
  let r = rng.next() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r < 0) return i;
  }
  return weights.length - 1;
}

/** A sub-stream: independent Rng for one part of a generator, so parts can be tuned separately. */
export function subRng(baseSeed: number, stream: number): Rng {
  // simple deterministic mixing of (seed, stream) into one u32 seed
  return new Rng((Math.imul(baseSeed ^ 0x9e3779b9, 0x85ebca6b) + Math.imul(stream + 1, 0xc2b2ae35)) >>> 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Time helpers (epoch MILLISECONDS, UTC; types.ts RawValue convention)
// ─────────────────────────────────────────────────────────────────────────────

export const SECOND_MS = 1000;
export const MINUTE_MS = 60_000;
export const HOUR_MS = 3_600_000;
export const DAY_MS = 86_400_000;

/** UTC epoch ms; month is 1-based. */
export function utc(y: number, m: number, d: number, hh = 0, mi = 0, ss = 0): number {
  return Date.UTC(y, m - 1, d, hh, mi, ss);
}

/** Global data span of all four DBs (BUILD_PLAN §3): 2024-01-01 .. 2025-06-30 (inclusive). */
export const DATA_START = utc(2024, 1, 1);
export const DATA_END = utc(2025, 6, 30, 23, 59, 59);

/** Forecast cutoffs shared by all DBs (BUILD_PLAN §3). RacingDB overrides val (see tasks.ts). */
export const VAL_TIMESTAMP = utc(2025, 3, 1);
export const TEST_TIMESTAMP = utc(2025, 4, 15);

/** Number of calendar months in the span (Jan 2024 .. Jun 2025). */
export const NUM_MONTHS = 18;

/** Start (epoch ms) of month i of the span; i = 0 → 2024-01-01, i = 18 → 2025-07-01. */
export function monthStart(i: number): number {
  return Date.UTC(2024, i, 1);
}

/** Calendar month index (0 = Jan 2024) containing epoch-ms `t`. */
export function monthIndexOf(t: number): number {
  const d = new Date(t);
  return (d.getUTCFullYear() - 2024) * 12 + d.getUTCMonth();
}

/** Calendar month (1..12) of month index i. */
export function calendarMonth(i: number): number {
  return (i % 12) + 1;
}

/** Truncate to whole seconds (database timestamps carry no sub-second part). */
export function wholeSeconds(t: number): number {
  return Math.floor(t / SECOND_MS) * SECOND_MS;
}

/** Uniform random whole-second time in [a, b). */
export function randomTime(rng: Rng, a: number, b: number): number {
  return wholeSeconds(a + (b - a) * rng.next());
}

/** ISO date (YYYY-MM-DD) for descriptions and logs. */
export function isoDate(t: number): string {
  return new Date(t).toISOString().slice(0, 10);
}

// ─────────────────────────────────────────────────────────────────────────────
// Table builder
// ─────────────────────────────────────────────────────────────────────────────

/** Shorthand column constructor. */
export function col(name: string, dtype: DType, description: string): ColumnSchema {
  return { name, dtype, description };
}

/**
 * Build a RawTable from records, re-keying every row in schema column order (so object key order
 * = "DataFrame order", which is the cell order pre.rs emits, spec §3.2/§3.8) and checking:
 *  - PK values equal row positions 0..n−1 (spec §1.2, pre.rs:452-459, 477);
 *  - at most 5 FK columns (MAX_F2P_NBRS = 5, fly.rs:25, spec §1.2, §4.16);
 *  - the time column, if any, is a datetime column (spec §1.4);
 *  - every value matches its declared dtype (or is null).
 */
export function makeTable(schema: TableSchema, records: Record<string, RawValue>[]): RawTable {
  const names = schema.columns.map((c) => c.name);
  if (new Set(names).size !== names.length) throw new Error(`${schema.name}: duplicate column names`);
  const fks = Object.keys(schema.fkeys);
  if (fks.length > 5) throw new Error(`${schema.name}: ${fks.length} FK columns > 5 (spec §1.2)`);
  for (const fk of fks) {
    if (!names.includes(fk)) throw new Error(`${schema.name}: FK column ${fk} missing`);
  }
  if (schema.pkey !== null && !names.includes(schema.pkey)) throw new Error(`${schema.name}: PK column missing`);
  if (schema.timeCol !== null) {
    const tc = schema.columns.find((c) => c.name === schema.timeCol);
    if (!tc || tc.dtype !== 'datetime') throw new Error(`${schema.name}: timeCol must be a datetime column`);
  }

  const rows = records.map((rec, i) => {
    const row: Record<string, RawValue> = {};
    for (const c of schema.columns) {
      const v = rec[c.name];
      if (v === undefined) throw new Error(`${schema.name}[${i}]: missing value for ${c.name}`);
      checkDType(schema.name, c, v, i);
      row[c.name] = v;
    }
    if (schema.pkey !== null && row[schema.pkey] !== i) {
      throw new Error(`${schema.name}[${i}]: PK ${String(row[schema.pkey])} != row position (spec §1.2)`);
    }
    return row;
  });
  return { schema, rows };
}

function checkDType(table: string, c: ColumnSchema, v: RawValue, i: number): void {
  if (v === null) return;
  const bad = () => new Error(`${table}[${i}].${c.name}: value ${String(v)} is not ${c.dtype}`);
  switch (c.dtype) {
    case 'int':
      if (typeof v !== 'number' || !Number.isInteger(v)) throw bad();
      break;
    case 'float':
      if (typeof v !== 'number') throw bad();
      break;
    case 'datetime':
      if (typeof v !== 'number' || !Number.isInteger(v)) throw bad();
      break;
    case 'bool':
      if (typeof v !== 'boolean') throw bad();
      break;
    case 'str':
      if (typeof v !== 'string') throw bad();
      break;
  }
}

/** Stable chronological sort of generated records by a time field, then a tie-breaker. */
export function sortByTime<T>(items: T[], time: (x: T) => number, tie: (x: T) => number): T[] {
  return items.sort((a, b) => time(a) - time(b) || tie(a) - tie(b));
}
