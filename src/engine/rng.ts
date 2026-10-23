/**
 * Seeded, deterministic PRNG used by the data generators and the sampler.
 *
 * The official sampler uses Rust's `StdRng::seed_from_u64(epoch + seed_node_idx + seed)`
 * (ChaCha12) with `random_range` and `rand::seq::index::sample` (spec §4.2). Bit-exact
 * parity is not required for fidelity: what matters is that every draw is uniform and
 * that a context is a deterministic function of (epoch, seed row, seed). We use
 * xoshiro128** seeded through splitmix32, which has the same distributional behavior.
 */
export class Rng {
  private s0: number;
  private s1: number;
  private s2: number;
  private s3: number;

  constructor(seed: number) {
    // splitmix32 to expand a single integer seed into 128 bits of state
    let x = (Math.trunc(seed) >>> 0) ^ 0x9e3779b9;
    const next = () => {
      x = (x + 0x9e3779b9) >>> 0;
      let z = x;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.s0 = next();
    this.s1 = next();
    this.s2 = next();
    this.s3 = next();
    if ((this.s0 | this.s1 | this.s2 | this.s3) === 0) this.s0 = 1;
  }

  /** Next raw 32-bit unsigned integer (xoshiro128**). */
  nextU32(): number {
    const result = Math.imul(rotl(Math.imul(this.s1, 5) >>> 0, 7), 9) >>> 0;
    const t = (this.s1 << 9) >>> 0;
    this.s2 ^= this.s0;
    this.s3 ^= this.s1;
    this.s1 ^= this.s2;
    this.s0 ^= this.s3;
    this.s2 ^= t;
    this.s3 = rotl(this.s3, 11);
    this.s0 >>>= 0;
    this.s1 >>>= 0;
    this.s2 >>>= 0;
    this.s3 >>>= 0;
    return result;
  }

  /** Uniform float in [0, 1). */
  next(): number {
    return this.nextU32() / 4294967296;
  }

  /** Uniform integer in [0, n) — mirrors `rng.random_range(0..n)` (fly.rs:381-396). Unbiased (rejection). */
  int(n: number): number {
    if (!(n >= 1)) throw new Error(`Rng.int: n must be ≥ 1, got ${n}`);
    if (n === 1) return 0;
    const limit = 4294967296 - (4294967296 % n);
    let r: number;
    do {
      r = this.nextU32();
    } while (r >= limit);
    return r % n;
  }

  /**
   * k distinct indices drawn uniformly without replacement from [0, n) —
   * mirrors `rand::seq::index::sample(rng, n, k)` (fly.rs:439-443).
   * Returned in draw order (like rand's IndexVec); callers must not rely on sortedness.
   */
  sampleIndices(n: number, k: number): number[] {
    if (k > n) throw new Error(`Rng.sampleIndices: k=${k} > n=${n}`);
    // partial Fisher–Yates over an index array
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < k; i++) {
      const j = i + this.int(n - i);
      const tmp = idx[i];
      idx[i] = idx[j];
      idx[j] = tmp;
    }
    return idx.slice(0, k);
  }

  /** In-place Fisher–Yates shuffle. */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }

  /** Uniform pick from a non-empty array. */
  choice<T>(arr: readonly T[]): T {
    return arr[this.int(arr.length)];
  }

  /** Weighted pick; weights need not be normalized. */
  weightedChoice<T>(arr: readonly T[], weights: readonly number[]): T {
    let total = 0;
    for (const w of weights) total += w;
    let r = this.next() * total;
    for (let i = 0; i < arr.length; i++) {
      r -= weights[i];
      if (r < 0) return arr[i];
    }
    return arr[arr.length - 1];
  }

  /** Bernoulli(p). */
  bool(p: number): boolean {
    return this.next() < p;
  }

  /** Uniform float in [a, b). */
  uniform(a: number, b: number): number {
    return a + (b - a) * this.next();
  }

  /** Standard normal (Box–Muller). */
  normal(mean = 0, std = 1): number {
    let u = 0;
    while (u === 0) u = this.next();
    const v = this.next();
    return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Poisson(λ) (Knuth for small λ, normal approximation for large λ). */
  poisson(lambda: number): number {
    if (lambda <= 0) return 0;
    if (lambda > 30) return Math.max(0, Math.round(this.normal(lambda, Math.sqrt(lambda))));
    const L = Math.exp(-lambda);
    let k = 0;
    let p = 1;
    do {
      k++;
      p *= this.next();
    } while (p > L);
    return k - 1;
  }

  /** Gamma(shape, scale) (Marsaglia–Tsang). */
  gamma(shape: number, scale = 1): number {
    if (shape < 1) {
      const u = this.next();
      return this.gamma(shape + 1, scale) * Math.pow(u, 1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x: number;
      let v: number;
      do {
        x = this.normal();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = this.next();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v * scale;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
    }
  }
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

/** fly.rs seeds each sequence with epoch + seed_node_idx + seed (wrapping u64 adds), spec §4.2. */
export function contextSeed(epoch: number, seedNodeIdx: number, seed: number): number {
  return (epoch + seedNodeIdx + seed) >>> 0;
}
