/**
 * Numeric kernels for the RT forward pass, on row-major Float32Array tensors.
 *
 * Layout conventions (identical to PyTorch so exported weights are used as-is):
 *  - an activation tensor [S, n] is `x[s * n + j]`;
 *  - an nn.Linear weight is [out, in] (`W[o * nIn + i]`), applied as y = x Wᵀ + b.
 *
 * Accumulations run in float64 (JS numbers) and results are stored as float32. PyTorch
 * accumulates in float32, so the two agree to ~1e-6 relative; the parity test allows 1e-4.
 *
 * Several kernels take an optional `rows` list (Int32Array of row indices, increasing). Rows not
 * in the list are left untouched (0 in a fresh output). The forward pass uses this to skip padding
 * tokens, whose activations are exactly 0 (spec §7.2): for a bias-free linear map W·0 = 0, so
 * skipping them is exact, not an approximation.
 */

/**
 * y[s, o] = Σ_i x[s, i] · W[o, i] + b[o]   (nn.Linear, model.py:25-28, 54-56, 115-131).
 * `x` is [S × nIn]; `W` is [nOut × nIn]; `b` is [nOut] or null; `rows` restricts the rows computed.
 */
export function linear(
  x: Float32Array,
  S: number,
  nIn: number,
  W: Float32Array,
  nOut: number,
  b: Float32Array | null = null,
  out: Float32Array = new Float32Array(S * nOut),
  rows: Int32Array | null = null,
): Float32Array {
  if (W.length !== nOut * nIn) throw new Error(`linear: W has ${W.length} values, expected ${nOut}×${nIn}`);
  if (x.length < S * nIn) throw new Error(`linear: x has ${x.length} values, expected ≥ ${S}×${nIn}`);
  const R = rows ? rows.length : S;
  const o4 = nOut - (nOut % 4);
  // Register blocking: 2 rows × 4 output units per pass (8 accumulators). Each output is still a
  // plain sequential dot product over i, so blocking does not change any result.
  let r = 0;
  for (; r + 1 < R; r += 2) {
    const sa = rows ? rows[r] : r;
    const sb = rows ? rows[r + 1] : r + 1;
    const xa = sa * nIn;
    const xb = sb * nIn;
    const ya = sa * nOut;
    const yb = sb * nOut;
    let o = 0;
    for (; o < o4; o += 4) {
      const w0 = o * nIn;
      const w1 = w0 + nIn;
      const w2 = w1 + nIn;
      const w3 = w2 + nIn;
      let a0 = 0, a1 = 0, a2 = 0, a3 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0; // prettier-ignore
      for (let i = 0; i < nIn; i++) {
        const p = x[xa + i];
        const q = x[xb + i];
        const u0 = W[w0 + i];
        const u1 = W[w1 + i];
        const u2 = W[w2 + i];
        const u3 = W[w3 + i];
        a0 += p * u0;
        a1 += p * u1;
        a2 += p * u2;
        a3 += p * u3;
        c0 += q * u0;
        c1 += q * u1;
        c2 += q * u2;
        c3 += q * u3;
      }
      const b0 = b ? b[o] : 0;
      const b1 = b ? b[o + 1] : 0;
      const b2 = b ? b[o + 2] : 0;
      const b3 = b ? b[o + 3] : 0;
      out[ya + o] = a0 + b0;
      out[ya + o + 1] = a1 + b1;
      out[ya + o + 2] = a2 + b2;
      out[ya + o + 3] = a3 + b3;
      out[yb + o] = c0 + b0;
      out[yb + o + 1] = c1 + b1;
      out[yb + o + 2] = c2 + b2;
      out[yb + o + 3] = c3 + b3;
    }
    for (; o < nOut; o++) {
      const w0 = o * nIn;
      let a = 0;
      let c = 0;
      for (let i = 0; i < nIn; i++) {
        a += x[xa + i] * W[w0 + i];
        c += x[xb + i] * W[w0 + i];
      }
      out[ya + o] = a + (b ? b[o] : 0);
      out[yb + o] = c + (b ? b[o] : 0);
    }
  }
  for (; r < R; r++) {
    const s = rows ? rows[r] : r;
    const xo = s * nIn;
    const yo = s * nOut;
    for (let o = 0; o < nOut; o++) {
      const w0 = o * nIn;
      let a = 0;
      for (let i = 0; i < nIn; i++) a += x[xo + i] * W[w0 + i];
      out[yo + o] = a + (b ? b[o] : 0);
    }
  }
  return out;
}

/** Linear map of a single vector (convenience for UI code). */
export function linearVec(x: Float32Array, W: Float32Array, nOut: number, b: Float32Array | null = null): Float32Array {
  return linear(x, 1, x.length, W, nOut, b);
}

/**
 * RMSNorm of one row (nn.RMSNorm, spec §7.2): y_j = x_j / sqrt(mean_k x_k² + ε) · γ_j.
 * Reads x[xo .. xo+d), writes out[oo .. oo+d). RMSNorm(0⃗) = 0⃗ exactly.
 */
export function rmsnormRow(
  x: Float32Array,
  xo: number,
  d: number,
  gamma: Float32Array,
  eps: number,
  out: Float32Array,
  oo: number,
): void {
  let ms = 0;
  for (let j = 0; j < d; j++) {
    const v = x[xo + j];
    ms += v * v;
  }
  const r = 1 / Math.sqrt(ms / d + eps);
  for (let j = 0; j < d; j++) out[oo + j] = x[xo + j] * r * gamma[j];
}

/** RMSNorm applied to every row (or the listed rows) of an [S × d] tensor. */
export function rmsnorm(
  x: Float32Array,
  S: number,
  d: number,
  gamma: Float32Array,
  eps: number,
  out: Float32Array = new Float32Array(S * d),
  rows: Int32Array | null = null,
): Float32Array {
  if (rows) for (const s of rows) rmsnormRow(x, s * d, d, gamma, eps, out, s * d);
  else for (let s = 0; s < S; s++) rmsnormRow(x, s * d, d, gamma, eps, out, s * d);
  return out;
}

/** SiLU(z) = z · σ(z) (F.silu, model.py:59). */
export function silu(z: number): number {
  return z / (1 + Math.exp(-z));
}

/** out = a + b element-wise (float32 result, i.e. the same rounding as a float32 tensor add). */
export function add(a: Float32Array, b: Float32Array, out: Float32Array = new Float32Array(a.length)): Float32Array {
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

/** L2 norm of x[off .. off+n). */
export function l2norm(x: Float32Array, off = 0, n = x.length - off): number {
  let s = 0;
  for (let i = 0; i < n; i++) s += x[off + i] * x[off + i];
  return Math.sqrt(s);
}

/** max_i |a_i − b_i| (Infinity if any difference is NaN). */
export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new Error(`maxAbsDiff: length ${a.length} != ${b.length}`);
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (Number.isNaN(d)) return Infinity;
    if (d > m) m = d;
  }
  return m;
}

/**
 * Compressed rows of a dense [S × S] boolean mask: for query q the allowed keys are
 * `keys[start[q] .. start[q+1])` in increasing order.
 */
export interface MaskRows {
  start: Int32Array;
  keys: Int32Array;
}

export function maskRows(mask: Uint8Array, S: number): MaskRows {
  const start = new Int32Array(S + 1);
  let n = 0;
  for (let q = 0; q < S; q++) {
    start[q] = n;
    for (let k = 0; k < S; k++) if (mask[q * S + k]) n++;
  }
  start[S] = n;
  const keys = new Int32Array(n);
  let p = 0;
  for (let q = 0; q < S; q++) for (let k = 0; k < S; k++) if (mask[q * S + k]) keys[p++] = k;
  return { start, keys };
}

/** Int32Array [0, 1, …, S−1]. */
export function allRows(S: number): Int32Array {
  const r = new Int32Array(S);
  for (let i = 0; i < S; i++) r[i] = i;
  return r;
}

export interface AttentionCaptureBuffers {
  /** [H × S × S] raw scaled scores q·k/√d_head for ALL pairs (before masking). */
  scores?: Float32Array;
  /** [H × S × S] masked softmax probabilities (0 where masked; all-0 rows when fully masked). */
  probs?: Float32Array;
}

/**
 * Raw scaled scores of one head for all (query, key) pairs among `rows`:
 *   out[outOff + qi·S + ki] = fround((Σ_j Q[qi, lo+j] · K[ki, lo+j]) · scale)
 * Pairs involving a row outside `rows` are left untouched (exactly 0 for padding rows, whose q
 * and k are 0). Blocked 2 queries × 4 keys; each score is a plain sequential dot product over j,
 * so it is bit-identical to the per-pair path in `maskedAttentionCore`.
 */
function headScores(
  q: Float32Array,
  k: Float32Array,
  S: number,
  d: number,
  lo: number,
  dh: number,
  scale: number,
  out: Float32Array,
  outOff: number,
  rows: Int32Array,
): void {
  const R = rows.length;
  const pair = (qi: number, ki: number) => {
    const qo = qi * d + lo;
    const ko = ki * d + lo;
    let acc = 0;
    for (let j = 0; j < dh; j++) acc += q[qo + j] * k[ko + j];
    out[outOff + qi * S + ki] = acc * scale;
  };
  let r = 0;
  for (; r + 1 < R; r += 2) {
    const qA = rows[r];
    const qB = rows[r + 1];
    const qa = qA * d + lo;
    const qb = qB * d + lo;
    const ra = outOff + qA * S;
    const rb = outOff + qB * S;
    let c = 0;
    for (; c + 3 < R; c += 4) {
      const kA = rows[c];
      const kB = rows[c + 1];
      const kC = rows[c + 2];
      const kD = rows[c + 3];
      const k0 = kA * d + lo;
      const k1 = kB * d + lo;
      const k2 = kC * d + lo;
      const k3 = kD * d + lo;
      let a0 = 0, a1 = 0, a2 = 0, a3 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0; // prettier-ignore
      for (let j = 0; j < dh; j++) {
        const x = q[qa + j];
        const y = q[qb + j];
        const u0 = k[k0 + j];
        const u1 = k[k1 + j];
        const u2 = k[k2 + j];
        const u3 = k[k3 + j];
        a0 += x * u0;
        a1 += x * u1;
        a2 += x * u2;
        a3 += x * u3;
        c0 += y * u0;
        c1 += y * u1;
        c2 += y * u2;
        c3 += y * u3;
      }
      out[ra + kA] = a0 * scale;
      out[ra + kB] = a1 * scale;
      out[ra + kC] = a2 * scale;
      out[ra + kD] = a3 * scale;
      out[rb + kA] = c0 * scale;
      out[rb + kB] = c1 * scale;
      out[rb + kC] = c2 * scale;
      out[rb + kD] = c3 * scale;
    }
    for (; c < R; c++) {
      pair(qA, rows[c]);
      pair(qB, rows[c]);
    }
  }
  for (; r < R; r++) for (let c = 0; c < R; c++) pair(rows[r], rows[c]);
}

/**
 * Masked multi-head scaled-dot-product attention core (no projections):
 *
 *   head j uses dims [j·dh, (j+1)·dh)            ("b s (h d) -> b h s d", model.py:35-37)
 *   s_qk = Q_j[q]·K_j[k] / √dh                    (flex_attention default scale 1/√E, model.py:43)
 *   s_qk = −∞ where mask[q,k] = 0                 (paper p.5 §3.2)
 *   p_q  = softmax_k(s_q);  O_j[q] = Σ_k p_qk V_j[k]
 *   a query row with NO allowed key outputs exactly 0 (not NaN, not uniform): FlexAttention
 *   sets l_i = 1 for fully-masked rows (torch 2.6 _inductor/kernel/flex_attention.py:422-427), spec §6.1.
 *
 * The same mask is shared by all heads (create_block_mask(..., H=None), model.py:89-101).
 * Returns the concatenated head outputs [S × d] ("b h s d -> b s (h d)", model.py:45).
 *
 * `active` lists the rows whose q/k may be non-zero (the non-padding rows; all rows if null).
 * Raw scores are computed only among them; the others are exactly 0. Scores are float32-rounded
 * before the softmax in every path, so results do not depend on whether `capture` is passed.
 */
export function maskedAttentionCore(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  S: number,
  d: number,
  nHeads: number,
  rows: MaskRows,
  capture: AttentionCaptureBuffers | null,
  out: Float32Array = new Float32Array(S * d),
  active: Int32Array | null = null,
): Float32Array {
  const dh = d / nHeads;
  const scale = 1 / Math.sqrt(dh);
  const { start, keys } = rows;
  const probs = capture?.probs;
  const act = active ?? allRows(S);
  // Dense score matrix when the UI wants all raw scores, or when the mask is dense enough that
  // a blocked all-pairs kernel beats per-pair dot products.
  const dense = !!capture?.scores || keys.length * 4 > act.length * act.length;
  const scoreBuf = capture?.scores ?? (dense ? new Float32Array(S * S) : null);
  const row = new Float64Array(S); // one query row's weights over its allowed keys
  const chunked = dh % 4 === 0;
  out.fill(0);
  for (let h = 0; h < nHeads; h++) {
    const lo = h * dh;
    const hOff = h * S * S;
    const sOff = capture?.scores ? hOff : 0;
    if (dense) headScores(q, k, S, d, lo, dh, scale, scoreBuf!, sOff, act);
    for (let qi = 0; qi < S; qi++) {
      const a = start[qi];
      const bEnd = start[qi + 1];
      if (a === bEnd) continue; // fully masked query row → output 0, probs row stays 0
      const qo = qi * d + lo;
      let mx = -Infinity;
      for (let p = a; p < bEnd; p++) {
        const ki = keys[p];
        let sc: number;
        if (dense) {
          sc = scoreBuf![sOff + qi * S + ki];
        } else {
          const ko = ki * d + lo;
          let acc = 0;
          for (let j = 0; j < dh; j++) acc += q[qo + j] * k[ko + j];
          sc = Math.fround(acc * scale); // same float32 rounding as the dense path
        }
        row[p - a] = sc;
        if (sc > mx) mx = sc;
      }
      let Z = 0;
      for (let p = a; p < bEnd; p++) {
        const e = Math.exp(row[p - a] - mx);
        row[p - a] = e;
        Z += e;
      }
      const invZ = 1 / Z;
      for (let p = a; p < bEnd; p++) row[p - a] *= invZ;
      if (probs) for (let p = a; p < bEnd; p++) probs[hOff + qi * S + keys[p]] = row[p - a];
      // O[q] = Σ_k p_qk V[k] over the allowed keys in increasing k (masked keys contribute 0)
      const oo = qi * d + lo;
      if (chunked) {
        for (let c = 0; c < dh; c += 4) {
          let a0 = 0, a1 = 0, a2 = 0, a3 = 0; // prettier-ignore
          for (let p = a; p < bEnd; p++) {
            const w = row[p - a];
            const vo = keys[p] * d + lo + c;
            a0 += w * v[vo];
            a1 += w * v[vo + 1];
            a2 += w * v[vo + 2];
            a3 += w * v[vo + 3];
          }
          out[oo + c] = a0;
          out[oo + c + 1] = a1;
          out[oo + c + 2] = a2;
          out[oo + c + 3] = a3;
        }
      } else {
        for (let j = 0; j < dh; j++) {
          let acc = 0;
          for (let p = a; p < bEnd; p++) acc += row[p - a] * v[keys[p] * d + lo + j];
          out[oo + j] = acc;
        }
      }
    }
  }
  return out;
}
