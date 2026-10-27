/**
 * Pure re-derivations for the "start to end" page. The page shows the engine's captured tensors
 * (ForwardResult, a mirror of rt/model.py:153-267); these helpers recompute single entries from
 * their operands so a reader can see where a number comes from (one dot product, one RMSNorm
 * row, one softmax row). math.test.ts checks every one of them against the capture on the real
 * trained weights, so the arithmetic shown is the forward pass, not an approximation of it.
 */
import { SemType, type AttnLayer, type Token } from '@/engine/types';

/** Row r of a row-major [rows × n] tensor (zero-copy view). */
export function rowOf(a: Float32Array, r: number, n: number): Float32Array {
  return a.subarray(r * n, (r + 1) * n);
}

/** Column c of a row-major [rows × n] tensor, for the first `rows` rows (a copy). */
export function colOf(a: ArrayLike<number>, c: number, n: number, rows: number): Float32Array {
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) out[r] = a[r * n + c];
  return out;
}

export interface Term {
  /** Index along the summed dimension. */
  i: number;
  a: number;
  b: number;
  /** a · b */
  p: number;
}

/** Σ_i a_i b_i in float64, with every term (for "where does this number come from"). */
export function dotTerms(a: ArrayLike<number>, b: ArrayLike<number>): { sum: number; terms: Term[] } {
  if (a.length !== b.length) throw new Error(`dotTerms: length ${a.length} != ${b.length}`);
  const terms: Term[] = [];
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const p = a[i] * b[i];
    sum += p;
    terms.push({ i, a: a[i], b: b[i], p });
  }
  return { sum, terms };
}

/** The n terms with the largest |a·b|, largest first. */
export function topTerms(terms: readonly Term[], n: number): Term[] {
  return [...terms].sort((x, y) => Math.abs(y.p) - Math.abs(x.p) || x.i - y.i).slice(0, n);
}

/**
 * One entry of nn.Linear, y = x Wᵀ + b: y[j] = Σ_k x[k] · W[j, k] (+ b[j]), with W stored [out, in]
 * (rt/model.py uses nn.Linear everywhere; spec §6.1, §7.1).
 */
export function linearEntry(x: ArrayLike<number>, Wt: Float32Array, nIn: number, j: number, b?: Float32Array | null) {
  const { sum, terms } = dotTerms(x, Wt.subarray(j * nIn, (j + 1) * nIn));
  return { value: sum + (b ? b[j] : 0), dot: sum, bias: b ? b[j] : 0, terms };
}

/**
 * RMSNorm of one row (nn.RMSNorm, spec §7.2): y_j = x_j / sqrt(mean_k x_k² + ε) · γ_j.
 * Returns the pieces the page shows: mean of squares, the divisor and the output.
 */
export function rmsRow(x: ArrayLike<number>, gamma: ArrayLike<number>, eps: number) {
  let ms = 0;
  for (let j = 0; j < x.length; j++) ms += x[j] * x[j];
  ms /= x.length;
  const denom = Math.sqrt(ms + eps);
  const y = new Float32Array(x.length);
  for (let j = 0; j < x.length; j++) y[j] = (x[j] / denom) * gamma[j];
  return { meanSq: ms, denom, y };
}

/** Head h's slice of a row-major [S × d] tensor as [S × dh] (the "b s (h d) -> b h s d" split, rt/model.py:35-37). */
export function headSlice(a: Float32Array, S: number, d: number, h: number, dh: number): Float32Array {
  const out = new Float32Array(S * dh);
  for (let s = 0; s < S; s++) out.set(a.subarray(s * d + h * dh, s * d + (h + 1) * dh), s * dh);
  return out;
}

/** The same slice transposed, [dh × S] (Kₕᵀ in Qₕ Kₕᵀ). */
export function headSliceT(a: Float32Array, S: number, d: number, h: number, dh: number): Float32Array {
  const out = new Float32Array(dh * S);
  for (let s = 0; s < S; s++) for (let j = 0; j < dh; j++) out[j * S + s] = a[s * d + h * dh + j];
  return out;
}

/** Head h's [S × S] panel of a captured [H × S × S] score or probability tensor (zero-copy view). */
export function headPanel(a: Float32Array, S: number, h: number): Float32Array {
  return a.subarray(h * S * S, (h + 1) * S * S);
}

/** One attention score: q_h[q] · k_h[k] / √dh (FlexAttention's default scale 1/√E, rt/model.py:43; spec §6.1). */
export function scoreEntry(qBuf: Float32Array, kBuf: Float32Array, d: number, dh: number, h: number, q: number, k: number) {
  const lo = h * dh;
  const { sum, terms } = dotTerms(qBuf.subarray(q * d + lo, q * d + lo + dh), kBuf.subarray(k * d + lo, k * d + lo + dh));
  const scale = 1 / Math.sqrt(dh);
  return { value: sum * scale, dot: sum, scale, terms };
}

/**
 * Masked softmax of one query row (paper p.5 §3.2; spec §6.1): weights over the allowed keys, 0 on
 * the others; a row with no allowed key gives all zeros (FlexAttention's fully-masked-row rule).
 */
export function maskedSoftmaxRow(scores: ArrayLike<number>, allowed: ArrayLike<number>) {
  const n = scores.length;
  const p = new Float64Array(n);
  let mx = -Infinity;
  for (let k = 0; k < n; k++) if (allowed[k] && scores[k] > mx) mx = scores[k];
  if (mx === -Infinity) return { p, max: mx, Z: 0 };
  let Z = 0;
  for (let k = 0; k < n; k++) {
    if (!allowed[k]) continue;
    p[k] = Math.exp(scores[k] - mx);
    Z += p[k];
  }
  for (let k = 0; k < n; k++) p[k] /= Z;
  return { p, max: mx, Z };
}

/** One entry of a head's output: O_h[q, j] = Σ_k P[q, k] · V_h[k, j] over the allowed keys. */
export function weightedSumEntry(probRow: ArrayLike<number>, vBuf: Float32Array, d: number, dh: number, h: number, j: number) {
  const terms: Term[] = [];
  let sum = 0;
  for (let k = 0; k < probRow.length; k++) {
    if (!probRow[k]) continue;
    const b = vBuf[k * d + h * dh + j];
    const p = probRow[k] * b;
    sum += p;
    terms.push({ i: k, a: probRow[k], b, p });
  }
  return { value: sum, terms };
}

/** SiLU(z) = z · σ(z) (F.silu, rt/model.py:59). */
export function silu(z: number): number {
  return z / (1 + Math.exp(-z));
}

/** Why query q may attend to key k in a layer, by the code's definition (rt/model.py:165-191; spec §6.2). */
export type MaskReason = 'same-column' | 'same-row' | 'parent-row' | 'child-row' | 'any';

export function maskReasons(tokens: readonly Token[], layer: AttnLayer, q: number, k: number): MaskReason[] {
  const a = tokens[q];
  const b = tokens[k];
  if (!a || !b || a.isPadding || b.isPadding) return [];
  switch (layer) {
    case 'col':
      return a.colNameIdx === b.colNameIdx && a.tableNameIdx === b.tableNameIdx ? ['same-column'] : [];
    case 'feat': {
      const out: MaskReason[] = [];
      if (a.nodeIdx === b.nodeIdx) out.push('same-row');
      if (a.f2pNbrIdxs.includes(b.nodeIdx)) out.push('parent-row');
      return out;
    }
    case 'nbr':
      return b.f2pNbrIdxs.includes(a.nodeIdx) ? ['child-row'] : [];
    case 'full':
      return ['any'];
  }
}

/** The scalar a token's value encoder reads (rt/model.py:216-220): its own type's slot. Text has none. */
export function scalarInput(t: Token): number | null {
  switch (t.sem) {
    case SemType.Number:
      return t.numberValue;
    case SemType.DateTime:
      return t.datetimeValue;
    case SemType.Boolean:
      return t.booleanValue;
    default:
      return null;
  }
}
