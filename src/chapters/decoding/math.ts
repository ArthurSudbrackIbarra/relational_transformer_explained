/**
 * Read-outs of the live forward pass for the decoding chapter. Only slicing and the decoders'
 * own linear maps (model.py:124-131, 238-239) are recomputed here, from the same weights, to show
 * their parts (w·x̂ and b); every result is checked against the engine's `ForwardResult.yhat`.
 */
import { tensor } from '@/engine/model/weights';
import type { ForwardResult, RTWeights } from '@/engine/types';

export type HeadKey = 'number' | 'text' | 'datetime' | 'boolean';
/** dec_dict registration order = sem-type code order (model.py:124-131, 238). */
export const HEADS: HeadKey[] = ['number', 'text', 'datetime', 'boolean'];

export function rowOf(t: Float32Array, pos: number, d: number): Float32Array {
  return t.subarray(pos * d, (pos + 1) * d);
}

export interface ScalarHead {
  key: Exclude<HeadKey, 'text'>;
  w: Float32Array;
  b: number;
  /** w·x̂ */
  dot: number;
  /** w·x̂ + b (recomputed) */
  out: number;
  /** The engine's value, fr.yhat[key][pos]. */
  engine: number;
  /** Elementwise w_j·x̂_j (the d terms of the sum). */
  terms: Float32Array;
}

export interface DecodeAt {
  pos: number;
  d: number;
  /** Residual stream after the last block (input of norm_out). */
  xLast: Float32Array;
  /** norm_out(xLast) = fr.xFinal row. */
  xHat: Float32Array;
  gamma: Float32Array;
  /** √(mean(x²) + ε) of xLast. */
  rms: number;
  eps: number;
  scalar: Record<Exclude<HeadKey, 'text'>, ScalarHead>;
  /** Text head output (d_text values) and its bias-only part. */
  text: { out: Float32Array; norm: number; dText: number };
}

export function decodeAt(fr: ForwardResult, w: RTWeights, pos: number): DecodeAt {
  const d = fr.d;
  const last = fr.blocks[fr.blocks.length - 1];
  const xLast = rowOf(last.ffn.xOut, pos, d);
  const xHat = rowOf(fr.xFinal, pos, d);
  const gamma = tensor(w, 'norm_out.weight');
  const eps = w.config.eps;
  let ms = 0;
  for (let j = 0; j < d; j++) ms += xLast[j] * xLast[j];
  const rms = Math.sqrt(ms / d + eps);

  const scalarHead = (key: Exclude<HeadKey, 'text'>): ScalarHead => {
    const W = tensor(w, `dec_dict.${key}.weight`);
    const b = tensor(w, `dec_dict.${key}.bias`)[0];
    const terms = new Float32Array(d);
    let dot = 0;
    for (let j = 0; j < d; j++) {
      terms[j] = W[j] * xHat[j];
      dot += W[j] * xHat[j];
    }
    return {
      key,
      w: W,
      b,
      dot,
      out: dot + b,
      engine: fr.yhat[key][pos],
      terms,
    };
  };

  const dText = w.config.dText;
  const Wt = tensor(w, 'dec_dict.text.weight'); // [dText, d]
  const bt = tensor(w, 'dec_dict.text.bias');
  const out = new Float32Array(dText);
  let n2 = 0;
  for (let o = 0; o < dText; o++) {
    let s = bt[o];
    for (let j = 0; j < d; j++) s += Wt[o * d + j] * xHat[j];
    out[o] = s;
    n2 += s * s;
  }

  return {
    pos,
    d,
    xLast,
    xHat,
    gamma,
    rms,
    eps,
    scalar: {
      number: scalarHead('number'),
      datetime: scalarHead('datetime'),
      boolean: scalarHead('boolean'),
    },
    text: { out, norm: Math.sqrt(n2), dText },
  };
}

export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
export const bce = (z: number, y: number) => Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));
export const huber = (d: number) => (Math.abs(d) < 1 ? 0.5 * d * d : Math.abs(d) - 0.5);

/** ROC points (FPR, TPR) from labels and scores, descending threshold, ties grouped (like sklearn). */
export function rocCurve(y: readonly (0 | 1)[], s: readonly number[]): { fpr: number; tpr: number; thr: number }[] {
  const idx = s.map((_, i) => i).sort((a, b) => s[b] - s[a]);
  let P = 0;
  for (const v of y) P += v;
  const N = y.length - P;
  const pts = [{ fpr: 0, tpr: 0, thr: Infinity }];
  let tp = 0;
  let fp = 0;
  for (let i = 0; i < idx.length;) {
    const thr = s[idx[i]];
    while (i < idx.length && s[idx[i]] === thr) {
      if (y[idx[i]] === 1) tp++;
      else fp++;
      i++;
    }
    pts.push({ fpr: N ? fp / N : 0, tpr: P ? tp / P : 0, thr });
  }
  return pts;
}

/** Ordered (positive, negative) pairs: correctly ranked, tied, wrong. AUROC = (correct + ties/2) / total. */
export function pairCounts(y: readonly (0 | 1)[], s: readonly number[]): { total: number; correct: number; ties: number } {
  const pos: number[] = [];
  const neg: number[] = [];
  y.forEach((v, i) => (v === 1 ? pos : neg).push(s[i]));
  let correct = 0;
  let ties = 0;
  for (const p of pos)
    for (const n of neg) {
      if (p > n) correct++;
      else if (p === n) ties++;
    }
  return { total: pos.length * neg.length, correct, ties };
}
