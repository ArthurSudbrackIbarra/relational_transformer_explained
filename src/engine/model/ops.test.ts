import { describe, expect, it } from 'vitest';
import { Rng } from '../rng';
import { createEmbeddingStore, parseEmbeddingStore } from '../embeddings';
import { MINI_CONFIG, PAPER_CONFIG, paramCount, parameterShapes } from './config';
import { bceWithLogits, huber } from './loss';
import { linear, maskRows, maskedAttentionCore, maxAbsDiff, rmsnorm, silu } from './ops';
import { initWeights, parseWeights, serializeWeights, validateWeights } from './weights';

function randArray(rng: Rng, n: number, scale = 1): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = rng.normal() * scale;
  return a;
}

describe('ops', () => {
  it('linear matches a naive x Wᵀ + b (PyTorch [out, in] layout)', () => {
    const rng = new Rng(1);
    const S = 5, nIn = 7, nOut = 3;
    const x = randArray(rng, S * nIn);
    const W = randArray(rng, nOut * nIn);
    const b = randArray(rng, nOut);
    const y = linear(x, S, nIn, W, nOut, b);
    for (let s = 0; s < S; s++)
      for (let o = 0; o < nOut; o++) {
        let acc = b[o];
        for (let i = 0; i < nIn; i++) acc += x[s * nIn + i] * W[o * nIn + i];
        expect(y[s * nOut + o]).toBeCloseTo(acc, 5);
      }
  });

  it('rmsnorm: y = x / sqrt(mean(x²) + eps) · γ, and RMSNorm(0) = 0 exactly', () => {
    const x = Float32Array.of(1, 2, 3, 4);
    const g = Float32Array.of(1, 0.5, 2, 1);
    const eps = 2 ** -7;
    const y = rmsnorm(x, 1, 4, g, eps);
    const r = 1 / Math.sqrt((1 + 4 + 9 + 16) / 4 + eps);
    expect(Array.from(y)).toEqual([1 * r, 2 * r * 0.5, 3 * r * 2, 4 * r].map(Math.fround));
    const z = rmsnorm(new Float32Array(4), 1, 4, g, eps);
    expect(Array.from(z)).toEqual([0, 0, 0, 0]);
  });

  it('silu(z) = z·σ(z)', () => {
    expect(silu(0)).toBe(0);
    expect(silu(2)).toBeCloseTo(2 / (1 + Math.exp(-2)), 12);
    expect(silu(-800)).toBe(-0);
  });

  it('masked attention equals dense softmax with −∞ masking; fully masked rows output exactly 0', () => {
    const rng = new Rng(7);
    const S = 9, d = 8, H = 2, dh = d / H;
    const q = randArray(rng, S * d), k = randArray(rng, S * d), v = randArray(rng, S * d);
    const mask = new Uint8Array(S * S);
    for (let i = 0; i < S * S; i++) mask[i] = rng.bool(0.5) ? 1 : 0;
    for (let kk = 0; kk < S; kk++) mask[3 * S + kk] = 0; // query 3 fully masked
    const cap = { scores: new Float32Array(H * S * S), probs: new Float32Array(H * S * S) };
    const out = maskedAttentionCore(q, k, v, S, d, H, maskRows(mask, S), cap);
    for (let h = 0; h < H; h++)
      for (let qi = 0; qi < S; qi++) {
        const s: number[] = [];
        for (let kk = 0; kk < S; kk++) {
          let a = 0;
          for (let j = 0; j < dh; j++) a += q[qi * d + h * dh + j] * k[kk * d + h * dh + j];
          a /= Math.sqrt(dh);
          expect(cap.scores[h * S * S + qi * S + kk]).toBeCloseTo(a, 5); // raw, before masking
          s.push(mask[qi * S + kk] ? a : -Infinity);
        }
        const mx = Math.max(...s);
        const e = s.map((z) => (mx === -Infinity ? 0 : Math.exp(z - mx)));
        const Z = e.reduce((a, b) => a + b, 0);
        for (let j = 0; j < dh; j++) {
          let o = 0;
          for (let kk = 0; kk < S; kk++) if (Z > 0) o += (e[kk] / Z) * v[kk * d + h * dh + j];
          expect(out[qi * d + h * dh + j]).toBeCloseTo(o, 5);
        }
      }
    for (let j = 0; j < d; j++) expect(out[3 * d + j]).toBe(0);
    for (let h = 0; h < H; h++) for (let kk = 0; kk < S; kk++) expect(cap.probs[h * S * S + 3 * S + kk]).toBe(0);
    expect(out.every((z) => Number.isFinite(z))).toBe(true);
  });
});

describe('loss', () => {
  it('huber δ=1 (F.huber_loss default): huber(3,0) = 2.5, quadratic inside', () => {
    expect(huber(3, 0)).toBe(2.5);
    expect(huber(0.5, 0)).toBe(0.125);
    expect(huber(-1, 0)).toBe(0.5);
  });
  it('bceWithLogits: bce(0,1) = ln 2, stable for large |z|', () => {
    expect(bceWithLogits(0, 1)).toBeCloseTo(Math.LN2, 12);
    expect(bceWithLogits(100, 1)).toBeCloseTo(0, 12);
    expect(bceWithLogits(-100, 1)).toBeCloseTo(100, 9);
    expect(bceWithLogits(2, 0)).toBeCloseTo(2 + Math.log1p(Math.exp(-2)), 12);
  });
});

describe('config & weights', () => {
  it('paper config has exactly 22,336,131 parameters (spec §7.3)', () => {
    expect(paramCount(PAPER_CONFIG)).toBe(22_336_131);
  });
  it('initWeights: every tensor present, PyTorch-default bounds, γ = 1, deterministic', () => {
    const w = initWeights(MINI_CONFIG, 3);
    validateWeights(w);
    const t = w.tensors;
    const bound = 1 / Math.sqrt(384);
    expect(t['enc_dict.col_name.weight'].data.every((x) => Math.abs(x) <= bound)).toBe(true);
    expect(t['enc_dict.col_name.bias'].data.every((x) => Math.abs(x) <= bound)).toBe(true);
    expect(t['enc_dict.number.weight'].data.every((x) => Math.abs(x) <= 1)).toBe(true);
    expect(t['blocks.0.ffn.w2.weight'].data.every((x) => Math.abs(x) <= 1 / 16)).toBe(true);
    expect(t['norm_out.weight'].data.every((x) => x === 1)).toBe(true);
    expect(t['blocks.3.norms.col.weight'].data.every((x) => x === 1)).toBe(true);
    const m = t['mask_embs.number'].data;
    expect(m.some((x) => Math.abs(x) > 1)).toBe(true); // N(0,1), not uniform
    expect(maxAbsDiff(initWeights(MINI_CONFIG, 3).tensors['blocks.1.attns.nbr.wq.weight'].data, t['blocks.1.attns.nbr.wq.weight'].data)).toBe(0);
  });
  it('serialize → parse round-trips and keeps state_dict order', () => {
    const w = initWeights(MINI_CONFIG, 5);
    const { manifest, bin } = serializeWeights(w);
    expect(Object.keys(manifest.tensors)).toEqual(parameterShapes(MINI_CONFIG).map(([n]) => n));
    const copy = new ArrayBuffer(bin.byteLength);
    new Float32Array(copy).set(bin);
    const w2 = parseWeights(manifest, copy);
    for (const [name] of parameterShapes(MINI_CONFIG)) expect(maxAbsDiff(w2.tensors[name].data, w.tensors[name].data)).toBe(0);
  });
  it('parseWeights rejects a missing or mis-shaped tensor', () => {
    const { manifest, bin } = serializeWeights(initWeights(MINI_CONFIG, 1));
    const bad = structuredClone(manifest);
    delete bad.tensors['norm_out.weight'];
    expect(() => parseWeights(bad, bin.buffer as ArrayBuffer)).toThrow(/missing/);
    const bad2 = structuredClone(manifest);
    bad2.tensors['blocks.0.ffn.w1.weight'].shape = [64, 256];
    expect(() => parseWeights(bad2, bin.buffer as ArrayBuffer)).toThrow(/shape/);
  });
});

describe('EmbeddingStore', () => {
  it('get() returns the stored row; unknown strings throw (never fabricated)', () => {
    const data = Float32Array.of(1, 2, 3, 4, 5, 6);
    const s = createEmbeddingStore(['a', 'b of c'], data);
    expect(s.dim).toBe(3);
    expect(Array.from(s.get('b of c'))).toEqual([4, 5, 6]);
    expect(s.has('a')).toBe(true);
    expect(s.has('zzz')).toBe(false);
    expect(() => s.get('zzz')).toThrow(/no precomputed embedding/);
    expect(s.strings).toEqual(['a', 'b of c']);
  });
  it('parses both index formats (string array or {strings, dim})', () => {
    const buf = Float32Array.of(1, 2, 3, 4).buffer as ArrayBuffer;
    expect(parseEmbeddingStore(['x', 'y'], buf).get('y')[1]).toBe(4);
    expect(parseEmbeddingStore({ strings: ['x'], dim: 4 }, buf).dim).toBe(4);
    expect(() => createEmbeddingStore(['x', 'y'], Float32Array.of(1, 2, 3))).toThrow();
  });
});
