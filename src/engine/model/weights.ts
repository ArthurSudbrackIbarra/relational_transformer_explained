/**
 * RT weights: binary format, loading, PyTorch-default initialization.
 *
 * File format (written by python/export_weights.py, read here):
 *   <name>.bin   float32 little-endian, every tensor's values concatenated (row-major / C order)
 *   <name>.json  {
 *     format: "rt-weights-v1",
 *     config: RTConfig,                              // dModel, nHeads, dFF, nBlocks, dText, eps
 *     meta:   {...},                                 // provenance (checkpoint, step, seed, ...)
 *     tensors: { "<state_dict name>": { shape: number[], offset: number, length: number } }
 *   }
 * `offset` and `length` count float32 ELEMENTS (not bytes). Tensor names are the PyTorch
 * state_dict names of PE/rt/model.py (see config.ts parameterShapes), so a checkpoint saved by
 * the official class or by python/rt_model.py exports without renaming.
 */
import type { RTConfig, RTWeights } from '../types';
import { Rng } from '../rng';
import { parameterShapes } from './config';
import { float32View, loadPublicBinary, loadPublicJson } from './io';

export const WEIGHTS_FORMAT = 'rt-weights-v1';

export interface WeightsManifest {
  format?: string;
  config: RTConfig;
  meta?: Record<string, unknown>;
  tensors: Record<string, { shape: number[]; offset: number; length: number }>;
}

/** Check that `w` holds exactly the parameters of `w.config` with the right shapes. Throws otherwise. */
export function validateWeights(w: RTWeights): void {
  const expected = parameterShapes(w.config);
  const missing: string[] = [];
  for (const [name, shape] of expected) {
    const t = w.tensors[name];
    if (!t) {
      missing.push(name);
      continue;
    }
    if (t.shape.length !== shape.length || t.shape.some((s, i) => s !== shape[i])) {
      throw new Error(`weights: ${name} has shape [${t.shape}], expected [${shape}]`);
    }
    const n = shape.reduce((a, b) => a * b, 1);
    if (t.data.length !== n) throw new Error(`weights: ${name} has ${t.data.length} values, expected ${n}`);
  }
  if (missing.length) throw new Error(`weights: missing tensors ${missing.slice(0, 5).join(', ')}${missing.length > 5 ? ', …' : ''}`);
  const extra = Object.keys(w.tensors).filter((k) => !expected.some(([n]) => n === k));
  if (extra.length) throw new Error(`weights: unexpected tensors ${extra.slice(0, 5).join(', ')}`);
}

/** Build RTWeights from a manifest and the .bin bytes (zero-copy views on little-endian hosts). */
export function parseWeights(manifest: WeightsManifest, bin: ArrayBuffer): RTWeights {
  if (manifest.format && manifest.format !== WEIGHTS_FORMAT) {
    throw new Error(`weights: unsupported format ${manifest.format}`);
  }
  const total = bin.byteLength / 4;
  const tensors: RTWeights['tensors'] = {};
  for (const [name, t] of Object.entries(manifest.tensors)) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    if (n !== t.length) throw new Error(`weights: ${name} length ${t.length} != prod(shape) ${n}`);
    if (t.offset < 0 || t.offset + t.length > total) throw new Error(`weights: ${name} out of bounds`);
    tensors[name] = { shape: t.shape.slice(), data: float32View(bin, t.offset, t.length) };
  }
  const w: RTWeights = { config: { ...manifest.config }, tensors, meta: { ...(manifest.meta ?? {}) } };
  validateWeights(w);
  return w;
}

/** Serialize RTWeights to the same format (manifest + one contiguous Float32Array). */
export function serializeWeights(w: RTWeights): { manifest: WeightsManifest; bin: Float32Array } {
  const order = parameterShapes(w.config);
  let total = 0;
  for (const [name] of order) total += w.tensors[name].data.length;
  const bin = new Float32Array(total);
  const tensors: WeightsManifest['tensors'] = {};
  let off = 0;
  for (const [name] of order) {
    const t = w.tensors[name];
    bin.set(t.data, off);
    tensors[name] = { shape: t.shape.slice(), offset: off, length: t.data.length };
    off += t.data.length;
  }
  return { manifest: { format: WEIGHTS_FORMAT, config: { ...w.config }, meta: { ...w.meta }, tensors }, bin };
}

const cache = new Map<string, Promise<RTWeights>>();

/**
 * Load a checkpoint `public/data/weights/<name>.{json,bin}`.
 * Browser: fetch under `${BASE_URL}data/weights/`. Node: fs from `<cwd>/public/data/weights/`.
 */
export function loadWeights(name: string): Promise<RTWeights> {
  let p = cache.get(name);
  if (!p) {
    p = (async () => {
      const [manifest, bin] = await Promise.all([
        loadPublicJson<WeightsManifest>(`data/weights/${name}.json`),
        loadPublicBinary(`data/weights/${name}.bin`),
      ]);
      const w = parseWeights(manifest, bin);
      w.meta = { ...w.meta, ckpt: name };
      return w;
    })();
    p.catch(() => cache.delete(name));
    cache.set(name, p);
  }
  return p;
}

/**
 * PyTorch-default initialization (spec §7.4; no custom init in PE/rt/model.py):
 *  - nn.Linear weight: kaiming_uniform_(a=√5) ⇒ U(−1/√fan_in, 1/√fan_in); bias: U(−1/√fan_in, 1/√fan_in);
 *  - nn.RMSNorm γ = 1;
 *  - mask_embs: torch.randn(d) ⇒ N(0, 1) (model.py:143).
 * Draws come from the site's seeded PRNG, so values differ from torch's generator but the
 * distributions are identical. Deterministic in `seed`.
 */
export function initWeights(config: RTConfig, seed: number): RTWeights {
  const rng = new Rng(seed);
  const tensors: RTWeights['tensors'] = {};
  const shapes = parameterShapes(config);
  const fanIn = new Map<string, number>();
  for (const [name, shape] of shapes) if (name.endsWith('.weight') && shape.length === 2) fanIn.set(name.slice(0, -7), shape[1]);
  for (const [name, shape] of shapes) {
    const n = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(n);
    if (name.startsWith('mask_embs.')) {
      for (let i = 0; i < n; i++) data[i] = rng.normal();
    } else if (shape.length === 2) {
      const bound = 1 / Math.sqrt(shape[1]);
      for (let i = 0; i < n; i++) data[i] = rng.uniform(-bound, bound);
    } else if (name.endsWith('.bias')) {
      const fi = fanIn.get(name.slice(0, -5));
      if (fi === undefined) throw new Error(`initWeights: no weight for bias ${name}`);
      const bound = 1 / Math.sqrt(fi);
      for (let i = 0; i < n; i++) data[i] = rng.uniform(-bound, bound);
    } else {
      data.fill(1); // RMSNorm γ (norm_dict.*, blocks.*.norms.*, norm_out)
    }
    tensors[name] = { shape: shape.slice(), data };
  }
  return { config: { ...config }, tensors, meta: { name: 'init', init: 'pytorch-default', seed } };
}

/** Typed accessor that throws on a missing tensor. */
export function tensor(w: RTWeights, name: string): Float32Array {
  const t = w.tensors[name];
  if (!t) throw new Error(`weights: missing tensor ${name}`);
  return t.data;
}
