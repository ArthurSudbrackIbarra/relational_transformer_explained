/**
 * Frozen text-encoder lookups (spec §3.6).
 *
 * RT never runs its text encoder at train or inference time: every string (table names,
 * "<col> of <table>" column phrases, string cell values) is embedded once offline with
 * all-MiniLM-L12-v2 (mean pooling + L2 normalisation, 384-d) and looked up by index
 * (`get_text_emb`, fly.rs:525-529). We mirror that with a precomputed table produced by
 * python/embed.py:
 *
 *   public/data/embeddings/minilm-l12-v2.f32   float32 LE, N×dim row-major
 *   public/data/embeddings/minilm-l12-v2.json  the N strings in the same order
 *                                              (a plain array, or {strings, dim?, ...})
 *
 * `get()` of a string that was not embedded THROWS: the simulation must never fabricate
 * an embedding (BUILD_PLAN §0 "never invent behaviour").
 */
import { float32View, loadPublicBinary, loadPublicJson } from './model/io';

export interface EmbeddingStore {
  /** Embedding width (384 for all-MiniLM-L12-v2). */
  dim: number;
  /** The embedding of `text` (a zero-copy view; do not mutate). Throws if `text` was never embedded. */
  get(text: string): Float32Array;
  has(text: string): boolean;
  /** All embedded strings, in storage order. */
  strings: readonly string[];
}

export const DEFAULT_EMBEDDING_NAME = 'minilm-l12-v2';

/**
 * Build a store from a string list and a row-major N×dim Float32Array.
 * Duplicate strings are allowed only if their vectors are identical (first wins).
 */
export function createEmbeddingStore(strings: readonly string[], data: Float32Array, dim?: number): EmbeddingStore {
  const n = strings.length;
  const D = dim ?? (n > 0 ? data.length / n : 0);
  if (!Number.isInteger(D) || D <= 0) throw new Error(`EmbeddingStore: cannot infer dim (data ${data.length}, strings ${n})`);
  if (data.length !== n * D) throw new Error(`EmbeddingStore: data length ${data.length} != ${n} strings × ${D}`);
  const index = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    const s = strings[i];
    const prev = index.get(s);
    if (prev === undefined) {
      index.set(s, i);
    } else {
      for (let j = 0; j < D; j++) {
        if (data[prev * D + j] !== data[i * D + j]) {
          throw new Error(`EmbeddingStore: duplicate string ${JSON.stringify(s)} with different vectors`);
        }
      }
    }
  }
  const frozen = Object.freeze(strings.slice());
  return {
    dim: D,
    strings: frozen,
    has: (text: string) => index.has(text),
    get(text: string): Float32Array {
      const i = index.get(text);
      if (i === undefined) {
        throw new Error(
          `EmbeddingStore: no precomputed embedding for ${JSON.stringify(text)}. ` +
            'Re-run scripts/export-strings.ts and python/embed.py; embeddings are never fabricated.',
        );
      }
      return data.subarray(i * D, (i + 1) * D);
    },
  };
}

interface EmbeddingIndexJson {
  strings: string[];
  dim?: number;
}

/** Parse the JSON index (plain string array or `{strings, dim}`) + the raw f32 bytes. */
export function parseEmbeddingStore(json: unknown, bin: ArrayBuffer): EmbeddingStore {
  let strings: string[];
  let dim: number | undefined;
  if (Array.isArray(json)) {
    strings = json as string[];
  } else if (json && typeof json === 'object' && Array.isArray((json as EmbeddingIndexJson).strings)) {
    strings = (json as EmbeddingIndexJson).strings;
    dim = (json as EmbeddingIndexJson).dim;
  } else {
    throw new Error('EmbeddingStore: index JSON must be a string array or {strings: string[]}');
  }
  if (bin.byteLength % 4 !== 0) throw new Error('EmbeddingStore: .f32 byte length not a multiple of 4');
  const data = float32View(bin, 0, bin.byteLength / 4);
  return createEmbeddingStore(strings, data, dim);
}

const cache = new Map<string, Promise<EmbeddingStore>>();

/**
 * Load the precomputed embedding table. Browser: fetch from `${BASE_URL}data/embeddings/`.
 * Node (tsx/vitest): read `<cwd>/public/data/embeddings/`. Cached per name.
 */
export function loadEmbeddingStore(name: string = DEFAULT_EMBEDDING_NAME): Promise<EmbeddingStore> {
  let p = cache.get(name);
  if (!p) {
    p = (async () => {
      const [json, bin] = await Promise.all([
        loadPublicJson(`data/embeddings/${name}.json`),
        loadPublicBinary(`data/embeddings/${name}.f32`),
      ]);
      return parseEmbeddingStore(json, bin);
    })();
    // do not cache failures (e.g. assets not generated yet)
    p.catch(() => cache.delete(name));
    cache.set(name, p);
  }
  return p;
}
