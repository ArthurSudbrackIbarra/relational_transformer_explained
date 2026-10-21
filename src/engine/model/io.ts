/**
 * Environment-agnostic asset loading for the engine (browser AND Node/tsx/vitest).
 *
 * - Browser: assets live under `public/data/...` and are served at
 *   `${import.meta.env.BASE_URL}data/...` (Vite copies `public/` verbatim).
 * - Node: the same files are read from `<process.cwd()>/public/data/...` with `fs`.
 *
 * `node:fs` / `node:path` are imported dynamically through a variable specifier so that
 * (a) Vite never tries to bundle them for the browser and (b) the browser-only
 * tsconfig (no @types/node) still type-checks this file.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

/** True when running under Node (tsx, vitest), false in a browser or web worker. */
export function isNodeEnv(): boolean {
  const g = globalThis as any;
  return typeof g.window === 'undefined' && typeof g.document === 'undefined' && !!g.process?.versions?.node;
}

/** Vite's base URL ('./' for this app), or '/' when not running under Vite. */
export function baseUrl(): string {
  const env = (import.meta as any).env;
  const b: unknown = env?.BASE_URL;
  return typeof b === 'string' && b.length > 0 ? b : '/';
}

async function nodeFs(): Promise<any> {
  const spec = 'node:fs/promises';
  return import(/* @vite-ignore */ spec);
}

async function nodePath(): Promise<any> {
  const spec = 'node:path';
  return import(/* @vite-ignore */ spec);
}

/** Absolute path of `public/<rel>` in Node (relative to process.cwd()). */
export async function publicPath(rel: string): Promise<string> {
  const path = await nodePath();
  return path.resolve((globalThis as any).process.cwd(), 'public', rel);
}

/** Read a file as a fresh, 4-byte-aligned ArrayBuffer (Node only). */
export async function readFileBuffer(absPath: string): Promise<ArrayBuffer> {
  const fs = await nodeFs();
  const buf: Uint8Array = await fs.readFile(absPath);
  // Node Buffers may be views into a shared pool at an unaligned offset: copy.
  const out = new ArrayBuffer(buf.byteLength);
  new Uint8Array(out).set(buf);
  return out;
}

/** Read and parse a JSON file (Node only). */
export async function readFileJson<T = unknown>(absPath: string): Promise<T> {
  const fs = await nodeFs();
  return JSON.parse(await fs.readFile(absPath, 'utf8')) as T;
}

/**
 * Load `public/<rel>` as JSON from wherever the engine runs.
 * Browser: fetch(`${BASE_URL}<rel>`). Node: fs.
 */
export async function loadPublicJson<T = unknown>(rel: string): Promise<T> {
  if (isNodeEnv()) return readFileJson<T>(await publicPath(rel));
  const res = await fetch(`${baseUrl()}${rel}`);
  if (!res.ok) throw new Error(`fetch ${rel}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

/** Load `public/<rel>` as raw bytes from wherever the engine runs. */
export async function loadPublicBinary(rel: string): Promise<ArrayBuffer> {
  if (isNodeEnv()) return readFileBuffer(await publicPath(rel));
  const res = await fetch(`${baseUrl()}${rel}`);
  if (!res.ok) throw new Error(`fetch ${rel}: HTTP ${res.status}`);
  return res.arrayBuffer();
}

/** True on little-endian hosts (every realistic browser/Node platform). */
export const IS_LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/**
 * View `count` float32 values starting at element `offset` of a little-endian buffer.
 * Zero-copy on little-endian hosts; decoded through a DataView otherwise.
 */
export function float32View(buf: ArrayBuffer, offset: number, count: number): Float32Array {
  if (IS_LITTLE_ENDIAN) return new Float32Array(buf, offset * 4, count);
  const dv = new DataView(buf, offset * 4, count * 4);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = dv.getFloat32(i * 4, true);
  return out;
}
