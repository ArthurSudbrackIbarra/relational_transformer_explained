/**
 * Pure helpers for the block chapter. Nothing here adds model semantics: every number comes from the
 * engine's forward-pass capture (ForwardResult.blocks[b].attn[l] / .ffn, a mirror of
 * rt/model.py:50-86 and :226-229) or from the masks (rt/model.py:164-191). The helpers only slice,
 * measure (norms, cosines) and label it, and trace which mask edges exist (spec §6.4, interpretation).
 */
import { tokenValueText } from '@/components';
import { ATTN_ORDER, type AttnLayer, type ForwardResult, type MaskSet, type Token } from '@/engine/types';

// ─────────────────────────────────────────────────────────────────────────────
// Sublayers
// ─────────────────────────────────────────────────────────────────────────────

/** The five sublayers of a block in execution order (model.py:83-85). */
export type SubKind = AttnLayer | 'ffn';
export const SUBLAYERS: readonly SubKind[] = ['col', 'feat', 'nbr', 'full', 'ffn'];

export const SUB_NAME: Record<SubKind, string> = {
  col: 'Column',
  feat: 'Feature',
  nbr: 'Neighbor',
  full: 'Full',
  ffn: 'SwiGLU',
};
export const SUB_SHORT: Record<SubKind, string> = { col: 'col', feat: 'feat', nbr: 'nbr', full: 'full', ffn: 'FFN' };

// ─────────────────────────────────────────────────────────────────────────────
// Small vector maths (display only)
// ─────────────────────────────────────────────────────────────────────────────

export function norm(v: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    s += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na > 0 && nb > 0 ? s / Math.sqrt(na * nb) : NaN;
}

/** Row `pos` of a row-major [S × width] buffer (a copy-free view). */
export function rowOf(buf: Float32Array, pos: number, width: number): Float32Array {
  return buf.subarray(pos * width, (pos + 1) * width);
}

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
/** SiLU(z) = z·σ(z) (F.silu, model.py:59). */
export const siluFn = (z: number) => z * sigmoid(z);

// ─────────────────────────────────────────────────────────────────────────────
// The residual stream of one token
// ─────────────────────────────────────────────────────────────────────────────

export interface Stage {
  /** 0 = embedding x⁽⁰⁾, 1..5·nBlocks = after sublayer, last = after norm_out. */
  index: number;
  kind: 'embed' | SubKind | 'out';
  /** Block index 0-based (−1 for the embedding and norm_out). */
  block: number;
  label: string;
  /** The vector after this stage, length d. */
  x: Float32Array;
  /** The residual update this stage added (sublayers only). */
  update: Float32Array | null;
  /** What the sublayer read: RMSNorm_ℓ(x_in) (sublayers only; null when skipped). */
  h: Float32Array | null;
  xNorm: number;
  updNorm: number;
  /** ‖RMSNorm_ℓ(x_in)‖ (sublayers only). */
  hNorm: number;
  /** Cosine with the output of the last block (before norm_out). */
  cosLast: number;
  /** Attention sublayer switched off by an inference-time ablation. */
  skipped: boolean;
}

/** Every stage of token `pos`'s residual stream, straight from the forward capture. */
export function streamStages(fr: ForwardResult, pos: number): Stage[] {
  const d = fr.d;
  const nB = fr.blocks.length;
  const last = rowOf(nB ? fr.blocks[nB - 1].ffn.xOut : fr.embedding.x0, pos, d);
  const out: Stage[] = [];
  const x0 = rowOf(fr.embedding.x0, pos, d);
  out.push({ index: 0, kind: 'embed', block: -1, label: 'x⁽⁰⁾ embedding', x: x0, update: null, h: null, xNorm: norm(x0), updNorm: NaN, hNorm: NaN, cosLast: cosine(x0, last), skipped: false });
  for (let b = 0; b < nB; b++) {
    for (const s of SUBLAYERS) {
      const cap = s === 'ffn' ? fr.blocks[b].ffn : fr.blocks[b].attn[s];
      const x = rowOf(cap.xOut, pos, d);
      const update = rowOf(cap.out, pos, d);
      const skipped = s !== 'ffn' && fr.blocks[b].attn[s].skipped;
      const h = cap.h.length ? rowOf(cap.h, pos, d) : null;
      out.push({
        index: out.length,
        kind: s,
        block: b,
        label: `B${b + 1} · ${SUB_SHORT[s]}`,
        x,
        update,
        h,
        xNorm: norm(x),
        updNorm: norm(update),
        hNorm: h ? norm(h) : NaN,
        cosLast: cosine(x, last),
        skipped,
      });
    }
  }
  const xf = rowOf(fr.xFinal, pos, d);
  out.push({ index: out.length, kind: 'out', block: -1, label: 'norm_out', x: xf, update: null, h: null, xNorm: norm(xf), updNorm: NaN, hNorm: NaN, cosLast: cosine(xf, last), skipped: false });
  return out;
}

/** Number of keys token `q` may attend to under a mask. */
export function allowedKeys(masks: MaskSet, layer: AttnLayer, q: number, S: number): number {
  const m = masks[layer];
  let n = 0;
  for (let k = 0; k < S; k++) n += m[q * S + k];
  return n;
}

/** Mean over heads of the attention weight query q gives key k in one sublayer (0 if masked). */
export function meanWeight(fr: ForwardResult, block: number, layer: AttnLayer, q: number, k: number, H: number): number {
  const p = fr.blocks[block]?.attn[layer]?.probs;
  if (!p || p.length === 0) return NaN;
  const S = fr.S;
  let s = 0;
  for (let h = 0; h < H; h++) s += p[(h * S + q) * S + k];
  return s / H;
}

// ─────────────────────────────────────────────────────────────────────────────
// Information flow through the masks (spec §6.4: a teaching interpretation)
// ─────────────────────────────────────────────────────────────────────────────

export interface FlowStep {
  block: number;
  layer: AttnLayer;
}

/**
 * The attention sublayers in execution order across the stack. FFNs act per token and move nothing between
 * tokens. `exclude` drops layers switched off by an inference-time ablation (the forward pass skips them,
 * so nothing flows through them).
 */
export function flowOrder(nBlocks: number, includeFull: boolean, exclude: readonly AttnLayer[] = []): FlowStep[] {
  const out: FlowStep[] = [];
  for (let b = 0; b < nBlocks; b++)
    for (const l of ATTN_ORDER) if ((includeFull || l !== 'full') && !exclude.includes(l)) out.push({ block: b, layer: l });
  return out;
}

export interface Hop {
  /** Key token (information comes from here). */
  from: number;
  /** Query token (information arrives here). */
  to: number;
  step: FlowStep;
  /** 1-based index of the step in the order. */
  t: number;
}

export interface FlowTrace {
  order: FlowStep[];
  /** time[i] = 1-based step after which token i first carries the source's information; 0 = source; −1 = never. */
  time: Int16Array;
  /** One shortest-in-time route from the source to `dest`, or null. */
  path: Hop[] | null;
  /** Tokens carrying the source's information after each step (index t-1). */
  reached: number[];
}

/**
 * Forward reachability from one source token. In step t (sublayer `l` of block `b`), query q receives
 * from key k iff mask_l[q,k] = 1 (model.py:186-191): every query reads the stream as it was BEFORE the
 * step, so a token reached at step t can only pass the information on from step t+1.
 */
export function traceFlow(masks: MaskSet, tokens: readonly Token[], src: number, dest: number, order: FlowStep[]): FlowTrace {
  const S = tokens.length;
  const time = new Int16Array(S).fill(-1);
  const parent = new Int32Array(S).fill(-1);
  time[src] = 0;
  const reached: number[] = [];
  let count = 1;
  for (let t = 1; t <= order.length; t++) {
    const m = masks[order[t - 1].layer];
    const newly: [number, number][] = [];
    for (let q = 0; q < S; q++) {
      if (time[q] !== -1 || tokens[q].isPadding) continue;
      let best = -1;
      for (let k = 0; k < S; k++) {
        if (!m[q * S + k]) continue;
        const tk = time[k];
        if (tk === -1 || tk >= t) continue;
        if (best === -1 || tk < time[best]) best = k;
      }
      if (best !== -1) newly.push([q, best]);
    }
    for (const [q, k] of newly) {
      time[q] = t;
      parent[q] = k;
    }
    count += newly.length;
    reached.push(count);
  }
  let path: Hop[] | null = null;
  if (time[dest] >= 0) {
    path = [];
    let cur = dest;
    while (cur !== src && parent[cur] !== -1) {
      const t = time[cur];
      path.unshift({ from: parent[cur], to: cur, step: order[t - 1], t });
      cur = parent[cur];
    }
  }
  return { order, time, path, reached };
}

/**
 * For every token, the first step after which its information can be in `dest`'s stream (bitset
 * propagation of all sources at once). arrival[i] = 0 for dest itself, −1 if never within the stack.
 */
export function arrivalAtDest(masks: MaskSet, tokens: readonly Token[], dest: number, order: FlowStep[]): Int16Array {
  const S = tokens.length;
  const W = Math.ceil(S / 32);
  let reach = new Uint32Array(S * W);
  for (let i = 0; i < S; i++) if (!tokens[i].isPadding) reach[i * W + (i >> 5)] |= 1 << (i & 31);
  const arrival = new Int16Array(S).fill(-1);
  arrival[dest] = 0;
  const mark = (t: number) => {
    for (let i = 0; i < S; i++) {
      if (arrival[i] !== -1) continue;
      if (reach[dest * W + (i >> 5)] & (1 << (i & 31))) arrival[i] = t;
    }
  };
  for (let t = 1; t <= order.length; t++) {
    const m = masks[order[t - 1].layer];
    const next = reach.slice();
    for (let q = 0; q < S; q++) {
      if (tokens[q].isPadding) continue;
      for (let k = 0; k < S; k++) {
        if (!m[q * S + k] || k === q) continue;
        for (let w = 0; w < W; w++) next[q * W + w] |= reach[k * W + w];
      }
    }
    reach = next;
    mark(t);
  }
  return arrival;
}

// ─────────────────────────────────────────────────────────────────────────────
// Names and labels (display only)
// ─────────────────────────────────────────────────────────────────────────────

/** Shortest unique prefix per table name, e.g. ShopDB → c, o, p, r. */
export function tablePrefixes(names: readonly string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const n of names) {
    let len = 1;
    while (len < n.length && names.some((m) => m !== n && m.startsWith(n.slice(0, len)))) len++;
    out[n] = n.slice(0, len);
  }
  return out;
}

/** Full row name: "customers[35]" or "customer-churn:Val[32]". */
export function rowName(t: Token): string {
  if (t.isPadding) return 'padding';
  if (t.tableType && t.tableType !== 'Db') return `${t.tableName}:${t.tableType}[${t.rowPos}]`;
  return `${t.tableName}[${t.rowPos}]`;
}

/** Short row tag: "c35", "o70", "val32", "train258". */
export function rowTag(t: Token, prefixes: Record<string, string>): string {
  if (t.isPadding) return 'pad';
  if (t.tableType && t.tableType !== 'Db') return `${t.tableType.toLowerCase()}${t.rowPos}`;
  return `${prefixes[t.tableName] ?? t.tableName[0]}${t.rowPos}`;
}

/** Short token label: "c35·age". */
export function tokenLabel(t: Token, prefixes: Record<string, string>): string {
  if (t.isPadding) return `pad #${t.pos}`;
  return `${rowTag(t, prefixes)}·${t.colName}`;
}

export const valueOf = (t: Token) => tokenValueText(t);

/** "customer" from "customers". */
export function singular(table: string): string {
  if (table.endsWith('ies')) return `${table.slice(0, -3)}y`;
  if (table.endsWith('s')) return table.slice(0, -1);
  return table;
}

export function article(word: string): string {
  return /^[aeiou]/i.test(word) ? 'an' : 'a';
}

export const int = (n: number) => n.toLocaleString('en-US');
export const fx = (v: number, digits = 2) => (Number.isFinite(v) ? v.toFixed(digits) : '–');
