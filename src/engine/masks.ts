/**
 * The four relational attention masks, mirroring `RelationalTransformer.forward` in the paper-era
 * `rt/model.py:164-191` (spec §6.2):
 *
 *   pad[q,k]  = ¬is_padding[q] ∧ ¬is_padding[k]                                  model.py:165
 *   col       = col_name_idx[q]==col_name_idx[k] ∧ table_name_idx[q]==table_name_idx[k] ∧ pad
 *                                                                               model.py:181-183, 189
 *   feat      = (node[q]==node[k] ∨ node[k] ∈ f2p[q]) ∧ pad                     model.py:168-173, 187
 *   nbr       = node[q] ∈ f2p[k] ∧ pad                                          model.py:176-178, 188
 *   full      = pad                                                             model.py:190
 *
 * Dense Uint8Array of size S×S, row-major with ROW = QUERY: m[q*S + k] = 1 iff q may attend to k.
 * The same masks are shared by all heads and all blocks (model.py:89-101, 226-227).
 *
 * The `∧ pad` matters: padding tokens have node_idx −1 and parent lists are −1-padded, so without it
 * `node[k] ∈ f2p[q]` would match padding keys to every empty parent slot (spec §6.2 gotchas).
 */
import { ATTN_ORDER, type AttnLayer, type MaskSet, type Token } from './types';

/** The token fields the masks read (model.py:154-158). Any Token satisfies it. */
export type MaskToken = Pick<Token, 'isPadding' | 'nodeIdx' | 'f2pNbrIdxs' | 'colNameIdx' | 'tableNameIdx'>;

export function buildMasks(tokens: readonly MaskToken[]): MaskSet {
  const S = tokens.length;
  const col = new Uint8Array(S * S);
  const feat = new Uint8Array(S * S);
  const nbr = new Uint8Array(S * S);
  const full = new Uint8Array(S * S);
  for (let q = 0; q < S; q++) {
    const tq = tokens[q];
    if (tq.isPadding) continue; // pad[q,·] = 0
    const row = q * S;
    for (let k = 0; k < S; k++) {
      const tk = tokens[k];
      if (tk.isPadding) continue; // pad[·,k] = 0
      const i = row + k;
      full[i] = 1;
      if (tq.colNameIdx === tk.colNameIdx && tq.tableNameIdx === tk.tableNameIdx) col[i] = 1;
      if (tq.nodeIdx === tk.nodeIdx || tq.f2pNbrIdxs.includes(tk.nodeIdx)) feat[i] = 1;
      if (tk.f2pNbrIdxs.includes(tq.nodeIdx)) nbr[i] = 1;
    }
  }
  return { col, feat, nbr, full };
}

/** Key positions query `q` may attend to in `mask` (S = sequence length). */
export function allowedKeys(mask: Uint8Array, S: number, q: number): number[] {
  const out: number[] = [];
  for (let k = 0; k < S; k++) if (mask[q * S + k]) out.push(k);
  return out;
}

/**
 * Non-padding queries whose row of `mask` is all zero. In `nbr` these are cells whose row has no
 * child row in the context: their attention output is exactly 0 (spec §6.1, torch 2.6 FlexAttention).
 */
export function emptyQueryRows(mask: Uint8Array, tokens: readonly MaskToken[]): number[] {
  const S = tokens.length;
  const out: number[] = [];
  for (let q = 0; q < S; q++) {
    if (tokens[q].isPadding) continue;
    let any = false;
    for (let k = 0; k < S && !any; k++) any = mask[q * S + k] === 1;
    if (!any) out.push(q);
  }
  return out;
}

/** Number of allowed (q,k) pairs per layer. */
export function maskDensity(masks: MaskSet): Record<AttnLayer, number> {
  const out = {} as Record<AttnLayer, number>;
  for (const l of ATTN_ORDER) {
    let c = 0;
    for (const v of masks[l]) c += v;
    out[l] = c;
  }
  return out;
}

/** ASCII rendering in the style of spec §6.3: one line per query, '#' = allowed, '.' = blocked. */
export function maskToAscii(mask: Uint8Array, S: number): string[] {
  const lines: string[] = [];
  for (let q = 0; q < S; q++) {
    let s = '';
    for (let k = 0; k < S; k++) s += mask[q * S + k] ? '#' : '.';
    lines.push(s);
  }
  return lines;
}
