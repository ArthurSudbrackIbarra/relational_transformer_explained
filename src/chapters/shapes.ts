/**
 * The wording of every ShapeTag (src/components/viz/ShapeTag.tsx): what one row or one column of a
 * matrix or vector is, and where its size comes from. Kept in one place so the same axis reads the
 * same way in every chapter. Sizes are passed in from the live model and context; the paper's sizes
 * come from PAPER_CONFIG and PAPER_L (spec §7.3, §9.2).
 */
import type { ReactNode } from 'react';
import type { Shape } from '@/components';
import { PAPER_CONFIG, PAPER_L } from '@/engine/model/config';

const P = PAPER_CONFIG;
const n = (x: number) => x.toLocaleString('en-US');

/** Which tokens of the context window an axis holds. */
export interface TokenSpan {
  /** Tokens drawn along the axis. */
  shown: number;
  /** The context length L: positions, padding included. */
  L: number;
  /** Real (non-padding) tokens of the context. */
  real: number;
  /** Display order: the context's own (default) or grouped by table and column. */
  order?: 'context' | 'grouped';
}

const ordered = (o?: TokenSpan['order']) =>
  o === 'grouped' ? 'grouped by table, then by column, then by context position' : 'in the order the sampler emitted them';

/** Which tokens an axis holds, as a phrase (for custom axis wording). */
export function span({ shown, L, real }: TokenSpan): string {
  if (shown >= L) {
    const pad = real < L ? `; the last ${n(L - real)} are padding` : '';
    return `all L = ${n(L)} positions of the context window (the “Context length L” setting at the top; ${n(PAPER_L)} in the paper)${pad}`;
  }
  if (shown === real) return `the ${n(real)} real tokens of the context window; its ${n(L - real)} padding positions (L = ${n(L)}) are left out`;
  return `the first ${n(shown)} of the context’s ${n(real)} tokens`;
}

export const axis = {
  /** Context tokens, one per row or column; `role` names them (e.g. "key" for the columns of Kᵀ). */
  tokens: (t: TokenSpan, role?: string) => `one per ${role ? `${role} token` : 'token'}, ${ordered(t.order)}: ${span(t)}`,
  /** Attention rows. */
  queries: (t: TokenSpan) => `one per query token, the one doing the looking, ${ordered(t.order)}: ${span(t)}`,
  /** Attention columns. */
  keys: (t: TokenSpan) => `one per key token, the one being looked at, in the same order as the rows: ${span(t)}`,
  /** The residual stream's width. */
  model: (d: number) =>
    `one per dimension of the model width d = ${d}, the length of every token’s vector inside the model (${P.dModel} in the paper)`,
  /** A frozen MiniLM embedding. */
  text: (dText: number) =>
    `one per dimension of the frozen text encoder all-MiniLM-L12-v2, which turns any string into ${dText} numbers (the encoder of the official code)`,
  /** One head's slice of Q, K, V or the attention output. */
  head: (dh: number, d: number, H: number) =>
    `one per dimension of one attention head: d / heads = ${d} / ${H} = ${dh} (${P.dModel} / ${P.nHeads} = ${P.dModel / P.nHeads} in the paper)`,
  /** The SwiGLU's hidden layer. */
  ffn: (dFF: number, d: number) => `one per hidden unit of the SwiGLU feed-forward: d_ff = ${dFF} = ${dFF / d} × d (${n(P.dFF)} in the paper)`,
  /** Attention heads. */
  heads: (H: number) => `one per attention head: ${H} here, ${P.nHeads} in the paper`,
};

/** A single vector: one row whose meaning is `row`, and columns as given. */
export function vec(row: ReactNode, cols: ReactNode, note?: ReactNode): Shape {
  return { rows: row, cols, note };
}
