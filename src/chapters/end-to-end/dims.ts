/**
 * ShapeTag wording for the "start to end" page: what the rows and columns of each matrix are. Axis
 * phrases come from ../shapes.ts, so this page explains an axis the same way as every other chapter.
 * Every token axis is the page's span (the real tokens, rows [0, numReal)); weights note how PyTorch
 * stores them ([out, in], rt/model.py nn.Linear) and that the page draws them transposed.
 */
import type { ReactNode } from 'react';
import type { Shape } from '@/components';
import { axis } from '../shapes';
import type { E2ELive } from './live';

type Live = Pick<E2ELive, 'span' | 'd' | 'H' | 'dh' | 'dFF' | 'dText'>;

/** Token rows × model-width columns (most activations). */
export const tokD = (l: Live, note?: ReactNode): Shape => ({ rows: axis.tokens(l.span), cols: axis.model(l.d), note });

/** Token rows × MiniLM-width columns (the two text fields). */
export const tokText = (l: Live, note?: ReactNode): Shape => ({ rows: axis.tokens(l.span), cols: axis.text(l.dText), note });

/** Token rows × SwiGLU hidden units. */
export const tokFF = (l: Live, note?: ReactNode): Shape => ({ rows: axis.tokens(l.span), cols: axis.ffn(l.dFF, l.d), note });

/** Token rows × one head's slice of `of` (Q, K, V). */
export const tokHead = (l: Live, head: number, of: string, role?: string): Shape => ({
  rows: axis.tokens(l.span, role),
  cols: `${axis.head(l.dh, l.d, l.H)}; here head ${head + 1}, dimensions ${head * l.dh}–${(head + 1) * l.dh - 1} of ${of}`,
});

/** Query rows × key columns (scores, masks, attention weights). */
export const qk = (l: Live, note?: ReactNode): Shape => ({ rows: axis.queries(l.span), cols: axis.keys(l.span), note });

/** An nn.Linear weight, stored [out, in] and drawn transposed ([in, out]) so x Wᵀ reads row × column. */
export const weightT = (input: string, output: string, stored: string, note?: string): Shape => ({
  rows: `one per input dimension: ${input}`,
  cols: `one per output dimension: ${output}`,
  note: `PyTorch stores this nn.Linear weight as [out, in] = ${stored}; it is drawn transposed, so the product reads row × column. Learned in training, the same for every token.${note ? ` ${note}` : ''}`,
});
