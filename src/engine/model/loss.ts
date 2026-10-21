/**
 * Losses of the RT (spec §8.2, mirrors model.py:231-265).
 *
 *   loss = Σ_i [mask_i] · ℓ_i / Σ_i mask_i
 *   ℓ_i  = Huber_δ=1(ŷ_number,   y_number)                 sem = number   (model.py:250-251)
 *          Huber_δ=1(ŷ_datetime, y_datetime)               sem = datetime
 *          BCEWithLogits(ŷ_boolean, 1{y_boolean > 0})      sem = boolean  (model.py:252-255)
 *          error "masking text not supported"              sem = text     (model.py:256-257)
 *
 * No per-task/type/class weighting, no label smoothing (spec §8.2). Targets y are the
 * z-scored values stored in the (masked) token itself — the value the input gate hid.
 */
import { SemType, type Token } from '../types';

/** F.huber_loss with the default δ = 1: 0.5·d² if |d| < δ else δ·(|d| − 0.5·δ). huber(3, 0) = 2.5. */
export function huber(yhat: number, y: number, delta = 1): number {
  const d = Math.abs(yhat - y);
  return d < delta ? 0.5 * d * d : delta * (d - 0.5 * delta);
}

/** F.binary_cross_entropy_with_logits(z, y) = max(z,0) − z·y + log(1 + e^{−|z|}). bce(0, 1) = ln 2. */
export function bceWithLogits(z: number, y: number): number {
  return Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));
}

export interface LossTerm {
  pos: number;
  sem: SemType;
  /** Which decoder produced ŷ. */
  head: 'number' | 'datetime' | 'boolean';
  yhat: number;
  /** z-scored target value read from the token (number/datetime), or the boolean z-score. */
  y: number;
  /** BCE label 1{y > 0} (boolean only). */
  label?: 0 | 1;
  loss: number;
}

export interface DecoderOutputs {
  number: Float32Array;
  datetime: Float32Array;
  boolean: Float32Array;
}

/**
 * Per-masked-cell loss terms, in position order. Throws for a masked text cell, exactly like
 * model.py:256-257 ("masking text not supported").
 */
export function lossTerms(tokens: readonly Token[], yhat: DecoderOutputs): LossTerm[] {
  const terms: LossTerm[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    // model.py:241 uses `masks` only (not is_padding); a padding token never has mask = true (fly.rs:65-84).
    if (!t.mask) continue;
    switch (t.sem) {
      case SemType.Number: {
        const yh = yhat.number[i];
        terms.push({ pos: i, sem: t.sem, head: 'number', yhat: yh, y: t.numberValue, loss: huber(yh, t.numberValue) });
        break;
      }
      case SemType.DateTime: {
        const yh = yhat.datetime[i];
        terms.push({ pos: i, sem: t.sem, head: 'datetime', yhat: yh, y: t.datetimeValue, loss: huber(yh, t.datetimeValue) });
        break;
      }
      case SemType.Boolean: {
        const yh = yhat.boolean[i];
        const label: 0 | 1 = t.booleanValue > 0 ? 1 : 0; // (y > 0).float(), model.py:254
        terms.push({ pos: i, sem: t.sem, head: 'boolean', yhat: yh, y: t.booleanValue, label, loss: bceWithLogits(yh, label) });
        break;
      }
      case SemType.Text:
        throw new Error('masking text not supported'); // model.py:257
      default:
        throw new Error(`unknown sem type ${t.sem as number}`);
    }
  }
  return terms;
}

/**
 * Mean over masked cells (model.py:265 `loss_out / masks.sum()`).
 * With no masked cell this is 0/0 = NaN, as in the official code.
 */
export function meanLoss(terms: readonly LossTerm[]): number {
  let s = 0;
  for (const t of terms) s += t.loss;
  return s / terms.length;
}
