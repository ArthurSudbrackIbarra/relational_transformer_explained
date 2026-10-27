/**
 * The "start to end" page recomputes single entries to show where they come from. These tests
 * recompute EVERY displayed quantity of the focus rows from its operands and compare it with the
 * engine's capture (model/forward.ts, itself checked against PyTorch and the official model.py),
 * on the real trained checkpoint and the live default context. If a re-derivation drifted from the
 * forward pass, the page would show arithmetic that is not the model's; these tests forbid that.
 */
import { describe, expect, it } from 'vitest';
import { loadEmbeddingStore } from '@/engine/embeddings';
import { denormalize, sigmoid } from '@/engine/metrics';
import { tensor } from '@/engine/model/weights';
import { runPipeline } from '@/engine/pipeline';
import { targetStats } from '@/engine/sampler';
import { ATTN_ORDER, SemType, type ForwardResult, type RTWeights, type SampledContext } from '@/engine/types';
import { textFields } from './live';
import {
  dotTerms,
  headPanel,
  headSlice,
  headSliceT,
  linearEntry,
  maskReasons,
  maskedSoftmaxRow,
  rmsRow,
  rowOf,
  scalarInput,
  scoreEntry,
  silu,
  weightedSumEntry,
} from './math';

const TOL = 2e-5;

function close(a: number, b: number, tol = TOL) {
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b)));
}

function closeVec(a: ArrayLike<number>, b: ArrayLike<number>, tol = TOL) {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) close(a[i], b[i], tol);
}

const SEM_KEY = ['number', 'text', 'datetime', 'boolean'] as const;

async function setup(taskId: string) {
  const run = await runPipeline({ dbName: taskId.split('/')[0] as 'shop', taskId, split: 'Val' });
  const fr: ForwardResult = run.forward;
  const w: RTWeights = run.assets.checkpoint.weights;
  return { ...run, fr, w, weightsKind: run.assets.checkpoint.kind };
}

/** Positions to check: the target plus one token of every sem type present (first occurrence). */
function probePositions(ctx: SampledContext): number[] {
  const out = new Set<number>([ctx.targetPos]);
  for (const s of [SemType.Number, SemType.Text, SemType.DateTime, SemType.Boolean]) {
    const t = ctx.tokens.find((x) => !x.isPadding && !x.isTarget && x.sem === s);
    if (t) out.add(t.pos);
  }
  return [...out];
}

describe.each(['shop/customer-churn', 'shop/product-sales'])('start-to-end re-derivations match the forward pass (%s)', (taskId) => {
  it('uses trained weights and a full forward capture', async () => {
    const { fr, w, weightsKind } = await setup(taskId);
    expect(weightsKind).toBe('trained');
    expect(fr.blocks.length).toBe(w.config.nBlocks);
    expect(fr.blocks[0].attn.col.scores.length).toBe(w.config.nHeads * fr.S * fr.S);
  });

  it('input embedding: text lookups, both encoders, both RMSNorms, the mask vector, the sum', async () => {
    const { fr, w, ctx, pdb } = await setup(taskId);
    const store = await loadEmbeddingStore();
    const { dModel: d, dText, eps } = w.config;
    const fields = textFields({ ctx, pdb, dText }, store);
    for (const pos of probePositions(ctx)) {
      const t = ctx.tokens[pos];
      // col_name_values row = E("<col> of <table>"); text_values row = E(text[text_idx]) (index 0 for non-text cells)
      closeVec(rowOf(fields.colName, pos, dText), store.get(pdb.text[t.colNameIdx]), 0);
      closeVec(rowOf(fields.text, pos, dText), store.get(pdb.text[t.sem === SemType.Text ? t.textIdx : 0]), 0);
      // column-name term
      const c = rowOf(fields.colName, pos, dText);
      const pre = Float32Array.from({ length: d }, (_, j) => linearEntry(c, tensor(w, 'enc_dict.col_name.weight'), dText, j, tensor(w, 'enc_dict.col_name.bias')).value);
      closeVec(pre, rowOf(fr.embedding.colPreNorm, pos, d));
      closeVec(rmsRow(pre, tensor(w, 'norm_dict.col_name.weight'), eps).y, rowOf(fr.embedding.colTerm, pos, d));
      // value term
      const key = SEM_KEY[t.sem];
      if (t.mask) {
        closeVec(rowOf(fr.embedding.valueTerm, pos, d), tensor(w, `mask_embs.${key}`), 0);
      } else {
        const Wt = tensor(w, `enc_dict.${key}.weight`);
        const bt = tensor(w, `enc_dict.${key}.bias`);
        const v = t.sem === SemType.Text ? rowOf(fields.text, pos, dText) : Float32Array.of(scalarInput(t)!);
        const vp = Float32Array.from({ length: d }, (_, j) => linearEntry(v, Wt, v.length, j, bt).value);
        closeVec(vp, rowOf(fr.embedding.valuePreNorm, pos, d));
        closeVec(rmsRow(vp, tensor(w, `norm_dict.${key}.weight`), eps).y, rowOf(fr.embedding.valueTerm, pos, d));
      }
      const x0 = rowOf(fr.embedding.x0, pos, d);
      for (let j = 0; j < d; j++) close(x0[j], fr.embedding.colTerm[pos * d + j] + fr.embedding.valueTerm[pos * d + j]);
    }
  });

  it('every attention sublayer: RMSNorm, Q/K/V, scores, masked softmax, weighted sum, W_o, residual', async () => {
    const { fr, w, ctx } = await setup(taskId);
    const { dModel: d, nHeads: H, eps } = w.config;
    const dh = d / H;
    const S = fr.S;
    for (let b = 0; b < fr.blocks.length; b++) {
      for (const l of ATTN_ORDER) {
        const cap = fr.blocks[b].attn[l];
        const p = `blocks.${b}.attns.${l}`;
        for (const pos of probePositions(ctx)) {
          closeVec(rmsRow(rowOf(cap.xIn, pos, d), tensor(w, `blocks.${b}.norms.${l}.weight`), eps).y, rowOf(cap.h, pos, d));
          for (const [name, buf] of [['wq', cap.q], ['wk', cap.k], ['wv', cap.v]] as const) {
            const row = Float32Array.from({ length: d }, (_, j) => linearEntry(rowOf(cap.h, pos, d), tensor(w, `${p}.${name}.weight`), d, j).value);
            closeVec(row, rowOf(buf, pos, d));
          }
          const allowed = fr.masks[l].subarray(pos * S, (pos + 1) * S);
          for (let h = 0; h < H; h++) {
            const sc = headPanel(cap.scores, S, h);
            for (let k = 0; k < ctx.numReal; k++) close(scoreEntry(cap.q, cap.k, d, dh, h, pos, k).value, sc[pos * S + k]);
            const pr = headPanel(cap.probs, S, h).subarray(pos * S, (pos + 1) * S);
            closeVec(maskedSoftmaxRow(sc.subarray(pos * S, (pos + 1) * S), allowed).p, pr);
            for (let j = 0; j < dh; j++) close(weightedSumEntry(pr, cap.v, d, dh, h, j).value, cap.headOut[pos * d + h * dh + j]);
          }
          const out = Float32Array.from({ length: d }, (_, j) => linearEntry(rowOf(cap.headOut, pos, d), tensor(w, `${p}.wo.weight`), d, j).value);
          closeVec(out, rowOf(cap.out, pos, d));
          for (let j = 0; j < d; j++) close(cap.xOut[pos * d + j], cap.xIn[pos * d + j] + cap.out[pos * d + j]);
        }
      }
    }
  });

  it('head slices and their transposes index the captured q / k as the page draws them', async () => {
    const { fr, w } = await setup(taskId);
    const { dModel: d, nHeads: H } = w.config;
    const dh = d / H;
    const cap = fr.blocks[0].attn.full;
    const S = fr.S;
    for (let h = 0; h < H; h++) {
      const qh = headSlice(cap.q, S, d, h, dh);
      const khT = headSliceT(cap.k, S, d, h, dh);
      // (Qₕ Kₕᵀ)[q, k] / √dh = captured score, for a few pairs
      for (const [q, k] of [[0, 1], [1, 0], [5, 9], [S - 1, 2]]) {
        let s = 0;
        for (let j = 0; j < dh; j++) s += qh[q * dh + j] * khT[j * S + k];
        close(s / Math.sqrt(dh), headPanel(cap.scores, S, h)[q * S + k]);
      }
    }
  });

  it('every SwiGLU sublayer: RMSNorm, W₁ and W₃, SiLU gate, W₂, residual', async () => {
    const { fr, w, ctx } = await setup(taskId);
    const { dModel: d, dFF, eps } = w.config;
    for (let b = 0; b < fr.blocks.length; b++) {
      const f = fr.blocks[b].ffn;
      for (const pos of probePositions(ctx)) {
        const h = rowOf(f.h, pos, d);
        closeVec(rmsRow(rowOf(f.xIn, pos, d), tensor(w, `blocks.${b}.norms.ffn.weight`), eps).y, h);
        const a = Float32Array.from({ length: dFF }, (_, j) => linearEntry(h, tensor(w, `blocks.${b}.ffn.w1.weight`), d, j).value);
        const c = Float32Array.from({ length: dFF }, (_, j) => linearEntry(h, tensor(w, `blocks.${b}.ffn.w3.weight`), d, j).value);
        closeVec(a, rowOf(f.a, pos, dFF));
        closeVec(c, rowOf(f.b, pos, dFF));
        const g = Float32Array.from({ length: dFF }, (_, j) => silu(f.a[pos * dFF + j]) * f.b[pos * dFF + j]);
        closeVec(g, rowOf(f.gated, pos, dFF));
        const out = Float32Array.from({ length: d }, (_, j) => linearEntry(rowOf(f.gated, pos, dFF), tensor(w, `blocks.${b}.ffn.w2.weight`), dFF, j).value);
        closeVec(out, rowOf(f.out, pos, d));
        for (let j = 0; j < d; j++) close(f.xOut[pos * d + j], f.xIn[pos * d + j] + f.out[pos * d + j]);
      }
    }
  });

  it('final RMSNorm, the three scalar decoders at every position, and the read-out', async () => {
    const { fr, w, ctx, pdb, task } = await setup(taskId);
    const { dModel: d, eps } = w.config;
    const last = fr.blocks[fr.blocks.length - 1].ffn.xOut;
    for (let pos = 0; pos < ctx.numReal; pos++) {
      const xf = rmsRow(rowOf(last, pos, d), tensor(w, 'norm_out.weight'), eps).y;
      closeVec(xf, rowOf(fr.xFinal, pos, d));
      for (const key of ['number', 'datetime', 'boolean'] as const) {
        close(linearEntry(rowOf(fr.xFinal, pos, d), tensor(w, `dec_dict.${key}.weight`), d, 0, tensor(w, `dec_dict.${key}.bias`)).value, fr.yhat[key][pos]);
      }
    }
    const t = ctx.tokens[ctx.targetPos];
    const head = t.sem === SemType.Boolean ? 'boolean' : 'number';
    expect(fr.prediction).toBe(fr.yhat[head][ctx.targetPos]);
    if (task.targetType === 'binary') {
      close(1 / (1 + Math.exp(-fr.prediction)), sigmoid(fr.prediction));
    } else {
      const st = targetStats(pdb, task);
      close(fr.prediction * st.std + st.mean, denormalize(fr.prediction, st));
    }
  });

  it('mask reasons agree with the four masks for every pair', async () => {
    const { fr, ctx } = await setup(taskId);
    const S = fr.S;
    for (const l of ATTN_ORDER) {
      for (let q = 0; q < S; q++) for (let k = 0; k < S; k++) expect(maskReasons(ctx.tokens, l, q, k).length > 0).toBe(fr.masks[l][q * S + k] === 1);
    }
  });

  it('dotTerms sums its terms', () => {
    const { sum, terms } = dotTerms([1, -2, 3], [4, 5, -6]);
    expect(sum).toBe(1 * 4 - 2 * 5 - 3 * 6);
    expect(terms.map((t) => t.p)).toEqual([4, -10, -18]);
  });
});
