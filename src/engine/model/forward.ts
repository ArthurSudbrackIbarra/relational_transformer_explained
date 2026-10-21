/**
 * The Relational Transformer forward pass with full capture.
 *
 * An exact, operation-by-operation mirror of `RelationalTransformer.forward`
 * (PE/rt/model.py:153-267), for one sequence (B = 1). Spec: §5.2 (input embedding),
 * §6.1-6.2 (masked attention, the four masks), §7 (block, RMSNorm), §8.1-8.3 (decoders,
 * loss, prediction), Appendix A (reference pseudo-code).
 *
 * The only departures from the paper environment are numeric, not operational:
 *  - float32 storage with float64 accumulation instead of bf16 (BUILD_PLAN §2);
 *  - dense masked softmax instead of FlexAttention's block-sparse kernel (identical maths,
 *    including "fully masked query row ⇒ output 0", spec §6.1).
 */
import type {
  AttnCapture,
  AttnLayer,
  BlockCapture,
  EmbeddingParts,
  FFNCapture,
  ForwardOptions,
  ForwardResult,
  MaskSet,
  PreprocessedDB,
  RTWeights,
  Token,
} from '../types';
import { ATTN_ORDER, SemType } from '../types';
import type { EmbeddingStore } from '../embeddings';
import { buildMasks } from '../masks';
import { SEM_KEYS, dHead, type SemKey } from './config';
import { lossTerms, meanLoss, type LossTerm } from './loss';
import { linear, maskRows, maskedAttentionCore, rmsnorm, rmsnormRow, silu, type MaskRows } from './ops';
import { tensor } from './weights';

/** ForwardResult plus the per-masked-cell loss decomposition (§8.2). Assignable to ForwardResult. */
export interface RTForwardResult extends ForwardResult {
  lossTerms: LossTerm[];
}

/** Maps a text index of the DB's string table (PreprocessedDB.text) to its frozen 384-d embedding. */
export type TextEmbedder = (textIdx: number) => Float32Array;

/**
 * Text lookup exactly as the sampler does it (fly.rs:476-481 `get_text_emb(dataset, idx)`):
 * index → string of the DB's text table → precomputed MiniLM vector. Throws on unknown index/string.
 */
export function textEmbedder(pdb: Pick<PreprocessedDB, 'text'>, store: EmbeddingStore): TextEmbedder {
  return (idx: number) => {
    const s = pdb.text[idx];
    if (s === undefined) throw new Error(`forward: text index ${idx} out of range (text table has ${pdb.text.length})`);
    return store.get(s);
  };
}

const EMPTY = new Float32Array(0);

/**
 * Value projection + its RMSNorm for one scalar value of a non-text type
 * (model.py:218 `norm_dict[t](enc_dict[t](v))`). Useful for the embedding chapter's slider.
 */
export function scalarValueTerm(
  w: RTWeights,
  sem: Exclude<SemKey, 'text'>,
  value: number,
): { preNorm: Float32Array; term: Float32Array } {
  const d = w.config.dModel;
  const pre = linear(Float32Array.of(value), 1, 1, tensor(w, `enc_dict.${sem}.weight`), d, tensor(w, `enc_dict.${sem}.bias`));
  const term = rmsnorm(pre, 1, d, tensor(w, `norm_dict.${sem}.weight`), w.config.eps);
  return { preNorm: pre, term };
}

/**
 * Input embedding of every token (model.py:208-224, spec §5.2):
 *
 *   x_i = [¬pad] · RMSNorm_col(W_col c_i + b_col)
 *       + Σ_t [sem=t ∧ ¬mask ∧ ¬pad] · RMSNorm_t(W_t v_i + b_t)  +  [sem=t ∧ mask ∧ ¬pad] · m_t
 *
 * Exactly one value term per non-padding token; m_t is added raw (no norm); padding → 0⃗.
 * Values of the other types are multiplied by a 0 gate in the official code, so they are not
 * computed here: for FINITE values 0·v = 0 and the sum is unchanged.
 *
 * Non-finite values: the official code evaluates every type's term for every position and only
 * then multiplies by the gate, so a NaN/±inf in ANY scalar value slot — another type's slot, a
 * padding token's slot, or the hidden label of a masked target — yields NaN·0 = NaN in x and, via
 * attention, in ŷ and the loss (spec §5.3). The sampler never produces such values (pre.rs drops
 * NaN cells, §3.5), so we throw instead of returning a finite result the official model would not
 * give (e.g. an "unchanged" prediction for a NaN hidden label).
 */
export function embedTokens(tokens: readonly Token[], textEmb: TextEmbedder, w: RTWeights): EmbeddingParts {
  const { dModel: d, dText, eps } = w.config;
  const S = tokens.length;
  const colPreNorm = new Float32Array(S * d);
  const colTerm = new Float32Array(S * d);
  const valuePreNorm = new Float32Array(S * d);
  const valueTerm = new Float32Array(S * d);
  const x0 = new Float32Array(S * d);

  const Wc = tensor(w, 'enc_dict.col_name.weight');
  const bc = tensor(w, 'enc_dict.col_name.bias');
  const gc = tensor(w, 'norm_dict.col_name.weight');
  // Column-name / text projections depend only on the text index: compute once per index.
  const colCache = new Map<number, Float32Array>();
  const textCache = new Map<number, Float32Array>();
  const project384 = (cache: Map<number, Float32Array>, idx: number, W: Float32Array, b: Float32Array) => {
    let p = cache.get(idx);
    if (!p) {
      const e = textEmb(idx);
      if (e.length !== dText) throw new Error(`forward: embedding dim ${e.length} != dText ${dText}`);
      p = linear(e, 1, dText, W, d, b);
      cache.set(idx, p);
    }
    return p;
  };
  const scalar = new Float32Array(1);

  for (let i = 0; i < S; i++) {
    const t = tokens[i];
    // Every scalar slot of every position goes through enc_dict/norm_dict before the ×gate
    // (model.py:216-220), so a non-finite one would poison the official output (see above).
    if (!Number.isFinite(t.numberValue) || !Number.isFinite(t.datetimeValue) || !Number.isFinite(t.booleanValue)) {
      throw new Error(
        `forward: token ${i} has a non-finite value slot (number ${t.numberValue}, datetime ${t.datetimeValue}, ` +
          `boolean ${t.booleanValue}); the official model would output NaN (NaN·0 = NaN, model.py:216-224, spec §5.3)`,
      );
    }
    if (t.isPadding) continue; // `* (~is_padding)[..., None]` on every term (model.py:213, 219, 223)
    const o = i * d;

    // column-name term (model.py:209-214)
    const cp = project384(colCache, t.colNameIdx, Wc, bc);
    colPreNorm.set(cp, o);
    rmsnormRow(cp, 0, d, gc, eps, colTerm, o);

    // value term (model.py:216-224)
    const key: SemKey = SEM_KEYS[t.sem];
    if (key === undefined) throw new Error(`forward: token ${i} has invalid sem type ${t.sem}`);
    if (t.mask) {
      valueTerm.set(tensor(w, `mask_embs.${key}`), o); // raw m_t, not normalised (model.py:221-224)
    } else {
      let pre: Float32Array;
      if (t.sem === SemType.Text) {
        pre = project384(textCache, t.textIdx, tensor(w, 'enc_dict.text.weight'), tensor(w, 'enc_dict.text.bias'));
      } else {
        scalar[0] = t.sem === SemType.Number ? t.numberValue : t.sem === SemType.DateTime ? t.datetimeValue : t.booleanValue;
        pre = linear(scalar, 1, 1, tensor(w, `enc_dict.${key}.weight`), d, tensor(w, `enc_dict.${key}.bias`));
      }
      valuePreNorm.set(pre, o);
      rmsnormRow(pre, 0, d, tensor(w, `norm_dict.${key}.weight`), eps, valueTerm, o);
    }
    for (let j = 0; j < d; j++) x0[o + j] = colTerm[o + j] + valueTerm[o + j];
  }
  return { colTerm, valuePreNorm, valueTerm, colPreNorm, x0 };
}

/**
 * One pre-norm attention sublayer: x + W_o · MHA(RMSNorm_l(x); M_l)  (model.py:20-47, 84).
 */
function attentionSublayer(
  layer: AttnLayer,
  xIn: Float32Array,
  S: number,
  w: RTWeights,
  block: number,
  rows: MaskRows,
  capture: boolean,
  active: Int32Array | null,
): AttnCapture {
  const { dModel: d, nHeads, eps } = w.config;
  const p = `blocks.${block}.attns.${layer}`;
  const h = rmsnorm(xIn, S, d, tensor(w, `blocks.${block}.norms.${layer}.weight`), eps, undefined, active);
  const q = linear(h, S, d, tensor(w, `${p}.wq.weight`), d, null, undefined, active); // bias=False (model.py:25-28)
  const k = linear(h, S, d, tensor(w, `${p}.wk.weight`), d, null, undefined, active);
  const v = linear(h, S, d, tensor(w, `${p}.wv.weight`), d, null, undefined, active);
  const bufs = capture
    ? { scores: new Float32Array(nHeads * S * S), probs: new Float32Array(nHeads * S * S) }
    : null;
  const headOut = maskedAttentionCore(q, k, v, S, d, nHeads, rows, bufs, undefined, active);
  const out = linear(headOut, S, d, tensor(w, `${p}.wo.weight`), d, null, undefined, active);
  const xOut = new Float32Array(S * d);
  for (let i = 0; i < S * d; i++) xOut[i] = xIn[i] + out[i];
  if (!capture) {
    return { layer, xIn, h: EMPTY, q: EMPTY, k: EMPTY, v: EMPTY, scores: EMPTY, probs: EMPTY, headOut: EMPTY, out, xOut, skipped: false };
  }
  return { layer, xIn, h, q, k, v, scores: bufs!.scores, probs: bufs!.probs, headOut, out, xOut, skipped: false };
}

/** A disabled sublayer (inference-time ablation): the residual stream passes through unchanged. */
function skippedSublayer(layer: AttnLayer, xIn: Float32Array, S: number, d: number): AttnCapture {
  return {
    layer,
    xIn,
    h: EMPTY,
    q: EMPTY,
    k: EMPTY,
    v: EMPTY,
    scores: EMPTY,
    probs: EMPTY,
    headOut: EMPTY,
    out: new Float32Array(S * d),
    xOut: xIn,
    skipped: true,
  };
}

/** Pre-norm SwiGLU: x + W2 · (SiLU(W1 h) ⊙ W3 h), h = RMSNorm_ffn(x)  (model.py:50-59, 85). */
function ffnSublayer(
  xIn: Float32Array,
  S: number,
  w: RTWeights,
  block: number,
  capture: boolean,
  active: Int32Array | null,
): FFNCapture {
  const { dModel: d, dFF, eps } = w.config;
  const p = `blocks.${block}.ffn`;
  const h = rmsnorm(xIn, S, d, tensor(w, `blocks.${block}.norms.ffn.weight`), eps, undefined, active);
  const a = linear(h, S, d, tensor(w, `${p}.w1.weight`), dFF, null, undefined, active);
  const b = linear(h, S, d, tensor(w, `${p}.w3.weight`), dFF, null, undefined, active);
  const gated = new Float32Array(S * dFF);
  const gate = (s: number) => {
    for (let i = s * dFF, e = i + dFF; i < e; i++) gated[i] = silu(a[i]) * b[i];
  };
  if (active) for (const s of active) gate(s);
  else for (let s = 0; s < S; s++) gate(s);
  const out = linear(gated, S, dFF, tensor(w, `${p}.w2.weight`), d, null, undefined, active);
  const xOut = new Float32Array(S * d);
  for (let i = 0; i < S * d; i++) xOut[i] = xIn[i] + out[i];
  if (!capture) return { xIn, h: EMPTY, a: EMPTY, b: EMPTY, gated: EMPTY, out, xOut };
  return { xIn, h, a, b, gated, out, xOut };
}

/**
 * Forward pass given precomputed masks (see `forward`). Exposed so tests and UI code can
 * supply masks from elsewhere (e.g. a PyTorch fixture) and compare.
 */
export function forwardWithMasks(
  tokens: readonly Token[],
  masks: MaskSet,
  textEmb: TextEmbedder,
  w: RTWeights,
  opts: ForwardOptions = {},
): RTForwardResult {
  const cfg = w.config;
  const S = tokens.length;
  const d = cfg.dModel;
  dHead(cfg); // validates d % H == 0
  if (S === 0) throw new Error('forward: empty token list');
  const capture = opts.capture !== false;
  const disabled = new Set<AttnLayer>(opts.disabledLayers ?? []);
  for (const l of ATTN_ORDER) {
    if (!masks[l] || masks[l].length !== S * S) throw new Error(`forward: mask ${l} must be S×S = ${S * S}`);
  }

  // 1. input embedding (model.py:208-224)
  const embedding = embedTokens(tokens, textEmb, w);

  // 2. masks → per-query allowed-key lists, built once and shared by every block (model.py:204-206, 226-227)
  const rows = {} as Record<AttnLayer, MaskRows>;
  for (const l of ATTN_ORDER) rows[l] = maskRows(masks[l], S);

  // Padding rows are exactly 0⃗ in every block (x0 = 0, RMSNorm(0) = 0, bias-free projections,
  // fully masked attention rows → 0, FFN(0) = 0; spec §7.2), so the kernels skip them. This is an
  // exact shortcut: computing them would also give 0. Only the decoders (which have biases) run
  // on every position, as in model.py:239.
  // (Guard: only valid when no padding query may attend to anything, which buildMasks guarantees
  // through `∧ pad`; with hand-made masks that break this, fall back to computing every row.)
  const nonPad: number[] = [];
  let padRowsEmpty = true;
  for (let i = 0; i < S; i++) {
    if (!tokens[i].isPadding) nonPad.push(i);
    else for (const l of ATTN_ORDER) if (rows[l].start[i] !== rows[l].start[i + 1]) padRowsEmpty = false;
  }
  const active = nonPad.length === S || !padRowsEmpty ? null : Int32Array.from(nonPad);

  // 3. blocks (model.py:226-227); inside each: col → feat → nbr → full, then FFN (model.py:82-86)
  let x = embedding.x0;
  const blocks: BlockCapture[] = [];
  for (let bi = 0; bi < cfg.nBlocks; bi++) {
    const attn = {} as Record<AttnLayer, AttnCapture>;
    for (const l of ATTN_ORDER) {
      const c = disabled.has(l) ? skippedSublayer(l, x, S, d) : attentionSublayer(l, x, S, w, bi, rows[l], capture, active);
      attn[l] = c;
      x = c.xOut;
    }
    const ffn = ffnSublayer(x, S, w, bi, capture, active);
    x = ffn.xOut;
    blocks.push({ attn, ffn });
  }

  // 4. final norm (model.py:229)
  const xFinal = rmsnorm(x, S, d, tensor(w, 'norm_out.weight'), cfg.eps);

  // 5. decoders at every position (model.py:238-239). The text decoder (Linear(d, 384)) exists in
  //    the checkpoint but is never trained or read (text is never masked, spec §8.1): skipped here.
  const dec = (key: 'number' | 'datetime' | 'boolean') =>
    linear(xFinal, S, d, tensor(w, `dec_dict.${key}.weight`), 1, tensor(w, `dec_dict.${key}.bias`));
  const yhat = { number: dec('number'), datetime: dec('datetime'), boolean: dec('boolean') };

  // 6. loss over masked cells (model.py:231-265)
  const terms = lossTerms(tokens, yhat);
  const loss = meanLoss(terms);

  // 7. prediction at the target (main.py:283-293 reads `is_targets`): boolean logit or number
  //    z-score (datetime z-score for a datetime target). No target ⇒ targetPos −1, NaN.
  //    main.py:228-241 picks the head from a hard-coded list of regression TASK names rather than
  //    from the cell's sem type; for every task of our four DBs the two agree (binary ↔ Boolean
  //    target, regression ↔ Number target), so this readout is the same. No paper task has a
  //    datetime target.
  const targetPos = tokens.findIndex((t) => t.isTarget && !t.isPadding);
  let prediction = NaN;
  if (targetPos >= 0) {
    const s = tokens[targetPos].sem;
    prediction =
      s === SemType.Boolean
        ? yhat.boolean[targetPos]
        : s === SemType.Number
          ? yhat.number[targetPos]
          : s === SemType.DateTime
            ? yhat.datetime[targetPos]
            : NaN;
  }

  return { S, d, masks, embedding, blocks, xFinal, yhat, loss, lossTerms: terms, targetPos, prediction };
}

/**
 * RT forward for one context window (mirror of model.py:153-267).
 *
 * @param tokens  the L tokens of a sampled context (padding included), §5.1
 * @param pdb     the preprocessed DB (only its `text` table is used: token.colNameIdx /
 *                token.textIdx → string → precomputed embedding)
 * @param store   precomputed MiniLM embeddings (throws on a string that was never embedded)
 * @param w       weights (PyTorch state_dict names)
 * @param opts    disabledLayers (inference-time ablation; residual passes unchanged), capture
 */
export function forward(
  tokens: readonly Token[],
  pdb: Pick<PreprocessedDB, 'text'>,
  store: EmbeddingStore,
  w: RTWeights,
  opts: ForwardOptions = {},
): RTForwardResult {
  if (store.dim !== w.config.dText) throw new Error(`forward: embedding dim ${store.dim} != model dText ${w.config.dText}`);
  return forwardWithMasks(tokens, buildMasks(tokens), textEmbedder(pdb, store), w, opts);
}
