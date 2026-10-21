/**
 * Evaluation metrics and display transforms (spec §8.3-§8.5).
 *
 * The paper-era eval (rt/main.py:328-337) does:
 *   regression:      r2_score(labels, preds)                    labels/preds are z-scores
 *   classification:  roc_auc_score([int(x > 0) for x in labels], preds)   preds are raw LOGITS
 * AUROC on logits equals AUROC on σ(logit) MATHEMATICALLY (σ is monotone); R² in z-space equals
 * raw-unit R² (both sums of squares scale by 1/σ²). Display uses σ(logit) and ẑ·σ_train + μ_train (§8.4).
 *
 * Numerical caveat: in float64, σ(z) rounds to exactly 1 for z ≳ 37 (and underflows towards 0 for
 * very negative z), which creates ties the logits do not have. Pass LOGITS to `auroc`, like
 * main.py:284, 337 does. (The paper itself computes both metrics on bf16 labels and predictions cast
 * to float32, spec §12, so tiny differences from a float64 demo are expected anyway.)
 *
 * Input validation follows sklearn 1.8.0 `check_array` (both arguments): empty input, NaN and ±inf
 * raise, as they would make roc_auc_score / r2_score raise a ValueError in the paper's eval.
 */

/** sklearn `check_array(..., ensure_all_finite=True, ensure_min_samples=1)` on a 1-D array. */
function checkFinite(xs: readonly number[]): void {
  if (xs.length === 0) throw new Error('Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required.');
  for (const x of xs) {
    if (Number.isNaN(x)) throw new Error('Input contains NaN.');
    if (!Number.isFinite(x)) throw new Error("Input contains infinity or a value too large for dtype('float64').");
  }
}

/** Positive-class label from a z-scored boolean, as main.py:336 `int(x > 0)` / model.py BCE target 1{z>0}. */
export function binaryLabel(z: number): 0 | 1 {
  return z > 0 ? 1 : 0;
}

/**
 * Area under the ROC curve, matching `sklearn.metrics.roc_auc_score(y_true, y_score)` for binary
 * labels. Tied scores are handled like sklearn's trapezoidal ROC curve, which is the Mann–Whitney
 * statistic with average ranks (a tied positive/negative pair counts ½).
 * Throws like sklearn when only one class is present, when the input is empty, or when a score is
 * NaN or ±inf. Pass raw logits (see the numerical caveat above).
 */
export function auroc(labels: readonly (number | boolean)[], scores: readonly number[]): number {
  const n = labels.length;
  if (scores.length !== n) throw new Error(`auroc: ${n} labels vs ${scores.length} scores`);
  const y = labels.map((v, i) => {
    if (v === true || v === 1) return 1;
    if (v === false || v === 0) return 0;
    throw new Error(`auroc: label ${i} is ${String(v)}, expected 0/1`);
  });
  checkFinite(scores);
  let nPos = 0;
  for (const v of y) nPos += v;
  const nNeg = n - nPos;
  if (nPos === 0 || nNeg === 0) {
    throw new Error('Only one class present in y_true. ROC AUC score is not defined in that case.');
  }
  // Average ranks (1-based) over ascending scores.
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[a] - scores[b]);
  let rankSumPos = 0;
  for (let i = 0; i < n; ) {
    let j = i;
    while (j + 1 < n && scores[order[j + 1]] === scores[order[i]]) j++;
    const avgRank = (i + j + 2) / 2; // ranks i+1 .. j+1
    for (let t = i; t <= j; t++) if (y[order[t]] === 1) rankSumPos += avgRank;
    i = j + 1;
  }
  return (rankSumPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

/**
 * Coefficient of determination, matching `sklearn.metrics.r2_score(y_true, y_pred)` (single output,
 * default force_finite=True): 1 − SS_res/SS_tot with ȳ the mean of the evaluated labels; if
 * SS_tot = 0 the score is 1 when SS_res = 0, else 0; exactly one sample → NaN (sklearn warns);
 * empty input, NaN or ±inf in either argument → throws (sklearn raises ValueError).
 */
export function r2(y: readonly number[], yhat: readonly number[]): number {
  const n = y.length;
  if (yhat.length !== n) throw new Error(`r2: ${n} labels vs ${yhat.length} predictions`);
  checkFinite(y);
  checkFinite(yhat);
  if (n < 2) return NaN; // sklearn: "R^2 score is not well-defined with less than two samples."
  let mean = 0;
  for (const v of y) mean += v;
  mean /= n;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    ssRes += (y[i] - yhat[i]) ** 2;
    ssTot += (y[i] - mean) ** 2;
  }
  if (ssTot === 0) return ssRes === 0 ? 1 : 0;
  if (ssRes === 0) return 1;
  return 1 - ssRes / ssTot;
}

/** Logistic sigmoid σ(z) = 1/(1+e^{−z}), numerically stable for large |z|. Probability display (§8.4). */
export function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

/** z-score → original units: ẑ·σ + μ with the target column's (Train) stats (§8.4, pre.rs:569 inverted). */
export function denormalize(z: number, stats: { mean: number; std: number }): number {
  return z * stats.std + stats.mean;
}

/** Original units → z-score (v − μ)/σ, the forward transform of pre.rs:569. */
export function normalize(v: number, stats: { mean: number; std: number }): number {
  return (v - stats.mean) / stats.std;
}
