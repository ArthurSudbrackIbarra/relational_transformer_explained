/**
 * Step 3 read-out: how short the column term comes out in the checkpoint this page runs, and why the
 * column projection starts below ε at initialization in both our model and the paper's. Every number
 * is computed from the live forward pass, the live weights or the exported init weights.
 */
import { memo } from 'react';
import { Cite, TeX } from '@/components';
import { tensor } from '@/engine/model';
import { SemType } from '@/engine/types';
import { fmt, int, useInitWeights, type EmbeddingStable } from './live';
import { columnProjection, meanSquare, norm, slice } from './math';

const median = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  if (!s.length) return NaN;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

export const EpsFinding = memo(function EpsFinding({ live }: { live: EmbeddingStable }) {
  const { fr, weights, ctx, store, pdb } = live;
  const init = useInitWeights();
  if (!fr || !weights) return null;
  const d = weights.config.dModel;
  const eps = weights.config.eps;
  const colMs: number[] = [];
  const colLen: number[] = [];
  const numLen: number[] = [];
  const phrases: string[] = [];
  const seen = new Set<number>();
  for (const t of ctx.tokens) {
    if (t.isPadding) continue;
    if (!seen.has(t.colNameIdx)) {
      seen.add(t.colNameIdx);
      phrases.push(pdb.text[t.colNameIdx]);
      colMs.push(meanSquare(slice(fr.embedding.colPreNorm, t.pos, d)));
      colLen.push(norm(slice(fr.embedding.colTerm, t.pos, d)));
    }
    if (!t.mask && t.sem === SemType.Number) numLen.push(norm(slice(fr.embedding.valueTerm, t.pos, d)));
  }
  const ms = median(colMs);
  if (!Number.isFinite(ms)) return null;
  const below = colMs.filter((m) => m < eps).length;
  const share = eps / (ms + eps);

  // the five gains: is γ_col the smallest? (a second, independent reason for a short column term)
  const gLen = (k: string) => norm(tensor(weights, `norm_dict.${k}.weight`));
  const gCol = gLen('col_name');
  const gVals = (['number', 'text', 'datetime', 'boolean'] as const).map((k) => ({ k, len: gLen(k) }));
  const gNum = gVals[0].len;
  const colLowest = gVals.every((v) => gCol < v.len);

  // initialization: the exported init weights on the same phrases
  const initMs =
    init && store && init.config.dModel === d ? median(phrases.map((p) => meanSquare(columnProjection(init, store, p).pre))) : NaN;
  const dText = weights.config.dText;

  const lenCol = median(colLen);
  const lenNum = median(numLen);
  const ratio = lenNum / lenCol;

  return (
    <>
      <p>
        Now the real inputs. In the checkpoint this page runs, <TeX tex="W_{\mathrm{col}}c + b_{\mathrm{col}}" /> has a median mean square of{' '}
        {ms.toFixed(4)} over the {int(colMs.length)} column phrases of this context, and{' '}
        {below === colMs.length ? `all ${int(below)}` : below === 0 ? 'none' : `${int(below)}`} of them {below === 1 ? 'is' : 'are'} below ε ={' '}
        {fmt(eps, 4)}. For a typical phrase, ε makes up {Math.round(share * 100)}% of mean(x²) + ε, the sum under the square root. So the column
        terms come out about {fmt(lenCol, 1)} long{Number.isFinite(lenNum) ? `, while number value terms reach about ${fmt(lenNum, 1)}` : ''}.
        {colLowest ? (
          <>
            {' '}
            ε is not the only reason: training also left <TeX tex="\gamma_{\mathrm{col}}" /> the smallest of the five gains (
            <TeX tex="\|\gamma_{\mathrm{col}}\|" /> = {fmt(gCol, 2)}, against {fmt(gNum, 2)} for <TeX tex="\gamma_{\mathrm{number}}" />; all
            ones would give {fmt(Math.sqrt(d), 0)}).
          </>
        ) : null}
      </p>
      {Number.isFinite(ratio) && ratio > 1.2 && (
        <p>
          This matters for what the model sees. In this checkpoint a number value term is about {fmt(ratio, 1)} times as long as a column term,
          so x₀ is dominated by the value. Two cells with the same z in different columns therefore start out with similar x₀; Step 4 measures
          how similar. What tells them apart is the shorter column component, and the attention layers have to work with that.
        </p>
      )}
      <p>
        Why the column projection starts small follows from the default initialization, in our model and in the paper’s alike. PyTorch draws
        every weight and bias of <code>nn.Linear({dText}, d)</code> uniformly from <TeX tex={String.raw`\pm 1/\sqrt{${dText}}`} />{' '}
        <Cite kind="spec" ref="§7.4" />, and c has length 1 <Cite kind="spec" ref="§3.6" />. Each output dimension then has{' '}
        <TeX tex={String.raw`\mathbb{E}[x_j^2] = (\|c\|^2 + 1)/(3\cdot ${dText}) = 1/${(3 * dText) / 2} \approx ${(2 / (3 * dText)).toFixed(4)}`} />
        , whatever d is: about {fmt(2 / (3 * dText) / eps, 2)} ε. Our exported initial weights give a median of{' '}
        {Number.isFinite(initMs) ? initMs.toFixed(4) : '…'} for this context’s phrases. The spec notes the same order of magnitude, mean(x²) ≈ 2·10⁻³{' '}
        <Cite kind="spec" ref="§7.2" />. How far training moves it in the paper’s 256-d model is not reported.
      </p>
    </>
  );
});
