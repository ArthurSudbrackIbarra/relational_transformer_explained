/**
 * Step 3 read-out: how short the column term comes out in the checkpoint this page runs, and why the
 * column projection starts below ε at initialization in both our model and the paper's. Every number
 * is computed from the live forward pass or the live weights.
 */
import { memo } from 'react';
import { Cite, TeX } from '@/components';
import { SemType } from '@/engine/types';
import { fmt, int, type EmbeddingStable } from './live';
import { meanSquare, norm, slice } from './math';

const median = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  if (!s.length) return NaN;
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

export const EpsFinding = memo(function EpsFinding({ live }: { live: EmbeddingStable }) {
  const { fr, weights, ctx } = live;
  if (!fr || !weights) return null;
  const d = weights.config.dModel;
  const eps = weights.config.eps;
  const colMs: number[] = [];
  const colLen: number[] = [];
  const numLen: number[] = [];
  const seen = new Set<number>();
  for (const t of ctx.tokens) {
    if (t.isPadding) continue;
    if (!seen.has(t.colNameIdx)) {
      seen.add(t.colNameIdx);
      colMs.push(meanSquare(slice(fr.embedding.colPreNorm, t.pos, d)));
      colLen.push(norm(slice(fr.embedding.colTerm, t.pos, d)));
    }
    if (!t.mask && t.sem === SemType.Number) numLen.push(norm(slice(fr.embedding.valueTerm, t.pos, d)));
  }
  const ms = median(colMs);
  if (!Number.isFinite(ms)) return null;
  const below = colMs.filter((m) => m < eps).length;
  const share = eps / (ms + eps);
  const dText = weights.config.dText;

  const lenCol = median(colLen);
  const lenNum = median(numLen);
  const ratio = lenNum / lenCol;

  return (
    <>
      <p>
        Now the real inputs. In the checkpoint this page runs, the column projection <TeX tex="W_{\mathrm{col}}c + b_{\mathrm{col}}" /> has a
        median mean square of {ms.toFixed(4)} over the {int(colMs.length)} column phrases of this context, and{' '}
        {below === colMs.length ? `all ${int(below)}` : below === 0 ? 'none' : `${int(below)}`} of them {below === 1 ? 'is' : 'are'} below ε ={' '}
        {fmt(eps, 4)}. For a typical phrase, ε makes up {Math.round(share * 100)}% of the sum under the square root, so the column terms come out
        about {fmt(lenCol, 1)} long{Number.isFinite(lenNum) ? `, while number value terms reach about ${fmt(lenNum, 1)}` : ''}.
      </p>
      {Number.isFinite(ratio) && ratio > 1.2 && (
        <p>
          In this checkpoint a number value term is about {fmt(ratio, 1)} times as long as a column term, so the value dominates x₀. Two cells
          with the same z in different columns start out with similar x₀ (Step 4 measures how similar), and the attention layers have to tell
          them apart by the shorter column component.
        </p>
      )}
      <p>
        The column projection’s mean square is already below ε at initialization, in our model and in the paper’s alike. PyTorch draws every weight and bias
        of <code>nn.Linear({dText}, d)</code> uniformly from <TeX tex={String.raw`\pm 1/\sqrt{${dText}}`} /> <Cite kind="spec" ref="§7.4" />, and
        c has length 1 <Cite kind="spec" ref="§3.6" />. Each output dimension then has{' '}
        <TeX tex={String.raw`\mathbb{E}[x_j^2] = (\|c\|^2 + 1)/(3\cdot ${dText}) = 1/${(3 * dText) / 2} \approx ${(2 / (3 * dText)).toFixed(4)}`} />
        , about {fmt(2 / (3 * dText) / eps, 2)} ε, whatever d is.
      </p>
    </>
  );
});
