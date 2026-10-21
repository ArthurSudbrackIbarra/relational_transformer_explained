/**
 * Section 5 live checks: properties of masked attention that hold exactly in the paper's FlexAttention
 * kernel (spec §6.1) and that our dense loop must reproduce, verified on the live forward pass:
 *  1. a query row with no allowed key outputs exactly 0 (so the residual update is exactly 0);
 *  2. blocked pairs get exactly 0 weight;
 *  3. every other row of weights sums to 1 (up to float32 rounding).
 */
import { useMemo } from 'react';
import { Cite, StatusBadge } from '@/components';
import { ATTN_ORDER } from '@/engine/types';
import { int } from './utils';
import type { AttnLive } from './useAttnLive';

export function ImplementationChecks({ live }: { live: AttnLive }) {
  const { fr, masks, S, H, d, nBlocks, ctx } = live;
  const res = useMemo(() => {
    if (!fr) return null;
    let nbrEmpty = 0; // non-padding queries whose neighbor row is empty
    let otherEmpty = 0; // non-padding queries with an empty row in col / feat / full (never, spec §6.2)
    let emptyChecked = 0;
    let maxEmptyOut = 0;
    let rowsChecked = 0;
    let maxSumErr = 0;
    let blockedPairs = 0;
    let maxBlockedP = 0;
    for (const l of ATTN_ORDER) {
      const m = masks[l];
      const empty: number[] = [];
      for (let q = 0; q < S; q++) {
        let n = 0;
        for (let k = 0; k < S; k++) n += m[q * S + k];
        if (n === 0) empty.push(q);
      }
      const real = empty.filter((q) => !ctx.tokens[q].isPadding).length;
      if (l === 'nbr') nbrEmpty += real;
      else otherEmpty += real;
      for (let b = 0; b < fr.blocks.length; b++) {
        const cap = fr.blocks[b].attn[l];
        if (cap.skipped || cap.probs.length < H * S * S) continue;
        for (const q of empty) {
          emptyChecked++;
          for (let j = 0; j < d; j++) maxEmptyOut = Math.max(maxEmptyOut, Math.abs(cap.out[q * d + j]));
        }
        for (let h = 0; h < H; h++) {
          for (let q = 0; q < S; q++) {
            const o = (h * S + q) * S;
            let sum = 0;
            let any = false;
            for (let k = 0; k < S; k++) {
              const p = cap.probs[o + k];
              if (m[q * S + k]) {
                sum += p;
                any = true;
              } else {
                blockedPairs++;
                if (p > maxBlockedP) maxBlockedP = p;
              }
            }
            if (any) {
              rowsChecked++;
              maxSumErr = Math.max(maxSumErr, Math.abs(sum - 1));
            }
          }
        }
      }
    }
    const padding = ctx.tokens.filter((t) => t.isPadding).length;
    return { nbrEmpty, otherEmpty, padding, emptyChecked, maxEmptyOut, rowsChecked, maxSumErr, blockedPairs, maxBlockedP };
  }, [fr, masks, S, H, d, ctx]);

  if (!res) return <p className="text-[16.5px] text-ink-2">Running the forward pass…</p>;
  return (
    <div className="my-6 grid gap-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge status={res.maxEmptyOut === 0 ? 'good' : 'critical'}>
          {int(res.emptyChecked)} query rows with no allowed key (all layers × {nBlocks} blocks): max |Δx| = {res.maxEmptyOut === 0 ? '0 exactly' : res.maxEmptyOut.toExponential(2)}
        </StatusBadge>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge status={res.maxBlockedP === 0 ? 'good' : 'critical'}>
          {int(res.blockedPairs)} blocked (query, key) pairs × heads: largest weight = {res.maxBlockedP === 0 ? '0 exactly' : res.maxBlockedP.toExponential(2)}
        </StatusBadge>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge status={res.maxSumErr < 1e-5 ? 'good' : 'warning'}>
          {int(res.rowsChecked)} rows of weights with at least one allowed key: max |Σp − 1| = {res.maxSumErr === 0 ? '0' : res.maxSumErr.toExponential(1)}
        </StatusBadge>
      </div>
      <p className="!mb-0 text-[15px] leading-snug text-ink-2">
        Checked on the live forward pass of this context. {int(res.nbrEmpty)} of the {int(ctx.numReal)} real tokens have an empty neighbor row
        {res.padding > 0 ? `, and the ${int(res.padding)} padding tokens have empty rows in every mask` : ''}; column, feature and full rows of real tokens
        are {res.otherEmpty === 0 ? 'never' : 'sometimes'} empty, since they always contain the query itself <Cite kind="spec" ref="§6.2" />. Each empty row
        leaves its token unchanged, in every block <Cite kind="spec" ref="§6.1" />. The sums miss 1 only by float32 rounding.
      </p>
    </div>
  );
}
