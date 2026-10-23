/**
 * "Paper vs our miniature": what is identical (every operation) and what is not (sizes, data,
 * precision, random numbers, training length, attention kernel). Sizes come from the engine's
 * configs, training facts from results.json.
 */
import { Check, Minus } from 'lucide-react';
import type { ReactNode } from 'react';
import { Cite, formatNumber } from '@/components';
import { DEFAULT_L, MINI_CONFIG, PAPER_CONFIG, PAPER_L, paramCount } from '@/engine/model';
import { int } from './live';
import { maxCheckpointDiff, useVerification } from '@/chapters/about/data';
import { useResultsJson } from './Results';

function Item({ icon, children }: { icon: 'same' | 'diff'; children: ReactNode }) {
  return (
    <li className="!m-0 flex gap-2.5 !pl-0 text-[16.5px] leading-[1.55]">
      <span
        aria-hidden="true"
        className="mt-[3px] inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full"
        style={{
          background: icon === 'same' ? 'var(--accent-soft)' : 'var(--surface-3)',
        }}
      >
        {icon === 'same' ? (
          <Check size={13} strokeWidth={2.6} className="text-accent-text" />
        ) : (
          <Minus size={13} strokeWidth={2.6} className="text-ink-2" />
        )}
      </span>
      <span className="min-w-0 text-ink">{children}</span>
    </li>
  );
}

export function SameAndDifferent() {
  const { data } = useResultsJson();
  const steps = data?.pretrain?.steps;
  const batch = data?.config?.batch;
  const { data: ver } = useVerification();
  // the official FlexAttention class vs this site's dense port, same trained weights, real contexts (chapter 16)
  const diff = ver ? maxCheckpointDiff(ver) : undefined;
  return (
    <div className="wide my-6 grid gap-4 md:grid-cols-2">
      <div className="card px-5 py-4">
        <div className="mb-2.5 text-[17px] font-semibold text-ink">Identical to the paper-era code</div>
        <ul className="grid gap-2 !p-0" style={{ listStyle: 'none' }}>
          <Item icon="same">
            Preprocessing: node indexing, the four semantic types, z-scoring <Cite kind="spec" ref="§3" />
          </Item>
          <Item icon="same">
            The context sampler, rule for rule: parents first, width bound, temporal filter, stop at L <Cite kind="spec" ref="§4" />
          </Item>
          <Item icon="same">
            The token embedding, the four masks, pre-norm blocks in the order column → feature → neighbor → full, SwiGLU, the final norm, the per-type
            decoders and losses <Cite kind="code" ref="rt/model.py:104-267" />
          </Item>
          <Item icon="same">
            The frozen text encoder, all-MiniLM-L12-v2 with 384-d vectors <Cite kind="spec" ref="§3.6" />
          </Item>
          <Item icon="same">
            The protocol: leave-one-database-out pretraining, the OneCycle schedule, the best validation checkpoint per target task{' '}
            <Cite kind="spec" ref="§9" />
          </Item>
        </ul>
      </div>
      <div className="card px-5 py-4">
        <div className="mb-2.5 text-[17px] font-semibold text-ink">Different, without changing any operation</div>
        <ul className="grid gap-2 !p-0" style={{ listStyle: 'none' }}>
          <Item icon="diff">
            Sizes: {int(paramCount(MINI_CONFIG))} parameters instead of {int(paramCount(PAPER_CONFIG))}, d = {MINI_CONFIG.dModel} instead of{' '}
            {PAPER_CONFIG.dModel}, {MINI_CONFIG.nBlocks} blocks instead of {PAPER_CONFIG.nBlocks}, L = {DEFAULT_L} instead of {PAPER_L}
          </Item>
          <Item icon="diff">Data: four small synthetic databases instead of RelBench, with ShopDB held out</Item>
          <Item icon="diff">
            Training: {steps ? int(steps) : '…'} steps with batch {batch ?? '…'} on a CPU, instead of 50,001 steps with batch 256 on 8 A100s{' '}
            <Cite kind="spec" ref="§9.2" />
          </Item>
          <Item icon="diff">
            Precision: float32 instead of bfloat16, keeping the paper environment’s RMSNorm ε = 2⁻⁷ <Cite kind="spec" ref="§7.2" />
          </Item>
          <Item icon="diff">
            Random numbers: a seeded xoshiro128** generator instead of Rust’s ChaCha12, so contexts follow the same distribution but are not
            bit-identical <Cite kind="spec" ref="§4.2" />
          </Item>
          <Item icon="diff">
            Attention kernel: plain masked attention instead of FlexAttention, the same function
            {diff !== undefined ? ` (the official class and this site’s port differ by at most ${formatNumber(diff, 2)} on a forward pass)` : ''};
            chapter 16 shows the parity checks
          </Item>
        </ul>
      </div>
    </div>
  );
}
