/**
 * "Paper vs our miniature": what is identical (every operation) and what is not (sizes, data,
 * training length, implementation). Sizes come from the engine's configs, training facts from
 * results.json.
 */
import { Check, Minus } from 'lucide-react';
import type { ReactNode } from 'react';
import { Cite } from '@/components';
import { DEFAULT_L, MINI_CONFIG, PAPER_CONFIG, PAPER_L, paramCount } from '@/engine/model';
import { int } from './live';
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
  return (
    <div className="wide my-6 grid gap-4 md:grid-cols-2">
      <div className="card px-5 py-4">
        <div className="mb-2.5 text-[17px] font-semibold text-ink">Identical to the paper-era code</div>
        <ul className="grid gap-2 !p-0" style={{ listStyle: 'none' }}>
          <Item icon="same">
            Preprocessing: the four semantic types and the z-scoring of numbers <Cite kind="spec" ref="§3" />
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
            The protocol: leave-one-database-out pretraining, the same learning-rate schedule, and for each target task the checkpoint with the best
            validation score{' '}
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
            Implementation: the sampler and the model run in your browser, ported to TypeScript from the official Rust and PyTorch code and checked
            against it in chapter 17
          </Item>
        </ul>
      </div>
    </div>
  );
}
