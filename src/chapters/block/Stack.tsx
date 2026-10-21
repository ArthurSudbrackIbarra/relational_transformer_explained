/**
 * Figure 4: the stack (rt/model.py:147-150, 226-229; spec §7.1, §7.3). nBlocks identical-shape blocks
 * (4 here, 12 in the paper), all fed the same four masks, then the final RMSNorm and the decoders.
 * The parameter table is computed: for the paper from parameterShapes(PAPER_CONFIG) (= 22,336,131,
 * spec §7.3), for this site from the tensors of the loaded checkpoint.
 */
import { Activity, ArrowRight } from 'lucide-react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { LAYER_INFO, LayerLabel, Segmented, StatusBadge, cx } from '@/components';
import { MINI_CONFIG, PAPER_CONFIG, paramCount, parameterShapes } from '@/engine/model/config';
import { ATTN_ORDER, type RTConfig, type RTWeights } from '@/engine/types';
import type { BlockLive } from './useBlockLive';
import { fx, int, streamStages } from './utils';

// ─────────────────────────────────────────────────────────────────────────────
// Parameter groups
// ─────────────────────────────────────────────────────────────────────────────

type Group = 'attn' | 'ffn' | 'bnorm' | 'enc' | 'innorm' | 'mask' | 'dec' | 'out';
const GROUPS: { id: Group; name: string; where: string; formula: (c: RTConfig) => string }[] = [
  { id: 'attn', name: 'Attention W_q, W_k, W_v, W_o', where: 'blocks', formula: (c) => `${c.nBlocks} × 4 layers × 4 × ${c.dModel}²` },
  { id: 'ffn', name: 'SwiGLU W₁, W₂, W₃', where: 'blocks', formula: (c) => `${c.nBlocks} × 3 × ${c.dModel} × ${c.dFF}` },
  { id: 'bnorm', name: 'Block RMSNorm γ', where: 'blocks', formula: (c) => `${c.nBlocks} × 5 × ${c.dModel}` },
  { id: 'enc', name: 'Input encoders (with bias)', where: 'embedding', formula: (c) => `3 × 2·${c.dModel} + 2 × (${c.dText}+1)·${c.dModel}` },
  { id: 'innorm', name: 'Input RMSNorm γ', where: 'embedding', formula: (c) => `5 × ${c.dModel}` },
  { id: 'mask', name: 'Mask vectors m_t', where: 'embedding', formula: (c) => `4 × ${c.dModel}` },
  { id: 'dec', name: 'Decoders (with bias)', where: 'output', formula: (c) => `3 × (${c.dModel}+1) + (${c.dModel}+1)·${c.dText}` },
  { id: 'out', name: 'norm_out γ', where: 'output', formula: (c) => `${c.dModel}` },
];

function groupOf(name: string): Group {
  if (/^blocks\.\d+\.attns\./.test(name)) return 'attn';
  if (/^blocks\.\d+\.ffn\./.test(name)) return 'ffn';
  if (/^blocks\.\d+\.norms\./.test(name)) return 'bnorm';
  if (name.startsWith('enc_dict.')) return 'enc';
  if (name.startsWith('norm_dict.')) return 'innorm';
  if (name.startsWith('mask_embs.')) return 'mask';
  if (name.startsWith('dec_dict.')) return 'dec';
  return 'out';
}

function countByGroup(entries: [string, number[]][]): Record<Group, number> {
  const out = { attn: 0, ffn: 0, bnorm: 0, enc: 0, innorm: 0, mask: 0, dec: 0, out: 0 } as Record<Group, number>;
  for (const [n, shape] of entries) out[groupOf(n)] += shape.reduce((a, b) => a * b, 1);
  return out;
}

function fromWeights(w: RTWeights): [string, number[]][] {
  return Object.entries(w.tensors).map(([n, t]) => [n, t.shape]);
}

export function ParamTable({ weights }: { weights: RTWeights | null }) {
  const paper = useMemo(() => countByGroup(parameterShapes(PAPER_CONFIG)), []);
  const oursShapes = useMemo(() => countByGroup(parameterShapes(MINI_CONFIG)), []);
  const oursLive = useMemo(() => (weights ? countByGroup(fromWeights(weights)) : null), [weights]);
  const ours = oursLive ?? oursShapes;
  const total = (g: Record<Group, number>) => Object.values(g).reduce((a, b) => a + b, 0);
  const tPaper = total(paper);
  const tOurs = total(ours);
  const match = oursLive ? tOurs === paramCount(MINI_CONFIG) : null;
  const blocksPaper = paper.attn + paper.ffn + paper.bnorm;
  const blocksOurs = ours.attn + ours.ffn + ours.bnorm;

  return (
    <div className="wide my-7">
      <div className="card scroll-quiet overflow-x-auto">
        <table className="w-full min-w-[640px] border-collapse text-[15px]">
          <caption className="sr-only">Parameter count by component, paper model vs this site’s model</caption>
          <thead>
            <tr className="border-b border-grid bg-surface-2 text-left text-ink-2">
              <th scope="col" className="px-3 py-2 font-semibold">
                Component
              </th>
              <th scope="col" className="px-3 py-2 font-semibold">
                Shape (paper)
              </th>
              <th scope="col" className="px-3 py-2 text-right font-semibold">
                Paper
              </th>
              <th scope="col" className="px-3 py-2 text-right font-semibold">
                This site
              </th>
              <th scope="col" className="w-[150px] px-3 py-2 font-semibold">
                Share here
              </th>
            </tr>
          </thead>
          <tbody>
            {GROUPS.map((g, i) => (
              <tr key={g.id} className={cx('border-b border-grid/70', i === 3 && 'border-t-2 border-t-grid')}>
                <th scope="row" className="px-3 py-1.5 text-left font-normal text-ink">
                  {g.name}
                  <span className="ml-1.5 text-[14px] text-muted">{g.where}</span>
                </th>
                <td className="num px-3 py-1.5 text-[14.5px] whitespace-nowrap text-ink-2">{g.formula(PAPER_CONFIG)}</td>
                <td className="num px-3 py-1.5 text-right text-ink">{int(paper[g.id])}</td>
                <td className="num px-3 py-1.5 text-right text-ink">{int(ours[g.id])}</td>
                <td className="px-3 py-1.5">
                  <ShareBar v={ours[g.id] / tOurs} />
                </td>
              </tr>
            ))}
            <tr className="border-b border-grid/70 bg-surface-2/60">
              <th scope="row" className="px-3 py-1.5 text-left font-medium text-ink">
                all blocks
              </th>
              <td className="num px-3 py-1.5 text-[14.5px] text-ink-2">
                {PAPER_CONFIG.nBlocks} × {int(blocksPaper / PAPER_CONFIG.nBlocks)}
              </td>
              <td className="num px-3 py-1.5 text-right text-ink">{int(blocksPaper)}</td>
              <td className="num px-3 py-1.5 text-right text-ink">{int(blocksOurs)}</td>
              <td className="px-3 py-1.5">
                <ShareBar v={blocksOurs / tOurs} />
              </td>
            </tr>
            <tr className="bg-surface-2">
              <th scope="row" className="px-3 py-2 text-left font-semibold text-ink">
                Total
              </th>
              <td className="px-3 py-2" />
              <td className="num px-3 py-2 text-right font-semibold text-ink">{int(tPaper)}</td>
              <td className="num px-3 py-2 text-right font-semibold text-ink">{int(tOurs)}</td>
              <td className="px-3 py-2" />
            </tr>
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[15px] text-ink-2">
        {match === null ? (
          <span>This site’s column is computed from the shapes of model.py at the mini configuration (weights still loading).</span>
        ) : (
          <>
            <StatusBadge status={match ? 'good' : 'critical'}>
              {match ? 'Loaded checkpoint = model.py shapes' : 'Checkpoint shapes differ from model.py'}
            </StatusBadge>
            <span>This site’s column is counted from the tensors of the loaded checkpoint; the paper column from model.py’s shapes at d = 256.</span>
          </>
        )}
      </div>
    </div>
  );
}

function ShareBar({ v }: { v: number }) {
  return (
    <span className="flex items-center gap-2">
      <span className="block h-2 w-[80px] shrink-0 overflow-hidden rounded-full bg-surface-3">
        <span className="block h-full rounded-full" style={{ width: `${Math.max(v > 0 ? 1.5 : 0, v * 100)}%`, background: 'var(--seq-500)' }} />
      </span>
      <span className="num text-[14px] text-ink-2">{v >= 0.001 ? `${(v * 100).toFixed(1)}%` : '<0.1%'}</span>
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// The stack diagram
// ─────────────────────────────────────────────────────────────────────────────

export function StackFigure({ live, follow }: { live: BlockLive; follow: number }) {
  const [mode, setMode] = useState<'ours' | 'paper'>('ours');
  const reduce = useReducedMotion();
  const cfg = mode === 'ours' ? live.cfg : PAPER_CONFIG;
  const perBlock = 4 * 4 * cfg.dModel * cfg.dModel + 3 * cfg.dModel * cfg.dFF + 5 * cfg.dModel;
  const stages = useMemo(() => (live.fr ? streamStages(live.fr, follow) : null), [live.fr, follow]);
  const blocks = Array.from({ length: cfg.nBlocks }, (_, b) => b);

  return (
    <div className="grid grid-cols-1 gap-4">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <Segmented
          label="Model size"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'ours', label: `This site: ${live.cfg.nBlocks} blocks` },
            { value: 'paper', label: `Paper: ${PAPER_CONFIG.nBlocks} blocks` },
          ]}
        />
        <span className="num text-[15px] text-ink-2">
          d = {cfg.dModel}, {cfg.nHeads} heads, d_ff = {cfg.dFF}
        </span>
      </div>
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <ol className="!m-0 grid gap-2 !p-0" style={{ listStyle: 'none' }} aria-label="The forward pass, top to bottom">
          <StackRow kind="io" title="Input embedding x⁽⁰⁾" detail={`one ${cfg.dModel}-d vector per token`} link="/embedding" linkLabel="chapter 8" right={mode === 'ours' && stages ? `‖x‖ ${fx(stages[0].xNorm, 2)}` : undefined} />
          <AnimatePresence initial={false}>
            {blocks.map((b) => (
              <motion.li
                key={b}
                layout={!reduce}
                initial={reduce ? false : { opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={reduce ? { opacity: 0 } : { opacity: 0, height: 0 }}
                transition={{ duration: reduce ? 0 : 0.28, delay: reduce ? 0 : Math.max(0, b - 4) * 0.04 }}
                className="!m-0 !p-0"
              >
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-lg border border-border bg-surface px-3 py-2">
                  <span className="w-[68px] shrink-0 text-[15.5px] font-semibold text-ink">Block {b + 1}</span>
                  <span className="order-last flex min-w-0 basis-full flex-wrap gap-1 sm:order-none sm:basis-auto sm:flex-1">
                    {ATTN_ORDER.map((l) => {
                      const Icon = LAYER_INFO[l].icon;
                      return (
                        <span key={l} title={`${LAYER_INFO[l].name} attention`} className="inline-flex h-7 items-center gap-1 rounded-md border border-border bg-surface-2 px-1.5 text-[13.5px] text-ink-2">
                          <Icon aria-hidden="true" size={14} />
                          {l}
                        </span>
                      );
                    })}
                    <span title="SwiGLU feed-forward" className="inline-flex h-7 items-center gap-1 rounded-md border border-border bg-surface-2 px-1.5 text-[13.5px] text-ink-2">
                      <Activity aria-hidden="true" size={14} />
                      ffn
                    </span>
                  </span>
                  <span className="num ml-auto shrink-0 text-[14.5px] text-ink-2 sm:ml-0">
                    {mode === 'ours' && stages ? `‖x‖ ${fx(stages[5 + b * 5].xNorm, 2)}` : `${int(perBlock)} params`}
                  </span>
                </div>
              </motion.li>
            ))}
          </AnimatePresence>
          <StackRow kind="io" title="RMSNorm_out" detail={`${cfg.dModel} weights γ, no other parameters`} right={mode === 'ours' && stages ? `‖x‖ ${fx(stages[stages.length - 1].xNorm, 2)}` : undefined} />
          <StackRow kind="io" title="Decoders" detail="number, text, datetime, boolean; only the masked cell’s output is read" link="/decoding" linkLabel="chapter 12" />
        </ol>
        <aside className="h-fit rounded-xl border border-dashed border-border-strong px-4 py-3.5">
          <div className="text-[16.5px] font-semibold text-ink">One set of masks for every block</div>
          <p className="!mb-2 mt-1 text-[15.5px] leading-snug text-ink-2">
            The four masks are built once per forward pass from the token ids and handed to every block. Blocks differ only in their weights.
          </p>
          <ul className="!m-0 grid gap-1 !p-0 text-[16px]" style={{ listStyle: 'none' }}>
            {ATTN_ORDER.map((l) => (
              <li key={l} className="!m-0 !p-0">
                <LayerLabel layer={l} />
              </li>
            ))}
          </ul>
          <p className="!mb-0 mt-2 text-[15px] leading-snug text-muted">
            Per block: 4 × 4 × d² attention + 3 × d × d_ff SwiGLU + 5 × d norm weights ={' '}
            <span className="num text-ink-2">{int(perBlock)}</span>.
          </p>
        </aside>
      </div>
      {mode === 'ours' && <p className="!mb-0 text-[15px] text-muted">‖x‖ values: the followed token’s residual stream after each block (live forward pass).</p>}
    </div>
  );
}

function StackRow({ title, detail, link, linkLabel, right }: { kind: 'io'; title: string; detail: string; link?: string; linkLabel?: string; right?: string }) {
  return (
    <li className="!m-0 !p-0">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border border-dashed border-border-strong bg-surface-2/60 px-3 py-2">
        <span className="text-[15.5px] font-semibold text-ink">{title}</span>
        <span className="min-w-[180px] flex-1 text-[14.5px] text-ink-2">{detail}</span>
        {right && <span className="num shrink-0 text-[14.5px] text-ink-2">{right}</span>}
        {link && (
          <Link to={link} data-plain className="inline-flex shrink-0 items-center gap-1 text-[14.5px] font-medium text-accent-text hover:underline">
            {linkLabel} <ArrowRight aria-hidden="true" size={14} />
          </Link>
        )}
      </div>
    </li>
  );
}
