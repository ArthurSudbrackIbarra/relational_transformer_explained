/**
 * "Where the masked cell looks": the attention weights of the masked target's query row (softmax
 * probabilities after masking, mean over heads; model.py:30-47) grouped by key type, per block,
 * for the baseline and for the ablated context. Read straight from the captured forward passes.
 */
import { useMemo, useState } from 'react';
import { ChartFrame, DataTable, LAYER_INFO, Segmented } from '@/components';
import { inContextLabels } from '@/engine/pipeline';
import type { AttnLayer, ForwardResult, Token } from '@/engine/types';
import { ATTN_ORDER } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { usePlayLive } from './context';

type Cat = 'self' | 'other' | 'target' | 'rest';
const CATS: Cat[] = ['self', 'other', 'target', 'rest'];
const CAT_LABEL: Record<Cat, string> = {
  self: 'self labels',
  other: 'other labels',
  target: 'the masked cell itself',
  rest: 'every other cell',
};

function fillOf(c: Cat): { fill: string; stroke?: string; opacity?: number; dash?: string } {
  switch (c) {
    case 'self':
      return { fill: catVar(5) };
    case 'other':
      return { fill: catVar(5), opacity: 0.42 };
    case 'target':
      return { fill: 'var(--surface)', stroke: 'var(--accent)', dash: '3 2' };
    default:
      return { fill: 'var(--axis)' };
  }
}

type Shares = Record<Cat, number> | 'skipped' | 'empty';

function shares(fr: ForwardResult, tokens: readonly Token[], catOfPos: (pos: number) => Cat, block: number, layer: AttnLayer): Shares {
  const a = fr.blocks[block]?.attn[layer];
  if (!a || a.skipped || a.probs.length === 0) return 'skipped';
  const S = fr.S;
  const H = a.probs.length / (S * S);
  const q = fr.targetPos;
  const out: Record<Cat, number> = { self: 0, other: 0, target: 0, rest: 0 };
  let total = 0;
  for (let h = 0; h < H; h++) {
    const o = h * S * S + q * S;
    for (let k = 0; k < S; k++) {
      const p = a.probs[o + k];
      if (p === 0 || tokens[k].isPadding) continue;
      out[catOfPos(k)] += p / H;
      total += p / H;
    }
  }
  return total < 1e-9 ? 'empty' : out;
}

function StackBar({ s, label }: { s: Shares; label: string }) {
  if (s === 'skipped' || s === 'empty') {
    return (
      <div className="flex h-5 items-center rounded-[4px] border border-dashed border-border-strong px-2 text-[13.5px] text-muted" aria-label={`${label}: ${s === 'skipped' ? 'layer switched off' : 'no keys'}`}>
        {s === 'skipped' ? 'switched off: no update' : 'no keys: zero update'}
      </div>
    );
  }
  let x = 0;
  return (
    <svg viewBox="0 0 100 10" preserveAspectRatio="none" className="block h-5 w-full overflow-visible" role="img" aria-label={`${label}: ${CATS.map((c) => `${CAT_LABEL[c]} ${(s[c] * 100).toFixed(0)}%`).join(', ')}`}>
      {CATS.map((c) => {
        const w = s[c] * 100;
        const f = fillOf(c);
        const r = (
          <rect
            key={c}
            x={x}
            y={0}
            width={Math.max(0, w)}
            height={10}
            style={{ fill: f.fill, stroke: f.stroke, opacity: f.opacity }}
            strokeWidth={f.stroke ? 0.8 : 0}
            strokeDasharray={f.dash}
            vectorEffect="non-scaling-stroke"
          >
            <title>
              {CAT_LABEL[c]}: {(s[c] * 100).toFixed(1)}%
            </title>
          </rect>
        );
        x += w;
        return r;
      })}
    </svg>
  );
}

export function AttentionShare() {
  const live = usePlayLive();
  const { baseFr, ablFr, ctx, abl, pdb, task, active } = live;
  const [layer, setLayer] = useState<AttnLayer>('col');

  const baseLabels = useMemo(() => inContextLabels(pdb, ctx, task), [pdb, ctx, task]);
  const baseCat = useMemo(() => {
    const role = new Map<number, Cat>();
    for (const l of baseLabels.self) role.set(l.pos, 'self');
    for (const l of baseLabels.other) role.set(l.pos, 'other');
    return (pos: number): Cat => (pos === ctx.targetPos ? 'target' : role.get(pos) ?? 'rest');
  }, [baseLabels, ctx]);
  const ablCat = useMemo(
    () =>
      (pos: number): Cat => {
        if (pos === abl.targetPos) return 'target';
        const r = abl.labelRole.get(abl.origPos[pos]);
        return r ?? 'rest';
      },
    [abl],
  );

  const nBlocks = baseFr?.blocks.length ?? 0;
  const rows = useMemo(() => {
    if (!baseFr) return [];
    return Array.from({ length: nBlocks }, (_, b) => ({
      b,
      base: shares(baseFr, ctx.tokens, baseCat, b, layer),
      cur: ablFr ? shares(ablFr, abl.tokens, ablCat, b, layer) : ('empty' as Shares),
    }));
  }, [baseFr, ablFr, ctx, abl, baseCat, ablCat, layer, nBlocks]);

  const pct = (s: Shares, c: Cat) => (typeof s === 'string' ? (s === 'skipped' ? 'off' : '0') : `${(s[c] * 100).toFixed(1)}%`);

  return (
    <ChartFrame
      title="Where the masked cell looks"
      subtitle={`Attention weights of the masked cell’s query in ${LAYER_INFO[layer].name.toLowerCase()} attention, averaged over heads, grouped by key.`}
      legend={
        <ul className="!m-0 flex !list-none flex-wrap items-center gap-x-4 gap-y-1 !p-0 text-[15px] text-ink-2" aria-label="Legend">
          {CATS.map((c) => {
            const f = fillOf(c);
            return (
              <li key={c} className="!m-0 inline-flex items-center gap-1.5 !p-0">
                <svg width="14" height="12" aria-hidden="true">
                  <rect x="1" y="1" width="12" height="10" rx="2" style={{ fill: f.fill, stroke: f.stroke, opacity: f.opacity }} strokeWidth={f.stroke ? 1.5 : 0} strokeDasharray={f.dash} />
                </svg>
                {CAT_LABEL[c]}
              </li>
            );
          })}
        </ul>
      }
      table={
        <DataTable
          caption="Attention shares of the masked cell"
          columns={['Block', ...CATS.flatMap((c) => [`baseline: ${CAT_LABEL[c]}`, `now: ${CAT_LABEL[c]}`])]}
          rows={rows.map((r) => [`block ${r.b + 1}`, ...CATS.flatMap((c) => [pct(r.base, c), pct(r.cur, c)])])}
        />
      }
    >
      <div className="mb-3">
        <Segmented<AttnLayer>
          label="Attention layer"
          value={layer}
          onChange={setLayer}
          options={ATTN_ORDER.map((l) => ({ value: l, label: LAYER_INFO[l].name }))}
        />
      </div>
      {!baseFr ? (
        <p className="text-[15px] text-muted">Loading the model…</p>
      ) : (
        <div className="grid gap-3">
          {rows.map((r) => (
            <div key={r.b} className="grid grid-cols-[72px_minmax(0,1fr)] items-center gap-x-3 gap-y-1">
              <div className="row-span-2 text-[15px] font-medium text-ink-2">block {r.b + 1}</div>
              <div className="grid grid-cols-[86px_minmax(0,1fr)] items-center gap-2">
                <span className="text-[14px] text-muted">baseline</span>
                <StackBar s={r.base} label={`Block ${r.b + 1}, baseline`} />
              </div>
              <div className="grid grid-cols-[86px_minmax(0,1fr)] items-center gap-2">
                <span className="text-[14px] text-ink-2">{active ? 'now' : 'now (same)'}</span>
                <StackBar s={r.cur} label={`Block ${r.b + 1}, now`} />
              </div>
            </div>
          ))}
        </div>
      )}
    </ChartFrame>
  );
}
