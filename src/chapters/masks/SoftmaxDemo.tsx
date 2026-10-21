/**
 * Section 1 figure: why masking means "−∞ before the softmax" and not "multiply the scores by M".
 * Uses the REAL scaled scores q·k/√d_head of the trained mini model at the masked target (captured by
 * the engine's forward pass) and the engine's softmax probabilities. The "⊙ M read literally" row is a
 * clearly labelled counterfactual: softmax(score · M), which is what the v1/v2 formula says on paper.
 */
import { useMemo, useState } from 'react';
import { Cite, Figure, LayerLabel, ScaleNote, Segmented, StatusBadge, cx, formatPct } from '@/components';
import type { AttnLayer } from '@/engine/types';
import { useAssets, useForward } from '@/state/hooks';
import { useResolvedTheme } from '@/styles/theme';
import { sequentialCss } from '@/components';
import { int } from './maskUtils';
import type { LiveMasks } from './useLiveMasks';

type DemoLayer = Exclude<AttnLayer, 'full'>;

export function SoftmaxDemo({ live }: { live: LiveMasks }) {
  const { masks, S, labels, target } = live;
  const fr = useForward();
  const assets = useAssets();
  const mode = useResolvedTheme();
  const [layer, setLayer] = useState<DemoLayer>('feat');
  const [block, setBlock] = useState(0);
  const [head, setHead] = useState(0);
  const nBlocks = fr?.blocks.length ?? 4;
  const H = assets.weights?.config.nHeads ?? 4;
  const dHead = fr ? fr.d / H : 16;
  const q = target;

  const data = useMemo(() => {
    if (!fr) return null;
    const cap = fr.blocks[Math.min(block, fr.blocks.length - 1)].attn[layer];
    if (cap.skipped || cap.scores.length < H * S * S) return { skipped: true as const };
    const off = (head * S + q) * S;
    const allowed: number[] = [];
    for (let k = 0; k < S; k++) if (masks[layer][q * S + k]) allowed.push(k);
    // code / paper v3: −∞ masking (engine probabilities)
    let massCode = 0;
    for (const k of allowed) massCode += cap.probs[off + k];
    // counterfactual: softmax(score ⊙ M) — blocked keys get logit 0
    const z = new Float64Array(S);
    let mx = -Infinity;
    for (let k = 0; k < S; k++) {
      z[k] = cap.scores[off + k] * masks[layer][q * S + k];
      if (z[k] > mx) mx = z[k];
    }
    let sum = 0;
    const p = new Float64Array(S);
    for (let k = 0; k < S; k++) {
      p[k] = Number.isFinite(z[k]) ? Math.exp(z[k] - mx) : 0;
      sum += p[k];
    }
    let massLit = 0;
    for (let k = 0; k < S; k++) {
      p[k] /= sum;
      if (masks[layer][q * S + k]) massLit += p[k];
    }
    const top = [...allowed]
      .sort((a, b) => cap.probs[off + b] - cap.probs[off + a])
      .slice(0, 5)
      .map((k) => ({ k, code: cap.probs[off + k], lit: p[k], score: cap.scores[off + k] }));
    return { skipped: false as const, allowed: allowed.length, massCode, massLit, top, blocked: S - allowed.length };
  }, [fr, block, layer, head, H, S, q, masks]);

  const fill = sequentialCss(mode === 'dark' ? 0.62 : 0.66, mode);

  return (
    <Figure
      wide
      title="Where the masked target’s attention goes, with real scores from the trained mini model"
      controls={
        <>
          <Segmented
            label="Layer"
            value={layer}
            onChange={setLayer}
            options={(['col', 'feat', 'nbr'] as DemoLayer[]).map((l) => ({ value: l, label: <LayerLabel layer={l} size={13} /> }))}
          />
          <Segmented label="Block" value={block} onChange={setBlock} options={Array.from({ length: nBlocks }, (_, i) => ({ value: i, label: `block ${i + 1}` }))} />
          <Segmented label="Head" value={head} onChange={setHead} options={Array.from({ length: H }, (_, i) => ({ value: i, label: `head ${i + 1}` }))} />
          <ScaleNote inline className="mb-1.5" />
        </>
      }
      caption={
        <>
          Query: the masked target (token #{q}, {labels[q]}). Both rows use the same scores q·k/√d<sub>head</sub> (d<sub>head</sub> = {dHead} here,
          32 in the paper) <Cite kind="spec" ref="§6.1" />. Top row: what the code computes. Bottom row: the v1/v2 formula taken literally, shown
          only to explain the correction; the code does not compute it.
        </>
      }
    >
      {!fr || !data ? (
        <p className="text-[16.5px] text-ink-2">{assets.status === 'error' ? `The forward pass is unavailable: ${assets.error}` : 'Loading the trained weights and running the forward pass…'}</p>
      ) : data.skipped ? (
        <p className="text-[16.5px] text-ink-2">This sublayer is switched off in the playground’s inference-time ablation, so it has no scores to show.</p>
      ) : (
        <div className="grid gap-5">
          {assets.weightsKind === 'init' && <StatusBadge status="warning">Untrained weights: the scores are from a random initialisation</StatusBadge>}
          <MassBar
            title={
              <>
                <strong className="font-semibold text-ink">−∞ masking</strong> <span className="text-muted">(paper v3, and the code)</span>
              </>
            }
            allowed={data.allowed}
            blocked={data.blocked}
            mass={data.massCode}
            fill={fill}
            empty={data.allowed === 0}
          />
          <MassBar
            title={
              <>
                <strong className="font-semibold text-ink">QKᵀ ⊙ M read literally</strong> <span className="text-muted">(v1/v2 notation, counterfactual)</span>
              </>
            }
            allowed={data.allowed}
            blocked={data.blocked}
            mass={data.massLit}
            fill={fill}
            empty={false}
          />
          {data.allowed > 0 ? (
            <div className="min-w-0 overflow-x-auto">
              <table className="w-full border-collapse text-[15px]">
                <thead>
                  <tr className="border-b border-grid text-left text-ink-2">
                    <th scope="col" className="py-1 pr-3 font-medium">
                      Top allowed keys (weights)
                    </th>
                    <th scope="col" className="hidden py-1 pr-3 text-right font-medium sm:table-cell">
                      score
                    </th>
                    <th scope="col" className="py-1 pr-3 text-right font-medium">
                      −∞ masking
                    </th>
                    <th scope="col" className="py-1 text-right font-medium">
                      ⊙ M literal
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.top.map((r) => (
                    <tr key={r.k} className="border-b border-grid/60 last:border-0">
                      <td className="num py-1 pr-3 text-ink">
                        #{r.k} {labels[r.k]}
                      </td>
                      <td className="num hidden py-1 pr-3 text-right text-ink-2 sm:table-cell">{r.score.toFixed(3).replace('-', '−')}</td>
                      <td className="num py-1 pr-3 text-right text-ink">{formatPct(r.code, 1)}</td>
                      <td className="num py-1 text-right text-ink-2">{formatPct(r.lit, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-[16px] text-ink-2">
              The target’s neighbor row has no allowed key. With −∞ masking every weight is 0 and the sublayer outputs exactly 0. Read literally,
              ⊙ M would instead give every one of the {int(S)} keys the same logit 0, so the weight would spread evenly over all of them.
            </p>
          )}
        </div>
      )}
    </Figure>
  );
}

function MassBar({
  title,
  allowed,
  blocked,
  mass,
  fill,
  empty,
}: {
  title: React.ReactNode;
  allowed: number;
  blocked: number;
  mass: number;
  fill: string;
  empty: boolean;
}) {
  const a = Math.max(0, Math.min(1, mass));
  return (
    <div className="min-w-0">
      <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5 text-[15.5px]">
        <span>{title}</span>
        <span className="num text-[15px] text-ink-2">
          {empty ? (
            'no allowed key: all weights 0, output 0'
          ) : (
            <>
              {allowed} allowed {allowed === 1 ? 'key' : 'keys'} get <strong className="text-ink">{formatPct(a, 1)}</strong> · {blocked} blocked get{' '}
              <strong className={cx(1 - a > 0.0005 ? 'text-ink' : 'text-muted')}>{formatPct(1 - a, 1)}</strong>
            </>
          )}
        </span>
      </div>
      <div className="flex h-5 w-full overflow-hidden rounded-[5px] border border-border bg-surface-2" role="img" aria-label={`${formatPct(a, 1)} of the attention weight on allowed keys`}>
        {!empty && <div style={{ width: `${a * 100}%`, background: fill }} className="h-full transition-[width] duration-500" />}
        {!empty && a < 1 && (
          <div
            className="h-full flex-1"
            style={{ backgroundImage: 'repeating-linear-gradient(135deg, var(--axis) 0 2px, transparent 2px 6px)' }}
          />
        )}
      </div>
    </div>
  );
}
