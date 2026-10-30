/**
 * Section 1 figure: masking means "−∞ before the softmax".
 * Uses the REAL scaled scores q·k/√d_head of the trained mini model at the masked target (captured by
 * the engine's forward pass, before masking) and the engine's softmax probabilities. A per-key strip
 * shows the mechanism for every key: raw score → logit after masking → softmax weight.
 */
import { useMemo, useState } from 'react';
import { Cite, Figure, LayerLabel, ScaleNote, Segmented, StatusBadge, cx, divergingCss, formatPct, sequentialCss } from '@/components';
import type { AttnLayer } from '@/engine/types';
import { useAssets, useForward } from '@/state/hooks';
import { useResolvedTheme } from '@/styles/theme';
import type { LiveMasks } from './useLiveMasks';

type DemoLayer = Exclude<AttnLayer, 'full'>;

const signed = (x: number, d = 3) => x.toFixed(d).replace('-', '−');
const HATCH = 'repeating-linear-gradient(135deg, var(--axis) 0 1.5px, transparent 1.5px 5px)';

export function SoftmaxDemo({ live }: { live: LiveMasks }) {
  const { masks, S, labels, target } = live;
  const fr = useForward();
  const assets = useAssets();
  const mode = useResolvedTheme();
  const [layer, setLayer] = useState<DemoLayer>('feat');
  const [block, setBlock] = useState(0);
  const [head, setHead] = useState(0);
  const [hoverK, setHoverK] = useState<number | null>(null);
  const nBlocks = fr?.blocks.length ?? 4;
  const H = assets.weights?.config.nHeads ?? 4;
  const dHead = fr ? fr.d / H : 16;
  const q = target;

  const data = useMemo(() => {
    if (!fr) return null;
    const cap = fr.blocks[Math.min(block, fr.blocks.length - 1)].attn[layer];
    if (cap.skipped || cap.scores.length < H * S * S) return { skipped: true as const };
    const off = (head * S + q) * S;
    const m = masks[layer].subarray(q * S, (q + 1) * S);
    const score = Float64Array.from(cap.scores.subarray(off, off + S));
    const pCode = Float64Array.from(cap.probs.subarray(off, off + S));
    const allowed: number[] = [];
    for (let k = 0; k < S; k++) if (m[k]) allowed.push(k);
    let massCode = 0;
    for (const k of allowed) massCode += pCode[k];
    let maxAbsScore = 0;
    for (let k = 0; k < S; k++) if (!live.ctx.tokens[k].isPadding) maxAbsScore = Math.max(maxAbsScore, Math.abs(score[k]));
    const top = [...allowed]
      .sort((a, b) => pCode[b] - pCode[a])
      .slice(0, 5)
      .map((k) => ({ k, code: pCode[k], score: score[k] }));
    return {
      skipped: false as const,
      m,
      score,
      pCode,
      allowed: allowed.length,
      massCode,
      maxAbsScore,
      top,
      blocked: S - allowed.length,
    };
  }, [fr, block, layer, head, H, S, q, masks, live.ctx]);

  const fill = sequentialCss(mode === 'dark' ? 0.62 : 0.66, mode);

  return (
    <Figure
      wide
      title={
        <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span>Where the masked target’s attention goes, with real scores from the trained mini model</span>
          <ScaleNote inline />
        </span>
      }
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
        </>
      }
      caption={
        <>
          Query: the masked target (token #{q}, {labels[q]}). The raw scores are q·k/√d_head, with d_head = {dHead} here and 32 in the paper{' '}
          <Cite kind="spec" ref="§6.1" />. Switching block or head changes how the weight is split among the allowed keys, but not which keys are
          allowed: every block and every head uses the same mask <Cite kind="code" ref="rt/model.py:89-101" />. Those keys always get 100% of the
          weight.
        </>
      }
    >
      {!fr || !data ? (
        <p className="text-[16.5px] text-ink-2">{assets.status === 'error' ? `The forward pass is unavailable: ${assets.error}` : 'Loading the trained weights and running the forward pass…'}</p>
      ) : data.skipped ? (
        <p className="text-[16.5px] text-ink-2">This sublayer is switched off in the playground’s inference-time ablation, so it has no scores to show.</p>
      ) : (
        <div className="grid gap-5">
          {assets.weightsKind === 'init' && <StatusBadge status="warning">Untrained weights: the scores are from a random initialization</StatusBadge>}
          <MassBar
            title={
              <>
                <strong className="font-semibold text-ink">Attention weight after −∞ masking</strong>
              </>
            }
            allowed={data.allowed}
            blocked={data.blocked}
            mass={data.massCode}
            fill={fill}
            empty={data.allowed === 0}
          />

          <KeyStrips
            S={S}
            labels={labels}
            m={data.m}
            score={data.score}
            weights={data.pCode}
            maxAbsScore={data.maxAbsScore}
            hoverK={hoverK}
            setHoverK={setHoverK}
            padding={live.ctx.tokens.map((t) => t.isPadding)}
          />

          {data.allowed > 0 ? (
            <div className="min-w-0 overflow-x-auto">
              <table className="w-full border-collapse text-[15px]">
                <thead>
                  <tr className="border-b border-grid text-left text-ink-2">
                    <th scope="col" className="py-1 pr-3 font-medium">
                      Top allowed keys
                    </th>
                    <th scope="col" className="py-1 pr-3 text-right font-medium">
                      score
                    </th>
                    <th scope="col" className="py-1 text-right font-medium whitespace-nowrap">
                      weight
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.top.map((r) => (
                    <tr
                      key={r.k}
                      className={cx('border-b border-grid/60 last:border-0', hoverK === r.k && 'bg-surface-2')}
                      onPointerEnter={() => setHoverK(r.k)}
                      onPointerLeave={() => setHoverK(null)}
                    >
                      <td className="num py-1 pr-3 text-ink">
                        #{r.k} <BreakAtDot text={labels[r.k]} />
                      </td>
                      <td className="num py-1 pr-3 text-right text-ink-2">{signed(r.score)}</td>
                      <td className="num py-1 text-right text-ink">{formatPct(r.code, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-[16px] text-ink-2">
              The target’s neighbor row has no allowed key, so every weight is 0 and the sublayer outputs exactly 0.
            </p>
          )}
        </div>
      )}
    </Figure>
  );
}

/** Every key of the query row, in emission order: raw score, logit after masking, softmax weight. */
function KeyStrips({
  S,
  labels,
  m,
  score,
  weights,
  maxAbsScore,
  hoverK,
  setHoverK,
  padding,
}: {
  S: number;
  labels: string[];
  m: Uint8Array;
  score: Float64Array;
  weights: Float64Array;
  maxAbsScore: number;
  hoverK: number | null;
  setHoverK: (k: number | null) => void;
  padding: boolean[];
}) {
  const mode = useResolvedTheme();
  let maxW = 0;
  for (let k = 0; k < S; k++) maxW = Math.max(maxW, weights[k]);
  const logit = (k: number) => (m[k] ? score[k] : -Infinity);
  const rows: { id: string; title: string; cell: (k: number) => { bg: string; hatch?: boolean } }[] = [
    { id: 'score', title: 'raw score q·k/√d_head', cell: (k) => ({ bg: divergingCss(score[k], maxAbsScore, mode) }) },
    {
      id: 'logit',
      title: 'logit after Mask(·; M) (blocked → −∞)',
      cell: (k) => (Number.isFinite(logit(k)) ? { bg: divergingCss(logit(k), maxAbsScore, mode) } : { bg: 'transparent', hatch: true }),
    },
    { id: 'weight', title: 'softmax weight', cell: (k) => ({ bg: weights[k] > 0 ? sequentialCss(maxW > 0 ? weights[k] / maxW : 0, mode) : 'var(--surface-2)' }) },
  ];
  const hk = hoverK;
  return (
    <div className="min-w-0 rounded-lg border border-border px-3 pt-2.5 pb-3">
      <div className="mb-2">
        <span className="text-[16px] font-medium text-ink">All {S} keys of the target’s row, in emission order</span>
      </div>
      <div className="grid gap-1.5" onPointerLeave={() => setHoverK(null)}>
        {rows.map((r) => (
          <div key={r.id} className="min-w-0">
            <div className="mb-0.5 text-[14px] text-ink-2">
              {r.title}
            </div>
            <div className="flex h-[18px] w-full overflow-hidden rounded-[3px] border border-border bg-surface" role="img" aria-label={`${r.title} for each of the ${S} keys`}>
              {Array.from({ length: S }, (_, k) => {
                const c = r.cell(k);
                return (
                  <span
                    key={k}
                    className={cx('h-full min-w-0 flex-1', hk === k && 'outline outline-2 -outline-offset-1 outline-[var(--ink)]')}
                    style={{ background: c.hatch ? HATCH : c.bg, opacity: padding[k] ? 0.35 : 1 }}
                    onPointerEnter={() => setHoverK(k)}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-2 min-h-[1.5em] text-[15px] text-ink-2" aria-live="polite">
        {hk !== null ? (
          <span className="num">
            <span className="text-ink">
              #{hk} {labels[hk]}
            </span>{' '}
            · {m[hk] ? 'allowed' : 'blocked'} · score {signed(score[hk])} · logit {Number.isFinite(logit(hk)) ? signed(logit(hk)) : '−∞'} · weight{' '}
            <strong className="text-ink">{formatPct(weights[hk], 2)}</strong>
          </span>
        ) : (
          <span className="text-muted">
            Hover a key. Hatched = −∞. Scores: blue negative, red positive; weights: {mode === 'dark' ? 'lighter' : 'darker'} = more of the attention.
          </span>
        )}
      </div>
    </div>
  );
}

/** "c35·signup_date" with a line-break opportunity after the "·" (never inside a name). */
function BreakAtDot({ text }: { text: string }) {
  const i = text.indexOf('·');
  if (i < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, i + 1)}
      <wbr />
      {text.slice(i + 1)}
    </>
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
