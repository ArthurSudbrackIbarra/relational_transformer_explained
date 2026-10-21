/**
 * Figure 5: the residual stream of one token through the whole stack. Left: ‖update‖ of each of the
 * 5·nBlocks sublayers (a block × sublayer grid). Right: the token's d-dim vector after every stage as a
 * diverging matrix, with its cosine to the last block's output. All numbers come from the forward capture.
 */
import { Activity } from 'lucide-react';
import { useMemo, useState } from 'react';
import { LAYER_INFO, ScaleLegend, cx, formatSigned, maxAbs } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import type { AttnLayer } from '@/engine/types';
import { useResolvedTheme } from '@/styles/theme';
import { divergingCss } from '@/components/viz/scales';
import type { BlockLive } from './useBlockLive';
import { TokenCaption, TokenPicker } from './TokenPicker';
import { allowedKeys, fx, int, streamStages, SUB_NAME, SUB_SHORT, SUBLAYERS, type Stage } from './utils';

export function ResidualStream({ live, follow, setFollow }: { live: BlockLive; follow: number; setFollow: (p: number) => void }) {
  const { fr } = live;
  const stages = useMemo(() => (fr ? streamStages(fr, follow) : null), [fr, follow]);
  const [sel, setSel] = useState<number | null>(null);

  return (
    <div className="grid grid-cols-1 gap-4">
      <div className="grid grid-cols-1 gap-1">
        <span className="text-[15px] font-medium text-ink-2">Token</span>
        <TokenPicker live={live} value={follow} onChange={setFollow} label="Token whose residual stream is shown" />
      </div>
      <TokenCaption live={live} pos={follow} prefix="Following" />
      {!stages ? (
        <div className="text-[16px] text-muted">Loading the trained weights…</div>
      ) : (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.25fr)]">
          <div className="min-w-0">
            <div className="mb-2 text-[16px] font-semibold text-ink">‖Δ‖: how much each sublayer adds</div>
            <UpdateGrid live={live} stages={stages} follow={follow} sel={sel} setSel={setSel} />
          </div>
          <div className="min-w-0">
            <div className="mb-2 text-[16px] font-semibold text-ink">The vector after every stage, and its cosine with the last block’s output</div>
            <StreamMatrix stages={stages} sel={sel} setSel={setSel} />
          </div>
        </div>
      )}
    </div>
  );
}

function UpdateGrid({ live, stages, follow, sel, setSel }: { live: BlockLive; stages: Stage[]; follow: number; sel: number | null; setSel: (i: number | null) => void }) {
  const nB = live.nBlocks;
  const max = Math.max(...stages.map((s) => (Number.isFinite(s.updNorm) ? s.updNorm : 0))) || 1;
  const keys = useMemo(() => {
    const out: Partial<Record<AttnLayer, number>> = {};
    for (const l of ['col', 'feat', 'nbr', 'full'] as AttnLayer[]) out[l] = allowedKeys(live.masks, l, follow, live.S);
    return out;
  }, [live.masks, live.S, follow]);
  const empty = (Object.keys(keys) as AttnLayer[]).filter((l) => keys[l] === 0).map((l) => SUB_NAME[l]);
  return (
    <div>
      <table className="w-full table-fixed border-separate border-spacing-1">
        <caption className="sr-only">Length of the residual update of each sublayer for the followed token</caption>
        <thead>
          <tr>
            <th scope="col" className="w-[54px] sm:w-[70px]" />
            {SUBLAYERS.map((s) => {
              const Icon = s === 'ffn' ? Activity : LAYER_INFO[s as AttnLayer].icon;
              return (
                <th key={s} scope="col" className="px-0.5 pb-1 text-center text-[14px] font-medium text-ink-2" title={`${SUB_NAME[s]} ${s === 'ffn' ? 'feed-forward' : 'attention'}`}>
                  <span className="inline-flex flex-col items-center gap-0.5">
                    <Icon aria-hidden="true" size={15} />
                    <span>{SUB_SHORT[s]}</span>
                  </span>
                  <span className="sr-only">{SUB_NAME[s]}</span>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: nB }, (_, b) => (
            <tr key={b}>
              <th scope="row" className="pr-1 text-left text-[15px] font-semibold whitespace-nowrap text-ink">
                <span className="sm:hidden">B{b + 1}</span>
                <span className="hidden sm:inline">Block {b + 1}</span>
              </th>
              {SUBLAYERS.map((s, i) => {
                const st = stages[1 + b * 5 + i];
                const on = sel === st.index;
                const zero = s !== 'ffn' && keys[s as AttnLayer] === 0;
                const v = st.updNorm;
                return (
                  <td key={s} className="p-0">
                    <button
                      type="button"
                      aria-pressed={on}
                      aria-label={`Block ${b + 1}, ${SUB_NAME[s]}: update length ${st.skipped ? 'off' : fx(v, 3)}`}
                      onClick={() => setSel(on ? null : st.index)}
                      onPointerEnter={() => setSel(st.index)}
                      className={cx(
                        'flex w-full flex-col gap-1 rounded-lg border px-1.5 py-1.5 text-left transition-colors sm:px-2',
                        on ? 'border-accent bg-accent-soft' : 'border-border bg-surface hover:bg-surface-2',
                      )}
                    >
                      <span className={cx('num text-[14.5px] leading-none', zero ? 'text-muted' : on ? 'font-semibold text-ink' : 'text-ink')}>
                        {st.skipped ? 'off' : zero ? '0' : fx(v, 2)}
                      </span>
                      <span className="block h-[6px] w-full overflow-hidden rounded-full bg-surface-3">
                        <span className="block h-full rounded-full transition-[width] duration-500" style={{ width: `${Number.isFinite(v) && v > 0 ? Math.max(3, (v / max) * 100) : 0}%`, background: on ? 'var(--accent)' : 'var(--seq-400)' }} />
                      </span>
                    </button>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="!mb-0 mt-2 text-[15px] leading-snug text-ink-2">
        Keys this token may read:{' '}
        {(['col', 'feat', 'nbr', 'full'] as AttnLayer[]).map((l, i) => (
          <span key={l} className="num whitespace-nowrap">
            {i ? ', ' : ''}
            {SUB_SHORT[l]} {int(keys[l] ?? 0)}
          </span>
        ))}
        .{' '}
        {empty.length > 0
          ? `${empty.join(' and ')} gives it none, so that sublayer adds exactly 0 in every block (shown as 0).`
          : 'Every layer gives it at least one key.'}
      </p>
    </div>
  );
}

function StreamMatrix({ stages, sel, setSel }: { stages: Stage[]; sel: number | null; setSel: (i: number | null) => void }) {
  const mode = useResolvedTheme();
  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(300, size.width || 600);
  const narrow = W < 520;
  const labelW = narrow ? 76 : 96;
  const cosW = narrow ? 50 : 112;
  const d = stages[0].x.length;
  const nB = Math.max(1, ...stages.map((s) => s.block + 1));
  const gridW = W - labelW - cosW - 8;
  const cw = gridW / d;
  const rh = 15;
  const gap = 2;
  const H = stages.length * (rh + gap);
  const m = useMemo(() => maxAbs(stages.flatMap((s) => Array.from(s.x))) || 1, [stages]);
  const [hover, setHover] = useState<{ r: number; j: number } | null>(null);
  const hs = hover ? stages[hover.r] : sel !== null ? stages[sel] : null;

  const rowLabel = (s: Stage) => (s.kind === 'embed' ? (narrow ? 'embed' : 'embedding') : s.kind === 'out' ? 'norm_out' : narrow ? `${s.block + 1}·${SUB_SHORT[s.kind]}` : `B${s.block + 1} · ${SUB_SHORT[s.kind]}`);

  return (
    <div>
      <div ref={ref} className="min-w-0">
        <svg
          width={W}
          height={H + 22}
          role="img"
          aria-label={`The followed token's ${d}-dimensional representation after each of ${stages.length} stages, as coloured cells, with its cosine similarity to the last block's output`}
          className="block"
          onPointerLeave={() => setHover(null)}
        >
          <text x={labelW + gridW + 10} y={13} fontSize={13.5} style={{ fill: 'var(--muted)' }}>
            {narrow ? 'cos' : `cos to B${nB}`}
          </text>
          <text x={labelW} y={13} fontSize={13.5} style={{ fill: 'var(--muted)' }}>
            dims 0 … {d - 1}
          </text>
          <g transform="translate(0,22)">
            {stages.map((s, r) => {
              const y = r * (rh + gap);
              const on = sel === s.index || hover?.r === r;
              const blockStart = s.kind === 'col' || s.kind === 'out';
              return (
                <g key={r}>
                  {blockStart && r > 0 && <line x1={0} x2={W} y1={y - gap / 2} y2={y - gap / 2} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />}
                  <text x={labelW - 6} y={y + rh - 3} textAnchor="end" fontSize={13} className="num" style={{ fill: on ? 'var(--ink)' : 'var(--ink-2)', fontWeight: on ? 600 : 400 }}>
                    {rowLabel(s)}
                  </text>
                  {Array.from(s.x, (v, j) => (
                    <rect key={j} x={labelW + j * cw} y={y} width={Math.max(0.6, cw - (cw > 5 ? 0.6 : 0))} height={rh} style={{ fill: divergingCss(v, m, mode) }} />
                  ))}
                  {on && <rect x={labelW - 1} y={y - 1} width={gridW + 2} height={rh + 2} fill="none" style={{ stroke: 'var(--ink)' }} strokeWidth={1.5} />}
                  {/* cosine */}
                  {!narrow && (
                    <>
                      <rect x={labelW + gridW + 10} y={y + 4} width={56} height={7} rx={3.5} style={{ fill: 'var(--surface-3)' }} />
                      <rect x={labelW + gridW + 10} y={y + 4} width={Math.max(0, Math.min(1, s.cosLast)) * 56} height={7} rx={3.5} style={{ fill: 'var(--seq-500)' }} />
                    </>
                  )}
                  <text x={W - 2} y={y + rh - 3} textAnchor="end" fontSize={13} className="num" style={{ fill: 'var(--ink-2)' }}>
                    {fx(s.cosLast, 2)}
                  </text>
                  <rect
                    x={0}
                    y={y}
                    width={W}
                    height={rh + gap}
                    fill="transparent"
                    onPointerMove={(e) => {
                      const r0 = e.currentTarget.getBoundingClientRect();
                      const px = e.clientX - r0.left - labelW;
                      const j = Math.floor(px / cw);
                      setHover({ r, j: j >= 0 && j < d ? j : -1 });
                    }}
                    onClick={() => setSel(sel === s.index ? null : s.index)}
                    className="cursor-crosshair"
                  />
                </g>
              );
            })}
          </g>
        </svg>
      </div>
      <div className="num mt-1.5 min-h-[2.6em] text-[14.5px] leading-snug text-ink-2" aria-live="polite">
        {hs ? (
          <>
            <span className="font-sans font-semibold text-ink">{hs.kind === 'embed' ? 'input embedding' : hs.kind === 'out' ? 'after norm_out' : `after block ${hs.block + 1}, ${SUB_NAME[hs.kind]}`}</span>
            {hover && hover.j >= 0 && (
              <>
                {' '}
                · dim {hover.j} = <span className="text-ink">{formatSigned(hs.x[hover.j], 3)}</span>
              </>
            )}{' '}
            · ‖x‖ = <span className="text-ink">{fx(hs.xNorm, 3)}</span>
            {Number.isFinite(hs.updNorm) && (
              <>
                {' '}
                · ‖Δ‖ = <span className="text-ink">{fx(hs.updNorm, 3)}</span>
              </>
            )}{' '}
            · cos = <span className="text-ink">{fx(hs.cosLast, 3)}</span>
          </>
        ) : (
          <span className="font-sans text-muted">Hover a row (or a cell of the ‖Δ‖ grid) to read it.</span>
        )}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1">
        <ScaleLegend scale="diverging" domain={[-m, m]} mode={mode} />
      </div>
    </div>
  );
}
