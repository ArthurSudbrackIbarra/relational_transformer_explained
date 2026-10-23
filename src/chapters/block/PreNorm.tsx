/**
 * Figure 2: where the norm sits. Paper Algorithm 2 (p.26) prints X ← X + Norm(MHA(X; M)); the code is
 * pre-norm, x ← x + MHA(RMSNorm(x); M) (rt/model.py:84-85; spec §7.1, §10 #1). The live chart shows the
 * consequence in the trained mini model: the residual stream is never rescaled inside the stack, only the
 * copy each sublayer reads, until norm_out.
 */
import { scaleLinear } from 'd3';
import { useMemo, useState } from 'react';
import { ChartFrame, DataTable, Legend, cx } from '@/components';
import { TooltipBox } from '@/components/viz/chartParts';
import { useElementSize } from '@/components/viz/useSize';
import type { ForwardResult } from '@/engine/types';
import { fx, streamStages, SUB_SHORT, type Stage } from './utils';

// ─────────────────────────────────────────────────────────────────────────────
// Two wirings
// ─────────────────────────────────────────────────────────────────────────────

function Wiring({ variant }: { variant: 'paper' | 'code' }) {
  const first = variant === 'paper' ? 'MHA(·; M)' : 'RMSNorm';
  const second = variant === 'paper' ? 'Norm' : 'MHA(·; M)';
  const W = 312;
  const H = 128;
  const y = 104;
  const by = 22;
  const bh = 38;
  const b1 = { x: 46, w: 104 };
  const b2 = { x: 168, w: 104 };
  const plusX = 290;
  const id = `wire-${variant}`;
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={variant === 'paper' ? 'Algorithm 2: the sublayer output is normalized, then added to X' : 'Code: a normalized copy of x goes into the sublayer, whose output is added to x'} className="block max-w-full">
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink-2)' }} />
        </marker>
      </defs>
      {/* identity path */}
      <line x1={6} y1={y} x2={W - 4} y2={y} style={{ stroke: 'var(--muted)', opacity: 0.45 }} strokeWidth={5} strokeLinecap="round" />
      <text x={8} y={y - 10} fontSize={15} fontStyle="italic" fontWeight={600} style={{ fill: 'var(--ink)' }}>
        x
      </text>
      <path d={`M24,${y - 4}V${by + bh / 2}H${b1.x - 3}`} fill="none" style={{ stroke: 'var(--ink-2)' }} strokeWidth={2} markerEnd={`url(#${id})`} />
      <path d={`M${b1.x + b1.w},${by + bh / 2}H${b2.x - 3}`} fill="none" style={{ stroke: 'var(--ink-2)' }} strokeWidth={2} markerEnd={`url(#${id})`} />
      <path d={`M${b2.x + b2.w},${by + bh / 2}H${plusX}V${y - 12}`} fill="none" style={{ stroke: 'var(--ink-2)' }} strokeWidth={2} markerEnd={`url(#${id})`} />
      {[
        { b: b1, t: first },
        { b: b2, t: second },
      ].map(({ b, t }) => {
        const isNorm = t === 'Norm' || t === 'RMSNorm';
        return (
          <g key={t}>
            <rect x={b.x} y={by} width={b.w} height={bh} rx={8} style={{ fill: isNorm ? 'var(--accent-soft)' : 'var(--surface)', stroke: isNorm ? 'var(--accent)' : 'var(--border-strong)' }} strokeWidth={isNorm ? 2 : 1.5} />
            <text x={b.x + b.w / 2} y={by + bh / 2 + 5} textAnchor="middle" fontSize={15} fontWeight={isNorm ? 600 : 500} style={{ fill: 'var(--ink)' }}>
              {t}
            </text>
          </g>
        );
      })}
      <circle cx={plusX} cy={y} r={10} style={{ fill: 'var(--surface)', stroke: 'var(--ink-2)' }} strokeWidth={2} />
      <path d={`M${plusX - 5},${y}H${plusX + 5}M${plusX},${y - 5}V${y + 5}`} style={{ stroke: 'var(--ink-2)' }} strokeWidth={2} />
    </svg>
  );
}

export function WiringCompare() {
  return (
    <div className="wide my-7 grid gap-4 md:grid-cols-2">
      {(['paper', 'code'] as const).map((v) => (
        <div key={v} className={cx('card min-w-0 px-4 py-3.5', v === 'code' && 'border-accent/40')}>
          <div className="eyebrow mb-1">{v === 'paper' ? 'Paper, Algorithm 2 as printed' : 'Code, rt/model.py:84-85 (this site)'}</div>
          <div className="mb-2 text-[17px] text-ink">
            {v === 'paper' ? (
              <span className="font-mono text-[16px]">X ← X + Norm(MHA(X; M))</span>
            ) : (
              <span className="font-mono text-[16px]">x = x + attns[l](norms[l](x), …)</span>
            )}
          </div>
          <div className="flex justify-center">
            <Wiring variant={v} />
          </div>
          <p className="!mb-0 mt-2 text-[15.5px] leading-snug text-ink-2">
            {v === 'paper'
              ? 'The sublayer reads the raw stream and its output is normalized before the add.'
              : 'The sublayer reads a normalized copy; its raw output is added. The stream itself is never normalized inside the block.'}
          </p>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Live chart: ‖x‖ of the residual stream vs ‖RMSNorm(x)‖ read by each sublayer
// ─────────────────────────────────────────────────────────────────────────────

const T = { tick: 13.5, axis: 14.5, label: 14 };

export function StreamNormChart({ fr, pos, tokenName }: { fr: ForwardResult; pos: number; tokenName: string }) {
  const stages = useMemo(() => streamStages(fr, pos), [fr, pos]);
  const nB = fr.blocks.length;
  const nStage = stages.length; // 1 + 5·nB + 1
  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(300, size.width || 700);
  const narrow = W < 560;
  const shortTicks = W < 820;
  const M = { top: 18, right: narrow ? 14 : 150, bottom: 58, left: 48 };
  const H = 300;
  const iw = W - M.left - M.right;
  const ih = H - M.top - M.bottom;
  const [hover, setHover] = useState<number | null>(null);

  const yMax = Math.max(...stages.map((s) => s.xNorm), ...stages.map((s) => (Number.isFinite(s.hNorm) ? s.hNorm : 0))) * 1.08;
  const x = scaleLinear().domain([0, nStage - 1]).range([0, iw]);
  const y = scaleLinear().domain([0, yMax]).range([ih, 0]).nice();
  const yTicks = y.ticks(5);

  const streamPts = stages.slice(0, nStage - 1).map((s) => [x(s.index), y(s.xNorm)] as const);
  const hPts = stages.filter((s) => Number.isFinite(s.hNorm)).map((s) => [x(s.index), y(s.hNorm)] as const);
  const last = stages[nStage - 1];
  const line = (pts: readonly (readonly [number, number])[]) => pts.map((p, i) => `${i ? 'L' : 'M'}${p[0]},${p[1]}`).join('');

  const hs = hover !== null ? stages[hover] : null;
  const endLabels = (() => {
    const ls = [
      { id: 'x', y: y(stages[nStage - 2].xNorm), text: `‖x‖ ${fx(stages[nStage - 2].xNorm, 1)}` },
      { id: 'out', y: y(last.xNorm), text: `norm_out ${fx(last.xNorm, 1)}` },
      ...(hPts.length ? [{ id: 'h', y: hPts[hPts.length - 1][1], text: `RMSNorm(x) ${fx(stages[nStage - 2].hNorm, 1)}` }] : []),
    ].sort((a, b) => a.y - b.y);
    for (let i = 1; i < ls.length; i++) if (ls[i].y - ls[i - 1].y < 17) ls[i].y = ls[i - 1].y + 17;
    return ls;
  })();
  const table = (
    <DataTable
      caption={`‖x‖ and ‖RMSNorm(x)‖ along the stack for ${tokenName}`}
      columns={['stage', '‖x‖ after the stage', '‖RMSNorm(x)‖ read by the sublayer']}
      rows={stages.map((s) => [s.label, fx(s.xNorm, 3), Number.isFinite(s.hNorm) ? fx(s.hNorm, 3) : '–'])}
    />
  );

  return (
    <ChartFrame
      title={`What the stream carries and what each sublayer reads: ${tokenName}`}
      legend={
        <Legend
          items={[
            { id: 'x', label: '‖x‖, the residual stream after each stage', slot: 1, mark: 'line' },
            { id: 'h', label: '‖RMSNorm(x)‖, the copy the next sublayer reads', slot: 2, mark: 'line' },
          ]}
        />
      }
      table={table}
    >
      <div ref={ref} className="relative min-w-0">
        <svg width={W} height={H} role="img" aria-label="Line chart of the residual-stream norm and the normalized sublayer inputs, per sublayer" className="block">
          <g transform={`translate(${M.left},${M.top})`}>
            {/* block bands */}
            {Array.from({ length: nB }, (_, b) => {
              const x0 = x(1 + b * 5) - x(0.5) + x(0);
              const x1 = x(5 + b * 5) + x(0.5) - x(0);
              return (
                <g key={b}>
                  {b % 2 === 0 && <rect x={x0} y={0} width={x1 - x0} height={ih} style={{ fill: 'var(--surface-2)' }} />}
                  <text x={(x0 + x1) / 2} y={ih + 22} textAnchor="middle" fontSize={T.tick} style={{ fill: 'var(--ink-2)' }}>
                    {shortTicks ? `B${b + 1}` : `Block ${b + 1}`}
                  </text>
                </g>
              );
            })}
            <text x={x(0)} y={ih + 22} textAnchor="middle" fontSize={T.tick} style={{ fill: 'var(--ink-2)' }}>
              x⁽⁰⁾
            </text>
            <text x={x(nStage - 1)} y={ih + 22} textAnchor="end" fontSize={T.tick} style={{ fill: 'var(--ink-2)' }}>
              {shortTicks ? 'out' : 'norm_out'}
            </text>
            <text x={iw / 2} y={ih + 46} textAnchor="middle" fontSize={T.axis} style={{ fill: 'var(--muted)' }}>
              {narrow ? 'stage (5 sublayers per block)' : 'stage of the forward pass (5 sublayers per block)'}
            </text>
            {yTicks.map((t) => (
              <g key={t}>
                <line x1={0} x2={iw} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
                <text x={-8} y={y(t) + 4.5} textAnchor="end" fontSize={T.tick} className="num" style={{ fill: 'var(--muted)' }}>
                  {t}
                </text>
              </g>
            ))}
            <path d={line(streamPts)} fill="none" style={{ stroke: 'var(--cat-1)' }} strokeWidth={2} />
            <path d={line(hPts)} fill="none" style={{ stroke: 'var(--cat-2)' }} strokeWidth={2} />
            {/* the jump through norm_out */}
            <path d={`M${x(nStage - 2)},${y(stages[nStage - 2].xNorm)}L${x(nStage - 1)},${y(last.xNorm)}`} fill="none" style={{ stroke: 'var(--cat-1)' }} strokeWidth={2} strokeDasharray="4 4" />
            {stages.map((s) => (
              <circle key={`x${s.index}`} cx={x(s.index)} cy={y(s.xNorm)} r={s.index === nStage - 1 || s.index === 0 ? 5 : 4} style={{ fill: 'var(--cat-1)', stroke: 'var(--surface)' }} strokeWidth={s.index === nStage - 1 || s.index === 0 ? 2 : 1} />
            ))}
            {/* direct end labels (wide only), pushed apart so they never collide */}
            {!narrow &&
              endLabels.map((l) => (
                <text key={l.id} x={iw + 10} y={l.y + 4.5} fontSize={T.label} style={{ fill: 'var(--ink-2)' }}>
                  {l.text}
                </text>
              ))}
            {hs && (
              <line x1={x(hs.index)} x2={x(hs.index)} y1={0} y2={ih} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1} strokeDasharray="3 3" />
            )}
            {/* hit areas */}
            {stages.map((s) => (
              <rect
                key={`hit${s.index}`}
                x={x(s.index) - iw / (nStage - 1) / 2}
                y={0}
                width={iw / (nStage - 1)}
                height={ih}
                fill="transparent"
                onPointerEnter={() => setHover(s.index)}
                onPointerLeave={() => setHover(null)}
              />
            ))}
          </g>
        </svg>
        {hs && (
          <TooltipBox
            style={{
              left: Math.min(W - 230, Math.max(0, M.left + x(hs.index) + 12)),
              top: 8,
            }}
          >
            <StageTip s={hs} />
          </TooltipBox>
        )}
      </div>
    </ChartFrame>
  );
}

function StageTip({ s }: { s: Stage }) {
  return (
    <div className="num grid gap-0.5 text-[14.5px]">
      <div className="font-sans text-[15px] font-semibold text-ink">
        {s.kind === 'embed' ? 'input embedding x⁽⁰⁾' : s.kind === 'out' ? 'after norm_out' : `block ${s.block + 1}, ${s.kind === 'ffn' ? 'SwiGLU' : SUB_SHORT[s.kind]}`}
      </div>
      <div className="text-ink-2">
        ‖x‖ = <span className="text-ink">{fx(s.xNorm, 3)}</span>
      </div>
      {Number.isFinite(s.hNorm) && (
        <div className="text-ink-2">
          ‖RMSNorm(x_in)‖ = <span className="text-ink">{fx(s.hNorm, 3)}</span>
        </div>
      )}
      {Number.isFinite(s.updNorm) && (
        <div className="text-ink-2">
          ‖Δ‖ = <span className="text-ink">{fx(s.updNorm, 3)}</span>
        </div>
      )}
    </div>
  );
}
