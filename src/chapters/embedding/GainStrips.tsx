/**
 * Step 3 figure: the learned gains γ of the five input RMSNorms (norm_dict.*.weight, model.py:132-140),
 * one dot per dimension. They start at 1 (PyTorch default, spec §7.4); the dots show where training
 * moved them in our mini model.
 */
import { scaleLinear } from 'd3';
import { useMemo, useState } from 'react';
import { Figure, SVG_TYPE, maxTextWidth } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { tensor } from '@/engine/model';
import type { RTWeights } from '@/engine/types';
import { fmt } from './live';
import { norm } from './math';

const NORMS = ['col_name', 'number', 'text', 'datetime', 'boolean'] as const;

export function GainStrips({ weights, kind }: { weights: RTWeights | null; kind: 'init' | 'trained' | null }) {
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<{ n: number; j: number } | null>(null);

  const data = useMemo(() => {
    if (!weights) return null;
    return NORMS.map((k) => {
      const g = tensor(weights, `norm_dict.${k}.weight`);
      let mn = Infinity;
      let mx = -Infinity;
      let mean = 0;
      for (const v of g) {
        mn = Math.min(mn, v);
        mx = Math.max(mx, v);
        mean += v / g.length;
      }
      return { key: k, g, mn, mx, mean, len: norm(g) };
    });
  }, [weights]);

  const W = Math.max(300, size.width || 700);
  const narrow = W < 520;
  const labels = NORMS.map((k) => (narrow ? k : `norm_dict["${k}"]`));
  const left = Math.ceil(maxTextWidth(labels, 14, { mono: true })) + 16;
  const right = 16;
  const rowH = 34;
  const top = 26;
  const H = top + NORMS.length * rowH + 40;
  const lo = data ? Math.min(1, ...data.map((r) => r.mn)) - 0.04 : 0.8;
  const hi = data ? Math.max(1, ...data.map((r) => r.mx)) + 0.04 : 1.2;
  const x = scaleLinear().domain([lo, hi]).nice().range([left, W - right]);
  const ticks = x.ticks(W < 500 ? 4 : 8);
  const hv = hover && data ? { row: data[hover.n], v: data[hover.n].g[hover.j] } : null;

  return (
    <Figure
      wide
      title="The five learned gains γ, one dot per dimension"
      caption={
        <>
          Each RMSNorm multiplies its output by its own γ ({weights?.config.dModel ?? 64} numbers), which starts at 1 in every dimension{' '}
          (the dashed line). {kind === 'trained' ? 'Training moved them in our mini model; this is the checkpoint the page runs.' : ''}
        </>
      }
    >
      <div ref={wrapRef} className="w-full">
        {!data ? (
          <div className="py-6 text-center text-[16px] text-ink-2">Loading the weights…</div>
        ) : (
          <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block max-w-full" role="img" aria-label="Learned RMSNorm gains of the five input norms">
            {ticks.map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={top - 6} y2={top + NORMS.length * rowH} style={{ stroke: 'var(--grid)' }} />
                <text x={x(t)} y={top + NORMS.length * rowH + 18} textAnchor="middle" fontSize={SVG_TYPE.tick} style={{ fill: 'var(--muted)' }}>
                  {t < 0 ? `−${Math.abs(t)}` : t}
                </text>
              </g>
            ))}
            <line x1={x(1)} x2={x(1)} y1={top - 10} y2={top + NORMS.length * rowH} style={{ stroke: 'var(--ink-2)' }} strokeDasharray="4 3" />
            <text x={x(1) + (x(1) > W - 110 ? -5 : 5)} y={top - 12} textAnchor={x(1) > W - 110 ? 'end' : 'start'} fontSize={SVG_TYPE.annotation} style={{ fill: 'var(--ink-2)' }}>
              start: γ = 1
            </text>
            <text x={(left + W - right) / 2} y={H - 4} textAnchor="middle" fontSize={SVG_TYPE.axisTitle} style={{ fill: 'var(--ink-2)' }}>
              γ value
            </text>
            {data.map((r, n) => {
              const y = top + n * rowH + rowH / 2;
              return (
                <g key={r.key}>
                  <text x={left - 10} y={y + 4.5} textAnchor="end" fontSize={14} className="font-mono" style={{ fill: 'var(--ink-2)' }}>
                    {labels[n]}
                  </text>
                  {Array.from(r.g).map((v, j) => {
                    const on = hover?.n === n && hover.j === j;
                    return (
                      <circle
                        key={j}
                        cx={x(v)}
                        cy={y + (((j * 37) % 11) / 10 - 0.5) * 14}
                        r={on ? 6 : 4}
                        style={{ fill: on ? 'var(--accent)' : 'var(--ink-2)', stroke: 'var(--surface)' }}
                        fillOpacity={on ? 1 : 0.5}
                        strokeWidth={1}
                        onPointerEnter={() => setHover({ n, j })}
                        onPointerLeave={() => setHover(null)}
                      />
                    );
                  })}
                </g>
              );
            })}
          </svg>
        )}
      </div>
      <div className="num mt-1 min-h-[1.4em] text-[15px] text-ink-2" aria-live="polite">
        {hv ? (
          <span className="font-mono">
            norm_dict["{hv.row.key}"].weight[{hover!.j}] = <strong className="text-ink">{fmt(hv.v, 3)}</strong>
          </span>
        ) : data ? (
          <>
            Range over all five: {fmt(Math.min(...data.map((r) => r.mn)), 2)} to {fmt(Math.max(...data.map((r) => r.mx)), 2)}. Hover a dot for its value.
          </>
        ) : null}
      </div>
    </Figure>
  );
}
