/**
 * Section 5: how many (query, key) pairs each mask allows in the live context, plus keys per query.
 */
import { useMemo } from 'react';
import { BarChart, Figure, LayerLabel } from '@/components';
import { formatNumber } from '@/components';
import { int, maskStats, pct } from './maskUtils';
import type { LiveMasks } from './useLiveMasks';

const NAME = { col: 'Column', feat: 'Feature', nbr: 'Neighbor', full: 'Full' } as const;

export function MaskCounts({ live }: { live: LiveMasks }) {
  const { ctx, masks, S } = live;
  const stats = useMemo(() => maskStats(masks, ctx.tokens), [masks, ctx]);
  const total = S * S;
  return (
    <Figure
      wide
      title="Allowed pairs per mask in the live context"
      caption={
        <>
          Bars: the share of all {int(total)} ordered pairs (S = {S}) that each mask allows, on a fixed 0-100% axis; the count of allowed pairs
          follows each mask’s name. Keys per query are over the {ctx.numReal} non-padding queries. The neighbor mask is the only one with empty rows.
        </>
      }
    >
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <BarChart
          title="Allowed (query, key) pairs"
          subtitle={`out of S² = ${int(total)} ordered pairs`}
          categories={stats.map((s) => `${NAME[s.layer]} · ${int(s.pairs)}`)}
          series={[{ id: 'pairs', label: 'share of S²', slot: 1, values: stats.map((s) => s.pairs / total) }]}
          format={(v) => (Math.abs(v * 100 - Math.round(v * 100)) < 1e-9 ? `${Math.round(v * 100)}%` : pct(v))}
          domain={[0, 1]}
          labelWidth={92}
          valueLabel="share of all S² pairs"
        />
        <div className="min-w-0 overflow-x-auto">
          <table className="w-full border-collapse text-[15.5px]">
            <caption className="mb-2 text-left text-[16.5px] font-semibold text-ink">Density and keys per query</caption>
            <thead>
              <tr className="border-b border-grid text-left text-[14.5px] text-ink-2">
                <th scope="col" className="py-1.5 pr-3 font-medium">
                  Mask
                </th>
                <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                  share of S²
                </th>
                <th scope="col" className="hidden py-1.5 pr-3 text-right font-medium sm:table-cell">
                  min
                </th>
                <th scope="col" className="hidden py-1.5 pr-3 text-right font-medium sm:table-cell">
                  median
                </th>
                <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                  mean
                </th>
                <th scope="col" className="hidden py-1.5 pr-3 text-right font-medium sm:table-cell">
                  max
                </th>
                <th scope="col" className="py-1.5 text-right font-medium">
                  no key
                </th>
              </tr>
            </thead>
            <tbody>
              {stats.map((s) => (
                <tr key={s.layer} className="border-b border-grid/70 last:border-0">
                  <th scope="row" className="py-1.5 pr-3 text-left font-normal">
                    <LayerLabel layer={s.layer} size={13} className="font-normal" />
                  </th>
                  <td className="num py-1.5 pr-3 text-right">{pct(s.density)}</td>
                  <td className="num hidden py-1.5 pr-3 text-right sm:table-cell">{int(s.min)}</td>
                  <td className="num hidden py-1.5 pr-3 text-right sm:table-cell">{formatNumber(s.median, 1)}</td>
                  <td className="num py-1.5 pr-3 text-right">{formatNumber(s.mean, 1)}</td>
                  <td className="num hidden py-1.5 pr-3 text-right sm:table-cell">{int(s.max)}</td>
                  <td className={s.empty > 0 ? 'num py-1.5 text-right font-semibold text-ink' : 'num py-1.5 text-right text-muted'}>{int(s.empty)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-[15px] leading-snug text-ink-2">
            “no key” counts non-padding queries whose mask row is all zero. Their attention output is exactly 0, so the residual stream passes
            through that sublayer unchanged.
          </p>
        </div>
      </div>
    </Figure>
  );
}
