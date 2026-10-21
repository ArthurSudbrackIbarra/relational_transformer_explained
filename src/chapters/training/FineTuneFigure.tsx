/**
 * Fine-tuning curves, a paper Fig. 3 analogue (p.7): ShopDB test (or val) metric vs fine-tuning
 * steps + 1 on a log axis, from the pretrained checkpoint (solid) and from the same random init
 * ("untrained RT", dashed), constant lr 1e-4, wd 0, fresh AdamW (spec §9.2). The first point of each
 * curve is step 0, i.e. before any fine-tuning update (zero-shot for the pretrained run).
 * A third view shows the training loss, to make overfitting visible.
 */
import { useMemo, useState } from 'react';
import { Cite, Segmented } from '@/components';
import { TrainChart, type TSeries } from './charts';
import { evalSeries, fx, int, movingAverage, SHOP_FORECAST, type Curves } from './data';

type View = 'Test' | 'Val' | 'loss';

export function FineTuneFigure({ curves }: { curves: Curves | null }) {
  const [view, setView] = useState<View>('Test');
  const charts = useMemo(() => {
    if (!curves) return [];
    return SHOP_FORECAST.map((t) => {
      const pre = curves.runs[`finetune-${t.ft}-pretrained`];
      const unt = curves.runs[`finetune-${t.ft}-untrained`];
      let series: TSeries[];
      if (view === 'loss') {
        const ma = (r: typeof pre) => {
          if (!r) return [];
          const m = movingAverage(r.loss, 31);
          return r.step.map((s, i) => ({ x: s, y: m[i] }));
        };
        series = [
          { id: 'pre', label: 'from pretrained', slot: 1, points: ma(pre) },
          { id: 'unt', label: 'from random init', slot: 2, dashed: true, points: ma(unt) },
        ];
      } else {
        series = [
          { id: 'pre', label: 'from pretrained', slot: 1, points: evalSeries(pre, `${t.id}/${view}`), dots: true },
          { id: 'unt', label: 'from random init', slot: 2, dashed: true, points: evalSeries(unt, `${t.id}/${view}`), dots: true },
        ];
      }
      const bestP = pre?.best?.[t.id];
      const bestU = unt?.best?.[t.id];
      const preS = evalSeries(pre, `${t.id}/${view === 'loss' ? 'Test' : view}`);
      return { t, series, bestP, bestU, steps: pre?.maxSteps ?? pre?.step.length, preS };
    });
  }, [curves, view]);

  if (!curves || charts.length === 0) return <div className="wide my-8 card px-5 py-6 text-[16px] text-ink-2">Loading curves.json…</div>;

  return (
    <figure className="wide my-8 min-w-0">
      <div className="card px-4 py-4 sm:px-5 sm:py-5">
        <div className="mb-3 flex flex-wrap items-end gap-3">
          <div className="mr-auto text-[17px] font-semibold text-ink">Fine-tuning on ShopDB: pretrained vs random init</div>
          <Segmented
            label="Show"
            value={view}
            onChange={setView}
            options={[
              { value: 'Test', label: 'test metric' },
              { value: 'Val', label: 'val metric' },
              { value: 'loss', label: 'training loss' },
            ]}
          />
        </div>
        <div className="grid gap-6 lg:grid-cols-2">
          {charts.map(({ t, series, bestP, bestU, steps }) => (
            <TrainChart
              key={t.id}
              title={`${t.short}: ${view === 'loss' ? 'training loss (moving average, 31 steps)' : `${view.toLowerCase()} ${t.metricLabel}`}`}
              subtitle={view === 'loss' ? 'on the Train split the model is fitting' : 'first point = step 0, before any fine-tuning update'}
              series={series}
              xScale="log"
              xPlot={(s) => s + 1}
              xDomain={[1, (steps ?? 1025) + 1]}
              xTicks={[0, 9, 99, 999]}
              formatX={(s) => int(s + 1)}
              xLabel="fine-tuning steps + 1 (log)"
              yLabel={view === 'loss' ? 'loss' : t.metricLabel}
              formatY={(v) => fx(v, 2)}
              markers={view === 'loss' ? undefined : [{ x: 0, label: '0-shot' }]}
              highlights={
                view === 'loss'
                  ? undefined
                  : [
                      ...(bestP
                        ? [{ x: bestP.step, y: view === 'Val' ? bestP.val : bestP.test, slot: 1, label: `best val: step ${int(bestP.step)}` }]
                        : []),
                      ...(bestU ? [{ x: bestU.step, y: view === 'Val' ? bestU.val : bestU.test, slot: 2 }] : []),
                    ]
              }
              height={300}
            />
          ))}
        </div>
      </div>
      <figcaption className="mx-auto mt-3 max-w-[var(--prose-max)] text-[16px] leading-snug text-ink-2">
        Same axes as the paper’s Figure 3: fine-tuning steps + 1 on a log scale, the first point being step 0 <Cite kind="paper" ref="p.7 Fig. 3" />.
        Rings mark the step each run would keep (best val). Evaluations at step 0, at every power of two and every{' '}
        {int(curves.runs[`finetune-churn-pretrained`]?.hp?.eval_freq)} steps.
      </figcaption>
    </figure>
  );
}
