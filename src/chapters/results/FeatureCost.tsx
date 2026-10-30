/**
 * RT as a feature extractor (Table 3, p.8) and the cost-quality trade-off (Fig. 4, p.8-9).
 * Fig. 4 prints no table, so only its settings and the paper's stated summary are shown.
 */
import { useMemo, useState } from 'react';
import { Cite, Figure, Segmented } from '@/components';
import { useSim } from '@/state/store';
import { FIG4, TABLE3, taskLabel, type Cell } from './data';
import { DotPlot, type DotRow } from './DotPlot';
import { f1, MethodLegend, PaperTableView, TitleWithBadge, ChartTableFrame } from './kit';
import { METHODS, T3_METHOD } from './methods';

type Metric = 'clf' | 'reg';

export function FeatureExtraction() {
  const [metric, setMetric] = useState<Metric>('clf');
  const t = TABLE3[metric];
  const methods = t.cols.map((c) => T3_METHOD[c]);
  const rows: DotRow[] = useMemo(
    () => [
      ...t.rows.map((r) => ({
        id: r.task,
        label: taskLabel(r.task),
        values: Object.fromEntries(t.cols.map((c, j) => [T3_METHOD[c], r.v[j]])) as Record<string, Cell>,
      })),
      {
        id: 'mean',
        label: 'Mean (as printed)',
        mean: true,
        values: Object.fromEntries(t.cols.map((c, j) => [T3_METHOD[c], t.mean[j]])) as Record<string, Cell>,
      },
    ],
    [t],
  );
  const table = (
    <PaperTableView
      headCols={['Dataset', 'Task']}
      columns={t.cols.map((c) => ({
        id: c,
        label: METHODS[T3_METHOD[c]].label,
        group: c === 'fullFt' ? 'Full FT' : 'MLP-only FT, frozen embedder',
        method: T3_METHOD[c],
      }))}
      rows={[
        ...t.rows.map((r) => {
          const [db, task] = taskLabel(r.task).split(' · ');
          return { id: r.task, head: [db, task], cells: r.v };
        }),
        { id: 'mean', head: ['Mean', ''], cells: t.mean, mean: true },
      ]}
      markMax
      caption={<>{t.caption} The best entry in each row is underlined.</>}
    />
  );
  const iP = t.cols.indexOf('rtP');
  const iF = t.cols.indexOf('fullFt');
  return (
    <Figure
      wide
      title={<TitleWithBadge title="A 2-layer MLP on frozen RT embeddings" kind="paper" badge="Table 3 · p.8" />}
      controls={
        <Segmented
          label="Metric"
          value={metric}
          onChange={setMetric}
          options={[
            { value: 'clf', label: 'AUROC · 5 tasks' },
            { value: 'reg', label: 'R² · 4 tasks' },
          ]}
        />
      }
      caption={
        <>
          Means as printed: MLP on pretrained RT {f1(t.mean[iP] as number)} vs full fine-tuning {f1(t.mean[iF] as number)}. The paper summarizes this
          as “4.3% (abs.) better avg. R² and only 1.6% (abs.) worse avg. AUROC than full FT, despite being orders of magnitude cheaper”{' '}
          <Cite kind="paper" ref="p.8 §4.4" />.
        </>
      }
    >
      <ChartTableFrame title={<MethodLegend ids={methods} />} table={table}>
        <DotPlot
          rows={rows}
          methods={methods}
          domain={metric === 'clf' ? [60, 95] : [0, 60]}
          axisLabel={metric === 'clf' ? 'test AUROC (%)' : 'test R² (%)'}
          focus="rtP3"
          ariaLabel={`RT as a feature extractor, ${metric === 'clf' ? 'AUROC' : 'R²'} per task, paper Table 3`}
        />
      </ChartTableFrame>
    </Figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Fig. 4: what was varied, and what the paper states
// ─────────────────────────────────────────────────────────────────────────────

export function CostQuality() {
  const L = useSim((s) => s.L);
  const maxL = Math.max(...FIG4.rtContexts);
  const s = FIG4.stated;
  return (
    <Figure
      wide
      title={<TitleWithBadge title="Shorter contexts at inference: what Figure 4 varies" kind="paper" badge="Fig. 4 · p.8-9" />}
      caption={
        <>
          Both models were pretrained at their largest setting and evaluated zero-shot with smaller ones. The figure averages AUROC over the 10
          classification tasks and R² over the 8 regression tasks <Cite kind="paper" ref="p.9 Fig. 4" />. It prints no table, so this panel shows
          its settings and the paper’s summary. Your context length (L = {L}) is drawn for scale.
        </>
      }
    >
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <div className="text-[15.5px] font-semibold text-ink">RT: context length L at inference (pretrained at 1024)</div>
          <div className="mt-2 grid gap-1.5">
            {FIG4.rtContexts.map((c) => (
              <div key={c} className="flex items-center gap-3">
                <span className="num w-[52px] shrink-0 text-right text-[15px] text-ink">{c}</span>
                <div className="h-4 min-w-0 flex-1">
                  <div
                    className="h-full rounded-r-[4px]"
                    style={{ width: `${(100 * c) / maxL}%`, background: 'var(--cat-7)', opacity: c === maxL ? 1 : 0.55 + (0.45 * c) / maxL }}
                  />
                </div>
              </div>
            ))}
            <div className="flex items-center gap-3">
              <span className="num w-[52px] shrink-0 text-right text-[15px] text-accent-text">{L}</span>
              <div className="h-4 min-w-0 flex-1">
                <div
                  className="h-full rounded-r-[4px] border-2 border-dashed border-accent"
                  style={{ width: `${Math.min(100, (100 * L) / maxL)}%` }}
                />
              </div>
            </div>
            <div className="pl-[64px] text-[14.5px] text-muted">dashed: this site’s L (cells)</div>
          </div>
          <div className="mt-4 text-[15.5px] font-semibold text-ink">Griffin: fanout:hops (pretrained at 20:2, its default)</div>
          <div className="mt-2 flex flex-wrap gap-2">
            {FIG4.griffinFanoutHops.map((f) => (
              <span key={f} className="num rounded-md border border-border bg-surface-2 px-2.5 py-1 text-[15px] text-ink">
                {f}
              </span>
            ))}
          </div>
        </div>
        <div className="min-w-0 rounded-xl bg-surface-2 px-4 py-3">
          <div className="eyebrow">Stated result</div>
          <p className="mt-1 text-[17px] leading-snug text-ink">
            “RT’s performance degrades gracefully, showing a drop of only {s.aurocDrop}% absolute AUROC and {s.r2Drop}% absolute R² with {s.lessTime}%
            lower time and {s.lessMemory}% lower memory” <Cite kind="paper" ref="p.8 §4.5" />.
          </p>
          <div className="mt-3 grid grid-cols-2 gap-2">
            {[
              { v: `−${s.aurocDrop}`, l: 'AUROC points' },
              { v: `−${s.r2Drop}`, l: 'R² points' },
              { v: `−${s.lessTime}%`, l: 'inference time' },
              { v: `−${s.lessMemory}%`, l: 'memory' },
            ].map((x) => (
              <div key={x.l} className="rounded-lg border border-border bg-surface px-3 py-2">
                <div className="text-[24px] leading-tight font-semibold text-ink">{x.v}</div>
                <div className="text-[15px] text-ink-2">{x.l}</div>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[15.5px] leading-snug text-ink-2">
            RT also dominates Griffin here, even though RT attends over cells and Griffin works on rows. The paper gives three reasons: RT’s
            FlexAttention kernels are efficient, Griffin pays a per-cell cost when it encodes rows, and at equal parameter count and similar context
            sizes the two use similar memory <Cite kind="paper" ref="p.8 §4.5" />.
          </p>
        </div>
      </div>
    </Figure>
  );
}
