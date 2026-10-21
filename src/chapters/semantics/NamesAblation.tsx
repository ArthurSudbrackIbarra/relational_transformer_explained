/**
 * Figures for "Why names matter": the paper's shuffled-names ablation (Table 4 p.9, per task in
 * Table 11 p.22) and its column-name overlap matrix (Table 6 p.17), plus the same overlap statistic
 * computed live on this site's four toy databases.
 */
import { useMemo, useState } from 'react';
import { ChartFrame, DataTable, Segmented, cx } from '@/components';
import { sequentialCss } from '@/components/viz/scales';
import { textWidth } from '@/components/viz/text';
import { useElementSize } from '@/components/viz/useSize';
import { DB_NAMES } from '@/engine/db';
import { catVar } from '@/styles/palette';
import { useResolvedTheme } from '@/styles/theme';
import { DB_SHORT, TABLE11_CLF, TABLE11_MEANS, TABLE11_REG, TABLE4, TABLE6, f1, jaccard, signed1, dbColumnNames, type Catalog } from './data';

// ─────────────────────────────────────────────────────────────────────────────
// Table 4 summary tiles
// ─────────────────────────────────────────────────────────────────────────────

export function Table4Tiles() {
  const tiles = [
    { label: 'Zero-shot AUROC', a: TABLE4.none.zsClf, b: TABLE4.colNames.zsClf, unit: '%' },
    { label: 'Zero-shot R²', a: TABLE4.none.zsReg, b: TABLE4.colNames.zsReg, unit: '%' },
    { label: 'Fine-tuned AUROC', a: TABLE4.none.ftClf, b: TABLE4.colNames.ftClf, unit: '%' },
    { label: 'Fine-tuned R²', a: TABLE4.none.ftReg, b: TABLE4.colNames.ftReg, unit: '%' },
  ];
  return (
    <div className="grid grid-cols-2 gap-2.5 md:grid-cols-4">
      {tiles.map((t) => {
        const d = Math.round((t.b - t.a) * 10) / 10;
        return (
          <div key={t.label} className="rounded-lg border border-border bg-surface px-3.5 py-3">
            <div className="text-[15px] font-medium text-ink-2">{t.label}</div>
            <div className="num mt-1 flex items-baseline gap-2 text-[22px] font-semibold text-ink">
              <span>{f1(t.a)}</span>
              <span className="text-[16px] font-normal text-muted">→</span>
              <span>{f1(t.b)}</span>
            </div>
            <div className="num mt-0.5 text-[15px] text-ink-2">{d === 0 ? 'no change' : `${signed1(d)} points`}</div>
          </div>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Table 11 per-task dumbbells
// ─────────────────────────────────────────────────────────────────────────────

type Metric = 'clf' | 'reg';
type Setting = 'zs' | 'ft';

export function AblationDumbbells() {
  const [metric, setMetric] = useState<Metric>('clf');
  const [setting, setSetting] = useState<Setting>('zs');
  const [hover, setHover] = useState<number | null>(null);
  const [ref, size] = useElementSize<HTMLDivElement>();

  const rows = useMemo(() => {
    const src = metric === 'clf' ? TABLE11_CLF : TABLE11_REG;
    return src
      .map((r) => ({ ...r, a: r[setting][0], b: r[setting][1], d: Math.round((r[setting][1] - r[setting][0]) * 10) / 10 }))
      .sort((x, y) => x.d - y.d);
  }, [metric, setting]);
  const means = TABLE11_MEANS[metric][setting];

  const W = Math.max(320, size.width || 760);
  const labelW = Math.ceil(Math.max(...rows.map((r) => textWidth(`${r.db.replace('rel-', '')} · ${r.task}`, 15, { mono: true })))) + 12;
  // narrow screens: the task name sits above its dumbbell instead of left of it
  const stacked = W - labelW - 64 < 300;
  const deltaW = 64;
  const plotL = stacked ? 4 : labelW;
  const plotR = W - deltaW;
  const rowH = stacked ? 50 : 34;
  const dotDy = stacked ? 9 : 0;
  const top = 34;
  const H = top + (rows.length + 1) * rowH + 40;
  const lo = Math.floor(Math.min(...rows.map((r) => Math.min(r.a, r.b)), means[0], means[1]) / 5) * 5;
  const hi = Math.ceil(Math.max(...rows.map((r) => Math.max(r.a, r.b)), means[0], means[1]) / 5) * 5;
  const x = (v: number) => plotL + ((v - lo) / (hi - lo || 1)) * (plotR - plotL - 12) + 6;
  const ticks: number[] = [];
  const span = hi - lo || 1;
  const step = [5, 10, 20, 25, 50].find((s) => ((plotR - plotL) / span) * s >= 44) ?? 50;
  for (let v = lo; v <= hi + 1e-9; v += step) ticks.push(v);
  const unit = metric === 'clf' ? 'AUROC (%)' : 'R² (%)';
  const all = [...rows.map((r) => ({ name: `${r.db} · ${r.task}`, a: r.a, b: r.b, d: r.d })), { name: 'Mean (as printed)', a: means[0], b: means[1], d: Math.round((means[1] - means[0]) * 10) / 10 }];

  return (
    <ChartFrame
      title={`Per task: names intact vs shuffled, ${setting === 'zs' ? 'zero-shot' : 'fine-tuned'} ${unit}`}
      subtitle="Paper Table 11 (p.22), sorted by the change. Shuffling hurts most at the top."
      legend={
        <ul className="!m-0 flex !list-none flex-wrap items-center gap-x-5 gap-y-1 !p-0 text-[15px] text-ink-2">
          <li className="!m-0 inline-flex items-center gap-1.5 !p-0">
            <svg width="14" height="14" aria-hidden="true">
              <circle cx="7" cy="7" r="5.5" style={{ fill: catVar(1) }} />
            </svg>
            names intact (“none”)
          </li>
          <li className="!m-0 inline-flex items-center gap-1.5 !p-0">
            <svg width="14" height="14" aria-hidden="true">
              <circle cx="7" cy="7" r="5" style={{ fill: 'var(--surface)', stroke: catVar(2) }} strokeWidth={2.2} />
            </svg>
            column and table names shuffled
          </li>
        </ul>
      }
      table={<DataTable columns={['Task', 'none', 'col names', 'change']} rows={all.map((r) => [r.name, f1(r.a), f1(r.b), signed1(r.d)])} caption={`Table 11 ${unit}`} />}
    >
      <div className="mb-3 flex flex-wrap gap-3">
        <Segmented<Metric>
          label="Metric"
          value={metric}
          onChange={setMetric}
          options={[
            { value: 'clf', label: 'AUROC · 10 binary tasks' },
            { value: 'reg', label: 'R² · 8 regression tasks' },
          ]}
        />
        <Segmented<Setting>
          label="Setting"
          value={setting}
          onChange={setSetting}
          options={[
            { value: 'zs', label: 'Zero-shot' },
            { value: 'ft', label: 'Fine-tuned' },
          ]}
        />
      </div>
      <div ref={ref} className="w-full">
        <svg width={W} height={H} role="img" aria-label={`Dumbbell chart of ${unit}, names intact vs shuffled, per task`} className="block">
          {ticks.map((v) => (
            <g key={v}>
              <line x1={x(v)} x2={x(v)} y1={top - 6} y2={H - 26} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
              <text x={x(v)} y={H - 8} textAnchor="middle" className="num" style={{ fontSize: 13.5, fill: 'var(--muted)' }}>
                {v}
              </text>
            </g>
          ))}
          <text x={plotR + deltaW - 4} y={top - 14} textAnchor="end" style={{ fontSize: 13.5, fill: 'var(--muted)' }}>
            change
          </text>
          {all.map((r, i) => {
            const y = top + i * rowH + (i === rows.length ? 10 : 0) + rowH / 2;
            const isMean = i === rows.length;
            const hot = hover === i;
            return (
              <g key={r.name} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
                <rect x={0} y={y - rowH / 2} width={W} height={rowH} style={{ fill: hot ? 'var(--surface-2)' : 'transparent' }} />
                {isMean && <line x1={0} x2={W} y1={y - rowH / 2 - 4} y2={y - rowH / 2 - 4} style={{ stroke: 'var(--axis)' }} strokeWidth={1} />}
                <text
                  x={0}
                  y={stacked ? y - 11 : y}
                  dominantBaseline="central"
                  className="font-mono"
                  style={{ fontSize: 15, fill: 'var(--ink)', fontWeight: isMean ? 650 : 400 }}
                >
                  {isMean ? 'mean' : r.name.replace('rel-', '')}
                </text>
                <line x1={x(r.a)} x2={x(r.b)} y1={y + dotDy} y2={y + dotDy} style={{ stroke: 'var(--ink-2)' }} strokeWidth={2} />
                <circle cx={x(r.a)} cy={y + dotDy} r={6} style={{ fill: catVar(1), stroke: 'var(--surface)' }} strokeWidth={2} />
                <circle cx={x(r.b)} cy={y + dotDy} r={5.5} style={{ fill: 'var(--surface)', stroke: catVar(2) }} strokeWidth={2.4} />
                <text
                  x={plotR + deltaW - 4}
                  y={y + dotDy}
                  dominantBaseline="central"
                  textAnchor="end"
                  className="num"
                  style={{ fontSize: 14.5, fill: 'var(--ink)', fontWeight: r.d < 0 ? 600 : 400 }}
                >
                  {signed1(r.d)}
                </text>
                {hot && (
                  <text
                    x={stacked ? plotR - 4 : Math.min(x(Math.max(r.a, r.b)) + 12, plotR - 110)}
                    y={stacked ? y - 11 : y - 11}
                    dominantBaseline={stacked ? 'central' : undefined}
                    textAnchor={stacked ? 'end' : 'start'}
                    className="num"
                    style={{ fontSize: 13.5, fill: 'var(--ink-2)' }}
                  >
                    {f1(r.a)} → {f1(r.b)}
                  </text>
                )}
              </g>
            );
          })}
          <text x={(plotL + plotR) / 2} y={top - 14} textAnchor="middle" style={{ fontSize: 14, fill: 'var(--ink-2)' }}>
            {unit}
          </text>
        </svg>
      </div>
    </ChartFrame>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Column-name overlap: paper Table 6 and the same statistic on our toy DBs
// ─────────────────────────────────────────────────────────────────────────────

function OverlapGrid({ labels, value, title, note }: { labels: string[]; value: (i: number, j: number) => number; title: string; note?: (i: number, j: number) => string }) {
  const mode = useResolvedTheme();
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  return (
    <div className="min-w-0">
      <div className="mb-2 text-[16px] font-semibold text-ink">{title}</div>
      <div className="scroll-quiet overflow-x-auto">
        <table className="border-separate border-spacing-[2px] text-[14.5px]" onPointerLeave={() => setHover(null)}>
          <thead>
            <tr>
              <th />
              {labels.map((l) => (
                <th key={l} scope="col" className="px-1 pb-1 text-center font-mono font-normal text-ink-2">
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {labels.map((r, i) => (
              <tr key={r}>
                <th scope="row" className="pr-2 text-right font-mono font-normal whitespace-nowrap text-ink-2">
                  {r}
                </th>
                {labels.map((c, j) => {
                  const v = value(i, j);
                  const diag = i === j;
                  const t = diag ? 0 : Math.min(1, v / 0.25);
                  const bg = diag || v === 0 ? 'var(--surface-2)' : sequentialCss(0.15 + 0.85 * t, mode);
                  const dark = !diag && v > 0 && 0.15 + 0.85 * t > 0.55;
                  return (
                    <td
                      key={c}
                      onPointerEnter={() => setHover({ i, j })}
                      title={note ? note(i, j) : undefined}
                      className={cx('num h-9 min-w-[46px] rounded px-1 text-center', hover && hover.i === i && hover.j === j && 'outline-2 outline-ink')}
                      style={{ background: bg, color: dark ? '#fff' : diag ? 'var(--muted)' : 'var(--ink)' }}
                    >
                      {v.toFixed(3)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-1.5 min-h-[1.4em] text-[14.5px] text-ink-2" aria-live="polite">
        {hover && hover.i !== hover.j && note ? note(hover.i, hover.j) : ' '}
      </div>
    </div>
  );
}

export function OverlapTables({ catalog }: { catalog: Catalog }) {
  const ours = useMemo(() => {
    const sets = DB_NAMES.map((d) => dbColumnNames(catalog.dbs[d]));
    return DB_NAMES.map((_, i) => DB_NAMES.map((__, j) => jaccard(sets[i], sets[j])));
  }, [catalog]);
  const best = useMemo(() => {
    let b = { i: 0, j: 1, j_: -1 };
    for (let i = 0; i < DB_NAMES.length; i++) for (let j = i + 1; j < DB_NAMES.length; j++) if (ours[i][j].j > b.j_) b = { i, j, j_: ours[i][j].j };
    return b;
  }, [ours]);
  const paperMax = Math.max(...TABLE6.m.flatMap((row, i) => row.filter((_, j) => j !== i)));
  return (
    <div>
    <div className="grid gap-6 xl:grid-cols-2">
      <OverlapGrid
        title="RelBench (paper Table 6, p.17)"
        labels={TABLE6.dbs.map((d) => d.replace('rel-', ''))}
        value={(i, j) => TABLE6.m[i][j]}
        note={(i, j) => `${TABLE6.dbs[i]} vs ${TABLE6.dbs[j]}: Jaccard ${TABLE6.m[i][j].toFixed(3)} (exact column-name matches)`}
      />
      <OverlapGrid
        title="This site’s toy databases (computed live)"
        labels={DB_NAMES.map((d) => DB_SHORT[d].replace('DB', ''))}
        value={(i, j) => ours[i][j].j}
        note={(i, j) => `${DB_SHORT[DB_NAMES[i]]} vs ${DB_SHORT[DB_NAMES[j]]}: shared ${ours[i][j].shared.length ? ours[i][j].shared.join(', ') : 'none'}`}
      />
    </div>
    <p className="mt-3 !mb-0 text-[15px] leading-snug text-ink-2">
      Largest overlap here: {DB_SHORT[DB_NAMES[best.i]]} vs {DB_SHORT[DB_NAMES[best.j]]}, {best.j_.toFixed(3)} (shared:{' '}
      <span className="font-mono text-ink">{ours[best.i][best.j].shared.join(', ') || 'none'}</span>), against {paperMax.toFixed(3)} in RelBench.
    </p>
    </div>
  );
}
