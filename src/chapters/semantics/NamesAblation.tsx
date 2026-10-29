/**
 * Figures for "Why names matter": the paper's shuffled-names ablation (Table 4 p.9, per task in
 * Table 11 p.22) and its column-name overlap matrix (Table 6 p.17), plus the same overlap statistic
 * computed live on this site's four toy databases.
 */
import { useMemo, useState } from 'react';
import { ChartFrame, DataTable, Segmented, cx } from '@/components';
import { divergingCss, sequentialCss } from '@/components/viz/scales';
import { textWidth } from '@/components/viz/text';
import { useElementSize } from '@/components/viz/useSize';
import { DB_NAMES } from '@/engine/db';
import { catVar } from '@/styles/palette';
import { useResolvedTheme } from '@/styles/theme';
import { DB_SHORT, TABLE11_CLF, TABLE11_MEANS, TABLE11_REG, TABLE4, TABLE6, f1, jaccard, signed1, dbColumnNames, type Catalog } from './data';

// ─────────────────────────────────────────────────────────────────────────────
// Table 4 summary tiles
// ─────────────────────────────────────────────────────────────────────────────

const T4_ROWS = [
  { key: 'none', label: 'none (full context)' },
  { key: 'colNames', label: 'column and table names shuffled' },
  { key: 'selfLabels', label: 'target entity’s past labels removed' },
  { key: 'otherLabels', label: 'other entities’ labels removed' },
] as const;
const T4_COLS = [
  { key: 'zsClf', setting: 'Zero-shot', metric: 'AUROC' },
  { key: 'zsReg', setting: 'Zero-shot', metric: 'R²' },
  { key: 'ftClf', setting: 'Fine-tuned', metric: 'AUROC' },
  { key: 'ftReg', setting: 'Fine-tuned', metric: 'R²' },
] as const;
type T4Col = (typeof T4_COLS)[number]['key'];

/**
 * Paper Table 4 (p.9), all four rows, with each change from "none" drawn as a bar on one shared scale,
 * so the names row can be compared with the label ablations. On narrow screens: one table per setting.
 */
export function Table4Grid({ narrow }: { narrow: boolean }) {
  const mode = useResolvedTheme();
  const maxAbs = Math.max(...T4_ROWS.flatMap((r) => T4_COLS.map((c) => Math.abs(TABLE4[r.key][c.key] - TABLE4.none[c.key]))));
  const groups: (typeof T4_COLS)[number][][] = narrow ? [T4_COLS.slice(0, 2), T4_COLS.slice(2)] : [[...T4_COLS]];
  const cellOf = (r: (typeof T4_ROWS)[number], c: T4Col) => {
    const v = TABLE4[r.key][c];
    const d = Math.round((v - TABLE4.none[c]) * 10) / 10;
    const isNone = r.key === 'none';
    return (
      <td key={c} className="num px-2 py-2 text-right align-top">
        <div className="text-[16px] font-semibold text-ink">{f1(v)}</div>
        {!isNone && (
          <div className="mt-0.5 flex items-center justify-end gap-1.5">
            <span className="relative inline-block h-2 w-[56px] rounded-full bg-surface-3" aria-hidden="true">
              <span
                className="absolute inset-y-0 rounded-full"
                style={{
                  width: `${Math.max(2, (50 * Math.abs(d)) / maxAbs)}%`,
                  [d < 0 ? 'right' : 'left']: '50%',
                  background: divergingCss(d, maxAbs, mode),
                }}
              />
              <span className="absolute inset-y-[-2px] left-1/2 w-px bg-axis" />
            </span>
            <span className={cx('w-[44px] text-[14.5px]', d < 0 ? 'text-ink' : 'text-ink-2')}>{d === 0 ? '0.0' : signed1(d)}</span>
          </div>
        )}
      </td>
    );
  };
  return (
    <div className={cx('grid gap-5', narrow ? '' : '')}>
      {groups.map((cols, gi) => (
        <div key={gi} className="scroll-quiet overflow-x-auto">
          <table className="w-full border-collapse text-[15px]">
            <caption className="sr-only">Paper Table 4: mean AUROC and R² (%) of each context ablation{narrow ? `, ${cols[0].setting.toLowerCase()}` : ''}</caption>
            <thead>
              {!narrow && (
                <tr className="text-ink-2">
                  <th />
                  <th scope="colgroup" colSpan={2} className="border-b border-grid pb-1 text-center font-semibold">
                    Zero-shot
                  </th>
                  <th scope="colgroup" colSpan={2} className="border-b border-grid pb-1 text-center font-semibold">
                    Fine-tuned
                  </th>
                </tr>
              )}
              <tr className="border-b border-grid text-ink-2">
                <th scope="col" className="py-1.5 pr-2 text-left font-semibold">
                  {narrow ? cols[0].setting : 'Ablation'}
                </th>
                {cols.map((c) => (
                  <th key={c.key} scope="col" className="px-2 py-1.5 text-right font-semibold">
                    {c.metric} (%)
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {T4_ROWS.map((r) => (
                <tr key={r.key} className={cx('border-b border-grid/70', r.key === 'colNames' && 'bg-accent-soft')}>
                  <th scope="row" className={cx('py-2 pr-2 text-left align-top font-normal leading-snug', r.key === 'colNames' ? 'font-semibold text-ink' : 'text-ink-2')}>
                    {r.label}
                  </th>
                  {cols.map((c) => cellOf(r, c.key))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      <p className="!m-0 text-[14.5px] leading-snug text-ink-2">
        Values from the paper. Under each value is its change from “none”, drawn on one scale for the whole table (the longest bar is{' '}
        {f1(maxAbs)} points), so you can compare the names row with the two label ablations.
      </p>
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
  const labelW = Math.ceil(Math.max(...rows.map((r) => textWidth(`${r.db.replace('rel-', '')} · ${r.task}`, 15)))) + 12;
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
  const all = [...rows.map((r) => ({ name: `${r.db} · ${r.task}`, a: r.a, b: r.b, d: r.d })), { name: 'Mean', a: means[0], b: means[1], d: Math.round((means[1] - means[0]) * 10) / 10 }];

  return (
    <ChartFrame
      title={`Per task: names intact vs shuffled, ${setting === 'zs' ? 'zero-shot' : 'fine-tuned'} ${unit}`}
      subtitle="Paper Table 11 (p.22), sorted by the change, so the tasks where shuffling hurts most come first. A ring around a dot means the two values are equal."
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
      table={<DataTable columns={['Task', 'names intact', 'names shuffled', 'change']} rows={all.map((r) => [r.name, f1(r.a), f1(r.b), signed1(r.d)])} caption={`Table 11 ${unit}`} />}
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
                  style={{ fontSize: 15, fill: 'var(--ink)', fontWeight: isMean ? 650 : 400 }}
                >
                  {isMean ? 'mean' : r.name.replace('rel-', '')}
                </text>
                <line x1={x(r.a)} x2={x(r.b)} y1={y + dotDy} y2={y + dotDy} style={{ stroke: 'var(--ink-2)' }} strokeWidth={2} />
                {Math.abs(r.a - r.b) < 0.05 ? (
                  // equal values: the ring sits around the dot, so both series stay visible
                  <>
                    <circle cx={x(r.b)} cy={y + dotDy} r={8} style={{ fill: 'var(--surface)', stroke: catVar(2) }} strokeWidth={2.4} />
                    <circle cx={x(r.a)} cy={y + dotDy} r={4.5} style={{ fill: catVar(1) }} />
                  </>
                ) : (
                  <>
                    <circle cx={x(r.a)} cy={y + dotDy} r={6} style={{ fill: catVar(1), stroke: 'var(--surface)' }} strokeWidth={2} />
                    <circle cx={x(r.b)} cy={y + dotDy} r={5.5} style={{ fill: 'var(--surface)', stroke: catVar(2) }} strokeWidth={2.4} />
                  </>
                )}
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
                <th key={l} scope="col" title={l} className="px-0 pb-1 text-center font-normal text-ink-2 sm:px-1">
                  <span className="inline-block align-bottom max-sm:rotate-180 max-sm:[writing-mode:vertical-rl]">{l}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {labels.map((r, i) => (
              <tr key={r}>
                <th scope="row" className="pr-2 text-right font-normal whitespace-nowrap text-ink-2">
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
                      className={cx('num h-9 min-w-[38px] rounded px-0 text-center sm:min-w-[46px] sm:px-1', hover && hover.i === i && hover.j === j && 'outline-2 outline-ink')}
                      style={{ background: bg, color: dark ? '#fff' : diag ? 'var(--muted)' : 'var(--ink)' }}
                    >
                      <span className="max-sm:hidden">{v.toFixed(3)}</span>
                      <span className="sm:hidden">{diag ? '1' : v.toFixed(3).replace(/^0(?=\.)/, '')}</span>
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
        note={(i, j) =>
          `${DB_SHORT[DB_NAMES[i]]} vs ${DB_SHORT[DB_NAMES[j]]}: ${ours[i][j].shared.length} shared / ${ours[i][j].union} distinct column names = ${ours[i][j].j.toFixed(3)}${ours[i][j].shared.length ? ` (${ours[i][j].shared.join(', ')})` : ''}`
        }
      />
    </div>
    <p className="mt-3 !mb-0 text-[15px] leading-snug text-ink-2">
      Largest overlap here: {DB_SHORT[DB_NAMES[best.i]]} vs {DB_SHORT[DB_NAMES[best.j]]}, {best.j_.toFixed(3)} ({ours[best.i][best.j].shared.length} shared of{' '}
      {ours[best.i][best.j].union} distinct names: <span className="text-ink">{ours[best.i][best.j].shared.join(', ') || 'none'}</span>),{' '}
      {best.j_ > paperMax ? 'higher than' : best.j_ < paperMax ? 'lower than' : 'equal to'} RelBench’s largest, {paperMax.toFixed(3)}.
    </p>
    </div>
  );
}
