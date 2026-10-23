/**
 * Ablation explorer (Tables 4/11, 5/13, 14, 15): a summary matrix of the printed means, linked to a
 * per-task dumbbell chart. Click a cell of the matrix (or use the selectors) to see, task by task,
 * how far each value moves from the reference column. Moves animate, so the direction of every change
 * is visible as motion. Shading follows the paper's: proportional to the difference from the reference.
 */
import { motion, useReducedMotion } from 'motion/react';
import { useMemo, useState, type ReactNode } from 'react';
import { Cite, cx, Segmented } from '@/components';
import { textWidth } from '@/components/viz/text';
import { useElementSize } from '@/components/viz/useSize';
import { taskLabel, type AblationTable, type TaskKey } from './data';
import { d1, f1, PaperTableView, Tip, xScale, ChartTableFrame } from './kit';

export type Setting = 'zs' | 'ft';
export type Metric = 'clf' | 'reg';

export interface Variant {
  /** Column id in the table. */
  id: string;
  label: string;
  /** Short label for the matrix / headers. */
  short: string;
}

export interface AblationSpec {
  table: AblationTable;
  /** Variants in display order; variants[0] is the reference column. */
  variants: Variant[];
  settings: Setting[];
  /** Source note per metric, e.g. { clf: 'Table 13 (p.23)', reg: 'Table 5 (p.10)' }. */
  source: Record<Metric, { label: string; page: number }>;
  /** Short name of the reference for prose ("none", "P(mask) = 0.0", "CFN"). */
  refName: string;
  /** Verb phrase for a variant ("removing self labels"). */
  describe: (v: Variant) => string;
}

const SETTING_LABEL: Record<Setting, string> = { zs: 'Zero-shot', ft: 'Fine-tuned' };
const METRIC_LABEL: Record<Metric, string> = { clf: 'AUROC', reg: 'R²' };

/** Values of a variant: per-task and mean, or null if that block has no such column. */
export function variantValues(spec: AblationSpec, v: Variant, setting: Setting, metric: Metric) {
  const part = spec.table[metric];
  if (!part) return null;
  const j = spec.table.cols.indexOf(v.id);
  const meanArr = part.mean[setting];
  if (j < 0 || !meanArr || j >= meanArr.length) return null;
  return {
    rows: part.rows.map((r) => ({ task: r.task, v: (r[setting] ?? [])[j] })),
    mean: meanArr[j],
  };
}

/** Shading for a difference (variant − reference): drops in red, gains in blue, like the paper's tables. */
export function shade(delta: number, maxAbs: number): string | undefined {
  if (!Number.isFinite(delta) || Math.abs(delta) < 0.05) return undefined;
  const a = Math.min(1, Math.abs(delta) / maxAbs);
  const pct = Math.round(8 + a * 42);
  return `color-mix(in srgb, var(${delta < 0 ? '--div-pos' : '--div-neg'}) ${pct}%, transparent)`;
}

// ─────────────────────────────────────────────────────────────────────────────
// The explorer
// ─────────────────────────────────────────────────────────────────────────────

export function AblationExplorer({
  spec,
  initial,
  intro,
}: {
  spec: AblationSpec;
  initial: { variant: string; setting: Setting; metric: Metric };
  intro?: ReactNode;
}) {
  const [sel, setSel] = useState(initial);
  const [sort, setSort] = useState<'paper' | 'delta'>('paper');
  const ref = spec.variants[0];
  const variant = spec.variants.find((v) => v.id === sel.variant) ?? spec.variants[1];

  // which (setting, metric) combos exist
  const cols: { setting: Setting; metric: Metric }[] = [];
  for (const s of spec.settings) for (const m of ['clf', 'reg'] as Metric[]) if (spec.table[m]?.mean[s]) cols.push({ setting: s, metric: m });

  // matrix shading scale (shared by every cell)
  const maxAbs = useMemo(() => {
    // floor of 5 points, so sub-point differences stay faint (the paper's shading is proportional too)
    let m = 5;
    for (const v of spec.variants.slice(1))
      for (const c of cols) {
        const a = variantValues(spec, v, c.setting, c.metric);
        const r = variantValues(spec, ref, c.setting, c.metric);
        if (a && r) m = Math.max(m, Math.abs(a.mean - r.mean));
      }
    return m;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec]);

  const available = variantValues(spec, variant, sel.setting, sel.metric) !== null;

  return (
    <div className="min-w-0">
      {intro}
      {/* summary matrix */}
      <div className="scroll-quiet overflow-x-auto">
        <table className="w-full border-collapse text-[15px] sm:min-w-[460px]">
          <caption className="sr-only">Printed means; click a cell to see that ablation task by task</caption>
          <thead>
            <tr className="text-ink-2">
              <th scope="col" className="sticky left-0 z-[1] bg-surface px-2 py-1.5 text-left text-[14px] font-semibold">
                {spec.refName === 'none' ? 'Ablated' : 'Variant'}
              </th>
              {cols.map((c) => (
                <th key={c.setting + c.metric} scope="col" className="px-2 py-1.5 text-right text-[14px] leading-tight font-semibold">
                  {SETTING_LABEL[c.setting]}
                  <br className="sm:hidden" /> {METRIC_LABEL[c.metric]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {spec.variants.map((v, vi) => (
              <tr key={v.id} className={cx('border-t border-grid', vi === 0 && 'border-t-axis')}>
                <th scope="row" className="sticky left-0 z-[1] bg-surface px-2 py-1 text-left font-normal text-ink sm:whitespace-nowrap">
                  <span className="hidden sm:inline">{v.label}</span>
                  <span className="sm:hidden">{v.short}</span>
                  {vi === 0 && <span className="ml-1.5 text-[14px] text-muted">reference</span>}
                </th>
                {cols.map((c) => {
                  const a = variantValues(spec, v, c.setting, c.metric);
                  const r = variantValues(spec, ref, c.setting, c.metric);
                  if (!a || !r) {
                    return (
                      <td key={c.setting + c.metric} className="px-2 py-1 text-right text-muted">
                        —
                      </td>
                    );
                  }
                  const delta = a.mean - r.mean;
                  const selected = vi > 0 && v.id === sel.variant && c.setting === sel.setting && c.metric === sel.metric;
                  const content = (
                    <span className="flex flex-col items-end leading-tight">
                      <span className={cx('num text-[15.5px]', vi > 0 ? 'font-semibold text-ink' : 'text-ink')}>{f1(a.mean)}</span>
                      {vi > 0 && <span className="num text-[13.5px] text-ink-2">{d1(delta)}</span>}
                    </span>
                  );
                  return (
                    <td key={c.setting + c.metric} className="p-0.5 text-right">
                      {vi === 0 ? (
                        <div className="px-2 py-1">{content}</div>
                      ) : (
                        <button
                          type="button"
                          aria-pressed={selected}
                          aria-label={`${v.label}, ${SETTING_LABEL[c.setting]} ${METRIC_LABEL[c.metric]}: ${f1(a.mean)} (${d1(delta)} vs ${spec.refName}). Show per task.`}
                          onClick={() => setSel({ variant: v.id, setting: c.setting, metric: c.metric })}
                          className={cx(
                            'w-full rounded-md px-2 py-1 transition-shadow hover:shadow-[inset_0_0_0_1.5px_var(--border-strong)]',
                            selected && '!shadow-[inset_0_0_0_2px_var(--accent)]',
                          )}
                          style={{ background: shade(delta, maxAbs) }}
                        >
                          {content}
                        </button>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[14.5px] text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3 w-5 rounded-[3px]" style={{ background: shade(-maxAbs * 0.6, maxAbs) }} />
          lower than {spec.refName}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3 w-5 rounded-[3px]" style={{ background: shade(maxAbs * 0.6, maxAbs) }} />
          higher than {spec.refName}
        </span>
        <span>Means as printed; click a cell to see it task by task.</span>
      </div>

      {/* selectors */}
      <div className="mt-5 flex flex-wrap items-end gap-x-4 gap-y-3">
        <Segmented
          label="Variant"
          value={variant.id}
          onChange={(id) => setSel((s) => ({ ...s, variant: id }))}
          options={spec.variants.slice(1).map((v) => ({ value: v.id, label: v.short }))}
        />
        {spec.settings.length > 1 && (
          <Segmented
            label="Setting"
            value={sel.setting}
            onChange={(s) => setSel((p) => ({ ...p, setting: s }))}
            options={spec.settings.map((s) => ({ value: s, label: SETTING_LABEL[s] }))}
          />
        )}
        <Segmented
          label="Metric"
          value={sel.metric}
          onChange={(m) => setSel((p) => ({ ...p, metric: m }))}
          options={(['clf', 'reg'] as Metric[])
            .filter((m) => spec.table[m])
            .map((m) => ({ value: m, label: m === 'clf' ? 'AUROC · 10 tasks' : 'R² · 8 tasks' }))}
        />
        <Segmented
          label="Row order"
          value={sort}
          onChange={setSort}
          options={[
            { value: 'paper', label: 'Paper order' },
            { value: 'delta', label: 'By change' },
          ]}
        />
      </div>

      <div className="mt-4">
        {available ? (
          <Dumbbells spec={spec} variant={variant} setting={sel.setting} metric={sel.metric} sort={sort} />
        ) : (
          <p className="rounded-lg border border-dashed border-border-strong px-4 py-3 text-[16px] text-ink-2">
            The paper has no “{variant.label}” column for the {SETTING_LABEL[sel.setting].toLowerCase()} setting.
          </p>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-task dumbbells
// ─────────────────────────────────────────────────────────────────────────────

function Dumbbells({
  spec,
  variant,
  setting,
  metric,
  sort,
}: {
  spec: AblationSpec;
  variant: Variant;
  setting: Setting;
  metric: Metric;
  sort: 'paper' | 'delta';
}) {
  const [box, size] = useElementSize<HTMLDivElement>();
  const reduce = useReducedMotion();
  const [hover, setHover] = useState<string | null>(null);
  const ref = spec.variants[0];
  const a = variantValues(spec, variant, setting, metric)!;
  const r = variantValues(spec, ref, setting, metric)!;

  type Row = { id: string; label: string; ref: number; v: number; d: number; mean?: boolean };
  const body: Row[] = a.rows.map((x, i) => ({ id: x.task, label: taskLabel(x.task as TaskKey), ref: r.rows[i].v, v: x.v, d: x.v - r.rows[i].v }));
  if (sort === 'delta') body.sort((p, q) => p.d - q.d);
  const rows: Row[] = [...body, { id: 'mean', label: 'Mean (as printed)', ref: r.mean, v: a.mean, d: a.mean - r.mean, mean: true }];

  const W = Math.max(300, size.width || 720);
  const labelW = Math.ceil(Math.max(...rows.map((x) => textWidth(x.label, 15, { weight: x.mean ? 600 : 400 })))) + 16;
  const deltaW = 70;
  const stacked = W - labelW - deltaW < 290;
  const rowH = stacked ? 56 : 34;
  const plotL = stacked ? 16 : labelW + 10;
  const plotR = W - deltaW - 14;
  const top = 30;
  const meanGap = 14;
  const plotH = rows.length * rowH + meanGap;
  const H = top + plotH + 46;
  const all = rows.flatMap((x) => [x.ref, x.v]);
  const lo = Math.floor(Math.min(...all) / 5) * 5;
  const hi = Math.ceil(Math.max(...all) / 5) * 5;
  const { x, ticks } = xScale([lo, hi === lo ? lo + 5 : hi], plotL, plotR, W < 520 ? 54 : 70);
  const yOf = (i: number) => top + i * rowH + (rows[i].mean ? meanGap : 0);
  const markY = stacked ? rowH - 17 : rowH / 2;
  const tr = reduce ? { duration: 0 } : { type: 'spring' as const, stiffness: 160, damping: 24 };
  const down = rows.slice(0, -1).filter((x) => x.d < -0.05).length;
  const up = rows.slice(0, -1).filter((x) => x.d > 0.05).length;
  const src = spec.source[metric];
  const unit = METRIC_LABEL[metric];

  const table = (
    <PaperTableView
      headCols={['Dataset', 'Task']}
      columns={[
        ...spec.variants
          .map((v, j) => ({ v, j }))
          .filter(({ v }) => variantValues(spec, v, 'zs', metric))
          .map(({ v }) => ({ id: 'zs' + v.id, label: v.short, group: spec.settings.length > 1 ? 'Zero-shot' : undefined })),
        ...(spec.settings.includes('ft')
          ? spec.variants.filter((v) => variantValues(spec, v, 'ft', metric)).map((v) => ({ id: 'ft' + v.id, label: v.short, group: 'Fine-tuned' }))
          : []),
      ]}
      rows={[
        ...(spec.table[metric]?.rows ?? []).map((row) => {
          const zsRef = row.zs[0];
          const ftRef = row.ft?.[0];
          const zsCells = spec.variants.filter((v) => variantValues(spec, v, 'zs', metric)).map((v) => row.zs[spec.table.cols.indexOf(v.id)]);
          const ftCells = spec.settings.includes('ft')
            ? spec.variants.filter((v) => variantValues(spec, v, 'ft', metric)).map((v) => row.ft![spec.table.cols.indexOf(v.id)])
            : [];
          const [db, task] = taskLabel(row.task).split(' · ');
          return {
            id: row.task,
            head: [db, task],
            cells: [...zsCells, ...ftCells],
            shade: [
              ...zsCells.map((c, k) => (k === 0 ? undefined : shade(c - zsRef, 15))),
              ...ftCells.map((c, k) => (k === 0 || ftRef === undefined ? undefined : shade(c - ftRef, 15))),
            ],
          };
        }),
        (() => {
          const m = spec.table[metric]!.mean;
          const zsCells = spec.variants.filter((v) => variantValues(spec, v, 'zs', metric)).map((v) => m.zs[spec.table.cols.indexOf(v.id)]);
          const ftCells =
            spec.settings.includes('ft') && m.ft
              ? spec.variants.filter((v) => variantValues(spec, v, 'ft', metric)).map((v) => m.ft![spec.table.cols.indexOf(v.id)])
              : [];
          return { id: 'mean', head: ['Mean', ''], cells: [...zsCells, ...ftCells], mean: true };
        })(),
      ]}
      caption={
        <>
          {spec.table.caption} Shading (ours, like the paper’s): difference from the {spec.refName} column of the same block. Source: {src.label}.
        </>
      }
    />
  );

  return (
    <ChartTableFrame
      title={
        <>
          {SETTING_LABEL[setting]} {unit}: {spec.describe(variant)}
        </>
      }
      subtitle={
        <>
          Mean {f1(r.mean)} → {f1(a.mean)} ({d1(a.mean - r.mean)}); lower on {down} and higher on {up} of {rows.length - 1} tasks. {src.label}{' '}
          <Cite kind="paper" ref={`p.${src.page}`} />
        </>
      }
      legend={
        <ul className="!m-0 flex !list-none flex-wrap items-center gap-x-5 gap-y-1 !p-0 text-[15px] text-ink-2">
          <li className="!m-0 inline-flex items-center gap-1.5 !p-0">
            <svg width="14" height="14" aria-hidden="true">
              <circle cx="7" cy="7" r="5" style={{ fill: 'var(--surface)', stroke: 'var(--ink-2)' }} strokeWidth={2} />
            </svg>
            {ref.short} (reference)
          </li>
          <li className="!m-0 inline-flex items-center gap-1.5 !p-0">
            <svg width="14" height="14" aria-hidden="true">
              <circle cx="7" cy="7" r="6" style={{ fill: 'var(--div-pos)' }} />
            </svg>
            {variant.short}: lower
          </li>
          <li className="!m-0 inline-flex items-center gap-1.5 !p-0">
            <svg width="14" height="14" aria-hidden="true">
              <circle cx="7" cy="7" r="6" style={{ fill: 'var(--div-neg)' }} />
            </svg>
            {variant.short}: higher
          </li>
        </ul>
      }
      table={table}
    >
      <div ref={box} className="relative w-full" onPointerLeave={() => setHover(null)}>
        <svg width={W} height={H} role="img" aria-label={`Per-task ${unit}, ${ref.label} vs ${variant.label}`} className="block">
          {ticks.map((t) => (
            <g key={t} transform={`translate(${x(t)},0)`}>
              <line y1={top - 6} y2={top + plotH} style={{ stroke: t === 0 ? 'var(--axis)' : 'var(--grid)' }} strokeWidth={t === 0 ? 1.5 : 1} />
              <text y={top + plotH + 20} textAnchor="middle" className="tabular" style={{ fill: 'var(--muted)', fontSize: 13.5 }}>
                {String(t).replace('-', '−')}
              </text>
            </g>
          ))}
          <text x={(plotL + plotR) / 2} y={top + plotH + 40} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: 14.5 }}>
            {SETTING_LABEL[setting].toLowerCase()} {unit} (%)
          </text>
          <text x={W - 4} y={top - 12} textAnchor="end" style={{ fill: 'var(--ink-2)', fontSize: 13.5, fontWeight: 600 }}>
            change
          </text>
          {rows.map((row, i) => {
            const y0 = yOf(i);
            const cy = y0 + markY;
            const col = row.d < -0.05 ? 'var(--div-pos)' : row.d > 0.05 ? 'var(--div-neg)' : 'var(--ink-2)';
            return (
              <motion.g key={row.id} initial={false} animate={{ y: y0 }} transition={tr}>
                <g
                  tabIndex={0}
                  role="listitem"
                  aria-label={`${row.label}: ${f1(row.ref)} → ${f1(row.v)} (${d1(row.d)})`}
                  onPointerEnter={() => setHover(row.id)}
                  onFocus={() => setHover(row.id)}
                  onBlur={() => setHover(null)}
                  className="outline-none"
                >
                  {row.mean && <line x1={0} x2={W} y1={-meanGap / 2} y2={-meanGap / 2} style={{ stroke: 'var(--axis)' }} strokeWidth={1.5} />}
                  <rect x={0} y={0} width={W} height={rowH} rx={6} style={{ fill: hover === row.id ? 'var(--surface-2)' : 'transparent' }} />
                  <text
                    x={stacked ? plotL : labelW - 6}
                    y={stacked ? 18 : rowH / 2}
                    dy="0.32em"
                    textAnchor={stacked ? 'start' : 'end'}
                    style={{
                      fill: 'var(--ink)',
                      fontSize: 15,
                      fontWeight: row.mean ? 600 : 400,
                      stroke: hover === row.id ? 'var(--surface-2)' : 'var(--surface)',
                      strokeWidth: 4,
                      paintOrder: 'stroke',
                      strokeLinejoin: 'round',
                    }}
                  >
                    {row.label}
                  </text>
                  <motion.line
                    key={`line-${col}`}
                    initial={false}
                    animate={{ x1: x(row.ref), x2: x(row.v) }}
                    transition={tr}
                    y1={markY}
                    y2={markY}
                    style={{ stroke: col }}
                    strokeWidth={3}
                    strokeLinecap="round"
                  />
                  <motion.circle
                    initial={false}
                    animate={{ cx: x(row.ref) }}
                    transition={tr}
                    cy={markY}
                    r={5.5}
                    style={{ fill: 'var(--surface)', stroke: 'var(--ink-2)' }}
                    strokeWidth={2}
                  />
                  <motion.circle
                    key={`dot-${col}`}
                    initial={false}
                    animate={{ cx: x(row.v) }}
                    transition={tr}
                    cy={markY}
                    r={row.mean ? 7.5 : 6.5}
                    style={{ fill: col, stroke: 'var(--surface)' }}
                    strokeWidth={2}
                  />
                  <text
                    x={W - 4}
                    y={markY}
                    dy="0.32em"
                    textAnchor="end"
                    className="tabular"
                    style={{ fill: 'var(--ink)', fontSize: 15, fontWeight: row.mean ? 700 : 500 }}
                  >
                    {d1(row.d)}
                  </text>
                </g>
              </motion.g>
            );
          })}
        </svg>
        {hover &&
          (() => {
            const i = rows.findIndex((x) => x.id === hover);
            const row = rows[i];
            if (!row) return null;
            return (
              <Tip x={W * 0.5} y={yOf(i) + rowH + 2} width={W}>
                <div className="mb-1 text-[14.5px] text-ink">{row.label}</div>
                <div className="num text-ink">
                  {ref.short}: {f1(row.ref)}
                </div>
                <div className="num font-semibold text-ink">
                  {variant.short}: {f1(row.v)}
                </div>
                <div className="num text-ink-2">change: {d1(row.d)} points</div>
              </Tip>
            );
          })()}
      </div>
    </ChartTableFrame>
  );
}
