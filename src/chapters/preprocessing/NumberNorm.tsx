/**
 * Section 4a figure: per-column z-scores of number cells (pre.rs:330-341,
 * spec §3.3). One histogram, two rulers: raw units above, z-scores below
 * (same data, same pixel positions: z is an affine relabeling). A probe value
 * shows z = (v − μ)/σ with the engine's live μ and σ (sample std, ddof = 1).
 */
import { bin as d3bin, scaleLinear } from 'd3';
import { motion } from 'motion/react';
import { useMemo, useState } from 'react';
import { ChartFrame, DataTable, Segmented, Slider, TeX, textWidth } from '@/components';
import { TooltipBox, vBarPath } from '@/components/viz/chartParts';
import { formatNumber } from '@/components/viz/format';
import { useElementSize } from '@/components/viz/useSize';
import { catVar } from '@/styles/palette';
import { usePrefersReducedMotion } from '@/styles/theme';
import { isSmallInteger, type StatColumn } from './data';
import { Readout, fixed, signed, texNum, type ChapterData } from './shared';

const nf = (n: number) => n.toLocaleString('en-US');

function colLabel(c: StatColumn) {
  return `${c.file.tableName}.${c.col.name}`;
}

function NumberChart({ c, slot, probe, setProbe, step }: { c: StatColumn; slot: number; probe: number; setProbe: (v: number) => void; step: number }) {
  const { mean, std } = c.stats;
  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(280, size.width || 700);
  const M = { top: 46, right: 14, bottom: 70, left: 58 };
  const PH = 150;
  const pw = W - M.left - M.right;
  const intBins = isSmallInteger(c.values);

  const { x, bins, yMax } = useMemo(() => {
    let lo = Math.min(...c.values);
    let hi = Math.max(...c.values);
    if (intBins) {
      lo -= 0.5;
      hi += 0.5;
    }
    const xs = scaleLinear().domain([lo, hi]);
    if (!intBins) xs.nice(8);
    const dom = xs.domain() as [number, number];
    const thresholds = intBins ? Array.from({ length: Math.round(dom[1] - dom[0]) - 1 }, (_, i) => dom[0] + i + 1) : xs.ticks(16);
    const b = d3bin().domain(dom).thresholds(thresholds)(c.values);
    return { x: xs.range([0, pw]), bins: b, yMax: Math.max(1, ...b.map((d) => d.length)) };
  }, [c, pw, intBins]);
  const y = scaleLinear().domain([0, yMax]).nice(4).range([PH, 0]);
  const [hover, setHover] = useState<number | null>(null);
  const reduce = usePrefersReducedMotion();

  const dom = x.domain();
  const zLo = Math.ceil((dom[0] - mean) / std);
  const zHi = Math.floor((dom[1] - mean) / std);
  const zTicks: number[] = [];
  const zStep = zHi - zLo > 10 ? 2 : 1;
  for (let z = zLo; z <= zHi; z++) if (z % zStep === 0) zTicks.push(z);
  const rawTicks = intBins ? Array.from(new Set(c.values)).sort((a, b) => a - b).filter((_, i, a) => a.length <= 12 || i % Math.ceil(a.length / 10) === 0) : x.ticks(Math.max(3, Math.floor(pw / 80)));
  const zProbe = (probe - mean) / std;
  const px = x(probe);

  const onPointer = (e: React.PointerEvent<SVGRectElement>) => {
    const r = (e.currentTarget as SVGRectElement).getBoundingClientRect();
    const v = x.invert(e.clientX - r.left);
    const clamped = Math.max(dom[0], Math.min(dom[1], v));
    setProbe(Math.round(clamped / step) * step);
  };

  const label = `v = ${formatNumber(probe, 2)} → z = ${signed(zProbe, 3)}`;
  // measured with the real font (13.5px semibold), plus 9px padding on each side
  const lw = Math.ceil(textWidth(label, 13.5, { weight: 600 })) + 18;
  const lx = Math.max(-M.left + 2, Math.min(pw - lw + M.right - 2, px - lw / 2));

  return (
    <div ref={ref} className="relative w-full">
      <svg width={W} height={M.top + PH + M.bottom} role="img" aria-label={`Histogram of ${colLabel(c)} with its z-score ruler`} className="block">
        <g transform={`translate(${M.left},${M.top})`}>
          {y.ticks(4).map((t) => (
            <g key={t} transform={`translate(0,${y(t)})`}>
              <line x2={pw} style={{ stroke: t === 0 ? 'var(--axis)' : 'var(--grid)' }} />
              <text x={-8} dy="0.32em" textAnchor="end" className="tabular" style={{ fill: 'var(--muted)', fontSize: 13 }}>
                {t}
              </text>
            </g>
          ))}
          {/* ±1σ wash, clipped to the plot */}
          {(() => {
            const b0 = Math.max(0, x(mean - std));
            const b1 = Math.min(pw, x(mean + std));
            return b1 > b0 ? <rect x={b0} y={0} width={b1 - b0} height={PH} style={{ fill: 'var(--accent)', opacity: 0.06 }} /> : null;
          })()}
          <text transform={`translate(${-M.left + 14},${PH / 2}) rotate(-90)`} textAnchor="middle" style={{ fill: 'var(--muted)', fontSize: 13 }}>
            rows
          </text>
          {bins.map((b, i) => {
            const x0 = x(b.x0 ?? 0) + 1;
            const x1 = x(b.x1 ?? 0) - 1;
            const w = Math.max(1, x1 - x0);
            return (
              <g key={i}>
                <motion.path
                  initial={reduce ? false : { d: vBarPath(x0, w, PH, PH) }}
                  animate={{ d: vBarPath(x0, w, PH, y(b.length)) }}
                  transition={reduce ? { duration: 0 } : { duration: 0.5, delay: i * 0.012, ease: 'easeOut' }}
                  style={{ fill: catVar(slot), opacity: hover === null || hover === i ? 1 : 0.55 }}
                />
              </g>
            );
          })}
          {/* hit area */}
          <rect
            x={0}
            y={0}
            width={pw}
            height={PH}
            fill="transparent"
            style={{ cursor: 'crosshair' }}
            onPointerMove={(e) => {
              const r = e.currentTarget.getBoundingClientRect();
              const v = x.invert(e.clientX - r.left);
              const i = bins.findIndex((b) => v >= (b.x0 ?? 0) && v < (b.x1 ?? 0));
              setHover(i >= 0 ? i : null);
            }}
            onPointerLeave={() => setHover(null)}
            onPointerDown={onPointer}
          />
          {/* mean */}
          <line x1={x(mean)} x2={x(mean)} y1={-6} y2={PH + 44} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.25} />
          {/* μ label on the side away from the probe line */}
          <text
            x={x(mean) + (px >= x(mean) && px - x(mean) < 24 ? -4 : 4)}
            y={-8}
            textAnchor={px >= x(mean) && px - x(mean) < 24 ? 'end' : 'start'}
            style={{ fill: 'var(--ink-2)', fontSize: 13.5, fontWeight: 600 }}
          >
            μ
          </text>
          {/* probe */}
          <line x1={px} x2={px} y1={-10} y2={PH + 44} style={{ stroke: 'var(--accent)', pointerEvents: 'none' }} strokeWidth={2} />
          <g transform={`translate(${lx},${-M.top + 2})`} style={{ pointerEvents: 'none' }}>
            <rect width={lw} height={20} rx={10} style={{ fill: 'var(--accent)' }} />
            <text x={lw / 2} y={10} dy="0.34em" textAnchor="middle" className="tabular" style={{ fill: 'var(--accent-ink)', fontSize: 13.5, fontWeight: 600 }}>
              {label}
            </text>
          </g>
          {/* raw axis */}
          <g transform={`translate(0,${PH})`}>
            {rawTicks.map((t) => (
              <g key={t} transform={`translate(${x(t)},0)`}>
                <line y2={5} style={{ stroke: 'var(--axis)' }} />
                <text y={17} textAnchor="middle" className="tabular" style={{ fill: 'var(--ink-2)', fontSize: 13 }}>
                  {formatNumber(t, 2)}
                </text>
              </g>
            ))}
            <text x={-14} y={17} textAnchor="end" style={{ fill: 'var(--muted)', fontSize: 13 }}>
              raw
            </text>
          </g>
          {/* z ruler */}
          <g transform={`translate(0,${PH + 40})`}>
            <line x1={0} x2={pw} style={{ stroke: 'var(--axis)' }} />
            {zTicks.map((z) => (
              <motion.g key={z} initial={reduce ? false : { opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4, delay: 0.25 }} transform={`translate(${x(mean + z * std)},0)`}>
                <line y1={-4} y2={4} style={{ stroke: z === 0 ? 'var(--ink-2)' : 'var(--axis)' }} strokeWidth={z === 0 ? 1.5 : 1} />
                <text y={17} textAnchor="middle" className="tabular" style={{ fill: z === 0 ? 'var(--ink)' : 'var(--ink-2)', fontSize: 13, fontWeight: z === 0 ? 600 : 400 }}>
                  {z === 0 ? '0' : z > 0 ? `+${z}` : `−${-z}`}
                </text>
              </motion.g>
            ))}
            <text x={-14} y={4} textAnchor="end" style={{ fill: 'var(--muted)', fontSize: 13 }}>
              z
            </text>
          </g>
        </g>
      </svg>
      {hover !== null && bins[hover] && (
        <TooltipBox style={{ left: Math.min(W - 200, M.left + x(bins[hover].x0 ?? 0) + 12), top: M.top + 10 }}>
          <div className="mb-1 text-[14px] text-muted">
            {intBins ? `${c.col.name} = ${formatNumber(((bins[hover].x0 ?? 0) + (bins[hover].x1 ?? 0)) / 2)}` : `${c.col.name} ${formatNumber(bins[hover].x0 ?? 0, 2)} to ${formatNumber(bins[hover].x1 ?? 0, 2)}`}
          </div>
          <div>
            <strong className="num text-ink">{bins[hover].length}</strong> <span className="text-ink-2">{bins[hover].length === 1 ? 'row' : 'rows'}</span>
          </div>
          <div className="num mt-0.5 text-ink-2">
            z {signed(((intBins ? ((bins[hover].x0 ?? 0) + (bins[hover].x1 ?? 0)) / 2 : bins[hover].x0 ?? 0) - mean) / std, 2)}
            {!intBins && ` to ${signed(((bins[hover].x1 ?? 0) - mean) / std, 2)}`}
          </div>
        </TooltipBox>
      )}
    </div>
  );
}

export function NumberNormFigure({ data, cols, defaultId, defaultProbe }: { data: ChapterData; cols: StatColumn[]; defaultId: string; defaultProbe: Record<string, number> }) {
  const [id, setId] = useState(defaultId);
  const c = cols.find((x) => x.id === id) ?? cols[0];
  const [probes, setProbes] = useState<Record<string, number>>(defaultProbe);
  if (!c) return <p className="text-[16.5px] text-ink-2">This database has no number columns.</p>;
  const sorted = [...c.values].sort((a, b) => a - b);
  const lo = sorted[0];
  const hi = sorted[sorted.length - 1];
  const intVals = c.values.every((v) => Number.isInteger(v));
  const step = intVals ? 1 : hi - lo > 100 ? 1 : hi - lo > 10 ? 0.5 : 0.1;
  const probe = probes[c.id] ?? sorted[Math.floor(sorted.length / 2)];
  const setProbe = (v: number) => setProbes((p) => ({ ...p, [c.id]: v }));
  const { mean, std } = c.stats;
  const z = (probe - mean) / std;
  const slot = data.slotOf(c.file.tableName);
  const isTask = c.file.isTask;
  // pre.rs:338-339 replaces σ by 1 when it is undefined (n ≤ 1) or exactly 0 (a constant column)
  const guard = c.values.length <= 1 || lo === hi;

  const table = (
    <DataTable
      caption={`Values of ${colLabel(c)}`}
      columns={['value', 'rows', 'z']}
      rows={Array.from(
        c.values.reduce((m, v) => m.set(v, (m.get(v) ?? 0) + 1), new Map<number, number>()),
      )
        .sort((a, b) => a[0] - b[0])
        .slice(0, 400)
        .map(([v, n]) => [formatNumber(v, 3), n, signed((v - mean) / std, 3)])}
    />
  );

  return (
    <div className="min-w-0">
      <div className="mb-4">
        <Segmented
          label="Number column"
          value={c.id}
          onChange={setId}
          options={cols.map((x) => ({ value: x.id, label: <span className="font-mono">{colLabel(x)}</span> }))}
        />
      </div>
      <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Readout label="non-null values n" hint={c.dropped ? `${c.dropped} null dropped` : 'no nulls'}>
          {nf(c.values.length)}
        </Readout>
        <Readout label="mean μ">{fixed(mean, 3)}</Readout>
        <Readout label={guard ? 'σ (guarded)' : 'sample std σ'} hint={guard ? 'guard: σ set to 1 (pre.rs:338-339)' : 'ddof = 1 (divide by n − 1)'}>
          {fixed(std, 3)}
        </Readout>
        <Readout label="rows that fed μ, σ" hint={isTask ? 'Val and Test reuse them' : 'every row of the table file'}>
          <span className="font-mono text-[15.5px]">{c.file.key}</span>
        </Readout>
      </div>
      <ChartFrame title={`${colLabel(c)}: histogram on two rulers`} subtitle="Click the plot or use the slider to move the probe." table={table}>
        <NumberChart key={c.id} c={c} slot={slot} probe={probe} setProbe={setProbe} step={step} />
      </ChartFrame>
      <div className="mt-2 grid items-end gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <Slider label={`Probe value v (${c.col.name})`} value={probe} min={lo} max={hi} step={step} onChange={setProbe} format={(v) => formatNumber(v, 2)} />
        <div className="scroll-quiet overflow-x-auto rounded-lg bg-surface-2 px-3 py-2 text-[18px]">
          <TeX tex={`z = \\dfrac{v - \\mu}{\\sigma} = \\dfrac{${texNum(probe, 2, { trim: true })} - ${texNum(mean, 3)}}{${texNum(std, 3)}} = ${texNum(z, 3, { sign: true })}`} />
        </div>
      </div>
    </div>
  );
}
