/**
 * Section 4c figure: ONE global datetime z-score per database (pre.rs:342-376,
 * spec §3.3). Every datetime column of every table file, task splits included,
 * is drawn on one shared time axis; the engine's μ_T and σ_T (population std)
 * are overlaid, with a z ruler under the date axis.
 */
import { max as d3max, scaleLinear, scaleUtc, utcMonth } from 'd3';
import { useMemo, useState } from 'react';
import { ChartFrame, DataTable, TeX, maxTextWidth, textWidth } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { catVar } from '@/styles/palette';
import { DAY_MS, datetimeColumns, fileLabel, isoDay } from './data';
import { Readout, Sub, signed, texNum, usePhone, type ChapterData } from './shared';

const nf = (n: number) => n.toLocaleString('en-US');

export function DatetimeNormFigure({ data }: { data: ChapterData }) {
  const { files, pdb, slotOf, seed } = data;
  const cols = useMemo(() => datetimeColumns(files), [files]);
  const { mean, std, count } = pdb.datetimeStats;
  const total = cols.reduce((s, c) => s + c.values.length, 0);
  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(280, size.width || 760);
  const LABEL_PX = 13;
  const labels = useMemo(() => cols.map((c) => `${fileLabel(c.file)}.${c.col.name}`), [cols]);
  const labelW = useMemo(() => Math.ceil(maxTextWidth(labels, LABEL_PX, { mono: true })) + 24, [labels]);
  // labels sit left of the bars when they fit in ~36% of the width; otherwise each row puts its label on its own line above its bars
  const above = labelW > W * 0.36;
  const LABEL_W = above ? 22 : labelW;
  const RIGHT_W = 52;
  const pw = W - LABEL_W - RIGHT_W;
  const BARS_H = 24;
  const LINE_H = above ? 18 : 0;
  const ROW_H = BARS_H + LINE_H;
  const TOP = 30;
  const plotH = cols.length * ROW_H;
  const H = TOP + plotH + 70;
  const narrow = W < 560;
  const phone = usePhone();

  const [lo, hi] = useMemo(() => {
    let a = Infinity;
    let b = -Infinity;
    for (const c of cols) for (const v of c.values) {
      if (v < a) a = v;
      if (v > b) b = v;
    }
    return [utcMonth.floor(new Date(a)).getTime(), utcMonth.ceil(new Date(b + 1)).getTime()];
  }, [cols]);
  const x = useMemo(() => scaleUtc().domain([lo, hi]).range([0, pw]), [lo, hi, pw]);
  const months = useMemo(() => utcMonth.range(new Date(lo), new Date(hi)).map((d) => d.getTime()), [lo, hi]);
  const hist = useMemo(
    () =>
      cols.map((c) => {
        const counts = new Array(months.length).fill(0) as number[];
        for (const v of c.values) {
          let i = months.length - 1;
          while (i > 0 && months[i] > v) i--;
          counts[i]++;
        }
        return counts;
      }),
    [cols, months],
  );
  const [hoverX, setHoverX] = useState<number | null>(null);
  const [hoverRow, setHoverRow] = useState<number | null>(null);

  const seedCell = seed?.cells.find((c) => c.sem === 2);
  const seedT = seedCell ? (seedCell.raw as number) : null;

  const zLo = Math.ceil((lo - mean) / std);
  const zHi = Math.floor((hi - mean) / std);
  const zTicks: number[] = [];
  for (let z = zLo; z <= zHi; z++) zTicks.push(z);
  const dateTicks = useMemo(() => {
    const out: Date[] = [];
    let last = -Infinity;
    for (const t of x.ticks(8)) {
      const px = x(t);
      if (px - last >= (narrow ? 62 : 70)) {
        out.push(t);
        last = px;
      }
    }
    return out;
  }, [x, narrow]);

  const table = (
    <DataTable
      caption="Datetime columns pooled into the global statistics"
      columns={['table file · column', 'values', 'first', 'last']}
      rows={[
        ...cols.map((c) => [`${c.file.key} · ${c.col.name}`, nf(c.values.length), isoDay(Math.min(...c.values)), isoDay(Math.max(...c.values))]),
        ['all (one pool)', nf(total), isoDay(lo), isoDay(hi)],
      ]}
    />
  );

  const hoverT = hoverX === null ? null : x.invert(hoverX).getTime();

  return (
    <div className="min-w-0">
      <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Readout label="values pooled n" hint={`${cols.length} columns, all files`}>
          {nf(count)}
        </Readout>
        <Readout label={<>global mean μ<Sub>T</Sub></>} hint={new Date(mean).toISOString().slice(11, 19) + ' UTC'}>
          {isoDay(mean)}
        </Readout>
        <Readout label={<>population std σ<Sub>T</Sub></>} hint="ddof = 0 (divide by n)">
          {(std / DAY_MS).toFixed(1)} days
        </Readout>
        <Readout label="statistics per column" hint="pre.rs stores only a placeholder (0, 0), never used">
          none
        </Readout>
      </div>
      <ChartFrame
        title="Every datetime column on one clock"
        subtitle={<>Monthly counts; each row is scaled to its own peak. The band is μ<Sub>T</Sub> ± σ<Sub>T</Sub>. Hover for a date’s z-score.</>}
        table={table}
      >
        <div ref={ref} className="relative w-full">
          <svg width={W} height={H} role="img" aria-label="Datetime columns of every table on a shared time axis with the global mean and standard deviation" className="block">
            <g transform={`translate(${LABEL_W},${TOP})`}>
              {/* σ band and mean */}
              <rect x={x(mean - std)} y={-6} width={x(mean + std) - x(mean - std)} height={plotH + 6} style={{ fill: 'var(--accent)', opacity: 0.07 }} />
              <line x1={x(mean)} x2={x(mean)} y1={-14} y2={plotH + 46} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.25} />
              <text x={x(mean) + 4} y={-16} style={{ fill: 'var(--ink-2)', fontSize: 15, fontWeight: 600 }}>
                μ<tspan dy={3} fontSize={13}>T</tspan>
              </text>
              <text x={x(mean + std) + 4} y={-16} style={{ fill: 'var(--muted)', fontSize: 14.5 }}>
                +σ<tspan dy={3} fontSize={13}>T</tspan>
              </text>
              {x(mean - std) > 40 && (
                <text x={x(mean - std) - 4} y={-16} textAnchor="end" style={{ fill: 'var(--muted)', fontSize: 14.5 }}>
                  −σ<tspan dy={3} fontSize={13}>T</tspan>
                </text>
              )}
            </g>
            {cols.map((c, ri) => {
              const counts = hist[ri];
              const m = d3max(counts) || 1;
              const yS = scaleLinear().domain([0, m]).range([0, BARS_H - 6]);
              const slot = slotOf(c.file.tableName);
              const rowY = TOP + ri * ROW_H;
              const labelY = above ? rowY + LINE_H / 2 + 2 : rowY + ROW_H / 2;
              return (
                <g key={c.id} onPointerEnter={() => setHoverRow(ri)} onPointerLeave={() => setHoverRow(null)}>
                  <rect x={0} y={rowY} width={W} height={ROW_H} style={{ fill: hoverRow === ri ? 'var(--surface-2)' : 'transparent' }} />
                  <rect x={2} y={labelY - 4.5} width={9} height={9} rx={2} style={{ fill: catVar(slot) }} />
                  <text x={16} y={labelY} dy="0.34em" className="font-mono" style={{ fill: 'var(--ink)', fontSize: LABEL_PX }}>
                    {labels[ri]}
                  </text>
                  <g transform={`translate(${LABEL_W},${rowY})`}>
                    <line x1={0} x2={pw} y1={ROW_H - 2} y2={ROW_H - 2} style={{ stroke: 'var(--grid)' }} />
                    {counts.map((n, i) =>
                      n > 0 ? (
                        <rect
                          key={i}
                          x={x(months[i]) + 0.5}
                          y={ROW_H - 2 - yS(n)}
                          width={Math.max(1, x(i + 1 < months.length ? months[i + 1] : hi) - x(months[i]) - 1)}
                          height={yS(n)}
                          rx={1.5}
                          style={{ fill: catVar(slot) }}
                        />
                      ) : null,
                    )}
                  </g>
                  <text x={W - 2} y={rowY + LINE_H + BARS_H / 2} dy="0.34em" textAnchor="end" className="tabular" style={{ fill: 'var(--ink-2)', fontSize: 13 }}>
                    {nf(c.values.length)}
                  </text>
                </g>
              );
            })}
            <g transform={`translate(${LABEL_W},${TOP + plotH})`}>
              {/* date axis */}
              {dateTicks.map((t) => {
                const label = t.toISOString().slice(0, 7);
                const half = textWidth(label, 13) / 2;
                // keep the label inside the svg: shift it, not the tick
                const dx = Math.max(half - LABEL_W + 2, Math.min(pw + RIGHT_W - half - 2, x(t))) - x(t);
                return (
                  <g key={t.getTime()} transform={`translate(${x(t)},0)`}>
                    <line y1={0} y2={5} style={{ stroke: 'var(--axis)' }} />
                    <text x={dx} y={17} textAnchor="middle" className="tabular" style={{ fill: 'var(--ink-2)', fontSize: 13 }}>
                      {label}
                    </text>
                  </g>
                );
              })}
              {/* z ruler */}
              <g transform="translate(0,46)">
                <line x1={0} x2={pw} style={{ stroke: 'var(--axis)' }} />
                {zTicks.map((z) => (
                  <g key={z} transform={`translate(${x(mean + z * std)},0)`}>
                    <line y1={-4} y2={4} style={{ stroke: z === 0 ? 'var(--ink-2)' : 'var(--axis)' }} strokeWidth={z === 0 ? 1.5 : 1} />
                    <text y={17} textAnchor="middle" className="tabular" style={{ fill: z === 0 ? 'var(--ink)' : 'var(--ink-2)', fontSize: 13, fontWeight: z === 0 ? 600 : 400 }}>
                      {z === 0 ? '0' : z > 0 ? `+${z}` : `−${-z}`}
                    </text>
                  </g>
                ))}
                <text x={-8} y={4} textAnchor="end" style={{ fill: 'var(--muted)', fontSize: 13 }}>
                  z
                </text>
                {seedT !== null && (
                  <g transform={`translate(${x(seedT)},0)`}>
                    <path d="M0,-4 L6,-14 L-6,-14 Z" style={{ fill: 'var(--accent)' }} />
                    <text
                      x={x(seedT) > pw - 50 ? -9 : 9}
                      y={-9}
                      textAnchor={x(seedT) > pw - 50 ? 'end' : 'start'}
                      style={{ fill: 'var(--accent-text)', fontSize: 13, fontWeight: 600 }}
                    >
                      seed
                    </text>
                  </g>
                )}
              </g>
            </g>
            {/* crosshair */}
            {hoverX !== null && hoverT !== null && (
              <g transform={`translate(${LABEL_W},0)`} style={{ pointerEvents: 'none' }}>
                <line x1={hoverX} x2={hoverX} y1={TOP - 6} y2={TOP + plotH + 46} style={{ stroke: 'var(--accent)' }} strokeWidth={1.5} />
              </g>
            )}
            <rect
              x={LABEL_W}
              y={TOP - 6}
              width={pw}
              height={plotH + 60}
              fill="transparent"
              onPointerMove={(e) => {
                const r = e.currentTarget.getBoundingClientRect();
                setHoverX(Math.max(0, Math.min(pw, e.clientX - r.left)));
                const ri = Math.floor((e.clientY - r.top - 6) / ROW_H);
                setHoverRow(ri >= 0 && ri < cols.length ? ri : null);
              }}
              onPointerLeave={() => {
                setHoverX(null);
                setHoverRow(null);
              }}
            />
          </svg>
          {hoverX !== null && hoverT !== null && (
            <div
              role="tooltip"
              className="pointer-events-none absolute z-20 rounded-[9px] border border-border bg-surface px-2.5 py-1.5 text-[14.5px] leading-tight shadow-[var(--shadow-pop)]"
              style={{ left: Math.max(0, Math.min(W - 250, LABEL_W + hoverX + 12)), top: TOP + 4 }}
            >
              <span className="num text-ink">{isoDay(hoverT)}</span> <span className="text-ink-2">→ z</span> <strong className="num text-ink">{signed((hoverT - mean) / std, 3)}</strong>
              {hoverRow !== null && (
                <div className="mt-0.5 text-ink-2">
                  {cols[hoverRow].file.key} · {cols[hoverRow].col.name}: {nf(cols[hoverRow].values.length)} values
                </div>
              )}
            </div>
          )}
        </div>
      </ChartFrame>
      <div className="mt-3 grid gap-2 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <div className="scroll-quiet overflow-x-auto rounded-lg bg-surface-2 px-3 py-2 text-[18px]">
          <TeX
            tex={
              phone
                ? `\\begin{aligned} \\mu_T &= \\dfrac{1}{n}\\sum t \\\\[4pt] \\sigma_T &= \\sqrt{\\dfrac{1}{n}\\sum t^2 - \\mu_T^2} \\end{aligned}`
                : `\\mu_T = \\dfrac{1}{n}\\sum t, \\qquad \\sigma_T = \\sqrt{\\dfrac{1}{n}\\sum t^2 - \\mu_T^2}`
            }
          />
        </div>
        {seedT !== null && seedCell && (
          <div className="scroll-quiet overflow-x-auto rounded-lg bg-surface-2 px-3 py-2 text-[18px]">
            <span className="mr-2 block align-middle text-[15px] text-ink-2 sm:inline">
              <span aria-hidden="true" className="mr-1 inline-block h-0 w-0 border-x-[5px] border-t-[8px] border-x-transparent border-t-[var(--accent)] align-middle" />
              seed {seedCell.colName}:
            </span>
            <TeX
              tex={
                phone
                  ? `\\begin{aligned} z &= \\dfrac{t - \\mu_T}{\\sigma_T} \\\\[4pt] &= \\dfrac{\\text{${isoDay(seedT)}} - \\text{${isoDay(mean)}}}{${texNum(std / DAY_MS, 1)}\\text{ days}} \\\\[4pt] &= ${texNum(seedCell.datetime, 3, { sign: true })} \\end{aligned}`
                  : `z = \\dfrac{t - \\mu_T}{\\sigma_T} = \\dfrac{\\text{${isoDay(seedT)}} - \\text{${isoDay(mean)}}}{${texNum(std / DAY_MS, 1)}\\text{ days}} = ${texNum(seedCell.datetime, 3, { sign: true })}`
              }
            />
          </div>
        )}
      </div>
    </div>
  );
}
