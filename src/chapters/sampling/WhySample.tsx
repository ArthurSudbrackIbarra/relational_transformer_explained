/**
 * Section 1 figure: how fast the neighborhood of the seed grows with hops, against L.
 * A plain BFS over foreign-key links in both directions (no time filter, no width bound), restricted
 * to the tables the sampler may visit for this task (DB tables + the seed's own task table, fly.rs:417).
 * The totals count every row of those tables: an upper bound, since rows that can only be reached as
 * P→F children and are later than the seed never pass the time filter (fly.rs:421-424).
 */
import { useMemo, useState } from 'react';
import { ChartFrame, DataTable, SVG_TYPE, Stat, maxTextWidth, textWidth } from '@/components';
import { hBarPath } from '@/components/viz/chartParts';
import { useElementSize } from '@/components/viz/useSize';
import { catVar } from '@/styles/palette';
import { useExplore } from './explore';

export function useHopGrowth(maxHops = 5) {
  const { pdb, ctx } = useExplore();
  return useMemo(() => {
    const seed = ctx.params.seedNode;
    const seedTable = pdb.nodes[seed].tableNameIdx;
    const allowed = (n: number) => {
      const g = pdb.nodes[n];
      return g.tableType === 'Db' || g.tableNameIdx === seedTable;
    };
    const dist = new Map<number, number>([[seed, 0]]);
    let frontier = [seed];
    const rows: number[] = [1];
    const cells: number[] = [pdb.nodes[seed].cells.length];
    for (let h = 1; h <= maxHops; h++) {
      const next: number[] = [];
      for (const n of frontier) {
        const nbrs = [...pdb.nodes[n].f2pEdges.map((e) => e.node), ...(pdb.p2fAdj[n] ?? []).map((e) => e.node)];
        for (const m of nbrs) {
          if (dist.has(m) || !allowed(m)) continue;
          dist.set(m, h);
          next.push(m);
        }
      }
      frontier = next;
      rows.push(rows[h - 1] + next.length);
      cells.push(cells[h - 1] + next.reduce((s, m) => s + pdb.nodes[m].cells.length, 0));
    }
    let totalRows = 0;
    let totalCells = 0;
    for (const g of pdb.nodes) {
      if (g.tableType !== 'Db' && g.tableNameIdx !== seedTable) continue;
      totalRows++;
      totalCells += g.cells.length;
    }
    return { rows, cells, totalRows, totalCells };
  }, [pdb, ctx.params.seedNode, maxHops]);
}

export function WhySampleFigure() {
  const { db, ctx } = useExplore();
  const g = useHopGrowth(5);
  const L = ctx.params.L;
  const cats = g.cells.map((_, h) => (h === 0 ? 'the seed row' : `≤ ${h} hop${h > 1 ? 's' : ''}`));
  return (
    <div className="wide my-8 grid min-w-0 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.7fr)]">
      <div className="grid content-start gap-3 sm:grid-cols-3 lg:grid-cols-1">
        <Stat
          label={`Rows in the tables the sampler may enter (${db.name})`}
          value={g.totalRows.toLocaleString('en-US')}
          caption="Every DB table plus this task’s table, all splits. It also counts rows the time filter would block, so it is an upper bound."
        />
        <Stat label="Feature cells in those tables" value={g.totalCells.toLocaleString('en-US')} caption="Every non-key, non-missing cell is a potential token." />
        <Stat label="Context length L" value={L} unit="tokens" emphasis caption="The model sees at most L cells per prediction (paper: 1024)." />
      </div>
      <div className="card min-w-0 px-4 py-4">
        <LogBars categories={cats} values={g.cells} L={L} />
      </div>
    </div>
  );
}

/**
 * Horizontal bars on a log axis: the counts span 1 to thousands, and the point of the figure (where the
 * cumulative count crosses L) sits at the small end, which a linear axis squeezes into a few pixels.
 */
function LogBars({ categories, values, L }: { categories: string[]; values: number[]; L: number }) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const width = Math.max(260, size.width || 600);
  const fmt = (v: number) => Math.round(v).toLocaleString('en-US');
  const T = SVG_TYPE;
  const maxV = Math.max(L, ...values, 10);
  const decades = Math.max(2, Math.ceil(Math.log10(maxV)));
  const top = 10 ** decades;
  const lw = Math.ceil(maxTextWidth(categories, T.category) + 14);
  const tipText = (v: number) => `${fmt(v)}${v > L ? ` · ${(v / L).toFixed(v / L >= 10 ? 0 : 1)}× L` : ''}`;
  const valuePad = Math.ceil(maxTextWidth(values.map(tipText), T.value)) + 12;
  const M = { top: 28, right: valuePad, bottom: 48, left: lw };
  const plotW = Math.max(80, width - M.left - M.right);
  const bandH = 34;
  const barH = 20;
  const plotH = categories.length * bandH;
  const x = (v: number) => (Math.log10(Math.max(1, v)) / decades) * plotW;
  const ticks = Array.from({ length: decades + 1 }, (_, i) => 10 ** i);
  const xL = x(L);
  const lLabel = `L = ${L}`;
  const lLabelW = textWidth(lLabel, T.annotation, { weight: 600 });
  return (
    <ChartFrame
      title="Cells within k hops of the seed row"
      subtitle="Plain BFS over foreign-key links in both directions, without RT’s time filter or width bound. Cumulative, log scale."
      table={<DataTable columns={['hops', 'feature cells', 'multiple of L']} rows={categories.map((c, i) => [c, fmt(values[i]), (values[i] / L).toFixed(2)])} caption="Cells within k hops of the seed row" />}
    >
      <div ref={ref} className="relative w-full">
        <svg width={width} height={plotH + M.top + M.bottom} role="img" aria-label={`Cells within k hops of the seed row, log scale: ${categories.map((c, i) => `${c} ${fmt(values[i])}`).join(', ')}; L = ${L}.`} className="block">
          <g transform={`translate(${M.left},${M.top})`}>
            {ticks.filter((_, i) => plotW >= 300 || i % 2 === 0 || i === decades).map((t) => (
              <g key={t} transform={`translate(${x(t)},0)`}>
                <line y2={plotH} style={{ stroke: t === 1 ? 'var(--axis)' : 'var(--grid)' }} />
                <text y={plotH + 20} textAnchor="middle" className="tabular" style={{ fill: 'var(--muted)', fontSize: T.tick }}>
                  {fmt(t)}
                </text>
              </g>
            ))}
            <text x={plotW / 2} y={plotH + 42} textAnchor="middle" style={{ fill: 'var(--ink-2)', fontSize: T.axisTitle }}>
              feature cells (log scale)
            </text>
            {categories.map((c, i) => {
              const yb = i * bandH + (bandH - barH) / 2;
              const v = values[i];
              const over = v > L;
              return (
                <g key={c} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
                  <rect x={-M.left} y={i * bandH} width={M.left + plotW + valuePad} height={bandH} style={{ fill: hover === i ? 'var(--surface-2)' : 'transparent' }} />
                  <text x={-10} y={yb + barH / 2} dy="0.32em" textAnchor="end" style={{ fill: 'var(--ink)', fontSize: T.category }}>
                    {c}
                  </text>
                  <path d={hBarPath(0, Math.max(0.5, x(v)), yb, barH)} style={{ fill: catVar(1), opacity: over ? 1 : 0.55 }} />
                  <text x={x(v) + 6} y={yb + barH / 2} dy="0.32em" className="tabular" style={{ fill: over ? 'var(--ink)' : 'var(--ink-2)', fontSize: T.value, fontWeight: over ? 600 : 400 }}>
                    {tipText(v)}
                  </text>
                </g>
              );
            })}
            <line x1={xL} x2={xL} y1={-6} y2={plotH} style={{ stroke: 'var(--accent)' }} strokeWidth={1.6} strokeDasharray="4 3" />
            <text x={xL + lLabelW / 2 + 4 > plotW ? xL - 4 : xL} y={-12} textAnchor={xL + lLabelW / 2 + 4 > plotW ? 'end' : 'middle'} style={{ fill: 'var(--accent-text)', fontSize: T.annotation, fontWeight: 600 }}>
              {lLabel}
            </text>
          </g>
        </svg>
      </div>
      <p className="mt-1 text-[14px] text-muted">Faded bars fit in L; solid bars exceed it.</p>
    </ChartFrame>
  );
}
