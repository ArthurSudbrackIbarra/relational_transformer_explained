/**
 * Section 1 figure: how fast the neighbourhood of the seed grows with hops, against L.
 * A plain BFS over foreign-key links in both directions (no time filter, no width bound), restricted
 * to the tables the sampler may visit for this task (DB tables + the seed's own task table, fly.rs:417).
 * The totals count every row of those tables: an upper bound, since rows that can only be reached as
 * P→F children and are later than the seed never pass the time filter (fly.rs:421-424).
 */
import { useMemo } from 'react';
import { BarChart, Stat } from '@/components';
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
          caption="Every DB table plus this task’s table, all splits: an upper bound, since a row later than the seed that can only be reached as a child never passes the time filter."
        />
        <Stat label="Feature cells in those tables" value={g.totalCells.toLocaleString('en-US')} caption="Every non-key, non-missing cell is a potential token." />
        <Stat label="Context length L" value={L} unit="tokens" emphasis caption="The model sees at most L cells per prediction (paper: 1024)." />
      </div>
      <div className="card min-w-0 px-4 py-4">
        <BarChart
          title="Cells within k hops of the seed row"
          subtitle="Plain BFS over foreign-key links in both directions, without RT’s time filter or width bound. Cumulative."
          categories={cats}
          series={[{ id: 'cells', label: 'feature cells', values: g.cells, slot: 1 }]}
          refLines={[{ value: L, label: `L = ${L}` }]}
          format={(v) => Math.round(v).toLocaleString('en-US')}
          labelWidth={110}
        />
      </div>
    </div>
  );
}
