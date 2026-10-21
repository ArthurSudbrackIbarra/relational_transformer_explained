/**
 * Section "Experiment": move L and w and watch what the context is made of. Counts come from the
 * live context; the two sweeps re-run the sampler (no model) for a range of L or w on the same seed,
 * in small chunks so the page stays responsive, and are cached.
 */
import { useEffect, useMemo, useState } from 'react';
import { LineChart, Stat, TableSwatch } from '@/components';
import { inContextLabels } from '@/engine/pipeline';
import { makeSampleParams } from '@/state/hooks';
import { sampleContext } from '@/engine/sampler';
import type { PreprocessedDB, TaskDef } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { useExplore } from './explore';
import { useLabels } from './Labels';
import { countByTable, cutRow, maxVisitedDepth, plural, rowName } from './trace';

interface SweepPoint {
  L: number;
  w: number;
  real: number;
  visited: number;
  self: number;
  other: number;
}

const cache = new WeakMap<PreprocessedDB, Map<string, SweepPoint>>();

function compute(pdb: PreprocessedDB, task: TaskDef, seed: number, L: number, w: number, rng: number): SweepPoint {
  let m = cache.get(pdb);
  if (!m) cache.set(pdb, (m = new Map()));
  const key = `${task.id}|${seed}|${L}|${w}|${rng}`;
  const hit = m.get(key);
  if (hit) return hit;
  const ctx = sampleContext(pdb, makeSampleParams(pdb, task, seed, L, w, rng), { recordEvents: false });
  const lab = inContextLabels(pdb, ctx, task);
  const p = { L, w, real: ctx.numReal, visited: ctx.visited.length, self: lab.self.length, other: lab.other.length };
  m.set(key, p);
  return p;
}

/**
 * Sweep: a few sampler runs per macrotask, cached per (task, seed, L, w, rng). It starts 150 ms after the
 * last settings change (so dragging a slider does not restart it on every step) and keeps showing the
 * previous curve until the new one is complete.
 */
function useSweep(points: { L: number; w: number }[]): { data: (SweepPoint | null)[]; pending: boolean } {
  const { pdb, task, seedNode, rngSeed } = useExplore();
  const key = `${task.id}|${seedNode}|${rngSeed}|${points.map((p) => `${p.L}:${p.w}`).join(',')}`;
  const [out, setOut] = useState<{ key: string; data: (SweepPoint | null)[] }>(() => ({ key: '', data: points.map(() => null) }));
  useEffect(() => {
    let alive = true;
    const res: (SweepPoint | null)[] = points.map(() => null);
    let i = 0;
    let id = 0;
    const tick = () => {
      if (!alive) return;
      const t0 = performance.now();
      while (i < points.length && performance.now() - t0 < 12) {
        res[i] = compute(pdb, task, seedNode, points[i].L, points[i].w, rngSeed);
        i++;
      }
      if (i < points.length) id = window.setTimeout(tick, 0);
      else setOut({ key, data: res });
    };
    id = window.setTimeout(tick, 150);
    return () => {
      alive = false;
      window.clearTimeout(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return { data: out.data, pending: out.key !== key };
}

const L_GRID = [16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512];
const W_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];

export function ExperimentStats() {
  const { ctx, pdb, db, task, slots } = useExplore();
  const labels = useLabels();
  const seed = pdb.nodes[ctx.params.seedNode];
  const counts = useMemo(() => countByTable(pdb, ctx), [pdb, ctx]);
  const cut = useMemo(() => cutRow(ctx, ctx.params.dropColNameIdxs, pdb), [ctx, pdb]);
  const same = labels.other.filter((l) => l.ts !== null && l.ts === seed.ts).length;
  const L = ctx.params.L;
  const pad = L - ctx.numReal;
  return (
    <div className="grid min-w-0 gap-4">
      <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Tokens" value={ctx.numReal} unit={`/ ${L}`} caption={cut ? `${rowName(db, pdb, cut.node).short} cut at L` : pad > 0 ? 'frontier ran dry' : 'full'} />
        <Stat label="Padding" value={pad} caption={pad > 0 ? 'positions left empty' : 'none'} />
        <Stat label="Rows visited" value={ctx.visited.length} caption={`deepest depth label ${maxVisitedDepth(ctx)}`} />
        <Stat label="Self labels" value={labels.self.length} emphasis caption={task.kind === 'forecast' ? 'the seed entity’s past rows' : 'n/a for autocomplete'} />
        <Stat label="Other-entity labels" value={labels.other.length} caption={same > 0 ? `${same} at the seed’s own timestamp` : task.kind === 'forecast' ? 'reached deeper in the graph' : 'other rows’ target cells'} />
        <Stat label="Tables in context" value={counts.length} caption={`${plural(counts.reduce((s, c) => s + c.rows, 0), 'row')}`} />
      </div>
      <div className="min-w-0">
        <div className="mb-1.5 text-[15px] font-semibold text-ink">What the {L} positions hold</div>
        <div className="flex h-6 w-full overflow-hidden rounded-md border border-border" role="img" aria-label={counts.map((c) => `${c.key}: ${c.tokens} tokens`).join(', ') + (pad ? `, padding: ${pad}` : '')}>
          {counts.map((c) => (
            <div key={c.key} title={`${c.key}: ${plural(c.tokens, 'token')} from ${plural(c.rows, 'row')}`} style={{ width: `${(100 * c.tokens) / L}%`, background: catVar(slots[c.tableName] ?? 0) }} className="h-full border-r-2 border-surface last:border-r-0" />
          ))}
          {pad > 0 && <div title={`padding: ${pad}`} style={{ width: `${(100 * pad) / L}%`, backgroundImage: 'repeating-linear-gradient(135deg, var(--grid) 0 4px, transparent 4px 8px)' }} className="h-full" />}
        </div>
        <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[14.5px] text-ink-2">
          {counts.map((c) => (
            <li key={c.key} className="inline-flex items-center gap-1.5">
              <TableSwatch slot={slots[c.tableName] ?? 0} label={c.tableType === 'Db' ? c.tableName : `${c.tableName} (${c.tableType})`} />
              <span className="num text-muted">
                {c.rows}r · {c.tokens}t
              </span>
            </li>
          ))}
          {pad > 0 && <li className="text-muted">padding · {pad}t</li>}
        </ul>
      </div>
    </div>
  );
}

export function SweepCharts() {
  const { ctx, task } = useExplore();
  const L = ctx.params.L;
  const w = ctx.params.w;
  const lPoints = useMemo(() => L_GRID.map((x) => ({ L: x, w })), [w]);
  const wPoints = useMemo(() => W_GRID.map((x) => ({ L, w: x })), [L]);
  const { data: byL, pending: pendingL } = useSweep(lPoints);
  const { data: byW, pending: pendingW } = useSweep(wPoints);
  const series = (data: (SweepPoint | null)[], x: (p: SweepPoint) => number) => {
    const ok = data.filter((p): p is SweepPoint => p !== null);
    const s = [{ id: 'other', label: task.kind === 'forecast' ? 'other-entity labels' : 'target cells of other rows', slot: 2, points: ok.map((p) => ({ x: x(p), y: p.other })) }];
    if (task.kind === 'forecast') s.unshift({ id: 'self', label: 'self labels', slot: 1, points: ok.map((p) => ({ x: x(p), y: p.self })) });
    return s;
  };
  const doneL = !pendingL;
  const doneW = !pendingW;
  const enough = (d: (SweepPoint | null)[]) => d.filter((p) => p !== null).length >= 2;
  const placeholder = <div className="flex h-[250px] items-center justify-center text-[15.5px] text-muted">Sampling contexts…</div>;
  return (
    <div className="wide my-6 grid min-w-0 gap-4 lg:grid-cols-2">
      <div className="card min-w-0 px-4 py-4">
        {!enough(byL) ? placeholder : <LineChart
          title={`Labels in the context as L grows (w = ${w})`}
          subtitle={doneL ? 'Same seed and RNG seed; only L changes.' : 'Sampling contexts for the new settings…'}
          series={series(byL, (p) => p.L)}
          xLabel="context length L"
          yLabel="labels"
          markers={[{ x: L, label: `L = ${L}` }]}
          formatX={(v) => String(Math.round(v))}
          formatY={(v) => String(Math.round(v))}
          height={250}
        />}
      </div>
      <div className="card min-w-0 px-4 py-4">
        {!enough(byW) ? placeholder : <LineChart
          title={`Labels in the context as w grows (L = ${L})`}
          subtitle={doneW ? 'Same seed and RNG seed; only w changes (log scale).' : 'Sampling contexts for the new settings…'}
          series={series(byW, (p) => p.w)}
          xLabel="BFS width w"
          yLabel="labels"
          xScale="log"
          markers={[{ x: w, label: `w = ${w}` }]}
          formatX={(v) => String(Math.round(v))}
          formatY={(v) => String(Math.round(v))}
          height={250}
        />}
      </div>
    </div>
  );
}
