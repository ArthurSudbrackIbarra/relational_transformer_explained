/**
 * The database as a relational entity graph (paper p.16 App. B): one node per
 * row, one edge per FK link, drawn for the k-hop neighbourhood of the focus row.
 * The layout is radial: the focus row in the centre and one ring per hop
 * distance, so the distance is read from the position, not from labels. Rows
 * with many children are drawn with a capped number of them (display limit
 * only, stated in the UI). The layout follows the focus row only while the
 * figure is near the viewport, so walking the database higher up stays fast.
 */
import { memo, startTransition, useEffect, useMemo, useState, useTransition } from 'react';
import { EntityGraph, Segmented, type GraphEdgeSpec, type GraphNodeSpec, type NodeId, type RingStyle } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import type { RawDatabase } from '@/engine/types';
import { useNearViewport } from './hooks';
import { parseRowKey, rowCaption, rowKey, rowLabel, sameRow, type LinkIndex, type RowRef } from './model';

const CAP_FOCUS = 60;
const CAP_DEEP = 12;
/** Screen px between neighbouring nodes on a ring (node diameter is 15 px). */
const NODE_GAP = 19;
/** Radial distance between the sub-bands of a crowded ring. */
const BAND_GAP = 15;

export const NeighbourhoodGraph = memo(function NeighbourhoodGraph({
  db,
  idx,
  slots,
  focus,
  onFocus,
}: {
  db: RawDatabase;
  idx: LinkIndex;
  slots: Record<string, number>;
  focus: RowRef;
  onFocus: (r: RowRef) => void;
}) {
  // `hopsUi` drives the control at once; the layout for `hops` follows as a transition.
  const [hopsUi, setHopsUi] = useState<1 | 2 | 3>(2);
  const [hops, setHops] = useState<1 | 2 | 3>(2);
  const [pending, startHops] = useTransition();
  const chooseHops = (h: 1 | 2 | 3) => {
    setHopsUi(h);
    startHops(() => setHops(h));
  };
  // The row actually drawn: follows `focus`, but only while the figure is near the viewport.
  const [box, near] = useNearViewport<HTMLDivElement>('400px 0px');
  const [drawn, setDrawn] = useState<RowRef>(focus);
  useEffect(() => {
    if (near && !sameRow(drawn, focus)) startTransition(() => setDrawn(focus));
  }, [near, focus, drawn]);
  const [wrap, size] = useElementSize<HTMLDivElement>();
  const width = Math.max(300, size.width || 720);
  const narrow = width < 560;

  // breadth-first neighbourhood: hop distance of every row, and the row it was reached from
  const g = useMemo(() => {
    const hopOf = new Map<string, number>([[rowKey(drawn), 0]]);
    const via = new Map<string, string>();
    let frontier: RowRef[] = [drawn];
    const hiddenKeys = new Set<string>();
    for (let h = 1; h <= hops; h++) {
      const next: RowRef[] = [];
      for (const r of frontier) {
        const from = rowKey(r);
        const add = (n: RowRef) => {
          const k = rowKey(n);
          if (!hopOf.has(k)) {
            hopOf.set(k, h);
            via.set(k, from);
            next.push(n);
          }
        };
        for (const p of idx.parentsOf(r)) if (p.parentRow !== null) add({ table: p.parentTable, row: p.parentRow });
        const cap = h === 1 ? CAP_FOCUS : CAP_DEEP;
        let n = 0;
        for (const grp of idx.childrenOf(r)) {
          for (const c of grp.rows) {
            const k = rowKey({ table: grp.childTable, row: c });
            if (hopOf.has(k)) continue;
            if (n >= cap) {
              hiddenKeys.add(k);
              continue;
            }
            add({ table: grp.childTable, row: c });
            n++;
          }
        }
      }
      frontier = next;
    }
    const nodes: GraphNodeSpec[] = [];
    const edges: GraphEdgeSpec[] = [];
    for (const k of hopOf.keys()) {
      const r = parseRowKey(k);
      const h = hopOf.get(k)!;
      nodes.push({ id: k, label: rowLabel(r), table: r.table, slot: slots[r.table] ?? 0, title: `${rowLabel(r)} ${rowCaption(db, r)} · ${h === 0 ? 'centre' : `${h} hop${h === 1 ? '' : 's'}`}` });
      for (const p of idx.parentsOf(r)) {
        if (p.parentRow === null) continue;
        const pk = rowKey({ table: p.parentTable, row: p.parentRow });
        if (hopOf.has(pk)) edges.push({ source: k, target: pk });
      }
    }
    let hidden = 0;
    for (const k of hiddenKeys) if (!hopOf.has(k)) hidden++;
    return { nodes, edges, hidden, hopOf, via };
  }, [db, idx, slots, drawn, hops]);

  // Radial layout in screen px (the graph is drawn at scale ≈ 1): ring h is an ellipse whose size grows
  // with h; its rows sit evenly along it, ordered by the angle of the row they were reached from, so
  // each row stays near its neighbour one ring in. A ring with more rows than fit gets sub-bands.
  const counts = useMemo(() => {
    const c = [0, 0, 0, 0];
    for (const h of g.hopOf.values()) c[h]++;
    return c;
  }, [g]);
  const height = hops === 1 ? (narrow ? 300 : 320) : hops === 2 ? (narrow ? 440 : 440) : narrow ? 560 : 520;
  const layout = useMemo(() => {
    const pad = 34;
    const rxMax = width / 2 - pad;
    const ryMax = height / 2 - pad - 8;
    const pos = new Map<NodeId, { x: number; y: number }>();
    const angle = new Map<string, number>();
    pos.set(rowKey(drawn), { x: 0, y: 0 });
    angle.set(rowKey(drawn), -Math.PI / 2);
    const guides: { rx: number; ry: number; label: string }[] = [];
    for (let h = 1; h <= hops; h++) {
      const ring = [...g.hopOf.entries()].filter(([, x]) => x === h).map(([k]) => k);
      const f = h / hops;
      const rx = rxMax * f;
      const ry = ryMax * f;
      guides.push({ rx, ry, label: `${h} hop${h === 1 ? '' : 's'}` });
      if (!ring.length) continue;
      // order: by the angle of the row each was reached from, then by table, then by row
      ring.sort((a, b) => {
        const da = angle.get(g.via.get(a) ?? '') ?? 0;
        const db_ = angle.get(g.via.get(b) ?? '') ?? 0;
        const na = norm(da);
        const nb = norm(db_);
        if (na !== nb) return na - nb;
        const ra = parseRowKey(a);
        const rb = parseRowKey(b);
        return ra.table === rb.table ? ra.row - rb.row : ra.table < rb.table ? -1 : 1;
      });
      // leave a gap at the top of the ring for its label
      const perimeter = Math.PI * (3 * (rx + ry) - Math.sqrt((3 * rx + ry) * (rx + 3 * ry)));
      const gap = Math.min(0.9, (64 / Math.max(1, perimeter)) * 2 * Math.PI);
      const span = 2 * Math.PI - gap;
      const bands = Math.max(1, Math.min(4, Math.ceil((ring.length * NODE_GAP) / Math.max(1, (perimeter * span) / (2 * Math.PI)))));
      ring.forEach((k, i) => {
        const t = (i + 0.5) / ring.length;
        const a = -Math.PI / 2 + gap / 2 + t * span;
        const band = bands > 1 ? (i % bands) - (bands - 1) / 2 : 0;
        const sx = (rx + band * BAND_GAP) / rx;
        const sy = (ry + band * BAND_GAP) / ry;
        pos.set(k, { x: Math.cos(a) * rx * sx, y: Math.sin(a) * ry * sy });
        angle.set(k, a);
      });
    }
    return { pos, guides };
  }, [g, hops, width, height, drawn]);

  const focusRing = useMemo(() => new Map<NodeId, RingStyle>([[rowKey(drawn), 'accent']]), [drawn]);
  const highlight = useMemo(() => [rowKey(drawn)], [drawn]);

  return (
    <div ref={box} className="min-w-0">
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
        <Segmented
          label="Neighbourhood size"
          value={hopsUi}
          onChange={chooseHops}
          options={[
            { value: 1, label: '1 hop' },
            { value: 2, label: '2 hops' },
            { value: 3, label: '3 hops' },
          ]}
        />
        <span className="num text-[15px] text-ink-2" aria-live="polite">
          {pending
            ? 'laying out the graph…'
            : `${g.nodes.length} rows (${counts
                .slice(1, hops + 1)
                .map((c, i) => `${c} at ${i + 1} hop${i === 0 ? '' : 's'}`)
                .join(', ')}) · ${g.edges.length} links`}
        </span>
      </div>
      <div ref={wrap} className="min-w-0">
        <EntityGraph
          nodes={g.nodes}
          edges={g.edges}
          rings={focusRing}
          highlight={highlight}
          dimOthers={false}
          positions={layout.pos}
          guides={layout.guides}
          onNodeClick={(id) => onFocus(parseRowKey(String(id)))}
          height={height}
          labels={g.nodes.length <= 16 ? 'all' : 'highlighted'}
          roving
          title={`Relational entity graph around ${rowLabel(drawn)}`}
          className={pending ? 'opacity-60 transition-opacity' : 'transition-opacity'}
        />
      </div>
      <p className="!mt-2 !mb-0 text-[15px] leading-snug text-ink-2">
        To keep the picture readable, {rowLabel(drawn)} is drawn with at most {CAP_FOCUS} children and every other row with at most {CAP_DEEP}
        {g.hidden > 0 ? `; ${g.hidden} more children are not drawn` : ''}. That is a display limit, not part of RT.
      </p>
    </div>
  );
});

/** Angle in [0, 2π), measured from the top of the ring (so sorting follows the drawing order). */
function norm(a: number): number {
  const t = a + Math.PI / 2;
  const m = t % (2 * Math.PI);
  return m < 0 ? m + 2 * Math.PI : m;
}
