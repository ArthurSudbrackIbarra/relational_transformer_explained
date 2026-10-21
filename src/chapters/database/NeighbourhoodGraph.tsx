/**
 * The database as a relational entity graph (paper p.16 App. B): one node per
 * row, one edge per FK link, drawn for the k-hop neighbourhood of the focus row.
 * Rows with many children are drawn with a capped number of them (display
 * limit only, stated in the UI). The force layout runs only while the figure is
 * near the viewport, so walking the database higher up the page stays fast.
 */
import { memo, startTransition, useEffect, useMemo, useRef, useState, useTransition } from 'react';
import { EntityGraph, Segmented, type GraphEdgeSpec, type GraphNodeSpec, type NodeId, type RingStyle } from '@/components';
import type { RawDatabase } from '@/engine/types';
import { parseRowKey, rowCaption, rowKey, rowLabel, sameRow, type LinkIndex, type RowRef } from './model';

const CAP_FOCUS = 60;
const CAP_DEEP = 12;

/** True once the element is within `margin` of the viewport (and while it stays there). */
function useNearViewport<T extends Element>(margin: string): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T | null>(null);
  const [near, setNear] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') {
      setNear(true);
      return;
    }
    const io = new IntersectionObserver((entries) => setNear(entries.some((e) => e.isIntersecting)), { rootMargin: margin });
    io.observe(el);
    return () => io.disconnect();
  }, [margin]);
  return [ref, near];
}

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
  // `hopsUi` drives the control at once; the (costly) layout for `hops` follows as a transition.
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

  const g = useMemo(() => {
    const hopOf = new Map<string, number>([[rowKey(drawn), 0]]);
    let frontier: RowRef[] = [drawn];
    const hiddenKeys = new Set<string>();
    for (let h = 1; h <= hops; h++) {
      const next: RowRef[] = [];
      for (const r of frontier) {
        const add = (n: RowRef) => {
          const k = rowKey(n);
          if (!hopOf.has(k)) {
            hopOf.set(k, h);
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
      nodes.push({ id: k, label: rowLabel(r), table: r.table, slot: slots[r.table] ?? 0, title: `${rowLabel(r)} ${rowCaption(db, r)} · hop ${hopOf.get(k)}` });
      for (const p of idx.parentsOf(r)) {
        if (p.parentRow === null) continue;
        const pk = rowKey({ table: p.parentTable, row: p.parentRow });
        if (hopOf.has(pk)) edges.push({ source: k, target: pk });
      }
    }
    let hidden = 0;
    for (const k of hiddenKeys) if (!hopOf.has(k)) hidden++;
    return { nodes, edges, hidden, hopOf };
  }, [db, idx, slots, drawn, hops]);

  const badges = useMemo(() => {
    const m = new Map<NodeId, string>();
    for (const [k, h] of g.hopOf) if (h > 1) m.set(k, String(h));
    return m;
  }, [g]);

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
          {pending ? 'laying out the graph…' : `${g.nodes.length} rows · ${g.edges.length} links`}
        </span>
      </div>
      <EntityGraph
        nodes={g.nodes}
        edges={g.edges}
        rings={focusRing}
        highlight={highlight}
        dimOthers={false}
        badges={badges}
        onNodeClick={(id) => onFocus(parseRowKey(String(id)))}
        height={420}
        labels="highlighted"
        linkDistance={hops === 1 ? 46 : 34}
        charge={hops === 3 ? -40 : -70}
        title={`Relational entity graph around ${rowLabel(drawn)}`}
        className={pending ? 'opacity-60 transition-opacity' : 'transition-opacity'}
      />
      <p className="!mt-2 !mb-0 text-[15px] leading-snug text-ink-2">
        Unmarked nodes are direct neighbours of {rowLabel(drawn)}; a 2 or 3 above a node is its hop distance. To keep the picture readable,{' '}
        {rowLabel(drawn)} is drawn with at most{' '}
        {CAP_FOCUS} children and every other row with at most {CAP_DEEP}
        {g.hidden > 0 ? `; ${g.hidden} more children are not drawn` : ''}. That is a display limit, not part of RT.
      </p>
    </div>
  );
});
