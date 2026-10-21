/**
 * Ego view of the context graph for the inspected token's row: its parents (the rows in its
 * f2p_nbr_idxs that are in the context) on top, the row itself in the middle, and the context rows
 * that list it as a parent at the bottom. Arrows go child → parent, as foreign keys do (spec §1.3).
 * Fixed height and node size, labels always shown (short row tags; the full names are listed under
 * the figure by the caller). Display only: nothing here is read by the model.
 */
import { useId } from 'react';
import { cx } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { catVar } from '@/styles/palette';

export interface EgoNode {
  id: number;
  /** Short tag drawn under the node, e.g. "c35". */
  tag: string;
  /** Full name for the accessible label and tooltip, e.g. "customers[35] · Anna Müller". */
  name: string;
  slot: number;
}

const R = 9;
const ROW_Y = [38, 90, 138];
const HEIGHT = 172;
const LABEL_PX = 14;

export function EgoGraph({
  parents,
  row,
  children,
  hiddenChildren,
  seed,
  onPick,
}: {
  parents: EgoNode[];
  row: EgoNode;
  children: EgoNode[];
  hiddenChildren: number;
  /** Node id of the seed row (dashed ring). */
  seed: number;
  onPick: (id: number) => void;
}) {
  const uid = useId().replace(/:/g, '');
  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(240, size.width || 480);
  const xs = (n: number) => Array.from({ length: n }, (_, i) => ((i + 1) * W) / (n + 1));
  const px = xs(parents.length);
  const cxs = xs(children.length + (hiddenChildren > 0 ? 1 : 0));
  const mid = { x: W / 2, y: ROW_Y[1] };

  const node = (n: EgoNode, x: number, y: number, kind: 'row' | 'parent' | 'child') => (
    <g
      key={`${kind}-${n.id}`}
      role="button"
      tabIndex={0}
      aria-label={`${kind === 'row' ? 'this row' : kind}: ${n.name}. Inspect its first token`}
      className="cursor-pointer outline-none [&:focus-visible>circle.focus]:stroke-accent"
      onClick={() => onPick(n.id)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onPick(n.id);
        }
      }}
    >
      <title>{n.name}</title>
      <circle className="focus" cx={x} cy={y} r={R + 5} fill="none" stroke="transparent" strokeWidth={2} />
      {n.id === seed && <circle cx={x} cy={y} r={R + 4} fill="none" style={{ stroke: 'var(--accent)' }} strokeWidth={1.5} strokeDasharray="3 2.5" />}
      <circle
        cx={x}
        cy={y}
        r={R}
        style={{ fill: catVar(n.slot), stroke: kind === 'row' ? 'var(--ink)' : 'var(--surface)' }}
        strokeWidth={kind === 'row' ? 2.5 : 1.5}
        opacity={kind === 'child' ? 0.75 : 1}
      />
      {/* labels away from the arrows: parents above, children below, the row itself to the right */}
      <text
        x={kind === 'row' ? x + R + 7 : x}
        y={kind === 'parent' ? y - R - 6 : kind === 'row' ? y + 5 : y + R + LABEL_PX + 1}
        textAnchor={kind === 'row' ? 'start' : 'middle'}
        style={{ fontSize: LABEL_PX, fontFamily: 'var(--font-mono)', fontWeight: kind === 'row' ? 700 : 500, fill: kind === 'child' ? 'var(--ink-2)' : 'var(--ink)' }}
      >
        {n.tag}
      </text>
    </g>
  );
  // An arrow from (x1, y1) to (x2, y2), stopping at the target circle.
  const arrow = (x1: number, y1: number, x2: number, y2: number, lit: boolean, key: string) => {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    return (
      <line
        key={key}
        x1={x1 + ux * (R + 2)}
        y1={y1 + uy * (R + 2)}
        x2={x2 - ux * (R + 4)}
        y2={y2 - uy * (R + 4)}
        style={{ stroke: lit ? 'var(--accent)' : 'var(--axis)' }}
        strokeWidth={lit ? 2 : 1.4}
        markerEnd={`url(#${uid}-${lit ? 'lit' : 'dim'})`}
      />
    );
  };

  return (
    <div ref={ref} className="w-full">
      <svg width={W} height={HEIGHT} viewBox={`0 0 ${W} ${HEIGHT}`} role="group" aria-label="The inspected row, its parent rows and its child rows in the context" className="block">
        <defs>
          {(['lit', 'dim'] as const).map((k) => (
            <marker key={k} id={`${uid}-${k}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" style={{ fill: k === 'lit' ? 'var(--accent)' : 'var(--axis)' }} />
            </marker>
          ))}
        </defs>
        {parents.map((p, i) => arrow(mid.x, mid.y, px[i], ROW_Y[0], true, `e-p-${p.id}`))}
        {children.map((c, i) => arrow(cxs[i], ROW_Y[2], mid.x, mid.y, false, `e-c-${c.id}`))}
        {parents.length === 0 && (
          <text x={W / 2} y={ROW_Y[0] + 5} textAnchor="middle" style={{ fontSize: 14, fill: 'var(--ink-2)' }}>
            no parent in the context
          </text>
        )}
        {parents.map((p, i) => node(p, px[i], ROW_Y[0], 'parent'))}
        {node(row, mid.x, mid.y, 'row')}
        {children.map((c, i) => node(c, cxs[i], ROW_Y[2], 'child'))}
        {hiddenChildren > 0 && (
          <text x={cxs[cxs.length - 1]} y={ROW_Y[2] + 5} textAnchor="middle" style={{ fontSize: 14, fill: 'var(--ink-2)' }}>
            +{hiddenChildren} more
          </text>
        )}
        {children.length === 0 && (
          <text x={W / 2} y={ROW_Y[2] + 5} textAnchor="middle" style={{ fontSize: 14, fill: 'var(--ink-2)' }}>
            no child row in the context
          </text>
        )}
      </svg>
      <div className={cx('sr-only')}>
        Parents: {parents.map((p) => p.name).join(', ') || 'none'}. Children: {children.map((c) => c.name).join(', ') || 'none'}.
      </div>
    </div>
  );
}
