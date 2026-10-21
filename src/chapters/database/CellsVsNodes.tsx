/**
 * Two views of the same rows. Graph view (relational deep learning, paper p.16
 * App. B): one node per row, one edge per PK–FK link. Cell view (RT): every
 * non-missing feature cell is its own token (v, c, t) (p.4 §3.1); key cells
 * become no token, the links survive as structure (p.16 App. B; spec §1.2).
 */
import { AnimatePresence, motion } from 'motion/react';
import { memo, useId, useMemo, useState } from 'react';
import { Segmented, formatDate, formatRaw, truncate } from '@/components';
import { shapeOfSlot, shapePath } from '@/components/viz/shapes';
import { useElementSize } from '@/components/viz/useSize';
import type { RawDatabase } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { childrenForDisplay, classifyRow, rowKey, rowLabel, type LinkIndex, type RowRef } from './model';

type View = 'graph' | 'cells';

const CW_MAX = 176; // cluster width in cell view (shrinks to CW_MIN on narrow screens; text never does)
const CW_MIN = 140;
const CHIP_H = 40;
const HEAD_H = 24;
const GAP_Y = 46;
/** Chip text sizes (px): glyph and column name, value. */
const NAME_PX = 13;
const VALUE_PX = 14;
/** Approximate advance of the monospace font, in em. */
const MONO_EM = 0.62;

interface Cluster {
  ref: RowRef;
  level: number;
  cells: { col: string; text: string; sem: string }[];
  nKeys: number;
  nMissing: number;
}

const SEM_GLYPH: Record<string, string> = { int: '#', float: '#', str: 'Aa', datetime: '◷', bool: '◐' };

export const CellsVsNodes = memo(function CellsVsNodes({ db, idx, slots, focus }: { db: RawDatabase; idx: LinkIndex; slots: Record<string, number>; focus: RowRef }) {
  const uid = useId().replace(/:/g, '');
  const [view, setView] = useState<View>('graph');
  const [wrap, size] = useElementSize<HTMLDivElement>();
  const width = Math.max(300, size.width || 720);
  const perLevel = Math.max(2, Math.min(6, Math.floor((width - 16) / (CW_MAX + 14))));
  const CW = Math.max(CW_MIN, Math.min(CW_MAX, Math.floor((width - 16) / perLevel) - 14));
  const nameChars = Math.max(8, Math.floor((CW - 12 - 32) / (NAME_PX * MONO_EM)));
  const valueChars = Math.max(8, Math.floor((CW - 12 - 14) / (VALUE_PX * MONO_EM)));

  const { clusters, edges, more } = useMemo(() => {
    const mk = (ref: RowRef, level: number): Cluster => {
      const t = idx.table(ref.table)!;
      const rec = t.rows[ref.row] ?? {};
      const cls = classifyRow(t, ref.row);
      return {
        ref,
        level,
        cells: cls.tokens.map((c) => {
          const dt = t.schema.columns.find((x) => x.name === c)!.dtype;
          const v = dt === 'datetime' && typeof rec[c] === 'number' ? formatDate(rec[c] as number, 'never') : formatRaw(rec[c], dt);
          return { col: c, text: v, sem: SEM_GLYPH[dt] ?? '#' };
        }),
        nKeys: cls.pk.length + cls.links.length,
        nMissing: cls.missing.length,
      };
    };
    const parents = idx
      .parentsOf(focus)
      .filter((p) => p.parentRow !== null)
      .map((p) => ({ table: p.parentTable, row: p.parentRow! }));
    const groups = idx.childrenOf(focus);
    const total = groups.reduce((s, g) => s + g.rows.length, 0);
    const children = childrenForDisplay(groups, perLevel);
    const cl: Cluster[] = [];
    let level = 0;
    if (parents.length) {
      parents.slice(0, perLevel).forEach((p) => cl.push(mk(p, level)));
      level++;
    }
    cl.push(mk(focus, level));
    const focusLevel = level;
    if (children.length) children.forEach((c) => cl.push(mk(c, focusLevel + 1)));
    const ed: { child: string; parent: string }[] = [];
    for (const c of cl) {
      for (const p of idx.parentsOf(c.ref)) {
        if (p.parentRow === null) continue;
        const pk = rowKey({ table: p.parentTable, row: p.parentRow });
        if (cl.some((x) => rowKey(x.ref) === pk)) ed.push({ child: rowKey(c.ref), parent: pk });
      }
    }
    return { clusters: cl, edges: ed, more: total - children.length };
  }, [idx, focus, perLevel]);

  // layout
  const nLevels = Math.max(...clusters.map((c) => c.level)) + 1;
  const levelH = (l: number, v: View) => {
    if (v === 'graph') return 74;
    const maxCells = Math.max(1, ...clusters.filter((c) => c.level === l).map((c) => c.cells.length));
    return HEAD_H + maxCells * (CHIP_H + 4) + 16 + GAP_Y;
  };
  const levelY = (l: number, v: View) => {
    let y = 30;
    for (let i = 0; i < l; i++) y += levelH(i, v);
    return y;
  };
  const H = levelY(nLevels, view) - (view === 'cells' ? GAP_Y - 10 : 20);
  const pos = new Map<string, { x: number; y: number }>();
  for (let l = 0; l < nLevels; l++) {
    const row = clusters.filter((c) => c.level === l);
    row.forEach((c, i) => pos.set(rowKey(c.ref), { x: (width * (i + 0.5)) / row.length, y: levelY(l, view) }));
  }

  const nTokens = clusters.reduce((s, c) => s + c.cells.length, 0);
  const nKeys = clusters.reduce((s, c) => s + c.nKeys, 0);
  const nMissing = clusters.reduce((s, c) => s + c.nMissing, 0);
  const spring = { type: 'spring' as const, stiffness: 170, damping: 24 };

  return (
    <div className="min-w-0">
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
        <Segmented
          label="Representation"
          value={view}
          onChange={setView}
          options={[
            { value: 'graph', label: 'Rows as nodes (graph view)' },
            { value: 'cells', label: 'Cells as tokens (RT)' },
          ]}
        />
      </div>
      <div ref={wrap} className="relative w-full overflow-hidden rounded-xl border border-border bg-surface-2/40">
        <motion.svg
          width={width}
          animate={{ height: H }}
          initial={false}
          transition={spring}
          role="img"
          aria-label={
            view === 'graph'
              ? `Graph view: ${clusters.length} rows as nodes and ${edges.length} links as edges.`
              : `Cell view: the same ${clusters.length} rows give ${nTokens} cell tokens; ${nKeys} key cells give no token.`
          }
          className="block"
        >
          <defs>
            <marker id={`${uid}-tip`} viewBox="0 0 8 8" refX="7.2" refY="4" markerWidth={7} markerHeight={7} markerUnits="userSpaceOnUse" orient="auto">
              <path d="M0,0.6 L7.6,4 L0,7.4 Z" style={{ fill: 'var(--axis)' }} />
            </marker>
          </defs>
          {edges.map((e) => {
            const a = pos.get(e.child)!;
            const b = pos.get(e.parent)!;
            const ca = clusters.find((c) => rowKey(c.ref) === e.parent)!;
            // graph: centre to centre (shortened); cells: child header top → parent cluster bottom
            const y1 = view === 'graph' ? a.y - 13 : a.y - 2;
            const y2 = view === 'graph' ? b.y + 14 : b.y + HEAD_H + ca.cells.length * (CHIP_H + 4) + 6;
            return (
              <motion.line
                key={`${e.child}->${e.parent}`}
                initial={false}
                animate={{ x1: a.x, y1, x2: b.x, y2 }}
                transition={spring}
                style={{ stroke: 'var(--axis)' }}
                strokeWidth={1.5}
                strokeDasharray={view === 'cells' ? '4 3' : undefined}
                markerEnd={`url(#${uid}-tip)`}
              />
            );
          })}
          {clusters.map((c) => {
            const k = rowKey(c.ref);
            const p = pos.get(k)!;
            const slot = slots[c.ref.table] ?? 0;
            const isFocus = k === rowKey(focus);
            return (
              <motion.g key={k} initial={false} animate={{ x: p.x, y: p.y }} transition={spring}>
                <AnimatePresence initial={false}>
                  {view === 'graph' ? (
                    <motion.g key="node" initial={{ opacity: 0, scale: 0.4 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.4 }}>
                      {isFocus && <circle r={16} fill="none" style={{ stroke: 'var(--accent)' }} strokeWidth={2} />}
                      <path d={shapePath(shapeOfSlot(slot), 10)} style={{ fill: catVar(slot), stroke: 'var(--surface)' }} strokeWidth={2} paintOrder="stroke" />
                      <text
                        y={29}
                        textAnchor="middle"
                        className="font-mono"
                        style={{ fill: 'var(--ink)', fontSize: 13, stroke: 'var(--surface)', strokeWidth: 4, paintOrder: 'stroke', strokeLinejoin: 'round' }}
                      >
                        {rowLabel(c.ref)}
                      </text>
                    </motion.g>
                  ) : (
                    <motion.g key="cells" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                      <rect
                        x={-CW / 2}
                        y={0}
                        width={CW}
                        height={HEAD_H}
                        rx={6}
                        style={{ fill: 'var(--surface)', stroke: isFocus ? 'var(--accent)' : catVar(slot) }}
                        strokeWidth={isFocus ? 2 : 1.5}
                      />
                      <text x={0} y={HEAD_H / 2 + 0.5} textAnchor="middle" dominantBaseline="central" className="font-mono" style={{ fill: 'var(--ink)', fontSize: 13, fontWeight: 600 }}>
                        {rowLabel(c.ref)}
                      </text>
                      {c.cells.map((cell, i) => (
                        <motion.g
                          key={cell.col}
                          initial={{ opacity: 0, y: 0, scale: 0.6 }}
                          animate={{ opacity: 1, y: HEAD_H + 4 + i * (CHIP_H + 4), scale: 1 }}
                          exit={{ opacity: 0, y: 0 }}
                          transition={{ ...spring, delay: 0.05 + i * 0.05 }}
                        >
                          <title>{`token (v, c, t) = (${cell.text}, ${cell.col}, ${c.ref.table})`}</title>
                          <rect x={-CW / 2 + 6} y={0} width={CW - 12} height={CHIP_H} rx={4} style={{ fill: 'var(--surface)', stroke: 'var(--border-strong)' }} strokeWidth={1} />
                          <rect x={-CW / 2 + 6} y={0} width={3} height={CHIP_H} rx={1.5} style={{ fill: catVar(slot) }} />
                          <text x={-CW / 2 + 14} y={12} dominantBaseline="central" style={{ fill: 'var(--ink-2)', fontSize: NAME_PX, fontWeight: 700 }}>
                            {cell.sem}
                          </text>
                          <text x={-CW / 2 + 40} y={12} dominantBaseline="central" className="font-mono" style={{ fill: 'var(--ink-2)', fontSize: NAME_PX }}>
                            {truncate(cell.col, nameChars)}
                          </text>
                          <text x={-CW / 2 + 14} y={29} dominantBaseline="central" className="font-mono" style={{ fill: 'var(--ink)', fontSize: VALUE_PX }}>
                            {truncate(cell.text, valueChars)}
                          </text>
                        </motion.g>
                      ))}
                    </motion.g>
                  )}
                </AnimatePresence>
              </motion.g>
            );
          })}
        </motion.svg>
      </div>
      <p className="!mt-2.5 !mb-0 text-[15.5px] leading-snug text-ink-2" aria-live="polite">
        {view === 'graph' ? (
          <>
            <strong className="text-ink">{clusters.length} rows → {clusters.length} nodes, {edges.length} edges.</strong> Each row is one node and
            each primary–foreign key link one edge.
          </>
        ) : (
          <>
            <strong className="text-ink">
              {clusters.length} rows → {nTokens} tokens.
            </strong>{' '}
            One token per non-missing feature cell. The {nKeys} key cells give no token; the dashed links survive only as structure
            {nMissing > 0 ? `, and ${nMissing} missing cell${nMissing === 1 ? '' : 's'} give nothing at all` : ''}.
          </>
        )}
        {more > 0 && (
          <span>
            {' '}
            ({more} more children of {rowLabel(focus)} are left out of this small picture.)
          </span>
        )}
      </p>
    </div>
  );
});
