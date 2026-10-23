/**
 * Section 2 summary chart: the fate of every raw value of every table file
 * (100% stacked bars): becomes a cell, becomes a link (FK), primary key, or
 * missing (null feature value or null FK). Counts: raw values from the
 * generator, cells from the engine's preprocessed nodes. The three splits of a
 * task table are drawn as one bar (they share one schema); the table view
 * keeps one row per file.
 *
 * Fates use a neutral gray ramp, not categorical colors: categorical color
 * means table identity on this site (BUILD_PLAN §5), shown by the swatches.
 */
import { useMemo, useState } from 'react';
import { ChartFrame, DataTable, maxTextWidth, textWidth } from '@/components';
import { TooltipBox } from '@/components/viz/chartParts';
import { useElementSize } from '@/components/viz/useSize';
import { catVar } from '@/styles/palette';
import { fateCounts, type FateCounts, type TableFile } from './data';
import type { ChapterData } from './shared';

const nf = (n: number) => n.toLocaleString('en-US');

const KINDS = [
  { id: 'cell', label: 'cell (feature value)', fill: 'var(--ink-2)' },
  { id: 'link', label: 'link (foreign key)', fill: 'var(--muted)' },
  { id: 'pk', label: 'primary key', fill: 'url(#pp-dots)' },
  { id: 'missing', label: 'missing (no cell, no link)', fill: 'url(#pp-hatch)' },
] as const;

type KindId = (typeof KINDS)[number]['id'];

const LABEL_PX = 13;
const NUM_PX = 13.5;
/** Minimum drawn width of a non-empty segment, so a handful of missing values stays visible. */
const MIN_SEG = 6;

interface Row {
  id: string;
  label: string;
  tableName: string;
  files: TableFile[];
  c: FateCounts;
}

function addCounts(a: FateCounts, b: FateCounts): FateCounts {
  return {
    raw: a.raw + b.raw,
    cell: a.cell + b.cell,
    link: a.link + b.link,
    pk: a.pk + b.pk,
    missing: a.missing + b.missing,
    bySem: [a.bySem[0] + b.bySem[0], a.bySem[1] + b.bySem[1], a.bySem[2] + b.bySem[2], a.bySem[3] + b.bySem[3]],
  };
}

const ZERO: FateCounts = { raw: 0, cell: 0, link: 0, pk: 0, missing: 0, bySem: [0, 0, 0, 0] };

/** Segment widths that sum to `barW`, with every non-empty segment at least MIN_SEG wide. */
function segWidths(values: number[], barW: number): number[] {
  const total = values.reduce((s, v) => s + v, 0) || 1;
  const raw = values.map((v) => (v / total) * barW);
  const small = raw.map((w, i) => values[i] > 0 && w < MIN_SEG);
  const need = small.reduce((s, isSmall, i) => s + (isSmall ? MIN_SEG - raw[i] : 0), 0);
  const bigTotal = raw.reduce((s, w, i) => s + (small[i] || values[i] === 0 ? 0 : w), 0) || 1;
  return raw.map((w, i) => (values[i] === 0 ? 0 : small[i] ? MIN_SEG : w - (need * w) / bigTotal));
}

export function CellFatesChart({ data }: { data: ChapterData }) {
  const { files, pdb, slotOf, db } = data;
  const perFile = useMemo(() => files.map((f) => ({ f, c: fateCounts(pdb, f) })), [files, pdb]);
  const rows = useMemo(() => {
    const out: Row[] = [];
    for (const { f, c } of perFile) {
      if (f.tableType === 'Db') {
        out.push({ id: f.key, label: f.tableName, tableName: f.tableName, files: [f], c });
        continue;
      }
      const g = out.find((r) => r.tableName === f.tableName && r.files[0].tableType !== 'Db');
      if (g) {
        g.files.push(f);
        g.c = addCounts(g.c, c);
      } else out.push({ id: `${f.tableName}:task`, label: f.tableName, tableName: f.tableName, files: [f], c });
    }
    for (const r of out) if (r.files[0].tableType !== 'Db') r.label = `${r.tableName} (${r.files.length} splits)`;
    return out;
  }, [perFile]);
  const totals = useMemo(() => rows.reduce((t, r) => addCounts(t, r.c), ZERO), [rows]);

  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(280, size.width || 700);
  const ratioText = (c: FateCounts) => `${nf(c.cell)} / ${nf(c.raw)}`;
  const labelW = Math.ceil(maxTextWidth(rows.map((r) => r.label), LABEL_PX)) + 26;
  const ratioW = Math.ceil(Math.max(textWidth('cells / raw', LABEL_PX), maxTextWidth(rows.map((r) => ratioText(r.c)), NUM_PX))) + 16;
  const missW = Math.ceil(textWidth('missing', LABEL_PX)) + 12;
  // wide: label | bar | cells/raw | missing. Narrow: label and numbers on one line, the bar under it.
  const narrow = W < labelW + ratioW + missW + 200;
  const HEAD_H = narrow ? 0 : 22;
  const BAR_H = 14;
  const ROW_H = narrow ? 44 : 28;
  const barX = narrow ? 0 : labelW;
  const barW = narrow ? W : W - labelW - ratioW - missW;
  const H = HEAD_H + rows.length * ROW_H + 2;
  const [hover, setHover] = useState<number | null>(null);

  const legend = (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[15px] text-ink-2" aria-label="Legend">
      {KINDS.map((k) => (
        <li key={k.id} className="inline-flex items-center gap-1.5">
          <svg width="12" height="12" aria-hidden="true">
            <rect x="0.5" y="0.5" width="11" height="11" rx="3" style={{ fill: k.fill, stroke: k.id === 'pk' || k.id === 'missing' ? 'var(--muted)' : 'none', strokeWidth: 1 }} />
          </svg>
          {k.label}
        </li>
      ))}
    </ul>
  );

  const table = (
    <DataTable
      caption="Fate of every raw value per table file"
      columns={['table file', 'raw values', 'cells', 'links', 'primary key', 'missing']}
      rows={[
        ...perFile.map(({ f, c }) => [f.key, nf(c.raw), nf(c.cell), nf(c.link), nf(c.pk), nf(c.missing)]),
        ['all files', nf(totals.raw), nf(totals.cell), nf(totals.link), nf(totals.pk), nf(totals.missing)],
      ]}
    />
  );

  return (
    <ChartFrame
      title={`What pre.rs makes of every raw value in ${db.title.split(' (')[0]}`}
      subtitle={
        <>
          {nf(totals.raw)} values in {files.length} table files → <strong className="text-ink">{nf(totals.cell)} cells</strong>, {nf(totals.link)} links, {nf(totals.pk)} primary keys
          skipped, {nf(totals.missing)} missing. The splits of a task table share one bar; the table view lists every file.
        </>
      }
      legend={legend}
      table={table}
    >
      <div ref={ref} className="relative w-full">
        <svg width={W} height={H} role="img" aria-label="Stacked bars: share of raw values that become cells, links, primary keys or missing values, per table" className="block">
          <defs>
            <pattern id="pp-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="5" height="5" style={{ fill: 'var(--surface)' }} />
              <line x1="0" y1="0" x2="0" y2="5" style={{ stroke: 'var(--ink-2)' }} strokeWidth="2.2" />
            </pattern>
            {/* primary key: dots in the muted ink (≥ 3:1 against the surface), distinct from the hatched "missing" */}
            <pattern id="pp-dots" width="4" height="4" patternUnits="userSpaceOnUse">
              <rect width="4" height="4" style={{ fill: 'var(--surface)' }} />
              <circle cx="2" cy="2" r="1.15" style={{ fill: 'var(--ink-2)' }} />
            </pattern>
          </defs>
          {!narrow && (
            <g style={{ fill: 'var(--muted)', fontSize: LABEL_PX }}>
              <text x={W - missW - 4} y={13} textAnchor="end">
                cells / raw
              </text>
              <text x={W - 4} y={13} textAnchor="end">
                missing
              </text>
            </g>
          )}
          {rows.map((r, i) => {
            const { c } = r;
            const top = HEAD_H + i * ROW_H;
            const labelY = narrow ? top + 11 : top + ROW_H / 2;
            const barY = narrow ? top + 22 : top + (ROW_H - BAR_H) / 2;
            const vals = KINDS.map((k) => c[k.id as KindId] as number);
            const ws = segWidths(vals, barW);
            let acc = barX;
            const segs = KINDS.map((k, j) => {
              const x0 = acc;
              acc += ws[j];
              return { k, v: vals[j], x0, w: ws[j] };
            }).filter((s) => s.v > 0);
            const pct = Math.round((100 * c.cell) / Math.max(1, c.raw));
            return (
              <g key={r.id} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
                <rect x={0} y={top} width={W} height={ROW_H} rx={4} style={{ fill: hover === i ? 'var(--surface-2)' : 'transparent' }} />
                <rect x={2} y={labelY - 5} width={10} height={10} rx={2.5} style={{ fill: catVar(slotOf(r.tableName)) }} />
                <text x={18} y={labelY} dy="0.34em" style={{ fill: 'var(--ink)', fontSize: LABEL_PX }}>
                  {r.label}
                </text>
                {segs.map((s, j) => (
                  <rect
                    key={s.k.id}
                    x={s.x0 + (j > 0 ? 1 : 0)}
                    y={barY}
                    width={Math.max(1, s.w - (j > 0 ? 1 : 0) - (j < segs.length - 1 ? 1 : 0))}
                    height={BAR_H}
                    rx={j === 0 || j === segs.length - 1 ? 3 : 0}
                    style={{ fill: s.k.fill, stroke: s.k.id === 'pk' || s.k.id === 'missing' ? 'var(--muted)' : 'none', strokeWidth: 1 }}
                  />
                ))}
                {narrow ? (
                  <text x={W - 2} y={labelY} dy="0.34em" textAnchor="end" className="tabular" style={{ fill: 'var(--ink-2)', fontSize: LABEL_PX }}>
                    {pct}% cells{c.missing > 0 ? ` · ${nf(c.missing)} missing` : ''}
                  </text>
                ) : (
                  <>
                    <text x={W - missW - 4} y={top + ROW_H / 2} dy="0.34em" textAnchor="end" className="tabular" style={{ fill: 'var(--ink-2)', fontSize: NUM_PX }}>
                      {ratioText(c)}
                    </text>
                    <text
                      x={W - 4}
                      y={top + ROW_H / 2}
                      dy="0.34em"
                      textAnchor="end"
                      className="tabular"
                      style={{ fill: c.missing > 0 ? 'var(--ink)' : 'var(--muted)', fontSize: NUM_PX, fontWeight: c.missing > 0 ? 600 : 400 }}
                    >
                      {nf(c.missing)}
                    </text>
                  </>
                )}
              </g>
            );
          })}
        </svg>
        {hover !== null && (
          <TooltipBox style={{ left: Math.max(0, Math.min(W - 250, barX + 10)), top: HEAD_H + hover * ROW_H + ROW_H + 2 }}>
            <div className="mb-1 font-mono text-[14px] text-muted">{rows[hover].files.map((f) => f.key).join(', ')}</div>
            {KINDS.map((k) => (
              <div key={k.id} className="flex items-center gap-2 py-px">
                <svg width="10" height="10" aria-hidden="true">
                  <rect x="0.5" y="0.5" width="9" height="9" rx="2" style={{ fill: k.fill, stroke: k.id === 'pk' || k.id === 'missing' ? 'var(--muted)' : 'none', strokeWidth: 1 }} />
                </svg>
                <strong className="num font-semibold text-ink">{nf(rows[hover].c[k.id as KindId] as number)}</strong>
                <span className="text-ink-2">{k.label}</span>
              </div>
            ))}
          </TooltipBox>
        )}
      </div>
    </ChartFrame>
  );
}
