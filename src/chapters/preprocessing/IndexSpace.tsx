/**
 * Section 1 figure: the global node index space (spec §1.5, pre.rs:79-307).
 * A StepPlayer replays phase 1 of pre.rs: files are read in glob order and each
 * one's node_idx_offset is the running row count `num_rows_sum`. A probe turns
 * (table, row position) into a global node index.
 */
import { color as d3color } from 'd3';
import { motion } from 'motion/react';
import { useMemo, useState } from 'react';
import { Select, Slider, StepPlayer, cx, fitText, inkOn, textWidth, useStepPlayer } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { catHex, catVar } from '@/styles/palette';
import { usePrefersReducedMotion, useResolvedTheme } from '@/styles/theme';
import { fileLabel, type TableFile } from './data';
import { PickChip, type ChapterData } from './shared';

const nf = (n: number) => n.toLocaleString('en-US');

/** First two non-key values of a row, for a short human description. */
function rowGist(f: TableFile, row: number): string {
  const rec = f.table.rows[row];
  if (!rec) return '';
  const s = f.table.schema;
  const parts: string[] = [];
  for (const c of s.columns) {
    if (c.name === s.pkey || c.name in s.fkeys) continue;
    const v = rec[c.name];
    if (v === null || v === undefined) continue;
    if (c.dtype === 'datetime') parts.push(`${c.name} ${new Date(v as number).toISOString().slice(0, 10)}`);
    else if (c.dtype === 'str') parts.push(`“${String(v).length > 22 ? String(v).slice(0, 21) + '…' : v}”`);
    else parts.push(`${c.name} ${String(v)}`);
    if (parts.length === 2) break;
  }
  return parts.join(', ');
}

export function IndexSpaceFigure({ data }: { data: ChapterData }) {
  const { files, slotOf, seedNode, entityNode, fileOfNode } = data;
  const mode = useResolvedTheme();
  const reduce = usePrefersReducedMotion();
  const player = useStepPlayer(files.length, { initial: files.length - 1, intervalMs: 1100 });
  const read = player.index;
  const total = files.reduce((s, f) => s + f.numNodes, 0);
  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(280, size.width || 720);
  const PAD = 2;
  const x = (n: number) => PAD + (n / Math.max(1, total)) * (W - 2 * PAD);

  // probe: default = the seed's entity row (forecast) or the seed row (autocomplete)
  const initialNode = entityNode ?? (seedNode >= 0 ? seedNode : 0);
  const initialFile = fileOfNode(initialNode) ?? files[0];
  const [probeKey, setProbeKey] = useState(initialFile.key);
  const [probeRow, setProbeRow] = useState(initialNode - initialFile.offset);
  const probeFile = files.find((f) => f.key === probeKey) ?? files[0];
  const row = Math.min(probeRow, probeFile.numNodes - 1);
  const probeNode = probeFile.offset + row;
  const probeIdx = files.indexOf(probeFile);
  const seedFile = seedNode >= 0 ? fileOfNode(seedNode) : undefined;
  const seedIdx = seedFile ? files.indexOf(seedFile) : -1;
  const [hover, setHover] = useState<number | null>(null);

  const setProbe = (node: number) => {
    const f = fileOfNode(node);
    if (!f) return;
    setProbeKey(f.key);
    setProbeRow(node - f.offset);
  };

  // axis labels at segment starts, skipping collisions (the final total always shows)
  const ticks = useMemo(() => {
    const out: { x: number; v: number; anchor: 'start' | 'middle' | 'end'; left: number; right: number }[] = [];
    const all = [...files.map((f) => f.offset), total];
    const GAP = 10;
    all.forEach((v, i) => {
      const px = x(v);
      const w = textWidth(nf(v), 13);
      const last = i === all.length - 1;
      const anchor = i === 0 ? 'start' : last ? 'end' : 'middle';
      const left = anchor === 'start' ? px : anchor === 'end' ? px - w : px - w / 2;
      if (last) {
        // drop earlier labels until the total has room
        while (out.length && out[out.length - 1].right + GAP > left) out.pop();
      } else if (out.length && left <= out[out.length - 1].right + GAP) return;
      out.push({ x: px, v, anchor, left, right: left + w });
    });
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files, total, W]);

  // pins
  const pins: { node: number; label: string; kind: 'seed' | 'probe'; visible: boolean }[] = [];
  if (seedNode >= 0) pins.push({ node: seedNode, label: `seed #${nf(seedNode)}`, kind: 'seed', visible: seedIdx <= read });
  if (probeNode !== seedNode) pins.push({ node: probeNode, label: `#${nf(probeNode)}`, kind: 'probe', visible: probeIdx <= read });
  const pinX = pins.map((p) => x(p.node + 0.5));
  const pinW = pins.map((p) => Math.ceil(textWidth(p.label, 13, { weight: 600 })) + 18);
  const stagger = pins.length === 2 && Math.abs(pinX[0] - pinX[1]) < (pinW[0] + pinW[1]) / 2 + 8;

  const H_PINS = stagger ? 50 : 30;
  const BAR_Y = H_PINS + 4;
  const BAR_H = 36;
  // brackets under the axis: the DB tables (sorted by name), then each task's splits (test < train < val)
  const BR_Y = BAR_Y + BAR_H + 30;
  const H = BR_Y + 30;
  const groups = useMemo(() => {
    const out: { id: string; label: string; short: string; n0: number; n1: number }[] = [];
    for (const f of files) {
      const id = f.tableType === 'Db' ? 'db' : f.tableName;
      const last = out[out.length - 1];
      if (last && last.id === id) last.n1 = f.offset + f.numNodes;
      else
        out.push({
          id,
          label: id === 'db' ? 'db/*.parquet (sorted by name)' : `tasks/${f.tableName}/*.parquet`,
          short: id === 'db' ? 'db/*.parquet' : f.tableName,
          n0: f.offset,
          n1: f.offset + f.numNodes,
        });
    }
    return out;
  }, [files]);

  const cur = files[read];
  const describe = (i: number) => {
    const f = files[i];
    return (
      <span>
        pre.rs reads <code>{f.path}</code>: {nf(f.numNodes)} rows. {nf(f.offset)} rows were read before it, so its{' '}
        <code>node_idx_offset</code> is {nf(f.offset)} and its rows get indices {nf(f.offset)}–{nf(f.offset + f.numNodes - 1)}.{' '}
        <span className="text-ink-2">
          <code>num_rows_sum</code> becomes {nf(f.offset + f.numNodes)}.
        </span>
      </span>
    );
  };

  return (
    <div className="min-w-0">
      <StepPlayer {...player} unit="file" label="Replay pre.rs reading the table files" describe={describe} />

      <div ref={ref} className="relative mt-5 w-full">
        <svg width={W} height={H} role="img" aria-label={`Global node index space: ${files.length} table files, ${nf(total)} rows`} className="block overflow-visible">
          {/* segments */}
          {files.map((f, i) => {
            const x0 = x(f.offset);
            const x1 = x(f.offset + f.numNodes);
            const w = Math.max(1.5, x1 - x0 - 2);
            const isRead = i <= read;
            const slot = slotOf(f.tableName);
            const label = fileLabel(f);
            const mono = { mono: true, weight: 600 };
            const fits = w > textWidth(label, 13.5, mono) + 12;
            const twoLine = !fits && f.tableType !== 'Db' && w > Math.max(textWidth(f.tableName, 13, mono), textWidth(f.tableType, 13, mono)) + 10;
            const shortFits = !fits && !twoLine && f.tableType !== 'Db' && w > textWidth(f.tableType, 13.5, mono) + 10;
            // tiny task splits: initials (Te / Tr / V), full name in the tooltip
            const initials = f.tableType === 'Test' ? 'Te' : f.tableType === 'Train' ? 'Tr' : 'V';
            const initFits = !fits && !twoLine && !shortFits && f.tableType !== 'Db' && w > textWidth(initials, 13, mono) + 4;
            const rgb = d3color(catHex(slot, mode))?.rgb();
            const ink = rgb ? inkOn([rgb.r, rgb.g, rgb.b]) : 'var(--ink)';
            return (
              <g
                key={f.key}
                onPointerEnter={() => setHover(i)}
                onPointerLeave={() => setHover(null)}
                onClick={() => setProbe(f.offset + Math.min(row, f.numNodes - 1))}
                style={{ cursor: 'pointer' }}
              >
                <rect
                  x={x0 + 0.5}
                  y={BAR_Y + 0.5}
                  width={Math.max(0.5, w - 1)}
                  height={BAR_H - 1}
                  rx={3}
                  style={{ fill: 'var(--surface-2)', stroke: 'var(--axis)', strokeDasharray: '3 3', strokeWidth: 1 }}
                />
                {isRead && (
                  <motion.rect
                    x={x0}
                    y={BAR_Y}
                    height={BAR_H}
                    rx={3}
                    initial={reduce ? false : { width: 0 }}
                    animate={{ width: w }}
                    transition={reduce ? { duration: 0 } : { duration: 0.45, ease: 'easeOut' }}
                    style={{ fill: catVar(slot) }}
                  />
                )}
                {i === read && <rect x={x0 - 1.5} y={BAR_Y - 1.5} width={w + 3} height={BAR_H + 3} rx={4} fill="none" style={{ stroke: 'var(--ink)' }} strokeWidth={1.5} />}
                {isRead && initFits && (
                  <text x={x0 + w / 2} y={BAR_Y + BAR_H / 2} dy="0.34em" textAnchor="middle" style={{ fill: ink, fontSize: 13, fontWeight: 600, pointerEvents: 'none' }} className="font-mono">
                    {initials}
                  </text>
                )}
                {isRead && (fits || shortFits) && (
                  <text x={x0 + w / 2} y={BAR_Y + BAR_H / 2} dy="0.34em" textAnchor="middle" style={{ fill: ink, fontSize: 13.5, fontWeight: 600, pointerEvents: 'none' }} className="font-mono">
                    {fits ? label : f.tableType}
                  </text>
                )}
                {isRead && twoLine && (
                  <text x={x0 + w / 2} y={BAR_Y + BAR_H / 2} textAnchor="middle" style={{ fill: ink, fontSize: 13, fontWeight: 600, pointerEvents: 'none' }} className="font-mono">
                    <tspan x={x0 + w / 2} dy="-0.2em">
                      {f.tableName}
                    </tspan>
                    <tspan x={x0 + w / 2} dy="1.12em">
                      {f.tableType}
                    </tspan>
                  </text>
                )}
              </g>
            );
          })}
          {/* axis ticks */}
          {ticks.map((t) => (
            <g key={t.v} transform={`translate(${t.x},${BAR_Y + BAR_H})`}>
              <line y1={2} y2={7} style={{ stroke: 'var(--axis)' }} />
              <text y={19} textAnchor={t.anchor} className="tabular" style={{ fill: 'var(--muted)', fontSize: 13 }}>
                {nf(t.v)}
              </text>
            </g>
          ))}
          {/* brackets: DB tables first, then each task's three splits */}
          {(() => {
            // labels stay inside their bracket (long form, then short, then cut), and never overlap the previous one
            let lastRight = -Infinity;
            return groups.map((g) => {
            const gx0 = x(g.n0) + 1;
            const gx1 = x(g.n1) - 2;
            const gw = Math.max(0, gx1 - gx0);
            const room = Math.max(0, gw);
            const text = textWidth(g.label, 13, { mono: true }) <= room ? g.label : fitText(g.short, room, 13, { mono: true });
            const tw = textWidth(text, 13, { mono: true });
            const tx = Math.max(lastRight + 8, Math.max(0, Math.min(W - tw, gx0 + gw / 2 - tw / 2)));
            const show = gw > 14 && text !== '…' && tx + tw <= W;
            if (show) lastRight = tx + tw;
            return (
              <g key={g.id}>
                <title>{g.label}</title>
                <path d={`M${gx0},${BR_Y - 6} V${BR_Y} H${gx1} V${BR_Y - 6}`} fill="none" style={{ stroke: 'var(--muted)' }} strokeWidth={1.25} />
                {show && (
                  <text x={tx} y={BR_Y + 16} className="font-mono" style={{ fill: 'var(--ink-2)', fontSize: 13 }}>
                    {text}
                  </text>
                )}
              </g>
            );
            });
          })()}
          {/* pins */}
          {pins.map((p, i) => {
            const px = pinX[i];
            const ly = stagger && i === 1 ? 4 : stagger ? 24 : 4;
            const lw = pinW[i];
            const lx = Math.max(0, Math.min(W - lw, px - lw / 2));
            const isSeed = p.kind === 'seed';
            return (
              <motion.g key={p.kind} initial={false} animate={{ opacity: p.visible ? 1 : 0.25 }} transition={{ duration: 0.3 }}>
                <line x1={px} x2={px} y1={ly + 18} y2={BAR_Y + BAR_H + 2} style={{ stroke: isSeed ? 'var(--accent)' : 'var(--ink)' }} strokeWidth={1.5} />
                <rect
                  x={lx}
                  y={ly}
                  width={lw}
                  height={18}
                  rx={9}
                  style={{ fill: isSeed ? 'var(--accent)' : 'var(--surface)', stroke: isSeed ? 'none' : 'var(--border-strong)' }}
                />
                <text x={lx + lw / 2} y={ly + 9} dy="0.34em" textAnchor="middle" className="tabular" style={{ fill: isSeed ? 'var(--accent-ink)' : 'var(--ink)', fontSize: 13, fontWeight: 600 }}>
                  {p.label}
                </text>
              </motion.g>
            );
          })}
        </svg>
        {hover !== null && (
          <div
            role="tooltip"
            className="pointer-events-none absolute z-20 rounded-[9px] border border-border bg-surface px-2.5 py-2 text-[14.5px] leading-tight shadow-[var(--shadow-pop)]"
            style={{ left: Math.max(0, Math.min(W - 230, x(files[hover].offset) - 20)), top: BR_Y + 22 }}
          >
            <div className="font-mono text-[14px] text-muted">{files[hover].path}</div>
            <div className="mt-1 text-ink">
              <strong className="num">{nf(files[hover].numNodes)}</strong> rows, nodes{' '}
              <span className="num">
                {nf(files[hover].offset)}–{nf(files[hover].offset + files[hover].numNodes - 1)}
              </span>
            </div>
            {hover > read && <div className="mt-1 text-muted">not read yet</div>}
          </div>
        )}
      </div>

      {/* table_info.json */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full border-collapse text-[15px]">
          <caption className="mb-1.5 text-left text-[15px] font-medium text-ink-2">
            <code>table_info.json</code>, as written by pre.rs (key → offset, count)
          </caption>
          <thead>
            <tr className="border-b border-grid text-left text-ink-2">
              <th scope="col" className="py-1.5 pr-2 font-semibold">
                key
              </th>
              <th scope="col" className="hidden py-1.5 pr-2 font-semibold sm:table-cell">
                file read
              </th>
              <th scope="col" className="py-1.5 pr-2 text-right font-semibold">
                <span className="hidden sm:inline">node_idx_offset</span>
                <span className="sm:hidden" title="node_idx_offset">
                  offset
                </span>
              </th>
              <th scope="col" className="py-1.5 text-right font-semibold">
                <span className="hidden sm:inline">num_nodes</span>
                <span className="sm:hidden" title="num_nodes">
                  rows
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {files.map((f, i) => (
              <tr
                key={f.key}
                className={cx(
                  'cursor-pointer border-b border-grid/70 transition-colors hover:bg-surface-2',
                  i > read && 'text-muted opacity-60',
                  f.key === probeKey && 'bg-accent-soft',
                )}
                onClick={() => setProbe(f.offset + Math.min(row, f.numNodes - 1))}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setProbe(f.offset + Math.min(row, f.numNodes - 1));
                  }
                }}
                tabIndex={0}
                aria-label={`${f.key}: offset ${f.offset}, ${f.numNodes} rows. Press Enter to probe this file.`}
                onPointerEnter={() => setHover(i)}
                onPointerLeave={() => setHover(null)}
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
              >
                <td className="py-1 pr-2">
                  <span className="inline-flex items-center gap-1.5">
                    <span aria-hidden="true" className="inline-block h-2.5 w-2.5 shrink-0 rounded-[3px]" style={{ background: catVar(slotOf(f.tableName)) }} />
                    <span className="font-mono whitespace-nowrap">{f.key}</span>
                  </span>
                </td>
                <td className="hidden py-1 pr-2 font-mono text-ink-2 sm:table-cell">{f.path}</td>
                <td className="num py-1 pr-2 text-right">{nf(f.offset)}</td>
                <td className="num py-1 text-right">{nf(f.numNodes)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* probe */}
      <div className="mt-5 rounded-xl border border-border bg-surface-2/60 px-4 py-3.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[15px] font-semibold text-ink">Find a row’s index</span>
          {entityNode !== null && (
            <PickChip active={probeNode === entityNode} onClick={() => setProbe(entityNode)}>
              the seed’s {fileOfNode(entityNode)?.tableName.replace(/s$/, '') ?? 'entity'}
            </PickChip>
          )}
          {seedNode >= 0 && (
            <PickChip active={probeNode === seedNode} onClick={() => setProbe(seedNode)}>
              the seed row
            </PickChip>
          )}
        </div>
        <div className="mt-3 grid gap-3 sm:grid-cols-[minmax(0,220px)_minmax(0,1fr)]">
          <Select label="Table file" value={probeKey} onChange={(e) => setProbeKey(e.target.value)}>
            {files.map((f) => (
              <option key={f.key} value={f.key}>
                {f.key}
              </option>
            ))}
          </Select>
          <Slider label="Row position" value={row} min={0} max={probeFile.numNodes - 1} onChange={setProbeRow} format={(v) => `row ${v}`} />
        </div>
        <div className="mt-3 flex flex-wrap items-baseline gap-x-3 gap-y-1 text-[17px]">
          <span className="num text-ink">
            node_idx = <span className="text-ink-2">offset(</span>
            {probeFile.key}
            <span className="text-ink-2">)</span> + row = {nf(probeFile.offset)} + {nf(row)} = <strong className="text-accent-text">{nf(probeNode)}</strong>
          </span>
          <span className="text-[15.5px] text-ink-2">{rowGist(probeFile, row)}</span>
        </div>
        {cur && probeIdx > read && <div className="mt-1 text-[14.5px] text-muted">This file has not been read yet in the replay above.</div>}
      </div>
    </div>
  );
}
