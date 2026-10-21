/**
 * Mini-figures for "Three ideas" (paper p.2 "Our contribution"), all on the live running example:
 *  - CellCards: what becomes of every cell of the seed row and of its entity's row (spec §1.2, §3.5, §5.3);
 *  - LabelTimeline: the entity's task rows over time, which of them are in the context, and the seed
 *    (spec §2.5, §4.6);
 *  - MaskRows: the masked cell's row of each relational attention mask (spec §6.2).
 */
import { scaleTime, timeMonth, timeFormat } from 'd3';
import { CircleSlash, KeyRound, Link2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import { LayerLabel, SemBadge, TableSwatch, formatRaw, LAYER_INFO } from '@/components';
import { textWidth } from '@/components/viz/text';
import { useElementSize } from '@/components/viz/useSize';
import { cx } from '@/components/ui/cx';
import { ATTN_ORDER, type ColumnSchema, type RawValue, type TableSchema } from '@/engine/types';
import { catVar } from '@/styles/palette';
import type { OverviewLive } from './live';
import { int, rawRow, rowLabel, singular } from './live';

// ─────────────────────────────────────────────────────────────────────────────
// 1 · Cells → tokens
// ─────────────────────────────────────────────────────────────────────────────

type CellKind = 'pk' | 'fk' | 'null' | 'feature' | 'target' | 'dropped' | 'cut';

interface CellCard {
  col: ColumnSchema;
  kind: CellKind;
  raw: RawValue;
  phrase?: string;
  sem?: number;
  link?: string;
}

function rowCards(live: OverviewLive, node: number): { schema: TableSchema; cards: CellCard[] } | null {
  const { db, pdb, task, ctx } = live;
  const n = pdb.nodes[node];
  if (!n) return null;
  const schema =
    n.tableType === 'Db'
      ? db.tables.find((t) => t.schema.name === n.tableName)?.schema
      : db.tasks.find((t) => t.kind === 'forecast' && t.name === n.tableName)?.tables?.[n.tableType]?.schema;
  const rec = rawRow(db, pdb, node);
  if (!schema || !rec) return null;
  const isSeed = node === ctx.params.seedNode;
  const inCtx = new Set(ctx.tokens.filter((t) => !t.isPadding && t.nodeIdx === node).map((t) => t.colName));
  const rowInCtx = inCtx.size > 0;
  const cards: CellCard[] = schema.columns.map((col) => {
    const raw = rec[col.name] ?? null;
    if (col.name === schema.pkey) return { col, kind: 'pk', raw };
    if (col.name in schema.fkeys) {
      const parentTable = schema.fkeys[col.name];
      return {
        col,
        kind: raw === null ? 'null' : 'fk',
        raw,
        link: raw === null ? undefined : `${singular(parentTable)} ${raw}`,
      };
    }
    const cell = n.cells.find((c) => c.colName === col.name);
    if (!cell) return { col, kind: 'null', raw };
    const target = isSeed && col.name === task.targetCol;
    const phrase = pdb.text[cell.colNameIdx];
    // a cell of a row in the context that has no token: a leakage column dropped on the fly (spec §4.9), or cut at L (§4.11)
    if (!target && rowInCtx && !inCtx.has(col.name))
      return {
        col,
        kind: task.columnsToDrop.includes(col.name) ? 'dropped' : 'cut',
        raw,
        phrase,
        sem: cell.sem,
      };
    return {
      col,
      kind: target ? 'target' : 'feature',
      raw,
      phrase,
      sem: cell.sem,
    };
  });
  return { schema, cards };
}

function Outcome({ c }: { c: CellCard }) {
  if (c.kind === 'pk')
    return (
      <span className="inline-flex items-center gap-1.5 text-ink-2">
        <KeyRound aria-hidden="true" size={14} /> no token: primary key
      </span>
    );
  if (c.kind === 'fk')
    return (
      <span className="inline-flex items-center gap-1.5 text-ink-2">
        <Link2 aria-hidden="true" size={14} /> no token: a link to {c.link}
      </span>
    );
  if (c.kind === 'null')
    return (
      <span className="inline-flex items-center gap-1.5 text-ink-2">
        <CircleSlash aria-hidden="true" size={14} /> no token: missing
      </span>
    );
  if (c.kind === 'dropped')
    return (
      <span className="inline-flex items-center gap-1.5 text-ink-2">
        <CircleSlash aria-hidden="true" size={14} /> no token here: a leakage column, dropped for this task
      </span>
    );
  if (c.kind === 'cut')
    return (
      <span className="inline-flex items-center gap-1.5 text-ink-2">
        <CircleSlash aria-hidden="true" size={14} /> no token here: the window was full
      </span>
    );
  return (
    <span className="flex min-w-0 items-start gap-1.5">
      <SemBadge sem={c.sem ?? 0} size="xs" />
      <span className="min-w-0 font-mono text-[14px] break-words text-ink">“{c.phrase}”</span>
    </span>
  );
}

export function CellCards({ live }: { live: OverviewLive }) {
  const rows = useMemo(() => {
    const out: {
      node: number;
      title: string;
      tag: string;
      data: NonNullable<ReturnType<typeof rowCards>>;
    }[] = [];
    const seed = rowCards(live, live.seedNode);
    if (seed)
      out.push({
        node: live.seedNode,
        title: rowLabel(live.db, live.pdb, live.seedNode).long,
        tag: 'seed row',
        data: seed,
      });
    if (live.entityNode !== null) {
      const ent = rowCards(live, live.entityNode);
      if (ent)
        out.push({
          node: live.entityNode,
          title: rowLabel(live.db, live.pdb, live.entityNode).long,
          tag: 'its parent row',
          data: ent,
        });
    }
    return out;
  }, [live]);

  return (
    <div className="grid gap-6">
      {rows.map((r) => (
        <div key={r.node}>
          <div className="mb-2 flex flex-wrap items-center gap-x-2.5 gap-y-1">
            <TableSwatch slot={live.slots[r.data.schema.name] ?? 0} label={r.data.schema.name} />
            <span className="text-[15.5px] text-ink-2">{r.title.replace(`${r.data.schema.name} `, '')}</span>
            <span className="rounded-full bg-surface-2 px-2 py-px text-[13.5px] font-medium text-ink-2">{r.tag}</span>
          </div>
          <ul
            className="grid gap-2 !p-0"
            style={{
              listStyle: 'none',
              gridTemplateColumns: 'repeat(auto-fill, minmax(min(100%, 205px), 1fr))',
            }}
          >
            {r.data.cards.map((c) => {
              const token = c.kind === 'feature' || c.kind === 'target';
              return (
                <li
                  key={c.col.name}
                  className={cx(
                    '!m-0 flex min-w-0 flex-col gap-1 rounded-lg border px-3 py-2',
                    c.kind === 'target'
                      ? 'border-dashed border-accent bg-accent-soft'
                      : token
                        ? 'border-border bg-surface'
                        : 'border-border bg-surface-2/60',
                  )}
                >
                  <span className="flex items-baseline justify-between gap-2">
                    <span className={cx('min-w-0 font-mono text-[14px] break-all', token ? 'text-ink-2' : 'text-muted')}>{c.col.name}</span>
                    {token && (
                      <span className={cx('shrink-0 text-[13.5px] font-medium', c.kind === 'target' ? 'text-accent-text' : 'text-ink-2')}>
                        {c.kind === 'target' ? 'masked token' : 'token'}
                      </span>
                    )}
                  </span>
                  <span
                    className={cx(
                      'font-mono text-[17px] leading-tight break-words',
                      c.kind === 'target' ? 'font-bold text-accent-text' : token ? 'text-ink' : 'text-muted',
                    )}
                  >
                    {c.kind === 'target' ? '?' : formatRaw(c.raw, c.col.dtype)}
                  </span>
                  <span className="min-w-0 text-[14px] leading-snug">
                    <Outcome c={c} />
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2 · The entity's task rows over time
// ─────────────────────────────────────────────────────────────────────────────

interface TaskRowMark {
  node: number;
  ts: number; // seconds
  split: string;
  raw: RawValue;
  isSeed: boolean;
  inContext: boolean;
  afterSeed: boolean;
}

export function useEntityTaskRows(live: OverviewLive): TaskRowMark[] {
  const { db, pdb, task, ctx, labels } = live;
  return useMemo(() => {
    const seed = pdb.nodes[ctx.params.seedNode];
    if (task.kind !== 'forecast' || live.entityNode === null || !seed || seed.ts === null) return [];
    const inCtx = new Set(labels.self.map((l) => l.nodeIdx));
    const out: TaskRowMark[] = [];
    for (const e of pdb.p2fAdj[live.entityNode] ?? []) {
      if (e.tableType === 'Db' || e.tableNameIdx !== seed.tableNameIdx || e.ts === null) continue;
      const rec = rawRow(db, pdb, e.node);
      out.push({
        node: e.node,
        ts: e.ts,
        split: e.tableType,
        raw: rec?.[task.targetCol] ?? null,
        isSeed: e.node === seed.nodeIdx,
        inContext: inCtx.has(e.node),
        afterSeed: e.ts > seed.ts,
      });
    }
    out.sort((a, b) => a.ts - b.ts);
    return out;
  }, [db, pdb, task, ctx, labels, live.entityNode]);
}

const fmtTick = timeFormat('%b %Y');

export function LabelTimeline({ live, rows }: { live: OverviewLive; rows: TaskRowMark[] }) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const W = Math.max(280, Math.round(size.width || 700));
  const slot = live.slots[live.task.name] ?? 5;
  const geo = useMemo(() => {
    if (!rows.length) return null;
    const t0 = new Date(rows[0].ts * 1000);
    const t1 = new Date(rows[rows.length - 1].ts * 1000);
    const pad = 22;
    const x = scaleTime()
      .domain([t0, t1])
      .range([pad, W - pad]);
    const minGap = textWidth('Jan 2025', 13.5) + 14;
    const step = Math.max(1, Math.ceil(minGap / ((W - 2 * pad) / Math.max(1, timeMonth.count(t0, t1)))));
    const every = timeMonth.every(step);
    const ticks = every ? x.ticks(every) : x.ticks(4);
    return { x, ticks };
  }, [rows, W]);
  if (!rows.length || !geo)
    return <p className="text-[16px] text-ink-2">This task has no task rows for an entity (autocomplete tasks seed from database rows).</p>;

  const cy = 46;
  const H = 112;
  const inCtx = rows.filter((r) => r.inContext);
  const xs = inCtx.map((r) => geo.x(new Date(r.ts * 1000)));
  const seedRow = rows.find((r) => r.isSeed);
  const seedX = seedRow ? geo.x(new Date(seedRow.ts * 1000)) : null;
  const hov = hover !== null ? rows[hover] : null;

  return (
    <div ref={ref} className="relative w-full">
      <svg
        width={W}
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        className="block max-w-full"
        role="img"
        aria-label={`${rows.length} task rows of ${live.entityShort} over time; ${inCtx.length} of them are in the context window.`}
      >
        {/* time axis */}
        <line x1={geo.x.range()[0] - 10} x2={geo.x.range()[1] + 10} y1={cy} y2={cy} style={{ stroke: 'var(--axis)' }} strokeWidth={1} />
        {geo.ticks.map((t) => (
          <g key={t.getTime()}>
            <line x1={geo.x(t)} x2={geo.x(t)} y1={cy + 14} y2={cy + 19} style={{ stroke: 'var(--axis)' }} />
            <text x={geo.x(t)} y={cy + 35} textAnchor="middle" style={{ fill: 'var(--muted)', fontSize: 13.5 }}>
              {fmtTick(t)}
            </text>
          </g>
        ))}
        {/* "in the context window" bracket */}
        {xs.length > 0 && (
          <g>
            <path
              d={`M${Math.min(...xs) - 9},${cy - 20} v-5 H${Math.max(...xs) + 9} v5`}
              fill="none"
              style={{ stroke: 'var(--ink-2)' }}
              strokeWidth={1.3}
            />
            <text
              x={Math.min(Math.max((Math.min(...xs) + Math.max(...xs)) / 2, 120), W - 120)}
              y={cy - 32}
              textAnchor="middle"
              style={{ fill: 'var(--ink-2)', fontSize: 13.5, fontWeight: 600 }}
            >
              {int(inCtx.length)} in the context window
            </text>
          </g>
        )}
        {seedX !== null && (
          <line x1={seedX} x2={seedX} y1={cy - 16} y2={cy + 14} style={{ stroke: 'var(--accent)' }} strokeWidth={1.5} strokeDasharray="3 3" />
        )}
        {rows.map((r, i) => {
          const x = geo.x(new Date(r.ts * 1000));
          const label = r.raw === true || r.raw === 1;
          const common = {
            onPointerEnter: () => setHover(i),
            onPointerLeave: () => setHover(null),
          };
          if (r.isSeed)
            return (
              <g key={r.node} {...common}>
                <rect
                  x={x - 10}
                  y={cy - 10}
                  width={20}
                  height={20}
                  rx={4}
                  style={{ fill: 'var(--surface)', stroke: 'var(--accent)' }}
                  strokeWidth={2}
                  strokeDasharray="3.5 2.5"
                />
                <text
                  x={x}
                  y={cy + 1}
                  textAnchor="middle"
                  dominantBaseline="central"
                  style={{
                    fill: 'var(--accent-text)',
                    fontSize: 14,
                    fontWeight: 800,
                  }}
                >
                  ?
                </text>
              </g>
            );
          return (
            <g key={r.node} {...common} style={{ cursor: 'default' }}>
              <circle cx={x} cy={cy} r={12} fill="transparent" />
              <circle
                cx={x}
                cy={cy}
                r={7}
                style={{
                  fill: r.afterSeed ? 'var(--surface)' : label ? catVar(slot) : 'var(--surface)',
                  stroke: r.afterSeed ? 'var(--muted)' : catVar(slot),
                }}
                strokeWidth={2}
                strokeDasharray={r.afterSeed ? '2.5 2' : undefined}
              />
            </g>
          );
        })}
      </svg>
      {hov && (
        <div
          className="pointer-events-none absolute z-10 rounded-lg border border-border bg-surface px-3 py-2 text-[14.5px] leading-snug shadow-lg"
          style={{
            left: Math.min(Math.max(4, geo.x(new Date(hov.ts * 1000)) - 110), W - 230),
            top: cy + 18,
          }}
        >
          <div className="font-semibold text-ink">
            {live.task.name} · {hov.split} · {new Date(hov.ts * 1000).toISOString().slice(0, 10)}
          </div>
          <div className="text-ink-2">
            {hov.isSeed
              ? `the seed row: ${live.task.targetCol} is masked`
              : hov.afterSeed
                ? 'dated after the seed: filtered out of the context'
                : `${live.task.targetCol} = ${formatRaw(hov.raw)}${hov.inContext ? ' · visible in the context' : ' · not in this context'}`}
          </div>
        </div>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1.5 text-[14.5px] text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <svg width="16" height="16" aria-hidden="true">
            <circle cx="8" cy="8" r="6" style={{ fill: catVar(slot), stroke: catVar(slot) }} strokeWidth={2} />
          </svg>
          {live.task.targetCol} = true
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width="16" height="16" aria-hidden="true">
            <circle cx="8" cy="8" r="6" style={{ fill: 'var(--surface)', stroke: catVar(slot) }} strokeWidth={2} />
          </svg>
          {live.task.targetCol} = false
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width="18" height="18" aria-hidden="true">
            <rect
              x="2"
              y="2"
              width="14"
              height="14"
              rx="3"
              style={{ fill: 'var(--surface)', stroke: 'var(--accent)' }}
              strokeWidth={1.8}
              strokeDasharray="3 2"
            />
          </svg>
          the seed row (label masked)
        </span>
        {rows.some((r) => r.afterSeed) && (
          <span className="inline-flex items-center gap-1.5">
            <svg width="16" height="16" aria-hidden="true">
              <circle cx="8" cy="8" r="6" style={{ fill: 'var(--surface)', stroke: 'var(--muted)' }} strokeWidth={2} strokeDasharray="2.5 2" />
            </svg>
            after the seed: never in its context
          </span>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3 · The masked cell's row of each mask
// ─────────────────────────────────────────────────────────────────────────────

export function MaskRows({ live }: { live: OverviewLive }) {
  const { ctx, masks, slots } = live;
  const [ref, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<{
    pos: number;
    x: number;
    y: number;
  } | null>(null);
  const W = Math.max(280, Math.round(size.width || 700));
  const n = ctx.numReal;
  const S = ctx.tokens.length;
  const q = ctx.targetPos;
  const wide = W >= 640;
  const labelW = wide ? 250 : 0;
  const stripW = W - labelW;
  const cw = Math.max(2, stripW / Math.max(1, n));
  const stripH = 22;
  const rowH = wide ? 46 : 64;
  const H = ATTN_ORDER.length * rowH;
  const tables = useMemo(() => [...new Set(ctx.tokens.filter((t) => !t.isPadding).map((t) => t.tableName))], [ctx]);

  return (
    <div ref={ref} className="relative w-full">
      <div className="relative" style={{ height: H }}>
        {ATTN_ORDER.map((l, li) => {
          let count = 0;
          for (let k = 0; k < S; k++) if (masks[l][q * S + k]) count++;
          return (
            <div key={l} className="absolute inset-x-0" style={{ top: li * rowH }}>
              <div className={cx('flex items-baseline gap-2 text-[15.5px]', wide ? 'absolute left-0 top-0.5 w-[240px]' : 'mb-1.5')}>
                <LayerLabel layer={l} />
                <span className="text-ink-2">
                  · {int(count)} {count === 1 ? 'key' : 'keys'}
                </span>
              </div>
              <svg
                width={cw * n}
                height={stripH}
                className="block"
                style={{ marginLeft: labelW }}
                role="img"
                aria-label={`${LAYER_INFO[l].name} attention: the masked cell may attend to ${count} of ${n} tokens.`}
                onPointerMove={(e) => {
                  const r = e.currentTarget.getBoundingClientRect();
                  const pos = Math.floor((e.clientX - r.left) / cw);
                  if (pos >= 0 && pos < n)
                    setHover({
                      pos,
                      x: labelW + pos * cw,
                      y: li * rowH + (wide ? 0 : 26),
                    });
                }}
                onPointerLeave={() => setHover(null)}
              >
                <rect x={0} y={0} width={cw * n} height={stripH} style={{ fill: 'var(--surface-2)' }} />
                {Array.from({ length: n }, (_, k) => {
                  const ok = masks[l][q * S + k] === 1;
                  if (!ok && k !== q) return null;
                  const t = ctx.tokens[k];
                  return (
                    <rect
                      key={k}
                      x={k * cw}
                      y={k === q ? 0 : 3}
                      width={Math.max(1, cw - (cw > 3.5 ? 1 : 0))}
                      height={k === q ? stripH : stripH - 6}
                      style={{
                        fill: k === q ? 'var(--accent)' : catVar(slots[t.tableName] ?? 0),
                      }}
                    />
                  );
                })}
              </svg>
            </div>
          );
        })}
        {hover && (
          <div
            className="pointer-events-none absolute z-10 rounded-lg border border-border bg-surface px-3 py-1.5 text-[14.5px] leading-snug shadow-lg"
            style={{
              left: Math.min(Math.max(0, hover.x - 90), W - 220),
              top: hover.y + 30,
            }}
          >
            <span className="font-mono text-ink">
              {ctx.tokens[hover.pos].tableName}.{ctx.tokens[hover.pos].colName}
            </span>
            <span className="text-ink-2"> · position {hover.pos}</span>
          </div>
        )}
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[14.5px] text-ink-2" style={{ marginLeft: labelW }}>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3.5 w-1.5 rounded-sm bg-accent" /> the masked cell
        </span>
        {tables.map((t) => (
          <TableSwatch key={t} slot={slots[t] ?? 0} label={t} />
        ))}
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="inline-block h-3.5 w-3.5 rounded-sm bg-surface-2 ring-1 ring-border" /> blocked
        </span>
      </div>
    </div>
  );
}
