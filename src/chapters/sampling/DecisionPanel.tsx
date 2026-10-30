/**
 * The "decision detail" of the current replay step: what exactly fly.rs looked at.
 *  - select: the row taken (and, for a bucket draw, the uniform draw over the bucket);
 *  - visit: the row and the parents its foreign keys push onto the stack;
 *  - filter: every P→F child on a time axis against the seed's timestamp (fly.rs:417-436);
 *  - width: the DB children that passed and the uniform w-subset kept (fly.rs:439-452);
 *  - emit: the tokens written, with dropped leakage cells and cells cut at L.
 */
import { Check, Clock, Dice5, KeyRound, Link2, Scissors, X } from 'lucide-react';
import { useMemo } from 'react';
import { Cite, SemBadge, ShapeGlyph, TokenStrip, cx, fitText, formatRaw, maxTextWidth, semOfDType, shapeOfSlot, shapePath, textWidth } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import type { PreprocessedDB, RawDatabase, SampledContext, TaskDef } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { dbCandidates, fitTicks, iterationEvents, plural, rawRowOf, rawTableOf, rowName, singular, tsDate, type Iteration, type ReplayState, type Step } from './trace';

export interface DecisionPanelProps {
  step: Step;
  iters: Iteration[];
  ctx: SampledContext;
  db: RawDatabase;
  pdb: PreprocessedDB;
  task: TaskDef;
  slots: Record<string, number>;
  before: ReplayState;
  state: ReplayState;
  hover: number | null;
  onHover: (n: number | null) => void;
}

function Heading({ children, aside }: { children: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
      <div className="eyebrow !text-[13.5px]">{children}</div>
      {aside && <div className="text-[14px] text-ink-2">{aside}</div>}
    </div>
  );
}

/** A row as a line of "column = value" pills; keys are marked because they produce no token. */
export function RowCard({ db, pdb, node, task, seed, slots, compact = false }: { db: RawDatabase; pdb: PreprocessedDB; node: number; task: TaskDef; seed: number; slots: Record<string, number>; compact?: boolean }) {
  const table = rawTableOf(db, pdb, node);
  const raw = rawRowOf(db, pdb, node);
  const g = pdb.nodes[node];
  if (!table || !raw) return null;
  const s = table.schema;
  const slot = slots[g.tableName] ?? 0;
  return (
    <div className="min-w-0 rounded-lg border border-border bg-surface px-2.5 py-2" style={{ borderLeft: `3px solid ${catVar(slot)}` }}>
      <div className="mb-1.5 flex items-center gap-2 text-[14.5px]">
        <span className="font-semibold text-ink">{rowName(db, pdb, node).long}</span>
        <span className="text-[13.5px] text-muted">
          {g.tableKey}[{g.rowPos}] · node {node}
        </span>
      </div>
      <div className="flex flex-wrap gap-1">
        {s.columns.map((c) => {
          const v = raw[c.name];
          const isPk = c.name === s.pkey;
          const fkTo = s.fkeys[c.name];
          const isNull = v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v));
          const isTarget = node === seed && c.name === task.targetCol && (task.kind === 'autocomplete' ? g.tableType === 'Db' : g.tableType !== 'Db');
          if (compact && (isPk || isNull)) return null;
          return (
            <span
              key={c.name}
              title={isPk ? 'primary key: no token' : fkTo ? `foreign key → ${fkTo}: no token, only a link` : isNull ? 'missing: no token' : undefined}
              className={cx(
                'inline-flex max-w-full items-center gap-1 rounded-md border px-1.5 py-[2px] text-[14px]',
                isPk || fkTo || isNull ? 'border-dashed border-border-strong text-muted' : 'border-border bg-surface-2 text-ink',
                isTarget && 'target-outline border-transparent bg-accent-soft',
              )}
            >
              {isPk ? <KeyRound aria-hidden="true" size={11} /> : fkTo ? <Link2 aria-hidden="true" size={11} /> : <SemBadge sem={semOfDType(c.dtype)} size="sm" bare className="-my-1 text-muted" />}
              <span className="text-ink-2">{c.name}</span>
              <span className={cx('truncate', isTarget && 'font-bold text-accent-text')}>
                {isTarget ? '?' : fkTo && !isNull ? `→ ${singular(fkTo)} ${v}` : formatRaw(v, c.dtype)}
              </span>
            </span>
          );
        })}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Time filter swimlanes
// ─────────────────────────────────────────────────────────────────────────────

type ChildStatus = 'future' | 'other-task' | 'task' | 'db';

const STATUS_TEXT: Record<ChildStatus, string> = {
  future: 'later than the seed, skipped',
  'other-task': 'another task table, skipped',
  task: 'task row, kept (never subsampled)',
  db: 'passes the time filter',
};

/** A lane with more marks than this is drawn as a histogram over time (exact counts in the summary below). */
const DENSE_LANE = 40;
const FS = 13;

function TimeFilter({ items, seedTs, db, pdb, slots, hover, onHover }: {
  items: { node: number; status: ChildStatus }[];
  seedTs: number | null;
  db: RawDatabase;
  pdb: PreprocessedDB;
  slots: Record<string, number>;
  hover: number | null;
  onHover: (n: number | null) => void;
}) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(260, size.width || 480);
  const lanes = useMemo(() => {
    const m = new Map<string, { key: string; label: string; slot: number; items: { node: number; status: ChildStatus; ts: number | null }[] }>();
    for (const it of items) {
      const g = pdb.nodes[it.node];
      const key = g.tableName;
      let lane = m.get(key);
      if (!lane) {
        lane = { key, label: g.tableName, slot: slots[g.tableName] ?? 0, items: [] };
        m.set(key, lane);
      }
      lane.items.push({ ...it, ts: g.ts });
    }
    return [...m.values()];
  }, [items, pdb, slots]);
  // label column: measured at the drawn size, at least 120px, long names truncated (full name in a tooltip)
  const labelW = Math.round(Math.min(W * 0.42, Math.max(120, maxTextWidth(lanes.map((l) => l.label), FS) + 10)));
  const hasNull = items.some((i) => pdb.nodes[i.node].ts === null);
  const noTsW = hasNull ? 48 : 0;
  const tss = items.map((i) => pdb.nodes[i.node].ts).filter((t): t is number => t !== null);
  if (seedTs !== null) tss.push(seedTs);
  const lo = tss.length ? Math.min(...tss) : 0;
  const hi = tss.length ? Math.max(...tss) : 1;
  const span = Math.max(1, hi - lo);
  const x0 = labelW + 10;
  const x1 = W - noTsW - 12;
  const x = (t: number) => x0 + ((t - lo + span * 0.03) / (span * 1.06)) * (x1 - x0);
  const laneH = 26;
  const top = 22;
  const H = top + lanes.length * laneH + 24;
  const nTicks = W < 440 ? 2 : 3;
  const ticks: number[] = [];
  for (let i = 0; i < nTicks; i++) ticks.push(lo + (span * i) / (nTicks - 1));
  const seedX = seedTs !== null ? x(seedTs) : null;
  const seedLabel = seedTs !== null ? `seed time ${tsDate(seedTs)}` : '';
  const seedLabelW = textWidth(seedLabel, FS, { weight: 600 });
  const seedLabelLeft = seedX !== null && seedX + 4 + seedLabelW > W;

  // histogram bins for dense lanes
  const nBins = Math.max(8, Math.floor((x1 - x0) / 6));
  const binOf = (t: number) => Math.min(nBins - 1, Math.max(0, Math.floor(((x(t) - x0) / (x1 - x0)) * nBins)));
  const binX = (b: number) => x0 + (b / nBins) * (x1 - x0);
  const binW = (x1 - x0) / nBins;

  return (
    <div ref={ref} className="min-w-0">
      <svg width={W} height={H} role="img" aria-label="Children of the visited row on a time axis against the seed time" className="block max-w-full">
        {seedX !== null && (
          <>
            <rect x={seedX} y={top - 4} width={Math.max(0, x1 + 4 - seedX)} height={lanes.length * laneH + 4} style={{ fill: 'var(--surface-2)' }} />
            <line x1={seedX} x2={seedX} y1={top - 8} y2={top + lanes.length * laneH} style={{ stroke: 'var(--accent)' }} strokeWidth={1.5} strokeDasharray="4 3" />
            <text x={seedLabelLeft ? seedX - 4 : seedX + 4} y={top - 8} textAnchor={seedLabelLeft ? 'end' : 'start'} style={{ fill: 'var(--accent-text)', fontSize: FS, fontWeight: 600 }}>
              {seedLabel}
            </text>
          </>
        )}
        {hasNull && (
          <text x={W - noTsW / 2 - 4} y={top - 8} textAnchor="middle" style={{ fill: 'var(--muted)', fontSize: FS }}>
            no ts
          </text>
        )}
        {lanes.map((lane, li) => {
          const yTop = top + li * laneH;
          const y = yTop + laneH / 2;
          const dense = lane.items.length > DENSE_LANE;
          const shown = fitText(lane.label, labelW - 6, FS);
          let body: React.ReactNode;
          if (dense) {
            // one stacked bar per time bin: passes (filled), later (hollow), other task table (gray)
            const bins = Array.from({ length: nBins }, () => ({ pass: 0, future: 0, other: 0 }));
            const nullBin = { pass: 0, future: 0, other: 0 };
            for (const it of lane.items) {
              const b = it.ts === null ? nullBin : bins[binOf(it.ts)];
              if (it.status === 'future') b.future++;
              else if (it.status === 'other-task') b.other++;
              else b.pass++;
            }
            const maxN = Math.max(1, ...bins.map((b) => b.pass + b.future + b.other), nullBin.pass + nullBin.future + nullBin.other);
            const hMax = laneH - 6;
            const col = (bx: number, bw: number, b: { pass: number; future: number; other: number }, key: string | number, what: string) => {
              const n = b.pass + b.future + b.other;
              if (!n) return null;
              const h = (v: number) => (v / maxN) * hMax;
              const base = yTop + laneH - 3;
              let yy = base;
              const segs: React.ReactNode[] = [];
              for (const [k, v] of [['pass', b.pass], ['future', b.future], ['other', b.other]] as const) {
                if (!v) continue;
                const hh = Math.max(1.5, h(v));
                yy -= hh;
                segs.push(
                  <rect
                    key={k}
                    x={bx + 0.5}
                    y={yy}
                    width={Math.max(1, bw - 1)}
                    height={hh}
                    style={{
                      fill: k === 'pass' ? catVar(lane.slot) : k === 'other' ? 'var(--axis)' : 'var(--surface)',
                      stroke: k === 'future' ? catVar(lane.slot) : 'none',
                    }}
                    strokeWidth={k === 'future' ? 1 : 0}
                  />,
                );
              }
              return (
                <g key={key}>
                  <title>{`${lane.label}, ${what}: ${n} ${n === 1 ? 'row' : 'rows'}${b.pass ? ` · ${b.pass} pass` : ''}${b.future ? ` · ${b.future} later than the seed` : ''}${b.other ? ` · ${b.other} other task table` : ''}`}</title>
                  {segs}
                </g>
              );
            };
            body = (
              <>
                {bins.map((b, i) => col(binX(i), binW, b, i, `around ${tsDate(Math.round(lo - span * 0.03 + ((i + 0.5) / nBins) * span * 1.06))}`))}
                {hasNull && col(W - noTsW / 2 - 10, 12, nullBin, 'null', 'no timestamp')}
              </>
            );
          } else {
            body = lane.items.map((it) => {
              const cx0 = it.ts === null ? W - noTsW / 2 - 4 : x(it.ts);
              const filled = it.status === 'db' || it.status === 'task';
              const gray = it.status === 'other-task';
              const r = hover === it.node ? 5.5 : 4.2;
              return (
                <g key={it.node} transform={`translate(${cx0},${y})`} onPointerEnter={() => onHover(it.node)} onPointerLeave={() => onHover(null)} style={{ cursor: 'default' }}>
                  <title>{`${rowName(db, pdb, it.node).long}: ${STATUS_TEXT[it.status]}`}</title>
                  <circle r={9} fill="transparent" />
                  <path
                    d={shapePath(shapeOfSlot(lane.slot), r)}
                    style={{
                      fill: filled ? catVar(lane.slot) : gray ? 'var(--axis)' : 'var(--surface)',
                      stroke: filled ? 'var(--surface)' : gray ? 'var(--axis)' : catVar(lane.slot),
                      opacity: it.status === 'future' ? 0.8 : 1,
                    }}
                    strokeWidth={1.2}
                  />
                  {hover === it.node && <circle r={r + 3.5} fill="none" style={{ stroke: 'var(--accent)' }} strokeWidth={1.6} />}
                </g>
              );
            });
          }
          return (
            <g key={lane.key}>
              <line x1={x0} x2={x1} y1={dense ? yTop + laneH - 3 : y} y2={dense ? yTop + laneH - 3 : y} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
              <text x={labelW} y={y} textAnchor="end" dominantBaseline="central" style={{ fill: 'var(--ink-2)', fontSize: FS }}>
                {shown !== lane.label && <title>{lane.label}</title>}
                {shown}
              </text>
              {body}
            </g>
          );
        })}
        {fitTicks(
          ticks.map((t) => ({ x: x(t), text: tsDate(Math.round(t)) })),
          (txt) => textWidth(txt, FS),
        ).map((t, i) => (
          <text key={i} x={t.x} y={H - 6} textAnchor={t.anchor} style={{ fill: 'var(--muted)', fontSize: FS }}>
            {t.text}
          </text>
        ))}
      </svg>
      <ul className="mt-1 grid gap-0.5 text-[14px] leading-snug text-ink-2" aria-label="Children per table">
        {lanes.map((lane) => {
          const c = { pass: 0, future: 0, other: 0 };
          for (const it of lane.items) {
            if (it.status === 'future') c.future++;
            else if (it.status === 'other-task') c.other++;
            else c.pass++;
          }
          return (
            <li key={lane.key} className="flex min-w-0 flex-wrap items-center gap-x-2">
              <ShapeGlyph shape={shapeOfSlot(lane.slot)} size={10} fill={catVar(lane.slot)} />
              <span className="text-ink">{lane.label}</span>
              <span className="num">
                {[c.pass ? `${c.pass} pass` : '', c.future ? `${c.future} later than the seed` : '', c.other ? `${c.other} in another task table` : ''].filter(Boolean).join(' · ')}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Panel
// ─────────────────────────────────────────────────────────────────────────────

export function DecisionPanel({ step, iters, ctx, db, pdb, task, slots, before, state, hover, onHover }: DecisionPanelProps) {
  const seed = ctx.params.seedNode;
  const seedNode = pdb.nodes[seed];
  const w = ctx.params.w;

  if (step.kind === 'start') {
    return (
      <div>
        <Heading aside="F→P stack = [(depth 0, seed)]">Start · fly.rs:366</Heading>
        <RowCard db={db} pdb={pdb} node={seed} task={task} seed={seed} slots={slots} />
        <p className="mt-2 text-[15px] leading-snug text-ink-2">
          The masked cell is the seed’s {task.targetCol}. Key columns (dashed) never become tokens; foreign keys only create links to parents.
        </p>
      </div>
    );
  }
  if (step.kind === 'end') {
    const pad = ctx.tokens.length - ctx.numReal;
    return (
      <div>
        <Heading aside="fly.rs:387-389">Frontier empty</Heading>
        <p className="text-[15.5px] leading-snug text-ink-2">
          Every row that entered the frontier has been taken off it (rows the width bound dropped never entered). The remaining{' '}
          {plural(pad, 'position')} {pad === 1 ? 'is' : 'are'} padding <Cite kind="spec" ref="§4.11" />, and every attention mask excludes them{' '}
          <Cite kind="spec" ref="§6.2" />.
        </p>
      </div>
    );
  }

  const it = iters[step.iter];
  const ie = iterationEvents(ctx, it);
  const kind = step.kind;

  const showSelect = kind === 'select' || kind === 'row';
  const showVisit = (kind === 'visit' || kind === 'row') && !it.skipped;
  const showFilter = (kind === 'filter' || (kind === 'row' && it.hasChildren)) && !it.skipped;
  const showWidth = (kind === 'width' || kind === 'row') && dbCandidates(ie).length > 0 && !it.skipped;
  const showEmit = (kind === 'emit' || kind === 'row') && !it.skipped;

  const children: { node: number; status: ChildStatus }[] = [];
  for (const e of ie.otherTask) children.push({ node: e.node, status: 'other-task' });
  for (const e of ie.future) children.push({ node: e.node, status: 'future' });
  for (const e of ie.taskKids) children.push({ node: e.node, status: 'task' });
  for (const n of dbCandidates(ie)) children.push({ node: n, status: 'db' });

  return (
    <div className="grid gap-3">
      {showSelect && (
        <div>
          <Heading aside={ie.pop.from === 'stack' ? 'fly.rs:378-379' : 'fly.rs:381-396'}>
            {ie.pop.from === 'stack' ? 'Select: pop the F→P stack' : 'Select: uniform draw from the lowest bucket'}
          </Heading>
          {ie.pop.from === 'bucket' && (
            <div className="mb-2">
              <div className="flex flex-wrap gap-[3px]" aria-label={`Bucket at depth ${ie.pop.depth} before the draw`}>
                {(before.buckets[ie.pop.depth] ?? []).map((n, i) => {
                  const slot = slots[pdb.nodes[n].tableName] ?? 0;
                  const picked = i === ie.pop.draw;
                  return (
                    <span
                      key={`${n}-${i}`}
                      title={`index ${i}: ${rowName(db, pdb, n).long}`}
                      onPointerEnter={() => onHover(n)}
                      onPointerLeave={() => onHover(null)}
                      className={cx('inline-block h-[12px] w-[12px] rounded-[3px] border', picked ? 'ring-2 ring-accent ring-offset-1 ring-offset-surface' : 'opacity-70')}
                      style={{ borderColor: catVar(slot), background: picked ? catVar(slot) : `color-mix(in srgb, ${catVar(slot)} 28%, transparent)` }}
                    />
                  );
                })}
              </div>
              <div className="mt-1.5 flex items-center gap-1.5 text-[14.5px] text-ink-2">
                <Dice5 aria-hidden="true" size={14} className="text-accent-text" />
                <span className="font-mono">
                  rng.random_range(0..{ie.pop.bucketSize}) = {ie.pop.draw}
                </span>
                <span className="text-muted">· each entry has probability 1/{ie.pop.bucketSize}</span>
              </div>
            </div>
          )}
          {it.skipped ? (
            <p className="rounded-lg border border-dashed border-border-strong px-2.5 py-2 text-[15px] leading-snug text-ink-2">
              <X aria-hidden="true" size={14} className="mr-1 inline -translate-y-px" />
              <span className="text-ink">{rowName(db, pdb, it.node).short}</span> was already visited (visit #{state.visitedOrder.indexOf(it.node) + 1}),
              so the loop moves on with <span className="font-mono">continue</span> (fly.rs:399-401). It adds no cells and pushes nothing.
            </p>
          ) : kind === 'select' ? (
            <RowCard db={db} pdb={pdb} node={it.node} task={task} seed={seed} slots={slots} compact />
          ) : null}
        </div>
      )}

      {showVisit && (
        <div>
          <Heading aside="fly.rs:399-408">Visit · push parents</Heading>
          <RowCard db={db} pdb={pdb} node={it.node} task={task} seed={seed} slots={slots} compact={kind === 'row'} />
          <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[14.5px] text-ink-2">
            {ie.parents.length === 0 ? (
              <span>No foreign key, so nothing goes onto the stack.</span>
            ) : (
              <>
                <span>Stack ←</span>
                {ie.parents.map((p, i) => {
                  const slot = slots[pdb.nodes[p.node].tableName] ?? 0;
                  const dup = before.visitedDepth.has(p.node);
                  return (
                    <span
                      key={i}
                      onPointerEnter={() => onHover(p.node)}
                      onPointerLeave={() => onHover(null)}
                      className={cx('inline-flex items-center gap-1 rounded-md border border-border bg-surface px-1.5 py-[2px]', dup && 'opacity-60')}
                    >
                      <ShapeGlyph shape={shapeOfSlot(slot)} size={10} fill={catVar(slot)} />
                      <span className={cx('text-ink', dup && 'line-through')}>{rowName(db, pdb, p.node).short}</span>
                      <span className="num text-muted">d{p.depth}</span>
                    </span>
                  );
                })}
                {ie.parents.length > 1 && <span className="text-muted">(last pushed = top)</span>}
              </>
            )}
          </div>
        </div>
      )}

      {showFilter && (
        <div>
          <Heading
            aside={
              <>
                <Clock aria-hidden="true" size={12} className="mr-1 inline -translate-y-px" />
                fly.rs:417-436
              </>
            }
          >
            P→F children: table and time filter
          </Heading>
          {children.length === 0 ? (
            <p className="text-[15px] text-ink-2">No child rows.</p>
          ) : (
            <TimeFilter items={children} seedTs={seedNode.ts} db={db} pdb={pdb} slots={slots} hover={hover} onHover={onHover} />
          )}
          <p className="mt-1 text-[14px] leading-snug text-muted">
            Filled: passes. Hollow: later than the seed, skipped. Gray: another task table, skipped. Busy tables are drawn as counts per time bin. Equal
            timestamps pass; rows without a timestamp always pass{seedNode.ts === null ? ' (and this seed has none, so nothing is time-filtered)' : ''}.
          </p>
        </div>
      )}

      {showWidth && (
        <WidthView ie={ie} w={w} db={db} pdb={pdb} slots={slots} before={before} hover={hover} onHover={onHover} task={task} />
      )}

      {showEmit && ie.emits.length + ie.drops.length > 0 && (
        <EmitView ie={ie} ctx={ctx} db={db} pdb={pdb} slots={slots} hover={hover} onHover={onHover} />
      )}
    </div>
  );
}

function WidthView({ ie, w, db, pdb, slots, before, hover, onHover, task }: {
  ie: ReturnType<typeof iterationEvents>;
  w: number;
  db: RawDatabase;
  pdb: PreprocessedDB;
  slots: Record<string, number>;
  before: ReplayState;
  hover: number | null;
  onHover: (n: number | null) => void;
  task: TaskDef;
}) {
  const cand = dbCandidates(ie);
  const kept = new Set(ie.dbKids.map((e) => e.node));
  const big = cand.length > 80;
  return (
    <div>
      <Heading aside="fly.rs:439-452">
        Width bound: {ie.subsample ? `keep ${w} of ${cand.length}` : `${cand.length} ≤ w = ${w}, keep all`}
      </Heading>
      <div className={cx('flex flex-wrap', big ? 'gap-[3px]' : 'gap-[4px]')}>
        {cand.map((n, i) => {
          const slot = slots[pdb.nodes[n].tableName] ?? 0;
          const k = kept.has(n);
          const vis = before.visitedDepth.has(n);
          return (
            <span
              key={`${n}-${i}`}
              title={`${rowName(db, pdb, n).long}: ${k ? 'kept' : 'not sampled'}${vis ? ' (already visited, still counts toward w)' : ''}`}
              onPointerEnter={() => onHover(n)}
              onPointerLeave={() => onHover(null)}
              className={cx(
                'relative inline-flex items-center justify-center rounded-[3px] border',
                big ? 'h-[10px] w-[10px]' : 'h-[16px] w-[16px]',
                !k && 'opacity-40',
                hover === n && 'ring-2 ring-accent',
              )}
              style={{ borderColor: catVar(slot), background: k ? catVar(slot) : 'transparent' }}
            >
              {vis && <Check aria-hidden="true" size={big ? 8 : 10} strokeWidth={3} style={{ color: k ? '#fff' : 'var(--ink-2)' }} />}
            </span>
          );
        })}
      </div>
      <p className="mt-1.5 text-[14px] leading-snug text-muted">
        Filled: kept and pushed. Faded: not sampled. ✓: already visited (it still counts toward w).
        {ie.taskKids.length > 0 ? ` The ${plural(ie.taskKids.length, `${task.kind === 'forecast' ? task.name : 'task'} row`)} bypass this bound.` : ''}
      </p>
    </div>
  );
}

function EmitView({ ie, ctx, db, pdb, slots, hover, onHover }: {
  ie: ReturnType<typeof iterationEvents>;
  ctx: SampledContext;
  db: RawDatabase;
  pdb: PreprocessedDB;
  slots: Record<string, number>;
  hover: number | null;
  onHover: (n: number | null) => void;
}) {
  const node = pdb.nodes[ie.it.node];
  const toks = ie.emits.map((e) => ctx.tokens[e.pos]);
  const emittedCols = new Set(ie.emits.map((e) => e.colName));
  const droppedCols = new Set(ie.drops.map((e) => e.colName));
  const cut = ie.truncated ? node.cells.filter((c) => !emittedCols.has(c.colName) && !droppedCols.has(c.colName)) : [];
  return (
    <div>
      <Heading aside={ie.truncated ? 'fly.rs:502-509' : 'fly.rs:454-506'}>
        Emit cells → tokens {toks.length ? `${toks[0].pos}${toks.length > 1 ? `–${toks[toks.length - 1].pos}` : ''}` : ''}
      </Heading>
      <div className="flex flex-wrap items-center gap-1.5">
        {toks.length > 0 && (
          <TokenStrip tokens={toks} tableSlots={slots} variant="chips" highlight={hover === node.nodeIdx ? toks.map((t) => t.pos) : []} onHover={(p) => onHover(p === null ? null : node.nodeIdx)} label="Tokens emitted in this step" />
        )}
        {ie.drops.map((d, i) => (
          <span key={`d${i}`} className="inline-flex h-[42px] items-center gap-1 rounded-[7px] border border-dashed border-border-strong px-2 text-[14px] text-muted line-through" title="leakage column: dropped (fly.rs:457-461)">
            {d.colName}
          </span>
        ))}
        {cut.map((c) => (
          <span key={`c${c.colName}`} className="inline-flex h-[42px] items-center gap-1 rounded-[7px] border border-dashed border-axis px-2 text-[14px] text-muted" title="not emitted: the context reached L">
            <Scissors aria-hidden="true" size={12} /> {c.colName}
          </span>
        ))}
      </div>
      {ie.truncated && (
        <p className="mt-1.5 text-[14px] leading-snug text-muted">
          {cut.length > 0 ? `Cut at L: ${rowName(db, pdb, node.nodeIdx).short} contributes ${toks.length} of its ${toks.length + cut.length} cells.` : `L is reached exactly at the end of this row.`}
        </p>
      )}
    </div>
  );
}
