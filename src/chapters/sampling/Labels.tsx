/**
 * Section "In-context labels": which label cells the sampled context contains (self vs other
 * entities, spec §2.5), how they were reached, which splits a seed can see, and the paper's Table 12.
 */
import { useEffect, useMemo, useState } from 'react';
import { EntityGraph, Segmented, ShapeGlyph, TokenStrip, cx, formatRaw, shapeOfSlot, shapePath, textWidth, type GraphEdgeSpec, type GraphNodeSpec, type NodeId, type RingStyle } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { inContextLabels, type InContextLabels } from '@/engine/pipeline';
import { sampleContext } from '@/engine/sampler';
import type { RawValue, SampledContext, Split } from '@/engine/types';
import { getContext, makeSampleParams } from '@/state/hooks';
import { catVar } from '@/styles/palette';
import { useExplore } from './explore';
import { useTrace } from './player';
import { buildLayout, fitTicks, plural, rowName, tsDate, type Layout } from './trace';

type Preset = { L: number; w: number; count: number };

/**
 * The first of `settings` (tried in order, one sampler run per macrotask so the page stays responsive)
 * whose context of the current seed scores > 0. Sampler only, no model.
 */
function useFirstSetting(enabled: boolean, settings: readonly { L: number; w: number }[], score: (lab: InContextLabels, seedTs: number | null) => number, tag: string): Preset | null | 'pending' {
  const { pdb, task, seedNode, rngSeed } = useExplore();
  const [res, setRes] = useState<{ key: string; value: Preset | null } | null>(null);
  const key = `${tag}|${task.id}|${seedNode}|${rngSeed}|${enabled}|${settings.map((x) => `${x.L}:${x.w}`).join(',')}`;
  useEffect(() => {
    if (!enabled) {
      setRes({ key, value: null });
      return;
    }
    let alive = true;
    let i = 0;
    let id = 0;
    const seedTs = pdb.nodes[seedNode].ts;
    const tick = () => {
      if (!alive) return;
      const t0 = performance.now();
      while (i < settings.length && performance.now() - t0 < 12) {
        const { L, w } = settings[i++];
        const ctx = sampleContext(pdb, makeSampleParams(pdb, task, seedNode, L, w, rngSeed), { recordEvents: false });
        const count = score(inContextLabels(pdb, ctx, task), seedTs);
        if (count > 0) {
          setRes({ key, value: { L, w, count } });
          return;
        }
      }
      if (i < settings.length) id = window.setTimeout(tick, 0);
      else setRes({ key, value: null });
    };
    id = window.setTimeout(tick, 80);
    return () => {
      alive = false;
      window.clearTimeout(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return res && res.key === key ? res.value : 'pending';
}

const SAME_HORIZON_SETTINGS = [128, 256, 512].flatMap((L) => [16, 8, 6, 4, 3, 2, 1].map((w) => ({ L, w })));

/**
 * The smallest sampler settings (L ascending, then w descending) under which the current seed's
 * context contains at least one same-horizon label.
 */
export function useSameHorizonPreset(): Preset | null | 'pending' {
  const { task } = useExplore();
  return useFirstSetting(task.kind === 'forecast', SAME_HORIZON_SETTINGS, (lab, ts) => lab.other.filter((l) => l.ts !== null && l.ts === ts).length, 'same');
}

/**
 * When the current context holds no other-entity label (forecasting), the first narrower search that
 * reaches some: w = 8 … 1 at the current L, then the same at L = 256.
 */
export function useOtherEntityPreset(): Preset | null | 'pending' {
  const { task, ctx } = useExplore();
  const labels = useLabels();
  const L = ctx.params.L;
  const w = ctx.params.w;
  const settings = useMemo(() => {
    const out: { L: number; w: number }[] = [];
    for (const l of [L, Math.max(L, 256)]) for (const x of [8, 6, 4, 3, 2, 1]) if (x < w && !out.some((o) => o.L === l && o.w === x)) out.push({ L: l, w: x });
    return out;
  }, [L, w]);
  return useFirstSetting(task.kind === 'forecast' && labels.other.length === 0, settings, (lab) => lab.other.length, 'other');
}

/** The context the label figures show: the live one, or the narrower search of `useOtherEntityPreset`. */
export interface LabelView {
  ctx: SampledContext;
  labels: InContextLabels;
  layout: Layout;
  alt: boolean;
}

export function useLabelView(alt: Preset | null): LabelView {
  const { pdb, task, ctx, seedNode, rngSeed } = useExplore();
  const { layout } = useTrace();
  const labels = useLabels();
  return useMemo(() => {
    if (!alt) return { ctx, labels, layout, alt: false };
    const c = getContext(pdb, makeSampleParams(pdb, task, seedNode, alt.L, alt.w, rngSeed));
    return { ctx: c, labels: inContextLabels(pdb, c, task), layout: buildLayout(c), alt: true };
  }, [alt, pdb, task, ctx, seedNode, rngSeed, labels, layout]);
}

/** Toggle between the live context and a narrower search that reaches other entities. */
export function LabelViewToggle({ preset, value, onChange }: { preset: Preset; value: boolean; onChange: (alt: boolean) => void }) {
  const { ctx, pdb } = useExplore();
  const labels = useLabels();
  const ent = labels.entityNode !== null ? pdb.nodes[labels.entityNode].tableName : 'entities';
  return (
    <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-2">
      <Segmented<'live' | 'alt'>
        label="Which context"
        value={value ? 'alt' : 'live'}
        onChange={(v) => onChange(v === 'alt')}
        options={[
          { value: 'live', label: `This context (w = ${ctx.params.w})`, title: `L = ${ctx.params.L}, w = ${ctx.params.w}: the context of the replay above` },
          { value: 'alt', label: `Narrower search (w = ${preset.w})`, title: `L = ${preset.L}, w = ${preset.w}: same seed and RNG seed` },
        ]}
      />
      <span className="text-[14.5px] text-ink-2">
        {value
          ? `L = ${preset.L}, w = ${preset.w}, same seed: this search reaches other ${ent} and ${plural(preset.count, 'other-entity label')}.`
          : `No other-entity label in this context (w = ${ctx.params.w}). A narrower search of the same seed reaches some.`}
      </span>
    </div>
  );
}

export function useLabels(): InContextLabels {
  const { pdb, ctx, task } = useExplore();
  return useMemo(() => inContextLabels(pdb, ctx, task), [pdb, ctx, task]);
}

function labelText(v: RawValue): string {
  if (typeof v === 'boolean') return v ? '1' : '0';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(1);
  return formatRaw(v);
}

/** Past task rows of the seed's entity that the sampler may include (same task table, any split, ts ≤ seed ts, not the seed). */
export function useEligibleSelf(): number[] {
  const { pdb, ctx, task } = useExplore();
  return useMemo(() => {
    if (task.kind !== 'forecast') return [];
    const seed = pdb.nodes[ctx.params.seedNode];
    const ent = seed.f2pNbrIdxs[0];
    if (ent === undefined) return [];
    return (pdb.p2fAdj[ent] ?? [])
      .filter((e) => e.tableType !== 'Db' && e.tableNameIdx === seed.tableNameIdx && e.node !== seed.nodeIdx && !(e.ts !== null && seed.ts !== null && e.ts > seed.ts))
      .map((e) => e.node);
  }, [pdb, ctx, task]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Timeline
// ─────────────────────────────────────────────────────────────────────────────

export function LabelTimeline({ view }: { view?: LabelView }) {
  const { pdb, db, task, slots } = useExplore();
  const live = useLabels();
  const liveCtx = useExplore().ctx;
  const ctx = view?.ctx ?? liveCtx;
  const labels = view?.labels ?? live;
  const eligible = useEligibleSelf();
  const [ref, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<string | null>(null);
  const W = Math.max(300, size.width || 700);
  const seed = pdb.nodes[ctx.params.seedNode];
  const slot = slots[seed.tableName] ?? 5;

  const selfSet = new Set(labels.self.map((l) => l.nodeIdx));
  const missing = eligible.filter((n) => !selfSet.has(n));

  // the time span: every label, the seed, and every row of the seed's table (to show what lies in the future)
  const allTs: number[] = [];
  for (const l of [...labels.self, ...labels.other]) if (l.ts !== null) allTs.push(l.ts);
  for (const n of missing) if (pdb.nodes[n].ts !== null) allTs.push(pdb.nodes[n].ts!);
  if (seed.ts !== null) allTs.push(seed.ts);
  if (task.kind === 'forecast') {
    for (const g of pdb.nodes) if (g.tableType !== 'Db' && g.tableNameIdx === seed.tableNameIdx && g.ts !== null) allTs.push(g.ts);
  }
  const lo = allTs.length ? Math.min(...allTs) : 0;
  const hi = allTs.length ? Math.max(...allTs) : 1;
  const span = Math.max(1, hi - lo);
  const narrow = W < 480;
  const labelW = narrow ? 96 : 136;
  const x0 = labelW + 14;
  const x1 = W - 18;
  const x = (t: number) => x0 + ((t - lo) / span) * (x1 - x0);

  const entityName = labels.entityNode !== null ? rowName(db, pdb, labels.entityNode).short : null;
  const lanes: { key: string; title: string; sub: string }[] =
    task.kind === 'forecast'
      ? [
          { key: 'self', title: entityName ?? 'self', sub: 'self labels' },
          { key: 'other', title: narrow ? 'others' : 'other entities', sub: narrow ? 'other labels' : 'other-entity labels' },
        ]
      : [{ key: 'other', title: narrow ? 'other rows' : `other ${seed.tableName}`, sub: 'target cells' }];
  const laneH = 52;
  const top = 30;
  const H = top + lanes.length * laneH + 28;
  const yOf = (k: string) => top + lanes.findIndex((l) => l.key === k) * laneH + laneH / 2 + 8;

  const nTicks = W < 480 ? 3 : 5;
  const ticks: number[] = [];
  for (let i = 0; i < nTicks; i++) ticks.push(lo + (span * i) / (nTicks - 1));

  // Marks grouped by (lane, timestamp): many task rows share a timestamp (one per entity and horizon), so
  // a group is one mark with a count, its values in the tooltip.
  type Item = { node: number; ts: number; raw: RawValue; kind: 'in' | 'missing'; same: boolean };
  const groups = useMemo(() => {
    const m = new Map<string, { lane: string; ts: number; items: Item[] }>();
    const add = (lane: string, it: Item) => {
      const k = `${lane}|${it.ts}|${it.kind}`;
      const g = m.get(k) ?? { lane, ts: it.ts, items: [] };
      g.items.push(it);
      m.set(k, g);
    };
    if (task.kind === 'forecast') for (const n of missing) if (pdb.nodes[n].ts !== null) add('self', { node: n, ts: pdb.nodes[n].ts!, raw: null, kind: 'missing', same: false });
    for (const l of labels.self) if (l.ts !== null) add('self', { node: l.nodeIdx, ts: l.ts, raw: l.raw, kind: 'in', same: false });
    for (const l of labels.other) if (l.ts !== null) add('other', { node: l.nodeIdx, ts: l.ts, raw: l.raw, kind: 'in', same: seed.ts !== null && l.ts === seed.ts });
    return [...m.values()].sort((a, b) => a.ts - b.ts);
  }, [labels, missing.join(','), pdb, seed.ts, task.kind]); // eslint-disable-line react-hooks/exhaustive-deps

  // value labels: drawn left to right, skipped where they would overlap the previous one (hover shows all)
  const VALUE_FS = 13;
  const placed = new Set<string>();
  const lastRight = new Map<string, number>();
  const textOf = (g: { items: Item[] }) => (g.items.length > 1 ? `×${g.items.length}` : labelText(g.items[0].raw));
  for (const g of groups) {
    if (g.items[0].kind !== 'in') continue;
    const txt = textOf(g);
    const wTxt = textWidth(txt, VALUE_FS, { mono: true });
    const cx0 = x(g.ts);
    const left = cx0 - wTxt / 2;
    if (left >= (lastRight.get(g.lane) ?? -Infinity) + 4) {
      placed.add(`${g.lane}|${g.ts}`);
      lastRight.set(g.lane, cx0 + wTxt / 2);
    }
  }
  const skipped = groups.some((g) => g.items[0].kind === 'in' && !placed.has(`${g.lane}|${g.ts}`));

  const groupMark = (g: { lane: string; ts: number; items: Item[] }) => {
    const first = g.items[0];
    const n = g.items.length;
    const cx0 = x(g.ts);
    const cy = yOf(g.lane);
    const r = 5.6;
    const id = `${g.lane}|${g.ts}|${first.kind}`;
    const on = hover === id;
    const same = g.items.some((i) => i.same);
    const vals = g.items.slice(0, 12).map((i) => labelText(i.raw)).join(', ') + (n > 12 ? ', …' : '');
    const title =
      first.kind === 'missing'
        ? `${n === 1 ? rowName(db, pdb, first.node).long : `${n} rows at ${tsDate(g.ts)}`}: eligible but not in this context`
        : n === 1
          ? `${rowName(db, pdb, first.node).long}: ${task.targetCol} = ${formatRaw(first.raw)}${same ? ' (same horizon as the seed)' : ''}`
          : `${n} labels at ${tsDate(g.ts)}${same ? ' (same horizon as the seed)' : ''}: ${task.targetCol} = ${vals}`;
    return (
      <g key={id} transform={`translate(${cx0},${cy})`} onPointerEnter={() => setHover(id)} onPointerLeave={() => setHover(null)}>
        <title>{title}</title>
        <circle r={11} fill="transparent" />
        {same && <circle r={r + 3.8} fill="none" style={{ stroke: 'var(--accent)' }} strokeWidth={1.6} />}
        {n > 1 && (
          <>
            <g transform="translate(3.5,-3.5)">
              <ShapeG slot={slot} r={r} filled={first.kind === 'in'} dashed={first.kind === 'missing'} />
            </g>
            {n > 2 && (
              <g transform="translate(1.75,-1.75)">
                <ShapeG slot={slot} r={r} filled={first.kind === 'in'} dashed={first.kind === 'missing'} />
              </g>
            )}
          </>
        )}
        <ShapeG slot={slot} r={r} filled={first.kind === 'in'} dashed={first.kind === 'missing'} />
        {on && <circle r={r + 5.5} fill="none" style={{ stroke: 'var(--ink)' }} strokeWidth={1.2} />}
        {first.kind === 'in' && placed.has(`${g.lane}|${g.ts}`) && (
          <text y={-r - (n > 1 ? 9 : 6)} textAnchor="middle" className="font-mono" style={{ fill: n > 1 ? 'var(--ink)' : 'var(--ink-2)', fontSize: VALUE_FS, fontWeight: n > 1 ? 600 : 400 }}>
            {textOf(g)}
          </text>
        )}
      </g>
    );
  };

  const seedLabel = seed.ts !== null ? `seed ${tsDate(seed.ts)}` : '';
  const seedLabelW = textWidth(seedLabel, 13.5, { weight: 600 });

  return (
    <div ref={ref} className="min-w-0">
      <svg width={W} height={H} role="img" aria-label={`Timeline of the ${plural(labels.self.length + labels.other.length, 'label')} in the context`} className="block">
        {seed.ts !== null && (
          <>
            <rect x={x(seed.ts)} y={top - 6} width={Math.max(0, x1 + 8 - x(seed.ts))} height={lanes.length * laneH + 8} style={{ fill: 'var(--surface-2)' }} />
            <line x1={x(seed.ts)} x2={x(seed.ts)} y1={top - 14} y2={top + lanes.length * laneH + 2} style={{ stroke: 'var(--accent)' }} strokeWidth={1.5} strokeDasharray="4 3" />
            <text
              x={x(seed.ts) + 5 + seedLabelW > W ? x(seed.ts) - 5 : x(seed.ts) + 5}
              textAnchor={x(seed.ts) + 5 + seedLabelW > W ? 'end' : 'start'}
              y={top - 16}
              style={{ fill: 'var(--accent-text)', fontSize: 13.5, fontWeight: 600 }}
            >
              {seedLabel}
            </text>
            {x1 - x(seed.ts) > textWidth('later rows: never in the context', 13) + 12 && (
              <text x={x1} y={top + lanes.length * laneH - 2} textAnchor="end" style={{ fill: 'var(--muted)', fontSize: 13 }}>
                later rows: never in the context
              </text>
            )}
          </>
        )}
        {lanes.map((l, i) => {
          const y = top + i * laneH + laneH / 2 + 8;
          return (
            <g key={l.key}>
              <line x1={x0 - 4} x2={x1} y1={y} y2={y} style={{ stroke: 'var(--grid)' }} />
              <text x={labelW} y={y - 5} textAnchor="end" className="font-mono" style={{ fill: 'var(--ink)', fontSize: 13.5 }}>
                {l.title.length > 18 ? l.title.slice(0, 17) + '…' : l.title}
              </text>
              <text x={labelW} y={y + 11} textAnchor="end" style={{ fill: 'var(--muted)', fontSize: 13 }}>
                {l.sub}
              </text>
              {(l.key === 'self' ? labels.self.length + missing.length : labels.other.length) === 0 && (
                <text x={x0 + 4} y={y - 8} style={{ fill: 'var(--muted)', fontSize: 13.5, fontStyle: 'italic' }}>
                  none in this context
                </text>
              )}
            </g>
          );
        })}
        {groups.map(groupMark)}
        {task.kind === 'forecast' && seed.ts !== null && (
          <g transform={`translate(${x(seed.ts)},${yOf('self')})`}>
            <title>{`${rowName(db, pdb, seed.nodeIdx).long}: the seed (masked target)`}</title>
            <circle r={12} fill="none" style={{ stroke: 'var(--accent)' }} strokeWidth={1.6} strokeDasharray="3 2.4" />
            <ShapeG slot={slot} r={8.5} filled />
            <text y={0.5} textAnchor="middle" dominantBaseline="central" style={{ fill: '#fff', fontSize: 13, fontWeight: 700 }}>
              ?
            </text>
          </g>
        )}
        {fitTicks(
          ticks.map((t) => ({ x: x(t), text: tsDate(Math.round(t)) })),
          (txt) => textWidth(txt, 13, { mono: true }),
        ).map((t, i) => (
          <text key={i} x={t.x} y={H - 6} textAnchor={t.anchor} className="font-mono" style={{ fill: 'var(--muted)', fontSize: 13 }}>
            {t.text}
          </text>
        ))}
      </svg>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[14.5px] text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <ShapeGlyph shape={shapeOfSlot(slot)} size={12} fill={catVar(slot)} /> label in the context (value above)
        </span>
        {groups.some((g) => g.items.length > 1) && (
        <span className="inline-flex items-center gap-1.5">
          <svg aria-hidden="true" width="18" height="16" viewBox="-8 -9 18 16">
            <g transform="translate(3.5,-3.5)">
              <ShapeG slot={slot} r={5} filled />
            </g>
            <ShapeG slot={slot} r={5} filled />
          </svg>
          ×n: n labels at the same timestamp (hover for the values)
        </span>
        )}
        {task.kind === 'forecast' && (
          <span className="inline-flex items-center gap-1.5">
            <ShapeGlyph shape={shapeOfSlot(slot)} size={12} fill="var(--surface)" stroke={catVar(slot)} /> eligible, but not sampled
          </span>
        )}
        <span className="inline-flex items-center gap-1.5">
          <svg aria-hidden="true" width="14" height="14" viewBox="-7 -7 14 14">
            <circle r={5.5} fill="none" style={{ stroke: 'var(--accent)' }} strokeWidth={1.6} />
          </svg>
          same horizon as the seed
        </span>
        {skipped && <span className="text-muted">Values are printed where they fit; hover a mark for its value.</span>}
      </div>
    </div>
  );
}

function ShapeG({ slot, r, filled, dashed }: { slot: number; r: number; filled: boolean; dashed?: boolean }) {
  return (
    <path
      d={shapePath(shapeOfSlot(slot), r)}
      style={{ fill: filled ? catVar(slot) : 'var(--surface)', stroke: filled ? 'var(--surface)' : catVar(slot) }}
      strokeWidth={1.3}
      strokeDasharray={dashed ? '2 1.6' : undefined}
    />
  );
}

/** The final context's tokens with every visible label cell ringed (shared TokenStrip). */
export function LabelTokens({ view }: { view?: LabelView }) {
  const ex = useExplore();
  const live = useLabels();
  const { slots, task } = ex;
  const ctx = view?.ctx ?? ex.ctx;
  const labels = view?.labels ?? live;
  const [mode, setMode] = useState<'all' | 'self' | 'other'>('all');
  const pos = useMemo(() => {
    const src = mode === 'self' ? labels.self : mode === 'other' ? labels.other : [...labels.self, ...labels.other];
    return src.map((l) => l.pos);
  }, [labels, mode]);
  const opts: { v: 'all' | 'self' | 'other'; label: string; n: number }[] =
    task.kind === 'forecast'
      ? [
          { v: 'all', label: 'all labels', n: labels.self.length + labels.other.length },
          { v: 'self', label: 'self', n: labels.self.length },
          { v: 'other', label: 'other entities', n: labels.other.length },
        ]
      : [{ v: 'all', label: 'label cells', n: labels.other.length }];
  return (
    <div className="min-w-0">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-[15px] text-ink-2">
        <span>Ring in the token strip:</span>
        {opts.map((o) => (
          <button
            key={o.v}
            type="button"
            aria-pressed={mode === o.v}
            onClick={() => setMode(o.v)}
            className={cx('rounded-md border px-2 py-0.5', mode === o.v ? 'border-accent bg-accent-soft text-accent-text' : 'border-border text-ink-2 hover:bg-surface-2')}
          >
            {o.label} <span className="num">{o.n}</span>
          </button>
        ))}
      </div>
      <TokenStrip tokens={ctx.tokens} tableSlots={slots} variant="cells" highlight={pos} dimOthers label="Context tokens with the label cells ringed" />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// How each label got in (shared EntityGraph)
// ─────────────────────────────────────────────────────────────────────────────

export function LabelPaths({ view }: { view?: LabelView }) {
  const ex = useExplore();
  const { pdb, db, slots } = ex;
  const trace = useTrace();
  const live = useLabels();
  const ctx = view?.ctx ?? ex.ctx;
  const layout = view?.layout ?? trace.layout;
  const labels = view?.labels ?? live;
  const MAX_OTHER = 24;

  const graph = useMemo(() => {
    const keep = new Set<number>([ctx.params.seedNode]);
    const targets = [...labels.self.map((l) => l.nodeIdx), ...labels.other.slice(0, MAX_OTHER).map((l) => l.nodeIdx)];
    for (const t of targets) {
      let cur: number | null = t;
      let guard = 0;
      while (cur !== null && guard++ < 64) {
        keep.add(cur);
        cur = layout.nodes.get(cur)?.parent ?? null;
      }
    }
    const nodes: GraphNodeSpec[] = [...keep].map((n) => {
      const g = pdb.nodes[n];
      const nm = rowName(db, pdb, n);
      const short = g.tableType !== 'Db' && g.ts !== null ? `${g.tableType} ${tsDate(g.ts).slice(2)}` : nm.short;
      return { id: n, label: n === ctx.params.seedNode ? 'seed' : short, table: g.tableName, slot: slots[g.tableName] ?? 0, title: nm.long };
    });
    const edges: GraphEdgeSpec[] = [];
    for (const n of keep) {
      const ln = layout.nodes.get(n);
      if (!ln || ln.parent === null || !keep.has(ln.parent)) continue;
      // FK direction child → parent
      if (ln.via === 'f2p') edges.push({ source: ln.parent, target: n });
      else edges.push({ source: n, target: ln.parent });
    }
    const rings = new Map<NodeId, RingStyle>();
    rings.set(ctx.params.seedNode, 'target');
    for (const l of labels.self) rings.set(l.nodeIdx, 'accent');
    for (const l of labels.other.slice(0, MAX_OTHER)) rings.set(l.nodeIdx, 'dashed');
    const badges = new Map<NodeId, string>();
    for (const n of keep) {
      const d = ctx.visited.find((v) => v.node === n)?.depth;
      if (d !== undefined && !rings.has(n)) badges.set(n, `d${d}`);
    }
    return { nodes, edges, rings, badges };
  }, [ctx, labels, layout, pdb, db, slots]);

  const shownOther = Math.min(labels.other.length, MAX_OTHER);
  const [wrap, size] = useElementSize<HTMLDivElement>();
  const narrow = (size.width || 600) < 520;
  return (
    <div ref={wrap} className="min-w-0">
      <EntityGraph
        nodes={graph.nodes}
        edges={graph.edges}
        rings={graph.rings}
        badges={graph.badges}
        labels={narrow ? 'none' : graph.nodes.length <= 40 ? 'all' : 'highlighted'}
        highlight={[...graph.rings.keys()]}
        dimOthers={false}
        height={420}
        linkDistance={52}
        charge={-190}
        title="How each in-context label entered the context"
      />
      <p className="mt-1 text-[14.5px] text-muted">
        Solid ring: self label · dashed ring: other-entity label · d = visit depth of the rows on the way. Each path is the one the row was
        visited through.{narrow ? ' Tap or hover a row for its name.' : ''}
        {labels.other.length > shownOther ? ` Showing ${shownOther} of ${labels.other.length} other-entity labels.` : ''}
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Which task rows can a seed of each split see?
// ─────────────────────────────────────────────────────────────────────────────

export function SplitVisibility() {
  const { task, split, pdb, ctx } = useExplore();
  if (task.kind !== 'forecast' || !task.tables) return null;
  const tsBySplit: Record<Split, number[]> = { Train: [], Val: [], Test: [] };
  for (const s of ['Train', 'Val', 'Test'] as Split[]) {
    const set = new Set<number>();
    const col = task.timeCol ?? 'timestamp';
    for (const r of task.tables[s].rows) if (typeof r[col] === 'number') set.add(Math.trunc((r[col] as number) / 1000));
    tsBySplit[s] = [...set].sort((a, b) => a - b);
  }
  const cols = [
    ...tsBySplit.Train.map((t) => ({ t, s: 'Train' as Split })),
    ...tsBySplit.Val.map((t) => ({ t, s: 'Val' as Split })),
    ...tsBySplit.Test.map((t) => ({ t, s: 'Test' as Split })),
  ];
  const seed = pdb.nodes[ctx.params.seedNode];
  const trainT = split === 'Train' && seed.ts !== null ? seed.ts : tsBySplit.Train[Math.max(0, tsBySplit.Train.length - 4)];
  const rows: { s: Split; t: number; label: string }[] = [
    { s: 'Train', t: trainT, label: `Train seed at ${tsDate(trainT)}` },
    { s: 'Val', t: tsBySplit.Val[0], label: `Val seed at ${tsDate(tsBySplit.Val[0])}` },
    { s: 'Test', t: tsBySplit.Test[0], label: `Test seed at ${tsDate(tsBySplit.Test[0])}` },
  ];
  const cell = (seedT: number, t: number) => (t < seedT ? 'past' : t === seedT ? 'same' : 'future');
  return (
    <div className="min-w-0">
      <div className="scroll-quiet overflow-x-auto">
        <table className="border-separate border-spacing-[3px] text-[14.5px]" aria-label="Which task rows a seed of each split can see">
          <thead>
            <tr>
              <th scope="col" className="sticky left-0 bg-surface pr-2 text-left font-medium text-ink-2">
                seed ↓ · task rows at →
              </th>
              {cols.map((c, i) => (
                <th key={i} scope="col" className="px-0 text-center font-normal text-muted" title={`${c.s} rows at ${tsDate(c.t)}`}>
                  <span className={cx('inline-block w-[18px] font-mono text-[13px]', c.s !== 'Train' && 'font-semibold text-ink')}>{c.s === 'Train' ? 'T' : c.s === 'Val' ? 'V' : 'Te'}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.s} className={cx(r.s === split && 'outline-1 outline-accent')}>
                <th scope="row" className={cx('sticky left-0 bg-surface pr-2 text-left font-normal whitespace-nowrap', r.s === split ? 'font-semibold text-ink' : 'text-ink-2')}>
                  {r.label}
                  {r.s === split ? ' (yours)' : ''}
                </th>
                {cols.map((c, i) => {
                  const k = cell(r.t, c.t);
                  return (
                    <td
                      key={i}
                      title={`${c.s} rows at ${tsDate(c.t)}: ${k === 'past' ? 'can appear (any entity)' : k === 'same' ? 'same horizon: other entities’ rows can appear; the seed’s own row is the target' : 'later than the seed: never'}`}
                      className="p-0"
                    >
                      <span
                        className={cx('block h-[18px] w-[18px] rounded-[4px]', k === 'future' && 'border border-dashed border-axis', k === 'same' && 'ring-2 ring-accent ring-inset')}
                        style={{ background: k === 'past' ? 'var(--cat-5)' : k === 'same' ? 'color-mix(in srgb, var(--cat-5) 45%, transparent)' : 'transparent' }}
                      />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[14.5px] text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-[3px]" style={{ background: 'var(--cat-5)' }} /> can appear
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-[3px] ring-2 ring-accent ring-inset" style={{ background: 'color-mix(in srgb, var(--cat-5) 45%, transparent)' }} /> same horizon: other
          entities only
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-[3px] border border-dashed border-axis" /> never (later than the seed)
        </span>
        <span className="text-muted">T = Train, V = Val, Te = Test timestamps of {task.name}</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Paper Table 12 (p.23)
// ─────────────────────────────────────────────────────────────────────────────

const T12: { group: string; rows: [string, string, string, string, string][] }[] = [
  {
    group: 'Binary classification tasks',
    rows: [
      ['rel-amazon', 'user-churn', '4 ± 4', '1 ± 5', '2 ± 3'],
      ['rel-hm', 'user-churn', '6 ± 4', '0 ± 0', '1 ± 0'],
      ['rel-stack', 'user-badge', '7 ± 6', '3 ± 6', '1 ± 1'],
      ['rel-amazon', 'item-churn', '8 ± 7', '2 ± 6', '2 ± 3'],
      ['rel-stack', 'user-engagement', '16 ± 10', '10 ± 10', '3 ± 1'],
      ['rel-avito', 'user-visits', '2 ± 2', '0 ± 0', '1 ± 0'],
      ['rel-avito', 'user-clicks', '1 ± 1', '0 ± 0', '0 ± 1'],
      ['rel-trial', 'study-outcome', '0 ± 0', '0 ± 0', '0 ± 0'],
      ['rel-f1', 'driver-dnf', '19 ± 14', '0 ± 0', '1 ± 0'],
      ['rel-f1', 'driver-top3', '17 ± 11', '0 ± 0', '1 ± 0'],
    ],
  },
  {
    group: 'Regression tasks',
    rows: [
      ['rel-hm', 'item-sales', '39 ± 13', '0 ± 3', '1 ± 1'],
      ['rel-amazon', 'user-ltv', '4 ± 4', '1 ± 5', '2 ± 3'],
      ['rel-amazon', 'item-ltv', '9 ± 8', '2 ± 6', '2 ± 3'],
      ['rel-stack', 'post-votes', '16 ± 10', '4 ± 14', '2 ± 2'],
      ['rel-trial', 'site-success', '1 ± 2', '0 ± 1', '1 ± 1'],
      ['rel-trial', 'study-adverse', '0 ± 0', '0 ± 0', '0 ± 0'],
      ['rel-f1', 'driver-position', '14 ± 10', '0 ± 0', '1 ± 0'],
      ['rel-avito', 'ad-ctr', '1 ± 1', '0 ± 0', '0 ± 1'],
    ],
  },
];

export function Table12() {
  return (
    <div className="scroll-quiet overflow-x-auto">
      <table className="w-full min-w-[520px] border-collapse text-[15.5px]">
        <caption className="mb-2 text-left text-[16px] font-semibold text-ink">Paper Table 12: in-context labels per 1024-cell context (mean ± sd)</caption>
        <thead>
          <tr className="border-b border-grid text-left text-ink-2">
            <th scope="col" className="py-1.5 pr-3 font-medium">Dataset</th>
            <th scope="col" className="py-1.5 pr-3 font-medium">Task</th>
            <th scope="col" className="py-1.5 pr-3 text-right font-medium">Target-entity labels</th>
            <th scope="col" className="py-1.5 pr-3 text-right font-medium">Other-entity labels</th>
            <th scope="col" className="py-1.5 text-right font-medium">Unique labeled entities</th>
          </tr>
        </thead>
        {T12.map((g) => (
          <tbody key={g.group}>
            <tr>
              <th colSpan={5} scope="colgroup" className="pt-3 pb-1 text-left text-[14.5px] font-semibold text-muted">
                {g.group}
              </th>
            </tr>
            {g.rows.map((r) => (
              <tr key={r[0] + r[1]} className={cx('border-b border-grid/60', r[2] === '0 ± 0' && 'text-ink-2')}>
                <td className="py-1 pr-3 font-mono text-[14.5px]">{r[0]}</td>
                <td className="py-1 pr-3 font-mono text-[14.5px]">{r[1]}</td>
                <td className="num py-1 pr-3 text-right">{r[2]}</td>
                <td className="num py-1 pr-3 text-right">{r[3]}</td>
                <td className="num py-1 text-right">{r[4]}</td>
              </tr>
            ))}
          </tbody>
        ))}
      </table>
    </div>
  );
}
