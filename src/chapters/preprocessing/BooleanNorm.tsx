/**
 * Section 4b figure: boolean z-scores (pre.rs:320-329, 553-564; spec §3.3).
 * true → (1 − μ)/σ > 0, false → −μ/σ < 0, with μ = share of trues and σ the
 * sample std of the 0/1 values. A what-if slider changes the number of trues
 * (n fixed) and shows the NaN hazard of a constant column (σ = 0, no guard).
 */
import { scaleLinear } from 'd3';
import { motion } from 'motion/react';
import { useMemo, useState } from 'react';
import { Button, Segmented, Slider, StatusBadge, TeX, textWidth } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { catVar } from '@/styles/palette';
import { usePrefersReducedMotion } from '@/styles/theme';
import { booleanStats, sharePct, type StatColumn } from './data';
import { PickChip, Readout, fixed, signed, texNum, usePhone, type ChapterData } from './shared';

const nf = (n: number) => n.toLocaleString('en-US');

function useSpring() {
  const reduce = usePrefersReducedMotion();
  return reduce ? { duration: 0 } : { type: 'spring' as const, stiffness: 200, damping: 28 };
}

function ZLine({
  zTrue,
  zFalse,
  data,
  slot,
  isTarget,
}: {
  zTrue: number;
  zFalse: number;
  data?: { zTrue: number; zFalse: number };
  slot: number;
  /** The column is some task's prediction target: then the sign is the BCE label (model.py:252-255). */
  isTarget: boolean;
}) {
  const spring = useSpring();
  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(280, size.width || 640);
  const M = { l: 30, r: 16 };
  const finite = Number.isFinite(zTrue) && Number.isFinite(zFalse);
  const ext = finite ? Math.max(2, Math.ceil(Math.max(Math.abs(zTrue), Math.abs(zFalse)) + 0.25)) : 2;
  const x = scaleLinear().domain([-ext, ext]).range([M.l, W - M.r]);
  const ticks = x.ticks(Math.min(2 * ext + 1, Math.floor(W / 60)));
  const Y = 58;
  const leftLabel = isTarget ? 'z < 0 → BCE target 0 (if masked)' : 'z < 0: false side';
  const rightLabel = isTarget ? 'z > 0 → BCE target 1 (if masked)' : 'z > 0: true side';
  // two lines when the side labels would collide (phones)
  const stackLabels = textWidth(leftLabel, 13) + textWidth(rightLabel, 13) + 16 > x(ext) - x(-ext);
  const dot = (z: number, label: string, kind: 'true' | 'false', ghost = false) => {
    const lx = x(z);
    const text = `${label} ${signed(z, 3)}`;
    const tw = Math.ceil(textWidth(text, 13.5, { weight: 600 })) + 16;
    const tx = Math.max(2, Math.min(W - tw - 2, lx - tw / 2));
    const ty = Y - 44;
    return (
      <g key={`${kind}${ghost ? 'g' : ''}`} opacity={ghost ? 0.45 : 1}>
        {!ghost && (
          <>
            <motion.line initial={false} animate={{ x1: lx, x2: lx }} y1={ty + 20} y2={Y - 7} style={{ stroke: 'var(--axis)' }} transition={spring} />
            <motion.g initial={false} animate={{ x: tx }} transition={spring}>
              <rect y={ty} width={tw} height={20} rx={6} style={{ fill: 'var(--surface)', stroke: 'var(--border-strong)' }} />
              <text x={tw / 2} y={ty + 10} dy="0.34em" textAnchor="middle" className="tabular" style={{ fill: 'var(--ink)', fontSize: 13.5, fontWeight: 600 }}>
                {text}
              </text>
            </motion.g>
          </>
        )}
        <motion.circle
          initial={false}
          animate={{ cx: lx }}
          transition={spring}
          cy={Y}
          r={kind === 'true' ? 7 : 6}
          style={{
            fill: ghost ? 'none' : kind === 'true' ? catVar(slot) : 'var(--surface)',
            stroke: ghost ? 'var(--muted)' : kind === 'true' ? 'var(--surface)' : catVar(slot),
            strokeWidth: 2,
            strokeDasharray: ghost ? '2 2' : undefined,
          }}
        />
      </g>
    );
  };
  return (
    <div ref={ref} className="w-full">
      <svg width={W} height={Y + (stackLabels ? 68 : 52)} role="img" aria-label={`z of true ${signed(zTrue)}, z of false ${signed(zFalse)}`} className="block">
        {/* halves */}
        <rect x={x(-ext)} y={Y - 12} width={x(0) - x(-ext)} height={24} rx={4} style={{ fill: 'var(--surface-2)' }} />
        <rect x={x(0)} y={Y - 12} width={x(ext) - x(0)} height={24} rx={4} style={{ fill: 'var(--accent-soft)' }} />
        <line x1={x(-ext)} x2={x(ext)} y1={Y} y2={Y} style={{ stroke: 'var(--axis)' }} />
        <line x1={x(0)} x2={x(0)} y1={Y - 16} y2={Y + 16} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.5} />
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={Y + 29} textAnchor="middle" className="tabular" style={{ fill: t === 0 ? 'var(--ink)' : 'var(--muted)', fontSize: 13, fontWeight: t === 0 ? 600 : 400 }}>
            {t === 0 ? '0' : t > 0 ? `+${t}` : `−${-t}`}
          </text>
        ))}
        <text x={M.l - 10} y={Y} dy="0.34em" textAnchor="end" style={{ fill: 'var(--muted)', fontSize: 13, fontStyle: 'italic' }}>
          z
        </text>
        <text x={x(-ext)} y={Y + 47} style={{ fill: 'var(--ink-2)', fontSize: 13 }}>
          {leftLabel}
        </text>
        <text x={x(ext)} y={Y + (stackLabels ? 64 : 47)} textAnchor="end" style={{ fill: 'var(--ink-2)', fontSize: 13 }}>
          {rightLabel}
        </text>
        {finite && data && dot(data.zFalse, 'false', 'false', true)}
        {finite && data && dot(data.zTrue, 'true', 'true', true)}
        {finite && dot(zFalse, 'false', 'false')}
        {finite && dot(zTrue, 'true', 'true')}
        {!finite && (
          <text x={x(0)} y={Y - 24} textAnchor="middle" style={{ fill: 'var(--ink)', fontSize: 14.5, fontWeight: 600 }}>
            every value: (b − μ) / 0 = NaN
          </text>
        )}
      </svg>
    </div>
  );
}

export function BooleanNormFigure({ data, cols, defaultId }: { data: ChapterData; cols: StatColumn[]; defaultId: string }) {
  const [id, setId] = useState(defaultId);
  const c = cols.find((x) => x.id === id) ?? cols[0];
  const n = c?.values.length ?? 0;
  const kData = c ? c.values.reduce((s, v) => s + v, 0) : 0;
  const [whatIf, setWhatIf] = useState<Record<string, number>>({});
  const spring = useSpring();
  const phone = usePhone();
  // the values the engine actually stored for a true and a false row of this column
  const stored = useMemo(() => {
    let zTrue: number | null = null;
    let zFalse: number | null = null;
    if (!c) return { zTrue, zFalse };
    const rows = c.file.table.rows;
    for (let r = 0; r < rows.length && (zTrue === null || zFalse === null); r++) {
      const v = rows[r][c.col.name];
      if (v === null || v === undefined) continue;
      const cell = data.pdb.nodes[c.file.offset + r]?.cells.find((x) => x.colName === c.col.name);
      if (!cell) continue;
      if (v && zTrue === null) zTrue = cell.boolean;
      else if (!v && zFalse === null) zFalse = cell.boolean;
    }
    return { zTrue, zFalse };
  }, [c, data.pdb]);
  if (!c) return <p className="text-[16.5px] text-ink-2">This database has no boolean columns.</p>;
  const k = whatIf[c.id] ?? kData;
  const setK = (v: number) => setWhatIf((w) => ({ ...w, [c.id]: Math.max(0, Math.min(n, Math.round(v))) }));
  const isWhatIf = k !== kData;
  // from the data: the engine's colStats and stored cell values; what-if: pre.rs:320-329 recomputed for k trues
  const real = {
    mean: c.stats.mean,
    std: c.stats.std,
    zTrue: stored.zTrue ?? (1 - c.stats.mean) / c.stats.std,
    zFalse: stored.zFalse ?? -c.stats.mean / c.stats.std,
  };
  const s = isWhatIf ? booleanStats(n, k) : real;
  const nan = !Number.isFinite(s.zTrue) || !Number.isFinite(s.zFalse);
  const slot = data.slotOf(c.file.tableName);
  const pTrue = n ? k / n : 0;
  // is this column a prediction target of some task of the DB (forecast label column, or an autocomplete target)?
  const isTarget = data.db.tasks.some((t) =>
    t.kind === 'forecast' ? c.file.isTask && t.name === c.file.tableName && t.targetCol === c.col.name : t.table === c.file.tableName && t.targetCol === c.col.name,
  );
  // quick picks; the label shows the share the chip really sets (k is rounded)
  const pctLabel = (kk: number) => sharePct(kk, n);
  const chips = [
    { k: Math.max(1, Math.round(n / 20)), text: (kk: number) => `${pctLabel(kk)} true` },
    { k: Math.round(n / 4), text: (kk: number) => `${pctLabel(kk)} true` },
    { k: Math.round(n / 2), text: (kk: number) => pctLabel(kk) },
    { k: n, text: () => 'all true' },
    { k: 0, text: () => 'all false' },
  ].filter((ch, i, a) => a.findIndex((o) => o.k === ch.k) === i);

  return (
    <div className="min-w-0">
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <Segmented
          label="Boolean column"
          value={c.id}
          onChange={setId}
          options={cols.map((x) => ({ value: x.id, label: <span className="font-mono">{`${x.file.tableName}.${x.col.name}`}</span> }))}
        />
        {c.file.isTask && <span className="text-[14.5px] text-ink-2">Train split: Val and Test reuse these stats.</span>}
      </div>

      {/* share of trues */}
      <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3 text-[15px] text-ink-2">
        <span>
          <strong className="num text-ink">{nf(k)}</strong> true · <strong className="num text-ink">{nf(n - k)}</strong> false (n = {nf(n)})
        </span>
        {isWhatIf ? <span className="font-medium text-accent-text">what-if</span> : <span>from the data</span>}
      </div>
      <div className="flex h-3.5 w-full overflow-hidden rounded-[4px]" aria-hidden="true">
        <motion.div initial={false} animate={{ width: `${pTrue * 100}%` }} transition={spring} style={{ background: catVar(slot) }} />
        <motion.div initial={false} animate={{ width: `${(1 - pTrue) * 100}%` }} transition={spring} style={{ background: 'var(--axis)', marginLeft: pTrue > 0 && pTrue < 1 ? 2 : 0 }} />
      </div>

      <div className="mt-4">
        <ZLine zTrue={s.zTrue} zFalse={s.zFalse} data={isWhatIf && Number.isFinite(real.zTrue) ? real : undefined} slot={slot} isTarget={isTarget} />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Readout label="μ = share of trues">{fixed(s.mean, 4)}</Readout>
        <Readout label="sample std σ" hint="ddof = 1, no zero guard">
          {fixed(s.std, 4)}
        </Readout>
        <Readout label="true is stored as" hint={k === 0 ? 'no true values in the column' : undefined}>
          {k === 0 ? '—' : signed(s.zTrue, 3)}
        </Readout>
        <Readout label="false is stored as" hint={k === n ? 'no false values in the column' : undefined}>
          {k === n ? '—' : signed(s.zFalse, 3)}
        </Readout>
      </div>

      <div className="mt-4 grid items-end gap-3 md:grid-cols-[minmax(0,1fr)_auto]">
        <Slider label="What if the column had this many trues? (n fixed)" value={k} min={0} max={n} onChange={setK} format={(v) => `${nf(v)} / ${nf(n)} (${((100 * v) / Math.max(1, n)).toFixed(1)}%)`} />
        <div className="flex flex-wrap items-center gap-1.5">
          {chips.map((ch) => (
            <PickChip key={ch.k} active={k === ch.k} onClick={() => setK(ch.k)} title={`${nf(ch.k)} of ${nf(n)} true`}>
              {ch.text(ch.k)}
            </PickChip>
          ))}
          <Button size="sm" variant="ghost" onClick={() => setK(kData)} disabled={!isWhatIf}>
            Reset to data
          </Button>
        </div>
      </div>

      <div className="mt-3 scroll-quiet overflow-x-auto rounded-lg bg-surface-2 px-3 py-2 text-[18px]" aria-live="polite">
        {nan ? (
          <div className="flex flex-wrap items-center gap-2 py-0.5">
            <StatusBadge status="critical">σ = 0: every value becomes 0/0 = NaN</StatusBadge>
            <span className="text-[16px] text-ink-2">A constant column has no zero guard (pre.rs:323), so the NaN goes straight into the batch.</span>
          </div>
        ) : (
          <TeX
            tex={
              phone
                ? `\\begin{aligned} z_{\\text{true}} &= \\dfrac{1 - ${texNum(s.mean, 4)}}{${texNum(s.std, 4)}} = ${texNum(s.zTrue, 3, { sign: true })} \\\\[4pt] z_{\\text{false}} &= \\dfrac{0 - ${texNum(s.mean, 4)}}{${texNum(s.std, 4)}} = ${texNum(s.zFalse, 3, { sign: true })} \\end{aligned}`
                : `z_{\\text{true}} = \\dfrac{1 - ${texNum(s.mean, 4)}}{${texNum(s.std, 4)}} = ${texNum(s.zTrue, 3, { sign: true })}, \\qquad z_{\\text{false}} = \\dfrac{0 - ${texNum(s.mean, 4)}}{${texNum(s.std, 4)}} = ${texNum(s.zFalse, 3, { sign: true })}`
            }
          />
        )}
      </div>
    </div>
  );
}
