/**
 * Step 4 figure: one encoder per type. A scalar z enters as RMSNorm_t(z·w_t + b_t) (model.py:216-220):
 * a straight line in ℝ^d bent onto (nearly) the sphere of radius √d, then scaled by γ_t. All values of
 * one type, from every column, lie on this one curve (spec §5.2 teaching note).
 *
 * Every point is computed with the engine's `scalarValueTerm` and the live weights; the 2-D view is a
 * PCA of the sweep z ∈ [−3, 3] (display only). The column's real cells are its stored z-scores from
 * preprocessing (all rows of its table file). The strips and the "same z, another column" panel embed
 * synthetic cells of the two columns with the engine's `embedTokens` (model.py:208-224).
 */
import { scaleLinear } from 'd3';
import { memo, useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { Cite, Figure, LineChart, Segmented, Select, SemBadge, Slider, SVG_TYPE, StatusBadge, VectorStrip, formatDate, maxAbs } from '@/components';
import { axis, vec } from '../shapes';
import { useElementSize } from '@/components/viz/useSize';
import { embedTokens, tensor, textEmbedder } from '@/engine/model';
import { SemType, type Token } from '@/engine/types';
import type { EmbeddingStable } from './live';
import { fmt, fmtSigned, int } from './live';
import {
  columnProjection,
  cosine,
  maxAbsDelta,
  meanSquare,
  norm,
  pca2,
  project,
  scalarProjection,
  syntheticCell,
  zToRaw,
  type FeatureColumn,
  type ScalarKey,
} from './math';

const SEM_OF: Record<ScalarKey, SemType> = { number: SemType.Number, datetime: SemType.DateTime, boolean: SemType.Boolean };
const Z_MIN = -3;
const Z_MAX = 3;
const N_SWEEP = 121;
/** Coarser sweep for the saturation chart: z in steps of 0.1 (61 points, distinct one-decimal labels). */
const N_CHART = 61;

function rawText(col: FeatureColumn, z: number): string {
  if (col.sem === SemType.Boolean) return z > 0 ? 'true' : 'false';
  const r = zToRaw(col, z);
  if (r === null) return '—';
  if (col.sem === SemType.DateTime) return formatDate(r, 'never');
  const a = Math.abs(r);
  return a >= 100 ? r.toFixed(0) : a >= 10 ? r.toFixed(1) : r.toFixed(2);
}

interface Dot {
  z: number;
  p: [number, number];
  n: number;
}

export const ValueCurve = memo(function ValueCurve({ live }: { live: EmbeddingStable }) {
  const { weights, store, ctx, pdb, contextColumns, assets } = live;
  const [key, setKey] = useState<ScalarKey>('number');
  const sem = SEM_OF[key];
  const cols = useMemo(() => contextColumns.filter((c) => c.sem === sem), [contextColumns, sem]);
  const inContext = useMemo(() => {
    const s = new Set<string>();
    for (const t of ctx.tokens) if (!t.isPadding) s.add(`${t.tableName}.${t.colName}`);
    return s;
  }, [ctx]);
  const [colKey, setColKey] = useState<string>('');
  const [otherKey, setOtherKey] = useState<string>('');
  const col = cols.find((c) => c.key === colKey) ?? cols.find((c) => inContext.has(c.key)) ?? cols[0];
  const other = cols.find((c) => c.key === otherKey && c.key !== col?.key) ?? cols.find((c) => c.key !== col?.key) ?? null;

  // the first real cell of the column in the context: the default z
  const firstZ = useMemo(() => {
    if (!col) return 0;
    const t = ctx.tokens.find((u) => !u.isPadding && !u.mask && `${u.tableName}.${u.colName}` === col.key);
    if (!t) return 0;
    return sem === SemType.Number ? t.numberValue : sem === SemType.DateTime ? t.datetimeValue : t.booleanValue;
  }, [col, ctx, sem]);
  const [z, setZ] = useState(0);
  useEffect(() => setZ(Math.max(Z_MIN, Math.min(Z_MAX, Math.round(firstZ * 100) / 100))), [firstZ]);

  // ── sweep, PCA basis, limits, saturation ──
  const curve = useMemo(() => {
    if (!weights) return null;
    const d = weights.config.dModel;
    const eps = weights.config.eps;
    const zs = Array.from({ length: N_SWEEP }, (_, i) => Z_MIN + ((Z_MAX - Z_MIN) * i) / (N_SWEEP - 1));
    const sweep = zs.map((zz) => scalarProjection(weights, key, zz));
    const terms = sweep.map((p) => p.term);
    const basis = pca2(terms, d);
    const mean = new Float64Array(d);
    for (const t of terms) for (let j = 0; j < d; j++) mean[j] += t[j] / terms.length;
    const proj = (v: ArrayLike<number>) => {
      const c = new Float64Array(d);
      for (let j = 0; j < d; j++) c[j] = v[j] - mean[j];
      return [project(c, basis.e1), project(c, basis.e2)] as [number, number];
    };
    // z → ±∞: z·w dominates the bias, RMSNorm(±w) without ε (ε/z² → 0), times γ
    const w = tensor(weights, `enc_dict.${key}.weight`);
    const g = tensor(weights, `norm_dict.${key}.weight`);
    const rmsW = Math.sqrt(meanSquare(w));
    const limPos = new Float32Array(d);
    const limNeg = new Float32Array(d);
    for (let j = 0; j < d; j++) {
      limPos[j] = (w[j] / rmsW) * g[j];
      limNeg[j] = -limPos[j];
    }
    // ‖normalized vector before γ‖ along the sweep: stays near √d unless mean(x²) is comparable to ε
    const preLen = sweep.map((p) => norm(p.pre) / Math.sqrt(meanSquare(p.pre) + eps));
    // saturation: cosine of the value term with each limit, z in steps of 0.1
    const zc = Array.from({ length: N_CHART }, (_, i) => Math.round((Z_MIN + ((Z_MAX - Z_MIN) * i) / (N_CHART - 1)) * 10) / 10);
    const chartTerms = zc.map((zz) => scalarProjection(weights, key, zz).term);
    const satPos = zc.map((x, i) => ({ x, y: cosine(chartTerms[i], limPos) }));
    const satNeg = zc.map((x, i) => ({ x, y: cosine(chartTerms[i], limNeg) }));
    return {
      d,
      zs,
      pts: terms.map(proj),
      proj,
      basis,
      limPos: proj(limPos),
      limNeg: proj(limNeg),
      limPosVec: limPos,
      limNegVec: limNeg,
      preLenMin: Math.min(...preLen),
      preLenMax: Math.max(...preLen),
      satPos,
      satNeg,
    };
  }, [weights, key]);

  const cur = useMemo(() => (weights ? scalarProjection(weights, key, z) : null), [weights, key, z]);

  /** The selected column's cell and the other column's cell at the same z, embedded by the engine. */
  const pair = useMemo(() => {
    if (!weights || !store || !col) return null;
    const d = weights.config.dModel;
    const a = syntheticCell(pdb, col, z);
    const b = other ? syntheticCell(pdb, other, z) : null;
    if (!a) {
      // phrase not in the string table (should not happen): fall back to the same kernels directly
      const colTerm = columnProjection(weights, store, col.phrase).term;
      const val = scalarProjection(weights, key, z).term;
      const x0 = new Float32Array(d);
      for (let j = 0; j < d; j++) x0[j] = colTerm[j] + val[j];
      return { a: { colTerm, valueTerm: val, x0 }, b: null };
    }
    const toks: Token[] = b ? [a, { ...b, pos: 1 }] : [a];
    const e = embedTokens(toks, textEmbedder(pdb, store), weights);
    const part = (i: number) => ({
      colTerm: e.colTerm.slice(i * d, (i + 1) * d),
      valueTerm: e.valueTerm.slice(i * d, (i + 1) * d),
      x0: e.x0.slice(i * d, (i + 1) * d),
    });
    return { a: part(0), b: b ? part(1) : null };
  }, [weights, store, pdb, col, other, z, key]);

  const real = useMemo<Dot[]>(() => {
    if (!col || !curve || !weights) return [];
    const counts = new Map<number, number>();
    for (const v of col.zs) {
      const r = Math.round(v * 1e6) / 1e6;
      counts.set(r, (counts.get(r) ?? 0) + 1);
    }
    return [...counts].sort((x, y) => x[0] - y[0]).map(([v, n]) => ({ z: v, p: curve.proj(scalarProjection(weights, key, v).term), n }));
  }, [col, curve, weights, key]);

  const extension = useMemo(() => {
    if (!col || !curve || !weights || !col.zs.length) return null;
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of col.zs) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    const seg = (a: number, b: number) => {
      const n = 30;
      return Array.from({ length: n + 1 }, (_, i) => curve.proj(scalarProjection(weights, key, a + ((b - a) * i) / n).term));
    };
    return { lo: lo < Z_MIN ? seg(lo, Z_MIN) : null, hi: hi > Z_MAX ? seg(Z_MAX, hi) : null, min: lo, max: hi };
  }, [col, curve, weights, key]);

  // ── plot geometry (equal aspect) ──
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const side = Math.round(Math.max(260, Math.min(size.width || 460, 480)));
  const pad = 34;
  const allPts = curve ? [...curve.pts, curve.limPos, curve.limNeg, ...real.map((r) => r.p), ...(extension?.lo ?? []), ...(extension?.hi ?? [])] : [];
  const xs = allPts.map((p) => p[0]);
  const ys = allPts.map((p) => p[1]);
  const cxD = xs.length ? (Math.min(...xs) + Math.max(...xs)) / 2 : 0;
  const cyD = ys.length ? (Math.min(...ys) + Math.max(...ys)) / 2 : 0;
  const span = xs.length ? Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 0.56 || 1 : 1;
  const sx = scaleLinear().domain([cxD - span, cxD + span]).range([pad, side - pad]);
  const sy = scaleLinear().domain([cyD - span, cyD + span]).range([side - pad, pad]);
  const path = (pts: [number, number][]) => pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join('');
  const curP = cur && curve ? curve.proj(cur.term) : null;
  const intLabels = side < 340 ? [-3, 0, 3] : [-3, -2, -1, 0, 1, 2, 3];
  const domain = pair ? Math.max(maxAbs(pair.a.valueTerm), maxAbs(pair.a.colTerm), maxAbs(pair.a.x0)) || 1 : 1;

  // stored-value dots: hover / focus tooltip, roving tab stop
  const [hoverDot, setHoverDot] = useState<number | null>(null);
  const [focusDot, setFocusDot] = useState<number | null>(null);
  const dotRefs = useRef(new Map<number, SVGGElement>());
  useEffect(() => {
    setHoverDot(null);
    setFocusDot(null);
  }, [col?.key, key]);
  const shownDot = hoverDot ?? focusDot;
  const goDot = (i: number) => {
    const j = Math.max(0, Math.min(real.length - 1, i));
    setFocusDot(j);
    dotRefs.current.get(j)?.focus();
  };
  const pickDot = (r: Dot) => setZ(Math.max(Z_MIN, Math.min(Z_MAX, Math.round(r.z * 100) / 100)));
  const onDotKey = (e: KeyboardEvent<SVGGElement>, i: number) => {
    const k = e.key;
    if (k === 'ArrowRight' || k === 'ArrowUp') goDot(i + 1);
    else if (k === 'ArrowLeft' || k === 'ArrowDown') goDot(i - 1);
    else if (k === 'Home') goDot(0);
    else if (k === 'End') goDot(real.length - 1);
    else if (k === 'Enter' || k === ' ') pickDot(real[i]);
    else if (k === 'Escape') setFocusDot(null);
    else return;
    e.preventDefault();
  };
  const dotLabel = (r: Dot) =>
    col ? `${col.col} = ${rawText(col, r.z)}, z = ${fmtSigned(r.z, 3)}, ${int(r.n)} ${r.n === 1 ? 'cell' : 'cells'}${r.z < Z_MIN || r.z > Z_MAX ? ' (beyond the slider range)' : ''}` : '';

  /** Integer-z tick labels, pushed outward along the curve's normal, skipped when a dot is in the way. */
  const ticks = useMemo(() => {
    if (!curve) return [];
    const scr = curve.pts.map((p) => [sx(p[0]), sy(p[1])] as const);
    const mx = scr.reduce((a, p) => a + p[0], 0) / scr.length;
    const my = scr.reduce((a, p) => a + p[1], 0) / scr.length;
    const dots = real.map((r) => [sx(r.p[0]), sy(r.p[1])] as const);
    return intLabels.flatMap((k) => {
      const i = Math.round(((k - Z_MIN) / (Z_MAX - Z_MIN)) * (N_SWEEP - 1));
      const p = scr[i];
      const a = scr[Math.max(0, i - 2)];
      const b = scr[Math.min(scr.length - 1, i + 2)];
      let nx = -(b[1] - a[1]);
      let ny = b[0] - a[0];
      const l = Math.hypot(nx, ny) || 1;
      nx /= l;
      ny /= l;
      // outward = away from the centroid of the arc
      if (nx * (p[0] - mx) + ny * (p[1] - my) < 0) {
        nx = -nx;
        ny = -ny;
      }
      const lx = p[0] + nx * 16;
      const ly = p[1] + ny * 16;
      const blocked = dots.some((q) => Math.hypot(q[0] - lx, q[1] - ly) < 10);
      return [{ k, p, lx, ly, anchor: nx > 0.35 ? 'start' : nx < -0.35 ? 'end' : 'middle', blocked } as const];
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [curve, real, side, span, cxD, cyD]);

  const typeOptions: { value: ScalarKey; label: ReactNode }[] = [
    { value: 'number', label: <span className="inline-flex items-center gap-1.5"><SemBadge sem={SemType.Number} size="sm" bare /> number</span> },
    { value: 'datetime', label: <span className="inline-flex items-center gap-1.5"><SemBadge sem={SemType.DateTime} size="sm" bare /> datetime</span> },
    { value: 'boolean', label: <span className="inline-flex items-center gap-1.5"><SemBadge sem={SemType.Boolean} size="sm" bare /> boolean</span> },
  ];

  const sat3 = curve ? curve.satPos[curve.satPos.length - 1].y : NaN;
  const satM3 = curve ? curve.satNeg[0].y : NaN;

  return (
    <Figure
      wide
      title={`Move a value along its type’s curve: RMSNorm_${key}(z·w + b)`}
      controls={
        <>
          <div className="flex flex-col gap-1">
            <span className="text-[15px] font-medium text-ink-2">Type</span>
            <Segmented
              label="Semantic type"
              value={key}
              onChange={(k) => {
                setKey(k);
                setColKey('');
                setOtherKey('');
              }}
              options={typeOptions}
            />
          </div>
          <Select label="Column" value={col?.key ?? ''} onChange={(e) => setColKey(e.target.value)} className="w-full sm:w-[280px]">
            {cols.map((c) => (
              <option key={c.key} value={c.key}>
                {c.phrase}
                {inContext.has(c.key) ? '' : ' (not in context)'}
              </option>
            ))}
          </Select>
          <Slider
            label="z (stored value)"
            value={z}
            min={Z_MIN}
            max={Z_MAX}
            step={0.01}
            onChange={setZ}
            format={(v) => (
              <>
                {fmtSigned(v, 2)} {col ? <span className="text-ink-2">≈ {rawText(col, v)}</span> : null}
              </>
            )}
            className="w-full min-w-[220px] sm:w-[300px]"
          />
        </>
      }
      caption={
        <>
          The curve is RMSNorm_{key}(z·w + b) for z from −3 to 3, computed with the weights of the checkpoint this page runs and drawn in its own two
          main directions (PCA, {curve ? `${Math.round(curve.basis.explained * 100)}%` : '…'} of its variance). Hollow dots: every distinct stored
          value of the selected column; hover, tap or focus one for its value, click or press Enter to move there. × marks the limits z → ±∞. Every{' '}
          {key} column of every table lands on this same curve.
        </>
      }
    >
      {assets.weightsKind === 'init' && <StatusBadge status="warning">Untrained weights: values from a random initialization</StatusBadge>}
      <div ref={wrapRef} className="h-0 w-full max-w-[480px]" aria-hidden="true" />
      {!curve || !cur || !col || !pair ? (
        <div className="rounded-lg border border-dashed border-border-strong px-4 py-8 text-center text-[16px] text-ink-2">Loading the weights…</div>
      ) : (
        <div className="grid items-start gap-6 min-[1200px]:grid-cols-[minmax(0,480px)_minmax(0,1fr)]">
          <div className="min-w-0">
            <div className="relative" style={{ width: side, maxWidth: '100%' }}>
              <svg
                width={side}
                height={side}
                viewBox={`0 0 ${side} ${side}`}
                className="block max-w-full"
                role="group"
                aria-label={`The ${key} value curve, current z = ${fmt(z, 2)}. The stored values of ${col.phrase} are buttons; arrow keys move between them.`}
                onPointerDown={(e) => {
                  if (e.target === e.currentTarget) setFocusDot(null);
                }}
              >
                <rect x={1} y={1} width={side - 2} height={side - 2} rx={8} style={{ fill: 'var(--surface-2)', stroke: 'var(--border)' }} />
                <line x1={sx(cxD - span)} x2={sx(cxD + span)} y1={sy(0)} y2={sy(0)} style={{ stroke: 'var(--grid)' }} />
                <line x1={sx(0)} x2={sx(0)} y1={sy(cyD - span)} y2={sy(cyD + span)} style={{ stroke: 'var(--grid)' }} />
                <text x={12} y={side - 12} fontSize={SVG_TYPE.annotation} style={{ fill: 'var(--muted)' }}>
                  axes: PC 1 →, PC 2 ↑
                </text>
                {extension?.lo && <path d={path(extension.lo)} fill="none" style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.5} strokeDasharray="4 3" />}
                {extension?.hi && <path d={path(extension.hi)} fill="none" style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.5} strokeDasharray="4 3" />}
                <path d={path(curve.pts)} fill="none" style={{ stroke: 'var(--ink-2)' }} strokeWidth={2.5} strokeLinecap="round" />
                {/* limits */}
                {(
                  [
                    [curve.limNeg, 'z → −∞'],
                    [curve.limPos, 'z → +∞'],
                  ] as const
                ).map(([p, label]) => (
                  <g key={label}>
                    <path
                      d={`M${sx(p[0]) - 5},${sy(p[1]) - 5}L${sx(p[0]) + 5},${sy(p[1]) + 5}M${sx(p[0]) - 5},${sy(p[1]) + 5}L${sx(p[0]) + 5},${sy(p[1]) - 5}`}
                      style={{ stroke: 'var(--ink)' }}
                      strokeWidth={2}
                    />
                    <text
                      x={sx(p[0]) + (sx(p[0]) > side / 2 ? -9 : 9)}
                      y={sy(p[1]) + (sy(p[1]) > side / 2 ? -9 : 17)}
                      textAnchor={sx(p[0]) > side / 2 ? 'end' : 'start'}
                      fontSize={SVG_TYPE.annotation}
                      style={{ fill: 'var(--ink)' }}
                    >
                      {label}
                    </text>
                  </g>
                ))}
                {/* integer z ticks: a dot on the curve, the label pushed outward */}
                {ticks.map((t) => (
                  <g key={t.k} pointerEvents="none">
                    <circle cx={t.p[0]} cy={t.p[1]} r={3} style={{ fill: 'var(--ink-2)' }} />
                    {!t.blocked && (
                      <text x={t.lx} y={t.ly + 4.5} textAnchor={t.anchor} fontSize={SVG_TYPE.label} style={{ fill: 'var(--ink-2)' }}>
                        {t.k > 0 ? `+${t.k}` : t.k < 0 ? `−${-t.k}` : '0'}
                      </text>
                    )}
                  </g>
                ))}
                {/* real cells: buttons with a roving tab stop */}
                {real.map((r, i) => {
                  const on = i === shownDot;
                  return (
                    <g
                      key={r.z}
                      ref={(el) => {
                        if (el) dotRefs.current.set(i, el);
                        else dotRefs.current.delete(i);
                      }}
                      role="button"
                      tabIndex={i === (focusDot ?? 0) ? 0 : -1}
                      aria-label={dotLabel(r)}
                      className="cursor-pointer outline-none"
                      onPointerEnter={(e) => e.pointerType === 'mouse' && setHoverDot(i)}
                      onPointerLeave={() => setHoverDot((h) => (h === i ? null : h))}
                      onClick={() => {
                        setFocusDot(i);
                        pickDot(r);
                      }}
                      onFocus={() => setFocusDot(i)}
                      onKeyDown={(e) => onDotKey(e, i)}
                    >
                      <circle cx={sx(r.p[0])} cy={sy(r.p[1])} r={10} fill="transparent" />
                      {on && <circle cx={sx(r.p[0])} cy={sy(r.p[1])} r={8.5} fill="none" style={{ stroke: 'var(--ink)' }} strokeWidth={1.5} />}
                      <circle cx={sx(r.p[0])} cy={sy(r.p[1])} r={5} fill="none" style={{ stroke: 'var(--accent)' }} strokeWidth={1.5} />
                    </g>
                  );
                })}
                {/* current */}
                {curP && (
                  <circle
                    cx={sx(curP[0])}
                    cy={sy(curP[1])}
                    r={8}
                    pointerEvents="none"
                    style={{ fill: 'var(--accent)', stroke: 'var(--surface)' }}
                    strokeWidth={2.5}
                  />
                )}
              </svg>
              {shownDot !== null && real[shownDot] && (
                <div
                  className="pointer-events-none absolute z-10 w-max max-w-[240px] rounded-lg border border-border bg-surface px-3 py-2 text-[14.5px] leading-snug shadow-lg"
                  style={{
                    left: Math.min(Math.max(4, sx(real[shownDot].p[0]) + 14), side - 200),
                    top: Math.min(Math.max(4, sy(real[shownDot].p[1]) - 58), side - 70),
                  }}
                >
                  <div className="font-medium text-ink">
                    {col.col} = {rawText(col, real[shownDot].z)}
                  </div>
                  <div className="num text-ink-2">
                    z = {fmtSigned(real[shownDot].z, 3)} · {int(real[shownDot].n)} {real[shownDot].n === 1 ? 'cell' : 'cells'}
                  </div>
                </div>
              )}
            </div>
            <div className="mt-2 text-[15px] text-ink-2">
              {int(real.length)} distinct stored {real.length === 1 ? 'value' : 'values'} in <span>{col.table}</span>
              {extension && (extension.min < Z_MIN || extension.max > Z_MAX) ? `, z from ${fmtSigned(extension.min, 2)} to ${fmtSigned(extension.max, 2)} (dashed beyond ±3)` : ''}
            </div>
          </div>

          <div className="min-w-0">
            <div className="grid gap-3">
              <StripRow label={`value term at z = ${fmtSigned(z, 2)}`} values={pair.a.valueTerm} domain={domain} />
              <StripRow label={`column term of “${col.phrase}”`} values={pair.a.colTerm} domain={domain} />
              <StripRow label="x₀ = column term + value term" values={pair.a.x0} domain={domain} strong />
            </div>
            <div className="mt-5">
              <LineChart
                title="How close the value term is to its limits"
                subtitle={
                  <span className="text-[15px]">
                    Cosine between the value term at z and the two limits ×, as z sweeps. Near 1: the curve has flattened toward that limit.
                  </span>
                }
                series={[
                  { id: 'pos', label: 'cos with the z → +∞ limit', points: curve.satPos },
                  { id: 'neg', label: 'cos with the z → −∞ limit', points: curve.satNeg },
                ]}
                xLabel="z"
                yLabel="cosine"
                height={210}
                yDomain={[-1, 1]}
                markers={[{ x: z, label: 'z' }]}
                formatY={(v) => fmtSigned(v, 2)}
                formatX={(v) => fmtSigned(v, 1)}
              />
              <p className="!mt-2 !mb-0 text-[15px] leading-snug text-ink-2">
                At z = +3 the value term already has cosine {fmt(sat3, 3)} with its +∞ limit, and at z = −3 cosine {fmt(satM3, 3)} with its −∞ limit.
                Before γ its length stays between {fmt(curve.preLenMin, 2)} and {fmt(curve.preLenMax, 2)} over the whole sweep, against √d ={' '}
                {fmt(Math.sqrt(curve.d), 0)}: ε shortens it by at most {((1 - curve.preLenMin / Math.sqrt(curve.d)) * 100).toFixed(1)}%, where z·w + b is
                shortest.
              </p>
            </div>
          </div>
        </div>
      )}

      {curve && cur && col && other && pair?.b && (
        <div className="mt-6 rounded-xl border border-border bg-surface-2/60 p-4">
          <div className="mb-3 flex flex-wrap items-end gap-x-4 gap-y-2">
            <div className="text-[16.5px] font-semibold text-ink">Same z, another {key} column</div>
            <Select label="Compare with" value={other.key} onChange={(e) => setOtherKey(e.target.value)} className="w-full sm:w-[280px]">
              {cols
                .filter((c) => c.key !== col.key)
                .map((c) => (
                  <option key={c.key} value={c.key}>
                    {c.phrase}
                  </option>
                ))}
            </Select>
          </div>
          <p className="!mt-0 !mb-3 text-[15px] leading-snug text-ink-2">
            Two cells with z = {fmtSigned(z, 2)}, one per column, embedded by the engine’s <code className="text-[14.5px]">embedTokens</code>{' '}
            <Cite kind="code" ref="rt/model.py:208-224" />.
          </p>
          <div className="grid gap-3 md:grid-cols-3">
            <CompareCell
              title="Value term"
              value={maxAbsDelta(pair.a.valueTerm, pair.b.valueTerm) === 0 ? 'the same vector' : `max |Δ| = ${maxAbsDelta(pair.a.valueTerm, pair.b.valueTerm).toExponential(1)}`}
              note={
                <>
                  Same type and same z: the same input to the same encoder, so the same {cur.term.length} numbers (max |Δ| ={' '}
                  {maxAbsDelta(pair.a.valueTerm, pair.b.valueTerm)}).{' '}
                  {sem === SemType.DateTime ? (
                    <>
                      Datetimes share one global z-score per database, so z = {fmtSigned(z, 2)} is the same moment ({rawText(col, z)}) in both columns{' '}
                      <Cite kind="spec" ref="§3.3" />. Only the column term says which event it was.
                    </>
                  ) : sem === SemType.Boolean ? (
                    <>
                      z = {fmtSigned(z, 2)} reads as {rawText(col, z)} in both columns (the sign carries the label <Cite kind="spec" ref="§3.3" />); the
                      encoder never sees which column.
                    </>
                  ) : (
                    <>
                      Yet z = {fmtSigned(z, 2)} means {col.col} ≈ {rawText(col, z)} but {other.col} ≈ {rawText(other, z)}.
                    </>
                  )}
                </>
              }
            />
            <CompareCell
              title="Column term"
              value={`cosine ${fmt(cosine(pair.a.colTerm, pair.b.colTerm), 3)}`}
              note={
                <>
                  “{col.phrase}” vs “{other.phrase}”: the only difference between the two tokens. Lengths {fmt(norm(pair.a.colTerm), 2)} and{' '}
                  {fmt(norm(pair.b.colTerm), 2)}.
                </>
              }
            />
            <CompareCell
              title="x₀"
              value={`cosine ${fmt(cosine(pair.a.x0, pair.b.x0), 3)}`}
              note={
                <>
                  The two x₀ differ exactly by the difference of their column terms. In the checkpoint this page runs the shared value term is{' '}
                  {fmt(norm(pair.a.valueTerm), 2)} long
                  {norm(pair.a.valueTerm) > Math.max(norm(pair.a.colTerm), norm(pair.b.colTerm))
                    ? ', longer than either column term, so the two x₀ point in similar directions: what tells the two cells apart is the smaller column component.'
                    : '.'}
                </>
              }
            />
          </div>
        </div>
      )}
    </Figure>
  );
});

function StripRow({ label, values, domain, strong }: { label: string; values: Float32Array; domain: number; strong?: boolean }) {
  return (
    <div className="min-w-0">
      <div className={strong ? 'mb-1 text-[15.5px] font-semibold text-ink' : 'mb-1 text-[15.5px] text-ink-2'}>{label}</div>
      <VectorStrip values={values} columns={32} cellHeight={12} domain={domain} shape={vec(label, axis.model(values.length))} />
    </div>
  );
}

function CompareCell({ title, value, note }: { title: string; value: string; note: ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-2.5">
      <div className="text-[15px] text-ink-2">{title}</div>
      <div className="num text-[18px] font-semibold text-ink">{value}</div>
      <div className="mt-1 text-[15px] leading-snug text-ink-2">{note}</div>
    </div>
  );
}

