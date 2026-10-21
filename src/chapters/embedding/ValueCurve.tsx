/**
 * Step 4 figure: one encoder per type. A scalar z enters as RMSNorm_t(z·w_t + b_t) (model.py:216-220):
 * a straight line in ℝ^d bent onto (nearly) the sphere of radius √d, then scaled by γ_t. All values of
 * one type, from every column, lie on this one curve (spec §5.2 teaching note).
 *
 * Every point is computed with the engine's `scalarValueTerm` and the live weights; the 2-D view is a
 * PCA of the sweep z ∈ [−3, 3] (display only). The column's real cells are its stored z-scores from
 * preprocessing (all rows of its table file).
 */
import { scaleLinear } from 'd3';
import { useEffect, useMemo, useState } from 'react';
import { Cite, Figure, LineChart, Segmented, Select, SemBadge, Slider, SVG_TYPE, StatusBadge, VectorStrip, formatDate, maxAbs } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { tensor } from '@/engine/model';
import { SemType } from '@/engine/types';
import type { EmbeddingLive } from './live';
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
  zToRaw,
  type FeatureColumn,
  type ScalarKey,
} from './math';

const SEM_OF: Record<ScalarKey, SemType> = { number: SemType.Number, datetime: SemType.DateTime, boolean: SemType.Boolean };
const Z_MIN = -3;
const Z_MAX = 3;
const N_SWEEP = 121;

function rawText(col: FeatureColumn, z: number): string {
  if (col.sem === SemType.Boolean) return z > 0 ? 'true' : 'false';
  const r = zToRaw(col, z);
  if (r === null) return '—';
  if (col.sem === SemType.DateTime) return formatDate(r, 'never');
  const a = Math.abs(r);
  return a >= 100 ? r.toFixed(0) : a >= 10 ? r.toFixed(1) : r.toFixed(2);
}

export function ValueCurve({ live }: { live: EmbeddingLive }) {
  const { weights, store, ctx, contextColumns, assets } = live;
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

  // ── sweep, PCA basis, limits ──
  const curve = useMemo(() => {
    if (!weights) return null;
    const d = weights.config.dModel;
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
    // ‖normalized vector before γ‖ along the sweep
    const preGain = sweep.map((p, i) => ({ x: zs[i], y: norm(p.pre) / Math.sqrt(meanSquare(p.pre) + weights.config.eps) }));
    return { d, zs, pts: terms.map(proj), proj, basis, limPos: proj(limPos), limNeg: proj(limNeg), preGain, eps: weights.config.eps };
  }, [weights, key]);

  const cur = useMemo(() => (weights ? scalarProjection(weights, key, z) : null), [weights, key, z]);
  const colTerm = useMemo(() => (weights && store && col ? columnProjection(weights, store, col.phrase).term : null), [weights, store, col]);
  const otherTerm = useMemo(() => (weights && store && other ? columnProjection(weights, store, other.phrase).term : null), [weights, store, other]);
  const otherVal = useMemo(() => (weights && other ? scalarProjection(weights, key, z) : null), [weights, other, key, z]);

  const real = useMemo(() => {
    if (!col || !curve || !weights) return [];
    const counts = new Map<number, number>();
    for (const v of col.zs) {
      const r = Math.round(v * 1e6) / 1e6;
      counts.set(r, (counts.get(r) ?? 0) + 1);
    }
    return [...counts].map(([v, n]) => ({ z: v, p: curve.proj(scalarProjection(weights, key, v).term), n }));
  }, [col, curve, weights, key]);

  const extension = useMemo(() => {
    if (!col || !curve || !weights || !col.zs.length) return null;
    const lo = Math.min(...col.zs);
    const hi = Math.max(...col.zs);
    const seg = (a: number, b: number) => {
      const n = 30;
      return Array.from({ length: n + 1 }, (_, i) => curve.proj(scalarProjection(weights, key, a + ((b - a) * i) / n).term));
    };
    return { lo: lo < Z_MIN ? seg(lo, Z_MIN) : null, hi: hi > Z_MAX ? seg(Z_MAX, hi) : null, min: lo, max: hi };
  }, [col, curve, weights, key]);

  const x0 = useMemo(() => {
    if (!cur || !colTerm) return null;
    const out = new Float32Array(cur.term.length);
    for (let j = 0; j < out.length; j++) out[j] = colTerm[j] + cur.term[j];
    return out;
  }, [cur, colTerm]);
  const x0Other = useMemo(() => {
    if (!otherVal || !otherTerm) return null;
    const out = new Float32Array(otherVal.term.length);
    for (let j = 0; j < out.length; j++) out[j] = otherTerm[j] + otherVal.term[j];
    return out;
  }, [otherVal, otherTerm]);

  // ── plot geometry (equal aspect) ──
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const side = Math.round(Math.max(260, Math.min(size.width || 460, 480)));
  const pad = 30;
  const allPts = curve ? [...curve.pts, curve.limPos, curve.limNeg, ...real.map((r) => r.p), ...(extension?.lo ?? []), ...(extension?.hi ?? [])] : [];
  const xs = allPts.map((p) => p[0]);
  const ys = allPts.map((p) => p[1]);
  const cx = xs.length ? (Math.min(...xs) + Math.max(...xs)) / 2 : 0;
  const cy = ys.length ? (Math.min(...ys) + Math.max(...ys)) / 2 : 0;
  const span = xs.length ? Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 0.56 || 1 : 1;
  const sx = scaleLinear().domain([cx - span, cx + span]).range([pad, side - pad]);
  const sy = scaleLinear().domain([cy - span, cy + span]).range([side - pad, pad]);
  const path = (pts: [number, number][]) => pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join('');
  const curP = cur && curve ? curve.proj(cur.term) : null;
  const intLabels = side < 340 ? [-3, 0, 3] : [-3, -2, -1, 0, 1, 2, 3];
  const domain = Math.max(maxAbs(cur?.term ?? []), maxAbs(colTerm ?? []), maxAbs(x0 ?? [])) || 1;

  const typeOptions: { value: ScalarKey; label: React.ReactNode }[] = [
    { value: 'number', label: <span className="inline-flex items-center gap-1.5"><SemBadge sem={SemType.Number} size="xs" bare /> number</span> },
    { value: 'datetime', label: <span className="inline-flex items-center gap-1.5"><SemBadge sem={SemType.DateTime} size="xs" bare /> datetime</span> },
    { value: 'boolean', label: <span className="inline-flex items-center gap-1.5"><SemBadge sem={SemType.Boolean} size="xs" bare /> boolean</span> },
  ];

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
          The curve is RMSNorm_{key}(z·w + b) for z from −3 to 3, computed with the trained weights and drawn in its own two main directions
          (PCA, {curve ? `${Math.round(curve.basis.explained * 100)}%` : '…'} of its variance). Hollow dots: every distinct stored value of the selected
          column; click one to move there. × marks the limits z → ±∞. Every {key} column of every table lands on this same curve.
        </>
      }
    >
      {assets.weightsKind === 'init' && <StatusBadge status="warning">Untrained weights: values from a random initialisation</StatusBadge>}
      <div ref={wrapRef} className="h-0 w-full max-w-[480px]" aria-hidden="true" />
      {!curve || !cur || !col ? (
        <div className="rounded-lg border border-dashed border-border-strong px-4 py-8 text-center text-[16px] text-ink-2">Loading the weights…</div>
      ) : (
        <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,480px)_minmax(0,1fr)]">
          <div className="min-w-0">
            <svg width={side} height={side} viewBox={`0 0 ${side} ${side}`} className="block max-w-full" role="img" aria-label={`The ${key} value curve, current z = ${fmt(z, 2)}`}>
              <rect x={1} y={1} width={side - 2} height={side - 2} rx={8} style={{ fill: 'var(--surface-2)', stroke: 'var(--border)' }} />
              <line x1={sx(cx - span)} x2={sx(cx + span)} y1={sy(0)} y2={sy(0)} style={{ stroke: 'var(--grid)' }} />
              <line x1={sx(0)} x2={sx(0)} y1={sy(cy - span)} y2={sy(cy + span)} style={{ stroke: 'var(--grid)' }} />
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
              {/* integer z ticks */}
              {intLabels.map((k) => {
                const i = Math.round(((k - Z_MIN) / (Z_MAX - Z_MIN)) * (N_SWEEP - 1));
                const p = curve.pts[i];
                return (
                  <g key={k}>
                    <circle cx={sx(p[0])} cy={sy(p[1])} r={3} style={{ fill: 'var(--ink-2)' }} />
                    <text x={sx(p[0]) + 7} y={sy(p[1]) - 6} fontSize={SVG_TYPE.label} style={{ fill: 'var(--ink-2)' }}>
                      {k > 0 ? `+${k}` : k < 0 ? `−${-k}` : '0'}
                    </text>
                  </g>
                );
              })}
              {/* real cells */}
              {real.map((r) => (
                <circle
                  key={r.z}
                  cx={sx(r.p[0])}
                  cy={sy(r.p[1])}
                  r={5}
                  fill="none"
                  style={{ stroke: 'var(--accent)' }}
                  strokeWidth={1.5}
                  className="cursor-pointer"
                  onClick={() => setZ(Math.max(Z_MIN, Math.min(Z_MAX, Math.round(r.z * 100) / 100)))}
                >
                  <title>
                    {col.col} = {rawText(col, r.z)} (z = {fmtSigned(r.z, 3)}), {r.n} {r.n === 1 ? 'cell' : 'cells'}
                  </title>
                </circle>
              ))}
              {/* current */}
              {curP && (
                <g>
                  <circle cx={sx(curP[0])} cy={sy(curP[1])} r={8} style={{ fill: 'var(--accent)', stroke: 'var(--surface)' }} strokeWidth={2.5} />
                </g>
              )}
            </svg>
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[15px] text-ink-2">
              <span>
                {int(real.length)} distinct stored {real.length === 1 ? 'value' : 'values'} in <span className="font-mono">{col.table}</span>
                {extension && (extension.min < Z_MIN || extension.max > Z_MAX) ? `, z from ${fmtSigned(extension.min, 2)} to ${fmtSigned(extension.max, 2)} (dashed beyond ±3)` : ''}
              </span>
            </div>
          </div>

          <div className="min-w-0">
            <div className="grid gap-3">
              <StripRow label={`value term at z = ${fmtSigned(z, 2)}`} values={cur.term} domain={domain} />
              <StripRow label={`column term of “${col.phrase}”`} values={colTerm ?? new Float32Array(0)} domain={domain} />
              <StripRow label="x₀ = column term + value term" values={x0 ?? new Float32Array(0)} domain={domain} strong />
            </div>
            <div className="mt-5">
              <LineChart
                title="Length of the normalised vector before γ"
                subtitle={<span className="text-[15px]">‖(z·w + b) / √(mean(·²) + ε)‖ as z sweeps; the line marks √d = {fmt(Math.sqrt(curve.d), 0)}</span>}
                series={[{ id: 'len', label: 'length before γ', points: curve.preGain }]}
                xLabel="z"
                yLabel="length"
                height={200}
                yDomain={[0, Math.ceil(Math.sqrt(curve.d) + 1)]}
                refLines={[{ y: Math.sqrt(curve.d), label: `√d = ${fmt(Math.sqrt(curve.d), 0)}` }]}
                markers={[{ x: z, label: 'z' }]}
                formatY={(v) => fmt(v, 1)}
                formatX={(v) => fmtSigned(v, 1)}
              />
            </div>
          </div>
        </div>
      )}

      {curve && cur && col && other && otherTerm && x0 && x0Other && otherVal && (
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
          <div className="grid gap-3 md:grid-cols-3">
            <CompareCell
              title="Value term"
              value={maxAbsDelta(cur.term, otherVal.term) === 0 ? 'identical' : `max |Δ| = ${maxAbsDelta(cur.term, otherVal.term).toExponential(1)}`}
              note={
                sem === SemType.DateTime ? (
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
                    z = {fmtSigned(z, 2)} means {col.col} ≈ {rawText(col, z)} but {other.col} ≈ {rawText(other, z)}; the encoder never sees which.
                  </>
                )
              }
            />
            <CompareCell
              title="Column term"
              value={`cosine ${fmt(cosine(colTerm!, otherTerm), 3)}`}
              note={
                <>
                  “{col.phrase}” vs “{other.phrase}”: the only difference between the two tokens.
                </>
              }
            />
            <CompareCell title="x₀" value={`cosine ${fmt(cosine(x0, x0Other), 3)}`} note={<>The two input vectors differ exactly by the difference of their column terms.</>} />
          </div>
        </div>
      )}
    </Figure>
  );
}

function StripRow({ label, values, domain, strong }: { label: string; values: Float32Array; domain: number; strong?: boolean }) {
  return (
    <div className="min-w-0">
      <div className={strong ? 'mb-1 text-[15.5px] font-semibold text-ink' : 'mb-1 text-[15.5px] text-ink-2'}>{label}</div>
      <VectorStrip values={values} columns={32} cellHeight={12} domain={domain} />
    </div>
  );
}

function CompareCell({ title, value, note }: { title: string; value: string; note: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-2.5">
      <div className="text-[15px] text-ink-2">{title}</div>
      <div className="num text-[18px] font-semibold text-ink">{value}</div>
      <div className="mt-1 text-[15px] leading-snug text-ink-2">{note}</div>
    </div>
  );
}
