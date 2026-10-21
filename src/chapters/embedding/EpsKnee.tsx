/**
 * Step 3 figure: where ε matters. Top: the RMSNorm transfer curve rms_out = r / √(r² + ε) (γ = 1) as a
 * function of the input's rms r, for ε = 2⁻⁷ (paper environment, spec §7.2) and for float32 ε. Bottom:
 * the real pre-norm inputs of the live forward pass (ForwardResult.embedding.colPreNorm / valuePreNorm),
 * one dot per distinct input, in one row per input kind. Hover a dot for its numbers, click to inspect it.
 */
import { scaleLinear, scaleLog } from 'd3';
import { useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { Figure, SemBadge, SVG_TYPE, StatusBadge, maxTextWidth } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { SemType } from '@/engine/types';
import type { EmbeddingLive } from './live';
import { fmt, int, phraseOf, rowRef, SEM_LABEL } from './live';
import { meanSquare, norm, slice } from './math';
import { useCrossPick } from './nav';

const EPS_FP32 = 2 ** -23;

type Kind = 'col' | SemType;

interface Pt {
  kind: Kind;
  /** rms of the pre-norm vector */
  r: number;
  ms: number;
  /** ‖term‖ after RMSNorm and γ (real) */
  termLen: number;
  /** representative token position */
  pos: number;
  label: string;
  /** deterministic jitter in [-1, 1] */
  jit: number;
}

const ROWS: { kind: Kind; label: string }[] = [
  { kind: 'col', label: 'column phrases' },
  { kind: SemType.Text, label: 'text values' },
  { kind: SemType.Number, label: 'number values' },
  { kind: SemType.DateTime, label: 'datetime values' },
  { kind: SemType.Boolean, label: 'boolean values' },
];

function median(a: number[]): number {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function EpsKnee({ live }: { live: EmbeddingLive }) {
  const { fr, weights, ctx, pdb, inspected, assets } = live;
  const { pick, chip } = useCrossPick(live);
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const [pointerPt, setPointerPt] = useState<Pt | null>(null);
  /** The dot picked by click, tap or keyboard focus: its tooltip stays until another dot is picked. */
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const dotRefs = useRef(new Map<number, SVGGElement>());

  const pts = useMemo<Pt[] | null>(() => {
    if (!fr || !weights) return null;
    const d = weights.config.dModel;
    const out: Pt[] = [];
    const seenCol = new Set<number>();
    const seenText = new Set<number>();
    const seenScalar = new Set<string>();
    let k = 0;
    const jit = () => {
      k++;
      return Math.sin(k * 12.9898) * 0.5 + Math.sin(k * 4.1414) * 0.5;
    };
    for (const t of ctx.tokens) {
      if (t.isPadding) continue;
      if (!seenCol.has(t.colNameIdx)) {
        seenCol.add(t.colNameIdx);
        const pre = slice(fr.embedding.colPreNorm, t.pos, d);
        const ms = meanSquare(pre);
        out.push({ kind: 'col', r: Math.sqrt(ms), ms, termLen: norm(slice(fr.embedding.colTerm, t.pos, d)), pos: t.pos, label: `“${phraseOf(t)}”`, jit: jit() });
      }
      if (t.mask) continue;
      if (t.sem === SemType.Text) {
        if (seenText.has(t.textIdx)) continue;
        seenText.add(t.textIdx);
      } else {
        const v = t.sem === SemType.Number ? t.numberValue : t.sem === SemType.DateTime ? t.datetimeValue : t.booleanValue;
        const key = `${t.sem}:${v}`;
        if (seenScalar.has(key)) continue;
        seenScalar.add(key);
      }
      const pre = slice(fr.embedding.valuePreNorm, t.pos, d);
      const ms = meanSquare(pre);
      const val = t.sem === SemType.Text ? `“${pdb.text[t.textIdx]}”` : String(t.raw);
      out.push({
        kind: t.sem,
        r: Math.sqrt(ms),
        ms,
        termLen: norm(slice(fr.embedding.valueTerm, t.pos, d)),
        pos: t.pos,
        label: `${t.colName} = ${val.length > 40 ? val.slice(0, 39) + '…”' : val}`,
        jit: jit(),
      });
    }
    // keyboard order: row by row, left to right
    const rowOf = (k: Kind) => ROWS.findIndex((r) => r.kind === k);
    return out.sort((a, b) => rowOf(a.kind) - rowOf(b.kind) || a.r - b.r);
  }, [fr, weights, ctx, pdb]);
  const active = pts && activeIdx !== null ? pts[activeIdx] ?? null : null;
  const hover = pointerPt ?? active;

  const eps = weights?.config.eps ?? 2 ** -7;
  const d = weights?.config.dModel ?? 64;

  const summary = useMemo(() => {
    if (!pts) return null;
    return ROWS.map((row) => {
      const p = pts.filter((q) => q.kind === row.kind);
      const ms = median(p.map((q) => q.ms));
      return { ...row, n: p.length, ms, share: eps / (ms + eps), len: median(p.map((q) => q.termLen)) };
    });
  }, [pts, eps]);

  // ── layout ──
  const W = Math.max(300, size.width || 720);
  const narrow = W < 520;
  const rowLabel = (r: (typeof ROWS)[number]) => (narrow ? r.label.split(' ')[0].replace('column', 'columns') : r.label);
  const labelW = Math.ceil(maxTextWidth(ROWS.map(rowLabel), 14)) + (narrow ? 14 : 34);
  const left = Math.max(labelW, 70);
  const right = 18;
  const topH = 176;
  const rowH = 36;
  const gap = 40;
  const top = 24;
  const H = top + topH + gap + ROWS.length * rowH + 52;
  const x = scaleLog()
    .domain([1e-3, 10])
    .range([left, W - right])
    .clamp(true);
  const yTop = scaleLinear().domain([0, 1.05]).range([top + topH, top]);
  const rowY = (i: number) => top + topH + gap + i * rowH + rowH / 2;

  const curve = (e: number) => {
    const n = 160;
    let dstr = '';
    for (let i = 0; i <= n; i++) {
      const r = 10 ** (-3 + (4 * i) / n);
      const v = r / Math.sqrt(r * r + e);
      dstr += `${i ? 'L' : 'M'}${x(r).toFixed(2)},${yTop(v).toFixed(2)}`;
    }
    return dstr;
  };
  const sqrtEps = Math.sqrt(eps);
  const ticks = [1e-3, 1e-2, 1e-1, 1, 10];
  const inspectedTok = ctx.tokens[inspected];

  const focusDot = (i: number) => {
    if (!pts) return;
    const j = Math.max(0, Math.min(pts.length - 1, i));
    setActiveIdx(j);
    dotRefs.current.get(j)?.focus();
  };
  const onDotKey = (e: KeyboardEvent<SVGGElement>, i: number) => {
    if (!pts) return;
    const k = e.key;
    if (k === 'ArrowRight' || k === 'ArrowDown') focusDot(i + 1);
    else if (k === 'ArrowLeft' || k === 'ArrowUp') focusDot(i - 1);
    else if (k === 'Home') focusDot(0);
    else if (k === 'End') focusDot(pts.length - 1);
    else if (k === 'Enter' || k === ' ') pick(pts[i].pos);
    else if (k === 'Escape') setActiveIdx(null);
    else return;
    e.preventDefault();
  };
  const rovingIdx = activeIdx ?? 0;
  const dotLabel = (p: Pt) =>
    `${p.kind === 'col' ? `column phrase ${p.label}` : `${rowRef(ctx.tokens[p.pos])} ${p.label}`}: mean(x²) ${p.ms.toExponential(2)}, ε is ${((eps / (p.ms + eps)) * 100).toFixed(0)}% of mean(x²) + ε, length after γ ${fmt(p.termLen, 2)}. Enter inspects it in Step 2.`;

  const isInspected = (p: Pt) => {
    if (!inspectedTok || inspectedTok.isPadding) return false;
    if (p.kind === 'col') return ctx.tokens[p.pos].colNameIdx === inspectedTok.colNameIdx;
    return p.pos === inspected;
  };

  return (
    <Figure
      wide
      title="Where ε takes over, with the real inputs of the five RMSNorms"
      caption={
        <>
          Top: the length RMSNorm gives a vector (as rms, before γ) as a function of the length it receives. Bottom: every distinct input of the live
          forward pass, one dot each, placed by its rms. Dots left of √ε come out short. Hover or tap a dot for its numbers; click it (or focus it
          and press Enter) to inspect that token in Step 2.
        </>
      }
    >
      {assets.weightsKind === 'init' && <StatusBadge status="warning">Untrained weights: the dots come from a random initialisation</StatusBadge>}
      <div className="mb-2 flex flex-wrap items-center gap-x-5 gap-y-1 text-[15px] text-ink-2" aria-label="Legend">
        <span className="inline-flex items-center gap-2">
          <svg width="22" height="8" aria-hidden="true">
            <line x1="1" x2="21" y1="4" y2="4" style={{ stroke: 'var(--accent)' }} strokeWidth={2.5} />
          </svg>
          ε = 2⁻⁷ (paper environment, this site)
        </span>
        <span className="inline-flex items-center gap-2">
          <svg width="22" height="8" aria-hidden="true">
            <line x1="1" x2="21" y1="4" y2="4" style={{ stroke: 'var(--muted)' }} strokeWidth={2} strokeDasharray="4 3" />
          </svg>
          ε = 2⁻²³ (float32, newer torch)
        </span>
      </div>
      <div ref={wrapRef} className="relative w-full">
        {!pts ? (
          <div className="rounded-lg border border-dashed border-border-strong px-4 py-8 text-center text-[16px] text-ink-2">Loading the forward pass…</div>
        ) : (
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            className="block max-w-full"
            role="group"
            aria-label="RMSNorm transfer curve and the live inputs of the five input RMSNorms; the dots are buttons, arrow keys move between them"
            onPointerDown={(e) => {
              if (e.target === e.currentTarget) setActiveIdx(null);
            }}
          >
            {/* top panel grid */}
            {[0, 0.5, 1].map((v) => (
              <g key={v}>
                <line x1={left} x2={W - right} y1={yTop(v)} y2={yTop(v)} style={{ stroke: 'var(--grid)' }} />
                <text x={left - 8} y={yTop(v) + 4.5} textAnchor="end" fontSize={SVG_TYPE.tick} style={{ fill: 'var(--muted)' }}>
                  {v}
                </text>
              </g>
            ))}
            <text x={left - 8} y={top - 8} textAnchor="end" fontSize={SVG_TYPE.axisTitle} style={{ fill: 'var(--ink-2)' }}>
              rms out
            </text>
            {/* vertical ticks through both panels */}
            {ticks.map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={top} y2={rowY(ROWS.length - 1) + rowH / 2} style={{ stroke: 'var(--grid)' }} />
                <text x={x(t)} y={rowY(ROWS.length - 1) + rowH / 2 + 18} textAnchor="middle" fontSize={SVG_TYPE.tick} style={{ fill: 'var(--muted)' }}>
                  {t >= 1 ? t : t.toString()}
                </text>
              </g>
            ))}
            <text x={(left + W - right) / 2} y={H - 6} textAnchor="middle" fontSize={SVG_TYPE.axisTitle} style={{ fill: 'var(--ink-2)' }}>
              {narrow ? 'input rms (log scale)' : 'rms of the input, √mean(x²), log scale'}
            </text>
            {/* √ε */}
            <line x1={x(sqrtEps)} x2={x(sqrtEps)} y1={top} y2={rowY(ROWS.length - 1) + rowH / 2} style={{ stroke: 'var(--accent)' }} strokeOpacity={0.5} strokeDasharray="3 3" />
            <text x={x(sqrtEps) + 6} y={yTop(0.12)} fontSize={SVG_TYPE.annotation} style={{ fill: 'var(--accent-text)' }}>
              √ε = {fmt(sqrtEps, 3)}
            </text>
            {/* curves */}
            <path d={curve(EPS_FP32)} fill="none" style={{ stroke: 'var(--muted)' }} strokeWidth={2} strokeDasharray="4 3" />
            <path d={curve(eps)} fill="none" style={{ stroke: 'var(--accent)' }} strokeWidth={2.5} />
            {/* hover guide */}
            {hover && (
              <g pointerEvents="none">
                <line
                  x1={x(hover.r)}
                  x2={x(hover.r)}
                  y1={yTop(hover.r / Math.sqrt(hover.r * hover.r + eps))}
                  y2={rowY(ROWS.findIndex((r) => r.kind === hover.kind))}
                  style={{ stroke: 'var(--ink-2)' }}
                  strokeDasharray="2 3"
                />
                <circle
                  cx={x(hover.r)}
                  cy={yTop(hover.r / Math.sqrt(hover.r * hover.r + eps))}
                  r={5}
                  style={{ fill: 'var(--accent)', stroke: 'var(--surface)' }}
                  strokeWidth={2}
                />
              </g>
            )}
            {/* divider between the curve panel and the dot rows */}
            <line x1={8} x2={W - right} y1={top + topH + 16} y2={top + topH + 16} style={{ stroke: 'var(--border-strong)' }} />
            <text x={left} y={top + topH + 33} fontSize={SVG_TYPE.annotation} style={{ fill: 'var(--ink-2)' }}>
              inputs of this forward pass, one dot per distinct input
            </text>
            {/* rows */}
            {ROWS.map((row, i) => {
              const y = rowY(i);
              const n = pts.filter((p) => p.kind === row.kind).length;
              return (
                <g key={String(row.kind)}>
                  <line x1={left} x2={W - right} y1={y} y2={y} style={{ stroke: 'var(--grid)' }} />
                  <text x={left - 10} y={y + 4.5} textAnchor="end" fontSize={14} style={{ fill: n ? 'var(--ink-2)' : 'var(--muted)' }}>
                    {rowLabel(row)}
                  </text>
                  {n === 0 && (
                    <text x={left + 8} y={y + 4.5} fontSize={SVG_TYPE.annotation} style={{ fill: 'var(--muted)' }}>
                      none in this context
                    </text>
                  )}
                </g>
              );
            })}
            {pts.map((p, idx) => {
              const i = ROWS.findIndex((r) => r.kind === p.kind);
              const cy = rowY(i) + p.jit * (rowH / 2 - 7);
              const on = isInspected(p);
              const act = idx === activeIdx;
              return (
                <g
                  key={idx}
                  ref={(el) => {
                    if (el) dotRefs.current.set(idx, el);
                    else dotRefs.current.delete(idx);
                  }}
                  role="button"
                  tabIndex={idx === rovingIdx ? 0 : -1}
                  aria-label={dotLabel(p)}
                  className="cursor-pointer outline-none"
                  onPointerEnter={(e) => e.pointerType === 'mouse' && setPointerPt(p)}
                  onPointerLeave={() => setPointerPt((h) => (h === p ? null : h))}
                  onClick={() => {
                    setActiveIdx(idx);
                    pick(p.pos);
                  }}
                  onFocus={() => setActiveIdx(idx)}
                  onKeyDown={(e) => onDotKey(e, idx)}
                >
                  {/* transparent hit area for touch */}
                  <circle cx={x(p.r)} cy={cy} r={11} fill="transparent" />
                  {act && <circle cx={x(p.r)} cy={cy} r={9} fill="none" style={{ stroke: 'var(--ink)' }} strokeWidth={1.5} />}
                  <circle
                    cx={x(p.r)}
                    cy={cy}
                    r={on ? 6 : 4.5}
                    style={{ fill: on ? 'var(--accent)' : 'var(--ink-2)', stroke: 'var(--surface)' }}
                    fillOpacity={on ? 1 : 0.55}
                    strokeWidth={1.5}
                  />
                </g>
              );
            })}
          </svg>
        )}
        {hover && (
          <div
            className="pointer-events-none absolute z-10 w-max max-w-[300px] rounded-lg border border-border bg-surface px-3 py-2 text-[14.5px] leading-snug shadow-lg"
            style={{
              // the upper-left of the top panel is always empty (the curve is near 0 there)
              left: Math.min(left + 10, Math.max(4, W - 304)),
              top: top + 22,
            }}
          >
            <div className="font-medium text-ink">{hover.kind === 'col' ? hover.label : `${rowRef(ctx.tokens[hover.pos])}`}</div>
            {hover.kind !== 'col' && <div className="text-ink-2">{hover.label}</div>}
            <div className="num mt-1 text-ink-2">mean(x²) = {hover.ms < 0.01 ? hover.ms.toExponential(2) : fmt(hover.ms, 3)}</div>
            <div className="num text-ink-2">ε is {((eps / (hover.ms + eps)) * 100).toFixed(1)}% of mean(x²) + ε</div>
            <div className="num text-ink-2">‖term‖ after γ = {fmt(hover.termLen, 2)}</div>
          </div>
        )}
      </div>

      {chip}
      {summary && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full border-collapse text-[15px]">
            <thead>
              <tr className="border-b border-grid text-left text-ink-2">
                <th scope="col" className="py-1.5 pr-3 font-medium">
                  Input
                </th>
                <th scope="col" className="hidden py-1.5 pr-3 text-right font-medium sm:table-cell">
                  distinct
                </th>
                <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                  mean(x²)
                </th>
                <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                  <span title="ε / (mean(x²) + ε): the share of ε in the sum under the square root">ε share of sum</span>
                </th>
                <th scope="col" className="py-1.5 text-right font-medium">
                  ‖term‖
                </th>
              </tr>
            </thead>
            <tbody>
              {summary.map((s) => (
                <tr key={String(s.kind)} className="border-b border-grid/60 last:border-0">
                  <th scope="row" className="py-1.5 pr-3 text-left font-normal text-ink">
                    <span className="inline-flex items-center gap-2">
                      {s.kind !== 'col' && <SemBadge sem={s.kind} size="sm" />}
                      {s.kind === 'col' ? 'column phrase' : `${SEM_LABEL[s.kind]} value`}
                    </span>
                  </th>
                  <td className="num hidden py-1.5 pr-3 text-right text-ink-2 sm:table-cell">{int(s.n)}</td>
                  <td className="num py-1.5 pr-3 text-right text-ink">{s.n ? (s.ms < 0.01 ? s.ms.toExponential(2) : fmt(s.ms, 3)) : '—'}</td>
                  <td className="num py-1.5 pr-3 text-right text-ink">{s.n ? `${(s.share * 100).toFixed(s.share < 0.01 ? 1 : 0)}%` : '—'}</td>
                  <td className="num py-1.5 text-right text-ink">{s.n ? fmt(s.len, 2) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="!mt-2 !mb-0 text-[15px] text-ink-2">
            Medians over the distinct inputs of each kind: mean(x²) before the norm, ε as a share of mean(x²) + ε (the sum under the square root),
            and the length of the term after RMSNorm and γ.
          </p>
        </div>
      )}
    </Figure>
  );
}
