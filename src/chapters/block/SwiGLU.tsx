/**
 * Figure 3: the feed-forward sublayer, FFN(h) = W2 · (SiLU(W1·h) ⊙ W3·h) (rt/model.py:50-59, spec §7.1).
 * Part A: the SiLU curve and the gate, with the live pre-activations of the followed token on it.
 * Part B: the token's d_ff hidden units in the selected block, straight from the forward capture.
 */
import { scaleLinear } from 'd3';
import { useMemo, useState } from 'react';
import { Segmented, Slider, VectorStrip, cx, formatSigned, maxAbs } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import type { BlockLive } from './useBlockLive';
import { TokenCaption, TokenPicker } from './TokenPicker';
import { fx, int, norm, rowOf, sigmoid, siluFn } from './utils';

/** argmin of SiLU on [−3, 0] by golden-section search (a property of the function, computed, not quoted). */
function siluMin(): { z: number; v: number } {
  let a = -3;
  let b = 0;
  const g = (Math.sqrt(5) - 1) / 2;
  for (let i = 0; i < 80; i++) {
    const c = b - g * (b - a);
    const d = a + g * (b - a);
    if (siluFn(c) < siluFn(d)) b = d;
    else a = c;
  }
  const z = (a + b) / 2;
  return { z, v: siluFn(z) };
}
const MIN = siluMin();
/** Plotted z range: the live pre-activations of the mini model sit well inside ±4. */
const ZR = 4;

export function SwiGLUFigure({ live, follow, setFollow }: { live: BlockLive; follow: number; setFollow: (p: number) => void }) {
  const { fr, nBlocks, dFF, d } = live;
  const [block, setBlockRaw] = useState(0);
  const b = Math.max(0, Math.min(nBlocks - 1, block));
  const [za, setZa] = useState(1.2);
  const [zb, setZb] = useState(-0.5);
  const [unit, setUnit] = useState<number | null>(null);

  const data = useMemo(() => {
    if (!fr) return null;
    const f = fr.blocks[b].ffn;
    const a = rowOf(f.a, follow, dFF);
    const bb = rowOf(f.b, follow, dFF);
    const gated = rowOf(f.gated, follow, dFF);
    const silu = Float32Array.from(a, siluFn);
    const out = rowOf(f.out, follow, d);
    const h = rowOf(f.h, follow, d);
    let neg = 0;
    let top = 0;
    for (let j = 0; j < dFF; j++) {
      if (a[j] < 0) neg++;
      if (Math.abs(gated[j]) > Math.abs(gated[top])) top = j;
    }
    return { a, b: bb, gated, silu, out, h, neg, top };
  }, [fr, b, follow, dFF, d]);

  const pickUnit = (j: number | null) => {
    setUnit(j);
    if (j !== null && data) {
      setZa(Math.max(-ZR, Math.min(ZR, data.a[j])));
      setZb(Math.max(-3, Math.min(3, data.b[j])));
    }
  };

  return (
    <div className="grid grid-cols-1 gap-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:gap-6">
        <div className="grid shrink-0 grid-cols-1 gap-1">
          <span className="text-[15px] font-medium text-ink-2">Block</span>
          <Segmented label="Block" value={b} onChange={setBlockRaw} options={Array.from({ length: nBlocks }, (_, i) => ({ value: i, label: `${i + 1}` }))} />
        </div>
        <div className="grid min-w-0 flex-1 gap-1">
          <span className="text-[15px] font-medium text-ink-2">Token</span>
          <TokenPicker live={live} value={follow} onChange={setFollow} label="Token for the SwiGLU view" />
        </div>
      </div>
      <TokenCaption live={live} pos={follow} />

      {/* Part A */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <div className="mb-1 text-[16px] font-semibold text-ink">SiLU(z) = z · σ(z), with this token’s {int(dFF)} values of W₁h</div>
          <SiluPlot za={za} setZa={(v) => { setZa(v); setUnit(null); }} dots={data?.a ?? null} unit={unit} onUnit={pickUnit} />
        </div>
        <div className="grid grid-cols-1 min-w-0 content-start gap-3">
          <Slider label="z = (W₁h)ⱼ, the SiLU input" value={za} min={-ZR} max={ZR} step={0.01} onChange={(v) => { setZa(v); setUnit(null); }} format={(v) => formatSigned(v, 2)} />
          <Slider label="g = (W₃h)ⱼ, the gate" value={zb} min={-3} max={3} step={0.01} onChange={(v) => { setZb(v); setUnit(null); }} format={(v) => formatSigned(v, 2)} />
          <GateReadout za={za} zb={zb} unit={unit} />
        </div>
      </div>

      {/* Part B */}
      {data ? (
        <div className="grid grid-cols-1 gap-3 rounded-xl border border-border bg-surface-2/50 p-4">
          <div className="text-[16px] font-semibold text-ink">
            The {int(dFF)} hidden units of this token in block {b + 1}
          </div>
          <VectorStrip values={data.h} label={`h = RMSNorm_ffn(x), the input (${d} dims)`} cellHeight={16} />
          <HiddenStrip label="W₁h, the SiLU inputs" values={data.a} domain={maxAbs([...data.a, ...data.silu])} unit={unit} onUnit={pickUnit} />
          <HiddenStrip label="SiLU(W₁h)" values={data.silu} domain={maxAbs([...data.a, ...data.silu])} unit={unit} onUnit={pickUnit} />
          <HiddenStrip label="W₃h, the gate" values={data.b} unit={unit} onUnit={pickUnit} />
          <HiddenStrip label="SiLU(W₁h) ⊙ W₃h, what W₂ reads" values={data.gated} unit={unit} onUnit={pickUnit} />
          <VectorStrip values={data.out} label={`W₂ · (…), the update added to x (${d} dims)`} cellHeight={16} />
          <ul className="!mb-0 grid gap-1 !pl-5 text-[15.5px] leading-snug text-ink-2" style={{ listStyle: 'disc' }}>
            <li>
              {int(data.neg)} of {int(dFF)} units have W₁h &lt; 0. There SiLU stays between {fx(MIN.v, 3)} and 0, so each of them hands W₂ at most{' '}
              {fx(-MIN.v, 3)} × |gate|.
            </li>
            <li>
              The strongest unit is <button type="button" className="num font-semibold text-accent-text underline decoration-dotted underline-offset-2" onClick={() => pickUnit(data.top)}>#{data.top}</button>: SiLU ={' '}
              {formatSigned(data.silu[data.top], 3)} times gate {formatSigned(data.b[data.top], 3)} = {formatSigned(data.gated[data.top], 3)}.
            </li>
            <li>
              ‖update‖ = {fx(norm(data.out), 3)}. Hover any strip to read one unit; the same unit is outlined in every strip.
            </li>
          </ul>
        </div>
      ) : (
        <div className="text-[16px] text-muted">Loading the trained weights…</div>
      )}
    </div>
  );
}

function HiddenStrip({ label, values, domain, unit, onUnit }: { label: string; values: Float32Array; domain?: number; unit: number | null; onUnit: (j: number | null) => void }) {
  const [hover, setHover] = useState<number | null>(null);
  return (
    <div onClick={() => hover !== null && onUnit(hover)} className="cursor-pointer">
      <VectorStrip
        values={values}
        label={label}
        columns={64}
        cellHeight={14}
        domain={domain}
        highlightIndex={unit}
        onHover={(i) => setHover(i)}
        showNorm={false}
      />
    </div>
  );
}

function GateReadout({ za, zb, unit }: { za: number; zb: number; unit: number | null }) {
  const s = sigmoid(za);
  const silu = za * s;
  const prod = silu * zb;
  const Row = ({ k, v, strong }: { k: string; v: string; strong?: boolean }) => (
    <>
      <dt className="text-ink-2">{k}</dt>
      <dd className={cx('text-right', strong ? 'font-semibold text-ink' : 'text-ink')}>{v}</dd>
    </>
  );
  return (
    <div className="rounded-xl border border-border bg-surface px-3.5 py-3">
      {unit !== null && <div className="mb-1.5 text-[15px] font-medium text-accent-text">hidden unit #{unit} of the live token</div>}
      <dl className="num grid grid-cols-[1fr_auto] gap-x-4 gap-y-1 text-[15.5px]">
        <Row k="σ(z)" v={s.toFixed(3)} />
        <Row k="SiLU(z) = z · σ(z)" v={formatSigned(silu, 3)} />
        <Row k="gate g" v={formatSigned(zb, 3)} />
        <Row k="SiLU(z) · g" v={formatSigned(prod, 3)} strong />
      </dl>
      <p className="!mb-0 mt-2 text-[15px] leading-snug text-ink-2">
        For large z, SiLU(z) ≈ z; for very negative z it fades to 0. The gate g, a second linear map of the same h, scales the result and can flip
        its sign: a product of two learned paths, not one activation alone.
      </p>
    </div>
  );
}

function SiluPlot({
  za,
  setZa,
  dots,
  unit,
  onUnit,
}: {
  za: number;
  setZa: (v: number) => void;
  dots: Float32Array | null;
  unit: number | null;
  onUnit: (j: number | null) => void;
}) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const W = Math.max(280, size.width || 560);
  const H = 300;
  const M = { top: 14, right: 14, bottom: 44, left: 40 };
  const iw = W - M.left - M.right;
  const ih = H - M.top - M.bottom;
  const x = scaleLinear().domain([-ZR, ZR]).range([0, iw]);
  const y = scaleLinear().domain([-0.6, ZR + 0.2]).range([ih, 0]);
  const [hoverDot, setHoverDot] = useState<number | null>(null);
  const [drag, setDrag] = useState(false);

  const curve = useMemo(() => {
    const pts: string[] = [];
    for (let i = 0; i <= 240; i++) {
      const z = -ZR + (2 * ZR * i) / 240;
      pts.push(`${i ? 'L' : 'M'}${x(z).toFixed(2)},${y(siluFn(z)).toFixed(2)}`);
    }
    return pts.join('');
  }, [x, y]);

  const setFromEvent = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const z = x.invert(e.clientX - r.left - M.left);
    setZa(Math.max(-ZR, Math.min(ZR, Math.round(z * 100) / 100)));
  };

  // nearest live dot to the pointer (for hover readout)
  const nearest = (px: number, py: number): number | null => {
    if (!dots) return null;
    let best = -1;
    let bd = 12 * 12;
    for (let j = 0; j < dots.length; j++) {
      const z = dots[j];
      if (z < -ZR || z > ZR) continue;
      const dx = x(z) - px;
      const dy = y(siluFn(z)) - py;
      const dd = dx * dx + dy * dy;
      if (dd < bd) {
        bd = dd;
        best = j;
      }
    }
    return best >= 0 ? best : null;
  };

  const shown = hoverDot ?? unit;

  return (
    <div ref={ref} className="relative min-w-0">
      <svg
        width={W}
        height={H}
        role="img"
        aria-label={`SiLU curve with ${dots?.length ?? 0} live pre-activations; z = ${za.toFixed(2)}`}
        className="block touch-none select-none"
        onPointerDown={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const j = nearest(e.clientX - r.left - M.left, e.clientY - r.top - M.top);
          if (j !== null) {
            onUnit(j);
            return;
          }
          setDrag(true);
          e.currentTarget.setPointerCapture(e.pointerId);
          setFromEvent(e);
        }}
        onPointerMove={(e) => {
          if (drag) {
            setFromEvent(e);
            return;
          }
          const r = e.currentTarget.getBoundingClientRect();
          setHoverDot(nearest(e.clientX - r.left - M.left, e.clientY - r.top - M.top));
        }}
        onPointerUp={() => setDrag(false)}
        onPointerLeave={() => {
          setHoverDot(null);
          setDrag(false);
        }}
      >
        <g transform={`translate(${M.left},${M.top})`}>
          {[0, 1, 2, 3, 4].map((t) => (
            <g key={`y${t}`}>
              <line x1={0} x2={iw} y1={y(t)} y2={y(t)} style={{ stroke: t === 0 ? 'var(--axis)' : 'var(--grid)' }} strokeWidth={1} />
              <text x={-8} y={y(t) + 4.5} textAnchor="end" fontSize={13.5} className="num" style={{ fill: 'var(--muted)' }}>
                {t}
              </text>
            </g>
          ))}
          {[-4, -3, -2, -1, 0, 1, 2, 3, 4].map((t) => (
            <g key={`x${t}`}>
              <line x1={x(t)} x2={x(t)} y1={0} y2={ih} style={{ stroke: t === 0 ? 'var(--axis)' : 'var(--grid)' }} strokeWidth={1} />
              <text x={x(t)} y={ih + 20} textAnchor="middle" fontSize={13.5} className="num" style={{ fill: 'var(--muted)' }}>
                {t}
              </text>
            </g>
          ))}
          <text x={iw / 2} y={ih + 40} textAnchor="middle" fontSize={14.5} style={{ fill: 'var(--muted)' }}>
            z
          </text>
          {/* live dots */}
          {dots &&
            Array.from(dots, (z, j) =>
              z < -ZR || z > ZR ? null : (
                <circle key={j} cx={x(z)} cy={y(siluFn(z))} r={4} style={{ fill: 'var(--seq-400)', stroke: 'var(--surface)', opacity: 0.75 }} strokeWidth={1} />
              ),
            )}
          {dots &&
            Array.from(dots, (z, j) =>
              z < -ZR || z > ZR ? null : <line key={`r${j}`} x1={x(z)} x2={x(z)} y1={ih - 7} y2={ih} style={{ stroke: 'var(--seq-400)', opacity: 0.6 }} strokeWidth={1} />,
            )}
          <path d={curve} fill="none" style={{ stroke: 'var(--cat-7)' }} strokeWidth={2.5} />
          {/* minimum annotation, with a leader line */}
          <line x1={x(MIN.z)} y1={y(MIN.v) - 5} x2={x(-2.9)} y2={y(1.05)} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1} />
          <circle cx={x(MIN.z)} cy={y(MIN.v)} r={3.5} style={{ fill: 'var(--ink-2)' }} />
          <text x={x(-3.9)} y={y(1.05) - 22} fontSize={13.5} style={{ fill: 'var(--ink-2)' }}>
            minimum {fx(MIN.v, 3)}
          </text>
          <text x={x(-3.9)} y={y(1.05) - 5} fontSize={13.5} style={{ fill: 'var(--ink-2)' }}>
            at z = {fx(MIN.z, 3)}
          </text>
          {shown !== null && dots && dots[shown] >= -ZR && dots[shown] <= ZR && (
            <circle cx={x(dots[shown])} cy={y(siluFn(dots[shown]))} r={7} fill="none" style={{ stroke: 'var(--ink)' }} strokeWidth={2} />
          )}
          {/* handle */}
          <line x1={x(za)} x2={x(za)} y1={y(siluFn(za))} y2={ih} style={{ stroke: 'var(--accent)' }} strokeWidth={1.5} strokeDasharray="3 3" />
          <circle cx={x(za)} cy={y(siluFn(za))} r={8} style={{ fill: 'var(--accent)', stroke: 'var(--surface)' }} strokeWidth={2.5} className="cursor-grab" />
        </g>
      </svg>
      {hoverDot !== null && dots && (
        <div className="pointer-events-none absolute top-1 right-1 rounded-lg border border-border bg-surface px-2.5 py-1 text-[14.5px] shadow-[var(--shadow-pop)]">
          <span className="num text-ink">
            unit #{hoverDot}: z = {formatSigned(dots[hoverDot], 2)}, SiLU = {formatSigned(siluFn(dots[hoverDot]), 2)}
          </span>
          <span className="block text-[14px] text-muted">click to load it into the readout</span>
        </div>
      )}
      <p className="!mb-0 mt-1 text-[14.5px] leading-snug text-muted">
        Drag along the curve or use the sliders. Blue dots: this token’s {dots?.length ?? 0} values of (W₁h)ⱼ in the chosen block
        {dots ? `, ${Array.from(dots).filter((z) => z < -ZR || z > ZR).length} outside the plotted range` : ''}.
      </p>
    </div>
  );
}
