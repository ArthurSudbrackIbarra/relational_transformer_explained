/**
 * Step 3 figure: RMSNorm in two dimensions. Drag x; y = x / √(mean(x²) + ε) ⊙ γ is computed with the
 * engine's own kernel (`rmsnormRow`, nn.RMSNorm semantics, spec §7.2) at d = 2. With γ = 1 the
 * output lands on the circle of radius √2 (rms = 1) unless ‖x‖ is comparable to √(2ε).
 */
import { RotateCcw } from 'lucide-react';
import { memo, useMemo, useRef, useState, type PointerEvent as RPointerEvent, type KeyboardEvent } from 'react';
import { Button, Figure, Segmented, Slider, TeX } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { rmsnormRow } from '@/engine/model';
import { fmt, fmtSigned } from './live';

const EPS_PAPER = 2 ** -7;
/** float32 machine epsilon: what newer torch uses for bf16 inputs (spec §7.2). */
const EPS_FP32 = 2 ** -23;
const X0: [number, number] = [0.5, 0.2];
/** Half-width of the zoom inset around the origin (in data units): a few times √(2ε) for ε = 2⁻⁷. */
const ZOOM = 0.3;

/**
 * @param presetRms the median rms of the live context's column projections (W_col·c + b_col), for the
 *   "a column projection" preset; omitted while the forward pass loads.
 */
export const RmsNormPlane = memo(function RmsNormPlane({ presetRms }: { presetRms?: number }) {
  const [x, setX] = useState<[number, number]>(X0);
  const [epsKey, setEpsKey] = useState<'paper' | 'fp32'>('paper');
  const [g1, setG1] = useState(1);
  const [g2, setG2] = useState(1);
  const eps = epsKey === 'paper' ? EPS_PAPER : EPS_FP32;
  // the plot always holds the target ellipse (radius √2·γ) and every possible output y
  const LIM = Math.max(2, Math.ceil(1.15 * Math.SQRT2 * Math.max(g1, g2) * 2) / 2);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [boxRef, box] = useElementSize<HTMLDivElement>();
  const dragging = useRef(false);

  const out = useMemo(() => {
    const xin = Float32Array.from(x);
    const y = new Float32Array(2);
    rmsnormRow(xin, 0, 2, Float32Array.of(g1, g2), eps, y, 0);
    const ms = (x[0] * x[0] + x[1] * x[1]) / 2;
    const scale = 1 / Math.sqrt(ms + eps);
    return { y: [y[0], y[1]] as [number, number], ms, scale, epsShare: eps / (ms + eps), rmsOut: Math.sqrt(ms) * scale };
  }, [x, eps, g1, g2]);

  // geometry: data [-LIM, LIM]² → SVG [0, S]²
  const S = Math.round(Math.max(260, Math.min(box.width || 360, 380)));
  const P = 20;
  const sx = (v: number) => P + ((v + LIM) / (2 * LIM)) * (S - 2 * P);
  const sy = (v: number) => P + ((LIM - v) / (2 * LIM)) * (S - 2 * P);
  const unit = (S - 2 * P) / (2 * LIM);

  const toData = (e: RPointerEvent<SVGSVGElement>): [number, number] | null => {
    const svg = svgRef.current;
    if (!svg) return null;
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * S;
    const py = ((e.clientY - r.top) / r.height) * S;
    const clamp = (v: number) => Math.max(-LIM, Math.min(LIM, v));
    const snap = (v: number) => (Math.abs(v) < 0.02 ? 0 : Math.round(v * 100) / 100);
    return [snap(clamp((px - P) / unit - LIM)), snap(clamp(LIM - (py - P) / unit))];
  };

  const onKey = (e: KeyboardEvent<SVGGElement>) => {
    const step = e.shiftKey ? 0.2 : 0.05;
    const m: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
    const dlt = m[e.key];
    if (!dlt) return;
    e.preventDefault();
    setX(([a, b]) => [
      Math.round(Math.max(-LIM, Math.min(LIM, a + dlt[0])) * 100) / 100,
      Math.round(Math.max(-LIM, Math.min(LIM, b + dlt[1])) * 100) / 100,
    ]);
  };

  const r2 = Math.SQRT2;
  const atOrigin = x[0] === 0 && x[1] === 0;
  // the input arrow would hide under the handle when x is this close to the origin
  const tinyX = Math.hypot(x[0], x[1]) * unit < 13;
  const zoneR = Math.sqrt(2 * eps) * unit;
  // x and y nearly on top of each other (γ ≈ 1 and rms(x) ≈ 1): put the y label on the other side
  const labelsClash = Math.hypot(sx(x[0]) - sx(out.y[0]), sy(x[1]) - sy(out.y[1])) < 20;
  const ticks = Array.from({ length: 2 * Math.floor(LIM) + 1 }, (_, i) => i - Math.floor(LIM));

  return (
    <Figure
      wide
      title="RMSNorm in two dimensions: drag the input"
      controls={
        <>
          <div className="flex flex-col gap-1">
            <span className="text-[15px] font-medium text-ink-2">ε</span>
            <Segmented
              label="Epsilon"
              value={epsKey}
              onChange={setEpsKey}
              options={[
                { value: 'paper', label: '2⁻⁷ (paper environment, this site)' },
                { value: 'fp32', label: '2⁻²³ (float32 ε)' },
              ]}
            />
          </div>
          <Slider label="γ₁" value={g1} min={0} max={2} step={0.05} onChange={setG1} format={(v) => v.toFixed(2)} className="w-[150px]" />
          <Slider label="γ₂" value={g2} min={0} max={2} step={0.05} onChange={setG2} format={(v) => v.toFixed(2)} className="w-[150px]" />
          <div className="flex flex-wrap gap-2">
            <Button size="sm" onClick={() => setX([0, 0])} aria-label="Move the input to the origin">
              x = 0
            </Button>
            {presetRms !== undefined && Number.isFinite(presetRms) && (
              <Button
                size="sm"
                onClick={() => {
                  // same mean(x²) as the median column projection of the live context: ‖x‖ = rms·√2 in d = 2
                  const r = presetRms * Math.SQRT2;
                  setX([r * Math.cos(0.38), r * Math.sin(0.38)]);
                }}
                title="Place x so that mean(x²) equals that of a typical column projection W_col·c + b_col of the live context"
              >
                a column projection (rms {fmt(presetRms, 3)})
              </Button>
            )}
            <Button size="sm" variant="ghost" icon={<RotateCcw size={14} aria-hidden="true" />} onClick={() => (setX(X0), setG1(1), setG2(1), setEpsKey('paper'))}>
              Reset
            </Button>
          </div>
        </>
      }
      caption={
        <>
          The output y is computed by the same RMSNorm kernel the engine uses, with d = 2. The dashed curve is where y lands when ε is negligible:
          the circle of radius √2 (rms = 1) stretched by γ. The shaded disk is the ε zone, mean(x²) &lt; ε: inside it ε is most of the sum under the
          square root; the magnified view shows it at a readable size. Drag x into it, use the arrow keys after focusing the handle
          (Shift for bigger steps), or try the column-projection preset. The plane widens when a γ needs the room.
        </>
      }
    >
      <div className="grid items-start gap-6 md:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
        <div ref={boxRef} className="min-w-0">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${S} ${S}`}
          width={S}
          height={S}
          className="block max-w-full touch-none select-none"
          role="img"
          aria-label={`Input x = (${fmt(x[0], 2)}, ${fmt(x[1], 2)}); output y = (${fmt(out.y[0], 3)}, ${fmt(out.y[1], 3)})`}
          onPointerDown={(e) => {
            const p = toData(e);
            if (!p) return;
            dragging.current = true;
            (e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId);
            setX(p);
          }}
          onPointerMove={(e) => {
            if (!dragging.current) return;
            const p = toData(e);
            if (p) setX(p);
          }}
          onPointerUp={() => (dragging.current = false)}
          onPointerCancel={() => (dragging.current = false)}
        >
          <defs>
            <marker id="rms-arrow-in" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink-2)' }} />
            </marker>
            <marker id="rms-arrow-out" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--accent)' }} />
            </marker>
          </defs>
          <rect x={P} y={P} width={S - 2 * P} height={S - 2 * P} rx={6} style={{ fill: 'var(--surface-2)' }} />
          {ticks.map((t) => (
            <g key={t}>
              <line x1={sx(t)} x2={sx(t)} y1={P} y2={S - P} style={{ stroke: t === 0 ? 'var(--axis)' : 'var(--grid)' }} strokeWidth={1} />
              <line x1={P} x2={S - P} y1={sy(t)} y2={sy(t)} style={{ stroke: t === 0 ? 'var(--axis)' : 'var(--grid)' }} strokeWidth={1} />
              {t !== 0 && (
                <>
                  <text x={sx(t)} y={sy(0) + 16} textAnchor="middle" fontSize={13} style={{ fill: 'var(--muted)' }}>
                    {t < 0 ? `−${-t}` : t}
                  </text>
                  <text x={sx(0) - 6} y={sy(t) + 4.5} textAnchor="end" fontSize={13} style={{ fill: 'var(--muted)' }}>
                    {t < 0 ? `−${-t}` : t}
                  </text>
                </>
              )}
            </g>
          ))}
          {/* ε zone: mean(x²) < ε, i.e. ‖x‖ < √(2ε) in d = 2 */}
          {zoneR >= 1 && (
            <g pointerEvents="none">
              <circle cx={sx(0)} cy={sy(0)} r={zoneR} style={{ fill: 'var(--accent)', stroke: 'var(--accent)' }} fillOpacity={0.16} strokeOpacity={0.6} strokeDasharray="2 2" />
              {zoneR >= 8 && (
              <text x={sx(0) - zoneR - 4} y={sy(0) + zoneR + 14} textAnchor="end" fontSize={13} style={{ fill: 'var(--accent-text)' }}>
                ε zone
              </text>
              )}
            </g>
          )}
          {/* target set: rms = 1 circle stretched by γ */}
          <ellipse
            cx={sx(0)}
            cy={sy(0)}
            rx={Math.max(0.5, r2 * g1 * unit)}
            ry={Math.max(0.5, r2 * g2 * unit)}
            fill="none"
            style={{ stroke: 'var(--accent)' }}
            strokeOpacity={0.55}
            strokeWidth={1.5}
            strokeDasharray="5 4"
          />
          {/* the ray through x, and the rescale */}
          {!atOrigin && (
            <line x1={sx(x[0])} y1={sy(x[1])} x2={sx(out.y[0])} y2={sy(out.y[1])} style={{ stroke: 'var(--muted)' }} strokeWidth={1} strokeDasharray="2 3" />
          )}
          {!atOrigin && !tinyX && (
            <line
              x1={sx(0)}
              y1={sy(0)}
              x2={sx(x[0])}
              y2={sy(x[1])}
              style={{ stroke: 'var(--ink-2)' }}
              strokeWidth={2}
              markerEnd="url(#rms-arrow-in)"
            />
          )}
          {(out.y[0] !== 0 || out.y[1] !== 0) && (
            <line
              x1={sx(0)}
              y1={sy(0)}
              x2={sx(out.y[0])}
              y2={sy(out.y[1])}
              style={{ stroke: 'var(--accent)' }}
              strokeWidth={2.5}
              markerEnd="url(#rms-arrow-out)"
            />
          )}
          <circle cx={sx(out.y[0])} cy={sy(out.y[1])} r={5} style={{ fill: 'var(--accent)', stroke: 'var(--surface)' }} strokeWidth={2} />
          {!atOrigin && (
          <text
            x={sx(out.y[0]) + (out.y[0] >= 0 ? 10 : -10)}
            y={sy(out.y[1]) + ((out.y[1] >= 0) !== labelsClash ? -10 : 18)}
            textAnchor={out.y[0] >= 0 ? 'start' : 'end'}
            fontSize={14}
            fontWeight={600}
            style={{ fill: 'var(--accent-text)' }}
          >
            y
          </text>
          )}
          <g
            tabIndex={0}
            role="slider"
            aria-label="Input vector x (arrow keys move it)"
            aria-valuetext={`x = (${fmt(x[0], 2)}, ${fmt(x[1], 2)})`}
            onKeyDown={onKey}
            className="cursor-grab outline-none focus-visible:[&>circle:first-child]:stroke-[var(--accent)]"
          >
            <circle cx={sx(x[0])} cy={sy(x[1])} r={14} fill="transparent" style={{ stroke: 'transparent' }} strokeWidth={2} />
            <circle cx={sx(x[0])} cy={sy(x[1])} r={7.5} style={{ fill: 'var(--ink)', stroke: 'var(--surface)' }} strokeWidth={2} />
          </g>
          <text
            x={sx(x[0]) + (x[0] >= 0 ? 12 : -12)}
            y={sy(x[1]) + (x[1] >= 0 ? -12 : 20)}
            textAnchor={x[0] >= 0 ? 'start' : 'end'}
            fontSize={14}
            fontWeight={600}
            style={{ fill: 'var(--ink)' }}
            pointerEvents="none"
          >
            {atOrigin ? 'x = y = 0' : 'x'}
          </text>
        </svg>
        </div>

        <div className="min-w-0">
          <dl className="num max-w-[460px] text-[16px]">
            <Row k="input x" v={`(${fmtSigned(x[0], 2)}, ${fmtSigned(x[1], 2)})`} />
            <Row k="mean(x²)" v={fmt(out.ms, 4)} />
            <Row k="+ ε" v={epsKey === 'paper' ? fmt(eps, 7) : eps.toExponential(2)} />
            <Row k="scale 1/√(mean(x²)+ε)" v={`×${out.scale >= 100 ? out.scale.toExponential(2) : fmt(out.scale, 3)}`} />
            <Row k="output y" v={`(${fmtSigned(out.y[0], 3)}, ${fmtSigned(out.y[1], 3)})`} strong />
            <Row k="rms before γ" v={fmt(out.rmsOut, 3)} />
            <Row k="ε as a share of mean(x²) + ε" v={atOrigin ? '100%' : `${(out.epsShare * 100).toFixed(out.epsShare < 0.001 ? 3 : 1)}%`} />
          </dl>
          <div className="mt-4 rounded-lg bg-surface-2 px-3.5 py-2.5 text-[16px] leading-snug text-ink-2">
            {atOrigin ? (
              <>
                <TeX tex="\mathrm{RMSNorm}(\vec 0) = \vec 0" /> exactly: 0 divided by √ε is still 0. This is why padding stays zero.
              </>
            ) : out.epsShare > 0.2 ? (
              <>
                Here ε is a large part of the sum under the square root, so the output is <strong className="text-ink">shorter</strong> than the
                circle: RMSNorm can scale a vector up by at most 1/√ε{epsKey === 'paper' ? ` ≈ ${fmt(1 / Math.sqrt(eps), 1)}` : ''}.
              </>
            ) : (
              <>
                The direction of x is kept (up to γ) and its length is reset: whatever the size of x, y has rms ≈ 1 before γ. Try a much longer or
                shorter x.
              </>
            )}
          </div>
          <div className="mt-4">
            <ZoomInset x={x} y={out.y} eps={eps} scale={LIM / ZOOM} />
          </div>
        </div>
      </div>
    </Figure>
  );
});

/**
 * A magnified view of the plane around the origin, so the ε zone and a short input are readable: the
 * disk mean(x²) < ε, the input x and the direction in which RMSNorm sends it (y itself lies far outside).
 */
function ZoomInset({ x, y, eps, scale }: { x: [number, number]; y: [number, number]; eps: number; scale: number }) {
  const S = 172;
  const P = 10;
  const u = (S - 2 * P) / (2 * ZOOM);
  const px = (v: number) => S / 2 + v * u;
  const py = (v: number) => S / 2 - v * u;
  const zr = Math.sqrt(2 * eps) * u;
  const inside = Math.abs(x[0]) <= ZOOM && Math.abs(x[1]) <= ZOOM;
  const yl = Math.hypot(y[0], y[1]);
  const yIn = Math.abs(y[0]) <= ZOOM && Math.abs(y[1]) <= ZOOM;
  // where the ray towards y leaves the inset
  const edge = yl > 0 ? Math.min((ZOOM * 0.92) / Math.max(Math.abs(y[0]), Math.abs(y[1])), 1e9) : 0;
  const ye: [number, number] = yIn ? y : [y[0] * edge, y[1] * edge];
  const zx = (v: number) => (Math.abs(v) < 0.05 ? v.toFixed(2) : v.toFixed(1));
  return (
    <figure className="m-0">
      <figcaption className="mb-1 text-[15px] font-medium text-ink-2">Around the origin, magnified ×{Math.round(scale)}</figcaption>
      <svg width={S} height={S} viewBox={`0 0 ${S} ${S}`} className="block" role="img" aria-label={`Magnified view of the square ±${ZOOM} around the origin`}>
        <rect x={P} y={P} width={S - 2 * P} height={S - 2 * P} rx={6} style={{ fill: 'var(--surface-2)', stroke: 'var(--border)' }} />
        <line x1={P} x2={S - P} y1={S / 2} y2={S / 2} style={{ stroke: 'var(--axis)' }} />
        <line y1={P} y2={S - P} x1={S / 2} x2={S / 2} style={{ stroke: 'var(--axis)' }} />
        <text x={S - P - 4} y={S / 2 + 15} textAnchor="end" fontSize={13} style={{ fill: 'var(--muted)' }}>
          {zx(ZOOM)}
        </text>
        {zr >= 1 && (
          <>
            <circle cx={S / 2} cy={S / 2} r={zr} style={{ fill: 'var(--accent)', stroke: 'var(--accent)' }} fillOpacity={0.16} strokeOpacity={0.6} strokeDasharray="2 2" />
            <text x={S / 2 - zr * 0.72 - 3} y={S / 2 + zr * 0.72 + 14} textAnchor="end" fontSize={13} style={{ fill: 'var(--accent-text)' }}>
              ε zone
            </text>
          </>
        )}
        {yl > 0 && (
          <line x1={S / 2} y1={S / 2} x2={px(ye[0])} y2={py(ye[1])} style={{ stroke: 'var(--accent)' }} strokeWidth={2} markerEnd={yIn ? undefined : 'url(#rms-arrow-out)'} />
        )}
        {inside && (
          <>
            <circle cx={px(x[0])} cy={py(x[1])} r={5} style={{ fill: 'var(--ink)', stroke: 'var(--surface)' }} strokeWidth={1.5} />
            <text x={px(x[0]) + (x[0] >= 0 ? 8 : -8)} y={py(x[1]) + (x[1] >= 0 ? -8 : 16)} textAnchor={x[0] >= 0 ? 'start' : 'end'} fontSize={13} fontWeight={600} style={{ fill: 'var(--ink)' }}>
              x
            </text>
          </>
        )}
      </svg>
      <p className="!mt-1 !mb-0 max-w-[260px] text-[14.5px] leading-snug text-ink-2">
        {inside ? 'The black dot is x.' : 'x lies outside this view.'}{' '}
        {yl === 0 ? 'y = 0.' : yIn ? 'The purple arrow ends at y.' : 'The purple arrow points towards y, which lies outside this view.'}
      </p>
    </figure>
  );
}

function Row({ k, v, strong }: { k: string; v: string; strong?: boolean }) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-x-4 border-b border-grid/60 py-1 last:border-0">
      <dt className="text-ink-2">{k}</dt>
      <dd className={strong ? 'font-semibold text-accent-text' : 'text-ink'}>{v}</dd>
    </div>
  );
}
