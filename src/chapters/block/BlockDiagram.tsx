/**
 * Figure 1: one RT block as an animated diagram (rt/model.py:82-86, spec §7.1).
 *
 *   x → [RMSNorm → MHA_col] ⊕ → [RMSNorm → MHA_feat] ⊕ → [RMSNorm → MHA_nbr] ⊕ → [RMSNorm → MHA_full] ⊕ → [RMSNorm → SwiGLU] ⊕ → x′
 *
 * A StepPlayer walks the residual stream through the five sublayers; every number is the live forward
 * pass of the followed token (default: the masked target). Clicking a box inspects that sublayer.
 */
import { ArrowRight, Activity } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import { useId, useMemo, useState, type ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Button,
  LAYER_INFO,
  LayerLabel,
  Segmented,
  StepPlayer,
  TeX,
  VectorStrip,
  cx,
  maxAbs,
  textWidth,
  useStepPlayer,
} from '@/components';
import { axis, vec } from '../shapes';
import { useElementSize } from '@/components/viz/useSize';
import { PAPER_CONFIG } from '@/engine/model/config';
import type { AttnLayer } from '@/engine/types';
import { TokenCaption, TokenPicker } from './TokenPicker';
import type { BlockLive } from './useBlockLive';
import { allowedKeys, fx, int, norm, streamStages, SUB_NAME, SUBLAYERS, type Stage, type SubKind } from './utils';

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}
interface StationGeom {
  split: { x: number; y: number };
  norm: Box;
  sub: Box;
  plus: { x: number; y: number };
  /** split → norm, norm → sub, sub → ⊕ */
  paths: [string, string, string];
  pulse: { x: number; y: number }[];
}
interface Geom {
  orient: 'h' | 'v';
  W: number;
  H: number;
  streamFrom: { x: number; y: number };
  streamTo: { x: number; y: number };
  inLabel: { x: number; y: number; anchor: 'start' | 'middle' | 'end' };
  outLabel: { x: number; y: number; anchor: 'start' | 'middle' | 'end' };
  stations: StationGeom[];
}

function layout(width: number): Geom {
  const W = Math.max(300, width);
  if (W >= 760) {
    const padL = 44;
    const padR = 52;
    const cw = (W - padL - padR) / 5;
    const bw = Math.min(178, cw * 0.72);
    const subY = 12;
    const subH = 68;
    const normY = subY + subH + 30;
    const normH = 34;
    const streamY = normY + normH + 40;
    const stations: StationGeom[] = [];
    for (let i = 0; i < 5; i++) {
      const left = padL + i * cw + 10;
      const cx = left + bw / 2;
      const plusX = left + bw + 20;
      const subMid = subY + subH / 2;
      stations.push({
        split: { x: cx, y: streamY },
        norm: { x: left, y: normY, w: bw, h: normH },
        sub: { x: left, y: subY, w: bw, h: subH },
        plus: { x: plusX, y: streamY },
        paths: [`M${cx},${streamY - 4}V${normY + normH + 3}`, `M${cx},${normY - 1}V${subY + subH + 3}`, `M${left + bw},${subMid}H${plusX}V${streamY - 12}`],
        pulse: [
          { x: cx, y: streamY },
          { x: cx, y: normY + normH / 2 },
          { x: cx, y: subMid },
          { x: plusX, y: subMid },
          { x: plusX, y: streamY },
        ],
      });
    }
    return {
      orient: 'h',
      W,
      H: streamY + 26,
      streamFrom: { x: 24, y: streamY },
      streamTo: { x: W - 30, y: streamY },
      inLabel: { x: 4, y: streamY + 6, anchor: 'start' },
      outLabel: { x: W - 2, y: streamY + 6, anchor: 'end' },
      stations,
    };
  }
  // vertical (phones): the stream runs top → bottom on the left, branches go right
  const streamX = 20;
  const boxL = 54;
  const boxW = Math.min(W - boxL - 4, 320);
  const sh = 126;
  const y0 = 36;
  const stations: StationGeom[] = [];
  for (let i = 0; i < 5; i++) {
    const ys = y0 + 14 + i * sh;
    const mid = boxL + boxW / 2;
    const subY = ys + 32;
    const subH = 60;
    const plusY = subY + subH / 2;
    stations.push({
      split: { x: streamX, y: ys },
      norm: { x: boxL, y: ys - 16, w: boxW, h: 32 },
      sub: { x: boxL, y: subY, w: boxW, h: subH },
      plus: { x: streamX, y: plusY },
      paths: [`M${streamX + 4},${ys}H${boxL - 3}`, `M${mid},${ys + 16}V${subY - 3}`, `M${boxL},${plusY}H${streamX + 12}`],
      pulse: [
        { x: streamX, y: ys },
        { x: mid, y: ys },
        { x: mid, y: plusY },
        { x: streamX, y: plusY },
      ],
    });
  }
  const H = y0 + 14 + 5 * sh + 6;
  return {
    orient: 'v',
    W,
    H,
    streamFrom: { x: streamX, y: 22 },
    streamTo: { x: streamX, y: H - 26 },
    inLabel: { x: streamX, y: 14, anchor: 'middle' },
    outLabel: { x: streamX, y: H - 4, anchor: 'middle' },
    stations,
  };
}

const STEP_TEXT = (i: number, block: number) => {
  const b = block + 1;
  if (i === 0) return `Block ${b} receives the residual stream x from ${block === 0 ? 'the input embedding' : `block ${block}`}.`;
  if (i === 6) return `Block ${b} hands x′ = x + five updates to ${block === 3 ? 'the final RMSNorm' : `block ${b + 1}`}. Nothing else changes the stream.`;
  const s = SUBLAYERS[i - 1];
  if (s === 'ffn') return 'RMSNorm_ffn rescales a copy of x, SwiGLU transforms each token on its own, and the result is added back to x.';
  return `RMSNorm_${s} rescales a copy of x, ${SUB_NAME[s].toLowerCase()} attention reads it under the ${SUB_NAME[s].toLowerCase()} mask, and the result is added back to x.`;
};

export function BlockDiagram({ live, follow, setFollow }: { live: BlockLive; follow: number; setFollow: (p: number) => void }) {
  const { fr, nBlocks } = live;
  const [block, setBlockLocal] = useBlockState(nBlocks);
  const player = useStepPlayer(7, { intervalMs: 1500 });
  const [ref, size] = useElementSize<HTMLDivElement>();
  const g = useMemo(() => layout(size.width || 900), [size.width]);
  const stages = useMemo(() => (fr ? streamStages(fr, follow) : null), [fr, follow]);
  const active = player.index - 1; // −1 input, 0..4 sublayers, 5 output

  return (
    <div className="grid grid-cols-1 gap-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:gap-6">
        <div className="grid shrink-0 grid-cols-1 gap-1">
          <span className="text-[15px] font-medium text-ink-2">Block</span>
          <Segmented
            label="Block"
            value={block}
            onChange={setBlockLocal}
            options={Array.from({ length: nBlocks }, (_, b) => ({ value: b, label: `${b + 1}` }))}
          />
        </div>
        <div className="grid min-w-0 flex-1 gap-1">
          <span className="text-[15px] font-medium text-ink-2">Follow the stream of</span>
          <TokenPicker live={live} value={follow} onChange={setFollow} label="Followed token" />
        </div>
      </div>
      <TokenCaption live={live} pos={follow} prefix="Following" />
      <div ref={ref} className="min-w-0">
        <DiagramSvg g={g} live={live} block={block} stages={stages} active={active} follow={follow} onPick={(i) => { player.setPlaying(false); player.setIndex(i + 1); }} />
      </div>
      <StepPlayer {...player} unit="step" label="Walk through the block" describe={(i) => STEP_TEXT(i, block)} />
      <Inspector live={live} block={block} active={active} stages={stages} follow={follow} />
    </div>
  );
}

/** Local block selection (starts at block 1), clamped to the model's depth. */
function useBlockState(nBlocks: number): [number, (b: number) => void] {
  const [b, setB] = useState(0);
  return [Math.max(0, Math.min(nBlocks - 1, b)), setB];
}

// ─────────────────────────────────────────────────────────────────────────────
// SVG
// ─────────────────────────────────────────────────────────────────────────────

function DiagramSvg({
  g,
  live,
  block,
  stages,
  active,
  follow,
  onPick,
}: {
  g: Geom;
  live: BlockLive;
  block: number;
  stages: Stage[] | null;
  active: number;
  follow: number;
  onPick: (i: number) => void;
}) {
  const reduce = useReducedMotion();
  const uid = useId().replace(/[^a-zA-Z0-9]/g, '');
  const arrow = `blk-arrow-${uid}`;
  const arrowOn = `blk-arrow-on-${uid}`;
  const st = (i: number) => stages?.[1 + block * 5 + i] ?? null;

  // progress point along the stream: the split of the active station (start / end for input / output)
  const progDone = active < 0 ? g.streamFrom : active >= 5 ? g.streamTo : g.stations[active].split;

  const pulseKeys =
    active < 0 ? [g.streamFrom] : active >= 5 ? [g.stations[4].plus, g.streamTo] : g.stations[active].pulse;

  return (
    <svg
      width={g.W}
      height={g.H}
      viewBox={`0 0 ${g.W} ${g.H}`}
      role="group"
      aria-label={`RT block ${block + 1}: five pre-norm residual sublayers in the order column, feature, neighbor, full attention, then SwiGLU`}
      className="block max-w-full overflow-visible"
    >
      <defs>
        <marker id={arrow} viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--muted)' }} />
        </marker>
        <marker id={arrowOn} viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--accent)' }} />
        </marker>
      </defs>

      {/* residual stream: base + progress */}
      <line
        x1={g.streamFrom.x}
        y1={g.streamFrom.y}
        x2={g.streamTo.x}
        y2={g.streamTo.y}
        style={{ stroke: 'var(--muted)', opacity: 0.45 }}
        strokeWidth={5}
        strokeLinecap="round"
      />
      <ArrowHead from={g.streamFrom} to={g.streamTo} />
      <motion.line
        x1={g.streamFrom.x}
        y1={g.streamFrom.y}
        initial={false}
        animate={{ x2: progDone.x, y2: progDone.y }}
        transition={reduce ? { duration: 0 } : { duration: 0.5, ease: 'easeInOut' }}
        style={{ stroke: 'var(--accent)', opacity: 0.55 }}
        strokeWidth={5}
        strokeLinecap="round"
      />
      <text x={g.inLabel.x} y={g.inLabel.y} textAnchor={g.inLabel.anchor} fontSize={16} fontStyle="italic" style={{ fill: 'var(--ink)' }} fontWeight={600}>
        x
      </text>
      <text x={g.outLabel.x} y={g.outLabel.y} textAnchor={g.outLabel.anchor} fontSize={16} fontStyle="italic" style={{ fill: 'var(--ink)' }} fontWeight={600}>
        x′
      </text>

      {g.stations.map((s, i) => {
        const kind = SUBLAYERS[i];
        const on = active === i;
        const done = active > i;
        const stroke = on ? 'var(--accent)' : done ? 'var(--ink-2)' : 'var(--muted)';
        const stage = st(i);
        const nKeys = kind !== 'ffn' ? allowedKeys(live.masks, kind, follow, live.S) : null;
        const zero = nKeys === 0;
        const dTxt = !stage ? '…' : stage.skipped ? 'off' : zero ? 'Δ = 0' : `‖Δ‖ ${fx(stage.updNorm)}`;
        const name = SUB_NAME[kind];
        const Icon = kind === 'ffn' ? Activity : LAYER_INFO[kind as AttnLayer].icon;
        const nameW = textWidth(name, 16, { weight: 600 }) + 22;
        const label = `${name} ${kind === 'ffn' ? 'feed-forward' : 'attention'} sublayer, block ${block + 1}${stage ? `: update length ${fx(stage.updNorm, 3)}` : ''}`;
        const pick = () => onPick(i);
        const keyDown = (e: React.KeyboardEvent) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            pick();
          }
        };
        return (
          <g key={kind}>
            {/* branch */}
            {s.paths.map((d, j) => (
              <path
                key={j}
                d={d}
                fill="none"
                style={{ stroke, transition: 'stroke 300ms' }}
                strokeWidth={on ? 2.5 : 2}
                strokeDasharray={zero && j === 2 ? '4 4' : undefined}
                markerEnd={`url(#${on ? arrowOn : arrow})`}
              />
            ))}
            <circle cx={s.split.x} cy={s.split.y} r={4} style={{ fill: stroke }} />
            {/* RMSNorm box */}
            <g role="button" tabIndex={0} aria-label={`RMSNorm before the ${label}`} onClick={pick} onKeyDown={keyDown} className="cursor-pointer outline-none [&:focus-visible>rect]:stroke-[3px]">
              <rect
                x={s.norm.x}
                y={s.norm.y}
                width={s.norm.w}
                height={s.norm.h}
                rx={8}
                style={{ fill: on ? 'var(--accent-soft)' : 'var(--surface-2)', stroke, transition: 'stroke 300ms, fill 300ms' }}
                strokeWidth={on ? 2 : 1.25}
              />
              <text x={s.norm.x + s.norm.w / 2} y={s.norm.y + s.norm.h / 2 + 5} textAnchor="middle" fontSize={14.5} style={{ fill: 'var(--ink-2)' }}>
                RMSNorm
                <tspan fontSize={13} dy={3}>
                  {kind}
                </tspan>
              </text>
            </g>
            {/* sublayer box */}
            <g role="button" tabIndex={0} aria-label={label} aria-pressed={on} onClick={pick} onKeyDown={keyDown} className="cursor-pointer outline-none [&:focus-visible>rect]:stroke-[3px]">
              <rect
                x={s.sub.x}
                y={s.sub.y}
                width={s.sub.w}
                height={s.sub.h}
                rx={10}
                style={{ fill: on ? 'var(--accent-soft)' : 'var(--surface)', stroke, transition: 'stroke 300ms, fill 300ms' }}
                strokeWidth={on ? 2.25 : 1.5}
              />
              <Icon
                x={s.sub.x + s.sub.w / 2 - nameW / 2}
                y={s.sub.y + 13}
                width={16}
                height={16}
                strokeWidth={2}
                style={{ color: 'var(--ink-2)' }}
                aria-hidden="true"
              />
              <text x={s.sub.x + s.sub.w / 2 - nameW / 2 + 22} y={s.sub.y + 26} fontSize={16} fontWeight={600} style={{ fill: 'var(--ink)' }}>
                {name}
              </text>
              <text x={s.sub.x + s.sub.w / 2} y={s.sub.y + 51} textAnchor="middle" fontSize={14.5} className="num" style={{ fill: zero ? 'var(--muted)' : 'var(--ink-2)' }}>
                {dTxt}
              </text>
            </g>
            {/* ⊕ */}
            <circle cx={s.plus.x} cy={s.plus.y} r={10} style={{ fill: 'var(--surface)', stroke, transition: 'stroke 300ms' }} strokeWidth={2} />
            <path d={`M${s.plus.x - 5},${s.plus.y}H${s.plus.x + 5}M${s.plus.x},${s.plus.y - 5}V${s.plus.y + 5}`} style={{ stroke }} strokeWidth={2} />
          </g>
        );
      })}

      {/* pulse */}
      <motion.circle
        key={`${active}-${block}-${g.orient}-${Math.round(g.W)}`}
        r={6.5}
        style={{ fill: 'var(--accent)', stroke: 'var(--surface)' }}
        strokeWidth={2}
        initial={{ cx: pulseKeys[0].x, cy: pulseKeys[0].y }}
        animate={
          reduce
            ? { cx: pulseKeys[pulseKeys.length - 1].x, cy: pulseKeys[pulseKeys.length - 1].y }
            : { cx: pulseKeys.map((p) => p.x), cy: pulseKeys.map((p) => p.y) }
        }
        transition={reduce ? { duration: 0 } : { duration: pulseKeys.length > 1 ? 1.1 : 0, ease: 'easeInOut' }}
      />
    </svg>
  );
}

/** A solid arrowhead at the end of the residual stream (a marker would scale with the 5px stroke). */
function ArrowHead({ from, to }: { from: { x: number; y: number }; to: { x: number; y: number } }) {
  const ang = Math.atan2(to.y - from.y, to.x - from.x);
  const L = 13;
  const Wd = 8;
  const bx = to.x - L * Math.cos(ang);
  const by = to.y - L * Math.sin(ang);
  const px = -Math.sin(ang) * Wd;
  const py = Math.cos(ang) * Wd;
  return <path d={`M${to.x},${to.y}L${bx + px},${by + py}L${bx - px},${by - py}Z`} style={{ fill: 'var(--muted)', opacity: 0.7 }} />;
}

// ─────────────────────────────────────────────────────────────────────────────
// Inspector
// ─────────────────────────────────────────────────────────────────────────────

const LAYER_TEX: Record<AttnLayer, string> = { col: 'column', feat: 'feature', nbr: 'neighbor', full: 'full' };

function Inspector({ live, block, active, stages, follow }: { live: BlockLive; block: number; active: number; stages: Stage[] | null; follow: number }) {
  const navigate = useNavigate();
  const { d, dFF, S, ctx } = live;
  if (!stages) return <div className="text-[16px] text-muted">Loading the trained weights…</div>;
  const inStage = stages[block * 5];
  const outStage = stages[block * 5 + 5];

  if (active < 0 || active > 4) {
    const v = active < 0 ? inStage : outStage;
    const which = active < 0 ? 'enters' : 'leaves';
    const diff = new Float32Array(d);
    for (let j = 0; j < d; j++) diff[j] = outStage.x[j] - inStage.x[j];
    const m = maxAbs([...inStage.x, ...outStage.x]);
    return (
      <div className="grid grid-cols-1 gap-4 rounded-xl border border-border bg-surface-2/50 p-4 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <div className="min-w-0">
          <div className="text-[17px] font-semibold text-ink">
            The residual stream {which} block {block + 1}
          </div>
          <p className="!mb-2 mt-1.5 text-[16px] leading-snug text-ink-2">
            {active < 0 ? (
              <>
                This is {block === 0 ? 'the input embedding x⁽⁰⁾ (chapter 8’s x₀)' : `the output of block ${block}`}. The block never rescales it:
                each sublayer reads a normalized copy and adds its update back.
              </>
            ) : (
              <>
                The output is the input plus five updates, x′ = x + Δ<sub>col</sub> + Δ<sub>feat</sub> + Δ<sub>nbr</sub> + Δ<sub>full</sub> + Δ
                <sub>ffn</sub>, each computed from the stream as it stood before that sublayer.
              </>
            )}
          </p>
          <dl className="num grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[15.5px]">
            <dt className="text-ink-2">‖x‖ entering</dt>
            <dd className="text-ink">{fx(inStage.xNorm, 3)}</dd>
            <dt className="text-ink-2">‖x′‖ leaving</dt>
            <dd className="text-ink">{fx(outStage.xNorm, 3)}</dd>
            <dt className="text-ink-2">‖x′ − x‖</dt>
            <dd className="text-ink">{fx(norm(diff), 3)}</dd>
          </dl>
        </div>
        <div className="grid grid-cols-1 min-w-0 gap-3">
          <VectorStrip
            values={v.x}
            label={active < 0 ? `x entering block ${block + 1} (${d} dims)` : `x′ leaving block ${block + 1} (${d} dims)`}
            domain={m}
            cellHeight={18}
            shape={vec(active < 0 ? `the followed token’s x entering block ${block + 1}` : `the followed token’s x′ leaving block ${block + 1}`, axis.model(d))}
          />
          <VectorStrip
            values={diff}
            label="x′ − x: everything this block added"
            domain={m}
            cellHeight={18}
            shape={vec(`x′ − x: everything block ${block + 1} added to the followed token`, axis.model(d))}
          />
        </div>
      </div>
    );
  }

  const kind: SubKind = SUBLAYERS[active];
  const stage = stages[1 + block * 5 + active];
  const prev = stages[block * 5 + active];
  const isAttn = kind !== 'ffn';
  const nKeys = isAttn ? allowedKeys(live.masks, kind, follow, S) : 0;
  const m = maxAbs([...prev.x, ...stage.x, ...(stage.update ?? [])]) || 1;
  const params = isAttn ? 4 * d * d + d : 3 * d * dFF + d;
  const P = PAPER_CONFIG;
  const paperParams = isAttn ? 4 * P.dModel * P.dModel + P.dModel : 3 * P.dModel * P.dFF + P.dModel;
  const tex = isAttn
    ? String.raw`x \;\leftarrow\; x + \mathrm{MHA}_{\text{${kind}}}\!\big(\mathrm{RMSNorm}_{\text{${kind}}}(x);\ \mathbf{M}^{\text{${LAYER_TEX[kind]}}}\big)`
    : String.raw`x \;\leftarrow\; x + \mathrm{SwiGLU}\!\big(\mathrm{RMSNorm}_{\text{ffn}}(x)\big)`;

  return (
    <div className="grid grid-cols-1 gap-4 rounded-xl border border-border bg-surface-2/50 p-4 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-x-2 text-[17px] font-semibold text-ink">
          <span>Block {block + 1} ·</span>
          {isAttn ? <LayerLabel layer={kind} size={16} /> : <span className="inline-flex items-center gap-1.5"><Activity aria-hidden="true" size={16} className="text-ink-2" />SwiGLU</span>}
          <span>{isAttn ? 'attention' : 'feed-forward'} sublayer</span>
        </div>
        <div className="scroll-quiet mt-2 overflow-x-auto py-1 text-[17px]">
          <TeX tex={tex} />
        </div>
        <dl className="num mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[15.5px]">
          <dt className="text-ink-2">‖x‖ entering</dt>
          <dd className="text-ink">{fx(prev.xNorm, 3)}</dd>
          <dt className="text-ink-2">‖RMSNorm(x)‖, what it reads</dt>
          <dd className="text-ink">{stage.skipped ? 'skipped' : fx(stage.hNorm, 3)}</dd>
          {isAttn ? (
            <>
              <dt className="text-ink-2">keys this token may read</dt>
              <dd className="text-ink">
                {int(nKeys)} of {int(ctx.numReal)}
              </dd>
            </>
          ) : (
            <>
              <dt className="text-ink-2">hidden units</dt>
              <dd className="text-ink">
                {d} → {dFF} → {d}
              </dd>
            </>
          )}
          <dt className="text-ink-2">‖Δ‖ added to x</dt>
          <dd className="font-semibold text-ink">{stage.skipped ? 'off (ablation)' : fx(stage.updNorm, 3)}</dd>
          <dt className="text-ink-2">‖x‖ leaving</dt>
          <dd className="text-ink">{fx(stage.xNorm, 3)}</dd>
          <dt className="text-ink-2">parameters</dt>
          <dd className="text-ink">
            {int(params)} <span className="text-muted">(paper {int(paperParams)})</span>
          </dd>
        </dl>
        {isAttn && nKeys === 0 && (
          <p className="!mb-0 mt-3 rounded-lg border border-border bg-surface px-3 py-2 text-[15.5px] leading-snug text-ink-2">
            This token’s neighbor row is empty: no row in the context points to its row. With no key to read, the attention output is 0, and
            W<sub>o</sub> has no bias, so the update is the zero vector and the stream passes through unchanged.
          </p>
        )}
        <div className="mt-3">
          {isAttn ? (
            <Button
              size="sm"
              className="!h-auto min-h-8 py-1 !whitespace-normal text-left"
              iconRight={<ArrowRight aria-hidden="true" size={15} className="shrink-0" />}
              onClick={() => {
                live.setBlock(block);
                live.setLayer(kind);
                navigate('/attention?section=steps');
              }}
            >
              See inside this sublayer (chapter 10)
            </Button>
          ) : (
            <Button size="sm" iconRight={<ArrowRight aria-hidden="true" size={15} />} onClick={() => navigate('/block?section=swiglu')}>
              SwiGLU in detail, below
            </Button>
          )}
        </div>
      </div>
      <div className="grid grid-cols-1 min-w-0 content-start gap-3">
        <VectorStrip values={prev.x} label="x entering" domain={m} cellHeight={18} shape={vec('the followed token’s x entering this sublayer', axis.model(prev.x.length))} />
        {stage.update && (
          <VectorStrip
            values={stage.update}
            label="Δ, the update the sublayer adds"
            domain={m}
            cellHeight={18}
            shape={vec('Δ, the update this sublayer adds to the followed token', axis.model(stage.update.length))}
          />
        )}
        <VectorStrip values={stage.x} label="x + Δ, leaving" domain={m} cellHeight={18} shape={vec('x + Δ, the followed token’s vector leaving this sublayer', axis.model(stage.x.length))} />
        <Note>All three strips share one color scale (±{fx(m)}), so you can compare them cell by cell.</Note>
      </div>
    </div>
  );
}

function Note({ children }: { children: ReactNode }) {
  return <p className={cx('!mb-0 text-[14.5px] leading-snug text-muted')}>{children}</p>;
}
