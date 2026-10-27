/**
 * One RMSNorm applied to every row of an [S × d] matrix (nn.RMSNorm, spec §7.2):
 *   y_ij = x_ij / sqrt(mean_j' x_ij'² + ε) · γ_j
 * The model has one per input type (embedding), five per block, and norm_out. The divisor column
 * is recomputed from x for display; the output panel is the engine's captured tensor (math.test.ts
 * checks that the two agree).
 */
import { useMemo, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { TeX } from '@/components';
import { fmtNum, int } from './format';
import type { E2ELive } from './live';
import { Panel, fitScale, type LinkedHover } from './MatMul';
import type { Cell } from './Matrix';
import { rmsRow, rowOf } from './math';
import { HoverHint, MathHead, Readout, Sym, tok } from './opParts';
import { OpLayout } from './shared';

export interface NormViewProps {
  live: E2ELive;
  x: Float32Array;
  y: Float32Array;
  gamma: Float32Array;
  /** TeX names of the input, the gain and the output. */
  xTex: string;
  gammaTex: string;
  yTex: string;
  width: number;
  hover: LinkedHover | null;
  setHover: (h: LinkedHover | null) => void;
  title: ReactNode;
  /** Prose above the arithmetic (what this norm is for). */
  children?: ReactNode;
  cite: ReactNode;
}

export function NormView({ live, x, y, gamma, xTex, gammaTex, yTex, width, hover, setHover, title, children, cite }: NormViewProps) {
  const { d, S, numReal, focus, cfg, tokenSlots } = live;
  const eps = cfg.eps;
  const tokens = useMemo(() => ({ drawRows: numReal, rowSlots: tokenSlots, tokenRows: true }), [numReal, tokenSlots]);
  const denom = useMemo(() => {
    const out = new Float32Array(S);
    for (let r = 0; r < numReal; r++) {
      let ms = 0;
      for (let j = 0; j < d; j++) ms += x[r * d + j] ** 2;
      out[r] = Math.sqrt(ms / d + eps);
    }
    return out;
  }, [x, S, numReal, d, eps]);
  const gRange = useMemo<[number, number]>(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of gamma) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    return [Math.min(0, lo), hi];
  }, [gamma]);
  const f = rmsRow(rowOf(x, focus, d), gamma, eps);
  // x ÷ rms = h, with γ drawn as a strip right above h so its columns line up with h's
  const c = 3 * fitScale(width, d * 3 * 2 + 16 + 3 * 30 + 4 * 12 + 12);
  const ch = 2.6 * Math.min(1, c / 3);
  const cell = hover && hover.which === 'C' ? { r: hover.r, c: hover.c } : null;
  const onH = (k: Cell | null) => setHover(k ? { which: 'C', r: k.r, c: k.c } : null);
  return (
    <OpLayout
      visual={
        <div className="flex flex-wrap items-end gap-x-3 gap-y-3">
          <Panel o={{ data: x, rows: S, cols: d, ...tokens, name: <TeX tex={xTex} />, label: 'input rows' }} cellW={c} cellH={ch} focusRow={focus} onFocusRow={live.setFocus} hover={cell} onHover={onH} />
          <Sym>÷</Sym>
          <Panel
            o={{ data: denom, rows: S, cols: 1, ...tokens, scale: 'sequential', name: <>rms</>, text: 'rms', label: 'per-row divisor', shape: `${int(numReal)} × 1` }}
            cellW={16}
            cellH={ch}
            focusRow={focus}
            onFocusRow={live.setFocus}
            markRow={cell?.r ?? null}
          />
          <Sym>=</Sym>
          <div className="grid grid-cols-1 gap-1.5">
            <Panel
              o={{ data: gamma, rows: 1, cols: d, scale: 'sequential', domain: gRange, name: <>⊙ <TeX tex={gammaTex} /></>, text: 'γ', label: 'RMSNorm gain, one per column', shape: `1 × ${d}` }}
              cellW={c}
              cellH={14}
              markCol={cell?.c ?? null}
              className="pl-[6px]"
            />
            <Panel o={{ data: y, rows: S, cols: d, ...tokens, name: <TeX tex={yTex} />, label: 'normalized rows' }} cellW={c} cellH={ch} focusRow={focus} onFocusRow={live.setFocus} hover={cell} onHover={onH} />
          </div>
        </div>
      }
      math={
        <MathHead
          title={title}
          tex={[
            `\\mathrm{rms}_i = \\sqrt{\\textstyle\\frac{1}{${d}}\\sum_{k} {${xTex}}_{ik}^{2} + \\varepsilon}`,
            `{${yTex}}_{ij} = {${xTex}}_{ij} \\,/\\, \\mathrm{rms}_i \\cdot {${gammaTex}}_j`,
          ]}
          cite={cite}
        >
          {children}
          <p className="!mt-2 !mb-0">
            Each row is divided by its root mean square, then scaled by the gain γ: one learned value per column, shared by all tokens. Followed token:
            mean of squares {fmtNum(f.meanSq, 4)}, rms √({fmtNum(f.meanSq, 4)} + ε) = {fmtNum(f.denom, 4)}, with ε = 2⁻⁷ = {fmtNum(eps, 4)}.
            {f.meanSq < eps && (
              <>
                {' '}
                Here the mean of squares is smaller than ε, so ε dominates the divisor and the row comes out with a root mean square of{' '}
                {fmtNum(Math.sqrt(f.meanSq) / f.denom, 2)} before γ instead of 1 (<Link to="/embedding?section=rmsnorm">chapter 8</Link>).
              </>
            )}
          </p>
          <Readout>
            {cell ? (
              <>
                [{tok(live, cell.r)}, {cell.c}]: {fmtNum(x[cell.r * d + cell.c], 3, true)} ÷ {fmtNum(denom[cell.r])} × {fmtNum(gamma[cell.c])} ={' '}
                <strong className="num text-ink">{fmtNum(y[cell.r * d + cell.c])}</strong>
              </>
            ) : (
              <HoverHint />
            )}
          </Readout>
        </MathHead>
      }
    />
  );
}
