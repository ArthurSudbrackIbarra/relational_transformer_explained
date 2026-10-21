/**
 * Colour scales for magnitudes and signed values (dataviz skill
 * references/color-formula.md):
 *  - sequential: ONE hue (blue), light→dark; the anchor flips in dark mode so
 *    "near zero" always recedes toward the surface;
 *  - diverging: two opposite hues (blue ↔ red) with a NEUTRAL gray midpoint,
 *    equal steps per arm.
 * Implemented as 256-entry lookup tables (fast for canvas ImageData).
 */
import { color as d3color, interpolateLab, piecewise } from 'd3';
import { TOKENS, type ThemeMode } from '@/styles/palette';

export type RGB = [number, number, number];

const N = 256;

function toRgb(c: string): RGB {
  const r = d3color(c)?.rgb();
  return r ? [Math.round(r.r), Math.round(r.g), Math.round(r.b)] : [128, 128, 128];
}

function buildLut(interp: (t: number) => string): RGB[] {
  return Array.from({ length: N }, (_, i) => toRgb(interp(i / (N - 1))));
}

const seqCache = new Map<ThemeMode, RGB[]>();
const divCache = new Map<ThemeMode, { neg: RGB[]; pos: RGB[] }>();

/** Sequential LUT for a theme (index 0 = near zero, 255 = max). */
export function sequentialLut(mode: ThemeMode): RGB[] {
  let lut = seqCache.get(mode);
  if (!lut) {
    lut = buildLut(piecewise(interpolateLab, [...TOKENS[mode].seqStops]));
    seqCache.set(mode, lut);
  }
  return lut;
}

function divergingLuts(mode: ThemeMode) {
  let d = divCache.get(mode);
  if (!d) {
    const t = TOKENS[mode];
    d = { neg: buildLut(interpolateLab(t.divMid, t.divNeg)), pos: buildLut(interpolateLab(t.divMid, t.divPos)) };
    divCache.set(mode, d);
  }
  return d;
}

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

/** Sequential colour for t ∈ [0,1]. */
export function sequentialRgb(t: number, mode: ThemeMode): RGB {
  const lut = sequentialLut(mode);
  return lut[Math.round(clamp01(Number.isFinite(t) ? t : 0) * (N - 1))];
}

/** Diverging colour for v ∈ [−maxAbs, +maxAbs] (negative = blue, positive = red, 0 = gray). */
export function divergingRgb(v: number, maxAbs: number, mode: ThemeMode): RGB {
  const { neg, pos } = divergingLuts(mode);
  if (!Number.isFinite(v) || maxAbs <= 0) return neg[0];
  const t = clamp01(Math.abs(v) / maxAbs);
  return (v < 0 ? neg : pos)[Math.round(t * (N - 1))];
}

export const rgbCss = ([r, g, b]: RGB) => `rgb(${r} ${g} ${b})`;

/** CSS colour for t ∈ [0,1] on the sequential scale. */
export function sequentialCss(t: number, mode: ThemeMode): string {
  return rgbCss(sequentialRgb(t, mode));
}

/** CSS colour for a signed value on the diverging scale. */
export function divergingCss(v: number, maxAbs: number, mode: ThemeMode): string {
  return rgbCss(divergingRgb(v, maxAbs, mode));
}

/** Pick white or near-black text for legibility on a fill. */
export function inkOn([r, g, b]: RGB): string {
  const lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return lum > 0.6 ? '#0b0b0b' : '#ffffff';
}

/** Max |v| over an array (ignores non-finite). */
export function maxAbs(values: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < values.length; i++) {
    const a = Math.abs(values[i]);
    if (Number.isFinite(a) && a > m) m = a;
  }
  return m;
}

/** Min/max over an array (ignores non-finite). */
export function extent(values: ArrayLike<number>): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? [lo, hi] : [0, 1];
}
