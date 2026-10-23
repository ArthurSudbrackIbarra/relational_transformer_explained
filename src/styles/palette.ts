/**
 * JS mirror of the colour tokens in src/styles/index.css (BUILD_PLAN §5,
 * dataviz skill references/palette.md). Canvas code (Heatmap, MaskMatrix)
 * cannot read `var(--…)`, so it picks the hex for the resolved theme here.
 *
 * Keep in lockstep with index.css. SVG/HTML components should prefer the CSS
 * variables (`catVar(slot)`) so they re-theme without a re-render.
 */

export type ThemeMode = 'light' | 'dark';

/** Categorical slots 1..8 (index 0 unused). Fixed order, never cycled. */
export const CATEGORICAL: Record<ThemeMode, readonly string[]> = {
  light: ['#898781', '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
  dark: ['#898781', '#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
};

export const CATEGORICAL_NAMES = ['neutral', 'blue', 'orange', 'aqua', 'yellow', 'magenta', 'green', 'violet', 'red'] as const;

/** Sequential blue ramp, light→dark (100..700). Same hexes in both modes; the anchor flips in dark. */
export const SEQ_RAMP = [
  '#cde2fb', // 100
  '#b7d3f6', // 150
  '#9ec5f4', // 200
  '#86b6ef', // 250
  '#6da7ec', // 300
  '#5598e7', // 350
  '#3987e5', // 400
  '#2a78d6', // 450
  '#256abf', // 500
  '#1c5cab', // 550
  '#184f95', // 600
  '#104281', // 650
  '#0d366b', // 700
] as const;

export interface ModeTokens {
  page: string;
  surface: string;
  surface2: string;
  ink: string;
  ink2: string;
  muted: string;
  grid: string;
  axis: string;
  accent: string;
  /** Diverging poles + neutral midpoint (blue ↔ red, gray midpoint). */
  divNeg: string;
  divMid: string;
  divPos: string;
  /** Sequential stops, from "near zero" (recedes to the surface) to "max". */
  seqStops: readonly string[];
}

const BASE_TOKENS: Record<ThemeMode, ModeTokens> = {
  light: {
    page: '#f9f9f7',
    surface: '#fcfcfb',
    surface2: '#f3f2ee',
    ink: '#0b0b0b',
    ink2: '#52514e',
    muted: '#898781',
    grid: '#e1e0d9',
    axis: '#c3c2b7',
    accent: '#6250d6',
    divNeg: '#2a78d6',
    divMid: '#f0efec',
    divPos: '#e34948',
    // light: 100 (near zero) → 700 (max)
    seqStops: ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'],
  },
  dark: {
    page: '#0d0d0d',
    surface: '#1a1a19',
    surface2: '#222221',
    ink: '#f0efec',
    ink2: '#c3c2b7',
    muted: '#898781',
    grid: '#2c2c2a',
    axis: '#383835',
    accent: '#9085e9',
    divNeg: '#3987e5',
    divMid: '#383835',
    divPos: '#e66767',
    // dark: anchor flips: 700 (near zero, sits by the dark surface) → 100 (max)
    seqStops: ['#0d366b', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb'],
  },
};

/**
 * Reading mode (<html data-reading="on">, index.css) swaps only the neutrals for paper tones;
 * data colours (categorical, sequential, diverging) keep their meaning.
 */
const READING_NEUTRALS: Record<ThemeMode, Partial<ModeTokens>> = {
  light: { page: '#f2e9d5', surface: '#f8f1e1', surface2: '#ece1c8', ink: '#2a2118', ink2: '#574838', muted: '#7d6c57', grid: '#e0d2b4', axis: '#c2b08d' },
  dark: { page: '#1a1713', surface: '#23201b', surface2: '#2b2721', ink: '#ece2cf', ink2: '#c9bba3', muted: '#968871', grid: '#342e27', axis: '#4a4238' },
};
const READING_TOKENS: Record<ThemeMode, ModeTokens> = {
  light: { ...BASE_TOKENS.light, ...READING_NEUTRALS.light },
  dark: { ...BASE_TOKENS.dark, ...READING_NEUTRALS.dark },
};

function readingOn(): boolean {
  return typeof document !== 'undefined' && document.documentElement.getAttribute('data-reading') === 'on';
}

/** Theme tokens for canvas code; follows reading mode (re-render with `useReadingMode()` from ./theme). */
export const TOKENS: Record<ThemeMode, ModeTokens> = {
  get light() {
    return readingOn() ? READING_TOKENS.light : BASE_TOKENS.light;
  },
  get dark() {
    return readingOn() ? READING_TOKENS.dark : BASE_TOKENS.dark;
  },
};

/** `var(--cat-N)` for a slot 1..8; slot 0 / out of range → muted neutral. */
export function catVar(slot: number | null | undefined): string {
  if (!slot || slot < 1 || slot > 8) return 'var(--muted)';
  return `var(--cat-${slot})`;
}

/** Hex of a categorical slot for a theme (canvas use). */
export function catHex(slot: number | null | undefined, mode: ThemeMode): string {
  const s = slot && slot >= 1 && slot <= 8 ? slot : 0;
  return CATEGORICAL[mode][s];
}

/** BUILD_PLAN §3.5: every task table uses slot 5 (magenta). */
export const TASK_TABLE_SLOT = 5;
