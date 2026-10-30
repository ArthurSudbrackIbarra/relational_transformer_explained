/**
 * Site mark: three cell tokens in table colors and the dashed "?" target. The "?" is a path, not text, so it
 * sits inside its box in every font; the dash pattern divides the box's perimeter into eight equal periods,
 * one dash centered on each side and each corner. Logo.tsx, certificate/render.ts and scripts/social-images.ts
 * (favicon, link-preview images) draw the mark from this data.
 */
export const LOGO_TARGET = {
  dash: [2.06, 1.618],
  dashOffset: 2.758,
  question: 'M16.2 16.75C16.2 15.95 16.9 15.45 17.75 15.45S19.3 15.95 19.3 16.7C19.3 17.35 18.85 17.6 18.4 17.9C17.95 18.2 17.75 18.4 17.75 18.75',
  dot: { cx: 17.75, cy: 20.3, r: 0.72 },
} as const;

/**
 * The mark as a standalone SVG in the light-theme colors, for files that cannot read CSS variables:
 * public/favicon.svg and the preview images, both written by scripts/social-images.ts.
 */
export function logoMarkSvg(): string {
  const t = LOGO_TARGET;
  return [
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">',
    '<rect x="1.5" y="1.5" width="9.5" height="9.5" rx="2.6" fill="#2a78d6"/>',
    '<rect x="13" y="1.5" width="9.5" height="9.5" rx="2.6" fill="#1baf7a"/>',
    '<rect x="1.5" y="13" width="9.5" height="9.5" rx="2.6" fill="#e87ba4"/>',
    '<g fill="none" stroke="#6250d6" stroke-width="1.3">',
    `<rect x="13.6" y="13.6" width="8.3" height="8.3" rx="2.2" stroke-dasharray="${t.dash.join(' ')}" stroke-dashoffset="${t.dashOffset}"/>`,
    `<path d="${t.question}" stroke-linecap="round" stroke-linejoin="round"/>`,
    '</g>',
    `<circle cx="${t.dot.cx}" cy="${t.dot.cy}" r="${t.dot.r}" fill="#6250d6"/>`,
    '</svg>',
  ].join('');
}
