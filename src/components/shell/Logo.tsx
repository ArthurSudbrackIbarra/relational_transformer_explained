/**
 * Site mark: three cell tokens in table colors and the dashed "?" target. The "?" is a path, not text, so it
 * sits inside its box in every font; the dash pattern divides the box's perimeter into eight equal periods,
 * one dash centered on each side and each corner. main.tsx (favicon) and certificate/render.ts draw the same mark.
 */
export const LOGO_TARGET = {
  dash: [2.06, 1.618],
  dashOffset: 2.758,
  question: 'M16.2 16.75C16.2 15.95 16.9 15.45 17.75 15.45S19.3 15.95 19.3 16.7C19.3 17.35 18.85 17.6 18.4 17.9C17.95 18.2 17.75 18.4 17.75 18.75',
  dot: { cx: 17.75, cy: 20.3, r: 0.72 },
} as const;

export function Logo({ size = 22 }: { size?: number }) {
  const t = LOGO_TARGET;
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" className="shrink-0">
      <rect x="1.5" y="1.5" width="9.5" height="9.5" rx="2.6" style={{ fill: 'var(--cat-1)' }} />
      <rect x="13" y="1.5" width="9.5" height="9.5" rx="2.6" style={{ fill: 'var(--cat-3)' }} />
      <rect x="1.5" y="13" width="9.5" height="9.5" rx="2.6" style={{ fill: 'var(--cat-5)' }} />
      <g style={{ stroke: 'var(--accent)' }} fill="none" strokeWidth="1.3">
        <rect x="13.6" y="13.6" width="8.3" height="8.3" rx="2.2" strokeDasharray={t.dash.join(' ')} strokeDashoffset={t.dashOffset} />
        <path d={t.question} strokeLinecap="round" strokeLinejoin="round" />
      </g>
      <circle {...t.dot} style={{ fill: 'var(--accent)' }} />
    </svg>
  );
}
