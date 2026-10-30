/** The site mark (data and geometry in logoMark.ts). */
import { LOGO_TARGET } from './logoMark';

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
