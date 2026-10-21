/** Site mark: three cell tokens in table colours and the dashed "?" target. */
export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" className="shrink-0">
      <rect x="1.5" y="1.5" width="9.5" height="9.5" rx="2.6" style={{ fill: 'var(--cat-1)' }} />
      <rect x="13" y="1.5" width="9.5" height="9.5" rx="2.6" style={{ fill: 'var(--cat-3)' }} />
      <rect x="1.5" y="13" width="9.5" height="9.5" rx="2.6" style={{ fill: 'var(--cat-5)' }} />
      <rect
        x="13.6"
        y="13.6"
        width="8.3"
        height="8.3"
        rx="2.2"
        fill="none"
        style={{ stroke: 'var(--accent)' }}
        strokeWidth="1.3"
        strokeDasharray="2 1.6"
      />
      <text x="17.75" y="20.3" textAnchor="middle" style={{ fill: 'var(--accent)', fontSize: 11.5, fontWeight: 800, fontFamily: 'system-ui, sans-serif' }}>
        ?
      </text>
    </svg>
  );
}
