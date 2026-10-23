/**
 * Horizontal dot plot on a log₁₀ x-axis, for absolute differences (|TS − PyTorch|, |official − port|).
 * One dot per row (single series in the accent color); an optional gray tick per row marks a reference
 * value (one float32 step at that row's magnitude); dashed vertical lines mark thresholds, labeled in a
 * band above the axis so they never collide with the dots.
 * Exact zeros (if any) sit in a separate "0" column on the left as hollow dots.
 * Below 520px the row labels move above their dot lines, so the plot keeps the full width.
 */
import { useMemo, useState } from 'react';
import { TooltipBox } from '@/components/viz/chartParts';
import { useElementSize } from '@/components/viz/useSize';
import { cx, maxTextWidth, SVG_TYPE } from '@/components';
import { sci } from './data';

export interface LogDotRow {
  id: string;
  /** Row label: a code name (stage key, checkpoint file), drawn in monospace unless `plain`. */
  label: string;
  /** The label is plain words, not a code name: draw it in the text font. */
  plain?: boolean;
  value: number;
  /** Optional per-row reference (gray tick). */
  ref?: number;
  /** Extra tooltip lines. */
  extra?: { k: string; v: string }[];
  /** Draw a separator above this row. */
  sepBefore?: boolean;
}

export interface LogDotChartProps {
  rows: LogDotRow[];
  thresholds?: { value: number; label: string }[];
  /** Name of the measured quantity, for the tooltip. */
  valueName: string;
  refName?: string;
  rowHeight?: number;
  ariaLabel: string;
  /** Force a domain [10^lo, 10^hi]. */
  domainExp?: [number, number];
}

const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
const pow10Label = (e: number) => `10${String(e)
  .split('')
  .map((c) => SUP[c] ?? c)
  .join('')}`;

const LABEL_PX = 14;
const NARROW = 520;

export function LogDotChart({ rows, thresholds = [], valueName, refName, rowHeight = 26, ariaLabel, domainExp }: LogDotChartProps) {
  const [wrapRef, { width }] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  const [lo, hi] = useMemo<[number, number]>(() => {
    if (domainExp) return domainExp;
    const vals = [...rows.flatMap((r) => [r.value, r.ref ?? 0]), ...thresholds.map((t) => t.value)].filter((v) => v > 0 && Number.isFinite(v));
    if (!vals.length) return [-9, -3];
    const mn = Math.floor(Math.log10(Math.min(...vals)));
    const mx = Math.ceil(Math.log10(Math.max(...vals)));
    return [Math.min(mn, mx - 2), Math.max(mx, mn + 2)];
  }, [rows, thresholds, domainExp]);

  const narrow = width > 0 && width < NARROW;
  const hasZero = rows.some((r) => r.value === 0);
  const rowH = narrow ? Math.max(rowHeight, 42) : rowHeight;
  const labelW = narrow
    ? 0
    : Math.ceil(
        Math.max(
          maxTextWidth(rows.filter((r) => !r.plain).map((r) => r.label), LABEL_PX, { mono: true }),
          maxTextWidth(rows.filter((r) => r.plain).map((r) => r.label), LABEL_PX),
        ),
      ) + 14;
  const zeroW = hasZero ? 30 : 0;
  const thrBand = thresholds.length ? 22 : 0;
  const top = 30 + thrBand;
  const right = 14;
  const left = narrow ? 6 : 0;
  const H = top + rows.length * rowH + 8;
  const plotX0 = left + labelW + zeroW + (narrow ? 16 : 0);
  const plotW = Math.max(60, width - plotX0 - right);
  const x = (v: number) => plotX0 + ((Math.log10(v) - lo) / (hi - lo)) * plotW;
  const ticks: number[] = [];
  const step = Math.max(1, Math.ceil((hi - lo) / Math.max(2, Math.floor(plotW / 58))));
  for (let e = lo; e <= hi; e += step) ticks.push(e);
  /** y of the dot line of row i */
  const cyOf = (i: number) => top + i * rowH + (narrow ? rowH - 12 : rowH / 2);

  const hv = hover !== null ? rows[hover] : null;

  return (
    <div ref={wrapRef} className="relative min-w-0" onPointerLeave={() => setHover(null)}>
      {width > 0 && (
        <svg width={width} height={H} role="img" aria-label={ariaLabel} className="block overflow-visible">
          {/* thresholds: line through the plot, label in the band above the ticks */}
          {thresholds.map((t) => {
            const tx = x(t.value);
            const toLeft = tx > plotX0 + plotW * 0.6;
            return (
              <g key={t.label}>
                <line x1={tx} x2={tx} y1={4} y2={20} stroke="var(--ink-2)" strokeWidth={1.5} />
                <line x1={tx} x2={tx} y1={top - 6} y2={H - 6} stroke="var(--ink-2)" strokeWidth={1.5} strokeDasharray="5 4" />
                <text x={toLeft ? tx - 6 : tx + 6} y={14} textAnchor={toLeft ? 'end' : 'start'} fontSize={SVG_TYPE.annotation} fill="var(--ink-2)">
                  {t.label}
                </text>
              </g>
            );
          })}
          {/* grid + ticks */}
          {ticks.map((e) => (
            <g key={e}>
              <line x1={x(10 ** e)} x2={x(10 ** e)} y1={top - 6} y2={H - 6} stroke="var(--grid)" strokeWidth={1} />
              <text x={x(10 ** e)} y={top - 12} textAnchor="middle" fontSize={SVG_TYPE.tick} fill="var(--ink-2)" className="tabular">
                {pow10Label(e)}
              </text>
            </g>
          ))}
          {hasZero && (
            <text x={plotX0 - zeroW / 2 - 2} y={top - 12} textAnchor="middle" fontSize={SVG_TYPE.tick} fill="var(--muted)">
              0
            </text>
          )}
          {/* rows */}
          {rows.map((r, i) => {
            const cy = cyOf(i);
            const rowTop = top + i * rowH;
            const isZero = r.value === 0;
            const cx = isZero ? plotX0 - zeroW / 2 - 2 : x(Math.max(r.value, 10 ** lo));
            const active = hover === i;
            return (
              <g
                key={r.id}
                onPointerEnter={() => setHover(i)}
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
                tabIndex={0}
                aria-label={`${r.label}: ${valueName} ${sci(r.value, 2)}`}
                style={{ outline: 'none' }}
              >
                {r.sepBefore && <line x1={0} x2={width - right} y1={rowTop} y2={rowTop} stroke="var(--grid)" strokeWidth={1} />}
                <rect x={0} y={rowTop} width={width} height={rowH} fill={active ? 'var(--surface-2)' : 'transparent'} fillOpacity={0.75} rx={4} />
                <text
                  x={narrow ? left : labelW - 10}
                  y={narrow ? rowTop + 15 : cy}
                  dy={narrow ? undefined : '0.35em'}
                  textAnchor={narrow ? 'start' : 'end'}
                  fontSize={LABEL_PX}
                  fill={active ? 'var(--ink)' : 'var(--ink-2)'}
                  fontFamily={r.plain ? undefined : 'var(--font-mono)'}
                >
                  {r.label}
                </text>
                <line x1={plotX0} x2={plotX0 + plotW} y1={cy} y2={cy} stroke="var(--grid)" strokeWidth={1} strokeDasharray="1 3" />
                {r.ref !== undefined && r.ref > 0 && (
                  <line x1={x(r.ref)} x2={x(r.ref)} y1={cy - 7} y2={cy + 7} stroke="var(--muted)" strokeWidth={2} strokeLinecap="round" />
                )}
                {isZero ? (
                  <circle cx={cx} cy={cy} r={5} fill="var(--surface)" stroke="var(--accent)" strokeWidth={2} />
                ) : (
                  <circle cx={cx} cy={cy} r={active ? 6 : 5} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
                )}
              </g>
            );
          })}
        </svg>
      )}
      {hv && hover !== null && (
        <TooltipBox
          style={{
            left: Math.min(Math.max(8, (hv.value === 0 ? plotX0 : x(Math.max(hv.value, 10 ** lo))) - 120), Math.max(8, width - 290)),
            top: top + hover * rowH + rowH + 4,
          }}
        >
          <div className={cx('text-[15px] font-semibold text-ink', !hv.plain && 'font-mono')}>{hv.label}</div>
          <div className="mt-1 flex justify-between gap-4 text-ink-2">
            <span>{valueName}</span>
            <span className="num text-ink">{sci(hv.value, 2)}</span>
          </div>
          {hv.ref !== undefined && refName && (
            <div className="flex justify-between gap-4 text-ink-2">
              <span>{refName}</span>
              <span className="num text-ink">{sci(hv.ref, 2)}</span>
            </div>
          )}
          {hv.extra?.map((e) => (
            <div key={e.k} className="flex justify-between gap-4 text-ink-2">
              <span>{e.k}</span>
              <span className="num text-ink">{e.v}</span>
            </div>
          ))}
        </TooltipBox>
      )}
    </div>
  );
}
