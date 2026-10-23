/**
 * TrainChart: the line chart of this chapter (dataviz rules: 2px lines, ≥ 8px markers with a 2px
 * surface ring, hairline solid grid, one y-axis, a crosshair that snaps to the nearest x with one
 * tooltip, a legend for ≥ 2 series, a table-view twin). It adds what the shared LineChart lacks and
 * this chapter needs:
 *   - a controlled hover x, so several charts share one crosshair (the training-run scrubber);
 *   - a log y-axis (the learning rate decays to 4·10⁻⁹);
 *   - dashed series (the paper's Fig. 3 draws untrained runs dashed), thin "raw" series under a
 *     smoothed one, eval points drawn as markers;
 *   - `clipX`: draw every series only up to an x (the checkpoint-selection replay);
 *   - highlighted points (the selected checkpoint) and keyboard scrubbing (← → when focused).
 * All text uses SVG_TYPE sizes (never below 13px here) and ink tokens, never the series color.
 */
import { scaleLinear, scaleLog } from "d3";
import { useMemo, useState, type ReactNode } from "react";
import {
  ChartFrame,
  DataTable,
  maxTextWidth,
  SVG_TYPE as T,
  textWidth,
} from "@/components";
import { useElementSize } from "@/components/viz/useSize";
import { catVar } from "@/styles/palette";

export interface TPoint {
  x: number;
  y: number;
}

export interface TSeries {
  id: string;
  label: string;
  slot: number;
  points: TPoint[];
  dashed?: boolean;
  /** Thin, translucent (e.g. the raw per-step loss under its moving average). */
  raw?: boolean;
  /** Draw a marker at every point (evaluations). */
  dots?: boolean;
  /** Leave out of legend and tooltip. */
  hidden?: boolean;
}

export interface THighlight {
  x: number;
  y: number;
  slot: number;
  label?: string;
}

export interface TrainChartProps {
  title: string;
  subtitle?: ReactNode;
  showTitle?: boolean;
  series: TSeries[];
  xLabel?: string;
  yLabel?: string;
  xScale?: "linear" | "log";
  yScale?: "linear" | "log";
  xDomain?: [number, number];
  yDomain?: [number, number];
  /** Map a data x to the plotted x (e.g. step → step + 1 on a log axis). */
  xPlot?: (x: number) => number;
  formatX?: (x: number) => string;
  formatY?: (y: number) => string;
  /** Explicit x ticks (data units). */
  xTicks?: number[];
  yTicks?: number[];
  height?: number;
  markers?: { x: number; label: string }[];
  /** `below`: put the label under the line (right-aligned); `\n` in the label starts a new line. */
  refLines?: { y: number; label: string; below?: boolean }[];
  highlights?: THighlight[];
  /** Only draw points with x ≤ clipX. */
  clipX?: number;
  /** Controlled hover (data x) for synchronized charts. */
  hoverX?: number | null;
  onHoverX?: (x: number | null) => void;
  /** Extra tooltip lines for the hovered x. */
  tooltipExtra?: (x: number) => ReactNode;
  /** Rows of the table-view twin (default: union of x, one column per series). */
  table?: { columns: string[]; rows: (string | number)[][] };
  className?: string;
}

const M0 = { top: 14, right: 18 };
/** Surface halo so annotation text stays readable where a line or gridline crosses it. */
const HALO = {
  paintOrder: "stroke",
  stroke: "var(--surface)",
  strokeWidth: 4,
  strokeLinejoin: "round",
} as const;

/** Greedy word wrap to a pixel width (annotation size); existing `\n` breaks are kept. */
function wrapLabel(label: string, maxW: number): string[] {
  const out: string[] = [];
  for (const para of label.split("\n")) {
    let cur = "";
    for (const w of para.split(" ")) {
      const next = cur ? `${cur} ${w}` : w;
      if (cur && textWidth(next, T.annotation) > maxW) {
        out.push(cur);
        cur = w;
      } else cur = next;
    }
    out.push(cur);
  }
  return out;
}

function nearestPoint(pts: TPoint[], x: number): TPoint | null {
  let best: TPoint | null = null;
  for (const p of pts)
    if (!best || Math.abs(p.x - x) < Math.abs(best.x - x)) best = p;
  return best;
}

export function TrainChart({
  title,
  subtitle,
  showTitle = true,
  series,
  xLabel,
  yLabel,
  xScale = "linear",
  yScale = "linear",
  xDomain,
  yDomain,
  xPlot = (x) => x,
  formatX = (x) => x.toLocaleString("en-US"),
  formatY = (y) => y.toFixed(2),
  xTicks: xTicksIn,
  yTicks: yTicksIn,
  height = 260,
  markers,
  refLines,
  highlights,
  clipX,
  hoverX: hoverXProp,
  onHoverX,
  tooltipExtra,
  table,
  className,
}: TrainChartProps) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const width = Math.max(280, size.width || 640);
  const [hoverLocal, setHoverLocal] = useState<number | null>(null);
  const controlled = hoverXProp !== undefined;
  const hoverX = controlled ? hoverXProp : hoverLocal;
  const setHover = (x: number | null) => {
    if (!controlled) setHoverLocal(x);
    onHoverX?.(x);
  };

  const visible = series.filter((s) => !s.hidden);
  const allPts = useMemo(() => series.flatMap((s) => s.points), [series]);
  const xs = useMemo(
    () => [...new Set(allPts.map((p) => p.x))].sort((a, b) => a - b),
    [allPts],
  );

  const bottom = xLabel ? 52 : 32;
  const plotH = height - (yLabel ? M0.top + 20 : M0.top) - bottom;

  const y = useMemo(() => {
    let dom = yDomain;
    if (!dom) {
      const ys = allPts
        .map((p) => p.y)
        .filter((v) => Number.isFinite(v) && (yScale !== "log" || v > 0));
      for (const r of refLines ?? []) ys.push(r.y);
      const lo = Math.min(...ys);
      const hi = Math.max(...ys);
      if (yScale === "log") dom = [lo, hi];
      else {
        const pad = (hi - lo || 1) * 0.08;
        dom = [lo >= 0 && lo - pad < 0 ? 0 : lo - pad, hi + pad];
      }
    }
    const s =
      yScale === "log"
        ? scaleLog().domain(dom)
        : scaleLinear().domain(dom).nice(5);
    return s.range([plotH, 0]);
  }, [allPts, yDomain, yScale, plotH, refLines]);

  const yTicks = useMemo(() => {
    if (yTicksIn) return yTicksIn;
    if (yScale === "log") {
      const [a, b] = y.domain() as [number, number];
      const out: number[] = [];
      for (
        let e = Math.ceil(Math.log10(a) - 1e-9);
        e <= Math.floor(Math.log10(b) + 1e-9);
        e++
      )
        out.push(10 ** e);
      // keep ≥ 22px between labels
      const step = Math.max(
        1,
        Math.ceil(out.length / Math.max(2, Math.floor(plotH / 26))),
      );
      return out.filter((_, i) => i % step === 0);
    }
    return (y as ReturnType<typeof scaleLinear>).ticks(
      Math.max(3, Math.min(6, Math.floor(plotH / 40))),
    );
  }, [y, yScale, yTicksIn, plotH]);

  const yTickW = maxTextWidth(
    yTicks.map((t) => formatY(t)),
    T.tick,
  );
  const M = {
    ...M0,
    top: yLabel ? M0.top + 20 : M0.top,
    bottom,
    left: Math.ceil(yTickW + 16 + 4),
  };
  const innerW = Math.max(80, width - M.left - M.right);

  const x = useMemo(() => {
    const dom = xDomain ?? [xPlot(xs[0] ?? 0), xPlot(xs[xs.length - 1] ?? 1)];
    const s =
      xScale === "log"
        ? scaleLog().domain([Math.max(dom[0], 1e-9), dom[1]])
        : scaleLinear().domain(dom[0] === dom[1] ? [dom[0], dom[0] + 1] : dom);
    return s.range([0, innerW]);
  }, [xs, xDomain, xScale, innerW, xPlot]);
  const px = (v: number) => x(xPlot(v));

  const xTicks = useMemo(() => {
    if (xTicksIn) {
      // drop ticks that would collide (≥ 18px between labels)
      const out: number[] = [];
      let lastRight = -Infinity;
      for (const t of xTicksIn) {
        const w = textWidth(formatX(t), T.tick);
        const left = px(t) - w / 2;
        if (left >= lastRight + 18) {
          out.push(t);
          lastRight = left + w;
        }
      }
      return out;
    }
    const lin = x as ReturnType<typeof scaleLinear>;
    for (let n = Math.max(2, Math.floor(innerW / 90)); n >= 2; n--) {
      const ts = lin.ticks(n);
      if (
        ts.length *
          (maxTextWidth(
            ts.map((t) => formatX(t)),
            T.tick,
          ) +
            22) <=
        innerW + 22
      )
        return ts;
    }
    return lin.ticks(2);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [x, xTicksIn, innerW]);

  const yv = (v: number) => {
    if (yScale === "log" && v <= 0) return plotH;
    const r = y(v);
    return Math.max(-4, Math.min(plotH + 4, r));
  };

  const path = (pts: TPoint[]) => {
    let d = "";
    let pen = false;
    for (const p of pts) {
      if (clipX !== undefined && p.x > clipX) break;
      if (!Number.isFinite(p.y) || (xScale === "log" && xPlot(p.x) <= 0)) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"}${px(p.x).toFixed(1)},${yv(p.y).toFixed(1)}`;
      pen = true;
    }
    return d;
  };

  const snap = (clientX: number, rect: DOMRect) => {
    const target = clientX - rect.left;
    let best = xs[0];
    let bd = Infinity;
    for (const v of xs) {
      if (clipX !== undefined && v > clipX) break;
      const dd = Math.abs(px(v) - target);
      if (dd < bd) {
        bd = dd;
        best = v;
      }
    }
    return best ?? null;
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight" && e.key !== "Escape")
      return;
    e.preventDefault();
    if (e.key === "Escape") return setHover(null);
    const lim = clipX === undefined ? xs : xs.filter((v) => v <= clipX);
    if (!lim.length) return;
    const i =
      hoverX === null || hoverX === undefined
        ? -1
        : lim.findIndex((v) => v >= hoverX);
    const big = e.shiftKey ? Math.max(1, Math.round(lim.length / 20)) : 1;
    const next =
      e.key === "ArrowRight"
        ? Math.min(lim.length - 1, (i < 0 ? -1 : i) + big)
        : Math.max(0, (i < 0 ? lim.length : i) - big);
    setHover(lim[next]);
  };

  const legend =
    visible.length >= 2 ? (
      <ul
        className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[15px] text-ink-2"
        aria-label="Legend"
      >
        {visible.map((s) => (
          <li key={s.id} className="inline-flex items-center gap-1.5">
            <svg width="22" height="10" aria-hidden="true">
              <line
                x1="1"
                x2="21"
                y1="5"
                y2="5"
                style={{ stroke: catVar(s.slot), opacity: s.raw ? 0.45 : 1 }}
                strokeWidth={s.raw ? 1.5 : 2.5}
                strokeDasharray={s.dashed ? "5 3" : undefined}
                strokeLinecap="round"
              />
            </svg>
            {s.label}
          </li>
        ))}
      </ul>
    ) : undefined;

  const tableTwin = table ?? {
    columns: [xLabel ?? "x", ...visible.map((s) => s.label)],
    rows: xs.map((xv) => [
      formatX(xv),
      ...visible.map((s) => {
        const p = s.points.find((q) => q.x === xv);
        return p ? formatY(p.y) : "—";
      }),
    ]),
  };

  const hx =
    hoverX !== null &&
    hoverX !== undefined &&
    (clipX === undefined || hoverX <= clipX)
      ? hoverX
      : null;
  const tipLeft = hx !== null ? M.left + px(hx) : 0;

  return (
    <ChartFrame
      title={showTitle ? title : undefined}
      subtitle={subtitle}
      legend={legend}
      table={
        <DataTable
          caption={title}
          columns={tableTwin.columns}
          rows={tableTwin.rows}
        />
      }
      className={className}
    >
      <div ref={ref} className="relative w-full">
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`${title}. Focus and use the arrow keys to read values.`}
          tabIndex={0}
          onKeyDown={onKey}
          onBlur={() => setHover(null)}
          className="block overflow-visible rounded-md focus-visible:outline-2 focus-visible:outline-accent"
        >
          <g transform={`translate(${M.left},${M.top})`}>
            {yTicks.map((t) => (
              <g key={t} transform={`translate(0,${y(t)})`}>
                <line
                  x1={0}
                  x2={innerW}
                  style={{ stroke: "var(--grid)" }}
                  strokeWidth={1}
                />
                <text
                  x={-12}
                  dy="0.32em"
                  textAnchor="end"
                  className="tabular"
                  style={{ fill: "var(--muted)", fontSize: T.tick }}
                >
                  {formatY(t)}
                </text>
              </g>
            ))}
            <line
              x1={0}
              x2={innerW}
              y1={plotH}
              y2={plotH}
              style={{ stroke: "var(--axis)" }}
              strokeWidth={1}
            />
            {xTicks.map((t) => (
              <text
                key={t}
                x={px(t)}
                y={plotH + 20}
                textAnchor="middle"
                className="tabular"
                style={{ fill: "var(--muted)", fontSize: T.tick }}
              >
                {formatX(t)}
              </text>
            ))}
            {xLabel && (
              <text
                x={innerW / 2}
                y={plotH + 43}
                textAnchor="middle"
                style={{ fill: "var(--ink-2)", fontSize: T.axisTitle }}
              >
                {xLabel}
              </text>
            )}
            {yLabel && (
              <text
                x={-M.left + 2}
                y={-12}
                textAnchor="start"
                style={{ fill: "var(--ink-2)", fontSize: T.axisTitle }}
              >
                {yLabel}
              </text>
            )}
            {refLines?.map((r) => (
              <line
                key={r.label}
                x1={0}
                x2={innerW}
                y1={y(r.y)}
                y2={y(r.y)}
                style={{ stroke: "var(--ink-2)", opacity: 0.55 }}
                strokeWidth={1}
              />
            ))}
            {markers?.map((m) => {
              const mx = px(m.x);
              const flip =
                mx + 6 + textWidth(m.label, T.annotation) >
                innerW + M.right - 2;
              return (
                <g key={`${m.label}-${m.x}`} transform={`translate(${mx},0)`}>
                  <line
                    y1={0}
                    y2={plotH}
                    style={{ stroke: "var(--axis)" }}
                    strokeWidth={1.5}
                  />
                  <text
                    x={flip ? -6 : 6}
                    y={2}
                    dy="0.71em"
                    textAnchor={flip ? "end" : "start"}
                    style={{
                      fill: "var(--ink-2)",
                      fontSize: T.annotation,
                      ...HALO,
                    }}
                  >
                    {m.label}
                  </text>
                </g>
              );
            })}
            {series.map((s) => {
              const d = path(s.points);
              const stroke = catVar(s.slot);
              return (
                <g key={s.id}>
                  <path
                    d={d}
                    fill="none"
                    style={{ stroke, opacity: s.raw ? 0.32 : 1 }}
                    strokeWidth={s.raw ? 1 : 2}
                    strokeDasharray={s.dashed ? "6 4" : undefined}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                  {s.dots &&
                    s.points
                      .filter(
                        (p) =>
                          Number.isFinite(p.y) &&
                          (clipX === undefined || p.x <= clipX) &&
                          (xScale !== "log" || xPlot(p.x) > 0),
                      )
                      .map((p) => (
                        <circle
                          key={p.x}
                          cx={px(p.x)}
                          cy={yv(p.y)}
                          r={3.2}
                          style={{ fill: stroke, stroke: "var(--surface)" }}
                          strokeWidth={1.5}
                        />
                      ))}
                </g>
              );
            })}
            {/* reference-line labels go above the series, with a surface halo, so a crossing line never hides them */}
            {refLines?.map((r) => (
              <text
                key={r.label}
                x={innerW - 4}
                y={r.below ? y(r.y) + 16 : y(r.y) - 6}
                textAnchor="end"
                style={{
                  fill: "var(--ink-2)",
                  fontSize: T.annotation,
                  ...HALO,
                }}
              >
                {(r.below
                  ? wrapLabel(r.label, innerW * 0.4)
                  : r.label.split("\n")
                ).map((ln, k) => (
                  <tspan key={k} x={innerW - 4} dy={k === 0 ? 0 : "1.2em"}>
                    {ln}
                  </tspan>
                ))}
              </text>
            ))}
            {highlights?.map((h) => {
              const hx0 = px(h.x);
              const hy0 = yv(h.y);
              // label to the right of the ring if it fits, else to the left, else centered above (clamped to the plot)
              const tw = h.label ? textWidth(h.label, T.annotation) * 1.04 : 0;
              let lx = 14;
              let anchor: "start" | "end" = "start";
              if (hx0 + 14 + tw > innerW + M.right - 2) {
                if (hx0 - 14 - tw >= 0) {
                  lx = -14;
                  anchor = "end";
                } else {
                  lx = Math.max(0, Math.min(innerW - tw, hx0 - tw / 2)) - hx0;
                }
              }
              const ly = lx === 14 || lx === -14 ? -12 : hy0 > 30 ? -16 : 24;
              return (
                <g
                  key={`${h.x}-${h.slot}-${h.label ?? ""}`}
                  transform={`translate(${hx0},${hy0})`}
                >
                  <circle
                    r={9}
                    style={{ fill: "none", stroke: "var(--ink)" }}
                    strokeWidth={2}
                  />
                  <circle
                    r={4.5}
                    style={{ fill: catVar(h.slot), stroke: "var(--surface)" }}
                    strokeWidth={2}
                  />
                  {h.label && (
                    <text
                      x={lx}
                      y={ly}
                      textAnchor={anchor}
                      style={{
                        fill: "var(--ink)",
                        fontSize: T.annotation,
                        fontWeight: 600,
                        paintOrder: "stroke",
                        stroke: "var(--surface)",
                        strokeWidth: 4,
                        strokeLinejoin: "round",
                      }}
                    >
                      {h.label}
                    </text>
                  )}
                </g>
              );
            })}
            {hx !== null && (
              <g>
                <line
                  x1={px(hx)}
                  x2={px(hx)}
                  y1={0}
                  y2={plotH}
                  style={{ stroke: "var(--ink-2)" }}
                  strokeWidth={1}
                />
                {visible.map((s) => {
                  const p = nearestPoint(s.points, hx);
                  if (
                    !p ||
                    !Number.isFinite(p.y) ||
                    Math.abs(px(p.x) - px(hx)) > 2
                  )
                    return null;
                  return (
                    <circle
                      key={s.id}
                      cx={px(p.x)}
                      cy={yv(p.y)}
                      r={4.5}
                      style={{ fill: catVar(s.slot), stroke: "var(--surface)" }}
                      strokeWidth={2}
                    />
                  );
                })}
              </g>
            )}
            <rect
              x={0}
              y={0}
              width={innerW}
              height={plotH}
              fill="transparent"
              className="cursor-crosshair"
              style={{ touchAction: "pan-y" }}
              onPointerMove={(e) =>
                setHover(
                  snap(
                    e.clientX,
                    (e.currentTarget as SVGRectElement).getBoundingClientRect(),
                  ),
                )
              }
              onPointerDown={(e) =>
                setHover(
                  snap(
                    e.clientX,
                    (e.currentTarget as SVGRectElement).getBoundingClientRect(),
                  ),
                )
              }
              onPointerLeave={() => setHover(null)}
            />
          </g>
        </svg>
        {hx !== null && (
          <div
            role="tooltip"
            className="pointer-events-none absolute z-20 max-w-[300px] min-w-[150px] rounded-[9px] border border-border bg-surface px-3 py-2 text-[15px] leading-snug shadow-[var(--shadow-pop)]"
            style={{
              left:
                tipLeft > width * 0.6
                  ? undefined
                  : Math.min(width - 160, tipLeft + 14),
              right:
                tipLeft > width * 0.6
                  ? Math.max(0, width - tipLeft + 14)
                  : undefined,
              top: M.top + 2,
            }}
          >
            <div className="mb-1 text-[14px] text-muted">
              {xLabel ?? "x"} {formatX(hx)}
            </div>
            {visible.map((s) => {
              const p = nearestPoint(s.points, hx);
              if (!p) return null;
              return (
                <div key={s.id} className="flex items-center gap-2 py-px">
                  <svg
                    width="14"
                    height="8"
                    aria-hidden="true"
                    className="shrink-0"
                  >
                    <line
                      x1="1"
                      x2="13"
                      y1="4"
                      y2="4"
                      style={{ stroke: catVar(s.slot) }}
                      strokeWidth={2.5}
                      strokeDasharray={s.dashed ? "4 2" : undefined}
                    />
                  </svg>
                  <strong className="num font-semibold text-ink">
                    {formatY(p.y)}
                  </strong>
                  <span className="truncate text-ink-2">{s.label}</span>
                </div>
              );
            })}
            {tooltipExtra?.(hx)}
          </div>
        )}
      </div>
    </ChartFrame>
  );
}

/** A small "label · value" readout cell used under the charts. */
export function Readout({
  label,
  value,
  sub,
}: {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
}) {
  return (
    <div className="min-w-0 rounded-lg bg-surface-2 px-3 py-2">
      <div className="text-[14px] text-ink-2">{label}</div>
      <div className="num text-[18px] font-semibold text-ink">{value}</div>
      {sub && <div className="text-[14px] text-muted">{sub}</div>}
    </div>
  );
}
