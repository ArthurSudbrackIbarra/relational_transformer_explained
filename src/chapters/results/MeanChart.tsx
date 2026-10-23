/**
 * MeanChart: the printed mean of each method as a lollipop that grows from a meaningful baseline
 * (random AUROC = 50, or the global-mean R² = 0), grouped by setting ("Maybe" / "No" / "Yes").
 * Stems animate in from the baseline the first time the chart scrolls into view.
 */
import { motion, useInView, useReducedMotion } from "motion/react";
import { useRef, useState } from "react";
import { textWidth } from "@/components/viz/text";
import { useElementSize } from "@/components/viz/useSize";
import { catVar } from "@/styles/palette";
import { LT9, type Cell } from "./data";
import { cellText, isNum, MethodMark, Tip, xScale } from "./kit";
import { METHODS } from "./methods";

export interface MeanGroup {
  label: string;
  sub?: string;
  items: { method: string; value: Cell; note?: string }[];
}

export function MeanChart({
  groups,
  domain,
  baseline,
  baselineLabel,
  axisLabel,
  emphasis,
  ariaLabel,
}: {
  groups: MeanGroup[];
  domain: [number, number];
  baseline: number;
  baselineLabel: string;
  axisLabel: string;
  /** Method ids drawn with a bold label. */
  emphasis?: string[];
  ariaLabel: string;
}) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const inViewRef = useRef<SVGSVGElement | null>(null);
  const inView = useInView(inViewRef, { once: true, amount: 0.3 });
  const reduce = useReducedMotion();
  const [hover, setHover] = useState<string | null>(null);
  const W = Math.max(300, size.width || 700);
  const narrow = W < 560;
  const emph = new Set(emphasis ?? []);

  const all = groups.flatMap((g) => g.items);
  const labelW =
    Math.ceil(
      Math.max(
        ...all.map((i) =>
          textWidth(METHODS[i.method]?.label ?? i.method, 15, { weight: 600 }),
        ),
      ),
    ) + 34;
  // narrow screens: the method label sits above its stem, and the stem gets the full width
  const stacked = W - labelW < 330;
  const plotL = stacked ? 16 : labelW + 8;
  const plotR = W - 70;
  const rowH = stacked ? 54 : 32;
  const groupH = 30;
  const top = 28;
  let y = top;
  const layout: {
    kind: "group" | "item";
    y: number;
    g: MeanGroup;
    item?: MeanGroup["items"][number];
  }[] = [];
  for (const g of groups) {
    layout.push({ kind: "group", y, g });
    y += groupH;
    for (const it of g.items) {
      layout.push({ kind: "item", y, g, item: it });
      y += rowH;
    }
    y += stacked ? 16 : 8;
  }
  const plotH = y - top;
  const H = y + 44;
  const { x, ticks } = xScale(domain, plotL, plotR, narrow ? 54 : 70);
  const x0 = x(baseline);
  const go = inView || reduce;
  const hovered = layout.find(
    (l) => l.kind === "item" && l.item?.method === hover,
  );

  return (
    <div
      ref={ref}
      className="relative w-full"
      onPointerLeave={() => setHover(null)}
    >
      <svg
        ref={inViewRef}
        width={W}
        height={H}
        role="img"
        aria-label={ariaLabel}
        className="block"
      >
        {/* grid and baseline drawn per item row, so group headers stay clear */}
        {layout
          .filter((l) => l.kind === "item")
          .map((l, i) => (
            <g key={`grid${i}`}>
              {ticks.map((t) => (
                <line
                  key={t}
                  x1={x(t)}
                  x2={x(t)}
                  y1={l.y}
                  y2={l.y + rowH}
                  style={{ stroke: "var(--grid)" }}
                />
              ))}
            </g>
          ))}
        <line
          x1={x0}
          x2={x0}
          y1={top - 8}
          y2={top + plotH}
          style={{ stroke: "var(--ink-2)" }}
          strokeWidth={1.25}
        />
        {ticks.map((t) => (
          <text
            key={t}
            x={x(t)}
            y={top + plotH + 20}
            textAnchor="middle"
            className="tabular"
            style={{ fill: "var(--muted)", fontSize: 13.5 }}
          >
            {t.toString().replace("-", "−")}
          </text>
        ))}
        <text
          x={(plotL + plotR) / 2}
          y={top + plotH + 40}
          textAnchor="middle"
          style={{ fill: "var(--ink-2)", fontSize: 14.5 }}
        >
          {axisLabel}
        </text>
        <text
          x={x0 - textWidth(baselineLabel, 13.5) / 2 < 2 ? x0 - 4 : x0}
          y={top - 12}
          textAnchor={
            x0 - textWidth(baselineLabel, 13.5) / 2 < 2 ? "start" : "middle"
          }
          style={{ fill: "var(--ink-2)", fontSize: 13.5 }}
        >
          {baselineLabel}
        </text>
        {layout.map((l, i) => {
          if (l.kind === "group") {
            return (
              <g key={`g${i}`}>
                <text
                  x={0}
                  y={l.y + 19}
                  style={{
                    fill: "var(--ink)",
                    fontSize: 14.5,
                    fontWeight: 650,
                    letterSpacing: "0.02em",
                    stroke: "var(--surface)",
                    strokeWidth: 5,
                    paintOrder: "stroke",
                    strokeLinejoin: "round",
                  }}
                >
                  {l.g.label}
                  {l.g.sub && !narrow && (
                    <tspan
                      style={{ fill: "var(--muted)", fontWeight: 400 }}
                    >{`  ${l.g.sub}`}</tspan>
                  )}
                </text>
              </g>
            );
          }
          const it = l.item!;
          const st = METHODS[it.method];
          const v = it.value;
          const off = v === LT9 || (isNum(v) && v < domain[0]);
          const xv = isNum(v)
            ? x(Math.max(domain[0], Math.min(domain[1], v)))
            : x(domain[0]);
          const cy = stacked ? l.y + 38 : l.y + rowH / 2;
          const ly = stacked ? l.y + 15 : cy;
          return (
            <g
              key={it.method}
              tabIndex={0}
              role="listitem"
              aria-label={`${st?.label}: ${cellText(v)}`}
              onPointerEnter={() => setHover(it.method)}
              onFocus={() => setHover(it.method)}
              onBlur={() => setHover(null)}
              className="outline-none"
            >
              <rect
                x={0}
                y={l.y}
                width={W}
                height={rowH}
                rx={6}
                style={{
                  fill:
                    hover === it.method ? "var(--surface-2)" : "transparent",
                }}
              />
              <g transform={`translate(14,${ly})`}>
                {st && <MethodMark m={st} r={5.5} />}
              </g>
              <text
                x={28}
                y={ly}
                dy="0.32em"
                style={{
                  fill: "var(--ink)",
                  fontSize: 15,
                  fontWeight: emph.has(it.method) ? 650 : 400,
                  stroke: "var(--surface)",
                  strokeWidth: 4,
                  paintOrder: "stroke",
                  strokeLinejoin: "round",
                }}
              >
                {st?.label ?? it.method}
              </text>
              {v !== null && !off && (
                <>
                  <motion.line
                    x1={x0}
                    y1={cy}
                    y2={cy}
                    initial={false}
                    animate={{ x2: go ? xv : x0 }}
                    transition={
                      reduce
                        ? { duration: 0 }
                        : {
                            duration: 0.7,
                            delay: 0.05 * i,
                            ease: [0.2, 0.7, 0.2, 1],
                          }
                    }
                    style={{ stroke: catVar(st?.slot ?? 0) }}
                    strokeWidth={2}
                  />
                  <motion.g
                    initial={false}
                    animate={{ x: go ? xv : x0, y: cy }}
                    transition={
                      reduce
                        ? { duration: 0 }
                        : {
                            duration: 0.7,
                            delay: 0.05 * i,
                            ease: [0.2, 0.7, 0.2, 1],
                          }
                    }
                  >
                    {st && <MethodMark m={st} r={6.5} />}
                  </motion.g>
                  <motion.text
                    initial={false}
                    animate={{ opacity: go ? 1 : 0 }}
                    transition={
                      reduce
                        ? { duration: 0 }
                        : { duration: 0.3, delay: 0.05 * i + 0.5 }
                    }
                    x={v >= baseline ? xv + 12 : xv - 12}
                    y={cy}
                    dy="0.32em"
                    textAnchor={v >= baseline ? "start" : "end"}
                    className="tabular"
                    style={{
                      fill: "var(--ink)",
                      fontSize: 15,
                      fontWeight: emph.has(it.method) ? 700 : 500,
                    }}
                  >
                    {cellText(v)}
                  </motion.text>
                </>
              )}
              {off && (
                <g transform={`translate(${plotL},${cy})`}>
                  <path
                    d="M6,-6 L-1,0 L6,6"
                    style={{ fill: "none", stroke: catVar(st?.slot ?? 0) }}
                    strokeWidth={2.4}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <text
                    x={12}
                    dy="0.32em"
                    style={{
                      fill: "var(--ink)",
                      fontSize: 15,
                      fontWeight: 500,
                    }}
                  >
                    {cellText(v)}
                  </text>
                </g>
              )}
              {v === null && (
                <text
                  x={plotL + 4}
                  y={cy}
                  dy="0.32em"
                  style={{ fill: "var(--muted)", fontSize: 14.5 }}
                >
                  not reported
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {hovered?.item && (
        <Tip x={W * 0.5} y={hovered.y + rowH + 2} width={W}>
          <div className="font-semibold text-ink">
            {METHODS[hovered.item.method]?.label}:{" "}
            <span className="num">{cellText(hovered.item.value)}</span>
          </div>
          <div className="mt-0.5 text-[14.5px] text-ink-2">
            {METHODS[hovered.item.method]?.blurb}
          </div>
          {hovered.item.note && (
            <div className="mt-1 text-[14px] text-muted">
              {hovered.item.note}
            </div>
          )}
        </Tip>
      )}
    </div>
  );
}
