/**
 * Results on ShopDB (results.json): for each forecast task, five models on one axis as a dot plot
 * (position, not length: AUROC differences of a few points would vanish in bars from 0), with the
 * val split used for selection (hollow) and the reported test split (filled). Plus: the spread of
 * seven untrained initializations (diagnostics.untrainedInitSeeds), the paper's means for context,
 * and a "re-run it in your browser" panel that recomputes each number with the live engine.
 */
import { Check, Play, X } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import {
  Button,
  Cite,
  DataTable,
  Segmented,
  SVG_TYPE as T,
  textWidth,
} from "@/components";
import { useElementSize } from "@/components/viz/useSize";
import type { CheckpointName } from "@/engine/pipeline";
import { loadPublicJson } from "@/engine/model/io";
import { catVar } from "@/styles/palette";
import { fx, int, pct, sci, SHOP_FORECAST, type Results } from "./data";
import { useBrowserEval } from "./evaluate";
import { T1_MEAN, T2_MEAN, T10_MEAN } from "./paper";

type Row = {
  id: string;
  label: string;
  val?: number;
  test?: number;
  emphasis?: boolean;
  note?: string;
};

export function rowsFor(results: Results, taskId: string): Row[] {
  const z = results.zeroShot?.[taskId];
  const ft = SHOP_FORECAST.find((t) => t.id === taskId)?.ft ?? "churn";
  const fp = results.finetune?.[`finetune-${ft}-pretrained`];
  const fu = results.finetune?.[`finetune-${ft}-untrained`];
  return [
    {
      id: "init",
      label: "Untrained RT, no gradient step",
      val: z?.untrainedInit?.val,
      test: z?.untrainedInit?.test,
    },
    {
      id: "em",
      label: "EntityMean baseline",
      val: z?.entityMean?.val,
      test: z?.entityMean?.test,
    },
    {
      id: "zs",
      label: "RT zero-shot (pretrained)",
      val: z?.pretrainBest?.val,
      test: z?.pretrainBest?.test,
      emphasis: true,
      note:
        z?.pretrainBest?.step !== undefined
          ? `step ${int(z.pretrainBest.step)}`
          : undefined,
    },
    {
      id: "fu",
      label: "RT fine-tuned from random init",
      val: fu?.best?.val,
      test: fu?.best?.test,
      note:
        fu?.best?.step !== undefined ? `step ${int(fu.best.step)}` : undefined,
    },
    {
      id: "fp",
      label: "RT fine-tuned from pretrained",
      val: fp?.best?.val,
      test: fp?.best?.test,
      note:
        fp?.best?.step !== undefined ? `step ${int(fp.best.step)}` : undefined,
    },
  ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Dot plot
// ─────────────────────────────────────────────────────────────────────────────

function DotPlot({
  rows,
  metric,
  title,
}: {
  rows: Row[];
  metric: "auroc" | "r2";
  title: string;
}) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const width = Math.max(280, size.width || 600);
  const [hover, setHover] = useState<string | null>(null);
  const vals = rows
    .flatMap((r) => [r.val, r.test])
    .filter((v): v is number => v !== undefined && Number.isFinite(v))
    .map((v) => v * 100);
  const lo0 = metric === "auroc" ? 40 : Math.min(-10, ...vals);
  const hi0 = metric === "auroc" ? 90 : Math.max(...vals, 50);
  const lo = Math.floor(Math.min(lo0, ...vals) / 10) * 10;
  const hi = Math.ceil(Math.max(hi0, ...vals) / 10) * 10;
  const narrow = width < 560;
  const labelW = narrow
    ? 0
    : Math.min(
        260,
        Math.max(...rows.map((r) => textWidth(r.label, T.category))) + 16,
      );
  // narrow: room for the half-width of the first / last tick label, and for the end dot's ring
  const left = narrow ? 24 : labelW + 8;
  const right = narrow ? 26 : 18;
  const rowH = narrow ? 58 : 40;
  const top = 8;
  const plotW = width - left - right;
  const H = top + rows.length * rowH + 34;
  const x = (v: number) => left + ((v - lo) / (hi - lo)) * plotW;
  const ticks: number[] = [];
  const stepT = hi - lo > 80 ? 20 : 10;
  for (let t = lo; t <= hi + 1e-9; t += stepT) ticks.push(t);
  const ref0 = metric === "auroc" ? 50 : 0;
  return (
    <div ref={ref} className="relative w-full">
      <svg
        width={width}
        height={H}
        role="img"
        aria-label={title}
        className="block"
      >
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={x(t)}
              x2={x(t)}
              y1={top}
              y2={H - 30}
              style={{ stroke: "var(--grid)" }}
              strokeWidth={1}
            />
            <text
              x={x(t)}
              y={H - 12}
              textAnchor="middle"
              className="tabular"
              style={{ fill: "var(--muted)", fontSize: T.tick }}
            >
              {t}
            </text>
          </g>
        ))}
        <line
          x1={x(ref0)}
          x2={x(ref0)}
          y1={top}
          y2={H - 30}
          style={{ stroke: "var(--ink-2)", opacity: 0.6 }}
          strokeWidth={1.5}
        />
        {rows.map((r, i) => {
          const cy = top + i * rowH + (narrow ? 38 : rowH / 2);
          const v = r.val !== undefined ? r.val * 100 : NaN;
          const t = r.test !== undefined ? r.test * 100 : NaN;
          const slot = r.emphasis ? 1 : 0;
          const color = r.emphasis ? catVar(1) : "var(--ink-2)";
          const hovered = hover === r.id;
          return (
            <g
              key={r.id}
              onPointerEnter={() => setHover(r.id)}
              onPointerLeave={() => setHover(null)}
            >
              <rect
                x={0}
                y={top + i * rowH}
                width={width}
                height={rowH}
                fill={hovered ? "var(--surface-2)" : "transparent"}
              />
              <text
                x={narrow ? left : labelW}
                y={narrow ? top + i * rowH + 16 : cy}
                dy={narrow ? undefined : "0.32em"}
                textAnchor={narrow ? "start" : "end"}
                style={{
                  fill: "var(--ink)",
                  fontSize: T.category,
                  fontWeight: r.emphasis ? 600 : 400,
                  paintOrder: "stroke",
                  stroke: hovered ? "var(--surface-2)" : "var(--surface)",
                  strokeWidth: narrow ? 5 : 0,
                  strokeLinejoin: "round",
                }}
              >
                {r.label}
              </text>
              {Number.isFinite(v) && Number.isFinite(t) && (
                <line
                  x1={x(Math.min(v, t))}
                  x2={x(Math.max(v, t))}
                  y1={cy}
                  y2={cy}
                  style={{ stroke: color, opacity: 0.45 }}
                  strokeWidth={2}
                />
              )}
              {Number.isFinite(v) && (
                <circle
                  cx={x(v)}
                  cy={cy}
                  r={6}
                  style={{ fill: "var(--surface)", stroke: color }}
                  strokeWidth={2.2}
                />
              )}
              {Number.isFinite(t) && (
                <circle
                  cx={x(t)}
                  cy={cy}
                  r={6.5}
                  style={{ fill: color, stroke: "var(--surface)" }}
                  strokeWidth={2}
                  data-slot={slot}
                />
              )}
              {Number.isFinite(t) && (
                <text
                  x={x(t) + (t >= v || !Number.isFinite(v) ? 12 : -12)}
                  y={cy}
                  dy="0.32em"
                  textAnchor={t >= v || !Number.isFinite(v) ? "start" : "end"}
                  className="tabular"
                  style={{
                    fill: "var(--ink)",
                    fontSize: T.value,
                    fontWeight: r.emphasis ? 600 : 400,
                    paintOrder: "stroke",
                    stroke: "var(--surface)",
                    strokeWidth: 4,
                  }}
                >
                  {fx(t, 1)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {hover &&
        (() => {
          const r = rows.find((q) => q.id === hover)!;
          const i = rows.indexOf(r);
          return (
            <div
              role="tooltip"
              className="pointer-events-none absolute z-20 rounded-[9px] border border-border bg-surface px-3 py-2 text-[15px] leading-snug shadow-[var(--shadow-pop)]"
              style={{
                left: Math.min(width - 230, Math.max(0, left)),
                top: top + i * rowH + rowH,
              }}
            >
              <div className="font-semibold text-ink">{r.label}</div>
              <div className="num text-ink-2">
                val {pct(r.val)} · test {pct(r.test)}
                {r.note ? ` · ${r.note}` : ""}
              </div>
            </div>
          );
        })()}
    </div>
  );
}

export function ResultsFigure({ results }: { results: Results | null }) {
  if (!results)
    return (
      <div className="wide my-8 card px-5 py-6 text-[16px] text-ink-2">
        Loading results.json…
      </div>
    );
  const lp = results.labelPersistence?.tasks ?? {};
  return (
    <figure className="wide my-8 min-w-0">
      <div className="card px-4 py-4 sm:px-5 sm:py-5">
        <div className="mb-1 text-[17px] font-semibold text-ink">
          Zero-shot on ShopDB, next to the baselines and to fine-tuning
        </div>
        <div
          className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-[15px] text-ink-2"
          aria-label="Legend"
        >
          <span className="inline-flex items-center gap-1.5">
            <svg width="14" height="14" aria-hidden="true">
              <circle cx="7" cy="7" r="5.5" style={{ fill: "var(--ink-2)" }} />
            </svg>
            test (reported)
          </span>
          <span className="inline-flex items-center gap-1.5">
            <svg width="14" height="14" aria-hidden="true">
              <circle
                cx="7"
                cy="7"
                r="5"
                style={{ fill: "var(--surface)", stroke: "var(--ink-2)" }}
                strokeWidth={2}
              />
            </svg>
            val (used to choose the step)
          </span>
          <span>values in %, like the paper’s tables</span>
        </div>
        <div className="grid gap-7">
          {SHOP_FORECAST.map((t) => {
            const rows = rowsFor(results, t.id);
            const nVal = lp[t.id]?.splits?.Val?.rows;
            const nTest = lp[t.id]?.splits?.Test?.rows;
            return (
              <div key={t.id} className="min-w-0">
                <div className="text-[16px] font-semibold text-ink">
                  <span className="font-mono">{t.short}</span>, {t.metricLabel}{" "}
                  (%)
                </div>
                <div className="text-[14.5px] text-ink-2">
                  {nVal !== undefined
                    ? `${nVal} val seeds, ${nTest} test seeds · `
                    : ""}
                  vertical rule:{" "}
                  {t.metric === "auroc"
                    ? "random ranking (50)"
                    : "always predicting the mean (0)"}
                </div>
                <DotPlot
                  rows={rows}
                  metric={t.metric}
                  title={`ShopDB ${t.short} ${t.metricLabel}`}
                />
                <details className="mt-1 text-[15px]">
                  <summary className="cursor-pointer text-ink-2">
                    Table view
                  </summary>
                  <DataTable
                    caption={`ShopDB ${t.short}`}
                    columns={[
                      "model",
                      `val ${t.metricLabel}`,
                      `test ${t.metricLabel}`,
                      "selected step",
                    ]}
                    rows={rows.map((r) => [
                      r.label,
                      pct(r.val),
                      pct(r.test),
                      r.note ?? "—",
                    ])}
                  />
                </details>
              </div>
            );
          })}
        </div>
        <PaperContext />
      </div>
      <figcaption className="mx-auto mt-3 max-w-[var(--prose-max)] text-[16px] leading-snug text-ink-2">
        Every RT number is the test metric recorded at the step with the best
        validation metric, as in the paper{" "}
        <Cite kind="code" ref="rt/main.py:377-392" />. EntityMean predicts the
        mean of the entity’s past labels, falling back to the Train mean{" "}
        <Cite kind="paper" ref="p.23 App. I.1" />.
      </figcaption>
    </figure>
  );
}

function PaperContext() {
  const rows: (string | number)[][] = [
    [
      "EntityMean (Tables 1, 2)",
      T1_MEAN.entityMean.toFixed(1),
      T2_MEAN.entityMean.toFixed(1),
    ],
    [
      "RT zero-shot, target DB unseen, “No” (Tables 1, 2)",
      T1_MEAN.rtNo.toFixed(1),
      T2_MEAN.rtNo.toFixed(1),
    ],
    [
      "RT fine-tuned from random init (Table 10)",
      T10_MEAN.auroc.rtUntrained.toFixed(1),
      T10_MEAN.r2.rtUntrained.toFixed(1),
    ],
    [
      "RT fine-tuned from pretrained (Table 10)",
      T10_MEAN.auroc.rtPretrained.toFixed(1),
      T10_MEAN.r2.rtPretrained.toFixed(1),
    ],
  ];
  return (
    <div className="mt-6 rounded-lg border border-border bg-surface-2 px-3.5 py-3">
      <div className="text-[16px] font-semibold text-ink">
        For context: the paper’s means over RelBench{" "}
        <Cite kind="paper" ref="p.8 Tables 1-2" />{" "}
        <Cite kind="paper" ref="p.21 Table 10" />
      </div>
      <div className="text-[14.5px] text-ink-2">
        10 classification and 8 regression tasks from 6 real databases.
        Different data and scale: compare the ordering, not the values.
      </div>
      <div className="relative mt-2 overflow-x-auto">
        <DataTable
          caption="Paper means"
          columns={["model (source)", "AUROC (%)", "R² (%)"]}
          rows={rows}
        />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Luck: seven untrained initializations
// ─────────────────────────────────────────────────────────────────────────────

export function InitSpread({ results }: { results: Results | null }) {
  const seeds = results?.diagnostics?.untrainedInitSeeds;
  if (!results || !seeds) return null;
  return <InitSpreadInner results={results} seeds={seeds} />;
}

function InitSpreadInner({
  results,
  seeds,
}: {
  results: Results;
  seeds: Record<string, Record<string, number>>;
}) {
  const [split, setSplit] = useState<"Val" | "Test">("Test");
  const [ref, size] = useElementSize<HTMLDivElement>();
  const width = Math.max(280, size.width || 600);
  const lanes = SHOP_FORECAST.map((t) => {
    const vals = Object.entries(seeds)
      .map(([s, m]) => ({ seed: s, v: m[`${t.id}/${split}`] }))
      .filter((d) => Number.isFinite(d.v));
    const z = results.zeroShot?.[t.id];
    const key = split === "Val" ? "val" : "test";
    return {
      t,
      vals,
      zs: z?.pretrainBest?.[key],
      em: z?.entityMean?.[key] ?? undefined,
    };
  });
  const laneH = 104;
  const left = 16;
  const right = 16;
  const H = lanes.length * laneH + 8;
  return (
    <figure className="wide my-8 min-w-0">
      <div className="card px-4 py-4 sm:px-5 sm:py-5">
        <div className="mb-3 flex flex-wrap items-end gap-3">
          <div className="mr-auto min-w-0">
            <div className="text-[17px] font-semibold text-ink">
              How lucky is an untrained network?
            </div>
            <div className="text-[15px] text-ink-2">
              Seven random initializations, no training at all, evaluated like
              the zero-shot model
            </div>
          </div>
          <Segmented
            label="Split"
            value={split}
            onChange={setSplit}
            options={[
              { value: "Val", label: "val" },
              { value: "Test", label: "test" },
            ]}
          />
        </div>
        <div ref={ref}>
          <svg
            width={width}
            height={H}
            role="img"
            aria-label="Metric of seven untrained initializations"
            className="block"
          >
            {lanes.map((l, li) => {
              const all = [...l.vals.map((d) => d.v), l.zs ?? NaN, l.em ?? NaN]
                .filter(Number.isFinite)
                .map((v) => v * 100);
              const isA = l.t.metric === "auroc";
              const lo = isA
                ? Math.min(25, Math.floor(Math.min(...all) / 5) * 5)
                : Math.floor(Math.min(...all, 0) / 10) * 10;
              const hi = isA
                ? Math.max(85, Math.ceil(Math.max(...all) / 5) * 5)
                : Math.ceil(Math.max(...all, 0) / 10) * 10;
              const x = (v: number) =>
                left + ((v - lo) / (hi - lo)) * (width - left - right);
              const y0 = li * laneH;
              const cy = y0 + 58;
              const ticks: number[] = [];
              const st = hi - lo > 60 ? 20 : 10;
              for (let t = Math.ceil(lo / st) * st; t <= hi; t += st)
                ticks.push(t);
              const marks: { v: number; label: string; slot: number }[] = [];
              if (l.zs !== undefined)
                marks.push({
                  v: l.zs * 100,
                  label: `RT zero-shot ${fx(l.zs * 100, 1)}`,
                  slot: 1,
                });
              if (l.em !== undefined)
                marks.push({
                  v: l.em * 100,
                  label: `EntityMean ${fx(l.em * 100, 1)}`,
                  slot: 0,
                });
              return (
                <g key={l.t.id}>
                  <text
                    x={left}
                    y={y0 + 16}
                    style={{
                      fill: "var(--ink)",
                      fontSize: T.category,
                      fontWeight: 600,
                    }}
                  >
                    {l.t.short} · {l.t.metricLabel} (%)
                  </text>
                  <line
                    x1={left}
                    x2={width - right}
                    y1={cy}
                    y2={cy}
                    style={{ stroke: "var(--axis)" }}
                    strokeWidth={1}
                  />
                  {ticks.map((t) => (
                    <g key={t}>
                      <line
                        x1={x(t)}
                        x2={x(t)}
                        y1={cy - 4}
                        y2={cy + 4}
                        style={{ stroke: "var(--axis)" }}
                      />
                      <text
                        x={x(t)}
                        y={cy + 22}
                        textAnchor="middle"
                        className="tabular"
                        style={{ fill: "var(--muted)", fontSize: T.tick }}
                      >
                        {t}
                      </text>
                    </g>
                  ))}
                  {marks.map((m, mi) => {
                    // first mark labeled above the axis, second below it, so they never collide
                    const right = x(m.v) > width * 0.7;
                    return (
                      <g key={m.label}>
                        <line
                          x1={x(m.v)}
                          x2={x(m.v)}
                          y1={cy - 16}
                          y2={cy + 8}
                          style={{
                            stroke: m.slot ? catVar(1) : "var(--ink-2)",
                          }}
                          strokeWidth={2}
                        />
                        <text
                          x={x(m.v) + (right ? -5 : 5)}
                          y={mi === 0 ? cy - 20 : cy + 38}
                          textAnchor={right ? "end" : "start"}
                          style={{
                            fill: "var(--ink-2)",
                            fontSize: T.annotation,
                            paintOrder: "stroke",
                            stroke: "var(--surface)",
                            strokeWidth: 4,
                          }}
                        >
                          {m.label}
                        </text>
                      </g>
                    );
                  })}
                  {l.vals.map((d) => (
                    <circle
                      key={d.seed}
                      cx={x(d.v * 100)}
                      cy={cy}
                      r={d.seed === "0" ? 7 : 6}
                      style={{
                        fill:
                          d.seed === "0" ? "var(--ink-2)" : "var(--surface)",
                        stroke:
                          d.seed === "0" ? "var(--surface)" : "var(--ink-2)",
                      }}
                      strokeWidth={2}
                    >
                      <title>{`init seed ${d.seed}${d.seed === "0" ? " (where our pretraining started)" : ""}: ${fx(d.v * 100, 1)}`}</title>
                    </circle>
                  ))}
                </g>
              );
            })}
          </svg>
        </div>
        <details className="mt-1 text-[15px]">
          <summary className="cursor-pointer text-ink-2">Table view</summary>
          <DataTable
            caption="Untrained initializations"
            columns={[
              "init seed",
              ...SHOP_FORECAST.map(
                (t) => `${t.short} ${split.toLowerCase()} (%)`,
              ),
            ]}
            rows={Object.entries(seeds).map(([s, m]) => [
              s === "0" ? "0 (our init)" : s,
              ...SHOP_FORECAST.map((t) => pct(m[`${t.id}/${split}`])),
            ])}
          />
        </details>
      </div>
      <figcaption className="mx-auto mt-3 max-w-[var(--prose-max)] text-[16px] leading-snug text-ink-2">
        Dots: PyTorch-default initializations after{" "}
        <code>torch.manual_seed(s)</code>, s = 0…{Object.keys(seeds).length - 1}
        . The filled dot, s = 0, is the starting point of our pretraining run.
        Vertical lines: the pretrained zero-shot model (blue) and EntityMean
        (gray) on the same split.
      </figcaption>
    </figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-run in the browser
// ─────────────────────────────────────────────────────────────────────────────

export function ReproducePanel({ results }: { results: Results | null }) {
  const [taskId, setTaskId] = useState(SHOP_FORECAST[0].id);
  // TS vs PyTorch parity, measured offline at every activation (public/data/verification.json)
  const [parity, setParity] = useState<number | undefined>();
  useEffect(() => {
    let alive = true;
    loadPublicJson<{ ts_vs_torch_max_abs_diff?: number }>(
      "data/verification.json",
    ).then(
      (v) => alive && setParity(v.ts_vs_torch_max_abs_diff),
      () => undefined,
    );
    return () => {
      alive = false;
    };
  }, []);
  const t = SHOP_FORECAST.find((q) => q.id === taskId)!;
  const ckpts: {
    name: CheckpointName;
    label: string;
    val?: number;
    test?: number;
  }[] = [];
  if (results) {
    const z = results.zeroShot?.[taskId];
    const info = (n: string) =>
      results.checkpoints?.[n] as { val?: number; test?: number } | undefined;
    ckpts.push({
      name: "init",
      label: "init (untrained)",
      val: z?.untrainedInit?.val,
      test: z?.untrainedInit?.test,
    });
    ckpts.push({
      name: `pretrain-${t.ft}` as CheckpointName,
      label: `pretrain-${t.ft} (zero-shot)`,
      val: info(`pretrain-${t.ft}`)?.val,
      test: info(`pretrain-${t.ft}`)?.test,
    });
    ckpts.push({
      name: `finetune-${t.ft}` as CheckpointName,
      label: `finetune-${t.ft} (fine-tuned)`,
      val: info(`finetune-${t.ft}`)?.val,
      test: info(`finetune-${t.ft}`)?.test,
    });
  }
  return (
    <figure className="wide my-8 min-w-0">
      <div className="card px-4 py-4 sm:px-5 sm:py-5">
        <div className="mb-3 flex flex-wrap items-end gap-3">
          <div className="mr-auto min-w-0">
            <div className="text-[17px] font-semibold text-ink">
              Check a number yourself
            </div>
            <div className="text-[15px] text-ink-2">
              Recompute any of them in this tab: one context and one forward
              pass per seed with the exported weights, then the metric
            </div>
          </div>
          <Segmented
            label="Task"
            value={taskId}
            onChange={setTaskId}
            options={SHOP_FORECAST.map((q) => ({
              value: q.id,
              label: q.short,
            }))}
          />
        </div>
        <div className="text-[15px]">
          <div
            className={`hidden border-b border-grid py-1.5 font-semibold text-ink-2 md:grid ${REPRO_COLS}`}
            aria-hidden="true"
          >
            <span>Checkpoint</span>
            <span>Split</span>
            <span className="text-right">results.json</span>
            <span className="text-right">Your browser</span>
            <span />
          </div>
          <ul
            className="!m-0 !p-0"
            style={{ listStyle: "none" }}
            aria-label="Recompute each metric"
          >
            {ckpts.flatMap((c) =>
              (["Val", "Test"] as const).map((sp) => (
                <ReproRow
                  key={`${taskId}-${c.name}-${sp}`}
                  taskId={taskId}
                  ckpt={c.name}
                  label={c.label}
                  split={sp}
                  expected={sp === "Val" ? c.val : c.test}
                  metric={t.metricLabel}
                />
              )),
            )}
          </ul>
        </div>
      </div>
      <figcaption className="mx-auto mt-3 max-w-[var(--prose-max)] text-[16px] leading-snug text-ink-2">
        L = 128, w = 16 and the evaluation RNG seed 0, as in training, whatever
        the global simulation settings say{" "}
        <Cite kind="code" ref="rt/main.py:158-160" />. The browser runs the
        TypeScript forward pass, which agrees with PyTorch to{" "}
        {parity !== undefined ? `at most ${sci(parity, 1)}` : "about 10⁻⁶"} at
        every activation of the verified contexts (
        <Link to="/about">chapter 16</Link>), so the metrics agree to the shown
        digits unless two seeds’ predictions are nearly tied.
      </figcaption>
    </figure>
  );
}

function ReproRow({
  taskId,
  ckpt,
  label,
  split,
  expected,
  metric,
}: {
  taskId: string;
  ckpt: CheckpointName;
  label: string;
  split: "Val" | "Test";
  expected?: number;
  metric: string;
}) {
  const ev = useBrowserEval({ taskId, split, ckpt, L: 128, w: 16 });
  const got = ev.result?.metric;
  const diff =
    got !== undefined && expected !== undefined
      ? Math.abs(got - expected)
      : undefined;
  let status: ReactNode = null;
  if (ev.status === "running" || ev.status === "waiting")
    status = (
      <span className="inline-flex items-center gap-2 text-ink-2">
        <span className="inline-block h-1.5 w-20 overflow-hidden rounded-full bg-surface-3">
          <span
            className="block h-full bg-accent"
            style={{ width: `${ev.total ? (100 * ev.done) / ev.total : 0}%` }}
          />
        </span>
        {ev.status === "waiting" ? "queued" : `${ev.done}/${ev.total}`}
        <button
          type="button"
          className="text-ink-2 underline hover:text-ink"
          onClick={ev.cancel}
        >
          cancel
        </button>
      </span>
    );
  else if (ev.status === "done" && diff !== undefined)
    status =
      diff < 5e-4 ? (
        <span className="inline-flex items-center gap-1 text-[var(--good-text)]">
          <Check aria-hidden="true" size={16} /> matches (Δ{" "}
          {diff < 1e-6 ? "< 10⁻⁶" : sci(diff, 1)})
        </span>
      ) : (
        <span className="inline-flex items-center gap-1 text-ink-2">
          <X aria-hidden="true" size={16} /> differs by {fx(diff, 4)}
        </span>
      );
  else if (ev.status === "error")
    status = <span className="text-ink-2">error: {ev.error}</span>;
  else
    status = (
      <Button
        size="sm"
        icon={<Play aria-hidden="true" size={14} />}
        onClick={ev.start}
      >
        Run
      </Button>
    );
  return (
    <li
      className={`!m-0 border-b border-grid/70 py-2 !pl-0 last:border-0 md:grid md:items-center md:gap-x-3 md:py-1.5 ${REPRO_COLS}`}
    >
      {/* narrow: two lines (checkpoint · split, then the two numbers); wide: one grid row, ordered with md:order-* */}
      <div className="flex items-center justify-between gap-3 md:contents">
        <span className="font-mono text-[14.5px] text-ink md:order-1">
          {label}
          <span className="font-sans text-ink-2 md:hidden">
            {" "}
            · {split.toLowerCase()}
          </span>
        </span>
        <span className="hidden text-ink-2 md:order-2 md:block">
          {split.toLowerCase()}
        </span>
        <span className="shrink-0 md:order-5">{status}</span>
      </div>
      <div className="mt-0.5 flex flex-wrap gap-x-4 md:contents">
        <span className="num text-ink md:order-3 md:text-right">
          <span className="font-sans text-ink-2 md:hidden">results.json </span>
          {metric} {fx(expected, 4)}
        </span>
        <span className="num text-ink md:order-4 md:text-right">
          <span className="font-sans text-ink-2 md:hidden">your browser </span>
          {got !== undefined ? fx(got, 4) : "–"}
        </span>
      </div>
    </li>
  );
}

const REPRO_COLS =
  "md:grid-cols-[minmax(0,1.5fr)_64px_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.5fr)]";
