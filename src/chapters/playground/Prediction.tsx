/**
 * Center-top: the prediction at the masked cell (spec §8.3-8.4), its change against the baseline,
 * and a short history of the last runs.
 */
import { ArrowDownRight, ArrowUpRight, Minus, Trash2 } from "lucide-react";
import { motion } from "motion/react";
import { LAYER_INFO, StatusBadge, cx } from "@/components";
import { ATTN_ORDER } from "@/engine/types";
import { useElementSize } from "@/components/viz/useSize";
import { usePrefersReducedMotion } from "@/styles/theme";
import { labelText, usePlayLive } from "./context";
import { usePlay } from "./state";
import {
  FIG4,
  LAYER_ABLATION,
  paperArrow,
  paperMetric,
  paperNum,
  TABLE10,
  TABLE4,
} from "./paper";
import { fmtDelta, fmtDisplay, targetPhrase } from "./usePlayground";

/** Value-axis domain shared by the scale bar and the history. */
function useDomain(): [number, number] {
  const { task, base, cur, truth, stats } = usePlayLive();
  const history = usePlay((s) => s.history);
  if (task.targetType === "binary") return [0, 1];
  const vals = [
    base?.display,
    cur?.display,
    typeof truth === "number" ? truth : undefined,
    stats.mean,
    ...history.map((h) => h.value),
  ].filter((v): v is number => v !== undefined && Number.isFinite(v));
  const lo = Math.min(0, ...vals);
  const hi = Math.max(...vals, stats.mean + stats.std);
  const pad = (hi - lo) * 0.06 || 1;
  return [lo === 0 ? 0 : lo - pad, hi + pad];
}

function ScaleBar() {
  const { task, base, cur, truth, active } = usePlayLive();
  const [ref, size] = useElementSize<HTMLDivElement>();
  const reduce = usePrefersReducedMotion();
  const [lo, hi] = useDomain();
  const W = Math.max(240, size.width || 480);
  const padX = 14;
  const x = (v: number) =>
    padX +
    ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo || 1)) * (W - 2 * padX);
  const binary = task.targetType === "binary";
  const ticks = binary
    ? [0, 0.25, 0.5, 0.75, 1]
    : niceTicks(lo, hi, W < 360 ? 4 : 6);
  const truthNum = typeof truth === "number" ? truth : null;
  const H = 64;
  const y = 22;
  return (
    <div ref={ref} className="w-full">
      <svg
        width={W}
        height={H}
        role="img"
        aria-label={`Prediction scale: baseline ${base ? fmtDisplay(base.display, task) : "loading"}, current ${cur ? fmtDisplay(cur.display, task) : "loading"}`}
        className="block"
      >
        <line
          x1={padX}
          x2={W - padX}
          y1={y}
          y2={y}
          style={{ stroke: "var(--grid)" }}
          strokeWidth={6}
          strokeLinecap="round"
        />
        {binary && (
          <line
            x1={x(0.5)}
            x2={x(0.5)}
            y1={y - 9}
            y2={y + 9}
            style={{ stroke: "var(--axis)" }}
            strokeWidth={1}
            strokeDasharray="3 3"
          />
        )}
        {ticks.map((t) => (
          <text
            key={t}
            x={x(t)}
            y={H - 6}
            textAnchor="middle"
            className="num"
            style={{ fontSize: 13, fill: "var(--muted)" }}
          >
            {binary ? `${Math.round(t * 100)}%` : formatTick(t)}
          </text>
        ))}
        {truthNum !== null && (
          <g>
            <path
              d={`M${x(truthNum)},${y - 11} l5,5 l-5,5 l-5,-5 Z`}
              style={{ fill: "var(--ink-2)" }}
            />
            <title>True value {labelText(truthNum)}</title>
          </g>
        )}
        {base && active && (
          <circle
            cx={x(base.display)}
            cy={y}
            r={7}
            style={{ fill: "var(--surface)", stroke: "var(--ink-2)" }}
            strokeWidth={2}
          >
            <title>Baseline {fmtDisplay(base.display, task)}</title>
          </circle>
        )}
        {cur && (
          <motion.circle
            initial={false}
            animate={{ cx: x(cur.display) }}
            transition={
              reduce
                ? { duration: 0 }
                : { type: "spring", stiffness: 260, damping: 26 }
            }
            cy={y}
            r={8}
            style={{ fill: "var(--accent)", stroke: "var(--surface)" }}
            strokeWidth={2}
          />
        )}
      </svg>
    </div>
  );
}

function niceTicks(lo: number, hi: number, n: number): number[] {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step =
    [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step)
    out.push(Math.round(v * 1e6) / 1e6);
  return out;
}

function formatTick(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

export function PredictionCard() {
  const live = usePlayLive();
  const { task, base, cur, truth, active, ablAssets, baseCkpt, ablCkpt } = live;
  const phrase = targetPhrase(task);
  const delta = base && cur ? cur.display - base.display : 0;
  const dir = Math.abs(delta) < 1e-9 ? 0 : Math.sign(delta);
  const Icon = dir > 0 ? ArrowUpRight : dir < 0 ? ArrowDownRight : Minus;
  const binary = task.targetType === "binary";

  return (
    <div className="min-w-0">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="eyebrow">Prediction at the masked cell</div>
        {ablAssets.weightsKind === "init" && (
          <StatusBadge status="warning">untrained weights</StatusBadge>
        )}
      </div>
      <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
        <div aria-live="polite">
          <div className="text-[15.5px] font-medium text-ink-2">
            {active ? "With your changes" : "Full model, live context"}
          </div>
          <div className="num mt-1 text-[46px] leading-none font-semibold tracking-tight text-ink">
            {cur ? fmtDisplay(cur.display, task) : "…"}
          </div>
          <div className="mt-1.5 text-[15px] text-ink-2">
            {phrase.short}
            {!binary && phrase.unit ? ` (${phrase.unit})` : ""} ·{" "}
            {binary ? "logit" : "z"}{" "}
            <span className="num">{cur ? cur.output.toFixed(3) : "…"}</span>
          </div>
        </div>
        <div className="flex flex-col gap-1 pb-1 text-[15px] text-ink-2">
          {active && (
            <div className="flex items-center gap-1.5">
              <Icon
                aria-hidden="true"
                size={16}
                strokeWidth={2.2}
                className="text-ink-2"
              />
              <span className="num font-semibold text-ink">
                {base && cur ? fmtDelta(delta, task) : "…"}
              </span>
              <span>vs baseline</span>
            </div>
          )}
          <div>
            Baseline{" "}
            <span className="num text-ink">
              {base ? fmtDisplay(base.display, task) : "…"}
            </span>{" "}
            <span className="text-muted">
              ({baseCkpt}
              {ablCkpt !== baseCkpt ? "" : ", all layers"})
            </span>
          </div>
          <div>
            Hidden label{" "}
            <span
              className="num rounded-md bg-surface-2 px-1.5 py-px text-ink"
              title="Stored in the target token but gated out of the input (spec §5.3)"
            >
              {task.targetCol} = {labelText(truth)}
            </span>
          </div>
        </div>
      </div>
      <div className="mt-3">
        <ScaleBar />
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[14.5px] text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <svg width="14" height="14" aria-hidden="true">
            <circle cx="7" cy="7" r="5.5" style={{ fill: "var(--accent)" }} />
          </svg>
          {active ? "with your changes" : "prediction"}
        </span>
        {active && (
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
            baseline
          </span>
        )}
        {!binary && (
          <span className="inline-flex items-center gap-1.5">
            <svg width="12" height="12" aria-hidden="true">
              <path
                d="M6,1 l5,5 l-5,5 l-5,-5 Z"
                style={{ fill: "var(--ink-2)" }}
              />
            </svg>
            hidden true value
          </span>
        )}
        {binary && <span>dashed line: 50%</span>}
      </div>
    </div>
  );
}

export function History() {
  const { task, base } = usePlayLive();
  const history = usePlay((s) => s.history);
  const clear = usePlay((s) => s.clearHistory);
  const [lo, hi] = useDomain();
  if (history.length === 0) return null;
  const pct = (v: number) =>
    ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo || 1)) * 100;
  const rows = [...history].reverse();
  return (
    <div className="min-w-0">
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <div className="text-[15.5px] font-semibold text-ink">Recent runs</div>
        <button
          type="button"
          onClick={clear}
          className="inline-flex items-center gap-1 text-[14.5px] text-ink-2 hover:text-ink max-sm:min-h-11 max-sm:px-2"
        >
          <Trash2 aria-hidden="true" size={14} /> Clear
        </button>
      </div>
      <ol
        className="!m-0 flex !list-none flex-col gap-1.5 !p-0"
        aria-label="Recent predictions, newest first"
      >
        {rows.map((h, i) => (
          <li
            key={h.id}
            className="!m-0 grid grid-cols-[minmax(0,1fr)_minmax(90px,38%)_auto] items-center gap-x-3 !p-0"
          >
            <span
              className={cx(
                "min-w-0 text-[14.5px] leading-snug",
                i === 0 ? "text-ink" : "text-ink-2",
              )}
            >
              {h.chips.join(" · ")}
            </span>
            <span className="relative block h-3.5 rounded-sm bg-surface-2">
              <span
                className="absolute inset-y-0 left-0 rounded-r-[3px]"
                style={{
                  width: `${pct(h.value)}%`,
                  background: i === 0 ? "var(--accent)" : "var(--axis)",
                }}
              />
              {base && (
                <span
                  aria-hidden="true"
                  className="absolute -inset-y-0.5 w-px bg-ink-2"
                  style={{ left: `${pct(base.display)}%` }}
                />
              )}
            </span>
            <span
              className={cx(
                "num text-right text-[14.5px]",
                i === 0 ? "font-semibold text-ink" : "text-ink-2",
              )}
            >
              {fmtDisplay(h.value, task)}
            </span>
          </li>
        ))}
      </ol>
      <p className="!mb-0 mt-1.5 text-[14px] text-muted">
        Thin vertical line: the baseline. The newest run is on top.
      </p>
    </div>
  );
}

/**
 * The paper's number for each switch that is on (mean zero-shot over RelBench tasks, the live task's
 * metric), shown right under the live prediction.
 */
export function PaperSideBySide() {
  const { task, tokenAb, disabled, effL, globalL, ablCkpt, baseCkpt } =
    usePlayLive();
  const binary = task.targetType === "binary";
  const m = paperMetric(binary);
  const items: { key: string; label: string; value: string; src: string }[] =
    [];
  if (tokenAb.removeSelf)
    items.push({
      key: "self",
      label: "Self labels removed",
      value: paperArrow(TABLE4.none, TABLE4.selfLabels, binary),
      src: "Table 4, p.9",
    });
  if (tokenAb.removeOther)
    items.push({
      key: "other",
      label: "Other-entity labels removed",
      value: paperArrow(TABLE4.none, TABLE4.otherLabels, binary),
      src: "Table 4, p.9",
    });
  if (tokenAb.shuffle)
    items.push({
      key: "names",
      label: "Column and table names shuffled",
      value: paperArrow(TABLE4.none, TABLE4.colNames, binary),
      src: "Table 4, p.9",
    });
  for (const l of ATTN_ORDER)
    if (disabled.includes(l))
      items.push({
        key: l,
        label: `${LAYER_INFO[l].name} attention removed, model retrained`,
        value: paperArrow(LAYER_ABLATION.none, LAYER_ABLATION[l], binary),
        src: binary ? "Table 13, p.23" : "Table 5, p.10",
      });
  if (effL < globalL)
    items.push({
      key: "L",
      label: `Context ${FIG4.pretrainedAt} → ${FIG4.contexts[0]} cells`,
      value: binary ? "AUROC −4 points" : "R² −7 points",
      src: "Fig. 4, p.8",
    });
  if (ablCkpt !== baseCkpt && ablCkpt.startsWith("finetune"))
    items.push({
      key: "ft",
      label: "Fully fine-tuned RT (not zero-shot)",
      value: `${m.name} ${paperNum(binary ? TABLE10.pretrained.clf : TABLE10.pretrained.reg)}`,
      src: "Table 10, p.21",
    });
  if (ablCkpt === "init" && ablCkpt !== baseCkpt)
    items.push({
      key: "init",
      label: "Untrained RT, zero-shot",
      value: "not reported",
      src: "",
    });
  if (items.length === 0) return null;
  return (
    <div className="rounded-lg border border-border bg-surface-2/60 px-3 py-2.5">
      <div className="text-[15px] font-semibold text-ink">
        The same switch in the paper
      </div>
      <p className="!m-0 text-[14px] leading-snug text-muted">
        Mean zero-shot {m.name} (%) over {m.tasks} of RelBench, a 22M-parameter
        model. Compare the direction, not the size.
      </p>
      <ul className="!m-0 mt-1.5 flex !list-none flex-col gap-1 !p-0">
        {items.map((it) => (
          <li
            key={it.key}
            className="!m-0 flex flex-wrap items-baseline justify-between gap-x-3 !p-0 text-[15px]"
          >
            <span className="text-ink-2">{it.label}</span>
            <span className="num text-ink">
              {it.value}
              {it.src && (
                <span className="ml-2 font-sans text-[14px] text-muted">
                  {it.src}
                </span>
              )}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
