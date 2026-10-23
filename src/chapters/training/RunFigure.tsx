/**
 * Our pretraining run, step by step (curves.json): the training loss of every optimizer step (the
 * model's own loss, mean over the batch's masked cells, model.py:265) and the global gradient norm
 * before clipping (main.py:413-418). One shared crosshair; the readout adds the lr / β1 of the step.
 */
import { useMemo, useState } from "react";
import { Cite, Segmented, Slider } from "@/components";
import { Readout, TrainChart } from "./charts";
import {
  fx,
  int,
  mainRun,
  mean,
  minutes,
  movingAverage,
  sci,
  type Curves,
  type Results,
} from "./data";

const WIN = 51;

export function RunFigure({
  curves,
  results,
}: {
  curves: Curves | null;
  results: Results | null;
}) {
  const run = mainRun(curves);
  const [hover, setHover] = useState<number | null>(null);
  const [gScale, setGScale] = useState<"linear" | "log">("log");

  const d = useMemo(() => {
    if (!run) return null;
    const ma = movingAverage(run.loss, WIN);
    const g = run.gradNorm ?? [];
    const gma = movingAverage(g, WIN);
    const clip = run.hp?.max_grad_norm ?? 1;
    return {
      raw: run.step.map((s, i) => ({ x: s, y: run.loss[i] })),
      ma: run.step.map((s, i) => ({ x: s, y: ma[i] })),
      g: run.step.map((s, i) => ({ x: s, y: g[i] })),
      gma: run.step.map((s, i) => ({ x: s, y: gma[i] })),
      clip,
      clippedFrac: g.length ? g.filter((v) => v > clip).length / g.length : NaN,
      firstLoss: mean(run.loss.slice(0, 100)),
      lastLoss: mean(run.loss.slice(-200)),
      clippedEarly:
        g.slice(0, 400).filter((v) => v > clip).length /
        Math.max(1, Math.min(400, g.length)),
      clippedLate:
        g.slice(-1000).filter((v) => v > clip).length /
        Math.max(1, Math.min(1000, g.length)),
    };
  }, [run]);

  if (!run || !d)
    return (
      <div className="wide my-8 card px-5 py-6 text-[16px] text-ink-2">
        Loading curves.json…
      </div>
    );

  const n = run.step.length;
  // readouts follow the crosshair; with nothing hovered they show the last step of the run
  const i =
    hover !== null
      ? Math.max(0, Math.min(n - 1, run.step.indexOf(hover)))
      : n - 1;
  const fmtStep = (s: number) => int(s);
  const wall = run.wallSeconds ?? results?.pretrain?.wallSeconds;

  return (
    <figure className="wide my-8 min-w-0">
      <div className="card px-4 py-4 sm:px-5 sm:py-5">
        <div className="mb-3 flex flex-wrap items-end justify-between gap-x-4 gap-y-2">
          <div className="text-[17px] font-semibold text-ink">
            Our pretraining run: {int(n)} optimizer steps
          </div>
          <Segmented
            label="Gradient-norm axis"
            value={gScale}
            onChange={setGScale}
            options={[
              { value: "linear", label: "grad norm: linear" },
              { value: "log", label: "grad norm: log" },
            ]}
          />
        </div>
        <div className="grid gap-2 sm:grid-cols-4">
          <Readout
            label="training loss, first 100 steps"
            value={fx(d.firstLoss, 3)}
            sub="mean"
          />
          <Readout
            label="training loss, last 200 steps"
            value={fx(d.lastLoss, 3)}
            sub="mean"
          />
          <Readout
            label="steps with grad norm > 1"
            value={`${fx(100 * d.clippedFrac, 1)}%`}
            sub="clipped to 1.0"
          />
          <Readout
            label="wall-clock time"
            value={minutes(wall)}
            sub={
              results?.config
                ? `${results.config.device ?? "cpu"}, ${results.config.threads ?? ""} threads`
                : undefined
            }
          />
        </div>
        <div className="mt-5 grid gap-6 lg:grid-cols-2">
          <TrainChart
            title="Training loss per step"
            subtitle={`thin: every step · bold: moving average over ${WIN} steps`}
            series={[
              {
                id: "raw",
                label: "every step",
                slot: 1,
                points: d.raw,
                raw: true,
              },
              {
                id: "ma",
                label: `moving average (${WIN})`,
                slot: 1,
                points: d.ma,
              },
            ]}
            xLabel="step"
            yLabel="loss"
            yDomain={[0, Math.max(0.9, ...run.loss) * 1.02]}
            formatX={fmtStep}
            formatY={(v) => fx(v, 2)}
            hoverX={hover}
            onHoverX={setHover}
            height={280}
            table={{
              columns: ["step", "loss", `moving average (${WIN})`, "grad norm"],
              rows: run.step
                .filter((s) => s % 50 === 0)
                .map((s) => [
                  int(s),
                  fx(run.loss[s], 4),
                  fx(d.ma[s].y, 4),
                  fx(run.gradNorm?.[s], 3),
                ]),
            }}
          />
          <TrainChart
            title="Gradient norm before clipping"
            subtitle="global L2 norm of all gradients"
            series={[
              { id: "g", label: "every step", slot: 2, points: d.g, raw: true },
              {
                id: "gma",
                label: `moving average (${WIN})`,
                slot: 2,
                points: d.gma,
              },
            ]}
            xLabel="step"
            yLabel="‖g‖"
            yScale={gScale}
            yDomain={
              gScale === "log"
                ? [0.1, 10]
                : [0, Math.max(...(run.gradNorm ?? [1])) * 1.05]
            }
            formatX={fmtStep}
            formatY={(v) => (v >= 1 ? fx(v, 0) : fx(v, 1))}
            refLines={[{ y: d.clip, label: `clip at max_norm = ${d.clip}` }]}
            hoverX={hover}
            onHoverX={setHover}
            height={280}
            table={{
              columns: ["step", "grad norm", "clipped?"],
              rows: run.step
                .filter((s) => s % 50 === 0)
                .map((s) => [
                  int(s),
                  fx(run.gradNorm?.[s], 3),
                  (run.gradNorm?.[s] ?? 0) > d.clip ? "yes" : "no",
                ]),
            }}
          />
        </div>
        <div className="mt-4 grid items-end gap-4 md:grid-cols-[minmax(0,1fr)_auto]">
          <Slider
            label="Scrub the run"
            value={i < 0 ? 0 : run.step[i]}
            min={0}
            max={run.step[n - 1]}
            onChange={(s) => setHover(s)}
            format={(s) => `step ${int(s)}`}
          />
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Readout label="loss" value={i < 0 ? "–" : fx(run.loss[i], 3)} />
            <Readout
              label="grad norm"
              value={i < 0 ? "–" : fx(run.gradNorm?.[i], 2)}
              sub={
                i >= 0 && (run.gradNorm?.[i] ?? 0) > d.clip
                  ? "clipped to 1"
                  : i >= 0
                    ? "not clipped"
                    : undefined
              }
            />
            <Readout label="η used" value={i < 0 ? "–" : sci(run.lr?.[i], 2)} />
            <Readout
              label="β1 used"
              value={i < 0 ? "–" : fx(run.beta1?.[i], 3)}
            />
          </div>
        </div>
      </div>
      <figcaption className="mx-auto mt-3 max-w-[var(--prose-max)] text-[16px] leading-snug text-ink-2">
        Every step is one batch of {results?.config?.batch ?? 32} contexts from
        all three pretraining databases, so the loss jumps with the batch’s mix
        of tasks; the moving average shows the trend. The norm is measured
        before clipping, as the code logs it{" "}
        <Cite kind="code" ref="rt/main.py:413-418" />{" "}
        <Cite kind="code" ref="rt/main.py:429-437" />.
      </figcaption>
    </figure>
  );
}
