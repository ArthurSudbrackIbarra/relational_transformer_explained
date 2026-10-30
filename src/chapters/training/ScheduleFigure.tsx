/**
 * The OneCycle learning-rate and β1 schedules (main.py:197-204, spec §9.2, §10 #13) as two separate
 * charts that share one crosshair (never a dual axis). "Our run" = the lr / β1 logged for every update
 * of our pretraining run; "Paper" = the same scheduler with the paper's 50,001 steps (schedule.json).
 * The closed form of torch's linear OneCycleLR is checked against the logged values live.
 */
import { useMemo, useState } from "react";
import { Cite, MathBlock, Segmented, Slider } from "@/components";
import { Readout, TrainChart } from "./charts";
import { fx, int, sci, type Schedule, type ScheduleSeries } from "./data";
import { ONECYCLE } from "./paper";

/** torch.optim.lr_scheduler.OneCycleLR, anneal_strategy='linear', three_phase=False (torch 2.6). */
export function oneCycle(step: number, total: number) {
  const { maxLr, pctStart, divFactor, finalDivFactor, beta1Max, beta1Min } =
    ONECYCLE;
  const lr0 = maxLr / divFactor;
  const lrEnd = lr0 / finalDivFactor;
  const s1 = pctStart * total - 1;
  const s2 = total - 1;
  if (step <= s1) {
    const p = step / s1;
    return {
      lr: lr0 + (maxLr - lr0) * p,
      beta1: beta1Max + (beta1Min - beta1Max) * p,
    };
  }
  const p = (step - s1) / (s2 - s1);
  return {
    lr: maxLr + (lrEnd - maxLr) * p,
    beta1: beta1Min + (beta1Max - beta1Min) * p,
  };
}

const LR_TEX = String.raw`\eta(s)=\begin{cases}\eta_0+(\eta_{\max}-\eta_0)\,\dfrac{s}{s^\ast} & s\le s^\ast\\[8pt]\eta_{\max}+(\eta_{\text{end}}-\eta_{\max})\,\dfrac{s-s^\ast}{T-1-s^\ast} & s>s^\ast\end{cases}`;
const DEF_TEX = String.raw`s^\ast=0.2\,T-1,\quad \eta_0=\eta_{\max}/25,\quad \eta_{\text{end}}=\eta_0/10^{4}`;
// phone versions: the same two formulas, one branch per aligned block so nothing needs sideways scrolling
const LR_TEX_PHONE = String.raw`\begin{aligned}\eta(s)&=\eta_0+(\eta_{\max}-\eta_0)\,\dfrac{s}{s^\ast}\\&\qquad\text{for } s\le s^\ast\\[8pt]\eta(s)&=\eta_{\max}+(\eta_{\text{end}}-\eta_{\max})\\&\quad\cdot\dfrac{s-s^\ast}{T-1-s^\ast}\\&\qquad\text{for } s>s^\ast\end{aligned}`;
const B1_TEX_PHONE = String.raw`\begin{aligned}\beta_1(s)&=0.95-0.10\,\dfrac{s}{s^\ast}\\&\qquad\text{for } s\le s^\ast\\[8pt]\beta_1(s)&=0.85+0.10\,\dfrac{s-s^\ast}{T-1-s^\ast}\\&\qquad\text{for } s>s^\ast\end{aligned}`;
const DEF_TEX_A = String.raw`s^\ast=0.2\,T-1,\quad \eta_0=\eta_{\max}/25`;
const DEF_TEX_B = String.raw`\eta_{\text{end}}=\eta_0/10^{4}`;
const B1_TEX = String.raw`\beta_1(s)=\begin{cases}0.95-0.10\,\dfrac{s}{s^\ast} & s\le s^\ast\\[8pt]0.85+0.10\,\dfrac{s-s^\ast}{T-1-s^\ast} & s>s^\ast\end{cases}`;

export function ScheduleFigure({ schedule }: { schedule: Schedule | null }) {
  const [which, setWhich] = useState<"mini" | "paper">("mini");
  const [yLog, setYLog] = useState<"linear" | "log">("linear");
  const [hover, setHover] = useState<number | null>(null);
  const ser: ScheduleSeries | undefined = schedule?.[which];

  const data = useMemo(() => {
    if (!ser) return null;
    const lr = ser.step.map((s, i) => ({ x: s, y: ser.lr[i] }));
    const b1 = ser.step.map((s, i) => ({ x: s, y: ser.beta1[i] }));
    let peakI = 0;
    for (let i = 1; i < ser.lr.length; i++)
      if (ser.lr[i] > ser.lr[peakI]) peakI = i;
    let maxErr = 0;
    for (let i = 0; i < ser.step.length; i++) {
      const f = oneCycle(ser.step[i], ser.totalSteps);
      maxErr = Math.max(
        maxErr,
        Math.abs(f.lr - ser.lr[i]) / ONECYCLE.maxLr,
        Math.abs(f.beta1 - ser.beta1[i]),
      );
    }
    return { lr, b1, peakI, maxErr };
  }, [ser]);

  if (!ser || !data)
    return (
      <div className="wide my-8 card px-5 py-6 text-[16px] text-ink-2">
        Loading schedule.json…
      </div>
    );

  const T = ser.totalSteps;
  const last = ser.step.length - 1;
  const at = hover ?? ser.step[data.peakI];
  const idx = Math.max(
    0,
    ser.step.findIndex((s) => s >= at),
  );
  const fmtStep = (s: number) =>
    s >= 10000 ? `${Math.round(s / 1000)}k` : int(s);
  const start = { x: ser.step[0], y: ser.lr[0] };
  const peak = { x: ser.step[data.peakI], y: ser.lr[data.peakI] };
  const end = { x: ser.step[last], y: ser.lr[last] };

  return (
    <figure className="wide my-8 min-w-0">
      <div className="card px-4 py-4 sm:px-5 sm:py-5">
        <div className="mb-3 flex flex-wrap items-end gap-x-4 gap-y-3">
          <div className="mr-auto text-[17px] font-semibold text-ink">
            The OneCycle schedule, step by step
          </div>
          <Segmented
            label="Which run"
            value={which}
            onChange={(v) => {
              setWhich(v);
              setHover(null);
            }}
            options={[
              {
                value: "mini",
                label: `Our run (${int(schedule?.mini?.totalSteps)} steps)`,
              },
              {
                value: "paper",
                label: `Paper (${int(schedule?.paper?.totalSteps)} steps)`,
              },
            ]}
          />
          <Segmented
            label="Learning-rate axis"
            value={yLog}
            onChange={setYLog}
            options={[
              { value: "linear", label: "linear" },
              { value: "log", label: "log" },
            ]}
          />
        </div>
        <div className="grid gap-5 lg:grid-cols-2">
          <TrainChart
            title="Learning rate η"
            subtitle={
              yLog === "log"
                ? "log scale: the end value 4·10⁻⁹ becomes visible"
                : "linear scale"
            }
            series={[
              { id: "lr", label: "learning rate", slot: 1, points: data.lr },
            ]}
            xLabel="step"
            yScale={yLog}
            yDomain={yLog === "log" ? [1e-9, 2e-3] : [0, 1.08e-3]}
            formatX={fmtStep}
            formatY={(v) => (v === 0 ? "0" : sci(v, 1))}
            markers={[{ x: peak.x, label: "" }]}
            highlights={[
              { ...start, slot: 1, label: `start ${sci(start.y, 1)}` },
              { ...peak, slot: 1, label: `peak ${sci(peak.y, 1)} at 20%` },
              ...(yLog === "log"
                ? [{ ...end, slot: 1, label: `end ${sci(end.y, 1)}` }]
                : []),
            ]}
            hoverX={hover}
            onHoverX={setHover}
            height={270}
          />
          <TrainChart
            title="AdamW β1 (momentum)"
            subtitle="cycled opposite to the learning rate"
            series={[{ id: "b1", label: "β1", slot: 2, points: data.b1 }]}
            xLabel="step"
            yDomain={[0.84, 0.96]}
            yTicks={[0.85, 0.875, 0.9, 0.925, 0.95]}
            formatX={fmtStep}
            formatY={(v) => fx(v, 3)}
            markers={[{ x: peak.x, label: "20%: lowest β1" }]}
            refLines={[
              {
                y: 0.9,
                label: "coded betas[0] = 0.9, overridden",
                below: true,
              },
            ]}
            hoverX={hover}
            onHoverX={setHover}
            height={270}
          />
        </div>
        <div className="mt-4 grid items-end gap-4 md:grid-cols-[minmax(0,1fr)_auto]">
          <Slider
            label="Training progress"
            value={Math.round((100 * at) / Math.max(1, T - 1))}
            min={0}
            max={100}
            onChange={(p) => setHover(Math.round((p / 100) * (T - 1)))}
            format={(p) => `${p}%`}
          />
          <div className="grid grid-cols-3 gap-2">
            <Readout
              label="step"
              value={int(ser.step[idx])}
              sub={`of ${int(T)}`}
            />
            <Readout label="η" value={sci(ser.lr[idx], 2)} />
            <Readout label="β1" value={fx(ser.beta1[idx], 4)} />
          </div>
        </div>
        <div className="relative mt-5 grid grid-cols-[minmax(0,1fr)] gap-1 overflow-x-auto rounded-lg bg-surface-2 px-3 py-1 text-[18px]">
          <MathBlock tex={LR_TEX_PHONE} className="my-3 min-w-0 sm:hidden" />
          <MathBlock tex={LR_TEX} className="my-3 hidden min-w-0 sm:block" />
          <MathBlock tex={DEF_TEX_A} className="my-1 min-w-0 sm:hidden" />
          <MathBlock tex={DEF_TEX_B} className="my-1 min-w-0 sm:hidden" />
          <MathBlock tex={DEF_TEX} className="my-1 hidden min-w-0 sm:block" />
          <MathBlock tex={B1_TEX_PHONE} className="my-3 min-w-0 sm:hidden" />
          <MathBlock tex={B1_TEX} className="my-3 hidden min-w-0 sm:block" />
        </div>
        <p className="mt-2 text-[14.5px] leading-snug text-muted">
          PyTorch’s OneCycleLR with linear annealing and the default settings
          the code leaves unchanged{" "}
          <Cite
            kind="url"
            ref="https://pytorch.org/docs/2.6/generated/torch.optim.lr_scheduler.OneCycleLR.html"
            title="PyTorch 2.6: OneCycleLR"
          />
          . These formulas reproduce every value of{" "}
          {which === "mini"
            ? "η and β1 logged by our run"
            : "the sampled 50,001-step schedule"}
          {data.maxErr < 1e-6 ? "" : ` to within ${sci(data.maxErr, 1)}`}.
        </p>
      </div>
      <figcaption className="mx-auto mt-3 max-w-[var(--prose-max)] text-[16px] leading-snug text-ink-2">
        Both runs use the same scheduler with T = total steps, so the curves
        have the same shape and only the step axis stretches. The paper’s
        curve is sampled every{" "}
        {int((schedule?.paper?.step[1] ?? 0) - (schedule?.paper?.step[0] ?? 0))}{" "}
        steps; ours is logged at every update <Cite kind="spec" ref="§9.2" />.
      </figcaption>
    </figure>
  );
}
