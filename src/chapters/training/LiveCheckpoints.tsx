/**
 * "One context, three sets of weights": the live context of the running example run through the
 * untrained init, the zero-shot checkpoint and the fine-tuned checkpoint of its task. Same tokens,
 * same masks; only the weights differ. Uses the store's seed when the global task is a ShopDB
 * forecast task, otherwise customer-churn's default seed (local, the global state is untouched).
 */
import { useEffect, useMemo, useState } from "react";
import { Cite, Segmented } from "@/components";
import {
  useContextWindow,
  usePreprocessed,
  useSeedNode,
  useTask,
  loadAssets,
} from "@/state/hooks";
import { useSim } from "@/state/store";
import { getDatabase } from "@/engine/db/registry";
import {
  getMasks,
  getContext,
  getPreprocessed,
  makeSampleParams,
  pickDefaultSeed,
  findTask,
  readPrediction,
  CHECKPOINT_INFO,
  type CheckpointName,
  type PredictionReadout,
} from "@/engine/pipeline";
import { forwardWithMasks, textEmbedder } from "@/engine/model";
import { getSeeds } from "@/engine/pipeline";
import type { PreprocessedDB, SampledContext, TaskDef } from "@/engine/types";
import { catVar } from "@/styles/palette";
import { fx, int, SHOP_FORECAST, type Results } from "./data";

interface Lane {
  ckpt: CheckpointName;
  title: string;
  pred: PredictionReadout | null;
  fallback: boolean;
}

const fcache = new Map<
  string,
  { pred: PredictionReadout | null; fallback: boolean }
>();

export function LiveCheckpoints({ results }: { results: Results | null }) {
  const globalTask = useTask();
  const globalCtx = useContextWindow();
  const globalPdb = usePreprocessed();
  const globalSeed = useSeedNode();
  const split = useSim((s) => s.split);
  const globalIsShopForecast = SHOP_FORECAST.some(
    (t) => t.id === globalTask.id,
  );
  const [taskId, setTaskId] = useState(
    globalIsShopForecast ? globalTask.id : SHOP_FORECAST[0].id,
  );
  const meta = SHOP_FORECAST.find((t) => t.id === taskId)!;

  const { ctx, pdb, task, seed, isGlobal } = useMemo(() => {
    if (taskId === globalTask.id)
      return {
        ctx: globalCtx,
        pdb: globalPdb,
        task: globalTask,
        seed: globalSeed,
        isGlobal: true,
      };
    const db = getDatabase("shop");
    const p = getPreprocessed(db);
    const t = findTask(db, taskId);
    const seeds = getSeeds(p, t, "Val");
    const s = pickDefaultSeed(p, db, t, "Val", seeds);
    return {
      ctx: getContext(p, makeSampleParams(p, t, s, 128, 16, 0)),
      pdb: p,
      task: t,
      seed: s,
      isGlobal: false,
    };
  }, [taskId, globalTask, globalCtx, globalPdb, globalSeed]);

  const ckpts: { ckpt: CheckpointName; title: string }[] = [
    { ckpt: "init", title: "Untrained" },
    { ckpt: `pretrain-${meta.ft}` as CheckpointName, title: "Zero-shot" },
    { ckpt: `finetune-${meta.ft}` as CheckpointName, title: "Fine-tuned" },
  ];
  const lanes = useLanes(ckpts, ctx, pdb, task);
  const entity = useMemo(() => {
    const n = pdb.nodes[seed];
    const parent =
      n?.f2pNbrIdxs[0] !== undefined ? pdb.nodes[n.f2pNbrIdxs[0]] : undefined;
    const name = parent?.cells.find((c) => c.colName === "name")?.raw;
    return parent
      ? `${parent.tableName.replace(/s$/, "")} ${parent.rowPos}${name ? ` (${String(name)})` : ""}`
      : `node ${seed}`;
  }, [pdb, seed]);

  return (
    <figure className="wide my-8 min-w-0">
      <div className="card px-4 py-4 sm:px-5 sm:py-5">
        <div className="mb-3 flex flex-wrap items-end gap-3">
          <div className="mr-auto min-w-0">
            <div className="text-[17px] font-semibold text-ink">
              One context, three sets of weights
            </div>
            <div className="text-[15px] text-ink-2">
              {isGlobal
                ? "The live context of the running example"
                : "The default context of this task"}
              :{" "}
              <span className="text-ink">
                {task.id.split("/")[1]}
              </span>{" "}
              for {entity}, {isGlobal ? split.toLowerCase() : "val"} row,{" "}
              {int(ctx.numReal)} tokens
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
        <div className="grid gap-3 md:grid-cols-3 md:grid-rows-[auto_auto_1fr]">
          {lanes.map((l) => (
            <LaneCard key={l.ckpt} lane={l} results={results} />
          ))}
        </div>
      </div>
      <figcaption className="mx-auto mt-3 max-w-[var(--prose-max)] text-[16px] leading-snug text-ink-2">
        The tokens and masks are identical in the three cards; the decoder
        output at the masked cell changes only because the weights do. One
        context is an anecdote: the metrics above average over every seed of the
        split.
      </figcaption>
    </figure>
  );
}

function useLanes(
  ckpts: { ckpt: CheckpointName; title: string }[],
  ctx: SampledContext,
  pdb: PreprocessedDB,
  task: TaskDef,
): Lane[] {
  const [, force] = useState(0);
  const keys = ckpts.map(
    (c) =>
      `${c.ckpt}|${ctx.params.seedNode}|${ctx.params.L}|${ctx.params.w}|${ctx.params.rngSeed}|${task.id}`,
  );
  useEffect(() => {
    let alive = true;
    (async () => {
      for (let i = 0; i < ckpts.length; i++) {
        if (fcache.has(keys[i])) continue;
        const a = await loadAssets(ckpts[i].ckpt);
        if (!alive) return;
        if (a.status !== "ready" || !a.store || !a.weights) continue;
        await new Promise((r) => window.setTimeout(r, 0));
        if (!alive) return;
        const fr = forwardWithMasks(
          ctx.tokens,
          getMasks(ctx),
          textEmbedder(pdb, a.store),
          a.weights,
          { capture: false },
        );
        fcache.set(keys[i], {
          pred: readPrediction(fr, ctx, pdb, task),
          fallback: a.fallback,
        });
        force((n) => n + 1);
      }
    })();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keys.join("#")]);
  return ckpts.map((c, i) => ({
    ...c,
    pred: fcache.get(keys[i])?.pred ?? null,
    fallback: fcache.get(keys[i])?.fallback ?? false,
  }));
}

function LaneCard({ lane, results }: { lane: Lane; results: Results | null }) {
  const info = results?.checkpoints?.[lane.ckpt] as
    { step?: number; selection?: string } | undefined;
  const p = lane.pred;
  return (
    <div
      className={`grid min-w-0 content-start rounded-lg border px-3.5 py-3 md:row-span-3 md:grid-rows-subgrid ${lane.title === "Zero-shot" ? "border-accent" : "border-border"}`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-[16px] font-semibold text-ink">{lane.title}</div>
        <div className="font-mono text-[14px] text-ink-2">{lane.ckpt}</div>
      </div>
      <div>
        <div
          className="text-[14.5px] leading-snug text-ink-2"
          title={CHECKPOINT_INFO[lane.ckpt]}
        >
          {lane.ckpt === "init"
            ? "step 0 of pretraining, no update yet"
            : info?.step !== undefined
              ? `${info.selection ?? ""}, step ${int(info.step)}`
              : CHECKPOINT_INFO[lane.ckpt]}
        </div>
        {lane.fallback && (
          <div className="mt-1 text-[14px] text-[var(--warning-text)]">
            file missing: running untrained fallback weights
          </div>
        )}
      </div>
      {!p ? (
        <div className="mt-3 text-[15px] text-ink-2">
          Running the forward pass…
        </div>
      ) : p.taskKind === "binary" ? (
        <BinaryReadout p={p} />
      ) : (
        <RegressionReadout p={p} />
      )}
    </div>
  );
}

function BinaryReadout({ p }: { p: PredictionReadout }) {
  const prob = p.probability ?? 0.5;
  const y = p.label.binary;
  return (
    <div className="mt-3">
      <div className="flex items-baseline gap-2">
        <span className="num text-[28px] leading-none font-semibold text-ink">
          {fx(prob, 3)}
        </span>
        <span className="text-[15px] text-ink-2">P(churn)</span>
      </div>
      <div
        className="relative mt-3 h-3 rounded-full bg-surface-3"
        role="img"
        aria-label={`Probability ${fx(prob, 3)}; label ${y ? "true" : "false"}`}
      >
        <div
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ width: `${prob * 100}%`, background: catVar(1) }}
        />
        <div
          className="absolute -top-1 h-5 w-[2px] bg-ink"
          style={{ left: `calc(${(y ?? 0) * 100}% - ${y ? 2 : 0}px)` }}
        />
      </div>
      <div className="mt-1 flex justify-between text-[14px] text-muted">
        <span>0</span>
        <span>1</span>
      </div>
      <div className="mt-1 text-[15px] text-ink-2">
        logit <span className="num text-ink">{fx(p.output, 3)}</span> · hidden
        label <span className="num text-ink">{String(p.label.raw)}</span>
      </div>
    </div>
  );
}

function RegressionReadout({ p }: { p: PredictionReadout }) {
  return (
    <div className="mt-3">
      <div className="flex items-baseline gap-2">
        <span className="num text-[28px] leading-none font-semibold text-ink">
          {fx(p.value, 2)}
        </span>
        <span className="text-[15px] text-ink-2">units sold, predicted</span>
      </div>
      <div className="mt-2 text-[15px] text-ink-2">
        z = <span className="num text-ink">{fx(p.output, 3)}</span> · hidden
        label <span className="num text-ink">{String(p.label.raw)}</span> (z ={" "}
        <span className="num text-ink">{fx(p.label.z, 3)}</span>)
      </div>
      <div className="mt-1 text-[15px] text-ink-2">
        error{" "}
        <span className="num text-ink">
          {fx((p.value ?? 0) - Number(p.label.raw), 2)}
        </span>{" "}
        units <Cite kind="spec" ref="§8.4" />
      </div>
    </div>
  );
}
