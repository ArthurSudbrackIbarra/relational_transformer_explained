/**
 * Right panel: the ablation switches. Label, name, L and checkpoint switches are local to this page;
 * the attention-layer switch is the course-wide, session-only `disabledLayers` (store.ts).
 * Under each switch that has a counterpart in the paper, the paper's mean zero-shot number is shown
 * for the live task's metric (Table 4, p.9).
 */
import { RotateCcw, Shuffle } from "lucide-react";
import type { ReactNode } from "react";
import { useShallow } from "zustand/react/shallow";
import {
  Button,
  IconButton,
  LAYER_INFO,
  Segmented,
  Slider,
  Switch,
  cx,
} from "@/components";
import { ATTN_ORDER, type AttnLayer } from "@/engine/types";
import { DEFAULT_L } from "@/engine/model/config";
import { useSim } from "@/state/store";
import type { RemoveWhat } from "./ablate";
import { RadioList } from "./bits";
import { usePlayLive } from "./context";
import { paperArrow, TABLE4 } from "./paper";
import { usePlay } from "./state";
import {
  ckptOptions,
  fmtDelta,
  fmtDisplay,
  targetPhrase,
} from "./usePlayground";

export const L_RANGE = { min: 16, max: 256, step: 8 } as const;

function Group({
  title,
  children,
  note,
}: {
  title: ReactNode;
  children: ReactNode;
  note?: ReactNode;
}) {
  return (
    <section className="border-t border-grid pt-3.5 first:border-0 first:pt-0">
      <h3 className="!m-0 mb-2 text-[15.5px] font-semibold text-ink">
        {title}
      </h3>
      <div className="flex flex-col gap-2.5">{children}</div>
      {note && (
        <div className="mt-2 text-[14px] leading-snug text-muted">{note}</div>
      )}
    </section>
  );
}

function Count({ n, title }: { n: number; title: string }) {
  return (
    <span
      className="num inline-flex h-6 min-w-[28px] shrink-0 items-center justify-center rounded-md bg-surface-2 px-1.5 text-[14px] text-ink-2"
      title={title}
    >
      {n}
    </span>
  );
}

/** "Paper: AUROC 70.1 → 53.8" in muted text. */
function PaperLine({ children }: { children: ReactNode }) {
  return (
    <div className="-mt-1 pl-[38px] text-[14px] leading-snug text-muted">
      {children}
    </div>
  );
}

/** Compact live readout for narrow screens, where the prediction card is far away. */
function MiniReadout() {
  const { base, cur, task, active } = usePlayLive();
  const phrase = targetPhrase(task);
  return (
    <div
      className="rounded-lg bg-surface-2 px-3 py-2 text-[15px] text-ink-2 min-[75rem]:hidden"
      aria-live="polite"
    >
      {phrase.short}:{" "}
      <strong className="num text-ink">
        {cur ? fmtDisplay(cur.display, task) : "…"}
      </strong>
      {active && base && cur && (
        <span className="num ml-2">
          ({fmtDelta(cur.display - base.display, task)} vs baseline)
        </span>
      )}
    </div>
  );
}

export function AblationPanel() {
  const live = usePlayLive();
  const { task, abl, globalL } = live;
  const p = usePlay(
    useShallow((s) => ({
      removeSelf: s.removeSelf,
      removeOther: s.removeOther,
      removeWhat: s.removeWhat,
      shuffle: s.shuffle,
      shuffleSeed: s.shuffleSeed,
      L: s.L,
      ckpt: s.ckpt,
      set: s.set,
      reshuffle: s.reshuffle,
      clearEdits: s.clearEdits,
      resetLocal: s.resetLocal,
    })),
  );
  const disabled = useSim(useShallow((s) => s.disabledLayers));
  const toggleLayer = useSim((s) => s.toggleLayer);
  const setDisabledLayers = useSim((s) => s.setDisabledLayers);
  const forecast = task.kind === "forecast";
  const binary = task.targetType === "binary";
  const nSelf = abl.labels.self.length;
  const nOther = abl.labels.other.length;
  const nEdits = abl.editedPos.size;
  const opts = ckptOptions(task, live.baseCkpt);
  const nCols = abl.perm?.size ?? 0;

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <div className="flex items-center justify-between gap-2">
        <div className="eyebrow">Ablations</div>
        <Button
          size="sm"
          variant="ghost"
          icon={<RotateCcw size={14} aria-hidden="true" />}
          disabled={!live.active}
          onClick={() => {
            p.resetLocal();
            setDisabledLayers([]);
          }}
        >
          Reset all
        </Button>
      </div>
      <MiniReadout />

      <Group
        title="Remove in-context labels"
        note={
          forecast
            ? nEdits > 0
              ? undefined
              : "Tip: click a label in the context to flip or edit it."
            : "Autocomplete has no task rows, so the self / other split of Table 4 does not apply. You can still edit the other rows’ values."
        }
      >
        <div className="flex items-center justify-between gap-2">
          <Switch
            checked={forecast && p.removeSelf}
            disabled={!forecast}
            onChange={(v) => p.set({ removeSelf: v })}
            label="Self labels"
            className="max-sm:min-h-11"
          />
          <Count
            n={forecast ? nSelf : 0}
            title="Self labels in the live context"
          />
        </div>
        <PaperLine>
          Paper: {paperArrow(TABLE4.none, TABLE4.selfLabels, binary)}
        </PaperLine>
        <div className="flex items-center justify-between gap-2">
          <Switch
            checked={forecast && p.removeOther}
            disabled={!forecast}
            onChange={(v) => p.set({ removeOther: v })}
            label="Other-entity labels"
            className="max-sm:min-h-11"
          />
          <Count
            n={nOther}
            title={
              forecast
                ? "Other-entity labels in the live context"
                : "Visible target cells of other rows"
            }
          />
        </div>
        <PaperLine>
          Paper: {paperArrow(TABLE4.none, TABLE4.otherLabels, binary)}
        </PaperLine>
        {forecast && (
          <div className="flex flex-col gap-1">
            <span className="text-[15px] text-ink-2">What is removed</span>
            <Segmented<RemoveWhat>
              label="What is removed"
              value={p.removeWhat}
              onChange={(v) => p.set({ removeWhat: v })}
              disabled={!p.removeSelf && !p.removeOther}
              options={[
                {
                  value: "cells",
                  label: "label cells",
                  title: "Only the target-column cells (Table 11/12 wording)",
                },
                {
                  value: "rows",
                  label: "whole rows",
                  title: "Every cell of those task rows (§5.1 wording)",
                },
              ]}
            />
          </div>
        )}
        {nEdits > 0 && (
          <div className="flex items-center justify-between gap-2 rounded-lg border border-accent/40 bg-accent-soft px-2.5 py-1.5 text-[15px] text-ink">
            <span>
              {nEdits} label{nEdits === 1 ? "" : "s"} edited
            </span>
            <button
              type="button"
              className="text-[15px] font-medium text-accent-text hover:underline"
              onClick={p.clearEdits}
            >
              Undo edits
            </button>
          </div>
        )}
      </Group>

      <Group
        title="Schema names"
        note={
          p.shuffle && nCols > 0
            ? `${nCols} column phrases permuted; the masks still use the true columns.`
            : undefined
        }
      >
        <div className="flex items-center justify-between gap-2">
          <Switch
            checked={p.shuffle}
            onChange={(v) => p.set({ shuffle: v })}
            label="Shuffle column names"
            className="max-sm:min-h-11"
          />
          <IconButton
            label="Draw another shuffle"
            title="Draw another shuffle"
            size="sm"
            onClick={p.reshuffle}
            className="border border-border-strong"
          >
            <Shuffle size={15} />
          </IconButton>
        </div>
        <PaperLine>
          Paper: {paperArrow(TABLE4.none, TABLE4.colNames, binary)}
        </PaperLine>
      </Group>

      <Group
        title="Context length"
        note={
          <>
            Trained at L = {DEFAULT_L}; the paper at 1024.{" "}
            {p.L !== null && (
              <button
                type="button"
                className="font-medium text-accent-text hover:underline"
                onClick={() => p.set({ L: null })}
              >
                Back to {globalL}
              </button>
            )}
          </>
        }
      >
        <Slider
          label="L (cells)"
          value={p.L ?? globalL}
          min={L_RANGE.min}
          max={L_RANGE.max}
          step={L_RANGE.step}
          onChange={(v) => p.set({ L: v === globalL ? null : v })}
          format={(v) => (v === globalL ? `${v} (course)` : String(v))}
        />
      </Group>

      <Group
        title="Attention layers"
        note={
          <>
            Pressed = switched off, in every block. The paper{" "}
            <strong className="font-semibold text-ink-2">retrains</strong>{" "}
            without the layer instead. This switch is shared with the other
            chapters for this session.
          </>
        }
      >
        <div
          className="grid grid-cols-2 gap-1.5"
          role="group"
          aria-label="Attention layers (pressed = switched off)"
        >
          {ATTN_ORDER.map((l: AttnLayer) => {
            const off = disabled.includes(l);
            const Icon = LAYER_INFO[l].icon;
            return (
              <button
                key={l}
                type="button"
                aria-pressed={off}
                title={
                  off
                    ? `Switch ${LAYER_INFO[l].name.toLowerCase()} attention back on`
                    : `Switch ${LAYER_INFO[l].name.toLowerCase()} attention off`
                }
                onClick={() => toggleLayer(l)}
                className={cx(
                  "inline-flex h-9 items-center justify-center gap-1.5 rounded-lg border px-2 text-[15px] font-medium transition-colors",
                  off
                    ? "border-dashed border-border-strong bg-surface-2 text-muted line-through"
                    : "border-border-strong bg-surface text-ink hover:bg-surface-2",
                )}
              >
                <Icon
                  aria-hidden="true"
                  size={15}
                  className={off ? "text-muted" : "text-ink-2"}
                />
                {LAYER_INFO[l].name}
                <span className="sr-only">{off ? " (off)" : " (on)"}</span>
              </button>
            );
          })}
        </div>
      </Group>

      <Group title="Checkpoint">
        <RadioList
          label="Checkpoint"
          value={live.ablCkpt}
          onChange={(name) =>
            p.set({ ckpt: name === live.baseCkpt ? null : name })
          }
          items={opts.map((o) => ({
            value: o.name,
            title: o.hint,
            label: (
              <>
                {o.label}
                {o.name === live.baseCkpt && (
                  <span className="ml-1.5 text-[14px] font-normal text-muted">
                    baseline
                  </span>
                )}
              </>
            ),
            sub: o.short,
          }))}
        />
        {live.ablAssets.status === "loading" && (
          <p className="!m-0 text-[14.5px] text-muted">Loading weights…</p>
        )}
      </Group>
    </div>
  );
}
