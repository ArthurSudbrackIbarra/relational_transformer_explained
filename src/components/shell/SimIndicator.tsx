/**
 * Top-bar "simulation" indicator (BUILD_PLAN §5, §7): a compact chip showing
 * the live DB / task / seed / L / w, opening a popover to change them. Reads
 * and writes the global store, so every chapter follows.
 */
import { FlaskConical, RotateCcw, SlidersHorizontal } from 'lucide-react';
import { lazy, Suspense } from 'react';
import { useShallow } from 'zustand/react/shallow';
import type { AttnLayer } from '@/engine/types';
import { ATTN_ORDER } from '@/engine/types';
import { CKPT_CHOICES, isNonDefault, selectSimParams, SIM_LIMITS, useSim, type CkptChoice } from '@/state/store';
import { Button, NumberField, Select, Slider } from '../ui/controls';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { Popover } from '../ui/floating';
import { cx } from '../ui/cx';
import { LAYER_INFO } from '../viz/layers';

const EngineFields = lazy(() => import('./SimEngine').then((m) => ({ default: m.EngineFields })));
const SeedLabel = lazy(() => import('./SimEngine').then((m) => ({ default: m.SeedLabel })));

const CKPT_LABEL: Record<CkptChoice, string> = {
  auto: 'auto (zero-shot)',
  init: 'init (untrained)',
  'pretrain-churn': 'pretrain-churn',
  'pretrain-sales': 'pretrain-sales',
  'finetune-churn': 'finetune-churn',
  'finetune-sales': 'finetune-sales',
};

function EngineFallback({ text }: { text: string }) {
  return <div className="rounded-lg bg-surface-2 px-3 py-2.5 text-[15px] text-ink-2">{text}</div>;
}

function SimPanel({ close }: { close: () => void }) {
  const s = useSim(
    useShallow((st) => ({
      L: st.L,
      w: st.w,
      rngSeed: st.rngSeed,
      ckpt: st.ckpt,
      disabledLayers: st.disabledLayers,
    })),
  );
  const a = useSim(
    useShallow((st) => ({
      setL: st.setL,
      setW: st.setW,
      setRngSeed: st.setRngSeed,
      setCkpt: st.setCkpt,
      toggleLayer: st.toggleLayer,
      setDisabledLayers: st.setDisabledLayers,
      resetSimulation: st.resetSimulation,
    })),
  );
  return (
    <div className="p-4">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <div className="text-[17px] font-semibold text-ink">Simulation</div>
          <p className="mt-0.5 text-[15px] leading-snug text-ink-2">Every chapter reads these settings, so the same context flows through the course.</p>
        </div>
      </div>

      <ErrorBoundary fallback={<EngineFallback text="Database and seed pickers are unavailable (engine failed to load)." />}>
        <Suspense fallback={<EngineFallback text="Loading the databases…" />}>
          <EngineFields />
        </Suspense>
      </ErrorBoundary>

      <div className="mt-3 grid gap-3">
        <div className="grid grid-cols-2 gap-3">
          <Slider label="Context length L" value={s.L} min={SIM_LIMITS.L.min} max={SIM_LIMITS.L.max} step={SIM_LIMITS.L.step} onChange={a.setL} hint="Paper: 1024" />
          <Slider label="BFS width w" value={s.w} min={SIM_LIMITS.w.min} max={SIM_LIMITS.w.max} onChange={a.setW} hint="Paper: 256" />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <NumberField label="RNG seed" value={s.rngSeed} min={SIM_LIMITS.rngSeed.min} max={SIM_LIMITS.rngSeed.max} onChange={a.setRngSeed} hint="fly.rs: epoch + seed row + seed" />
          <Select
            label="Checkpoint"
            value={s.ckpt}
            onChange={(e) => a.setCkpt(e.target.value as CkptChoice)}
            hint={s.ckpt === 'auto' ? 'Best val checkpoint per task (spec §9.1)' : s.ckpt.startsWith('finetune') ? 'Trained on ShopDB: not zero-shot' : undefined}
          >
            {CKPT_CHOICES.map((c) => (
              <option key={c} value={c}>
                {CKPT_LABEL[c]}
              </option>
            ))}
          </Select>
        </div>
        {s.disabledLayers.length > 0 && (
          <div className="rounded-lg border border-dashed border-border-strong px-3 py-2.5">
            <div className="flex items-center gap-1.5 text-[15px] font-semibold text-ink">
              <FlaskConical aria-hidden="true" size={14} className="text-ink-2" /> Inference-time ablation active
            </div>
            <p className="mt-0.5 text-[14.5px] leading-snug text-ink-2">
              These layers are skipped in every block. The paper’s ablations retrain the model instead (spec §11.4).
            </p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {ATTN_ORDER.map((l: AttnLayer) => (
                <button
                  key={l}
                  type="button"
                  aria-pressed={s.disabledLayers.includes(l)}
                  onClick={() => a.toggleLayer(l)}
                  className={cx(
                    'rounded-md border px-2 py-0.5 text-[15px]',
                    s.disabledLayers.includes(l) ? 'border-accent bg-accent-soft text-accent-text line-through' : 'border-border text-ink-2',
                  )}
                >
                  {LAYER_INFO[l].name}
                </button>
              ))}
              <button type="button" onClick={() => a.setDisabledLayers([])} className="ml-auto text-[15px] font-medium text-accent-text hover:underline">
                Clear
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="mt-4 flex items-center justify-between gap-2 border-t border-grid pt-3">
        <Button variant="ghost" size="sm" icon={<RotateCcw size={14} />} onClick={a.resetSimulation}>
          Reset to defaults
        </Button>
        <Button variant="secondary" size="sm" onClick={close}>
          Done
        </Button>
      </div>
    </div>
  );
}

/** Top-bar chip + popover. */
export function SimIndicator({ className }: { className?: string }) {
  const p = useSim(useShallow(selectSimParams));
  const ablated = useSim((s) => s.disabledLayers.length > 0);
  const task = p.taskId.split('/')[1] ?? p.taskId;
  const changed = isNonDefault(p);
  return (
    <Popover
      label="Simulation settings"
      width={400}
      align="end"
      trigger={(t, open) => (
        <button
          type="button"
          {...t}
          ref={t.ref}
          className={cx(
            'group inline-flex h-9 min-w-0 items-center gap-2 rounded-lg border px-2.5 text-[15px] transition-colors',
            open ? 'border-accent bg-accent-soft' : 'border-border bg-surface hover:bg-surface-2',
            className,
          )}
          title="Simulation settings"
        >
          <SlidersHorizontal aria-hidden="true" size={15} className="shrink-0 text-ink-2" />
          <span className="sr-only">Simulation: </span>
          <span className="hidden min-w-0 items-center gap-1.5 truncate lg:inline-flex">
            <span className="font-mono text-ink">{p.dbName}</span>
            <span className="text-muted">·</span>
            <span className="truncate font-mono text-ink">{task}</span>
            <span className="text-muted">·</span>
            <span className="num text-ink-2">
              seed{' '}
              <ErrorBoundary fallback={<>auto</>}>
                <Suspense fallback={<>auto</>}>
                  <SeedLabel />
                </Suspense>
              </ErrorBoundary>
            </span>
            <span className="text-muted">·</span>
            <span className="num text-ink-2">L {p.L}</span>
            <span className="num text-ink-2">w {p.w}</span>
          </span>
          <span className="font-medium text-ink lg:hidden">Sim</span>
          {(changed || ablated) && (
            <span
              aria-label={ablated ? 'ablation active' : 'non-default settings'}
              className={cx('h-1.5 w-1.5 shrink-0 rounded-full', ablated ? 'bg-[var(--warning)]' : 'bg-accent')}
            />
          )}
        </button>
      )}
    >
      {(close) => <SimPanel close={close} />}
    </Popover>
  );
}
