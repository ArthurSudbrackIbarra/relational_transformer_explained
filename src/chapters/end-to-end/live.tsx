/**
 * One live state for the whole "start to end" page, computed once and shared by every stage figure:
 * the global simulation (src/state/hooks.ts → src/engine/pipeline.ts), the forward pass with full
 * capture (model/forward.ts, a mirror of rt/model.py:153-267), and the token in focus.
 *
 * The focus token is the store's `selectedPos` (null = the masked target), the same "inspected
 * token" the masks and attention chapters use, so a token pinned there is the token followed here.
 * Nothing in this file computes model semantics; it only gathers and labels engine output.
 */
import { createContext, useCallback, useContext, useMemo, type ReactNode } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { MINI_CONFIG } from '@/engine/model/config';
import { tensor } from '@/engine/model/weights';
import type { EmbeddingStore } from '@/engine/embeddings';
import type { RTWeights } from '@/engine/types';
import {
  useAssets,
  useContextWindow,
  useDatabase,
  useForward,
  useInContextLabels,
  useMasks,
  usePrediction,
  usePreprocessed,
  useSeedNode,
  useTableSlots,
  useTargetStats,
  useTask,
} from '@/state/hooks';
import { useSim } from '@/state/store';
import { listDatabases } from '@/engine/db/registry';
import { tablePrefixes, tokenLabel } from '../attention/utils';
import type { TokenSpan } from '../shapes';

function useE2EState() {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const task = useTask();
  const seedNode = useSeedNode();
  const ctx = useContextWindow();
  const masks = useMasks();
  const slots = useTableSlots();
  const labels = useInContextLabels();
  const stats = useTargetStats();
  const assets = useAssets();
  const fr = useForward();
  const pred = usePrediction();
  const sel = useSim(
    useShallow((s) => ({
      selectedPos: s.selectedPos,
      setSelectedPos: s.setSelectedPos,
      disabledLayers: s.disabledLayers,
      setDisabledLayers: s.setDisabledLayers,
    })),
  );

  const cfg = assets.weights?.config ?? MINI_CONFIG;
  const S = ctx.tokens.length;
  const d = cfg.dModel;
  const H = cfg.nHeads;
  const target = ctx.targetPos;
  const valid = sel.selectedPos !== null && sel.selectedPos >= 0 && sel.selectedPos < S && !ctx.tokens[sel.selectedPos].isPadding;
  const focus = valid ? (sel.selectedPos as number) : Math.max(0, target);
  const { setSelectedPos } = sel;
  const setFocus = useCallback((pos: number | null) => setSelectedPos(pos === null || pos === target ? null : pos), [setSelectedPos, target]);

  const prefixes = useMemo(() => tablePrefixes(db.tables.map((t) => t.schema.name)), [db]);
  const tokenLabels = useMemo(() => ctx.tokens.map((t) => tokenLabel(t, prefixes)), [ctx, prefixes]);
  const tokenSlots = useMemo(() => ctx.tokens.map((t) => (t.isPadding ? 0 : (slots[t.tableName] ?? 0))), [ctx, slots]);
  // every figure draws the real tokens only (rows [0, numReal)); ShapeTags describe that span
  const span = useMemo<TokenSpan>(() => ({ shown: ctx.numReal, L: ctx.params.L, real: ctx.numReal }), [ctx]);

  return {
    db,
    pdb,
    task,
    seedNode,
    ctx,
    masks,
    slots,
    labels,
    stats,
    assets,
    fr,
    pred,
    weights: assets.weights,
    store: assets.store,
    cfg,
    S,
    d,
    H,
    dh: d / H,
    dFF: cfg.dFF,
    dText: cfg.dText,
    nBlocks: cfg.nBlocks,
    /** Non-padding tokens come first (fly.rs pads the tail), so rows [0, numReal) are the real ones. */
    numReal: ctx.numReal,
    target,
    focus,
    setFocus,
    prefixes,
    tokenLabels,
    tokenSlots,
    span,
    disabledLayers: sel.disabledLayers,
    setDisabledLayers: sel.setDisabledLayers,
  };
}

export type E2ELive = ReturnType<typeof useE2EState>;

/**
 * What the loaded weights mean for this database: zero-shot (pretrained without it, the held-out
 * database), seen in pretraining, fine-tuned on it, or untrained. Read from the registry's database
 * role and the checkpoint name; see src/engine/pipeline.ts (CHECKPOINT_INFO, autoCheckpoint).
 */
export type RunKind = 'zero-shot' | 'seen' | 'finetuned' | 'init';
export function runKind(live: Pick<E2ELive, 'db' | 'assets'>): RunKind {
  const { assets, db } = live;
  if (assets.weightsKind === 'init' || assets.ckpt === 'init') return 'init';
  if (assets.ckpt.startsWith('finetune')) return 'finetuned';
  return listDatabases().find((d) => d.name === db.name)?.role === 'target' ? 'zero-shot' : 'seen';
}

const Ctx = createContext<E2ELive | null>(null);

export function E2EProvider({ children }: { children: ReactNode }) {
  const live = useE2EState();
  return <Ctx.Provider value={live}>{children}</Ctx.Provider>;
}

export function useE2E(): E2ELive {
  const v = useContext(Ctx);
  if (!v) throw new Error('useE2E outside <E2EProvider>');
  return v;
}

/** A weight tensor of the loaded checkpoint by its PyTorch state_dict name (throws if absent). */
export function W(weights: RTWeights, name: string): Float32Array {
  return tensor(weights, name);
}

/**
 * The two 384-wide batch fields of the context, exactly as fly.rs:472-486 writes them:
 * `col_name_values` = E("<col> of <table>") and `text_values` = E(text[text_idx]) for every emitted
 * cell (pre.rs stores text index 0 for non-text cells, so they carry the embedding of string 0),
 * and zeros for padding (Vecs::new, fly.rs:65-85).
 */
export function textFields(live: Pick<E2ELive, 'ctx' | 'pdb' | 'dText'>, store: EmbeddingStore) {
  const { ctx, pdb, dText } = live;
  const S = ctx.tokens.length;
  const colName = new Float32Array(S * dText);
  const text = new Float32Array(S * dText);
  for (const t of ctx.tokens) {
    if (t.isPadding) continue;
    colName.set(store.get(pdb.text[t.colNameIdx]), t.pos * dText);
    text.set(store.get(pdb.text[t.textIdx]), t.pos * dText);
  }
  return { colName, text };
}
