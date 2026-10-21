/**
 * Live data for the attention chapter: the global context, its masks, the forward pass of the trained
 * mini model (with full capture) and the reader's selection (block, layer, head, query token).
 *
 * The selection lives in the global store's "inspect" fields (`block`, `layer`, `head`, `selectedPos`),
 * which exist for exactly this purpose and are shared with the masks chapter: a query pinned there is
 * the query here. `selectedPos = null` means "the masked target".
 */
import { useCallback, useMemo } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { MINI_CONFIG } from '@/engine/model/config';
import { tensor } from '@/engine/model/weights';
import type { AttnCapture, AttnLayer, Token } from '@/engine/types';
import {
  useAssets,
  useContextWindow,
  useDatabase,
  useForward,
  useInContextLabels,
  useMasks,
  usePreprocessed,
  useTableSlots,
  useTask,
} from '@/state/hooks';
import { useSim } from '@/state/store';
import { groupKeys, rowOf, tablePrefixes, tokenLabel } from './utils';

export function useAttnLive() {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const task = useTask();
  const ctx = useContextWindow();
  const masks = useMasks();
  const slots = useTableSlots();
  const inCtx = useInContextLabels();
  const assets = useAssets();
  const fr = useForward();
  const sel = useSim(
    useShallow((s) => ({
      block: s.block,
      layer: s.layer,
      head: s.head,
      selectedPos: s.selectedPos,
      disabledLayers: s.disabledLayers,
      setBlock: s.setBlock,
      setLayer: s.setLayer,
      setHead: s.setHead,
      setSelectedPos: s.setSelectedPos,
      setDisabledLayers: s.setDisabledLayers,
    })),
  );

  const cfg = assets.weights?.config ?? MINI_CONFIG;
  const nBlocks = fr?.blocks.length ?? cfg.nBlocks;
  const H = cfg.nHeads;
  const d = fr?.d ?? cfg.dModel;
  const dh = d / H;
  const S = ctx.tokens.length;

  const block = Math.max(0, Math.min(nBlocks - 1, sel.block));
  const head = Math.max(0, Math.min(H - 1, sel.head));
  const layer: AttnLayer = sel.layer;
  const target = ctx.targetPos >= 0 ? ctx.targetPos : 0;
  const valid = sel.selectedPos !== null && sel.selectedPos >= 0 && sel.selectedPos < S && !ctx.tokens[sel.selectedPos].isPadding;
  const query = valid ? (sel.selectedPos as number) : target;
  const setQuery = useCallback(
    (pos: number | null) => sel.setSelectedPos(pos === null || pos === target ? null : pos),
    [sel, target],
  );

  const prefixes = useMemo(() => tablePrefixes(db.tables.map((t) => t.schema.name)), [db]);
  const labels = useMemo(() => ctx.tokens.map((t) => tokenLabel(t, prefixes)), [ctx, prefixes]);
  const tokenSlots = useMemo(() => ctx.tokens.map((t) => (t.isPadding ? 0 : slots[t.tableName] ?? 0)), [ctx, slots]);
  const keys = useMemo(() => groupKeys(ctx.tokens, slots), [ctx, slots]);

  return {
    db,
    pdb,
    task,
    ctx,
    masks,
    slots,
    inCtx,
    assets,
    fr,
    weights: assets.weights,
    cfg,
    nBlocks,
    H,
    d,
    dh,
    S,
    numReal: ctx.numReal,
    block,
    head,
    layer,
    query,
    target,
    setBlock: sel.setBlock,
    setLayer: sel.setLayer,
    setHead: sel.setHead,
    setQuery,
    disabledLayers: sel.disabledLayers,
    setDisabledLayers: sel.setDisabledLayers,
    prefixes,
    labels,
    tokenSlots,
    keys,
  };
}

export type AttnLive = ReturnType<typeof useAttnLive>;

/** Everything one query sees in one head of one attention sublayer (all views into the capture). */
export interface QueryView {
  cap: AttnCapture;
  /** x entering the sublayer, h = RMSNorm_l(x), γ of that RMSNorm. */
  x: Float32Array;
  h: Float32Array;
  gamma: Float32Array;
  /** q, k, v rows of the query (all heads, d dims). */
  q: Float32Array;
  /** This head's slice of the query (d_head dims). */
  qh: Float32Array;
  /** Scaled scores q·k/√d_head for every key (before masking), this head. */
  scores: Float32Array;
  /** Masked softmax weights for every key (0 where masked), this head. */
  probs: Float32Array;
  /** The layer's mask row for the query (shared by all heads). */
  allowed: Uint8Array;
  allowedList: number[];
  /** Concatenated head outputs at the query (d) and this head's slice (d_head). */
  headOut: Float32Array;
  headSlice: Float32Array;
  /** W_o · headOut (the residual update) and x + update. */
  out: Float32Array;
  xOut: Float32Array;
  /** Weight matrices of this sublayer, [d × d] row-major (PyTorch [out, in]). */
  Wq: Float32Array;
  Wk: Float32Array;
  Wv: Float32Array;
  Wo: Float32Array;
}

/** Slice the capture for (block, layer, head, query). null while loading or when the sublayer is ablated. */
export function useQueryView(live: AttnLive, override?: { block?: number; layer?: AttnLayer; head?: number; query?: number }): QueryView | null | 'skipped' {
  const { fr, weights, masks, S, d, dh, H } = live;
  const block = override?.block ?? live.block;
  const layer = override?.layer ?? live.layer;
  const head = override?.head ?? live.head;
  const query = override?.query ?? live.query;
  return useMemo(() => {
    if (!fr || !weights) return null;
    const cap = fr.blocks[Math.min(block, fr.blocks.length - 1)].attn[layer];
    if (cap.skipped || cap.scores.length < H * S * S) return 'skipped';
    const p = `blocks.${block}.attns.${layer}`;
    const allowed = masks[layer].subarray(query * S, (query + 1) * S);
    const allowedList: number[] = [];
    for (let k = 0; k < S; k++) if (allowed[k]) allowedList.push(k);
    const q = rowOf(cap.q, query, d);
    const headOut = rowOf(cap.headOut, query, d);
    return {
      cap,
      x: rowOf(cap.xIn, query, d),
      h: rowOf(cap.h, query, d),
      gamma: tensor(weights, `blocks.${block}.norms.${layer}.weight`),
      q,
      qh: q.subarray(head * dh, (head + 1) * dh),
      scores: cap.scores.subarray((head * S + query) * S, (head * S + query + 1) * S),
      probs: cap.probs.subarray((head * S + query) * S, (head * S + query + 1) * S),
      allowed,
      allowedList,
      headOut,
      headSlice: headOut.subarray(head * dh, (head + 1) * dh),
      out: rowOf(cap.out, query, d),
      xOut: rowOf(cap.xOut, query, d),
      Wq: tensor(weights, `${p}.wq.weight`),
      Wk: tensor(weights, `${p}.wk.weight`),
      Wv: tensor(weights, `${p}.wv.weight`),
      Wo: tensor(weights, `${p}.wo.weight`),
    };
  }, [fr, weights, masks, block, layer, head, query, S, d, dh, H]);
}

/** Key row of this head: k or v of token `pos`, head slice. */
export function headRow(t: Float32Array, pos: number, head: number, d: number, dh: number): Float32Array {
  return t.subarray(pos * d + head * dh, pos * d + (head + 1) * dh);
}

/** The first token of each row in context order (for presets). */
export function firstOfRow(tokens: readonly Token[], node: number): Token | undefined {
  return tokens.find((t) => !t.isPadding && t.nodeIdx === node);
}
