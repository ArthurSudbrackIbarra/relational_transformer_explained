/**
 * Live data shared by the masks chapter's figures: the global context, its engine masks, the table
 * grouping and the pinned query token.
 *
 * The pinned query is the store's `selectedPos` ("the token being inspected", shared with the
 * attention chapter); null means the masked target. Hover state stays local to each figure.
 */
import { useCallback, useMemo } from 'react';
import { useContextWindow, useDatabase, useMasks, usePreprocessed, useTableSlots } from '@/state/hooks';
import { useSim } from '@/state/store';
import { groupContext, holeClassifier, tablePrefixes, tokenLabel } from './maskUtils';

export function useLiveMasks() {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const ctx = useContextWindow();
  const masks = useMasks();
  const slots = useTableSlots();
  const selectedPos = useSim((s) => s.selectedPos);
  const setSelectedPos = useSim((s) => s.setSelectedPos);

  const S = ctx.tokens.length;
  const prefixes = useMemo(() => tablePrefixes(db.tables.map((t) => t.schema.name)), [db]);
  const holeKind = useMemo(() => holeClassifier(ctx, pdb), [ctx, pdb]);
  const grouping = useMemo(() => groupContext(ctx.tokens, slots, holeKind), [ctx, slots, holeKind]);
  const labels = useMemo(() => ctx.tokens.map((t) => tokenLabel(t, prefixes)), [ctx, prefixes]);
  const tokenSlots = useMemo(() => ctx.tokens.map((t) => (t.isPadding ? 0 : slots[t.tableName] ?? 0)), [ctx, slots]);

  const target = ctx.targetPos >= 0 ? ctx.targetPos : 0;
  const valid = selectedPos !== null && selectedPos >= 0 && selectedPos < S && !ctx.tokens[selectedPos].isPadding;
  const pinned = valid ? selectedPos : target;
  const setPinned = useCallback((pos: number | null) => setSelectedPos(pos === null || pos === target ? null : pos), [setSelectedPos, target]);

  return { db, pdb, ctx, masks, slots, S, prefixes, grouping, holeKind, labels, tokenSlots, target, pinned, setPinned };
}

export type LiveMasks = ReturnType<typeof useLiveMasks>;
