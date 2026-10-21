/**
 * Live data for the block chapter: the global context and masks, the forward pass of the mini model
 * (full capture, cached by the engine pipeline: computed once and shared with every chapter), the
 * table colours and the token presets used by the figures. Nothing here recomputes the model.
 */
import { useMemo } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { MINI_CONFIG } from '@/engine/model/config';
import type { Token } from '@/engine/types';
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
import { article, singular, tablePrefixes, tokenLabel } from './utils';

export interface TokenPreset {
  id: string;
  label: string;
  pos: number;
}

export function useBlockLive() {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const task = useTask();
  const ctx = useContextWindow();
  const masks = useMasks();
  const slots = useTableSlots();
  const inCtx = useInContextLabels();
  const assets = useAssets();
  const fr = useForward();
  const { disabledLayers, setDisabledLayers, setBlock, setLayer } = useSim(
    useShallow((s) => ({ disabledLayers: s.disabledLayers, setDisabledLayers: s.setDisabledLayers, setBlock: s.setBlock, setLayer: s.setLayer })),
  );

  const cfg = assets.weights?.config ?? MINI_CONFIG;
  const nBlocks = fr?.blocks.length ?? cfg.nBlocks;
  const d = fr?.d ?? cfg.dModel;
  const H = cfg.nHeads;
  const dFF = cfg.dFF;
  const S = ctx.tokens.length;
  const target = ctx.targetPos >= 0 ? ctx.targetPos : 0;

  const prefixes = useMemo(() => tablePrefixes(db.tables.map((t) => t.schema.name)), [db]);
  const labels = useMemo(() => ctx.tokens.map((t) => tokenLabel(t, prefixes)), [ctx, prefixes]);

  const presets = useMemo(() => tokenPresets(ctx.tokens, target, inCtx.self.map((l) => l.pos), inCtx.other.map((l) => l.pos), task.entityTable), [ctx, target, inCtx, task]);

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
    cfg,
    nBlocks,
    d,
    H,
    dFF,
    S,
    target,
    prefixes,
    labels,
    presets,
    disabledLayers,
    setDisabledLayers,
    setBlock,
    setLayer,
  };
}

export type BlockLive = ReturnType<typeof useBlockLive>;

/** Token presets derived from the live context (never hard-coded positions). */
export function tokenPresets(tokens: readonly Token[], target: number, self: number[], other: number[], entityTable?: string): TokenPreset[] {
  const out: TokenPreset[] = [];
  const seed = tokens[target];
  if (!seed || seed.isPadding) return out;
  out.push({ id: 'target', label: 'the masked target', pos: target });
  const entity = seed.f2pNbrIdxs.find((p) => p >= 0);
  const ent = entity !== undefined ? tokens.find((t) => !t.isPadding && t.nodeIdx === entity) : undefined;
  const entName = entityTable ? singular(entityTable) : ent ? singular(ent.tableName) : 'entity';
  if (self.length) out.push({ id: 'self', label: `a past label of the same ${entName}`, pos: self[0] });
  if (other.length) out.push({ id: 'other', label: `another ${entName}’s label`, pos: other[0] });
  if (ent) {
    out.push({ id: 'entity', label: `the ${entName}’s ${ent.colName}`, pos: ent.pos });
    const child = tokens.find((t) => !t.isPadding && t.tableType === 'Db' && t.f2pNbrIdxs.includes(entity!) && t.nodeIdx !== seed.nodeIdx);
    if (child) {
      out.push({ id: 'child', label: `${article(singular(child.tableName))} ${singular(child.tableName)} of this ${entName}`, pos: child.pos });
      const op = child.f2pNbrIdxs.find((p) => p >= 0 && p !== entity && tokens.some((t) => !t.isPadding && t.nodeIdx === p));
      if (op !== undefined) {
        const o = tokens.find((t) => !t.isPadding && t.nodeIdx === op)!;
        out.push({ id: 'other-parent', label: `${article(singular(o.tableName))} ${singular(o.tableName)} it points to`, pos: o.pos });
      }
    }
    // a DB row that is NOT linked to the entity at all (to show multi-hop paths)
    const far = tokens.find(
      (t) =>
        !t.isPadding &&
        t.tableType === 'Db' &&
        t.nodeIdx !== entity &&
        t.tableName === ent.tableName,
    );
    if (far) out.push({ id: 'far', label: `another ${entName}`, pos: far.pos });
  }
  return out;
}
