/**
 * Attention-layer identity: NAME + ICON (BUILD_PLAN §5 "Attention layers are
 * identified by name label + icon"), never a colour of their own.
 * Order inside a block is col → feat → nbr → full (model.py:83, spec §7.1).
 */
import { Columns3, Grid3x3, Network, Rows3, type LucideIcon } from 'lucide-react';
import type { AttnLayer } from '@/engine/types';
import { cx } from '../ui/cx';

export interface LayerInfo {
  id: AttnLayer;
  /** Display name (the paper also calls "full" "global", spec §6.2). */
  name: string;
  icon: LucideIcon;
  /** Paper mask, p.5 §3.2. */
  paperMask: string;
  /** What it lets flow (spec §6.2). */
  flow: string;
}

export const LAYER_INFO: Record<AttnLayer, LayerInfo> = {
  col: {
    id: 'col',
    name: 'Column',
    icon: Columns3,
    paperMask: '1{Col(k) = Col(q)}',
    flow: 'between cells of the same column of the same table',
  },
  feat: {
    id: 'feat',
    name: 'Feature',
    icon: Rows3,
    paperMask: '1{Row(k) = Row(q) ∨ Row(k) ∈ OutLinks(q)}',
    flow: 'from a cell’s own row and its parent rows',
  },
  nbr: {
    id: 'nbr',
    name: 'Neighbor',
    icon: Network,
    paperMask: '1{Row(q) ∈ OutLinks(k)}',
    flow: 'from child rows into the parent’s cells',
  },
  full: {
    id: 'full',
    name: 'Full',
    icon: Grid3x3,
    paperMask: '1',
    flow: 'between all non-padding tokens',
  },
};

/** Inline layer label: icon + name. */
export function LayerLabel({ layer, className, size = 14 }: { layer: AttnLayer; className?: string; size?: number }) {
  const info = LAYER_INFO[layer];
  const Icon = info.icon;
  return (
    <span className={cx('inline-flex items-center gap-1.5 align-middle font-medium text-ink', className)}>
      <Icon aria-hidden="true" size={size} strokeWidth={2} className="text-ink-2" />
      {info.name}
    </span>
  );
}
