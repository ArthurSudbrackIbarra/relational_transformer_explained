/**
 * MaskMatrix: a Heatmap specialised for one boolean attention mask (spec §6.2).
 * mask[q * S + k] = 1 if query q may attend to key k. Rows = queries, columns = keys.
 * Uses one sequential hue for "allowed" (BUILD_PLAN §5).
 */
import type { AttnLayer } from '@/engine/types';
import { Heatmap, type HeatmapCell } from './Heatmap';
import { LayerLabel } from './layers';
import { cx } from '../ui/cx';

export interface MaskMatrixProps {
  /** Dense S×S mask (MaskSet[layer]). */
  mask: Uint8Array | ArrayLike<number>;
  S: number;
  /** Which layer this is (adds the name + icon heading). */
  layer?: AttnLayer;
  /** Show only the first n tokens (e.g. numReal, hiding padding). */
  crop?: number;
  /** Per-token labels (e.g. "customers.age"). */
  labels?: string[];
  /** Per-token table colour slots (identity strips along both axes). */
  slots?: number[];
  highlightRows?: ReadonlySet<number> | readonly number[];
  highlightCols?: ReadonlySet<number> | readonly number[];
  selected?: { row: number; col: number } | null;
  /** (query, key, allowed) or null. */
  onHover?: (cell: { q: number; k: number; allowed: boolean } | null) => void;
  onCellClick?: (cell: { q: number; k: number; allowed: boolean }) => void;
  cellSize?: number;
  maxCellSize?: number;
  title?: string;
  showLegend?: boolean;
  showReadout?: boolean;
  className?: string;
}

const toCell = (c: HeatmapCell) => ({ q: c.row, k: c.col, allowed: c.value > 0 });

export function MaskMatrix({
  mask,
  S,
  layer,
  crop,
  labels,
  slots,
  highlightRows,
  highlightCols,
  selected,
  onHover,
  onCellClick,
  cellSize,
  maxCellSize = 18,
  title,
  showLegend = true,
  showReadout = true,
  className,
}: MaskMatrixProps) {
  const n = Math.min(S, crop ?? S);
  return (
    <div className={cx('min-w-0', className)}>
      {layer && (
        <div className="mb-2 text-[17px]">
          <LayerLabel layer={layer} />
          <span className="text-muted"> mask</span>
        </div>
      )}
      <Heatmap
        data={mask}
        rows={S}
        cols={S}
        crop={{ rows: n, cols: n }}
        scale="binary"
        rowLabels={labels}
        colLabels={labels}
        rowSlots={slots}
        colSlots={slots}
        highlightRows={highlightRows}
        highlightCols={highlightCols}
        selected={selected}
        cellSize={cellSize}
        maxCellSize={maxCellSize}
        rowName="query"
        colName="key"
        binaryLabels={['may attend', 'blocked']}
        title={title ?? (layer ? `${layer} attention mask` : 'attention mask')}
        onHover={onHover ? (c) => onHover(c ? toCell(c) : null) : undefined}
        onCellClick={onCellClick ? (c) => onCellClick(toCell(c)) : undefined}
        showLegend={showLegend}
        showReadout={showReadout}
      />
    </div>
  );
}
