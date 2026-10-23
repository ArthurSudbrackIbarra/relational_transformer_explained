/**
 * Node shapes for composite encoding (hue × shape). The ShopDB table slots
 * fail the dataviz all-pairs color check (see index.css), so wherever table
 * identity is shown as a free-floating mark (EntityGraph, legends), the slot
 * also picks a shape. Fixed mapping by slot, never cycled.
 */

export type ShapeName = 'circle' | 'square' | 'diamond' | 'triangle' | 'hexagon' | 'triangle-down' | 'pentagon' | 'cross';

/** Slot 1..8 → shape. Slot 5 (task tables) is the hexagon. */
export const SLOT_SHAPES: readonly ShapeName[] = [
  'circle', // 0 neutral
  'circle', // 1 blue
  'square', // 2 orange
  'diamond', // 3 aqua
  'triangle', // 4 yellow
  'hexagon', // 5 magenta (task tables)
  'triangle-down', // 6 green
  'pentagon', // 7 violet
  'cross', // 8 red
];

export function shapeOfSlot(slot: number | null | undefined): ShapeName {
  return SLOT_SHAPES[slot && slot >= 1 && slot <= 8 ? slot : 0];
}

function polygon(n: number, r: number, rot: number): string {
  const pts: string[] = [];
  for (let i = 0; i < n; i++) {
    const a = rot + (i * 2 * Math.PI) / n;
    pts.push(`${(r * Math.cos(a)).toFixed(2)},${(r * Math.sin(a)).toFixed(2)}`);
  }
  return `M${pts.join('L')}Z`;
}

/**
 * SVG path for a shape centered at (0,0) with "radius" r. Areas are roughly
 * equalised so no table looks heavier than another.
 */
export function shapePath(shape: ShapeName, r: number): string {
  switch (shape) {
    case 'circle':
      return `M${r},0A${r},${r} 0 1,1 ${-r},0A${r},${r} 0 1,1 ${r},0Z`;
    case 'square': {
      const s = r * 0.89;
      return `M${-s},${-s}H${s}V${s}H${-s}Z`;
    }
    case 'diamond':
      return polygon(4, r * 1.2, -Math.PI / 2);
    case 'triangle':
      return polygon(3, r * 1.3, -Math.PI / 2);
    case 'triangle-down':
      return polygon(3, r * 1.3, Math.PI / 2);
    case 'hexagon':
      return polygon(6, r * 1.08, 0);
    case 'pentagon':
      return polygon(5, r * 1.12, -Math.PI / 2);
    case 'cross': {
      const a = r * 0.36;
      const b = r * 1.05;
      return `M${-a},${-b}H${a}V${-a}H${b}V${a}H${a}V${b}H${-a}V${a}H${-b}V${-a}H${-a}Z`;
    }
  }
}

/** Inline SVG glyph of a shape (legends, swatches). */
export function ShapeGlyph({
  shape,
  size = 12,
  fill,
  stroke,
  className,
}: {
  shape: ShapeName;
  size?: number;
  fill: string;
  stroke?: string;
  className?: string;
}) {
  const r = size / 2 - 1;
  return (
    <svg aria-hidden="true" width={size} height={size} viewBox={`${-size / 2} ${-size / 2} ${size} ${size}`} className={className}>
      <path d={shapePath(shape, r * 0.92)} style={{ fill, stroke: stroke ?? 'none' }} strokeWidth={1} />
    </svg>
  );
}
