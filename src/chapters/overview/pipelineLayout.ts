/**
 * Geometry and live data for the "RT in one picture" figure.
 *
 * The figure is one SVG drawn 1:1 in CSS pixels (never scaled), so every label keeps its real size
 * (≥ 13px). Three regions:
 *   - the database map: one dot per row of every DB table plus the active task table (its three
 *     splits side by side), packed into shelves;
 *   - the context window: one slot per token position (L of them);
 *   - the model column: embedding → 4 × (col, feat, nbr, full, MLP) → norm_out → decoder → output.
 *
 * Everything shown is read from the engine (pdb.tableInfo, ctx.visited, ctx.events, masks, the
 * forward capture). The only thing computed here beyond geometry is which earlier row "found" each
 * visited row, read back from the sampler's own event trace (push → pop → visit).
 */
import { textWidth } from '@/components/viz/text';
import type { PreprocessedDB, RawDatabase, SampledContext, Split, TaskDef } from '@/engine/types';

export const T = {
  band: 14.5, // band titles
  title: 14.5, // table names (semibold)
  count: 13.5,
  sub: 13.5, // split labels
  box: 14.5, // model boxes
  boxSmall: 13.5,
  value: 15,
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// Database map
// ─────────────────────────────────────────────────────────────────────────────

export interface GridSpec {
  key: string; // "<table>:<type>"
  label: string | null; // split label for task tables
  nodes: number[];
}

export interface BlockSpec {
  key: string;
  title: string;
  slot: number;
  isTask: boolean;
  grids: GridSpec[];
}

export interface DotGrid extends GridSpec {
  cols: number;
  rows: number;
  x: number;
  y: number; // top of the dots
  w: number;
  h: number;
}

export interface DbBlock extends Omit<BlockSpec, 'grids'> {
  grids: DotGrid[];
  count: number;
  x: number;
  y: number; // top of the block (title baseline area)
  w: number;
  h: number;
}

export interface MapLayout {
  blocks: DbBlock[];
  pitch: number;
  dot: number;
  width: number;
  height: number;
}

const SPLITS: Split[] = ['Train', 'Val', 'Test'];

/** DB tables in schema order, then the active forecast task table (Train, Val, Test). */
export function blockSpecs(db: RawDatabase, pdb: PreprocessedDB, task: TaskDef, slots: Record<string, number>): BlockSpec[] {
  const range = (key: string) => {
    const info = pdb.tableInfo[key];
    if (!info) return [];
    return Array.from({ length: info.numNodes }, (_, i) => info.nodeIdxOffset + i);
  };
  const out: BlockSpec[] = db.tables.map((t) => ({
    key: `${t.schema.name}:Db`,
    title: t.schema.name,
    slot: slots[t.schema.name] ?? 0,
    isTask: false,
    grids: [
      {
        key: `${t.schema.name}:Db`,
        label: null,
        nodes: range(`${t.schema.name}:Db`),
      },
    ],
  }));
  if (task.kind === 'forecast') {
    out.push({
      key: `${task.name}:task`,
      title: task.name,
      slot: slots[task.name] ?? 5,
      isTask: true,
      grids: SPLITS.map((s) => ({
        key: `${task.name}:${s}`,
        label: s,
        nodes: range(`${task.name}:${s}`),
      })).filter((g) => g.nodes.length > 0),
    });
  }
  return out;
}

const HEADER_H = 24;
const SUB_H = 20;
const GAP_X = 26;
const GAP_Y = 22;
const SUB_GAP = 14;

function layoutBlocksFor(specs: BlockSpec[], width: number, pitch: number, R: number) {
  const maxCols = Math.max(4, Math.floor(width / pitch));
  const sized = specs.map((b) => {
    const count = b.grids.reduce((s, g) => s + g.nodes.length, 0);
    const titleW = textWidth(b.title, T.title, { weight: 600 }) + 16 + textWidth(count.toLocaleString('en-US'), T.count) + 6;
    // grids flow left to right inside the block, wrapping when needed
    const single = b.grids.length === 1;
    const grids = b.grids.map((g) => {
      const labelW = g.label ? textWidth(`${g.label} ${g.nodes.length}`, T.sub) + 4 : 0;
      // a one-grid block is at least as wide as its title, so small tables stay squat
      const minCols = Math.ceil((single ? titleW : labelW) / pitch);
      const cols = Math.min(maxCols, g.nodes.length, Math.max(Math.ceil(g.nodes.length / R), minCols, 2));
      const rows = Math.ceil(g.nodes.length / cols);
      return {
        ...g,
        cols,
        rows,
        w: Math.max(cols * pitch, labelW),
        h: rows * pitch + (g.label ? SUB_H : 0),
      };
    });
    let x = 0;
    let y = HEADER_H;
    let lineH = 0;
    let bw = titleW;
    const placed: DotGrid[] = [];
    for (const g of grids) {
      if (x > 0 && x + g.w > width) {
        x = 0;
        y += lineH + 10;
        lineH = 0;
      }
      placed.push({ ...g, x, y: y + (g.label ? SUB_H : 0), h: g.rows * pitch });
      x += g.w + SUB_GAP;
      lineH = Math.max(lineH, g.h);
      bw = Math.max(bw, x - SUB_GAP);
    }
    return {
      spec: b,
      count,
      grids: placed,
      w: Math.min(width, bw),
      h: y + lineH,
    };
  });
  // shelf packing
  const blocks: DbBlock[] = [];
  let x = 0;
  let y = 0;
  let shelfH = 0;
  let empty = 0;
  let used = 0;
  for (const s of sized) {
    if (x > 0 && x + s.w > width) {
      empty += (width - used) * shelfH;
      x = 0;
      y += shelfH + GAP_Y;
      shelfH = 0;
      used = 0;
    }
    used = x + s.w;
    blocks.push({
      ...s.spec,
      count: s.count,
      x,
      y,
      w: s.w,
      h: s.h,
      grids: s.grids.map((g) => ({ ...g, x: g.x + x, y: g.y + y })),
    });
    x += s.w + GAP_X;
    shelfH = Math.max(shelfH, s.h);
  }
  empty += (width - used) * shelfH;
  return { blocks, height: y + shelfH, empty };
}

/** Pack the map into `width` px, choosing the grid height that gives the most compact layout. */
export function layoutMap(specs: BlockSpec[], width: number): MapLayout {
  const pitch = width >= 460 ? 6 : 5;
  const dot = pitch - 1.5;
  // most compact layout, with a mild penalty for empty shelf space (keeps the map from looking ragged)
  let best: { blocks: DbBlock[]; height: number; cost: number } | null = null;
  for (let R = 5; R <= 30; R++) {
    const l = layoutBlocksFor(specs, width, pitch, R);
    const cost = l.height + (0.35 * l.empty) / width;
    if (!best || cost < best.cost - 0.5) best = { ...l, cost };
  }
  return { blocks: best!.blocks, pitch, dot, width, height: best!.height };
}

/** Top-left corner of a node's dot in map coordinates (null if the node is not on the map). */
export function dotIndex(map: MapLayout): Map<number, { x: number; y: number; block: DbBlock }> {
  const out = new Map<number, { x: number; y: number; block: DbBlock }>();
  for (const b of map.blocks)
    for (const g of b.grids)
      g.nodes.forEach((n, i) => {
        out.set(n, {
          x: g.x + (i % g.cols) * map.pitch,
          y: g.y + Math.floor(i / g.cols) * map.pitch,
          block: b,
        });
      });
  return out;
}

/** Hit-test a point in map coordinates → node idx (or null). */
export function nodeAt(map: MapLayout, x: number, y: number): number | null {
  for (const b of map.blocks)
    for (const g of b.grids) {
      if (x < g.x - 1 || y < g.y - 1 || x >= g.x + g.cols * map.pitch || y >= g.y + g.rows * map.pitch) continue;
      const c = Math.floor((x - g.x) / map.pitch);
      const r = Math.floor((y - g.y) / map.pitch);
      const i = r * g.cols + c;
      if (c >= 0 && r >= 0 && i < g.nodes.length) return g.nodes[i];
    }
  return null;
}

/** One SVG path drawing every dot of a grid as a small square (one DOM node per grid). */
export function gridPath(g: DotGrid, pitch: number, dot: number): string {
  const parts: string[] = [];
  for (let i = 0; i < g.nodes.length; i++) {
    const x = g.x + (i % g.cols) * pitch;
    const y = g.y + Math.floor(i / g.cols) * pitch;
    parts.push(`M${x},${y}h${dot}v${dot}h${-dot}z`);
  }
  return parts.join('');
}

// ─────────────────────────────────────────────────────────────────────────────
// BFS discovery edges, read back from the sampler's event trace
// ─────────────────────────────────────────────────────────────────────────────

export interface Discovery {
  node: number;
  depth: number;
  order: number;
  /** Row whose expansion pushed the frontier entry that was popped for this visit (null for the seed). */
  from: number | null;
  /** 'parent' = F→P (the found row is a parent of `from`); 'child' = P→F. */
  via: 'parent' | 'child' | null;
}

export function discoveries(ctx: SampledContext): Discovery[] {
  const pushes = new Map<number, { from: number; depth: number; via: 'parent' | 'child' }[]>();
  const found = new Map<number, { from: number; via: 'parent' | 'child' } | null>();
  for (const e of ctx.events) {
    if (e.kind === 'push-parent' || e.kind === 'push-task-child' || e.kind === 'push-db-child') {
      const list = pushes.get(e.node) ?? [];
      list.push({
        from: e.from,
        depth: e.depth,
        via: e.kind === 'push-parent' ? 'parent' : 'child',
      });
      pushes.set(e.node, list);
    } else if (e.kind === 'visit') {
      const list = pushes.get(e.node) ?? [];
      // the popped entry carries the visit's depth label; take the latest such push
      let hit: { from: number; via: 'parent' | 'child' } | null = null;
      for (let i = list.length - 1; i >= 0; i--)
        if (list[i].depth === e.depth) {
          hit = { from: list[i].from, via: list[i].via };
          break;
        }
      found.set(e.node, hit);
    }
  }
  return ctx.visited.map((v) => {
    const f = found.get(v.node) ?? null;
    return {
      node: v.node,
      depth: v.depth,
      order: v.order,
      from: f?.from ?? null,
      via: f?.via ?? null,
    };
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Token grid
// ─────────────────────────────────────────────────────────────────────────────

export interface TokenGrid {
  x: number;
  y: number;
  chip: number;
  gap: number;
  perRow: number;
  rows: number;
  w: number;
  h: number;
}

export function layoutTokens(n: number, width: number, x: number, y: number): TokenGrid {
  const chip = width >= 520 ? 21 : 19;
  const gap = 4;
  const pitch = chip + gap;
  const perRow = Math.max(4, Math.min(n, Math.floor((width + gap) / pitch)));
  const rows = Math.ceil(n / perRow);
  return {
    x,
    y,
    chip,
    gap,
    perRow,
    rows,
    w: perRow * pitch - gap,
    h: rows * pitch - gap,
  };
}

export function tokenXY(g: TokenGrid, i: number): { x: number; y: number } {
  const pitch = g.chip + g.gap;
  return {
    x: g.x + (i % g.perRow) * pitch,
    y: g.y + Math.floor(i / g.perRow) * pitch,
  };
}

export function tokenAt(g: TokenGrid, n: number, x: number, y: number): number | null {
  const pitch = g.chip + g.gap;
  if (x < g.x || y < g.y) return null;
  const c = Math.floor((x - g.x) / pitch);
  const r = Math.floor((y - g.y) / pitch);
  if (c >= g.perRow || (x - g.x) % pitch > g.chip || (y - g.y) % pitch > g.chip) return null;
  const i = r * g.perRow + c;
  return i >= 0 && i < n ? i : null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Whole figure
// ─────────────────────────────────────────────────────────────────────────────

export interface ModelBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface ModelLayout {
  x: number;
  y: number;
  w: number;
  /** Narrow screens: layers as a 3 + 2 grid of tiles, norm and decoder side by side. */
  compact: boolean;
  embed: ModelBox;
  block: ModelBox; // dashed container
  layers: ModelBox[]; // col, feat, nbr, full, mlp
  norm: ModelBox;
  decoder: ModelBox;
  out: ModelBox;
  /** Connector paths (with arrow heads) between the boxes. */
  arrows: string[];
  h: number;
}

function layoutModel(x: number, y: number, w: number, compact: boolean): ModelLayout {
  const BOX = 34;
  const ax = x + 24;
  const v = (y1: number, y2: number, xx = ax) => `M${xx},${y1 + 2} L${xx},${y2 - 4}`;
  let cy = y;
  const embed = { x, y: cy, w, h: BOX };
  cy += BOX + 22;
  const blockTop = cy;
  cy += 30; // container title
  const layers: ModelBox[] = [];
  if (!compact) {
    const LAYER = 31;
    for (let i = 0; i < 5; i++) {
      layers.push({ x: x + 12, y: cy, w: w - 24, h: LAYER });
      cy += LAYER + (i === 3 ? 10 : 6);
    }
    cy += 6;
  } else {
    const TILE = 46;
    const gap = 6;
    const inner = w - 24;
    const w3 = (inner - 2 * gap) / 3;
    const w2 = (inner - gap) / 2;
    for (let i = 0; i < 3; i++) layers.push({ x: x + 12 + i * (w3 + gap), y: cy, w: w3, h: TILE });
    cy += TILE + gap;
    for (let i = 0; i < 2; i++) layers.push({ x: x + 12 + i * (w2 + gap), y: cy, w: w2, h: TILE });
    cy += TILE + 12;
  }
  const block = { x, y: blockTop, w, h: cy - blockTop };
  cy += 22;
  const arrows = [v(embed.y + embed.h, block.y)];
  let norm: ModelBox;
  let decoder: ModelBox;
  if (!compact) {
    norm = { x, y: cy, w, h: BOX };
    arrows.push(v(block.y + block.h, norm.y));
    cy += BOX + 18;
    decoder = { x, y: cy, w, h: BOX };
    arrows.push(v(norm.y + norm.h, decoder.y));
    cy += BOX + 18;
  } else {
    const half = (w - 22) / 2;
    norm = { x, y: cy, w: half, h: BOX };
    decoder = { x: x + half + 22, y: cy, w: half, h: BOX };
    arrows.push(v(block.y + block.h, norm.y));
    arrows.push(`M${norm.x + norm.w + 2},${cy + BOX / 2} L${decoder.x - 4},${cy + BOX / 2}`);
    cy += BOX + 18;
  }
  const out = { x, y: cy, w, h: 78 };
  arrows.push(v(decoder.y + decoder.h, out.y, compact ? decoder.x + 24 : ax));
  cy += 78;
  return {
    x,
    y,
    w,
    compact,
    embed,
    block,
    layers,
    norm,
    decoder,
    out,
    arrows,
    h: cy - y,
  };
}

export interface FigureLayout {
  W: number;
  H: number;
  wide: boolean;
  mapX: number;
  mapY: number;
  map: MapLayout;
  tokensLabelY: number;
  tokens: TokenGrid;
  model: ModelLayout;
  modelLabelY: number;
  mapLabelY: number;
}

export function layoutFigure(W: number, specs: BlockSpec[], nTokens: number): FigureLayout {
  const wide = W >= 860;
  const modelW = wide ? (W >= 1060 ? 330 : 300) : W;
  const leftW = wide ? W - modelW - 40 : W;
  const mapLabelY = 16;
  const mapY = 34;
  const map = layoutMap(specs, leftW);
  const tokensLabelY = mapY + map.height + 40;
  const tokens = layoutTokens(nTokens, leftW, 0, tokensLabelY + 18);
  const leftBottom = tokens.y + tokens.h;
  const modelLabelY = wide ? mapLabelY : leftBottom + 44;
  const model = layoutModel(wide ? W - modelW : 0, wide ? mapY : modelLabelY + 18, modelW, !wide);
  const H = Math.ceil(Math.max(leftBottom, model.y + model.h) + 6);
  return {
    W,
    H,
    wide,
    mapX: 0,
    mapY,
    map,
    tokensLabelY,
    tokens,
    model,
    modelLabelY,
    mapLabelY,
  };
}
