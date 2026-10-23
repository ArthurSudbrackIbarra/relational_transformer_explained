/**
 * Pure helpers that turn the sampler's event trace (src/engine/sampler.ts, a mirror of
 * rustler/src/fly.rs:357-511) into what the chapter displays:
 *
 *  - `parseIterations`: one record per loop iteration of fly.rs (`loop { select; visited?; expand; emit }`),
 *    with the event-index boundaries of each phase;
 *  - `buildSteps`: the step list driven by the StepPlayer, either one step per iteration ("rows") or one
 *    step per decision ("decisions": select → visit/parents → P→F filter → width bound → emit);
 *  - `replay`: the exact frontier state after any prefix of the trace (stack, depth buckets with the
 *    swap-remove draws, visited set, token count). It re-applies the recorded decisions; it never
 *    re-decides anything;
 *  - `buildLayout`: a deterministic radial layout of every row the BFS ever touched (seed at the center).
 *    A visited row sits on the ring of the depth it was visited with and hangs off the row that pushed
 *    the frontier entry actually popped ("the first pop wins", spec §4.5); a row that never left the
 *    frontier sits where it was first pushed.
 *
 * Nothing here adds sampler semantics: every decision comes from the engine's event trace.
 */
import { formatDate } from '@/components';
import type { PreprocessedDB, RawDatabase, RawTable, RawValue, SampledContext, SamplerEvent, TaskDef } from '@/engine/types';

// ─────────────────────────────────────────────────────────────────────────────
// Iterations and steps
// ─────────────────────────────────────────────────────────────────────────────

export type Phase = 'select' | 'visit' | 'filter' | 'width' | 'emit' | 'end';

function phaseOf(e: SamplerEvent): Phase {
  switch (e.kind) {
    case 'pop':
    case 'skip-visited':
      return 'select';
    case 'visit':
    case 'push-parent':
      return 'visit';
    case 'child-other-task-table':
    case 'child-future':
    case 'push-task-child':
      return 'filter';
    case 'db-children-subsampled':
    case 'push-db-child':
      return 'width';
    case 'drop-leakage-cell':
    case 'emit':
    case 'truncated':
      return 'emit';
    case 'frontier-empty':
    case 'pad':
      return 'end';
  }
}

/** One loop iteration of fly.rs `seq` (one pop). Boundaries are event indices (end exclusive). */
export interface Iteration {
  iter: number;
  node: number;
  depth: number;
  from: 'stack' | 'bucket';
  popAt: number;
  skipped: boolean;
  /** [popAt, selectEnd) = pop (+ skip-visited) */
  selectEnd: number;
  /** [selectEnd, visitEnd) = visit + push-parent* */
  visitEnd: number;
  /** [visitEnd, filterEnd) = per-edge filter events (other task table, future, task rows pushed) */
  filterEnd: number;
  /** [filterEnd, widthEnd) = db-children-subsampled? + push-db-child* */
  widthEnd: number;
  /** [widthEnd, end) = drop-leakage-cell / emit / truncated */
  end: number;
  hasChildren: boolean;
}

export function parseIterations(events: readonly SamplerEvent[]): Iteration[] {
  const out: Iteration[] = [];
  let i = 0;
  while (i < events.length) {
    const e = events[i];
    if (e.kind !== 'pop') {
      i++;
      continue;
    }
    const it: Iteration = {
      iter: out.length,
      node: e.node,
      depth: e.depth,
      from: e.from,
      popAt: i,
      skipped: false,
      selectEnd: i + 1,
      visitEnd: i + 1,
      filterEnd: i + 1,
      widthEnd: i + 1,
      end: i + 1,
      hasChildren: false,
    };
    let j = i + 1;
    // phases appear in a fixed order inside one iteration (sampler.ts header comment)
    const order: Phase[] = ['select', 'visit', 'filter', 'width', 'emit'];
    let cur = 0;
    while (j < events.length && events[j].kind !== 'pop') {
      const ph = phaseOf(events[j]);
      if (ph === 'end') break;
      const k = order.indexOf(ph);
      if (k < cur) break; // defensive: should not happen
      cur = k;
      if (events[j].kind === 'skip-visited') it.skipped = true;
      j++;
      if (ph === 'select') it.selectEnd = j;
      if (ph === 'select' || ph === 'visit') it.visitEnd = j;
      if (ph !== 'width' && ph !== 'emit') it.filterEnd = j;
      if (ph !== 'emit') it.widthEnd = j;
      it.end = j;
    }
    // make the boundaries monotone
    it.visitEnd = Math.max(it.visitEnd, it.selectEnd);
    it.filterEnd = Math.max(it.filterEnd, it.visitEnd);
    it.widthEnd = Math.max(it.widthEnd, it.filterEnd);
    it.end = Math.max(it.end, it.widthEnd);
    it.hasChildren = it.widthEnd > it.visitEnd;
    out.push(it);
    i = j;
  }
  return out;
}

export type StepKind = 'start' | 'row' | Phase;
export type Granularity = 'rows' | 'decisions';

export interface Step {
  kind: StepKind;
  /** Events [start, end) belong to this step; the state shown is the state after `end`. */
  start: number;
  end: number;
  /** Iteration index, −1 for start / end. */
  iter: number;
}

export function buildSteps(events: readonly SamplerEvent[], iters: readonly Iteration[], g: Granularity): Step[] {
  const steps: Step[] = [{ kind: 'start', start: 0, end: 0, iter: -1 }];
  for (const it of iters) {
    if (g === 'rows') {
      steps.push({ kind: 'row', start: it.popAt, end: it.end, iter: it.iter });
      continue;
    }
    steps.push({ kind: 'select', start: it.popAt, end: it.selectEnd, iter: it.iter });
    if (it.skipped) continue;
    steps.push({ kind: 'visit', start: it.selectEnd, end: it.visitEnd, iter: it.iter });
    if (it.hasChildren) {
      steps.push({ kind: 'filter', start: it.visitEnd, end: it.filterEnd, iter: it.iter });
      if (it.widthEnd > it.filterEnd) steps.push({ kind: 'width', start: it.filterEnd, end: it.widthEnd, iter: it.iter });
    }
    if (it.end > it.widthEnd) steps.push({ kind: 'emit', start: it.widthEnd, end: it.end, iter: it.iter });
  }
  const last = iters.length ? iters[iters.length - 1].end : 0;
  if (last < events.length) steps.push({ kind: 'end', start: last, end: events.length, iter: -1 });
  return steps;
}

/** Index of the step whose event range contains `cursor` (the state after `cursor` events). */
export function stepAtCursor(steps: readonly Step[], cursor: number): number {
  let best = 0;
  for (let i = 0; i < steps.length; i++) if (steps[i].end <= cursor) best = i;
  return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// Replay
// ─────────────────────────────────────────────────────────────────────────────

export interface ReplayState {
  /** F→P stack, bottom → top (fly.rs `f2p_ftr`). */
  stack: { node: number; depth: number }[];
  /** P→F depth buckets (fly.rs `p2f_ftr[depth]`), exact contents and order. */
  buckets: number[][];
  /** Visit order (node ids) and the depth label each row was expanded with. */
  visitedOrder: number[];
  visitedDepth: Map<number, number>;
  /** Every row that has entered the frontier so far (and the seed). */
  discovered: Set<number>;
  /** Token positions emitted per row. */
  emittedByNode: Map<number, number[]>;
  emitted: number;
  pops: number;
  skips: number;
  bucketDraws: number;
  /** Set once the hard stop at L fired or the frontier emptied. */
  finished: null | { kind: 'full'; node: number } | { kind: 'empty'; tokens: number };
}

/** Re-apply the first `upto` events of the trace, starting from fly.rs:366 (`f2p_ftr = [(0, seed)]`). */
export function replay(ctx: SampledContext, upto: number): ReplayState {
  const seed = ctx.params.seedNode;
  const st: ReplayState = {
    stack: [{ node: seed, depth: 0 }],
    buckets: [],
    visitedOrder: [],
    visitedDepth: new Map(),
    discovered: new Set([seed]),
    emittedByNode: new Map(),
    emitted: 0,
    pops: 0,
    skips: 0,
    bucketDraws: 0,
    finished: null,
  };
  const ev = ctx.events;
  const n = Math.min(upto, ev.length);
  const push = (d: number, node: number) => {
    while (st.buckets.length <= d) st.buckets.push([]);
    st.buckets[d].push(node);
    st.discovered.add(node);
  };
  for (let i = 0; i < n; i++) {
    const e = ev[i];
    switch (e.kind) {
      case 'pop':
        st.pops++;
        if (e.from === 'stack') st.stack.pop();
        else {
          const b = st.buckets[e.depth];
          const r = e.draw ?? 0;
          const last = b.length - 1;
          [b[r], b[last]] = [b[last], b[r]]; // swap-remove, fly.rs:392-394
          b.pop();
          st.bucketDraws++;
        }
        break;
      case 'skip-visited':
        st.skips++;
        break;
      case 'visit':
        st.visitedOrder.push(e.node);
        st.visitedDepth.set(e.node, e.depth);
        break;
      case 'push-parent':
        st.stack.push({ node: e.node, depth: e.depth });
        st.discovered.add(e.node);
        break;
      case 'push-task-child':
      case 'push-db-child':
        push(e.depth, e.node);
        break;
      case 'emit': {
        st.emitted++;
        const arr = st.emittedByNode.get(e.node);
        if (arr) arr.push(e.pos);
        else st.emittedByNode.set(e.node, [e.pos]);
        break;
      }
      case 'truncated':
        st.finished = { kind: 'full', node: e.node };
        break;
      case 'frontier-empty':
        st.finished = { kind: 'empty', tokens: e.tokens };
        break;
      default:
        break;
    }
  }
  return st;
}

// ─────────────────────────────────────────────────────────────────────────────
// Radial layout of every row the BFS touched
// ─────────────────────────────────────────────────────────────────────────────

export interface LayoutNode {
  node: number;
  /**
   * The row whose expansion pushed the entry this row was visited from (for a visited row), else the row
   * that first pushed it. Null for the seed.
   */
  parent: number | null;
  /** F→P: pushed as a parent of `parent`; P→F: pushed as a child of `parent`. */
  via: 'seed' | 'f2p' | 'p2f';
  /** Visit depth for a visited row, else the depth label of its first frontier entry (spec §4.5). */
  depth: number;
  /** Event index at which the row first entered the frontier (−1 for the seed). */
  firstEvent: number;
  /** Event index of the push that `parent`/`via` describe (−1 for the seed). */
  pathEvent: number;
  /** The first push, when it differs from the path push (a row first pushed by one row, visited through another). */
  first: { parent: number; via: 'f2p' | 'p2f'; depth: number } | null;
  x: number;
  y: number;
}

export interface Layout {
  nodes: Map<number, LayoutNode>;
  maxDepth: number;
  radius: number;
  /** Ring radius per depth label. */
  ring: (d: number) => number;
  /** Number of laid-out rows per depth label. */
  perDepth: number[];
}

/**
 * Radial tree on the discovery tree: the seed in the center, each row on the ring of its depth label,
 * and each subtree in an angular sector proportional to its number of leaves (a radial "cluster").
 */
export function buildLayout(ctx: SampledContext, radius = 260): Layout {
  const seed = ctx.params.seedNode;
  const nodes = new Map<number, LayoutNode>();
  const kids = new Map<number, number[]>();
  nodes.set(seed, { node: seed, parent: null, via: 'seed', depth: 0, firstEvent: -1, pathEvent: -1, first: null, x: 0, y: 0 });
  kids.set(seed, []);

  // Re-apply the frontier with push records (same operations as `replay`), so every visit is tied to the
  // push that produced the popped entry.
  interface PushRec { node: number; from: number; via: 'f2p' | 'p2f'; depth: number; ev: number }
  const stack: (PushRec | null)[] = [null]; // null = the seed's initial entry
  const buckets: PushRec[][] = [];
  const firstPush = new Map<number, PushRec>();
  const pathPush = new Map<number, PushRec>();
  let popped: PushRec | null | undefined;
  ctx.events.forEach((e, i) => {
    switch (e.kind) {
      case 'push-parent':
      case 'push-task-child':
      case 'push-db-child': {
        const rec: PushRec = { node: e.node, from: e.from, via: e.kind === 'push-parent' ? 'f2p' : 'p2f', depth: e.depth, ev: i };
        if (e.kind === 'push-parent') stack.push(rec);
        else {
          while (buckets.length <= e.depth) buckets.push([]);
          buckets[e.depth].push(rec);
        }
        if (!firstPush.has(e.node) && e.node !== seed) firstPush.set(e.node, rec);
        break;
      }
      case 'pop':
        if (e.from === 'stack') popped = stack.pop();
        else {
          const b = buckets[e.depth] ?? [];
          const r = e.draw ?? 0;
          const last = b.length - 1;
          [b[r], b[last]] = [b[last], b[r]];
          popped = b.pop();
        }
        break;
      case 'visit':
        if (popped) pathPush.set(e.node, popped);
        break;
      default:
        break;
    }
  });
  // first-discovery order keeps the angular order stable; parent = the path push for visited rows
  const order = [...firstPush.values()].sort((a, b) => a.ev - b.ev);
  for (const f of order) {
    const p = pathPush.get(f.node) ?? f;
    nodes.set(f.node, {
      node: f.node,
      parent: p.from,
      via: p.via,
      depth: p.depth,
      firstEvent: f.ev,
      pathEvent: p.ev,
      first: p === f ? null : { parent: f.from, via: f.via, depth: f.depth },
      x: 0,
      y: 0,
    });
    kids.set(f.node, []);
  }
  // children lists in the order their path push happened (parents always exist: they were visited earlier)
  for (const n of [...nodes.values()].filter((x) => x.parent !== null).sort((a, b) => a.pathEvent - b.pathEvent)) kids.get(n.parent!)?.push(n.node);

  // leaf weights; visited rows get more room than frontier entries so the rows of the context stay legible
  const visitedSet = new Set(ctx.visited.map((v) => v.node));
  const weight = new Map<number, number>();
  const w = (n: number): number => {
    const c = kids.get(n) ?? [];
    let s = 0;
    for (const k of c) s += w(k);
    const v = Math.max(visitedSet.has(n) ? 3 : 1, s);
    weight.set(n, v);
    return v;
  };
  // iterative-safe: trees here are shallow (≤ ~20 levels), recursion is fine
  w(seed);

  let maxDepth = 1;
  for (const n of nodes.values()) maxDepth = Math.max(maxDepth, n.depth);
  const ring = (d: number) => (d <= 0 ? 0 : (radius * d) / maxDepth);

  const place = (n: number, a0: number, a1: number) => {
    const ln = nodes.get(n)!;
    const a = (a0 + a1) / 2;
    const r = ring(ln.depth);
    ln.x = r * Math.cos(a);
    ln.y = r * Math.sin(a);
    const c = kids.get(n) ?? [];
    const total = c.reduce((s, k) => s + (weight.get(k) ?? 1), 0) || 1;
    let cur = a0;
    for (const k of c) {
      const span = ((a1 - a0) * (weight.get(k) ?? 1)) / total;
      place(k, cur, cur + span);
      cur += span;
    }
  };
  // clockwise from 12 o'clock, leaving a small wedge at the top free for the ring labels
  const gap = 0.42;
  place(seed, -Math.PI / 2 + gap / 2, (3 * Math.PI) / 2 - gap / 2);

  const perDepth: number[] = [];
  for (const n of nodes.values()) perDepth[n.depth] = (perDepth[n.depth] ?? 0) + 1;
  for (let d = 0; d < perDepth.length; d++) perDepth[d] = perDepth[d] ?? 0;
  return { nodes, maxDepth, radius, ring, perDepth };
}

// ─────────────────────────────────────────────────────────────────────────────
// Row naming (display only)
// ─────────────────────────────────────────────────────────────────────────────

const NAME_COLS = ['name', 'display_name', 'title', 'circuit'];

/** "customers" → "customer" (all our table names are regular plurals). */
export function singular(table: string): string {
  if (table.endsWith('ies')) return table.slice(0, -3) + 'y';
  return table.endsWith('s') ? table.slice(0, -1) : table;
}

/** The raw table a node lives in (DB table or one split of a task table). */
export function rawTableOf(db: RawDatabase, pdb: PreprocessedDB, node: number): RawTable | undefined {
  const n = pdb.nodes[node];
  if (!n) return undefined;
  if (n.tableType === 'Db') return db.tables.find((t) => t.schema.name === n.tableName);
  const task = db.tasks.find((t) => t.kind === 'forecast' && t.name === n.tableName);
  return task?.tables?.[n.tableType];
}

export function rawRowOf(db: RawDatabase, pdb: PreprocessedDB, node: number): Record<string, RawValue> | undefined {
  const t = rawTableOf(db, pdb, node);
  const n = pdb.nodes[node];
  return t && n ? t.rows[n.rowPos] : undefined;
}

export interface RowName {
  /** Short, e.g. "customer 35", "order 70", "Train row 258". */
  short: string;
  /** Longer, e.g. "customer 35 (Anna Müller)", "Train row of customer 35 at 2024-06-01". */
  long: string;
  /** table_info key, e.g. "customer-churn:Train". */
  key: string;
}

const nameCache = new WeakMap<PreprocessedDB, Map<number, RowName>>();

/** Display names of a row (cached per preprocessed DB; the rows never change). */
export function rowName(db: RawDatabase, pdb: PreprocessedDB, node: number): RowName {
  let m = nameCache.get(pdb);
  if (!m) nameCache.set(pdb, (m = new Map()));
  let r = m.get(node);
  if (!r) {
    r = computeRowName(db, pdb, node);
    m.set(node, r);
  }
  return r;
}

function computeRowName(db: RawDatabase, pdb: PreprocessedDB, node: number): RowName {
  const n = pdb.nodes[node];
  if (!n) return { short: `#${node}`, long: `row #${node}`, key: '' };
  const raw = rawRowOf(db, pdb, node);
  if (n.tableType === 'Db') {
    const short = `${singular(n.tableName)} ${n.rowPos}`;
    const nameCol = NAME_COLS.find((c) => raw && typeof raw[c] === 'string');
    const extra = nameCol ? String(raw![nameCol]) : n.ts !== null ? formatDate(n.ts * 1000, 'never') : '';
    return { short, long: extra ? `${short} (${extra})` : short, key: n.tableKey };
  }
  const parent = n.f2pNbrIdxs[0];
  const p = parent !== undefined ? pdb.nodes[parent] : undefined;
  const ent = p ? `${singular(p.tableName)} ${p.rowPos}` : '';
  const date = n.ts !== null ? formatDate(n.ts * 1000, 'never') : '';
  const extra = [ent, date].filter(Boolean).join(', ');
  return {
    short: `${n.tableType} row ${n.rowPos}`,
    long: `${n.tableName} ${n.tableType} row ${n.rowPos}${extra ? ` (${extra})` : ''}`,
    key: n.tableKey,
  };
}

/** Seconds → "YYYY-MM-DD". */
export function tsDate(ts: number | null): string {
  return ts === null ? 'no timestamp' : formatDate(ts * 1000, 'never');
}

/** "3 cells" / "1 cell". */
export function plural(n: number, word: string, pluralWord?: string): string {
  return `${n.toLocaleString('en-US')} ${n === 1 ? word : (pluralWord ?? `${word}s`)}`;
}

/** Events of one iteration split by phase (convenience for the narration and detail panels). */
export interface IterationEvents {
  it: Iteration;
  pop: Extract<SamplerEvent, { kind: 'pop' }>;
  parents: Extract<SamplerEvent, { kind: 'push-parent' }>[];
  otherTask: Extract<SamplerEvent, { kind: 'child-other-task-table' }>[];
  future: Extract<SamplerEvent, { kind: 'child-future' }>[];
  taskKids: Extract<SamplerEvent, { kind: 'push-task-child' }>[];
  subsample: Extract<SamplerEvent, { kind: 'db-children-subsampled' }> | null;
  dbKids: Extract<SamplerEvent, { kind: 'push-db-child' }>[];
  emits: Extract<SamplerEvent, { kind: 'emit' }>[];
  drops: Extract<SamplerEvent, { kind: 'drop-leakage-cell' }>[];
  truncated: boolean;
}

export function iterationEvents(ctx: SampledContext, it: Iteration): IterationEvents {
  const out: IterationEvents = {
    it,
    pop: ctx.events[it.popAt] as Extract<SamplerEvent, { kind: 'pop' }>,
    parents: [],
    otherTask: [],
    future: [],
    taskKids: [],
    subsample: null,
    dbKids: [],
    emits: [],
    drops: [],
    truncated: false,
  };
  for (let i = it.popAt; i < it.end; i++) {
    const e = ctx.events[i];
    switch (e.kind) {
      case 'push-parent':
        out.parents.push(e);
        break;
      case 'child-other-task-table':
        out.otherTask.push(e);
        break;
      case 'child-future':
        out.future.push(e);
        break;
      case 'push-task-child':
        out.taskKids.push(e);
        break;
      case 'db-children-subsampled':
        out.subsample = e;
        break;
      case 'push-db-child':
        out.dbKids.push(e);
        break;
      case 'emit':
        out.emits.push(e);
        break;
      case 'drop-leakage-cell':
        out.drops.push(e);
        break;
      case 'truncated':
        out.truncated = true;
        break;
      default:
        break;
    }
  }
  return out;
}

/** DB children that passed the filters (= the candidates of the width bound), in edge order. */
export function dbCandidates(ie: IterationEvents): number[] {
  return ie.subsample ? ie.subsample.candidates : ie.dbKids.map((e) => e.node);
}

// ─────────────────────────────────────────────────────────────────────────────
// Context summaries (final context)
// ─────────────────────────────────────────────────────────────────────────────

export interface TableCount {
  key: string;
  tableName: string;
  tableType: string;
  rows: number;
  tokens: number;
}

/** Rows and tokens per table_info key, in order of first appearance. */
export function countByTable(pdb: PreprocessedDB, ctx: SampledContext): TableCount[] {
  const m = new Map<string, TableCount>();
  const rowsSeen = new Set<number>();
  for (const t of ctx.tokens) {
    if (t.isPadding) continue;
    const n = pdb.nodes[t.nodeIdx];
    let c = m.get(n.tableKey);
    if (!c) {
      c = { key: n.tableKey, tableName: n.tableName, tableType: n.tableType, rows: 0, tokens: 0 };
      m.set(n.tableKey, c);
    }
    c.tokens++;
    if (!rowsSeen.has(t.nodeIdx)) {
      rowsSeen.add(t.nodeIdx);
      c.rows++;
    }
  }
  return [...m.values()];
}

/** The row cut by the hard stop at L, if any: emitted vs total (non-dropped) cells. */
export function cutRow(ctx: SampledContext, dropCols: readonly number[], pdb: PreprocessedDB): { node: number; emitted: number; total: number } | null {
  const tr = ctx.events.find((e) => e.kind === 'truncated');
  if (!tr || tr.kind !== 'truncated') return null;
  const node = pdb.nodes[tr.node];
  const seed = pdb.nodes[ctx.params.seedNode];
  const drop = new Set(dropCols);
  const total = node.cells.filter((c) => !(drop.has(c.colNameIdx) && (node.nodeIdx === seed.nodeIdx || node.ts === seed.ts))).length;
  const emitted = ctx.tokens.filter((t) => !t.isPadding && t.nodeIdx === tr.node).length;
  return emitted < total ? { node: tr.node, emitted, total } : null;
}

/** Depth label of every visited row of the final context. */
export function maxVisitedDepth(ctx: SampledContext): number {
  let m = 0;
  for (const v of ctx.visited) m = Math.max(m, v.depth);
  return m;
}

/** Re-export for local components. */
export type { TaskDef };

/**
 * Axis tick labels that fit without overlapping: the first (anchored start) and the last (anchored end)
 * win, then the middle ones (anchored middle) where there is room. Labels keep their font size.
 */
export function fitTicks(
  ticks: { x: number; text: string }[],
  width: (text: string) => number,
  gap = 10,
): { x: number; text: string; anchor: 'start' | 'middle' | 'end' }[] {
  if (ticks.length === 0) return [];
  const n = ticks.length;
  const anchorOf = (i: number): 'start' | 'middle' | 'end' => (i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle');
  const span = (i: number): [number, number] => {
    const w = width(ticks[i].text);
    const a = anchorOf(i);
    return a === 'start' ? [ticks[i].x, ticks[i].x + w] : a === 'end' ? [ticks[i].x - w, ticks[i].x] : [ticks[i].x - w / 2, ticks[i].x + w / 2];
  };
  const order = [0, n - 1, ...Array.from({ length: Math.max(0, n - 2) }, (_, i) => i + 1)].filter((v, i, arr) => arr.indexOf(v) === i);
  const kept: number[] = [];
  for (const i of order) {
    const [a, b] = span(i);
    if (kept.every((k) => {
      const [c, d] = span(k);
      return b + gap <= c || d + gap <= a;
    }))
      kept.push(i);
  }
  return kept.sort((a, b) => a - b).map((i) => ({ ...ticks[i], anchor: anchorOf(i) }));
}
