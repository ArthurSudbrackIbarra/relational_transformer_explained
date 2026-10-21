/**
 * Live data shared by the "tokens" chapter's figures: the global context window (fly.rs mirror),
 * its forward pass and prediction, plus small display helpers.
 *
 * The inspected token is the store's `selectedPos` (shared with the masks and attention chapters as
 * "the token being inspected"); null means the masked target. Hover state stays local to each figure.
 */
import { useCallback, useMemo } from 'react';
import type { InContextLabels } from '@/engine/pipeline';
import type { GraphNode, PreprocessedDB, RawDatabase, RawTable, SampledContext, TaskDef, Token } from '@/engine/types';
import {
  useAssets,
  useContextWindow,
  useDatabase,
  useForward,
  useInContextLabels,
  usePrediction,
  usePreprocessed,
  useSeedNode,
  useTableSlots,
  useTask,
} from '@/state/hooks';
import { useSim } from '@/state/store';

export function useTokensLive() {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const task = useTask();
  const seedNode = useSeedNode();
  const ctx = useContextWindow();
  const slots = useTableSlots();
  const assets = useAssets();
  const fr = useForward();
  const pred = usePrediction();
  const labels = useInContextLabels();
  const selectedPos = useSim((s) => s.selectedPos);
  const setSelectedPos = useSim((s) => s.setSelectedPos);

  const S = ctx.tokens.length;
  const target = ctx.targetPos >= 0 ? ctx.targetPos : 0;
  const valid = selectedPos !== null && selectedPos >= 0 && selectedPos < S;
  const inspected = valid ? selectedPos : target;
  const setInspected = useCallback(
    (pos: number | null) => setSelectedPos(pos === null || pos === target ? null : pos),
    [setSelectedPos, target],
  );
  /** Inspect a token and bring the Step 1 inspector into view (for links from later sections). */
  const inspectAndGo = useCallback(
    (pos: number) => {
      setInspected(pos);
      const reduce = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
      requestAnimationFrame(() => document.getElementById(INSPECTOR_ID)?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' }));
    },
    [setInspected],
  );

  const seed = pdb.nodes[seedNode];
  const entityNode = task.kind === 'forecast' && seed && seed.f2pNbrIdxs.length ? seed.f2pNbrIdxs[0] : null;

  /** nodeIdx → positions of its tokens in the context (emission order). */
  const tokensByNode = useMemo(() => {
    const m = new Map<number, number[]>();
    for (const t of ctx.tokens) {
      if (t.isPadding) continue;
      let a = m.get(t.nodeIdx);
      if (!a) m.set(t.nodeIdx, (a = []));
      a.push(t.pos);
    }
    return m;
  }, [ctx]);

  return {
    db,
    pdb,
    task,
    seedNode,
    seed,
    entityNode,
    ctx,
    S,
    slots,
    assets,
    fr,
    pred,
    labels,
    target,
    inspected,
    setInspected,
    inspectAndGo,
    tokensByNode,
  };
}

/** DOM id of the Step 1 token inspector figure (scroll target of `inspectAndGo`). */
export const INSPECTOR_ID = 'token-inspector';

/**
 * The visible label every figure of the chapter compares with the target: the most recent label of the
 * seed's own entity if there is one, else the first label of another row. Display choice only.
 */
export function pickVisibleLabel(labels: InContextLabels): { pos: number; kind: 'self' | 'other' } | null {
  const self = [...labels.self].sort((a, b) => (b.ts ?? 0) - (a.ts ?? 0))[0];
  if (self) return { pos: self.pos, kind: 'self' };
  return labels.other[0] ? { pos: labels.other[0].pos, kind: 'other' } : null;
}

export type TokensLive = ReturnType<typeof useTokensLive>;

// ─────────────────────────────────────────────────────────────────────────────
// Display helpers (never read by the model)
// ─────────────────────────────────────────────────────────────────────────────

/** "customers[35]" or "customer-churn:Val[32]". */
export function nodeRef(node: { tableName: string; tableType: GraphNode['tableType'] | null; rowPos: number }): string {
  return `${node.tableName}${node.tableType && node.tableType !== 'Db' ? `:${node.tableType}` : ''}[${node.rowPos}]`;
}

/** Same as nodeRef for a token (padding → "padding"). */
export function tokenRowRef(t: Token): string {
  return t.isPadding ? 'padding' : nodeRef({ tableName: t.tableName, tableType: t.tableType, rowPos: t.rowPos });
}

/** The first text cell of a row (e.g. a customer's name), if any, for friendly labels. */
export function nodeNickname(node: GraphNode | undefined): string | null {
  if (!node) return null;
  const c = node.cells.find((x) => typeof x.raw === 'string');
  return c ? String(c.raw) : null;
}

/** "customers[35] · Anna Müller". */
export function nodeLabel(pdb: PreprocessedDB, nodeIdx: number): string {
  const n = pdb.nodes[nodeIdx];
  if (!n) return `node ${nodeIdx}`;
  const nick = nodeNickname(n);
  return nick ? `${nodeRef(n)} · ${nick}` : nodeRef(n);
}

/** The RawTable a token's row lives in (a DB table, or one split of a forecast task table). */
export function sourceTable(db: RawDatabase, tableName: string, tableType: string | null): RawTable | undefined {
  if (!tableType || tableType === 'Db') return db.tables.find((t) => t.schema.name === tableName);
  const task = db.tasks.find((t) => t.kind === 'forecast' && t.name === tableName);
  return task?.tables?.[tableType as 'Train' | 'Val' | 'Test'];
}

/** Seconds → "YYYY-MM-DD". */
export function tsDate(ts: number | null | undefined): string {
  if (ts === null || ts === undefined) return '—';
  return new Date(ts * 1000).toISOString().slice(0, 10);
}

/**
 * Cells the sampler would emit for a visited row (fly.rs:454-461): every stored cell, minus
 * leakage columns when the row is the seed or shares the seed's timestamp (null == null counts).
 */
export function emittableCells(pdb: PreprocessedDB, ctx: SampledContext, nodeIdx: number): number {
  const node = pdb.nodes[nodeIdx];
  const seed = pdb.nodes[ctx.params.seedNode];
  if (!node || !seed) return 0;
  const drop = ctx.params.dropColNameIdxs;
  return node.cells.filter((c) => !(drop.includes(c.colNameIdx) && (node.nodeIdx === seed.nodeIdx || node.ts === seed.ts))).length;
}

/** The last real row of a full context, if the hard stop at L cut it (fly.rs:502-509). */
export function cutRow(pdb: PreprocessedDB, ctx: SampledContext): { node: number; emitted: number; total: number } | null {
  if (ctx.numReal < ctx.tokens.length) return null;
  const last = ctx.tokens[ctx.numReal - 1];
  if (!last || last.isPadding) return null;
  const emitted = ctx.tokens.filter((t) => !t.isPadding && t.nodeIdx === last.nodeIdx).length;
  const total = emittableCells(pdb, ctx, last.nodeIdx);
  return emitted < total ? { node: last.nodeIdx, emitted, total } : null;
}

/**
 * Short graph tags for the rows of a context: "c35", "o70" for DB rows (shortest unique table-name
 * prefix), "val32", "train564" for task rows. Display only.
 */
export function rowTags(pdb: PreprocessedDB, ctx: SampledContext): Map<number, string> {
  const dbTables = [...new Set(ctx.tokens.filter((t) => !t.isPadding && (!t.tableType || t.tableType === 'Db')).map((t) => t.tableName))];
  const prefix: Record<string, string> = {};
  for (const name of dbTables) {
    let k = 1;
    while (k < name.length && dbTables.some((o) => o !== name && o.startsWith(name.slice(0, k)))) k++;
    prefix[name] = name.slice(0, k);
  }
  const out = new Map<number, string>();
  for (const t of ctx.tokens) {
    if (t.isPadding || out.has(t.nodeIdx)) continue;
    const n = pdb.nodes[t.nodeIdx];
    out.set(t.nodeIdx, n.tableType && n.tableType !== 'Db' ? `${n.tableType.toLowerCase()}${n.rowPos}` : `${prefix[n.tableName] ?? n.tableName[0]}${n.rowPos}`);
  }
  return out;
}

/** Short description of the task for prose. */
export function taskShort(task: TaskDef): string {
  return task.id.split('/')[1] ?? task.name;
}

export const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString('en-US')} ${n === 1 ? one : many}`;
