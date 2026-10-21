/**
 * Live facts of the running example for the overview chapter.
 *
 * Everything comes from the shared simulation hooks (src/state/hooks.ts), so this chapter shows
 * the same seed, context and forward pass as every other chapter. Nothing here adds model or
 * sampler semantics; it only names rows for the prose.
 */
import { useMemo } from 'react';
import type { GraphNode, PreprocessedDB, RawDatabase, RawValue, TaskDef } from '@/engine/types';
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
  useTask,
} from '@/state/hooks';
import { formatDate } from '@/components';

/** "customers" → "customer" (display only). */
export function singular(table: string): string {
  return table.endsWith('s') ? table.slice(0, -1) : table;
}

/** The raw record behind a node (DB row or task-table row), for display only. */
export function rawRow(db: RawDatabase, pdb: PreprocessedDB, node: number): Record<string, RawValue> | null {
  const n = pdb.nodes[node];
  if (!n) return null;
  if (n.tableType === 'Db') return db.tables.find((t) => t.schema.name === n.tableName)?.rows[n.rowPos] ?? null;
  const task = db.tasks.find((t) => t.kind === 'forecast' && t.name === n.tableName);
  return task?.tables?.[n.tableType]?.rows[n.rowPos] ?? null;
}

/** A human name for a DB row: its name/title column if it has one. */
export function rowTitle(db: RawDatabase, pdb: PreprocessedDB, node: number): string | null {
  const n = pdb.nodes[node];
  if (!n || n.tableType !== 'Db') return null;
  const t = db.tables.find((x) => x.schema.name === n.tableName);
  const rec = t?.rows[n.rowPos];
  if (!t || !rec) return null;
  const col = t.schema.columns.find((c) => c.dtype === 'str' && /name|title/.test(c.name) && typeof rec[c.name] === 'string');
  return col ? String(rec[col.name]) : null;
}

export interface RowLabel {
  /** "customer 35", "Val row 32". */
  short: string;
  /** "customer 35 (Anna Müller)", "customer-churn Val row 32 (customer 35, 2025-03-01)". */
  long: string;
}

export function rowLabel(db: RawDatabase, pdb: PreprocessedDB, node: number): RowLabel {
  const n = pdb.nodes[node];
  if (!n) return { short: `#${node}`, long: `row #${node}` };
  if (n.tableType === 'Db') {
    const short = `${singular(n.tableName)} ${n.rowPos}`;
    const title = rowTitle(db, pdb, node);
    const extra = title ?? (n.ts !== null ? formatDate(n.ts * 1000, 'never') : '');
    return { short, long: extra ? `${short} (${extra})` : short };
  }
  const parent = n.f2pNbrIdxs[0];
  const p = parent !== undefined ? pdb.nodes[parent] : undefined;
  const bits = [p ? `${singular(p.tableName)} ${p.rowPos}` : '', n.ts !== null ? formatDate(n.ts * 1000, 'never') : ''].filter(Boolean);
  return {
    short: `${n.tableType} row ${n.rowPos}`,
    long: `${n.tableName} ${n.tableType} row ${n.rowPos}${bits.length ? ` (${bits.join(', ')})` : ''}`,
  };
}

/** "2025-03-01" from a node timestamp in seconds. */
export function tsDate(ts: number | null | undefined): string {
  return ts === null || ts === undefined ? 'no timestamp' : formatDate(ts * 1000, 'never');
}

export const int = (n: number) => n.toLocaleString('en-US');

export function plural(n: number, word: string, pluralWord?: string): string {
  return `${int(n)} ${n === 1 ? word : (pluralWord ?? `${word}s`)}`;
}

/** Join ["a", "b", "c"] → "a, b and c". */
export function listJoin(items: string[]): string {
  if (items.length <= 1) return items.join('');
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

/** Display name of the task's target, e.g. "churn". */
export function targetName(task: TaskDef): string {
  return task.targetCol;
}

/** The whole live state the overview needs, from the shared hooks. */
export function useOverviewLive() {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const task = useTask();
  const seedNode = useSeedNode();
  const ctx = useContextWindow();
  const masks = useMasks();
  const labels = useInContextLabels();
  const slots = useTableSlots();
  const assets = useAssets();
  const fr = useForward();
  const pred = usePrediction();

  const facts = useMemo(() => {
    const seed: GraphNode | undefined = pdb.nodes[seedNode];
    const entityNode = task.kind === 'forecast' && seed && seed.f2pNbrIdxs.length ? seed.f2pNbrIdxs[0] : null;
    const entity = entityNode !== null ? pdb.nodes[entityNode] : undefined;
    const entityTitle = entityNode !== null ? rowTitle(db, pdb, entityNode) : null;
    const entityRec = entityNode !== null ? rawRow(db, pdb, entityNode) : null;
    const country = entityRec && typeof entityRec.country === 'string' ? entityRec.country : null;
    const entityShort = entityNode !== null ? rowLabel(db, pdb, entityNode).short : null;
    /** "Anna Müller (customer 35)" or "customer 35". */
    const entityName = entityShort ? (entityTitle ? `${entityTitle} (${entityShort})` : entityShort) : null;
    /** First name for friendly prose, when the entity has a person-like name. */
    const firstName = entityTitle && /\s/.test(entityTitle) ? entityTitle.split(/\s+/)[0] : entityTitle;
    const rowsByTable = new Map<string, number>();
    for (const v of ctx.visited) {
      const n = pdb.nodes[v.node];
      const k = n.tableType === 'Db' ? n.tableName : `${n.tableName}:${n.tableType}`;
      rowsByTable.set(k, (rowsByTable.get(k) ?? 0) + 1);
    }
    return {
      seed,
      seedLabel: seed ? rowLabel(db, pdb, seedNode) : null,
      seedDate: tsDate(seed?.ts),
      entityNode,
      entity,
      entityTitle,
      entityShort,
      entityName,
      firstName,
      country,
      rowsByTable,
      target: ctx.targetPos >= 0 ? ctx.tokens[ctx.targetPos] : undefined,
    };
  }, [db, pdb, task, seedNode, ctx]);

  return {
    db,
    pdb,
    task,
    seedNode,
    ctx,
    masks,
    labels,
    slots,
    assets,
    fr,
    pred,
    isRunningExample: task.id === 'shop/customer-churn',
    ...facts,
  };
}

export type OverviewLive = ReturnType<typeof useOverviewLive>;
