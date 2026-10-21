/**
 * Live simulation data for chapter 5, read from the global store through the shared hooks, so this
 * chapter shows the same database, task, seed and context as every other chapter.
 */
import { useMemo } from 'react';
import type { DbName } from '@/engine/db';
import { SemType, type Token } from '@/engine/types';
import { useContextWindow, useDatabase, usePreprocessed, useSeedNode, useTableSlots, useTask } from '@/state/hooks';

export function useSemanticsLive() {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const task = useTask();
  const ctx = useContextWindow();
  const slots = useTableSlots();
  const seedNode = useSeedNode();

  return useMemo(() => {
    const dbName = db.name as DbName;
    const real = ctx.tokens.filter((t) => !t.isPadding);
    const target = ctx.targetPos >= 0 ? ctx.tokens[ctx.targetPos] : undefined;
    const targetTable = task.kind === 'autocomplete' ? task.table ?? task.name : task.name;
    const targetPhrase = `${task.targetCol} of ${targetTable}`;

    /** Number of live tokens carrying each column phrase. */
    const tokensPerPhrase = new Map<string, number>();
    for (const t of real) {
      const p = `${t.colName} of ${t.tableName}`;
      tokensPerPhrase.set(p, (tokensPerPhrase.get(p) ?? 0) + 1);
    }

    /** Text tokens of the live context, in sequence order. */
    const textTokens: Token[] = real.filter((t) => t.sem === SemType.Text);

    const seed = seedNode >= 0 ? pdb.nodes[seedNode] : undefined;
    const entityNode = task.kind === 'forecast' && seed && seed.f2pNbrIdxs.length ? pdb.nodes[seed.f2pNbrIdxs[0]] : undefined;
    const entityName = entityNode?.cells.find((c) => c.sem === SemType.Text)?.raw;

    return {
      db,
      dbName,
      pdb,
      task,
      ctx,
      slots,
      seed,
      entityNode,
      entityName: typeof entityName === 'string' ? entityName : undefined,
      target,
      targetPhrase,
      tokensPerPhrase,
      textTokens,
      numReal: real.length,
    };
  }, [db, pdb, task, ctx, slots, seedNode]);
}

export type SemanticsLive = ReturnType<typeof useSemanticsLive>;

/** "customers[35]" or "customer-churn:Val[32]" (display only). */
export function rowRef(t: { tableName: string; tableType: string | null; rowPos: number }): string {
  return `${t.tableName}${t.tableType && t.tableType !== 'Db' ? `:${t.tableType}` : ''}[${t.rowPos}]`;
}
