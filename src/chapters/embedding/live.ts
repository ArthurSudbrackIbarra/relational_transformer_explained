/**
 * Live data for the "embedding" chapter: the global context window and its forward pass (whose
 * `embedding` capture holds the column term, value term and x₀ of every token, model.py:208-224),
 * plus the weights and the embedding store for the extra read-outs.
 *
 * The inspected token is the store's `selectedPos`, shared with the tokens, masks and attention
 * chapters as "the token being inspected" (session-scoped, not persisted); null = the masked target.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { loadWeights } from '@/engine/model';
import type { RTWeights, Token } from '@/engine/types';
import { SemType } from '@/engine/types';
import {
  useAssets,
  useContextWindow,
  useDatabase,
  useForward,
  usePreprocessed,
  useSeedNode,
  useTableSlots,
  useTask,
} from '@/state/hooks';
import { useSim } from '@/state/store';
import { featureColumns } from './math';

export function useEmbeddingLive() {
  const db = useDatabase();
  const pdb = usePreprocessed();
  const task = useTask();
  const seedNode = useSeedNode();
  const ctx = useContextWindow();
  const slots = useTableSlots();
  const assets = useAssets();
  const fr = useForward();
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

  const columns = useMemo(() => featureColumns(db, pdb), [db, pdb]);

  /** The first position of each sem type in the context (presets for the token picker). */
  const firstOfType = useMemo(() => {
    const out: Partial<Record<SemType, number>> = {};
    for (const t of ctx.tokens) {
      if (t.isPadding || t.isTarget) continue;
      if (out[t.sem] === undefined) out[t.sem] = t.pos;
    }
    return out;
  }, [ctx]);

  /** Which of the eight value gates of model.py:216-224 is 1 for each token (padding: none). */
  const gates = useMemo(() => {
    const value: number[][] = [[], [], [], []];
    const mask: number[][] = [[], [], [], []];
    const padding: number[] = [];
    for (const t of ctx.tokens) {
      if (t.isPadding) padding.push(t.pos);
      else (t.mask ? mask : value)[t.sem].push(t.pos);
    }
    return { value, mask, padding };
  }, [ctx]);

  /** Feature columns that can appear in this context: DB tables + the active task table only. */
  const contextColumns = useMemo(
    () => columns.filter((c) => !c.isTaskTable || c.table === task.name),
    [columns, task],
  );

  const padCount = S - ctx.numReal;
  const weights = assets.weights;
  const store = assets.store;
  const ready = assets.status === 'ready' && !!weights && !!store && !!fr;

  return {
    db,
    pdb,
    task,
    seedNode,
    ctx,
    S,
    slots,
    assets,
    weights,
    store,
    fr,
    ready,
    target,
    inspected,
    setInspected,
    columns,
    contextColumns,
    gates,
    firstOfType,
    padCount,
  };
}

export type EmbeddingLive = ReturnType<typeof useEmbeddingLive>;

/**
 * The exported untrained initialisation of our mini model (public/data/weights/init.*, PyTorch
 * defaults, spec §7.4), for "where training started" read-outs. null while loading or if missing.
 */
export function useInitWeights(): RTWeights | null {
  const [w, setW] = useState<RTWeights | null>(null);
  useEffect(() => {
    let alive = true;
    loadWeights('init').then(
      (x) => alive && setW(x),
      () => alive && setW(null),
    );
    return () => {
      alive = false;
    };
  }, []);
  return w;
}

// ─────────────────────────────────────────────────────────────────────────────
// Display helpers (never read by the model)
// ─────────────────────────────────────────────────────────────────────────────

/** "customers[35]" or "customer-churn:Val[32]". */
export function rowRef(t: Pick<Token, 'tableName' | 'tableType' | 'rowPos' | 'isPadding'>): string {
  if (t.isPadding) return 'padding';
  return `${t.tableName}${t.tableType && t.tableType !== 'Db' ? `:${t.tableType}` : ''}[${t.rowPos}]`;
}

/** The column phrase of a token, exactly as interned by pre.rs:432-437: "<col> of <table>". */
export function phraseOf(t: Pick<Token, 'colName' | 'tableName'>): string {
  return `${t.colName} of ${t.tableName}`;
}

export const SEM_LABEL: Record<SemType, string> = {
  [SemType.Number]: 'number',
  [SemType.Text]: 'text',
  [SemType.DateTime]: 'datetime',
  [SemType.Boolean]: 'boolean',
};

/** The batch field that carries the value of a sem type (spec §5.1). */
export const VALUE_FIELD: Record<SemType, string> = {
  [SemType.Number]: 'number_values',
  [SemType.Text]: 'text_values',
  [SemType.DateTime]: 'datetime_values',
  [SemType.Boolean]: 'boolean_values',
};

export const fmt = (v: number, digits = 3) => {
  if (!Number.isFinite(v)) return '—';
  const s = Math.abs(v).toFixed(digits);
  return v < 0 && Number(s) !== 0 ? `−${s}` : s;
};

export const fmtSigned = (v: number, digits = 3) => {
  if (!Number.isFinite(v)) return '—';
  const s = Math.abs(v).toFixed(digits);
  return Number(s) === 0 ? `${s}` : `${v < 0 ? '−' : '+'}${s}`;
};

export const int = (n: number) => n.toLocaleString('en-US');
