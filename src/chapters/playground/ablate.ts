/**
 * Inference-time ablations of one sampled context, built only from the engine's own mirrors
 * (sampleContext → tokens, buildMasks = model.py:164-191, forwardWithMasks = model.py:153-267).
 * Nothing here changes what the model computes; it only changes what goes INTO it:
 *
 *  - edit an in-context label: the label cell's stored value is replaced by the new value z-scored
 *    with the same Train statistics that pre.rs used for every label of that task (pre.rs:553-569,
 *    spec §3.3; Val/Test task rows reuse Train stats);
 *  - remove self / other-entity labels: those tokens are dropped from the token list before the
 *    forward. Padding is invisible to every mask on both sides (model.py:165, 186-191), so dropping a
 *    token is the same as turning it into padding. The context is not refilled: the paper applies its
 *    zero-shot ablations "directly to the sampled context used as input" (p.22 App. G). Its exact code
 *    is unreleased (spec §13 #3);
 *  - shuffle column names: the 384-d column-phrase embeddings are permuted among the columns present
 *    in the context. The masks keep using the true column ids, because the sampler stores the id
 *    (col_name_idxs, for the masks) and the embedding (col_name_values, for the input) separately
 *    (fly.rs:469-477, model.py:181-183, 209-214);
 *  - disable attention layers: the residual update of that sublayer is skipped in every block
 *    (model.py:83-84); the paper instead retrains without the layer (p.9 §5.2).
 */
import type { EmbeddingStore } from '@/engine/embeddings';
import { buildMasks } from '@/engine/masks';
import { normalize } from '@/engine/metrics';
import { forwardWithMasks, textEmbedder, type RTForwardResult, type TextEmbedder } from '@/engine/model';
import { inContextLabels, type InContextLabels } from '@/engine/pipeline';
import { Rng } from '@/engine/rng';
import { targetStats } from '@/engine/sampler';
import type { AttnLayer, ColumnStats, PreprocessedDB, RawValue, RTWeights, SampledContext, TaskDef, Token } from '@/engine/types';
import { SemType } from '@/engine/types';

export type RemoveWhat = 'cells' | 'rows';

/** The token-level ablations (everything except L, checkpoint and disabled layers). */
export interface TokenAblation {
  removeSelf: boolean;
  removeOther: boolean;
  /** 'cells' = only the label cells (Table 11/12 wording); 'rows' = every cell of those task rows (§5.1 wording). */
  removeWhat: RemoveWhat;
  shuffle: boolean;
  shuffleSeed: number;
  /** nodeIdx of a row whose target-column cell is visible → its new raw value. */
  edits: ReadonlyMap<number, RawValue>;
}

export const NO_TOKEN_ABLATION: TokenAblation = {
  removeSelf: false,
  removeOther: false,
  removeWhat: 'cells',
  shuffle: false,
  shuffleSeed: 1,
  edits: new Map(),
};

export type LabelRole = 'self' | 'other';

export interface AblatedContext {
  /** The token list fed to the model (dropped tokens removed, edits applied, `pos` re-indexed). */
  tokens: Token[];
  /** tokens[i] came from ctx.tokens[origPos[i]]. */
  origPos: number[];
  /** Original positions that were dropped. */
  removedPos: ReadonlySet<number>;
  /** Original positions whose label value was edited. */
  editedPos: ReadonlySet<number>;
  /** In-context labels of the ORIGINAL context (spec §2.5). */
  labels: InContextLabels;
  /** Original position → self / other label role (label cells only). */
  labelRole: ReadonlyMap<number, LabelRole>;
  /** Column-phrase permutation (text idx of a phrase → text idx whose embedding is used), or null. */
  perm: ReadonlyMap<number, number> | null;
  /** Position of the masked target in `tokens`. */
  targetPos: number;
  /** True when nothing token-level changed (tokens === ctx.tokens). */
  untouched: boolean;
}

/** Train stats of the target column (also used for Val/Test task rows, pre.rs:383-396). */
export function labelStats(pdb: PreprocessedDB, task: TaskDef): ColumnStats {
  return targetStats(pdb, task);
}

/** The stored (z-scored, f32) value of a label cell for a raw value, exactly as pre.rs:553-569 computes it. */
export function encodeLabel(raw: RawValue, task: TaskDef, stats: ColumnStats): number {
  if (task.targetType === 'binary') return Math.fround(((raw === true ? 1 : 0) - stats.mean) / stats.std);
  return Math.fround(normalize(Number(raw), stats));
}

/** The entity node of a forecast task row (its single FK parent), or null. */
function entityOf(pdb: PreprocessedDB, node: number): number | null {
  const n = pdb.nodes[node];
  return n && n.f2pNbrIdxs.length > 0 ? n.f2pNbrIdxs[0] : null;
}

/** Sattolo's algorithm: a uniformly random cyclic permutation, so every column gets another column's name. */
function cyclicPermutation(items: number[], rng: Rng): number[] {
  const a = items.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = rng.int(i);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** PRNG seed of a shuffle: the reader's shuffle number and the seed row (one draw per context). */
export function shuffleRngSeed(shuffleSeed: number, seedNode: number): number {
  return (Math.imul(shuffleSeed + 1, 2654435761) ^ (seedNode * 40503)) >>> 0;
}

/**
 * Apply the token-level ablations to a sampled context. Pure; never mutates `ctx`.
 */
export function ablateContext(pdb: PreprocessedDB, task: TaskDef, ctx: SampledContext, ab: TokenAblation): AblatedContext {
  const labels = inContextLabels(pdb, ctx, task);
  const labelRole = new Map<number, LabelRole>();
  for (const l of labels.self) labelRole.set(l.pos, 'self');
  for (const l of labels.other) labelRole.set(l.pos, 'other');

  // ── which tokens are dropped ──────────────────────────────────────────────
  const removed = new Set<number>();
  if (task.kind === 'forecast' && (ab.removeSelf || ab.removeOther)) {
    if (ab.removeWhat === 'cells') {
      for (const [pos, role] of labelRole) if ((role === 'self' && ab.removeSelf) || (role === 'other' && ab.removeOther)) removed.add(pos);
    } else {
      // every cell of a task row of the seed's task table (any split), except the seed row itself
      const seed = pdb.nodes[ctx.params.seedNode];
      const seedEntity = entityOf(pdb, seed.nodeIdx);
      for (const t of ctx.tokens) {
        if (t.isPadding || t.nodeIdx === seed.nodeIdx) continue;
        const n = pdb.nodes[t.nodeIdx];
        if (!n || n.tableType === 'Db' || n.tableNameIdx !== seed.tableNameIdx) continue;
        const self = seedEntity !== null && entityOf(pdb, n.nodeIdx) === seedEntity;
        if ((self && ab.removeSelf) || (!self && ab.removeOther)) removed.add(t.pos);
      }
    }
  }

  // ── edits ─────────────────────────────────────────────────────────────────
  const edited = new Set<number>();
  const stats = ab.edits.size > 0 ? labelStats(pdb, task) : null;
  const editOf = new Map<number, RawValue>();
  if (stats) {
    for (const [pos] of labelRole) {
      if (removed.has(pos)) continue;
      const t = ctx.tokens[pos];
      if (!ab.edits.has(t.nodeIdx)) continue;
      const v = ab.edits.get(t.nodeIdx)!;
      if (v === t.raw) continue;
      editOf.set(pos, v);
      edited.add(pos);
    }
  }

  const untouchedTokens = removed.size === 0 && edited.size === 0;
  const tokens: Token[] = [];
  const origPos: number[] = [];
  if (untouchedTokens) {
    for (const t of ctx.tokens) {
      tokens.push(t);
      origPos.push(t.pos);
    }
  } else {
    for (const t of ctx.tokens) {
      if (removed.has(t.pos)) continue;
      let nt: Token = { ...t, pos: tokens.length };
      const v = editOf.get(t.pos);
      if (v !== undefined && stats) {
        const z = encodeLabel(v, task, stats);
        nt = task.targetType === 'binary' ? { ...nt, raw: v, booleanValue: z } : { ...nt, raw: v, numberValue: z };
      }
      tokens.push(nt);
      origPos.push(t.pos);
    }
  }

  // ── column-name shuffle ───────────────────────────────────────────────────
  let perm: Map<number, number> | null = null;
  if (ab.shuffle) {
    const textValueIdxs = new Set<number>();
    for (const t of tokens) if (!t.isPadding && t.sem === SemType.Text) textValueIdxs.add(t.textIdx);
    const cols = [...new Set(tokens.filter((t) => !t.isPadding).map((t) => t.colNameIdx))]
      .filter((c) => !textValueIdxs.has(c)) // guard: never remap a string that is also a text value
      .sort((a, b) => a - b);
    if (cols.length >= 2) {
      const shuffled = cyclicPermutation(cols, new Rng(shuffleRngSeed(ab.shuffleSeed, ctx.params.seedNode)));
      perm = new Map(cols.map((c, i) => [c, shuffled[i]]));
    }
  }

  return {
    tokens,
    origPos,
    removedPos: removed,
    editedPos: edited,
    labels,
    labelRole,
    perm,
    targetPos: tokens.findIndex((t) => t.isTarget && !t.isPadding),
    untouched: untouchedTokens && perm === null,
  };
}

/** The text lookup with the column-phrase permutation applied (identity when perm is null). */
export function ablatedEmbedder(pdb: PreprocessedDB, store: EmbeddingStore, perm: ReadonlyMap<number, number> | null): TextEmbedder {
  const base = textEmbedder(pdb, store);
  if (!perm) return base;
  return (idx: number) => base(perm.get(idx) ?? idx);
}

/** One forward pass of an ablated context (model.py:153-267 via the engine). */
export function ablatedForward(
  abl: AblatedContext,
  pdb: PreprocessedDB,
  store: EmbeddingStore,
  weights: RTWeights,
  disabled: readonly AttnLayer[],
  capture: boolean,
): RTForwardResult {
  return forwardWithMasks(abl.tokens, buildMasks(abl.tokens), ablatedEmbedder(pdb, store, abl.perm), weights, {
    disabledLayers: [...disabled],
    capture,
  });
}

/** Stable key of a token ablation (for caches). */
export function tokenAblationKey(ab: TokenAblation): string {
  const edits = [...ab.edits.entries()].sort((a, b) => a[0] - b[0]).map(([n, v]) => `${n}=${String(v)}`);
  return [
    ab.removeSelf ? 'S' : '',
    ab.removeOther ? 'O' : '',
    ab.removeSelf || ab.removeOther ? ab.removeWhat : '',
    ab.shuffle ? `sh${ab.shuffleSeed}` : '',
    edits.join(','),
  ].join('|');
}
