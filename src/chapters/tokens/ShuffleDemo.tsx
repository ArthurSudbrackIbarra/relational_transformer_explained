/**
 * Section 4 figure: shuffle the token order and run the model again.
 *
 * RT has no positional encodings: x₀ depends only on values, types and column-name embeddings
 * (model.py:208-224) and the masks only on ids (model.py:165-191), so permuting the tokens permutes
 * every output the same way and leaves the prediction at the target unchanged (spec §6.5).
 * The demo really permutes the arrays, rebuilds the masks from the permuted ids and reruns the
 * forward pass; nothing is assumed.
 */
import { RotateCcw, Shuffle } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { Button, Cite, Figure, LayerLabel, MaskMatrix, SCALE_ROWS, ScaleNote, StatusBadge, cx, formatNumber, formatPct, formatSigned } from '@/components';
import { axis } from '../shapes';
import { useElementSize } from '@/components/viz/useSize';
import { sigmoid } from '@/engine/metrics';
import { getMasks } from '@/engine/pipeline';
import { ATTN_ORDER, SemType } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { afterPaint, forwardShuffled, randomPermutation, type ShuffleResult } from './compute';
import { nodeLabel, tokenRowRef, type TokensLive } from './live';

const CELL = 18;
const GAP = 3;
/** PRNG seed of shuffle round r (deterministic, so a round always gives the same permutation). */
const seedOf = (r: number) => 7919 * r + 17;
const PAPER_D = SCALE_ROWS.find((r) => r.what === 'Hidden size d')?.paper ?? '256';

export function ShuffleDemo({ live }: { live: TokensLive }) {
  const { ctx, pdb, assets, fr, slots } = live;
  const reduce = useReducedMotion();
  const [ref, size] = useElementSize<HTMLDivElement>();
  const [round, setRound] = useState(0); // 0 = original order
  const [res, setRes] = useState<ShuffleResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    setRound(0);
    setRes(null);
  }, [ctx]);

  useEffect(() => {
    if (round === 0 || !fr || !assets.store || !assets.weights) return;
    let alive = true;
    setRunning(true);
    setError(null);
    afterPaint(() => forwardShuffled(ctx, pdb, assets.store!, assets.weights!, { xFinal: fr.xFinal, d: fr.d }, seedOf(round)))
      .then((r) => alive && setRes(r))
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => alive && setRunning(false));
    return () => {
      alive = false;
    };
  }, [round, ctx, pdb, assets.store, assets.weights, fr]);

  const S = ctx.tokens.length;
  const perm = useMemo(() => (round > 0 ? randomPermutation(S, seedOf(round)) : null), [round, S]);
  // newPos[i] = where original token i sits now (perm[newPos] = oldPos).
  const newPos = useMemo(() => {
    const out = new Array<number>(S);
    if (perm) perm.forEach((old, j) => (out[old] = j));
    else for (let i = 0; i < S; i++) out[i] = i;
    return out;
  }, [S, perm]);
  const current = res && perm && res.perm.length === perm.length && res.perm.every((v, i) => v === perm[i]) ? res : null;
  // Fly the squares only when the order changes (a shuffle or a reset), never on a resize.
  const lastRound = useRef(round);
  const flying = lastRound.current !== round;
  useEffect(() => {
    lastRound.current = round;
  }, [round]);

  const width = Math.max(200, size.width || 700);
  const cols = Math.max(8, Math.floor((width + GAP) / (CELL + GAP)));
  const rows = Math.ceil(S / cols);
  const H = rows * (CELL + GAP) - GAP;
  const targetNow = ctx.targetPos >= 0 ? newPos[ctx.targetPos] : -1;
  const baseMasks = useMemo(() => getMasks(ctx), [ctx]);
  const tokenSlots = useMemo(() => ctx.tokens.map((t) => (t.isPadding ? 0 : slots[t.tableName] ?? 0)), [ctx, slots]);
  const shuffledSlots = useMemo(() => (current ? current.tokens.map((t) => (t.isPadding ? 0 : slots[t.tableName] ?? 0)) : tokenSlots), [current, slots, tokenSlots]);
  const binary = ctx.targetPos >= 0 && ctx.tokens[ctx.targetPos].sem === SemType.Boolean;
  const hovered = hover !== null ? ctx.tokens[hover] : null;
  const same = current && fr ? current.fr.prediction === fr.prediction : null;
  // M_shuffled = P·M·Pᵀ, checked entry by entry on all four masks: entry (newPos[q], newPos[k]) of
  // the rebuilt mask must equal entry (q, k) of the sampler-order mask.
  const maskCheck = useMemo(() => {
    if (!current) return null;
    let bad = 0;
    for (const l of ATTN_ORDER) {
      const a = baseMasks[l];
      const b = current.masks[l];
      for (let q = 0; q < S; q++) {
        const rowA = q * S;
        const rowB = newPos[q] * S;
        for (let k = 0; k < S; k++) if (a[rowA + k] !== b[rowB + newPos[k]]) bad++;
      }
    }
    return { bad, total: ATTN_ORDER.length * S * S };
  }, [current, baseMasks, newPos, S]);
  // The parent row whose cells the most other cells may read (the widest vertical band of the feature mask).
  const band = useMemo(() => {
    const present = new Set(ctx.tokens.filter((t) => !t.isPadding).map((t) => t.nodeIdx));
    const readers = new Map<number, number>();
    for (const t of ctx.tokens) {
      if (t.isPadding) continue;
      for (const p of new Set(t.f2pNbrIdxs)) if (p >= 0 && present.has(p)) readers.set(p, (readers.get(p) ?? 0) + 1);
    }
    let best: { node: number; readers: number } | null = null;
    for (const [node, n] of readers) if (!best || n > best.readers) best = { node, readers: n };
    return best ? { label: nodeLabel(pdb, best.node), readers: best.readers } : null;
  }, [ctx, pdb]);
  const hlBase = hover !== null ? [hover] : [];
  const hlShuf = hover !== null ? [newPos[hover]] : [];
  const onKey = (e: KeyboardEvent) => {
    const cur = hover ?? (ctx.targetPos >= 0 ? ctx.targetPos : 0);
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') setHover(Math.min(S - 1, hover === null ? cur : cur + 1));
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') setHover(Math.max(0, hover === null ? cur : cur - 1));
    else if (e.key === 'Home') setHover(0);
    else if (e.key === 'End') setHover(S - 1);
    else if (e.key === 'Escape') setHover(null);
    else return;
    e.preventDefault();
  };
  const predFmt = (p: number) => (binary ? `${formatSigned(p, 6)} (P = ${formatPct(sigmoid(p), 2)})` : formatSigned(p, 6));

  return (
    <Figure
      wide
      title="Experiment: shuffle the tokens, run the model again"
      controls={
        <>
          <Button variant="primary" size="sm" icon={<Shuffle size={15} />} onClick={() => setRound((r) => r + 1)} disabled={!fr || running}>
            {round === 0 ? 'Shuffle the tokens' : 'Shuffle again'}
          </Button>
          <Button size="sm" icon={<RotateCcw size={15} />} onClick={() => setRound(0)} disabled={round === 0}>
            Sampler order
          </Button>
          <span className="text-[15px] text-ink-2">
            {round === 0 ? 'Showing the order the sampler emitted.' : running ? 'Running the forward pass…' : `Random permutation #${round}.`}
          </span>
        </>
      }
      caption={
        <>
          Each square is one token (color = table, “?” = the masked target). Shuffling moves the squares to random positions: the token arrays are
          permuted, the four masks are rebuilt from the permuted ids <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:165-191" />,</span> and the full forward pass runs
          again in your browser. Mathematically the result is exact. In floating point a different key order can change the last bits, so the
          official bfloat16 model gives equality up to rounding. This site’s engine adds up its dot products and weighted sums in 64-bit floats and
          stores 32-bit results (<span className="font-mono">src/engine/model/ops.ts</span>), so a different summation order moves the sums far
          below what a 32-bit float can hold, and the comparison reports what it actually gets. Focus the squares and use the arrow keys to follow
          one token.
        </>
      }
    >
      <div
        ref={ref}
        className="relative w-full rounded-[4px] focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-accent"
        style={{ height: H }}
        onPointerLeave={() => setHover(null)}
        tabIndex={0}
        role="group"
        aria-label="Token squares: use the arrow keys to follow one token through the shuffle"
        onKeyDown={onKey}
        onBlur={() => setHover(null)}
      >
        {ctx.tokens.map((t, i) => {
          const p = newPos[i];
          const x = (p % cols) * (CELL + GAP);
          const y = Math.floor(p / cols) * (CELL + GAP);
          return (
            <motion.div
              key={i}
              aria-hidden="true"
              initial={false}
              animate={{ x, y }}
              transition={reduce || !flying ? { duration: 0 } : { type: 'spring', stiffness: 170, damping: 22, mass: 0.7, delay: (i % 29) * 0.006 }}
              onPointerEnter={() => setHover(i)}
              className={cx(
                'absolute top-0 left-0 flex items-center justify-center rounded-[4px] text-[13px] font-bold',
                t.isPadding && 'border border-dashed border-axis',
                t.isTarget && 'target-outline z-10 bg-accent-soft text-accent-text',
                hover === i && 'ring-2 ring-ink ring-offset-1 ring-offset-surface',
              )}
              style={{ width: CELL, height: CELL, background: !t.isPadding && !t.isTarget ? catVar(tokenSlots[i]) : undefined }}
            >
              {t.isTarget ? '?' : null}
            </motion.div>
          );
        })}
      </div>
      <div className="mt-2 min-h-[20px] text-[15px] text-ink-2" aria-live="polite">
        {hovered ? (
          <>
            <span className="text-ink">
              {hovered.isPadding ? 'padding' : `${tokenRowRef(hovered)}.${hovered.colName}`}
            </span>{' '}
            · sampler position <span className="num">{hover}</span> · now at position <span className="num">{newPos[hover!]}</span>
          </>
        ) : (
          <>
            The target sits at position <span className="num font-semibold text-ink">{targetNow}</span>
            {round > 0 ? (
              <>
                {' '}
                (the sampler put it at <span className="num">{ctx.targetPos}</span>)
              </>
            ) : null}
            . Hover a square to follow one token.
          </>
        )}
      </div>

      <div className="mt-5 grid gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-border bg-surface-2/50 px-4 py-3">
          <div className="text-[15px] font-medium text-ink-2">Prediction, sampler order</div>
          <div className="num mt-1.5 text-[16px] font-semibold text-ink">{fr ? predFmt(fr.prediction) : '…'}</div>
          <div className="mt-1 text-[14.5px] text-ink-2">read at position {ctx.targetPos}</div>
        </div>
        <div className="rounded-xl border border-border bg-surface px-4 py-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[15px] font-medium text-ink-2">Prediction, shuffled</span>
            {running ? (
              <StatusBadge status="info">running…</StatusBadge>
            ) : current && same !== null ? (
              <StatusBadge status={same ? 'good' : Math.abs(current.fr.prediction - fr!.prediction) < 1e-5 ? 'good' : 'warning'}>
                {same ? 'identical' : `|Δ| = ${Math.abs(current.fr.prediction - fr!.prediction).toExponential(1)}`}
              </StatusBadge>
            ) : null}
          </div>
          <div className="num mt-1.5 text-[16px] font-semibold text-ink">{current ? predFmt(current.fr.prediction) : '—'}</div>
          <div className="mt-1 text-[14.5px] text-ink-2">{current ? `read at position ${current.targetPos}` : 'press Shuffle'}</div>
        </div>
        <div className="rounded-xl border border-border bg-surface px-4 py-3">
          <div className="text-[15px] font-medium text-ink-2">Every output moved with its token</div>
          <div className="num mt-1.5 text-[16px] font-semibold text-ink">
            {current ? (current.maxEquivDiff === 0 ? 'max |Δ| = 0' : `max |Δ| = ${current.maxEquivDiff.toExponential(1)}`) : '—'}
          </div>
          <div className="mt-1 text-[14.5px] text-ink-2">
            {current
              ? `over all ${ctx.numReal} × ${fr?.d ?? 0} numbers of the final hidden states, matched token by token${current.maxEquivDiff > 0 ? ' (float32 rounding from a different summation order)' : ''}.`
              : 'compares the final vector of every token'}{' '}
            d = {fr?.d ?? 0} here, {PAPER_D} in the paper <ScaleNote inline className="ml-0.5" />
          </div>
        </div>
      </div>
      {error && (
        <div className="mt-3">
          <StatusBadge status="critical">{error}</StatusBadge>
        </div>
      )}

      <div className="mt-6 flex flex-wrap items-center gap-x-3 gap-y-1.5">
        {maskCheck ? (
          <StatusBadge status={maskCheck.bad === 0 ? 'good' : 'critical'}>
            rebuilt masks = P·M·Pᵀ: {maskCheck.bad.toLocaleString('en-US')} of {maskCheck.total.toLocaleString('en-US')} entries differ
          </StatusBadge>
        ) : (
          <span className="text-[15px] text-ink-2">After a shuffle, every entry of the four rebuilt masks is checked against the original.</span>
        )}
        <span className="text-[14.5px] text-ink-2">rows = queries ↓ · columns = keys →</span>
      </div>
      <div className="mt-3 grid gap-5 md:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-1 text-[15.5px] text-ink-2">
            <LayerLabel layer="feat" /> mask, sampler order
          </div>
          <div className="flex justify-center">
            <MaskMatrix
              mask={baseMasks.feat}
              S={S}
              slots={tokenSlots}
              maxCellSize={3}
              showLegend={false}
              showReadout={false}
              highlightRows={hlBase}
              highlightCols={hlBase}
              onHover={(c) => setHover(c ? c.q : null)}
              title="feature mask, sampler order"
              shape={{
                rows: axis.queries({ shown: S, L: ctx.params.L, real: ctx.numReal }),
                cols: axis.keys({ shown: S, L: ctx.params.L, real: ctx.numReal }),
              }}
            />
          </div>
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-[15.5px] text-ink-2">
            <LayerLabel layer="feat" /> mask, {current ? 'shuffled order' : 'shuffled order (press Shuffle)'}
          </div>
          <div className="flex justify-center">
            <MaskMatrix
              mask={current ? current.masks.feat : baseMasks.feat}
              S={S}
              slots={current ? shuffledSlots : tokenSlots}
              maxCellSize={3}
              showLegend={false}
              showReadout={false}
              highlightRows={current ? hlShuf : hlBase}
              highlightCols={current ? hlShuf : hlBase}
              onHover={(c) => setHover(c ? (current ? current.perm[c.q] : c.q) : null)}
              title="feature mask, shuffled order"
              shape={{
                rows: `one per query token, in the random order of the shuffle: the same ${S} positions as the mask on the left, permuted`,
                cols: 'one per key token, in the same shuffled order as the rows',
                note: 'Rebuilding the masks after the shuffle gives P·M·Pᵀ: the same entries, moved along with their tokens.',
              }}
              className={cx(!current && 'opacity-40')}
            />
          </div>
        </div>
      </div>
      <p className="mt-3 text-[15px] leading-snug text-ink-2">
        The same mask drawn in two orders. The sampler emits a row’s cells together, so each row is a block on the diagonal (same_node). The
        vertical bands are parent rows whose cells every child row may read (kv_in_f2p) <Cite kind="code" ref="rt/model.py:171-173" />{' '}
        <Cite kind="code" ref="rt/model.py:187" />
        {band ? (
          <>
            ; the widest here is <span className="text-ink">{band.label}</span>, whose cells {band.readers} child cells may read
          </>
        ) : null}
        . After the shuffle the picture looks scattered, yet every entry still says the same thing about the same pair of cells: it is the same
        relation, written in a different order. Hover a square or a matrix row to light up that token’s row and column in both drawings.
        {fr && current ? ` Forward pass: ${formatNumber(current.ms, 0)} ms.` : ''}
      </p>
    </Figure>
  );
}
