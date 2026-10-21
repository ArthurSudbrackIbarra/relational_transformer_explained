/**
 * Step 1 figure: the last residual vector of one position → norm_out → the four linear decoders
 * (model.py:229, 124-131, 238-239) → only the target's type is read (model.py:241; main.py:283-293).
 * Every number is the live forward pass; the decoders are recomputed from the same weights only to
 * show w·x̂ and b separately, and checked against the engine's ŷ.
 */
import { motion, useReducedMotion } from 'motion/react';
import { useMemo } from 'react';
import { Figure, SemBadge, Select, StatusBadge, StepPlayer, useStepPlayer, VectorStrip } from '@/components';
import { SemType, type Token } from '@/engine/types';
import { fmt, fmtS, rowName, type Lane } from './live';
import { decodeAt, HEADS, type HeadKey } from './math';
import { LaneHeader } from './LaneHeader';

const SEM_OF: Record<HeadKey, SemType> = {
  number: SemType.Number,
  text: SemType.Text,
  datetime: SemType.DateTime,
  boolean: SemType.Boolean,
};

const MEANING: Record<HeadKey, string> = {
  number: 'a z-score, read in the target column’s own (μ, σ)',
  text: 'a 384-d vector in MiniLM space',
  datetime: 'a z-score on the database’s one datetime scale',
  boolean: 'a logit: positive means “true” is more likely',
};

export function tokenOption(t: Token): string {
  if (t.isPadding) return `#${t.pos} · padding`;
  return `#${t.pos} · ${rowName(t)}.${t.colName}${t.isTarget ? ' · masked target' : ''}`;
}

export function DecoderFan({ lane, pos, setPos }: { lane: Lane; pos: number; setPos: (p: number) => void }) {
  const { fr, assets, ctx } = lane;
  const weights = assets.weights;
  const reduce = useReducedMotion();
  const player = useStepPlayer(4, { initial: 3, intervalMs: 1500 });
  const step = player.index;
  const dec = useMemo(() => (fr && weights ? decodeAt(fr, weights, pos) : null), [fr, weights, pos]);
  const tok = ctx.tokens[pos];
  const isTarget = !!tok?.isTarget;
  const readKey: HeadKey | null = isTarget ? (['number', 'text', 'datetime', 'boolean'] as const)[tok.sem] : null;
  const nBlocks = fr?.blocks.length ?? 4;
  const d = dec?.d ?? 64;
  const domain = dec ? Math.max(...Array.from(dec.xHat, Math.abs), 1e-6) : 1;
  const agree = dec ? Math.max(...(['number', 'datetime', 'boolean'] as const).map((k) => Math.abs(dec.scalar[k].out - dec.scalar[k].engine))) : 0;
  const tr = (delay = 0) => (reduce ? { duration: 0 } : { duration: 0.45, delay, ease: [0.22, 1, 0.36, 1] as const });

  const describe = (i: number) =>
    [
      `After block ${nBlocks}, position #${pos} holds a ${d}-dimensional residual vector x.`,
      `norm_out divides x by its root-mean-square √(mean(x²) + ε) and multiplies by the learned gain γ (model.py:229).`,
      'All four decoders run on x̂, at this position and at every other one (model.py:238-239).',
      isTarget
        ? `Only the ${readKey} head is read here: the masked cell is a ${readKey} cell (model.py:241; main.py:283-288).`
        : 'This position is not the masked cell, so none of its four outputs is read or enters the loss.',
    ][i];

  return (
    <Figure
      wide
      title="From the last block to four predictions"
      controls={
        <Select label="Position" value={pos} onChange={(e) => setPos(Number(e.target.value))} className="w-full max-w-[560px]">
          {ctx.tokens.map((t) => (
            <option key={t.pos} value={t.pos}>
              {tokenOption(t)}
            </option>
          ))}
        </Select>
      }
      caption={
        <>
          All values are the live forward pass of the mini model. The colour of a cell is the sign and size of one coordinate (blue negative, red
          positive). Pick another position to see that the four heads produce outputs there too; only the masked cell’s own type is ever read.
        </>
      }
    >
      <LaneHeader lane={lane} />
      <StepPlayer {...player} describe={describe} unit="stage" label="Decoding stages" className="mb-5" />
      {!dec ? (
        <div className="rounded-lg border border-dashed border-border-strong px-4 py-8 text-center text-[16px] text-ink-2">
          Running the forward pass…
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] lg:gap-8">
          {/* left: x → x̂ */}
          <div className="min-w-0">
            <div className="rounded-lg border border-border bg-surface p-3">
              <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-2">
                <span className="text-[15.5px] font-semibold text-ink">x after block {nBlocks}</span>
                <span className="num text-[15px] text-ink-2">{tok?.isPadding ? 'padding: x = 0' : `RMS = ${fmt(dec.rms, 3)}`}</span>
              </div>
              <VectorStrip values={dec.xLast} columns={32} cellHeight={13} />
            </div>
            <motion.div
              initial={false}
              animate={{ opacity: step >= 1 ? 1 : 0.25 }}
              transition={tr()}
              className="my-2 flex items-center gap-2 pl-3 text-[15.5px] text-ink-2"
            >
              <span aria-hidden="true" className="text-[20px] leading-none text-accent-text">
                ↓
              </span>
              <span>
                <span className="font-mono text-ink">norm_out</span>: x̂ = x / √(mean(x²) + ε) ⊙ γ, with ε = 2⁻⁷
              </span>
            </motion.div>
            <motion.div
              initial={false}
              animate={{ opacity: step >= 1 ? 1 : 0.25, y: step >= 1 ? 0 : 6 }}
              transition={tr()}
              className="rounded-lg border border-border bg-surface p-3"
            >
              <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-2">
                <span className="text-[15.5px] font-semibold text-ink">x̂ = norm_out(x)</span>
                <span className="num text-[15px] text-ink-2">{d} values</span>
              </div>
              <VectorStrip values={dec.xHat} columns={32} cellHeight={13} domain={domain} />
            </motion.div>
            {readKey && readKey !== 'text' && step >= 3 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={tr(0.1)}
                className="mt-4 rounded-lg border border-border bg-surface p-3"
              >
                <div className="mb-1.5 text-[15.5px] font-semibold text-ink">
                  The {d} terms w_j · x̂_j of the {readKey} head
                </div>
                <VectorStrip values={dec.scalar[readKey].terms} columns={32} cellHeight={13} />
                <div className="mt-1 text-[15px] text-ink-2">
                  Their sum is w·x̂ = <span className="num text-ink">{fmtS(dec.scalar[readKey].dot)}</span>; adding the bias gives the prediction.
                </div>
              </motion.div>
            )}
          </div>

          {/* right: the four heads */}
          <div className="relative min-w-0 pl-5">
            <motion.span
              aria-hidden="true"
              initial={false}
              animate={{ scaleY: step >= 2 ? 1 : 0 }}
              transition={tr()}
              style={{ originY: 0 }}
              className="absolute top-3 bottom-3 left-0 w-[2px] rounded-full bg-accent"
            />
            <div className="mb-2 text-[15.5px] font-semibold text-ink-2">dec_dict: four linear heads, applied to x̂</div>
            <div className="grid gap-2.5">
              {HEADS.map((k, i) => {
                const read = step >= 3 && readKey === k;
                const dim = step >= 3 && readKey !== k;
                const visible = step >= 2;
                return (
                  <motion.div
                    key={k}
                    initial={false}
                    animate={{
                      opacity: visible ? (dim ? 0.6 : 1) : 0.2,
                      x: visible ? 0 : -8,
                    }}
                    transition={tr(visible && step === 2 ? i * 0.12 : 0)}
                    className={`relative rounded-lg border bg-surface px-3 py-2.5 ${read ? 'border-accent ring-2 ring-[var(--accent-ring)]' : 'border-border'}`}
                  >
                    <span aria-hidden="true" className="absolute top-1/2 -left-5 h-[2px] w-5 -translate-y-1/2 bg-accent opacity-70" />
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <SemBadge sem={SEM_OF[k]} size="sm" />
                      <span className="font-mono text-[15.5px] text-ink">dec_dict["{k}"]</span>
                      <span className="text-[15px] text-ink-2">
                        Linear({d} → {k === 'text' ? dec.text.dText : 1})
                      </span>
                      <span className="ml-auto">
                        {read ? (
                          <span className="rounded-full bg-accent px-2 py-0.5 text-[14px] font-semibold text-accent-ink">read</span>
                        ) : step >= 3 ? (
                          <span className="text-[14px] text-ink-2">computed, not read</span>
                        ) : null}
                      </span>
                    </div>
                    {k === 'text' ? (
                      <div className="mt-2">
                        <VectorStrip values={dec.text.out} columns={64} cellHeight={7} showReadout={false} />
                        <div className="mt-1 text-[15px] text-ink-2">
                          {dec.text.dText} outputs, ‖ŷ‖ = <span className="num text-ink">{fmt(dec.text.norm, 2)}</span>. {MEANING.text}; never read,
                          because text cells are never masked.
                        </div>
                      </div>
                    ) : (
                      <div className="mt-1.5">
                        <div className="num text-[16px] text-ink">
                          w·x̂ + b = {fmtS(dec.scalar[k].dot, 4)} {dec.scalar[k].b < 0 ? '−' : '+'} {fmt(Math.abs(dec.scalar[k].b), 4)} ={' '}
                          <span className="font-semibold">{fmtS(dec.scalar[k].out, 4)}</span>
                        </div>
                        <div className="text-[15px] text-ink-2">{MEANING[k]}</div>
                      </div>
                    )}
                  </motion.div>
                );
              })}
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {agree < 1e-4 ? (
                <StatusBadge status="good">matches the engine’s decoder outputs (max |Δ| {agree === 0 ? '0' : agree.toExponential(0)})</StatusBadge>
              ) : (
                <StatusBadge status="critical">differs from the engine by {agree.toExponential(1)}</StatusBadge>
              )}
              {tok?.isPadding && <StatusBadge status="info">padding: x̂ = 0, so every head outputs its bias</StatusBadge>}
            </div>
          </div>
        </div>
      )}
    </Figure>
  );
}
