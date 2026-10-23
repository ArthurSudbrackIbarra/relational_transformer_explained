/**
 * Section 3 figures: the single masked token.
 *
 *  - `OneMask`: the literal masks[0, :] array of the live context (exactly one true, fly.rs:492-496);
 *  - `TargetVsLabel`: the masked target next to a visible label cell of the same column;
 *  - `HiddenLabelDemo`: overwrite the label stored in the target token and rerun the forward pass
 *    with the input embedding evaluated literally (every term × its gate, model.py:216-224; see
 *    compute.ts literalWithStoredLabel). The prediction does not move (the value term is multiplied
 *    by ~masks = 0, model.py:219); the loss does, because the loss reads that stored value as its
 *    target (model.py:240) (spec §5.3).
 */
import { ArrowDown, ArrowRight, ArrowUp, Equal, EqualNot, Lock } from 'lucide-react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import {
  Cite,
  Figure,
  SCALE_ROWS,
  ScaleNote,
  Segmented,
  StatusBadge,
  TableSwatch,
  VectorStrip,
  cx,
  formatNumber,
  formatPct,
  formatRaw,
  formatSigned,
} from '@/components';
import { l2norm, scalarValueTerm, tensor } from '@/engine/model';
import { sigmoid, denormalize } from '@/engine/metrics';
import { SemType, type ForwardResult, type RTWeights, type Token } from '@/engine/types';
import { useTargetStats } from '@/state/hooks';
import { ArrayStrip } from './ArrayStrip';
import { afterPaint, literalSelfCheck, literalWithStoredLabel, type LiteralResult } from './compute';
import { pickVisibleLabel, tokenRowRef, type TokensLive } from './live';
import { ZeroVec } from './ZeroVec';

/** The paper's hidden size, from the shared scale table (paper d = 256, spec §5.2). */
const PAPER_D = SCALE_ROWS.find((r) => r.what === 'Hidden size d')?.paper ?? '256';

// ─────────────────────────────────────────────────────────────────────────────
// One mask
// ─────────────────────────────────────────────────────────────────────────────

export function OneMask({ live }: { live: TokensLive }) {
  const { ctx, slots, inspectAndGo } = live;
  const masks = useMemo(() => ctx.tokens.map((t) => t.mask), [ctx]);
  const [hover, setHover] = useState<number | null>(null);
  const t = ctx.tokens[hover ?? ctx.targetPos] ?? ctx.tokens[0];
  return (
    <Figure
      title="The masks field of the live context"
      caption={
        <>
          The array exactly as the model receives it: one boolean per position. Hover a cell to see which token it belongs to. The single true entry
          is the seed row’s target cell <span className="whitespace-nowrap"><Cite kind="code" ref="rustler/src/fly.rs:492-496" />.</span>
        </>
      }
    >
      <ArrayStrip name="masks" values={masks} highlight={ctx.targetPos} onHover={setHover} />
      <div className="mt-3 flex min-h-[28px] flex-wrap items-center gap-x-2 gap-y-1 text-[15.5px] text-ink-2">
        <span className="num text-ink">pos {t.pos}</span>
        {t.isPadding ? (
          <span>padding</span>
        ) : (
          <>
            <TableSwatch slot={slots[t.tableName] ?? 0} label={tokenRowRef(t)} />
            <span>.{t.colName}</span>
            {t.isTarget ? (
              <span className="target-outline rounded-[3px] px-1 font-bold text-accent-text">?</span>
            ) : (
              <span className="text-ink">= {formatRaw(t.raw, t.sem === SemType.DateTime ? 'datetime' : undefined)}</span>
            )}
          </>
        )}
        <button
          type="button"
          className="ml-auto inline-flex items-center gap-1 text-[15px] text-accent-text hover:underline"
          onClick={() => inspectAndGo(t.pos, 'Back to Step 3')}
        >
          Inspect pos {t.pos} in Step 1 <ArrowUp aria-hidden="true" size={15} />
        </button>
      </div>
    </Figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Target vs a visible label
// ─────────────────────────────────────────────────────────────────────────────

function valueSlot(t: Token): { field: string; v: number } {
  switch (t.sem) {
    case SemType.Number:
      return { field: 'number_values', v: t.numberValue };
    case SemType.DateTime:
      return { field: 'datetime_values', v: t.datetimeValue };
    case SemType.Boolean:
      return { field: 'boolean_values', v: t.booleanValue };
    default:
      return { field: 'text_values', v: NaN };
  }
}

/** "same parents" / "share k of n parents" / "no shared parent" for two f2p_nbr_idxs lists. */
function parentNote(a: readonly number[], b: readonly number[], kind: 'self' | 'other'): string {
  if (a.join() === b.join()) return kind === 'self' ? 'same parent entity' : 'same parents';
  const pa = a.filter((p) => p >= 0);
  const pb = new Set(b.filter((p) => p >= 0));
  const shared = pa.filter((p) => pb.has(p)).length;
  const n = Math.max(pa.length, pb.size);
  return shared === 0 ? 'no shared parent' : `share ${shared} of ${n} parents`;
}

export function TargetVsLabel({ live }: { live: TokensLive }) {
  const { ctx, pdb, labels, slots, inspectAndGo } = live;
  const target = ctx.targetPos >= 0 ? ctx.tokens[ctx.targetPos] : null;
  // The same visible label the Step 1 jump chip uses (live.ts pickVisibleLabel).
  const pick = useMemo(() => pickVisibleLabel(labels), [labels]);
  if (!target || !pick) return null;
  const lab = ctx.tokens[pick.pos];
  const ts = valueSlot(target);
  const ls = valueSlot(lab);
  const sameParents = target.f2pNbrIdxs.join() === lab.f2pNbrIdxs.join();
  const sameValue = ts.v === ls.v;
  const rows: { name: string; a: ReactNode; b: ReactNode; same: boolean; note?: ReactNode; accent?: boolean; plain?: boolean }[] = [
    {
      name: 'node_idxs',
      a: target.nodeIdx,
      b: lab.nodeIdx,
      same: target.nodeIdx === lab.nodeIdx,
      note: 'different rows',
    },
    {
      name: 'f2p_nbr_idxs',
      a: `[${target.f2pNbrIdxs.join(', ')}]`,
      b: `[${lab.f2pNbrIdxs.join(', ')}]`,
      same: sameParents,
      note: parentNote(target.f2pNbrIdxs, lab.f2pNbrIdxs, pick.kind),
    },
    {
      name: 'table_name_idxs',
      a: `${target.tableNameIdx}`,
      b: `${lab.tableNameIdx}`,
      same: target.tableNameIdx === lab.tableNameIdx,
      note:
        target.tableType !== lab.tableType
          ? `“${pdb.text[target.tableNameIdx]}”: the ${target.tableType} and ${lab.tableType} splits share one table id`
          : `“${pdb.text[target.tableNameIdx]}”`,
    },
    {
      name: 'col_name_idxs',
      a: `${target.colNameIdx}`,
      b: `${lab.colNameIdx}`,
      same: target.colNameIdx === lab.colNameIdx,
      note: `“${pdb.text[target.colNameIdx]}”`,
    },
    { name: 'sem_types', a: target.sem, b: lab.sem, same: target.sem === lab.sem, note: 'same type, same encoder' },
    {
      name: ts.field,
      a: (
        <span className="inline-flex items-center gap-1">
          {formatSigned(ts.v)} <Lock aria-label="stored but gated out" size={14} className="text-accent-text" />
        </span>
      ),
      b: formatSigned(ls.v),
      same: sameValue,
      note: 'both slots hold a real, z-scored label',
    },
    { name: 'masks', a: 'true', b: 'false', same: false, accent: true, note: 'the only flag the model reads that differs' },
    { name: 'is_targets', a: 'true', b: 'false', same: false, note: 'differs too, but only evaluation reads it' },
    {
      name: 'value term of x₀',
      plain: true,
      a: <span className="font-sans">mask vector m (learned)</span>,
      b: <span className="font-sans">RMSNorm(W·v + b)</span>,
      same: false,
      note: (
        <>
          what the model sees: computed in <Link to="/embedding">chapter 8</Link>
        </>
      ),
    },
  ];
  const otherDiffs = [!sameParents && 'parents', !sameValue && 'stored value'].filter(Boolean) as string[];
  const sameIcon = (same: boolean) =>
    same ? (
      <Equal aria-label="same" size={16} className="inline shrink-0 text-ink-2" />
    ) : (
      <EqualNot aria-label="different" size={16} className="inline shrink-0 text-accent-text" />
    );
  const targetHead = (
    <button type="button" className="inline-flex items-center gap-1.5 text-left hover:text-accent-text" onClick={() => inspectAndGo(target.pos, 'Back to Step 3')} title="Inspect in Step 1">
      <span className="target-outline rounded-[3px] px-1 font-bold text-accent-text">?</span> target · pos {target.pos}
    </button>
  );
  const labelHead = (
    <button type="button" className="inline-flex items-center gap-1.5 text-left hover:text-accent-text" onClick={() => inspectAndGo(lab.pos, 'Back to Step 3')} title="Inspect in Step 1">
      <TableSwatch slot={slots[lab.tableName] ?? 0} /> {pick.kind === 'self' ? 'self label (own past label)' : 'other row’s label'} · pos {lab.pos}
    </button>
  );
  return (
    <Figure
      wide
      title="The masked target next to a visible label of the same column"
      caption={
        <>
          Left: the target, {tokenRowRef(target)}. Right: {pick.kind === 'self' ? 'a past label of the same entity' : 'a label of another row'},{' '}
          {tokenRowRef(lab)} = {formatRaw(lab.raw)}. Table, column and type match, so both tokens get the same column-name term and the same
          encoder. They differ in row id{otherDiffs.length ? <> (and here also in {otherDiffs.join(' and ')})</> : null}. Of the flags the model
          reads, only <span className="font-mono">masks</span> differs, and it swaps the value term for the mask vector{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:216-224" />.</span> Click a column header to inspect that token in Step 1.
        </>
      }
    >
      {/* phones: one card per field, target and label side by side */}
      <div className="md:hidden">
        <div className="mb-2 grid grid-cols-2 gap-3 text-[15px] font-medium text-ink-2">
          <div className="min-w-0">{targetHead}</div>
          <div className="min-w-0">{labelHead}</div>
        </div>
        <ul className="space-y-2">
          {rows.map((r) => (
            <li key={r.name} className={cx('rounded-lg border border-border px-3 py-2', !r.same && 'bg-accent-soft/40')}>
              <div className="flex items-center gap-2">
                <span className={cx('min-w-0 text-[15px] font-semibold text-ink [overflow-wrap:anywhere]', !r.plain && 'font-mono')}>{r.name}</span>
                <span className="ml-auto">{sameIcon(r.same)}</span>
              </div>
              <div className="mt-1 grid grid-cols-2 gap-3 text-[15px]">
                <div className={cx('num min-w-0 [overflow-wrap:anywhere]', r.accent ? 'font-semibold text-accent-text' : 'text-ink')}>{r.a}</div>
                <div className="num min-w-0 text-ink [overflow-wrap:anywhere]">{r.b}</div>
              </div>
              {r.note && <div className="mt-1 text-[14.5px] leading-snug text-ink-2">{r.note}</div>}
            </li>
          ))}
        </ul>
      </div>

      {/* wider screens: one table */}
      <div className="scroll-quiet hidden overflow-x-auto md:block">
        <table className="w-full border-collapse text-[15px]">
          <thead>
            <tr className="border-b border-grid text-left text-[15px] text-ink-2">
              <th scope="col" className="py-2 pr-3 font-medium">
                field
              </th>
              <th scope="col" className="py-2 pr-3 font-medium whitespace-nowrap">
                {targetHead}
              </th>
              <th scope="col" className="w-8 py-2" aria-label="same?" />
              <th scope="col" className="py-2 pr-3 font-medium whitespace-nowrap">
                {labelHead}
              </th>
              <th scope="col" className="py-2 font-medium">
                <span className="sr-only">note</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name} className={cx('border-b border-grid/60 last:border-0', !r.same && 'bg-accent-soft/40')}>
                <th scope="row" className={cx('py-2 pr-3 text-left text-[15px] font-semibold whitespace-nowrap text-ink', !r.plain && 'font-mono')}>
                  {r.name}
                </th>
                <td className={cx('num py-2 pr-3 whitespace-nowrap', r.accent ? 'font-semibold text-accent-text' : 'text-ink')}>{r.a}</td>
                <td className="py-2 text-center">{sameIcon(r.same)}</td>
                <td className="num py-2 pr-3 whitespace-nowrap text-ink">{r.b}</td>
                <td className="py-2 text-[14.5px] leading-snug text-ink-2">{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// The gate, computed literally (model.py:216-224)
// ─────────────────────────────────────────────────────────────────────────────

type ScalarKey = 'number' | 'datetime' | 'boolean';

/**
 * The value branch and the mask branch of the target's input embedding, computed term by term as
 * model.py:216-224 writes them: the type's encoder output for the stored value is evaluated and then
 * multiplied by [sem = t ∧ ¬mask ∧ ¬pad] = 0; the raw mask vector is multiplied by
 * [sem = t ∧ mask ∧ ¬pad] = 1. The engine skips a term whose gate is 0 (identical for finite
 * values), so this panel evaluates it to show what the gate removes.
 */
function GateFlow({
  pos,
  semKey,
  semCode,
  z,
  optKey,
  weights,
  base,
}: {
  pos: number;
  semKey: ScalarKey;
  semCode: number;
  z: number;
  optKey: string;
  weights: RTWeights;
  base: ForwardResult;
}) {
  const reduce = useReducedMotion();
  const d = base.d;
  const term = useMemo(() => scalarValueTerm(weights, semKey, z).term, [weights, semKey, z]);
  const m = useMemo(() => tensor(weights, `mask_embs.${semKey}`), [weights, semKey]);
  const col = useMemo(() => base.embedding.colTerm.slice(pos * d, (pos + 1) * d), [base, pos, d]);
  // x₀ = colTerm + 0·term + 1·m, exactly the sum of model.py:208-224 for this token.
  const x0Literal = useMemo(() => {
    const out = new Float32Array(d);
    for (let j = 0; j < d; j++) out[j] = col[j] + 0 * term[j] + 1 * m[j];
    return out;
  }, [col, term, m, d]);
  const zero = useMemo(() => new Float32Array(d), [d]);
  const engineX0 = base.embedding.x0.subarray(pos * d, (pos + 1) * d);
  let diff = 0;
  for (let j = 0; j < d; j++) diff = Math.max(diff, Math.abs(x0Literal[j] - engineX0[j]));
  const domain = useMemo(() => {
    let mx = 0;
    for (const a of [term, m]) for (const v of a) mx = Math.max(mx, Math.abs(v));
    return mx || 1;
  }, [term, m]);
  const field = `${semKey}_values`;
  const gateTex = (masked: boolean) => (
    <>
      [sem = {semCode} ∧ {masked ? '' : '¬'}mask ∧ ¬pad]
    </>
  );

  const Arrow = () => (
    <>
      <ArrowRight aria-hidden="true" size={18} className="hidden shrink-0 text-muted md:block" />
      <ArrowDown aria-hidden="true" size={18} className="mx-auto shrink-0 text-muted md:hidden" />
    </>
  );
  const Box = ({ children, className }: { children: ReactNode; className?: string }) => (
    <div className={cx('min-w-0 rounded-xl border border-border bg-surface px-3 py-2.5', className)}>{children}</div>
  );
  const row = 'flex flex-col items-stretch gap-2 md:grid md:grid-cols-[190px_18px_minmax(0,1fr)_18px_minmax(150px,190px)_18px_112px] md:items-center md:gap-3';

  return (
    <div className="mt-6 rounded-xl border border-border bg-surface-2/50 px-4 py-4">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <span className="text-[16px] font-semibold text-ink">Inside the input embedding of position {pos}</span>
        <span className="text-[14.5px] text-ink-2">
          computed term by term as written in <Cite kind="code" ref="rt/model.py:216-224" />
        </span>
      </div>
      <div className="mb-3 flex flex-wrap items-center gap-x-2 gap-y-1 text-[14.5px] text-ink-2">
        <span>
          Each strip is one vector of <span className="num">d = {d}</span> numbers (d = {PAPER_D} in the paper), one square per number.
        </span>
        <ScaleNote inline />
      </div>

      {/* value branch */}
      <div className={row}>
        <Box>
          <div className="font-mono text-[14.5px] text-ink-2">
            {field}[{pos}]
          </div>
          <div className="relative h-[30px] overflow-hidden">
            <AnimatePresence mode="popLayout" initial={false}>
              <motion.div
                key={optKey}
                initial={reduce ? false : { y: 22, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={reduce ? undefined : { y: -22, opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="num text-[20px] leading-[30px] font-semibold text-ink"
              >
                {formatSigned(z, 3)}
              </motion.div>
            </AnimatePresence>
          </div>
        </Box>
        <Arrow />
        <Box>
          <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-2 text-[15px]">
            <span className="text-ink">
              RMSNorm<sub className="text-[13px] leading-none">{semKey}</sub>(W·v + b)
            </span>
            <span className="num text-ink-2">‖·‖ = {formatNumber(l2norm(term), 2)}</span>
          </div>
          <motion.div key={optKey} initial={reduce ? false : { opacity: 0.25 }} animate={{ opacity: 1 }} transition={{ duration: 0.45 }}>
            <VectorStrip values={term} columns={32} cellHeight={11} domain={domain} showReadout={false} />
          </motion.div>
          <div className="mt-1 text-[14px] text-ink-2">changes with the stored value</div>
        </Box>
        <Arrow />
        <div className="rounded-xl border border-dashed border-[var(--critical)] px-3 py-2 text-center">
          <div className="num text-[14.5px] text-ink-2">× {gateTex(false)}</div>
          <div className="num text-[22px] leading-tight font-bold text-[var(--critical)]">= 0</div>
        </div>
        <Arrow />
        <div className="rounded-xl border border-dashed border-axis px-3 py-2 text-center">
          <ZeroVec className="text-[19px] text-ink-2" />
          <VectorStrip values={zero} columns={16} cellHeight={7} domain={domain} showReadout={false} className="mt-1" />
          <div className="mt-0.5 text-[14px] text-ink-2">{d} zeros</div>
        </div>
      </div>

      {/* mask branch */}
      <div className={cx(row, 'mt-4 md:mt-3')}>
        <Box>
          <div className="font-mono text-[14.5px] text-ink-2">masks[{pos}]</div>
          <div className="num text-[20px] leading-[30px] font-semibold text-accent-text">true</div>
        </Box>
        <Arrow />
        <Box>
          <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-2 text-[15px]">
            <span className="text-ink">
              m<sub className="text-[13px] leading-none">{semKey}</sub>, learned, added raw
            </span>
            <span className="num text-ink-2">‖·‖ = {formatNumber(l2norm(m), 2)}</span>
          </div>
          <VectorStrip values={m} columns={32} cellHeight={11} domain={domain} showReadout={false} />
          <div className="mt-1 text-[14px] text-ink-2">the same vector for every masked {semKey} cell</div>
        </Box>
        <Arrow />
        <div className="rounded-xl border border-dashed border-accent px-3 py-2 text-center">
          <div className="num text-[14.5px] text-ink-2">× {gateTex(true)}</div>
          <div className="num text-[22px] leading-tight font-bold text-accent-text">= 1</div>
        </div>
        <Arrow />
        <div className="num rounded-xl border border-accent/50 bg-accent-soft px-3 py-2 text-center text-[17px] text-accent-text">
          m<sub className="text-[13px] leading-none">{semKey}</sub>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1.5 border-t border-grid pt-3 text-[15.5px] text-ink-2">
        <span>
          x₀[{pos}] = column-name term + <ZeroVec /> + m<sub className="text-[13px] leading-none">{semKey}</sub>, whatever the stored value.
        </span>
        <StatusBadge status={diff === 0 ? 'good' : 'warning'}>
          {diff === 0 ? 'same x₀ as the engine’s forward pass, bit for bit' : `differs from the engine by ${diff.toExponential(1)}`}
        </StatusBadge>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// The hidden-label experiment
// ─────────────────────────────────────────────────────────────────────────────

interface Option {
  key: string;
  z: number;
  label: string;
  hint: string;
}

export function HiddenLabelDemo({ live }: { live: TokensLive }) {
  const { ctx, pdb, assets, fr, task } = live;
  const stats = useTargetStats();
  const reduce = useReducedMotion();
  const target = ctx.targetPos >= 0 ? ctx.tokens[ctx.targetPos] : null;
  const binary = target?.sem === SemType.Boolean;
  const storedZ = target ? (binary ? target.booleanValue : target.sem === SemType.Number ? target.numberValue : target.datetimeValue) : 0;

  const options: Option[] = useMemo(() => {
    if (!target) return [];
    if (binary && stats) {
      // pre.rs:554-555: (1.0 or 0.0 − mean) / std with the label column's stats (Train split of a task
      // table, all rows of a DB table for autocomplete; spec §3.4), as useTargetStats returns them.
      const zTrue = (1 - stats.mean) / stats.std;
      const zFalse = (0 - stats.mean) / stats.std;
      const isTrue = storedZ > 0;
      return [
        { key: 'stored', z: storedZ, label: `stored: ${isTrue ? 'true' : 'false'}`, hint: 'what the sampler wrote' },
        { key: 'flip', z: isTrue ? zFalse : zTrue, label: `flipped: ${isTrue ? 'false' : 'true'}`, hint: 'the other class, z-scored the same way' },
        { key: 'any', z: 1000, label: 'any number: 1000', hint: 'an absurd but finite value' },
      ];
    }
    return [
      { key: 'stored', z: storedZ, label: `stored: z = ${formatSigned(storedZ, 2)}`, hint: 'what the sampler wrote' },
      {
        key: 'zero',
        z: 0,
        label: task.kind === 'forecast' ? 'z = 0 (the Train mean)' : 'z = 0 (the column mean)',
        hint: task.kind === 'forecast' ? 'the mean of the Train labels' : 'the mean over all rows of the column (spec §3.4)',
      },
      { key: 'any', z: 1000, label: 'any number: 1000', hint: 'an absurd but finite value' },
    ];
  }, [target, binary, stats, storedZ, task.kind]);

  const [key, setKey] = useState('stored');
  const opt = options.find((o) => o.key === key) ?? options[0];
  const [res, setRes] = useState<{ key: string; ctx: typeof ctx; lit: LiteralResult } | null>(null);
  // The literal path on the sampler's own tokens, compared with the engine's forward pass.
  const [check, setCheck] = useState<{ ctx: typeof ctx; ok: boolean } | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setRes(null);
    setKey('stored');
  }, [ctx]);

  useEffect(() => {
    if (!opt || !fr || !assets.store || !assets.weights) return;
    let alive = true;
    setRunning(true);
    setError(null);
    afterPaint(() => {
      const lit = literalWithStoredLabel(ctx, pdb, assets.store!, assets.weights!, opt.z);
      const base = opt.key === 'stored' ? lit : literalWithStoredLabel(ctx, pdb, assets.store!, assets.weights!, storedZ);
      const c = literalSelfCheck(base, fr);
      return { lit, ok: c.x0 === 0 && c.xFinal === 0 && c.prediction && c.loss };
    })
      .then((r) => {
        if (!alive) return;
        setRes({ key: opt.key, ctx, lit: r.lit });
        setCheck({ ctx, ok: r.ok });
      })
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => alive && setRunning(false));
    return () => {
      alive = false;
    };
  }, [opt, fr, assets.store, assets.weights, ctx, pdb, storedZ]);

  if (!target || !opt) return null;
  const field = binary ? 'boolean_values' : target.sem === SemType.Number ? 'number_values' : 'datetime_values';
  const cur = res && res.key === opt.key && res.ctx === ctx ? res.lit : null;
  const checked = check && check.ctx === ctx ? check.ok : null;
  // Measured: the literal path's prediction for this stored value against the engine's, bit for bit.
  const same = cur && fr ? Object.is(cur.prediction, fr.prediction) : null;
  const lossLabel = (z: number) => (binary ? `1{z > 0} = ${z > 0 ? 1 : 0}` : `y = ${formatSigned(z, 3)}`);
  const isBaseline = opt?.key === 'stored';
  const predText = (p: number) =>
    binary ? (
      <>
        logit <span className="font-semibold text-ink">{formatSigned(p, 6)}</span> · P = {formatPct(sigmoid(p), 2)}
      </>
    ) : (
      <>
        ẑ <span className="font-semibold text-ink">{formatSigned(p, 6)}</span>
        {stats ? <> · {formatNumber(denormalize(p, stats))} {task.targetCol}</> : null}
      </>
    );

  return (
    <Figure
      wide
      title="Experiment: overwrite the hidden label"
      controls={
        <Segmented
          label={`Value stored in the target's ${field} slot`}
          value={opt.key}
          onChange={setKey}
          options={options.map((o) => ({ value: o.key, label: o.label, title: o.hint }))}
        />
      }
      caption={
        <>
          The forward pass runs again in your browser on the same {ctx.tokens.length} tokens, with only the target’s stored value replaced. This
          rerun evaluates the input embedding literally, the way <Cite kind="code" ref="rt/model.py:216-224" /> writes it: at every position the value
          term of every type is computed from its slot, the replaced label included, multiplied by its gate and added up. That input then passes
          through the four blocks. The prediction is read at the masked position, as evaluation does{' '}
          <span className="whitespace-nowrap"><Cite kind="code" ref="rt/main.py:283-288" />,</span> and the loss uses the stored value as its target{' '}
          <span className="whitespace-nowrap">
            <span className="whitespace-nowrap"><Cite kind="code" ref="rt/model.py:240" />.</span>
          </span>{' '}
          The bottom panel opens the gate for the target alone.
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-border bg-surface-2/50 px-4 py-3">
          <div className="text-[15px] font-medium text-ink-2">In the token arrays</div>
          <div className="font-mono mt-1.5 text-[15.5px] text-ink-2">
            {field}[{target.pos}] =
          </div>
          <div className="relative mt-0.5 h-[34px] overflow-hidden">
            <AnimatePresence mode="popLayout" initial={false}>
              <motion.div
                key={opt.key}
                initial={reduce ? false : { y: 26, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={reduce ? undefined : { y: -26, opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="num text-[26px] leading-[34px] font-semibold tracking-tight text-ink"
              >
                {formatSigned(opt.z, 3)}
              </motion.div>
            </AnimatePresence>
          </div>
          <div className="mt-1 text-[14.5px] text-ink-2">
            masks[{target.pos}] = <span className="num font-semibold text-accent-text">true</span>, so the input embedding multiplies this value’s
            term by 0.
          </div>
        </div>

        <div className="rounded-xl border border-border bg-surface px-4 py-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[15px] font-medium text-ink-2">Prediction at the target</span>
            {running ? (
              <StatusBadge status="info">running…</StatusBadge>
            ) : isBaseline ? (
              <span className="rounded-full border border-border px-2 text-[14px] text-ink-2">baseline</span>
            ) : same === true ? (
              <StatusBadge status="good">unchanged</StatusBadge>
            ) : same === false ? (
              <StatusBadge status="warning">changed</StatusBadge>
            ) : null}
          </div>
          <div className="num mt-2 text-[15.5px] text-ink-2">{fr ? predText(cur ? cur.prediction : fr.prediction) : 'loading the model…'}</div>
          <div className="mt-1.5 text-[14.5px] text-ink-2">
            {fr && cur && opt.key !== 'stored'
              ? same
                ? 'Measured: bit for bit the same output as with the stored label.'
                : `Differs by ${formatNumber(Math.abs(cur.prediction - fr.prediction))}.`
              : 'Pick another value above and the forward pass runs again.'}
          </div>
        </div>

        <div className="rounded-xl border border-border bg-surface px-4 py-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[15px] font-medium text-ink-2">Loss at the target</span>
            {cur && fr && opt.key !== 'stored' && !running && cur.loss !== fr.loss && <StatusBadge status="info">moves with the label</StatusBadge>}
          </div>
          <div className="num mt-2 text-[15.5px] text-ink-2">
            {binary ? 'BCE' : 'Huber'} = <span className="font-semibold text-ink">{cur ? formatNumber(cur.loss, 4) : fr ? formatNumber(fr.loss, 4) : '…'}</span>
          </div>
          <div className="mt-1.5 text-[14.5px] text-ink-2">
            target read from the token: <span className="num whitespace-nowrap text-ink">{lossLabel(opt.z)}</span>
          </div>
          {binary && <div className="mt-0.5 text-[14px] leading-snug text-ink-2">the label is 1 (true) if the stored z is above 0, else 0</div>}
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[14.5px] text-ink-2">
        {checked === null ? (
          <StatusBadge status="info">checking the literal path against the engine…</StatusBadge>
        ) : checked ? (
          <StatusBadge status="good">literal path = the engine’s forward pass, bit for bit</StatusBadge>
        ) : (
          <StatusBadge status="warning">literal path differs from the engine</StatusBadge>
        )}
        <span>
          On the sampler’s own tokens, the literal evaluation gives the same x₀, final states, prediction and loss as the engine.
          {cur ? ` This rerun: ${formatNumber(cur.ms, 0)} ms.` : ''}
        </span>
      </div>
      {fr && assets.weights && target.sem !== SemType.Text && (
        <GateFlow
          pos={target.pos}
          semKey={(binary ? 'boolean' : target.sem === SemType.Number ? 'number' : 'datetime') as ScalarKey}
          semCode={target.sem}
          z={opt.z}
          optKey={opt.key}
          weights={assets.weights}
          base={fr}
        />
      )}
      {error && (
        <div className="mt-3">
          <StatusBadge status="critical">{error}</StatusBadge>
        </div>
      )}
      {assets.weightsKind === 'init' && (
        <div className="mt-3">
          <StatusBadge status="warning">Untrained weights: the prediction is not meaningful, but the invariance holds for any weights</StatusBadge>
        </div>
      )}
    </Figure>
  );
}
