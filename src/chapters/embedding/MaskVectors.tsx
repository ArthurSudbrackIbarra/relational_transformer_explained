/**
 * Step 5 figure: the four learned mask vectors m_t (mask_embs, model.py:141-146) and the masked target
 * of the live context. Everything is read from the live weights and the forward pass; the "change the
 * hidden label" check re-runs the engine's own `embedTokens` (model.py:208-224) on a copy of the
 * target token whose stored value is replaced by another finite number.
 */
import { Check, X } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';
import { Cite, Figure, Segmented, SemBadge, StatusBadge, VectorStrip, cx, maxAbs, semOfDType } from '@/components';
import { SEM_KEYS, embedTokens, tensor, textEmbedder } from '@/engine/model';
import { SemType, type RawDatabase, type Token } from '@/engine/types';
import type { EmbeddingLive } from './live';
import { fmt, fmtSigned, phraseOf, rowRef, VALUE_FIELD } from './live';
import { maxAbsDelta, norm, slice, tokenScalar } from './math';

const TYPES = [SemType.Number, SemType.Text, SemType.DateTime, SemType.Boolean];

function median(a: number[]): number {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function withStoredValue(t: Token, v: number): Token {
  if (t.sem === SemType.Number) return { ...t, numberValue: v };
  if (t.sem === SemType.DateTime) return { ...t, datetimeValue: v };
  return { ...t, booleanValue: v };
}

export function MaskVectors({ live }: { live: EmbeddingLive }) {
  const { weights, store, fr, ctx, pdb, target, assets } = live;
  const tgt = ctx.tokens[target];
  const [alt, setAlt] = useState<'stored' | 'neg' | 'zero' | 'big'>('neg');

  const data = useMemo(() => {
    if (!weights || !fr) return null;
    const d = weights.config.dModel;
    const m = TYPES.map((s) => tensor(weights, `mask_embs.${SEM_KEYS[s]}`));
    const typical = TYPES.map((s) =>
      median(ctx.tokens.filter((t) => !t.isPadding && !t.mask && t.sem === s).map((t) => norm(slice(fr.embedding.valueTerm, t.pos, d)))),
    );
    return { d, m, typical, domain: Math.max(...m.map((v) => maxAbs(v))) || 1 };
  }, [weights, fr, ctx]);

  const check = useMemo(() => {
    if (!weights || !store || !fr || !tgt || !tgt.mask) return null;
    const d = weights.config.dModel;
    const stored = tokenScalar(tgt);
    const v = alt === 'stored' ? stored : alt === 'neg' ? -stored : alt === 'zero' ? 0 : 1000;
    const emb = embedTokens([withStoredValue({ ...tgt, pos: 0 }, v)], textEmbedder(pdb, store), weights);
    const live0 = slice(fr.embedding.x0, tgt.pos, d);
    return { stored, v, delta: maxAbsDelta(emb.x0, live0), x0: emb.x0 };
  }, [weights, store, fr, tgt, pdb, alt]);

  if (!data || !tgt) {
    return (
      <Figure wide title="The four mask vectors">
        <div className="py-6 text-center text-[16px] text-ink-2">Loading the weights…</div>
      </Figure>
    );
  }
  const tKey = SEM_KEYS[tgt.sem];
  const tasksBySem = targetSems(live.db);
  const parent = tgt.f2pNbrIdxs.find((p) => p >= 0);
  const parentTok = parent !== undefined ? ctx.tokens.find((t) => !t.isPadding && t.nodeIdx === parent) : undefined;

  return (
    <Figure
      wide
      title="The four mask vectors, and what the masked target still knows"
      caption={
        <>
          Left: mask_embs of the checkpoint this page runs, on one shared colour scale. Right: the live masked target. The check at the bottom re-runs
          the engine’s embedding on a copy of the target with a different stored label and compares x₀ with the forward pass.
        </>
      }
    >
      {assets.weightsKind === 'init' && <StatusBadge status="warning">Untrained weights: values from a random initialisation</StatusBadge>}
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div className="grid gap-3.5">
          {TYPES.map((s, i) => {
            const used = s === tgt.sem;
            return (
              <div key={s} className={cx('rounded-lg border px-3 py-2.5', used ? 'border-accent bg-accent-soft/40' : 'border-border bg-surface')}>
                <div className="mb-1.5 flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
                  <span className="inline-flex items-center gap-2 text-[16px] text-ink">
                    <SemBadge sem={s} size="sm" />
                    <span className="font-mono">m_{SEM_KEYS[s]}</span>
                    {used && <span className="rounded-full bg-accent px-2 py-px text-[13.5px] font-medium text-accent-ink">live target</span>}
                  </span>
                  <span className="num text-[15px] text-ink-2">
                    ‖m‖ = <span className="text-ink">{fmt(norm(data.m[i]), 2)}</span>
                    {Number.isFinite(data.typical[i]) && (
                      <>
                        {' '}
                        · value terms ≈ <span className="text-ink">{fmt(data.typical[i], 2)}</span>
                      </>
                    )}
                  </span>
                </div>
                <VectorStrip values={data.m[i]} columns={32} cellHeight={11} domain={data.domain} showNorm={false} />
              </div>
            );
          })}
          <p className="!m-0 text-[15px] leading-snug text-ink-2">
            “value terms ≈” is the median length of that type’s value terms in the live context, for comparison. Each m_t starts as{' '}
            <code>torch.randn({data.d})</code>, a standard normal vector of length about √{data.d} = {fmt(Math.sqrt(data.d), 0)}{' '}
            <Cite kind="spec" ref="§7.4" />.
          </p>
        </div>

        <div className="min-w-0">
          <div className="rounded-xl border border-border bg-surface-2/60 p-4">
            <div className="mb-1 text-[15px] text-ink-2">The masked target of the live context</div>
            <div className="mb-3 flex flex-wrap items-center gap-2 text-[16.5px]">
              <span className="font-mono text-ink">{rowRef(tgt)}</span>
              <SemBadge sem={tgt.sem} size="xs" showLabel />
              <span className="target-outline rounded px-1.5 font-mono font-bold text-accent-text">?</span>
            </div>
            <ul className="grid gap-2.5">
              <Knows ok>
                <strong className="font-semibold text-ink">Which column of which table</strong>: its column term embeds “{phraseOf(tgt)}”.
              </Knows>
              <Knows ok>
                <strong className="font-semibold text-ink">Which type</strong>: the gate picks m_{tKey}, one of four different vectors.
              </Knows>
              <Knows ok>
                <strong className="font-semibold text-ink">Which row, and its parent</strong>: node {tgt.nodeIdx}
                {parentTok ? `, linked to ${rowRef(parentTok)}` : ''}. These ids reach the attention masks, not x₀{' '}
                <Cite kind="spec" ref="§5.3" />.
              </Knows>
              <Knows ok={false}>
                <strong className="font-semibold text-ink">Its value</strong>: the label is still stored in {VALUE_FIELD[tgt.sem]} ={' '}
                {fmtSigned(tokenScalar(tgt))}, but its gate is 0 <Cite kind="code" ref="rt/model.py:219" />.
              </Knows>
            </ul>
          </div>

          {check && (
            <div className="mt-4 rounded-xl border border-border bg-surface p-4">
              <div className="mb-2 text-[16px] font-semibold text-ink">Change the hidden label, recompute x₀</div>
              <Segmented
                label="Stored label used for the check"
                value={alt}
                onChange={setAlt}
                options={[
                  { value: 'stored', label: `as stored (${fmtSigned(check.stored, 2)})` },
                  { value: 'neg', label: `flipped (${fmtSigned(-check.stored, 2)})` },
                  { value: 'zero', label: '0' },
                  { value: 'big', label: '1000' },
                ]}
              />
              <div className="mt-3 flex items-center gap-2 text-[16px]">
                {check.delta === 0 ? (
                  <StatusBadge status="good">x₀ identical: max |Δ| = 0</StatusBadge>
                ) : (
                  <StatusBadge status="critical">max |Δ| = {check.delta.toExponential(2)}</StatusBadge>
                )}
              </div>
              <p className="!mt-2 !mb-0 text-[15px] leading-snug text-ink-2">
                {VALUE_FIELD[tgt.sem]} = {fmtSigned(check.v, 2)} gives the same {data.d} numbers as the forward pass. Any finite value would; a NaN
                would not, because NaN·0 = NaN <Cite kind="spec" ref="§5.3" />.
              </p>
            </div>
          )}
        </div>
      </div>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[560px] border-collapse text-[15px]">
          <thead>
            <tr className="border-b border-grid text-left text-ink-2">
              <th scope="col" className="py-1.5 pr-3 font-medium">
                Mask vector
              </th>
              <th scope="col" className="py-1.5 pr-3 font-medium">
                Masked as a target?
              </th>
            </tr>
          </thead>
          <tbody>
            {TYPES.map((s) => (
              <tr key={s} className="border-b border-grid/60 last:border-0">
                <th scope="row" className="py-1.5 pr-3 text-left font-normal text-ink">
                  <span className="inline-flex items-center gap-2">
                    <SemBadge sem={s} size="xs" />
                    <span className="font-mono">m_{SEM_KEYS[s]}</span>
                  </span>
                </th>
                <td className="py-1.5 pr-3 text-ink-2">
                  {s === SemType.Number || s === SemType.Boolean ? (
                    <>
                      Yes: every RelBench task is regression or binary classification <Cite kind="paper" ref="p.6 §3.3" />.
                      {tasksBySem[s].length > 0 && <> Here: {tasksBySem[s].join(', ')}.</>}
                    </>
                  ) : s === SemType.DateTime ? (
                    <>
                      Not in the paper’s tasks. The code would accept it and train it with the Huber loss <Cite kind="code" ref="rt/model.py:250-251" />.
                    </>
                  ) : (
                    <>
                      Never: the loss raises “masking text not supported” <Cite kind="code" ref="rt/model.py:256-257" />.
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Figure>
  );
}

/** Task names of a DB grouped by the sem type of their target column (sem type = physical dtype, spec §3.2). */
function targetSems(db: RawDatabase): Record<SemType, string[]> {
  const out: Record<SemType, string[]> = { [SemType.Number]: [], [SemType.Text]: [], [SemType.DateTime]: [], [SemType.Boolean]: [] };
  for (const task of db.tasks) {
    const schema = task.kind === 'forecast' ? task.tables?.Train.schema : db.tables.find((t) => t.schema.name === task.table)?.schema;
    const c = schema?.columns.find((x) => x.name === task.targetCol);
    if (c) out[semOfDType(c.dtype)].push(task.kind === 'forecast' ? task.name : `${task.table ?? task.name}.${task.targetCol}`);
  }
  return out;
}

function Knows({ ok, children }: { ok: boolean; children: ReactNode }) {
  return (
    <li className="flex items-start gap-2.5 text-[16px] leading-snug text-ink-2">
      <span
        className={cx(
          'mt-[2px] inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full',
          ok ? 'bg-surface-3 text-good-text' : 'bg-surface-3 text-critical',
        )}
        aria-label={ok ? 'known' : 'not known'}
      >
        {ok ? <Check size={14} strokeWidth={2.6} aria-hidden="true" /> : <X size={14} strokeWidth={2.6} aria-hidden="true" />}
      </span>
      <span>{children}</span>
    </li>
  );
}
