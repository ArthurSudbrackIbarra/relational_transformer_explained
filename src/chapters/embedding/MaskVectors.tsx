/**
 * Step 5 figure: the four mask vectors m_t (mask_embs, model.py:141-146) and the masked target of
 * the live context. Everything is read from the live weights, the exported init weights and the
 * forward pass.
 *
 * The "change the hidden label" panel shows the official mechanism (model.py:216-224): the code
 * evaluates the target type's value branch on the stored label, a nonzero vector, and multiplies it
 * by the gate 0. Our engine's `embedTokens` skips gate-0 branches, so its recomputed x₀ equals the
 * forward pass by construction; the panel says so.
 */
import { Check, X } from 'lucide-react';
import { memo, useMemo, useState, type ReactNode } from 'react';
import { Cite, Figure, Segmented, SemBadge, StatusBadge, TeX, VectorStrip, cx, maxAbs, semOfDType } from '@/components';
import { axis, vec } from '../shapes';
import { SEM_KEYS, embedTokens, scalarValueTerm, tensor, textEmbedder } from '@/engine/model';
import { SemType, type RawDatabase, type Token } from '@/engine/types';
import type { EmbeddingStable } from './live';
import { fmt, fmtSigned, phraseOf, rowRef, useInitWeights, VALUE_FIELD } from './live';
import { cosine, maxAbsDelta, norm, slice, tokenScalar, type ScalarKey } from './math';

const TYPES = [SemType.Number, SemType.Text, SemType.DateTime, SemType.Boolean];

function withStoredValue(t: Token, v: number): Token {
  if (t.sem === SemType.Number) return { ...t, numberValue: v };
  if (t.sem === SemType.DateTime) return { ...t, datetimeValue: v };
  return { ...t, booleanValue: v };
}

export const MaskVectors = memo(function MaskVectors({ live }: { live: EmbeddingStable }) {
  const { weights, store, fr, ctx, pdb, target, assets, db } = live;
  const tgt = ctx.tokens[target];
  const init = useInitWeights();
  const [alt, setAlt] = useState<'stored' | 'neg' | 'zero' | 'big'>('neg');

  const data = useMemo(() => {
    if (!weights) return null;
    const d = weights.config.dModel;
    const m = TYPES.map((s) => tensor(weights, `mask_embs.${SEM_KEYS[s]}`));
    const m0 = init && init.config.dModel === d ? TYPES.map((s) => tensor(init, `mask_embs.${SEM_KEYS[s]}`)) : null;
    return { d, m, m0, domain: Math.max(...m.map((v) => maxAbs(v))) || 1 };
  }, [weights, init]);

  const check = useMemo(() => {
    if (!weights || !store || !fr || !tgt || !tgt.mask || tgt.sem === SemType.Text) return null;
    const d = weights.config.dModel;
    const key = SEM_KEYS[tgt.sem] as ScalarKey;
    const stored = tokenScalar(tgt);
    const v = alt === 'stored' ? stored : alt === 'neg' ? -stored : alt === 'zero' ? 0 : 1000;
    // the branch the official code evaluates for this position before the ×gate (model.py:218)
    const branch = scalarValueTerm(weights, key, v).term;
    const emb = embedTokens([withStoredValue({ ...tgt, pos: 0 }, v)], textEmbedder(pdb, store), weights);
    const live0 = slice(fr.embedding.x0, tgt.pos, d);
    return { key, stored, v, branch, delta: maxAbsDelta(emb.x0, live0) };
  }, [weights, store, fr, tgt, pdb, alt]);

  if (!data || !tgt) {
    return (
      <Figure wide title="The four mask vectors">
        <div className="py-6 text-center text-[16px] text-ink-2">Loading the weights…</div>
      </Figure>
    );
  }
  const tKey = SEM_KEYS[tgt.sem];
  const tasksBySem = targetSems(db);
  const parent = tgt.f2pNbrIdxs.find((p) => p >= 0);
  const parentTok = parent !== undefined ? ctx.tokens.find((t) => !t.isPadding && t.nodeIdx === parent) : undefined;
  const cosInit = (i: number) => (data.m0 ? cosine(data.m[i], data.m0[i]) : NaN);
  const minCos = data.m0 ? Math.min(...TYPES.map((_, i) => cosInit(i))) : NaN;

  return (
    <Figure
      wide
      title="The four mask vectors, and what the masked target still knows"
      caption={
        <>
          Left: the four mask vectors of the checkpoint this page runs, on one shared color scale, each compared with its starting value. Right:
          the live masked target, and what the code does with its hidden label.
        </>
      }
    >
      {assets.weightsKind === 'init' && <StatusBadge status="warning">Untrained weights: values from a random initialization</StatusBadge>}
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div className="grid gap-3.5">
          {TYPES.map((s, i) => {
            const used = s === tgt.sem;
            return (
              <div key={s} className={cx('rounded-lg border px-3 py-2.5', used ? 'border-accent bg-accent-soft/40' : 'border-border bg-surface')}>
                <div className="mb-1.5 flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
                  <span className="inline-flex items-center gap-2 text-[16px] text-ink">
                    <SemBadge sem={s} size="sm" />
                    <span>m_{SEM_KEYS[s]}</span>
                    {used && <span className="rounded-full bg-accent px-2 py-px text-[13.5px] font-medium text-accent-ink">live target</span>}
                  </span>
                  <span className="num text-[15px] text-ink-2">
                    {data.m0 ? (
                      <>
                        ‖m‖ {fmt(norm(data.m0[i]), 2)} → <span className="text-ink">{fmt(norm(data.m[i]), 2)}</span> · cos with start{' '}
                        <span className="text-ink">{cosInit(i).toFixed(4)}</span>
                      </>
                    ) : (
                      <>
                        ‖m‖ = <span className="text-ink">{fmt(norm(data.m[i]), 2)}</span>
                      </>
                    )}
                  </span>
                </div>
                <VectorStrip
                  values={data.m[i]}
                  columns={32}
                  cellHeight={11}
                  domain={data.domain}
                  showNorm={false}
                  shape={vec(`m_${SEM_KEYS[s]} (mask_embs.${SEM_KEYS[s]}), the learned vector added in place of a masked ${SEM_KEYS[s]} value`, axis.model(data.d))}
                />
              </div>
            );
          })}
          <div className="grid gap-2 text-[15px] leading-snug text-ink-2 [&_code]:text-[14.5px]">
            <p className="!m-0">
              Each <Mt /> starts as <code>torch.randn({data.d})</code>, a standard normal vector of length about √{data.d} = {fmt(Math.sqrt(data.d), 0)}, as
              long as a normalized value term with γ = 1 <Cite kind="spec" ref="§7.4" />. Only <code>m_number</code> and <code>m_boolean</code> are
              ever trained, because only a masked token of type t adds <Mt /> <Cite kind="code" ref="rt/model.py:221-224" />. RT never masks text{' '}
              <Cite kind="code" ref="rt/model.py:256-257" />, and no task in the paper has a datetime target <Cite kind="code" ref="rt/tasks.py:3-88" />,
              so <code>m_text</code> and <code>m_datetime</code> keep their random start, shrunk only by weight decay{' '}
              <Cite kind="code" ref="rt/main.py:188-195" />.
            </p>
            {data.m0 && (
              <p className="!m-0">
                In the checkpoint this page runs, every <Mt /> still has a cosine of at least {minCos.toFixed(4)} with its starting value
                {minCos > 0.98 ? ', so these patterns say more about the random start than about anything learned.' : '.'}
              </p>
            )}
          </div>
        </div>

        <div className="min-w-0">
          <div className="rounded-xl border border-border bg-surface-2/60 p-4">
            <div className="mb-1 text-[15px] text-ink-2">The masked target of the live context</div>
            <div className="mb-3 flex flex-wrap items-center gap-2 text-[16.5px]">
              <span className="text-ink">{rowRef(tgt)}</span>
              <SemBadge sem={tgt.sem} size="sm" showLabel />
              <span className="target-outline rounded px-1.5 font-bold text-accent-text">?</span>
            </div>
            <ul className="grid gap-2.5">
              <Knows ok>
                <strong className="font-semibold text-ink">Which column of which table</strong>: its column term embeds “{phraseOf(tgt)}”.
              </Knows>
              <Knows ok>
                <strong className="font-semibold text-ink">Which type</strong>: the gate picks <span>m_{tKey}</span>, one of
                four different vectors.
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
              <div className="mb-2 text-[16px] font-semibold text-ink">Change the hidden label, follow it through the code</div>
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
              <ol className="mt-3 grid list-none gap-2.5 pl-0 text-[15px] leading-snug text-ink-2">
                <li>
                  <div className="mb-1">
                    <span className="font-semibold text-ink">a.</span> The code evaluates the {check.key} branch on the stored label{' '}
                    {VALUE_FIELD[tgt.sem]} = {fmtSigned(check.v, 2)}, like at every position <Cite kind="code" ref="rt/model.py:218" />:
                  </div>
                  <VectorStrip
                    values={check.branch}
                    columns={32}
                    cellHeight={10}
                    domain={data.domain}
                    label={undefined}
                    hint={false}
                    shape={vec(`RMSNorm_${check.key}(W v + b) on the hidden label: the branch the gate multiplies by 0`, axis.model(data.d))}
                  />
                  <div className="mt-0.5">
                    <span className="text-[17px]">
                      <TeX tex={String.raw`\mathrm{RMSNorm}_{\mathrm{${check.key}}}(W v + b)`} />
                    </span>{' '}
                    has length <span className="num text-ink">{fmt(norm(check.branch), 2)}</span>: not zero.
                  </div>
                </li>
                <li>
                  <span className="font-semibold text-ink">b.</span> It multiplies that vector by its gate{' '}
                  <span className="whitespace-nowrap text-[17px]">
                    <TeX tex={String.raw`[t_i{=}\mathrm{${check.key}} \land \lnot\mathrm{mask}_i \land \lnot\mathrm{pad}_i] = 0`} />
                  </span>
                  , which gives <TeX tex="\vec 0" />, and adds <span>m_{check.key}</span> times its gate 1{' '}
                  <Cite kind="code" ref="rt/model.py:216-224" />. So x₀ = column term + <span>m_{check.key}</span>, whatever the
                  label.
                </li>
                <li>
                  <span className="font-semibold text-ink">c.</span> The engine’s <code className="text-[14.5px]">embedTokens</code> on a copy of the
                  target with this label, against the forward pass:{' '}
                  {check.delta === 0 ? (
                    <StatusBadge status="good">x₀ identical: max |Δ| = 0</StatusBadge>
                  ) : (
                    <StatusBadge status="critical">max |Δ| = {check.delta.toExponential(2)}</StatusBadge>
                  )}
                </li>
              </ol>
              <p className="!mt-2.5 !mb-0 text-[15px] leading-snug text-ink-2">
                Step b is why the hidden label never reaches x₀: its branch is multiplied by 0 before the sum <Cite kind="spec" ref="§5.3" />. The
                engine on this site skips that branch altogether, so check c finds the same x₀ for every label.
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="mt-5">
        <div className="mb-1.5 text-[15.5px] font-semibold text-ink">Which mask vectors are ever used</div>
        <dl className="divide-y divide-grid/60 border-y border-grid text-[15px]">
          {TYPES.map((s) => (
            <div key={s} className="grid gap-x-4 gap-y-1 py-2 sm:grid-cols-[170px_minmax(0,1fr)]">
              <dt className="inline-flex items-center gap-2 text-ink">
                <SemBadge sem={s} size="sm" />
                <span>m_{SEM_KEYS[s]}</span>
              </dt>
              <dd className="m-0 min-w-0 text-ink-2">
                {s === SemType.Number || s === SemType.Boolean ? (
                  <>
                    Masked as a target: every RelBench task is regression or binary classification <Cite kind="paper" ref="p.6 §3.3" />.
                    {tasksBySem[s].length > 0 && (
                      <>
                        {' '}
                        Here: {tasksBySem[s].join(', ')}.
                      </>
                    )}
                  </>
                ) : s === SemType.DateTime ? (
                  <>
                    No task in the paper predicts a datetime <Cite kind="code" ref="rt/tasks.py:3-88" />.
                    {tasksBySem[s].length > 0 ? <> This database does: {tasksBySem[s].join(', ')}.</> : <> This database has none either.</>}
                  </>
                ) : (
                  <>
                    Never: RT does not mask text, and the loss raises “masking text not supported” <Cite kind="code" ref="rt/model.py:256-257" />.
                  </>
                )}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </Figure>
  );
});

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

/** m_t inline at 17.5px, so its subscript stays above the 12px floor inside 15px text. */
function Mt() {
  return (
    <span className="text-[17.5px]">
      <TeX tex="m_t" />
    </span>
  );
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
