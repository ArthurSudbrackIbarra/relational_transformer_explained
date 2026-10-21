/**
 * Section 7: the worked example of spec §6.3. The engine's `buildMasks` runs on the 24 toy tokens; its
 * output is printed in the spec's '#'/'.' style and compared with the masks printed in the spec. A
 * StepPlayer walks through the query rows and explains every allowed key.
 */
import { useMemo, useState, type ReactNode } from 'react';
import { Figure, LayerLabel, Segmented, StatusBadge, StepPlayer, TokenStrip, cx, useStepPlayer } from '@/components';
import { buildMasks, maskToAscii } from '@/engine/masks';
import type { AttnLayer, Token } from '@/engine/types';
import { FitMask } from './FitMask';
import { LAYER_ORDER } from './maskUtils';
import { SPEC_PRINTED, TOY_L, TOY_REAL, TOY_SLOTS, TOY_TOKENS, toyRowName } from './toy';

const LABELS = TOY_TOKENS.map((t) => (t.isPadding ? `PAD #${t.pos}` : `${toyRowName(t)}·${t.colName}`));
const SLOTS = TOY_TOKENS.map((t) => (t.isPadding ? 0 : TOY_SLOTS[t.tableName] ?? 0));

function positionsOf(pred: (t: Token) => boolean): number[] {
  return TOY_TOKENS.filter((t) => !t.isPadding && pred(t)).map((t) => t.pos);
}

function ranges(ps: number[]): string {
  if (ps.length === 0) return 'none';
  const out: string[] = [];
  let a = ps[0];
  let b = ps[0];
  for (const p of [...ps.slice(1), Infinity]) {
    if (p === b + 1) b = p;
    else {
      out.push(a === b ? `${a}` : `${a}-${b}`);
      a = b = p;
    }
  }
  return out.join(', ');
}

function rowLabel(node: number): string {
  const t = TOY_TOKENS.find((x) => !x.isPadding && x.nodeIdx === node);
  return t ? toyRowName(t) : `node ${node}`;
}

/** Why each key is allowed for query q (derived from the token ids, the same tests as model.py:165-191). */
function explain(layer: AttnLayer, q: number, allowed: number[]): ReactNode {
  const t = TOY_TOKENS[q];
  if (t.isPadding) return <>Token {q} is padding (is_padding = true): the AND with <code>pad</code> blocks its whole row in every mask.</>;
  const name = `${toyRowName(t)}·${t.colName}`;
  switch (layer) {
    case 'col': {
      return (
        <>
          Query {q} is <code>{t.colName} of {t.tableName}</code>. Keys with the same column id and table id: <strong>{ranges(allowed)}</strong>.
          {t.isTarget && ' These are the target itself and the three visible churn labels (6, 12: c0’s own; 18: c2’s).'}
        </>
      );
    }
    case 'feat': {
      const own = positionsOf((k) => k.nodeIdx === t.nodeIdx);
      const parents = t.f2pNbrIdxs.filter((p) => p >= 0);
      const parts = parents.map((p) => {
        const ps = positionsOf((k) => k.nodeIdx === p);
        return `${rowLabel(p)} → ${ps.length ? ranges(ps) : 'not in the context'}`;
      });
      return (
        <>
          Query {q} ({name}): own row {toyRowName(t)} → <strong>{ranges(own)}</strong>
          {parents.length ? (
            <>
              ; parents (f2p) {parts.map((s, i) => (
                <span key={i}>
                  {i > 0 && ', '}
                  <strong>{s}</strong>
                </span>
              ))}
            </>
          ) : (
            '; no parents (a customers or products row has no foreign key)'
          )}
          .
        </>
      );
    }
    case 'nbr': {
      const kids = [...new Set(allowed.map((k) => TOY_TOKENS[k].nodeIdx))].map(rowLabel);
      return allowed.length ? (
        <>
          Query {q} ({name}): rows whose parent list contains {toyRowName(t)}: {kids.join(', ')} → <strong>{ranges(allowed)}</strong>.
        </>
      ) : (
        <>
          Query {q} ({name}): no row in the context has {toyRowName(t)} among its parents, so the row is empty and this token gets a zero update.
        </>
      );
    }
    case 'full':
      return (
        <>
          Query {q} ({name}): every non-padding key, <strong>0-{TOY_REAL - 1}</strong>.
        </>
      );
  }
}

export function WorkedExample() {
  const [layer, setLayer] = useState<AttnLayer>('col');
  const masks = useMemo(() => buildMasks(TOY_TOKENS), []);
  const ascii = useMemo(() => {
    const out = {} as Record<AttnLayer, string[]>;
    for (const l of LAYER_ORDER) out[l] = maskToAscii(masks[l], TOY_L);
    return out;
  }, [masks]);
  const matches = useMemo(() => {
    const out = {} as Record<AttnLayer, boolean>;
    for (const l of LAYER_ORDER) out[l] = ascii[l].every((row, i) => row === SPEC_PRINTED[l][i]);
    return out;
  }, [ascii]);
  const player = useStepPlayer(TOY_L, { intervalMs: 1400, initial: 1 });
  const q = player.index;
  const allowed = useMemo(() => {
    const out: number[] = [];
    for (let k = 0; k < TOY_L; k++) if (masks[layer][q * TOY_L + k]) out.push(k);
    return out;
  }, [masks, layer, q]);

  return (
    <Figure
      wide
      title="The spec’s worked example, rebuilt by the engine"
      controls={
        <>
          <Segmented label="Mask" value={layer} onChange={setLayer} options={LAYER_ORDER.map((l) => ({ value: l, label: <LayerLabel layer={l} size={13} /> }))} />
          <div className="flex flex-wrap gap-1.5">
            {LAYER_ORDER.map((l) => (
              <StatusBadge key={l} status={matches[l] ? 'good' : 'critical'} className="!text-[14px]">
                {l}: {matches[l] ? 'matches spec §6.3' : 'differs'}
              </StatusBadge>
            ))}
          </div>
        </>
      }
      caption={
        <>
          The toy ShopDB of spec §6.3: seed = the churn Val row #16 (customer c0, 2024-03-01), L = 24, w = 1. Row and column numbers are token
          positions; <code>#</code> = allowed, <code>.</code> = blocked. The strings on the left are printed from the engine’s{' '}
          <code>buildMasks</code> and compared character by character with the spec.
        </>
      }
    >
      <div className="mb-4">
        <TokenStrip
          tokens={TOY_TOKENS}
          tableSlots={TOY_SLOTS}
          highlight={allowed}
          dimOthers
          selected={q}
          onSelect={(p) => player.setIndex(p)}
          showPadding
          label="Toy context tokens (click one to make it the query)"
        />
      </div>
      <div className="grid items-start gap-6 lg:grid-cols-[auto_minmax(0,1fr)]">
        <AsciiGrid rows={ascii[layer]} q={q} allowed={allowed} />
        <FitMask
          n={TOY_L}
          labelled
          maxCell={15}
          bandRow={q}
          matrix={{
            mask: masks[layer],
            S: TOY_L,
            labels: LABELS,
            slots: SLOTS,
            onCellClick: (c) => player.setIndex(c.q),
            title: `${layer} mask of the toy context`,
          }}
        />
      </div>
      <StepPlayer {...player} unit="step" className="mt-5" label="Step through the query rows" describe={() => explain(layer, q, allowed)} />
    </Figure>
  );
}

function AsciiGrid({ rows, q, allowed }: { rows: string[]; q: number; allowed: number[] }) {
  const header = Array.from({ length: TOY_L }, (_, i) => String(i % 10)).join('');
  const allow = new Set(allowed);
  return (
    <div className="scroll-quiet min-w-0 overflow-x-auto rounded-lg bg-[var(--code-bg)] px-3 py-2.5">
      <pre className="code-block !text-[15px] !leading-[1.45] text-[var(--code-ink)]" aria-label="Mask as text: one line per query">
        <div className="text-[var(--code-comment)]">
          {'    '}
          {header}
        </div>
        {rows.map((r, i) => (
          <div key={i} className={cx('rounded-[3px]', i === q && 'bg-[var(--code-line-hl)] outline outline-1 outline-[var(--accent)]')}>
            <span className="text-[var(--code-comment)]">{String(i).padStart(2, ' ')}  </span>
            {r.split('').map((ch, k) =>
              ch === '#' ? (
                <span key={k} className={i === q && allow.has(k) ? 'font-bold text-accent-text' : 'text-[var(--code-function)]'}>
                  #
                </span>
              ) : (
                <span key={k} className="text-[var(--code-comment)] opacity-60">
                  .
                </span>
              ),
            )}
          </div>
        ))}
      </pre>
    </div>
  );
}
