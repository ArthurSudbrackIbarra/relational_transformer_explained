/**
 * Section 6 figures:
 *  - PaddingDemo: the spec §6.3 toy context (5 padding tokens) with and without the `& pad` of
 *    model.py:187-188. "Without" is a labeled counterfactual on the raw ids.
 *  - TransposeFlip: the parent part of the feature mask, flipped across the diagonal, is the neighbor
 *    mask (live context; checked on every pair).
 *  - ZeroUpdateCheck: in the live forward pass, the neighbor sublayer's update is exactly 0 wherever the
 *    neighbor mask row is empty (spec §6.1).
 *  - ColumnTableCheck: the extra table test of the column mask changes no pair (spec §6.2, §10 #12).
 */
import { FlipHorizontal2 } from 'lucide-react';
import { animate, motion, useMotionValue, useReducedMotion, useTransform } from 'motion/react';
import { useMemo, useRef, useState } from 'react';
import { Button, Cite, Figure, LayerLabel, MaskMatrix, ScaleNote, Segmented, ShapeTag, StatusBadge, formatNumber } from '@/components';
import { axis, span } from '../shapes';
import { buildMasks } from '@/engine/masks';
import { useForward } from '@/state/hooks';
import { FitMask } from './FitMask';
import { TableKey } from './FourMasks';
import { countRow, diffCount, int, masksWithoutPad, parentPart, sameColumnOtherTable, transpose } from './maskUtils';
import { TOY_L, TOY_REAL, TOY_SHAPE, TOY_SLOTS, TOY_TOKENS, toyRowName } from './toy';
import type { LiveMasks } from './useLiveMasks';

const TOY_LABELS = TOY_TOKENS.map((t) => (t.isPadding ? `PAD #${t.pos}` : `${toyRowName(t)}·${t.colName}`));
const TOY_TOKEN_SLOTS = TOY_TOKENS.map((t) => (t.isPadding ? 0 : TOY_SLOTS[t.tableName] ?? 0));
const TOY_TABLES = Object.entries(TOY_SLOTS)
  .filter(([name]) => TOY_TOKENS.some((t) => !t.isPadding && t.tableName === name))
  .map(([name, slot]) => ({ name, slot }));

// ─────────────────────────────────────────────────────────────────────────────

export function PaddingDemo() {
  const [layer, setLayer] = useState<'feat' | 'nbr'>('feat');
  const withPad = useMemo(() => buildMasks(TOY_TOKENS), []);
  const noPad = useMemo(() => masksWithoutPad(TOY_TOKENS), []);
  const changed = diffCount(withPad[layer], noPad[layer]);
  // pairs that differ from the code's mask (q·24 + k), outlined in the counterfactual matrix
  const diffCells = useMemo(() => {
    const out: number[] = [];
    for (let i = 0; i < TOY_L * TOY_L; i++) if (withPad[layer][i] !== noPad[layer][i]) out.push(i);
    return out;
  }, [withPad, noPad, layer]);
  const changedRows = useMemo(() => {
    const rows: number[] = [];
    for (let q = 0; q < TOY_L; q++) {
      let d = false;
      for (let k = 0; k < TOY_L && !d; k++) d = withPad[layer][q * TOY_L + k] !== noPad[layer][q * TOY_L + k];
      if (d) rows.push(q);
    }
    return rows;
  }, [withPad, noPad, layer]);
  return (
    <Figure
      wide
      title="Why the masks are ANDed with “not padding” (toy context of spec §6.3, 19 cells + 5 padding)"
      controls={
        <Segmented
          label="Mask"
          value={layer}
          onChange={setLayer}
          options={[
            { value: 'feat', label: <LayerLabel layer="feat" size={13} /> },
            { value: 'nbr', label: <LayerLabel layer="nbr" size={13} /> },
          ]}
        />
      }
      caption={
        <>
          Left: the mask the code builds, <code>{layer === 'feat' ? '(same_node | kv_in_f2p) & pad' : 'q_in_f2p & pad'}</code>. Right: the same
          expression without <code>& pad</code>, evaluated on the raw ids (padding has node −1 and parents −1×5). The five padding tokens are the
          last rows and columns (gray strips). Without the AND, {changedRows.length} of the 24 query rows change; the changed pairs are outlined
          in red. Where the labels do not fit (narrow screens), the axes show token positions instead.
        </>
      }
    >
      <div className="grid gap-x-6 gap-y-6 md:grid-cols-2">
        <FitMask
          n={TOY_L}
          labeled
          fitLabels
          ticks={4}
          maxCell={15}
          header={<div className="mb-2 text-[16px] font-medium text-ink">With &amp; pad (the code)</div>}
          matrix={{ mask: withPad[layer], S: TOY_L, labels: TOY_LABELS, slots: TOY_TOKEN_SLOTS, title: `${layer} mask with the padding AND`, showLegend: false, shape: TOY_SHAPE }}
        />
        <FitMask
          n={TOY_L}
          labeled
          fitLabels
          ticks={4}
          maxCell={15}
          marks={diffCells}
          header={
            <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3 text-[16px] font-medium text-ink">
              <span>
                Without &amp; pad <span className="font-normal text-muted">(counterfactual)</span>
              </span>
              <span className="num text-[15px] font-normal text-ink-2">
                <span aria-hidden="true" className="mr-1 inline-block h-3 w-3 rounded-[2px] align-[-1px]" style={{ border: '2px solid var(--critical)' }} />
                {int(changed)} pairs differ
              </span>
            </div>
          }
          matrix={{ mask: noPad[layer], S: TOY_L, labels: TOY_LABELS, slots: TOY_TOKEN_SLOTS, title: `${layer} mask without the padding AND`, showLegend: false, shape: TOY_SHAPE }}
        />
      </div>
      <div className="mt-3">
        <TableKey tables={TOY_TABLES} padding />
      </div>
      <p className="mt-4 text-[16px] leading-snug text-ink-2">
        <strong className="text-ink">{int(changed)} pairs</strong> would change.{' '}
        {layer === 'feat'
          ? `Every real cell whose parent list has an empty (−1) slot would match the padding keys (node −1) as if they were a parent, and the ${TOY_L - TOY_REAL} padding tokens would all share one “row”.`
          : `Every padding query (node −1) would be found in the parent list of every key that has an empty (−1) slot, which here is every token, since no row has five parents; so it would “see” those tokens as its children.`}
      </p>
    </Figure>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

const FLIP_N = 40;

export function TransposeFlip({ live }: { live: LiveMasks }) {
  const { ctx, masks, S, tokenSlots, grouping } = live;
  const reduce = useReducedMotion();
  const parent = useMemo(() => parentPart(ctx.tokens), [ctx]);
  const diff = useMemo(() => diffCount(transpose(parent, S), masks.nbr), [parent, masks, S]);
  const n = Math.min(FLIP_N, ctx.numReal);
  const spanN = { shown: n, L: ctx.params.L, real: ctx.numReal };
  const cell = 6;
  const side = 7 + n * cell; // identity strip (5 + 2 px) + n cells
  const angle = useMotionValue(0);
  const transform = useTransform(angle, (a) => `perspective(1100px) rotate3d(1, 1, 0, ${a}deg)`);
  const [flipped, setFlipped] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  // The rotation axis must run along the matrix diagonal, i.e. through the canvas's top-left corner.
  const [origin, setOrigin] = useState('0px 0px');

  const flip = () => {
    const to = flipped ? 0 : 180;
    if (!flipped && angle.get() === 0) {
      const wrap = wrapRef.current;
      const canvas = wrap?.querySelector('canvas');
      if (wrap && canvas) {
        const a = wrap.getBoundingClientRect();
        const b = canvas.getBoundingClientRect();
        setOrigin(`${b.left - a.left}px ${b.top - a.top}px`);
      }
    }
    setFlipped(!flipped);
    if (reduce) angle.set(to);
    else animate(angle, to, { duration: 1.3, ease: [0.45, 0, 0.2, 1] });
  };

  return (
    <Figure
      wide
      title="Neighbor is the feature mask’s parent part, transposed"
      controls={
        <Button variant="primary" size="sm" icon={<FlipHorizontal2 aria-hidden="true" size={15} />} onClick={flip}>
          {flipped ? 'Flip back' : 'Flip across the diagonal'}
        </Button>
      }
      caption={
        <>
          Left: <code>kv_in_f2p &amp; pad</code>, the part of the feature mask where the key’s row is a parent of the query’s row{' '}
          <Cite kind="code" ref="rt/model.py:171-173" />. Flipping it across the diagonal swaps query and key, which is exactly{' '}
          <code>q_in_f2p &amp; pad</code> <Cite kind="code" ref="rt/model.py:176-178" />. First {n} tokens of the live context shown; the check
          below covers all {int(S * S)} pairs.
        </>
      }
    >
      <div className="flex flex-wrap items-start justify-center gap-x-10 gap-y-6">
        <div className="min-w-0">
          <div className="mb-2 text-[15.5px] font-medium text-ink">
            Feature, parent part{flipped ? <span className="text-accent-text">, transposed</span> : null}
          </div>
          <AxisTitles side={side}>
          <div style={{ width: side }} className="pointer-events-none">
            <motion.div ref={wrapRef} style={{ transform, transformOrigin: origin, width: side, transformStyle: 'flat' }}>
              <MaskMatrix
                mask={parent}
                S={S}
                crop={n}
                slots={tokenSlots}
                cellSize={cell}
                title="feature mask, parent part"
                showLegend={false}
                showReadout={false}
                shape={null}
              />
            </motion.div>
          </div>
          </AxisTitles>
          <div className="mt-1 pl-[22px]">
            <ShapeTag
              rows={n}
              cols={n}
              shape={
                flipped
                  ? {
                      rows: `one per key token, the one being looked at (transposed, so the keys now run down the rows): ${span(spanN)}`,
                      cols: `one per query token, in the same order: ${span(spanN)}`,
                    }
                  : { rows: axis.queries(spanN), cols: axis.keys(spanN) }
              }
            />
          </div>
        </div>
        <div className="min-w-0">
          <div className="mb-2 text-[15.5px] font-medium text-ink">
            <LayerLabel layer="nbr" size={13} /> <span className="font-normal text-muted">mask (engine)</span>
          </div>
          <AxisTitles side={side}>
            <div style={{ width: side }}>
              <MaskMatrix mask={masks.nbr} S={S} crop={n} slots={tokenSlots} cellSize={cell} title="neighbor mask" showLegend={false} showReadout={false} shape={null} />
            </div>
          </AxisTitles>
          <div className="mt-1 pl-[22px]">
            <ShapeTag rows={n} cols={n} shape={{ rows: axis.queries(spanN), cols: axis.keys(spanN) }} />
          </div>
        </div>
      </div>
      <div className="mt-4 flex justify-center">
        <TableKey tables={grouping.tables} />
      </div>
      <div className="mt-3 flex justify-center">
        {diff === 0 ? (
          <StatusBadge status="good" className="!rounded-lg">
            Checked on the live context: nbr = (parent part of feat)ᵀ on all {int(S * S)} pairs
          </StatusBadge>
        ) : (
          <StatusBadge status="critical" className="!rounded-lg">
            {int(diff)} pairs differ
          </StatusBadge>
        )}
      </div>
    </Figure>
  );
}

/** "keys →" above and "queries ↓" left of a fixed-size matrix. */
function AxisTitles({ side, children }: { side: number; children: React.ReactNode }) {
  return (
    <div style={{ width: side + 22 }}>
      <div className="pl-[29px] text-[14px] leading-tight text-muted" aria-hidden="true">
        keys →
      </div>
      <div className="flex">
        <div className="relative w-[22px] shrink-0" aria-hidden="true">
          <span className="absolute top-[7px] left-0 text-[14px] leading-none whitespace-nowrap text-muted" style={{ writingMode: 'vertical-rl' }}>
            queries →
          </span>
        </div>
        {children}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

/** Live check: wherever a neighbor mask row is empty, the sublayer's update is exactly 0 (spec §6.1). */
export function ZeroUpdateCheck({ live }: { live: LiveMasks }) {
  const { ctx, masks, S } = live;
  const fr = useForward();
  const res = useMemo(() => {
    if (!fr) return null;
    const empty: number[] = [];
    for (let q = 0; q < S; q++) if (!ctx.tokens[q].isPadding && countRow(masks.nbr, S, q) === 0) empty.push(q);
    let maxEmpty = 0;
    let minOther = Infinity;
    let blocks = 0;
    for (const b of fr.blocks) {
      const cap = b.attn.nbr;
      if (cap.skipped || cap.out.length < S * fr.d) continue;
      blocks++;
      const emptySet = new Set(empty);
      for (let q = 0; q < S; q++) {
        if (ctx.tokens[q].isPadding) continue;
        let m = 0;
        for (let j = 0; j < fr.d; j++) m = Math.max(m, Math.abs(cap.out[q * fr.d + j]));
        if (emptySet.has(q)) maxEmpty = Math.max(maxEmpty, m);
        else minOther = Math.min(minOther, m);
      }
    }
    return { empty: empty.length, maxEmpty, minOther, blocks };
  }, [fr, ctx, masks, S]);
  if (!res) return <p className="text-[16.5px] text-ink-2">Running the forward pass to check this live…</p>;
  if (res.blocks === 0) return <p className="text-[16.5px] text-ink-2">The neighbor sublayer is switched off in the playground’s ablation settings.</p>;
  return (
    <div className="my-5 flex flex-wrap items-center gap-2">
      <StatusBadge status={res.maxEmpty === 0 ? 'good' : 'critical'} className="!rounded-lg">
        max |update| = {res.maxEmpty === 0 ? '0' : formatNumber(res.maxEmpty, 6)} at {int(res.empty)} empty-neighbor tokens, all {res.blocks} blocks
      </StatusBadge>
      {Number.isFinite(res.minOther) && (
        <StatusBadge status="info" className="!rounded-lg">
          other tokens: smallest max |update| {formatNumber(res.minOther, 4)}
        </StatusBadge>
      )}
      <ScaleNote inline />
    </div>
  );
}

/** Live check: requiring the same table on top of the same column id changes no pair. */
export function ColumnTableCheck({ live }: { live: LiveMasks }) {
  const n = useMemo(() => sameColumnOtherTable(live.ctx.tokens), [live.ctx]);
  return (
    <StatusBadge status={n === 0 ? 'good' : 'warning'} className="!rounded-lg">
      Live context: {n === 0 ? 'no pair' : `${int(n)} pairs`} with the same column id but a different table id
    </StatusBadge>
  );
}
