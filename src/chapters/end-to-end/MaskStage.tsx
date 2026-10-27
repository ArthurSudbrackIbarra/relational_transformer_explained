/**
 * Stage 5: the four relational attention masks of the live context (rt/model.py:164-191; spec §6.2),
 * built once from the token ids and shared by every head and every block. Hovering an entry marks
 * the same (query, key) pair in all four and says which condition allows it (./math.ts `maskReasons`,
 * checked against the masks in math.test.ts).
 */
import { Cite, Figure, LayerLabel } from '@/components';
import { useElementSize } from '@/components/viz/useSize';
import { ATTN_ORDER, type AttnLayer } from '@/engine/types';
import { int, plural } from './format';
import { useE2E } from './live';
import { Matrix, type Cell } from './Matrix';
import { maskReasons, type MaskReason } from './math';
import { FocusLine, useViewHover } from './shared';

const REASON: Record<MaskReason, string> = {
  'same-column': 'same column, same table',
  'same-row': 'same row',
  'parent-row': 'key row is a parent of the query row',
  'child-row': 'key row is a child of the query row',
  any: 'any two non-padding tokens',
};

const RULE: Record<AttnLayer, string> = {
  col: 'col[q] = col[k] and table[q] = table[k]',
  feat: 'node[q] = node[k], or node[k] in parents(q)',
  nbr: 'node[q] in parents(k)',
  full: 'both are not padding',
};

export function MaskStage() {
  const live = useE2E();
  const { masks, ctx, numReal, focus, tokenSlots, tokenLabels, setFocus } = live;
  const S = ctx.tokens.length;
  const [hover, setHover] = useViewHover<Cell>([ctx]);
  const [ref, size] = useElementSize<HTMLDivElement>();
  const w = size.width || 640;
  // as many masks per row as fit with cells of at least 0.9 px (4, 2 or 1 per row)
  let perRow = w >= 760 ? 4 : w >= 480 ? 2 : 1;
  while (perRow > 1 && (w / perRow - 28) / numReal < 0.9) perRow = perRow === 4 ? 2 : 1;
  const cell = Math.max(0.4, Math.min(3.2, (w / perRow - 28) / numReal));
  const qTok = ctx.tokens[hover?.r ?? focus];
  const kTok = hover ? ctx.tokens[hover.c] : null;
  const count = (l: AttnLayer) => {
    let n = 0;
    for (let k = 0; k < S; k++) n += masks[l][focus * S + k];
    return n;
  };
  const ids = (t: (typeof ctx.tokens)[number]) => `node ${t.nodeIdx} · parents [${t.f2pNbrIdxs.filter((x) => x >= 0).join(', ')}] · col ${t.colNameIdx} · table ${t.tableNameIdx}`;
  return (
    <Figure
      wide
      title={`Four ${int(numReal)} × ${int(numReal)} masks from the token ids`}
      caption={
        <>
          Rows are queries, columns are keys, in token order; a filled entry means the query may attend to the key. The followed token’s row is
          outlined. Hover any entry to compare the four layers on the same pair; click it to follow the query token.
        </>
      }
    >
      <FocusLine />
      <div ref={ref} className="mt-4 grid grid-cols-1 gap-x-4 gap-y-5" style={{ gridTemplateColumns: `repeat(${perRow}, minmax(0, 1fr))` }}>
        {ATTN_ORDER.map((l) => (
          <div key={l} className="min-w-0">
            <div className="mb-1 text-[16px]">
              <LayerLabel layer={l} />
            </div>
            <div className="mb-1.5 text-[14px] leading-snug text-muted">{RULE[l]}</div>
            <Matrix
              data={masks[l]}
              rows={S}
              cols={S}
              drawRows={numReal}
              drawCols={numReal}
              cellW={cell}
              cellH={cell}
              scale="binary"
              rowSlots={tokenSlots}
              colSlots={tokenSlots}
              focusRow={focus}
              markCell={hover}
              onHover={setHover}
              onPick={(c) => setFocus(c.r)}
              label={`${l} mask`}
            />
            <div className="mt-1.5 text-[14.5px] text-ink-2">
              followed token: <span className="num text-ink">{plural(count(l), 'key')}</span>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 grid grid-cols-1 gap-3 rounded-xl bg-surface-2 px-4 py-3 text-[15.5px] leading-snug text-ink-2 md:grid-cols-2" aria-live="polite">
        <div className="min-w-0">
          <div className="text-[14.5px] text-muted">query</div>
          <div className="font-semibold text-ink">{tokenLabels[qTok.pos]}</div>
          <div className="num text-[14.5px]">{ids(qTok)}</div>
        </div>
        <div className="min-w-0">
          <div className="text-[14.5px] text-muted">key</div>
          {kTok ? (
            <>
              <div className="font-semibold text-ink">{tokenLabels[kTok.pos]}</div>
              <div className="num text-[14.5px]">{ids(kTok)}</div>
            </>
          ) : (
            <div className="text-muted">Hover an entry of any mask.</div>
          )}
        </div>
        {kTok && (
          <ul className="!m-0 grid grid-cols-1 gap-1 !pl-0 md:col-span-2" style={{ listStyle: 'none' }}>
            {ATTN_ORDER.map((l) => {
              const r = maskReasons(ctx.tokens, l, qTok.pos, kTok.pos);
              return (
                <li key={l} className="!m-0 flex flex-wrap items-center gap-x-2 !pl-0">
                  <span className="w-[120px] shrink-0">
                    <LayerLabel layer={l} />
                  </span>
                  {r.length ? <span className="text-ink">allowed: {r.map((x) => REASON[x]).join('; ')}</span> : <span className="text-muted">blocked</span>}
                </li>
              );
            })}
          </ul>
        )}
      </div>
      <p className="!mt-3 !mb-0 text-[15px] leading-snug text-ink-2">
        The code builds the four boolean S × S masks once per forward pass, ANDs each with “neither token is padding” and turns them into block masks
        for FlexAttention <Cite kind="code" ref="rt/model.py:164-206" />. Every head of every block reuses them <Cite kind="spec" ref="§6.1" />.
      </p>
    </Figure>
  );
}
