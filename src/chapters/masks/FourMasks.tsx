/**
 * Section 2 figure: the four masks of the live context as S×S matrices, side by side, with a synced
 * hover (one (query, key) pair compared across all four masks) and click-to-pin the query.
 */
import { Check, X } from 'lucide-react';
import { useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Figure, LayerLabel, ScaleLegend, ScaleNote, Segmented, Switch, TableSwatch } from '@/components';
import { axis } from '../shapes';
import { useResolvedTheme } from '@/styles/theme';
import { FitMask } from './FitMask';
import { countRow, int, LAYER_ORDER, maskStats, pct, rowName } from './maskUtils';
import type { LiveMasks } from './useLiveMasks';

/**
 * Tokens in the labeled view: 20 is the most that fits two matrices side by side at 1440 px with
 * 15 px cells, the size at which the Heatmap draws its labels at 13 px.
 */
export const FOURMASKS_ZOOM = 20;
const ZOOM = FOURMASKS_ZOOM;

export function FourMasks({ live }: { live: LiveMasks }) {
  const { ctx, masks, S, labels, tokenSlots, pinned, setPinned, grouping } = live;
  const mode = useResolvedTheme();
  const [view, setView] = useState<'all' | 'zoom'>('all');
  const [hidePad, setHidePad] = useState(true);
  const [hover, setHover] = useState<{ q: number; k: number } | null>(null);

  const hasPad = ctx.numReal < S;
  const zoomN = Math.min(ZOOM, ctx.numReal);
  const crop = view === 'zoom' ? zoomN : hidePad ? ctx.numReal : S;
  const stats = useMemo(() => maskStats(masks, ctx.tokens), [masks, ctx]);

  const q = hover?.q ?? pinned;
  const k = hover?.k ?? null;
  const qTok = ctx.tokens[q];
  const kTok = k !== null ? ctx.tokens[k] : null;
  // floating readout next to the cursor (positioned imperatively, so pointer moves inside a cell do not re-render)
  const gridRef = useRef<HTMLDivElement | null>(null);
  const tipRef = useRef<HTMLDivElement | null>(null);
  const lastPt = useRef<{ x: number; y: number } | null>(null);
  const positionTip = () => {
    const g = gridRef.current;
    const t = tipRef.current;
    const p = lastPt.current;
    if (!g || !t || !p) return;
    const r = g.getBoundingClientRect();
    const x = p.x - r.left;
    const y = p.y - r.top;
    const w = t.offsetWidth || 300;
    const left = x + 16 + w > r.width ? Math.max(0, x - 16 - w) : x + 16;
    t.style.transform = `translate(${Math.round(left)}px, ${Math.round(y + 18)}px)`;
  };
  const placeTip = (e: React.PointerEvent) => {
    lastPt.current = { x: e.clientX, y: e.clientY };
    positionTip();
  };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useLayoutEffect(positionTip, [hover]);

  return (
    <Figure
      wide
      title="The four attention masks of the live context"
      controls={
        <>
          <Segmented
            label="Matrix view"
            value={view}
            onChange={setView}
            options={[
              { value: 'all', label: `All ${hidePad && hasPad ? ctx.numReal : S} tokens` },
              { value: 'zoom', label: `First ${zoomN}, labeled` },
            ]}
          />
          <ScaleNote inline className="mb-1.5" />
          {hasPad ? (
            <Switch checked={hidePad} onChange={setHidePad} label="Hide padding" disabled={view === 'zoom'} />
          ) : (
            <span className="mb-1.5 max-w-[34rem] text-[15px] leading-snug text-muted">
              No padding to hide: the sampler filled all {S} positions. The toy context in Step 6 has 5 padding tokens.
            </span>
          )}
        </>
      }
      caption={
        <>
          Rows are queries, columns are keys; a filled square means the query may attend to that key. The colored strips along both axes give
          each token’s table (key below), in emission order, so the blocks you see are rows and tables. Hover any square to compare that one pair
          in all four masks; click to pin the query (it carries over to the next chapters). The pinned query’s row is framed in violet.
          {view === 'zoom' && ' The labeled view shows the top-left corner: the first tokens the sampler emitted.'}
        </>
      }
    >
      <div
        ref={gridRef}
        className="relative grid gap-x-8 gap-y-7 md:grid-cols-2"
        onPointerMove={placeTip}
        onPointerLeave={() => setHover(null)}
      >
        {hover && qTok && kTok && (
          <div
            ref={tipRef}
            role="tooltip"
            className="pointer-events-none absolute top-0 left-0 z-20 w-max max-w-[320px] rounded-lg border border-border bg-surface px-3 py-2 text-[15px] leading-snug text-ink shadow-[var(--shadow-pop)]"
          >
            <div className="num">
              <span className="text-muted">q </span>#{hover.q} {labels[hover.q]}
            </div>
            <div className="num">
              <span className="text-muted">k </span>#{hover.k} {labels[hover.k]}
            </div>
            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5">
              {LAYER_ORDER.map((l) => {
                const ok = masks[l][hover.q * S + hover.k] === 1;
                return (
                  <span key={l} className={ok ? 'inline-flex items-center gap-0.5 font-semibold text-ink' : 'inline-flex items-center gap-0.5 text-muted'}>
                    {ok ? <Check aria-hidden="true" size={14} style={{ color: 'var(--good)' }} /> : <X aria-hidden="true" size={14} />}
                    {LAYER_SHORT[l]}
                  </span>
                );
              })}
            </div>
          </div>
        )}
        {LAYER_ORDER.map((l, i) => {
          const s = stats[i];
          const allowedPair = k !== null ? masks[l][q * S + k] === 1 : null;
          return (
            <FitMask
              key={l}
              n={crop}
              labeled={view === 'zoom'}
              maxCell={view === 'zoom' ? 16 : 7}
              minLabelCell={15}
              ticks={crop > 64 ? 32 : 16}
              bandRow={q < crop ? q : null}
              header={
                <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
                  <span className="text-[16px]">
                    <LayerLabel layer={l} />
                    <span className="text-muted"> mask</span>
                  </span>
                  <span className="num text-[15px] text-ink-2">
                    {int(s.pairs)} pairs · {pct(s.density)}
                    {allowedPair !== null && (
                      <span className={allowedPair ? 'ml-2 font-semibold text-ink' : 'ml-2 text-muted'}>{allowedPair ? '✓ allowed' : '✗ blocked'}</span>
                    )}
                  </span>
                </div>
              }
              matrix={{
                mask: masks[l],
                S,
                crop,
                labels,
                slots: tokenSlots,
                selected: k !== null && q < crop && k < crop ? { row: q, col: k } : null,
                onHover: (c) => setHover(c ? { q: c.q, k: c.k } : null),
                onCellClick: (c) => setPinned(c.q),
                title: `${l} attention mask`,
                showLegend: false,
                showReadout: false,
                shape: {
                  rows: axis.queries({ shown: crop, L: ctx.params.L, real: ctx.numReal }),
                  cols: axis.keys({ shown: crop, L: ctx.params.L, real: ctx.numReal }),
                },
              }}
            />
          );
        })}
      </div>

      <div className="mt-5 flex flex-col items-start justify-between gap-x-6 gap-y-3 border-t border-grid pt-3 text-[15.5px] lg:flex-row" aria-live="polite">
        <div className="min-w-0 flex-1">
          <div className="text-ink-2">
            <span className="text-muted">query </span>
            <span className="num text-ink">
              #{q} {labels[q]}
            </span>
            {qTok && !qTok.isPadding && <span className="text-muted"> ({rowName(qTok)})</span>}
            {q === ctx.targetPos && <span className="ml-1 font-semibold text-accent-text">masked target</span>}
            {kTok && (
              <>
                <span className="text-muted"> · key </span>
                <span className="num text-ink">
                  #{k} {labels[k!]}
                </span>
              </>
            )}
          </div>
          <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1">
            {LAYER_ORDER.map((l) => {
              const ok = k !== null ? masks[l][q * S + k] === 1 : null;
              return (
                <span key={l} className="inline-flex items-center gap-1.5">
                  <LayerLabel layer={l} size={13} className="text-[15px] font-normal" />
                  {ok === null ? (
                    <span className="num text-ink">{int(countRow(masks[l], S, q))} keys</span>
                  ) : ok ? (
                    <span className="inline-flex items-center gap-0.5 font-semibold text-ink">
                      <Check aria-hidden="true" size={14} style={{ color: 'var(--good)' }} /> allowed
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-0.5 text-muted">
                      <X aria-hidden="true" size={14} /> blocked
                    </span>
                  )}
                </span>
              );
            })}
          </div>
        </div>
        <div className="flex flex-col items-start gap-1.5 lg:items-end">
          <ScaleLegend scale="binary" domain={[0, 1]} mode={mode} binaryLabels={['may attend', 'blocked']} />
          <TableKey tables={grouping.tables} padding={hasPad && crop > ctx.numReal} />
        </div>
      </div>
    </Figure>
  );
}

const LAYER_SHORT = { col: 'column', feat: 'feature', nbr: 'neighbor', full: 'full' } as const;

/** Color key for the identity strips: one swatch per table in the context (task splits share one). */
export function TableKey({ tables, padding = false }: { tables: readonly { name: string; slot: number }[]; padding?: boolean }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3.5 gap-y-1 text-[15px] text-ink-2" aria-label="Table colors of the axis strips">
      <span className="text-muted">strips:</span>
      {tables.map((t) => (
        <TableSwatch key={t.name} slot={t.slot} label={t.name} size={10} className="text-[15px]" />
      ))}
      {padding && <TableSwatch slot={0} label="padding" size={10} mono={false} />}
    </div>
  );
}
