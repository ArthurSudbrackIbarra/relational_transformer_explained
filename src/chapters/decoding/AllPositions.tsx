/**
 * Step 1, second figure: the decoders' outputs at EVERY position of the live context, and the
 * selection `sem_type_mask = (sem_types == i) & masks` (model.py:241) that lets exactly one of them
 * into the loss. Values are the engine's `ForwardResult.yhat` (model.py:239).
 */
import { useMemo, useState } from 'react';
import { divergingCss, Figure, Segmented, SemBadge, ShapeTag } from '@/components';
import { axis } from '../shapes';
import { useResolvedTheme } from '@/styles/theme';
import { useElementSize } from '@/components/viz/useSize';
import { SemType } from '@/engine/types';
import { fmtS, int, rowName, type Lane } from './live';

type Mode = 'outputs' | 'loss';
const SCALAR: { key: 'number' | 'datetime' | 'boolean'; sem: SemType }[] = [
  { key: 'number', sem: SemType.Number },
  { key: 'datetime', sem: SemType.DateTime },
  { key: 'boolean', sem: SemType.Boolean },
];
const ALL: { key: string; sem: SemType }[] = [
  { key: 'number', sem: SemType.Number },
  { key: 'text', sem: SemType.Text },
  { key: 'datetime', sem: SemType.DateTime },
  { key: 'boolean', sem: SemType.Boolean },
];

const ROW_H = 26;
const LABEL_W_OUT = 116;
const LABEL_W_LOSS = 156;

export function AllPositions({ lane, pos, setPos }: { lane: Lane; pos: number; setPos: (p: number) => void }) {
  const { fr, ctx } = lane;
  const [mode, setMode] = useState<Mode>('outputs');
  const [hover, setHover] = useState<number | null>(null);
  const theme = useResolvedTheme();
  const S = ctx.tokens.length;
  const [boxRef, box] = useElementSize<HTMLDivElement>();
  const tight = !!box.width && box.width < 480; // phone: no badges, narrower label column
  const labelW = tight ? (mode === 'loss' ? 116 : 80) : mode === 'loss' ? LABEL_W_LOSS : LABEL_W_OUT;
  const avail = Math.max(200, (box.width || 900) - labelW);
  // as wide as fits (6-14 px per position); below 6 px the grid scrolls sideways instead of shrinking
  const cw = Math.max(6, Math.min(14, Math.floor(avail / S)));
  const scrolls = S * cw > avail + 1;
  const domains = useMemo(() => {
    const out: Record<string, number> = {};
    if (!fr) return out;
    for (const { key } of SCALAR) {
      let m = 0;
      for (let i = 0; i < S; i++) if (!ctx.tokens[i].isPadding) m = Math.max(m, Math.abs(fr.yhat[key][i]));
      out[key] = m || 1;
    }
    return out;
  }, [fr, ctx, S]);
  const counts = useMemo(() => {
    const c = [0, 0, 0, 0];
    // sem_types == i over ALL positions: padding stores sem_type 0, so it counts as number (its masks entry is 0)
    for (const t of ctx.tokens) c[t.sem]++;
    return c;
  }, [ctx]);
  const target = ctx.targetPos;
  const shown = hover ?? pos;
  const st = ctx.tokens[shown];
  const rows = mode === 'outputs' ? SCALAR : ALL;
  const extraRow = mode === 'loss' ? 1 : 0;
  const H = (rows.length + extraRow) * ROW_H + 18;
  const W = S * cw;

  return (
    <Figure
      wide
      title={
        mode === 'outputs' ? `Every position gets all four outputs (${int(S)} positions)` : 'Which output enters the loss: (sem_types == i) & masks'
      }
      controls={
        <Segmented
          label="View"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'outputs', label: 'Decoder outputs ŷ' },
            { value: 'loss', label: 'What enters the loss' },
          ]}
        />
      }
      caption={
        mode === 'outputs' ? (
          <>
            One column per position of the context, in token order. Each row is one decoder, colored by its output (blue negative, red positive,
            scaled per row). The text head’s 384 outputs per position are left out. The column with the dashed outline is the masked target. Hover or
            tap a column to read it; clicking also selects it in the figure above.
          </>
        ) : (
          <>
            A gray cell means “this token has this type” (sem_types == i). The loss multiplies each decoder’s per-position loss by that row AND the
            masks row, so out of {int(S)} positions × 4 decoders, exactly one value survives: the masked target’s own type.
            {ctx.numReal < S && <> Padding positions store sem_type 0, so they are gray in the number row; their masks entry is 0.</>}
          </>
        )
      }
    >
      <div ref={boxRef} className="min-w-0">
        {!fr ? (
          <div className="rounded-lg border border-dashed border-border-strong px-4 py-8 text-center text-[16px] text-ink-2">
            Running the forward pass…
          </div>
        ) : (
          <>
            <div className="flex min-w-0">
              <div className="shrink-0" style={{ width: labelW }}>
                {rows.map((r) => (
                  <div key={r.key} className="flex items-center gap-1.5 text-[15px] text-ink-2" style={{ height: ROW_H }}>
                    {!tight && <SemBadge sem={r.sem} size="sm" />}
                    <span className="text-ink">{r.key}</span>
                    {mode === 'loss' && <span className="num text-[14px] text-muted">({counts[r.sem]})</span>}
                  </div>
                ))}
                {mode === 'loss' && (
                  <div className="flex items-center text-[15px] font-semibold text-ink" style={{ height: ROW_H }}>
                    masks
                  </div>
                )}
              </div>
              <div className="scroll-quiet min-w-0 flex-1 overflow-x-auto pb-1">
                <svg
                  width={W}
                  height={H}
                  role="img"
                  aria-label={mode === 'outputs' ? 'Decoder outputs at every position' : 'Loss selection mask at every position'}
                  className="block"
                  onPointerLeave={() => setHover(null)}
                >
                  {rows.map((r, ri) =>
                    ctx.tokens.map((t, i) => {
                      const x = i * cw;
                      const y = ri * ROW_H + 3;
                      let fill: string;
                      if (mode === 'outputs') {
                        const v = fr.yhat[r.key as 'number' | 'datetime' | 'boolean'][i];
                        fill = divergingCss(v, domains[r.key], theme);
                      } else {
                        const has = t.sem === r.sem;
                        const sel = has && t.mask;
                        fill = sel ? 'var(--accent)' : has ? 'var(--muted)' : 'var(--surface-2)';
                      }
                      return (
                        <rect
                          key={`${r.key}-${i}`}
                          x={x + 0.5}
                          y={y}
                          width={cw - 1}
                          height={ROW_H - 6}
                          rx={1.5}
                          style={{ fill, transition: 'fill 350ms ease' }}
                        />
                      );
                    }),
                  )}
                  {mode === 'loss' &&
                    ctx.tokens.map((t, i) => (
                      <rect
                        key={`m${i}`}
                        x={i * cw + 0.5}
                        y={rows.length * ROW_H + 3}
                        width={cw - 1}
                        height={ROW_H - 6}
                        rx={1.5}
                        style={{
                          fill: t.mask ? 'var(--accent)' : 'var(--surface-2)',
                        }}
                      />
                    ))}
                  {/* target outline */}
                  {target >= 0 && (
                    <rect
                      x={target * cw - 1}
                      y={1}
                      width={cw + 2}
                      height={(rows.length + extraRow) * ROW_H - 2}
                      rx={3}
                      fill="none"
                      style={{ stroke: 'var(--accent)' }}
                      strokeWidth={2}
                      strokeDasharray="4 3"
                    />
                  )}
                  {/* selected column */}
                  {pos !== target && (
                    <rect
                      x={pos * cw - 1}
                      y={1}
                      width={cw + 2}
                      height={(rows.length + extraRow) * ROW_H - 2}
                      rx={3}
                      fill="none"
                      style={{ stroke: 'var(--ink)' }}
                      strokeWidth={1.5}
                    />
                  )}
                  {hover !== null && (
                    <rect x={hover * cw} y={0} width={cw} height={(rows.length + extraRow) * ROW_H} style={{ fill: 'var(--ink)', opacity: 0.12 }} />
                  )}
                  <text
                    x={Math.min(W - 4, Math.max(4, target * cw + cw / 2))}
                    y={H - 3}
                    textAnchor="middle"
                    style={{
                      fill: 'var(--accent-text)',
                      fontSize: 13,
                      fontWeight: 600,
                    }}
                  >
                    ?
                  </text>
                  {/* hit targets */}
                  {ctx.tokens.map((t, i) => (
                    <rect
                      key={`h${i}`}
                      x={i * cw}
                      y={0}
                      width={cw}
                      height={H}
                      fill="transparent"
                      className="cursor-pointer"
                      onPointerEnter={() => setHover(i)}
                      onClick={() => setPos(i)}
                    >
                      <title>{t.isPadding ? `#${i} padding` : `#${i} ${rowName(t)}.${t.colName}`}</title>
                    </rect>
                  ))}
                </svg>
              </div>
            </div>
            <div className="mt-1" style={{ paddingLeft: labelW }}>
              <ShapeTag
                rows={rows.length + extraRow}
                cols={S}
                shape={
                  mode === 'outputs'
                    ? {
                        rows: 'one per scalar decoder: number, datetime, boolean (the text head’s 384 outputs per position are left out)',
                        cols: axis.tokens({ shown: S, L: ctx.params.L, real: ctx.numReal }),
                        note: 'Every head outputs one number at every position; here the three are stacked as rows.',
                      }
                    : {
                        rows: 'one per data type i (number, text, datetime, boolean), marking where sem_types == i and, in accent, where (sem_types == i) & masks lets that output into the loss; the last row is masks itself',
                        cols: axis.tokens({ shown: S, L: ctx.params.L, real: ctx.numReal }),
                      }
                }
              />
            </div>
            {scrolls && <div className="mt-1 text-[14px] text-ink-2">Scroll the grid sideways to see all {int(S)} positions.</div>}
            <div className="mt-2 min-h-[48px] rounded-lg bg-surface-2 px-3 py-2 text-[15px] leading-snug text-ink-2" aria-live="polite">
              {st && (
                <>
                  <span className="font-semibold text-ink">#{shown}</span>{' '}
                  {st.isPadding ? (
                    <>padding: x̂ = 0, so each head outputs exactly its bias</>
                  ) : (
                    <>
                      <span className="text-ink">
                        {rowName(st)}.{st.colName}
                      </span>{' '}
                      ({['number', 'text', 'datetime', 'boolean'][st.sem]}){st.mask ? ', the masked target' : ''}
                    </>
                  )}
                  <span className="mx-1.5 text-axis">·</span>
                  number <span className="num text-ink">{fmtS(fr.yhat.number[shown], 4)}</span>, datetime{' '}
                  <span className="num text-ink">{fmtS(fr.yhat.datetime[shown], 4)}</span>, boolean{' '}
                  <span className="num text-ink">{fmtS(fr.yhat.boolean[shown], 4)}</span>
                  <span className="mx-1.5 text-axis">·</span>
                  {st.mask ? (
                    <span className="font-semibold text-accent-text">enters the loss</span>
                  ) : (
                    <span>masks = 0: none of these enters the loss</span>
                  )}
                </>
              )}
            </div>
          </>
        )}
      </div>
    </Figure>
  );
}
