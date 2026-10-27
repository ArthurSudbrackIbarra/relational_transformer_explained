/**
 * Matrix products and small labeled panels for the "start to end" page (display only).
 *
 *  - MatMul: the textbook layout of C = A · B (+ b). B sits above C and A to its left, so row i of A
 *    and column j of B meet at C[i, j]; a bias b sits between B and C, aligned with C's columns.
 *    Hovering any entry marks the row and column it is made of in the other matrices; the caller
 *    renders the arithmetic of a hovered C entry, and MatMul itself reads out A, B and b entries.
 *  - Panel: one matrix with its name (weights read out their hovered entry).
 *  Every matrix carries a ShapeTag (its drawn rows × columns and what they are) under its bottom-left corner.
 *  - DotLine: Σ a·b written out with its largest terms, for the readouts.
 *
 * Linear layers are nn.Linear, y = x Wᵀ (+ b) with W stored [out, in] (rt/model.py), so the B of a
 * linear layer is Wᵀ, drawn from the stored W with Matrix's `transpose`.
 */
import { useState, type ReactNode } from 'react';
import { cx, type Shape } from '@/components';
import { fmtNum, int } from './format';
import { useE2E } from './live';
import { Matrix, type Cell, type MatScale } from './Matrix';
import { topTerms, type Term } from './math';

export interface Operand {
  data: ArrayLike<number>;
  rows: number;
  cols: number;
  transpose?: boolean;
  drawRows?: number;
  drawCols?: number;
  scale?: MatScale;
  domain?: number | [number, number];
  blocked?: ArrayLike<number> | null;
  rowSlots?: readonly number[];
  colSlots?: readonly number[];
  bandRows?: readonly [number, number] | null;
  bandCols?: readonly [number, number] | null;
  fadeOutside?: boolean;
  /** Rows are tokens: outline the focus row and pick a token on click. */
  tokenRows?: boolean;
  /** Rows / columns are tokens, for readouts only (no focus outline), e.g. V's rows or Kᵀ's columns. */
  labelRows?: boolean;
  tokenCols?: boolean;
  /** Displayed name (TeX or text), plain name for readouts (default: `label`), accessible name. */
  name: ReactNode;
  text?: string;
  label: string;
  /** What the drawn rows and columns are, for the ShapeTag under the matrix. */
  dims: Shape | null;
}

export type Which = 'A' | 'B' | 'C' | 'bias';
export interface LinkedHover {
  which: Which;
  r: number;
  c: number;
}

/**
 * Cell size for `n` square-celled panels of `cols` columns laid side by side (in a wrapping row):
 * the largest size up to `pref` that fits them in `width`, with fewer panels per row when the cells
 * would get smaller than `min` (one panel per row always fits: its cells may go down to 0.4 px).
 */
export function panelCell(width: number, cols: number, n: number, pref: number, min = 0.9, gutter = 30): number {
  for (let k = n; k >= 1; k--) {
    const c = Math.min(pref, (width - k * gutter) / (k * cols));
    if (c >= min || k === 1) return Math.max(0.4, c);
  }
  return Math.max(0.4, pref);
}

/** Shrink preferred cell sizes uniformly so a layout `need` px wide fits in `width` px. */
export function fitScale(width: number, need: number, minScale = 0.35): number {
  if (!(width > 0) || need <= width) return 1;
  return Math.max(minScale, width / need);
}

export function PanelTitle({ name, className }: { name: ReactNode; className?: string }) {
  return (
    <div className={cx('mb-1.5 flex min-h-[26px] flex-wrap items-baseline gap-x-2 gap-y-0.5', className)}>
      <span className="text-[18px] leading-tight font-semibold text-ink">{name}</span>
    </div>
  );
}

export interface MatMulProps {
  A: Operand;
  B: Operand;
  C: Operand;
  cellS: number;
  cellK: number;
  cellN: number;
  focusRow: number;
  onFocusRow?: (r: number) => void;
  hover: LinkedHover | null;
  setHover: (h: LinkedHover | null) => void;
  /** A bias vector added to every row of C, drawn under B and aligned with C's columns. */
  bias?: { data: ArrayLike<number>; name: ReactNode; text: string; label: string; dims: Shape };
  /** Token names, for reading out entries of token rows / columns. */
  tokenLabels?: readonly string[];
  /** Text placed in the top-left corner (e.g. the operation). */
  corner?: ReactNode;
  className?: string;
}

/** "h[c35·age, 12] = 0.314" for one entry of an operand (display coordinates). */
export function entryText(o: Pick<Operand, 'data' | 'rows' | 'cols' | 'transpose' | 'tokenRows' | 'labelRows' | 'tokenCols' | 'text' | 'label'>, r: number, c: number, tokenLabels?: readonly string[]): string {
  const v = o.transpose ? o.data[c * o.rows + r] : o.data[r * o.cols + c];
  const rn = (o.tokenRows || o.labelRows) && tokenLabels ? tokenLabels[r] : String(r);
  const cn = o.tokenCols && tokenLabels ? tokenLabels[c] : String(c);
  const idx = o.rows === 1 ? cn : o.cols === 1 ? rn : `${rn}, ${cn}`;
  return `${o.text ?? o.label}[${idx}] = ${fmtNum(Number(v), 4)}`;
}

export function MatMul({ A, B, C, cellS, cellK, cellN, focusRow, onFocusRow, hover, setHover, bias, tokenLabels, corner, className }: MatMulProps) {
  const live = useE2E();
  const labels = tokenLabels ?? live.tokenLabels;
  const hv = hover;
  // what each panel marks, from the hovered entry
  const markA = hv?.which === 'C' ? { row: hv.r } : hv?.which === 'B' ? { col: hv.r } : null;
  const markB = hv?.which === 'C' ? { col: hv.c } : hv?.which === 'A' ? { row: hv.c } : hv?.which === 'bias' ? { col: hv.c } : null;
  const markC = hv?.which === 'A' ? { row: hv.r } : hv?.which === 'B' || hv?.which === 'bias' ? { col: hv.c } : null;
  const markBias = hv && hv.which !== 'A' ? { col: hv.c } : null;
  const biasOp: Operand | null = bias ? { data: bias.data, rows: 1, cols: bias.data.length, name: bias.name, text: bias.text, label: bias.label, dims: bias.dims } : null;
  const readout =
    hv?.which === 'A' ? entryText(A, hv.r, hv.c, labels) : hv?.which === 'B' ? entryText(B, hv.r, hv.c, labels) : hv?.which === 'bias' && biasOp ? entryText(biasOp, 0, hv.c) : null;
  const panel = (o: Operand, which: Which, cw: number, ch: number, mark: { row?: number; col?: number } | null) => (
    <Matrix
      data={o.data}
      rows={o.rows}
      cols={o.cols}
      transpose={o.transpose}
      drawRows={o.drawRows}
      drawCols={o.drawCols}
      cellW={cw}
      cellH={ch}
      scale={o.scale}
      domain={o.domain}
      blocked={o.blocked}
      rowSlots={o.rowSlots}
      colSlots={o.colSlots}
      bandRows={o.bandRows}
      bandCols={o.bandCols}
      fadeOutside={o.fadeOutside}
      focusRow={o.tokenRows ? focusRow : null}
      markRow={mark?.row ?? null}
      markCol={mark?.col ?? null}
      markCell={hv && hv.which === which ? { r: hv.r, c: hv.c } : null}
      onHover={(c: Cell | null) => setHover(c ? { which, r: c.r, c: c.c } : null)}
      onPick={o.tokenRows && onFocusRow ? (c) => onFocusRow(c.r) : undefined}
      label={o.label}
      shape={o.dims}
    />
  );
  return (
    <div className={cx('scroll-quiet max-w-full overflow-x-auto', className)}>
      <div className="inline-grid gap-x-4 gap-y-1" style={{ gridTemplateColumns: 'auto auto' }}>
        <div className="flex items-end justify-end pr-1 pb-1 text-right text-[14.5px] leading-snug text-muted">{corner}</div>
        <div>
          <PanelTitle name={B.name} />
          {panel(B, 'B', cellN, cellK, markB)}
          {biasOp && (
            <div className="mt-2">
              <PanelTitle name={biasOp.name} />
              {panel(biasOp, 'bias', cellN, 12, markBias)}
            </div>
          )}
        </div>
        <div className="mt-3">
          <PanelTitle name={A.name} />
          {panel(A, 'A', cellK, cellS, markA)}
        </div>
        <div className="mt-3">
          <PanelTitle name={C.name} />
          {panel(C, 'C', cellN, cellS, markC)}
        </div>
      </div>
      <div className="num mt-1.5 min-h-[1.4em] text-[14.5px] text-ink-2" aria-live="polite">
        {readout}
      </div>
    </div>
  );
}

/** One matrix with its title (for element-wise steps). */
export function Panel({
  o,
  cellW,
  cellH,
  focusRow,
  onFocusRow,
  hover,
  onHover,
  markRow,
  markCol,
  className,
}: {
  o: Operand;
  cellW: number;
  cellH: number;
  focusRow?: number;
  onFocusRow?: (r: number) => void;
  hover?: Cell | null;
  onHover?: (c: Cell | null) => void;
  markRow?: number | null;
  markCol?: number | null;
  className?: string;
}) {
  // weights and other panels that nobody else explains read out their own hovered entry
  const { tokenLabels } = useE2E();
  const own = !onHover;
  const [mine, setMine] = useState<Cell | null>(null);
  return (
    <div className={cx('min-w-0', className)}>
      <PanelTitle name={o.name} />
      <Matrix
        data={o.data}
        rows={o.rows}
        cols={o.cols}
        transpose={o.transpose}
        drawRows={o.drawRows}
        drawCols={o.drawCols}
        cellW={cellW}
        cellH={cellH}
        scale={o.scale}
        domain={o.domain}
        blocked={o.blocked}
        rowSlots={o.rowSlots}
        colSlots={o.colSlots}
        bandRows={o.bandRows}
        bandCols={o.bandCols}
        fadeOutside={o.fadeOutside}
        focusRow={o.tokenRows ? (focusRow ?? null) : null}
        markRow={markRow ?? null}
        markCol={markCol ?? null}
        markCell={hover ?? null}
        onHover={own ? setMine : onHover}
        onPick={o.tokenRows && onFocusRow ? (c) => onFocusRow(c.r) : undefined}
        label={o.label}
        shape={o.dims}
      />
      {own && <div className="num mt-1 min-h-[1.4em] text-[14px] text-ink-2">{mine ? entryText(o, mine.r, mine.c, tokenLabels) : ''}</div>}
    </div>
  );
}

/**
 * "a₁·b₁ + a₂·b₂ + … + n more = s": the `show` terms with the largest |a·b|, then the total.
 * Negative factors are parenthesized, so every term is joined by "+" and the line reads exactly.
 */
export function DotLine({ terms, total, show = 3, digits = 3, extra }: { terms: readonly Term[]; total: number; show?: number; digits?: number; extra?: ReactNode }) {
  const top = topTerms(terms, show);
  const rest = terms.length - top.length;
  return (
    <span className="num break-words">
      {top.map((t, i) => (
        <span key={t.i}>
          {i > 0 && ' + '}
          <span className="whitespace-nowrap">
            {fmtNum(t.a, digits, true)}·{fmtNum(t.b, digits, true)}
          </span>
        </span>
      ))}
      {rest > 0 && <span className="text-muted"> + {int(rest)} more</span>}
      {extra}
      <span> = </span>
      <strong className="font-semibold text-ink">{fmtNum(total, digits)}</strong>
    </span>
  );
}
