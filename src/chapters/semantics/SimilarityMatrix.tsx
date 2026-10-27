/**
 * Figure "Similarity across databases": cosine similarity between the column phrases of the live
 * database (rows) and those of the other three databases (columns), computed in the browser from the
 * stored 384-d MiniLM vectors. Nothing is precomputed or assumed.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { ScaleLegend, Segmented, ShapeTag, Switch, cx } from '@/components';
import { sequentialCss } from '@/components/viz/scales';
import { fitText, textWidth } from '@/components/viz/text';
import { useElementSize } from '@/components/viz/useSize';
import { DB_NAMES, type DbName } from '@/engine/db';
import type { EmbeddingStore } from '@/engine/embeddings';
import { catVar } from '@/styles/palette';
import { useResolvedTheme } from '@/styles/theme';
import { DbTag, Phrase } from './bits';
import { DB_SHORT, cosine, f3, percentileOf, type Catalog, type PhraseInfo } from './data';

export interface SimilarityMatrixProps {
  store: EmbeddingStore;
  catalog: Catalog;
  home: DbName;
  focus: string;
  onFocus: (phrase: string) => void;
}

type ColsChoice = 'all' | DbName;

const ROW_PX = 14.5; // row label size
const COL_PX = 13.5; // rotated column label size
const TABLE_PX = 13.5; // table-name labels (row groups and column groups)
const ROW_GROUP_H = 22; // a table-name line above each group of rows
const TABLE_ROW_H = 19; // one line of column-group table names
const TABLE_GAP_COL = 3;
const DB_GAP_COL = 12;

interface Group {
  db: DbName;
  table: string;
  slot: number;
  j0: number;
  j1: number;
}

export function SimilarityMatrix({ store, catalog, home, focus, onFocus }: SimilarityMatrixProps) {
  const mode = useResolvedTheme();
  const others = useMemo(() => DB_NAMES.filter((d) => d !== home), [home]);
  const [cols, setCols] = useState<ColsChoice>('all');
  const [keys, setKeys] = useState(false);
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [scroll, setScroll] = useState({ left: 0, max: 0 });

  const rowPhrases = useMemo(() => catalog.byDb[home].filter((p) => keys || p.onToken), [catalog, home, keys]);
  const colPhrases = useMemo(
    () => (cols === 'all' ? others : [cols]).flatMap((d) => catalog.byDb[d].filter((p) => keys || p.onToken)),
    [catalog, cols, others, keys],
  );

  const M = useMemo(() => {
    const out = new Float32Array(rowPhrases.length * colPhrases.length);
    rowPhrases.forEach((r, i) => {
      const a = store.get(r.text);
      colPhrases.forEach((c, j) => (out[i * colPhrases.length + j] = cosine(a, store.get(c.text))));
    });
    return out;
  }, [store, rowPhrases, colPhrases]);

  const R = rowPhrases.length;
  const C = colPhrases.length;
  const all = useMemo(() => Array.from(M), [M]);
  const maxCos = useMemo(() => Math.max(...all), [all]);
  // legend maximum: the largest cosine shown, rounded up to the next 0.1 (at least 0.5)
  const vmax = Math.max(0.5, Math.ceil(maxCos * 10) / 10);
  const mean = all.reduce((a, b) => a + b, 0) / Math.max(1, all.length);

  // ── groups (tables) ───────────────────────────────────────────────────────
  const colGroups = useMemo(() => {
    const g: Group[] = [];
    colPhrases.forEach((p, j) => {
      const last = g[g.length - 1];
      if (last && last.db === p.db && last.table === p.table) last.j1 = j;
      else g.push({ db: p.db, table: p.table, slot: p.slot, j0: j, j1: j });
    });
    return g;
  }, [colPhrases]);
  const rowGroupStart = useMemo(() => new Set(rowPhrases.map((p, i) => (i === 0 || rowPhrases[i - 1].table !== p.table ? i : -1)).filter((i) => i >= 0)), [rowPhrases]);

  // ── layout ────────────────────────────────────────────────────────────────
  const rowLabelW = useMemo(
    () =>
      Math.ceil(
        Math.max(
          ...rowPhrases.map((p) => textWidth(p.col, ROW_PX)),
          ...rowPhrases.map((p) => textWidth(p.table, TABLE_PX) + 16),
          60,
        ),
      ) + 18,
    [rowPhrases],
  );
  const colLabelH = useMemo(() => Math.ceil(Math.max(...colPhrases.map((p) => textWidth(p.col, COL_PX)), 40)) + 12, [colPhrases]);
  const dbHeaderH = 30;
  const stripH = 6;

  // column x offsets with gaps between tables and databases
  const colGaps = useMemo(() => {
    const g: number[] = [];
    let acc = 0;
    colPhrases.forEach((p, j) => {
      if (j > 0) {
        const prev = colPhrases[j - 1];
        if (prev.db !== p.db) acc += DB_GAP_COL;
        else if (prev.table !== p.table) acc += TABLE_GAP_COL;
      }
      g.push(acc);
    });
    return { g, total: acc };
  }, [colPhrases]);
  const rowOff = useMemo(() => {
    const g: number[] = [];
    let acc = 0;
    rowPhrases.forEach((_, i) => {
      if (rowGroupStart.has(i)) acc += ROW_GROUP_H;
      g.push(acc);
    });
    return g;
  }, [rowPhrases, rowGroupStart]);

  const avail = Math.max(200, (size.width || 900) - rowLabelW - 2);
  const cell = Math.max(17, Math.min(28, Math.floor((avail - colGaps.total - 2) / Math.max(1, C))));
  const cx0 = (j: number) => j * cell + colGaps.g[j];

  // table names over the column groups, staggered over n lines (the fewest, up to 4, with which every
  // name fits): the k-th label may run until the (k+n)-th group starts; a longer one is shortened
  // (full name on hover)
  const tableLabels = useMemo(() => {
    const x0s = colGroups.map((g) => g.j0 * cell + colGaps.g[g.j0]);
    const x1s = colGroups.map((g) => g.j1 * cell + colGaps.g[g.j1] + cell - 1);
    // the last labels may run past the last column, into free width (the SVG widens to hold them)
    const room = Math.max(C * cell + colGaps.total + 2, avail);
    const limit = (k: number, n: number) => (k + n < colGroups.length ? x0s[k + n] - 6 : room - 2);
    const need = (k: number) => textWidth(colGroups[k].table, TABLE_PX) + 14;
    let n = 1;
    while (n < 4 && colGroups.some((_, k) => x0s[k] + need(k) > limit(k, n))) n++;
    const out = colGroups.map((g, k) => ({
      g,
      line: k % n,
      text: fitText(g.table, limit(k, n) - x0s[k] - 14, TABLE_PX),
      x0: x0s[k],
      x1: x1s[k],
    }));
    const extent = Math.max(0, ...out.map((o) => o.x0 + textWidth(o.text, TABLE_PX) + 16));
    return { out, n, extent };
  }, [colGroups, cell, colGaps, C, avail]);
  // the tail labels may use the free width to the right of the cells, but never cause a scroll
  const cellsW = C * cell + colGaps.total + 2;
  const tableBandH = tableLabels.n * TABLE_ROW_H + 6;

  const plotY = dbHeaderH + tableBandH + colLabelH + stripH + 4;
  const Wm = Math.max(cellsW, Math.min(tableLabels.extent, Math.max(cellsW, avail)));
  const H = plotY + R * cell + (rowOff[R - 1] ?? 0) + 2;
  const cy0 = (i: number) => plotY + i * cell + rowOff[i];

  // per-row best match
  const best = useMemo(() => {
    const b: number[] = [];
    for (let i = 0; i < R; i++) {
      let k = 0;
      for (let j = 1; j < C; j++) if (M[i * C + j] > M[i * C + k]) k = j;
      b.push(k);
    }
    return b;
  }, [M, R, C]);

  const focusRow = rowPhrases.findIndex((p) => p.text === focus);
  const shown = hover ?? (focusRow >= 0 ? { r: focusRow, c: best[focusRow] } : null);

  // database header spans
  const dbSpans = useMemo(() => {
    const spans: { db: DbName; j0: number; j1: number }[] = [];
    colPhrases.forEach((p, j) => {
      const last = spans[spans.length - 1];
      if (last && last.db === p.db) last.j1 = j;
      else spans.push({ db: p.db, j0: j, j1: j });
    });
    return spans;
  }, [colPhrases]);

  // ── horizontal scrolling (narrow screens): keep a column in view, show fades at the cut edges ──
  const readScroll = useCallback(() => {
    const el = scrollRef.current;
    if (el) setScroll({ left: el.scrollLeft, max: Math.max(0, el.scrollWidth - el.clientWidth) });
  }, []);
  const ensureVisible = useCallback(
    (c: number, center = false) => {
      const el = scrollRef.current;
      if (!el || el.scrollWidth <= el.clientWidth + 1 || c < 0 || c >= C) return;
      const x = c * cell + colGaps.g[c];
      if (center) el.scrollLeft = Math.max(0, x - el.clientWidth / 2 + cell / 2);
      else if (x < el.scrollLeft) el.scrollLeft = Math.max(0, x - 8);
      else if (x + cell > el.scrollLeft + el.clientWidth) el.scrollLeft = x + cell - el.clientWidth + 8;
    },
    [C, cell, colGaps],
  );
  // bring the followed row's best match into view whenever the row, the columns or the size change
  const bestOfFocus = focusRow >= 0 ? best[focusRow] : -1;
  useEffect(() => {
    ensureVisible(bestOfFocus, true);
    readScroll();
  }, [bestOfFocus, cols, keys, size.width, ensureVisible, readScroll]);

  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    const cur = hover ?? { r: Math.max(0, focusRow), c: focusRow >= 0 ? best[focusRow] : 0 };
    let { r, c } = cur;
    if (e.key === 'ArrowRight') c = Math.min(C - 1, c + 1);
    else if (e.key === 'ArrowLeft') c = Math.max(0, c - 1);
    else if (e.key === 'ArrowDown') r = Math.min(R - 1, r + 1);
    else if (e.key === 'ArrowUp') r = Math.max(0, r - 1);
    else if (e.key === 'Enter' || e.key === ' ') {
      onFocus(rowPhrases[r].text);
      e.preventDefault();
      return;
    } else if (e.key === 'Escape') {
      setHover(null);
      return;
    } else return;
    e.preventDefault();
    setHover({ r, c });
    ensureVisible(c);
  };

  const rankInRow = (r: number, c: number) => {
    const v = M[r * C + c];
    let rank = 1;
    for (let j = 0; j < C; j++) if (M[r * C + j] > v) rank++;
    return rank;
  };

  const keyStyle = (p: PhraseInfo) => (p.onToken ? {} : { fontStyle: 'italic' as const, textDecoration: 'line-through', fill: 'var(--muted)' });

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-3">
        <Segmented<ColsChoice>
          label="Columns of the matrix"
          value={cols}
          onChange={(v) => {
            setCols(v);
            setHover(null);
          }}
          options={[{ value: 'all', label: 'All three' }, ...others.map((d) => ({ value: d, label: DB_SHORT[d] }))]}
        />
        <Switch checked={keys} onChange={(v) => (setKeys(v), setHover(null))} label="Include key columns" />
      </div>

      <div ref={wrapRef} className="flex w-full items-start">
        {/* row labels: outside the scrolling area, so they stay visible when the cells scroll */}
        <svg width={rowLabelW} height={H} className="block shrink-0" aria-hidden="true">
          {rowPhrases.map((p, i) => {
            const y = cy0(i);
            const isFocus = i === focusRow;
            const active = shown?.r === i;
            return (
              <g key={p.text}>
                {rowGroupStart.has(i) && (
                  <g>
                    <rect x={2} y={y - ROW_GROUP_H + 6} width={10} height={10} rx={2} style={{ fill: catVar(p.slot) }} />
                    <text x={17} y={y - ROW_GROUP_H + 11} dominantBaseline="central" style={{ fontSize: TABLE_PX, fill: 'var(--ink-2)', fontWeight: 600 }}>
                      {p.table}
                    </text>
                  </g>
                )}
                <g className="cursor-pointer" onClick={() => onFocus(p.text)}>
                  <rect x={0} y={y} width={rowLabelW - 6} height={cell - 1} rx={4} style={{ fill: isFocus ? 'var(--accent-soft)' : 'transparent' }} />
                  <text
                    x={17}
                    y={y + (cell - 1) / 2}
                    dominantBaseline="central"
                    style={{ fontSize: ROW_PX, fill: active || isFocus ? 'var(--ink)' : 'var(--ink-2)', fontWeight: isFocus ? 650 : 400, ...keyStyle(p) }}
                  >
                    {p.col}
                    <title>{p.onToken ? p.text : `${p.text} (key column: never on a token)`}</title>
                  </text>
                </g>
              </g>
            );
          })}
        </svg>

        <div className="relative min-w-0 flex-1">
          <div ref={scrollRef} onScroll={readScroll} className="scroll-quiet w-full overflow-x-auto pb-1">
            <svg
              width={Wm}
              height={H}
              role="img"
              aria-label={`Cosine similarity of ${R} ${DB_SHORT[home]} column phrases (rows) with ${C} phrases of other databases (columns). Use arrow keys to move, Enter to select a row.`}
              tabIndex={0}
              onKeyDown={onKey}
              onPointerLeave={() => setHover(null)}
              className="block rounded-md focus-visible:outline-2 focus-visible:outline-accent"
            >
              {/* database headers: shape + name in ink (color means table in this figure) */}
              {dbSpans.map((s) => {
                const x0 = cx0(s.j0);
                const x1 = cx0(s.j1) + cell;
                return (
                  <g key={s.db}>
                    <foreignObject x={x0} y={0} width={Math.max(60, x1 - x0)} height={24}>
                      <div className="flex h-6 items-center text-[14.5px] font-semibold whitespace-nowrap">
                        <DbTag db={s.db} neutral />
                      </div>
                    </foreignObject>
                    <line x1={x0} x2={x1} y1={dbHeaderH - 3} y2={dbHeaderH - 3} style={{ stroke: 'var(--axis)' }} strokeWidth={1} />
                  </g>
                );
              })}
              {/* table names over the column groups */}
              {tableLabels.out.map(({ g, line, text, x0, x1 }) => {
                const y = dbHeaderH + 3 + line * TABLE_ROW_H;
                const yb = dbHeaderH + tableBandH - 2;
                return (
                  <g key={`${g.db}-${g.table}`}>
                    <line x1={x0 + 0.5} x2={x0 + 0.5} y1={y + 13} y2={yb} style={{ stroke: 'var(--grid)' }} strokeWidth={1} />
                    <line x1={x0} x2={x1} y1={yb} y2={yb} style={{ stroke: catVar(g.slot) }} strokeWidth={2} />
                    <rect x={x0} y={y + 3} width={9} height={9} rx={2} style={{ fill: catVar(g.slot) }} />
                    <text x={x0 + 13} y={y + 8} dominantBaseline="central" style={{ fontSize: TABLE_PX, fill: 'var(--ink-2)' }}>
                      {text}
                      <title>{`${g.table} (${DB_SHORT[g.db]})`}</title>
                    </text>
                  </g>
                );
              })}
              {/* column labels, rotated */}
              {colPhrases.map((p, j) => {
                const x = cx0(j) + cell / 2;
                const active = shown?.c === j;
                return (
                  <text
                    key={p.text}
                    transform={`translate(${x + 4.5},${dbHeaderH + tableBandH + colLabelH - 4}) rotate(-90)`}
                    style={{ fontSize: COL_PX, fill: active ? 'var(--ink)' : 'var(--ink-2)', fontWeight: active ? 650 : 400, ...keyStyle(p) }}
                  >
                    {p.col}
                    <title>{p.onToken ? p.text : `${p.text} (key column: never on a token)`}</title>
                  </text>
                );
              })}
              {/* table identity strips */}
              {colPhrases.map((p, j) => (
                <rect key={`s-${p.text}`} x={cx0(j)} y={dbHeaderH + tableBandH + colLabelH} width={cell - 1} height={stripH - 1} rx={1} style={{ fill: catVar(p.slot) }} />
              ))}
              {/* cells */}
              {rowPhrases.map((rp, i) =>
                colPhrases.map((_cp, j) => {
                  const v = M[i * C + j];
                  const t = Math.max(0, v) / vmax;
                  return (
                    <rect
                      key={`${i}-${j}`}
                      x={cx0(j)}
                      y={cy0(i)}
                      width={cell - 1}
                      height={cell - 1}
                      rx={2}
                      style={{ fill: sequentialCss(t, mode) }}
                      onPointerEnter={() => setHover({ r: i, c: j })}
                      onClick={() => onFocus(rp.text)}
                    />
                  );
                }),
              )}
              {/* best match per row */}
              {best.map((j, i) => (
                <circle
                  key={`b-${i}`}
                  cx={cx0(j) + (cell - 1) / 2}
                  cy={cy0(i) + (cell - 1) / 2}
                  r={Math.max(3, cell / 2 - 5)}
                  fill="none"
                  style={{ stroke: M[i * C + j] / vmax > 0.55 ? 'var(--page)' : 'var(--ink)' }}
                  strokeWidth={1.6}
                  pointerEvents="none"
                />
              ))}
              {/* hover crosshair */}
              {shown && (
                <rect
                  x={cx0(shown.c) - 1.5}
                  y={cy0(shown.r) - 1.5}
                  width={cell + 2}
                  height={cell + 2}
                  rx={3}
                  fill="none"
                  style={{ stroke: 'var(--ink)' }}
                  strokeWidth={2}
                  pointerEvents="none"
                />
              )}
            </svg>
          </div>
          {/* fades where the matrix is cut off */}
          {scroll.left > 2 && <div aria-hidden="true" className="pointer-events-none absolute inset-y-0 left-0 w-6 bg-gradient-to-r from-surface to-transparent" />}
          {scroll.max - scroll.left > 2 && (
            <div aria-hidden="true" className="pointer-events-none absolute inset-y-0 right-0 w-10 bg-gradient-to-l from-surface to-transparent" />
          )}
        </div>
      </div>
      {scroll.max > 2 && (
        <p className="mt-1 !mb-0 text-[14.5px] text-muted">
          The matrix is wider than the screen: scroll it sideways (row labels stay in place).
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
        <div className="flex min-h-[1.5em] min-w-0 flex-wrap items-baseline gap-x-2.5 text-[15.5px] text-ink-2" aria-live="polite">
          <ShapeTag
            rows={R}
            cols={C}
            shape={{
              rows: `one per column phrase “<col> of <table>” of ${DB_SHORT[home]}, the live database, grouped by table${keys ? ', key columns included' : '; key columns, which never become tokens, are left out'}`,
              cols: `one per column phrase of ${cols === 'all' ? `the other three databases (${others.map((o) => DB_SHORT[o]).join(', ')})` : DB_SHORT[cols]}, grouped by database and table`,
              note: 'Each entry is the cosine between the two phrases’ frozen MiniLM vectors, computed in your browser.',
            }}
          />
          <span className="min-w-0">
          {shown ? (
            <>
              <Phrase text={rowPhrases[shown.r].text} /> <span className="text-muted">·</span> <Phrase text={colPhrases[shown.c].text} />{' '}
              <span className="text-muted">→</span> <strong className="num text-ink">cos {f3(M[shown.r * C + shown.c])}</strong>{' '}
              <span className="text-muted">
                (rank {rankInRow(shown.r, shown.c)} of {C} in this row)
              </span>
            </>
          ) : (
            'Hover or focus a cell to read its cosine.'
          )}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-4 text-[14.5px] text-ink-2">
          <ScaleLegend scale="sequential" domain={[0, vmax]} mode={mode} />
          <span className="inline-flex items-center gap-1.5">
            <svg width="14" height="14" aria-hidden="true">
              <circle cx="7" cy="7" r="5" fill="none" style={{ stroke: 'var(--ink)' }} strokeWidth={1.6} />
            </svg>
            best match in the row
          </span>
          {keys && (
            <span className="inline-flex items-center gap-1.5">
              <span className="text-muted italic line-through">customer_id</span> key column: never on a token
            </span>
          )}
        </div>
      </div>
      <p className="mt-2 text-[14.5px] leading-snug text-muted">
        Largest cosine shown: <span className="num text-ink-2">{f3(maxCos)}</span> (the legend rounds it up to {vmax.toFixed(1)}). Cells at or
        below 0 take the lightest step (the hover read-out gives the signed value). Average over the {R * C} pairs shown:{' '}
        <span className="num text-ink-2">{f3(mean)}</span>. Rows: {DB_SHORT[home]}, the live database; click a row to follow that phrase.
      </p>

      {focusRow >= 0 ? (
        <NearestList store={store} phrase={rowPhrases[focusRow]} pool={colPhrases} all={all} />
      ) : (
        <p className="mt-4 rounded-lg bg-surface-2 px-4 py-3 text-[15px] text-ink-2">
          The phrase you are following, <Phrase text={focus} plain={!catalog.byText.has(focus)} />, is not a row of this matrix. Click a row to see
          its closest phrases.
        </p>
      )}
    </div>
  );
}

/** Ranked list of the phrases in `pool` closest to `phrase`. */
function NearestList({ store, phrase, pool, all }: { store: EmbeddingStore; phrase: PhraseInfo; pool: PhraseInfo[]; all: number[] }) {
  const ranked = useMemo(() => {
    const q = store.get(phrase.text);
    return pool
      .map((p) => ({ p, cos: cosine(q, store.get(p.text)) }))
      .sort((a, b) => b.cos - a.cos)
      .slice(0, 6);
  }, [store, phrase, pool]);
  const max = Math.max(0.0001, ...ranked.map((r) => r.cos));
  return (
    <div className="mt-5 rounded-lg border border-border bg-surface px-4 py-3">
      <div className="mb-2 text-[15.5px] font-semibold text-ink">
        Closest to <Phrase text={phrase.text} /> among the columns shown
      </div>
      <ol className="!m-0 grid !list-none gap-1.5 !p-0">
        {ranked.map(({ p, cos }, i) => (
          <li key={p.text} className="!m-0 grid grid-cols-[22px_minmax(0,1fr)_56px] items-center gap-2 !p-0 text-[15px] sm:grid-cols-[22px_minmax(0,1fr)_minmax(80px,30%)_56px]">
            <span className="num text-muted">{i + 1}</span>
            <span className="flex min-w-0 items-center gap-2">
              <DbTag db={p.db} showName={false} neutral />
              <span className="truncate" title={`${p.text} (${DB_SHORT[p.db]})`}>
                <Phrase text={p.text} />
              </span>
            </span>
            <span className="relative hidden h-2.5 rounded-full bg-surface-3 sm:block" aria-hidden="true">
              <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${(100 * Math.max(0, cos)) / max}%`, background: 'var(--seq-450)' }} />
            </span>
            <span className={cx('num text-right text-ink')}>{f3(cos)}</span>
          </li>
        ))}
      </ol>
      <p className="mt-2 !mb-0 text-[14.5px] text-ink-2">
        The best of these sits above {percentileOf(ranked[0]?.cos ?? 0, all).toFixed(0)}% of all pairs in the matrix.
      </p>
    </div>
  );
}
