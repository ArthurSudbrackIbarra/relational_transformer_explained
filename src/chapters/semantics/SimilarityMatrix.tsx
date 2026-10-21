/**
 * Figure "Similarity across databases": cosine similarity between the column phrases of the live
 * database (rows) and those of the other three databases (columns), computed in the browser from the
 * stored 384-d MiniLM vectors. Nothing is precomputed or assumed.
 */
import { useMemo, useState, type KeyboardEvent } from 'react';
import { ScaleLegend, Segmented, Switch, cx } from '@/components';
import { sequentialCss } from '@/components/viz/scales';
import { textWidth } from '@/components/viz/text';
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
const GROUP_GAP_ROW = 6;
const TABLE_GAP_COL = 3;
const DB_GAP_COL = 12;

export function SimilarityMatrix({ store, catalog, home, focus, onFocus }: SimilarityMatrixProps) {
  const mode = useResolvedTheme();
  const others = DB_NAMES.filter((d) => d !== home);
  const [cols, setCols] = useState<ColsChoice>('all');
  const [keys, setKeys] = useState(false);
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const [wrapRef, size] = useElementSize<HTMLDivElement>();

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
  const vmax = useMemo(() => Math.max(0.5, Math.ceil(Math.max(...all) * 10) / 10), [all]);
  const mean = all.reduce((a, b) => a + b, 0) / Math.max(1, all.length);

  // ── layout ────────────────────────────────────────────────────────────────
  const rowLabelW = useMemo(() => Math.ceil(Math.max(...rowPhrases.map((p) => textWidth(p.col, ROW_PX, { mono: true })), 60)) + 26, [rowPhrases]);
  const colLabelH = useMemo(() => Math.ceil(Math.max(...colPhrases.map((p) => textWidth(p.col, COL_PX, { mono: true })), 40)) + 12, [colPhrases]);
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
  const rowGaps = useMemo(() => {
    const g: number[] = [];
    let acc = 0;
    rowPhrases.forEach((p, i) => {
      if (i > 0 && rowPhrases[i - 1].table !== p.table) acc += GROUP_GAP_ROW;
      g.push(acc);
    });
    return { g, total: acc };
  }, [rowPhrases]);

  const avail = Math.max(320, (size.width || 900) - 4);
  const cell = Math.max(17, Math.min(28, Math.floor((avail - rowLabelW - colGaps.total) / Math.max(1, C))));
  const plotX = rowLabelW;
  const plotY = dbHeaderH + colLabelH + stripH + 4;
  const W = plotX + C * cell + colGaps.total + 2;
  const H = plotY + R * cell + rowGaps.total + 2;
  const cx0 = (j: number) => plotX + j * cell + colGaps.g[j];
  const cy0 = (i: number) => plotY + i * cell + rowGaps.g[i];

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

  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    const cur = hover ?? { r: Math.max(0, focusRow), c: 0 };
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
  };

  const rankInRow = (r: number, c: number) => {
    const v = M[r * C + c];
    let rank = 1;
    for (let j = 0; j < C; j++) if (M[r * C + j] > v) rank++;
    return rank;
  };

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

      <div ref={wrapRef} className="scroll-quiet w-full overflow-x-auto pb-1">
        <svg
          width={W}
          height={H}
          role="img"
          aria-label={`Cosine similarity of ${R} ${DB_SHORT[home]} column phrases (rows) with ${C} phrases of other databases (columns). Use arrow keys to move, Enter to select a row.`}
          tabIndex={0}
          onKeyDown={onKey}
          onPointerLeave={() => setHover(null)}
          className="block rounded-md focus-visible:outline-2 focus-visible:outline-accent"
        >
          {/* database headers */}
          {dbSpans.map((s) => {
            const x0 = cx0(s.j0);
            const x1 = cx0(s.j1) + cell;
            return (
              <g key={s.db}>
                <foreignObject x={x0} y={0} width={Math.max(60, x1 - x0)} height={24}>
                  <div className="flex h-6 items-center text-[14.5px] whitespace-nowrap">
                    <DbTag db={s.db} />
                  </div>
                </foreignObject>
                <line x1={x0} x2={x1} y1={dbHeaderH - 3} y2={dbHeaderH - 3} style={{ stroke: 'var(--axis)' }} strokeWidth={1} />
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
                transform={`translate(${x + 4.5},${dbHeaderH + colLabelH - 4}) rotate(-90)`}
                className="font-mono"
                style={{ fontSize: COL_PX, fill: active ? 'var(--ink)' : 'var(--ink-2)', fontWeight: active ? 650 : 400 }}
              >
                {p.col}
                <title>{p.text}</title>
              </text>
            );
          })}
          {/* table identity strips */}
          {colPhrases.map((p, j) => (
            <rect key={`s-${p.text}`} x={cx0(j)} y={dbHeaderH + colLabelH} width={cell - 1} height={stripH - 1} rx={1} style={{ fill: catVar(p.slot) }} />
          ))}
          {/* row labels */}
          {rowPhrases.map((p, i) => {
            const y = cy0(i);
            const isFocus = i === focusRow;
            const active = shown?.r === i;
            return (
              <g key={p.text} className="cursor-pointer" onClick={() => onFocus(p.text)}>
                <rect x={0} y={y} width={rowLabelW - 6} height={cell - 1} rx={4} style={{ fill: isFocus ? 'var(--accent-soft)' : 'transparent' }} />
                <rect x={4} y={y + (cell - 1) / 2 - 5} width={10} height={10} rx={2} style={{ fill: catVar(p.slot) }} />
                <text
                  x={20}
                  y={y + (cell - 1) / 2}
                  dominantBaseline="central"
                  className="font-mono"
                  style={{ fontSize: ROW_PX, fill: active || isFocus ? 'var(--ink)' : 'var(--ink-2)', fontWeight: isFocus ? 650 : 400 }}
                >
                  {p.col}
                  <title>{p.text}</title>
                </text>
              </g>
            );
          })}
          {/* cells */}
          {rowPhrases.map((rp, i) =>
            colPhrases.map((cp, j) => {
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

      <div className="mt-3 flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
        <div className="min-h-[1.5em] min-w-0 text-[15.5px] text-ink-2" aria-live="polite">
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
        </div>
        <div className="flex flex-wrap items-center gap-4 text-[14.5px] text-ink-2">
          <ScaleLegend scale="sequential" domain={[0, vmax]} mode={mode} />
          <span className="inline-flex items-center gap-1.5">
            <svg width="14" height="14" aria-hidden="true">
              <circle cx="7" cy="7" r="5" fill="none" style={{ stroke: 'var(--ink)' }} strokeWidth={1.6} />
            </svg>
            best match in the row
          </span>
        </div>
      </div>
      <p className="mt-2 text-[14.5px] leading-snug text-muted">
        Cells at or below 0 take the lightest step (the hover read-out gives the signed value). Average over the {R * C} pairs shown:{' '}
        <span className="num text-ink-2">{f3(mean)}</span>. Rows: {DB_SHORT[home]}, the live database; click a row to follow that phrase.
      </p>

      {focusRow >= 0 ? (
        <NearestList store={store} catalog={catalog} phrase={rowPhrases[focusRow]} pool={colPhrases} all={all} />
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
function NearestList({ store, catalog, phrase, pool, all }: { store: EmbeddingStore; catalog: Catalog; phrase: PhraseInfo; pool: PhraseInfo[]; all: number[] }) {
  const ranked = useMemo(() => {
    const q = store.get(phrase.text);
    return pool
      .map((p) => ({ p, cos: cosine(q, store.get(p.text)) }))
      .sort((a, b) => b.cos - a.cos)
      .slice(0, 6);
  }, [store, phrase, pool]);
  const max = Math.max(0.0001, ...ranked.map((r) => r.cos));
  void catalog;
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
              <DbTag db={p.db} showName={false} />
              <span className="truncate" title={p.text}>
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
