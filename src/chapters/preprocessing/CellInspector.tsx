/**
 * Section 5 figure: "pick a cell, see its encoding". Click any cell of any
 * table file; the panel walks it through pre.rs (role, sem type,
 * normalization, the record slots it fills, the bf16 value a batch would
 * carry) and names the embedding term that reads it (model.py:208-224). Every
 * number is read from the engine (pdb nodes, colStats, datetimeStats), the
 * text embeddings from the precomputed MiniLM table (public/data/embeddings).
 */
import { motion } from 'motion/react';
import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { Cite, NumberField, ScaleNote, Select, SemBadge, StatusBadge, TableView, TeX, VectorStrip, cx, formatRaw } from '@/components';
import { SEM_TYPE_NAMES, SemType, type Cell } from '@/engine/types';
import { catVar } from '@/styles/palette';
import { useAssets, useContextWindow } from '@/state/hooks';
import { DAY_MS, fileLabel, isoDay, isoSeconds, rowFates, toBf16, type Fate, type TableFile } from './data';
import { PickChip, Sub, signed, texNum, texText, usePhone, type ChapterData } from './shared';

export interface CellPick {
  label: string;
  key: string;
  row: number;
  col: string;
}

const TYPE_NAME = ['Number', 'Text', 'DateTime', 'Boolean'] as const;

function Step({ n, title, children, last, delay }: { n: number; title: ReactNode; children: ReactNode; last?: boolean; delay: number }) {
  return (
    <motion.li
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, delay }}
      className="!m-0 grid grid-cols-[26px_minmax(0,1fr)] gap-x-3 !pl-0"
    >
      <div className="flex flex-col items-center">
        <span className="num flex h-[22px] w-[22px] shrink-0 items-center justify-center rounded-full border border-border-strong bg-surface text-[14px] font-semibold text-ink-2">{n}</span>
        {!last && <span aria-hidden="true" className="w-px flex-1 bg-[var(--grid)]" />}
      </div>
      <div className={cx('min-w-0', last ? 'pb-0' : 'pb-4')}>
        <div className="mb-1 text-[15px] font-semibold tracking-[0.01em] text-ink">{title}</div>
        <div className="text-[16px] leading-snug text-ink-2">{children}</div>
      </div>
    </motion.li>
  );
}

/** KaTeX box at 18px, so script-size subscripts (W_col, z_true) stay at 13px or more. */
function TexBox({ tex }: { tex: string }) {
  return (
    <div className="scroll-quiet mt-1 overflow-x-auto rounded-lg bg-surface-2 px-2.5 py-1.5 text-[18px] text-ink">
      <TeX tex={tex} />
    </div>
  );
}

const ARBITRARY = (
  <>
    string index; the numbering is arbitrary <Cite kind="spec" ref="§3.6" />
  </>
);

function slotRows(cell: Cell, phrase: string): { k: string; v: string; active: boolean; note?: ReactNode }[] {
  return [
    { k: 'sem_types', v: `${cell.sem} (${TYPE_NAME[cell.sem]})`, active: true },
    {
      k: 'col_name_idxs',
      v: String(cell.colNameIdx),
      active: true,
      note: (
        <>
          “{phrase}” · {ARBITRARY}
        </>
      ),
    },
    { k: 'number_values', v: cell.sem === SemType.Number ? signed(cell.number, 5) : '0', active: cell.sem === SemType.Number },
    { k: 'text_values', v: String(cell.textIdx), active: cell.sem === SemType.Text, note: cell.sem === SemType.Text ? ARBITRARY : 'index 0, gated out later' },
    { k: 'datetime_values', v: cell.sem === SemType.DateTime ? signed(cell.datetime, 5) : '0', active: cell.sem === SemType.DateTime },
    { k: 'boolean_values', v: cell.sem === SemType.Boolean ? signed(cell.boolean, 5) : '0', active: cell.sem === SemType.Boolean },
    { k: 'class_value_idx', v: cell.sem === SemType.Text ? String(cell.textIdx) : '−1', active: false, note: 'carried along, never used by the model' },
  ];
}

function Panel({ data, file, row, col, roomy }: { data: ChapterData; file: TableFile; row: number; col: string; roomy: boolean }) {
  const { pdb, files, task, seedNode } = data;
  const fate = useMemo(() => rowFates(pdb, file, row).find((f) => f.col.name === col), [pdb, file, row, col]);
  const ctx = useContextWindow();
  const assets = useAssets();
  const phone = usePhone();
  const node = pdb.nodes[file.offset + row];
  if (!fate) return null;
  const missingInRow = rowFates(pdb, file, row).filter((f) => f.kind === 'missing').length;
  const rawShown = fate.kind === 'missing' || fate.kind === 'null-fk' ? '∅ (null)' : formatRaw(fate.value, fate.col.dtype);
  const steps: { title: ReactNode; body: ReactNode }[] = [];

  steps.push({
    title: 'Raw value',
    body: (
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className={cx('num text-[19px] font-semibold', fate.kind === 'missing' || fate.kind === 'null-fk' ? 'text-muted' : 'text-ink')}>
          {fate.col.dtype === 'str' && fate.kind === 'cell' ? `“${rawShown}”` : rawShown}
        </span>
        <span className="text-[15px]">
          physical dtype <code>{fate.col.dtype}</code>
          {fate.col.dtype === 'datetime' && fate.kind === 'cell' ? ` (${new Date(fate.value as number).toISOString().slice(11, 19)} UTC)` : ''}
        </span>
      </div>
    ),
  });

  if (fate.kind !== 'cell') {
    const why: Record<Exclude<Fate['kind'], 'cell'>, ReactNode> = {
      pk: (
        <>
          <code>{col}</code> is the primary key. pre.rs makes no cell and no edge from it (its column phrase gets a text index, which is never used). Its value equals the row
          position, so the node index {file.offset} + {row} = {file.offset + row} already identifies the row.
        </>
      ),
      link: (
        <>
          <code>{col}</code> is a foreign key. No cell: pre.rs pushes the parent’s node index{' '}
          <strong className="num text-ink">#{fate.kind === 'link' ? fate.parentNode : ''}</strong> onto this row’s <code>f2p_nbr_idxs</code> (now [{node?.f2pNbrIdxs.join(', ')}]) and adds a P→F
          edge on the parent that carries this row’s timestamp ({node?.ts != null ? isoSeconds(node.ts) : 'None'}). The column’s name is never used.
        </>
      ),
      'null-fk': (
        <>
          <code>{col}</code> is a foreign key with a null value. pre.rs skips it: no link, no cell.
        </>
      ),
      missing: (
        <>
          A missing value produces no cell at all. It is not imputed and not masked: this row simply has {node?.cells.length ?? 0} cells instead of{' '}
          {(node?.cells.length ?? 0) + missingInRow}.
        </>
      ),
    };
    steps.push({ title: fate.kind === 'pk' ? 'Primary key' : fate.kind === 'missing' ? 'Missing value' : 'Foreign key', body: why[fate.kind] });
    steps.push({
      title: 'Result',
      body: <StatusBadge status="info">{fate.kind === 'link' ? 'a link, no token' : 'nothing: no cell, no token'}</StatusBadge>,
    });
  } else {
    const cell = fate.cell;
    const phrase = pdb.text[cell.colNameIdx];
    steps.push({
      title: 'Semantic type',
      body: (
        <span className="inline-flex flex-wrap items-center gap-2">
          <SemBadge sem={cell.sem} size="sm" showLabel />
          <span>
            code {cell.sem}, read off dtype <code>{fate.col.dtype}</code>
          </span>
        </span>
      ),
    });

    // normalization
    let norm: ReactNode;
    const statsKey = file.tableType === 'Val' || file.tableType === 'Test' ? `${file.tableName}:Train` : file.key;
    const st = pdb.colStats[file.key]?.[col];
    // the number guard fired: σ undefined (n ≤ 1) or 0 (constant column) became 1 (pre.rs:338-339)
    const guarded =
      cell.sem === SemType.Number &&
      !!st &&
      (() => {
        const src = files.find((f) => f.key === statsKey);
        const vals = (src?.table.rows ?? []).map((r) => r[col]).filter((v): v is number => typeof v === 'number' && !Number.isNaN(v));
        return vals.length <= 1 || vals.every((v) => v === vals[0]);
      })();
    if (cell.sem === SemType.Number && st) {
      norm = (
        <>
          Per-column z-score with the statistics of <code>{statsKey}</code>
          {statsKey !== file.key ? ' (copied to this split)' : ''}: {st.count} values,{' '}
          {guarded ? (
            <>
              σ set to 1 by the guard <Cite kind="code" ref="rustler/src/pre.rs:338-339" />
            </>
          ) : (
            'sample std'
          )}
          .
          <TexBox tex={`z = \\dfrac{${texNum(fate.value as number, 3, { trim: true })} - ${texNum(st.mean, 3)}}{${texNum(st.std, 3)}} = ${texNum(cell.number, 4, { sign: true })}`} />
        </>
      );
    } else if (cell.sem === SemType.Boolean && st) {
      norm = (
        <>
          Cast to {fate.value ? 1 : 0}, then the per-column z-score of <code>{statsKey}</code>
          {statsKey !== file.key ? ' (copied to this split)' : ''}. {cell.boolean > 0 ? 'Positive, because it is true.' : 'Negative, because it is false.'}
          <TexBox tex={`z = \\dfrac{${fate.value ? 1 : 0} - ${texNum(st.mean, 4)}}{${texNum(st.std, 4)}} = ${texNum(cell.boolean, 4, { sign: true })}`} />
        </>
      );
    } else if (cell.sem === SemType.DateTime) {
      const { mean, std, count } = pdb.datetimeStats;
      norm = (
        <>
          The one global z-score of the database: μ<Sub>T</Sub> and σ<Sub>T</Sub> over all {count.toLocaleString('en-US')} datetime values, population std.
          <TexBox
            tex={`z = \\dfrac{\\text{${isoDay(fate.value as number)}} - \\text{${isoDay(mean)}}}{${texNum(std / DAY_MS, 1)}\\text{ days}} = ${texNum(cell.datetime, 4, { sign: true })}`}
          />
          {file.table.schema.timeCol === col && (
            <div className="mt-1.5 text-[15px]">
              This is the table’s time column, so it is also the source of the row’s timestamp: {node?.ts != null ? `${node.ts} s, used only by the sampler` : 'None for this row (no non-null foreign key)'}.
            </div>
          )}
        </>
      );
    } else {
      norm = (
        <>
          Not normalized. The string is interned in the database’s one string table and stored as its index <strong className="num text-ink">#{cell.textIdx}</strong>. Every cell
          holding the same string shares that index, and the text encoder embeds the raw string, without the column name.
        </>
      );
    }
    steps.push({ title: cell.sem === SemType.Text ? 'Text index' : 'Normalization', body: norm });

    steps.push({
      title: 'Cell record',
      body: (
        <table className="mt-0.5 w-full border-collapse text-[14.5px]">
          <tbody>
            {slotRows(cell, phrase).map((r) => (
              <tr key={r.k} className={cx('border-b border-grid/60 last:border-0', r.active ? 'text-ink' : 'text-muted', r.active && r.k.endsWith('_values') && 'bg-accent-soft')}>
                <td className="py-[3px] pr-2 pl-1 align-top">
                  <div className="font-mono">{r.k}</div>
                  {r.note && <div className="text-[14px] leading-tight text-muted">{r.note}</div>}
                </td>
                <td className={cx('num py-[3px] pr-1 text-right align-top whitespace-nowrap', r.active && r.k.endsWith('_values') && 'font-semibold')}>
                  {r.active && r.k.endsWith('_values') && <span aria-hidden="true" className="mr-1 inline-block h-1.5 w-1.5 rounded-full bg-accent align-middle" />}
                  {r.v}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ),
    });

    if (cell.sem !== SemType.Text) {
      const v = cell.sem === SemType.Number ? cell.number : cell.sem === SemType.DateTime ? cell.datetime : cell.boolean;
      const b = toBf16(v);
      steps.push({
        title: 'In a batch: bfloat16',
        body: (
          <>
            fly.rs rounds the stored float32 to bf16: <span className="num text-ink">{signed(v, 6)}</span> → <span className="num font-semibold text-ink">{signed(b, 6)}</span>{' '}
            <span className="text-muted">(error {Math.abs(b - v).toExponential(1)})</span>. This site keeps float32.
          </>
        ),
      });
    }

    // embedding
    const tok = ctx.tokens.find((t) => !t.isPadding && t.nodeIdx === file.offset + row && t.colName === col);
    const tname = SEM_TYPE_NAMES[cell.sem];
    const store = assets.store;
    const phraseVec = store && store.has(phrase) ? store.get(phrase) : null;
    const valueVec = cell.sem === SemType.Text && store && store.has(pdb.text[cell.textIdx]) ? store.get(pdb.text[cell.textIdx]) : null;
    const stored = cell.sem === SemType.Number ? cell.number : cell.sem === SemType.DateTime ? cell.datetime : cell.boolean;
    // fly.rs:457-460: the task's leakage columns ("<col> of <seed table>") are left out for the seed row and rows with its timestamp
    const seedN = seedNode >= 0 ? pdb.nodes[seedNode] : undefined;
    const seedF = seedNode >= 0 ? data.fileOfNode(seedNode) : undefined;
    const leaked =
      !!seedN && !!seedF && seedF.tableName === file.tableName && task.columnsToDrop.includes(col) && (file.offset + row === seedNode || (node?.ts ?? null) === (seedN.ts ?? null));
    const taskLabel = task.kind === 'autocomplete' ? `${task.table}.${task.targetCol}` : task.name;
    // long arguments become symbols (c, s, v), spelled out under the formula, so it never needs a sideways scroll
    const textVal = pdb.text[cell.textIdx];
    // the literal form only when the panel spans the figure (stacked layout, tablet and up) and the strings are short
    const sym = phone || !roomy || phrase.length > 36 || (cell.sem === SemType.Text && textVal.length > 24);
    const valueTex =
      cell.sem === SemType.Text
        ? `\\mathrm{RMSNorm}_{\\text{text}}\\!\\big(W_{\\text{text}}\\,E(${sym ? 's' : `\\text{“${texText(textVal, 26)}”}`}) + b\\big)`
        : `\\mathrm{RMSNorm}_{\\text{${tname}}}\\!\\big(W_{\\text{${tname}}} \\cdot ${sym ? 'v' : `(${texNum(stored, 3, { sign: true })})`} + b\\big)`;
    const colArg = sym ? 'c' : `\\text{“${texText(phrase, 48)}”}`;
    // phones: one bracket per line, so the 18px formula fits without a sideways scroll
    const embTex = phone
      ? `\\begin{aligned} x &= \\mathrm{RMSNorm}_{\\text{col}}\\big( \\\\ &\\qquad W_{\\text{col}}\\,E(c) + b\\big) \\\\ &\\quad + ${
          tok?.isTarget
            ? `m_{\\text{${tname}}}`
            : cell.sem === SemType.Text
              ? `\\mathrm{RMSNorm}_{\\text{text}}\\big( \\\\ &\\qquad W_{\\text{text}}\\,E(s) + b\\big)`
              : `\\mathrm{RMSNorm}_{\\text{${tname}}}\\big( \\\\ &\\qquad W_{\\text{${tname}}} \\cdot v + b\\big)`
        } \\end{aligned}`
      : `\\begin{aligned} x &= \\mathrm{RMSNorm}_{\\text{col}}\\!\\big(W_{\\text{col}}\\,E(${colArg}) + b\\big) \\\\ &\\quad + ${tok?.isTarget ? `m_{\\text{${tname}}}` : valueTex} \\end{aligned}`;
    const ctxStepNo = steps.length + 2;
    steps.push({
      title: leaked ? 'Embedding term it would feed (chapter 8)' : 'Embedding term it will feed (chapter 8)',
      body: (
        <>
          <TexBox
            tex={embTex}
          />
          {sym && (
            <div className="mt-1 text-[15px]">
              <span className="font-mono">c</span> = “{phrase}”
              {!tok?.isTarget && (
                <>
                  ,{' '}
                  {cell.sem === SemType.Text ? (
                    <>
                      <span className="font-mono">s</span> = “{textVal}”
                    </>
                  ) : (
                    <>
                      <span className="font-mono">v</span> = <span className="num">{signed(stored, 3)}</span>
                    </>
                  )}
                </>
              )}
            </div>
          )}
          <div className="mt-1.5">
            {tok?.isTarget ? (
              <>
                In the live context this cell is the <strong className="text-ink">masked target</strong>: its stored value stays in the batch, but the model adds the learned mask
                vector <code>mask_embs["{tname}"]</code> instead of the value term.
              </>
            ) : leaked ? (
              <>
                This term arises only in other tasks’ contexts: the live task drops this column for this row (step {ctxStepNo}).
              </>
            ) : (
              <>
                The value goes through <code>enc_dict["{tname}"]</code> and <code>norm_dict["{tname}"]</code>; the other three types’ terms are multiplied by 0.
              </>
            )}
          </div>
          {phraseVec && (
            <div className="mt-2">
              <VectorStrip values={phraseVec} label={`E(“${phrase}”), 384-d MiniLM`} columns={64} cellHeight={7} showNorm={false} />
            </div>
          )}
          {valueVec && (
            <div className="mt-2">
              <VectorStrip values={valueVec} label={`E(“${textVal}”), the raw string only`} columns={64} cellHeight={7} showNorm={false} />
            </div>
          )}
          {!store && assets.status === 'loading' && <div className="mt-1.5 text-[14.5px] text-muted">Loading the precomputed MiniLM embeddings…</div>}
        </>
      ),
    });

    steps.push({
      title: 'In the live context window',
      body: tok ? (
        <>
          Token at position <strong className="num text-ink">{tok.pos}</strong> of the current context window (seed #{ctx.params.seedNode}, L = {ctx.params.L}{' '}
          <ScaleNote inline />
          ).{tok.isTarget && ' It is the one masked cell.'}
        </>
      ) : leaked ? (
        <>
          Never a token in this task’s contexts: <code>{taskLabel}</code> drops <code>{col}</code> as a leakage column for the seed row and for every row with the seed’s
          timestamp <Cite kind="code" ref="rustler/src/fly.rs:457-460" /> <Cite kind="spec" ref="§4.9" />. The cell record itself is unchanged; preprocessing knows nothing about
          tasks.
        </>
      ) : (
        <>Not in the current context window: the sampler did not emit this cell for seed #{ctx.params.seedNode} (chapter 6). Its record still exists, ready for other contexts.</>
      ),
    });
  }

  return (
    <div className="min-w-0">
      <div className="mb-3 flex flex-wrap items-baseline gap-x-2 gap-y-1 border-b border-grid pb-2">
        <span aria-hidden="true" className="inline-block h-2.5 w-2.5 translate-y-[1px] rounded-[3px]" style={{ background: catVar(data.slotOf(file.tableName)) }} />
        <span className="font-mono text-[16.5px] font-semibold text-ink">
          {fileLabel(file)}[{row}].{col}
        </span>
        <span className="num text-[15px] text-ink-2">node #{file.offset + row}</span>
      </div>
      <ol key={`${file.key}:${row}:${col}`} className="!m-0 !list-none !p-0">
        {steps.map((s, i) => (
          <Step key={i} n={i + 1} title={s.title} last={i === steps.length - 1} delay={i * 0.05}>
            {s.body}
          </Step>
        ))}
      </ol>
    </div>
  );
}

/** True below the xl breakpoint (the inspector stacks there, table on top at full width). */
function useNarrow(): boolean {
  const q = '(max-width: 1279px)';
  return useSyncExternalStore(
    (cb) => {
      try {
        const m = window.matchMedia(q);
        m.addEventListener('change', cb);
        return () => m.removeEventListener('change', cb);
      } catch {
        return () => {};
      }
    },
    () => {
      try {
        return window.matchMedia(q).matches;
      } catch {
        return false;
      }
    },
    () => false,
  );
}

/**
 * A fade on the right edge of the TableView's scroller while columns are hidden to the right,
 * so it is clear that the table scrolls sideways (the scroller is found by its role="table").
 */
function useEdgeFade(dep: unknown) {
  const wrap = useRef<HTMLDivElement | null>(null);
  const [fade, setFade] = useState<{ top: number; height: number; hidden: boolean } | null>(null);
  useEffect(() => {
    const el = wrap.current?.querySelector<HTMLDivElement>('[role="table"]');
    if (!el || !wrap.current) return;
    const update = () => {
      const w = wrap.current!.getBoundingClientRect();
      const r = el.getBoundingClientRect();
      const more = el.scrollWidth - el.clientWidth - el.scrollLeft > 2;
      setFade(el.scrollWidth > el.clientWidth + 1 ? { top: r.top - w.top, height: el.clientHeight, hidden: !more } : null);
    };
    update();
    el.addEventListener('scroll', update, { passive: true });
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(update) : null;
    ro?.observe(el);
    return () => {
      el.removeEventListener('scroll', update);
      ro?.disconnect();
    };
  }, [dep]);
  return { wrap, fade };
}

export function CellInspectorFigure({ data, picks }: { data: ChapterData; picks: CellPick[] }) {
  const { files, pdb, seedNode } = data;
  const first = picks[0] ?? { key: files[0].key, row: 0, col: files[0].table.schema.columns[0].name };
  const [sel, setSel] = useState({ key: first.key, row: first.row, col: first.col });
  const [mode, setMode] = useState<'raw' | 'z'>('raw');
  const narrow = useNarrow();
  const file = files.find((f) => f.key === sel.key) ?? files[0];
  const row = Math.min(sel.row, file.numNodes - 1);
  const col = file.table.schema.columns.some((c) => c.name === sel.col) ? sel.col : file.table.schema.columns[0].name;
  const seedRow = seedNode >= file.offset && seedNode < file.offset + file.numNodes ? seedNode - file.offset : null;
  const { wrap, fade } = useEdgeFade(file.key);

  const zValue = (r: number, c: string) => {
    const cell = pdb.nodes[file.offset + r]?.cells.find((x) => x.colName === c);
    if (!cell) return null;
    return cell.sem === SemType.Number ? cell.number : cell.sem === SemType.DateTime ? cell.datetime : cell.sem === SemType.Boolean ? cell.boolean : null;
  };

  return (
    <div className="min-w-0">
      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[14.5px] font-medium text-ink-2">Try</span>
        {picks.map((p) => (
          <PickChip key={p.label} active={p.key === file.key && p.row === row && p.col === col} onClick={() => setSel({ key: p.key, row: p.row, col: p.col })}>
            {p.label}
          </PickChip>
        ))}
      </div>
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <div className="min-w-0 xl:sticky xl:top-[calc(var(--topbar-h)+16px)] xl:self-start">
          <div className="mb-3 grid grid-cols-[84px_minmax(0,1fr)] gap-2 sm:grid-cols-[minmax(0,1.3fr)_84px_minmax(0,1fr)]">
            <Select label="Table file" className="col-span-2 sm:col-span-1" value={file.key} onChange={(e) => setSel({ key: e.target.value, row: 0, col: sel.col })}>
              {files.map((f) => (
                <option key={f.key} value={f.key}>
                  {f.key}
                </option>
              ))}
            </Select>
            <NumberField
              label="Row"
              value={row}
              min={0}
              max={file.numNodes - 1}
              onChange={(v) => setSel({ key: file.key, row: Math.max(0, Math.min(file.numNodes - 1, Math.round(v))), col })}
            />
            <Select label="Column" value={col} onChange={(e) => setSel({ key: file.key, row, col: e.target.value })}>
              {file.table.schema.columns.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                </option>
              ))}
            </Select>
          </div>
          <div ref={wrap} className="relative">
          <TableView
            schema={file.table.schema}
            rows={file.table.rows}
            slot={data.slotOf(file.tableName)}
            title={fileLabel(file)}
            zValue={zValue}
            mode={mode}
            onModeChange={setMode}
            highlightCells={[`${row}:${col}`]}
            highlightRows={seedRow !== null ? [seedRow] : []}
            onCellClick={(r, c) => setSel({ key: file.key, row: r, col: c })}
            onRowClick={(r) => setSel({ key: file.key, row: r, col })}
            scrollToRow={row}
            scrollToCol={col}
            maxHeight={narrow ? 300 : 560}
            footer={
              <>
                Click any cell{fade ? ' (the table scrolls sideways)' : ''}. {seedRow !== null ? `Shaded row ${seedRow}: the live seed. ` : ''}The z-scored toggle shows
                numbers, booleans and datetimes exactly as pre.rs stores them; text is stored as a string index, and keys store no cell.
              </>
            }
          />
          {fade && (
            <div
              aria-hidden="true"
              className="pointer-events-none absolute right-px w-9 transition-opacity duration-200"
              style={{ top: fade.top, height: fade.height, opacity: fade.hidden ? 0 : 1, background: 'linear-gradient(to right, transparent, var(--surface))' }}
            />
          )}
          </div>
        </div>
        <div className="min-w-0 rounded-xl border border-border bg-surface px-4 py-3.5">
          <Panel data={data} file={file} row={row} col={col} roomy={narrow} />
        </div>
      </div>
    </div>
  );
}
