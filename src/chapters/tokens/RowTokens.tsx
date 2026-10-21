/**
 * Section 2 figure: one row of the context becomes k tokens (one per emitted cell). The row-level
 * fields (node_idxs, f2p_nbr_idxs, table_name_idxs) are written again for every cell
 * (fly.rs:462-469); the cell-level fields differ (fly.rs:470-500). The PK becomes the node id, each
 * FK a parent id, a missing value nothing (pre.rs:439-611; spec §1.2, §3.5, §4.13).
 */
import { ArrowUp, KeyRound, Link2, Lock } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import { useMemo, useState, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Figure, Segmented, Select, SemBadge, TableSwatch, cx, formatRaw, formatSigned, semOfDType, truncate } from '@/components';
import { SEM_TYPE_NAMES, SemType, type Token } from '@/engine/types';
import { ParentSlots } from './fields';
import { nodeLabel, nodeRef, sourceTable, type TokensLive } from './live';

interface RowOption {
  node: number;
  table: string;
  tableType: string | null;
  rowPos: number;
  positions: number[];
  parents: number;
}

/** The stored value slot as the model receives it (the target's hidden label included, fly.rs:482-490). */
function valueText(t: Token, text: readonly string[]): string {
  switch (t.sem) {
    case SemType.Number:
      return formatSigned(t.numberValue);
    case SemType.DateTime:
      return formatSigned(t.datetimeValue);
    case SemType.Boolean:
      return formatSigned(t.booleanValue);
    default:
      return `E(“${truncate(text[t.textIdx] ?? '', 16)}”)`;
  }
}

const VALUE_FIELD = ['number_values', 'text_values', 'datetime_values', 'boolean_values'] as const;

export function RowTokens({ live }: { live: TokensLive }) {
  const { db, pdb, ctx, slots, tokensByNode, inspectAndGo } = live;
  const reduce = useReducedMotion();

  const rows: RowOption[] = useMemo(
    () =>
      ctx.visited
        .filter((v) => tokensByNode.has(v.node))
        .map((v) => {
          const n = pdb.nodes[v.node];
          return {
            node: v.node,
            table: n.tableName,
            tableType: n.tableType,
            rowPos: n.rowPos,
            positions: tokensByNode.get(v.node) ?? [],
            parents: n.f2pNbrIdxs.length,
          };
        }),
    [ctx, pdb, tokensByNode],
  );
  const tables = useMemo(() => {
    const seen: string[] = [];
    for (const r of rows) if (!seen.includes(r.table)) seen.push(r.table);
    return seen;
  }, [rows]);

  const defaultNode = useMemo(() => {
    const maxParents = Math.max(0, ...rows.map((r) => r.parents));
    return (rows.find((r) => r.parents === maxParents && r.positions.length > 1) ?? rows[0])?.node ?? -1;
  }, [rows]);
  const [picked, setPicked] = useState<number | null>(null);
  const node = picked !== null && rows.some((r) => r.node === picked) ? picked : defaultNode;
  const row = rows.find((r) => r.node === node);
  const gnode = pdb.nodes[node];
  const table = row?.table ?? '';
  const inTable = rows.filter((r) => r.table === table);
  const raw = row ? sourceTable(db, row.table, row.tableType) : undefined;
  const rec = raw && row ? raw.rows[row.rowPos] : undefined;
  const toks = (row?.positions ?? []).map((p) => ctx.tokens[p]);
  const k = toks.length;
  const present = (n: number) => tokensByNode.get(n)?.[0];
  const emitted = new Set(toks.map((t) => t.colName));
  const seedNode = pdb.nodes[ctx.params.seedNode];

  if (!row || !gnode || !raw || !rec) return null;
  // Global node ids: table offset + row position (pre.rs:302-305, 537); FK value + parent offset (pre.rs:477).
  const offset = pdb.tableInfo[`${row.table}:${row.tableType ?? 'Db'}`]?.nodeIdxOffset ?? node - row.rowPos;

  // Fate of every column of the raw row.
  const fates = raw.schema.columns.map((c) => {
    const v = rec[c.name];
    const isNull = v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v));
    if (c.name === raw.schema.pkey)
      return {
        col: c,
        kind: 'pk' as const,
        v,
        note: (
          <>
            no token · node_idxs = offset {offset} + row {row.rowPos} = <span className="num font-semibold text-ink">{node}</span>
          </>
        ),
      };
    if (c.name in raw.schema.fkeys) {
      if (isNull) return { col: c, kind: 'null' as const, v, note: <>missing: no parent</> };
      const parentTable = raw.schema.fkeys[c.name];
      const info = pdb.tableInfo[`${parentTable}:Db`];
      const pid = info ? info.nodeIdxOffset + Number(v) : undefined;
      return {
        col: c,
        kind: 'fk' as const,
        v,
        note: (
          <>
            no token · parent = {parentTable} offset {info?.nodeIdxOffset ?? '?'} + {String(v)} = <span className="num font-semibold text-ink">{pid ?? '?'}</span>{' '}
            in <span className="font-mono">f2p_nbr_idxs</span>
          </>
        ),
      };
    }
    if (isNull) return { col: c, kind: 'null' as const, v, note: <>missing → no token</> };
    const tok = toks.find((t) => t.colName === c.name);
    if (tok)
      return {
        col: c,
        kind: 'cell' as const,
        v,
        tok,
        note: (
          <>
            → token <span className="num">#{tok.pos}</span>
          </>
        ),
      };
    const colIdx = pdb.columnIndex[`${c.name} of ${row.table}`];
    const dropped =
      colIdx !== undefined && ctx.params.dropColNameIdxs.includes(colIdx) && (gnode.nodeIdx === seedNode?.nodeIdx || gnode.ts === seedNode?.ts);
    return { col: c, kind: 'cut' as const, v, note: dropped ? <>leakage column, dropped</> : <>after the stop at L</> };
  });

  const cellCols = `minmax(124px,auto) repeat(${k}, minmax(118px, 1fr))`;
  const merged = (key: string, content: ReactNode, i: number) => (
    <motion.div
      key={`${node}-${key}`}
      initial={reduce ? false : { scaleX: 0, opacity: 0 }}
      animate={{ scaleX: 1, opacity: 1 }}
      transition={{ duration: 0.55, delay: 0.08 * i, ease: [0.22, 1, 0.36, 1] }}
      style={{ gridColumn: `2 / span ${k}`, transformOrigin: 'left center' }}
      className="relative flex min-h-[34px] min-w-0 items-center justify-between gap-3 rounded-[7px] border border-accent/40 bg-accent-soft px-2.5"
    >
      <span className="min-w-0 truncate">{content}</span>
      {k > 1 && (
        <span className="num shrink-0 text-[14px] font-semibold text-accent-text" title={`written ${k} times, once per cell`}>
          × {k}
        </span>
      )}
    </motion.div>
  );
  const per = (key: string, render: (t: Token) => ReactNode) =>
    toks.map((t, j) => (
      <motion.div
        key={`${node}-${key}-${t.pos}`}
        initial={reduce ? false : { opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.3 + 0.05 * j }}
        className={cx(
          'flex min-h-[36px] min-w-0 items-center gap-1.5 rounded-[7px] border bg-surface px-2 text-[15px]',
          t.isTarget ? 'target-outline border-transparent' : 'border-border',
        )}
      >
        {render(t)}
      </motion.div>
    ));
  const label = (name: string, sub: string) => (
    <div className="flex min-w-0 flex-col justify-center pr-1">
      <code className="!border-0 !bg-transparent !p-0 text-[15px] font-semibold text-ink">{name}</code>
      <span className="text-[14px] text-ink-2">{sub}</span>
    </div>
  );

  return (
    <Figure
      wide
      title="One row in, k tokens out"
      controls={
        <>
          <Segmented
            label="Table"
            value={table}
            onChange={(tb) => {
              const first = rows.find((r) => r.table === tb && r.positions.length > 0);
              if (first) setPicked(first.node);
            }}
            options={tables.map((tb) => ({ value: tb, label: <TableSwatch slot={slots[tb] ?? 0} label={tb} /> }))}
          />
          <Select label="Row (visit order)" value={String(node)} onChange={(e) => setPicked(Number(e.target.value))} className="min-w-[200px]">
            {inTable.map((r) => (
              <option key={r.node} value={r.node}>
                {truncate(nodeLabel(pdb, r.node), 40)} · {r.positions.length} {r.positions.length === 1 ? 'token' : 'tokens'}
              </option>
            ))}
          </Select>
        </>
      }
      caption={
        <>
          Top: the raw row, column by column, and what becomes of each value. Bottom: the row’s tokens as the model receives them. Shaded bars are
          the row-level fields, written again for every cell; the per-cell fields differ from token to token. The masked target still stores its
          label (lock icon), which the input embedding gates out (Step 3). E(“…”) is the frozen MiniLM embedding of a string (
          <Link to="/semantics">chapter 5</Link>). Click a token number to inspect it in Step 1.
        </>
      }
    >
      {/* raw row */}
      <div className="pb-1">
        <div className="flex flex-wrap gap-2">
          {fates.map((f) => (
            <div
              key={f.col.name}
              className={cx(
                'flex w-[calc(50%-4px)] shrink-0 flex-col rounded-[9px] border px-2.5 py-2 sm:w-[156px]',
                f.kind === 'cell' ? 'border-border bg-surface' : 'border-dashed border-border-strong bg-surface-2/60',
              )}
            >
              <div className="flex min-w-0 items-center gap-1.5 text-[14.5px] text-ink-2" title={f.col.name}>
                {f.kind === 'pk' ? (
                  <KeyRound aria-label="primary key" size={14} className="shrink-0" />
                ) : f.kind === 'fk' ? (
                  <Link2 aria-label="foreign key" size={14} className="shrink-0" />
                ) : (
                  <SemBadge sem={semOfDType(f.col.dtype)} size="sm" bare />
                )}
                <span className="truncate font-mono font-semibold">{f.col.name}</span>
              </div>
              <div className={cx('mt-1 font-mono text-[15.5px] leading-snug [overflow-wrap:anywhere]', f.kind === 'null' ? 'text-ink-2' : 'text-ink')}>
                {'tok' in f && f.tok?.isTarget ? (
                  <span className="target-outline rounded-[3px] px-1 font-bold text-accent-text">?</span>
                ) : (
                  formatRaw(f.v, f.col.dtype)
                )}
              </div>
              <div className="mt-1.5 text-[14px] leading-snug text-ink-2">
                {'tok' in f && f.tok ? (
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 text-left text-accent-text hover:underline"
                    onClick={() => inspectAndGo(f.tok!.pos, 'Back to Step 2')}
                    title="Inspect this token in Step 1"
                  >
                    {f.note} <ArrowUp aria-hidden="true" size={14} />
                  </button>
                ) : (
                  f.note
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="my-4 flex items-center gap-3 text-[14.5px] text-ink-2">
        <span className="h-px flex-1 bg-grid" />
        <span>
          {nodeRef(gnode)}: {fates.filter((f) => f.kind === 'cell').length} feature cells in the context → {k} {k === 1 ? 'token' : 'tokens'}
          {emitted.size < fates.filter((f) => f.kind === 'cell' || f.kind === 'cut').length ? ' (some cells missing, see above)' : ''}
        </span>
        <span className="h-px flex-1 bg-grid" />
      </div>

      {/* tokens grid */}
      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-[14.5px] text-ink-2">
        {k > 1 && (
          <span>
            <span className="num font-semibold text-accent-text">× {k}</span> = the same value written {k} times, once per cell
          </span>
        )}
        {k > 1 && <span className="sm:hidden">Scroll sideways to see all {k} tokens →</span>}
      </div>
      <div className="scroll-quiet relative -mx-1 overflow-x-auto px-1 pb-1">
        <div className="grid min-w-max gap-x-1.5 gap-y-1.5" style={{ gridTemplateColumns: cellCols }}>
          <div />
          {toks.map((t) => (
            <button
              key={t.pos}
              type="button"
              onClick={() => inspectAndGo(t.pos, 'Back to Step 2')}
              className="num inline-flex items-center justify-center gap-0.5 rounded-[6px] py-0.5 text-[14.5px] text-accent-text hover:bg-surface-2 hover:underline"
              title="Inspect this token in Step 1"
            >
              pos #{t.pos} <ArrowUp aria-hidden="true" size={13} />
            </button>
          ))}

          {label('node_idxs', 'row-level')}
          {merged('node', <span className="num text-[15px] font-semibold text-ink">{node}</span>, 0)}

          {label('f2p_nbr_idxs', 'row-level')}
          {merged('f2p', <span className="num text-[15px] text-ink">[{gnode.f2pNbrIdxs.concat([-1, -1, -1, -1, -1]).slice(0, 5).join(', ')}]</span>, 1)}

          {label('table_name_idxs', 'row-level')}
          {merged(
            'table',
            <span className="text-[15px]">
              <span className="num font-semibold text-ink">{gnode.tableNameIdx}</span>{' '}
              <span className="font-mono text-ink-2">“{pdb.text[gnode.tableNameIdx]}”</span>
            </span>,
            2,
          )}

          {label('col_name_idxs', 'per cell')}
          {per('col', (t) => (
            <span className="min-w-0 truncate">
              <span className="num font-semibold text-ink">{t.colNameIdx}</span> <span className="font-mono text-[14px] text-ink-2">{t.colName}</span>
            </span>
          ))}

          {label('sem_types', 'per cell')}
          {per('sem', (t) => (
            <>
              <span className="num font-semibold text-ink">{t.sem}</span>
              <SemBadge sem={t.sem} size="sm" bare />
              <span className="truncate text-[14px] text-ink-2">{SEM_TYPE_NAMES[t.sem]}</span>
            </>
          ))}

          {label('value slot', 'per cell')}
          {per('val', (t) => (
            <span
              className="inline-flex min-w-0 items-center gap-1"
              title={
                t.isTarget
                  ? 'The hidden label: stored in the arrays, gated out of the input by masks (model.py:219)'
                  : t.sem === SemType.Text
                    ? `E(“${pdb.text[t.textIdx] ?? ''}”), text[${t.textIdx}]`
                    : undefined
              }
            >
              <span className="shrink-0 font-mono text-[14px] text-ink-2">{VALUE_FIELD[t.sem]?.replace('_values', '')}</span>
              <span className={cx('num truncate', t.isTarget ? 'font-bold text-accent-text' : 'text-ink')}>{valueText(t, pdb.text)}</span>
              {t.isTarget && <Lock aria-label="hidden label, gated out" size={14} className="shrink-0 text-accent-text" />}
            </span>
          ))}

          {label('masks', 'per cell')}
          {per('mask', (t) => (
            <span className={cx('num', t.mask ? 'font-semibold text-accent-text' : 'text-ink-2')}>{t.mask ? 'true' : 'false'}</span>
          ))}
        </div>
      </div>

      <div className="mt-5 rounded-xl bg-surface-2 px-3.5 py-3">
        <div className="mb-2 flex flex-wrap items-baseline gap-x-2 text-[15.5px]">
          <span className="font-semibold text-ink">The parent list, slot by slot</span>
          <span className="text-ink-2">
            {gnode.f2pNbrIdxs.length === 0
              ? 'This row has no foreign key, so all five slots stay −1.'
              : `${gnode.f2pNbrIdxs.length} of 5 slots filled, in foreign-key column order; the rest stay −1.`}
          </span>
        </div>
        <ParentSlots
          ids={gnode.f2pNbrIdxs.concat([-1, -1, -1, -1, -1]).slice(0, 5)}
          pdb={pdb}
          slots={slots}
          present={present}
          onPick={(p) => inspectAndGo(p, 'Back to Step 2')}
        />
        {gnode.f2pNbrIdxs.length > 0 && (
          <div className="mt-2 text-[14.5px] text-ink-2">
            {gnode.f2pNbrIdxs.every((p) => present(p) !== undefined)
              ? 'Every listed parent is in this context. Click one to inspect its first token.'
              : 'A faded slot is a parent that did not make it into the context. It is still listed: the list holds all parents, in the context or not.'}
          </div>
        )}
      </div>
    </Figure>
  );
}
